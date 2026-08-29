"""Prompt signatures, prewarm identity, and resolved model metadata."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from typing import Any, Iterator, Mapping

if __package__:
    from .llm_profiles import (
        generated_profile_manifest_path,
        llm_model_references_from_payload,
    )
    from .remote_graph_analysis import _is_link, _is_transportable_output_type
    from .remote_plan_types import (
        BoundaryInputSpec,
        LinkedOutputRef,
        ModalPromptValidationError,
    )
    from .settings import ModalSyncSettings
else:  # pragma: no cover - flat import inside the Modal container.
    from llm_profiles import (
        generated_profile_manifest_path,
        llm_model_references_from_payload,
    )
    from remote_graph_analysis import _is_link, _is_transportable_output_type
    from remote_plan_types import (
        BoundaryInputSpec,
        LinkedOutputRef,
        ModalPromptValidationError,
    )
    from settings import ModalSyncSettings

logger = logging.getLogger(__name__)

_ROOT_LOADER_PREWARM_CLASS_TYPES = frozenset(
    {
        "CheckpointLoaderSimple",
        "UNETLoader",
        "CLIPLoader",
        "DualCLIPLoader",
    }
)


def _prompt_value_signature_fragment(
    prompt: dict[str, Any],
    value: Any,
    *,
    memo: dict[str, str],
) -> Any:
    """Return a stable structural signature fragment for one prompt input value."""
    if _is_link(value):
        source_node_id = str(value[0])
        return {
            "kind": "link",
            "source_node_id": source_node_id,
            "output_index": int(value[1]),
            "source_digest": _prompt_node_signature_digest(
                prompt,
                source_node_id,
                memo=memo,
            ),
        }
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        if value != value:
            return {"kind": "float", "value": "nan"}
        if value == float("inf"):
            return {"kind": "float", "value": "inf"}
        if value == float("-inf"):
            return {"kind": "float", "value": "-inf"}
        return value
    if isinstance(value, list):
        return {
            "kind": "list",
            "items": [
                _prompt_value_signature_fragment(prompt, item, memo=memo)
                for item in value
            ],
        }
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [
                _prompt_value_signature_fragment(prompt, item, memo=memo)
                for item in value
            ],
        }
    if isinstance(value, dict):
        return {
            "kind": "dict",
            "items": [
                {
                    "key": str(key),
                    "value": _prompt_value_signature_fragment(
                        prompt, value[key], memo=memo
                    ),
                }
                for key in sorted(value)
            ],
        }
    return {
        "kind": "repr",
        "type": type(value).__name__,
        "value": repr(value),
    }


def _prompt_node_signature_digest(
    prompt: dict[str, Any],
    node_id: str,
    *,
    memo: dict[str, str],
) -> str:
    """Return a stable digest for one prompt node and its upstream prompt inputs."""
    if node_id in memo:
        return memo[node_id]

    prompt_node = prompt.get(str(node_id))
    if prompt_node is None:
        digest = hashlib.sha256(
            json.dumps(
                {"kind": "missing-node", "node_id": str(node_id)},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        memo[str(node_id)] = digest
        return digest

    payload = {
        "kind": "prompt-node",
        "node_id": str(node_id),
        "class_type": str(prompt_node.get("class_type", "")),
        "inputs": [
            {
                "name": str(input_name),
                "value": _prompt_value_signature_fragment(
                    prompt,
                    input_value,
                    memo=memo,
                ),
            }
            for input_name, input_value in sorted(
                (prompt_node.get("inputs") or {}).items()
            )
        ],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    memo[str(node_id)] = digest
    return digest


def _iter_loader_snapshot_prompt_payloads(
    payload: Mapping[str, Any]
) -> Iterator[Mapping[str, Any]]:
    """Yield prompt-bearing payload fragments that may contain root loader nodes."""
    split_proxy_payloads = payload.get("split_proxy_payloads")
    if isinstance(split_proxy_payloads, dict):
        for phase_payload in split_proxy_payloads.values():
            if isinstance(phase_payload, Mapping):
                yield phase_payload
        return
    if isinstance(split_proxy_payloads, list):
        for phase_payload in split_proxy_payloads:
            if isinstance(phase_payload, Mapping):
                yield phase_payload
        return
    if isinstance(payload.get("subgraph_prompt"), Mapping):
        yield payload


def _is_root_literal_loader_node(prompt_node: Mapping[str, Any]) -> bool:
    """Return whether one prompt node is a supported root loader with literal inputs."""
    class_type = str(prompt_node.get("class_type") or "")
    if class_type not in _ROOT_LOADER_PREWARM_CLASS_TYPES:
        return False
    inputs = prompt_node.get("inputs")
    if not isinstance(inputs, Mapping):
        return False
    return not any(_is_link(input_value) for input_value in inputs.values())


def _loader_prewarm_plan_signature(class_type: str, inputs: Mapping[str, Any]) -> str:
    """Return a stable signature for one synthetic loader-prewarm plan."""
    return json.dumps(
        {
            "class_type": class_type,
            "inputs": copy.deepcopy(dict(inputs)),
        },
        sort_keys=True,
        default=str,
    )


def _uses_llm_worker_affinity(payload: Mapping[str, Any]) -> bool:
    """Return whether an execution payload belongs to the isolated LLM worker pool."""
    affinity_group = (
        str(payload.get("remote_worker_affinity_group") or "").strip().lower()
    )
    return affinity_group == "llm"


def _payload_loader_snapshot_profile_key(payload: Mapping[str, Any]) -> str:
    """Return the stable loader snapshot profile key derivable from one payload."""
    prompt_id = payload.get("prompt_id")
    normalized_prompt_id = str(prompt_id) if prompt_id is not None else None
    plan_signatures: set[str] = set()
    for prompt_payload in _iter_loader_snapshot_prompt_payloads(payload):
        subgraph_prompt = prompt_payload.get("subgraph_prompt")
        if not isinstance(subgraph_prompt, Mapping):
            continue
        for node_id, prompt_node in subgraph_prompt.items():
            if not isinstance(prompt_node, Mapping) or not _is_root_literal_loader_node(
                prompt_node
            ):
                continue
            class_type = str(prompt_node.get("class_type") or "")
            inputs = prompt_node.get("inputs")
            if not isinstance(inputs, Mapping):
                continue
            plan_signatures.add(_loader_prewarm_plan_signature(class_type, inputs))
            logger.debug(
                "Derived rewrite-time loader prewarm plan for component=%s node=%s class_type=%s prompt_id=%s.",
                payload.get("component_id"),
                node_id,
                class_type,
                normalized_prompt_id,
            )
    if not plan_signatures:
        return ""
    profile_digest = hashlib.sha256(
        json.dumps({"plan_signatures": sorted(plan_signatures)}, sort_keys=True).encode(
            "utf-8"
        )
    ).hexdigest()
    return f"loader-profile:{profile_digest}"


def _stamp_snapshot_profile_key(
    payload: dict[str, Any], snapshot_profile_key: str
) -> None:
    """Attach one loader snapshot profile to eligible Comfy payload descendants."""
    if not snapshot_profile_key:
        return
    if _uses_llm_worker_affinity(payload):
        payload.pop("snapshot_profile_key", None)
        logger.info(
            "Omitting Comfy loader snapshot profile from LLM worker component=%s.",
            payload.get("component_id"),
        )
    else:
        payload["snapshot_profile_key"] = snapshot_profile_key
    split_proxy_payloads = payload.get("split_proxy_payloads")
    if isinstance(split_proxy_payloads, dict):
        for phase_payload in split_proxy_payloads.values():
            if isinstance(phase_payload, dict):
                _stamp_snapshot_profile_key(phase_payload, snapshot_profile_key)
        return
    if isinstance(split_proxy_payloads, list):
        for phase_payload in split_proxy_payloads:
            if isinstance(phase_payload, dict):
                _stamp_snapshot_profile_key(phase_payload, snapshot_profile_key)


def _attach_snapshot_profile_key(
    payload: dict[str, Any], settings: ModalSyncSettings
) -> dict[str, Any]:
    """Stamp a deterministic loader snapshot profile onto one payload when enabled."""
    if not settings.enable_gpu_memory_snapshot or not settings.enable_loader_prewarm:
        return payload
    snapshot_profile_key = _payload_loader_snapshot_profile_key(payload)
    if snapshot_profile_key:
        _stamp_snapshot_profile_key(payload, snapshot_profile_key)
        logger.info(
            "Attached rewrite-time loader snapshot profile %s to component=%s payload_kind=%s.",
            snapshot_profile_key,
            payload.get("component_id"),
            payload.get("payload_kind"),
        )
    return payload


def _resolved_llm_profile_entry(
    profile: Any,
    settings: ModalSyncSettings,
) -> dict[str, Any]:
    """Serialize planner metadata plus its persisted security-scan result."""
    security_scan_complete = True
    if getattr(profile, "source", "curated") == "generated":
        manifest_path = generated_profile_manifest_path(
            settings.local_storage_root,
            profile.profile_id,
        )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError) as exc:
            raise ModalPromptValidationError(
                f"Planner-resolved LLM manifest {manifest_path} is unavailable."
            ) from exc
        if not isinstance(manifest, Mapping):
            raise ModalPromptValidationError(
                f"Planner-resolved LLM manifest {manifest_path} is invalid."
            )
        security_scan_complete = bool(
            manifest.get("security_scan_complete", False)
        )
    return {
        "profile": profile.to_mapping(),
        "security_scan_complete": security_scan_complete,
    }


def _attach_resolved_llm_profiles(
    payload: dict[str, Any],
    resolved_profiles: Mapping[str, Any],
    settings: ModalSyncSettings,
) -> None:
    """Attach only the planner-resolved profiles used by each payload subtree."""
    if not resolved_profiles:
        return
    references = llm_model_references_from_payload(payload)
    entries = {
        reference: _resolved_llm_profile_entry(resolved_profiles[reference], settings)
        for reference in references
        if reference in resolved_profiles
    }
    if entries:
        payload["resolved_llm_profiles"] = entries
    split_payloads = payload.get("split_proxy_payloads")
    if isinstance(split_payloads, Mapping):
        children = split_payloads.values()
    elif isinstance(split_payloads, list):
        children = split_payloads
    else:
        children = ()
    for child in children:
        if isinstance(child, dict):
            _attach_resolved_llm_profiles(child, resolved_profiles, settings)


def _boundary_source_signature(
    prompt: dict[str, Any],
    source: LinkedOutputRef,
) -> str:
    """Return a stable prompt-structural fingerprint for one boundary source output."""
    payload = {
        "kind": "boundary-source",
        "source_node_id": source.node_id,
        "output_index": int(source.output_index),
        "source_digest": _prompt_node_signature_digest(
            prompt,
            source.node_id,
            memo={},
        ),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"SRC_{digest}"


def _serialize_boundary_input_specs(
    boundary_inputs: list[BoundaryInputSpec],
    *,
    signature_prompt: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return serialized boundary input payloads with stable provenance for non-transportable inputs."""
    serialized_boundary_inputs: list[dict[str, Any]] = []
    for boundary_input in boundary_inputs:
        serialized_boundary_input = {
            "proxy_input_name": boundary_input.proxy_input_name,
            "io_type": boundary_input.io_type,
            "targets": [
                {"node_id": target.node_id, "input_name": target.input_name}
                for target in boundary_input.targets
            ],
        }
        if not _is_transportable_output_type(boundary_input.io_type):
            serialized_boundary_input["source_signature"] = _boundary_source_signature(
                signature_prompt,
                boundary_input.source,
            )
        serialized_boundary_inputs.append(serialized_boundary_input)
    return serialized_boundary_inputs
