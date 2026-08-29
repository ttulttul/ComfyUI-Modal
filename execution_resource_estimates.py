"""Resource requirements and workload signatures for remote execution planning."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any, Iterator, Mapping

if __package__:
    from .execution_environments import ExecutionProvider, WorkflowExecutionPreferences
    from .llm_profiles import get_llm_profile
    from .llm_resolver import resolve_model_profile
    from .remote_graph_analysis import _is_link
    from .remote_plan_types import (
        ComponentMemoryEstimate,
        ModalPromptValidationError,
        RemoteComponentPlan,
    )
    from .settings import ModalSyncSettings
    from .sync_engine import MODEL_FILE_EXTENSIONS, resolve_model_path
else:  # pragma: no cover - flat import inside the Modal container.
    from execution_environments import ExecutionProvider, WorkflowExecutionPreferences
    from llm_profiles import get_llm_profile
    from llm_resolver import resolve_model_profile
    from remote_graph_analysis import _is_link
    from remote_plan_types import (
        ComponentMemoryEstimate,
        ModalPromptValidationError,
        RemoteComponentPlan,
    )
    from settings import ModalSyncSettings
    from sync_engine import MODEL_FILE_EXTENSIONS, resolve_model_path

logger = logging.getLogger(__name__)

_MODEL_VRAM_WEIGHT_MULTIPLIER = 1.20
_MODEL_VRAM_HEADROOM_BYTES = 4 * 1024**3
_MODEL_RAM_HEADROOM_BYTES = 4 * 1024**3
_ADDITIVE_MODEL_CLASS_TOKENS = (
    "adapter",
    "controlnet",
    "ipadapter",
    "lora",
)


def _prompt_llm_model_references(prompt: Mapping[str, Any]) -> tuple[str, ...]:
    """Return every fixed Modal LLM model reference in a prompt."""
    references: set[str] = set()
    for prompt_node in prompt.values():
        if not isinstance(prompt_node, Mapping):
            continue
        if str(prompt_node.get("class_type") or "") != "ModalLLM":
            continue
        inputs = prompt_node.get("inputs")
        if not isinstance(inputs, Mapping):
            continue
        model_reference = inputs.get("model_profile")
        if isinstance(model_reference, str) and model_reference.strip():
            references.add(model_reference.strip())
    return tuple(sorted(references))


def _resolve_prompt_llm_profiles(
    prompt: Mapping[str, Any],
    settings: ModalSyncSettings,
) -> dict[str, Any]:
    """Resolve LLM metadata before environment admission and cost ranking."""
    storage_root = Path(
        getattr(settings, "local_storage_root", "/tmp/comfyui-modal-sync-storage")
    )
    profiles: dict[str, Any] = {}
    for model_reference in _prompt_llm_model_references(prompt):
        try:
            profile = get_llm_profile(model_reference, storage_root=storage_root)
        except ValueError as profile_error:
            if model_reference.startswith("hf-"):
                raise ModalPromptValidationError(str(profile_error)) from profile_error
            try:
                profile = resolve_model_profile(model_reference, storage_root).profile
            except ValueError as resolution_error:
                raise ModalPromptValidationError(
                    str(resolution_error)
                ) from resolution_error
        profiles[model_reference] = profile
        logger.info(
            "Resolved planner LLM profile model=%s profile=%s weights_gib=%.2f "
            "estimated_vram_gib=%.2f.",
            model_reference,
            profile.profile_id,
            profile.artifact_bytes / 1024**3,
            profile.estimated_vram_gb,
        )
    return profiles


def _component_profile_memory_estimate(
    component: RemoteComponentPlan,
    prompt: Mapping[str, Any],
    preferences: WorkflowExecutionPreferences,
    resolved_profiles: Mapping[str, Any],
) -> ComponentMemoryEstimate:
    """Estimate a component's RAM and VRAM floors from resolved LLM profiles."""
    minimum_vram_bytes = preferences.minimum_vram_bytes
    profiles: dict[str, Any] = {}
    for node_id in component.node_ids:
        prompt_node = prompt.get(node_id)
        if not isinstance(prompt_node, Mapping):
            continue
        if str(prompt_node.get("class_type") or "") != "ModalLLM":
            continue
        inputs = prompt_node.get("inputs")
        if not isinstance(inputs, Mapping):
            continue
        model_profile = inputs.get("model_profile")
        if not isinstance(model_profile, str) or not model_profile.strip():
            continue
        profile = resolved_profiles.get(model_profile.strip())
        if profile is None:
            continue
        profiles[profile.profile_id] = profile
        minimum_vram_bytes = max(
            minimum_vram_bytes,
            int(max(0.0, profile.estimated_vram_gb) * 1024**3),
        )
    artifact_sizes = [int(profile.artifact_bytes) for profile in profiles.values()]
    return ComponentMemoryEstimate(
        minimum_vram_bytes=minimum_vram_bytes,
        minimum_ram_bytes=(sum(artifact_sizes) + _MODEL_RAM_HEADROOM_BYTES)
        if artifact_sizes
        else 0,
        model_asset_count=len(artifact_sizes),
        largest_model_bytes=max(artifact_sizes, default=0),
    )


def _component_required_provider(
    component: RemoteComponentPlan,
    prompt: Mapping[str, Any],
    resolved_profiles: Mapping[str, Any],
) -> ExecutionProvider | None:
    """Require SSH when a component uses an SSH-only resident backend."""
    for node_id in component.node_ids:
        prompt_node = prompt.get(node_id)
        if not isinstance(prompt_node, Mapping):
            continue
        if str(prompt_node.get("class_type") or "") != "ModalLLM":
            continue
        inputs = prompt_node.get("inputs")
        if not isinstance(inputs, Mapping):
            continue
        model_reference = inputs.get("model_profile")
        if not isinstance(model_reference, str):
            continue
        profile = resolved_profiles.get(model_reference.strip())
        if (
            profile is not None
            and getattr(profile, "backend", "") == "llama_cpp_server"
        ):
            return ExecutionProvider.SSH_DOCKER
    return None


def _iter_prompt_string_values(value: object) -> Iterator[str]:
    """Yield nested prompt string values while ignoring graph links and scalars."""
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _iter_prompt_string_values(child)
        return
    if isinstance(value, (list, tuple)) and not _is_link(value):
        for child in value:
            yield from _iter_prompt_string_values(child)


def _is_additive_model_node(class_type: str) -> bool:
    """Return whether a loader contributes weights alongside a primary model."""
    normalized = class_type.casefold()
    return any(token in normalized for token in _ADDITIVE_MODEL_CLASS_TOKENS)


def _component_model_asset_sizes(
    component: RemoteComponentPlan,
    prompt: Mapping[str, Any],
    settings: ModalSyncSettings,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return unique primary and additive model sizes referenced by a component."""
    asset_roles: dict[Path, bool] = {}
    for node_id in component.node_ids:
        prompt_node = prompt.get(node_id)
        if not isinstance(prompt_node, Mapping):
            continue
        inputs = prompt_node.get("inputs")
        if not isinstance(inputs, Mapping):
            continue
        additive = _is_additive_model_node(str(prompt_node.get("class_type") or ""))
        for value in _iter_prompt_string_values(inputs):
            resolved_path = resolve_model_path(
                value,
                comfyui_root=getattr(settings, "comfyui_root", None),
                extensions=MODEL_FILE_EXTENSIONS,
            )
            if resolved_path is None:
                continue
            asset_roles[resolved_path] = (
                asset_roles.get(resolved_path, True) and additive
            )

    primary_sizes: list[int] = []
    additive_sizes: list[int] = []
    for asset_path, additive in asset_roles.items():
        try:
            size_bytes = asset_path.stat().st_size
        except OSError as exc:
            logger.warning(
                "Unable to inspect model size for placement path=%s: %s",
                asset_path,
                exc,
            )
            continue
        if size_bytes <= 0:
            continue
        target = additive_sizes if additive else primary_sizes
        target.append(size_bytes)
    return tuple(primary_sizes), tuple(additive_sizes)


def _component_memory_estimate(
    component: RemoteComponentPlan,
    prompt: Mapping[str, Any],
    preferences: WorkflowExecutionPreferences,
    settings: ModalSyncSettings,
    resolved_llm_profiles: Mapping[str, Any] | None = None,
) -> ComponentMemoryEstimate:
    """Infer conservative RAM and VRAM floors from resident model weight sizes."""
    profile_estimate = _component_profile_memory_estimate(
        component,
        prompt,
        preferences,
        resolved_llm_profiles
        if resolved_llm_profiles is not None
        else _resolve_prompt_llm_profiles(prompt, settings),
    )
    primary_sizes, additive_sizes = _component_model_asset_sizes(
        component,
        prompt,
        settings,
    )
    all_sizes = primary_sizes + additive_sizes
    if not all_sizes:
        return profile_estimate

    primary_peak_bytes = max(primary_sizes or all_sizes)
    additive_bytes = sum(additive_sizes) if primary_sizes else 0
    resident_weight_bytes = primary_peak_bytes + additive_bytes
    model_vram_bytes = (
        math.ceil(resident_weight_bytes * _MODEL_VRAM_WEIGHT_MULTIPLIER)
        + _MODEL_VRAM_HEADROOM_BYTES
    )
    return ComponentMemoryEstimate(
        minimum_vram_bytes=max(
            profile_estimate.minimum_vram_bytes,
            model_vram_bytes,
        ),
        minimum_ram_bytes=max(
            profile_estimate.minimum_ram_bytes,
            resident_weight_bytes + _MODEL_RAM_HEADROOM_BYTES,
        ),
        model_asset_count=len(all_sizes) + profile_estimate.model_asset_count,
        largest_model_bytes=max(
            max(all_sizes),
            profile_estimate.largest_model_bytes,
        ),
    )


def _component_execution_signature(
    component: RemoteComponentPlan,
    prompt: Mapping[str, Any],
) -> str:
    """Return a stable workload signature for runtime-history estimates."""
    cost_shaping_input_names = frozenset(
        {
            "batch_size",
            "duration",
            "frames",
            "height",
            "max_new_tokens",
            "model_profile",
            "num_frames",
            "steps",
            "width",
        }
    )
    nodes: list[dict[str, Any]] = []
    for node_id in sorted(component.node_ids):
        prompt_node = prompt.get(node_id)
        if not isinstance(prompt_node, Mapping):
            continue
        inputs = prompt_node.get("inputs")
        input_mapping = inputs if isinstance(inputs, Mapping) else {}
        nodes.append(
            {
                "class_type": str(prompt_node.get("class_type") or ""),
                "inputs": {
                    name: input_mapping[name]
                    for name in sorted(cost_shaping_input_names & set(input_mapping))
                    if not _is_link(input_mapping[name])
                },
            }
        )
    serialized = json.dumps(nodes, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    return hashlib.sha256(serialized).hexdigest()


