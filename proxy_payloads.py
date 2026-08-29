"""Proxy payload normalization, cache surfaces, and run-scoped registries."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import logging
import threading
from typing import Any

from comfy_api.latest import _io as io

if __package__:
    from .serialization import split_mapped_value
else:  # pragma: no cover - flat import inside the Modal container.
    from serialization import split_mapped_value

logger = logging.getLogger(__name__)

MODAL_PROMPT_ID_EXTRA_PNGINFO_KEY = "comfy_modal_prompt_id"
_PROXY_CACHE_CONTEXT_ID_KEY = "__comfy_modal_proxy_cache_context_id__"
_VOLATILE_PROXY_CACHE_KEYS = frozenset({
    "prompt_id", "remote_session", "clear_remote_session", "extra_data",
    "requires_volume_reload", "volume_reload_marker", "uploaded_volume_paths",
    "speculative_remote_prewarm_target",
})


@dataclass(frozen=True)
class _ProxyExecutionContext:
    """Run-scoped execution context used to rehydrate cache-friendly proxy payloads."""

    execution_payload: dict[str, Any]


@dataclass(frozen=True)
class _ModalMapWarmupContext:
    """Run-scoped warmup context used by one local Modal Map Input node."""

    execution_payload: dict[str, Any]
    mapped_io_type: str


_PROXY_EXECUTION_CONTEXTS_LOCK = threading.Lock()
_PROXY_EXECUTION_CONTEXTS: dict[str, _ProxyExecutionContext] = {}
_PROXY_EXECUTION_CONTEXTS_BY_PROMPT: OrderedDict[
    tuple[str, str], _ProxyExecutionContext
] = OrderedDict()
_PROXY_EXECUTION_CONTEXT_LIMIT = 2048
_MODAL_MAP_WARMUP_CONTEXTS_LOCK = threading.Lock()
_MODAL_MAP_WARMUP_CONTEXTS: dict[str, _ModalMapWarmupContext] = {}

def _unwrap_proxy_singleton(value: Any) -> Any:
    """Unwrap one value wrapped by ComfyUI for an INPUT_IS_LIST proxy."""
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def _pop_proxy_hidden_value(
    proxy_class: type[Any],
    kwargs: dict[str, Any],
    hidden_input: io.Hidden,
) -> Any:
    """Read a V3 class-clone hidden value with a legacy kwargs fallback."""
    legacy_value = kwargs.pop(hidden_input.name, None)
    hidden_holder = getattr(proxy_class, "hidden", None)
    hidden_value = getattr(hidden_holder, hidden_input.name, None)
    return legacy_value if hidden_value is None else hidden_value


def _normalize_proxy_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Convert ComfyUI INPUT_IS_LIST proxy kwargs back into ordinary runtime values."""
    return {
        str(input_name): _unwrap_proxy_singleton(input_value)
        for input_name, input_value in kwargs.items()
    }


def _normalize_proxy_payload(payload: Any) -> Mapping[str, Any]:
    """Convert ComfyUI INPUT_IS_LIST payload wrappers back into one payload mapping."""
    payload = _unwrap_proxy_singleton(payload)
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, Mapping):
        raise TypeError("original_node_data must be a mapping or JSON object.")
    return payload


def _normalize_scheduler_list_outputs(
    payload: Mapping[str, Any],
    outputs: Sequence[Any],
) -> tuple[Any, ...]:
    """Make remote output containers match the proxy's scheduler-list declarations."""
    normalized_outputs = list(outputs)
    boundary_outputs = payload.get("boundary_outputs")
    if not isinstance(boundary_outputs, Sequence) or isinstance(
        boundary_outputs,
        str | bytes | bytearray,
    ):
        return tuple(normalized_outputs)

    for output_index, boundary_output in enumerate(boundary_outputs):
        if output_index >= len(normalized_outputs):
            break
        if not isinstance(boundary_output, Mapping):
            continue
        if not bool(boundary_output.get("scheduler_is_list", False)):
            continue
        if isinstance(normalized_outputs[output_index], list):
            continue
        normalized_outputs[output_index] = [normalized_outputs[output_index]]
        logger.debug(
            "Wrapped singleton remote output %d for component=%s to satisfy its scheduler-list contract.",
            output_index,
            payload.get("component_id"),
        )

    return tuple(normalized_outputs)


def _normalize_prompt_id(value: Any) -> str | None:
    """Return one non-empty prompt id string when available."""
    if value is None:
        return None
    prompt_id = str(value).strip()
    return prompt_id or None


def _payload_is_local_cache_safe(payload: Mapping[str, Any]) -> bool:
    """Return whether one proxy payload can safely reuse local ComfyUI outputs across prompt runs."""
    split_proxy_payloads = payload.get("split_proxy_payloads")
    if isinstance(split_proxy_payloads, Mapping):
        return all(
            isinstance(nested_payload, Mapping) and _payload_is_local_cache_safe(nested_payload)
            for nested_payload in split_proxy_payloads.values()
        )
    if isinstance(split_proxy_payloads, Sequence) and not isinstance(split_proxy_payloads, (str, bytes, bytearray)):
        return all(
            isinstance(nested_payload, Mapping) and _payload_is_local_cache_safe(nested_payload)
            for nested_payload in split_proxy_payloads
        )

    for phase_name in ("static_phase", "mapped_phase"):
        phase_payload = payload.get(phase_name)
        if isinstance(phase_payload, Mapping) and not _payload_is_local_cache_safe(phase_payload):
            return False
    return True


def _sanitize_cache_surface_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Strip run-scoped fields from one proxy payload before exposing it to ComfyUI caching."""
    sanitized_payload = dict(payload)
    for field_name in _VOLATILE_PROXY_CACHE_KEYS:
        sanitized_payload.pop(field_name, None)

    split_proxy_payloads = sanitized_payload.get("split_proxy_payloads")
    if isinstance(split_proxy_payloads, Mapping):
        sanitized_payload["split_proxy_payloads"] = {
            str(phase_name): _sanitize_cache_surface_payload(dict(phase_payload))
            for phase_name, phase_payload in split_proxy_payloads.items()
            if isinstance(phase_payload, Mapping)
        }
    elif isinstance(split_proxy_payloads, Sequence) and not isinstance(
        split_proxy_payloads,
        (str, bytes, bytearray),
    ):
        sanitized_payload["split_proxy_payloads"] = [
            _sanitize_cache_surface_payload(dict(phase_payload))
            for phase_payload in split_proxy_payloads
            if isinstance(phase_payload, Mapping)
        ]

    for phase_name in ("static_phase", "mapped_phase"):
        phase_payload = sanitized_payload.get(phase_name)
        if isinstance(phase_payload, Mapping):
            sanitized_payload[phase_name] = _sanitize_cache_surface_payload(dict(phase_payload))
    return sanitized_payload


def register_cache_friendly_proxy_payload(
    node_id: str,
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Return the payload that should be embedded in the proxy node input for local cache reuse."""
    if not _payload_is_local_cache_safe(payload):
        return dict(payload)

    sanitized_payload = _sanitize_cache_surface_payload(payload)
    sanitized_payload[_PROXY_CACHE_CONTEXT_ID_KEY] = str(node_id)
    prompt_id = _normalize_prompt_id(payload.get("prompt_id"))
    context = _ProxyExecutionContext(execution_payload=dict(payload))
    with _PROXY_EXECUTION_CONTEXTS_LOCK:
        _PROXY_EXECUTION_CONTEXTS[str(node_id)] = context
        if prompt_id is not None:
            context_key = (prompt_id, str(node_id))
            _PROXY_EXECUTION_CONTEXTS_BY_PROMPT[context_key] = context
            _PROXY_EXECUTION_CONTEXTS_BY_PROMPT.move_to_end(context_key)
            while len(_PROXY_EXECUTION_CONTEXTS_BY_PROMPT) > _PROXY_EXECUTION_CONTEXT_LIMIT:
                _PROXY_EXECUTION_CONTEXTS_BY_PROMPT.popitem(last=False)
    logger.debug(
        "Registered cache-friendly Modal proxy payload for node_id=%s prompt_id=%s session_backed=%s.",
        node_id,
        prompt_id,
        payload.get("remote_session") is not None,
    )
    return sanitized_payload


def update_registered_proxy_payload_fields(
    node_id: str,
    embedded_payload: Mapping[str, Any],
    fields: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Update both the embedded cache surface and run-scoped execution payload."""
    with _PROXY_EXECUTION_CONTEXTS_LOCK:
        context = _PROXY_EXECUTION_CONTEXTS.get(str(node_id))
        execution_payload = dict(
            context.execution_payload if context is not None else embedded_payload
        )
    execution_payload.update(fields)
    return register_cache_friendly_proxy_payload(node_id, execution_payload)


def registered_proxy_execution_payload(
    node_id: str,
    embedded_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the full run-scoped payload registered for one proxy node."""
    with _PROXY_EXECUTION_CONTEXTS_LOCK:
        context = _PROXY_EXECUTION_CONTEXTS.get(str(node_id))
        return dict(
            context.execution_payload if context is not None else embedded_payload
        )


def register_modal_map_input_warmup_context(
    node_id: str,
    payload: Mapping[str, Any],
    mapped_io_type: str,
) -> None:
    """Register prompt-scoped warmup metadata for one local Modal Map Input node."""
    with _MODAL_MAP_WARMUP_CONTEXTS_LOCK:
        _MODAL_MAP_WARMUP_CONTEXTS[str(node_id)] = _ModalMapWarmupContext(
            execution_payload=dict(payload),
            mapped_io_type=str(mapped_io_type or "*"),
        )
    logger.debug(
        "Registered Modal Map Input warmup context for node_id=%s component_id=%s prompt_id=%s io_type=%s.",
        node_id,
        payload.get("component_id"),
        _normalize_prompt_id(payload.get("prompt_id")),
        mapped_io_type,
    )


def _boost_modal_map_input_warmup(
    unique_id: str | None,
    value: Any,
) -> None:
    """Best-effort exact warmup boost for one local Modal Map Input execution."""
    if unique_id is None:
        return

    with _MODAL_MAP_WARMUP_CONTEXTS_LOCK:
        context = _MODAL_MAP_WARMUP_CONTEXTS.get(str(unique_id))
    if context is None:
        return

    try:
        total_items = len(split_mapped_value(value, context.mapped_io_type))
    except (TypeError, ValueError) as exc:
        logger.debug(
            "Skipping Modal Map Input warmup boost for node_id=%s because the runtime value was not splittable as io_type=%s: %s",
            unique_id,
            context.mapped_io_type,
            exc,
        )
        return

    from .remote.modal_app import boost_mapped_component_warmup

    boost_mapped_component_warmup(
        payload=context.execution_payload,
        total_items=total_items,
        reason="modal_map_input_execute",
    )


def _rehydrate_proxy_payload(
    payload: Mapping[str, Any],
    *,
    unique_id: str | None,
    prompt_id: str | None = None,
) -> Mapping[str, Any]:
    """Restore any execution-scoped fields stripped from a cache-friendly proxy payload."""
    candidate_context_id = payload.get(_PROXY_CACHE_CONTEXT_ID_KEY)
    if candidate_context_id is None:
        return payload

    context_id = unique_id
    if context_id is None:
        normalized_context_id = str(candidate_context_id).strip()
        context_id = normalized_context_id or None
    if context_id is None:
        return payload

    normalized_prompt_id = _normalize_prompt_id(prompt_id)
    with _PROXY_EXECUTION_CONTEXTS_LOCK:
        if normalized_prompt_id is not None:
            context_key = (normalized_prompt_id, str(context_id))
            context = _PROXY_EXECUTION_CONTEXTS_BY_PROMPT.pop(context_key, None)
        else:
            context = _PROXY_EXECUTION_CONTEXTS.get(str(context_id))
    if context is None:
        if normalized_prompt_id is not None:
            logger.warning(
                "No prompt-scoped Modal proxy payload found for prompt_id=%s node_id=%s; using embedded cache surface.",
                normalized_prompt_id,
                context_id,
            )
        return payload

    return dict(context.execution_payload)


def _prompt_id_from_extra_pnginfo(extra_pnginfo: Any) -> str | None:
    """Read the queue prompt id carried through ComfyUI's hidden PNG metadata input."""
    extra_pnginfo = _unwrap_proxy_singleton(extra_pnginfo)
    if not isinstance(extra_pnginfo, Mapping):
        return None
    return _normalize_prompt_id(extra_pnginfo.get(MODAL_PROMPT_ID_EXTRA_PNGINFO_KEY))



