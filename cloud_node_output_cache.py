"""Persisted ComfyUI node-output cache keys, records, restore, and write-back."""

from __future__ import annotations

import asyncio
import base64
from contextlib import AbstractContextManager
from dataclasses import dataclass
import hashlib
import inspect
import json
import logging
import math
from typing import Any, Callable, Iterable, Mapping, Sequence
import zlib

try:
    from .cloud_comfy_bootstrap import _load_nodes_module
    from .cloud_runtime_context import node_output_cache_store
    from .remote_protocol import (
        BOUNDARY_INPUT_SIGNATURES_KEY as _BOUNDARY_INPUT_SIGNATURES_KEY,
    )
    from .serialization import (
        deserialize_node_outputs,
        deserialize_value,
        serialize_node_outputs,
        serialize_value,
    )
    from .settings import get_settings
except ImportError:  # pragma: no cover - flat Modal-container import.
    from cloud_comfy_bootstrap import _load_nodes_module
    from cloud_runtime_context import node_output_cache_store
    from remote_protocol import (
        BOUNDARY_INPUT_SIGNATURES_KEY as _BOUNDARY_INPUT_SIGNATURES_KEY,
    )
    from serialization import (
        deserialize_node_outputs,
        deserialize_value,
        serialize_node_outputs,
        serialize_value,
    )
    from settings import get_settings

logger = logging.getLogger(__name__)

_NODE_OUTPUT_CACHE_KEY_PREFIX = "NC_"
_NODE_OUTPUT_CACHE_RECORD_VERSION = 1


@dataclass(frozen=True)
class CloudNodeOutputCacheHooks:
    """Callbacks supplied by the stable cloud entrypoint."""

    emit_cloud_info: Callable[..., None]
    timed_phase: Callable[..., AbstractContextManager[None]]


_NODE_OUTPUT_CACHE_HOOKS: CloudNodeOutputCacheHooks | None = None


def configure_cloud_node_output_cache_hooks(
    hooks: CloudNodeOutputCacheHooks,
) -> None:
    """Install cloud logging callbacks without importing upward into the entrypoint."""
    global _NODE_OUTPUT_CACHE_HOOKS
    _NODE_OUTPUT_CACHE_HOOKS = hooks


def _node_output_cache_hooks() -> CloudNodeOutputCacheHooks:
    """Return configured callbacks or fail with a clear import-order error."""
    if _NODE_OUTPUT_CACHE_HOOKS is None:
        raise RuntimeError("Cloud node-output cache hooks have not been configured.")
    return _NODE_OUTPUT_CACHE_HOOKS


def _emit_cloud_info(message: str, *args: Any) -> None:
    """Delegate timestamped cloud logging to the stable entrypoint."""
    _node_output_cache_hooks().emit_cloud_info(message, *args)


def _timed_phase(phase: str, **fields: Any) -> AbstractContextManager[None]:
    """Delegate phase timing to the stable entrypoint."""
    return _node_output_cache_hooks().timed_phase(phase, **fields)


@dataclass
class _PersistedNodeCacheRestoreState:
    """Track distributed cache entries restored into one prompt execution."""

    restored_node_ids: list[str]
    restored_cache_keys_by_node_id: dict[str, str]
    restore_original_method: Callable[[], None]


@dataclass(frozen=True)
class _NodeOutputCacheLookupResult:
    """Hold one distributed cache lookup result before live-cache hydration."""

    node_id: str
    cache_key: str | None
    raw_record: Any | None
    cache_entry: Any | None


def _is_link(value: Any) -> bool:
    """Return whether a prompt input value is a ComfyUI link."""
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(not isinstance(item, dict) for item in value)
    )


def _normalize_link_output_index(value: Any) -> Any:
    """Unwrap a singleton list around a prompt-link output index when present."""
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return value


def _node_output_cache_store() -> Any | None:
    """Return the shared Modal Dict used for persisted transport-safe node outputs."""
    return node_output_cache_store()


def _node_output_cache_key_preview(
    cache_key: str | None, *, max_chars: int = 32
) -> str:
    """Return a short human-readable prefix of one persisted node-cache key."""
    if cache_key is None:
        return "<none>"
    return cache_key[:max_chars]


def _node_output_cache_value_preview(value: Any, *, max_chars: int = 160) -> str:
    """Return a truncated repr for node-cache debug logging."""
    try:
        rendered = repr(value)
    except Exception as exc:  # pragma: no cover - defensive logging path.
        rendered = f"<repr failed: {type(exc).__name__}: {exc}>"
    if len(rendered) <= max_chars:
        return rendered
    return f"{rendered[:max_chars]}..."


def _tensor_cache_key_digest(value: Any) -> dict[str, Any]:
    """Return a stable digest payload for one tensor used inside a cache key."""
    from safetensors.torch import save

    tensor = value.detach().contiguous().cpu()
    tensor_bytes = save({"value": tensor})
    return {
        "kind": "tensor",
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "sha256": hashlib.sha256(tensor_bytes).hexdigest(),
    }


async def _node_output_cache_store_get(cache_store: Any, cache_key: str) -> Any:
    """Return one persisted node-cache record, preferring Modal's async Dict interface."""
    aio_get = getattr(getattr(cache_store, "get", None), "aio", None)
    if callable(aio_get):
        return await aio_get(cache_key)
    return cache_store.get(cache_key)


async def _node_output_cache_store_put(
    cache_store: Any,
    cache_key: str,
    record: dict[str, Any],
) -> None:
    """Persist one node-cache record without blocking an active event loop."""
    aio_put = getattr(getattr(cache_store, "put", None), "aio", None)
    if callable(aio_put):
        await aio_put(cache_key, record)
        return
    cache_store[cache_key] = record


def _is_input_signature_cache_key_set(cache_key_set: Any) -> bool:
    """Return whether one cache-key set uses ComfyUI input-signature semantics."""
    return all(
        hasattr(cache_key_set, attribute)
        for attribute in (
            "dynprompt",
            "is_changed_cache",
            "get_ordered_ancestry",
            "include_node_id_in_input",
        )
    )


def _include_unique_id_in_input_signature(class_type: str) -> bool:
    """Return whether ComfyUI includes the unique node id in this input signature."""
    from comfy_execution.caching import include_unique_id_in_input

    return bool(include_unique_id_in_input(class_type))


def _build_node_output_cache_immediate_signature(
    cache_key_set: Any,
    *,
    dynprompt: Any,
    node_id: str,
    ancestor_order_mapping: dict[str, int],
    is_changed_value: Any,
) -> list[Any]:
    """Return one raw ComfyUI input-signature fragment before `to_hashable()` runs."""
    if not dynprompt.has_node(node_id):
        return [float("NaN")]

    node = dynprompt.get_node(node_id)
    class_type = node["class_type"]
    class_def = _load_nodes_module().NODE_CLASS_MAPPINGS[class_type]
    signature: list[Any] = [class_type, is_changed_value]
    if (
        cache_key_set.include_node_id_in_input()
        or (hasattr(class_def, "NOT_IDEMPOTENT") and class_def.NOT_IDEMPOTENT)
        or _include_unique_id_in_input_signature(class_type)
    ):
        signature.append(node_id)

    inputs = node["inputs"]
    boundary_input_signatures = node.get(_BOUNDARY_INPUT_SIGNATURES_KEY)
    for key in sorted(inputs.keys()):
        input_value = inputs[key]
        ancestor_socket = _cache_signature_link_output_index(input_value)
        ancestor_id = str(input_value[0]) if ancestor_socket is not None else None
        if ancestor_socket is not None and ancestor_id in ancestor_order_mapping:
            ancestor_index = int(ancestor_order_mapping[ancestor_id])
            signature.append((key, ("ANCESTOR", ancestor_index, ancestor_socket)))
        else:
            boundary_signature = None
            if isinstance(boundary_input_signatures, dict):
                candidate_signature = boundary_input_signatures.get(str(key))
                if candidate_signature is not None:
                    boundary_signature = candidate_signature
            if boundary_signature is not None:
                signature.append((key, ("BOUNDARY_SOURCE", boundary_signature)))
            else:
                signature.append((key, input_value))
    return signature


def _cache_signature_link_output_index(value: Any) -> int | None:
    """Return a prompt-link output index when `value` is safe to treat as graph wiring."""
    if not _is_link(value):
        return None

    output_index = _normalize_link_output_index(value[1])
    if isinstance(output_index, bool):
        return None

    try:
        return int(output_index)
    except (TypeError, ValueError):
        return None


async def _build_node_output_cache_signature_from_key_set_async(
    cache_key_set: Any,
    node_id: str,
) -> Any:
    """Return one distributed cache signature derived from a live ComfyUI cache-key set."""
    if not _is_input_signature_cache_key_set(cache_key_set):
        return cache_key_set.get_data_key(node_id)

    dynprompt = cache_key_set.dynprompt
    ancestors, order_mapping = cache_key_set.get_ordered_ancestry(dynprompt, node_id)
    signature = [
        _build_node_output_cache_immediate_signature(
            cache_key_set,
            dynprompt=dynprompt,
            node_id=node_id,
            ancestor_order_mapping=order_mapping,
            is_changed_value=await cache_key_set.is_changed_cache.get(node_id),
        )
    ]
    for ancestor_id in ancestors:
        signature.append(
            _build_node_output_cache_immediate_signature(
                cache_key_set,
                dynprompt=dynprompt,
                node_id=str(ancestor_id),
                ancestor_order_mapping=order_mapping,
                is_changed_value=await cache_key_set.is_changed_cache.get(ancestor_id),
            )
        )
    return signature


def _build_node_output_cache_signature_from_key_set_sync(
    cache_key_set: Any,
    node_id: str,
) -> Any | None:
    """Return one distributed cache signature using cached `is_changed` values only."""
    if not _is_input_signature_cache_key_set(cache_key_set):
        return cache_key_set.get_data_key(node_id)

    cached_is_changed = getattr(cache_key_set.is_changed_cache, "is_changed", None)
    if not isinstance(cached_is_changed, dict):
        return None

    dynprompt = cache_key_set.dynprompt
    ancestors, order_mapping = cache_key_set.get_ordered_ancestry(dynprompt, node_id)
    all_node_ids = [str(node_id), *[str(ancestor_id) for ancestor_id in ancestors]]
    missing_node_ids = [
        candidate for candidate in all_node_ids if candidate not in cached_is_changed
    ]
    if missing_node_ids:
        _emit_cloud_info(
            "Node output cache signature rebuild node=%s result=skip reason=missing-is-changed values=%s",
            node_id,
            missing_node_ids,
        )
        return None

    return [
        _build_node_output_cache_immediate_signature(
            cache_key_set,
            dynprompt=dynprompt,
            node_id=candidate,
            ancestor_order_mapping=order_mapping,
            is_changed_value=cached_is_changed[candidate],
        )
        for candidate in all_node_ids
    ]


def _canonicalize_node_output_cache_key_part(
    value: Any,
    *,
    path: str = "root",
) -> Any | None:
    """Return a JSON-stable representation of one CacheKeySetInputSignature fragment."""
    value_type_name = type(value).__name__
    if value_type_name == "Unhashable":
        _emit_cloud_info(
            "Node output cache canonicalization path=%s result=unhashable reason=comfy-unhashable-marker type=%s value=%s",
            path,
            value_type_name,
            _node_output_cache_value_preview(value),
        )
        return None
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"kind": "float", "value": "nan"}
        if math.isinf(value):
            return {"kind": "float", "value": "inf" if value > 0 else "-inf"}
        return value
    if isinstance(value, bytes):
        return {
            "kind": "bytes",
            "value": base64.b64encode(value).decode("ascii"),
        }
    try:
        import torch
    except ModuleNotFoundError:
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        return _tensor_cache_key_digest(value)
    if isinstance(value, tuple):
        items = []
        for index, item in enumerate(value):
            child_path = f"{path}[{index}]"
            canonical_item = _canonicalize_node_output_cache_key_part(
                item,
                path=child_path,
            )
            if canonical_item is None:
                _emit_cloud_info(
                    "Node output cache canonicalization path=%s result=unhashable reason=tuple-child child_path=%s parent_type=%s parent_value=%s",
                    path,
                    child_path,
                    value_type_name,
                    _node_output_cache_value_preview(value),
                )
                return None
            items.append(canonical_item)
        return {"kind": "tuple", "items": items}
    if isinstance(value, list):
        items = []
        for index, item in enumerate(value):
            child_path = f"{path}[{index}]"
            canonical_item = _canonicalize_node_output_cache_key_part(
                item,
                path=child_path,
            )
            if canonical_item is None:
                _emit_cloud_info(
                    "Node output cache canonicalization path=%s result=unhashable reason=list-child child_path=%s parent_type=%s parent_value=%s",
                    path,
                    child_path,
                    value_type_name,
                    _node_output_cache_value_preview(value),
                )
                return None
            items.append(canonical_item)
        return {"kind": "list", "items": items}
    if isinstance(value, dict):
        items: list[dict[str, Any]] = []
        for key in sorted(value):
            rendered_key = _node_output_cache_value_preview(key, max_chars=48)
            key_path = f"{path}.key[{rendered_key}]"
            value_path = f"{path}[{rendered_key}]"
            canonical_key = _canonicalize_node_output_cache_key_part(
                key,
                path=key_path,
            )
            canonical_value = _canonicalize_node_output_cache_key_part(
                value[key],
                path=value_path,
            )
            if canonical_key is None:
                _emit_cloud_info(
                    "Node output cache canonicalization path=%s result=unhashable reason=dict-key child_path=%s parent_type=%s parent_value=%s",
                    path,
                    key_path,
                    value_type_name,
                    _node_output_cache_value_preview(value),
                )
                return None
            if canonical_value is None:
                _emit_cloud_info(
                    "Node output cache canonicalization path=%s result=unhashable reason=dict-value key=%s child_path=%s parent_type=%s parent_value=%s",
                    path,
                    rendered_key,
                    value_path,
                    value_type_name,
                    _node_output_cache_value_preview(value),
                )
                return None
            items.append({"key": canonical_key, "value": canonical_value})
        return {"kind": "dict", "items": items}
    if isinstance(value, frozenset):
        canonical_items: list[Any] = []
        for index, item in enumerate(
            sorted(
                value,
                key=lambda item: _node_output_cache_value_preview(item, max_chars=120),
            )
        ):
            child_path = f"{path}{{{index}}}"
            canonical_item = _canonicalize_node_output_cache_key_part(
                item,
                path=child_path,
            )
            if canonical_item is None:
                _emit_cloud_info(
                    "Node output cache canonicalization path=%s result=unhashable reason=frozenset-child child_path=%s parent_type=%s parent_value=%s",
                    path,
                    child_path,
                    value_type_name,
                    _node_output_cache_value_preview(value),
                )
                return None
            canonical_items.append(canonical_item)
        canonical_items.sort(
            key=lambda item: json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return {"kind": "frozenset", "items": canonical_items}
    _emit_cloud_info(
        "Node output cache canonicalization path=%s result=unhashable reason=unsupported-type type=%s value=%s",
        path,
        value_type_name,
        _node_output_cache_value_preview(value),
    )
    return None


def _node_output_cache_key(signature: Any) -> str | None:
    """Return the persisted Modal Dict key for one ComfyUI cache signature."""
    canonical_signature = _canonicalize_node_output_cache_key_part(signature)
    if canonical_signature is None:
        return None
    signature_payload = json.dumps(
        canonical_signature,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    signature_digest = hashlib.sha256(signature_payload).hexdigest()
    return f"{_NODE_OUTPUT_CACHE_KEY_PREFIX}{signature_digest}"


async def _node_output_cache_key_from_key_set_async(
    cache_key_set: Any,
    node_id: str,
) -> str | None:
    """Return the persisted Modal Dict key for one live ComfyUI cache-key-set node."""
    return _node_output_cache_key(
        await _build_node_output_cache_signature_from_key_set_async(
            cache_key_set,
            node_id,
        )
    )


def _node_output_cache_key_from_key_set_sync(
    cache_key_set: Any,
    node_id: str,
) -> str | None:
    """Return the persisted Modal Dict key for one executed ComfyUI cache-key-set node."""
    signature = _build_node_output_cache_signature_from_key_set_sync(
        cache_key_set, node_id
    )
    if signature is None:
        return None
    return _node_output_cache_key(signature)


def _estimate_node_output_cache_value_size_bytes(
    value: Any,
    *,
    byte_limit: int,
) -> int | None:
    """Return a best-effort raw-size estimate for one transport-safe value."""
    if byte_limit < 0:
        return None
    if value is None or isinstance(value, bool):
        return 1
    if isinstance(value, int):
        return 8
    if isinstance(value, float):
        return 8
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    if isinstance(value, bytes):
        return len(value)

    try:
        import torch
    except ModuleNotFoundError:
        torch = None

    if torch is not None and isinstance(value, torch.Tensor):
        return int(value.numel()) * int(value.element_size())

    if isinstance(value, tuple | list):
        total_size = 0
        for item in value:
            item_size = _estimate_node_output_cache_value_size_bytes(
                item, byte_limit=byte_limit
            )
            if item_size is None:
                return None
            total_size += item_size
            if total_size > byte_limit:
                return total_size
        return total_size

    if isinstance(value, dict):
        total_size = 0
        for key, item in value.items():
            total_size += len(str(key).encode("utf-8"))
            if total_size > byte_limit:
                return total_size
            item_size = _estimate_node_output_cache_value_size_bytes(
                item, byte_limit=byte_limit
            )
            if item_size is None:
                return None
            total_size += item_size
            if total_size > byte_limit:
                return total_size
        return total_size

    return None


def _serialize_node_output_cache_entry(
    cache_entry: Any,
    *,
    max_bytes: int,
) -> dict[str, Any] | None:
    """Return a persisted node-cache record when the outputs are safe and small enough."""
    if max_bytes <= 0:
        return None

    outputs_size = _estimate_node_output_cache_value_size_bytes(
        list(getattr(cache_entry, "outputs", [])),
        byte_limit=max_bytes,
    )
    if outputs_size is None or outputs_size > max_bytes:
        return None

    try:
        serialized_outputs = serialize_node_outputs(
            tuple(getattr(cache_entry, "outputs", []))
        )
    except TypeError:
        return None

    ui_payload: Any | None = None
    ui_value = getattr(cache_entry, "ui", None)
    if ui_value is not None:
        ui_size = _estimate_node_output_cache_value_size_bytes(
            ui_value, byte_limit=max_bytes
        )
        if ui_size is not None and ui_size <= max_bytes:
            try:
                ui_payload = serialize_value(ui_value)
            except TypeError:
                ui_payload = None

    return {
        "version": _NODE_OUTPUT_CACHE_RECORD_VERSION,
        "outputs_zlib": zlib.compress(serialized_outputs),
        "outputs_size_bytes": outputs_size,
        "ui": ui_payload,
    }


def _deserialize_node_output_cache_entry(
    execution: Any,
    record: Any,
) -> Any | None:
    """Return a ComfyUI CacheEntry reconstructed from one persisted Modal Dict record."""
    if not isinstance(record, dict):
        return None
    if int(record.get("version", -1)) != _NODE_OUTPUT_CACHE_RECORD_VERSION:
        return None
    compressed_outputs = record.get("outputs_zlib")
    if not isinstance(compressed_outputs, (bytes, bytearray)):
        return None

    try:
        outputs = list(
            deserialize_node_outputs(zlib.decompress(bytes(compressed_outputs)))
        )
    except (TypeError, ValueError, zlib.error):
        return None

    ui_payload = record.get("ui")
    try:
        ui_value = deserialize_value(ui_payload) if ui_payload is not None else None
    except TypeError:
        ui_value = None
    return execution.CacheEntry(ui=ui_value, outputs=outputs)


async def _await_maybe(value: Any) -> Any:
    """Await an asynchronous compatibility result or return a synchronous value."""
    if inspect.isawaitable(value):
        return await value
    return value


def _prompt_executor_cache_get_sync(outputs_cache: Any, node_id: str) -> Any | None:
    """Read one prepared cache entry without leaking a coroutine into synchronous code."""
    local_getter = getattr(outputs_cache, "get_local", None)
    if callable(local_getter):
        return local_getter(node_id)

    cache_entry = outputs_cache.get(node_id)
    if not inspect.isawaitable(cache_entry):
        return cache_entry
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(cache_entry)
    if inspect.iscoroutine(cache_entry):
        cache_entry.close()
    raise RuntimeError(
        "ComfyUI exposed an asynchronous outputs cache without the synchronous get_local API."
    )


async def _restore_persisted_node_output_cache_entries(
    execution: Any,
    executor: Any,
    *,
    prompt_id: str,
    prompt: dict[str, Any],
    cache_store: Any,
    required_materialized_node_ids: Iterable[str] | None = None,
    restored_cache_keys_by_node_id: dict[str, str] | None = None,
) -> list[str]:
    """Hydrate PromptExecutor output-cache misses from the shared Modal Dict."""
    outputs_cache = executor.caches.outputs
    dynamic_prompt = execution.DynamicPrompt(prompt)
    is_changed_cache = execution.IsChangedCache(
        prompt_id, dynamic_prompt, outputs_cache
    )
    await _await_maybe(
        outputs_cache.set_prompt(dynamic_prompt, prompt.keys(), is_changed_cache)
    )
    outputs_cache.clean_unused()

    return await _restore_persisted_node_output_cache_entries_into_prepared_cache(
        execution,
        outputs_cache,
        prompt=prompt,
        cache_store=cache_store,
        required_materialized_node_ids=required_materialized_node_ids,
        restored_cache_keys_by_node_id=restored_cache_keys_by_node_id,
    )


def _node_output_cache_ancestor_ids(cache_key_set: Any, node_id: str) -> set[str]:
    """Return the prompt ancestor ids used by one ComfyUI input-signature cache key."""
    if not _is_input_signature_cache_key_set(cache_key_set):
        return set()
    dynprompt = cache_key_set.dynprompt
    if not dynprompt.has_node(node_id):
        return set()
    ancestors, _order_mapping = cache_key_set.get_ordered_ancestry(dynprompt, node_id)
    return {str(ancestor_id) for ancestor_id in ancestors}


async def _restore_persisted_node_output_cache_entries_into_prepared_cache(
    execution: Any,
    outputs_cache: Any,
    *,
    prompt: dict[str, Any],
    cache_store: Any,
    required_materialized_node_ids: Iterable[str] | None = None,
    restored_cache_keys_by_node_id: dict[str, str] | None = None,
) -> list[str]:
    """Hydrate one already-prepared PromptExecutor outputs cache from the shared Modal Dict."""
    restored_node_ids: list[str] = []
    pending_lookup_tasks: list[asyncio.Task[_NodeOutputCacheLookupResult]] = []
    required_node_ids = {
        str(node_id)
        for node_id in (required_materialized_node_ids or [])
        if str(node_id) in prompt
    }
    local_node_ids: set[str] = set()

    async def lookup_node(node_id: str) -> _NodeOutputCacheLookupResult:
        """Resolve one distributed cache candidate without mutating the live outputs cache."""
        cache_key = await _node_output_cache_key_from_key_set_async(
            outputs_cache.cache_key_set, node_id
        )
        if cache_key is None:
            return _NodeOutputCacheLookupResult(
                node_id=node_id,
                cache_key=None,
                raw_record=None,
                cache_entry=None,
            )
        raw_record = await _node_output_cache_store_get(cache_store, cache_key)
        return _NodeOutputCacheLookupResult(
            node_id=node_id,
            cache_key=cache_key,
            raw_record=raw_record,
            cache_entry=_deserialize_node_output_cache_entry(execution, raw_record),
        )

    for node_id in prompt:
        if await _await_maybe(outputs_cache.get(node_id)) is not None:
            _emit_cloud_info(
                "Node output cache lookup node=%s result=local-hit",
                node_id,
            )
            local_node_ids.add(str(node_id))
            continue
        pending_lookup_tasks.append(asyncio.create_task(lookup_node(str(node_id))))

    if not pending_lookup_tasks:
        return restored_node_ids

    lookup_results = await asyncio.gather(*pending_lookup_tasks)
    restorable_node_ids = {
        lookup_result.node_id
        for lookup_result in lookup_results
        if lookup_result.cache_key is not None and lookup_result.cache_entry is not None
    }
    available_node_ids = local_node_ids | restorable_node_ids

    for lookup_result in lookup_results:
        node_id = lookup_result.node_id
        cache_key = lookup_result.cache_key
        if cache_key is None:
            _emit_cloud_info(
                "Node output cache lookup node=%s key_prefix=%s result=skip reason=key-unhashable",
                node_id,
                _node_output_cache_key_preview(cache_key),
            )
            continue
        raw_record = lookup_result.raw_record
        cache_entry = lookup_result.cache_entry
        if cache_entry is None:
            result = "miss"
            if raw_record is not None:
                result = "miss-invalid"
            _emit_cloud_info(
                "Node output cache lookup node=%s key_prefix=%s result=%s",
                node_id,
                _node_output_cache_key_preview(cache_key),
                result,
            )
            continue
        missing_required_ancestors = sorted(
            required_node_ids
            & _node_output_cache_ancestor_ids(outputs_cache.cache_key_set, node_id)
            - available_node_ids
        )
        if missing_required_ancestors:
            _emit_cloud_info(
                "Node output cache lookup node=%s key_prefix=%s result=skip reason=missing-required-boundary-ancestors ancestors=%s",
                node_id,
                _node_output_cache_key_preview(cache_key),
                missing_required_ancestors,
            )
            continue
        await _await_maybe(outputs_cache.set(node_id, cache_entry))
        _emit_cloud_info(
            "Node output cache lookup node=%s key_prefix=%s result=hit",
            node_id,
            _node_output_cache_key_preview(cache_key),
        )
        if restored_cache_keys_by_node_id is not None:
            restored_cache_keys_by_node_id[str(node_id)] = cache_key
        restored_node_ids.append(str(node_id))
    return restored_node_ids


def _install_prompt_executor_persisted_cache_restore(
    execution: Any,
    executor: Any,
    *,
    component_id: str,
    prompt: dict[str, Any],
    cache_store: Any,
    required_materialized_node_ids: Iterable[str] | None = None,
) -> _PersistedNodeCacheRestoreState:
    """Patch one executor so persisted-cache restore runs after its live `set_prompt()` call."""
    restored_node_ids: list[str] = []
    restored_cache_keys_by_node_id: dict[str, str] = {}
    outputs_cache = executor.caches.outputs
    original_set_prompt = outputs_cache.set_prompt

    async def wrapped_set_prompt(
        dynprompt: Any, node_ids: Any, is_changed_cache: Any
    ) -> None:
        await original_set_prompt(dynprompt, node_ids, is_changed_cache)
        with _timed_phase("restore_persisted_node_cache", component=component_id):
            restored_cache_keys_by_node_id.clear()
            restored_node_ids[
                :
            ] = await _restore_persisted_node_output_cache_entries_into_prepared_cache(
                execution,
                outputs_cache,
                prompt=prompt,
                cache_store=cache_store,
                required_materialized_node_ids=required_materialized_node_ids,
                restored_cache_keys_by_node_id=restored_cache_keys_by_node_id,
            )

    outputs_cache.set_prompt = wrapped_set_prompt

    def restore_original_method() -> None:
        outputs_cache.set_prompt = original_set_prompt

    return _PersistedNodeCacheRestoreState(
        restored_node_ids=restored_node_ids,
        restored_cache_keys_by_node_id=restored_cache_keys_by_node_id,
        restore_original_method=restore_original_method,
    )


def _emit_restored_node_cache_events(
    status_callback: Callable[[dict[str, Any]], None] | None,
    restored_node_ids: Sequence[str],
) -> None:
    """Publish one status event per node whose outputs were restored from the persisted cache."""
    if status_callback is None:
        return

    for node_id in restored_node_ids:
        safe_node_id = str(node_id)
        status_callback(
            {
                "event_type": "node_cached",
                "node_id": safe_node_id,
                "display_node_id": safe_node_id,
                "real_node_id": safe_node_id,
            }
        )


def _boundary_output_node_ids(
    boundary_outputs: Iterable[Mapping[str, Any]]
) -> set[str]:
    """Return node ids whose local cache entries must exist for boundary collection."""
    return {
        str(boundary_output.get("node_id"))
        for boundary_output in boundary_outputs
        if boundary_output.get("node_id") is not None
    }


async def _persist_node_output_cache_entries(
    executor: Any,
    *,
    prompt: dict[str, Any],
    cache_store: Any,
    restored_cache_keys_by_node_id: dict[str, str] | None = None,
) -> list[str]:
    """Persist eligible PromptExecutor cache entries into the shared Modal Dict."""
    max_bytes = get_settings().node_output_cache_max_bytes
    if max_bytes <= 0:
        return []

    outputs_cache = executor.caches.outputs
    cache_key_set = getattr(outputs_cache, "cache_key_set", None)
    if cache_key_set is None:
        return []

    persisted_node_ids: list[str] = []
    for node_id in prompt:
        cache_entry = await _await_maybe(outputs_cache.get(node_id))
        if cache_entry is None:
            _emit_cloud_info(
                "Node output cache write node=%s result=skip reason=no-local-cache-entry",
                node_id,
            )
            continue
        cache_key = _node_output_cache_key_from_key_set_sync(
            cache_key_set, str(node_id)
        )
        if cache_key is None:
            _emit_cloud_info(
                "Node output cache write node=%s key_prefix=%s result=skip reason=key-unhashable",
                node_id,
                _node_output_cache_key_preview(cache_key),
            )
            continue
        restored_cache_key = None
        if restored_cache_keys_by_node_id is not None:
            restored_cache_key = restored_cache_keys_by_node_id.get(str(node_id))
        if restored_cache_key == cache_key:
            _emit_cloud_info(
                "Node output cache write node=%s key_prefix=%s result=skip reason=restored-hit",
                node_id,
                _node_output_cache_key_preview(cache_key),
            )
            continue
        record = _serialize_node_output_cache_entry(cache_entry, max_bytes=max_bytes)
        if record is None:
            _emit_cloud_info(
                "Node output cache write node=%s key_prefix=%s result=skip reason=ineligible-or-oversize",
                node_id,
                _node_output_cache_key_preview(cache_key),
            )
            continue
        await _node_output_cache_store_put(cache_store, cache_key, record)
        _emit_cloud_info(
            "Node output cache write node=%s key_prefix=%s result=write outputs_size_bytes=%s",
            node_id,
            _node_output_cache_key_preview(cache_key),
            record.get("outputs_size_bytes"),
        )
        persisted_node_ids.append(str(node_id))
    return persisted_node_ids
