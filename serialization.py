"""Serialization helpers for Modal node execution payloads."""

from __future__ import annotations

import base64
import copy
import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any

logger = logging.getLogger(__name__)

_KIND_KEY = "__comfy_modal_kind__"
_VALUE_KEY = "value"
_TENSOR_KIND = "tensor"
_BYTES_KIND = "bytes"
_TUPLE_KIND = "tuple"
_BATCHABLE_TENSOR_IO_TYPES = frozenset({"IMAGE", "MASK", "NOISE", "SIGMAS"})


def _is_scalar(value: Any) -> bool:
    """Return whether a value is natively representable in JSON."""
    return value is None or isinstance(value, bool | int | float | str)


def _import_torch() -> Any:
    """Import torch lazily so the module stays importable in light environments."""
    import torch

    return torch


def _serialize_tensor(value: Any) -> dict[str, str]:
    """Serialize a torch.Tensor into a base64 safetensors payload."""
    from safetensors.torch import save

    torch = _import_torch()
    if not isinstance(value, torch.Tensor):
        raise TypeError("Expected a torch.Tensor payload.")

    tensor_bytes = save({_VALUE_KEY: value.detach().contiguous()})
    encoded = base64.b64encode(tensor_bytes).decode("ascii")
    return {_KIND_KEY: _TENSOR_KIND, "payload": encoded}


def _deserialize_tensor(payload: Mapping[str, Any]) -> Any:
    """Deserialize a base64 safetensors payload back into a tensor."""
    from safetensors.torch import load

    encoded = payload["payload"]
    tensor_map = load(base64.b64decode(encoded.encode("ascii")))
    return tensor_map[_VALUE_KEY]


def serialize_value(value: Any) -> Any:
    """Convert a Python value into a JSON-safe execution payload."""
    if _is_scalar(value):
        return value

    try:
        torch = _import_torch()
    except ModuleNotFoundError:
        torch = None

    if torch is not None and isinstance(value, torch.Tensor):
        return _serialize_tensor(value)

    if isinstance(value, bytes):
        return {
            _KIND_KEY: _BYTES_KIND,
            "payload": base64.b64encode(value).decode("ascii"),
        }

    if isinstance(value, tuple):
        return {
            _KIND_KEY: _TUPLE_KIND,
            "items": [serialize_value(item) for item in value],
        }

    if isinstance(value, list):
        return [serialize_value(item) for item in value]

    if isinstance(value, Mapping):
        return {str(key): serialize_value(item) for key, item in value.items()}

    raise TypeError(
        "ComfyUI-Modal can only serialize JSON-compatible values, bytes, "
        "and torch tensors. Unsupported value type: "
        f"{type(value)!r}"
    )


def serialize_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a mapping into a JSON-safe payload using the Modal transport rules."""
    return {str(key): serialize_value(item) for key, item in mapping.items()}


def deserialize_value(payload: Any) -> Any:
    """Reconstruct a serialized execution payload back into Python values."""
    if _is_scalar(payload):
        return payload

    if isinstance(payload, list):
        return [deserialize_value(item) for item in payload]

    if not isinstance(payload, Mapping):
        raise TypeError(f"Unsupported payload type: {type(payload)!r}")

    kind = payload.get(_KIND_KEY)
    if kind == _TENSOR_KIND:
        return _deserialize_tensor(payload)
    if kind == _BYTES_KIND:
        encoded = payload["payload"]
        return base64.b64decode(encoded.encode("ascii"))
    if kind == _TUPLE_KIND:
        return tuple(deserialize_value(item) for item in payload["items"])

    return {str(key): deserialize_value(value) for key, value in payload.items()}


def serialize_node_inputs(inputs: Mapping[str, Any]) -> bytes:
    """Serialize node keyword arguments into transport bytes."""
    payload = serialize_mapping(inputs)
    return json.dumps(payload, sort_keys=True).encode("utf-8")


def deserialize_node_inputs(payload: bytes | bytearray | str | Mapping[str, Any]) -> dict[str, Any]:
    """Deserialize node keyword arguments from transport bytes."""
    if isinstance(payload, Mapping):
        raw_payload = dict(payload)
    else:
        if isinstance(payload, (bytes, bytearray)):
            payload = payload.decode("utf-8")
        raw_payload = json.loads(payload)

    if not isinstance(raw_payload, Mapping):
        raise TypeError("Serialized node inputs must decode to a mapping.")
    return {str(key): deserialize_value(value) for key, value in raw_payload.items()}


def serialize_node_outputs(outputs: Sequence[Any]) -> bytes:
    """Serialize node outputs into transport bytes."""
    payload = [serialize_value(value) for value in outputs]
    return json.dumps(payload).encode("utf-8")


def coerce_serialized_node_outputs(outputs: bytes | bytearray | str | Sequence[Any] | Any) -> bytes:
    """Normalize raw or pre-serialized node outputs into transport bytes."""
    if isinstance(outputs, bytes):
        return outputs
    if isinstance(outputs, bytearray):
        return bytes(outputs)
    if isinstance(outputs, str):
        return outputs.encode("utf-8")
    if isinstance(outputs, (list, tuple)):
        return serialize_node_outputs(tuple(outputs))
    return serialize_node_outputs((outputs,))


def deserialize_node_outputs(payload: bytes | bytearray | str | Sequence[Any]) -> tuple[Any, ...]:
    """Deserialize node outputs from transport bytes."""
    if isinstance(payload, Sequence) and not isinstance(payload, (bytes, bytearray, str)):
        raw_payload = list(payload)
    else:
        if isinstance(payload, (bytes, bytearray)):
            payload = payload.decode("utf-8")
        raw_payload = json.loads(payload)

    if not isinstance(raw_payload, list):
        raise TypeError("Serialized node outputs must decode to a list.")
    return tuple(deserialize_value(value) for value in raw_payload)


def _split_tensor_batch(value: Any) -> list[Any]:
    """Split one tensor batch into per-item tensors that retain the batch dimension."""
    torch = _import_torch()
    if not isinstance(value, torch.Tensor):
        raise TypeError("Expected a tensor batch.")
    if value.ndim == 0 or value.shape[0] == 0:
        raise ValueError("Mapped tensor batches must have a non-zero leading batch dimension.")
    return [value[index : index + 1] for index in range(int(value.shape[0]))]


def _latent_batch_size(value: Mapping[str, Any]) -> int:
    """Return the batch size of a ComfyUI LATENT-like mapping."""
    torch = _import_torch()
    samples = value.get("samples")
    if not isinstance(samples, torch.Tensor) or samples.ndim == 0 or samples.shape[0] == 0:
        raise TypeError("Mapped LATENT values must contain a batched 'samples' tensor.")
    return int(samples.shape[0])


def _split_latent_batch(value: Mapping[str, Any]) -> list[Any]:
    """Split one ComfyUI LATENT mapping into per-item latent mappings."""
    torch = _import_torch()
    batch_size = _latent_batch_size(value)
    items: list[dict[str, Any]] = []
    for index in range(batch_size):
        item: dict[str, Any] = {}
        for key, entry in value.items():
            if isinstance(entry, torch.Tensor) and entry.ndim > 0 and entry.shape[0] == batch_size:
                item[str(key)] = entry[index : index + 1]
                continue
            if isinstance(entry, list) and len(entry) == batch_size:
                item[str(key)] = [entry[index]]
                continue
            item[str(key)] = copy.deepcopy(entry)
        items.append(item)
    return items


def split_mapped_value(value: Any, io_type: str) -> list[Any]:
    """Split one mapped input value into ordered per-item values."""
    if isinstance(value, list):
        if len(value) == 0:
            raise ValueError("Mapped list inputs must contain at least one item.")
        return list(value)

    normalized_io_type = str(io_type)
    if normalized_io_type == "LATENT" and isinstance(value, Mapping):
        return _split_latent_batch(value)

    try:
        torch = _import_torch()
    except ModuleNotFoundError as exc:
        raise TypeError(
            f"Mapped input type {normalized_io_type!r} requires torch to split batched values."
        ) from exc

    if isinstance(value, torch.Tensor) and (
        normalized_io_type in _BATCHABLE_TENSOR_IO_TYPES or value.ndim > 0
    ):
        return _split_tensor_batch(value)

    raise TypeError(
        "Mapped execution only supports Python lists, tensor batches, and LATENT dictionaries. "
        f"Unsupported mapped value type {type(value)!r} for io_type={normalized_io_type!r}."
    )


def _join_latent_batches(values: Sequence[Any]) -> Any:
    """Reassemble ordered per-item LATENT mappings into one batched latent mapping."""
    torch = _import_torch()
    if not values:
        raise ValueError("Expected at least one mapped LATENT output to aggregate.")
    if not all(isinstance(value, Mapping) for value in values):
        raise TypeError("Mapped LATENT outputs must all be mappings.")

    first_value = values[0]
    merged: dict[str, Any] = {}
    for key, first_entry in first_value.items():
        entries = [value[key] for value in values]
        if isinstance(first_entry, torch.Tensor):
            merged[str(key)] = torch.cat(entries, dim=0)
            continue
        if isinstance(first_entry, list):
            flattened: list[Any] = []
            for entry in entries:
                flattened.extend(entry)
            merged[str(key)] = flattened
            continue
        merged[str(key)] = copy.deepcopy(first_entry)
    return merged


def _join_mapped_values_as_list(values: Sequence[Any]) -> list[Any]:
    """Return mapped outputs as an ordered Python list."""
    return list(values)


def join_mapped_values(values: Sequence[Any], io_type: str, is_list: bool) -> Any:
    """Reassemble ordered per-item mapped outputs into one proxy output value."""
    if not values:
        raise ValueError("Mapped execution produced no outputs to aggregate.")

    if is_list:
        flattened: list[Any] = []
        for value in values:
            if isinstance(value, list):
                flattened.extend(value)
                continue
            flattened.append(value)
        return flattened

    normalized_io_type = str(io_type)
    if normalized_io_type == "LATENT":
        try:
            return _join_latent_batches(values)
        except RuntimeError as exc:
            logger.info(
                "Falling back to list aggregation for mapped LATENT outputs because batch concatenation "
                "failed: %s",
                exc,
            )
            return _join_mapped_values_as_list(values)

    try:
        torch = _import_torch()
    except ModuleNotFoundError:
        torch = None

    if torch is not None and all(isinstance(value, torch.Tensor) for value in values):
        try:
            return torch.cat(list(values), dim=0)
        except RuntimeError as exc:
            logger.info(
                "Falling back to list aggregation for mapped %s outputs because tensor concatenation "
                "failed: %s",
                normalized_io_type,
                exc,
            )
            return _join_mapped_values_as_list(values)

    return _join_mapped_values_as_list(values)
