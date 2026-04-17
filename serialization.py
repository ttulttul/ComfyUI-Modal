"""Serialization helpers for Modal node execution payloads."""

from __future__ import annotations

import base64
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
    payload = {str(key): serialize_value(value) for key, value in inputs.items()}
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
