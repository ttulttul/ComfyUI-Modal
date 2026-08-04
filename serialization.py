"""Serialization helpers for Modal node execution payloads."""

from __future__ import annotations

import base64
import copy
import json
import logging
import struct
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any, Callable

logger = logging.getLogger(__name__)

_KIND_KEY = "__comfy_modal_kind__"
_VALUE_KEY = "value"
_TENSOR_KIND = "tensor"
_BYTES_KIND = "bytes"
_TUPLE_KIND = "tuple"
_MAPPED_OUTPUT_KIND = "mapped_output"
_VIDEO_KIND = "video"
_VIDEO_PAYLOAD_VERSION = 1
_BATCHABLE_TENSOR_IO_TYPES = frozenset({"IMAGE", "MASK", "NOISE", "SIGMAS"})
_BINARY_ENVELOPE_MAGIC = b"CMODALB1"
_BINARY_ENVELOPE_VERSION = 1
_BINARY_HEADER_LENGTH_BYTES = 8
_MAX_BINARY_HEADER_BYTES = 16 * 1024 * 1024


class MappedOutputValue(list[Any]):
    """Ordered value produced by mapped execution and intended for per-item reuse."""

    def __init__(self, items: Sequence[Any], io_type: str, is_list: bool) -> None:
        """Initialize a list-like mapped output with scheduler metadata."""
        super().__init__(items)
        self.io_type = str(io_type)
        self.is_list = bool(is_list)

    @property
    def items(self) -> tuple[Any, ...]:
        """Return mapped items as an immutable tuple."""
        return tuple(self)


def unwrap_mapped_output_value(value: Any) -> Any:
    """Return the ordinary runtime value represented by a mapped output wrapper."""
    if isinstance(value, MappedOutputValue):
        if value.is_list:
            flattened: list[Any] = []
            for item in value.items:
                if isinstance(item, list):
                    flattened.extend(item)
                    continue
                flattened.append(item)
            return flattened
        return list(value.items)
    return value


def is_mapped_output_value(value: Any) -> bool:
    """Return whether a value carries mapped-output item metadata."""
    return isinstance(value, MappedOutputValue)


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


def _serialize_tensor_bytes(value: Any) -> bytes:
    """Serialize a tensor directly to safetensors bytes without base64 expansion."""
    from safetensors.torch import save

    torch = _import_torch()
    if not isinstance(value, torch.Tensor):
        raise TypeError("Expected a torch.Tensor payload.")
    return save({_VALUE_KEY: value.detach().contiguous()})


def _deserialize_tensor_bytes(payload: bytes) -> Any:
    """Deserialize direct safetensors bytes back into a tensor."""
    from safetensors.torch import load

    return load(payload)[_VALUE_KEY]


def _load_video_transport_types() -> tuple[type[Any], type[Any], type[Any]] | None:
    """Load current ComfyUI video protocol and implementation types when available."""
    try:
        from comfy_api.latest import Input, InputImpl, Types
    except (ImportError, AttributeError):
        return None

    video_input_type = getattr(Input, "Video", None)
    video_impl_type = getattr(InputImpl, "VideoFromComponents", None)
    video_components_type = getattr(Types, "VideoComponents", None)
    if not all(
        isinstance(candidate, type)
        for candidate in (video_input_type, video_impl_type, video_components_type)
    ):
        return None
    return video_input_type, video_impl_type, video_components_type


def _is_comfy_video_input(value: Any) -> bool:
    """Return whether a value implements the current ComfyUI VIDEO protocol."""
    video_types = _load_video_transport_types()
    return video_types is not None and isinstance(value, video_types[0])


def _serialize_video(
    value: Any,
    serialize_item: Callable[[Any], Any],
) -> dict[str, Any]:
    """Serialize a ComfyUI VIDEO as transportable tensor-backed components."""
    components = value.get_components()
    frame_rate = Fraction(components.frame_rate).limit_denominator(1_000_000)
    get_bit_depth = getattr(value, "get_bit_depth", None)
    bit_depth = int(get_bit_depth()) if callable(get_bit_depth) else 8
    return {
        _KIND_KEY: _VIDEO_KIND,
        "version": _VIDEO_PAYLOAD_VERSION,
        "images": serialize_item(components.images),
        "audio": serialize_item(getattr(components, "audio", None)),
        "alpha": serialize_item(getattr(components, "alpha", None)),
        "metadata": serialize_item(getattr(components, "metadata", None)),
        "frame_rate_numerator": frame_rate.numerator,
        "frame_rate_denominator": frame_rate.denominator,
        "bit_depth": bit_depth,
    }


def _video_payload_integer(payload: Mapping[str, Any], key: str) -> int:
    """Return one validated integer field from a serialized VIDEO payload."""
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Serialized VIDEO field '{key}' must be an integer.")
    return value


def _deserialize_video(
    payload: Mapping[str, Any],
    deserialize_item: Callable[[Any], Any],
) -> Any:
    """Reconstruct a current ComfyUI VIDEO from tensor-backed components."""
    if payload.get("version") != _VIDEO_PAYLOAD_VERSION:
        raise ValueError(f"Unsupported serialized VIDEO version {payload.get('version')!r}.")
    video_types = _load_video_transport_types()
    if video_types is None:
        raise TypeError(
            "Deserializing VIDEO values requires a ComfyUI runtime with the current video API."
        )

    _, video_impl_type, video_components_type = video_types
    numerator = _video_payload_integer(payload, "frame_rate_numerator")
    denominator = _video_payload_integer(payload, "frame_rate_denominator")
    if denominator <= 0:
        raise ValueError("Serialized VIDEO frame-rate denominator must be positive.")
    bit_depth = _video_payload_integer(payload, "bit_depth")
    components = video_components_type(
        images=deserialize_item(payload.get("images")),
        frame_rate=Fraction(numerator, denominator),
        audio=deserialize_item(payload.get("audio")),
        metadata=deserialize_item(payload.get("metadata")),
        alpha=deserialize_item(payload.get("alpha")),
    )
    return video_impl_type(components, bit_depth=bit_depth)


def serialize_value(value: Any) -> Any:
    """Convert a Python value into a JSON-safe execution payload."""
    if _is_scalar(value):
        return value

    if isinstance(value, MappedOutputValue):
        return {
            _KIND_KEY: _MAPPED_OUTPUT_KIND,
            "items": [serialize_value(item) for item in value.items],
            "io_type": value.io_type,
            "is_list": value.is_list,
        }

    try:
        torch = _import_torch()
    except ModuleNotFoundError:
        torch = None

    if torch is not None and isinstance(value, torch.Tensor):
        return _serialize_tensor(value)

    if _is_comfy_video_input(value):
        return _serialize_video(value, serialize_value)

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
        "torch tensors, and ComfyUI VIDEO values. Unsupported value type: "
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
    if kind == _MAPPED_OUTPUT_KIND:
        return MappedOutputValue(
            items=tuple(deserialize_value(item) for item in payload["items"]),
            io_type=str(payload.get("io_type", "*")),
            is_list=bool(payload.get("is_list", False)),
        )
    if kind == _VIDEO_KIND:
        return _deserialize_video(payload, deserialize_value)

    return {str(key): deserialize_value(value) for key, value in payload.items()}


def _serialize_transport_value(value: Any, attachments: list[bytes]) -> Any:
    """Convert one value into JSON metadata plus raw binary attachments."""
    if _is_scalar(value):
        return value

    if isinstance(value, MappedOutputValue):
        return {
            _KIND_KEY: _MAPPED_OUTPUT_KIND,
            "items": [_serialize_transport_value(item, attachments) for item in value.items],
            "io_type": value.io_type,
            "is_list": value.is_list,
        }

    try:
        torch = _import_torch()
    except ModuleNotFoundError:
        torch = None

    if torch is not None and isinstance(value, torch.Tensor):
        attachment_index = len(attachments)
        attachments.append(_serialize_tensor_bytes(value))
        return {_KIND_KEY: _TENSOR_KIND, "attachment": attachment_index}

    if _is_comfy_video_input(value):
        return _serialize_video(
            value,
            lambda item: _serialize_transport_value(item, attachments),
        )

    if isinstance(value, bytes):
        attachment_index = len(attachments)
        attachments.append(value)
        return {_KIND_KEY: _BYTES_KIND, "attachment": attachment_index}

    if isinstance(value, tuple):
        return {
            _KIND_KEY: _TUPLE_KIND,
            "items": [_serialize_transport_value(item, attachments) for item in value],
        }

    if isinstance(value, list):
        return [_serialize_transport_value(item, attachments) for item in value]

    if isinstance(value, Mapping):
        return {
            str(key): _serialize_transport_value(item, attachments)
            for key, item in value.items()
        }

    raise TypeError(
        "ComfyUI-Modal can only serialize JSON-compatible values, bytes, "
        "torch tensors, and ComfyUI VIDEO values. Unsupported value type: "
        f"{type(value)!r}"
    )


def _serialize_transport_payload(value: Any, *, sort_keys: bool) -> bytes:
    """Serialize one transport payload, using raw attachments when needed."""
    attachments: list[bytes] = []
    metadata = _serialize_transport_value(value, attachments)
    if not attachments:
        return json.dumps(metadata, sort_keys=sort_keys).encode("utf-8")

    header = json.dumps(
        {
            "version": _BINARY_ENVELOPE_VERSION,
            "payload": metadata,
            "attachment_lengths": [len(attachment) for attachment in attachments],
        },
        sort_keys=sort_keys,
        separators=(",", ":"),
    ).encode("utf-8")
    return b"".join(
        (
            _BINARY_ENVELOPE_MAGIC,
            struct.pack(">Q", len(header)),
            header,
            *attachments,
        )
    )


def _deserialize_transport_value(payload: Any, attachments: tuple[bytes, ...]) -> Any:
    """Reconstruct one value from metadata and binary attachments."""
    if _is_scalar(payload):
        return payload
    if isinstance(payload, list):
        return [_deserialize_transport_value(item, attachments) for item in payload]
    if not isinstance(payload, Mapping):
        raise TypeError(f"Unsupported payload type: {type(payload)!r}")

    kind = payload.get(_KIND_KEY)
    attachment_index = payload.get("attachment")
    if kind in {_TENSOR_KIND, _BYTES_KIND} and attachment_index is not None:
        if isinstance(attachment_index, bool) or not isinstance(attachment_index, int):
            raise ValueError("Binary attachment indices must be integers.")
        if attachment_index < 0 or attachment_index >= len(attachments):
            raise ValueError(f"Binary attachment index {attachment_index} is out of range.")
        attachment = attachments[attachment_index]
        if kind == _TENSOR_KIND:
            return _deserialize_tensor_bytes(attachment)
        return attachment
    if kind == _TENSOR_KIND:
        return _deserialize_tensor(payload)
    if kind == _BYTES_KIND:
        encoded = payload["payload"]
        return base64.b64decode(encoded.encode("ascii"))
    if kind == _TUPLE_KIND:
        return tuple(
            _deserialize_transport_value(item, attachments)
            for item in payload["items"]
        )
    if kind == _MAPPED_OUTPUT_KIND:
        return MappedOutputValue(
            items=tuple(
                _deserialize_transport_value(item, attachments)
                for item in payload["items"]
            ),
            io_type=str(payload.get("io_type", "*")),
            is_list=bool(payload.get("is_list", False)),
        )
    if kind == _VIDEO_KIND:
        return _deserialize_video(
            payload,
            lambda item: _deserialize_transport_value(item, attachments),
        )
    return {
        str(key): _deserialize_transport_value(value, attachments)
        for key, value in payload.items()
    }


def _deserialize_binary_envelope(payload: bytes) -> tuple[Any, tuple[bytes, ...]]:
    """Parse one validated binary transport envelope."""
    length_offset = len(_BINARY_ENVELOPE_MAGIC)
    header_offset = length_offset + _BINARY_HEADER_LENGTH_BYTES
    if len(payload) < header_offset:
        raise ValueError("Binary transport envelope is truncated before its header.")
    header_length = struct.unpack(">Q", payload[length_offset:header_offset])[0]
    if header_length > _MAX_BINARY_HEADER_BYTES:
        raise ValueError(
            f"Binary transport header exceeds {_MAX_BINARY_HEADER_BYTES} bytes."
        )
    attachment_offset = header_offset + header_length
    if attachment_offset > len(payload):
        raise ValueError("Binary transport envelope is truncated inside its header.")
    try:
        header = json.loads(payload[header_offset:attachment_offset].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Binary transport envelope header is invalid JSON.") from exc
    if not isinstance(header, Mapping):
        raise ValueError("Binary transport envelope header must be a mapping.")
    if header.get("version") != _BINARY_ENVELOPE_VERSION:
        raise ValueError(f"Unsupported binary transport version {header.get('version')!r}.")
    attachment_lengths = header.get("attachment_lengths")
    if not isinstance(attachment_lengths, list):
        raise ValueError("Binary transport envelope is missing attachment lengths.")

    attachments: list[bytes] = []
    next_offset = attachment_offset
    for attachment_length in attachment_lengths:
        if (
            isinstance(attachment_length, bool)
            or not isinstance(attachment_length, int)
            or attachment_length < 0
        ):
            raise ValueError("Binary attachment lengths must be non-negative integers.")
        next_attachment_offset = next_offset + attachment_length
        if next_attachment_offset > len(payload):
            raise ValueError("Binary transport envelope is truncated inside an attachment.")
        attachments.append(bytes(memoryview(payload)[next_offset:next_attachment_offset]))
        next_offset = next_attachment_offset
    if next_offset != len(payload):
        raise ValueError("Binary transport envelope contains trailing bytes.")
    return header.get("payload"), tuple(attachments)


def _deserialize_transport_payload(payload: bytes | bytearray | str) -> tuple[Any, tuple[bytes, ...]]:
    """Decode either the binary transport or the legacy JSON representation."""
    if isinstance(payload, str):
        return json.loads(payload), ()
    normalized_payload = bytes(payload)
    if normalized_payload.startswith(_BINARY_ENVELOPE_MAGIC):
        return _deserialize_binary_envelope(normalized_payload)
    return json.loads(normalized_payload.decode("utf-8")), ()


def serialize_node_inputs(inputs: Mapping[str, Any]) -> bytes:
    """Serialize node keyword arguments into transport bytes."""
    return _serialize_transport_payload(dict(inputs), sort_keys=True)


def deserialize_node_inputs(payload: bytes | bytearray | str | Mapping[str, Any]) -> dict[str, Any]:
    """Deserialize node keyword arguments from transport bytes."""
    if isinstance(payload, Mapping):
        raw_payload = dict(payload)
        attachments: tuple[bytes, ...] = ()
    else:
        raw_payload, attachments = _deserialize_transport_payload(payload)

    if not isinstance(raw_payload, Mapping):
        raise TypeError("Serialized node inputs must decode to a mapping.")
    return {
        str(key): _deserialize_transport_value(value, attachments)
        for key, value in raw_payload.items()
    }


def serialize_node_outputs(outputs: Sequence[Any]) -> bytes:
    """Serialize node outputs into transport bytes."""
    return _serialize_transport_payload(list(outputs), sort_keys=False)


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
        attachments: tuple[bytes, ...] = ()
    else:
        raw_payload, attachments = _deserialize_transport_payload(payload)

    if not isinstance(raw_payload, list):
        raise TypeError("Serialized node outputs must decode to a list.")
    return tuple(
        _deserialize_transport_value(value, attachments)
        for value in raw_payload
    )


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
    if isinstance(value, MappedOutputValue):
        return list(value.items)

    if isinstance(value, list):
        if len(value) == 0:
            raise ValueError("Mapped list inputs must contain at least one item.")
        return list(value)

    normalized_io_type = str(io_type)
    if _is_scalar(value):
        return [value]

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
        "Mapped execution only supports scalar values, Python lists, tensor batches, and LATENT dictionaries. "
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

    normalized_io_type = str(io_type)
    if normalized_io_type == "CONDITIONING":
        return MappedOutputValue(
            items=tuple(copy.deepcopy(value) for value in values),
            io_type=normalized_io_type,
            is_list=is_list,
        )

    if is_list:
        flattened: list[Any] = []
        for value in values:
            if isinstance(value, list):
                flattened.extend(value)
                continue
            flattened.append(value)
        return flattened

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
