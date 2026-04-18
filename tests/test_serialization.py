"""Tests for Modal payload serialization helpers."""

from __future__ import annotations

from typing import Any

import pytest


def test_nested_payload_round_trip(serialization_module: Any) -> None:
    """Nested JSON-safe values should survive a full round trip."""
    payload = {
        "integer": 7,
        "text": "hello",
        "items": [1, True, None, {"nested": ("a", "b")}],
        "raw_bytes": b"abc",
    }

    encoded = serialization_module.serialize_node_inputs(payload)
    decoded = serialization_module.deserialize_node_inputs(encoded)

    assert decoded["integer"] == 7
    assert decoded["text"] == "hello"
    assert decoded["items"][3]["nested"] == ("a", "b")
    assert decoded["raw_bytes"] == b"abc"


def test_tensor_round_trip(serialization_module: Any) -> None:
    """Torch tensors should round-trip through the safetensors transport."""
    torch = pytest.importorskip("torch")
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    encoded = serialization_module.serialize_node_outputs((tensor,))
    decoded = serialization_module.deserialize_node_outputs(encoded)

    assert len(decoded) == 1
    assert torch.equal(decoded[0], tensor)


def test_serialize_mapping_supports_nested_tensors(serialization_module: Any) -> None:
    """Transport mapping helpers should encode tensor values safely."""
    torch = pytest.importorskip("torch")
    tensor = torch.arange(4, dtype=torch.float32).reshape(2, 2)

    payload = serialization_module.serialize_mapping(
        {
            "phase": "executing",
            "preview": tensor,
        }
    )

    assert payload["phase"] == "executing"
    decoded_tensor = serialization_module.deserialize_value(payload["preview"])
    assert torch.equal(decoded_tensor, tensor)


def test_coerce_serialized_node_outputs_accepts_raw_tensor_outputs(serialization_module: Any) -> None:
    """Raw node outputs should be normalized into transport bytes before crossing the wire."""
    torch = pytest.importorskip("torch")
    tensor = torch.arange(3, dtype=torch.float32)

    payload = serialization_module.coerce_serialized_node_outputs((tensor,))
    decoded = serialization_module.deserialize_node_outputs(payload)

    assert len(decoded) == 1
    assert torch.equal(decoded[0], tensor)
