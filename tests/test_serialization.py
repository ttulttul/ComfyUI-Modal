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


def test_split_mapped_value_accepts_python_lists(serialization_module: Any) -> None:
    """Mapped execution should split list inputs item-by-item without changing order."""
    items = serialization_module.split_mapped_value(["a", "b", "c"], "STRING")

    assert items == ["a", "b", "c"]


def test_split_and_join_tensor_batch_for_mapped_execution(serialization_module: Any) -> None:
    """Mapped execution should split and reassemble tensor batches on the leading dimension."""
    torch = pytest.importorskip("torch")
    batch = torch.arange(24, dtype=torch.float32).reshape(3, 2, 4)

    split_items = serialization_module.split_mapped_value(batch, "IMAGE")
    rejoined = serialization_module.join_mapped_values(split_items, "IMAGE", is_list=False)

    assert len(split_items) == 3
    assert all(item.shape[0] == 1 for item in split_items)
    assert torch.equal(rejoined, batch)


def test_split_and_join_latent_batch_for_mapped_execution(serialization_module: Any) -> None:
    """Mapped execution should split and reassemble ComfyUI LATENT dictionaries."""
    torch = pytest.importorskip("torch")
    latent = {
        "samples": torch.arange(48, dtype=torch.float32).reshape(3, 4, 2, 2),
        "batch_index": [0, 1, 2],
    }

    split_items = serialization_module.split_mapped_value(latent, "LATENT")
    rejoined = serialization_module.join_mapped_values(split_items, "LATENT", is_list=False)

    assert len(split_items) == 3
    assert all(item["samples"].shape[0] == 1 for item in split_items)
    assert torch.equal(rejoined["samples"], latent["samples"])
    assert rejoined["batch_index"] == [0, 1, 2]
