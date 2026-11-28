"""Tests for Modal payload serialization helpers."""

from __future__ import annotations

import json
import tracemalloc
from fractions import Fraction
from types import SimpleNamespace
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


def test_nested_tensor_round_trip_preserves_multimodal_members(
    serialization_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LATENT NestedTensor members should survive binary and legacy transport."""
    torch = pytest.importorskip("torch")

    class FakeNestedTensor:
        """Minimal stand-in for ComfyUI's NestedTensor wrapper."""

        def __init__(self, tensors: list[Any]) -> None:
            """Store ordered tensor members."""
            self.tensors = list(tensors)

    monkeypatch.setattr(
        serialization_module,
        "_load_nested_tensor_transport_type",
        lambda: FakeNestedTensor,
    )
    video_samples = torch.arange(24, dtype=torch.float32).reshape(1, 3, 2, 4)
    audio_samples = torch.arange(12, dtype=torch.float32).reshape(1, 2, 6)
    latent = {
        "samples": FakeNestedTensor([video_samples, audio_samples]),
        "batch_index": [0],
    }

    encoded = serialization_module.serialize_node_outputs((latent,))
    decoded = serialization_module.deserialize_node_outputs(encoded)[0]

    assert encoded.startswith(serialization_module._BINARY_ENVELOPE_MAGIC)
    assert isinstance(decoded["samples"], FakeNestedTensor)
    assert len(decoded["samples"].tensors) == 2
    assert torch.equal(decoded["samples"].tensors[0], video_samples)
    assert torch.equal(decoded["samples"].tensors[1], audio_samples)

    legacy_decoded = serialization_module.deserialize_value(
        serialization_module.serialize_value(latent)
    )
    assert isinstance(legacy_decoded["samples"], FakeNestedTensor)
    assert torch.equal(legacy_decoded["samples"].tensors[0], video_samples)
    assert torch.equal(legacy_decoded["samples"].tensors[1], audio_samples)


def test_video_round_trip_preserves_tensor_backed_components(
    serialization_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VIDEO transport should preserve frames, audio, alpha, rate, metadata, and depth."""
    torch = pytest.importorskip("torch")

    class FakeVideo:
        """Minimal stand-in for ComfyUI's VideoInput protocol."""

        def __init__(self, components: Any, bit_depth: int = 8) -> None:
            """Store video components and encoding bit depth."""
            self._components = components
            self._bit_depth = bit_depth

        def get_components(self) -> Any:
            """Return the tensor-backed video components."""
            return self._components

        def get_bit_depth(self) -> int:
            """Return the preferred encoded bit depth."""
            return self._bit_depth

    class FakeVideoFromComponents(FakeVideo):
        """Stand in for ComfyUI's receiving-side VideoFromComponents implementation."""

    class FakeVideoComponents(SimpleNamespace):
        """Stand in for ComfyUI's VideoComponents dataclass."""

    monkeypatch.setattr(
        serialization_module,
        "_load_video_transport_types",
        lambda: (FakeVideo, FakeVideoFromComponents, FakeVideoComponents),
    )
    images = torch.arange(72, dtype=torch.float32).reshape(2, 3, 4, 3)
    waveform = torch.arange(16, dtype=torch.float32).reshape(1, 2, 8)
    alpha = torch.ones((2, 3, 4), dtype=torch.float32)
    video = FakeVideo(
        FakeVideoComponents(
            images=images,
            frame_rate=Fraction(30_000, 1_001),
            audio={"waveform": waveform, "sample_rate": 48_000},
            metadata={"source": "modal"},
            alpha=alpha,
        ),
        bit_depth=10,
    )

    encoded = serialization_module.serialize_node_outputs((video,))
    decoded = serialization_module.deserialize_node_outputs(encoded)[0]
    decoded_components = decoded.get_components()

    assert encoded.startswith(serialization_module._BINARY_ENVELOPE_MAGIC)
    assert isinstance(decoded, FakeVideoFromComponents)
    assert decoded.get_bit_depth() == 10
    assert decoded_components.frame_rate == Fraction(30_000, 1_001)
    assert decoded_components.audio["sample_rate"] == 48_000
    assert decoded_components.metadata == {"source": "modal"}
    assert torch.equal(decoded_components.images, images)
    assert torch.equal(decoded_components.audio["waveform"], waveform)
    assert torch.equal(decoded_components.alpha, alpha)

    legacy_decoded = serialization_module.deserialize_value(
        serialization_module.serialize_value(video)
    )
    assert legacy_decoded.get_components().frame_rate == Fraction(30_000, 1_001)
    assert torch.equal(legacy_decoded.get_components().images, images)


def test_tensor_transport_uses_raw_binary_attachments(serialization_module: Any) -> None:
    """Tensor transport should avoid base64's one-third payload expansion."""
    torch = pytest.importorskip("torch")
    tensor = torch.arange(256 * 1024, dtype=torch.float32)

    encoded = serialization_module.serialize_node_inputs({"tensor": tensor})
    legacy_json = json.dumps(
        serialization_module.serialize_mapping({"tensor": tensor}),
        sort_keys=True,
    ).encode("utf-8")

    assert encoded.startswith(serialization_module._BINARY_ENVELOPE_MAGIC)
    assert len(encoded) < len(legacy_json) * 0.8
    assert torch.equal(
        serialization_module.deserialize_node_inputs(encoded)["tensor"],
        tensor,
    )


def test_binary_transport_keeps_legacy_json_compatibility(serialization_module: Any) -> None:
    """New readers should continue to accept already-deployed JSON/base64 payloads."""
    torch = pytest.importorskip("torch")
    tensor = torch.arange(8, dtype=torch.float32)
    legacy_payload = json.dumps(
        serialization_module.serialize_mapping({"tensor": tensor, "bytes": b"abc"})
    ).encode("utf-8")

    decoded = serialization_module.deserialize_node_inputs(legacy_payload)

    assert torch.equal(decoded["tensor"], tensor)
    assert decoded["bytes"] == b"abc"


def test_binary_transport_rejects_truncated_attachments(serialization_module: Any) -> None:
    """A partial Modal payload should fail explicitly instead of decoding corrupted data."""
    encoded = serialization_module.serialize_node_inputs({"bytes": b"abcdef"})

    with pytest.raises(ValueError, match="truncated inside an attachment"):
        serialization_module.deserialize_node_inputs(encoded[:-1])


def _transport_test_tensors(torch: Any) -> list[Any]:
    """Return tensors covering dtypes, empty and scalar shapes, and strided layouts."""
    tensors = [
        torch.arange(24, dtype=torch.float32).reshape(2, 3, 4),
        torch.arange(24, dtype=torch.float32).reshape(4, 6).t(),
        torch.linspace(-2, 2, 10, dtype=torch.float16),
        torch.linspace(-2, 2, 10, dtype=torch.bfloat16),
        torch.arange(-5, 5, dtype=torch.int64),
        torch.arange(7, dtype=torch.uint8),
        torch.tensor([True, False, True]),
        torch.tensor(3.5, dtype=torch.float64),
        torch.empty((0, 3), dtype=torch.float32),
    ]
    if hasattr(torch, "float8_e4m3fn"):
        tensors.append(torch.linspace(-1, 1, 8).to(torch.float8_e4m3fn))
    return tensors


def test_tensor_attachment_matches_safetensors_save_bytes(serialization_module: Any) -> None:
    """Directly written tensor attachments must stay byte-identical to safetensors.save."""
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")

    for tensor in _transport_test_tensors(torch):
        expected = safetensors_torch.save({"value": tensor.contiguous()})
        assert serialization_module._serialize_tensor_bytes(tensor) == expected, tensor.dtype


def test_tensor_transport_round_trips_every_test_dtype(serialization_module: Any) -> None:
    """Every supported dtype and layout should decode to an equal, writable tensor."""
    torch = pytest.importorskip("torch")
    tensors = _transport_test_tensors(torch)

    decoded = serialization_module.deserialize_node_outputs(
        serialization_module.serialize_node_outputs(tuple(tensors))
    )

    for original, restored in zip(tensors, decoded, strict=True):
        assert restored.dtype == original.dtype
        assert restored.shape == original.shape
        assert torch.equal(
            restored.reshape(-1).view(torch.uint8),
            original.contiguous().reshape(-1).view(torch.uint8),
        )
    decoded[0].add_(1)


def test_tensor_transport_decodes_writers_that_used_safetensors_save(
    serialization_module: Any,
) -> None:
    """Envelopes produced by already-deployed workers must still decode."""
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    attachment = safetensors_torch.save({"value": tensor})
    header = json.dumps(
        {
            "version": 1,
            "payload": [{"__comfy_modal_kind__": "tensor", "attachment": 0}],
            "attachment_lengths": [len(attachment)],
        },
        separators=(",", ":"),
    ).encode("utf-8")
    legacy_envelope = b"".join(
        (b"CMODALB1", len(header).to_bytes(8, "big"), header, attachment)
    )

    decoded = serialization_module.deserialize_node_outputs(legacy_envelope)

    assert torch.equal(decoded[0], tensor)


def test_tensor_transport_accepts_bytearray_and_memoryview(serialization_module: Any) -> None:
    """Readers should decode binary envelopes from any contiguous byte buffer."""
    torch = pytest.importorskip("torch")
    tensor = torch.arange(6, dtype=torch.float32)
    encoded = serialization_module.serialize_node_inputs({"tensor": tensor, "raw": b"xy"})

    for payload in (bytearray(encoded), memoryview(encoded)):
        decoded = serialization_module.deserialize_node_inputs(payload)
        assert torch.equal(decoded["tensor"], tensor)
        assert decoded["raw"] == b"xy"
        assert isinstance(decoded["raw"], bytes)


def test_decoded_tensors_do_not_keep_the_payload_alive(serialization_module: Any) -> None:
    """Decoded tensors must own their storage so the transport payload can be freed."""
    torch = pytest.importorskip("torch")
    payload = bytearray(
        serialization_module.serialize_node_outputs((torch.zeros(16, dtype=torch.float32),))
    )

    decoded = serialization_module.deserialize_node_outputs(payload)[0]
    payload[:] = b"\xff" * len(payload)

    assert torch.equal(decoded, torch.zeros(16, dtype=torch.float32))


def test_tensor_transport_decodes_to_cpu_under_a_non_cpu_default_device(
    serialization_module: Any,
) -> None:
    """Decoding must not follow a default device set by another node, as safetensors.load did not."""
    torch = pytest.importorskip("torch")
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    encoded = serialization_module.serialize_node_outputs((tensor,))

    with torch.device("meta"):
        decoded = serialization_module.deserialize_node_outputs(encoded)[0]

    assert decoded.device.type == "cpu"
    assert torch.equal(decoded, tensor)


def test_tensor_attachment_rejects_mismatched_data_offsets(serialization_module: Any) -> None:
    """A tensor attachment whose offsets disagree with its shape must fail loudly."""
    torch = pytest.importorskip("torch")
    attachment = bytearray(
        serialization_module._serialize_tensor_bytes(torch.zeros(4, dtype=torch.float32))
    )
    corrupted = bytes(attachment).replace(b'"data_offsets":[0,16]', b'"data_offsets":[0,12]')

    with pytest.raises(ValueError, match="do not match its shape and dtype"):
        serialization_module._deserialize_tensor_bytes(corrupted)


def test_binary_transport_copies_tensor_data_once(serialization_module: Any) -> None:
    """Serializing a large tensor should allocate about one payload, not two or three."""
    torch = pytest.importorskip("torch")
    tensor = torch.zeros(8 * 1024 * 1024, dtype=torch.float32)
    serialization_module.serialize_node_inputs({"warmup": torch.zeros(1)})

    tracemalloc.start()
    try:
        encoded = serialization_module.serialize_node_inputs({"tensor": tensor})
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert len(encoded) > tensor.numel() * tensor.element_size()
    assert peak_bytes < len(encoded) * 1.1


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


def test_conditioning_with_reference_latents_round_trips(serialization_module: Any) -> None:
    """CONDITIONING payloads may include reference latent tensors in their metadata."""
    torch = pytest.importorskip("torch")
    conditioning = [
        [
            torch.arange(12, dtype=torch.float32).reshape(1, 3, 4),
            {
                "pooled_output": None,
                "attention_mask": torch.ones((1, 4), dtype=torch.int64),
                "reference_latents": [
                    torch.arange(16, dtype=torch.float16).reshape(1, 4, 2, 2),
                ],
            },
        ]
    ]

    encoded = serialization_module.serialize_node_inputs({"positive": conditioning})
    decoded = serialization_module.deserialize_node_inputs(encoded)

    decoded_conditioning = decoded["positive"]
    assert len(decoded_conditioning) == 1
    assert torch.equal(decoded_conditioning[0][0], conditioning[0][0])
    assert decoded_conditioning[0][1]["pooled_output"] is None
    assert torch.equal(
        decoded_conditioning[0][1]["attention_mask"],
        conditioning[0][1]["attention_mask"],
    )
    assert torch.equal(
        decoded_conditioning[0][1]["reference_latents"][0],
        conditioning[0][1]["reference_latents"][0],
    )


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


def test_split_mapped_value_treats_scalar_as_single_item(serialization_module: Any) -> None:
    """Mapped execution should treat scalar marker inputs as a one-item map."""
    items = serialization_module.split_mapped_value(7, "INT")

    assert items == [7]


def test_mapped_conditioning_output_round_trips_with_item_metadata(serialization_module: Any) -> None:
    """Mapped CONDITIONING outputs should preserve item boundaries across transport."""
    values = [
        [["cond-a", {"pooled_output": "pool-a"}]],
        [["cond-b", {"pooled_output": "pool-b"}]],
    ]
    mapped_value = serialization_module.join_mapped_values(values, "CONDITIONING", is_list=False)
    encoded = serialization_module.serialize_node_outputs((mapped_value,))
    decoded = serialization_module.deserialize_node_outputs(encoded)[0]

    assert isinstance(decoded, list)
    assert serialization_module.split_mapped_value(decoded, "CONDITIONING") == values
    assert serialization_module.unwrap_mapped_output_value(decoded) == values


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


def test_join_mapped_latents_falls_back_to_list_when_shapes_differ(
    serialization_module: Any,
) -> None:
    """Mapped LATENT outputs should stay ordered as a list when batch concatenation is impossible."""
    torch = pytest.importorskip("torch")
    latents = [
        {
            "samples": torch.zeros((1, 4, 32, 32), dtype=torch.float32),
            "batch_index": [0],
        },
        {
            "samples": torch.zeros((1, 4, 35, 35), dtype=torch.float32),
            "batch_index": [1],
        },
    ]

    rejoined = serialization_module.join_mapped_values(latents, "LATENT", is_list=False)

    assert isinstance(rejoined, list)
    assert len(rejoined) == 2
    assert rejoined[0]["samples"].shape == (1, 4, 32, 32)
    assert rejoined[1]["samples"].shape == (1, 4, 35, 35)


def _latent_bridge_refs(session_state_module: Any, count: int) -> list[dict[str, Any]]:
    """Return one distinct durable bridge ref per mapped item for LATENT node 8."""
    return [
        session_state_module.RemoteSessionBridgeRef(
            bridge_key=f"RSB_item_{index}",
            node_id="8",
            output_index=0,
            session_id="shared-session",
        ).to_payload()
        for index in range(count)
    ]


def test_join_mapped_latent_session_refs_keeps_every_item(
    serialization_module: Any,
    session_state_module: Any,
) -> None:
    """Per-item LATENT bridge refs must not collapse into the first item's ref."""
    refs = _latent_bridge_refs(session_state_module, 16)

    rejoined = serialization_module.join_mapped_values_for_scheduler(
        refs, "LATENT", is_list=False, scheduler_is_list=True
    )

    assert [ref["bridge_key"] for ref in rejoined] == [
        f"RSB_item_{index}" for index in range(16)
    ]


def test_join_mapped_latents_adapts_batches_to_scheduler_list_contract(
    serialization_module: Any,
) -> None:
    """Mapped LATENT proxies should expose one batch item or several heterogeneous items."""
    torch = pytest.importorskip("torch")
    compatible_latents = [
        {"samples": torch.zeros((1, 4, 32, 32), dtype=torch.float32)},
        {"samples": torch.ones((1, 4, 32, 32), dtype=torch.float32)},
    ]
    heterogeneous_latents = [
        compatible_latents[0],
        {"samples": torch.ones((1, 4, 35, 35), dtype=torch.float32)},
    ]

    compatible_result = serialization_module.join_mapped_values_for_scheduler(
        compatible_latents,
        "LATENT",
        is_list=False,
        scheduler_is_list=True,
    )
    heterogeneous_result = serialization_module.join_mapped_values_for_scheduler(
        heterogeneous_latents,
        "LATENT",
        is_list=False,
        scheduler_is_list=True,
    )

    assert len(compatible_result) == 1
    assert compatible_result[0]["samples"].shape == (2, 4, 32, 32)
    assert len(heterogeneous_result) == 2
    assert heterogeneous_result[0]["samples"].shape == (1, 4, 32, 32)
    assert heterogeneous_result[1]["samples"].shape == (1, 4, 35, 35)
