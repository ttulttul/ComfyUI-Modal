"""Tests for progressive remote previews of images, masks, audio, and text."""

from __future__ import annotations

import asyncio
import importlib
import types
from pathlib import Path
from typing import Any

import pytest

from conftest import PACKAGE_NAME
from modal_executor_test_support import _FakeRewritePreviewImageNode
from modal_executor_test_support import _FakeRewriteRemoteImageNode
from modal_executor_test_support import _FakeRewriteRemoteTextNode
from modal_executor_test_support import _FakeRewriteLocalSinkNode


@pytest.fixture(scope="session")
def progressive_preview_module(extension_package: object) -> Any:
    """Return the shared progressive preview state module."""
    return importlib.import_module(f"{PACKAGE_NAME}.progressive_preview")


@pytest.fixture(scope="session")
def progressive_preview_node_module(extension_package: object) -> Any:
    """Return the RemoteProgressivePreview node module."""
    return importlib.import_module(f"{PACKAGE_NAME}.progressive_preview_node")


@pytest.fixture(autouse=True)
def _clear_progressive_registry(progressive_preview_module: Any) -> None:
    """Start every test with an empty global registry."""
    progressive_preview_module.PROGRESSIVE_PREVIEW_REGISTRY.clear()


def _image_ref(progressive_preview_module: Any, name: str) -> Any:
    """Return one temp preview image entry."""
    return progressive_preview_module.PreviewEntry(
        kind="image", filename=name, subfolder="", type="temp"
    )


def _bridge_ref(item: int) -> dict[str, Any]:
    """Return a minimal payload recognized as a remote session bridge ref."""
    return {"__comfy_modal_remote_session_bridge_ref__": True, "item": item}


def test_registry_tracks_slots_until_complete(progressive_preview_module: Any) -> None:
    """Slots fill out of order, report completeness, and return images in slot order."""
    registry = progressive_preview_module.ProgressivePreviewRegistry()
    registry.begin("prompt-1", "18", 3)
    registry.fill("prompt-1", "18", 2, [_image_ref(progressive_preview_module, "c.png")])
    state = registry.fill(
        "prompt-1", "18", 0, [_image_ref(progressive_preview_module, "a.png")]
    )

    assert not state.is_complete()
    assert state.effective_slot_count() == 3

    registry.fill("prompt-1", "18", 1, [_image_ref(progressive_preview_module, "b.png")])
    final_state = registry.pop("prompt-1", "18")

    assert final_state is not None and final_state.is_complete()
    assert [image.filename for image in final_state.ordered_entries()] == [
        "a.png",
        "b.png",
        "c.png",
    ]
    assert registry.get("prompt-1", "18") is None


def test_registry_fill_without_begin_grows_slot_count(
    progressive_preview_module: Any,
) -> None:
    """A non-mapped stream fills slot 0 and never claims to know the full count."""
    registry = progressive_preview_module.ProgressivePreviewRegistry()
    state = registry.fill(
        "prompt-1", "18", 0, [_image_ref(progressive_preview_module, "a.png")]
    )

    assert state.slot_count is None
    assert state.effective_slot_count() == 1
    assert not state.is_complete()


def test_registry_drops_earlier_prompts_for_the_same_node(
    progressive_preview_module: Any,
) -> None:
    """A new prompt replaces a node's stale slots and the registry stays bounded."""
    registry = progressive_preview_module.ProgressivePreviewRegistry(entry_limit=2)
    registry.begin("prompt-1", "18", 2)
    registry.begin("prompt-2", "18", 4)
    registry.begin("prompt-2", "19", 1)
    registry.begin("prompt-2", "20", 1)

    assert registry.get("prompt-1", "18") is None
    assert registry.get("prompt-2", "18") is None
    assert registry.get("prompt-2", "19") is not None
    assert registry.get("prompt-2", "20") is not None


def test_iter_preview_values_sorts_each_value_by_type(
    progressive_preview_module: Any,
) -> None:
    """Lists flatten; tensors, audio dicts, text, and numbers each get their own preview kind."""
    torch = pytest.importorskip("torch")
    audio = {"waveform": torch.zeros(1, 2, 8), "sample_rate": 44100}
    values = list(
        progressive_preview_module.iter_preview_values(
            [
                torch.zeros(2, 4, 4, 3),
                [torch.zeros(4, 4, 3)],
                torch.zeros(4, 4),
                torch.zeros(2, 4, 5),
                audio,
                "a caption",
                7,
                None,
                {"samples": torch.zeros(1, 4, 8, 8)},
            ]
        )
    )

    assert [value.kind for value in values] == [
        "image",
        "image",
        "mask",
        "mask",
        "audio",
        "text",
        "text",
    ]
    assert tuple(values[1].value.shape) == (1, 4, 4, 3)
    assert values[4].value is audio
    assert [values[5].value, values[6].value] == ["a caption", "7"]


def test_iter_preview_values_trusts_the_mask_io_type(
    progressive_preview_module: Any,
) -> None:
    """A MASK batch whose width happens to be 3 is still a mask when the type says so."""
    torch = pytest.importorskip("torch")
    mask = torch.zeros(2, 4, 3)

    [guessed] = progressive_preview_module.iter_preview_values(mask)
    [typed] = progressive_preview_module.iter_preview_values(mask, "MASK")

    assert guessed.kind == "image"
    assert typed.kind == "mask"


def test_save_preview_value_routes_each_kind(
    progressive_preview_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Masks save as grayscale images, audio saves as clips, and text needs no file."""
    torch = pytest.importorskip("torch")
    saved_batches: list[Any] = []
    monkeypatch.setattr(
        progressive_preview_module,
        "_save_image_batch",
        lambda batch: saved_batches.append(batch) or [_image_ref(progressive_preview_module, "m.png")],
    )
    monkeypatch.setattr(
        progressive_preview_module,
        "_save_audio",
        lambda audio: [
            progressive_preview_module.PreviewEntry(kind="audio", filename="a.flac")
        ],
    )

    entries = progressive_preview_module.save_temp_preview_entries(
        [torch.ones(2, 4, 5), {"waveform": torch.zeros(1, 1, 8), "sample_rate": 8000}, "hi"]
    )

    assert [entry.as_ui() for entry in entries] == [
        {"kind": "image", "filename": "m.png", "subfolder": "", "type": "temp"},
        {"kind": "audio", "filename": "a.flac", "subfolder": "", "type": "temp"},
        {"kind": "text", "text": "hi"},
    ]
    assert tuple(saved_batches[0].shape) == (2, 4, 5, 3)


def test_progressive_preview_ui_lists_images_for_history(
    progressive_preview_module: Any,
) -> None:
    """Every entry reaches the panel; images are also listed for ComfyUI's history."""
    entries = [
        progressive_preview_module.PreviewEntry.from_text("caption"),
        _image_ref(progressive_preview_module, "a.png"),
    ]

    assert progressive_preview_module.progressive_preview_ui(entries) == {
        "modal_preview": [
            {"kind": "text", "text": "caption"},
            {"kind": "image", "filename": "a.png", "subfolder": "", "type": "temp"},
        ],
        "images": [{"filename": "a.png", "subfolder": "", "type": "temp"}],
    }
    assert progressive_preview_module.progressive_preview_ui(entries[:1]) == {
        "modal_preview": [{"kind": "text", "text": "caption"}]
    }


def test_resolve_reuses_complete_streamed_slots(
    progressive_preview_module: Any,
    progressive_preview_node_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fully streamed preview never saves or downloads the delivered value again."""
    monkeypatch.setattr(
        progressive_preview_node_module,
        "save_temp_preview_entries",
        lambda value: pytest.fail("streamed slots should be reused"),
    )
    state = progressive_preview_module.ProgressivePreviewState(
        slot_count=2,
        slots={
            1: [_image_ref(progressive_preview_module, "b.png")],
            0: [_image_ref(progressive_preview_module, "a.png")],
        },
    )

    images = asyncio.run(
        progressive_preview_node_module.resolve_progressive_preview_entries(
            [_bridge_ref(0), _bridge_ref(1)], state
        )
    )

    assert [image.filename for image in images] == ["a.png", "b.png"]


def test_resolve_materializes_only_missing_bridge_slots(
    progressive_preview_module: Any,
    progressive_preview_node_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Partially streamed refs reuse streamed slots and download only the gaps."""
    materialized: list[dict[str, Any]] = []
    monkeypatch.setattr(
        progressive_preview_node_module,
        "_materialize_bridge_ref",
        lambda ref: materialized.append(ref) or f"tensor-{ref['item']}",
    )
    monkeypatch.setattr(
        progressive_preview_node_module,
        "save_temp_preview_entries",
        lambda value: [_image_ref(progressive_preview_module, f"{value}.png")],
    )
    state = progressive_preview_module.ProgressivePreviewState(
        slot_count=2, slots={0: [_image_ref(progressive_preview_module, "streamed.png")]}
    )

    images = asyncio.run(
        progressive_preview_node_module.resolve_progressive_preview_entries(
            [_bridge_ref(0), _bridge_ref(1)], state
        )
    )

    assert [image.filename for image in images] == ["streamed.png", "tensor-1.png"]
    assert materialized == [_bridge_ref(1)]


def test_node_schema_is_list_input_output_node(
    progressive_preview_node_module: Any,
) -> None:
    """The node behaves like PreviewImage but receives the whole mapped list at once."""
    schema = progressive_preview_node_module.RemoteProgressivePreview.define_schema()

    assert schema.node_id == "RemoteProgressivePreview"
    assert schema.is_output_node is True
    assert schema.is_input_list is True
    assert schema.outputs == []
    assert schema.inputs[0].id == "images"
    assert schema.inputs[0].get_io_type() == "IMAGE,MASK,AUDIO,STRING"


def test_node_execute_pops_registry_and_returns_ui_images(
    progressive_preview_module: Any,
    progressive_preview_node_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Final execution returns the streamed images and clears the registry entry."""
    registry = progressive_preview_module.PROGRESSIVE_PREVIEW_REGISTRY
    registry.begin("prompt-1", "18", 1)
    registry.fill("prompt-1", "18", 0, [_image_ref(progressive_preview_module, "a.png")])
    monkeypatch.setattr(
        progressive_preview_node_module,
        "_executing_prompt_and_node",
        lambda: ("prompt-1", "18"),
    )

    output = asyncio.run(
        progressive_preview_node_module.RemoteProgressivePreview.execute(
            [_bridge_ref(0)]
        )
    )

    assert output.ui == {
        "modal_preview": [
            {"kind": "image", "filename": "a.png", "subfolder": "", "type": "temp"}
        ],
        "images": [{"filename": "a.png", "subfolder": "", "type": "temp"}],
    }
    assert registry.get("prompt-1", "18") is None


def test_node_execute_previews_local_text_values(
    progressive_preview_node_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local text and number items become text entries without saving any file."""
    monkeypatch.setattr(
        progressive_preview_node_module,
        "_executing_prompt_and_node",
        lambda: ("prompt-1", "18"),
    )

    output = asyncio.run(
        progressive_preview_node_module.RemoteProgressivePreview.execute(
            ["first caption", 2]
        )
    )

    assert output.ui == {
        "modal_preview": [
            {"kind": "text", "text": "first caption"},
            {"kind": "text", "text": "2"},
        ]
    }


class _FakePromptServer:
    """Collect websocket events sent by local UI emitters."""

    def __init__(self) -> None:
        """Create an empty event log."""
        self.events: list[tuple[str, dict[str, Any], str | None]] = []

    def send_sync(self, event: str, data: dict[str, Any], client_id: str | None) -> None:
        """Record one websocket event."""
        self.events.append((event, data, client_id))


def test_local_emitters_send_start_and_slot_events(
    progressive_preview_module: Any,
    local_ui_events_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Start lays out empty slots; each slot event carries its saved images."""
    prompt_server = _FakePromptServer()
    monkeypatch.setattr(
        local_ui_events_module, "_lookup_local_prompt_server", lambda: prompt_server
    )
    monkeypatch.setattr(
        local_ui_events_module, "_record_local_modal_ui_event", lambda *args: None
    )
    monkeypatch.setattr(
        progressive_preview_module,
        "save_temp_preview_entries",
        lambda value, io_type=None: [_image_ref(progressive_preview_module, f"{value}.png")],
    )

    local_ui_events_module._emit_local_progressive_preview_start(
        prompt_id="prompt-1", client_id="client-1", node_ids=["18"], slot_count=3
    )
    local_ui_events_module._emit_local_progressive_preview_slot(
        prompt_id="prompt-1",
        client_id="client-1",
        node_ids=["18"],
        slot_index=1,
        slot_count=3,
        value="item-1",
    )

    assert prompt_server.events == [
        (
            "modal_progressive_preview",
            {"phase": "start", "prompt_id": "prompt-1", "node": "18", "slot_count": 3},
            "client-1",
        ),
        (
            "modal_progressive_preview",
            {
                "phase": "slot",
                "prompt_id": "prompt-1",
                "node": "18",
                "slot_index": 1,
                "slot_count": 3,
                "items": [
                    {"kind": "image", "filename": "item-1.png", "subfolder": "", "type": "temp"}
                ],
            },
            "client-1",
        ),
    ]
    state = progressive_preview_module.PROGRESSIVE_PREVIEW_REGISTRY.get("prompt-1", "18")
    assert state is not None and sorted(state.slots) == [1]


def test_boundary_stream_event_fills_the_mapped_item_slot(
    payload_stream_module: Any,
    serialization_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A streamed boundary output fills the slot named by the item payload."""
    slot_calls: list[dict[str, Any]] = []
    preview_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        payload_stream_module,
        "_emit_local_progressive_preview_slot",
        lambda **kwargs: slot_calls.append(kwargs),
    )
    monkeypatch.setattr(
        payload_stream_module,
        "_emit_local_preview_boundary_output",
        lambda **kwargs: preview_calls.append(kwargs),
    )
    state = types.SimpleNamespace(
        prompt_id="prompt-1",
        client_id="client-1",
        payload={"component_id": "1__mapped::item:2", "map_item_index": 2, "map_item_count": 4},
    )

    payload_stream_module._handle_boundary_output_event(
        state,
        {
            "event_type": "boundary_output",
            "node_id": "9",
            "preview_target_node_ids": [],
            "progressive_preview_target_node_ids": ["18"],
            "value": serialization_module.serialize_value(["image-2"]),
        },
    )

    assert slot_calls == [
        {
            "prompt_id": "prompt-1",
            "client_id": "client-1",
            "node_ids": ["18"],
            "slot_index": 2,
            "slot_count": 4,
            "value": ["image-2"],
            "io_type": None,
        }
    ]
    assert preview_calls == []


def test_mapped_item_payloads_carry_item_count(mapped_execution_module: Any) -> None:
    """Per-item payloads tell the host how many slots the mapped run fills."""
    payload = {
        "component_id": "1__mapped",
        "mapped_input": {"proxy_input_name": "remote_input_0"},
        "boundary_outputs": [
            {
                "node_id": "9",
                "output_index": 0,
                "io_type": "IMAGE",
                "progressive_preview_target_node_ids": ["18"],
            }
        ],
        "execute_node_ids": ["9"],
    }

    item_payload = mapped_execution_module._build_mapped_item_payload(
        payload, 1, 0, item_count=3
    )

    assert item_payload["map_item_index"] == 1
    assert item_payload["map_item_count"] == 3
    assert mapped_execution_module._mapped_progressive_preview_targets(payload) == ["18"]


def test_mapped_start_lays_out_slots_for_progressive_targets(
    mapped_execution_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mapped dispatch announces one slot per item before any item runs."""
    start_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        mapped_execution_module,
        "_emit_local_progressive_preview_start",
        lambda **kwargs: start_calls.append(kwargs),
    )
    payload = {
        "component_id": "1__mapped",
        "prompt_id": "prompt-1",
        "extra_data": {"client_id": "client-1"},
        "mapped_input": {"proxy_input_name": "remote_input_0"},
        "boundary_outputs": [
            {"node_id": "9", "progressive_preview_target_node_ids": ["18"]}
        ],
    }

    mapped_execution_module._emit_mapped_progressive_preview_start(payload, 5)

    assert start_calls == [
        {
            "prompt_id": "prompt-1",
            "client_id": "client-1",
            "node_ids": ["18"],
            "slot_count": 5,
        }
    ]


class _FakeProgressivePreviewNode(_FakeRewritePreviewImageNode):
    """Fake local RemoteProgressivePreview output node."""


def test_progressive_preview_consumer_gets_bridge_without_materializer(
    api_intercept_module: Any,
    modal_executor_module: Any,
    settings_module: Any,
    sync_engine_module: Any,
    tmp_path: Path,
) -> None:
    """A progressive preview beside a remote continuation is streamed, not materialized."""
    settings = settings_module.ModalSyncSettings(
        app_name="app",
        auto_deploy=True,
        allow_ephemeral_fallback=False,
        enable_memory_snapshot=True,
        enable_gpu_memory_snapshot=False,
        execution_mode="local",
        sync_custom_nodes=False,
        volume_name="volume",
        route_path="/modal/queue_prompt",
        marker_property="is_modal_remote",
        local_storage_root=tmp_path / "storage",
        remote_storage_root="/storage",
        custom_nodes_archive_name="custom_nodes_bundle.zip",
        comfyui_root=None,
        custom_nodes_dir=tmp_path / "custom_nodes",
    )
    settings.custom_nodes_dir.mkdir()
    sync_engine = sync_engine_module.ModalAssetSyncEngine.from_environment(settings)
    fake_nodes_module = types.SimpleNamespace(
        NODE_CLASS_MAPPINGS={
            "RemoteImage": _FakeRewriteRemoteImageNode,
            "RemoteText": _FakeRewriteRemoteTextNode,
            "RemoteProgressivePreview": _FakeProgressivePreviewNode,
            "LocalSink": _FakeRewriteLocalSinkNode,
        },
        NODE_DISPLAY_NAME_MAPPINGS={},
    )
    workflow = {
        "nodes": [
            {"id": 1, "properties": {"is_modal_remote": True}},
            {"id": 2, "properties": {"is_modal_remote": True}},
            {"id": 3, "properties": {"is_modal_remote": False}},
            {"id": 4, "properties": {"is_modal_remote": False}},
        ]
    }
    prompt = {
        "1": {"class_type": "RemoteImage", "inputs": {}},
        "2": {"class_type": "RemoteText", "inputs": {"image": ["1", 0]}},
        "3": {"class_type": "RemoteProgressivePreview", "inputs": {"images": ["1", 0]}},
        "4": {"class_type": "LocalSink", "inputs": {"text": ["2", 0]}},
    }

    rewritten_prompt, summary = api_intercept_module.rewrite_prompt_for_modal(
        prompt=prompt,
        workflow=workflow,
        sync_engine=sync_engine,
        settings=settings,
        nodes_module=fake_nodes_module,
        extra_data={"prompt_id": "prompt-progressive"},
    )

    producer_payload = modal_executor_module._rehydrate_proxy_payload(
        rewritten_prompt["1"]["inputs"]["original_node_data"],
        unique_id="1",
    )
    boundary_output = producer_payload["boundary_outputs"][0]
    assert boundary_output["session_output"] is True
    assert boundary_output["progressive_preview_target_node_ids"] == ["3"]
    assert not any(
        prompt_node["class_type"]
        == api_intercept_module.MODAL_LOCAL_BRIDGE_MATERIALIZER_NODE_ID
        for prompt_node in rewritten_prompt.values()
    )
    dispatch_gate_node_id = summary.parallel_local_branch_node_ids[0]
    assert rewritten_prompt["3"]["inputs"]["images"] == [dispatch_gate_node_id, 0]
    assert rewritten_prompt[dispatch_gate_node_id]["inputs"]["value"] == ["1", 0]


class _FakeRemoteSaveNode:
    """Fake remote output node consuming a mapped sampler result."""

    RETURN_TYPES: tuple[str, ...] = ()
    RETURN_NAMES: tuple[str, ...] = ()
    OUTPUT_IS_LIST: tuple[bool, ...] = ()
    OUTPUT_NODE = True


def test_hybrid_mapped_output_streams_to_progressive_preview(
    api_intercept_module: Any,
    modal_executor_module: Any,
    settings_module: Any,
    sync_engine_module: Any,
    tmp_path: Path,
) -> None:
    """The logged failure shape: a mapped output feeding a remote save and a local preview.

    The remote save runs inside the same per-item execution, so the preview output
    is returned directly rather than through a session bridge.
    """
    import json

    from test_modal_test_workflow import (
        _FakeRegressionLatentSourceNode,
        _FakeRegressionLocalSinkNode,
        _FakeRegressionModalMapInputNode,
        _FakeRegressionRemoteModelNode,
        _FakeRegressionRemoteSamplerNode,
    )

    artifact = json.loads(
        (Path(__file__).parent / "fixtures" / "modal_hybrid_split_regression.json").read_text()
    )
    prompt = artifact["prompt"]
    prompt["8"] = {"class_type": "RemoteProgressivePreview", "inputs": {"images": ["7", 0]}}
    prompt["9"] = {"class_type": "RemoteSave", "inputs": {"images": ["7", 0]}}
    workflow = artifact["workflow"]
    workflow["nodes"].append({"id": 9, "properties": {"is_modal_remote": True}})
    settings = settings_module.ModalSyncSettings(
        app_name="app",
        auto_deploy=True,
        allow_ephemeral_fallback=False,
        enable_memory_snapshot=True,
        enable_gpu_memory_snapshot=False,
        execution_mode="local",
        sync_custom_nodes=False,
        volume_name="volume",
        route_path="/modal/queue_prompt",
        marker_property="is_modal_remote",
        local_storage_root=tmp_path / "storage",
        remote_storage_root="/storage",
        custom_nodes_archive_name="custom_nodes_bundle.zip",
        comfyui_root=None,
        custom_nodes_dir=tmp_path / "custom_nodes",
    )
    settings.custom_nodes_dir.mkdir()
    sync_engine = sync_engine_module.ModalAssetSyncEngine.from_environment(settings)
    fake_nodes_module = types.SimpleNamespace(
        NODE_CLASS_MAPPINGS={
            "RemoteModel": _FakeRegressionRemoteModelNode,
            "RemoteSampler": _FakeRegressionRemoteSamplerNode,
            "LatentSource": _FakeRegressionLatentSourceNode,
            "ModalMapInput": _FakeRegressionModalMapInputNode,
            "LocalSink": _FakeRegressionLocalSinkNode,
            "RemoteProgressivePreview": _FakeProgressivePreviewNode,
            "RemoteSave": _FakeRemoteSaveNode,
        },
        NODE_DISPLAY_NAME_MAPPINGS={},
    )

    rewritten_prompt, summary = api_intercept_module.rewrite_prompt_for_modal(
        prompt=prompt,
        workflow=workflow,
        sync_engine=sync_engine,
        settings=settings,
        nodes_module=fake_nodes_module,
        extra_data={"prompt_id": "prompt-hybrid-progressive"},
    )

    preview_source = rewritten_prompt["8"]["inputs"]["images"]
    if preview_source[0] in summary.parallel_local_branch_node_ids:
        preview_source = rewritten_prompt[preview_source[0]]["inputs"]["value"]
    producer_id = str(preview_source[0])
    producer_payload = modal_executor_module._rehydrate_proxy_payload(
        rewritten_prompt[producer_id]["inputs"]["original_node_data"],
        unique_id=producer_id,
    )
    progressive_outputs = [
        output
        for output in producer_payload["boundary_outputs"]
        if output.get("progressive_preview_target_node_ids") == ["8"]
    ]
    assert producer_id.endswith("__mapped")
    assert progressive_outputs, producer_payload["boundary_outputs"]
    assert "session_output" not in progressive_outputs[0]
    assert "9" in producer_payload["component_node_ids"]
    assert not any(
        prompt_node["class_type"]
        == api_intercept_module.MODAL_LOCAL_BRIDGE_MATERIALIZER_NODE_ID
        for prompt_node in rewritten_prompt.values()
    )
