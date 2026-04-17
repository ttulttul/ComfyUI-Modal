"""Tests for prompt rewriting and asset sync integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class _FakeOriginalNode:
    """Simple fake source node for rewrite tests."""

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("image", "count")
    OUTPUT_IS_LIST = (False, False)


def test_rewrite_prompt_for_remote_nodes(
    api_intercept_module: Any,
    settings_module: Any,
    sync_engine_module: Any,
    tmp_path: Path,
) -> None:
    """Prompt rewrite should proxy remote-marked nodes and mirror their assets."""
    model_path = tmp_path / "weights.safetensors"
    model_path.write_bytes(b"weights")
    custom_nodes_dir = tmp_path / "custom_nodes"
    custom_nodes_dir.mkdir()
    (custom_nodes_dir / "__init__.py").write_text("NODE_CLASS_MAPPINGS = {}\n", encoding="utf-8")

    settings = settings_module.ModalSyncSettings(
        app_name="app",
        execution_mode="local",
        sync_custom_nodes=False,
        volume_name="volume",
        route_path="/modal/queue_prompt",
        marker_property="is_modal_remote",
        local_storage_root=tmp_path / "storage",
        remote_storage_root="/storage",
        custom_nodes_archive_name="custom_nodes_bundle.zip",
        comfyui_root=None,
        custom_nodes_dir=custom_nodes_dir,
    )
    sync_engine = sync_engine_module.ModalAssetSyncEngine.from_environment(settings)
    fake_nodes_module = type(
        "FakeNodesModule",
        (),
        {
            "NODE_CLASS_MAPPINGS": {"OriginalNode": _FakeOriginalNode},
            "NODE_DISPLAY_NAME_MAPPINGS": {},
        },
    )()

    workflow = {
        "nodes": [
            {"id": 1, "properties": {"is_modal_remote": True}},
            {"id": 2, "properties": {"is_modal_remote": False}},
        ]
    }
    prompt = {
        "1": {
            "class_type": "OriginalNode",
            "inputs": {
                "model_name": str(model_path),
                "strength": 0.5,
            },
            "_meta": {"title": "Remote Node"},
        },
        "2": {
            "class_type": "OtherNode",
            "inputs": {},
            "_meta": {"title": "Local Node"},
        },
    }

    rewritten_prompt, summary = api_intercept_module.rewrite_prompt_for_modal(
        prompt=prompt,
        workflow=workflow,
        sync_engine=sync_engine,
        settings=settings,
        nodes_module=fake_nodes_module,
    )

    rewritten_node = rewritten_prompt["1"]
    assert rewritten_node["class_type"].startswith("ModalUniversalExecutor_")
    assert rewritten_node["inputs"]["model_name"].startswith("/assets/")
    assert rewritten_node["inputs"]["original_node_data"]["class_type"] == "OriginalNode"
    assert "original_node_data" not in rewritten_node["inputs"]["original_node_data"]["inputs"]
    assert rewritten_node["inputs"] is not rewritten_node["inputs"]["original_node_data"]["inputs"]
    assert rewritten_prompt["2"]["class_type"] == "OtherNode"
    assert summary.remote_node_ids == ["1"]
    assert len(summary.synced_assets) == 1
    assert summary.custom_nodes_bundle is None
