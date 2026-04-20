"""Tests for the captured working Modal workflow artifact."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class _FakeRegressionRemoteModelNode:
    """Fake remote node that produces a non-transportable MODEL output."""

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    OUTPUT_IS_LIST = (False,)


class _FakeRegressionRemoteSamplerNode:
    """Fake remote node that produces a transportable LATENT output."""

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    OUTPUT_IS_LIST = (False,)


class _FakeRegressionLatentSourceNode:
    """Fake local node that produces a LATENT input for remote samplers."""

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    OUTPUT_IS_LIST = (False,)


class _FakeRegressionModalMapInputNode:
    """Fake Modal map marker node used during rewrite regression tests."""

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("value",)
    OUTPUT_IS_LIST = (False,)


class _FakeRegressionLocalSinkNode:
    """Fake local sink node used to validate downstream rewiring."""

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_IS_LIST = (False,)


def test_modal_test_workflow_captures_known_working_remote_component() -> None:
    """The checked-in workflow artifact should preserve the first known working remote graph shape."""
    workflow_path = Path(__file__).resolve().parents[1] / "modal_test_workflow.json"
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))

    nodes = workflow["nodes"]
    node_types_by_id = {int(node["id"]): str(node["type"]) for node in nodes}
    remote_node_ids = {
        int(node["id"])
        for node in nodes
        if bool(node.get("properties", {}).get("is_modal_remote"))
    }

    assert workflow_path.exists()
    assert workflow["version"] == 0.4
    assert node_types_by_id[2] == "KSampler"
    assert node_types_by_id[7] == "UNETLoader"
    assert node_types_by_id[11] == "CLIPLoader"
    assert remote_node_ids == {2, 4, 5, 6, 7, 11}
    assert any(str(node["type"]).startswith("ModalUniversalExecutor") for node in nodes)
    assert node_types_by_id[8] == "VAEDecode"
    assert node_types_by_id[9] == "PreviewImage"


def test_hybrid_split_workflow_artifact_rewrites_into_static_and_mapped_proxies(
    api_intercept_module: Any,
    settings_module: Any,
    sync_engine_module: Any,
    tmp_path: Path,
) -> None:
    """The checked-in hybrid artifact should keep rewriting into static and mapped split proxies."""
    artifact_path = (
        Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "modal_hybrid_split_regression.json"
    )
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    workflow = artifact["workflow"]
    prompt = artifact["prompt"]

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
    fake_nodes_module = type(
        "FakeNodesModule",
        (),
        {
            "NODE_CLASS_MAPPINGS": {
                "RemoteModel": _FakeRegressionRemoteModelNode,
                "RemoteSampler": _FakeRegressionRemoteSamplerNode,
                "LatentSource": _FakeRegressionLatentSourceNode,
                "ModalMapInput": _FakeRegressionModalMapInputNode,
                "LocalSink": _FakeRegressionLocalSinkNode,
            },
            "NODE_DISPLAY_NAME_MAPPINGS": {},
        },
    )()

    rewritten_prompt, summary = api_intercept_module.rewrite_prompt_for_modal(
        prompt=prompt,
        workflow=workflow,
        sync_engine=sync_engine,
        settings=settings,
        nodes_module=fake_nodes_module,
    )

    assert artifact_path.exists()
    assert artifact["name"] == "hybrid_split_static_then_mapped"
    assert {int(node["id"]) for node in workflow["nodes"] if node["properties"]["is_modal_remote"]} == {
        1,
        3,
        6,
        7,
    }
    assert rewritten_prompt["4"]["inputs"]["image"] == ["1", 0]
    assert rewritten_prompt["8"]["inputs"]["image"] == ["1__mapped", 0]
    assert summary.remote_component_ids == ["1", "1__mapped"]
    assert summary.component_dependency_ids_by_representative == {
        "1": [],
        "1__mapped": ["1"],
    }
    assert summary.component_execution_stages == [["1"], ["1__mapped"]]

    static_payload = rewritten_prompt["1"]["inputs"]["original_node_data"]
    mapped_payload = rewritten_prompt["1__mapped"]["inputs"]["original_node_data"]

    assert static_payload["component_node_ids"] == ["1", "3"]
    assert mapped_payload["component_node_ids"] == ["6", "7"]
    assert static_payload["boundary_outputs"][1]["session_output"] is True
    assert static_payload["remote_session"]["session_id"] == mapped_payload["remote_session"]["session_id"]
    assert mapped_payload["clear_remote_session"] is True
