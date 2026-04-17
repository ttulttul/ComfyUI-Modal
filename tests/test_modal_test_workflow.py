"""Tests for the captured working Modal workflow artifact."""

from __future__ import annotations

import json
from pathlib import Path


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
