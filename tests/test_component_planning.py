"""Tests for the component planning boundary."""

from __future__ import annotations

from api_intercept_test_support import *  # noqa: F401,F403

def test_rewritten_prompt_diagnostics_reports_dependency_cycles(
    api_intercept_module: Any,
    prompt_diagnostics_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rewritten prompt diagnostics should name local dependency cycles before Comfy executes."""
    prompt = {
        "1": {
            "class_type": "ModalUniversalExecutor_a",
            "inputs": {"remote_input_0": ["2", 0]},
        },
        "2": {
            "class_type": "ModalUniversalExecutor_b",
            "inputs": {"remote_input_0": ["1", 0]},
        },
    }

    diagnostics = api_intercept_module._modal_rewritten_prompt_diagnostics(prompt)

    assert diagnostics["cycles"] == [["1", "2", "1"]]

    warning_messages: list[str] = []
    log_messages: list[str] = []

    def record_warning(message: str, *args: Any, **_kwargs: Any) -> None:
        """Record one warning log message."""
        warning_messages.append(message % args)

    def record_log(_level: int, message: str, *args: Any, **_kwargs: Any) -> None:
        """Record one generic log message."""
        log_messages.append(message % args)

    monkeypatch.setattr(prompt_diagnostics_module.logger, "warning", record_warning)
    monkeypatch.setattr(prompt_diagnostics_module.logger, "log", record_log)

    api_intercept_module._log_modal_rewritten_prompt_diagnostics(
        prompt_id="prompt-cycle",
        prompt=prompt,
        reason="test",
    )

    assert any("Modal rewritten prompt contains dependency cycle(s)" in item for item in warning_messages)
    assert any("prompt-cycle" in item for item in warning_messages)
    assert any("Modal rewritten prompt diagnostics" in item for item in log_messages)

def test_component_local_reentry_dependency_detection(
    api_intercept_module: Any,
    component_planning_module: Any,
) -> None:
    """Boundary inputs that trace back to the same component require a split-capable proxy."""
    prompt = {
        "1": {
            "class_type": "RemoteImage",
            "inputs": {},
            "_meta": {"title": "Remote Image"},
        },
        "4": {
            "class_type": "LocalSink",
            "inputs": {"image": ["1", 0]},
            "_meta": {"title": "Local Transform"},
        },
        "2": {
            "class_type": "RemoteImageConsumer",
            "inputs": {"image": ["4", 0]},
            "_meta": {"title": "Remote Consumer"},
        },
    }
    component = api_intercept_module.RemoteComponentPlan(
        node_ids=["1", "2"],
        representative_node_id="1",
        boundary_inputs=[
            api_intercept_module.BoundaryInputSpec(
                proxy_input_name="remote_input_0",
                source=api_intercept_module.LinkedOutputRef(node_id="4", output_index=0),
                io_type="IMAGE",
                targets=[
                    api_intercept_module.InputTarget(
                        node_id="2",
                        input_name="image",
                    )
                ],
            )
        ],
        boundary_outputs=[],
        execute_node_ids=["1", "2"],
        contains_output_node=False,
        local_tap_node_ids=["9"],
    )

    assert component_planning_module._component_has_local_reentry_dependency(
        prompt=prompt,
        component=component,
    )

def test_sandwiched_local_nodes_include_only_remote_reentry_paths(
    api_intercept_module: Any,
) -> None:
    """Planner warnings should cover local chains that leave and re-enter remote work."""
    prompt = {
        "1": {"class_type": "RemoteSource", "inputs": {}},
        "2": {"class_type": "LocalTransform", "inputs": {"value": ["1", 0]}},
        "3": {
            "class_type": "LocalTransform",
            "inputs": {"value": ["2", 0], "local_only": ["7", 0]},
        },
        "4": {"class_type": "RemoteSink", "inputs": {"value": ["3", 0]}},
        "5": {"class_type": "LocalPreview", "inputs": {"value": ["1", 0]}},
        "6": {"class_type": "RemoteSink", "inputs": {"value": ["8", 0]}},
        "7": {"class_type": "LocalSource", "inputs": {}},
        "8": {"class_type": "LocalSource", "inputs": {}},
        "9": {"class_type": "RemoteSink", "inputs": {"value": ["1", 1]}},
    }

    assert api_intercept_module._sandwiched_local_node_ids(
        prompt,
        {"1", "4", "6", "9"},
    ) == {"2", "3"}

def test_analyze_remote_node_selection_returns_nodes_to_mark_and_reasons(
    api_intercept_module: Any,
    remote_graph_analysis_module: Any,
    settings_module: Any,
) -> None:
    """Dry-run analysis should surface the clicked node plus required upstream nodes."""
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
        local_storage_root=Path("/tmp/storage"),
        remote_storage_root="/storage",
        custom_nodes_archive_name="custom_nodes_bundle.zip",
        comfyui_root=None,
        custom_nodes_dir=Path("/tmp/custom_nodes"),
    )
    fake_nodes_module = type(
        "FakeNodesModule",
        (),
        {
            "NODE_CLASS_MAPPINGS": {
                "ModelSource": _FakeRemoteModelNode,
                "ConditioningSource": _FakeRemoteConditioningNode,
                "RemoteConsumer": _FakeRemoteSamplerNode,
            },
            "NODE_DISPLAY_NAME_MAPPINGS": {},
        },
    )()
    workflow = {
        "nodes": [
            {"id": 1, "properties": {"is_modal_remote": False}},
            {"id": 2, "properties": {"is_modal_remote": False}},
            {"id": 3, "properties": {"is_modal_remote": False}},
        ]
    }
    prompt = {
        "1": {"class_type": "ModelSource", "inputs": {}, "_meta": {"title": "Model"}},
        "2": {
            "class_type": "ConditioningSource",
            "inputs": {},
            "_meta": {"title": "Conditioning"},
        },
        "3": {
            "class_type": "RemoteConsumer",
            "inputs": {"model": ["1", 0], "conditioning": ["2", 0]},
            "_meta": {"title": "Remote Consumer"},
        },
    }

    analysis = remote_graph_analysis_module.analyze_remote_node_selection(
        prompt=prompt,
        workflow=workflow,
        seed_workflow_node_paths=["3"],
        settings=settings,
        nodes_module=fake_nodes_module,
    )

    assert analysis.requested_node_ids == ["3"]
    assert analysis.requested_workflow_node_paths == ["3"]
    assert analysis.current_remote_node_ids == []
    assert analysis.current_remote_workflow_node_paths == []
    assert analysis.resolved_remote_node_ids == ["1", "2", "3"]
    assert analysis.resolved_workflow_node_paths == ["1", "2", "3"]
    assert analysis.added_node_ids == ["1", "2", "3"]
    assert analysis.added_workflow_node_paths == ["1", "2", "3"]
    assert [(reason.node_id, reason.required_by_node_id) for reason in analysis.reasons] == [
        ("1", "3"),
        ("2", "3"),
    ]

def test_analyze_remote_node_selection_reports_local_reentry(
    api_intercept_module: Any,
    remote_graph_analysis_module: Any,
    settings_module: Any,
) -> None:
    """Dry-run analysis should report local nodes between existing remote regions."""
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
        local_storage_root=Path("/tmp/storage"),
        remote_storage_root="/storage",
        custom_nodes_archive_name="custom_nodes_bundle.zip",
        comfyui_root=None,
        custom_nodes_dir=Path("/tmp/custom_nodes"),
    )
    fake_nodes_module = type(
        "FakeNodesModule",
        (),
        {
            "NODE_CLASS_MAPPINGS": {
                "RemoteString": _FakeRemoteStringEchoNode,
                "LocalString": _FakeLocalStringSinkNode,
            },
            "NODE_DISPLAY_NAME_MAPPINGS": {},
        },
    )()
    workflow = {
        "nodes": [
            {"id": 1, "properties": {"is_modal_remote": True}},
            {"id": 2, "properties": {"is_modal_remote": False}},
            {"id": 3, "properties": {"is_modal_remote": True}},
        ]
    }
    prompt = {
        "1": {"class_type": "RemoteString", "inputs": {}},
        "2": {"class_type": "LocalString", "inputs": {"text": ["1", 0]}},
        "3": {"class_type": "RemoteString", "inputs": {"text": ["2", 0]}},
    }

    analysis = remote_graph_analysis_module.analyze_remote_node_selection(
        prompt=prompt,
        workflow=workflow,
        seed_workflow_node_paths=[],
        settings=settings,
        nodes_module=fake_nodes_module,
    )

    assert analysis.resolved_remote_node_ids == ["1", "3"]
    assert analysis.sandwiched_local_node_ids == ["2"]

def test_analyze_remote_node_selection_prefers_nested_workflow_paths(
    api_intercept_module: Any,
    remote_graph_analysis_module: Any,
    settings_module: Any,
) -> None:
    """Nested prompt ids should map back to the specific inner workflow node path."""
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
        local_storage_root=Path("/tmp/storage"),
        remote_storage_root="/storage",
        custom_nodes_archive_name="custom_nodes_bundle.zip",
        comfyui_root=None,
        custom_nodes_dir=Path("/tmp/custom_nodes"),
    )
    fake_nodes_module = type(
        "FakeNodesModule",
        (),
        {
            "NODE_CLASS_MAPPINGS": {
                "ModelSource": _FakeRemoteModelNode,
                "RemoteConsumer": _FakeRemoteSamplerNode,
            },
            "NODE_DISPLAY_NAME_MAPPINGS": {},
        },
    )()
    workflow = {
        "nodes": [
            {
                "id": 24,
                "properties": {"is_modal_remote": False},
                "subgraph": {
                    "nodes": [
                        {"id": 23, "properties": {"is_modal_remote": True}},
                    ]
                },
            },
            {"id": 30, "properties": {"is_modal_remote": False}},
        ]
    }
    prompt = {
        "30": {"class_type": "ModelSource", "inputs": {}, "_meta": {"title": "Model"}},
        "24:23": {
            "class_type": "RemoteConsumer",
            "inputs": {"model": ["30", 0]},
            "_meta": {"title": "Nested Consumer"},
        },
    }

    analysis = remote_graph_analysis_module.analyze_remote_node_selection(
        prompt=prompt,
        workflow=workflow,
        seed_workflow_node_paths=["24:23"],
        settings=settings,
        nodes_module=fake_nodes_module,
    )

    assert analysis.requested_node_ids == ["24:23"]
    assert analysis.current_remote_node_ids == ["24:23"]
    assert analysis.current_remote_workflow_node_paths == ["24:23"]
    assert analysis.resolved_remote_node_ids == ["24:23", "30"]
    assert analysis.resolved_workflow_node_paths == ["24:23", "30"]
    assert analysis.added_node_ids == ["30"]
    assert analysis.added_workflow_node_paths == ["30"]
    assert [(reason.node_id, reason.required_by_node_id) for reason in analysis.reasons] == [
        ("30", "24:23"),
    ]

