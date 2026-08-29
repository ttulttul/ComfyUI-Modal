"""Tests split from the Modal executor integration suite."""

from __future__ import annotations

from modal_executor_test_support import *  # noqa: F401,F403

def test_mapped_boundary_payload_requests_scheduler_list_output(
    api_intercept_module: Any,
    component_planning_module: Any,
) -> None:
    """Explicit mapped outputs should tell the local proxy to expose ComfyUI list semantics."""
    boundary_output = api_intercept_module.BoundaryOutputSpec(
        proxy_output_name="sampler_samples",
        source=api_intercept_module.LinkedOutputRef(node_id="12", output_index=0),
        io_type="LATENT",
        is_list=False,
    )

    payload = component_planning_module._boundary_output_payload(
        boundary_output,
        mapped_output=True,
    )

    assert payload["mapped_output"] is True
    assert payload["scheduler_is_list"] is True
    assert component_planning_module._proxy_boundary_output_is_list(payload) is True

def test_modal_cloud_mirrors_phase_logs_to_stdout_in_modal_runtime(
    modal_cloud_module: Any,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    """Timed cloud phases should write directly to stdout inside Modal containers."""
    monkeypatch.setenv("MODAL_IS_REMOTE", "1")

    with modal_cloud_module._timed_phase("phase_under_test", component="component-1"):
        pass

    captured = capsys.readouterr()
    assert "Starting phase_under_test component=component-1" in captured.out
    assert "Finished phase_under_test in " in captured.out
    assert "component=component-1" in captured.out

def test_modal_cloud_finds_memory_mapped_compile_cache_files(
    modal_cloud_module: Any,
    tmp_path: Path,
) -> None:
    """Identify native cache files visible only in process memory maps."""
    volume_root = tmp_path / "compile-cache"
    mapped_library = volume_root / "triton" / "cuda_utils.so"
    mapped_library.parent.mkdir(parents=True)
    mapped_library.touch()
    proc_root = tmp_path / "proc"
    process_root = proc_root / "191"
    process_root.mkdir(parents=True)
    (process_root / "maps").write_text(
        f"7f000000-7f001000 r-xp 00000000 00:1e 144 {mapped_library}\n"
        f"7f001000-7f002000 r--p 00001000 00:1e 144 {mapped_library}\n",
        encoding="utf-8",
    )

    mapped_files = modal_cloud_module._mapped_process_files_under(
        volume_root,
        proc_root=proc_root,
    )

    assert mapped_files == ((191, str(mapped_library)),)

def test_invoke_mapped_remote_engine_async_runs_explicit_mapped_phase_items(
    remote_modal_app_module: Any,
    mapped_execution_module: Any,
    serialization_module: Any,
    monkeypatch: Any,
) -> None:
    """Mapped remote execution should run the explicit mapped phase once per item in order."""
    observed_calls: list[tuple[str, dict[str, Any]]] = []
    progress_updates: list[dict[str, Any]] = []

    def fake_execute_subgraph_prompt(
        payload: dict[str, Any],
        hydrated_inputs: dict[str, Any],
        node_mapping: Any = None,
    ) -> tuple[str]:
        assert payload["payload_kind"] == "subgraph"
        assert payload["suppress_status_stream"] is True
        observed_calls.append((str(payload["component_id"]), dict(hydrated_inputs)))
        return (f"done:{hydrated_inputs['remote_input_0']}",)

    monkeypatch.setattr(
        mapped_execution_module,
        "_execute_subgraph_prompt",
        fake_execute_subgraph_prompt,
    )
    monkeypatch.setattr(
        mapped_execution_module,
        "_emit_local_modal_progress",
        lambda **kwargs: progress_updates.append(kwargs),
    )
    payload = {
        "payload_kind": "mapped_subgraph",
        "component_id": "6",
        "prompt_id": "prompt-1",
        "mapped_input": {"proxy_input_name": "remote_input_0", "io_type": "STRING"},
        "boundary_outputs": [
            {
                "proxy_output_name": "7_text",
                "node_id": "7",
                "output_index": 0,
                "io_type": "STRING",
                "is_list": False,
                "mapped_output": True,
            }
        ],
        "static_to_mapped_boundaries": [],
        "static_phase": {
            "component_node_ids": [],
            "subgraph_prompt": {},
            "boundary_inputs": [],
            "boundary_outputs": [],
            "execute_node_ids": [],
        },
        "mapped_phase": {
            "component_node_ids": ["7"],
            "subgraph_prompt": {
                "7": {
                    "class_type": "RemoteStringEcho",
                    "inputs": {"text": ["remote_input_0", 0]},
                }
            },
            "boundary_inputs": [
                {
                    "proxy_input_name": "remote_input_0",
                    "io_type": "STRING",
                    "targets": [{"node_id": "7", "input_name": "text"}],
                }
            ],
            "boundary_outputs": [
                {
                    "proxy_output_name": "7_text",
                    "node_id": "7",
                    "output_index": 0,
                    "io_type": "STRING",
                    "is_list": False,
                    "mapped_output": True,
                }
            ],
            "execute_node_ids": ["7"],
        },
        "extra_data": {"client_id": "client-1"},
    }
    response = asyncio.run(
        remote_modal_app_module._invoke_mapped_remote_engine_async(
            payload,
            serialization_module.serialize_node_inputs(
                {"remote_input_0": ["a", "b", "c", "d"]}
            ),
        )
    )

    assert serialization_module.deserialize_node_outputs(response) == (
        ["done:a", "done:b", "done:c", "done:d"],
    )
    assert observed_calls == [
        ("6::item:0", {"remote_input_0": "a"}),
        ("6::item:1", {"remote_input_0": "b"}),
        ("6::item:2", {"remote_input_0": "c"}),
        ("6::item:3", {"remote_input_0": "d"}),
    ]
    assert progress_updates[0]["value"] == 0.0
    assert progress_updates[0].get("lane_id") is None
    assert progress_updates[-1]["value"] == 4.0

@pytest.mark.parametrize(
    ("module_fixture_name", "aggregate_function_name"),
    [
        ("remote_modal_app_module", "_aggregate_mapped_outputs"),
        ("modal_cloud_module", "_aggregate_mapped_phase_outputs"),
    ],
)
def test_mapped_latent_aggregation_preserves_scheduler_items_when_shapes_differ(
    request: Any,
    module_fixture_name: str,
    aggregate_function_name: str,
) -> None:
    """Both execution paths should return heterogeneous LATENT values as scheduler items."""
    torch = pytest.importorskip("torch")
    target_module = request.getfixturevalue(module_fixture_name)
    aggregate_outputs = getattr(target_module, aggregate_function_name)
    first_latent = {"samples": torch.zeros((1, 4, 32, 32), dtype=torch.float32)}
    second_latent = {"samples": torch.zeros((1, 4, 35, 35), dtype=torch.float32)}
    payload = {
        "boundary_outputs": [
            {
                "io_type": "LATENT",
                "is_list": False,
                "mapped_output": True,
                "scheduler_is_list": True,
            }
        ]
    }

    result = aggregate_outputs([(first_latent,), (second_latent,)], payload)

    assert len(result) == 1
    assert len(result[0]) == 2
    assert result[0][0] is first_latent
    assert result[0][1] is second_latent

def test_invoke_mapped_remote_engine_async_splits_int_inputs_for_direct_targets(
    remote_modal_app_module: Any,
    mapped_execution_module: Any,
    serialization_module: Any,
    monkeypatch: Any,
) -> None:
    """Mapped remote execution should itemize list(INT) inputs even when ModalMapInput stays local."""
    observed_calls: list[tuple[str, dict[str, Any]]] = []

    def fake_execute_subgraph_prompt(
        payload: dict[str, Any],
        hydrated_inputs: dict[str, Any],
        node_mapping: Any = None,
    ) -> tuple[str]:
        observed_calls.append((str(payload["component_id"]), dict(hydrated_inputs)))
        return (f"seed:{hydrated_inputs['remote_input_0']}",)

    monkeypatch.setattr(
        mapped_execution_module,
        "_execute_subgraph_prompt",
        fake_execute_subgraph_prompt,
    )

    payload = {
        "payload_kind": "mapped_subgraph",
        "component_id": "12",
        "prompt_id": "prompt-1",
        "mapped_input": {"proxy_input_name": "remote_input_0", "io_type": "INT"},
        "boundary_outputs": [
            {
                "proxy_output_name": "12_latent",
                "node_id": "12",
                "output_index": 0,
                "io_type": "STRING",
                "is_list": False,
                "mapped_output": True,
            }
        ],
        "static_to_mapped_boundaries": [],
        "static_phase": {
            "component_node_ids": [],
            "subgraph_prompt": {},
            "boundary_inputs": [],
            "boundary_outputs": [],
            "execute_node_ids": [],
        },
        "mapped_phase": {
            "component_node_ids": ["12"],
            "subgraph_prompt": {
                "12": {
                    "class_type": "RemoteSampler",
                    "inputs": {"seed": ["remote_input_0", 0]},
                }
            },
            "boundary_inputs": [
                {
                    "proxy_input_name": "remote_input_0",
                    "io_type": "INT",
                    "targets": [{"node_id": "12", "input_name": "seed"}],
                }
            ],
            "boundary_outputs": [
                {
                    "proxy_output_name": "12_latent",
                    "node_id": "12",
                    "output_index": 0,
                    "io_type": "STRING",
                    "is_list": False,
                    "mapped_output": True,
                }
            ],
            "execute_node_ids": ["12"],
        },
        "extra_data": {"client_id": "client-1"},
    }

    response = asyncio.run(
        remote_modal_app_module._invoke_mapped_remote_engine_async(
            payload,
            serialization_module.serialize_node_inputs(
                {"remote_input_0": [10, 11, 12]}
            ),
        )
    )

    assert serialization_module.deserialize_node_outputs(response) == (
        [
            "seed:10",
            "seed:11",
            "seed:12",
        ],
    )
    assert observed_calls == [
        ("12::item:0", {"remote_input_0": 10}),
        ("12::item:1", {"remote_input_0": 11}),
        ("12::item:2", {"remote_input_0": 12}),
    ]

def test_invoke_mapped_remote_engine_async_executes_static_branch_once(
    remote_modal_app_module: Any,
    mapped_execution_module: Any,
    serialization_module: Any,
    monkeypatch: Any,
) -> None:
    """Mapped remote execution should run the explicit static phase once and inject its bridge outputs."""
    observed_execute_node_ids: list[tuple[str, tuple[str, ...]]] = []

    def fake_execute_subgraph_prompt(
        payload: dict[str, Any],
        hydrated_inputs: dict[str, Any],
        node_mapping: Any = None,
    ) -> tuple[str, ...]:
        observed_execute_node_ids.append(
            (
                str(payload["component_id"]),
                tuple(str(node_id) for node_id in payload.get("execute_node_ids", [])),
            )
        )
        if str(payload["component_id"]).endswith("::static"):
            assert tuple(payload.get("execute_node_ids", [])) == ("1", "3")
            assert [output["proxy_output_name"] for output in payload.get("boundary_outputs", [])] == [
                "3_text",
                "static_input_0",
            ]
            return ("static-output", "shared-model")

        assert tuple(payload.get("execute_node_ids", [])) == ("7",)
        assert hydrated_inputs["static_input_0"] == "shared-model"
        assert [output["proxy_output_name"] for output in payload.get("boundary_outputs", [])] == ["7_text"]
        return (f"mapped:{hydrated_inputs['remote_input_1']}",)

    monkeypatch.setattr(
        mapped_execution_module,
        "_execute_subgraph_prompt",
        fake_execute_subgraph_prompt,
    )

    payload = {
        "payload_kind": "mapped_subgraph",
        "component_id": "1",
        "prompt_id": "prompt-1",
        "mapped_input": {"proxy_input_name": "remote_input_1", "io_type": "STRING"},
        "static_to_mapped_boundaries": [
            {
                "proxy_name": "static_input_0",
                "node_id": "1",
                "output_index": 0,
                "io_type": "MODEL",
                "is_list": False,
                "targets": [{"node_id": "7", "input_name": "model"}],
            }
        ],
        "static_phase": {
            "component_node_ids": ["1", "3"],
            "subgraph_prompt": {
                "1": {"class_type": "RemoteModel", "inputs": {}},
                "3": {"class_type": "RemoteSampler", "inputs": {"model": ["1", 0]}},
            },
            "boundary_inputs": [],
            "boundary_outputs": [
                {
                    "proxy_output_name": "3_text",
                    "node_id": "3",
                    "output_index": 0,
                    "io_type": "STRING",
                    "is_list": False,
                },
                {
                    "proxy_output_name": "static_input_0",
                    "node_id": "1",
                    "output_index": 0,
                    "io_type": "MODEL",
                    "is_list": False,
                },
            ],
            "execute_node_ids": ["1", "3"],
        },
        "mapped_phase": {
            "component_node_ids": ["6", "7"],
            "subgraph_prompt": {
                "6": {"class_type": "ModalMapInput", "inputs": {"value": ["remote_input_1", 0]}},
                "7": {
                    "class_type": "RemoteSampler",
                    "inputs": {"model": ["static_input_0", 0], "latent": ["6", 0]},
                },
            },
            "boundary_inputs": [
                {
                    "proxy_input_name": "remote_input_1",
                    "io_type": "STRING",
                    "targets": [{"node_id": "6", "input_name": "value"}],
                },
                {
                    "proxy_input_name": "static_input_0",
                    "io_type": "MODEL",
                    "targets": [{"node_id": "7", "input_name": "model"}],
                },
            ],
            "boundary_outputs": [
                {
                    "proxy_output_name": "7_text",
                    "node_id": "7",
                    "output_index": 0,
                    "io_type": "STRING",
                    "is_list": False,
                    "mapped_output": True,
                }
            ],
            "execute_node_ids": ["7"],
        },
        "boundary_outputs": [
            {
                "proxy_output_name": "3_text",
                "node_id": "3",
                "output_index": 0,
                "io_type": "STRING",
                "is_list": False,
                "mapped_output": False,
            },
            {
                "proxy_output_name": "7_text",
                "node_id": "7",
                "output_index": 0,
                "io_type": "STRING",
                "is_list": False,
                "mapped_output": True,
            },
        ],
        "execute_node_ids": ["3", "7"],
        "static_execute_node_ids": ["1", "3"],
        "mapped_execute_node_ids": ["7"],
        "extra_data": {"client_id": "client-1"},
    }

    response = asyncio.run(
        remote_modal_app_module._invoke_mapped_remote_engine_async(
            payload,
            serialization_module.serialize_node_inputs(
                {"remote_input_1": ["a", "b"]}
            ),
        )
    )

    assert serialization_module.deserialize_node_outputs(response) == (
        "static-output",
        ["mapped:a", "mapped:b"],
    )
    assert observed_execute_node_ids[0] == ("1::static", ("1", "3"))
    assert observed_execute_node_ids[1:] == [
        ("1::item:0", ("7",)),
        ("1::item:1", ("7",)),
    ]

def test_modal_cloud_execute_mapped_subgraph_payload_injects_static_bridges(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """The cloud runtime should execute the static phase once and feed its outputs into each mapped item."""
    observed_calls: list[tuple[str, tuple[str, ...], dict[str, Any]]] = []

    def fake_execute_subgraph_prompt(
        payload: dict[str, Any],
        hydrated_inputs: dict[str, Any],
        custom_nodes_root: Any,
        status_callback: Any = None,
        cancellation_event: Any = None,
        interrupt_store: Any = None,
        interrupt_flag_key: Any = None,
    ) -> tuple[str, ...]:
        observed_calls.append(
            (
                str(payload["component_id"]),
                tuple(str(node_id) for node_id in payload.get("execute_node_ids", [])),
                dict(hydrated_inputs),
            )
        )
        if str(payload["component_id"]).endswith("::static"):
            return ("static-output", "shared-model")
        return (f"mapped:{hydrated_inputs['remote_input_1']}:{hydrated_inputs['static_input_0']}",)

    monkeypatch.setattr(
        _cloud_mapped_execution_owner(),
        "_execute_prompt_subgraph",
        fake_execute_subgraph_prompt,
    )

    payload = {
        "payload_kind": "mapped_subgraph",
        "component_id": "cloud-1",
        "prompt_id": "prompt-1",
        "mapped_input": {"proxy_input_name": "remote_input_1", "io_type": "STRING"},
        "static_to_mapped_boundaries": [
            {
                "proxy_name": "static_input_0",
                "node_id": "1",
                "output_index": 0,
                "io_type": "MODEL",
                "is_list": False,
                "targets": [{"node_id": "7", "input_name": "model"}],
            }
        ],
        "static_phase": {
            "component_node_ids": ["1", "3"],
            "subgraph_prompt": {
                "1": {"class_type": "RemoteModel", "inputs": {}},
                "3": {"class_type": "RemoteSampler", "inputs": {"model": ["1", 0]}},
            },
            "boundary_inputs": [],
            "boundary_outputs": [
                {
                    "proxy_output_name": "3_text",
                    "node_id": "3",
                    "output_index": 0,
                    "io_type": "STRING",
                    "is_list": False,
                },
                {
                    "proxy_output_name": "static_input_0",
                    "node_id": "1",
                    "output_index": 0,
                    "io_type": "MODEL",
                    "is_list": False,
                },
            ],
            "execute_node_ids": ["1", "3"],
        },
        "mapped_phase": {
            "component_node_ids": ["6", "7"],
            "subgraph_prompt": {
                "6": {"class_type": "ModalMapInput", "inputs": {"value": ["remote_input_1", 0]}},
                "7": {
                    "class_type": "RemoteSampler",
                    "inputs": {"model": ["static_input_0", 0], "latent": ["6", 0]},
                },
            },
            "boundary_inputs": [
                {
                    "proxy_input_name": "remote_input_1",
                    "io_type": "STRING",
                    "targets": [{"node_id": "6", "input_name": "value"}],
                },
                {
                    "proxy_input_name": "static_input_0",
                    "io_type": "MODEL",
                    "targets": [{"node_id": "7", "input_name": "model"}],
                },
            ],
            "boundary_outputs": [
                {
                    "proxy_output_name": "7_text",
                    "node_id": "7",
                    "output_index": 0,
                    "io_type": "STRING",
                    "is_list": False,
                    "mapped_output": True,
                }
            ],
            "execute_node_ids": ["7"],
        },
        "boundary_outputs": [
            {
                "proxy_output_name": "3_text",
                "node_id": "3",
                "output_index": 0,
                "io_type": "STRING",
                "is_list": False,
                "mapped_output": False,
            },
            {
                "proxy_output_name": "7_text",
                "node_id": "7",
                "output_index": 0,
                "io_type": "STRING",
                "is_list": False,
                "mapped_output": True,
            },
        ],
        "extra_data": {"client_id": "client-1"},
    }

    outputs = modal_cloud_module._execute_mapped_subgraph_payload(
        payload,
        {"remote_input_1": ["a", "b"]},
        None,
    )

    assert outputs == (
        "static-output",
        ["mapped:a:shared-model", "mapped:b:shared-model"],
    )
    assert observed_calls == [
        ("cloud-1::static", ("1", "3"), {}),
        ("cloud-1::item:0", ("7",), {"static_input_0": "shared-model", "remote_input_1": "a"}),
        ("cloud-1::item:1", ("7",), {"static_input_0": "shared-model", "remote_input_1": "b"}),
    ]

def test_modal_cloud_execute_mapped_subgraph_payload_preserves_assigned_lane_id(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """Mapped cloud progress should keep the caller-assigned lane id instead of collapsing to `0`."""
    observed_progress_events: list[dict[str, Any]] = []

    def fake_execute_subgraph_prompt(
        payload: dict[str, Any],
        hydrated_inputs: dict[str, Any],
        custom_nodes_root: Any,
        status_callback: Any = None,
        cancellation_event: Any = None,
        interrupt_store: Any = None,
        interrupt_flag_key: Any = None,
    ) -> tuple[str, ...]:
        del custom_nodes_root, cancellation_event, interrupt_store, interrupt_flag_key
        if status_callback is not None:
            status_callback(
                {
                    "event_type": "node_progress",
                    "node_id": "12",
                    "display_node_id": "12",
                    "real_node_id": "12",
                    "value": 3.0,
                    "max": 9.0,
                }
            )
        return (f"mapped:{hydrated_inputs['remote_input_1']}",)

    monkeypatch.setattr(
        _cloud_mapped_execution_owner(),
        "_execute_prompt_subgraph",
        fake_execute_subgraph_prompt,
    )

    payload = {
        "payload_kind": "mapped_subgraph",
        "component_id": "cloud-2",
        "prompt_id": "prompt-1",
        "mapped_progress_lane_id": "3",
        "mapped_input": {"proxy_input_name": "remote_input_1", "io_type": "STRING"},
        "static_to_mapped_boundaries": [],
        "mapped_phase": {
            "component_node_ids": ["12", "39"],
            "subgraph_prompt": {
                "39": {
                    "class_type": "RemoteSampler",
                    "inputs": {"latent": ["remote_input_1", 0]},
                },
            },
            "boundary_inputs": [
                {
                    "proxy_input_name": "remote_input_1",
                    "io_type": "STRING",
                    "targets": [{"node_id": "39", "input_name": "latent"}],
                }
            ],
            "boundary_outputs": [
                {
                    "proxy_output_name": "39_text",
                    "node_id": "39",
                    "output_index": 0,
                    "io_type": "STRING",
                    "is_list": False,
                    "mapped_output": True,
                }
            ],
            "execute_node_ids": ["39"],
        },
        "boundary_outputs": [
            {
                "proxy_output_name": "39_text",
                "node_id": "39",
                "output_index": 0,
                "io_type": "STRING",
                "is_list": False,
                "mapped_output": True,
            },
        ],
    }

    outputs = modal_cloud_module._execute_mapped_subgraph_payload(
        payload,
        {"remote_input_1": ["a"]},
        None,
        status_callback=lambda event: observed_progress_events.append(dict(event)),
    )

    assert outputs == (["mapped:a"],)
    assert any(
        event.get("event_type") == "node_progress"
        and event.get("real_node_id") == "12"
        and event.get("lane_id") == "3"
        for event in observed_progress_events
    )
    assert any(
        event.get("event_type") == "node_progress"
        and event.get("clear") is True
        and event.get("lane_id") == "3"
        for event in observed_progress_events
    )

def test_mapped_component_with_local_reentry_rewrites_to_ordered_acyclic_proxies(
    api_intercept_module: Any,
    modal_executor_module: Any,
    settings_module: Any,
    sync_engine_module: Any,
    tmp_path: Path,
) -> None:
    """A mapped input should not collapse local feedback back into the same remote proxy."""
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
                "RemoteModel": _FakeRewriteRemoteModelNode,
                "RemoteSampler": _FakeRewriteRemoteSamplerNode,
                "LatentSource": _FakeRewriteLatentSourceNode,
                "ModalMapInput": _FakeRewriteModalMapInputNode,
                "LocalFeedback": _FakeRewriteLocalFeedbackNode,
                "LocalSink": _FakeRewriteLocalSinkNode,
            },
            "NODE_DISPLAY_NAME_MAPPINGS": {},
        },
    )()
    workflow = {
        "nodes": [
            {"id": 1, "properties": {"is_modal_remote": True}},
            {"id": 2, "properties": {"is_modal_remote": False}},
            {"id": 3, "properties": {"is_modal_remote": True}},
            {"id": 4, "properties": {"is_modal_remote": False}},
            {"id": 5, "properties": {"is_modal_remote": False}},
            {"id": 6, "properties": {"is_modal_remote": False}},
            {"id": 7, "properties": {"is_modal_remote": True}},
            {"id": 8, "properties": {"is_modal_remote": False}},
            {"id": 9, "properties": {"is_modal_remote": False}},
        ]
    }
    prompt = {
        "1": {
            "class_type": "RemoteModel",
            "inputs": {},
            "_meta": {"title": "Shared Model"},
        },
        "2": {
            "class_type": "LatentSource",
            "inputs": {},
            "_meta": {"title": "Single Latent"},
        },
        "3": {
            "class_type": "RemoteSampler",
            "inputs": {"model": ["1", 0], "latent": ["2", 0]},
            "_meta": {"title": "First Sampler"},
        },
        "4": {
            "class_type": "LocalFeedback",
            "inputs": {"image": ["3", 0]},
            "_meta": {"title": "Local Feedback"},
        },
        "5": {
            "class_type": "LatentSource",
            "inputs": {},
            "_meta": {"title": "Batch Values"},
        },
        "6": {
            "class_type": "ModalMapInput",
            "inputs": {"value": ["5", 0]},
            "_meta": {"title": "Map Input"},
        },
        "7": {
            "class_type": "RemoteSampler",
            "inputs": {"model": ["1", 0], "latent": ["6", 0], "prompt": ["4", 0]},
            "_meta": {"title": "Mapped Sampler"},
        },
        "8": {
            "class_type": "LocalSink",
            "inputs": {"image": ["7", 0]},
            "_meta": {"title": "Local Sink"},
        },
        "9": {
            "class_type": "LocalSink",
            "inputs": {"image": ["3", 0]},
            "_meta": {"title": "Independent Local Preview"},
        },
    }

    rewritten_prompt, summary = api_intercept_module.rewrite_prompt_for_modal(
        prompt=prompt,
        workflow=workflow,
        sync_engine=sync_engine,
        settings=settings,
        nodes_module=fake_nodes_module,
    )

    assert api_intercept_module._find_prompt_dependency_cycles(rewritten_prompt) == []
    assert summary.mapped_component_ids == ["7"]
    assert summary.component_execution_stages == [["3"], ["7"]]
    assert rewritten_prompt["4"]["inputs"]["image"] == ["3", 0]
    assert rewritten_prompt["7"]["inputs"]["remote_input_2"] == ["4", 0]
    assert len(summary.parallel_local_branch_node_ids) == 1
    passthrough_node_id = summary.parallel_local_branch_node_ids[0]
    passthrough_node = rewritten_prompt[passthrough_node_id]
    assert passthrough_node["class_type"] == (
        api_intercept_module.MODAL_PARALLEL_LOCAL_PASSTHROUGH_NODE_ID
    )
    assert passthrough_node["inputs"]["value"] == ["3", 0]
    dispatch_context = passthrough_node["inputs"]["dispatch_context"]
    assert dispatch_context["component_ids"] == ["7"]
    assert rewritten_prompt["9"]["inputs"]["image"] == [passthrough_node_id, 0]
    first_payload = rewritten_prompt["3"]["inputs"]["original_node_data"]
    mapped_payload = rewritten_prompt["7"]["inputs"]["original_node_data"]
    assert first_payload["component_node_ids"] == ["1", "3"]
    assert mapped_payload["component_node_ids"] == ["7"]
    assert first_payload["remote_local_gap_pool"] is True
    assert first_payload["keepalive_after_remote_component"] is True
    assert "stop_local_gap_keepalive_before_remote_component" not in first_payload
    assert mapped_payload["remote_local_gap_pool"] is True
    assert "keepalive_after_remote_component" not in mapped_payload
    assert mapped_payload["stop_local_gap_keepalive_before_remote_component"] is True
    assert mapped_payload["parallel_local_dispatch_group_id"] == (
        dispatch_context["dispatch_group_id"]
    )
    assert mapped_payload["signal_parallel_local_dispatch"] is True
    rehydrated_mapped_payload = modal_executor_module._rehydrate_proxy_payload(
        mapped_payload,
        unique_id="7",
    )
    assert rehydrated_mapped_payload["signal_parallel_local_dispatch"] is True
    assert {
        boundary_input["proxy_input_name"]
        for boundary_input in mapped_payload["boundary_inputs"]
    } == {"phase_bridge_0", "remote_input_1", "remote_input_2"}

def test_invoke_implicitly_mapped_subgraph_async_zips_batched_boundary_inputs(
    remote_modal_app_module: Any,
    serialization_module: Any,
    monkeypatch: Any,
) -> None:
    """Ordinary remote subgraphs should fan out when multiple boundary inputs arrive batched."""
    observed_inputs: list[dict[str, Any]] = []

    async def fake_invoke_remote_engine_async(payload: dict[str, Any], kwargs_payload: bytes) -> bytes:
        assert payload["payload_kind"] == "subgraph"
        assert payload["suppress_status_stream"] is True
        hydrated_inputs = serialization_module.deserialize_node_inputs(kwargs_payload)
        observed_inputs.append(hydrated_inputs)
        return serialization_module.serialize_node_outputs(
            (f"{hydrated_inputs['remote_input_0']}:{hydrated_inputs['remote_input_1']}",)
        )

    monkeypatch.setattr(
        remote_modal_app_module,
        "invoke_remote_engine_async",
        fake_invoke_remote_engine_async,
    )

    payload = {
        "payload_kind": "subgraph",
        "component_id": "12",
        "prompt_id": "prompt-1",
        "execute_node_ids": ["12"],
        "subgraph_prompt": {
            "12": {
                "class_type": "KSampler",
                "inputs": {
                    "latent_image": ["remote_input_0", 0],
                    "seed": ["remote_input_1", 0],
                },
            }
        },
        "boundary_inputs": [
            {
                "proxy_input_name": "remote_input_0",
                "io_type": "LATENT",
                "targets": [{"node_id": "12", "input_name": "latent_image"}],
            },
            {
                "proxy_input_name": "remote_input_1",
                "io_type": "INT",
                "targets": [{"node_id": "12", "input_name": "seed"}],
            },
        ],
        "boundary_outputs": [{"node_id": "12", "io_type": "STRING", "is_list": False}],
        "extra_data": {"client_id": "client-1"},
    }

    response = asyncio.run(
        remote_modal_app_module._invoke_implicitly_mapped_subgraph_async(
            payload,
            serialization_module.serialize_node_inputs(
                {
                    "remote_input_0": ["latent-a", "latent-b"],
                    "remote_input_1": [10, 11],
                }
            ),
        )
    )

    assert observed_inputs == [
        {"remote_input_0": "latent-a", "remote_input_1": 10},
        {"remote_input_0": "latent-b", "remote_input_1": 11},
    ]
    assert serialization_module.deserialize_node_outputs(response) == (
        ["latent-a:10", "latent-b:11"],
    )

def test_implicit_mapping_preserves_create_video_frame_sequence(
    remote_modal_app_module: Any,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """CreateVideo should receive one complete IMAGE frame sequence."""
    torch = pytest.importorskip("torch")
    frame_sequence = torch.zeros((124, 8, 8, 3), dtype=torch.float32)
    payload = {
        "payload_kind": "subgraph",
        "component_id": "105:91",
        "subgraph_prompt": {
            "105:91": {
                "class_type": "CreateVideo",
                "inputs": {"images": ["105:10", 0]},
            }
        },
        "boundary_inputs": [
            {
                "proxy_input_name": "remote_input_0",
                "io_type": "IMAGE",
                "targets": [{"node_id": "105:91", "input_name": "images"}],
            }
        ],
    }
    caplog.set_level(logging.INFO)

    split_inputs = remote_modal_app_module._split_batch_boundary_inputs(
        payload,
        {"remote_input_0": frame_sequence},
    )

    assert split_inputs is None
    assert "target sockets consume the complete tensor batch" in caplog.text
    assert "105:91.images" in caplog.text

def test_implicitly_mapped_subgraph_shared_model_keeps_unbatched_sampler_single_run(
    remote_modal_app_module: Any,
    serialization_module: Any,
    monkeypatch: Any,
) -> None:
    """A shared MODEL with mixed batch-size INT seeds should run sampler 4 once and sampler 12 four times."""
    observed_calls: list[tuple[str, tuple[str, ...], dict[str, Any]]] = []

    async def fake_invoke_remote_engine_async(payload: dict[str, Any], kwargs_payload: bytes) -> bytes:
        hydrated_inputs = serialization_module.deserialize_node_inputs(kwargs_payload)
        execute_node_ids = tuple(str(node_id) for node_id in payload.get("execute_node_ids", []))
        observed_calls.append((str(payload["component_id"]), execute_node_ids, hydrated_inputs))

        if execute_node_ids == ("4",):
            return serialization_module.serialize_node_outputs(("sampler-4",))
        if execute_node_ids == ("12",):
            return serialization_module.serialize_node_outputs((f"sampler-12:{hydrated_inputs['remote_input_0']}",))
        raise AssertionError(f"Unexpected execute nodes for implicit mapped regression: {execute_node_ids!r}")

    monkeypatch.setattr(
        remote_modal_app_module,
        "invoke_remote_engine_async",
        fake_invoke_remote_engine_async,
    )

    payload = {
        "payload_kind": "subgraph",
        "component_id": "17",
        "prompt_id": "prompt-1",
        "component_node_ids": ["4", "12", "17"],
        "execute_node_ids": ["4", "12"],
        "subgraph_prompt": {
            "17": {"class_type": "LoraLoaderModelOnly", "inputs": {}},
            "4": {"class_type": "KSampler", "inputs": {"model": ["17", 0], "seed": 0}},
            "12": {
                "class_type": "KSampler",
                "inputs": {"model": ["17", 0], "seed": 0},
            },
        },
        "boundary_inputs": [
            {
                "proxy_input_name": "remote_input_0",
                "io_type": "INT",
                "targets": [{"node_id": "12", "input_name": "seed"}],
            },
            {
                "proxy_input_name": "remote_input_1",
                "io_type": "INT",
                "targets": [{"node_id": "4", "input_name": "seed"}],
            },
        ],
        "boundary_outputs": [
            {"node_id": "4", "io_type": "STRING", "is_list": False},
            {"node_id": "12", "io_type": "STRING", "is_list": False},
        ],
        "extra_data": {"client_id": "client-1"},
    }

    response = asyncio.run(
        remote_modal_app_module._invoke_implicitly_mapped_subgraph_async(
            payload,
            serialization_module.serialize_node_inputs(
                {
                    "remote_input_0": [10, 11, 12, 13],
                    "remote_input_1": [28],
                }
            ),
        )
    )

    assert serialization_module.deserialize_node_outputs(response) == (
        "sampler-4",
        ["sampler-12:10", "sampler-12:11", "sampler-12:12", "sampler-12:13"],
    )
    assert observed_calls == [
        ("17::static", ("4",), {"remote_input_1": [28]}),
        ("17::item:0", ("12",), {"remote_input_0": 10, "remote_input_1": [28]}),
        ("17::item:1", ("12",), {"remote_input_0": 11, "remote_input_1": [28]}),
        ("17::item:2", ("12",), {"remote_input_0": 12, "remote_input_1": [28]}),
        ("17::item:3", ("12",), {"remote_input_0": 13, "remote_input_1": [28]}),
    ]

def test_implicitly_mapped_subgraph_seeds_remote_lanes_before_item_dispatch(
    remote_modal_app_module: Any,
    mapped_execution_module: Any,
    serialization_module: Any,
    session_state_module: Any,
    monkeypatch: Any,
) -> None:
    """Mapped remote scheduling should seed one bound worker lane before sending per-item calls there."""
    observed_calls: list[tuple[str, str, bool, tuple[str, ...], dict[str, Any]]] = []
    observed_lane_setup_starts: list[tuple[str, int]] = []

    monkeypatch.setenv("COMFY_MODAL_EXECUTION_MODE", "remote")
    monkeypatch.setattr(remote_modal_app_module, "modal", object())
    monkeypatch.setattr(mapped_execution_module, "modal", object())
    monkeypatch.setattr(remote_modal_app_module, "_mapped_execution_parallelism", lambda total_items: 2)
    monkeypatch.setattr(remote_modal_app_module, "ensure_remote_warm_capacity", lambda *args, **kwargs: 0)
    monkeypatch.setattr(
        mapped_execution_module,
        "_lookup_deployed_remote_engine",
        lambda payload, affinity_key_override=None: (
            f"engine:{affinity_key_override or remote_modal_app_module._remote_worker_affinity_key(payload)}"
        ),
    )

    async def fake_invoke_bound_remote_engine_async(
        remote_engine: Any,
        payload: dict[str, Any],
        kwargs_payload: bytes,
    ) -> bytes:
        hydrated_inputs = serialization_module.deserialize_node_inputs(kwargs_payload)
        execute_node_ids = tuple(str(node_id) for node_id in payload.get("execute_node_ids", []))
        observed_calls.append(
            (
                str(remote_engine),
                str(payload["component_id"]),
                bool(payload.get("clear_remote_session")),
                execute_node_ids,
                hydrated_inputs,
            )
        )
        if str(payload["component_id"]).startswith("17::seed:"):
            return serialization_module.serialize_node_outputs(())
        if str(payload["component_id"]).startswith("17::cleanup:"):
            return serialization_module.serialize_node_outputs(())
        if execute_node_ids == ("12",):
            await asyncio.sleep(0)
            return serialization_module.serialize_node_outputs((f"{remote_engine}:{hydrated_inputs['remote_input_0']}",))
        raise AssertionError(f"Unexpected seeded-lane payload: {payload!r}")

    monkeypatch.setattr(
        remote_modal_app_module,
        "_invoke_bound_remote_engine_async",
        fake_invoke_bound_remote_engine_async,
    )
    monkeypatch.setattr(
        mapped_execution_module,
        "_emit_local_mapped_lane_progress_start",
        lambda payload, lane_index, item_index=None: observed_lane_setup_starts.append(
            (str(payload["component_id"]), int(lane_index))
        ),
    )

    payload = {
        "payload_kind": "subgraph",
        "component_id": "17",
        "prompt_id": "prompt-1",
        "component_node_ids": ["12"],
        "execute_node_ids": ["12"],
        "subgraph_prompt": {
            "12": {
                "class_type": "KSampler",
                "inputs": {"model": 0, "seed": 0},
            },
        },
        "boundary_inputs": [
            {
                "proxy_input_name": "remote_input_0",
                "io_type": "INT",
                "targets": [{"node_id": "12", "input_name": "seed"}],
            },
            {
                "proxy_input_name": "static_input_0",
                "io_type": "MODEL",
                "targets": [{"node_id": "12", "input_name": "model"}],
            },
        ],
        "boundary_outputs": [
            {"node_id": "12", "io_type": "STRING", "is_list": False},
        ],
        "extra_data": {"client_id": "client-1"},
        "remote_session": session_state_module.RemoteSessionHandle(
            session_id="session-1",
            prompt_id="prompt-1",
            owner_component_id="17",
        ).to_payload(),
        "clear_remote_session": True,
        "static_to_mapped_boundaries": [
            {
                "proxy_name": "static_input_0",
                "node_id": "4",
                "output_index": 0,
                "io_type": "MODEL",
                "is_list": False,
                "targets": [{"node_id": "12", "input_name": "model"}],
            }
        ],
        "static_phase": {
            "component_node_ids": ["4"],
            "subgraph_prompt": {
                "4": {"class_type": "LoraLoaderModelOnly", "inputs": {}},
            },
            "boundary_inputs": [],
            "boundary_outputs": [
                {
                    "proxy_output_name": "static_input_0",
                    "node_id": "4",
                    "output_index": 0,
                    "io_type": "MODEL",
                    "is_list": False,
                    "session_output": True,
                    "preview_target_node_ids": [],
                }
            ],
            "execute_node_ids": ["4"],
        },
    }

    response = asyncio.run(
        remote_modal_app_module._invoke_implicitly_mapped_subgraph_async(
            payload,
            serialization_module.serialize_node_inputs(
                {
                    "remote_input_0": [10, 11, 12, 13],
                    "static_input_0": {"__comfy_modal_remote_session_bridge_ref__": True},
                }
            ),
        )
    )

    seed_calls = [call for call in observed_calls if call[1].startswith("17::seed:")]
    item_calls = [call for call in observed_calls if call[1].startswith("17::item:")]
    cleanup_calls = [call for call in observed_calls if call[1].startswith("17::cleanup:")]

    assert sorted(seed_calls) == [
        (
            "engine:worker-pool:slot:0",
            "17::seed:0",
            False,
            ("4",),
            {"static_input_0": {"__comfy_modal_remote_session_bridge_ref__": True}},
        ),
        (
            "engine:worker-pool:slot:1",
            "17::seed:1",
            False,
            ("4",),
            {"static_input_0": {"__comfy_modal_remote_session_bridge_ref__": True}},
        ),
    ]
    assert {call[0] for call in item_calls} <= {"engine:worker-pool:slot:0", "engine:worker-pool:slot:1"}
    assert {call[0] for call in item_calls} <= {call[0] for call in seed_calls}
    assert sorted(cleanup_calls) == [
        ("engine:worker-pool:slot:0", "17::cleanup:0", True, (), {}),
        ("engine:worker-pool:slot:1", "17::cleanup:1", True, (), {}),
    ]
    assert sorted(observed_lane_setup_starts) == [("17", 0), ("17", 1)]
    response_outputs = serialization_module.deserialize_node_outputs(response)[0]
    assert len(response_outputs) == 4
    assert [output.rsplit(":", 1)[-1] for output in response_outputs] == ["10", "11", "12", "13"]

def test_implicitly_mapped_subgraph_allows_ready_lanes_to_start_before_slowest_seed_finishes(
    remote_modal_app_module: Any,
    mapped_execution_module: Any,
    serialization_module: Any,
    session_state_module: Any,
    monkeypatch: Any,
) -> None:
    """A fast seeded lane should start item execution before another lane finishes seeding."""
    observed_events: list[tuple[str, str, str]] = []

    monkeypatch.setenv("COMFY_MODAL_EXECUTION_MODE", "remote")
    monkeypatch.setattr(remote_modal_app_module, "modal", object())
    monkeypatch.setattr(mapped_execution_module, "modal", object())
    monkeypatch.setattr(remote_modal_app_module, "_mapped_execution_parallelism", lambda total_items: 2)
    monkeypatch.setattr(remote_modal_app_module, "ensure_remote_warm_capacity", lambda *args, **kwargs: 0)
    monkeypatch.setattr(
        mapped_execution_module,
        "_lookup_deployed_remote_engine",
        lambda payload, affinity_key_override=None: (
            f"engine:{affinity_key_override or remote_modal_app_module._remote_worker_affinity_key(payload)}"
        ),
    )

    async def fake_invoke_bound_remote_engine_async(
        remote_engine: Any,
        payload: dict[str, Any],
        kwargs_payload: bytes,
    ) -> bytes:
        hydrated_inputs = serialization_module.deserialize_node_inputs(kwargs_payload)
        component_id = str(payload["component_id"])
        if component_id == "17::seed:0":
            observed_events.append(("seed_start", component_id, str(remote_engine)))
            await asyncio.sleep(0.05)
            observed_events.append(("seed_done", component_id, str(remote_engine)))
            return serialization_module.serialize_node_outputs(())
        if component_id == "17::seed:1":
            observed_events.append(("seed_start", component_id, str(remote_engine)))
            await asyncio.sleep(0)
            observed_events.append(("seed_done", component_id, str(remote_engine)))
            return serialization_module.serialize_node_outputs(())
        if component_id.startswith("17::cleanup:"):
            observed_events.append(("cleanup", component_id, str(remote_engine)))
            return serialization_module.serialize_node_outputs(())
        if component_id.startswith("17::item:"):
            observed_events.append(("item", component_id, str(remote_engine)))
            return serialization_module.serialize_node_outputs(
                (f"{remote_engine}:{hydrated_inputs['remote_input_0']}",)
            )
        raise AssertionError(f"Unexpected seeded-lane payload: {payload!r}")

    monkeypatch.setattr(
        remote_modal_app_module,
        "_invoke_bound_remote_engine_async",
        fake_invoke_bound_remote_engine_async,
    )

    payload = {
        "payload_kind": "subgraph",
        "component_id": "17",
        "prompt_id": "prompt-1",
        "component_node_ids": ["12"],
        "execute_node_ids": ["12"],
        "subgraph_prompt": {
            "12": {
                "class_type": "KSampler",
                "inputs": {"model": 0, "seed": 0},
            },
        },
        "boundary_inputs": [
            {
                "proxy_input_name": "remote_input_0",
                "io_type": "INT",
                "targets": [{"node_id": "12", "input_name": "seed"}],
            },
            {
                "proxy_input_name": "static_input_0",
                "io_type": "MODEL",
                "targets": [{"node_id": "12", "input_name": "model"}],
            },
        ],
        "boundary_outputs": [
            {"node_id": "12", "io_type": "STRING", "is_list": False},
        ],
        "extra_data": {"client_id": "client-1"},
        "remote_session": session_state_module.RemoteSessionHandle(
            session_id="session-1",
            prompt_id="prompt-1",
            owner_component_id="17",
        ).to_payload(),
        "clear_remote_session": True,
        "static_to_mapped_boundaries": [
            {
                "proxy_name": "static_input_0",
                "node_id": "4",
                "output_index": 0,
                "io_type": "MODEL",
                "is_list": False,
                "targets": [{"node_id": "12", "input_name": "model"}],
            }
        ],
        "static_phase": {
            "component_node_ids": ["4"],
            "subgraph_prompt": {
                "4": {"class_type": "LoraLoaderModelOnly", "inputs": {}},
            },
            "boundary_inputs": [],
            "boundary_outputs": [
                {
                    "proxy_output_name": "static_input_0",
                    "node_id": "4",
                    "output_index": 0,
                    "io_type": "MODEL",
                    "is_list": False,
                    "session_output": True,
                    "preview_target_node_ids": [],
                }
            ],
            "execute_node_ids": ["4"],
        },
    }

    response = asyncio.run(
        remote_modal_app_module._invoke_implicitly_mapped_subgraph_async(
            payload,
            serialization_module.serialize_node_inputs(
                {
                    "remote_input_0": [10, 11, 12, 13],
                    "static_input_0": {"__comfy_modal_remote_session_bridge_ref__": True},
                }
            ),
        )
    )

    assert len(serialization_module.deserialize_node_outputs(response)[0]) == 4
    assert ("seed_done", "17::seed:1", "engine:worker-pool:slot:1") in observed_events
    first_item_index = next(
        index for index, event in enumerate(observed_events) if event[0] == "item"
    )
    fast_seed_done_index = observed_events.index(
        ("seed_done", "17::seed:1", "engine:worker-pool:slot:1")
    )
    assert first_item_index > fast_seed_done_index

def test_implicitly_mapped_subgraph_returns_once_all_items_finish_without_waiting_for_unused_lane(
    remote_modal_app_module: Any,
    mapped_execution_module: Any,
    serialization_module: Any,
    session_state_module: Any,
    monkeypatch: Any,
) -> None:
    """Prompt completion should not wait for a slow seeded lane after other lanes finish all mapped items."""
    observed_events: list[tuple[str, str, str]] = []

    monkeypatch.setenv("COMFY_MODAL_EXECUTION_MODE", "remote")
    monkeypatch.setattr(remote_modal_app_module, "modal", object())
    monkeypatch.setattr(mapped_execution_module, "modal", object())
    monkeypatch.setattr(remote_modal_app_module, "_mapped_execution_parallelism", lambda total_items: 2)
    monkeypatch.setattr(remote_modal_app_module, "ensure_remote_warm_capacity", lambda *args, **kwargs: 0)
    monkeypatch.setattr(
        mapped_execution_module,
        "_lookup_deployed_remote_engine",
        lambda payload, affinity_key_override=None: (
            f"engine:{affinity_key_override or remote_modal_app_module._remote_worker_affinity_key(payload)}"
        ),
    )

    async def fake_invoke_bound_remote_engine_async(
        remote_engine: Any,
        payload: dict[str, Any],
        kwargs_payload: bytes,
    ) -> bytes:
        hydrated_inputs = serialization_module.deserialize_node_inputs(kwargs_payload)
        component_id = str(payload["component_id"])
        if component_id == "17::seed:0":
            observed_events.append(("seed_start", component_id, str(remote_engine)))
            await asyncio.sleep(1.0)
            observed_events.append(("seed_done", component_id, str(remote_engine)))
            return serialization_module.serialize_node_outputs(())
        if component_id == "17::seed:1":
            observed_events.append(("seed_start", component_id, str(remote_engine)))
            await asyncio.sleep(0)
            observed_events.append(("seed_done", component_id, str(remote_engine)))
            return serialization_module.serialize_node_outputs(())
        if component_id.startswith("17::cleanup:"):
            observed_events.append(("cleanup", component_id, str(remote_engine)))
            return serialization_module.serialize_node_outputs(())
        if component_id.startswith("17::item:"):
            observed_events.append(("item", component_id, str(remote_engine)))
            return serialization_module.serialize_node_outputs(
                (f"{remote_engine}:{hydrated_inputs['remote_input_0']}",)
            )
        raise AssertionError(f"Unexpected seeded-lane payload: {payload!r}")

    monkeypatch.setattr(
        remote_modal_app_module,
        "_invoke_bound_remote_engine_async",
        fake_invoke_bound_remote_engine_async,
    )

    payload = {
        "payload_kind": "subgraph",
        "component_id": "17",
        "prompt_id": "prompt-1",
        "component_node_ids": ["12"],
        "execute_node_ids": ["12"],
        "subgraph_prompt": {
            "12": {
                "class_type": "KSampler",
                "inputs": {"model": 0, "seed": 0},
            },
        },
        "boundary_inputs": [
            {
                "proxy_input_name": "remote_input_0",
                "io_type": "INT",
                "targets": [{"node_id": "12", "input_name": "seed"}],
            },
            {
                "proxy_input_name": "static_input_0",
                "io_type": "MODEL",
                "targets": [{"node_id": "12", "input_name": "model"}],
            },
        ],
        "boundary_outputs": [
            {"node_id": "12", "io_type": "STRING", "is_list": False},
        ],
        "extra_data": {"client_id": "client-1"},
        "remote_session": session_state_module.RemoteSessionHandle(
            session_id="session-1",
            prompt_id="prompt-1",
            owner_component_id="17",
        ).to_payload(),
        "clear_remote_session": True,
        "static_to_mapped_boundaries": [
            {
                "proxy_name": "static_input_0",
                "node_id": "4",
                "output_index": 0,
                "io_type": "MODEL",
                "is_list": False,
                "targets": [{"node_id": "12", "input_name": "model"}],
            }
        ],
        "static_phase": {
            "component_node_ids": ["4"],
            "subgraph_prompt": {
                "4": {"class_type": "LoraLoaderModelOnly", "inputs": {}},
            },
            "boundary_inputs": [],
            "boundary_outputs": [
                {
                    "proxy_output_name": "static_input_0",
                    "node_id": "4",
                    "output_index": 0,
                    "io_type": "MODEL",
                    "is_list": False,
                    "session_output": True,
                    "preview_target_node_ids": [],
                }
            ],
            "execute_node_ids": ["4"],
        },
    }

    started_at = time.perf_counter()
    response = asyncio.run(
        remote_modal_app_module._invoke_implicitly_mapped_subgraph_async(
            payload,
            serialization_module.serialize_node_inputs(
                {
                    "remote_input_0": [10, 11],
                    "static_input_0": {"__comfy_modal_remote_session_bridge_ref__": True},
                }
            ),
        )
    )
    elapsed = time.perf_counter() - started_at

    assert serialization_module.deserialize_node_outputs(response)[0] == [
        "engine:worker-pool:slot:1:10",
        "engine:worker-pool:slot:1:11",
    ]
    assert elapsed < 0.5
    assert ("item", "17::item:0", "engine:worker-pool:slot:1") in observed_events
    assert ("cleanup", "17::cleanup:1", "engine:worker-pool:slot:1") in observed_events
    assert ("cleanup", "17::cleanup:0", "engine:worker-pool:slot:0") not in observed_events
    assert ("seed_done", "17::seed:0", "engine:worker-pool:slot:0") not in observed_events

def test_implicitly_mapped_subgraph_ignores_late_lane_failure_after_all_items_complete(
    remote_modal_app_module: Any,
    mapped_execution_module: Any,
    serialization_module: Any,
    session_state_module: Any,
    monkeypatch: Any,
) -> None:
    """A detached late lane must not fail the prompt after all mapped outputs are already complete."""
    observed_events: list[tuple[str, str, str]] = []

    monkeypatch.setenv("COMFY_MODAL_EXECUTION_MODE", "remote")
    monkeypatch.setattr(remote_modal_app_module, "modal", object())
    monkeypatch.setattr(mapped_execution_module, "modal", object())
    monkeypatch.setattr(remote_modal_app_module, "_mapped_execution_parallelism", lambda total_items: 2)
    monkeypatch.setattr(remote_modal_app_module, "ensure_remote_warm_capacity", lambda *args, **kwargs: 0)
    monkeypatch.setattr(
        mapped_execution_module,
        "_lookup_deployed_remote_engine",
        lambda payload, affinity_key_override=None: (
            f"engine:{affinity_key_override or remote_modal_app_module._remote_worker_affinity_key(payload)}"
        ),
    )

    async def fake_invoke_bound_remote_engine_async(
        remote_engine: Any,
        payload: dict[str, Any],
        kwargs_payload: bytes,
    ) -> bytes:
        hydrated_inputs = serialization_module.deserialize_node_inputs(kwargs_payload)
        component_id = str(payload["component_id"])
        if component_id == "17::seed:0":
            observed_events.append(("seed_start", component_id, str(remote_engine)))
            await asyncio.sleep(0.05)
            raise RuntimeError("late lane failed after prompt completion")
        if component_id == "17::seed:1":
            observed_events.append(("seed_start", component_id, str(remote_engine)))
            await asyncio.sleep(0)
            observed_events.append(("seed_done", component_id, str(remote_engine)))
            return serialization_module.serialize_node_outputs(())
        if component_id.startswith("17::cleanup:"):
            observed_events.append(("cleanup", component_id, str(remote_engine)))
            return serialization_module.serialize_node_outputs(())
        if component_id.startswith("17::item:"):
            observed_events.append(("item", component_id, str(remote_engine)))
            return serialization_module.serialize_node_outputs(
                (f"{remote_engine}:{hydrated_inputs['remote_input_0']}",)
            )
        raise AssertionError(f"Unexpected seeded-lane payload: {payload!r}")

    monkeypatch.setattr(
        remote_modal_app_module,
        "_invoke_bound_remote_engine_async",
        fake_invoke_bound_remote_engine_async,
    )

    payload = {
        "payload_kind": "subgraph",
        "component_id": "17",
        "prompt_id": "prompt-1",
        "component_node_ids": ["12"],
        "execute_node_ids": ["12"],
        "subgraph_prompt": {
            "12": {
                "class_type": "KSampler",
                "inputs": {"model": 0, "seed": 0},
            },
        },
        "boundary_inputs": [
            {
                "proxy_input_name": "remote_input_0",
                "io_type": "INT",
                "targets": [{"node_id": "12", "input_name": "seed"}],
            },
            {
                "proxy_input_name": "static_input_0",
                "io_type": "MODEL",
                "targets": [{"node_id": "12", "input_name": "model"}],
            },
        ],
        "boundary_outputs": [
            {"node_id": "12", "io_type": "STRING", "is_list": False},
        ],
        "extra_data": {"client_id": "client-1"},
        "remote_session": session_state_module.RemoteSessionHandle(
            session_id="session-1",
            prompt_id="prompt-1",
            owner_component_id="17",
        ).to_payload(),
        "clear_remote_session": True,
        "static_to_mapped_boundaries": [
            {
                "proxy_name": "static_input_0",
                "node_id": "4",
                "output_index": 0,
                "io_type": "MODEL",
                "is_list": False,
                "targets": [{"node_id": "12", "input_name": "model"}],
            }
        ],
        "static_phase": {
            "component_node_ids": ["4"],
            "subgraph_prompt": {
                "4": {"class_type": "LoraLoaderModelOnly", "inputs": {}},
            },
            "boundary_inputs": [],
            "boundary_outputs": [
                {
                    "proxy_output_name": "static_input_0",
                    "node_id": "4",
                    "output_index": 0,
                    "io_type": "MODEL",
                    "is_list": False,
                    "session_output": True,
                    "preview_target_node_ids": [],
                }
            ],
            "execute_node_ids": ["4"],
        },
    }

    response = asyncio.run(
        remote_modal_app_module._invoke_implicitly_mapped_subgraph_async(
            payload,
            serialization_module.serialize_node_inputs(
                {
                    "remote_input_0": [10, 11],
                    "static_input_0": {"__comfy_modal_remote_session_bridge_ref__": True},
                }
            ),
        )
    )

    assert serialization_module.deserialize_node_outputs(response)[0] == [
        "engine:worker-pool:slot:1:10",
        "engine:worker-pool:slot:1:11",
    ]
    assert ("cleanup", "17::cleanup:1", "engine:worker-pool:slot:1") in observed_events
    assert ("cleanup", "17::cleanup:0", "engine:worker-pool:slot:0") not in observed_events

def test_implicitly_mapped_subgraph_skips_outer_fanout_for_input_is_list_targets(
    remote_modal_app_module: Any,
    mapped_execution_module: Any,
    serialization_module: Any,
    monkeypatch: Any,
) -> None:
    """Implicit fan-out should not split components whose list boundary lands on INPUT_IS_LIST nodes."""
    observed_calls: list[tuple[str, tuple[str, ...], dict[str, Any]]] = []

    fake_nodes_module = type(
        "FakeNodesModule",
        (),
        {
            "NODE_CLASS_MAPPINGS": {
                "ImplicitBatchListSource": _ImplicitBatchListSourceNode,
                "ImplicitBatchScalarConsumer": _ImplicitBatchScalarConsumerNode,
                "ImplicitBatchListConsumer": _ImplicitBatchListConsumerNode,
            }
        },
    )()
    monkeypatch.setattr(mapped_execution_module, "_load_nodes_module", lambda: fake_nodes_module)

    async def fake_invoke_remote_engine_async(
        payload: dict[str, Any],
        kwargs_payload: bytes,
        *,
        allow_implicit_mapping: bool = True,
    ) -> bytes:
        del allow_implicit_mapping
        hydrated_inputs = serialization_module.deserialize_node_inputs(kwargs_payload)
        execute_node_ids = tuple(str(node_id) for node_id in payload.get("execute_node_ids", []))
        observed_calls.append((str(payload["component_id"]), execute_node_ids, hydrated_inputs))
        return serialization_module.serialize_node_outputs(
            ("scalar-output", ["list-output:0", "list-output:1", "list-output:2"])
        )

    monkeypatch.setattr(
        remote_modal_app_module,
        "invoke_remote_engine_async",
        fake_invoke_remote_engine_async,
    )

    payload = {
        "payload_kind": "subgraph",
        "component_id": "17",
        "prompt_id": "prompt-1",
        "component_node_ids": ["4", "12", "17"],
        "execute_node_ids": ["4", "12"],
        "subgraph_prompt": {
            "17": {"class_type": "ImplicitBatchListSource", "inputs": {"values": 0}},
            "4": {
                "class_type": "ImplicitBatchScalarConsumer",
                "inputs": {"value": ["17", 0]},
            },
            "12": {
                "class_type": "ImplicitBatchListConsumer",
                "inputs": {"values": ["17", 1]},
            },
        },
        "boundary_inputs": [
            {
                "proxy_input_name": "remote_input_0",
                "io_type": "INT",
                "targets": [{"node_id": "17", "input_name": "values"}],
            }
        ],
        "boundary_outputs": [
            {"node_id": "4", "io_type": "STRING", "is_list": False},
            {"node_id": "12", "io_type": "STRING", "is_list": True},
        ],
        "extra_data": {"client_id": "client-1"},
    }

    response = asyncio.run(
        remote_modal_app_module._invoke_implicitly_mapped_subgraph_async(
            payload,
            serialization_module.serialize_node_inputs({"remote_input_0": [10, 11, 12]}),
        )
    )

    assert serialization_module.deserialize_node_outputs(response) == (
        "scalar-output",
        ["list-output:0", "list-output:1", "list-output:2"],
    )
    assert observed_calls == [
        ("17", ("4", "12"), {"remote_input_0": [10, 11, 12]}),
    ]

def test_invoke_remote_engine_async_bypasses_implicit_mapping_once_for_input_is_list_targets(
    remote_modal_app_module: Any,
    mapped_execution_module: Any,
    serialization_module: Any,
    monkeypatch: Any,
) -> None:
    """The async dispatcher should not recurse when implicit fan-out is suppressed."""
    observed_calls: list[tuple[bool, dict[str, Any]]] = []

    fake_nodes_module = type(
        "FakeNodesModule",
        (),
        {
            "NODE_CLASS_MAPPINGS": {
                "ImplicitBatchListSource": _ImplicitBatchListSourceNode,
                "ImplicitBatchScalarConsumer": _ImplicitBatchScalarConsumerNode,
                "ImplicitBatchListConsumer": _ImplicitBatchListConsumerNode,
            }
        },
    )()
    monkeypatch.setattr(mapped_execution_module, "_load_nodes_module", lambda: fake_nodes_module)

    def fake_invoke_remote_engine(
        payload: dict[str, Any],
        kwargs_payload: bytes,
        *,
        allow_implicit_mapping: bool = True,
    ) -> bytes:
        hydrated_inputs = serialization_module.deserialize_node_inputs(kwargs_payload)
        observed_calls.append((allow_implicit_mapping, hydrated_inputs))
        return serialization_module.serialize_node_outputs(("ordinary-subgraph",))

    monkeypatch.setattr(
        remote_modal_app_module,
        "invoke_remote_engine",
        fake_invoke_remote_engine,
    )

    payload = {
        "payload_kind": "subgraph",
        "component_id": "17",
        "prompt_id": "prompt-1",
        "component_node_ids": ["4", "12", "17"],
        "execute_node_ids": ["4", "12"],
        "subgraph_prompt": {
            "17": {"class_type": "ImplicitBatchListSource", "inputs": {"values": 0}},
            "4": {
                "class_type": "ImplicitBatchScalarConsumer",
                "inputs": {"value": ["17", 0]},
            },
            "12": {
                "class_type": "ImplicitBatchListConsumer",
                "inputs": {"values": ["17", 1]},
            },
        },
        "boundary_inputs": [
            {
                "proxy_input_name": "remote_input_0",
                "io_type": "INT",
                "targets": [{"node_id": "17", "input_name": "values"}],
            }
        ],
        "boundary_outputs": [
            {"node_id": "4", "io_type": "STRING", "is_list": False},
        ],
        "extra_data": {"client_id": "client-1"},
    }

    response = asyncio.run(
        remote_modal_app_module.invoke_remote_engine_async(
            payload,
            serialization_module.serialize_node_inputs({"remote_input_0": [10, 11, 12]}),
        )
    )

    assert serialization_module.deserialize_node_outputs(response) == ("ordinary-subgraph",)
    assert observed_calls == [
        (False, {"remote_input_0": [10, 11, 12]}),
    ]

def test_implicitly_mapped_subgraph_clears_remote_session_once_after_all_items_finish(
    remote_modal_app_module: Any,
    serialization_module: Any,
    session_state_module: Any,
    monkeypatch: Any,
) -> None:
    """Implicit mapped execution should reserve remote-session cleanup for one final cleanup payload."""
    observed_calls: list[tuple[str, bool, tuple[str, ...], dict[str, Any]]] = []
    session_cleared = False

    monkeypatch.setattr(remote_modal_app_module, "_mapped_execution_parallelism", lambda total_items: 2)

    async def fake_invoke_remote_engine_async(payload: dict[str, Any], kwargs_payload: bytes) -> bytes:
        nonlocal session_cleared
        hydrated_inputs = serialization_module.deserialize_node_inputs(kwargs_payload)
        execute_node_ids = tuple(str(node_id) for node_id in payload.get("execute_node_ids", []))
        clear_remote_session = bool(payload.get("clear_remote_session"))
        observed_calls.append(
            (str(payload["component_id"]), clear_remote_session, execute_node_ids, hydrated_inputs)
        )

        if clear_remote_session:
            session_cleared = True
        elif session_cleared:
            raise session_state_module.RemoteSessionStateError(
                "Remote session 'session-1' was not found."
            )

        if execute_node_ids == ("4",):
            return serialization_module.serialize_node_outputs(("sampler-4",))
        if execute_node_ids == ("12",):
            return serialization_module.serialize_node_outputs((f"sampler-12:{hydrated_inputs['remote_input_0']}",))
        if execute_node_ids == ():
            return serialization_module.serialize_node_outputs(())
        raise AssertionError(f"Unexpected execute nodes for implicit cleanup regression: {execute_node_ids!r}")

    monkeypatch.setattr(
        remote_modal_app_module,
        "invoke_remote_engine_async",
        fake_invoke_remote_engine_async,
    )

    payload = {
        "payload_kind": "subgraph",
        "component_id": "17",
        "prompt_id": "prompt-1",
        "component_node_ids": ["4", "12", "17"],
        "execute_node_ids": ["4", "12"],
        "subgraph_prompt": {
            "17": {"class_type": "LoraLoaderModelOnly", "inputs": {}},
            "4": {"class_type": "KSampler", "inputs": {"model": ["17", 0], "seed": 0}},
            "12": {
                "class_type": "KSampler",
                "inputs": {"model": ["17", 0], "seed": 0},
            },
        },
        "boundary_inputs": [
            {
                "proxy_input_name": "remote_input_0",
                "io_type": "INT",
                "targets": [{"node_id": "12", "input_name": "seed"}],
            },
            {
                "proxy_input_name": "remote_input_1",
                "io_type": "INT",
                "targets": [{"node_id": "4", "input_name": "seed"}],
            },
        ],
        "boundary_outputs": [
            {"node_id": "4", "io_type": "STRING", "is_list": False},
            {"node_id": "12", "io_type": "STRING", "is_list": False},
        ],
        "extra_data": {"client_id": "client-1"},
        "remote_session": session_state_module.RemoteSessionHandle(
            session_id="session-1",
            prompt_id="prompt-1",
            owner_component_id="17",
        ).to_payload(),
        "clear_remote_session": True,
    }

    response = asyncio.run(
        remote_modal_app_module._invoke_implicitly_mapped_subgraph_async(
            payload,
            serialization_module.serialize_node_inputs(
                {
                    "remote_input_0": [10, 11, 12],
                    "remote_input_1": [28],
                }
            ),
        )
    )

    assert serialization_module.deserialize_node_outputs(response) == (
        "sampler-4",
        ["sampler-12:10", "sampler-12:11", "sampler-12:12"],
    )
    assert [
        (component_id, execute_node_ids)
        for component_id, _, execute_node_ids, _ in observed_calls
    ] == [
        ("17::static", ("4",)),
        ("17::item:0", ("12",)),
        ("17::item:1", ("12",)),
        ("17::item:2", ("12",)),
        ("17::cleanup", ()),
    ]
    assert all(
        not clear_remote_session
        for component_id, clear_remote_session, _, _ in observed_calls
        if component_id != "17::cleanup"
    )
    assert observed_calls[-1] == ("17::cleanup", True, (), {})

def test_implicitly_mapped_subgraph_stops_queued_items_after_local_interrupt(
    remote_modal_app_module: Any,
    serialization_module: Any,
    session_state_module: Any,
    monkeypatch: Any,
) -> None:
    """Implicit mapped cancellation should not dispatch queued items or cleanup work."""

    class FakeInterrupt(Exception):
        """Stand-in for ComfyUI's InterruptProcessingException."""

    observed_calls: list[str] = []
    local_interrupted = False

    async def fake_invoke_remote_engine_async(payload: dict[str, Any], kwargs_payload: bytes) -> bytes:
        nonlocal local_interrupted
        del kwargs_payload
        component_id = str(payload["component_id"])
        observed_calls.append(component_id)
        if component_id.endswith("::cleanup"):
            raise AssertionError("interrupted mapped execution should not run cleanup")
        local_interrupted = True
        return serialization_module.serialize_node_outputs((component_id,))

    def fake_local_processing_interrupted() -> bool:
        return local_interrupted

    monkeypatch.setattr(
        remote_modal_app_module,
        "boost_mapped_component_warmup",
        lambda payload, *, total_items, reason: (1, 1),
    )
    monkeypatch.setattr(
        remote_modal_app_module,
        "invoke_remote_engine_async",
        fake_invoke_remote_engine_async,
    )
    monkeypatch.setattr(
        remote_modal_app_module,
        "_local_processing_interrupted",
        fake_local_processing_interrupted,
    )
    monkeypatch.setattr(
        remote_modal_app_module,
        "_raise_local_interrupt",
        lambda: (_ for _ in ()).throw(FakeInterrupt()),
    )

    payload = {
        "payload_kind": "subgraph",
        "component_id": "17",
        "prompt_id": "prompt-1",
        "component_node_ids": ["12"],
        "execute_node_ids": ["12"],
        "subgraph_prompt": {
            "12": {
                "class_type": "KSampler",
                "inputs": {"seed": 0},
            },
        },
        "boundary_inputs": [
            {
                "proxy_input_name": "remote_input_0",
                "io_type": "INT",
                "targets": [{"node_id": "12", "input_name": "seed"}],
            },
        ],
        "boundary_outputs": [{"node_id": "12", "io_type": "STRING", "is_list": False}],
        "extra_data": {"client_id": "client-1"},
        "remote_session": session_state_module.RemoteSessionHandle(
            session_id="session-1",
            prompt_id="prompt-1",
            owner_component_id="17",
        ).to_payload(),
        "clear_remote_session": True,
    }

    with pytest.raises(FakeInterrupt):
        asyncio.run(
            remote_modal_app_module._invoke_implicitly_mapped_subgraph_async(
                payload,
                serialization_module.serialize_node_inputs({"remote_input_0": [10, 11, 12]}),
            )
        )

    assert observed_calls == ["17::item:0"]

def test_implicitly_mapped_subgraph_keeps_conditioning_lists_broadcast(
    remote_modal_app_module: Any,
    serialization_module: Any,
    monkeypatch: Any,
) -> None:
    """Implicit mapped execution must not split list-backed CONDITIONING inputs per item."""
    conditioning = [
        ["cond-a", {"pooled_output": "pool-a"}],
        ["cond-b", {"pooled_output": "pool-b"}],
    ]
    observed_calls: list[tuple[str, tuple[str, ...], dict[str, Any]]] = []

    async def fake_invoke_remote_engine_async(payload: dict[str, Any], kwargs_payload: bytes) -> bytes:
        hydrated_inputs = serialization_module.deserialize_node_inputs(kwargs_payload)
        execute_node_ids = tuple(str(node_id) for node_id in payload.get("execute_node_ids", []))
        observed_calls.append((str(payload["component_id"]), execute_node_ids, hydrated_inputs))

        if execute_node_ids != ("12",):
            raise AssertionError(
                f"Unexpected execute nodes for conditioning implicit mapped regression: {execute_node_ids!r}"
            )
        return serialization_module.serialize_node_outputs(
            (f"{hydrated_inputs['remote_input_1']}:{len(hydrated_inputs['remote_input_0'])}",)
        )

    monkeypatch.setattr(
        remote_modal_app_module,
        "invoke_remote_engine_async",
        fake_invoke_remote_engine_async,
    )

    payload = {
        "payload_kind": "subgraph",
        "component_id": "17",
        "prompt_id": "prompt-1",
        "component_node_ids": ["12"],
        "execute_node_ids": ["12"],
        "subgraph_prompt": {
            "12": {
                "class_type": "KSampler",
                "inputs": {
                    "positive": 0,
                    "seed": 0,
                },
            },
        },
        "boundary_inputs": [
            {
                "proxy_input_name": "remote_input_0",
                "io_type": "CONDITIONING",
                "targets": [{"node_id": "12", "input_name": "positive"}],
            },
            {
                "proxy_input_name": "remote_input_1",
                "io_type": "INT",
                "targets": [{"node_id": "12", "input_name": "seed"}],
            },
        ],
        "boundary_outputs": [
            {"node_id": "12", "io_type": "STRING", "is_list": False},
        ],
        "extra_data": {"client_id": "client-1"},
    }

    response = asyncio.run(
        remote_modal_app_module._invoke_implicitly_mapped_subgraph_async(
            payload,
            serialization_module.serialize_node_inputs(
                {
                    "remote_input_0": conditioning,
                    "remote_input_1": [10, 11, 12, 13],
                }
            ),
        )
    )

    assert serialization_module.deserialize_node_outputs(response) == (
        ["10:2", "11:2", "12:2", "13:2"],
    )
    assert observed_calls == [
        ("17::item:0", ("12",), {"remote_input_0": conditioning, "remote_input_1": 10}),
        ("17::item:1", ("12",), {"remote_input_0": conditioning, "remote_input_1": 11}),
        ("17::item:2", ("12",), {"remote_input_0": conditioning, "remote_input_1": 12}),
        ("17::item:3", ("12",), {"remote_input_0": conditioning, "remote_input_1": 13}),
    ]

def test_implicitly_mapped_subgraph_splits_mapped_conditioning_outputs(
    remote_modal_app_module: Any,
    serialization_module: Any,
    monkeypatch: Any,
) -> None:
    """Mapped CONDITIONING aggregates should feed downstream mapped items in lockstep."""
    conditioning_items = [
        [["cond-a", {"pooled_output": "pool-a"}]],
        [["cond-b", {"pooled_output": "pool-b"}]],
        [["cond-c", {"pooled_output": "pool-c"}]],
    ]
    mapped_conditioning = serialization_module.join_mapped_values(
        conditioning_items,
        "CONDITIONING",
        is_list=False,
    )
    latent_items = ["latent-a", "latent-b", "latent-c"]
    observed_calls: list[tuple[str, dict[str, Any]]] = []

    async def fake_invoke_remote_engine_async(payload: dict[str, Any], kwargs_payload: bytes) -> bytes:
        hydrated_inputs = serialization_module.deserialize_node_inputs(kwargs_payload)
        observed_calls.append((str(payload["component_id"]), hydrated_inputs))
        return serialization_module.serialize_node_outputs(
            (f"{hydrated_inputs['remote_input_0'][0][0]}:{hydrated_inputs['remote_input_1']}",)
        )

    monkeypatch.setattr(
        remote_modal_app_module,
        "invoke_remote_engine_async",
        fake_invoke_remote_engine_async,
    )

    payload = {
        "payload_kind": "subgraph",
        "component_id": "17",
        "prompt_id": "prompt-1",
        "component_node_ids": ["12"],
        "execute_node_ids": ["12"],
        "subgraph_prompt": {
            "12": {
                "class_type": "KSampler",
                "inputs": {
                    "positive": 0,
                    "latent_image": 0,
                },
            },
        },
        "boundary_inputs": [
            {
                "proxy_input_name": "remote_input_0",
                "io_type": "CONDITIONING",
                "targets": [{"node_id": "12", "input_name": "positive"}],
            },
            {
                "proxy_input_name": "remote_input_1",
                "io_type": "LATENT",
                "targets": [{"node_id": "12", "input_name": "latent_image"}],
            },
        ],
        "boundary_outputs": [
            {"node_id": "12", "io_type": "STRING", "is_list": False},
        ],
        "extra_data": {"client_id": "client-1"},
    }

    response = asyncio.run(
        remote_modal_app_module._invoke_implicitly_mapped_subgraph_async(
            payload,
            serialization_module.serialize_node_inputs(
                {
                    "remote_input_0": mapped_conditioning,
                    "remote_input_1": latent_items,
                }
            ),
        )
    )

    assert serialization_module.deserialize_node_outputs(response) == (
        ["cond-a:latent-a", "cond-b:latent-b", "cond-c:latent-c"],
    )
    assert observed_calls == [
        (
            "17::item:0",
            {"remote_input_0": conditioning_items[0], "remote_input_1": latent_items[0]},
        ),
        (
            "17::item:1",
            {"remote_input_0": conditioning_items[1], "remote_input_1": latent_items[1]},
        ),
        (
            "17::item:2",
            {"remote_input_0": conditioning_items[2], "remote_input_1": latent_items[2]},
        ),
    ]

def test_implicitly_mapped_subgraph_splits_wildcard_latent_lists_for_latent_targets(
    remote_modal_app_module: Any,
    serialization_module: Any,
    monkeypatch: Any,
) -> None:
    """Wildcard boundary lists of LATENT-like mappings should still itemize for LATENT sockets."""
    latent_items = [
        {"samples": "latent-a", "batch_index": [0]},
        {"samples": "latent-b", "batch_index": [1]},
        {"samples": "latent-c", "batch_index": [2]},
    ]
    observed_calls: list[tuple[str, tuple[str, ...], dict[str, Any]]] = []

    fake_nodes_module = type(
        "FakeNodesModule",
        (),
        {"NODE_CLASS_MAPPINGS": {"KSampler": _FakeImplicitBatchKSamplerNode}},
    )()
    monkeypatch.setattr(remote_modal_app_module, "_load_nodes_module", lambda: fake_nodes_module)

    async def fake_invoke_remote_engine_async(payload: dict[str, Any], kwargs_payload: bytes) -> bytes:
        hydrated_inputs = serialization_module.deserialize_node_inputs(kwargs_payload)
        execute_node_ids = tuple(str(node_id) for node_id in payload.get("execute_node_ids", []))
        observed_calls.append((str(payload["component_id"]), execute_node_ids, hydrated_inputs))

        if execute_node_ids != ("12",):
            raise AssertionError(
                f"Unexpected execute nodes for wildcard LATENT implicit mapped regression: {execute_node_ids!r}"
            )
        return serialization_module.serialize_node_outputs(
            (f"{hydrated_inputs['remote_input_1']}:{hydrated_inputs['remote_input_0']['samples']}",)
        )

    monkeypatch.setattr(
        remote_modal_app_module,
        "invoke_remote_engine_async",
        fake_invoke_remote_engine_async,
    )

    payload = {
        "payload_kind": "subgraph",
        "component_id": "17",
        "prompt_id": "prompt-1",
        "component_node_ids": ["12"],
        "execute_node_ids": ["12"],
        "subgraph_prompt": {
            "12": {
                "class_type": "KSampler",
                "inputs": {
                    "latent_image": 0,
                    "seed": 0,
                },
            },
        },
        "boundary_inputs": [
            {
                "proxy_input_name": "remote_input_0",
                "io_type": "*",
                "targets": [{"node_id": "12", "input_name": "latent_image"}],
            },
            {
                "proxy_input_name": "remote_input_1",
                "io_type": "INT",
                "targets": [{"node_id": "12", "input_name": "seed"}],
            },
        ],
        "boundary_outputs": [
            {"node_id": "12", "io_type": "STRING", "is_list": False},
        ],
        "extra_data": {"client_id": "client-1"},
    }

    response = asyncio.run(
        remote_modal_app_module._invoke_implicitly_mapped_subgraph_async(
            payload,
            serialization_module.serialize_node_inputs(
                {
                    "remote_input_0": latent_items,
                    "remote_input_1": [10, 11, 12],
                }
            ),
        )
    )

    assert serialization_module.deserialize_node_outputs(response) == (
        ["10:latent-a", "11:latent-b", "12:latent-c"],
    )
    assert observed_calls == [
        ("17::item:0", ("12",), {"remote_input_0": latent_items[0], "remote_input_1": 10}),
        ("17::item:1", ("12",), {"remote_input_0": latent_items[1], "remote_input_1": 11}),
        ("17::item:2", ("12",), {"remote_input_0": latent_items[2], "remote_input_1": 12}),
    ]

def test_implicitly_mapped_subgraph_splits_session_ref_lists_for_nontransportable_inputs(
    remote_modal_app_module: Any,
    serialization_module: Any,
    monkeypatch: Any,
) -> None:
    """Implicit mapped execution should itemize lists of remote session refs even for MODEL/CONDITIONING."""
    model_ref = {
        "__comfy_modal_remote_session_value_ref__": True,
        "session_id": "session-1",
        "node_id": "31",
        "output_index": 0,
    }
    conditioning_ref = {
        "__comfy_modal_remote_session_value_ref__": True,
        "session_id": "session-1",
        "node_id": "2",
        "output_index": 0,
    }
    observed_calls: list[tuple[str, tuple[str, ...], dict[str, Any]]] = []

    async def fake_invoke_remote_engine_async(payload: dict[str, Any], kwargs_payload: bytes) -> bytes:
        hydrated_inputs = serialization_module.deserialize_node_inputs(kwargs_payload)
        execute_node_ids = tuple(str(node_id) for node_id in payload.get("execute_node_ids", []))
        observed_calls.append((str(payload["component_id"]), execute_node_ids, hydrated_inputs))

        if execute_node_ids != ("12",):
            raise AssertionError(
                f"Unexpected execute nodes for implicit session-ref regression: {execute_node_ids!r}"
            )
        return serialization_module.serialize_node_outputs((f"seed:{hydrated_inputs['remote_input_2']}",))

    monkeypatch.setattr(
        remote_modal_app_module,
        "invoke_remote_engine_async",
        fake_invoke_remote_engine_async,
    )

    payload = {
        "payload_kind": "subgraph",
        "component_id": "17",
        "prompt_id": "prompt-1",
        "component_node_ids": ["12"],
        "execute_node_ids": ["12"],
        "subgraph_prompt": {
            "12": {
                "class_type": "KSampler",
                "inputs": {
                    "model": 0,
                    "positive": 0,
                    "seed": 0,
                },
            },
        },
        "boundary_inputs": [
            {
                "proxy_input_name": "remote_input_0",
                "io_type": "MODEL",
                "targets": [{"node_id": "12", "input_name": "model"}],
            },
            {
                "proxy_input_name": "remote_input_1",
                "io_type": "CONDITIONING",
                "targets": [{"node_id": "12", "input_name": "positive"}],
            },
            {
                "proxy_input_name": "remote_input_2",
                "io_type": "INT",
                "targets": [{"node_id": "12", "input_name": "seed"}],
            },
        ],
        "boundary_outputs": [
            {"node_id": "12", "io_type": "STRING", "is_list": False},
        ],
        "extra_data": {"client_id": "client-1"},
    }

    response = asyncio.run(
        remote_modal_app_module._invoke_implicitly_mapped_subgraph_async(
            payload,
            serialization_module.serialize_node_inputs(
                {
                    "remote_input_0": [model_ref, model_ref, model_ref],
                    "remote_input_1": [conditioning_ref, conditioning_ref, conditioning_ref],
                    "remote_input_2": [10, 11, 12],
                }
            ),
        )
    )

    assert serialization_module.deserialize_node_outputs(response) == (
        ["seed:10", "seed:11", "seed:12"],
    )
    assert observed_calls == [
        (
            "17::item:0",
            ("12",),
            {"remote_input_0": model_ref, "remote_input_1": conditioning_ref, "remote_input_2": 10},
        ),
        (
            "17::item:1",
            ("12",),
            {"remote_input_0": model_ref, "remote_input_1": conditioning_ref, "remote_input_2": 11},
        ),
        (
            "17::item:2",
            ("12",),
            {"remote_input_0": model_ref, "remote_input_1": conditioning_ref, "remote_input_2": 12},
        ),
    ]

def test_implicitly_mapped_subgraph_splits_singleton_wrapped_bridge_ref_lists(
    remote_modal_app_module: Any,
    serialization_module: Any,
    monkeypatch: Any,
) -> None:
    """Comfy output wrappers around bridge-ref lists should still itemize mapped conditioning."""
    first_conditioning_ref = {
        "__comfy_modal_remote_session_bridge_ref__": True,
        "bridge_key": "RSB_first",
        "session_id": "session-1",
        "node_id": "508",
        "output_index": 0,
    }
    second_conditioning_ref = {
        "__comfy_modal_remote_session_bridge_ref__": True,
        "bridge_key": "RSB_second",
        "session_id": "session-1",
        "node_id": "508",
        "output_index": 0,
    }
    observed_calls: list[tuple[str, dict[str, Any]]] = []

    async def fake_invoke_remote_engine_async(payload: dict[str, Any], kwargs_payload: bytes) -> bytes:
        hydrated_inputs = serialization_module.deserialize_node_inputs(kwargs_payload)
        observed_calls.append((str(payload["component_id"]), hydrated_inputs))
        return serialization_module.serialize_node_outputs((f"seed:{hydrated_inputs['remote_input_1']}",))

    monkeypatch.setattr(
        remote_modal_app_module,
        "invoke_remote_engine_async",
        fake_invoke_remote_engine_async,
    )

    payload = {
        "payload_kind": "subgraph",
        "component_id": "507",
        "prompt_id": "prompt-1",
        "component_node_ids": ["507"],
        "execute_node_ids": ["507"],
        "subgraph_prompt": {
            "507": {
                "class_type": "KSampler",
                "inputs": {
                    "positive": 0,
                    "seed": 0,
                },
            },
        },
        "boundary_inputs": [
            {
                "proxy_input_name": "phase_bridge_3",
                "io_type": "CONDITIONING",
                "targets": [{"node_id": "507", "input_name": "positive"}],
            },
            {
                "proxy_input_name": "remote_input_1",
                "io_type": "INT",
                "targets": [{"node_id": "507", "input_name": "seed"}],
            },
        ],
        "boundary_outputs": [
            {"node_id": "507", "io_type": "STRING", "is_list": False},
        ],
        "extra_data": {"client_id": "client-1"},
    }

    response = asyncio.run(
        remote_modal_app_module._invoke_implicitly_mapped_subgraph_async(
            payload,
            serialization_module.serialize_node_inputs(
                {
                    "phase_bridge_3": [[first_conditioning_ref, second_conditioning_ref]],
                    "remote_input_1": [10, 11],
                }
            ),
        )
    )

    assert serialization_module.deserialize_node_outputs(response) == (
        ["seed:10", "seed:11"],
    )
    assert observed_calls == [
        (
            "507::item:0",
            {"phase_bridge_3": first_conditioning_ref, "remote_input_1": 10},
        ),
        (
            "507::item:1",
            {"phase_bridge_3": second_conditioning_ref, "remote_input_1": 11},
        ),
    ]

@pytest.mark.parametrize(
    ("module_fixture_name",),
    [
        ("local_execution_module",),
        ("modal_cloud_module",),
    ],
)
def test_trim_subgraph_payload_to_required_nodes_drops_unrelated_mapped_branch(
    request: Any,
    module_fixture_name: str,
) -> None:
    """Static or per-item sub-runs should exclude unrelated nodes from the mapped sibling branch."""
    target_module = request.getfixturevalue(module_fixture_name)
    payload = {
        "component_id": "1::static",
        "component_node_ids": ["1", "2", "3", "7"],
        "subgraph_prompt": {
            "1": {"class_type": "LoadDiffusionModel", "inputs": {}},
            "2": {"class_type": "ModalMapInput", "inputs": {"value": ["remote_input_1", 0]}},
            "3": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "steps": 20}},
            "7": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "latent_image": ["2", 0]}},
        },
        "boundary_inputs": [
            {
                "proxy_input_name": "remote_input_1",
                "targets": [{"node_id": "2", "input_name": "value"}],
            }
        ],
        "boundary_outputs": [
            {"node_id": "3", "output_index": 0, "io_type": "LATENT", "is_list": False},
        ],
        "execute_node_ids": ["3"],
        "mapped_execute_node_ids": ["7"],
        "static_execute_node_ids": ["3"],
    }

    trimmed_payload = target_module._trim_subgraph_payload_to_required_nodes(payload)

    assert trimmed_payload["component_node_ids"] == ["1", "3"]
    assert list(trimmed_payload["subgraph_prompt"].keys()) == ["1", "3"]
    assert trimmed_payload["boundary_inputs"] == []
    assert trimmed_payload["boundary_outputs"] == payload["boundary_outputs"]
    assert trimmed_payload["execute_node_ids"] == ["3"]
    assert trimmed_payload["mapped_execute_node_ids"] == []
    assert trimmed_payload["static_execute_node_ids"] == ["3"]
