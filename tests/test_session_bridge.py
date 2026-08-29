"""Tests split from the Modal executor integration suite."""

from __future__ import annotations

from modal_executor_test_support import *  # noqa: F401,F403

def test_modal_local_bridge_materializer_downloads_without_blocking_event_loop(
    modal_executor_module: Any,
    remote_modal_app_module: Any,
    session_state_module: Any,
    monkeypatch: Any,
) -> None:
    """The local bridge node should perform its durable download in a worker thread."""
    materialization_started = threading.Event()
    release_materialization = threading.Event()

    def fake_materialize(ref_payload: dict[str, Any]) -> str:
        """Block the worker thread until the async test permits completion."""
        assert session_state_module.is_remote_session_bridge_ref_payload(ref_payload)
        materialization_started.set()
        assert release_materialization.wait(timeout=1.0)
        return "local-image"

    monkeypatch.setattr(
        remote_modal_app_module,
        "materialize_remote_session_bridge_ref_locally",
        fake_materialize,
    )
    bridge_ref = session_state_module.RemoteSessionBridgeRef(
        bridge_key="RSB_local_materializer",
        node_id="251",
        output_index=0,
        session_id="session-source",
    ).to_payload()

    async def run_scenario() -> tuple[Any, ...]:
        """Verify another coroutine runs while durable materialization is blocked."""
        materializer_task = asyncio.create_task(
            modal_executor_module.ModalLocalBridgeMaterializer.execute(bridge_ref)
        )
        await asyncio.to_thread(materialization_started.wait, 1.0)
        assert materializer_task.done() is False
        await asyncio.sleep(0)
        release_materialization.set()
        result = await asyncio.wait_for(materializer_task, timeout=1.0)
        return result.result

    assert asyncio.run(run_scenario()) == ("local-image",)

def test_local_bridge_materialization_restores_durable_inline_output(
    remote_modal_app_module: Any,
    host_session_bridge_module: Any,
    monkeypatch: Any,
) -> None:
    """A local branch should restore the producer's durable serialized output."""
    import torch

    bridge_ref = remote_modal_app_module.RemoteSessionBridgeRef(
        bridge_key="RSB_local_inline_bridge",
        node_id="251",
        output_index=0,
        session_id="session-source",
    )
    image = torch.arange(12, dtype=torch.float32).reshape(1, 2, 2, 3)
    record = remote_modal_app_module.RemoteSessionBridgeRecord(
        bridge_key=bridge_ref.bridge_key,
        node_id=bridge_ref.node_id,
        output_index=bridge_ref.output_index,
        producer_payload={"component_id": "251"},
        producer_inputs={},
        serialized_output=remote_modal_app_module.serialize_value(image),
        serialized_output_io_type="IMAGE",
    )
    monkeypatch.setattr(
        host_session_bridge_module._REMOTE_SESSION_BRIDGE_STORE,
        "get_record",
        lambda bridge_key: record,
    )

    restored = host_session_bridge_module.materialize_remote_session_bridge_ref_locally(
        bridge_ref.to_payload()
    )

    assert torch.equal(restored, image)

def test_local_bridge_materialization_downloads_object_backed_output(
    remote_modal_app_module: Any,
    host_session_bridge_module: Any,
    monkeypatch: Any,
) -> None:
    """A large bridge should download its content-addressed output object locally."""
    monkeypatch.delenv("MODAL_ENVIRONMENT", raising=False)
    bridge_ref = remote_modal_app_module.RemoteSessionBridgeRef(
        bridge_key="RSB_local_object_bridge",
        node_id="251",
        output_index=0,
        session_id="session-source",
    )
    stored_payload = remote_modal_app_module.serialize_node_outputs(("large-image",))
    object_ref = remote_modal_app_module.DurableObjectRef(
        object_path="bridge-outputs/sha256/value.bin",
        sha256=hashlib.sha256(stored_payload).hexdigest(),
        size_bytes=len(stored_payload),
    )
    record = remote_modal_app_module.RemoteSessionBridgeRecord(
        bridge_key=bridge_ref.bridge_key,
        node_id=bridge_ref.node_id,
        output_index=bridge_ref.output_index,
        producer_payload={"component_id": "251"},
        producer_inputs={},
        serialized_output_object=object_ref,
        serialized_output_io_type="IMAGE",
    )
    dict_calls: list[tuple[Any, ...]] = []
    volume_calls: list[tuple[Any, ...]] = []

    class FakeDict:
        """Expose the durable bridge record through Modal's shared Dict API."""

        @staticmethod
        def from_name(*args: Any, **kwargs: Any) -> Any:
            """Return a Dict handle that records bridge lookups."""
            dict_calls.append((*args, kwargs))
            return types.SimpleNamespace(get=lambda bridge_key: record.to_payload())

    class FakeVolume:
        """Expose the durable object through Modal's direct Volume API."""

        @staticmethod
        def from_name(*args: Any, **kwargs: Any) -> Any:
            """Return a Volume handle that records object downloads."""
            volume_calls.append((*args, kwargs))
            return types.SimpleNamespace(
                read_file=lambda volume_path: iter((stored_payload[:7], stored_payload[7:]))
            )

    monkeypatch.setattr(
        host_session_bridge_module,
        "get_settings",
        lambda: types.SimpleNamespace(
            execution_mode="remote",
            session_bridge_dict_name="bridge-dict",
            volume_name="bridge-volume",
        ),
    )
    monkeypatch.setattr(
        host_session_bridge_module,
        "modal",
        types.SimpleNamespace(Dict=FakeDict, Volume=FakeVolume),
    )
    host_session_bridge_module._MODAL_SESSION_BRIDGE_DICTS.clear()
    host_session_bridge_module._MODAL_DURABLE_VOLUMES.clear()

    restored = host_session_bridge_module.materialize_remote_session_bridge_ref_locally(
        bridge_ref.to_payload()
    )

    assert restored == "large-image"
    assert dict_calls == [
        (
            "bridge-dict",
            {"environment_name": None, "create_if_missing": True},
        )
    ]
    assert volume_calls == [
        (
            "bridge-volume",
            {"environment_name": None, "create_if_missing": True},
        )
    ]
    host_session_bridge_module._MODAL_SESSION_BRIDGE_DICTS.clear()
    host_session_bridge_module._MODAL_DURABLE_VOLUMES.clear()

@pytest.mark.parametrize("hidden_delivery", ["v3_class_clone", "legacy_kwargs"])
def test_cache_friendly_proxy_payload_rehydrates_prompt_id_at_execution(
    modal_executor_module: Any,
    hidden_delivery: str,
) -> None:
    """V3 and legacy hidden inputs should select the correct overlapping prompt."""
    fake_nodes_module = type(
        "FakeNodesModule",
        (),
        {
            "NODE_CLASS_MAPPINGS": {"OriginalNode": _FakeOriginalNode},
            "NODE_DISPLAY_NAME_MAPPINGS": {},
        },
    )()

    proxy_id = modal_executor_module.ensure_modal_proxy_node_registered(
        original_class_type="OriginalNode",
        original_class=_FakeOriginalNode,
        nodes_module=fake_nodes_module,
    )
    proxy_class = fake_nodes_module.NODE_CLASS_MAPPINGS[proxy_id]

    payload = modal_executor_module.register_cache_friendly_proxy_payload(
        "node-1",
        {
            "payload_kind": "subgraph",
            "component_id": "component-1",
            "prompt_id": "prompt-1",
            "boundary_outputs": [],
            "execute_node_ids": [],
        },
    )
    next_payload = modal_executor_module.register_cache_friendly_proxy_payload(
        "node-1",
        {
            "payload_kind": "subgraph",
            "component_id": "component-1",
            "prompt_id": "prompt-2",
            "boundary_outputs": [],
            "execute_node_ids": [],
        },
    )

    class FakeClient:
        """Test client that captures the rehydrated payload."""

        async def execute_payload_async(
            self,
            payload: dict[str, Any],
            kwargs: dict[str, Any],
        ) -> tuple[str, int]:
            """Return values derived from the restored prompt id."""
            return (str(payload.get("prompt_id")), len(kwargs))

    modal_executor_module.set_remote_executor_client_factory(lambda: FakeClient())
    try:
        hidden_metadata = {
            modal_executor_module.MODAL_PROMPT_ID_EXTRA_PNGINFO_KEY: "prompt-1"
        }
        execution_proxy_class = proxy_class
        hidden_kwargs: dict[str, Any] = {}
        if hidden_delivery == "v3_class_clone":
            execution_proxy_class = proxy_class.PREPARE_CLASS_CLONE(
                {
                    "hidden_inputs": {
                        modal_executor_module.io.Hidden.unique_id: "node-1",
                        modal_executor_module.io.Hidden.extra_pnginfo: hidden_metadata,
                    }
                }
            )
        else:
            hidden_kwargs = {
                "unique_id": ["node-1"],
                "extra_pnginfo": [hidden_metadata],
            }
        result = asyncio.run(
            execution_proxy_class.execute(
                original_node_data=[payload],
                value=["payload"],
                **hidden_kwargs,
            )
        )
    finally:
        modal_executor_module.set_remote_executor_client_factory(None)

    assert "prompt_id" not in payload
    assert payload == next_payload
    assert result.result == ("prompt-1", 1)

def test_register_cache_friendly_proxy_payload_strips_session_fields_and_rehydrates_them(
    modal_executor_module: Any,
) -> None:
    """Session-backed proxy payloads should strip run-scoped fields from the local cache surface."""
    payload = modal_executor_module.register_cache_friendly_proxy_payload(
        "node-1",
        {
            "payload_kind": "subgraph",
            "component_id": "component-1",
            "prompt_id": "prompt-1",
            "remote_session": {
                "session_id": "session-1",
                "prompt_id": "prompt-1",
                "owner_component_id": "component-1",
            },
            "boundary_outputs": [],
            "execute_node_ids": [],
            "clear_remote_session": True,
            "extra_data": {
                "prompt_id": "prompt-1",
                "create_time": 1234567890,
                "modal": {"remote_node_ids": ["12"], "estimated_max_parallel_requests": 1},
            },
            "requires_volume_reload": True,
            "volume_reload_marker": "marker-1",
            "uploaded_volume_paths": ["/storage/assets/example.safetensors"],
        },
    )

    assert "prompt_id" not in payload
    assert "remote_session" not in payload
    assert "clear_remote_session" not in payload
    assert "extra_data" not in payload
    assert "requires_volume_reload" not in payload
    assert "volume_reload_marker" not in payload
    assert "uploaded_volume_paths" not in payload
    assert modal_executor_module._rehydrate_proxy_payload(payload, unique_id="node-1") == {
        "payload_kind": "subgraph",
        "component_id": "component-1",
        "prompt_id": "prompt-1",
        "remote_session": {
            "session_id": "session-1",
            "prompt_id": "prompt-1",
            "owner_component_id": "component-1",
        },
        "boundary_outputs": [],
        "execute_node_ids": [],
        "clear_remote_session": True,
        "extra_data": {
            "prompt_id": "prompt-1",
            "create_time": 1234567890,
            "modal": {"remote_node_ids": ["12"], "estimated_max_parallel_requests": 1},
        },
        "requires_volume_reload": True,
        "volume_reload_marker": "marker-1",
        "uploaded_volume_paths": ["/storage/assets/example.safetensors"],
    }

def test_cache_friendly_proxy_payload_rehydrates_without_hidden_unique_id(
    modal_executor_module: Any,
) -> None:
    """Cache-friendly proxy payloads should still rehydrate when ComfyUI omits hidden unique_id."""
    payload = modal_executor_module.register_cache_friendly_proxy_payload(
        "node-7",
        {
            "payload_kind": "subgraph",
            "component_id": "component-7",
            "prompt_id": "prompt-7",
            "remote_session": {
                "session_id": "session-7",
                "prompt_id": "prompt-7",
                "owner_component_id": "component-7",
            },
            "boundary_outputs": [],
            "execute_node_ids": [],
        },
    )

    assert modal_executor_module._rehydrate_proxy_payload(payload, unique_id=None) == {
        "payload_kind": "subgraph",
        "component_id": "component-7",
        "prompt_id": "prompt-7",
        "remote_session": {
            "session_id": "session-7",
            "prompt_id": "prompt-7",
            "owner_component_id": "component-7",
        },
        "boundary_outputs": [],
        "execute_node_ids": [],
    }

def test_modal_cloud_rehydrates_conditioning_bridge_refs_from_durable_record_without_replay(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """Durably serialized CONDITIONING bridge values should restore without replay on fresh workers."""
    import torch

    target_handle = modal_cloud_module.RemoteSessionHandle(
        session_id="session-target",
        prompt_id="prompt-1",
        owner_component_id="component-1",
    )
    bridge_ref = modal_cloud_module.RemoteSessionBridgeRef(
        bridge_key="RSB_conditioning_bridge",
        node_id="node-9",
        output_index=0,
        session_id="session-source",
    )
    conditioning = [
        [
            torch.arange(6, dtype=torch.float32).reshape(1, 2, 3),
            {"pooled_output": torch.arange(4, dtype=torch.float32).reshape(1, 4)},
        ]
    ]
    _patch_cloud_session_bridge(
        monkeypatch,
        "_load_remote_session_bridge_record",
        lambda bridge_key: modal_cloud_module.RemoteSessionBridgeRecord(
            bridge_key=bridge_key,
            node_id="node-9",
            output_index=0,
            producer_payload={"component_id": "should-not-replay"},
            producer_inputs={},
            serialized_output=modal_cloud_module.serialize_value(conditioning),
            serialized_output_io_type="CONDITIONING",
        ),
    )
    _patch_cloud_session_bridge(
        monkeypatch,
        "_execute_subgraph_prompt",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("durable CONDITIONING restore should skip replay")
        ),
    )
    resolution_stats = modal_cloud_module._RemoteSessionBridgeResolutionStats()

    try:
        restored_value = modal_cloud_module._rehydrate_remote_session_bridge_value(
            bridge_ref,
            target_session_handle=target_handle,
            custom_nodes_root=None,
            cancellation_event=None,
            interrupt_store=None,
            interrupt_flag_key=None,
            resolution_stats=resolution_stats,
        )
        stored_value = _cloud_remote_session_store().get_output(
            modal_cloud_module.RemoteSessionValueRef(
                session_id=target_handle.session_id,
                node_id="node-9",
                output_index=0,
            )
        )
    finally:
        _cloud_remote_session_store().clear_session(target_handle)

    assert torch.equal(restored_value[0][0], conditioning[0][0])
    assert torch.equal(restored_value[0][1]["pooled_output"], conditioning[0][1]["pooled_output"])
    assert torch.equal(stored_value[0][0], conditioning[0][0])
    assert resolution_stats.bridge_cache_hits == 0
    assert resolution_stats.durable_bridge_hits == 1
    assert resolution_stats.bridge_record_lookups == 1
    assert resolution_stats.replay_count == 0
    assert resolution_stats.session_restore_writes == 1

@pytest.mark.parametrize(
    ("io_type", "node_id", "class_type", "node_inputs"),
    [
        ("NOISE", "15", "RandomNoise", {"noise_seed": 42}),
        ("SAMPLER", "17", "KSamplerSelect", {"sampler_name": "euler"}),
    ],
)
def test_modal_cloud_rehydrates_literal_sampling_strategy_bridges_without_replay(
    modal_cloud_module: Any,
    monkeypatch: Any,
    io_type: str,
    node_id: str,
    class_type: str,
    node_inputs: dict[str, Any],
) -> None:
    """Literal NOISE and SAMPLER strategy nodes should rebuild without producer replay."""
    target_handle = modal_cloud_module.RemoteSessionHandle(
        session_id="session-target",
        prompt_id="prompt-1",
        owner_component_id="component-1",
    )
    bridge_ref = modal_cloud_module.RemoteSessionBridgeRef(
        bridge_key=f"RSB_{io_type.lower()}_bridge",
        node_id=node_id,
        output_index=0,
        session_id="session-source",
    )
    record = modal_cloud_module._build_remote_session_bridge_record(
        payload={
            "component_id": "image-preview-component",
            "execute_node_ids": ["251"],
            "subgraph_prompt": {
                "14": {
                    "class_type": "SamplerCustomAdvanced",
                    "inputs": {"noise": ["15", 0], "sampler": ["17", 0]},
                },
                "15": {
                    "class_type": "RandomNoise",
                    "inputs": {"noise_seed": 42},
                },
                "17": {
                    "class_type": "KSamplerSelect",
                    "inputs": {"sampler_name": "euler"},
                },
                "250": {
                    "class_type": "VAEDecode",
                    "inputs": {"samples": ["14", 0]},
                },
                "251": {
                    "class_type": "ImageFromBatch",
                    "inputs": {"image": ["250", 0]},
                },
            },
        },
        hydrated_inputs={},
        node_id=node_id,
        output_index=0,
        io_type=io_type,
        output_value=object(),
    )
    assert record.recovery_kind is modal_cloud_module.RemoteSessionBridgeRecoveryKind.SINGLE_NODE_PLAN
    assert record.rehydration_plan == {
        "kind": "single_node_output",
        "node_data": {"class_type": class_type},
        "node_inputs": node_inputs,
    }
    execute_calls: list[tuple[dict[str, Any], dict[str, Any]]] = []
    _patch_cloud_session_bridge(
        monkeypatch,
        "_load_remote_session_bridge_record",
        lambda bridge_key: record,
    )
    _patch_cloud_session_bridge(
        monkeypatch,
        "_execute_subgraph_prompt",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("literal strategy bridge should not replay its producer component")
        ),
    )
    _patch_cloud_session_bridge(
        monkeypatch,
        "_execute_node_locally_raw",
        lambda node_data, kwargs_payload, **kwargs: (
            execute_calls.append((dict(node_data), dict(kwargs_payload))),
            (f"restored-{io_type.lower()}",),
        )[1],
    )
    _patch_cloud_session_bridge(
        monkeypatch,
        "_store_remote_session_bridge_value",
        lambda bridge_key, value: None,
    )
    resolution_stats = modal_cloud_module._RemoteSessionBridgeResolutionStats()

    try:
        restored_value = modal_cloud_module._rehydrate_remote_session_bridge_value(
            bridge_ref,
            target_session_handle=target_handle,
            custom_nodes_root=None,
            cancellation_event=None,
            interrupt_store=None,
            interrupt_flag_key=None,
            resolution_stats=resolution_stats,
        )
    finally:
        _cloud_remote_session_store().clear_session(target_handle)

    assert restored_value == f"restored-{io_type.lower()}"
    assert execute_calls == [({"class_type": class_type}, node_inputs)]
    assert resolution_stats.durable_bridge_hits == 1
    assert resolution_stats.replay_count == 0

def test_modal_cloud_rehydrates_sampler_latent_bridge_refs_from_durable_record_without_replay(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """Durably serialized LATENT bridge values should restore without replaying sampler producers."""
    import torch

    target_handle = modal_cloud_module.RemoteSessionHandle(
        session_id="session-target",
        prompt_id="prompt-1",
        owner_component_id="component-1",
    )
    bridge_ref = modal_cloud_module.RemoteSessionBridgeRef(
        bridge_key="RSB_sampler_latent_bridge",
        node_id="sampler-1",
        output_index=0,
        session_id="session-source",
    )
    latent = {"samples": torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2)}
    record = modal_cloud_module._build_remote_session_bridge_record(
        payload={
            "component_id": "sampler-component",
            "execute_node_ids": ["sampler-1"],
            "subgraph_prompt": {
                "sampler-1": {
                    "class_type": "KSampler",
                    "inputs": {},
                }
            },
        },
        hydrated_inputs={},
        node_id="sampler-1",
        output_index=0,
        io_type="LATENT",
        output_value=latent,
    )
    assert record.serialized_output is not None
    assert record.serialized_output_io_type == "LATENT"
    _patch_cloud_session_bridge(
        monkeypatch,
        "_load_remote_session_bridge_record",
        lambda bridge_key: record,
    )
    _patch_cloud_session_bridge(
        monkeypatch,
        "_execute_subgraph_prompt",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("durable LATENT restore should skip sampler replay")
        ),
    )
    resolution_stats = modal_cloud_module._RemoteSessionBridgeResolutionStats()

    try:
        restored_value = modal_cloud_module._rehydrate_remote_session_bridge_value(
            bridge_ref,
            target_session_handle=target_handle,
            custom_nodes_root=None,
            cancellation_event=None,
            interrupt_store=None,
            interrupt_flag_key=None,
            resolution_stats=resolution_stats,
        )
        stored_value = _cloud_remote_session_store().get_output(
            modal_cloud_module.RemoteSessionValueRef(
                session_id=target_handle.session_id,
                node_id="sampler-1",
                output_index=0,
            )
        )
    finally:
        _cloud_remote_session_store().clear_session(target_handle)

    assert torch.equal(restored_value["samples"], latent["samples"])
    assert torch.equal(stored_value["samples"], latent["samples"])
    assert resolution_stats.bridge_cache_hits == 0
    assert resolution_stats.durable_bridge_hits == 1
    assert resolution_stats.bridge_record_lookups == 1
    assert resolution_stats.replay_count == 0
    assert resolution_stats.session_restore_writes == 1

def test_modal_cloud_rehydrates_model_bridge_refs_from_durable_plan_without_replay(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """Durable MODEL bridge plans should rebuild one self-contained loader output without replay."""
    target_handle = modal_cloud_module.RemoteSessionHandle(
        session_id="session-target",
        prompt_id="prompt-1",
        owner_component_id="component-1",
    )
    bridge_ref = modal_cloud_module.RemoteSessionBridgeRef(
        bridge_key="RSB_model_bridge",
        node_id="node-5",
        output_index=0,
        session_id="session-source",
    )
    record = modal_cloud_module._build_remote_session_bridge_record(
        payload={
            "component_id": "component-seed",
            "subgraph_prompt": {
                "node-5": {
                    "class_type": "CheckpointLoaderSimple",
                    "inputs": {"ckpt_name": "model.safetensors"},
                }
            },
        },
        hydrated_inputs={},
        node_id="node-5",
        output_index=0,
        io_type="MODEL",
        output_value=_FakeModelValue("seed-model"),
    )
    execute_calls: list[tuple[dict[str, Any], dict[str, Any]]] = []
    _patch_cloud_session_bridge(monkeypatch, "_load_remote_session_bridge_record", lambda bridge_key: record)
    _patch_cloud_session_bridge(
        monkeypatch,
        "_execute_subgraph_prompt",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("durable MODEL rehydration should skip replay")
        ),
    )
    _patch_cloud_session_bridge(
        monkeypatch,
        "_execute_node_locally_raw",
        lambda node_data, kwargs_payload, **kwargs: (
            execute_calls.append((dict(node_data), dict(kwargs_payload))),
            (_FakeModelValue("restored-model"),),
        )[1],
    )
    resolution_stats = modal_cloud_module._RemoteSessionBridgeResolutionStats()

    try:
        restored_value = modal_cloud_module._rehydrate_remote_session_bridge_value(
            bridge_ref,
            target_session_handle=target_handle,
            custom_nodes_root=None,
            cancellation_event=None,
            interrupt_store=None,
            interrupt_flag_key=None,
            resolution_stats=resolution_stats,
        )
        stored_value = _cloud_remote_session_store().get_output(
            modal_cloud_module.RemoteSessionValueRef(
                session_id=target_handle.session_id,
                node_id="node-5",
                output_index=0,
            )
        )
    finally:
        _cloud_remote_session_store().clear_session(target_handle)

    assert isinstance(restored_value, _FakeModelValue)
    assert restored_value.value == "restored-model"
    assert stored_value is restored_value
    assert execute_calls == [
        (
            {"class_type": "CheckpointLoaderSimple"},
            {"ckpt_name": "model.safetensors"},
        )
    ]
    assert resolution_stats.bridge_cache_hits == 0
    assert resolution_stats.durable_bridge_hits == 1
    assert resolution_stats.bridge_record_lookups == 1
    assert resolution_stats.replay_count == 0
    assert resolution_stats.session_restore_writes == 1

def test_modal_cloud_rehydrates_clip_bridge_refs_from_durable_plan_without_replay(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """Durable CLIP bridge plans should rebuild one self-contained loader output without replay."""
    target_handle = modal_cloud_module.RemoteSessionHandle(
        session_id="session-target",
        prompt_id="prompt-1",
        owner_component_id="component-1",
    )
    bridge_ref = modal_cloud_module.RemoteSessionBridgeRef(
        bridge_key="RSB_clip_bridge",
        node_id="clip-1",
        output_index=0,
        session_id="session-source",
    )
    record = modal_cloud_module._build_remote_session_bridge_record(
        payload={
            "component_id": "sampler-component",
            "execute_node_ids": ["sampler-1"],
            "subgraph_prompt": {
                "clip-1": {
                    "class_type": "CLIPLoader",
                    "inputs": {
                        "clip_name": "text.safetensors",
                        "type": "flux",
                        "device": "default",
                    },
                },
                "sampler-1": {
                    "class_type": "KSampler",
                    "inputs": {"clip": ["clip-1", 0]},
                },
            },
        },
        hydrated_inputs={},
        node_id="clip-1",
        output_index=0,
        io_type="CLIP",
        output_value="seed-clip",
    )
    assert record.rehydration_plan is not None
    assert record.rehydration_plan_io_type == "CLIP"
    _patch_cloud_session_bridge(monkeypatch, "_load_remote_session_bridge_record", lambda bridge_key: record)
    _patch_cloud_session_bridge(
        monkeypatch,
        "_execute_subgraph_prompt",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("durable CLIP rehydration should skip replay")
        ),
    )
    _patch_cloud_session_bridge(
        monkeypatch,
        "_execute_node_locally_raw",
        lambda node_data, kwargs_payload, **kwargs: (
            "clip::"
            f"{kwargs_payload['clip_name']}:"
            f"{kwargs_payload['type']}:"
            f"{kwargs_payload['device']}",
        ),
    )
    resolution_stats = modal_cloud_module._RemoteSessionBridgeResolutionStats()

    try:
        restored_value = modal_cloud_module._rehydrate_remote_session_bridge_value(
            bridge_ref,
            target_session_handle=target_handle,
            custom_nodes_root=None,
            cancellation_event=None,
            interrupt_store=None,
            interrupt_flag_key=None,
            resolution_stats=resolution_stats,
        )
    finally:
        _cloud_remote_session_store().clear_session(target_handle)

    assert restored_value == "clip::text.safetensors:flux:default"
    assert resolution_stats.durable_bridge_hits == 1
    assert resolution_stats.replay_count == 0

def test_modal_cloud_rehydrates_linked_model_bridge_with_non_sampler_subgraph_plan(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """Linked MODEL bridge plans should rerun only the non-sampler model dependency closure."""
    target_handle = modal_cloud_module.RemoteSessionHandle(
        session_id="session-target",
        prompt_id="prompt-1",
        owner_component_id="component-1",
    )
    bridge_ref = modal_cloud_module.RemoteSessionBridgeRef(
        bridge_key="RSB_linked_model_bridge",
        node_id="lora-1",
        output_index=0,
        session_id="session-source",
    )
    record = modal_cloud_module._build_remote_session_bridge_record(
        payload={
            "component_id": "sampler-component",
            "execute_node_ids": ["sampler-1"],
            "subgraph_prompt": {
                "loader-1": {
                    "class_type": "UNETLoader",
                    "inputs": {"unet_name": "base.safetensors"},
                },
                "lora-1": {
                    "class_type": "LoraLoaderModelOnly",
                    "inputs": {"model": ["loader-1", 0], "lora_name": "style.safetensors"},
                },
                "sampler-1": {
                    "class_type": "KSampler",
                    "inputs": {"model": ["lora-1", 0]},
                },
            },
        },
        hydrated_inputs={},
        node_id="lora-1",
        output_index=0,
        io_type="MODEL",
        output_value=_FakeModelValue("lora-model"),
    )
    assert record.rehydration_plan is not None
    assert record.rehydration_plan["kind"] == "subgraph_output"
    _patch_cloud_session_bridge(monkeypatch, "_load_remote_session_bridge_record", lambda bridge_key: record)
    observed_payloads: list[dict[str, Any]] = []

    def fake_execute_subgraph_prompt(
        payload: dict[str, Any],
        hydrated_inputs: dict[str, Any],
        *_args: Any,
    ) -> tuple[_FakeModelValue]:
        """Assert the durable plan avoids rerunning the sampler component."""
        del hydrated_inputs
        observed_payloads.append(copy.deepcopy(payload))
        assert payload["execute_node_ids"] == ["lora-1"]
        assert payload["remote_session"]["session_id"] == target_handle.session_id
        assert sorted(payload["subgraph_prompt"]) == ["loader-1", "lora-1"]
        assert "sampler-1" not in payload["subgraph_prompt"]
        return (_FakeModelValue("restored-lora-model"),)

    _patch_cloud_session_bridge(monkeypatch, "_execute_subgraph_prompt", fake_execute_subgraph_prompt)
    resolution_stats = modal_cloud_module._RemoteSessionBridgeResolutionStats()

    try:
        restored_value = modal_cloud_module._rehydrate_remote_session_bridge_value(
            bridge_ref,
            target_session_handle=target_handle,
            custom_nodes_root=None,
            cancellation_event=None,
            interrupt_store=None,
            interrupt_flag_key=None,
            resolution_stats=resolution_stats,
        )
    finally:
        _cloud_remote_session_store().clear_session(target_handle)

    assert isinstance(restored_value, _FakeModelValue)
    assert restored_value.value == "restored-lora-model"
    assert len(observed_payloads) == 1
    assert resolution_stats.durable_bridge_hits == 1
    assert resolution_stats.replay_count == 0

def test_modal_cloud_refuses_sampler_ancestor_bridge_replay(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """Producer replay should fail when a terminal node's dependency closure samples."""
    target_handle = modal_cloud_module.RemoteSessionHandle(
        session_id="session-target",
        prompt_id="prompt-1",
        owner_component_id="component-1",
    )
    bridge_ref = modal_cloud_module.RemoteSessionBridgeRef(
        bridge_key="RSB_sampler_bridge",
        node_id="unsupported-bridge",
        output_index=0,
        session_id="session-source",
    )
    _patch_cloud_session_bridge(
        monkeypatch,
        "_load_remote_session_bridge_record",
        lambda bridge_key: modal_cloud_module.RemoteSessionBridgeRecord(
            bridge_key=bridge_key,
            node_id="unsupported-bridge",
            output_index=0,
            producer_payload={
                "component_id": "sampler-component",
                "execute_node_ids": ["251"],
                "subgraph_prompt": {
                    "14": {"class_type": "KSampler", "inputs": {}},
                    "250": {
                        "class_type": "VAEDecode",
                        "inputs": {"samples": ["14", 0]},
                    },
                    "251": {
                        "class_type": "ImageFromBatch",
                        "inputs": {"image": ["250", 0]},
                    },
                },
            },
            producer_inputs={},
        ),
    )
    _patch_cloud_session_bridge(
        monkeypatch,
        "_execute_subgraph_prompt",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("sampler bridge replay should be blocked")
        ),
    )
    resolution_stats = modal_cloud_module._RemoteSessionBridgeResolutionStats()

    with pytest.raises(modal_cloud_module.RemoteSessionStateError, match="rerun a sampler"):
        modal_cloud_module._rehydrate_remote_session_bridge_value(
            bridge_ref,
            target_session_handle=target_handle,
            custom_nodes_root=None,
            cancellation_event=None,
            interrupt_store=None,
            interrupt_flag_key=None,
            resolution_stats=resolution_stats,
        )

    assert resolution_stats.bridge_record_lookups == 1
    assert resolution_stats.replay_count == 0

def test_modal_cloud_skips_seed_execution_when_session_outputs_are_already_restored(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """Cloud seed payloads should no-op when bridge outputs are already in session memory."""
    session_handle = modal_cloud_module.RemoteSessionHandle(
        session_id="session-restored",
        prompt_id="prompt-1",
        owner_component_id="component-1",
    )
    restored_value = _FakeModelValue("restored-model")
    _cloud_remote_session_store().put_output(
        session_handle,
        node_id="5",
        output_index=0,
        value=restored_value,
    )
    payload = {
        "payload_kind": "subgraph",
        "component_id": "1__mapped::seed:0",
        "prompt_id": "prompt-1",
        "component_node_ids": ["5"],
        "subgraph_prompt": {
            "5": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "model.safetensors"},
            }
        },
        "boundary_inputs": [
            {
                "proxy_input_name": "static_input_0",
                "io_type": "MODEL",
                "targets": [{"node_id": "5", "input_name": "model"}],
            }
        ],
        "boundary_outputs": [
            {
                "proxy_output_name": "static_input_0",
                "node_id": "5",
                "output_index": 0,
                "io_type": "MODEL",
                "is_list": False,
                "session_output": True,
            }
        ],
        "execute_node_ids": ["5"],
        "remote_session": session_handle.to_payload(),
    }
    hydrated_inputs = {
        "static_input_0": modal_cloud_module.RemoteSessionValueRef(
            session_id=session_handle.session_id,
            node_id="5",
            output_index=0,
        ).to_payload()
    }
    monkeypatch.setattr(
        _cloud_prompt_execution_owner(),
        "_load_execution_module",
        lambda: (_ for _ in ()).throw(AssertionError("seed short-circuit should skip PromptExecutor")),
    )

    try:
        outputs = modal_cloud_module._execute_subgraph_prompt(
            payload,
            hydrated_inputs,
            None,
        )
    finally:
        _cloud_remote_session_store().clear_session(session_handle)

    assert len(outputs) == 1
    assert modal_cloud_module.is_remote_session_bridge_ref_payload(outputs[0])

def test_modal_cloud_logs_remote_session_resolution_summary(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """Mapped-phase bridge resolution should emit one summary block with replay and loader deltas."""
    observed_logs: list[tuple[str, tuple[Any, ...]]] = []
    _patch_cloud_session_bridge(
        monkeypatch,
        "_emit_cloud_info",
        lambda message, *args: observed_logs.append((message, args)),
    )

    modal_cloud_module._log_remote_session_resolution_summary(
        component_id="1::mapped",
        resolution_stats=modal_cloud_module._RemoteSessionBridgeResolutionStats(
            input_ref_count=2,
            live_session_hits=1,
            bridge_cache_hits=1,
            durable_bridge_hits=1,
            bridge_record_lookups=1,
            bridge_record_lookup_seconds=0.25,
            replay_count=1,
            replay_seconds=1.5,
            direct_restore_seconds=0.02,
            session_restore_writes=1,
        ),
        loader_cache_before={"hit": 3, "miss": 4},
        loader_cache_after={"hit": 5, "miss": 5},
    )

    assert observed_logs == [
        (
            "Remote session resolution summary component=%s refs=%d live_hits=%d warm_bridge_hits=%d durable_bridge_hits=%d bridge_record_lookups=%d bridge_record_lookup_seconds=%.3f replay_count=%d replay_seconds=%.3f direct_restore_seconds=%.3f session_restore_writes=%d loader_cache_hits=%d loader_cache_misses=%d",
            ("1::mapped", 2, 1, 1, 1, 1, 0.25, 1, 1.5, 0.02, 1, 2, 1),
        )
    ]

def test_modal_cloud_node_cache_key_includes_remote_session_bridge_key(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """Mapped remote-session bridge inputs should not collide in the distributed cache."""

    class FakeDynPrompt:
        """Minimal dynamic prompt wrapper backed by one mutable prompt dict."""

        def __init__(self, prompt: dict[str, Any]) -> None:
            """Store the prompt used by the cache-key rebuild."""
            self._prompt = prompt

        def has_node(self, node_id: str) -> bool:
            """Return whether the requested node exists in the prompt."""
            return str(node_id) in self._prompt

        def get_node(self, node_id: str) -> dict[str, Any]:
            """Return the stored node payload."""
            return self._prompt[str(node_id)]

    class FakeConditioning:
        """Stand-in for a resolved CONDITIONING object."""

    def build_prompt(bridge_key: str) -> dict[str, Any]:
        """Build one sampler prompt with a resolved conditioning and original bridge ref metadata."""
        prompt = {
            "507": {
                "class_type": "FakeSampler",
                "inputs": {},
            }
        }
        boundary_spec = [
            {
                "proxy_input_name": "phase_bridge_3",
                "io_type": "CONDITIONING",
                "source_signature": "SRC_node_508_output_0",
                "targets": [{"node_id": "507", "input_name": "positive"}],
            }
        ]
        bridge_ref = modal_cloud_module.RemoteSessionBridgeRef(
            bridge_key=bridge_key,
            node_id="508",
            output_index=0,
            session_id="session_1",
        ).to_payload()
        modal_cloud_module._apply_boundary_inputs(
            prompt,
            boundary_spec,
            {"phase_bridge_3": FakeConditioning()},
            cache_signature_inputs={"phase_bridge_3": bridge_ref},
        )
        return prompt

    monkeypatch.setattr(
        _cloud_node_output_cache_owner(),
        "_load_nodes_module",
        lambda: types.SimpleNamespace(
            NODE_CLASS_MAPPINGS={"FakeSampler": type("FakeSampler", (), {})}
        ),
    )
    monkeypatch.setattr(
        _cloud_node_output_cache_owner(),
        "_include_unique_id_in_input_signature",
        lambda class_type: False,
    )

    def cache_key_for(prompt: dict[str, Any]) -> str | None:
        """Build one distributed cache key for the prepared prompt."""
        cache_key_set = types.SimpleNamespace(
            dynprompt=FakeDynPrompt(prompt),
            is_changed_cache=types.SimpleNamespace(is_changed={"507": False}),
            get_ordered_ancestry=lambda current_dynprompt, node_id: ([], {}),
            include_node_id_in_input=lambda: False,
            get_data_key=lambda node_id: None,
        )
        return modal_cloud_module._node_output_cache_key_from_key_set_sync(cache_key_set, "507")

    first_key = cache_key_for(build_prompt("RSB_first"))
    same_key = cache_key_for(build_prompt("RSB_first"))
    second_key = cache_key_for(build_prompt("RSB_second"))

    assert isinstance(first_key, str)
    assert first_key.startswith("NC_")
    assert same_key == first_key
    assert second_key != first_key

def test_execute_subgraph_locally_round_trips_remote_session_bridge_refs(
    remote_modal_app_module: Any,
    host_session_bridge_module: Any,
    serialization_module: Any,
    session_state_module: Any,
) -> None:
    """Split proxy subgraphs should return durable bridge refs that survive lost sessions."""
    static_payload = {
        "payload_kind": "subgraph",
        "component_id": "1",
        "prompt_id": "prompt-1",
        "component_node_ids": ["1"],
        "subgraph_prompt": {
            "1": {
                "class_type": "SessionValueNode",
                "inputs": {},
            }
        },
        "boundary_inputs": [],
        "boundary_outputs": [
            {
                "proxy_output_name": "static_input_0",
                "node_id": "1",
                "output_index": 0,
                "io_type": "MODEL",
                "is_list": False,
                "session_output": True,
            }
        ],
        "execute_node_ids": ["1"],
        "remote_session": session_state_module.RemoteSessionHandle(
            session_id="session-runtime-1",
            prompt_id="prompt-1",
            owner_component_id="1",
        ).to_payload(),
    }

    static_outputs = serialization_module.deserialize_node_outputs(
        remote_modal_app_module.execute_subgraph_locally(
            static_payload,
            serialization_module.serialize_node_inputs({}),
            node_mapping={"SessionValueNode": _FakeSessionValueNode},
        )
    )
    assert session_state_module.is_remote_session_bridge_ref_payload(static_outputs[0])
    host_session_bridge_module._REMOTE_SESSION_STORE.clear_session(
        session_state_module.RemoteSessionHandle.from_payload(static_payload["remote_session"])
    )

    mapped_payload = {
        "payload_kind": "subgraph",
        "component_id": "1__mapped",
        "prompt_id": "prompt-1",
        "component_node_ids": ["7"],
        "subgraph_prompt": {
            "7": {
                "class_type": "SessionEchoNode",
                "inputs": {"text": ["static_input_0", 0]},
            }
        },
        "boundary_inputs": [
            {
                "proxy_input_name": "static_input_0",
                "io_type": "MODEL",
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
            }
        ],
        "execute_node_ids": ["7"],
        "remote_session": session_state_module.RemoteSessionHandle(
            session_id="session-runtime-2",
            prompt_id="prompt-1",
            owner_component_id="1",
        ).to_payload(),
        "clear_remote_session": True,
    }

    mapped_outputs = serialization_module.deserialize_node_outputs(
        remote_modal_app_module.execute_subgraph_locally(
            mapped_payload,
            serialization_module.serialize_node_inputs({"static_input_0": static_outputs[0]}),
            node_mapping={
                "SessionEchoNode": _FakeSessionEchoNode,
                "SessionValueNode": _FakeSessionValueNode,
            },
        )
    )

    assert mapped_outputs == ("shared-session-value",)
    replay_outputs = remote_modal_app_module._execute_subgraph_prompt(
        {
            **mapped_payload,
            "remote_session": session_state_module.RemoteSessionHandle(
                session_id="session-runtime-3",
                prompt_id="prompt-1",
                owner_component_id="1",
            ).to_payload(),
            "clear_remote_session": False,
        },
        {"static_input_0": static_outputs[0]},
        {
            "SessionEchoNode": _FakeSessionEchoNode,
            "SessionValueNode": _FakeSessionValueNode,
        },
    )
    assert replay_outputs == ("shared-session-value",)

def test_local_fallback_skips_seed_execution_when_session_outputs_are_already_restored(
    remote_modal_app_module: Any,
    host_session_bridge_module: Any,
    local_execution_module: Any,
    monkeypatch: Any,
) -> None:
    """Local fallback seed payloads should no-op when bridge outputs are already in session memory."""
    session_handle = remote_modal_app_module.RemoteSessionHandle(
        session_id="session-restored",
        prompt_id="prompt-1",
        owner_component_id="component-1",
    )
    restored_value = _FakeModelValue("restored-model")
    host_session_bridge_module._REMOTE_SESSION_STORE.put_output(
        session_handle,
        node_id="5",
        output_index=0,
        value=restored_value,
    )
    payload = {
        "payload_kind": "subgraph",
        "component_id": "1__mapped::seed:0",
        "prompt_id": "prompt-1",
        "component_node_ids": ["5"],
        "subgraph_prompt": {
            "5": {
                "class_type": "ShouldNotRun",
                "inputs": {"model": ["static_input_0", 0]},
            }
        },
        "boundary_inputs": [
            {
                "proxy_input_name": "static_input_0",
                "io_type": "MODEL",
                "targets": [{"node_id": "5", "input_name": "model"}],
            }
        ],
        "boundary_outputs": [
            {
                "proxy_output_name": "static_input_0",
                "node_id": "5",
                "output_index": 0,
                "io_type": "MODEL",
                "is_list": False,
                "session_output": True,
            }
        ],
        "execute_node_ids": ["5"],
        "remote_session": session_handle.to_payload(),
    }
    hydrated_inputs = {
        "static_input_0": remote_modal_app_module.RemoteSessionValueRef(
            session_id=session_handle.session_id,
            node_id="5",
            output_index=0,
        ).to_payload()
    }
    monkeypatch.setattr(
        local_execution_module,
        "_invoke_original_node",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("seed short-circuit should skip local node execution")
        ),
    )

    try:
        outputs = remote_modal_app_module._execute_subgraph_prompt(
            payload,
            hydrated_inputs,
            node_mapping={"ShouldNotRun": type("ShouldNotRun", (), {})},
        )
    finally:
        host_session_bridge_module._REMOTE_SESSION_STORE.clear_session(session_handle)

    assert len(outputs) == 1
    assert remote_modal_app_module.is_remote_session_bridge_ref_payload(outputs[0])

def test_local_remote_app_rehydrates_conditioning_bridge_refs_from_durable_record_without_replay(
    remote_modal_app_module: Any,
    host_session_bridge_module: Any,
    monkeypatch: Any,
) -> None:
    """The local fallback should restore durably serialized CONDITIONING bridge values without replay."""
    import torch

    target_handle = remote_modal_app_module.RemoteSessionHandle(
        session_id="session-target",
        prompt_id="prompt-1",
        owner_component_id="component-1",
    )
    bridge_ref = remote_modal_app_module.RemoteSessionBridgeRef(
        bridge_key="RSB_local_conditioning_bridge",
        node_id="node-9",
        output_index=0,
        session_id="session-source",
    )
    conditioning = [
        [
            torch.arange(6, dtype=torch.float32).reshape(1, 2, 3),
            {"pooled_output": torch.arange(4, dtype=torch.float32).reshape(1, 4)},
        ]
    ]
    monkeypatch.setattr(
        host_session_bridge_module._REMOTE_SESSION_BRIDGE_STORE,
        "get_record",
        lambda bridge_key: remote_modal_app_module.RemoteSessionBridgeRecord(
            bridge_key=bridge_key,
            node_id="node-9",
            output_index=0,
            producer_payload={"component_id": "should-not-replay"},
            producer_inputs={},
            serialized_output=remote_modal_app_module.serialize_value(conditioning),
            serialized_output_io_type="CONDITIONING",
        ),
    )
    monkeypatch.setattr(
        remote_modal_app_module,
        "_execute_subgraph_prompt",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("durable CONDITIONING restore should skip local replay")
        ),
    )
    resolution_stats = remote_modal_app_module._RemoteSessionBridgeResolutionStats()

    try:
        restored_value = host_session_bridge_module._rehydrate_remote_session_bridge_value(
            bridge_ref,
            target_session_handle=target_handle,
            node_mapping=None,
            resolution_stats=resolution_stats,
        )
        stored_value = host_session_bridge_module._REMOTE_SESSION_STORE.get_output(
            remote_modal_app_module.RemoteSessionValueRef(
                session_id=target_handle.session_id,
                node_id="node-9",
                output_index=0,
            )
        )
    finally:
        host_session_bridge_module._REMOTE_SESSION_STORE.clear_session(target_handle)

    assert torch.equal(restored_value[0][0], conditioning[0][0])
    assert torch.equal(restored_value[0][1]["pooled_output"], conditioning[0][1]["pooled_output"])
    assert torch.equal(stored_value[0][0], conditioning[0][0])
    assert resolution_stats.bridge_cache_hits == 0
    assert resolution_stats.durable_bridge_hits == 1
    assert resolution_stats.bridge_record_lookups == 1
    assert resolution_stats.replay_count == 0
    assert resolution_stats.session_restore_writes == 1

def test_remote_session_input_resolution_handles_nested_conditioning_bridge_refs(
    remote_modal_app_module: Any,
    host_session_bridge_module: Any,
) -> None:
    """Boundary input resolution should resolve bridge refs nested inside list outputs."""
    source_handle = remote_modal_app_module.RemoteSessionHandle(
        session_id="session-source",
        prompt_id="prompt-1",
        owner_component_id="component-1",
    )
    first_ref = host_session_bridge_module._REMOTE_SESSION_STORE.put_bridge_output(
        source_handle,
        bridge_key="RSB_a",
        node_id="508:item:0",
        output_index=0,
        value=[["conditioning-a", {"pooled_output": "pool-a"}]],
    )
    second_ref = host_session_bridge_module._REMOTE_SESSION_STORE.put_bridge_output(
        source_handle,
        bridge_key="RSB_b",
        node_id="508:item:1",
        output_index=0,
        value=[["conditioning-b", {"pooled_output": "pool-b"}]],
    )
    resolution_stats = remote_modal_app_module._RemoteSessionBridgeResolutionStats()
    try:
        resolved_inputs = host_session_bridge_module._resolve_remote_session_inputs(
            {
                "positive": [
                    remote_modal_app_module.RemoteSessionBridgeRef(
                        bridge_key="RSB_a",
                        node_id=first_ref.node_id,
                        output_index=first_ref.output_index,
                        session_id=first_ref.session_id,
                    ).to_payload(),
                    remote_modal_app_module.RemoteSessionBridgeRef(
                        bridge_key="RSB_b",
                        node_id=second_ref.node_id,
                        output_index=second_ref.output_index,
                        session_id=second_ref.session_id,
                    ).to_payload(),
                ]
            },
            component_id="sampler-component",
            resolution_stats=resolution_stats,
        )
    finally:
        host_session_bridge_module._REMOTE_SESSION_STORE.clear_session(source_handle)

    assert resolved_inputs["positive"] == [
        [["conditioning-a", {"pooled_output": "pool-a"}]],
        [["conditioning-b", {"pooled_output": "pool-b"}]],
    ]
    assert resolution_stats.input_ref_count == 2

def test_local_remote_app_rehydrates_sampler_latent_bridge_refs_from_durable_record_without_replay(
    remote_modal_app_module: Any,
    host_session_bridge_module: Any,
    monkeypatch: Any,
) -> None:
    """The local fallback should restore serialized LATENT bridge values without sampler replay."""
    import torch

    target_handle = remote_modal_app_module.RemoteSessionHandle(
        session_id="session-target",
        prompt_id="prompt-1",
        owner_component_id="component-1",
    )
    bridge_ref = remote_modal_app_module.RemoteSessionBridgeRef(
        bridge_key="RSB_local_sampler_latent_bridge",
        node_id="sampler-1",
        output_index=0,
        session_id="session-source",
    )
    latent = {"samples": torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2)}
    record = host_session_bridge_module._build_remote_session_bridge_record(
        payload={
            "component_id": "sampler-component",
            "execute_node_ids": ["sampler-1"],
            "subgraph_prompt": {
                "sampler-1": {
                    "class_type": "KSampler",
                    "inputs": {},
                }
            },
        },
        hydrated_inputs={},
        node_id="sampler-1",
        output_index=0,
        io_type="LATENT",
        output_value=latent,
    )
    assert record.serialized_output is not None
    assert record.serialized_output_io_type == "LATENT"
    monkeypatch.setattr(
        host_session_bridge_module._REMOTE_SESSION_BRIDGE_STORE,
        "get_record",
        lambda bridge_key: record,
    )
    monkeypatch.setattr(
        remote_modal_app_module,
        "_execute_subgraph_prompt",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("durable LATENT restore should skip local sampler replay")
        ),
    )
    resolution_stats = remote_modal_app_module._RemoteSessionBridgeResolutionStats()

    try:
        restored_value = host_session_bridge_module._rehydrate_remote_session_bridge_value(
            bridge_ref,
            target_session_handle=target_handle,
            node_mapping=None,
            resolution_stats=resolution_stats,
        )
        stored_value = host_session_bridge_module._REMOTE_SESSION_STORE.get_output(
            remote_modal_app_module.RemoteSessionValueRef(
                session_id=target_handle.session_id,
                node_id="sampler-1",
                output_index=0,
            )
        )
    finally:
        host_session_bridge_module._REMOTE_SESSION_STORE.clear_session(target_handle)

    assert torch.equal(restored_value["samples"], latent["samples"])
    assert torch.equal(stored_value["samples"], latent["samples"])
    assert resolution_stats.bridge_cache_hits == 0
    assert resolution_stats.durable_bridge_hits == 1
    assert resolution_stats.bridge_record_lookups == 1
    assert resolution_stats.replay_count == 0
    assert resolution_stats.session_restore_writes == 1

@pytest.mark.parametrize(
    ("io_type", "node_id", "class_type", "node_inputs"),
    [
        ("NOISE", "15", "RandomNoise", {"noise_seed": 42}),
        ("SAMPLER", "17", "KSamplerSelect", {"sampler_name": "euler"}),
    ],
)
def test_local_remote_app_rehydrates_literal_sampling_strategy_bridges_without_replay(
    remote_modal_app_module: Any,
    host_session_bridge_module: Any,
    local_execution_module: Any,
    monkeypatch: Any,
    io_type: str,
    node_id: str,
    class_type: str,
    node_inputs: dict[str, Any],
) -> None:
    """The local fallback should rebuild literal sampling strategies without replay."""
    target_handle = remote_modal_app_module.RemoteSessionHandle(
        session_id="session-target",
        prompt_id="prompt-1",
        owner_component_id="component-1",
    )
    bridge_ref = remote_modal_app_module.RemoteSessionBridgeRef(
        bridge_key=f"RSB_local_{io_type.lower()}_bridge",
        node_id=node_id,
        output_index=0,
        session_id="session-source",
    )
    record = host_session_bridge_module._build_remote_session_bridge_record(
        payload={
            "component_id": "image-preview-component",
            "execute_node_ids": ["251"],
            "subgraph_prompt": {
                "14": {
                    "class_type": "SamplerCustomAdvanced",
                    "inputs": {"noise": ["15", 0], "sampler": ["17", 0]},
                },
                "15": {
                    "class_type": "RandomNoise",
                    "inputs": {"noise_seed": 42},
                },
                "17": {
                    "class_type": "KSamplerSelect",
                    "inputs": {"sampler_name": "euler"},
                },
                "250": {
                    "class_type": "VAEDecode",
                    "inputs": {"samples": ["14", 0]},
                },
                "251": {
                    "class_type": "ImageFromBatch",
                    "inputs": {"image": ["250", 0]},
                },
            },
        },
        hydrated_inputs={},
        node_id=node_id,
        output_index=0,
        io_type=io_type,
        output_value=object(),
    )
    assert (
        record.recovery_kind
        is remote_modal_app_module.RemoteSessionBridgeRecoveryKind.SINGLE_NODE_PLAN
    )
    assert record.rehydration_plan == {
        "kind": "single_node_output",
        "node_data": {"class_type": class_type},
        "node_inputs": node_inputs,
    }
    execute_calls: list[tuple[dict[str, Any], dict[str, Any]]] = []
    monkeypatch.setattr(
        host_session_bridge_module._REMOTE_SESSION_BRIDGE_STORE,
        "get_record",
        lambda bridge_key: record,
    )
    monkeypatch.setattr(
        local_execution_module,
        "_execute_subgraph_prompt",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("literal strategy bridge should not replay its producer component")
        ),
    )
    monkeypatch.setattr(
        local_execution_module,
        "_execute_node_locally_raw",
        lambda node_data, kwargs_payload, **kwargs: (
            execute_calls.append((dict(node_data), dict(kwargs_payload))),
            (f"restored-{io_type.lower()}",),
        )[1],
    )
    monkeypatch.setattr(
        remote_modal_app_module,
        "_store_remote_session_bridge_value",
        lambda bridge_key, value: None,
    )
    resolution_stats = remote_modal_app_module._RemoteSessionBridgeResolutionStats()

    try:
        restored_value = host_session_bridge_module._rehydrate_remote_session_bridge_value(
            bridge_ref,
            target_session_handle=target_handle,
            node_mapping=None,
            resolution_stats=resolution_stats,
        )
    finally:
        host_session_bridge_module._REMOTE_SESSION_STORE.clear_session(target_handle)

    assert restored_value == f"restored-{io_type.lower()}"
    assert execute_calls == [({"class_type": class_type}, node_inputs)]
    assert resolution_stats.durable_bridge_hits == 1
    assert resolution_stats.replay_count == 0

def test_local_remote_app_rehydrates_model_bridge_refs_from_durable_plan_without_replay(
    remote_modal_app_module: Any,
    host_session_bridge_module: Any,
    monkeypatch: Any,
) -> None:
    """The local fallback should rebuild one MODEL bridge output from a durable plan without replay."""
    target_handle = remote_modal_app_module.RemoteSessionHandle(
        session_id="session-target",
        prompt_id="prompt-1",
        owner_component_id="component-1",
    )
    bridge_ref = remote_modal_app_module.RemoteSessionBridgeRef(
        bridge_key="RSB_local_model_bridge",
        node_id="node-5",
        output_index=0,
        session_id="session-source",
    )
    record = host_session_bridge_module._build_remote_session_bridge_record(
        payload={
            "component_id": "component-seed",
            "subgraph_prompt": {
                "node-5": {
                    "class_type": "CheckpointLoaderSimple",
                    "inputs": {"ckpt_name": "model.safetensors"},
                }
            },
        },
        hydrated_inputs={},
        node_id="node-5",
        output_index=0,
        io_type="MODEL",
        output_value=_FakeModelValue("seed-model"),
    )
    monkeypatch.setattr(host_session_bridge_module._REMOTE_SESSION_BRIDGE_STORE, "get_record", lambda bridge_key: record)
    monkeypatch.setattr(
        remote_modal_app_module,
        "_execute_subgraph_prompt",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("durable MODEL rehydration should skip local replay")
        ),
    )
    resolution_stats = remote_modal_app_module._RemoteSessionBridgeResolutionStats()

    try:
        restored_value = host_session_bridge_module._rehydrate_remote_session_bridge_value(
            bridge_ref,
            target_session_handle=target_handle,
            node_mapping={"CheckpointLoaderSimple": _FakeModelLoaderNode},
            resolution_stats=resolution_stats,
        )
        stored_value = host_session_bridge_module._REMOTE_SESSION_STORE.get_output(
            remote_modal_app_module.RemoteSessionValueRef(
                session_id=target_handle.session_id,
                node_id="node-5",
                output_index=0,
            )
        )
    finally:
        host_session_bridge_module._REMOTE_SESSION_STORE.clear_session(target_handle)

    assert isinstance(restored_value, _FakeModelValue)
    assert restored_value.value == "model::model.safetensors"
    assert stored_value is restored_value
    assert resolution_stats.bridge_cache_hits == 0
    assert resolution_stats.durable_bridge_hits == 1
    assert resolution_stats.bridge_record_lookups == 1
    assert resolution_stats.replay_count == 0
    assert resolution_stats.session_restore_writes == 1

def test_local_remote_app_rehydrates_clip_bridge_refs_from_durable_plan_without_replay(
    remote_modal_app_module: Any,
    host_session_bridge_module: Any,
    monkeypatch: Any,
) -> None:
    """The local fallback should rebuild one CLIP bridge output from a durable plan."""
    target_handle = remote_modal_app_module.RemoteSessionHandle(
        session_id="session-target",
        prompt_id="prompt-1",
        owner_component_id="component-1",
    )
    bridge_ref = remote_modal_app_module.RemoteSessionBridgeRef(
        bridge_key="RSB_local_clip_bridge",
        node_id="clip-1",
        output_index=0,
        session_id="session-source",
    )
    record = host_session_bridge_module._build_remote_session_bridge_record(
        payload={
            "component_id": "sampler-component",
            "execute_node_ids": ["sampler-1"],
            "subgraph_prompt": {
                "clip-1": {
                    "class_type": "CLIPLoader",
                    "inputs": {
                        "clip_name": "text.safetensors",
                        "type": "flux",
                        "device": "default",
                    },
                },
                "sampler-1": {
                    "class_type": "KSampler",
                    "inputs": {"clip": ["clip-1", 0]},
                },
            },
        },
        hydrated_inputs={},
        node_id="clip-1",
        output_index=0,
        io_type="CLIP",
        output_value="seed-clip",
    )
    assert record.rehydration_plan is not None
    assert record.rehydration_plan_io_type == "CLIP"
    monkeypatch.setattr(host_session_bridge_module._REMOTE_SESSION_BRIDGE_STORE, "get_record", lambda bridge_key: record)
    monkeypatch.setattr(
        remote_modal_app_module,
        "_execute_subgraph_prompt",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("durable CLIP rehydration should skip local replay")
        ),
    )
    resolution_stats = remote_modal_app_module._RemoteSessionBridgeResolutionStats()

    try:
        restored_value = host_session_bridge_module._rehydrate_remote_session_bridge_value(
            bridge_ref,
            target_session_handle=target_handle,
            node_mapping={"CLIPLoader": _FakeClipLoaderNode},
            resolution_stats=resolution_stats,
        )
    finally:
        host_session_bridge_module._REMOTE_SESSION_STORE.clear_session(target_handle)

    assert restored_value == "clip::text.safetensors:flux:default"
    assert resolution_stats.durable_bridge_hits == 1
    assert resolution_stats.replay_count == 0

def test_local_remote_app_rehydrates_linked_model_bridge_with_non_sampler_subgraph_plan(
    remote_modal_app_module: Any,
    host_session_bridge_module: Any,
    local_execution_module: Any,
    monkeypatch: Any,
) -> None:
    """The local fallback should rehydrate linked MODEL bridges without sampler replay."""
    target_handle = remote_modal_app_module.RemoteSessionHandle(
        session_id="session-target",
        prompt_id="prompt-1",
        owner_component_id="component-1",
    )
    bridge_ref = remote_modal_app_module.RemoteSessionBridgeRef(
        bridge_key="RSB_local_linked_model_bridge",
        node_id="lora-1",
        output_index=0,
        session_id="session-source",
    )
    record = host_session_bridge_module._build_remote_session_bridge_record(
        payload={
            "component_id": "sampler-component",
            "execute_node_ids": ["sampler-1"],
            "subgraph_prompt": {
                "loader-1": {
                    "class_type": "UNETLoader",
                    "inputs": {"unet_name": "base.safetensors"},
                },
                "lora-1": {
                    "class_type": "LoraLoaderModelOnly",
                    "inputs": {"model": ["loader-1", 0], "lora_name": "style.safetensors"},
                },
                "sampler-1": {
                    "class_type": "KSampler",
                    "inputs": {"model": ["lora-1", 0]},
                },
            },
        },
        hydrated_inputs={},
        node_id="lora-1",
        output_index=0,
        io_type="MODEL",
        output_value=_FakeModelValue("lora-model"),
    )
    assert record.rehydration_plan is not None
    assert record.rehydration_plan["kind"] == "subgraph_output"
    monkeypatch.setattr(host_session_bridge_module._REMOTE_SESSION_BRIDGE_STORE, "get_record", lambda bridge_key: record)
    observed_payloads: list[dict[str, Any]] = []

    def fake_execute_subgraph_prompt(
        payload: dict[str, Any],
        hydrated_inputs: dict[str, Any],
        _node_mapping: dict[str, type[Any]] | None = None,
    ) -> tuple[_FakeModelValue]:
        """Assert the durable plan avoids rerunning the sampler component."""
        del hydrated_inputs, _node_mapping
        observed_payloads.append(copy.deepcopy(payload))
        assert payload["execute_node_ids"] == ["lora-1"]
        assert payload["remote_session"]["session_id"] == target_handle.session_id
        assert sorted(payload["subgraph_prompt"]) == ["loader-1", "lora-1"]
        assert "sampler-1" not in payload["subgraph_prompt"]
        return (_FakeModelValue("restored-lora-model"),)

    monkeypatch.setattr(local_execution_module, "_execute_subgraph_prompt", fake_execute_subgraph_prompt)
    resolution_stats = remote_modal_app_module._RemoteSessionBridgeResolutionStats()

    try:
        restored_value = host_session_bridge_module._rehydrate_remote_session_bridge_value(
            bridge_ref,
            target_session_handle=target_handle,
            node_mapping=None,
            resolution_stats=resolution_stats,
        )
    finally:
        host_session_bridge_module._REMOTE_SESSION_STORE.clear_session(target_handle)

    assert isinstance(restored_value, _FakeModelValue)
    assert restored_value.value == "restored-lora-model"
    assert len(observed_payloads) == 1
    assert resolution_stats.durable_bridge_hits == 1
    assert resolution_stats.replay_count == 0

def test_local_remote_app_refuses_sampler_ancestor_bridge_replay(
    remote_modal_app_module: Any,
    host_session_bridge_module: Any,
    monkeypatch: Any,
) -> None:
    """The local fallback should reject replay when a terminal depends on sampling."""
    target_handle = remote_modal_app_module.RemoteSessionHandle(
        session_id="session-target",
        prompt_id="prompt-1",
        owner_component_id="component-1",
    )
    bridge_ref = remote_modal_app_module.RemoteSessionBridgeRef(
        bridge_key="RSB_local_sampler_bridge",
        node_id="unsupported-bridge",
        output_index=0,
        session_id="session-source",
    )
    monkeypatch.setattr(
        host_session_bridge_module._REMOTE_SESSION_BRIDGE_STORE,
        "get_record",
        lambda bridge_key: remote_modal_app_module.RemoteSessionBridgeRecord(
            bridge_key=bridge_key,
            node_id="unsupported-bridge",
            output_index=0,
            producer_payload={
                "component_id": "sampler-component",
                "execute_node_ids": ["251"],
                "subgraph_prompt": {
                    "14": {"class_type": "KSampler", "inputs": {}},
                    "250": {
                        "class_type": "VAEDecode",
                        "inputs": {"samples": ["14", 0]},
                    },
                    "251": {
                        "class_type": "ImageFromBatch",
                        "inputs": {"image": ["250", 0]},
                    },
                },
            },
            producer_inputs={},
        ),
    )
    monkeypatch.setattr(
        remote_modal_app_module,
        "_execute_subgraph_prompt",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("sampler bridge replay should be blocked")
        ),
    )
    resolution_stats = remote_modal_app_module._RemoteSessionBridgeResolutionStats()

    try:
        with pytest.raises(remote_modal_app_module.RemoteSessionStateError, match="rerun a sampler"):
            host_session_bridge_module._rehydrate_remote_session_bridge_value(
                bridge_ref,
                target_session_handle=target_handle,
                node_mapping=None,
                resolution_stats=resolution_stats,
            )
    finally:
        host_session_bridge_module._REMOTE_SESSION_STORE.clear_session(target_handle)

    assert resolution_stats.bridge_record_lookups == 1
    assert resolution_stats.replay_count == 0
