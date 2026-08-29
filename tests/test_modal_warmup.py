"""Tests split from the Modal executor integration suite."""

from __future__ import annotations

from modal_executor_test_support import *  # noqa: F401,F403

def test_modal_map_input_execute_boosts_exact_warmup_for_registered_context(
    modal_executor_module: Any,
    proxy_payloads_module: Any,
    remote_modal_app_module: Any,
    monkeypatch: Any,
) -> None:
    """Local Modal Map Input execution should kick off exact warmup once the real list value is known."""
    observed_boost: dict[str, Any] = {}

    def fake_boost_mapped_component_warmup(
        payload: dict[str, Any],
        *,
        total_items: int,
        reason: str,
    ) -> tuple[int, int]:
        """Record the exact warmup boost request without touching Modal."""
        observed_boost["payload"] = dict(payload)
        observed_boost["total_items"] = total_items
        observed_boost["reason"] = reason
        return 2, 2

    monkeypatch.setattr(
        remote_modal_app_module,
        "boost_mapped_component_warmup",
        fake_boost_mapped_component_warmup,
    )
    with proxy_payloads_module._MODAL_MAP_WARMUP_CONTEXTS_LOCK:
        proxy_payloads_module._MODAL_MAP_WARMUP_CONTEXTS.clear()
    modal_executor_module.register_modal_map_input_warmup_context(
        "map-node-1",
        {
            "prompt_id": "prompt-1",
            "component_id": "mapped-component-1",
            "extra_data": {"modal": {"mapped_component_ids": ["mapped-component-1"]}},
        },
        "INT",
    )

    result = modal_executor_module.ModalMapInput.execute(
        value=[10, 11, 12],
        unique_id="map-node-1",
    )

    assert result.result == ([10, 11, 12],)
    assert observed_boost == {
        "payload": {
            "prompt_id": "prompt-1",
            "component_id": "mapped-component-1",
            "extra_data": {"modal": {"mapped_component_ids": ["mapped-component-1"]}},
        },
        "total_items": 3,
        "reason": "modal_map_input_execute",
    }

def test_modal_map_input_execute_accepts_scalar_for_single_item_warmup(
    modal_executor_module: Any,
    proxy_payloads_module: Any,
    remote_modal_app_module: Any,
    monkeypatch: Any,
) -> None:
    """A scalar Modal Map Input should pass through and warm one mapped item."""
    observed_boost: dict[str, Any] = {}

    def fake_boost_mapped_component_warmup(
        payload: dict[str, Any],
        *,
        total_items: int,
        reason: str,
    ) -> tuple[int, int]:
        """Record the scalar warmup boost request without touching Modal."""
        observed_boost["payload"] = dict(payload)
        observed_boost["total_items"] = total_items
        observed_boost["reason"] = reason
        return 1, 1

    monkeypatch.setattr(
        remote_modal_app_module,
        "boost_mapped_component_warmup",
        fake_boost_mapped_component_warmup,
    )
    with proxy_payloads_module._MODAL_MAP_WARMUP_CONTEXTS_LOCK:
        proxy_payloads_module._MODAL_MAP_WARMUP_CONTEXTS.clear()
    modal_executor_module.register_modal_map_input_warmup_context(
        "map-node-1",
        {
            "prompt_id": "prompt-1",
            "component_id": "mapped-component-1",
            "extra_data": {"modal": {"mapped_component_ids": ["mapped-component-1"]}},
        },
        "INT",
    )

    result = modal_executor_module.ModalMapInput.execute(
        value=7,
        unique_id="map-node-1",
    )

    assert result.result == (7,)
    assert observed_boost == {
        "payload": {
            "prompt_id": "prompt-1",
            "component_id": "mapped-component-1",
            "extra_data": {"modal": {"mapped_component_ids": ["mapped-component-1"]}},
        },
        "total_items": 1,
        "reason": "modal_map_input_execute",
    }

def test_ensure_remote_warm_capacity_deduplicates_prompt_slots(
    remote_modal_app_module: Any,
    modal_warmup_module: Any,
    monkeypatch: Any,
) -> None:
    """Prompt-scoped proactive warmup should only schedule each target slot once."""
    submitted_tasks: list[tuple[Any, tuple[Any, ...]]] = []

    class FakeExecutor:
        """Minimal executor that records submitted warmup jobs."""

        def submit(self, fn: Any, *args: Any) -> Future[Any]:
            """Capture one scheduled warmup task without running it."""
            submitted_tasks.append((fn, args))
            return Future()

    monkeypatch.setenv("COMFY_MODAL_EXECUTION_MODE", "remote")
    monkeypatch.setattr(remote_modal_app_module, "modal", object())
    monkeypatch.setattr(modal_warmup_module, "modal", object())
    monkeypatch.setattr(modal_warmup_module, "_REMOTE_MODAL_WARMUP_EXECUTOR", FakeExecutor())
    remote_modal_app_module.get_settings.cache_clear()
    with modal_warmup_module._PROMPT_WARMUP_STATES_LOCK:
        modal_warmup_module._PROMPT_WARMUP_STATES.clear()
        modal_warmup_module._PROMPT_WARMUP_STATE_ORDER = None

    try:
        warmup_request = {"prompt_id": "prompt-1", "component_id": "component-1"}
        first_target = remote_modal_app_module.ensure_remote_warm_capacity(
            warmup_request,
            warmup_target=2,
            reason="queue_time",
        )
        second_target = remote_modal_app_module.ensure_remote_warm_capacity(
            warmup_request,
            warmup_target=2,
            reason="queue_time_repeat",
        )
        third_target = remote_modal_app_module.ensure_remote_warm_capacity(
            warmup_request,
            warmup_target=4,
            reason="runtime_top_up",
        )
    finally:
        remote_modal_app_module.get_settings.cache_clear()

    assert first_target == 2
    assert second_target == 2
    assert third_target == 4
    assert len(submitted_tasks) == 4
    assert [args[1] for _fn, args in submitted_tasks] == [0, 1, 2, 3]

def test_speculative_affinity_prewarm_is_distinct_and_deduplicated(
    remote_modal_app_module: Any,
    modal_warmup_module: Any,
    monkeypatch: Any,
) -> None:
    """A running component should schedule its next distinct affinity exactly once."""
    submitted_tasks: list[tuple[Any, tuple[Any, ...]]] = []

    class FakeExecutor:
        """Minimal executor that records speculative warmup jobs."""

        def submit(self, fn: Any, *args: Any) -> Future[Any]:
            """Capture one scheduled task without running it."""
            submitted_tasks.append((fn, args))
            return Future()

    monkeypatch.setenv("COMFY_MODAL_EXECUTION_MODE", "remote")
    monkeypatch.setenv("COMFY_MODAL_ENABLE_GPU_MEMORY_SNAPSHOT", "false")
    monkeypatch.setattr(remote_modal_app_module, "modal", object())
    monkeypatch.setattr(modal_warmup_module, "modal", object())
    monkeypatch.setattr(
        modal_warmup_module,
        "_REMOTE_MODAL_WARMUP_EXECUTOR",
        FakeExecutor(),
    )
    remote_modal_app_module.get_settings.cache_clear()
    with modal_warmup_module._PROMPT_WARMUP_STATES_LOCK:
        modal_warmup_module._PROMPT_WARMUP_STATES.clear()
        modal_warmup_module._PROMPT_WARMUP_STATE_ORDER = None

    payload = {
        "prompt_id": "prompt-spec",
        "component_id": "llm-component",
        "execution_provider": "modal",
        "execution_environment_id": "modal:RTX-PRO-6000",
        "remote_worker_affinity_group": "llm",
        "speculative_remote_prewarm_target": {
            "prompt_id": "prompt-spec",
            "component_id": "comfy-component",
            "modal_gpu": "RTX-PRO-6000",
            "execution_provider": "modal",
            "execution_environment_id": "modal:RTX-PRO-6000",
            "remote_worker_affinity_group": "comfy",
            "remote_local_gap_pool": True,
            "subgraph_prompt": {
                "11": {
                    "class_type": "UNETLoader",
                    "inputs": {"unet_name": "video-model.safetensors"},
                }
            },
        },
    }

    try:
        assert modal_warmup_module._schedule_speculative_affinity_prewarm(
            payload,
            reason="test_first_event",
        ) is True
        assert modal_warmup_module._schedule_speculative_affinity_prewarm(
            payload,
            reason="test_duplicate_event",
        ) is False
    finally:
        remote_modal_app_module.get_settings.cache_clear()

    assert len(submitted_tasks) == 1
    scheduled_function, scheduled_args = submitted_tasks[0]
    assert scheduled_function is modal_warmup_module._run_speculative_affinity_prewarm
    assert scheduled_args[0] == "prompt-spec"
    scheduled_identity = json.loads(scheduled_args[1])
    assert scheduled_identity["affinity"] == "worker-pool:comfy:slot:0"
    assert scheduled_identity["snapshot_variant"] == "direct"
    assert scheduled_args[2]["remote_worker_affinity_group"] == "comfy"
    assert scheduled_args[2]["remote_local_gap_pool"] is True
    assert [
        plan["class_type"] for plan in scheduled_args[2]["loader_prewarm_plans"]
    ] == ["UNETLoader"]

def test_speculative_affinity_prewarm_rejects_cross_provider_target(
    remote_modal_app_module: Any,
    modal_warmup_module: Any,
    monkeypatch: Any,
) -> None:
    """An SSH continuation must never stage its payload on a Modal worker."""
    submitted_tasks: list[tuple[Any, tuple[Any, ...]]] = []

    class FakeExecutor:
        """Executor double that records unexpected warmup jobs."""

        def submit(self, fn: Any, *args: Any) -> Future[Any]:
            """Capture one scheduled job."""
            submitted_tasks.append((fn, args))
            return Future()

    monkeypatch.setattr(
        modal_warmup_module,
        "_REMOTE_MODAL_WARMUP_EXECUTOR",
        FakeExecutor(),
    )
    payload = {
        "prompt_id": "prompt-cross-provider",
        "component_id": "modal-component",
        "execution_provider": "modal",
        "execution_environment_id": "modal:H200",
        "speculative_remote_prewarm_target": {
            "prompt_id": "prompt-cross-provider",
            "component_id": "ssh-component",
            "execution_provider": "ssh_docker",
            "execution_environment_id": "lambda",
        },
    }

    assert (
        modal_warmup_module._schedule_speculative_affinity_prewarm(
            payload,
            reason="provider_boundary",
        )
        is False
    )
    assert submitted_tasks == []

def test_await_prompt_warmup_slots_waits_for_inflight_futures(
    remote_modal_app_module: Any,
    modal_warmup_module: Any,
) -> None:
    """Mapped execution should be able to wait briefly for already scheduled warmup slots."""
    prompt_id = "prompt-head-start"
    with modal_warmup_module._PROMPT_WARMUP_STATES_LOCK:
        modal_warmup_module._PROMPT_WARMUP_STATES.clear()
        modal_warmup_module._PROMPT_WARMUP_STATE_ORDER = None
        modal_warmup_module._ensure_prompt_warmup_state(prompt_id)

    future: Future[Any] = Future()
    modal_warmup_module._track_prompt_warmup_future(prompt_id, 0, future)

    def complete_future() -> None:
        """Complete the synthetic warmup slot after a short delay."""
        time.sleep(0.01)
        future.set_result({"ok": True})

    threading.Thread(target=complete_future, daemon=True).start()

    completed_count = asyncio.run(
        modal_warmup_module._await_prompt_warmup_slots(
            prompt_id,
            [0],
            0.2,
        )
    )

    assert completed_count == 1

def test_build_prompt_warmup_request_includes_root_loader_prewarm_plans(
    remote_modal_app_module: Any,
    monkeypatch: Any,
) -> None:
    """Warmup requests should synthesize one-node plans for root literal loader nodes only."""
    monkeypatch.setenv("COMFY_MODAL_ENABLE_LOADER_PREWARM", "true")
    monkeypatch.setenv("COMFY_MODAL_ENABLE_GPU_MEMORY_SNAPSHOT", "false")
    remote_modal_app_module.get_settings.cache_clear()
    try:
        warmup_request = remote_modal_app_module._build_prompt_warmup_request(
            {
                "prompt_id": "prompt-1",
                "component_id": "component-1",
                "modal_gpu": "B300",
                "subgraph_prompt": {
                    "1": {
                        "class_type": "UNETLoader",
                        "inputs": {"unet_name": "model-a.safetensors", "weight_dtype": "default"},
                    },
                    "2": {
                        "class_type": "CLIPLoader",
                        "inputs": {"clip_name": "clip-a.safetensors", "type": "stable_diffusion"},
                    },
                    "3": {
                        "class_type": "DualCLIPLoader",
                        "inputs": {"clip_name1": "clip-a.safetensors", "clip_name2": "clip-b.safetensors", "type": "flux"},
                    },
                    "4": {
                        "class_type": "UNETLoader",
                        "inputs": {"unet_name": ["99", 0]},
                    },
                    "5": {
                        "class_type": "KSampler",
                        "inputs": {"model": ["1", 0]},
                    },
                },
            }
        )
    finally:
        remote_modal_app_module.get_settings.cache_clear()

    loader_plans = warmup_request["loader_prewarm_plans"]
    assert warmup_request["modal_gpu"] == "B300"
    assert [plan["node_id"] for plan in loader_plans] == ["1", "2", "3"]
    assert [plan["class_type"] for plan in loader_plans] == [
        "UNETLoader",
        "CLIPLoader",
        "DualCLIPLoader",
    ]
    assert all(plan["execute_node_ids"] == [plan["node_id"]] for plan in loader_plans)

def test_build_prompt_warmup_request_registers_snapshot_profile_when_gpu_snapshots_enabled(
    remote_modal_app_module: Any,
    modal_warmup_module: Any,
    monkeypatch: Any,
) -> None:
    """Warmup requests should register one snapshot profile for GPU-snapshot loader plans."""

    class FakeSnapshotProfiles(dict[str, Any]):
        """Minimal modal.Dict shim for snapshot profile writes."""

    snapshot_profiles = FakeSnapshotProfiles()

    class FakeModal:
        """Minimal modal SDK double that exposes Dict.from_name."""

        class Dict:
            """Namespace for fake dict lookups."""

            @staticmethod
            def from_name(dict_name: str, create_if_missing: bool = False) -> Any:
                """Return the shared fake snapshot profile store."""
                assert dict_name == "comfy-modal-sync-snapshot-profiles"
                assert create_if_missing is True
                return snapshot_profiles

    monkeypatch.setattr(remote_modal_app_module, "modal", FakeModal)
    monkeypatch.setattr(modal_warmup_module, "modal", FakeModal)
    monkeypatch.setenv("COMFY_MODAL_ENABLE_LOADER_PREWARM", "true")
    monkeypatch.setenv("COMFY_MODAL_ENABLE_GPU_MEMORY_SNAPSHOT", "true")
    remote_modal_app_module.get_settings.cache_clear()
    modal_warmup_module._SNAPSHOT_PROFILE_RECORDS.clear()
    try:
        warmup_request = modal_warmup_module._build_prompt_warmup_request(
            {
                "prompt_id": "prompt-1",
                "component_id": "component-1",
                "subgraph_prompt": {
                    "1": {
                        "class_type": "UNETLoader",
                        "inputs": {"unet_name": "model-a.safetensors", "weight_dtype": "default"},
                    },
                    "2": {
                        "class_type": "CLIPLoader",
                        "inputs": {"clip_name": "clip-a.safetensors", "type": "flux"},
                    },
                },
            }
        )
    finally:
        remote_modal_app_module.get_settings.cache_clear()
        modal_warmup_module._SNAPSHOT_PROFILE_RECORDS.clear()

    snapshot_profile_key = warmup_request["snapshot_profile_key"]
    assert snapshot_profile_key.startswith("loader-profile:")
    assert snapshot_profile_key in snapshot_profiles
    assert snapshot_profiles[snapshot_profile_key]["loader_prewarm_plans"] == warmup_request["loader_prewarm_plans"]

def test_build_prompt_warmup_request_skips_generic_gpu_snapshot_warmup_without_profile(
    remote_modal_app_module: Any,
    monkeypatch: Any,
) -> None:
    """Generic proactive warmup should be skipped when GPU snapshots are enabled without a loader profile."""
    monkeypatch.setenv("COMFY_MODAL_ENABLE_GPU_MEMORY_SNAPSHOT", "true")
    monkeypatch.setenv("COMFY_MODAL_ENABLE_LOADER_PREWARM", "true")
    remote_modal_app_module.get_settings.cache_clear()
    try:
        warmup_request = remote_modal_app_module._build_prompt_warmup_request(
            {
                "prompt_id": "prompt-1",
                "component_id": "component-1",
                "subgraph_prompt": {
                    "5": {
                        "class_type": "KSampler",
                        "inputs": {"model": ["1", 0]},
                    }
                },
            }
        )
    finally:
        remote_modal_app_module.get_settings.cache_clear()

    assert warmup_request is None

def test_build_prompt_warmup_request_includes_llm_load_and_jit_plan(
    remote_modal_app_module: Any,
    monkeypatch: Any,
) -> None:
    """An LLM-only future component should warm even without a Comfy loader profile."""
    monkeypatch.setenv("COMFY_MODAL_ENABLE_GPU_MEMORY_SNAPSHOT", "true")
    remote_modal_app_module.get_settings.cache_clear()
    try:
        warmup_request = remote_modal_app_module._build_prompt_warmup_request(
            {
                "prompt_id": "prompt-llm",
                "component_id": "component-llm",
                "remote_worker_affinity_group": "llm",
                "execute_node_ids": ["263"],
                "subgraph_prompt": {
                    "263": {
                        "class_type": "ModalLLM",
                        "inputs": {
                            "model_profile": "qwen-test",
                            "images": ["251", 0],
                        },
                    }
                },
            }
        )
    finally:
        remote_modal_app_module.get_settings.cache_clear()

    assert warmup_request is not None
    assert warmup_request["snapshot_profile_key"] == ""
    assert warmup_request["loader_prewarm_plans"] == []
    assert len(warmup_request["llm_prewarm_plans"]) == 1
    assert warmup_request["llm_prewarm_plans"][0]["model_profile"] == "qwen-test"
    assert warmup_request["llm_prewarm_plans"][0]["representative_request_count"] == 3

def test_build_prompt_warmup_request_ignores_llm_outside_executable_closure(
    remote_modal_app_module: Any,
    monkeypatch: Any,
) -> None:
    """Disconnected nodes and serialized planning metadata must not warm an LLM."""
    monkeypatch.setenv("COMFY_MODAL_ENABLE_GPU_MEMORY_SNAPSHOT", "true")
    remote_modal_app_module.get_settings.cache_clear()
    try:
        warmup_request = remote_modal_app_module._build_prompt_warmup_request(
            {
                "prompt_id": "prompt-comfy",
                "component_id": "251",
                "execute_node_ids": ["251"],
                "subgraph_prompt": {
                    "251": {
                        "class_type": "ImageFromBatch",
                        "inputs": {"image": ["250", 0]},
                    },
                    "250": {
                        "class_type": "VAEDecode",
                        "inputs": {"samples": ["14", 0]},
                    },
                    "263": {
                        "class_type": "ModalLLM",
                        "inputs": {"model_profile": "disconnected-profile"},
                    },
                },
                "extra_data": {
                    "modal": {
                        "future_component": {
                            "class_type": "ModalLLM",
                            "inputs": {"model_profile": "metadata-profile"},
                        }
                    }
                },
            }
        )
    finally:
        remote_modal_app_module.get_settings.cache_clear()

    assert warmup_request is None

def test_dispatch_joins_matching_speculative_warmup(
    remote_modal_app_module: Any,
    modal_warmup_module: Any,
    monkeypatch: Any,
) -> None:
    """A real dispatch should wait on the exact active speculative worker identity."""
    monkeypatch.setenv("COMFY_MODAL_ENABLE_GPU_MEMORY_SNAPSHOT", "false")
    remote_modal_app_module.get_settings.cache_clear()
    payload = {
        "prompt_id": "prompt-join",
        "component_id": "component-llm",
        "remote_worker_affinity_group": "llm",
    }
    future: Future[Any] = Future()
    identity = modal_warmup_module._speculative_warmup_identity(
        {**payload, "gpu_snapshot_variant": "direct"}
    )
    with modal_warmup_module._PROMPT_WARMUP_STATES_LOCK:
        modal_warmup_module._PROMPT_WARMUP_STATES.clear()
        state = modal_warmup_module._ensure_prompt_warmup_state("prompt-join")
        state.speculative_affinity_futures[identity] = future

    timer = threading.Timer(0.03, lambda: future.set_result({"ready": True}))
    timer.start()
    started_at = time.perf_counter()
    try:
        modal_warmup_module._await_matching_speculative_prewarm(payload, None)
    finally:
        timer.join()
        remote_modal_app_module.get_settings.cache_clear()
        with modal_warmup_module._PROMPT_WARMUP_STATES_LOCK:
            modal_warmup_module._PROMPT_WARMUP_STATES.clear()

    assert time.perf_counter() - started_at >= 0.02

def test_snapshot_policy_samples_both_variants_then_selects_faster(
    remote_modal_app_module: Any,
    modal_warmup_module: Any,
    monkeypatch: Any,
) -> None:
    """Loader profiles should learn snapshot versus direct startup independently."""
    profile_key = "loader-profile:test"
    store: dict[str, Any] = {
        profile_key: {
            "snapshot_profile_key": profile_key,
            "loader_prewarm_plans": [],
            "snapshot_policy": {
                "selected_variant": None,
                "measurements": {"snapshot": [], "direct": []},
            },
        }
    }

    class FakeModal:
        """Expose the shared policy dict through Modal's named-object shape."""

        class Dict:
            """Return the in-memory profile store."""

            @staticmethod
            def from_name(name: str, create_if_missing: bool = False) -> Any:
                """Return the fake store."""
                del name, create_if_missing
                return store

    monkeypatch.setattr(remote_modal_app_module, "modal", FakeModal)
    monkeypatch.setattr(modal_warmup_module, "modal", FakeModal)
    monkeypatch.setenv("COMFY_MODAL_ENABLE_GPU_MEMORY_SNAPSHOT", "true")
    remote_modal_app_module.get_settings.cache_clear()
    modal_warmup_module._SNAPSHOT_PROFILE_RECORDS.clear()
    try:
        snapshot_payload = {"snapshot_profile_key": profile_key}
        assert modal_warmup_module._select_gpu_snapshot_for_profile(
            snapshot_payload, profile_key
        ) is True
        modal_warmup_module._record_snapshot_warmup_measurement(
            snapshot_payload, 80.0
        )
        direct_payload = {"snapshot_profile_key": profile_key}
        assert modal_warmup_module._select_gpu_snapshot_for_profile(
            direct_payload, profile_key
        ) is False
        modal_warmup_module._record_snapshot_warmup_measurement(
            direct_payload, 50.0
        )
        selected_payload = {"snapshot_profile_key": profile_key}
        assert modal_warmup_module._select_gpu_snapshot_for_profile(
            selected_payload, profile_key
        ) is False
    finally:
        remote_modal_app_module.get_settings.cache_clear()
        modal_warmup_module._SNAPSHOT_PROFILE_RECORDS.clear()

    assert store[profile_key]["snapshot_policy"]["selected_variant"] == "direct"

def test_modal_cloud_does_not_schedule_container_exit_for_interruptions(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """Expected interruption-style failures should not tear down the warm Modal container."""
    scheduled_exits: list[tuple[float, int]] = []
    original_flag = modal_cloud_module._CONTAINER_TERMINATION_SCHEDULED
    monkeypatch.setattr(modal_cloud_module, "_CONTAINER_TERMINATION_SCHEDULED", False)
    monkeypatch.setattr(modal_cloud_module, "_is_modal_container_runtime", lambda: True)
    monkeypatch.setattr(
        modal_cloud_module,
        "_schedule_process_exit",
        lambda delay_seconds, exit_code: scheduled_exits.append((delay_seconds, exit_code)),
    )
    try:
        scheduled = modal_cloud_module._maybe_schedule_container_termination_on_error(
            {"component_id": "component-1", "terminate_container_on_error": True},
            modal_cloud_module.RemoteSubgraphExecutionError(
                "Remote subgraph execution was interrupted."
            ),
        )
    finally:
        monkeypatch.setattr(modal_cloud_module, "_CONTAINER_TERMINATION_SCHEDULED", original_flag)

    assert scheduled is False
    assert scheduled_exits == []

def test_modal_cloud_does_not_schedule_container_exit_for_session_state_misses(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """Prompt-scoped session misses are routing problems, not poisoned-container crashes."""
    scheduled_exits: list[tuple[float, int]] = []
    original_flag = modal_cloud_module._CONTAINER_TERMINATION_SCHEDULED
    monkeypatch.setattr(modal_cloud_module, "_CONTAINER_TERMINATION_SCHEDULED", False)
    monkeypatch.setattr(modal_cloud_module, "_is_modal_container_runtime", lambda: True)
    monkeypatch.setattr(
        modal_cloud_module,
        "_schedule_process_exit",
        lambda delay_seconds, exit_code: scheduled_exits.append((delay_seconds, exit_code)),
    )
    try:
        scheduled = modal_cloud_module._maybe_schedule_container_termination_on_error(
            {"component_id": "component-1", "terminate_container_on_error": True},
            modal_cloud_module.RemoteSessionStateError("Remote session 'abc' was not found."),
        )
    finally:
        monkeypatch.setattr(modal_cloud_module, "_CONTAINER_TERMINATION_SCHEDULED", original_flag)

    assert scheduled is False
    assert scheduled_exits == []

def test_modal_cloud_skips_duplicate_reload_markers_in_same_container(
    modal_cloud_module: Any,
) -> None:
    """One container should reload a given uploaded-asset marker only once."""

    class FakeVolume:
        """Simple Modal volume double that tracks reload calls."""

        def __init__(self) -> None:
            """Initialize the reload counter."""
            self.reload_calls = 0

        def reload(self) -> None:
            """Record one reload attempt."""
            self.reload_calls += 1

    volume_reload_owner = _cloud_volume_reload_owner()
    original_marker_queue = volume_reload_owner._MODAL_VOLUME_RELOAD_MARKERS
    original_marker_set = set(volume_reload_owner._MODAL_VOLUME_RELOAD_MARKER_SET)
    volume_reload_owner._MODAL_VOLUME_RELOAD_MARKERS = None
    volume_reload_owner._MODAL_VOLUME_RELOAD_MARKER_SET.clear()
    try:
        payload = {"requires_volume_reload": True, "volume_reload_marker": "marker-1"}
        assert modal_cloud_module._should_reload_modal_volume(payload) is True

        volume = FakeVolume()
        modal_cloud_module._reload_modal_volume_for_request(
            volume,
            "component-1",
            reload_marker="marker-1",
        )

        assert volume.reload_calls == 1
        assert modal_cloud_module._should_reload_modal_volume(payload) is False
    finally:
        volume_reload_owner._MODAL_VOLUME_RELOAD_MARKERS = original_marker_queue
        volume_reload_owner._MODAL_VOLUME_RELOAD_MARKER_SET.clear()
        volume_reload_owner._MODAL_VOLUME_RELOAD_MARKER_SET.update(original_marker_set)

def test_modal_cloud_retries_volume_reload_after_clearing_warm_state(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """Modal volume reload should retry after unloading warm caches when open files block it."""

    class FakeVolume:
        """Simple Modal volume double that fails twice before succeeding."""

        def __init__(self) -> None:
            """Initialize the reload attempt counter."""
            self.reload_calls = 0

        def reload(self) -> None:
            """Raise on the first two calls and succeed on the third."""
            self.reload_calls += 1
            if self.reload_calls < 3:
                raise RuntimeError("there are open files preventing the operation")

    prepare_calls: list[str] = []
    sleep_calls: list[float] = []
    monkeypatch.setattr(
        _cloud_volume_reload_owner(),
        "_prepare_for_modal_volume_reload",
        lambda: prepare_calls.append("prepared"),
    )
    monkeypatch.setattr(
        _cloud_volume_reload_owner(),
        "_sleep_before_modal_volume_reload_retry",
        lambda delay_seconds: sleep_calls.append(delay_seconds),
    )

    volume = FakeVolume()
    modal_cloud_module._reload_modal_volume_for_request(volume, "component-1")

    assert volume.reload_calls == 3
    assert prepare_calls == ["prepared", "prepared"]
    assert sleep_calls == [0.25, 0.5]

def test_modal_cloud_rehydrates_bridge_refs_from_warm_value_cache_without_replay(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """Warm-worker bridge values should restore directly into a fresh session without replay."""
    bridge_key = "RSB_cached_bridge"
    target_handle = modal_cloud_module.RemoteSessionHandle(
        session_id="session-target",
        prompt_id="prompt-1",
        owner_component_id="component-1",
    )
    bridge_ref = modal_cloud_module.RemoteSessionBridgeRef(
        bridge_key=bridge_key,
        node_id="node-7",
        output_index=0,
        session_id="session-source",
    )
    original_cache = dict(_cloud_bridge_value_cache())
    original_order = list(_cloud_bridge_value_cache_order())
    try:
        seed_value = _CloneableCacheValue("warm-bridge-value")
        _cloud_bridge_value_cache().clear()
        _cloud_bridge_value_cache_order().clear()
        modal_cloud_module._store_remote_session_bridge_value(bridge_key, seed_value)
        _patch_cloud_session_bridge(
            monkeypatch,
            "_load_remote_session_bridge_record",
            lambda bridge_key: (_ for _ in ()).throw(
                AssertionError(f"warm bridge cache hit should skip record lookup for {bridge_key}")
            ),
        )
        resolution_stats = modal_cloud_module._RemoteSessionBridgeResolutionStats()

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
                node_id="node-7",
                output_index=0,
            )
        )
    finally:
        _cloud_remote_session_store().clear_session(target_handle)
        _cloud_bridge_value_cache().clear()
        _cloud_bridge_value_cache().update(original_cache)
        _cloud_bridge_value_cache_order().clear()
        _cloud_bridge_value_cache_order().extend(original_order)

    assert isinstance(restored_value, _CloneableCacheValue)
    assert restored_value.value == "warm-bridge-value"
    assert restored_value is not seed_value
    assert stored_value is restored_value
    assert resolution_stats.bridge_cache_hits == 1
    assert resolution_stats.bridge_record_lookups == 0
    assert resolution_stats.replay_count == 0
    assert resolution_stats.session_restore_writes == 1

def test_modal_cloud_existing_app_guard_skips_remote_container_import(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """The guard should not run inside Modal worker containers that import the same module."""

    class FakeApp:
        """Modal App namespace that would fail if the guard tried a lookup."""

        @staticmethod
        def lookup(app_name: str, create_if_missing: bool = True) -> object:
            """Raise if remote-container imports accidentally perform local deploy checks."""
            del app_name, create_if_missing
            raise AssertionError("remote container import should not query Modal apps")

    monkeypatch.setenv("MODAL_TASK_ID", "ta-123")
    fake_modal = types.SimpleNamespace(App=FakeApp, exception=types.SimpleNamespace(), is_local=lambda: False)

    modal_cloud_module._guard_against_existing_modal_app(
        types.SimpleNamespace(app_name="comfy-modal-sync"),
        fake_modal,
    )

def test_modal_cloud_builds_snapshot_enabled_cls_options(
    modal_cloud_module: Any,
) -> None:
    """The remote engine should default to CPU memory snapshots and optional GPU snapshots."""
    base_settings = types.SimpleNamespace(
        remote_storage_root="/storage",
        enable_memory_snapshot=True,
        enable_gpu_memory_snapshot=False,
        modal_gpu="L40S",
        scaledown_window_seconds=600,
        min_containers=0,
    )

    modal_secret = object()
    options = modal_cloud_module._remote_engine_cls_options(
        base_settings,
        "volume",
        "image",
        modal_secret,
        "compile-cache-volume",
    )

    assert options["enable_memory_snapshot"] is True
    assert "experimental_options" not in options
    assert options["gpu"] == "L40S"
    assert options["volumes"] == {
        "/storage": "volume",
        "/root/.cache/comfy-modal-llm": "compile-cache-volume",
    }
    assert options["scaledown_window"] == 600
    assert options["min_containers"] == 0
    assert options["timeout"] == 3600
    assert options["startup_timeout"] == 900
    assert options["secrets"] == [modal_secret]

    gpu_snapshot_settings = types.SimpleNamespace(
        remote_storage_root="/storage",
        enable_memory_snapshot=True,
        enable_gpu_memory_snapshot=True,
        modal_gpu="A100",
        scaledown_window_seconds=900,
        min_containers=1,
    )
    gpu_snapshot_options = modal_cloud_module._remote_engine_cls_options(
        gpu_snapshot_settings,
        "volume",
        "image",
    )
    assert gpu_snapshot_options["experimental_options"] == {"enable_gpu_snapshot": True}
    assert gpu_snapshot_options["gpu"] == "A100"
    assert gpu_snapshot_options["scaledown_window"] == 900
    assert gpu_snapshot_options["min_containers"] == 1

def test_modal_cloud_prewarms_snapshot_state_without_gpu_runtime_by_default(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """CPU-only snapshot prewarm should avoid full ComfyUI runtime initialization."""
    calls: list[str] = []

    monkeypatch.setattr(
        _cloud_prewarm_owner(),
        "_ensure_comfyui_support_packages",
        lambda: calls.append("support"),
    )
    monkeypatch.setattr(
        _cloud_prewarm_owner(),
        "_ensure_comfy_runtime_initialized",
        lambda custom_nodes_root: calls.append(f"runtime:{custom_nodes_root}"),
    )
    monkeypatch.setattr(
        _cloud_prewarm_owner(),
        "_load_execution_module",
        lambda: calls.append("execution"),
    )

    modal_cloud_module._prewarm_snapshot_state(
        gpu_snapshot_enabled=False,
    )

    assert calls == ["support"]

def test_modal_cloud_prewarms_snapshot_loader_profile_when_gpu_snapshots_enabled(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """GPU snapshot prewarm should execute registered loader plans for one snapshot profile."""
    calls: list[tuple[str, Any]] = []

    monkeypatch.setattr(
        _cloud_prewarm_owner(),
        "_ensure_comfyui_support_packages",
        lambda: calls.append(("support", None)),
    )
    monkeypatch.setattr(
        _cloud_prewarm_owner(),
        "_ensure_comfy_runtime_initialized",
        lambda custom_nodes_root: calls.append(("runtime", custom_nodes_root)),
    )
    monkeypatch.setattr(
        _cloud_prewarm_owner(),
        "_load_execution_module",
        lambda: calls.append(("execution", None)),
    )
    monkeypatch.setattr(
        _cloud_prewarm_owner(),
        "_load_loader_snapshot_profile",
        lambda snapshot_profile_key: [
            {
                "signature": "loader-plan-1",
                "node_id": "7",
                "subgraph_prompt": {"7": {"class_type": "UNETLoader", "inputs": {"unet_name": "model.safetensors"}}},
                "execute_node_ids": ["7"],
            }
        ]
        if snapshot_profile_key == "loader-profile:abc"
        else [],
    )
    monkeypatch.setattr(
        _cloud_prewarm_owner(),
        "_execute_loader_prewarm_plans",
        lambda *, component_id, loader_prewarm_plans, custom_nodes_root: calls.append(
            ("prewarm", component_id, tuple(plan["signature"] for plan in loader_prewarm_plans), custom_nodes_root)
        ),
    )

    modal_cloud_module._prewarm_snapshot_state(
        gpu_snapshot_enabled=True,
        snapshot_profile_key="loader-profile:abc",
    )

    assert calls == [
        ("support", None),
        ("runtime", None),
        ("execution", None),
        ("prewarm", "snapshot-profile:loader-profile:abc", ("loader-plan-1",), None),
    ]

def test_modal_cloud_skips_generic_gpu_snapshot_prewarm_without_profile(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """GPU snapshot prewarm should skip runtime init when no stable snapshot profile exists."""
    calls: list[str] = []

    monkeypatch.setattr(
        _cloud_prewarm_owner(),
        "_ensure_comfyui_support_packages",
        lambda: calls.append("support"),
    )
    monkeypatch.setattr(
        _cloud_prewarm_owner(),
        "_ensure_comfy_runtime_initialized",
        lambda custom_nodes_root: calls.append(f"runtime:{custom_nodes_root}"),
    )
    monkeypatch.setattr(
        _cloud_prewarm_owner(),
        "_load_execution_module",
        lambda: calls.append("execution"),
    )

    modal_cloud_module._prewarm_snapshot_state(
        gpu_snapshot_enabled=True,
    )

    assert calls == ["support"]

def test_modal_cloud_prewarms_restored_runtime(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """Post-restore prewarm should fully initialize the request-serving runtime."""
    calls: list[str] = []

    monkeypatch.setattr(
        _cloud_prewarm_owner(),
        "_ensure_comfy_runtime_initialized",
        lambda custom_nodes_root: calls.append(f"runtime:{custom_nodes_root}"),
    )
    monkeypatch.setattr(
        _cloud_prewarm_owner(),
        "_load_execution_module",
        lambda: calls.append("execution"),
    )

    modal_cloud_module._prewarm_restored_runtime()

    assert calls == ["runtime:None", "execution"]

def test_modal_cloud_prepares_warm_container_for_request(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """Warmup requests should prime volume visibility and extracted custom nodes without executing a payload."""
    calls: list[tuple[str, Any]] = []

    monkeypatch.setattr(
        _cloud_prewarm_owner(), "_should_reload_modal_volume", lambda payload: True
    )
    monkeypatch.setattr(
        _cloud_prewarm_owner(),
        "_reload_modal_volume_for_request",
        lambda volume, component_id, reload_marker=None, payload=None: calls.append(
            ("reload", component_id, reload_marker, payload.get("uploaded_volume_paths"))
        ),
    )
    monkeypatch.setattr(
        _cloud_prewarm_owner(),
        "_extract_custom_nodes_bundle",
        lambda bundle_path: Path("/tmp/extracted-bundle") if bundle_path else None,
    )
    monkeypatch.setattr(
        _cloud_prewarm_owner(),
        "_register_custom_nodes_root",
        lambda custom_nodes_root: calls.append(("register", custom_nodes_root)),
    )
    monkeypatch.setattr(
        _cloud_prewarm_owner(),
        "_execute_loader_prewarm_plans",
        lambda *, component_id, loader_prewarm_plans, custom_nodes_root: calls.append(
            ("prewarm", component_id, tuple(plan["signature"] for plan in loader_prewarm_plans), custom_nodes_root)
        ),
    )
    monkeypatch.setenv("MODAL_TASK_ID", "task-123")

    result = modal_cloud_module._prepare_warm_container_for_request(
        object(),
        {
            "component_id": "component-1",
            "volume_reload_marker": "marker-1",
            "uploaded_volume_paths": ["/storage/example.bin"],
            "custom_nodes_bundle": "custom_nodes_bundle.zip",
            "warmup_slot_index": 2,
            "loader_prewarm_plans": [{"signature": "loader-plan-1"}],
        },
    )

    assert calls == [
        ("reload", "component-1", "marker-1", ["/storage/example.bin"]),
        ("register", Path("/tmp/extracted-bundle")),
        ("prewarm", "component-1", ("loader-plan-1",), Path("/tmp/extracted-bundle")),
    ]
    assert result == {
        "component_id": "component-1",
        "task_id": "task-123",
        "warmup_slot_index": 2,
        "reloaded_volume": True,
        "llm_prewarm_results": [],
    }

def test_modal_cloud_executes_loader_prewarm_plans_once_per_worker(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """Worker-local loader prewarm plans should execute once and then be skipped on reuse."""
    prewarm_owner = _cloud_prewarm_owner()
    original_plan_keys = set(prewarm_owner._LOADER_PREWARM_PLAN_KEYS)
    prewarm_owner._LOADER_PREWARM_PLAN_KEYS.clear()
    observed_calls: list[tuple[str, tuple[str, ...]]] = []

    monkeypatch.setattr(
        prewarm_owner,
        "_ensure_comfy_runtime_initialized",
        lambda custom_nodes_root: observed_calls.append(("runtime", (str(custom_nodes_root),))),
    )
    monkeypatch.setattr(
        prewarm_owner,
        "_execute_subgraph_prompt",
        lambda payload, hydrated_inputs, custom_nodes_root, **kwargs: observed_calls.append(
            ("execute", (str(payload["component_id"]), str(tuple(payload["execute_node_ids"]))))
        ) or tuple(),
    )
    monkeypatch.setenv("COMFY_MODAL_ENABLE_LOADER_PREWARM", "true")
    modal_cloud_module.get_settings.cache_clear()
    try:
        plans = [
            {
                "signature": "loader-plan-1",
                "node_id": "7",
                "prompt_id": "prompt-1",
                "subgraph_prompt": {"7": {"class_type": "UNETLoader", "inputs": {"unet_name": "model.safetensors"}}},
                "execute_node_ids": ["7"],
            }
        ]
        modal_cloud_module._execute_loader_prewarm_plans(
            component_id="component-1",
            loader_prewarm_plans=plans,
            custom_nodes_root=Path("/tmp/extracted-bundle"),
        )
        modal_cloud_module._execute_loader_prewarm_plans(
            component_id="component-1",
            loader_prewarm_plans=plans,
            custom_nodes_root=Path("/tmp/extracted-bundle"),
        )
    finally:
        modal_cloud_module.get_settings.cache_clear()
        prewarm_owner._LOADER_PREWARM_PLAN_KEYS.clear()
        prewarm_owner._LOADER_PREWARM_PLAN_KEYS.update(original_plan_keys)

    assert observed_calls == [
        ("runtime", ("/tmp/extracted-bundle",)),
        ("execute", ("component-1::loader-prewarm:7", "('7',)")),
        ("runtime", ("/tmp/extracted-bundle",)),
    ]

def test_modal_cloud_parallelizes_independent_loader_prewarms(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """Independent CLIP and UNET loads should overlap under bounded concurrency."""
    active_count = 0
    maximum_active_count = 0
    counter_lock = threading.Lock()
    prewarm_owner = _cloud_prewarm_owner()
    original_plan_keys = set(prewarm_owner._LOADER_PREWARM_PLAN_KEYS)
    monkeypatch.setattr(
        prewarm_owner,
        "_ensure_comfy_runtime_initialized",
        lambda custom_nodes_root: None,
    )

    def execute_plan(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
        """Record overlap while simulating one heavyweight loader."""
        del args, kwargs
        nonlocal active_count, maximum_active_count
        with counter_lock:
            active_count += 1
            maximum_active_count = max(maximum_active_count, active_count)
        time.sleep(0.04)
        with counter_lock:
            active_count -= 1
        return ()

    monkeypatch.setattr(prewarm_owner, "_execute_subgraph_prompt", execute_plan)
    monkeypatch.setenv("COMFY_MODAL_ENABLE_LOADER_PREWARM", "true")
    monkeypatch.setenv("COMFY_MODAL_LOADER_PREWARM_WORKERS", "2")
    modal_cloud_module.get_settings.cache_clear()
    plans = [
        {
            "signature": f"parallel-loader-{index}",
            "node_id": str(index),
            "class_type": class_type,
            "subgraph_prompt": {
                str(index): {"class_type": class_type, "inputs": {}}
            },
            "execute_node_ids": [str(index)],
        }
        for index, class_type in enumerate(("CLIPLoader", "UNETLoader"), start=1)
    ]
    try:
        modal_cloud_module._execute_loader_prewarm_plans(
            component_id="parallel-component",
            loader_prewarm_plans=plans,
            custom_nodes_root=None,
        )
    finally:
        modal_cloud_module.get_settings.cache_clear()
        prewarm_owner._LOADER_PREWARM_PLAN_KEYS.clear()
        prewarm_owner._LOADER_PREWARM_PLAN_KEYS.update(original_plan_keys)

    assert maximum_active_count == 2

def test_modal_cloud_llm_prewarm_commits_content_addressed_manifest(
    modal_cloud_module: Any,
    modal_llm_runtime_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """A representative LLM cache miss should publish and commit its artifacts."""
    observed_requests: list[tuple[str, int, str | None]] = []
    compile_miss_signal = {"size": 0}

    def prewarm_profile(
        *,
        model_profile: str,
        representative_request_count: int,
        workflow_execution_id: str | None,
    ) -> dict[str, Any]:
        """Return deterministic warmup telemetry."""
        observed_requests.append(
            (model_profile, representative_request_count, workflow_execution_id)
        )
        compile_miss_signal["size"] += 100
        return {
            "profile_id": model_profile,
            "representative_request_count": representative_request_count,
        }

    class FakeVolume:
        """Count durable compile-cache commits."""

        def __init__(self) -> None:
            """Initialize counters."""
            self.commits = 0

        def commit(self) -> None:
            """Record one commit."""
            self.commits += 1

    cloud_llm_runtime_module = sys.modules.get("modal_llm_runtime")
    if cloud_llm_runtime_module is None:
        cloud_llm_runtime_module = modal_llm_runtime_module
        monkeypatch.setitem(
            sys.modules,
            "modal_llm_runtime",
            cloud_llm_runtime_module,
        )
    monkeypatch.setattr(
        cloud_llm_runtime_module,
        "prewarm_modal_llm_profile",
        prewarm_profile,
    )
    monkeypatch.setattr(
        cloud_llm_runtime_module,
        "triton_compile_miss_signal_size",
        lambda: compile_miss_signal["size"],
    )
    monkeypatch.setattr(
        cloud_llm_runtime_module,
        "triton_compile_listener_engine_pids",
        lambda: (321,),
    )
    prewarm_owner = _cloud_prewarm_owner()
    monkeypatch.setattr(
        prewarm_owner,
        "_llm_compile_manifest_path",
        lambda signature: tmp_path / "manifests" / f"{signature}.json",
    )
    original_plan_keys = set(prewarm_owner._LLM_PREWARM_PLAN_KEYS)
    prewarm_owner._LLM_PREWARM_PLAN_KEYS.clear()
    volume = FakeVolume()
    try:
        results = modal_cloud_module._execute_llm_prewarm_plans(
            component_id="llm-component",
            prompt_id="prompt-1",
            llm_prewarm_plans=[
                {
                    "signature": "plan-signature",
                    "model_profile": "old-profile",
                    "representative_request_count": 3,
                    "prompt_node": {
                        "class_type": "ModalLLM",
                        "inputs": {"model_profile": "staged-profile"},
                    },
                }
            ],
            compile_cache_volume=volume,
        )
    finally:
        prewarm_owner._LLM_PREWARM_PLAN_KEYS.clear()
        prewarm_owner._LLM_PREWARM_PLAN_KEYS.update(original_plan_keys)

    assert observed_requests == [("staged-profile", 3, "prompt-1")]
    assert volume.commits == 1
    assert Path(results[0]["manifest_path"]).exists()
    assert results[0]["manifest_cache_hit"] is False
    assert results[0]["compile_cache_committed"] is True

def test_modal_cloud_llm_prewarm_does_not_commit_persistent_cache_hit(
    modal_cloud_module: Any,
    modal_llm_runtime_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """A successful representative request with only disk hits must not commit."""

    class FakeVolume:
        """Reject an unnecessary persistent cache commit."""

        def commit(self) -> None:
            """Fail if a disk-cache hit is committed."""
            raise AssertionError("persistent cache hits must not be committed")

    cloud_llm_runtime_module = sys.modules.get(
        "modal_llm_runtime",
        modal_llm_runtime_module,
    )
    monkeypatch.setitem(sys.modules, "modal_llm_runtime", cloud_llm_runtime_module)
    monkeypatch.setattr(
        cloud_llm_runtime_module,
        "prewarm_modal_llm_profile",
        lambda **_kwargs: {"profile_id": "cached-profile"},
    )
    monkeypatch.setattr(
        cloud_llm_runtime_module,
        "triton_compile_miss_signal_size",
        lambda: 50,
    )
    monkeypatch.setattr(
        cloud_llm_runtime_module,
        "triton_compile_listener_engine_pids",
        lambda: (654,),
    )
    prewarm_owner = _cloud_prewarm_owner()
    monkeypatch.setattr(
        prewarm_owner,
        "_llm_compile_manifest_path",
        lambda signature: tmp_path / "manifests" / f"{signature}.json",
    )
    original_plan_keys = set(prewarm_owner._LLM_PREWARM_PLAN_KEYS)
    prewarm_owner._LLM_PREWARM_PLAN_KEYS.clear()
    try:
        results = modal_cloud_module._execute_llm_prewarm_plans(
            component_id="llm-component",
            prompt_id="prompt-hit",
            llm_prewarm_plans=[
                {
                    "signature": "cached-plan",
                    "model_profile": "cached-profile",
                    "representative_request_count": 1,
                }
            ],
            compile_cache_volume=FakeVolume(),
        )
    finally:
        prewarm_owner._LLM_PREWARM_PLAN_KEYS.clear()
        prewarm_owner._LLM_PREWARM_PLAN_KEYS.update(original_plan_keys)

    assert results[0]["compile_cache_committed"] is False

def test_modal_cloud_does_not_reload_compile_cache_during_request_warmup(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """Do not reload a cache Volume after native libraries may be mapped."""

    class FakeCompileCacheVolume:
        """Fail if request-time warmup tries to reload the compile cache."""

        def reload(self) -> None:
            """Reject the unsafe request-time reload."""
            raise AssertionError("request-time compile-cache reload is unsafe")

    monkeypatch.setattr(
        _cloud_prewarm_owner(),
        "_hydrate_missing_payload_volume_paths",
        lambda volume, payload: [],
    )
    monkeypatch.setattr(
        _cloud_prewarm_owner(),
        "_should_reload_modal_volume",
        lambda payload: False,
    )
    monkeypatch.setattr(
        _cloud_prewarm_owner(),
        "_emit_modal_volume_reload_skip",
        lambda component_id, payload: None,
    )

    result = modal_cloud_module._prepare_warm_container_for_request(
        object(),
        {"component_id": "llm-component"},
        FakeCompileCacheVolume(),
    )

    assert result["component_id"] == "llm-component"

def test_list_active_modal_containers_filters_and_classifies_managed_apps(
    modal_container_logs_module: Any,
    monkeypatch: Any,
) -> None:
    """Container status should include every active GPU app for this ComfyUI instance."""
    settings = modal_container_logs_module.get_settings()
    b300_app_name = modal_container_logs_module.modal_deployment_app_name(
        modal_container_logs_module.settings_for_modal_gpu(settings, "B300")
    )
    request_environment_names: list[str] = []
    request_thread_ids: list[int] = []
    request_event_loops: list[Any] = []
    stopped_container_ids: list[str] = []
    caller_thread_id = threading.get_ident()
    sdk_event_loop = asyncio.new_event_loop()

    class FakeStub:
        """Return a mixed active-container list."""

        async def TaskList(self, request: Any) -> Any:
            """Capture the environment and return managed plus unrelated tasks."""
            request_environment_names.append(request.environment_name)
            request_thread_ids.append(threading.get_ident())
            request_event_loops.append(asyncio.get_running_loop())
            return types.SimpleNamespace(
                tasks=[
                    types.SimpleNamespace(
                        task_id="ta-starting",
                        app_id="ap-managed",
                        app_description=b300_app_name,
                        started_at=0.0,
                        enqueued_at=100.0,
                    ),
                    types.SimpleNamespace(
                        task_id="ta-running",
                        app_id="ap-managed",
                        app_description=b300_app_name,
                        started_at=101.0,
                        enqueued_at=99.0,
                    ),
                    types.SimpleNamespace(
                        task_id="ta-unrelated",
                        app_id="ap-other",
                        app_description="another-modal-app",
                        started_at=102.0,
                        enqueued_at=98.0,
                    ),
                ]
            )

        async def ContainerStop(self, request: Any) -> Any:
            """Capture one exact Modal task termination request."""
            stopped_container_ids.append(request.task_id)
            request_thread_ids.append(threading.get_ident())
            request_event_loops.append(asyncio.get_running_loop())
            return types.SimpleNamespace()

    class FakeClient:
        """Expose the synchronized Modal TaskList stub."""

        def __init__(self) -> None:
            """Create the fake TaskList stub."""
            self.stub = FakeStub()

    class FakeClientFactory:
        """Create the fake synchronized Modal client."""

        @staticmethod
        async def from_env() -> FakeClient:
            """Return one fake authenticated client."""
            return FakeClient()

    def synchronize_api(async_callable: Any) -> Any:
        """Run the fake SDK operation on one persistent event loop."""

        def blocking_callable(*args: Any) -> Any:
            """Synchronously marshal one call onto the fake SDK loop."""
            return sdk_event_loop.run_until_complete(async_callable(*args))

        return blocking_callable

    original_import_module = modal_container_logs_module.importlib.import_module

    def fake_import_module(name: str) -> Any:
        """Supply the Modal SDK modules used by the container status query."""
        if name == "modal._object":
            return types.SimpleNamespace(_get_environment_name=lambda _environment: "test-env")
        if name == "modal.environments":
            return types.SimpleNamespace(ensure_env=lambda environment: environment or "test-env")
        if name == "modal.client":
            return types.SimpleNamespace(_Client=FakeClientFactory)
        if name == "modal._utils.async_utils":
            return types.SimpleNamespace(synchronize_api=synchronize_api)
        if name == "modal.exception":
            return types.SimpleNamespace(Error=RuntimeError)
        if name == "modal_proto.api_pb2":
            return types.SimpleNamespace(
                TaskListRequest=lambda **kwargs: types.SimpleNamespace(**kwargs),
                ContainerStopRequest=lambda **kwargs: types.SimpleNamespace(**kwargs),
            )
        return original_import_module(name)

    monkeypatch.setattr(modal_container_logs_module, "modal", object())
    monkeypatch.setattr(modal_container_logs_module.importlib, "import_module", fake_import_module)

    first_containers = asyncio.run(
        modal_container_logs_module.list_active_modal_containers(settings)
    )
    containers = asyncio.run(modal_container_logs_module.list_active_modal_containers(settings))
    stopped = asyncio.run(
        modal_container_logs_module.stop_managed_modal_container("ta-running", settings)
    )

    assert first_containers == containers
    assert [container.container_id for container in containers] == ["ta-running", "ta-starting"]
    assert [container.state for container in containers] == ["running", "starting"]
    assert all(container.modal_gpu == "B300" for container in containers)
    assert containers[0].as_dict()["started_at"] == 101.0
    assert containers[0].as_dict()["estimated_gpu_cost_per_second"] == 0.001972
    assert stopped
    assert stopped_container_ids == ["ta-running"]
    assert request_environment_names == ["test-env", "test-env", "test-env"]
    assert request_thread_ids
    assert all(thread_id != caller_thread_id for thread_id in request_thread_ids)
    assert request_event_loops == [sdk_event_loop] * 4
    sdk_event_loop.close()

def test_stop_managed_modal_container_verifies_ownership_before_exact_stop(
    modal_container_logs_module: Any,
    monkeypatch: Any,
) -> None:
    """Container termination must target only a currently listed managed task."""
    stopped_container_ids: list[str] = []

    async def fake_list_active(_settings: Any) -> list[Any]:
        """Return one container already filtered to this ComfyUI installation."""
        return [
            modal_container_logs_module.ModalContainerStatus(
                container_id="ta-managed",
                app_id="ap-managed",
                app_name="comfy-modal-sync-B300",
                modal_gpu="B300",
                estimated_gpu_cost_per_second=0.001972,
                state="running",
                enqueued_at=100.0,
                started_at=101.0,
            )
        ]

    def fake_stop(
        _client_module: Any,
        _api_pb2: Any,
        container_id: str,
    ) -> None:
        """Record the exact task passed to the synchronized SDK bridge."""
        stopped_container_ids.append(container_id)

    original_import_module = modal_container_logs_module.importlib.import_module

    def fake_import_module(name: str) -> Any:
        """Supply the minimal modules needed after ownership verification."""
        if name == "modal.client":
            return types.SimpleNamespace()
        if name == "modal.exception":
            return types.SimpleNamespace(Error=RuntimeError)
        if name == "modal_proto.api_pb2":
            return types.SimpleNamespace()
        return original_import_module(name)

    monkeypatch.setattr(
        modal_container_logs_module,
        "list_active_modal_containers",
        fake_list_active,
    )
    monkeypatch.setattr(
        modal_container_logs_module,
        "_stop_modal_task_synchronously",
        fake_stop,
    )
    monkeypatch.setattr(modal_container_logs_module.importlib, "import_module", fake_import_module)

    assert asyncio.run(
        modal_container_logs_module.stop_managed_modal_container(
            "ta-managed",
            modal_container_logs_module.get_settings(),
        )
    )
    assert not asyncio.run(
        modal_container_logs_module.stop_managed_modal_container(
            "ta-unrelated",
            modal_container_logs_module.get_settings(),
        )
    )
    assert stopped_container_ids == ["ta-managed"]

def test_local_remote_app_rehydrates_bridge_refs_from_warm_value_cache_without_replay(
    remote_modal_app_module: Any,
    host_session_bridge_module: Any,
) -> None:
    """The local fallback bridge fast path should restore retained values without replaying producers."""
    bridge_key = "RSB_local_cached_bridge"
    target_handle = remote_modal_app_module.RemoteSessionHandle(
        session_id="session-target",
        prompt_id="prompt-1",
        owner_component_id="component-1",
    )
    bridge_ref = remote_modal_app_module.RemoteSessionBridgeRef(
        bridge_key=bridge_key,
        node_id="node-7",
        output_index=0,
        session_id="session-source",
    )
    original_cache = dict(host_session_bridge_module._REMOTE_SESSION_BRIDGE_VALUE_CACHE)
    original_order = list(host_session_bridge_module._REMOTE_SESSION_BRIDGE_VALUE_CACHE_ORDER)
    try:
        seed_value = _CloneableCacheValue("warm-local-bridge-value")
        host_session_bridge_module._REMOTE_SESSION_BRIDGE_VALUE_CACHE.clear()
        host_session_bridge_module._REMOTE_SESSION_BRIDGE_VALUE_CACHE_ORDER.clear()
        host_session_bridge_module._store_remote_session_bridge_value(bridge_key, seed_value)
        resolution_stats = remote_modal_app_module._RemoteSessionBridgeResolutionStats()

        restored_value = host_session_bridge_module._rehydrate_remote_session_bridge_value(
            bridge_ref,
            target_session_handle=target_handle,
            node_mapping=None,
            resolution_stats=resolution_stats,
        )

        stored_value = host_session_bridge_module._REMOTE_SESSION_STORE.get_output(
            remote_modal_app_module.RemoteSessionValueRef(
                session_id=target_handle.session_id,
                node_id="node-7",
                output_index=0,
            )
        )
    finally:
        host_session_bridge_module._REMOTE_SESSION_STORE.clear_session(target_handle)
        host_session_bridge_module._REMOTE_SESSION_BRIDGE_VALUE_CACHE.clear()
        host_session_bridge_module._REMOTE_SESSION_BRIDGE_VALUE_CACHE.update(original_cache)
        host_session_bridge_module._REMOTE_SESSION_BRIDGE_VALUE_CACHE_ORDER.clear()
        host_session_bridge_module._REMOTE_SESSION_BRIDGE_VALUE_CACHE_ORDER.extend(original_order)

    assert isinstance(restored_value, _CloneableCacheValue)
    assert restored_value.value == "warm-local-bridge-value"
    assert restored_value is not seed_value
    assert stored_value is restored_value
    assert resolution_stats.bridge_cache_hits == 1
    assert resolution_stats.bridge_record_lookups == 0
    assert resolution_stats.replay_count == 0
    assert resolution_stats.session_restore_writes == 1

def test_local_phase_payload_builder_preserves_remote_session_and_snapshot_profile(
    remote_modal_app_module: Any,
) -> None:
    """Explicit local mapped phase payloads should keep remote_session context and snapshot profile."""
    payload = {
        "prompt_id": "prompt-1",
        "extra_data": {"client_id": "c-1"},
        "snapshot_profile_key": "loader-profile:abc",
        "remote_session": {
            "__comfy_modal_remote_session_handle__": True,
            "session_id": "session-1",
            "prompt_id": "prompt-1",
            "owner_component_id": "component-1",
        },
        "clear_remote_session": True,
        "static_phase": {
            "component_node_ids": ["1"],
            "subgraph_prompt": {},
            "boundary_inputs": [],
            "boundary_outputs": [],
            "execute_node_ids": ["1"],
        },
    }

    phase_payload = remote_modal_app_module._build_phase_subgraph_payload(
        payload,
        "static_phase",
        "component-1::static",
        suppress_status_stream=True,
    )

    assert phase_payload["remote_session"]["session_id"] == "session-1"
    assert phase_payload["clear_remote_session"] is True
    assert phase_payload["snapshot_profile_key"] == "loader-profile:abc"

def test_cloud_phase_payload_builder_preserves_remote_session_and_snapshot_profile(
    modal_cloud_module: Any,
) -> None:
    """Explicit cloud mapped phase payloads should keep remote_session context and snapshot profile."""
    payload = {
        "prompt_id": "prompt-1",
        "extra_data": {"client_id": "c-1"},
        "snapshot_profile_key": "loader-profile:abc",
        "remote_session": {
            "__comfy_modal_remote_session_handle__": True,
            "session_id": "session-1",
            "prompt_id": "prompt-1",
            "owner_component_id": "component-1",
        },
        "clear_remote_session": True,
        "mapped_phase": {
            "component_node_ids": ["7"],
            "subgraph_prompt": {},
            "boundary_inputs": [],
            "boundary_outputs": [],
            "execute_node_ids": ["7"],
        },
    }

    phase_payload = modal_cloud_module._build_phase_subgraph_payload(
        payload,
        "mapped_phase",
        "component-1::mapped",
    )

    assert phase_payload["remote_session"]["session_id"] == "session-1"
    assert phase_payload["clear_remote_session"] is True
    assert phase_payload["snapshot_profile_key"] == "loader-profile:abc"

def test_modal_cloud_interrupt_monitor_schedules_container_restart_after_cancel(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """Remote cancellation should restart the worker if the prompt ignores ComfyUI interrupt."""

    class FakeInterruptFlags:
        """Simple Modal Dict double that exposes one shared cancel flag."""

        def __init__(self) -> None:
            """Initialize the backing key set."""
            self.keys = {"prompt-1:component-2"}

        def contains(self, key: str) -> bool:
            """Report whether the shared interrupt flag exists."""
            return key in self.keys

        def pop(self, key: str, default: Any = None) -> Any:
            """Remove the shared interrupt flag once consumed."""
            del default
            self.keys.discard(key)
            return None

    scheduled_restarts: list[dict[str, Any]] = []
    interrupt_calls: list[str] = []
    cancellation_event = threading.Event()

    def fake_schedule_process_exit_unless_cancelled(**kwargs: Any) -> None:
        """Record the delayed restart request without exiting pytest."""
        scheduled_restarts.append(kwargs)

    monkeypatch.setenv("COMFY_MODAL_REMOTE_CANCEL_RESTART_SECONDS", "0.25")
    modal_cloud_module.get_settings.cache_clear()
    monkeypatch.setattr(modal_cloud_module, "_is_modal_container_runtime", lambda: True)
    monkeypatch.setattr(
        modal_cloud_module,
        "_schedule_process_exit_unless_cancelled",
        fake_schedule_process_exit_unless_cancelled,
    )
    monkeypatch.setitem(
        sys.modules,
        "nodes",
        types.SimpleNamespace(interrupt_processing=lambda: interrupt_calls.append("interrupt")),
    )

    try:
        with modal_cloud_module._temporary_remote_interrupt_monitor(
            "component-2",
            cancellation_event,
            interrupt_store=FakeInterruptFlags(),
            interrupt_flag_key="prompt-1:component-2",
        ):
            deadline = time.time() + 1.0
            while not scheduled_restarts and time.time() < deadline:
                time.sleep(0.01)
    finally:
        modal_cloud_module.get_settings.cache_clear()

    assert interrupt_calls == ["interrupt"]
    assert len(scheduled_restarts) == 1
    assert scheduled_restarts[0]["delay_seconds"] == 0.25
    assert scheduled_restarts[0]["exit_code"] == 0
    assert scheduled_restarts[0]["cancel_event"].is_set()
    assert "component=component-2" in scheduled_restarts[0]["reason"]
