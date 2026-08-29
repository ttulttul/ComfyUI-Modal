"""Tests for the modal llm runtime boundary."""

from __future__ import annotations

from modal_llm_test_support import *  # noqa: F401,F403

def test_curated_huihui_gguf_profile_pins_bounded_multimodal_artifacts(
    llm_profiles_module: Any,
) -> None:
    """The lambda profile should name one bounded GGUF rather than the whole repo."""
    profile = llm_profiles_module.get_llm_profile(
        "huihui-qwen3.8-27b-abliterated-q2-k-gguf"
    )

    assert profile.backend == "llama_cpp_server"
    assert profile.repository == "huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF"
    assert profile.estimated_vram_gb == 16.0
    assert profile.modalities == frozenset({"text", "image", "file"})
    assert profile.backend_option("model_filename") == (
        "Huihui-Qwen3.8-27B-abliterated-Q2_K.gguf"
    )
    assert profile.backend_option("mmproj_filename") == "mmproj-model-bf16.gguf"

def test_resident_manager_waits_for_post_eviction_memory_recovery(
    modal_llm_runtime_module: Any,
    llm_profiles_module: Any,
    tmp_path: Path,
) -> None:
    """Admission should poll until an asynchronous vLLM release becomes visible."""
    gib = 1024**3
    profile = replace(
        llm_profiles_module.get_llm_profile("smolvlm2-2.2b-instruct"),
        backend="vllm",
        estimated_vram_gb=67.9,
    )
    controller = modal_llm_runtime_module.VLLMExecutionModeController("auto")
    controller.force_throughput_after_memory_recovery("workflow-2")
    runtime_profile = modal_llm_runtime_module._profile_for_vllm_execution(
        profile,
        controller,
    )
    memory_samples = iter(
        [
            (68 * gib, 95 * gib),
            (75 * gib, 95 * gib),
            (85 * gib, 95 * gib),
        ]
    )
    clock = [0.0]
    sleeps: list[float] = []
    releases: list[int] = []

    def sleep(seconds: float) -> None:
        """Advance the deterministic recovery clock."""
        sleeps.append(seconds)
        clock[0] += seconds

    manager = modal_llm_runtime_module.ResidentLLMManager(
        storage_root=tmp_path,
        memory_info=lambda: next(memory_samples),
        empty_cache=lambda: None,
        snapshot_ready=lambda storage_root, selected_profile: True,
        comfy_memory_release=releases.append,
        vllm_mode_controller=controller,
        memory_recovery_timeout_seconds=1.0,
        memory_recovery_poll_interval_seconds=0.25,
        monotonic=lambda: clock[0],
        sleep=sleep,
    )

    manager._make_room(
        runtime_profile,
        16.0,
        evicted_before_load=True,
    )

    assert len(releases) == 1
    assert sleeps == [0.25, 0.25]

def test_resident_manager_marks_exhausted_post_eviction_memory_for_retry(
    modal_llm_runtime_module: Any,
    llm_profiles_module: Any,
    tmp_path: Path,
) -> None:
    """A bounded recovery timeout should emit the cross-process retry marker."""
    gib = 1024**3
    profile = replace(
        llm_profiles_module.get_llm_profile("smolvlm2-2.2b-instruct"),
        backend="vllm",
        estimated_vram_gb=67.9,
    )
    controller = modal_llm_runtime_module.VLLMExecutionModeController("auto")
    controller.force_throughput_after_memory_recovery("workflow-2")
    runtime_profile = modal_llm_runtime_module._profile_for_vllm_execution(
        profile,
        controller,
    )
    clock = [0.0]

    def sleep(seconds: float) -> None:
        """Advance the deterministic recovery clock."""
        clock[0] += seconds

    manager = modal_llm_runtime_module.ResidentLLMManager(
        storage_root=tmp_path,
        memory_info=lambda: (68 * gib, 95 * gib),
        empty_cache=lambda: None,
        snapshot_ready=lambda storage_root, selected_profile: True,
        comfy_memory_release=lambda required_bytes: None,
        vllm_mode_controller=controller,
        memory_recovery_timeout_seconds=0.5,
        memory_recovery_poll_interval_seconds=0.25,
        monotonic=lambda: clock[0],
        sleep=sleep,
    )

    with pytest.raises(
        RuntimeError,
        match="comfy-modal-llm-memory-recovery-exhausted.*vllm_mode=throughput",
    ):
        manager._make_room(
            runtime_profile,
            16.0,
            evicted_before_load=True,
        )

    assert clock[0] == 0.5

def test_resident_manager_reuses_and_lru_evicts_models(
    modal_llm_runtime_module: Any,
    llm_profiles_module: Any,
    tmp_path: Path,
) -> None:
    """Warm requests should hit cache and evict the least-recently used model."""
    base_profile = llm_profiles_module.get_llm_profile("smolvlm2-2.2b-instruct")
    second_profile = replace(
        base_profile,
        profile_id="second-profile",
        revision="1" * 40,
    )
    backends: dict[str, _FakeBackend] = {}

    def backend_factory(
        profile: Any,
        snapshot_path: Path,
        progress_callback: Callable[[Any], None],
    ) -> _FakeBackend:
        """Create one fake backend per load."""
        del snapshot_path, progress_callback
        backend = _FakeBackend(profile.profile_id)
        backends[profile.profile_id] = backend
        return backend

    manager = modal_llm_runtime_module.ResidentLLMManager(
        storage_root=tmp_path,
        backend_factory=backend_factory,
        max_resident_models=1,
        memory_info=lambda: (200 * 1024**3, 256 * 1024**3),
        empty_cache=lambda: None,
        snapshot_ready=lambda storage_root, profile: True,
        comfy_memory_release=lambda required_bytes: None,
    )
    prepared = modal_llm_runtime_module.PreparedLLMInputs(
        prompt="hello",
        system_prompt="",
        images=(),
        video=None,
        file_characters=0,
        file_count=0,
    )
    settings = modal_llm_runtime_module.LLMGenerationSettings(
        max_new_tokens=8,
        temperature=0.0,
        top_p=1.0,
        seed=3,
    )

    first = manager.infer(
        profile=base_profile,
        prepared_inputs=prepared,
        generation_settings=settings,
        reserve_free_vram_gb=0,
        keep_model_loaded=True,
        progress_callback=lambda value: None,
    )
    second = manager.infer(
        profile=base_profile,
        prepared_inputs=prepared,
        generation_settings=settings,
        reserve_free_vram_gb=0,
        keep_model_loaded=True,
        progress_callback=lambda value: None,
    )
    manager.infer(
        profile=second_profile,
        prepared_inputs=prepared,
        generation_settings=settings,
        reserve_free_vram_gb=0,
        keep_model_loaded=True,
        progress_callback=lambda value: None,
    )

    assert first.metadata["cache_hit"] is False
    assert first.metadata["reasoning_enabled"] is True
    assert first.metadata["execution_target"] == "modal"
    assert first.metadata["device"] == "cuda"
    assert first.metadata["memory_total_gib"] == 256
    assert first.metadata["gpu_total_gib"] == 256
    assert second.metadata["cache_hit"] is True
    assert backends[base_profile.profile_id].generate_calls == 2
    assert backends[base_profile.profile_id].unloaded is True
    assert manager.resident_profiles() == (second_profile.profile_id,)

def test_resident_manager_reports_apple_unified_memory_metadata(
    modal_llm_runtime_module: Any,
    llm_profiles_module: Any,
    tmp_path: Path,
) -> None:
    """Local inference telemetry should not mislabel unified memory as VRAM."""
    profile = replace(
        llm_profiles_module.get_llm_profile("smolvlm2-2.2b-instruct"),
        backend="mlx_vlm",
        execution_target="local_apple",
    )
    manager = modal_llm_runtime_module.ResidentLLMManager(
        storage_root=tmp_path,
        backend_factory=lambda *args: _FakeBackend(profile.profile_id),
        memory_info=lambda: (48 * 1024**3, 64 * 1024**3),
        empty_cache=lambda: None,
        snapshot_ready=lambda storage_root, selected_profile: True,
        comfy_memory_release=lambda required_bytes: None,
        execution_target="local_apple",
        device_name="metal",
        memory_label="unified memory",
    )

    result = manager.infer(
        profile=profile,
        prepared_inputs=modal_llm_runtime_module.PreparedLLMInputs(
            prompt="hello",
            system_prompt="",
            images=(),
            video=None,
            file_characters=0,
            file_count=0,
        ),
        generation_settings=modal_llm_runtime_module.LLMGenerationSettings(
            max_new_tokens=8,
            temperature=0.0,
            top_p=1.0,
            seed=0,
        ),
        reserve_free_vram_gb=1,
        keep_model_loaded=False,
        progress_callback=lambda progress: None,
    )

    assert result.metadata["execution_target"] == "local_apple"
    assert result.metadata["device"] == "metal"
    assert result.metadata["memory_total_gib"] == 64
    assert result.metadata["memory_available_before_gib"] == 48
    assert "gpu_total_gib" not in result.metadata

def test_modal_llm_node_routes_between_local_and_remote_runtimes(
    modal_llm_node_module: Any,
    modal_llm_runtime_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The V3 node toggle state should select local or remote execution."""
    schema = modal_llm_node_module.ModalLLM.define_schema()
    assert schema.node_id == "ModalLLM"
    assert [output.display_name for output in schema.outputs] == [
        "response",
        "metadata_json",
        "reasoning",
    ]
    assert "enable_reasoning" in [input_value.id for input_value in schema.inputs]

    calls: list[tuple[str, float, bool]] = []

    def fake_result(target: str, **kwargs: Any) -> Any:
        """Return deterministic output while recording the selected runtime."""
        calls.append(
            (
                target,
                kwargs["reserve_free_vram_gb"],
                kwargs["enable_reasoning"],
            )
        )
        return modal_llm_runtime_module.LLMInferenceResult(
            text=f"{target}-done",
            metadata={
                "output_tokens": 1,
                "cache_hit": True,
                "execution_target": target,
            },
            reasoning="working",
        )

    monkeypatch.delenv("COMFY_MODAL_REMOTE_WORKER", raising=False)
    monkeypatch.setattr(
        modal_llm_node_module,
        "run_local_llm_inference",
        lambda **kwargs: fake_result("local_apple", **kwargs),
    )
    local_output = modal_llm_node_module.ModalLLM.execute(
        prompt="hello",
        model_profile="smolvlm2-2.2b-instruct",
        max_new_tokens=4,
        local_reserve_free_memory_gb=3.0,
    )
    monkeypatch.setenv("COMFY_MODAL_REMOTE_WORKER", "1")
    monkeypatch.setattr(
        modal_llm_node_module,
        "run_modal_llm_inference",
        lambda **kwargs: fake_result("modal", **kwargs),
    )
    remote_output = modal_llm_node_module.ModalLLM.execute(
        prompt="hello",
        model_profile="smolvlm2-2.2b-instruct",
        enable_reasoning=False,
        max_new_tokens=4,
        reserve_free_vram_gb=17.0,
    )

    assert local_output.result[0] == "local_apple-done"
    assert json.loads(local_output.result[1])["execution_target"] == "local_apple"
    assert remote_output.result[0] == "modal-done"
    assert json.loads(remote_output.result[1])["execution_target"] == "modal"
    assert remote_output.result[2] == "working"
    assert calls == [
        ("local_apple", 3.0, True),
        ("modal", 17.0, False),
    ]

def test_resident_manager_reports_ssh_docker_execution_target(
    modal_llm_runtime_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A self-hosted CUDA worker must not identify its inference as Modal."""
    monkeypatch.setenv("COMFY_MODAL_LLM_EXECUTION_TARGET", "ssh_docker")
    monkeypatch.setenv("COMFY_MODAL_REMOTE_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setattr(modal_llm_runtime_module, "_RESIDENT_MANAGER", None)

    manager = modal_llm_runtime_module.get_resident_llm_manager()

    assert manager.execution_target == "ssh_docker"
    assert manager.device_name == "cuda (SSH Docker)"

def test_remote_dispatch_stages_llm_once_and_forces_volume_reload(
    remote_modal_app_module: Any,
    modal_llm_profile_staging_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The local dispatcher should finish CPU staging before sending work to a GPU."""
    stage_calls: list[list[str]] = []

    class FakeStageMethod:
        """Expose the Modal method remote-call surface."""

        def remote(self, profile_ids: list[str]) -> list[dict[str, str]]:
            """Record and confirm every requested profile."""
            stage_calls.append(profile_ids)
            return [
                {
                    "requested_reference": profile_id,
                    "profile_id": profile_id,
                    "revision": "482adb537c021c86670beed01cd58990d01e72e4",
                }
                for profile_id in profile_ids
            ]

    class FakeStager:
        """Expose the deployed CPU staging method."""

        stage_profiles = FakeStageMethod()

    class FakeRemoteClass:
        """Construct one fake deployed stager instance."""

        def __call__(self) -> FakeStager:
            """Return a fake stager."""
            return FakeStager()

    class FakeCls:
        """Resolve only the expected deployed stager class."""

        @staticmethod
        def from_name(app_name: str, class_name: str) -> FakeRemoteClass:
            """Return a fake class handle after validating lookup identity."""
            assert app_name == "test-b300-app"
            assert class_name == "ModelStager"
            return FakeRemoteClass()

    monkeypatch.setattr(
        modal_llm_profile_staging_module,
        "modal",
        SimpleNamespace(Cls=FakeCls),
    )
    with modal_llm_profile_staging_module._STAGED_LLM_PROFILES_LOCK:
        modal_llm_profile_staging_module._STAGED_LLM_PROFILES.clear()
        modal_llm_profile_staging_module._STAGED_LLM_PROFILE_RESULTS.clear()
    payload = {
        "component_id": "llm-component",
        "subgraph_prompt": {
            "1": {
                "class_type": "ModalLLM",
                "inputs": {"model_profile": "smolvlm2-2.2b-instruct"},
            }
        },
    }

    modal_llm_profile_staging_module._ensure_llm_profiles_staged(payload, "test-b300-app")
    first_marker = payload["volume_reload_marker"]
    modal_llm_profile_staging_module._ensure_llm_profiles_staged(payload, "test-b300-app")

    assert stage_calls == [["smolvlm2-2.2b-instruct"]]
    assert payload["requires_volume_reload"] is True
    assert payload["volume_reload_marker"] == first_marker

def test_remote_dispatch_streams_cpu_llm_staging_progress(
    remote_modal_app_module: Any,
    modal_llm_profile_staging_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First-run Hugging Face progress should render before GPU allocation."""
    observed_progress: list[dict[str, Any]] = []

    class FakeStageStream:
        """Yield one download update followed by the immutable profile result."""

        def remote_gen(self, model_references: list[str]) -> Any:
            """Stream deterministic CPU staging envelopes."""
            assert model_references == ["owner/model"]
            yield {
                "kind": "progress",
                "stage": "download",
                "message": "Fetching 8 files",
                "value": 3,
                "max": 8,
                "unit": "files",
                "model_reference": "owner/model",
            }
            yield {
                "kind": "result",
                "results": [
                    {
                        "requested_reference": "owner/model",
                        "profile_id": "hf-" + "b" * 64,
                        "revision": "8" * 40,
                    }
                ],
            }

    class FakeCls:
        """Resolve the streaming CPU ModelStager."""

        @staticmethod
        def from_name(app_name: str, class_name: str) -> Callable[[], Any]:
            """Return a deterministic streaming stager instance."""
            assert app_name == "test-app"
            assert class_name == "ModelStager"
            return lambda: SimpleNamespace(stage_profiles_stream=FakeStageStream())

    monkeypatch.setattr(
        modal_llm_profile_staging_module,
        "modal",
        SimpleNamespace(Cls=FakeCls),
    )
    monkeypatch.setattr(
        modal_llm_profile_staging_module,
        "_emit_local_llm_staging_progress",
        lambda payload, event: observed_progress.append(dict(event)),
    )
    with modal_llm_profile_staging_module._STAGED_LLM_PROFILES_LOCK:
        modal_llm_profile_staging_module._STAGED_LLM_PROFILES.clear()
        modal_llm_profile_staging_module._STAGED_LLM_PROFILE_RESULTS.clear()
    payload = {
        "prompt_id": "prompt-1",
        "component_id": "llm-node",
        "extra_data": {"client_id": "client-1"},
        "subgraph_prompt": {
            "llm-node": {
                "class_type": "ModalLLM",
                "inputs": {"model_profile": "owner/model"},
            }
        },
    }

    modal_llm_profile_staging_module._ensure_llm_profiles_staged(payload, "test-app")

    assert observed_progress == [
        {
            "kind": "progress",
            "stage": "download",
            "message": "Fetching 8 files",
            "value": 3,
            "max": 8,
            "unit": "files",
            "model_reference": "owner/model",
        }
    ]
    assert (
        payload["subgraph_prompt"]["llm-node"]["inputs"]["model_profile"]
        == "hf-" + "b" * 64
    )

def test_modal_staging_stream_has_bounded_no_progress_wait(
    remote_modal_app_module: Any,
    modal_llm_profile_staging_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A silent Modal generator should be closed instead of hanging forever."""

    class SilentStream:
        """Block until the controller asks this fake remote stream to close."""

        def __init__(self) -> None:
            self.closed = threading.Event()

        def __iter__(self) -> Any:
            """Wait without producing a staging event."""
            self.closed.wait(timeout=5)
            return
            yield  # pragma: no cover - makes this method a generator.

        def close(self) -> None:
            """Model successful cancellation of the remote generator."""
            self.closed.set()

    monkeypatch.setenv(
        "COMFY_MODAL_LLM_STAGE_NO_PROGRESS_TIMEOUT_SECONDS",
        "0.05",
    )
    stream = SilentStream()

    with pytest.raises(
        remote_modal_app_module.ModalRemoteInvocationError,
        match="produced no progress",
    ):
        list(modal_llm_profile_staging_module._bounded_modal_stage_events(stream))

    assert stream.closed.is_set()

def test_llm_staging_progress_targets_the_matching_llm_node(
    remote_modal_app_module: Any,
    modal_llm_profile_staging_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A component representative must not receive an inner LLM staging bar."""
    observed_progress: list[dict[str, Any]] = []
    monkeypatch.setattr(
        modal_llm_profile_staging_module,
        "_emit_local_modal_progress",
        lambda **kwargs: observed_progress.append(kwargs),
    )
    payload = {
        "prompt_id": "prompt-1",
        "component_id": "11",
        "component_node_ids": ["11", "249:263", "289:288"],
        "extra_data": {"client_id": "client-1"},
        "subgraph_prompt": {
            "11": {"class_type": "VAELoader", "inputs": {}},
            "249:263": {
                "class_type": "ModalLLM",
                "inputs": {"model_profile": "owner/model"},
            },
            "289:288": {
                "class_type": "ModalLLM",
                "inputs": {"model_profile": "owner/other-model"},
            },
        },
    }

    modal_llm_profile_staging_module._emit_local_llm_staging_progress(
        payload,
        {
            "stage": "metadata",
            "message": "Inspecting Hugging Face metadata for owner/model",
            "indeterminate": True,
            "model_reference": "owner/model",
        },
    )

    assert [event["node_id"] for event in observed_progress] == ["249:263"]
    assert observed_progress[0]["stage"] == "metadata"
    assert observed_progress[0]["pre_gpu"] is True

def test_remote_runtime_registers_deployment_owned_llm_node(
    modal_cloud_module: Any,
) -> None:
    """The remote image should expose ModalLLM even when custom-node sync is absent."""
    nodes_module = SimpleNamespace(
        NODE_CLASS_MAPPINGS={},
        NODE_DISPLAY_NAME_MAPPINGS={},
    )

    modal_cloud_module._register_modal_sync_runtime_nodes(nodes_module)

    assert nodes_module.NODE_CLASS_MAPPINGS["ModalLLM"].__name__ == "ModalLLM"
    assert nodes_module.NODE_DISPLAY_NAME_MAPPINGS["ModalLLM"] == "Modal LLM"

