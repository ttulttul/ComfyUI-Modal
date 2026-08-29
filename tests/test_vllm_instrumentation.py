"""Tests for the vllm instrumentation boundary."""

from __future__ import annotations

from modal_llm_test_support import *  # noqa: F401,F403

def test_accurate_triton_listener_distinguishes_persistent_cache_hits(
    modal_llm_runtime_module: Any,
    vllm_instrumentation_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The vLLM EngineCore wrapper should signal only genuine compilations."""
    signal_path = tmp_path / "triton-compile-misses.jsonl"
    status_path = tmp_path / "triton-listeners.jsonl"
    previous_events: list[dict[str, Any]] = []
    engine_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    fake_triton = SimpleNamespace(
        knobs=SimpleNamespace(
            compilation=SimpleNamespace(
                listener=lambda **event: previous_events.append(event)
            )
        )
    )
    original_setup_hook = object()
    fake_jit_monitor = SimpleNamespace(
        _setup_triton_jit_hook=original_setup_hook,
    )

    class FakeEngineCoreProc:
        """Expose the spawn entrypoint patched by the Modal runtime."""

        @staticmethod
        def run_engine_core(*args: Any, **kwargs: Any) -> str:
            """Record execution after the child listener is installed."""
            engine_calls.append((args, kwargs))
            return "engine-complete"

    fake_engine_core = SimpleNamespace(EngineCoreProc=FakeEngineCoreProc)

    def import_module(name: str) -> Any:
        """Return the two runtime modules used by listener installation."""
        if name == "triton":
            return fake_triton
        if name == "vllm.utils.jit_monitor":
            return fake_jit_monitor
        if name == "vllm.v1.engine.core":
            return fake_engine_core
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(
        vllm_instrumentation_module,
        "_TRITON_COMPILE_MISS_SIGNAL_PATH",
        signal_path,
    )
    monkeypatch.setattr(
        vllm_instrumentation_module,
        "_TRITON_COMPILE_LISTENER_STATUS_PATH",
        status_path,
    )
    monkeypatch.setattr(
        vllm_instrumentation_module,
        "_TRITON_COMPILE_LISTENER_INSTALLED_PID",
        None,
    )
    monkeypatch.setattr(
        vllm_instrumentation_module,
        "_TRITON_ENGINE_CORE_LISTENER_RECORDED_PID",
        None,
    )
    monkeypatch.setattr(
        vllm_instrumentation_module,
        "_VLLM_ENGINE_CORE_ENTRYPOINT_PATCHED_PID",
        None,
    )
    monkeypatch.setattr(
        vllm_instrumentation_module,
        "_VLLM_ENGINE_CORE_ORIGINAL_ENTRYPOINT",
        None,
    )
    monkeypatch.setattr(
        vllm_instrumentation_module.importlib,
        "import_module",
        import_module,
    )
    vllm_instrumentation_module._install_accurate_triton_compile_listener()
    result = fake_engine_core.EngineCoreProc.run_engine_core("spawned")

    listener = fake_triton.knobs.compilation.listener
    compile_times = SimpleNamespace(
        ir_initialization=1200,
        lowering_stages=[("ttir", 2300), ("cubin", 3400)],
        store_results=500,
        total=7400,
    )
    common_event = {
        "src": SimpleNamespace(name="example_kernel"),
        "metadata": {"hash": "artifact-1"},
        "times": compile_times,
    }
    with caplog.at_level(logging.INFO):
        listener(**common_event, cache_hit=True)
    assert modal_llm_runtime_module.triton_compile_miss_signal_size() == 0
    assert "cache_hit=True" in caplog.text

    listener(**common_event, cache_hit=False)

    assert modal_llm_runtime_module.triton_compile_miss_signal_size() > 0
    miss_event = json.loads(signal_path.read_text(encoding="utf-8"))
    assert miss_event["cache_hit"] is False
    assert miss_event["kernel"] == "example_kernel"
    assert miss_event["timing"]["total_ms"] == 7.4
    assert len(previous_events) == 2
    assert fake_jit_monitor._setup_triton_jit_hook is not original_setup_hook
    assert result == "engine-complete"
    assert engine_calls == [(("spawned",), {})]
    assert modal_llm_runtime_module.triton_compile_listener_engine_pids() == (
        modal_llm_runtime_module.os.getpid(),
    )

def test_spawned_engine_core_recovers_original_entrypoint_and_installs_listener(
    modal_llm_runtime_module: Any,
    vllm_instrumentation_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A spawn-fresh module must install telemetry before EngineCore startup."""
    startup_observations: list[tuple[int, ...]] = []
    fake_triton = SimpleNamespace(
        knobs=SimpleNamespace(compilation=SimpleNamespace(listener=None))
    )
    fake_jit_monitor = SimpleNamespace(_setup_triton_jit_hook=lambda: None)

    class FakeEngineCoreProc:
        """Represent vLLM's unpatched class in a spawned interpreter."""

        @staticmethod
        def run_engine_core() -> str:
            """Observe listener readiness at the first EngineCore instruction."""
            startup_observations.append(
                modal_llm_runtime_module.triton_compile_listener_engine_pids()
            )
            return "started"

    modules = {
        "triton": fake_triton,
        "vllm.utils.jit_monitor": fake_jit_monitor,
        "vllm.v1.engine.core": SimpleNamespace(EngineCoreProc=FakeEngineCoreProc),
    }
    monkeypatch.setattr(
        vllm_instrumentation_module.importlib,
        "import_module",
        lambda name: modules[name],
    )
    monkeypatch.setattr(
        vllm_instrumentation_module,
        "_TRITON_COMPILE_LISTENER_STATUS_PATH",
        tmp_path / "listeners.jsonl",
    )
    monkeypatch.setattr(
        vllm_instrumentation_module,
        "_TRITON_COMPILE_LISTENER_INSTALLED_PID",
        None,
    )
    monkeypatch.setattr(
        vllm_instrumentation_module,
        "_TRITON_ENGINE_CORE_LISTENER_RECORDED_PID",
        None,
    )
    monkeypatch.setattr(
        vllm_instrumentation_module,
        "_VLLM_ENGINE_CORE_ORIGINAL_ENTRYPOINT",
        None,
    )

    result = (
        vllm_instrumentation_module._run_vllm_engine_core_with_accurate_triton_listener()
    )

    assert result == "started"
    assert startup_observations == [(modal_llm_runtime_module.os.getpid(),)]
    assert fake_triton.knobs.compilation.listener._comfy_modal_cache_aware

@pytest.mark.parametrize(
    ("configured_mode", "expected_mode", "expected_enforce_eager"),
    [
        ("auto", "eager", True),
        ("eager", "eager", True),
        ("throughput", "throughput", False),
    ],
)
def test_vllm_execution_setting_selects_initial_engine_policy(
    modal_llm_runtime_module: Any,
    llm_profiles_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    configured_mode: str,
    expected_mode: str,
    expected_enforce_eager: bool,
) -> None:
    """Pinned settings should win while auto starts with a low-latency eager engine."""
    profile = replace(
        llm_profiles_module.get_llm_profile("smolvlm2-2.2b-instruct"),
        backend="vllm",
        backend_options=(("enforce_eager", True),),
    )
    monkeypatch.setenv("COMFY_MODAL_LLM_VLLM_EXECUTION_MODE", configured_mode)

    mode, enforce_eager = modal_llm_runtime_module._vllm_execution_policy(profile)

    assert mode == expected_mode
    assert enforce_eager is expected_enforce_eager

@pytest.mark.parametrize("setting", ["eager", "throughput"])
def test_vllm_pinned_mode_never_auto_promotes(
    modal_llm_runtime_module: Any,
    setting: str,
) -> None:
    """Pinned controllers must ignore container workflow-count changes."""
    controller = modal_llm_runtime_module.VLLMExecutionModeController(setting)

    assert controller.observe("workflow-1") is False
    assert controller.observe("workflow-2") is False
    assert controller.effective_mode() == setting
    assert controller.promoted is False
    assert controller.observed_workflow_count == 0

def test_vllm_auto_mode_can_promote_before_first_llm_load(
    modal_llm_runtime_module: Any,
    llm_profiles_module: Any,
) -> None:
    """An image-only first workflow should still make a later LLM use throughput."""
    profile = replace(
        llm_profiles_module.get_llm_profile("smolvlm2-2.2b-instruct"),
        backend="vllm",
    )
    controller = modal_llm_runtime_module.VLLMExecutionModeController("auto")

    controller.observe("image-workflow")
    controller.observe("llm-workflow")
    runtime_profile = modal_llm_runtime_module._profile_for_vllm_execution(
        profile,
        controller,
    )

    assert (
        runtime_profile.backend_option(
            modal_llm_runtime_module._VLLM_RUNTIME_MODE_OPTION
        )
        == "throughput"
    )

def test_vllm_auto_mode_preserves_throughput_on_memory_recovery_worker(
    modal_llm_runtime_module: Any,
) -> None:
    """A fresh recovery worker must not reset a promoted retry back to eager."""
    controller = modal_llm_runtime_module.VLLMExecutionModeController("auto")

    promoted = controller.force_throughput_after_memory_recovery("workflow-2")

    assert promoted is True
    assert controller.effective_mode() == "throughput"
    assert controller.promoted is True
    assert controller.observed_workflow_count == 2
    assert controller.observe("workflow-2") is False

def test_vllm_auto_mode_promotes_on_second_distinct_workflow(
    modal_llm_runtime_module: Any,
    llm_profiles_module: Any,
    tmp_path: Path,
) -> None:
    """Repeat components stay eager, then the next workflow rebuilds for throughput."""
    profile = replace(
        llm_profiles_module.get_llm_profile("smolvlm2-2.2b-instruct"),
        backend="vllm",
    )
    controller = modal_llm_runtime_module.VLLMExecutionModeController("auto")
    loaded_backends: list[Any] = []

    class ModeBackend(_FakeBackend):
        """Expose the ephemeral vLLM mode attached by the resident manager."""

        def __init__(self, runtime_profile: Any) -> None:
            """Record the mode selected for this fake engine construction."""
            super().__init__(runtime_profile.profile_id)
            self.mode = runtime_profile.backend_option(
                modal_llm_runtime_module._VLLM_RUNTIME_MODE_OPTION
            )

        def runtime_metadata(self) -> dict[str, Any]:
            """Mirror the real vLLM backend's effective-mode telemetry."""
            return {"vllm_execution_mode": self.mode}

    def backend_factory(
        runtime_profile: Any,
        snapshot_path: Path,
        progress_callback: Callable[[Any], None],
    ) -> ModeBackend:
        """Create a mode-aware fake backend for each engine rebuild."""
        del snapshot_path, progress_callback
        backend = ModeBackend(runtime_profile)
        loaded_backends.append(backend)
        return backend

    comfy_memory_releases: list[int] = []
    manager = modal_llm_runtime_module.ResidentLLMManager(
        storage_root=tmp_path,
        backend_factory=backend_factory,
        memory_info=lambda: (200 * 1024**3, 256 * 1024**3),
        empty_cache=lambda: None,
        snapshot_ready=lambda storage_root, selected_profile: True,
        comfy_memory_release=comfy_memory_releases.append,
        vllm_mode_controller=controller,
    )
    prepared = modal_llm_runtime_module.PreparedLLMInputs(
        prompt="hello",
        system_prompt="",
        images=(),
        video=None,
        file_characters=0,
        file_count=0,
    )
    generation = modal_llm_runtime_module.LLMGenerationSettings(
        max_new_tokens=8,
        temperature=0.0,
        top_p=1.0,
        seed=0,
    )
    progress: list[Any] = []

    def infer(workflow_execution_id: str) -> Any:
        """Run one fake workflow through the shared resident manager."""
        return manager.infer(
            profile=profile,
            prepared_inputs=prepared,
            generation_settings=generation,
            reserve_free_vram_gb=0,
            keep_model_loaded=True,
            progress_callback=progress.append,
            workflow_execution_id=workflow_execution_id,
        )

    first = infer("workflow-1")
    same_workflow = infer("workflow-1")
    promoted = infer("workflow-2")
    warm_throughput = infer("workflow-3")

    assert first.metadata["vllm_execution_mode"] == "eager"
    assert first.metadata["cache_hit"] is False
    assert same_workflow.metadata["cache_hit"] is True
    assert promoted.metadata["vllm_execution_mode"] == "throughput"
    assert promoted.metadata["vllm_execution_setting"] == "auto"
    assert promoted.metadata["vllm_auto_promoted"] is True
    assert promoted.metadata["vllm_observed_workflow_count"] == 2
    assert promoted.metadata["cache_hit"] is False
    assert warm_throughput.metadata["cache_hit"] is True
    assert [backend.mode for backend in loaded_backends] == ["eager", "throughput"]
    assert loaded_backends[0].unloaded is True
    assert len(comfy_memory_releases) == 2
    assert any(
        event.stage == "engine"
        and event.message == "Optimizing vLLM for repeat workflows"
        for event in progress
    )

