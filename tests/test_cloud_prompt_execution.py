"""Tests split from the Modal executor integration suite."""

from __future__ import annotations

from modal_executor_test_support import *  # noqa: F401,F403

def test_stable_modal_cloud_entry_imports_without_modal_sdk(
    modal_cloud_module: Any,
) -> None:
    """The stable Modal cloud module should stay importable when modal is unavailable."""
    assert modal_cloud_module.__name__ == "comfyui_modal_sync_cloud"
    assert hasattr(modal_cloud_module, "RemoteEngine")

def test_modal_cloud_exposes_affinity_parameter_and_local_gap_keepalive() -> None:
    """The deployed class must make affinity part of identity and expose a heartbeat."""
    source = (Path(__file__).resolve().parents[1] / "comfyui_modal_sync_cloud.py").read_text()

    assert "worker_affinity_key: str = modal.parameter" in source
    assert "def keepalive_for_local_gap" in source

def test_modal_cloud_image_environment_preserves_unique_app_name(
    modal_cloud_module: Any,
) -> None:
    """Remote workers should receive the same per-ComfyUI app name resolved locally."""
    settings = types.SimpleNamespace(
        app_name="comfy-modal-sync-AAECAwQFBgc",
        modal_gpu="B300",
        modal_secret_name="workflow-credentials",
        stream_event_queue_maxsize=256,
        bridge_inline_max_bytes=1024,
        invocation_result_inline_max_bytes=2048,
        execution_timeout_seconds=3600,
        startup_timeout_seconds=900,
        llm_vllm_execution_mode="throughput",
    )

    image_environment = modal_cloud_module._modal_image_environment(settings, "fingerprint-1")

    assert image_environment["COMFY_MODAL_APP_NAME"] == "comfy-modal-sync-AAECAwQFBgc"
    assert image_environment["COMFY_MODAL_GPU"] == "B300"
    assert image_environment["COMFY_MODAL_REMOTE_STORAGE_ROOT"] == "/storage"
    assert image_environment["COMFY_MODAL_REMOTE_WORKER"] == "1"
    assert image_environment["COMFY_MODAL_LLM_VLLM_EXECUTION_MODE"] == "throughput"
    assert image_environment["VLLM_CACHE_ROOT"].startswith(
        "/root/.cache/comfy-modal-llm/"
    )
    assert image_environment["TORCHINDUCTOR_CACHE_DIR"].startswith(
        "/root/.cache/comfy-modal-llm/"
    )
    assert image_environment["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    assert image_environment["COMFY_MODAL_LLM_MAX_RESIDENT_MODELS"] == "2"
    assert image_environment["COMFY_MODAL_LLM_MEMORY_RECOVERY_TIMEOUT_SECONDS"] == "15.0"
    assert image_environment["COMFY_MODAL_LLM_RESERVE_FREE_GB"] == "24.0"
    assert image_environment["COMFY_MODAL_SECRET_NAME"] == "workflow-credentials"
    assert image_environment["COMFY_MODAL_RUNTIME_FINGERPRINT"] == "fingerprint-1"

def test_modal_cloud_observes_distinct_workflows_for_llm_auto_mode(
    modal_cloud_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The container boundary should count prompts while excluding canary traffic."""
    observed_prompt_ids: list[str | None] = []
    forced_prompt_ids: list[str | None] = []
    runtime_module = types.ModuleType("modal_llm_runtime")
    runtime_module.observe_modal_workflow_execution = observed_prompt_ids.append
    runtime_module.force_modal_vllm_throughput_after_memory_recovery = (
        forced_prompt_ids.append
    )
    monkeypatch.setitem(sys.modules, "modal_llm_runtime", runtime_module)

    modal_cloud_module._observe_remote_workflow_for_llm_mode(
        {"prompt_id": "prompt-1", "payload_kind": "subgraph"}
    )
    modal_cloud_module._observe_remote_workflow_for_llm_mode(
        {"prompt_id": "prompt-1", "payload_kind": "mapped_subgraph"}
    )
    modal_cloud_module._observe_remote_workflow_for_llm_mode(
        {"prompt_id": "health-check", "payload_kind": "canary"}
    )
    modal_cloud_module._observe_remote_workflow_for_llm_mode(
        {"prompt_id": "prompt-2", "payload_kind": "subgraph"}
    )
    modal_cloud_module._observe_remote_workflow_for_llm_mode(
        {
            "prompt_id": "prompt-3",
            "payload_kind": "subgraph",
            "force_vllm_throughput_after_memory_recovery": True,
        }
    )

    assert observed_prompt_ids == ["prompt-1", "prompt-1", "prompt-2"]
    assert forced_prompt_ids == ["prompt-3"]

def test_modal_cloud_resolves_configured_named_secret(
    modal_cloud_module: Any,
) -> None:
    """Cloud app construction should reference the configured collection without reading .env."""
    observed_names: list[str] = []
    expected_secret = object()
    fake_modal = types.SimpleNamespace(
        Secret=types.SimpleNamespace(
            from_name=lambda name: observed_names.append(name) or expected_secret,
        )
    )

    resolved_secret = modal_cloud_module._modal_secret_from_settings(
        types.SimpleNamespace(modal_secret_name="workflow-credentials"),
        fake_modal,
    )

    assert resolved_secret is expected_secret
    assert observed_names == ["workflow-credentials"]

def test_modal_cloud_installs_timestamped_logger_handler(
    modal_cloud_module: Any,
) -> None:
    """The cloud runtime should install its own timestamped logger handler."""
    matching_handlers = [
        handler
        for handler in modal_cloud_module.logger.handlers
        if getattr(handler, "name", "") == modal_cloud_module._CLOUD_HANDLER_NAME
    ]

    assert len(matching_handlers) == 1
    assert modal_cloud_module.logger.propagate is False
    assert modal_cloud_module.logger.level == logging.INFO
    assert matching_handlers[0].stream is sys.stdout
    formatter = matching_handlers[0].formatter
    assert isinstance(formatter, logging.Formatter)
    assert "%(asctime)s" in formatter._fmt
    assert "%(relativeCreated)" in formatter._fmt

def test_modal_cloud_traces_remote_node_execution_spans(
    modal_cloud_module: Any,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    """The tracing prompt server should emit per-node timing lines."""
    prompt = {
        "7": {"class_type": "UNETLoader", "inputs": {}},
        "2": {"class_type": "KSampler", "inputs": {}},
    }
    monkeypatch.setenv("MODAL_IS_REMOTE", "1")
    server = modal_cloud_module._TracingPromptServer("component-1", prompt)

    server.send_sync("executing", {"node": "7"}, None)
    server.send_sync("executed", {"node": "7"}, None)
    server.send_sync("executing", {"node": "2"}, None)
    server.send_sync("execution_success", {"prompt_id": "component-1"}, None)

    captured = capsys.readouterr()
    assert "Remote node 7 class_type=UNETLoader role=model_load started" in captured.out
    assert "Remote node 7 class_type=UNETLoader role=model_load finished in " in captured.out
    assert "Remote node 2 class_type=KSampler role=sampling started" in captured.out
    assert "Remote node 2 class_type=KSampler role=sampling finished in " in captured.out

def test_modal_cloud_installs_headless_prompt_server_instance(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """Remote custom-node init should get a minimal PromptServer.instance shim."""
    class FakeNodeReplaceManager:
        """Record replacement registrations made through the headless server."""

        def __init__(self) -> None:
            """Initialize an empty replacement registry."""
            self.registrations: list[Any] = []

        def register(self, replacement: Any) -> None:
            """Record one replacement registered by a ComfyUI extension."""
            self.registrations.append(replacement)

    fake_prompt_server_class = type("PromptServer", (), {})
    fake_server_module = types.SimpleNamespace(
        NodeReplaceManager=FakeNodeReplaceManager,
        PromptServer=fake_prompt_server_class,
    )

    monkeypatch.setitem(sys.modules, "server", fake_server_module)
    modal_cloud_module._ensure_headless_prompt_server_instance()

    instance = fake_prompt_server_class.instance
    assert instance is not None
    assert hasattr(instance, "routes")
    assert hasattr(instance, "app")
    assert isinstance(instance.node_replace_manager, FakeNodeReplaceManager)
    assert instance.supports == ["custom_nodes_from_web"]
    assert instance.client_id is None
    assert instance.last_node_id is None
    assert instance.number == 0
    assert instance.prompt_queue.currently_running == {}
    assert instance.prompt_queue.get_current_queue() == ([], [])

    instance.send_progress_text("width: 1024, height: 768", "104")

    instance.add_on_prompt_handler("handler")
    assert instance.on_prompt_handlers == ["handler"]

    instance.node_replace_manager.register("replacement")
    assert instance.node_replace_manager.registrations == ["replacement"]

    instance.prompt_queue.put((1, "prompt-id", {}, {}, []))
    assert instance.prompt_queue.get_current_queue() == (
        [],
        [(1, "prompt-id", {}, {}, [])],
    )
    assert instance.prompt_queue.get_tasks_remaining() == 1

    instance.prompt_queue.set_flag("free_memory", True)
    assert instance.prompt_queue.get_flags() == {"free_memory": True}
    assert instance.prompt_queue.get_flags() == {}

def test_modal_cloud_dynamic_prompt_preserves_thread_local_metadata_graph(
    modal_cloud_module: Any,
) -> None:
    """Hydrated execution graphs should not replace the JSON-safe hidden PROMPT graph."""

    class FakeDynamicPrompt:
        """Minimal ComfyUI DynamicPrompt stand-in."""

        def __init__(self, original_prompt: dict[str, Any]) -> None:
            """Retain the graph used for runtime dependency resolution."""
            self.original_prompt = original_prompt

        def get_original_prompt(self) -> dict[str, Any]:
            """Return the runtime graph when no metadata override is active."""
            return self.original_prompt

    fake_execution = types.SimpleNamespace(DynamicPrompt=FakeDynamicPrompt)
    hydrated_prompt = {"1": {"inputs": {"image": object()}}}
    metadata_prompt = {"1": {"inputs": {"image": ["upstream", 0]}}}

    modal_cloud_module._install_metadata_safe_dynamic_prompt_wrapper(fake_execution)
    with modal_cloud_module._temporary_prompt_metadata(metadata_prompt):
        dynamic_prompt = fake_execution.DynamicPrompt(hydrated_prompt)

    ordinary_prompt = fake_execution.DynamicPrompt(hydrated_prompt)

    assert dynamic_prompt.original_prompt is hydrated_prompt
    assert dynamic_prompt.get_original_prompt() is metadata_prompt
    assert ordinary_prompt.get_original_prompt() is hydrated_prompt

def test_modal_cloud_hidden_prompt_stays_json_safe_after_tensor_boundary_hydration(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """Metadata-writing output nodes should receive links, not hydrated tensors, in PROMPT."""
    import torch

    monkeypatch.setitem(sys.modules, "torchsde", types.ModuleType("torchsde"))
    import comfy_execution.caching as comfy_caching

    modal_cloud_module._ensure_comfy_runtime_initialized(None)
    nodes_module = modal_cloud_module._load_nodes_module()
    monkeypatch.setitem(
        nodes_module.NODE_CLASS_MAPPINGS,
        "PromptMetadataSerializationNode",
        _PromptMetadataSerializationNode,
    )
    monkeypatch.setitem(
        comfy_caching.nodes.NODE_CLASS_MAPPINGS,
        "PromptMetadataSerializationNode",
        _PromptMetadataSerializationNode,
    )
    monkeypatch.setitem(
        nodes_module.NODE_DISPLAY_NAME_MAPPINGS,
        "PromptMetadataSerializationNode",
        "PromptMetadataSerializationNode",
    )
    monkeypatch.setattr(
        _cloud_prompt_execution_owner(), "_node_output_cache_store", lambda: None
    )
    payload = {
        "payload_kind": "subgraph",
        "component_id": "metadata-output",
        "prompt_id": "prompt-metadata-output",
        "component_node_ids": ["output"],
        "subgraph_prompt": {
            "output": {
                "class_type": "PromptMetadataSerializationNode",
                "inputs": {"image": ["upstream", 0]},
            }
        },
        "boundary_inputs": [
            {
                "proxy_input_name": "remote_image",
                "io_type": "IMAGE",
                "source_signature": "image-signature",
                "targets": [{"node_id": "output", "input_name": "image"}],
            }
        ],
        "boundary_outputs": [
            {
                "proxy_output_name": "metadata",
                "node_id": "output",
                "output_index": 0,
                "io_type": "STRING",
                "is_list": False,
            }
        ],
        "execute_node_ids": ["output"],
        "extra_data": {},
    }

    outputs = modal_cloud_module._execute_subgraph_prompt(
        payload,
        {"remote_image": torch.zeros((1, 2, 2, 3), dtype=torch.float32)},
        None,
    )

    assert len(outputs) == 1
    assert json.loads(outputs[0]) == payload["subgraph_prompt"]

def test_modal_cloud_only_reloads_volume_for_requests_with_new_uploads(
    modal_cloud_module: Any,
) -> None:
    """Steady-state requests should skip Modal volume reload when queue-time sync uploaded nothing."""

    assert modal_cloud_module._should_reload_modal_volume({"requires_volume_reload": True}) is True
    assert modal_cloud_module._should_reload_modal_volume({"requires_volume_reload": False}) is False
    assert modal_cloud_module._should_reload_modal_volume({}) is True

def test_modal_cloud_skips_reload_when_uploaded_paths_are_already_visible(
    modal_cloud_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Visible immutable uploaded paths should not force an extra Modal volume reload."""
    storage_root = tmp_path / "storage"
    storage_root.mkdir()
    uploaded_path = storage_root / "assets" / "hash_model.safetensors"
    uploaded_path.parent.mkdir(parents=True, exist_ok=True)
    uploaded_path.write_bytes(b"weights")
    recorded_markers: list[str] = []

    _patch_cloud_storage_root(monkeypatch, modal_cloud_module, storage_root)
    monkeypatch.setattr(
        _cloud_volume_reload_owner(),
        "_record_modal_volume_reload_marker",
        lambda marker: recorded_markers.append(marker),
    )

    payload = {
        "requires_volume_reload": True,
        "volume_reload_marker": "marker-1",
        "uploaded_volume_paths": ["/assets/hash_model.safetensors"],
    }

    assert modal_cloud_module._should_reload_modal_volume(payload) is False
    assert recorded_markers == ["marker-1"]

def test_modal_cloud_reads_missing_committed_asset_without_reloading_volume(
    modal_cloud_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """A stale warm mount should read a committed asset into ephemeral storage."""

    class FakeVolume:
        """Modal Volume double that exposes committed file reads."""

        def __init__(self) -> None:
            """Record direct read paths."""
            self.read_paths: list[str] = []

        def read_file(self, path: str) -> Iterator[bytes]:
            """Yield one committed model file in chunks."""
            self.read_paths.append(path)
            yield b"model-"
            yield b"weights"

    storage_root = tmp_path / "storage"
    storage_root.mkdir()
    readthrough_root = tmp_path / "readthrough"
    _patch_cloud_storage_root(monkeypatch, modal_cloud_module, storage_root)
    bootstrap_owner = _cloud_comfy_bootstrap_owner()
    monkeypatch.setattr(
        bootstrap_owner,
        "_REMOTE_VOLUME_READTHROUGH_ROOT",
        readthrough_root,
    )
    payload = {
        "component_id": "component-1",
        "requires_volume_reload": False,
        "subgraph_prompt": {
            "14": {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {"lora_name": "/assets/hash_style.safetensors"},
            }
        },
    }

    volume = FakeVolume()
    hydrated_paths = modal_cloud_module._hydrate_missing_payload_volume_paths(volume, payload)

    cached_path = readthrough_root / "assets" / "hash_style.safetensors"
    assert hydrated_paths == [cached_path]
    assert cached_path.read_bytes() == b"model-weights"
    assert volume.read_paths == ["assets/hash_style.safetensors"]
    assert modal_cloud_module._should_reload_modal_volume(payload) is False
    assert (
        modal_cloud_module._resolve_runtime_asset_path("/assets/hash_style.safetensors")
        == str(cached_path)
    )

def test_modal_cloud_reloads_when_reused_referenced_asset_is_still_missing(
    modal_cloud_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """A sync-index hit must not suppress recovery for a missing runtime asset."""
    storage_root = tmp_path / "storage"
    storage_root.mkdir()
    _patch_cloud_storage_root(monkeypatch, modal_cloud_module, storage_root)
    monkeypatch.setattr(
        _cloud_comfy_bootstrap_owner(),
        "_REMOTE_VOLUME_READTHROUGH_ROOT",
        tmp_path / "readthrough",
    )
    payload = {
        "requires_volume_reload": False,
        "subgraph_prompt": {
            "14": {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {"lora_name": "/assets/missing_style.safetensors"},
            }
        },
    }

    assert modal_cloud_module._should_reload_modal_volume(payload) is True

def test_modal_cloud_preserves_worker_for_deterministic_remote_failures(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """Ordinary node failures should not discard a healthy warm Modal worker."""
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
            RuntimeError("boom"),
        )
    finally:
        monkeypatch.setattr(modal_cloud_module, "_CONTAINER_TERMINATION_SCHEDULED", original_flag)

    assert scheduled is False
    assert scheduled_exits == []

def test_modal_cloud_retires_worker_for_poisoned_cuda_runtime(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """CUDA failures that can poison process state should retire the worker."""
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
            RuntimeError("CUDA error: an illegal memory access was encountered"),
        )
    finally:
        monkeypatch.setattr(modal_cloud_module, "_CONTAINER_TERMINATION_SCHEDULED", original_flag)

    assert scheduled is True
    assert scheduled_exits == [(modal_cloud_module._REMOTE_ERROR_CONTAINER_EXIT_DELAY_SECONDS, 1)]

def test_modal_cloud_retires_worker_after_llm_memory_recovery_timeout(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """An unrecovered eviction should retire dirty process state before retry."""
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
            {"component_id": "llm-1", "terminate_container_on_error": True},
            RuntimeError("[comfy-modal-llm-memory-recovery-exhausted] still low"),
        )
    finally:
        monkeypatch.setattr(modal_cloud_module, "_CONTAINER_TERMINATION_SCHEDULED", original_flag)

    assert scheduled is True
    assert scheduled_exits == [(modal_cloud_module._REMOTE_ERROR_CONTAINER_EXIT_DELAY_SECONDS, 1)]

def test_modal_cloud_raises_after_exhausting_open_file_reload_retries(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """Modal volume reload should surface persistent open-file errors after bounded retries."""

    class FakeVolume:
        """Simple Modal volume double that always fails with open files."""

        def __init__(self) -> None:
            """Initialize the reload attempt counter."""
            self.reload_calls = 0

        def reload(self) -> None:
            """Always fail with the same open-file reload error."""
            self.reload_calls += 1
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
    with pytest.raises(RuntimeError, match="open files"):
        modal_cloud_module._reload_modal_volume_for_request(volume, "component-1")

    assert volume.reload_calls == len(
        modal_cloud_module._MODAL_VOLUME_RELOAD_OPEN_FILE_RETRY_DELAYS_SECONDS
    )
    assert prepare_calls == ["prepared"] * (
        len(modal_cloud_module._MODAL_VOLUME_RELOAD_OPEN_FILE_RETRY_DELAYS_SECONDS) - 1
    )
    assert sleep_calls == list(
        modal_cloud_module._MODAL_VOLUME_RELOAD_OPEN_FILE_RETRY_DELAYS_SECONDS[1:]
    )

def test_modal_cloud_proceeds_when_referenced_volume_paths_are_already_visible(
    modal_cloud_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Persistent open-file reload errors may be ignored when the payload's mounted files are already visible."""

    class FakeVolume:
        """Simple Modal volume double that always fails with open files."""

        def __init__(self) -> None:
            """Initialize the reload attempt counter."""
            self.reload_calls = 0

        def reload(self) -> None:
            """Always fail with the same open-file reload error."""
            self.reload_calls += 1
            raise RuntimeError("there are open files preventing the operation")

    storage_root = tmp_path / "storage"
    storage_root.mkdir()
    asset_path = storage_root / "assets" / "hash_model.safetensors"
    asset_path.parent.mkdir(parents=True, exist_ok=True)
    asset_path.write_bytes(b"weights")
    bundle_path = storage_root / "custom_nodes" / "hash_bundle.zip"
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path.write_bytes(b"bundle")

    recorded_markers: list[str] = []
    prepare_calls: list[str] = []
    sleep_calls: list[float] = []
    _patch_cloud_storage_root(monkeypatch, modal_cloud_module, storage_root)
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
    monkeypatch.setattr(
        _cloud_volume_reload_owner(),
        "_record_modal_volume_reload_marker",
        lambda marker: recorded_markers.append(marker),
    )

    payload = {
        "custom_nodes_bundle": "/custom_nodes/hash_bundle.zip",
        "subgraph_prompt": {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "/assets/hash_model.safetensors"},
            }
        },
    }

    volume = FakeVolume()
    modal_cloud_module._reload_modal_volume_for_request(
        volume,
        "component-1",
        reload_marker="marker-1",
        payload=payload,
    )

    assert volume.reload_calls == len(
        modal_cloud_module._MODAL_VOLUME_RELOAD_OPEN_FILE_RETRY_DELAYS_SECONDS
    )
    assert recorded_markers == ["marker-1"]
    assert prepare_calls == ["prepared"] * (
        len(modal_cloud_module._MODAL_VOLUME_RELOAD_OPEN_FILE_RETRY_DELAYS_SECONDS) - 1
    )
    assert sleep_calls == list(
        modal_cloud_module._MODAL_VOLUME_RELOAD_OPEN_FILE_RETRY_DELAYS_SECONDS[1:]
    )

def test_modal_cloud_logs_volume_reload_diagnostics_for_open_file_retries(
    modal_cloud_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Open-file retries should log which uploaded and referenced volume paths matter."""

    class FakeVolume:
        """Simple Modal volume double that always fails with open files."""

        def reload(self) -> None:
            """Always fail with the same open-file reload error."""
            raise RuntimeError("there are open files preventing the operation")

    storage_root = tmp_path / "storage"
    storage_root.mkdir()
    uploaded_path = storage_root / "assets" / "missing_model.safetensors"
    logged_messages: list[tuple[str, tuple[Any, ...]]] = []

    _patch_cloud_storage_root(monkeypatch, modal_cloud_module, storage_root)
    monkeypatch.setattr(
        _cloud_volume_reload_owner(),
        "_prepare_for_modal_volume_reload",
        lambda: None,
    )
    monkeypatch.setattr(
        _cloud_volume_reload_owner(),
        "_sleep_before_modal_volume_reload_retry",
        lambda delay_seconds: None,
    )
    monkeypatch.setattr(
        _cloud_volume_reload_owner().logger,
        "info",
        lambda message, *args: logged_messages.append((message, args)),
    )

    payload = {
        "uploaded_volume_paths": ["/assets/missing_model.safetensors"],
        "subgraph_prompt": {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "/assets/missing_model.safetensors"},
            }
        },
    }

    with pytest.raises(RuntimeError, match="open files"):
        modal_cloud_module._reload_modal_volume_for_request(
            FakeVolume(),
            "component-1",
            payload=payload,
        )

    assert any(
        "Modal volume reload diagnostics for component=%s context=%s uploaded_paths=%s referenced_paths=%s visible_uploaded=%s visible_referenced=%s."
        in message
        and args[0] == "component-1"
        and args[1] == "open_files_retry"
        and args[2] == [str(uploaded_path)]
        and args[3] == [str(uploaded_path)]
        and args[4] is False
        and args[5] is False
        for message, args in logged_messages
    )

def test_modal_cloud_ignores_heavy_comfyui_paths(
    modal_cloud_module: Any,
) -> None:
    """The Modal cloud module should skip heavyweight ComfyUI runtime artifacts."""
    from pathlib import Path

    assert modal_cloud_module._should_ignore_comfyui_path(Path("models/checkpoint.safetensors"))
    assert modal_cloud_module._should_ignore_comfyui_path(Path("custom_nodes/example/__init__.py"))
    assert modal_cloud_module._should_ignore_comfyui_path(Path("output/run/output.png"))
    assert modal_cloud_module._should_ignore_comfyui_path(
        Path(".cache/strings/acfca22bde9a1a1fee53fe6e1299f4fe54a78a6f1d306dbb6cac2e71cf35d2c2.txt")
    )
    assert modal_cloud_module._should_ignore_comfyui_path(Path("__pycache__/execution.pyc"))
    assert modal_cloud_module._should_ignore_comfyui_path(Path("False/checkpoints/model.pth"))
    assert modal_cloud_module._should_ignore_comfyui_path(Path("tests/test_execution.py"))
    assert modal_cloud_module._should_ignore_comfyui_path(Path("unexpected/code.py"))
    assert not modal_cloud_module._should_ignore_comfyui_path(Path("execution.py"))
    assert not modal_cloud_module._should_ignore_comfyui_path(Path("requirements.txt"))
    assert not modal_cloud_module._should_ignore_comfyui_path(Path("comfy/model_management.py"))
    assert not modal_cloud_module._should_ignore_comfyui_path(Path("comfy/ldm/models/diffusion/ddpm.py"))

def test_modal_cloud_installs_comfyui_runtime_packages(
    modal_cloud_module: Any,
) -> None:
    """The Modal cloud image should include the core packages ComfyUI imports at runtime."""
    assert modal_cloud_module._comfyui_apt_packages() == (
        "build-essential",
        "libgl1",
        "libglib2.0-0",
    )

    packages = modal_cloud_module._comfyui_runtime_packages()
    package_names = {package.split("==", maxsplit=1)[0] for package in packages}

    assert all("==" in package for package in packages)
    assert {
        "aiohttp",
        "alembic",
        "blake3",
        "comfy-angle",
        "comfy-aimdo",
        "comfy-kitchen",
        "hf-xet",
        "huggingface-hub",
        "kornia",
        "opencv-python-headless",
        "pydantic-settings",
        "psutil",
        "pyopengl",
        "sentencepiece",
        "simpleeval",
        "spandrel",
        "torchsde",
        "transformers",
    } <= package_names

def test_modal_cloud_selects_gpu_compatible_pytorch_stack(
    modal_cloud_module: Any,
) -> None:
    """Every Modal GPU image should expose the shared pinned PyTorch build."""
    default_build = modal_cloud_module._select_remote_torch_build("A100")
    b300_build = modal_cloud_module._select_remote_torch_build("B300")

    assert default_build.install_layers[0].packages == (
        "torch==2.13.0",
        "torchvision==0.28.0",
    )
    assert default_build.install_layers[0].index_url == "https://download.pytorch.org/whl/cu130"
    assert default_build == b300_build
    assert b300_build.install_layers[0].packages == (
        "torch==2.13.0",
        "torchvision==0.28.0",
    )
    assert b300_build.install_layers[0].index_url == "https://download.pytorch.org/whl/cu130"
    assert b300_build.install_layers[1].packages == ("torchaudio==2.11.0+cpu",)
    assert b300_build.install_layers[1].index_url == "https://download.pytorch.org/whl/cpu"
    assert b300_build.install_layers[1].extra_options == "--no-deps"

def test_modal_cloud_installs_and_validates_torch_layers_in_order(
    modal_cloud_module: Any,
) -> None:
    """The CPU TorchAudio layer must not replace the CUDA-enabled Torch package."""

    class RecordingImage:
        """Record Modal image package and command layers."""

        def __init__(self) -> None:
            """Initialize an empty call record."""
            self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

        def pip_install(self, *packages: str, **options: Any) -> "RecordingImage":
            """Record one package installation layer."""
            self.calls.append(("pip_install", packages, options))
            return self

        def run_commands(self, *commands: str) -> "RecordingImage":
            """Record one image validation command."""
            self.calls.append(("run_commands", commands, {}))
            return self

    build = modal_cloud_module._select_remote_torch_build("B300")
    image = RecordingImage()

    result = modal_cloud_module._install_remote_torch_build(image, build)

    assert result is image
    assert image.calls[:2] == [
        (
            "pip_install",
            ("torch==2.13.0", "torchvision==0.28.0"),
            {
                "index_url": "https://download.pytorch.org/whl/cu130",
                "extra_options": "",
            },
        ),
        (
            "pip_install",
            ("torchaudio==2.11.0+cpu",),
            {
                "index_url": "https://download.pytorch.org/whl/cpu",
                "extra_options": "--no-deps",
            },
        ),
    ]
    assert image.calls[2][0] == "run_commands"
    assert "import torch, torchaudio, torchvision" in image.calls[2][1][0]

def test_modal_cloud_installs_and_validates_vllm_for_non_b300_gpu(
    modal_cloud_module: Any,
) -> None:
    """A lower-cost GPU image should install and import the pinned vLLM wheel."""

    class RecordingImage:
        """Record Modal image package and command layers."""

        def __init__(self) -> None:
            """Initialize an empty call record."""
            self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

        def pip_install(self, *packages: str, **options: Any) -> "RecordingImage":
            """Record one package installation layer."""
            self.calls.append(("pip_install", packages, options))
            return self

        def run_commands(self, *commands: str) -> "RecordingImage":
            """Record one image validation command."""
            self.calls.append(("run_commands", commands, {}))
            return self

    image = RecordingImage()

    result = modal_cloud_module._install_remote_accelerator_packages(
        image,
        "RTX-PRO-6000",
    )

    assert result is image
    assert image.calls[0] == (
        "pip_install",
        (modal_cloud_module._remote_accelerator_packages("RTX-PRO-6000")[0],),
        {},
    )
    assert image.calls[1][0] == "run_commands"
    assert "import cv2, numpy, torch, vllm" in image.calls[1][1][0]

def test_modal_cloud_missing_prompt_node_class_raises_clear_error(
    modal_cloud_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Missing custom-node classes should fail before ComfyUI cache setup raises KeyError."""
    async def fake_init_external_custom_nodes() -> None:
        """Leave the missing custom node unregistered."""

    fake_nodes_module = types.SimpleNamespace(
        NODE_CLASS_MAPPINGS={"KnownNode": object},
        init_external_custom_nodes=fake_init_external_custom_nodes,
    )
    monkeypatch.setitem(sys.modules, "nodes", fake_nodes_module)
    custom_nodes_root = tmp_path / "custom_nodes"
    package_dir = custom_nodes_root / "Skoogeer-Noise" / "src"
    package_dir.mkdir(parents=True)
    (package_dir / "qwen_noise_nodes.py").write_text(
        "class KSamplerLoraSigmaInverse:\n    pass\n",
        encoding="utf-8",
    )

    with pytest.raises(modal_cloud_module.RemoteSubgraphExecutionError) as exc_info:
        modal_cloud_module._ensure_prompt_node_classes_registered(
            component_id="component-1",
            prompt={
                "1": {"class_type": "KnownNode", "inputs": {}},
                "2": {"class_type": "KSamplerLoraSigmaInverse", "inputs": {}},
            },
            custom_nodes_root=custom_nodes_root,
        )

    message = str(exc_info.value)
    assert "KSamplerLoraSigmaInverse" in message
    assert "custom-node sync is enabled" in message
    assert "Skoogeer-Noise/src/qwen_noise_nodes.py" in message
    assert "package=Skoogeer-Noise" in message

def test_modal_cloud_existing_app_guard_uses_non_creating_sdk_lookup(
    modal_cloud_module: Any,
) -> None:
    """Deploy-time app construction should fail if the configured Modal app already exists."""

    class FakeNotFoundError(Exception):
        """Stand-in for Modal app lookup misses."""

    lookup_calls: list[tuple[str, bool]] = []

    class FakeApp:
        """Minimal Modal App namespace with non-creating lookup support."""

        @staticmethod
        def lookup(app_name: str, create_if_missing: bool = True) -> object:
            """Record lookup arguments and return an existing app."""
            lookup_calls.append((app_name, create_if_missing))
            return object()

    fake_modal = types.SimpleNamespace(
        App=FakeApp,
        exception=types.SimpleNamespace(NotFoundError=FakeNotFoundError),
        is_local=lambda: True,
    )
    settings = types.SimpleNamespace(app_name="comfy-modal-sync")

    with pytest.raises(modal_cloud_module.ExistingModalAppError) as exc_info:
        modal_cloud_module._guard_against_existing_modal_app(settings, fake_modal)

    assert lookup_calls == [(DEFAULT_TEST_DEPLOYMENT_APP_NAME, False)]
    assert "Delete the existing app" in str(exc_info.value)
    assert "COMFY_MODAL_GPU" in str(exc_info.value)

def test_modal_cloud_existing_app_guard_allows_missing_sdk_lookup(
    modal_cloud_module: Any,
) -> None:
    """A missing Modal app should not block first-run app construction."""

    class FakeNotFoundError(Exception):
        """Stand-in for Modal app lookup misses."""

    class FakeApp:
        """Minimal Modal App namespace that reports a missing app."""

        @staticmethod
        def lookup(app_name: str, create_if_missing: bool = True) -> object:
            """Raise the SDK's not-found error without creating the app."""
            del app_name, create_if_missing
            raise FakeNotFoundError("app not found")

    fake_modal = types.SimpleNamespace(
        App=FakeApp,
        exception=types.SimpleNamespace(NotFoundError=FakeNotFoundError),
        is_local=lambda: True,
    )

    modal_cloud_module._guard_against_existing_modal_app(
        types.SimpleNamespace(app_name="comfy-modal-sync"),
        fake_modal,
    )

def test_modal_cloud_existing_app_guard_falls_back_to_cli_json(
    modal_cloud_module: Any,
    cloud_app_guard_module: Any,
    monkeypatch: Any,
) -> None:
    """SDKs without a non-creating lookup should fall back to `modal app list --json`."""

    class FakeApp:
        """Modal App namespace whose lookup signature cannot safely check existence."""

        @staticmethod
        def lookup(app_name: str) -> object:
            """This must not be called because it may create the app."""
            raise AssertionError(f"unsafe lookup called for {app_name}")

    completed = subprocess.CompletedProcess(
        args=["modal", "app", "list", "--json"],
        returncode=0,
        stdout=json.dumps([{"name": "other"}, {"name": DEFAULT_TEST_DEPLOYMENT_APP_NAME}]),
        stderr="",
    )
    observed_commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        """Return a deterministic Modal CLI app list response."""
        del kwargs
        observed_commands.append(command)
        return completed

    monkeypatch.setattr(
        cloud_app_guard_module.shutil, "which", lambda name: f"/usr/bin/{name}"
    )
    monkeypatch.setattr(cloud_app_guard_module.subprocess, "run", fake_run)
    fake_modal = types.SimpleNamespace(App=FakeApp, exception=types.SimpleNamespace(), is_local=lambda: True)

    with pytest.raises(modal_cloud_module.ExistingModalAppError):
        modal_cloud_module._guard_against_existing_modal_app(
            types.SimpleNamespace(app_name="comfy-modal-sync"),
            fake_modal,
        )

    assert observed_commands == [["/usr/bin/modal", "app", "list", "--json"]]

def test_modal_cloud_llm_compile_profiles_are_limited_to_executable_subgraph(
    modal_cloud_module: Any,
) -> None:
    """Nested metadata and disconnected LLM nodes must not trigger cache commits."""
    payload = {
        "payload_kind": "subgraph",
        "execute_node_ids": ["3"],
        "subgraph_prompt": {
            "1": {
                "class_type": "ModalLLM",
                "inputs": {"model_profile": "executed-profile"},
            },
            "2": {
                "class_type": "ModalLLM",
                "inputs": {"model_profile": "disconnected-profile"},
            },
            "3": {
                "class_type": "PreviewText",
                "inputs": {"text": ["1", 0]},
            },
        },
        "boundary_inputs": [
            {
                "source_signature": {
                    "class_type": "ModalLLM",
                    "inputs": {"model_profile": "metadata-profile"},
                }
            }
        ],
    }

    assert modal_cloud_module._llm_profiles_in_payload(payload) == (
        "executed-profile",
    )
    assert (
        modal_cloud_module._llm_profiles_in_payload(
            {**payload, "payload_kind": "canary"}
        )
        == ()
    )

def test_load_modal_cloud_module_reloads_stale_partial_module(
    remote_modal_app_module: Any,
    modal_deployment_module: Any,
    monkeypatch: Any,
) -> None:
    """Stale partially imported cloud modules should be discarded and reloaded."""
    original_module = sys.modules.get(modal_deployment_module._MODAL_CLOUD_MODULE_NAME)
    stale_module = types.SimpleNamespace(app=None)
    sys.modules[modal_deployment_module._MODAL_CLOUD_MODULE_NAME] = stale_module

    loaded_module = types.SimpleNamespace(app="fresh-app")

    class FakeLoader:
        """Populate the fresh replacement module during exec."""

        def create_module(self, spec: Any) -> None:
            """Use the default module creation path."""
            del spec
            return None

        def exec_module(self, module: Any) -> None:
            """Install the expected deployable app onto the reloaded module."""
            module.app = loaded_module.app

    monkeypatch.setattr(
        modal_deployment_module.importlib.util,
        "spec_from_file_location",
        lambda *args, **kwargs: importlib.util.spec_from_loader(
            modal_deployment_module._MODAL_CLOUD_MODULE_NAME,
            FakeLoader(),
        ),
    )
    try:
        reloaded_module = modal_deployment_module._load_modal_cloud_module()
    finally:
        sys.modules.pop(modal_deployment_module._MODAL_CLOUD_MODULE_NAME, None)
        if original_module is not None:
            sys.modules[modal_deployment_module._MODAL_CLOUD_MODULE_NAME] = original_module

    assert reloaded_module is not stale_module
    assert getattr(reloaded_module, "app", None) == "fresh-app"

def test_load_modal_cloud_module_reloads_for_workflow_gpu_change(
    remote_modal_app_module: Any,
    modal_deployment_module: Any,
    monkeypatch: Any,
) -> None:
    """Cloud app construction should receive the workflow-selected GPU settings."""
    module_name = modal_deployment_module._MODAL_CLOUD_MODULE_NAME
    original_module = sys.modules.get(module_name)
    sys.modules[module_name] = types.SimpleNamespace(
        app="a100-app",
        __comfy_modal_gpu__="A100",
    )
    observed_gpu_values: list[str] = []

    class FakeLoader:
        """Populate a replacement cloud module from its injected settings."""

        def create_module(self, spec: Any) -> None:
            """Use the default module creation path."""
            del spec
            return None

        def exec_module(self, module: Any) -> None:
            """Capture the settings override and expose a deployable app."""
            settings_override = module.__comfy_modal_settings_override__
            observed_gpu_values.append(settings_override.modal_gpu)
            module.app = "b300-app"

    monkeypatch.setattr(
        modal_deployment_module.importlib.util,
        "spec_from_file_location",
        lambda *args, **kwargs: importlib.util.spec_from_loader(module_name, FakeLoader()),
    )
    settings = remote_modal_app_module.settings_for_modal_gpu(
        remote_modal_app_module.get_settings(),
        "B300",
    )
    try:
        with modal_deployment_module._modal_cloud_settings_override(settings):
            reloaded_module = modal_deployment_module._load_modal_cloud_module()
    finally:
        sys.modules.pop(module_name, None)
        if original_module is not None:
            sys.modules[module_name] = original_module

    assert observed_gpu_values == ["B300"]
    assert reloaded_module.app == "b300-app"
    assert reloaded_module.__comfy_modal_gpu__ == "B300"
    assert reloaded_module.__comfy_modal_app_name__ == "comfy-modal-sync-gpu-b300"

def test_load_modal_cloud_module_reloads_for_secret_name_change(
    remote_modal_app_module: Any,
    modal_deployment_module: Any,
    monkeypatch: Any,
) -> None:
    """Changing only the secret name should rebuild the cloud module with new settings."""
    module_name = modal_deployment_module._MODAL_CLOUD_MODULE_NAME
    original_module = sys.modules.get(module_name)
    base_settings = remote_modal_app_module.get_settings()
    sys.modules[module_name] = types.SimpleNamespace(
        app="old-secret-app",
        __comfy_modal_gpu__=base_settings.modal_gpu,
        __comfy_modal_app_name__=remote_modal_app_module.modal_deployment_app_name(
            base_settings
        ),
        __comfy_modal_secret_name__="comfy",
    )
    observed_secret_names: list[str] = []

    class FakeLoader:
        """Populate a replacement cloud module from its injected settings."""

        def create_module(self, spec: Any) -> None:
            """Use the default module creation path."""
            del spec
            return None

        def exec_module(self, module: Any) -> None:
            """Capture the secret setting and expose a deployable app."""
            settings_override = module.__comfy_modal_settings_override__
            observed_secret_names.append(settings_override.modal_secret_name)
            module.app = "new-secret-app"

    monkeypatch.setattr(
        modal_deployment_module.importlib.util,
        "spec_from_file_location",
        lambda *args, **kwargs: importlib.util.spec_from_loader(module_name, FakeLoader()),
    )
    new_settings = replace(base_settings, modal_secret_name="workflow-credentials")
    try:
        with modal_deployment_module._modal_cloud_settings_override(new_settings):
            reloaded_module = modal_deployment_module._load_modal_cloud_module()
    finally:
        sys.modules.pop(module_name, None)
        if original_module is not None:
            sys.modules[module_name] = original_module

    assert observed_secret_names == ["workflow-credentials"]
    assert reloaded_module.app == "new-secret-app"
    assert reloaded_module.__comfy_modal_secret_name__ == "workflow-credentials"

def test_remote_modal_installs_cloud_exception_compatibility_module(
    remote_modal_app_module: Any,
    modal_deployment_module: Any,
) -> None:
    """Deployed remote exceptions should have importable local definitions."""
    module_name = modal_deployment_module._MODAL_CLOUD_MODULE_NAME
    original_module = sys.modules.pop(module_name, None)
    serialization_module = types.ModuleType(module_name)
    serialization_module.RemoteSubgraphExecutionError = type(
        "RemoteSubgraphExecutionError",
        (RuntimeError,),
        {"__module__": module_name},
    )
    sys.modules[module_name] = serialization_module
    serialized_error = pickle.dumps(
        serialization_module.RemoteSubgraphExecutionError("remote failure")
    )
    sys.modules.pop(module_name)
    try:
        modal_deployment_module._install_modal_cloud_exception_compatibility_module()

        compatibility_module = sys.modules[module_name]
        assert (
            compatibility_module.RemoteSubgraphExecutionError
            is remote_modal_app_module.RemoteSubgraphExecutionError
        )
        assert issubclass(compatibility_module.RemoteInvocationAbandonedError, RuntimeError)
        restored_error = pickle.loads(serialized_error)
        assert isinstance(restored_error, remote_modal_app_module.RemoteSubgraphExecutionError)
        assert str(restored_error) == "remote failure"
    finally:
        sys.modules.pop(module_name, None)
        if original_module is not None:
            sys.modules[module_name] = original_module

def test_load_modal_cloud_module_clears_failed_import_from_sys_modules(
    remote_modal_app_module: Any,
    modal_deployment_module: Any,
    monkeypatch: Any,
) -> None:
    """Failed cloud-module imports should not leave a poisoned cache entry behind."""
    original_module = sys.modules.pop(
        modal_deployment_module._MODAL_CLOUD_MODULE_NAME,
        None,
    )

    class FakeLoader:
        """Raise during module execution to simulate a partial import failure."""

        def create_module(self, spec: Any) -> None:
            """Use the default module creation path."""
            del spec
            return None

        def exec_module(self, module: Any) -> None:
            """Fail while the module is being initialized."""
            module.app = None
            raise RuntimeError("boom")

    monkeypatch.setattr(
        modal_deployment_module.importlib.util,
        "spec_from_file_location",
        lambda *args, **kwargs: importlib.util.spec_from_loader(
            modal_deployment_module._MODAL_CLOUD_MODULE_NAME,
            FakeLoader(),
        ),
    )
    try:
        with pytest.raises(RuntimeError, match="boom"):
            modal_deployment_module._load_modal_cloud_module()
        assert modal_deployment_module._MODAL_CLOUD_MODULE_NAME not in sys.modules
    finally:
        sys.modules.pop(modal_deployment_module._MODAL_CLOUD_MODULE_NAME, None)
        if original_module is not None:
            sys.modules[modal_deployment_module._MODAL_CLOUD_MODULE_NAME] = original_module

def test_modal_cloud_tracing_prompt_server_emits_executed_outputs(
    modal_cloud_module: Any,
) -> None:
    """The cloud tracing prompt server should stream node UI outputs as executed events."""
    observed_updates: list[dict[str, Any]] = []
    server = modal_cloud_module._TracingPromptServer(
        "component-1",
        {"7": {"class_type": "PreviewImage", "inputs": {}}},
        status_callback=observed_updates.append,
    )

    server.send_sync(
        "executed",
        {
            "node": "7",
            "display_node": "7",
            "output": {"images": [{"filename": "preview.png"}]},
        },
        None,
    )

    assert observed_updates == [
        {
            "event_type": "executed",
            "node_id": "7",
            "display_node_id": "7",
            "output": {"images": [{"filename": "preview.png"}]},
        }
    ]

def test_modal_cloud_tracing_prompt_server_emits_boundary_image_outputs(
    modal_cloud_module: Any,
) -> None:
    """The cloud tracing prompt server should stream configured boundary IMAGE outputs once cached."""
    torch = pytest.importorskip("torch")
    image_tensor = torch.zeros((1, 4, 4, 3), dtype=torch.float32)
    observed_updates: list[dict[str, Any]] = []
    server = modal_cloud_module._TracingPromptServer(
        "component-1",
        {
            "7": {"class_type": "VAEDecode", "inputs": {}},
            "8": {"class_type": "OtherNode", "inputs": {}},
        },
        status_callback=observed_updates.append,
    )
    cache_entries = {
        "7": types.SimpleNamespace(outputs=[image_tensor]),
    }
    server.configure_boundary_output_stream(
        boundary_outputs=[
            {
                "node_id": "7",
                "output_index": 0,
                "io_type": "IMAGE",
                "is_list": False,
                "preview_target_node_ids": ["9"],
            }
        ],
        lookup_cache_entry=lambda node_id: cache_entries.get(node_id),
    )

    server.send_sync("executing", {"node": "7"}, None)
    server.send_sync("executing", {"node": "8"}, None)

    assert observed_updates[0]["phase"] == "executing"
    assert observed_updates[1] == {
        "event_type": "boundary_output",
        "node_id": "7",
        "output_index": 0,
        "io_type": "IMAGE",
        "is_list": False,
        "preview_target_node_ids": ["9"],
        "value": image_tensor,
    }
    assert observed_updates[2]["phase"] == "executing"

@pytest.mark.parametrize(
    ("module_fixture_name",),
    [
        ("local_execution_module",),
        ("modal_cloud_module",),
    ],
)
def test_trim_subgraph_payload_to_required_nodes_drops_stale_execute_targets(
    request: Any,
    module_fixture_name: str,
) -> None:
    """Trimmed subgraph payloads should ignore execute targets that are absent from the current prompt."""
    target_module = request.getfixturevalue(module_fixture_name)
    payload = {
        "component_id": "1::static",
        "component_node_ids": ["1", "3"],
        "subgraph_prompt": {
            "1": {"class_type": "LoadDiffusionModel", "inputs": {}},
            "3": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "steps": 20}},
        },
        "boundary_inputs": [],
        "boundary_outputs": [
            {"node_id": "3", "output_index": 0, "io_type": "LATENT", "is_list": False},
        ],
        "execute_node_ids": ["3", "5"],
        "mapped_execute_node_ids": [],
        "static_execute_node_ids": ["3", "5"],
    }

    trimmed_payload = target_module._trim_subgraph_payload_to_required_nodes(payload)

    assert trimmed_payload["component_node_ids"] == ["1", "3"]
    assert list(trimmed_payload["subgraph_prompt"].keys()) == ["1", "3"]
    assert trimmed_payload["execute_node_ids"] == ["3"]
    assert trimmed_payload["mapped_execute_node_ids"] == []
    assert trimmed_payload["static_execute_node_ids"] == ["3"]

def test_modal_cloud_uses_current_comfy_explicit_ram_pressure_thresholds(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """Two current cache-ram values should flow into active and inactive thresholds."""
    fake_args = types.SimpleNamespace(
        cache_classic=False,
        cache_lru=0,
        cache_ram=[3.0, 20.0],
        cache_none=False,
    )
    fake_cli_args_module = types.SimpleNamespace(args=fake_args)
    fake_model_management_module = types.SimpleNamespace(total_ram=64 * 1024)
    fake_execution_module = types.SimpleNamespace(
        CacheType=types.SimpleNamespace(
            CLASSIC="classic",
            LRU="lru",
            RAM_PRESSURE="ram-pressure",
            NONE="none",
        )
    )
    monkeypatch.setitem(sys.modules, "comfy.cli_args", fake_cli_args_module)
    monkeypatch.setitem(
        sys.modules,
        "comfy.model_management",
        fake_model_management_module,
    )

    cache_type, cache_args = modal_cloud_module._prompt_executor_cache_config(fake_execution_module)

    assert cache_type == "ram-pressure"
    assert cache_args == {"lru": 0, "ram": 3.0, "ram_inactive": 20.0}

def test_modal_cloud_awaits_async_prompt_executor_api(
    modal_cloud_module: Any,
) -> None:
    """An asynchronous PromptExecutor compatibility API should finish before cache reads."""
    observed_calls: list[tuple[dict[str, Any], str, dict[str, Any], list[str]]] = []

    class FakeExecutor:
        """Minimal executor exposing a coroutine-based execute method."""

        async def execute(
            self,
            *,
            prompt: dict[str, Any],
            prompt_id: str,
            extra_data: dict[str, Any],
            execute_outputs: list[str],
        ) -> None:
            """Record execution after yielding once."""
            await asyncio.sleep(0)
            observed_calls.append((prompt, prompt_id, extra_data, execute_outputs))

    modal_cloud_module._execute_prompt_executor_compat(
        FakeExecutor(),
        prompt={"1": {"class_type": "Example", "inputs": {}}},
        prompt_id="prompt-1",
        extra_data={"client_id": "client-1"},
        execute_outputs=["1"],
    )

    assert observed_calls == [
        (
            {"1": {"class_type": "Example", "inputs": {}}},
            "prompt-1",
            {"client_id": "client-1"},
            ["1"],
        )
    ]

def test_modal_cloud_class_options_do_not_use_deprecated_concurrency_flag(
    modal_cloud_module: Any,
) -> None:
    """The deployed Modal class options should avoid deprecated concurrency flags."""
    fake_settings = types.SimpleNamespace(
        modal_gpu="A100",
        remote_storage_root="/vol/data",
        scaledown_window_seconds=60,
        min_containers=0,
        max_containers=4,
        buffer_containers=1,
        execution_timeout_seconds=1800,
        startup_timeout_seconds=600,
        enable_memory_snapshot=True,
        enable_gpu_memory_snapshot=False,
    )

    options = modal_cloud_module._remote_engine_cls_options(
        fake_settings,
        vol=object(),
        image=object(),
    )

    assert "allow_concurrent_inputs" not in options
    assert options["max_containers"] == 4
    assert options["buffer_containers"] == 1
    assert options["timeout"] == 1800
    assert options["startup_timeout"] == 600
    module_source = Path(modal_cloud_module.__file__).read_text(encoding="utf-8")
    assert "@modal.concurrent(max_inputs=1)" in module_source

def test_modal_cloud_registered_execution_preserves_pre_start_interrupt_flags(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """Remote execution registration should not delete a valid cancel flag on entry."""

    class FakeInterruptFlags:
        """Simple Modal Dict double that records cleared keys."""

        def __init__(self) -> None:
            """Initialize captured pop calls."""
            self.pop_calls: list[tuple[str, Any]] = []

        def pop(self, key: str, default: Any = None) -> Any:
            """Record one cleared interrupt flag."""
            self.pop_calls.append((key, default))
            return None

    interrupt_flags = FakeInterruptFlags()
    monkeypatch.setattr(_cloud_execution_control_owner(), "modal", object())
    monkeypatch.setattr(
        _cloud_execution_control_owner(),
        "interrupt_flag_store",
        lambda: interrupt_flags,
    )

    with modal_cloud_module._registered_remote_execution(
        {"prompt_id": "prompt-1", "component_id": "component-2"}
    ) as execution_control:
        assert execution_control.interrupt_flag_key == "prompt-1:component-2"
        assert interrupt_flags.pop_calls == []

    assert interrupt_flags.pop_calls == [
        ("prompt-1:component-2", None),
    ]

def test_modal_cloud_interrupt_monitor_consumes_shared_cancel_flag(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """The remote interrupt monitor should trip when the shared Modal Dict flag appears."""

    class FakeInterruptFlags:
        """Simple Modal Dict double that exposes one shared cancel flag."""

        def __init__(self) -> None:
            """Initialize the backing key set."""
            self.keys = {"prompt-1:component-2"}
            self.contains_calls = 0
            self.pop_calls: list[tuple[str, Any]] = []

        def contains(self, key: str) -> bool:
            """Report whether the shared interrupt flag exists."""
            self.contains_calls += 1
            return key in self.keys

        def pop(self, key: str, default: Any = None) -> Any:
            """Remove the shared interrupt flag once consumed."""
            self.pop_calls.append((key, default))
            self.keys.discard(key)
            return None

    interrupt_calls: list[str] = []
    cancellation_event = threading.Event()
    monkeypatch.setitem(
        sys.modules,
        "nodes",
        types.SimpleNamespace(interrupt_processing=lambda: interrupt_calls.append("interrupt")),
    )

    with modal_cloud_module._temporary_remote_interrupt_monitor(
        "component-2",
        cancellation_event,
        interrupt_store=FakeInterruptFlags(),
        interrupt_flag_key="prompt-1:component-2",
    ):
        deadline = time.time() + 1.0
        while not interrupt_calls and time.time() < deadline:
            time.sleep(0.01)

    assert interrupt_calls == ["interrupt"]
    assert cancellation_event.is_set()

def test_modal_cloud_interrupt_monitor_ignores_modal_client_shutdown(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """The remote interrupt monitor should exit quietly if Modal tears down the client first."""

    class FakeClientClosedError(RuntimeError):
        """Stand-in for `modal.exception.ClientClosed`."""

    class FakeInterruptFlags:
        """Simple Modal Dict double that fails once the client is already shutting down."""

        def __init__(self) -> None:
            """Initialize the poll counter."""
            self.contains_calls = 0
            self.pop_calls: list[tuple[str, Any]] = []

        def contains(self, key: str) -> bool:
            """Raise the same client-shutdown error Modal emits during teardown."""
            del key
            self.contains_calls += 1
            raise FakeClientClosedError("client closed")

        def pop(self, key: str, default: Any = None) -> Any:
            """Record unexpected cleanup attempts."""
            self.pop_calls.append((key, default))
            return None

    interrupt_calls: list[str] = []
    thread_exceptions: list[BaseException] = []
    cancellation_event = threading.Event()
    fake_modal_module = types.ModuleType("modal")
    fake_modal_exception_module = types.ModuleType("modal.exception")
    fake_modal_exception_module.ClientClosed = FakeClientClosedError
    fake_modal_module.exception = fake_modal_exception_module
    monkeypatch.setitem(sys.modules, "modal", fake_modal_module)
    monkeypatch.setitem(sys.modules, "modal.exception", fake_modal_exception_module)
    monkeypatch.setitem(
        sys.modules,
        "nodes",
        types.SimpleNamespace(interrupt_processing=lambda: interrupt_calls.append("interrupt")),
    )
    monkeypatch.setattr(
        threading,
        "excepthook",
        lambda args: thread_exceptions.append(args.exc_value),
    )
    interrupt_store = FakeInterruptFlags()

    with modal_cloud_module._temporary_remote_interrupt_monitor(
        "component-2",
        cancellation_event,
        interrupt_store=interrupt_store,
        interrupt_flag_key="prompt-1:component-2",
    ):
        deadline = time.time() + 1.0
        while interrupt_store.contains_calls == 0 and time.time() < deadline:
            time.sleep(0.01)

    assert interrupt_store.contains_calls >= 1
    assert interrupt_store.pop_calls == []
    assert interrupt_calls == []
    assert not cancellation_event.is_set()
    assert thread_exceptions == []

def test_modal_cloud_serializes_only_small_transport_safe_node_outputs(
    modal_cloud_module: Any,
) -> None:
    """Persisted node-cache records should keep small tensor outputs and skip oversized ones."""
    import torch

    execution = modal_cloud_module._load_execution_module()
    small_entry = execution.CacheEntry(ui=None, outputs=[[torch.zeros((8,), dtype=torch.float32)]])
    large_entry = execution.CacheEntry(ui=None, outputs=[[torch.zeros((512,), dtype=torch.float32)]])

    small_record = modal_cloud_module._serialize_node_output_cache_entry(
        small_entry,
        max_bytes=1024,
    )
    large_record = modal_cloud_module._serialize_node_output_cache_entry(
        large_entry,
        max_bytes=1024,
    )

    assert small_record is not None
    assert large_record is None

def test_modal_cloud_materializes_synced_asset_paths(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """Remote asset references should resolve to absolute files under the storage root."""
    monkeypatch.setenv("COMFY_MODAL_REMOTE_STORAGE_ROOT", "/storage")
    modal_cloud_module.get_settings.cache_clear()
    try:
        assert modal_cloud_module._materialize_remote_asset_path("/assets/model.safetensors") == (
            "/storage/assets/model.safetensors"
        )
        assert modal_cloud_module._rewrite_modal_asset_references(
            {"clip_name": "/assets/model.safetensors", "nested": ["/assets/other.pt", 3]}
        ) == {
            "clip_name": "/storage/assets/model.safetensors",
            "nested": ["/storage/assets/other.pt", 3],
        }
    finally:
        modal_cloud_module.get_settings.cache_clear()

def test_modal_cloud_summarizes_suspicious_wrapped_prompt_inputs(
    modal_cloud_module: Any,
) -> None:
    """Remote failure diagnostics should flag remaining singleton-list prompt wrappers."""
    prompt = {
        "12": {
            "class_type": "ExampleNode",
            "inputs": {
                "steps": [20],
                "latent": ["7", [0]],
                "ok_link": ["8", 0],
            },
        }
    }

    findings = modal_cloud_module._summarize_suspicious_prompt_inputs(prompt)

    assert findings == [
        "12.steps=[20]",
        "12.latent=['7', [0]]",
    ]

def test_modal_cloud_accepts_absolute_asset_paths_in_folder_lookup(
    modal_cloud_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Patched folder lookups should return already-materialized absolute asset paths."""
    remote_storage_root = tmp_path / "storage"
    asset_path = remote_storage_root / "assets" / "clip.safetensors"
    asset_path.parent.mkdir(parents=True)
    asset_path.write_bytes(b"clip")

    fake_folder_paths_module = types.SimpleNamespace(
        get_full_path=lambda folder_name, filename: None,
        get_full_path_or_raise=lambda folder_name, filename: (_ for _ in ()).throw(
            FileNotFoundError(filename)
        ),
    )
    monkeypatch.setitem(sys.modules, "folder_paths", fake_folder_paths_module)
    monkeypatch.setenv("COMFY_MODAL_REMOTE_STORAGE_ROOT", str(remote_storage_root))
    modal_cloud_module.get_settings.cache_clear()
    try:
        with modal_cloud_module._patched_folder_paths_absolute_lookup():
            resolved = fake_folder_paths_module.get_full_path(
                "text_encoders",
                "/assets/clip.safetensors",
            )
            assert resolved == str(asset_path)
            assert (
                fake_folder_paths_module.get_full_path_or_raise(
                    "text_encoders",
                    "/assets/clip.safetensors",
                )
                == str(asset_path)
            )
    finally:
        modal_cloud_module.get_settings.cache_clear()

def test_modal_cloud_preserves_absolute_lookup_until_overlapping_contexts_exit(
    modal_cloud_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """One loader finishing must not remove the lookup patch from another loader."""
    asset_path = tmp_path / "unet.safetensors"
    asset_path.write_bytes(b"unet")

    def original_get_full_path(folder_name: str, filename: str) -> None:
        """Represent ComfyUI's original lookup missing the absolute asset."""
        del folder_name, filename
        return None

    def original_get_full_path_or_raise(folder_name: str, filename: str) -> str:
        """Represent ComfyUI's original raising lookup."""
        del folder_name
        raise FileNotFoundError(filename)

    fake_folder_paths_module = types.SimpleNamespace(
        get_full_path=original_get_full_path,
        get_full_path_or_raise=original_get_full_path_or_raise,
    )
    monkeypatch.setitem(sys.modules, "folder_paths", fake_folder_paths_module)
    first_entered = threading.Event()
    second_entered = threading.Event()
    release_first = threading.Event()
    first_exited = threading.Event()
    release_second = threading.Event()
    results: list[str | None] = []

    def first_loader() -> None:
        """Enter first and exit while the second lookup context remains active."""
        with modal_cloud_module._patched_folder_paths_absolute_lookup():
            first_entered.set()
            assert release_first.wait(timeout=2.0)
        first_exited.set()

    def second_loader() -> None:
        """Resolve an absolute model after the first overlapping loader exits."""
        assert first_entered.wait(timeout=2.0)
        with modal_cloud_module._patched_folder_paths_absolute_lookup():
            second_entered.set()
            assert first_exited.wait(timeout=2.0)
            results.append(
                fake_folder_paths_module.get_full_path("diffusion_models", str(asset_path))
            )
            assert release_second.wait(timeout=2.0)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(first_loader)
        second_future = executor.submit(second_loader)
        assert second_entered.wait(timeout=2.0)
        release_first.set()
        assert first_exited.wait(timeout=2.0)
        first_future.result(timeout=2.0)
        release_second.set()
        second_future.result(timeout=2.0)

    assert results == [str(asset_path)]
    assert fake_folder_paths_module.get_full_path is original_get_full_path
    assert fake_folder_paths_module.get_full_path_or_raise is original_get_full_path_or_raise

def test_modal_cloud_aliases_flux_rms_norm_weight_keys_for_model_detection(
    modal_cloud_module: Any,
) -> None:
    """Saved Flux models using RMSNorm `.weight` keys should gain `.scale` aliases."""
    state_dict = {
        "model.diffusion_model.double_blocks.0.img_attn.norm.key_norm.weight": object(),
        "model.diffusion_model.double_blocks.0.img_attn.norm.query_norm.weight": object(),
        "model.diffusion_model.double_blocks.0.txt_attn.norm.key_norm.weight": object(),
        "model.diffusion_model.double_blocks.0.txt_attn.norm.query_norm.weight": object(),
        "model.diffusion_model.img_in.weight": object(),
    }

    alias_count = modal_cloud_module._alias_flux_rms_norm_weight_keys(state_dict)

    assert alias_count == 4
    assert (
        state_dict["model.diffusion_model.double_blocks.0.img_attn.norm.key_norm.scale"]
        is state_dict["model.diffusion_model.double_blocks.0.img_attn.norm.key_norm.weight"]
    )
    assert (
        state_dict["model.diffusion_model.double_blocks.0.img_attn.norm.query_norm.scale"]
        is state_dict["model.diffusion_model.double_blocks.0.img_attn.norm.query_norm.weight"]
    )
    assert (
        state_dict["model.diffusion_model.double_blocks.0.txt_attn.norm.key_norm.scale"]
        is state_dict["model.diffusion_model.double_blocks.0.txt_attn.norm.key_norm.weight"]
    )
    assert (
        state_dict["model.diffusion_model.double_blocks.0.txt_attn.norm.query_norm.scale"]
        is state_dict["model.diffusion_model.double_blocks.0.txt_attn.norm.query_norm.weight"]
    )
    assert modal_cloud_module._alias_flux_rms_norm_weight_keys(state_dict) == 0

def test_modal_cloud_model_state_dict_wrapper_forwards_new_comfyui_arguments(
    modal_cloud_module: Any,
    monkeypatch: Any,
) -> None:
    """The compatibility wrapper should preserve evolving ComfyUI loader arguments."""
    calls: list[tuple[dict[str, Any], tuple[Any, ...], dict[str, Any]]] = []

    def original_loader(
        state_dict: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """Record the arguments forwarded by the compatibility wrapper."""
        calls.append((state_dict, args, kwargs))
        return "loaded"

    fake_comfy_module = types.ModuleType("comfy")
    fake_comfy_module.__path__ = []
    fake_sd_module = types.ModuleType("comfy.sd")
    fake_sd_module.load_diffusion_model_state_dict = original_loader
    fake_comfy_module.sd = fake_sd_module
    monkeypatch.setitem(sys.modules, "comfy", fake_comfy_module)
    monkeypatch.setitem(sys.modules, "comfy.sd", fake_sd_module)
    monkeypatch.setattr(
        _cloud_comfy_bootstrap_owner(),
        "_MODEL_STATE_DICT_COMPAT_WRAPPED",
        False,
    )

    modal_cloud_module._install_model_state_dict_compatibility_wrappers()
    state_dict = {
        "model.diffusion_model.double_blocks.0.img_attn.norm.key_norm.weight": object()
    }

    result = fake_sd_module.load_diffusion_model_state_dict(
        state_dict,
        {"dtype": "default"},
        metadata={"source": "test"},
        disable_dynamic=True,
        future_option="preserved",
    )

    assert result == "loaded"
    assert calls == [
        (
            state_dict,
            ({"dtype": "default"},),
            {
                "metadata": {"source": "test"},
                "disable_dynamic": True,
                "future_option": "preserved",
            },
        )
    ]
    assert (
        state_dict["model.diffusion_model.double_blocks.0.img_attn.norm.key_norm.scale"]
        is state_dict["model.diffusion_model.double_blocks.0.img_attn.norm.key_norm.weight"]
    )

def test_modal_cloud_force_imports_comfyui_utils_package(
    modal_cloud_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """ComfyUI's utils package should override a shadowing non-package module."""
    package_root = tmp_path / "comfyui"
    utils_dir = package_root / "utils"
    utils_dir.mkdir(parents=True)
    (utils_dir / "__init__.py").write_text("SENTINEL = 'comfy-utils'\n", encoding="utf-8")

    shadow_module = types.ModuleType("utils")
    shadow_module.__file__ = str(tmp_path / "utils.py")
    monkeypatch.setitem(sys.modules, "utils", shadow_module)

    modal_cloud_module._force_import_package_from_root("utils", package_root)

    imported_module = sys.modules["utils"]
    assert getattr(imported_module, "SENTINEL", None) == "comfy-utils"
    assert list(getattr(imported_module, "__path__", [])) == [str(utils_dir)]

def test_local_remote_app_executes_subgraph_payload(
    remote_modal_app_module: Any,
    serialization_module: Any,
) -> None:
    """The local fallback remote app should execute rewritten subgraph payloads."""
    payload = remote_modal_app_module.execute_subgraph_locally(
        payload={
            "payload_kind": "subgraph",
            "component_id": "component-1",
            "subgraph_prompt": {
                "remote_1": {
                    "class_type": "BoundarySource",
                    "inputs": {"value": 0},
                    "_meta": {},
                },
                "remote_2": {
                    "class_type": "BoundarySink",
                    "inputs": {"value": ["remote_1", 0]},
                    "_meta": {},
                },
            },
            "boundary_inputs": [
                {
                    "proxy_input_name": "remote_input_0",
                    "targets": [{"node_id": "remote_1", "input_name": "value"}],
                }
            ],
            "boundary_outputs": [
                {
                    "proxy_output_name": "remote_2_value",
                    "node_id": "remote_2",
                    "output_index": 0,
                    "io_type": "INT",
                    "is_list": False,
                }
            ],
            "execute_node_ids": ["remote_2"],
            "extra_data": {},
            "custom_nodes_bundle": None,
        },
        kwargs_payload='{"remote_input_0": 4}',
        node_mapping={
            "BoundarySource": _BoundarySourceNode,
            "BoundarySink": _BoundarySinkNode,
        },
    )
    outputs = serialization_module.deserialize_node_outputs(payload)
    assert outputs == (10,)

def test_local_remote_app_normalizes_wrapped_subgraph_link_indexes(
    remote_modal_app_module: Any,
    serialization_module: Any,
) -> None:
    """The local fallback runner should canonicalize singleton-list prompt link indexes."""
    payload = remote_modal_app_module.execute_subgraph_locally(
        payload={
            "payload_kind": "subgraph",
            "component_id": "component-1",
            "subgraph_prompt": {
                "remote_1": {
                    "class_type": "BoundarySource",
                    "inputs": {"value": 0},
                    "_meta": {},
                },
                "remote_2": {
                    "class_type": "BoundarySink",
                    "inputs": {"value": [[["remote_1", [0]]]]},
                    "_meta": {},
                },
            },
            "boundary_inputs": [
                {
                    "proxy_input_name": "remote_input_0",
                    "targets": [{"node_id": "remote_1", "input_name": "value"}],
                }
            ],
            "boundary_outputs": [
                {
                    "proxy_output_name": "remote_2_value",
                    "node_id": "remote_2",
                    "output_index": [[0]],
                    "io_type": "INT",
                    "is_list": False,
                }
            ],
            "execute_node_ids": ["remote_2"],
            "extra_data": {},
            "custom_nodes_bundle": None,
        },
        kwargs_payload='{"remote_input_0": 4}',
        node_mapping={
            "BoundarySource": _BoundarySourceNode,
            "BoundarySink": _BoundarySinkNode,
        },
    )
    outputs = serialization_module.deserialize_node_outputs(payload)
    assert outputs == (10,)

@pytest.mark.parametrize(
    ("module_fixture_name",),
    [
        ("local_execution_module",),
        ("modal_cloud_module",),
    ],
)
def test_format_prompt_executor_error_payload_includes_node_context(
    request: Any,
    module_fixture_name: str,
) -> None:
    """PromptExecutor failure formatting should surface the failing node and current inputs."""
    target_module = request.getfixturevalue(module_fixture_name)

    message = target_module._format_prompt_executor_error_payload(
        {
            "exception_message": "int() argument must be a string, a bytes-like object or a real number, not 'list'",
            "node_id": "12",
            "node_type": "KSampler",
            "current_inputs": [{"input_name": "steps", "value": [4, 5]}],
        }
    )

    assert "node_id='12'" in message
    assert "node_type='KSampler'" in message
    assert "current_inputs=" in message
