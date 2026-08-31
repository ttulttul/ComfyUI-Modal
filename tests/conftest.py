"""Pytest fixtures for loading the ComfyUI Modal-Sync extension package."""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "comfyui_modal_sync_under_test"
os.environ.setdefault("COMFY_MODAL_APP_NAME", "comfy-modal-sync")


def _comfyui_root() -> Path:
    """Return the ComfyUI checkout to expose during tests."""
    configured_root = os.getenv("COMFYUI_ROOT")
    if configured_root:
        return Path(configured_root).expanduser().resolve()
    preferred_root = Path.home() / "git" / "ComfyUI"
    if preferred_root.exists():
        return preferred_root
    return Path.home() / "git" / "Latest_ComfyUI"


def _ensure_import_paths() -> None:
    """Add the repository and local ComfyUI checkout to sys.path when present."""
    if "av" not in sys.modules:
        av_module = types.ModuleType("av")
        av_module.__spec__ = importlib.machinery.ModuleSpec("av", loader=None)
        av_module.__path__ = []
        av_module.open = lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("The av stub should not be used in tests.")
        )
        av_module.time_base = 1
        av_module.VideoStream = type("VideoStream", (), {})
        av_module.AVError = RuntimeError
        av_module.FFmpegError = RuntimeError
        av_module.logging = types.SimpleNamespace(
            ERROR="ERROR",
            set_level=lambda *args, **kwargs: None,
        )
        av_module.video = types.SimpleNamespace(
            frame=types.SimpleNamespace(
                VideoFrame=type("VideoFrame", (), {"pict_type": None})
            )
        )

        av_container_module = types.ModuleType("av.container")
        av_container_module.InputContainer = type("InputContainer", (), {})

        av_subtitles_module = types.ModuleType("av.subtitles")
        av_subtitles_stream_module = types.ModuleType("av.subtitles.stream")
        av_subtitles_stream_module.SubtitleStream = type("SubtitleStream", (), {})
        av_video_module = types.ModuleType("av.video")
        av_video_reformatter_module = types.ModuleType("av.video.reformatter")
        av_video_reformatter_module.ColorPrimaries = type(
            "ColorPrimaries", (), {"BT2020": 9, "BT709": 1}
        )
        av_video_reformatter_module.ColorRange = type(
            "ColorRange", (), {"JPEG": 2, "MPEG": 1}
        )
        av_video_reformatter_module.ColorTrc = type(
            "ColorTrc",
            (),
            {
                "ARIB_STD_B67": 18,
                "BT709": 1,
                "IEC61966_2_1": 13,
                "SMPTE2084": 16,
            },
        )

        sys.modules["av"] = av_module
        sys.modules["av.container"] = av_container_module
        sys.modules["av.subtitles"] = av_subtitles_module
        sys.modules["av.subtitles.stream"] = av_subtitles_stream_module
        sys.modules["av.video"] = av_video_module
        sys.modules["av.video.reformatter"] = av_video_reformatter_module
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    comfyui_root = _comfyui_root()
    if comfyui_root.exists() and str(comfyui_root) not in sys.path:
        sys.path.insert(0, str(comfyui_root))


def _load_extension_package() -> object:
    """Load the extension root as an importable package for test modules."""
    _ensure_import_paths()
    if PACKAGE_NAME in sys.modules:
        return sys.modules[PACKAGE_NAME]

    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME,
        REPO_ROOT / "__init__.py",
        submodule_search_locations=[str(REPO_ROOT)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to create an import spec for the extension package.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE_NAME] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def extension_package() -> object:
    """Return the loaded extension package."""
    return _load_extension_package()


@pytest.fixture(scope="session")
def modal_sdk_module(extension_package: object) -> object:
    """Return the Modal SDK startup bootstrap module."""
    return importlib.import_module(f"{PACKAGE_NAME}.modal_sdk")


@pytest.fixture(scope="session")
def instance_identity_module(extension_package: object) -> object:
    """Return the persistent ComfyUI instance identity module."""
    return importlib.import_module(f"{PACKAGE_NAME}.instance_identity")


@pytest.fixture(scope="session")
def execution_environments_module(extension_package: object) -> object:
    """Return the provider-neutral execution environment module."""
    return importlib.import_module(f"{PACKAGE_NAME}.execution_environments")


@pytest.fixture(scope="session")
def remote_configurations_module(extension_package: object) -> object:
    """Return workflow-declared remote configuration models."""
    return importlib.import_module(f"{PACKAGE_NAME}.remote_configurations")


@pytest.fixture(scope="session")
def remote_configuration_nodes_module(extension_package: object) -> object:
    """Return the workflow remote configuration v3 nodes and compiler."""
    return importlib.import_module(f"{PACKAGE_NAME}.remote_configuration_nodes")


@pytest.fixture(scope="session")
def subrosa_credentials_module(extension_package: object) -> object:
    """Return the Subrosa OS-keyring credential module."""
    return importlib.import_module(f"{PACKAGE_NAME}.subrosa_credentials")


@pytest.fixture(scope="session")
def subrosa_executor_module(extension_package: object) -> object:
    """Return the Subrosa relay executor module."""
    return importlib.import_module(f"{PACKAGE_NAME}.subrosa_executor")


@pytest.fixture(scope="session")
def subrosa_login_module(extension_package: object) -> object:
    """Return local Subrosa Login routes and token validation helpers."""
    return importlib.import_module(f"{PACKAGE_NAME}.subrosa_login")


@pytest.fixture(scope="session")
def subrosa_sync_module(extension_package: object) -> object:
    """Return the milestone Subrosa sync module."""
    return importlib.import_module(f"{PACKAGE_NAME}.subrosa_sync")


@pytest.fixture(scope="session")
def r2_credentials_module(extension_package: object) -> object:
    """Return secure Cloudflare R2 credential persistence helpers."""
    return importlib.import_module(f"{PACKAGE_NAME}.r2_credentials")


@pytest.fixture(scope="session")
def cloudflare_oauth_module(extension_package: object) -> object:
    """Return Cloudflare OAuth and R2 provisioning helpers."""
    return importlib.import_module(f"{PACKAGE_NAME}.cloudflare_oauth")


@pytest.fixture(scope="session")
def execution_history_module(extension_package: object) -> object:
    """Return the persistent remote execution timing history module."""
    return importlib.import_module(f"{PACKAGE_NAME}.execution_history")


@pytest.fixture(scope="session")
def remote_hosts_module(extension_package: object) -> object:
    """Return the persistent SSH host registry module."""
    return importlib.import_module(f"{PACKAGE_NAME}.remote_hosts")


@pytest.fixture(scope="session")
def ssh_docker_module(extension_package: object) -> object:
    """Return the SSH Docker transport module."""
    return importlib.import_module(f"{PACKAGE_NAME}.ssh_docker")


@pytest.fixture(scope="session")
def ssh_runtime_module(extension_package: object) -> object:
    """Return the SSH OCI runtime lifecycle module."""
    return importlib.import_module(f"{PACKAGE_NAME}.ssh_runtime")


@pytest.fixture(scope="session")
def ssh_executor_module(extension_package: object) -> object:
    """Return the SSH remote executor client module."""
    return importlib.import_module(f"{PACKAGE_NAME}.ssh_executor")


@pytest.fixture(scope="session")
def remote_protocol_module(extension_package: object) -> object:
    """Return the generic remote framing protocol module."""
    return importlib.import_module(f"{PACKAGE_NAME}.remote_protocol")


@pytest.fixture(scope="session")
def ssh_worker_module(extension_package: object) -> object:
    """Return the persistent SSH worker module."""
    return importlib.import_module(f"{PACKAGE_NAME}.remote.ssh_worker")


@pytest.fixture(scope="session")
def vast_models_module(extension_package: object) -> object:
    """Return the Vast marketplace and instance models."""
    return importlib.import_module(f"{PACKAGE_NAME}.vast_models")


@pytest.fixture(scope="session")
def vast_api_module(extension_package: object) -> object:
    """Return the asynchronous Vast API client."""
    return importlib.import_module(f"{PACKAGE_NAME}.vast_api")


@pytest.fixture(scope="session")
def vast_simulator_module(extension_package: object) -> object:
    """Return the local Vast API simulator."""
    return importlib.import_module(f"{PACKAGE_NAME}.vast_simulator")


@pytest.fixture(scope="session")
def vast_config_node_module(extension_package: object) -> object:
    """Return the disconnected Vast configuration node module."""
    return importlib.import_module(f"{PACKAGE_NAME}.vast_config_node")


@pytest.fixture(scope="session")
def vast_leases_module(extension_package: object) -> object:
    """Return persistent Vast lease lifecycle helpers."""
    return importlib.import_module(f"{PACKAGE_NAME}.vast_leases")


@pytest.fixture(scope="session")
def vast_ssh_module(extension_package: object) -> object:
    """Return direct Vast SSH and storage adapters."""
    return importlib.import_module(f"{PACKAGE_NAME}.vast_ssh")


@pytest.fixture(scope="session")
def r2_cache_module(extension_package: object) -> object:
    """Return the controller-side Cloudflare R2 cache module."""
    return importlib.import_module(f"{PACKAGE_NAME}.r2_cache")


@pytest.fixture(scope="session")
def r2_materializer_module(extension_package: object) -> object:
    """Return the worker-side signed R2 transfer module."""
    return importlib.import_module(f"{PACKAGE_NAME}.remote.r2_materializer")


@pytest.fixture(scope="session")
def huggingface_assets_module(extension_package: object) -> object:
    """Return persistent Hugging Face asset provenance helpers."""
    return importlib.import_module(f"{PACKAGE_NAME}.huggingface_assets")


@pytest.fixture(scope="session")
def huggingface_discovery_module(extension_package: object) -> object:
    """Return automatic Hugging Face asset provenance discovery helpers."""
    return importlib.import_module(f"{PACKAGE_NAME}.huggingface_discovery")


@pytest.fixture(scope="session")
def huggingface_materializer_module(extension_package: object) -> object:
    """Return the remote Hugging Face file materializer."""
    return importlib.import_module(
        f"{PACKAGE_NAME}.remote.huggingface_materializer"
    )


@pytest.fixture(scope="session")
def vast_runtime_module(extension_package: object) -> object:
    """Return direct Vast worker runtime management."""
    return importlib.import_module(f"{PACKAGE_NAME}.vast_runtime")


@pytest.fixture(scope="session")
def vast_image_build_module(extension_package: object) -> object:
    """Return automatic Vast worker image publication helpers."""
    return importlib.import_module(f"{PACKAGE_NAME}.vast_image_build")


@pytest.fixture(scope="session")
def vast_executor_module(extension_package: object) -> object:
    """Return the direct Vast remote executor client."""
    return importlib.import_module(f"{PACKAGE_NAME}.vast_executor")


@pytest.fixture(scope="session")
def vast_supervisor_module(extension_package: object) -> object:
    """Return the Vast direct-worker supervisor module."""
    return importlib.import_module(f"{PACKAGE_NAME}.remote.vast_supervisor")


@pytest.fixture(scope="session")
def vast_service_module(extension_package: object) -> object:
    """Return the application-level Vast controller service."""
    return importlib.import_module(f"{PACKAGE_NAME}.vast_service")


@pytest.fixture(scope="session")
def api_intercept_module(extension_package: object) -> object:
    """Return the prompt interception module."""
    return importlib.import_module(f"{PACKAGE_NAME}.api_intercept")


@pytest.fixture(scope="session")
def modal_executor_module(extension_package: object) -> object:
    """Return the dynamic proxy node module."""
    return importlib.import_module(f"{PACKAGE_NAME}.modal_executor_node")


@pytest.fixture(scope="session")
def modal_admin_ops_module(extension_package: object) -> object:
    """Return the Modal persistent-object administration module."""
    return importlib.import_module(f"{PACKAGE_NAME}.modal_admin_ops")


@pytest.fixture(scope="session")
def modal_ui_events_module(extension_package: object) -> object:
    """Return the client-scoped Modal UI event module."""
    return importlib.import_module(f"{PACKAGE_NAME}.modal_ui_events")


@pytest.fixture(scope="session")
def remote_graph_analysis_module(extension_package: object) -> object:
    """Return the provider-aware remote graph analysis module."""
    return importlib.import_module(f"{PACKAGE_NAME}.remote_graph_analysis")


@pytest.fixture(scope="session")
def component_planning_module(extension_package: object) -> object:
    """Return the remote component construction and validation module."""
    return importlib.import_module(f"{PACKAGE_NAME}.component_planning")


@pytest.fixture(scope="session")
def prompt_payload_metadata_module(extension_package: object) -> object:
    """Return prompt signatures and resolved-model metadata helpers."""
    return importlib.import_module(f"{PACKAGE_NAME}.prompt_payload_metadata")


@pytest.fixture(scope="session")
def prompt_rewrite_module(extension_package: object) -> object:
    """Return remote payload construction and proxy rewrite helpers."""
    return importlib.import_module(f"{PACKAGE_NAME}.prompt_rewrite")


@pytest.fixture(scope="session")
def prompt_payload_building_module(extension_package: object) -> object:
    """Return remote component payload construction helpers."""
    return importlib.import_module(f"{PACKAGE_NAME}.prompt_payload_building")


@pytest.fixture(scope="session")
def prompt_affinity_planning_module(extension_package: object) -> object:
    """Return affinity, keepalive, and speculative prewarm rewrite helpers."""
    return importlib.import_module(f"{PACKAGE_NAME}.prompt_affinity_planning")


@pytest.fixture(scope="session")
def prompt_interception_module(extension_package: object) -> object:
    """Return queue-time prompt analysis, asset preparation, and rewrite orchestration."""
    return importlib.import_module(f"{PACKAGE_NAME}.prompt_interception")


@pytest.fixture(scope="session")
def execution_scheduling_module(extension_package: object) -> object:
    """Return provider-neutral scheduling and capacity helpers."""
    return importlib.import_module(f"{PACKAGE_NAME}.execution_scheduling")


@pytest.fixture(scope="session")
def execution_assignment_runtime_module(extension_package: object) -> object:
    """Return execution backend and assignment metadata helpers."""
    return importlib.import_module(f"{PACKAGE_NAME}.execution_assignment_runtime")


@pytest.fixture(scope="session")
def execution_plan_reporting_module(extension_package: object) -> object:
    """Return credential-safe execution plan reporting helpers."""
    return importlib.import_module(f"{PACKAGE_NAME}.execution_plan_reporting")


@pytest.fixture(scope="session")
def execution_resource_estimates_module(extension_package: object) -> object:
    """Return execution resource estimation and signature helpers."""
    return importlib.import_module(f"{PACKAGE_NAME}.execution_resource_estimates")


@pytest.fixture(scope="session")
def queue_bridge_module(extension_package: object) -> object:
    """Return ComfyUI queue insertion and remote preparation helpers."""
    return importlib.import_module(f"{PACKAGE_NAME}.queue_bridge")


@pytest.fixture(scope="session")
def prompt_diagnostics_module(extension_package: object) -> object:
    """Return rewritten prompt diagnostic helpers."""
    return importlib.import_module(f"{PACKAGE_NAME}.prompt_diagnostics")


@pytest.fixture(scope="session")
def routes_r2_module(extension_package: object) -> object:
    """Return R2 route registration helpers."""
    return importlib.import_module(f"{PACKAGE_NAME}.routes_r2")


@pytest.fixture(scope="session")
def routes_remote_environments_module(extension_package: object) -> object:
    """Return remote-environment route registration helpers."""
    return importlib.import_module(f"{PACKAGE_NAME}.routes_remote_environments")


@pytest.fixture(scope="session")
def modal_endpoint_module(extension_package: object) -> object:
    """Return the Modal hosted-model endpoint node module."""
    return importlib.import_module(f"{PACKAGE_NAME}.modal_endpoint_node")


@pytest.fixture(scope="session")
def llm_profiles_module(extension_package: object) -> object:
    """Return the curated Modal LLM profile registry module."""
    return importlib.import_module(f"{PACKAGE_NAME}.llm_profiles")


@pytest.fixture
def llm_compatibility_module(extension_package: object) -> object:
    """Import the LLM compatibility module under the synthetic package."""
    del extension_package
    return importlib.import_module(f"{PACKAGE_NAME}.llm_compatibility")


@pytest.fixture
def llm_resolver_module(extension_package: object) -> object:
    """Import the CPU-side LLM profile resolver under the synthetic package."""
    del extension_package
    return importlib.import_module(f"{PACKAGE_NAME}.llm_resolver")


@pytest.fixture(scope="session")
def llm_staging_module(extension_package: object) -> object:
    """Return the CPU-side Modal LLM staging module."""
    return importlib.import_module(f"{PACKAGE_NAME}.llm_staging")


@pytest.fixture(scope="session")
def snapshot_lease_module(extension_package: object) -> object:
    """Return the cross-process snapshot lease module."""
    return importlib.import_module(f"{PACKAGE_NAME}.snapshot_lease")


@pytest.fixture(scope="session")
def staging_process_module(extension_package: object) -> object:
    """Return the bounded remote staging process controller module."""
    return importlib.import_module(f"{PACKAGE_NAME}.staging_process")


@pytest.fixture(scope="session")
def modal_llm_runtime_module(extension_package: object) -> object:
    """Return the resident Modal LLM runtime module."""
    return importlib.import_module(f"{PACKAGE_NAME}.modal_llm_runtime")


@pytest.fixture(scope="session")
def vllm_instrumentation_module(extension_package: object) -> object:
    """Return the vLLM execution-policy and Triton instrumentation module."""
    return importlib.import_module(f"{PACKAGE_NAME}.vllm_instrumentation")


@pytest.fixture(scope="session")
def llm_backend_llamacpp_module(extension_package: object) -> object:
    """Return the resident llama.cpp backend module."""
    return importlib.import_module(f"{PACKAGE_NAME}.llm_backend_llamacpp")


@pytest.fixture(scope="session")
def llm_backend_transformers_module(extension_package: object) -> object:
    """Return the resident Transformers backend module."""
    return importlib.import_module(f"{PACKAGE_NAME}.llm_backend_transformers")


@pytest.fixture(scope="session")
def llm_backend_vllm_module(extension_package: object) -> object:
    """Return the resident vLLM backend module."""
    return importlib.import_module(f"{PACKAGE_NAME}.llm_backend_vllm")


@pytest.fixture(scope="session")
def local_llm_runtime_module(extension_package: object) -> object:
    """Return the Apple-local resident LLM runtime module."""
    return importlib.import_module(f"{PACKAGE_NAME}.local_llm_runtime")


@pytest.fixture(scope="session")
def llm_reasoning_module(extension_package: object) -> object:
    """Return the backend-neutral LLM reasoning-output module."""
    return importlib.import_module(f"{PACKAGE_NAME}.llm_reasoning")


@pytest.fixture(scope="session")
def modal_llm_node_module(extension_package: object) -> object:
    """Return the V3 resident Modal LLM node module."""
    return importlib.import_module(f"{PACKAGE_NAME}.modal_llm_node")


@pytest.fixture(scope="session")
def remote_executor_router_module(extension_package: object) -> object:
    """Return the provider-aware remote executor router module."""
    return importlib.import_module(f"{PACKAGE_NAME}.remote_executor_router")


@pytest.fixture(scope="session")
def proxy_payloads_module(extension_package: object) -> object:
    """Return the run-scoped proxy payload registry module."""
    return importlib.import_module(f"{PACKAGE_NAME}.proxy_payloads")


@pytest.fixture(scope="session")
def proxy_node_factory_module(extension_package: object) -> object:
    """Return the dynamic ComfyUI proxy-node factory module."""
    return importlib.import_module(f"{PACKAGE_NAME}.proxy_node_factory")


@pytest.fixture(scope="session")
def remote_modal_app_module(extension_package: object) -> object:
    """Return the remote execution module."""
    return importlib.import_module(f"{PACKAGE_NAME}.remote.modal_app")


@pytest.fixture(scope="session")
def modal_billing_module(extension_package: object) -> object:
    """Return Modal billing queries and cache state."""
    return importlib.import_module(f"{PACKAGE_NAME}.remote.modal_billing")


@pytest.fixture(scope="session")
def modal_container_logs_module(extension_package: object) -> object:
    """Return managed Modal container and log-stream state."""
    return importlib.import_module(f"{PACKAGE_NAME}.remote.modal_container_logs")


@pytest.fixture(scope="session")
def host_session_bridge_module(extension_package: object) -> object:
    """Return durable host session-bridge state and recovery helpers."""
    return importlib.import_module(f"{PACKAGE_NAME}.remote.host_session_bridge")


@pytest.fixture(scope="session")
def local_execution_module(extension_package: object) -> object:
    """Return headless local node and subgraph execution helpers."""
    return importlib.import_module(f"{PACKAGE_NAME}.remote.local_execution")


@pytest.fixture(scope="session")
def local_ui_events_module(extension_package: object) -> object:
    """Return local ComfyUI event emission helpers."""
    return importlib.import_module(f"{PACKAGE_NAME}.remote.local_ui_events")


@pytest.fixture(scope="session")
def modal_deployment_module(extension_package: object) -> object:
    """Return Modal deployment and runtime compatibility helpers."""
    return importlib.import_module(f"{PACKAGE_NAME}.remote.modal_deployment")


@pytest.fixture(scope="session")
def modal_warmup_module(extension_package: object) -> object:
    """Return Modal warmup, snapshot, and keepalive helpers."""
    return importlib.import_module(f"{PACKAGE_NAME}.remote.modal_warmup")


@pytest.fixture(scope="session")
def mapped_execution_module(extension_package: object) -> object:
    """Return mapped and implicit-batch execution helpers."""
    return importlib.import_module(f"{PACKAGE_NAME}.remote.mapped_execution")


@pytest.fixture(scope="session")
def modal_interrupts_module(extension_package: object) -> object:
    """Return Modal prompt cancellation and interrupt state."""
    return importlib.import_module(f"{PACKAGE_NAME}.remote.modal_interrupts")


@pytest.fixture(scope="session")
def payload_stream_module(extension_package: object) -> object:
    """Return Modal stream-to-local-UI forwarding helpers."""
    return importlib.import_module(f"{PACKAGE_NAME}.remote.payload_stream")


@pytest.fixture(scope="session")
def modal_llm_profile_staging_module(extension_package: object) -> object:
    """Return host-side Modal LLM staging helpers and registry state."""
    return importlib.import_module(
        f"{PACKAGE_NAME}.remote.modal_llm_profile_staging"
    )


@pytest.fixture(scope="session")
def serialization_module(extension_package: object) -> object:
    """Return the serialization helpers module."""
    return importlib.import_module(f"{PACKAGE_NAME}.serialization")


@pytest.fixture(scope="session")
def durable_state_module(extension_package: object) -> object:
    """Return the durable invocation and object-storage helpers module."""
    return importlib.import_module(f"{PACKAGE_NAME}.durable_state")


@pytest.fixture(scope="session")
def output_artifacts_module(extension_package: object) -> object:
    """Return the remote output artifact transfer helpers module."""
    return importlib.import_module(f"{PACKAGE_NAME}.output_artifacts")


@pytest.fixture(scope="session")
def settings_module(extension_package: object) -> object:
    """Return the settings module."""
    return importlib.import_module(f"{PACKAGE_NAME}.settings")


@pytest.fixture(scope="session")
def session_state_module(extension_package: object) -> object:
    """Return the prompt-scoped remote session helpers module."""
    return importlib.import_module(f"{PACKAGE_NAME}.session_state")


@pytest.fixture(scope="session")
def runtime_environment_module(extension_package: object) -> object:
    """Return the pinned remote runtime identity module."""
    return importlib.import_module(f"{PACKAGE_NAME}.runtime_environment")


@pytest.fixture(scope="session")
def resource_telemetry_module(extension_package: object) -> object:
    """Return the backend-neutral resource telemetry module."""
    return importlib.import_module(f"{PACKAGE_NAME}.resource_telemetry")


@pytest.fixture(scope="session")
def sync_engine_module(extension_package: object) -> object:
    """Return the sync engine module."""
    return importlib.import_module(f"{PACKAGE_NAME}.sync_engine")


@pytest.fixture(scope="session")
def sync_backends_module(extension_package: object) -> object:
    """Return local and Modal synchronization backends."""
    return importlib.import_module(f"{PACKAGE_NAME}.sync_backends")


@pytest.fixture(scope="session")
def sync_hashing_module(extension_package: object) -> object:
    """Return the persistent synchronization hashing module."""
    return importlib.import_module(f"{PACKAGE_NAME}.sync_hashing")


@pytest.fixture(scope="session")
def sync_custom_nodes_module(extension_package: object) -> object:
    """Return the custom-node synchronization module."""
    return importlib.import_module(f"{PACKAGE_NAME}.sync_custom_nodes")


@pytest.fixture(scope="session")
def sync_r2_transfer_module(extension_package: object) -> object:
    """Return the R2 transfer and write-back coordination module."""
    return importlib.import_module(f"{PACKAGE_NAME}.sync_r2_transfer")


@pytest.fixture(scope="session")
def modal_cloud_module() -> object:
    """Return the stable Modal cloud entry module."""
    _ensure_import_paths()
    return importlib.import_module("comfyui_modal_sync_cloud")


@pytest.fixture(scope="session")
def cloud_app_guard_module(modal_cloud_module: object) -> object:
    """Return the extracted cloud app-existence guard module."""
    del modal_cloud_module
    return importlib.import_module("cloud_app_guard")


@pytest.fixture(scope="session")
def cloud_durable_invocation_module(modal_cloud_module: object) -> object:
    """Return the extracted durable cloud invocation module."""
    del modal_cloud_module
    return importlib.import_module("cloud_durable_invocation")


@pytest.fixture(autouse=True)
def reset_modal_environment(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> None:
    """Isolate Modal-Sync environment variables between tests."""
    if (
        request.node.get_closest_marker("live_modal") is not None
        or request.node.get_closest_marker("live_vast") is not None
    ):
        return
    sync_r2_transfer_module = sys.modules.get(f"{PACKAGE_NAME}.sync_r2_transfer")
    coordinator = getattr(
        sync_r2_transfer_module,
        "_R2_WRITE_BACK_COORDINATOR",
        None,
    )
    reset_reservations = getattr(
        coordinator,
        "reset_prompt_reservations_for_tests",
        None,
    )
    if callable(reset_reservations):
        reset_reservations()
    monkeypatch.setenv("COMFY_MODAL_EXECUTION_MODE", "local")
    monkeypatch.setenv("COMFY_MODAL_APP_NAME", "comfy-modal-sync")
    monkeypatch.setenv("COMFY_MODAL_LOCAL_STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.delenv("COMFY_MODAL_INSTANCE_ID_PATH", raising=False)
    monkeypatch.delenv("COMFY_MODAL_AUTO_DEPLOY", raising=False)
    monkeypatch.delenv("COMFY_MODAL_ALLOW_EPHEMERAL_FALLBACK", raising=False)
    monkeypatch.delenv("COMFY_MODAL_CUSTOM_NODES_DIR", raising=False)
    monkeypatch.delenv("COMFY_MODAL_ENABLE_MEMORY_SNAPSHOT", raising=False)
    monkeypatch.delenv("COMFY_MODAL_ENABLE_GPU_MEMORY_SNAPSHOT", raising=False)
    monkeypatch.delenv("COMFY_MODAL_SECRET_NAME", raising=False)
    monkeypatch.delenv("COMFY_MODAL_INTERRUPT_DICT_NAME", raising=False)
    monkeypatch.delenv("COMFY_MODAL_INVOCATION_DICT_NAME", raising=False)
    monkeypatch.delenv("COMFY_MODAL_INVOCATION_RESULT_INLINE_MAX_BYTES", raising=False)
    monkeypatch.delenv("COMFY_MODAL_TERMINATE_CONTAINER_ON_ERROR", raising=False)
    monkeypatch.delenv("COMFY_MODAL_GPU", raising=False)
    monkeypatch.delenv("COMFY_MODAL_MAX_CONTAINERS", raising=False)
    monkeypatch.delenv("COMFY_MODAL_BUFFER_CONTAINERS", raising=False)
    monkeypatch.delenv("COMFY_MODAL_MAX_INFLIGHT_CALLS", raising=False)
    monkeypatch.delenv("COMFY_MODAL_EXECUTION_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("COMFY_MODAL_STARTUP_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("COMFY_MODAL_ENABLE_PROACTIVE_WARMUP", raising=False)
    monkeypatch.delenv("COMFY_MODAL_REMOTE_CANCEL_GRACE_SECONDS", raising=False)
    monkeypatch.delenv("COMFY_MODAL_REMOTE_CANCEL_RESTART_SECONDS", raising=False)
    monkeypatch.delenv("COMFY_MODAL_SESSION_BRIDGE_DICT_NAME", raising=False)
    monkeypatch.delenv("COMFY_MODAL_BRIDGE_INLINE_MAX_BYTES", raising=False)
    monkeypatch.delenv("COMFY_MODAL_STREAM_EVENT_QUEUE_MAXSIZE", raising=False)
    monkeypatch.delenv("COMFY_MODAL_STREAM_REMOTE_CONTAINER_LOGS", raising=False)
    monkeypatch.delenv("COMFY_MODAL_SCALEDOWN_WINDOW", raising=False)
    monkeypatch.delenv("COMFY_MODAL_MIN_CONTAINERS", raising=False)
    monkeypatch.delenv("COMFYUI_ROOT", raising=False)
    monkeypatch.delenv("MODAL_KEY", raising=False)
    monkeypatch.delenv("MODAL_SECRET", raising=False)
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    monkeypatch.delenv("COMFY_MODAL_VAST_API_BASE_URL", raising=False)
    monkeypatch.delenv("COMFY_MODAL_VAST_IMAGE", raising=False)
    monkeypatch.delenv("COMFY_MODAL_VAST_SSH_IDENTITY_FILE", raising=False)
    for r2_name in (
        "COMFY_MODAL_R2_ENABLED",
        "COMFY_MODAL_R2_ACCOUNT_ID",
        "COMFY_MODAL_R2_BUCKET",
        "COMFY_MODAL_R2_ACCESS_KEY_ID",
        "COMFY_MODAL_R2_SECRET_ACCESS_KEY",
        "COMFY_MODAL_R2_ENDPOINT_URL",
        "COMFY_MODAL_R2_KEY_PREFIX",
        "COMFY_MODAL_R2_WRITE_BACK",
        "COMFY_MODAL_R2_URL_TTL_SECONDS",
        "COMFY_MODAL_R2_MULTIPART_PART_MIB",
        "COMFY_MODAL_R2_SINGLE_UPLOAD_MAX_MIB",
    ):
        monkeypatch.delenv(r2_name, raising=False)
