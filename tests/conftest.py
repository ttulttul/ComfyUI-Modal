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
        av_video_reformatter_module.ColorRange = type("ColorRange", (), {})

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
def vast_runtime_module(extension_package: object) -> object:
    """Return direct Vast worker runtime management."""
    return importlib.import_module(f"{PACKAGE_NAME}.vast_runtime")


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
def modal_llm_runtime_module(extension_package: object) -> object:
    """Return the resident Modal LLM runtime module."""
    return importlib.import_module(f"{PACKAGE_NAME}.modal_llm_runtime")


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
def remote_modal_app_module(extension_package: object) -> object:
    """Return the remote execution module."""
    return importlib.import_module(f"{PACKAGE_NAME}.remote.modal_app")


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
def sync_engine_module(extension_package: object) -> object:
    """Return the sync engine module."""
    return importlib.import_module(f"{PACKAGE_NAME}.sync_engine")


@pytest.fixture(scope="session")
def modal_cloud_module() -> object:
    """Return the stable Modal cloud entry module."""
    _ensure_import_paths()
    return importlib.import_module("comfyui_modal_sync_cloud")


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
