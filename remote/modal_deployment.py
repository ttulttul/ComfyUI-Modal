"""Modal cloud-module loading, compatibility checks, and deployment lifecycle."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from functools import lru_cache
import hashlib
import importlib
import importlib.util
import inspect
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from types import ModuleType
from typing import Any, Callable, Iterator, Mapping

from ..runtime_environment import (
    REMOTE_APP_PROTOCOL_VERSION,
    RemoteRuntimeIdentity,
    build_remote_runtime_identity,
)
from ..settings import (
    ModalSyncSettings,
    get_settings,
    modal_deployment_app_name,
    settings_for_modal_gpu,
)
from .local_execution import RemoteSubgraphExecutionError
from .local_ui_events import (
    _emit_local_remote_dispatch_status,
    _emit_local_remote_startup_status,
)

logger = logging.getLogger(__name__)

try:
    import modal  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - local fallback environments.
    modal = None

_MODAL_CLOUD_MODULE_NAME = "comfyui_modal_sync_cloud"
_REMOTE_APP_PROTOCOL_VERSION = REMOTE_APP_PROTOCOL_VERSION
_MODAL_CLOUD_MODULE_LOCK = threading.Lock()
_MODAL_CLOUD_SETTINGS_STATE = threading.local()
_MODAL_AUTO_DEPLOY_LOCK = threading.Lock()
_MODAL_AUTO_DEPLOY_STATES: dict[tuple[str, str | None], "_ModalAutoDeployState"] = {}
_MODAL_REMOTE_APP_VERSION_OK: set[tuple[str, str | None, str]] = set()
_MODAL_APP_STOP_TIMEOUT_SECONDS = 120.0


@dataclass(frozen=True)
class ModalDeploymentHooks:
    """Callbacks supplied by the host orchestrator for adjacent runtime concerns."""

    ensure_llm_profiles_staged: Callable[[dict[str, Any], str], None]
    schedule_post_deploy_runtime_seed: Callable[[Mapping[str, Any]], bool]
    await_matching_speculative_prewarm: Callable[
        [dict[str, Any], threading.Event | None], None
    ]
    prepare_snapshot_profile_fields: Callable[[dict[str, Any]], str]
    select_gpu_snapshot_for_profile: Callable[[dict[str, Any], str], Any]
    prompt_parallelism_target: Callable[[Mapping[str, Any]], int]


_DEPLOYMENT_HOOKS: ModalDeploymentHooks | None = None


def configure_modal_deployment_hooks(hooks: ModalDeploymentHooks) -> None:
    """Install host callbacks without importing upward into the orchestrator."""
    global _DEPLOYMENT_HOOKS
    _DEPLOYMENT_HOOKS = hooks


def _deployment_hooks() -> ModalDeploymentHooks:
    """Return the configured host callbacks or fail with a clear import-order error."""
    if _DEPLOYMENT_HOOKS is None:
        raise RuntimeError("Modal deployment hooks have not been configured.")
    return _DEPLOYMENT_HOOKS


def _ensure_llm_profiles_staged(*args: Any, **kwargs: Any) -> Any:
    """Delegate LLM staging through the injected host callback."""
    return _deployment_hooks().ensure_llm_profiles_staged(*args, **kwargs)


def _schedule_post_deploy_runtime_seed(*args: Any, **kwargs: Any) -> Any:
    """Delegate post-deploy runtime seeding through the injected host callback."""
    return _deployment_hooks().schedule_post_deploy_runtime_seed(*args, **kwargs)


def _await_matching_speculative_prewarm(*args: Any, **kwargs: Any) -> Any:
    """Delegate speculative prewarm joining through the injected host callback."""
    return _deployment_hooks().await_matching_speculative_prewarm(*args, **kwargs)


def _prepare_snapshot_profile_fields(*args: Any, **kwargs: Any) -> Any:
    """Delegate snapshot-profile preparation through the injected host callback."""
    return _deployment_hooks().prepare_snapshot_profile_fields(*args, **kwargs)


def _select_gpu_snapshot_for_profile(*args: Any, **kwargs: Any) -> Any:
    """Delegate snapshot selection through the injected host callback."""
    return _deployment_hooks().select_gpu_snapshot_for_profile(*args, **kwargs)


def _prompt_parallelism_target(*args: Any, **kwargs: Any) -> Any:
    """Delegate prompt-pool sizing through the injected host callback."""
    return _deployment_hooks().prompt_parallelism_target(*args, **kwargs)


def _settings_for_payload(payload: Mapping[str, Any]) -> ModalSyncSettings:
    """Resolve settings with the workflow-selected GPU carried by one payload."""
    settings = get_settings()
    modal_gpu = payload.get("modal_gpu")
    if modal_gpu is not None:
        settings = settings_for_modal_gpu(settings, modal_gpu)
    raw_max_containers = payload.get("modal_max_containers")
    if raw_max_containers is None:
        return settings
    max_containers = int(raw_max_containers)
    if max_containers <= 0:
        raise ValueError("modal_max_containers must be positive.")
    return replace(settings, max_containers=max_containers)

@contextmanager
def _modal_cloud_settings_override(settings: ModalSyncSettings) -> Iterator[None]:
    """Expose request-specific settings while constructing the deployable cloud module."""
    previous_settings = getattr(_MODAL_CLOUD_SETTINGS_STATE, "settings", None)
    _MODAL_CLOUD_SETTINGS_STATE.settings = settings
    try:
        yield
    finally:
        if previous_settings is None:
            try:
                delattr(_MODAL_CLOUD_SETTINGS_STATE, "settings")
            except AttributeError:
                pass
        else:
            _MODAL_CLOUD_SETTINGS_STATE.settings = previous_settings

@dataclass
class _ModalAutoDeployState:
    """Track one thread-safe deployed-app readiness lifecycle."""

    condition: threading.Condition = field(default_factory=threading.Condition)
    deploy_in_progress: bool = False
    ready: bool = False
    last_error: BaseException | None = None

class ModalRemoteInvocationError(RuntimeError):
    """Raised when the Modal client cannot invoke the remote runtime."""

class ModalRemoteAppOutOfDateError(ModalRemoteInvocationError):
    """Raised when a deployed Modal app is incompatible with the local client."""

def _remote_worker_pool_affinity_key(slot_index: int) -> str:
    """Return the stable worker-pool affinity key for one reusable remote slot."""
    return f"worker-pool:slot:{int(slot_index)}"

def _component_pool_slot_index(payload: dict[str, Any]) -> int:
    """Return the reusable worker-pool slot index for one ordinary remote payload."""
    warmup_slot_index = payload.get("warmup_slot_index")
    if warmup_slot_index is not None:
        return max(0, int(warmup_slot_index))
    if bool(payload.get("remote_local_gap_pool")):
        return 0

    prompt_parallelism_target = _prompt_parallelism_target(payload)
    slot_count = max(1, int(prompt_parallelism_target))
    component_id = str(payload.get("component_id") or "")
    if not component_id:
        return 0

    component_hash = hashlib.sha256(component_id.encode("utf-8")).digest()
    return int.from_bytes(component_hash[:8], "big") % slot_count

def _remote_worker_affinity_key(payload: dict[str, Any]) -> str:
    """Return the reusable worker-pool affinity key for one remote payload."""
    configured_affinity_group = payload.get("remote_worker_affinity_group")
    if configured_affinity_group is None:
        return _remote_worker_pool_affinity_key(_component_pool_slot_index(payload))
    affinity_group = str(configured_affinity_group).strip().lower()
    if affinity_group not in {"comfy", "llm"}:
        affinity_group = "comfy"
    slot_index = _component_pool_slot_index(payload)
    return f"worker-pool:{affinity_group}:slot:{slot_index}"

def _mapped_lane_affinity_key(payload: dict[str, Any], lane_index: int) -> str | None:
    """Return the stable per-lane worker-pool affinity key used for mapped execution."""
    del payload
    return _remote_worker_pool_affinity_key(lane_index)

def _modal_lookup_error_types() -> tuple[type[BaseException], ...]:
    """Return Modal exception types that indicate lookup or hydration failure."""
    if modal is None:
        return tuple()
    exception_module = getattr(modal, "exception", None)
    if exception_module is None:
        return tuple()

    error_types: list[type[BaseException]] = []
    for error_name in ("NotFoundError", "ExecutionError", "InvalidError"):
        error_type = getattr(exception_module, error_name, None)
        if isinstance(error_type, type) and issubclass(error_type, BaseException):
            error_types.append(error_type)
    return tuple(error_types)

def _is_missing_modal_deployment_error(exc: BaseException) -> bool:
    """Return whether one Modal lookup failure indicates missing deployed app state."""
    message = str(exc).strip().lower()
    if any(
        marker in message
        for marker in (
            "could not deserialize remote exception",
            "remote traceback",
            "remotesubgraphexecutionerror",
            "object of type",
            "is not json serializable",
        )
    ):
        return False
    if "not deployed" in message:
        return True
    if "not found" not in message:
        return False
    return any(
        marker in message
        for marker in (
            "lookup failed for cls",
            "app '",
            'app "',
            "class '",
            'class "',
            "not found in environment",
        )
    )

def _load_modal_cloud_module() -> Any:
    """Load the stable Modal cloud entry module under a valid Python name."""
    settings = getattr(_MODAL_CLOUD_SETTINGS_STATE, "settings", None) or get_settings()
    deployment_app_name = modal_deployment_app_name(settings)
    with _MODAL_CLOUD_MODULE_LOCK:
        existing_module = sys.modules.get(_MODAL_CLOUD_MODULE_NAME)
        existing_gpu = getattr(
            existing_module, "__comfy_modal_gpu__", settings.modal_gpu
        )
        existing_app_name = getattr(existing_module, "__comfy_modal_app_name__", None)
        existing_secret_name = getattr(
            existing_module, "__comfy_modal_secret_name__", None
        )
        if (
            existing_module is not None
            and getattr(existing_module, "app", None) is not None
            and existing_gpu == settings.modal_gpu
            and existing_app_name == deployment_app_name
            and existing_secret_name == settings.modal_secret_name
        ):
            return existing_module
        if existing_module is not None:
            logger.warning(
                "Discarding Modal cloud module %s before reload for app=%s gpu=%s secret=%s (previous_app=%s previous_gpu=%s previous_secret=%s).",
                _MODAL_CLOUD_MODULE_NAME,
                deployment_app_name,
                settings.modal_gpu,
                settings.modal_secret_name,
                existing_app_name,
                existing_gpu,
                existing_secret_name,
            )
            sys.modules.pop(_MODAL_CLOUD_MODULE_NAME, None)

        cloud_module_path = (
            Path(__file__).resolve().parents[1] / f"{_MODAL_CLOUD_MODULE_NAME}.py"
        )
        module_spec = importlib.util.spec_from_file_location(
            _MODAL_CLOUD_MODULE_NAME,
            cloud_module_path,
        )
        if module_spec is None or module_spec.loader is None:
            raise ModalRemoteInvocationError(
                f"Unable to create module spec for Modal cloud entrypoint at {cloud_module_path}."
            )

        cloud_module = importlib.util.module_from_spec(module_spec)
        setattr(cloud_module, "__comfy_modal_settings_override__", settings)
        setattr(cloud_module, "__comfy_modal_gpu__", settings.modal_gpu)
        setattr(cloud_module, "__comfy_modal_app_name__", deployment_app_name)
        setattr(cloud_module, "__comfy_modal_secret_name__", settings.modal_secret_name)
        sys.modules[_MODAL_CLOUD_MODULE_NAME] = cloud_module
        try:
            module_spec.loader.exec_module(cloud_module)
        except BaseException:
            sys.modules.pop(_MODAL_CLOUD_MODULE_NAME, None)
            raise
        return cloud_module

def _install_modal_cloud_exception_compatibility_module() -> None:
    """Expose cloud exception definitions without loading the deployable cloud app."""
    if _MODAL_CLOUD_MODULE_NAME in sys.modules:
        return

    compatibility_module = ModuleType(_MODAL_CLOUD_MODULE_NAME)
    setattr(
        compatibility_module,
        "RemoteSubgraphExecutionError",
        RemoteSubgraphExecutionError,
    )
    exception_bases: dict[str, type[BaseException]] = {
        "RemoteInvocationInProgressError": RuntimeError,
        "RemoteInvocationAbandonedError": RuntimeError,
        "RemoteCanaryInterruptedError": RuntimeError,
        "RemoteCanaryBarrierTimeoutError": TimeoutError,
        "ExistingModalAppError": RuntimeError,
    }
    for exception_name, exception_base in exception_bases.items():
        setattr(
            compatibility_module,
            exception_name,
            type(
                exception_name,
                (exception_base,),
                {"__module__": _MODAL_CLOUD_MODULE_NAME},
            ),
        )
    sys.modules[_MODAL_CLOUD_MODULE_NAME] = compatibility_module

def _lookup_deployed_remote_engine(
    payload: dict[str, Any],
    *,
    affinity_key_override: str | None = None,
    protocol_probe: bool = False,
) -> Any:
    """Look up the deployed runtime, optionally omitting new parameters for probing."""
    if modal is None:
        raise ModalRemoteInvocationError("Modal SDK is unavailable.")

    settings = _settings_for_payload(payload)
    deployment_app_name = modal_deployment_app_name(settings)
    snapshot_profile_key = _prepare_snapshot_profile_fields(payload)
    gpu_snapshot_enabled = (
        False
        if protocol_probe
        else _select_gpu_snapshot_for_profile(payload, snapshot_profile_key)
    )
    worker_affinity_key = affinity_key_override or _remote_worker_affinity_key(payload)
    logger.info(
        "Attempting deployed Modal invocation for app=%s class=%s component=%s worker_affinity=%s snapshot_profile=%s gpu_snapshot_enabled=%s.",
        deployment_app_name,
        "RemoteEngine",
        payload.get("component_id"),
        worker_affinity_key,
        snapshot_profile_key or None,
        gpu_snapshot_enabled,
    )
    remote_cls = modal.Cls.from_name(deployment_app_name, "RemoteEngine")
    if bool(payload.get("remote_local_gap_pool")) and hasattr(remote_cls, "with_options"):
        local_gap_scaledown_seconds = max(
            int(settings.scaledown_window_seconds),
            int(settings.local_gap_keepalive_seconds),
        )
        remote_cls = remote_cls.with_options(
            scaledown_window=local_gap_scaledown_seconds,
        )
        logger.info(
            "Using local-gap Modal pool component=%s worker_affinity=%s scaledown_window=%ds.",
            payload.get("component_id"),
            worker_affinity_key,
            local_gap_scaledown_seconds,
        )
    remote_engine_kwargs: dict[str, Any] = {
        "gpu_snapshot_enabled": gpu_snapshot_enabled,
    }
    if snapshot_profile_key:
        remote_engine_kwargs["snapshot_profile_key"] = snapshot_profile_key
    if not protocol_probe:
        remote_engine_kwargs["worker_affinity_key"] = worker_affinity_key
    else:
        logger.info(
            "Probing deployed RemoteEngine protocol without affinity parameters component=%s.",
            payload.get("component_id"),
        )
    remote_engine = remote_cls(**remote_engine_kwargs)
    with _MODAL_AUTO_DEPLOY_LOCK:
        runtime_is_known_current = (
            _modal_runtime_cache_key(payload) in _MODAL_REMOTE_APP_VERSION_OK
        )
    if runtime_is_known_current:
        _ensure_llm_profiles_staged(payload, deployment_app_name)
    return remote_engine

def _modal_environment_name() -> str | None:
    """Return the active Modal environment name when explicitly configured."""
    environment_name = os.getenv("MODAL_ENVIRONMENT")
    if environment_name is None:
        return None
    normalized = environment_name.strip()
    return normalized or None

def _modal_deploy_cache_key(
    payload: Mapping[str, Any] | None = None,
) -> tuple[str, str | None]:
    """Return the cache key for auto-deployed Modal apps."""
    settings = _settings_for_payload(payload) if payload is not None else get_settings()
    return (modal_deployment_app_name(settings), _modal_environment_name())

@lru_cache(maxsize=8)
def _remote_runtime_identity_for_settings(
    settings: ModalSyncSettings,
) -> RemoteRuntimeIdentity:
    """Build and cache the local runtime identity for one resolved settings object."""
    return build_remote_runtime_identity(
        repo_root=Path(__file__).resolve().parents[1],
        comfyui_root=settings.comfyui_root,
        custom_nodes_dir=settings.custom_nodes_dir,
        settings=settings,
    )

def _expected_remote_runtime_fingerprint(
    payload: Mapping[str, Any] | None = None
) -> str:
    """Return the exact deployment fingerprint required by this local client."""
    settings = _settings_for_payload(payload) if payload is not None else get_settings()
    return _remote_runtime_identity_for_settings(settings).fingerprint

def _modal_runtime_cache_key(
    payload: Mapping[str, Any] | None = None,
) -> tuple[str, str | None, str]:
    """Return the version cache key for the current app, environment, and runtime."""
    settings = _settings_for_payload(payload) if payload is not None else get_settings()
    return (
        modal_deployment_app_name(settings),
        _modal_environment_name(),
        _expected_remote_runtime_fingerprint(payload),
    )

def _call_modal_method(method: Any, *args: Any, **kwargs: Any) -> Any:
    """Invoke a Modal method handle or an in-process test double."""
    remote_method = getattr(method, "remote", None)
    if callable(remote_method):
        return remote_method(*args, **kwargs)
    return method(*args, **kwargs)

def _remote_engine_runtime_version(remote_engine: Any) -> dict[str, Any] | None:
    """Return runtime version metadata from a deployed engine when available."""
    version_method = getattr(remote_engine, "runtime_version", None)
    if version_method is None:
        return None
    version_payload = _call_modal_method(version_method)
    if not isinstance(version_payload, dict):
        return None
    return version_payload

def _remote_runtime_version_from_cpu_stager(
    payload: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Read deployment identity from the CPU stager when that method is available."""
    if modal is None:
        return None
    settings = _settings_for_payload(payload)
    deployment_app_name = modal_deployment_app_name(settings)
    stager_cls = modal.Cls.from_name(deployment_app_name, "ModelStager")
    stager = stager_cls()
    version_method = getattr(stager, "runtime_version", None)
    if version_method is None:
        return None
    try:
        version_payload = _call_modal_method(version_method)
    except AttributeError:
        return None
    except _modal_lookup_error_types() as exc:
        if _is_missing_modal_deployment_error(exc):
            raise
        logger.info(
            "CPU ModelStager runtime identity is unavailable for app=%s; "
            "falling back to the legacy GPU protocol probe: %s",
            deployment_app_name,
            exc,
        )
        return None
    if not isinstance(version_payload, dict):
        return None
    return version_payload

def _runtime_fingerprint_from_payload(
    version_payload: dict[str, Any] | None
) -> str | None:
    """Return a normalized runtime fingerprint from remote version metadata."""
    if version_payload is None:
        return None
    runtime_fingerprint = version_payload.get("runtime_fingerprint")
    if not isinstance(runtime_fingerprint, str):
        return None
    normalized = runtime_fingerprint.strip()
    return normalized or None

def _is_runtime_version_payload_current(
    version_payload: dict[str, Any] | None,
    payload: Mapping[str, Any] | None = None,
) -> bool:
    """Return whether remote version metadata exactly matches this local runtime."""
    if version_payload is None:
        return False
    protocol_version = version_payload.get("protocol_version")
    if isinstance(protocol_version, bool) or not isinstance(protocol_version, int):
        return False
    return (
        protocol_version == _REMOTE_APP_PROTOCOL_VERSION
        and _runtime_fingerprint_from_payload(version_payload)
        == _expected_remote_runtime_fingerprint(payload)
    )

def _is_remote_engine_runtime_current(
    remote_engine: Any,
    payload: Mapping[str, Any] | None = None,
) -> bool:
    """Return whether a deployed engine exactly matches this local runtime."""
    return _is_runtime_version_payload_current(
        _remote_engine_runtime_version(remote_engine),
        payload,
    )

def _stop_modal_app_via_sdk(app_name: str) -> bool:
    """Try to stop a Modal app through the SDK if this SDK version exposes app stopping."""
    if modal is None:
        return False
    try:
        experimental_namespace = importlib.import_module("modal.experimental")
    except ModuleNotFoundError as exc:
        if exc.name not in {"modal", "modal.experimental"}:
            raise
    else:
        stop_app = getattr(experimental_namespace, "stop_app", None)
        if callable(stop_app):
            try:
                _call_modal_method(
                    stop_app,
                    app_name,
                    environment_name=_modal_environment_name(),
                )
            except _modal_lookup_error_types():
                return True
            return True

    app_namespace = getattr(modal, "App", None)
    app_lookup = getattr(app_namespace, "lookup", None)
    if not callable(app_lookup):
        return False
    try:
        lookup_signature = inspect.signature(app_lookup)
    except (TypeError, ValueError):
        lookup_signature = None
    lookup_kwargs: dict[str, Any] = {}
    if lookup_signature is not None:
        if "create_if_missing" in lookup_signature.parameters:
            lookup_kwargs["create_if_missing"] = False
        elif "create" in lookup_signature.parameters:
            lookup_kwargs["create"] = False
    try:
        app_handle = app_lookup(app_name, **lookup_kwargs)
    except _modal_lookup_error_types():
        return True
    stop_method = getattr(app_handle, "stop", None)
    if not callable(stop_method):
        return False
    _call_modal_method(stop_method)
    return True

def _stop_modal_app_via_cli(app_name: str) -> bool:
    """Try to stop a Modal app non-interactively through the Modal CLI."""
    modal_cli = shutil.which("modal")
    if modal_cli is None:
        return False
    command = [modal_cli, "app", "stop", app_name, "--yes"]
    environment_name = _modal_environment_name()
    if environment_name is not None:
        command.extend(("--env", environment_name))
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=_MODAL_APP_STOP_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        logger.warning(
            "Modal CLI app stop timed out for app=%s environment=%s after %.1fs.",
            app_name,
            environment_name or "<default>",
            _MODAL_APP_STOP_TIMEOUT_SECONDS,
        )
        return False
    if completed.returncode != 0:
        logger.warning(
            "Modal CLI app stop failed for app=%s exit_code=%s stderr=%s",
            app_name,
            completed.returncode,
            completed.stderr.strip(),
        )
        return False
    return True

def _stop_modal_app_for_replacement(app_name: str) -> None:
    """Stop an out-of-date Modal app before replacing it with a fresh deployment."""
    if _stop_modal_app_via_sdk(app_name):
        logger.warning(
            "Stopped out-of-date Modal app %s through the Modal SDK.", app_name
        )
        return
    if _stop_modal_app_via_cli(app_name):
        logger.warning(
            "Stopped out-of-date Modal app %s through the Modal CLI.", app_name
        )
        return
    raise ModalRemoteInvocationError(
        f"Deployed Modal app {app_name!r} is out of date, but Modal-Sync could not stop it automatically. "
        f"Stop it manually with `modal app stop {app_name}` and retry."
    )

def _mark_modal_deploy_state_not_ready(deploy_key: tuple[str, str | None]) -> None:
    """Invalidate the local in-process deploy readiness cache for one app."""
    with _MODAL_AUTO_DEPLOY_LOCK:
        deploy_state = _MODAL_AUTO_DEPLOY_STATES.get(deploy_key)
    if deploy_state is None:
        return
    with deploy_state.condition:
        deploy_state.ready = False
        deploy_state.last_error = None
        deploy_state.condition.notify_all()

def _replace_outdated_modal_app(
    payload: dict[str, Any],
    remote_engine: Any,
    *,
    version_payload: dict[str, Any] | None = None,
) -> Any:
    """Stop and auto-deploy a replacement for an incompatible deployed app."""
    settings = _settings_for_payload(payload)
    deployment_app_name = modal_deployment_app_name(settings)
    deploy_key = _modal_deploy_cache_key(payload)
    runtime_cache_key = _modal_runtime_cache_key(payload)
    if version_payload is None:
        version_payload = _remote_engine_runtime_version(remote_engine)
    protocol_version = (
        version_payload.get("protocol_version")
        if isinstance(version_payload, dict)
        else None
    )
    remote_fingerprint = _runtime_fingerprint_from_payload(version_payload)
    local_fingerprint = _expected_remote_runtime_fingerprint(payload)
    logger.warning(
        "Deployed Modal app %s is out of date for component=%s remote_protocol=%s local_protocol=%s remote_fingerprint=%s local_fingerprint=%s; stopping and replacing it.",
        deployment_app_name,
        payload.get("component_id"),
        protocol_version,
        _REMOTE_APP_PROTOCOL_VERSION,
        remote_fingerprint,
        local_fingerprint,
    )
    with _MODAL_AUTO_DEPLOY_LOCK:
        _MODAL_REMOTE_APP_VERSION_OK.discard(runtime_cache_key)
    _mark_modal_deploy_state_not_ready(deploy_key)
    _stop_modal_app_for_replacement(deployment_app_name)
    stale_error = ModalRemoteAppOutOfDateError(
        f"Modal app {deployment_app_name!r} runtime identity does not match the local client "
        f"(remote protocol={protocol_version!r}, local protocol={_REMOTE_APP_PROTOCOL_VERSION}, "
        f"remote fingerprint={remote_fingerprint!r}, local fingerprint={local_fingerprint!r})."
    )
    replacement_engine = _auto_deploy_modal_app(payload, stale_error)
    with _MODAL_AUTO_DEPLOY_LOCK:
        _MODAL_REMOTE_APP_VERSION_OK.add(runtime_cache_key)
    return replacement_engine

def _ensure_remote_engine_protocol_current(
    remote_engine: Any, payload: dict[str, Any]
) -> Any:
    """Return a compatible remote engine, replacing the deployed app when allowed."""
    settings = _settings_for_payload(payload)
    runtime_cache_key = _modal_runtime_cache_key(payload)
    with _MODAL_AUTO_DEPLOY_LOCK:
        runtime_is_known_current = runtime_cache_key in _MODAL_REMOTE_APP_VERSION_OK
    if runtime_is_known_current:
        logger.info(
            "Rebinding deployed RemoteEngine after cached protocol validation "
            "component=%s worker_affinity=%s.",
            payload.get("component_id"),
            _remote_worker_affinity_key(payload),
        )
        return _lookup_deployed_remote_engine(payload)
    version_payload = _remote_engine_runtime_version(remote_engine)
    if _is_runtime_version_payload_current(version_payload, payload):
        with _MODAL_AUTO_DEPLOY_LOCK:
            _MODAL_REMOTE_APP_VERSION_OK.add(runtime_cache_key)
        return _lookup_deployed_remote_engine(payload)
    if not settings.auto_deploy:
        raise ModalRemoteInvocationError(
            "Deployed Modal app runtime fingerprint is out of date and "
            "COMFY_MODAL_AUTO_DEPLOY=false prevents automatic replacement."
        )
    return _replace_outdated_modal_app(
        payload,
        remote_engine,
        version_payload=version_payload,
    )

def _lookup_protocol_current_remote_engine(payload: dict[str, Any]) -> Any:
    """Return an affinity-bound engine without allocating a cached protocol probe."""
    runtime_cache_key = _modal_runtime_cache_key(payload)
    with _MODAL_AUTO_DEPLOY_LOCK:
        runtime_is_known_current = runtime_cache_key in _MODAL_REMOTE_APP_VERSION_OK
    if runtime_is_known_current:
        logger.info(
            "Using cached Modal protocol validation without creating a parameterless "
            "probe component=%s worker_affinity=%s.",
            payload.get("component_id"),
            _remote_worker_affinity_key(payload),
        )
        return _lookup_deployed_remote_engine(payload)

    version_payload = _remote_runtime_version_from_cpu_stager(payload)
    if version_payload is not None:
        if _is_runtime_version_payload_current(version_payload, payload):
            with _MODAL_AUTO_DEPLOY_LOCK:
                _MODAL_REMOTE_APP_VERSION_OK.add(runtime_cache_key)
            logger.info(
                "Validated Modal runtime through CPU ModelStager without allocating "
                "a GPU protocol probe component=%s worker_affinity=%s.",
                payload.get("component_id"),
                _remote_worker_affinity_key(payload),
            )
            return _lookup_deployed_remote_engine(payload)
        settings = _settings_for_payload(payload)
        if not settings.auto_deploy:
            raise ModalRemoteInvocationError(
                "Deployed Modal app runtime fingerprint is out of date and "
                "COMFY_MODAL_AUTO_DEPLOY=false prevents automatic replacement."
            )
        return _replace_outdated_modal_app(
            payload,
            None,
            version_payload=version_payload,
        )

    logger.info(
        "CPU Modal runtime validation method is unavailable for component=%s; "
        "using the legacy parameterless GPU protocol probe.",
        payload.get("component_id"),
    )
    return _ensure_remote_engine_protocol_current(
        _lookup_deployed_remote_engine(payload, protocol_probe=True),
        payload,
    )

def _modal_auto_deploy_state(
    deploy_key: tuple[str, str | None],
) -> _ModalAutoDeployState:
    """Return the shared auto-deploy state bucket for one Modal app/environment."""
    with _MODAL_AUTO_DEPLOY_LOCK:
        state = _MODAL_AUTO_DEPLOY_STATES.get(deploy_key)
        if state is None:
            state = _ModalAutoDeployState()
            _MODAL_AUTO_DEPLOY_STATES[deploy_key] = state
        return state

def _lookup_deployed_remote_engine_with_retry(
    payload: dict[str, Any],
    *,
    timeout_seconds: float = 15.0,
    initial_delay_seconds: float = 0.25,
    max_delay_seconds: float = 2.0,
) -> Any:
    """Poll deployed Modal lookup until the freshly deployed app becomes discoverable."""
    lookup_error_types = _modal_lookup_error_types()
    if not lookup_error_types:
        return _lookup_deployed_remote_engine(payload)

    deadline = time.monotonic() + max(0.0, timeout_seconds)
    delay_seconds = max(0.0, initial_delay_seconds)
    last_error: BaseException | None = None
    while True:
        try:
            return _lookup_deployed_remote_engine(payload)
        except lookup_error_types as exc:
            last_error = exc
            if not _is_missing_modal_deployment_error(exc):
                raise
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0.0:
                raise
            time.sleep(min(delay_seconds, remaining_seconds))
            delay_seconds = min(
                max_delay_seconds, max(delay_seconds * 2.0, initial_delay_seconds)
            )
    if last_error is not None:
        raise last_error
    raise ModalRemoteInvocationError(
        "Deployed Modal lookup retry loop exited unexpectedly."
    )

def _auto_deploy_modal_app(payload: dict[str, Any], lookup_error: BaseException) -> Any:
    """Deploy the stable Modal cloud app once and wait until deployed lookup becomes ready."""
    if modal is None:
        raise ModalRemoteInvocationError("Modal SDK is unavailable.")

    settings = _settings_for_payload(payload)
    deployment_app_name = modal_deployment_app_name(settings)
    deploy_key = _modal_deploy_cache_key(payload)
    deploy_state = _modal_auto_deploy_state(deploy_key)
    with _modal_cloud_settings_override(settings):
        cloud_module = _load_modal_cloud_module()
    cloud_app = getattr(cloud_module, "app", None)
    if cloud_app is None:
        raise ModalRemoteInvocationError(
            "Stable Modal cloud entry module did not expose a deployable app."
        )

    while True:
        with deploy_state.condition:
            if (
                not _is_missing_modal_deployment_error(lookup_error)
                and deploy_state.ready
            ):
                logger.info(
                    "Auto-deploy already completed for app=%s env=%s; reusing cached deployment state.",
                    deployment_app_name,
                    deploy_key[1] or "<default>",
                )
                return _lookup_deployed_remote_engine(payload)
            if _is_missing_modal_deployment_error(lookup_error) and deploy_state.ready:
                logger.warning(
                    "Discarding stale auto-deploy ready state for app=%s env=%s after missing deployment lookup failure: %s",
                    deployment_app_name,
                    deploy_key[1] or "<default>",
                    lookup_error,
                )
                deploy_state.ready = False
            if deploy_state.deploy_in_progress:
                logger.info(
                    "Waiting for in-flight auto-deploy readiness for app=%s env=%s component=%s.",
                    deployment_app_name,
                    deploy_key[1] or "<default>",
                    payload.get("component_id"),
                )
                _emit_local_remote_startup_status(
                    payload,
                    phase="setup",
                    status_message="Waiting for Modal app rebuild",
                )
                deploy_state.condition.wait()
                if deploy_state.ready:
                    _emit_local_remote_dispatch_status(payload)
                    return _lookup_deployed_remote_engine(payload)
                if deploy_state.last_error is not None:
                    logger.warning(
                        "Retrying Modal auto-deploy for app=%s env=%s after previous readiness failure: %s",
                        deployment_app_name,
                        deploy_key[1] or "<default>",
                        deploy_state.last_error,
                    )
                continue
            deploy_state.deploy_in_progress = True
            deploy_state.last_error = None
            break

    _emit_local_remote_startup_status(
        payload,
        phase="setup",
        status_message="Rebuilding Modal app",
    )
    try:
        logger.warning(
            "Deployed Modal app lookup failed for app=%s component=%s: %s. "
            "Attempting first-run auto-deploy from the custom node.",
            deployment_app_name,
            payload.get("component_id"),
            lookup_error,
        )
        output_context = (
            modal.enable_output() if hasattr(modal, "enable_output") else nullcontext()
        )
        deploy_started_at = time.perf_counter()
        with output_context:
            cloud_app.deploy(
                name=deployment_app_name,
                environment_name=deploy_key[1],
            )
        logger.info(
            "Auto-deployed Modal app %s for env=%s in %.3fs; waiting for deployed lookup readiness.",
            deployment_app_name,
            deploy_key[1] or "<default>",
            time.perf_counter() - deploy_started_at,
        )
        _ensure_llm_profiles_staged(payload, deployment_app_name)
        remote_engine = _lookup_deployed_remote_engine_with_retry(payload)
        if not _is_remote_engine_runtime_current(remote_engine, payload):
            raise ModalRemoteInvocationError(
                f"Auto-deployed Modal app {deployment_app_name!r} did not report expected protocol "
                f"{_REMOTE_APP_PROTOCOL_VERSION} and fingerprint "
                f"{_expected_remote_runtime_fingerprint(payload)!r}."
            )
    except BaseException as exc:
        with deploy_state.condition:
            deploy_state.deploy_in_progress = False
            deploy_state.last_error = exc
            deploy_state.condition.notify_all()
        raise

    with deploy_state.condition:
        deploy_state.deploy_in_progress = False
        deploy_state.ready = True
        deploy_state.last_error = None
        deploy_state.condition.notify_all()
    with _MODAL_AUTO_DEPLOY_LOCK:
        _MODAL_REMOTE_APP_VERSION_OK.add(_modal_runtime_cache_key(payload))
    logger.info(
        "Deployed Modal app %s for env=%s is now lookup-ready.",
        deployment_app_name,
        deploy_key[1] or "<default>",
    )
    if _schedule_post_deploy_runtime_seed(payload):
        _await_matching_speculative_prewarm(payload, None)
    _emit_local_remote_dispatch_status(payload)
    return remote_engine
