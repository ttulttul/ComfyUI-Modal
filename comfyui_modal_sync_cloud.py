"""Stable Modal cloud entrypoint for ComfyUI Modal-Sync."""

import importlib
import importlib.metadata
import importlib.util
import logging
import os
import queue
import sys
import threading
import time
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterator, Mapping

_REPO_ROOT = Path(__file__).resolve().parent
_REMOTE_REPO_ROOT = Path("/root/comfyui_modal_sync_repo")
_LOCAL_COMFYUI_ROOT = (Path.home() / "git" / "ComfyUI").resolve()
_REMOTE_COMFYUI_ROOT = Path("/root/comfyui_src")
for candidate in (
    _REPO_ROOT,
    _REMOTE_REPO_ROOT,
    _LOCAL_COMFYUI_ROOT,
    _REMOTE_COMFYUI_ROOT,
):
    candidate_str = str(candidate)
    try:
        candidate_exists = candidate.exists()
    except PermissionError:
        candidate_exists = False
    if candidate_exists and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from runtime_environment import (  # noqa: E402 - paths are bootstrapped above.
    REMOTE_APP_PROTOCOL_VERSION as _REMOTE_APP_PROTOCOL_VERSION,
    REMOTE_PYTHON_VERSION,
    build_remote_runtime_identity,
    custom_node_runtime_packages as _custom_node_runtime_packages,
    remote_apt_packages as _comfyui_apt_packages,
    remote_huggingface_packages as _remote_huggingface_packages,
    remote_huggingface_validation_command as _remote_huggingface_validation_command,
    remote_runtime_packages as _comfyui_runtime_packages,
    select_remote_torch_build as _select_remote_torch_build,
)
from llm_recovery import (  # noqa: E402
    LLM_FORCE_VLLM_THROUGHPUT_PAYLOAD_KEY,
    is_llm_memory_recovery_exhausted,
)
from llm_staging import resolve_and_stage_model_references  # noqa: E402
from serialization import (  # noqa: E402 - paths are bootstrapped above.
    deserialize_node_inputs,
    serialize_node_outputs,
)
from session_state import (  # noqa: E402 - paths are bootstrapped above.
    RemoteSessionStateError,
)
from settings import (  # noqa: E402 - paths are bootstrapped above.
    DEFAULT_MODAL_SECRET_NAME,
    get_settings,
    modal_deployment_app_name,
)
try:  # noqa: E402 - support package and flat Modal-container imports.
    from .cloud_comfy_bootstrap import (
        CloudComfyBootstrapHooks,
        _clone_loader_cache_value,
        _ensure_comfy_runtime_initialized,
        _extract_custom_nodes_bundle,
        configure_cloud_comfy_bootstrap_hooks,
    )
except ImportError:  # pragma: no cover - exercised by flat cloud imports.
    from cloud_comfy_bootstrap import (
        CloudComfyBootstrapHooks,
        _clone_loader_cache_value,
        _ensure_comfy_runtime_initialized,
        _extract_custom_nodes_bundle,
        configure_cloud_comfy_bootstrap_hooks,
    )
try:  # noqa: E402 - support package and flat Modal-container imports.
    from .cloud_node_output_cache import (
        CloudNodeOutputCacheHooks,
        configure_cloud_node_output_cache_hooks,
    )
except ImportError:  # pragma: no cover - exercised by flat cloud imports.
    from cloud_node_output_cache import (
        CloudNodeOutputCacheHooks,
        configure_cloud_node_output_cache_hooks,
    )
try:  # noqa: E402 - support package and flat Modal-container imports.
    from .cloud_mapped_execution import (
        _execute_mapped_subgraph_payload,
    )
    from .cloud_prompt_validation import configure_cloud_prompt_validation_error
    from .cloud_prompt_execution import (
        CloudPromptExecutionHooks,
        _collapse_cache_slot,
        _execute_node_locally_raw,
        _execute_subgraph_prompt,
        _is_link,
        _normalize_prompt_input_value,
        _resolve_required_subgraph_nodes,
        configure_cloud_prompt_execution_hooks,
        execute_node_locally,
        execute_subgraph_locally,
    )
except ImportError:  # pragma: no cover - exercised by flat cloud imports.
    from cloud_mapped_execution import (
        _execute_mapped_subgraph_payload,
    )
    from cloud_prompt_validation import configure_cloud_prompt_validation_error
    from cloud_prompt_execution import (
        CloudPromptExecutionHooks,
        _collapse_cache_slot,
        _execute_node_locally_raw,
        _execute_subgraph_prompt,
        _is_link,
        _normalize_prompt_input_value,
        _resolve_required_subgraph_nodes,
        configure_cloud_prompt_execution_hooks,
        execute_node_locally,
        execute_subgraph_locally,
    )
try:  # noqa: E402 - support package and flat Modal-container imports.
    from .cloud_prompt_server_shims import (
        CloudPromptServerHooks,
        configure_cloud_prompt_server_hooks,
    )
except ImportError:  # pragma: no cover - exercised by flat cloud imports.
    from cloud_prompt_server_shims import (
        CloudPromptServerHooks,
        configure_cloud_prompt_server_hooks,
    )
try:  # noqa: E402 - support package and flat Modal-container imports.
    from .cloud_streaming import (
        CloudStreamingErrors,
        _stream_remote_payload_events as _stream_remote_payload_events_impl,
        configure_cloud_streaming_errors,
    )
except ImportError:  # pragma: no cover - exercised by flat cloud imports.
    from cloud_streaming import (
        CloudStreamingErrors,
        _stream_remote_payload_events as _stream_remote_payload_events_impl,
        configure_cloud_streaming_errors,
    )
try:  # noqa: E402 - support package and flat Modal-container imports.
    from .cloud_volume_reload import (
        CloudVolumeReloadHooks,
        _emit_modal_volume_reload_skip,
        _modal_volume_reload_marker,
        _reload_modal_volume_for_request,
        _should_reload_modal_volume,
        configure_cloud_volume_reload_hooks,
    )
except ImportError:  # pragma: no cover - exercised by flat cloud imports.
    from cloud_volume_reload import (
        CloudVolumeReloadHooks,
        _emit_modal_volume_reload_skip,
        _modal_volume_reload_marker,
        _reload_modal_volume_for_request,
        _should_reload_modal_volume,
        configure_cloud_volume_reload_hooks,
    )
try:  # noqa: E402 - support package and flat Modal-container imports.
    from .cloud_prewarm import (
        CloudPrewarmHooks,
        _commit_actual_llm_compile_cache,
        _llm_compile_miss_checkpoint,
        _prepare_warm_container_for_request,
        _prewarm_restored_runtime,
        _prewarm_snapshot_state,
        configure_cloud_prewarm_hooks,
    )
except ImportError:  # pragma: no cover - exercised by flat cloud imports.
    from cloud_prewarm import (
        CloudPrewarmHooks,
        _commit_actual_llm_compile_cache,
        _llm_compile_miss_checkpoint,
        _prepare_warm_container_for_request,
        _prewarm_restored_runtime,
        _prewarm_snapshot_state,
        configure_cloud_prewarm_hooks,
    )
try:  # noqa: E402 - support package and flat Modal-container imports.
    from .cloud_session_bridge import (
        CloudSessionBridgeHooks,
        configure_cloud_session_bridge_hooks,
    )
except ImportError:  # pragma: no cover - exercised by flat cloud imports.
    from cloud_session_bridge import (
        CloudSessionBridgeHooks,
        configure_cloud_session_bridge_hooks,
    )
try:  # noqa: E402 - support package and flat Modal-container imports.
    from .cloud_durable_invocation import (
        DurableInvocationErrors,
        _execute_canary_payload,
        _execute_with_durable_invocation,
        configure_durable_invocation_errors,
    )
except ImportError:  # pragma: no cover - exercised by flat cloud imports.
    from cloud_durable_invocation import (
        DurableInvocationErrors,
        _execute_canary_payload,
        _execute_with_durable_invocation,
        configure_durable_invocation_errors,
    )
try:  # noqa: E402 - support package and flat Modal-container imports.
    from .cloud_image_env import (
        _install_custom_node_packages,
        _install_remote_accelerator_packages,
        _install_remote_torch_build,
        _modal_image_environment,
        _modal_secret_from_settings,
        _model_stager_image_environment,
        _remote_engine_cls_options,
        _should_ignore_comfyui_path,
        _should_ignore_repo_path,
    )
except ImportError:  # pragma: no cover - exercised by flat cloud imports.
    from cloud_image_env import (
        _install_custom_node_packages,
        _install_remote_accelerator_packages,
        _install_remote_torch_build,
        _modal_image_environment,
        _modal_secret_from_settings,
        _model_stager_image_environment,
        _remote_engine_cls_options,
        _should_ignore_comfyui_path,
        _should_ignore_repo_path,
    )
try:  # noqa: E402 - support package and flat Modal-container imports.
    from .cloud_app_guard import guard_against_existing_modal_app
except ImportError:  # pragma: no cover - exercised by flat cloud imports.
    from cloud_app_guard import guard_against_existing_modal_app
try:  # noqa: E402 - support package and flat Modal-container imports.
    from .cloud_runtime_context import (
        register_cloud_runtime_stores,
    )
    from .cloud_execution_control import (
        _registered_remote_execution,
    )
    from .cloud_runtime_logging import (
        _emit_cloud_info,
        _is_modal_container_runtime,
        _timed_phase,
        configure_cloud_runtime_logging,
    )
except ImportError:  # pragma: no cover - exercised by flat cloud imports.
    from cloud_runtime_context import (
        register_cloud_runtime_stores,
    )
    from cloud_execution_control import (
        _registered_remote_execution,
    )
    from cloud_runtime_logging import (
        _emit_cloud_info,
        _is_modal_container_runtime,
        _timed_phase,
        configure_cloud_runtime_logging,
    )


_COMPATIBILITY_EXPORT_MODULES: tuple[ModuleType, ...] = tuple(
    importlib.import_module(module_name)
    for module_name in (
        "runtime_environment",
        "durable_state",
        "output_artifacts",
        "remote_protocol",
        "serialization",
        "session_state",
        "cloud_comfy_bootstrap",
        "cloud_node_output_cache",
        "cloud_mapped_execution",
        "cloud_prompt_validation",
        "cloud_prompt_execution",
        "cloud_prompt_server_shims",
        "cloud_streaming",
        "cloud_volume_reload",
        "cloud_prewarm",
        "cloud_session_bridge",
        "cloud_durable_invocation",
        "cloud_image_env",
        "cloud_app_guard",
        "cloud_runtime_context",
        "cloud_execution_control",
        "cloud_runtime_logging",
    )
)


def __getattr__(name: str) -> Any:
    """Read legacy private exports from the focused module that now owns them."""
    for compatibility_module in _COMPATIBILITY_EXPORT_MODULES:
        try:
            return getattr(compatibility_module, name)
        except AttributeError:
            continue
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


logger = logging.getLogger(__name__)

# Cloud app and ComfyUI bootstrap state.
_CLOUD_HANDLER_NAME = "comfyui-modal-sync-cloud-timestamped"
configure_cloud_runtime_logging(logger, _CLOUD_HANDLER_NAME)

# Poisoned-container retirement state.
_CONTAINER_TERMINATION_LOCK = threading.Lock()
_REMOTE_ERROR_CONTAINER_EXIT_DELAY_SECONDS = 1.0
_CONTAINER_TERMINATION_SCHEDULED = False
try:
    import modal  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - remote entrypoint only.
    modal = None


class RemoteSubgraphExecutionError(RuntimeError):
    """Raised when remote subgraph execution fails."""


class RemoteInvocationInProgressError(RuntimeError):
    """Raised when an idempotent invocation is already running remotely."""


class RemoteInvocationAbandonedError(RuntimeError):
    """Raised when a streamed invocation loses its consumer before completion."""


class RemoteCanaryInterruptedError(RuntimeError):
    """Raised when a live remote canary observes its shared interrupt flag."""


class RemoteCanaryBarrierTimeoutError(TimeoutError):
    """Raised when live canary calls fail to overlap before their deadline."""


class ExistingModalAppError(RuntimeError):
    """Raised when deploying would overwrite an existing Modal app."""


configure_durable_invocation_errors(
    DurableInvocationErrors(
        invocation_in_progress=RemoteInvocationInProgressError,
        canary_interrupted=RemoteCanaryInterruptedError,
        canary_barrier_timeout=RemoteCanaryBarrierTimeoutError,
    )
)

configure_cloud_streaming_errors(
    CloudStreamingErrors(invocation_abandoned=RemoteInvocationAbandonedError)
)


class RemoteFailureDisposition(str, Enum):
    """Describe whether one remote failure implies poisoned worker state."""

    EXPECTED = "expected"
    DETERMINISTIC = "deterministic"
    POISONED_WORKER = "poisoned-worker"


def _guard_against_existing_modal_app(settings: Any, modal_module: Any) -> None:
    """Fail local Modal app construction when the configured app already exists."""
    guard_against_existing_modal_app(
        settings,
        modal_module,
        error_type=ExistingModalAppError,
    )


def _meaningful_progress_values(
    node_state: dict[str, Any]
) -> tuple[float, float] | None:
    """Return numeric progress values only for node states that represent real progress."""
    try:
        progress_value = float(node_state.get("value", 0.0))
        max_value = float(node_state.get("max", 1.0))
    except (TypeError, ValueError):
        return None

    if max_value <= 1.0:
        return None
    return progress_value, max_value


def _schedule_process_exit(delay_seconds: float, exit_code: int) -> None:
    """Exit the current process after a short delay to retire a bad Modal container."""

    def exit_later() -> None:
        """Sleep briefly so Modal can ship the error response before exiting the worker."""
        if delay_seconds > 0:
            time.sleep(delay_seconds)
        logger.error(
            "Exiting Modal container process with code=%s after remote failure.",
            exit_code,
        )
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)

    threading.Thread(
        target=exit_later,
        name="modal-container-exit",
        daemon=True,
    ).start()


def _schedule_process_exit_unless_cancelled(
    *,
    delay_seconds: float,
    exit_code: int,
    cancel_event: threading.Event,
    reason: str,
) -> None:
    """Exit the current process after a delay unless the caller cancels first."""

    def exit_later() -> None:
        """Wait for cancellation or exit the worker if the delay expires."""
        if delay_seconds > 0 and cancel_event.wait(timeout=delay_seconds):
            logger.debug("Cancelled delayed Modal container restart for %s.", reason)
            return
        logger.error(
            "Exiting Modal container process with code=%s after %s.", exit_code, reason
        )
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)

    threading.Thread(
        target=exit_later,
        name="modal-container-cancel-restart",
        daemon=True,
    ).start()


def _schedule_remote_cancel_restart(
    *,
    component_id: str,
    completion_event: threading.Event,
) -> bool:
    """Restart the Modal worker if a cancelled remote prompt keeps executing."""
    if not _is_modal_container_runtime():
        return False

    delay_seconds = max(0.0, get_settings().remote_cancel_restart_seconds)
    logger.warning(
        "Remote cancellation requested for component=%s; scheduling container restart in %.3fs unless execution stops first.",
        component_id,
        delay_seconds,
    )
    _schedule_process_exit_unless_cancelled(
        delay_seconds=delay_seconds,
        exit_code=0,
        cancel_event=completion_event,
        reason=f"remote cancellation timeout for component={component_id}",
    )
    return True


def _is_interrupt_like_failure(exc: Exception) -> bool:
    """Return whether one remote failure represents an expected interruption rather than a crash."""
    return "interrupt" in str(exc).lower()


def _is_session_state_like_failure(exc: Exception) -> bool:
    """Return whether one remote failure came from prompt-scoped session routing/state issues."""
    if isinstance(exc, RemoteSessionStateError):
        return True
    return "remote session" in str(exc).lower()


def _remote_failure_disposition(exc: Exception) -> RemoteFailureDisposition:
    """Classify one execution failure for worker-retirement decisions."""
    if _is_interrupt_like_failure(exc) or _is_session_state_like_failure(exc):
        return RemoteFailureDisposition.EXPECTED
    if isinstance(exc, MemoryError) or is_llm_memory_recovery_exhausted(exc):
        return RemoteFailureDisposition.POISONED_WORKER

    message = str(exc).lower()
    poisoned_runtime_markers = (
        "cuda out of memory",
        "cuda error",
        "device-side assert",
        "illegal memory access",
        "cublas_status",
        "cudnn_status",
        "hip error",
    )
    if any(marker in message for marker in poisoned_runtime_markers):
        return RemoteFailureDisposition.POISONED_WORKER
    return RemoteFailureDisposition.DETERMINISTIC


def _maybe_schedule_container_termination_on_error(
    payload: dict[str, Any],
    exc: Exception,
) -> bool:
    """Retire the current Modal container after a remote execution crash when configured."""
    if not _is_modal_container_runtime():
        return False
    if not bool(payload.get("terminate_container_on_error", True)):
        return False
    disposition = _remote_failure_disposition(exc)
    if disposition is not RemoteFailureDisposition.POISONED_WORKER:
        logger.warning(
            "Skipping Modal container termination for component=%s failure_disposition=%s because the worker is safe to reuse.",
            payload.get("component_id"),
            disposition.value,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return False

    global _CONTAINER_TERMINATION_SCHEDULED
    with _CONTAINER_TERMINATION_LOCK:
        if _CONTAINER_TERMINATION_SCHEDULED:
            return False
        _CONTAINER_TERMINATION_SCHEDULED = True

    logger.error(
        "Scheduling Modal container termination after remote execution failure for component=%s.",
        payload.get("component_id"),
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    _schedule_process_exit(_REMOTE_ERROR_CONTAINER_EXIT_DELAY_SECONDS, 1)
    return True




def _observe_remote_workflow_for_llm_mode(payload: dict[str, Any]) -> None:
    """Record real workflow arrivals for container-local vLLM auto promotion."""
    if payload.get("payload_kind") == "canary":
        return
    from modal_llm_runtime import (
        force_modal_vllm_throughput_after_memory_recovery,
        observe_modal_workflow_execution,
    )

    prompt_id = payload.get("prompt_id")
    normalized_prompt_id = str(prompt_id).strip() if prompt_id is not None else None
    if bool(payload.get(LLM_FORCE_VLLM_THROUGHPUT_PAYLOAD_KEY)):
        force_modal_vllm_throughput_after_memory_recovery(normalized_prompt_id)
        return
    observe_modal_workflow_execution(normalized_prompt_id)




def _stream_remote_payload_events(
    payload: dict[str, Any],
    kwargs_payload: bytes | bytearray | str | dict[str, Any],
    cancellation_event: threading.Event | None = None,
    interrupt_store: Any | None = None,
    interrupt_flag_key: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Delegate remote event streaming through the stable cloud entrypoint name."""
    yield from _stream_remote_payload_events_impl(
        payload,
        kwargs_payload,
        cancellation_event=cancellation_event,
        interrupt_store=interrupt_store,
        interrupt_flag_key=interrupt_flag_key,
    )


configure_cloud_session_bridge_hooks(
    CloudSessionBridgeHooks(
        clone_loader_cache_value=_clone_loader_cache_value,
        emit_cloud_info=_emit_cloud_info,
        execute_node_locally_raw=_execute_node_locally_raw,
        execute_subgraph_prompt=_execute_subgraph_prompt,
        is_link=_is_link,
        normalize_prompt_input_value=_normalize_prompt_input_value,
        resolve_required_subgraph_nodes=_resolve_required_subgraph_nodes,
    )
)

configure_cloud_comfy_bootstrap_hooks(
    CloudComfyBootstrapHooks(
        emit_cloud_info=_emit_cloud_info,
        timed_phase=_timed_phase,
        remote_subgraph_error=RemoteSubgraphExecutionError,
    )
)

configure_cloud_node_output_cache_hooks(
    CloudNodeOutputCacheHooks(
        emit_cloud_info=_emit_cloud_info,
        timed_phase=_timed_phase,
    )
)

configure_cloud_volume_reload_hooks(
    CloudVolumeReloadHooks(
        emit_cloud_info=_emit_cloud_info,
        timed_phase=_timed_phase,
    )
)

configure_cloud_prewarm_hooks(
    CloudPrewarmHooks(
        emit_cloud_info=_emit_cloud_info,
        timed_phase=_timed_phase,
    )
)

configure_cloud_prompt_execution_hooks(
    CloudPromptExecutionHooks(
        emit_cloud_info=_emit_cloud_info,
        timed_phase=_timed_phase,
        schedule_remote_cancel_restart=_schedule_remote_cancel_restart,
        remote_subgraph_error=RemoteSubgraphExecutionError,
    )
)
configure_cloud_prompt_validation_error(RemoteSubgraphExecutionError)

configure_cloud_prompt_server_hooks(
    CloudPromptServerHooks(
        collapse_cache_slot=_collapse_cache_slot,
        emit_cloud_info=_emit_cloud_info,
        meaningful_progress_values=_meaningful_progress_values,
    )
)


if modal is not None:  # pragma: no branch - remote entrypoint configuration.
    settings = globals().get("__comfy_modal_settings_override__") or get_settings()
    __comfy_modal_gpu__ = settings.modal_gpu
    __comfy_modal_app_name__ = modal_deployment_app_name(settings)
    __comfy_modal_secret_name__ = str(
        getattr(settings, "modal_secret_name", DEFAULT_MODAL_SECRET_NAME)
    ).strip()
    _guard_against_existing_modal_app(settings, modal)
    app = modal.App(__comfy_modal_app_name__)
    modal_secret = _modal_secret_from_settings(settings, modal)
    vol = modal.Volume.from_name(settings.volume_name, create_if_missing=True)
    llm_compile_cache_vol = modal.Volume.from_name(
        getattr(
            settings,
            "llm_compile_cache_volume_name",
            f"{settings.volume_name}-llm-compile-cache",
        ),
        create_if_missing=True,
    )
    interrupt_flags = modal.Dict.from_name(
        settings.interrupt_dict_name,
        create_if_missing=True,
    )
    node_output_cache = modal.Dict.from_name(
        settings.node_output_cache_dict_name,
        create_if_missing=True,
    )
    session_bridge_cache = modal.Dict.from_name(
        settings.session_bridge_dict_name,
        create_if_missing=True,
    )
    invocation_records = modal.Dict.from_name(
        settings.invocation_dict_name,
        create_if_missing=True,
    )
    snapshot_profiles = modal.Dict.from_name(
        settings.snapshot_profile_dict_name,
        create_if_missing=True,
    )
    register_cloud_runtime_stores(
        session_bridge_cache=session_bridge_cache,
        invocation_records=invocation_records,
        volume=vol,
        snapshot_profiles=snapshot_profiles,
        node_output_cache=node_output_cache,
        interrupt_flags=interrupt_flags,
    )
    custom_node_packages = _custom_node_runtime_packages(settings.custom_nodes_dir)
    torch_build = _select_remote_torch_build(settings.modal_gpu)
    runtime_identity = build_remote_runtime_identity(
        repo_root=_REPO_ROOT,
        comfyui_root=settings.comfyui_root,
        custom_nodes_dir=settings.custom_nodes_dir,
        settings=settings,
    )
    logger.info(
        "Building Modal runtime fingerprint=%s protocol=%d python=%s.",
        runtime_identity.fingerprint,
        _REMOTE_APP_PROTOCOL_VERSION,
        REMOTE_PYTHON_VERSION,
    )
    logger.info(
        "Selected Modal PyTorch build gpu=%s cuda=%s install_layers=%s.",
        settings.modal_gpu,
        torch_build.cuda_version,
        torch_build.install_layers,
    )
    image = (
        modal.Image.debian_slim(python_version=REMOTE_PYTHON_VERSION)
        .apt_install(*_comfyui_apt_packages())
        .pip_install(*_comfyui_runtime_packages())
    )
    image = _install_custom_node_packages(image, custom_node_packages)
    image = _install_remote_torch_build(image, torch_build)
    image = _install_remote_accelerator_packages(image, settings.modal_gpu)
    image = image.env(
        _modal_image_environment(settings, runtime_identity.fingerprint)
    )
    image = image.add_local_dir(
        _REPO_ROOT,
        remote_path="/root/comfyui_modal_sync_repo",
        ignore=_should_ignore_repo_path,
    )
    if settings.comfyui_root is not None and settings.comfyui_root.exists():
        image = image.add_local_dir(
            settings.comfyui_root,
            remote_path=str(_REMOTE_COMFYUI_ROOT),
            ignore=_should_ignore_comfyui_path,
        )
        logger.info(
            "Including local ComfyUI checkout %s in Modal image at %s.",
            settings.comfyui_root,
            _REMOTE_COMFYUI_ROOT,
        )
    else:
        logger.warning(
            "No local ComfyUI checkout was discovered; remote Modal execution may fail to import ComfyUI core modules."
        )

    stager_image = (
        modal.Image.debian_slim(python_version=REMOTE_PYTHON_VERSION)
        .env(
            _model_stager_image_environment(
                settings,
                runtime_identity.fingerprint,
            )
        )
        .pip_install(*_remote_huggingface_packages())
        .run_commands(_remote_huggingface_validation_command())
        .add_local_dir(
            _REPO_ROOT,
            remote_path="/root/comfyui_modal_sync_repo",
            ignore=_should_ignore_repo_path,
        )
    )

    @app.cls(
        image=stager_image,
        volumes={settings.remote_storage_root: vol},
        secrets=[modal_secret],
        cpu=4.0,
        memory=16384,
        max_containers=1,
        scaledown_window=300,
        timeout=7200,
    )
    @modal.concurrent(max_inputs=1)
    class ModelStager:
        """Resolve and stage pinned Hugging Face snapshots without consuming GPU time."""

        def _stage_profiles(
            self,
            model_references: list[str],
            resolved_profiles: Mapping[str, Any] | None = None,
            progress_callback: Callable[[dict[str, Any]], None] | None = None,
        ) -> list[dict[str, Any]]:
            """Resolve and stage profiles while optionally publishing progress."""
            staged_profiles = resolve_and_stage_model_references(
                model_references,
                settings.remote_storage_root,
                resolved_profiles=resolved_profiles,
                owner_id=f"modal:{os.getpid()}:{time.time_ns()}",
                progress_callback=(
                    lambda progress: progress_callback(
                        {
                            "stage": progress.stage,
                            "message": progress.message,
                            "value": progress.value,
                            "max": progress.maximum,
                            "unit": progress.unit,
                            "indeterminate": progress.indeterminate,
                            "model_reference": progress.model_reference,
                        }
                    )
                    if progress_callback is not None
                    else None
                ),
            )
            results = [profile.to_dict() for profile in staged_profiles]
            if any(
                result["downloaded"]
                or result["manifest_created"]
                for result in results
            ):
                vol.commit()
            else:
                logger.info(
                    "Skipping Modal Volume commit because all requested LLM "
                    "profiles and weights were already durable."
                )
            logger.info(
                "Modal LLM CPU resolution and staging completed for models=%s.",
                model_references,
            )
            return results

        @modal.method()
        def stage_profiles(
            self,
            model_references: list[str],
            resolved_profiles: Mapping[str, Any] | None = None,
        ) -> list[dict[str, Any]]:
            """Resolve model references, stage snapshots, and return metadata."""
            return self._stage_profiles(model_references, resolved_profiles)

        @modal.method()
        def stage_profiles_stream(
            self,
            model_references: list[str],
            resolved_profiles: Mapping[str, Any] | None = None,
        ) -> Iterator[dict[str, Any]]:
            """Stream CPU staging progress and finish with resolved profile data."""
            progress_events: queue.Queue[dict[str, Any]] = queue.Queue()
            results: list[list[dict[str, Any]]] = []
            errors: list[Exception] = []

            def run_staging() -> None:
                """Run blocking Hugging Face work while the generator yields events."""
                try:
                    results.append(
                        self._stage_profiles(
                            model_references,
                            resolved_profiles,
                            progress_events.put,
                        )
                    )
                except Exception as error:
                    errors.append(error)

            staging_thread = threading.Thread(
                target=run_staging,
                name="modal-llm-model-stager",
                daemon=True,
            )
            staging_thread.start()
            while staging_thread.is_alive() or not progress_events.empty():
                try:
                    progress = progress_events.get(timeout=0.25)
                except queue.Empty:
                    continue
                yield {"kind": "progress", **progress}
            staging_thread.join()
            if errors:
                raise errors[0]
            yield {"kind": "result", "results": results[0] if results else []}

        @modal.method()
        def runtime_version(self) -> dict[str, Any]:
            """Return deployment identity without allocating a GPU container."""
            return {
                "protocol_version": _REMOTE_APP_PROTOCOL_VERSION,
                "app_name": __comfy_modal_app_name__,
                "runtime_fingerprint": os.environ.get(
                    "COMFY_MODAL_RUNTIME_FINGERPRINT",
                    "",
                ),
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            }

    @app.cls(
        **_remote_engine_cls_options(
            settings,
            vol,
            image,
            modal_secret,
            llm_compile_cache_vol,
        )
    )
    @modal.concurrent(max_inputs=1)
    class RemoteEngine:
        """Modal runtime class that executes proxied ComfyUI payloads."""

        snapshot_profile_key: str = modal.parameter(default="")
        gpu_snapshot_enabled: bool = modal.parameter(default=False)
        worker_affinity_key: str = modal.parameter(default="worker-pool:slot:0")

        @modal.enter(snap=True)
        def setup_snapshot_state(self) -> None:
            """Prepare snapshot-friendly runtime state before Modal captures memory."""
            with _timed_phase("remote_engine_setup_snapshot"):
                _prewarm_snapshot_state(
                    gpu_snapshot_enabled=bool(self.gpu_snapshot_enabled),
                    snapshot_profile_key=self.snapshot_profile_key,
                )
                logger.info(
                    "RemoteEngine snapshot setup complete for snapshot_profile_key=%s gpu_snapshot_enabled=%s worker_affinity=%s.",
                    self.snapshot_profile_key or None,
                    bool(self.gpu_snapshot_enabled),
                    self.worker_affinity_key,
                )

        @modal.enter(snap=False)
        def setup_restored_runtime(self) -> None:
            """Prepare request-serving runtime state after a fresh boot or snapshot restore."""
            with _timed_phase("remote_engine_setup_restored"):
                _prewarm_restored_runtime(llm_compile_cache_vol)
                logger.info(
                    "RemoteEngine restored-runtime setup complete for snapshot_profile_key=%s.",
                    self.snapshot_profile_key or None,
                )

        @modal.method()
        def execute_payload(
            self, payload: dict[str, Any], kwargs_payload: bytes
        ) -> bytes:
            """Execute a proxied node or subgraph inside the Modal container."""
            _observe_remote_workflow_for_llm_mode(payload)
            component_id = payload.get("component_id", "single-node")
            reload_marker = _modal_volume_reload_marker(payload)
            try:
                with _registered_remote_execution(payload) as execution_control:
                    with _timed_phase(
                        "remote_engine_execute_payload",
                        component=component_id,
                        payload_kind=payload.get("payload_kind"),
                    ):
                        _hydrate_missing_payload_volume_paths(vol, payload)
                        if _should_reload_modal_volume(payload):
                            _reload_modal_volume_for_request(
                                vol,
                                str(component_id),
                                reload_marker=reload_marker,
                                payload=payload,
                            )
                        else:
                            _emit_modal_volume_reload_skip(component_id, payload)

                        def execute_once() -> bytes:
                            """Execute the underlying payload once inside this request context."""
                            if payload.get("payload_kind") == "canary":
                                return _execute_canary_payload(
                                    payload,
                                    kwargs_payload,
                                    cancellation_event=execution_control.cancellation_event,
                                    interrupt_store=interrupt_flags,
                                    interrupt_flag_key=execution_control.interrupt_flag_key,
                                )
                            if payload.get("payload_kind") == "mapped_subgraph":
                                custom_nodes_root = _extract_custom_nodes_bundle(
                                    payload.get("custom_nodes_bundle")
                                )
                                _ensure_comfy_runtime_initialized(custom_nodes_root)
                                hydrated_inputs = deserialize_node_inputs(
                                    kwargs_payload
                                )
                                return serialize_node_outputs(
                                    _execute_mapped_subgraph_payload(
                                        payload,
                                        hydrated_inputs,
                                        custom_nodes_root,
                                        cancellation_event=execution_control.cancellation_event,
                                        interrupt_store=interrupt_flags,
                                        interrupt_flag_key=execution_control.interrupt_flag_key,
                                    )
                                )
                            if payload.get("payload_kind") == "subgraph":
                                return execute_subgraph_locally(
                                    payload,
                                    kwargs_payload,
                                    cancellation_event=execution_control.cancellation_event,
                                    interrupt_store=interrupt_flags,
                                    interrupt_flag_key=execution_control.interrupt_flag_key,
                                )
                            return execute_node_locally(
                                payload,
                                kwargs_payload,
                                cancellation_event=execution_control.cancellation_event,
                                interrupt_store=interrupt_flags,
                                interrupt_flag_key=execution_control.interrupt_flag_key,
                            )

                        compile_miss_checkpoint = _llm_compile_miss_checkpoint(payload)
                        result = _execute_with_durable_invocation(
                            payload, execute_once
                        )
                        _commit_actual_llm_compile_cache(
                            compile_miss_checkpoint,
                            llm_compile_cache_vol,
                        )
                        return result
            except Exception as exc:
                _maybe_schedule_container_termination_on_error(payload, exc)
                raise

        @modal.method()
        def warmup_for_request(self, payload: dict[str, Any]) -> dict[str, Any]:
            """Prime the current or a newly started Modal container for one prompt."""
            return _prepare_warm_container_for_request(
                vol,
                payload,
                llm_compile_cache_vol,
            )

        @modal.method()
        def keepalive_for_local_gap(self, payload: dict[str, Any]) -> dict[str, Any]:
            """Keep this affinity slot active while the workflow executes locally."""
            logger.info(
                "Remote local-gap keepalive prompt=%s component=%s worker_affinity=%s.",
                payload.get("prompt_id"),
                payload.get("component_id"),
                self.worker_affinity_key,
            )
            return {
                "component_id": str(payload.get("component_id") or "modal-keepalive"),
                "task_id": os.getenv("MODAL_TASK_ID"),
                "worker_affinity_key": self.worker_affinity_key,
            }

        @modal.method()
        def runtime_version(self) -> dict[str, Any]:
            """Return the deployed runtime identity expected by the local client."""
            return {
                "protocol_version": _REMOTE_APP_PROTOCOL_VERSION,
                "app_name": __comfy_modal_app_name__,
                "runtime_fingerprint": os.environ.get(
                    "COMFY_MODAL_RUNTIME_FINGERPRINT",
                    "",
                ),
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
                "vllm_version": importlib.metadata.version("vllm"),
            }

        @modal.method()
        def execute_payload_stream(
            self,
            payload: dict[str, Any],
            kwargs_payload: bytes,
        ) -> Iterator[dict[str, Any]]:
            """Stream progress envelopes and a final serialized result for one payload."""
            _observe_remote_workflow_for_llm_mode(payload)
            component_id = payload.get("component_id", "single-node")
            reload_marker = _modal_volume_reload_marker(payload)
            try:
                with _registered_remote_execution(payload) as execution_control:
                    with _timed_phase(
                        "remote_engine_execute_payload",
                        component=component_id,
                        payload_kind=payload.get("payload_kind"),
                    ):
                        _hydrate_missing_payload_volume_paths(vol, payload)
                        if _should_reload_modal_volume(payload):
                            _reload_modal_volume_for_request(
                                vol,
                                str(component_id),
                                reload_marker=reload_marker,
                                payload=payload,
                            )
                        else:
                            _emit_modal_volume_reload_skip(component_id, payload)
                        compile_miss_checkpoint = _llm_compile_miss_checkpoint(payload)
                        yield from _stream_remote_payload_events(
                            payload,
                            kwargs_payload,
                            cancellation_event=execution_control.cancellation_event,
                            interrupt_store=interrupt_flags,
                            interrupt_flag_key=execution_control.interrupt_flag_key,
                        )
                        _commit_actual_llm_compile_cache(
                            compile_miss_checkpoint,
                            llm_compile_cache_vol,
                        )
            except Exception as exc:
                _maybe_schedule_container_termination_on_error(payload, exc)
                raise

else:
    app = None
    RemoteEngine = None
