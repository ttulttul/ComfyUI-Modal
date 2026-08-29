"""vLLM execution policy and process-wide Triton compile telemetry."""

from __future__ import annotations

from dataclasses import replace
import importlib
import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Mapping

if __package__:
    from .llm_profiles import LLMModelProfile
else:  # pragma: no cover - remote node bundles may import top-level modules.
    from llm_profiles import LLMModelProfile

logger = logging.getLogger(__name__)

_VLLM_EXECUTION_SETTINGS = frozenset({"auto", "eager", "throughput"})
_VLLM_EFFECTIVE_MODES = frozenset({"eager", "throughput"})
_VLLM_RUNTIME_MODE_OPTION = "_runtime_vllm_execution_mode"
_VLLM_RUNTIME_SETTING_OPTION = "_runtime_vllm_execution_setting"
_TRITON_COMPILE_MISS_SIGNAL_PATH = Path(
    os.getenv(
        "COMFY_MODAL_TRITON_COMPILE_MISS_SIGNAL_PATH",
        "/tmp/comfy-modal-triton-compile-misses.jsonl",
    )
)
_TRITON_COMPILE_LISTENER_STATUS_PATH = Path(
    os.getenv(
        "COMFY_MODAL_TRITON_COMPILE_LISTENER_STATUS_PATH",
        "/tmp/comfy-modal-triton-listeners.jsonl",
    )
)
_TRITON_COMPILE_LISTENER_LOCK = threading.Lock()
_TRITON_COMPILE_LISTENER_INSTALLED_PID: int | None = None
_TRITON_ENGINE_CORE_LISTENER_RECORDED_PID: int | None = None
_VLLM_ENGINE_CORE_ENTRYPOINT_PATCHED_PID: int | None = None
_VLLM_ENGINE_CORE_ORIGINAL_ENTRYPOINT: Callable[..., Any] | None = None

def triton_compile_miss_signal_size() -> int:
    """Return the process-shared signal offset for genuine Triton cache misses."""
    try:
        return _TRITON_COMPILE_MISS_SIGNAL_PATH.stat().st_size
    except FileNotFoundError:
        return 0


def _append_process_signal(path: Path, event: Mapping[str, Any]) -> None:
    """Atomically append one JSON event shared by worker processes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(dict(event), sort_keys=True, default=str).encode("utf-8") + b"\n"
    )
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        written_bytes = os.write(descriptor, payload)
        if written_bytes != len(payload):
            raise OSError(
                f"Short write for process signal {path}: "
                f"expected={len(payload)} actual={written_bytes}."
            )
    finally:
        os.close(descriptor)


def _process_is_alive(pid: int) -> bool:
    """Return whether a process id still identifies a visible live process."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def triton_compile_listener_engine_pids() -> tuple[int, ...]:
    """Return live EngineCore processes that installed the accurate listener."""
    try:
        records = _TRITON_COMPILE_LISTENER_STATUS_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
    except FileNotFoundError:
        return ()
    active_pids: set[int] = set()
    for record in records:
        try:
            event = json.loads(record)
            pid = int(event.get("pid"))
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if event.get("role") == "engine_core" and _process_is_alive(pid):
            active_pids.add(pid)
    return tuple(sorted(active_pids))


def _triton_compile_timing_payload(times: Any) -> dict[str, Any]:
    """Return JSON-safe millisecond timings from Triton's compile listener."""
    lowering_stages = [
        [str(stage_name), int(duration_microseconds) / 1000.0]
        for stage_name, duration_microseconds in getattr(times, "lowering_stages", ())
    ]
    return {
        "ir_initialization_ms": int(getattr(times, "ir_initialization", 0)) / 1000.0,
        "lowering_stages_ms": lowering_stages,
        "store_results_ms": int(getattr(times, "store_results", 0)) / 1000.0,
        "total_ms": int(getattr(times, "total", 0)) / 1000.0,
    }


def _record_triton_compile_miss(event: Mapping[str, Any]) -> None:
    """Append one genuine compile miss for observation by the Modal parent process."""
    _append_process_signal(_TRITON_COMPILE_MISS_SIGNAL_PATH, event)


def _triton_compile_listener(**event: Any) -> None:
    """Log persistent-cache outcomes and signal only genuine Triton compilations."""
    source = event.get("src")
    metadata = event.get("metadata")
    times = event.get("times")
    cache_hit = bool(event.get("cache_hit"))
    kernel_name = str(getattr(source, "name", "<unknown>"))
    artifact_hash = (
        str(metadata.get("hash") or "") if isinstance(metadata, Mapping) else ""
    )
    timing = _triton_compile_timing_payload(times)
    log_method = logger.info if cache_hit else logger.warning
    log_method(
        "Triton persistent compile cache kernel=%s cache_hit=%s "
        "artifact_hash=%s total_ms=%.3f ir_ms=%.3f lowering_ms=%.3f "
        "store_ms=%.3f stages=%s.",
        kernel_name,
        cache_hit,
        artifact_hash or None,
        timing["total_ms"],
        timing["ir_initialization_ms"],
        sum(stage[1] for stage in timing["lowering_stages_ms"]),
        timing["store_results_ms"],
        timing["lowering_stages_ms"],
    )
    if cache_hit:
        return
    _record_triton_compile_miss(
        {
            "artifact_hash": artifact_hash,
            "cache_hit": False,
            "kernel": kernel_name,
            "pid": os.getpid(),
            "recorded_at": time.time(),
            "timing": timing,
        }
    )


def _record_engine_core_listener_installation() -> None:
    """Publish that this live EngineCore installed cache-aware telemetry."""
    global _TRITON_ENGINE_CORE_LISTENER_RECORDED_PID
    current_pid = os.getpid()
    if _TRITON_ENGINE_CORE_LISTENER_RECORDED_PID == current_pid:
        return
    _append_process_signal(
        _TRITON_COMPILE_LISTENER_STATUS_PATH,
        {
            "installed_at": time.time(),
            "pid": current_pid,
            "ppid": os.getppid(),
            "role": "engine_core",
        },
    )
    _TRITON_ENGINE_CORE_LISTENER_RECORDED_PID = current_pid


def _install_triton_compile_listener_in_current_process(*, engine_core: bool) -> None:
    """Install cache-aware Triton telemetry in the current Python process."""
    global _TRITON_COMPILE_LISTENER_INSTALLED_PID
    current_pid = os.getpid()
    with _TRITON_COMPILE_LISTENER_LOCK:
        if _TRITON_COMPILE_LISTENER_INSTALLED_PID == current_pid:
            if engine_core:
                _record_engine_core_listener_installation()
            return
        triton = importlib.import_module("triton")
        jit_monitor = importlib.import_module("vllm.utils.jit_monitor")
        previous_listener = triton.knobs.compilation.listener
        if getattr(previous_listener, "_comfy_modal_cache_aware", False):
            previous_listener = None

        def listener(**event: Any) -> None:
            """Preserve an existing listener before recording Modal telemetry."""
            if previous_listener is not None:
                previous_listener(**event)
            _triton_compile_listener(**event)

        listener._comfy_modal_cache_aware = True  # type: ignore[attr-defined]

        def disable_cache_blind_post_compile_hook() -> None:
            """Leave the cache-blind Triton post-compile hook unset."""
            logger.info(
                "Using cache-aware Triton compilation listener instead of "
                "vLLM's cache-blind jit_post_compile_hook."
            )

        triton.knobs.compilation.listener = listener
        jit_monitor._setup_triton_jit_hook = disable_cache_blind_post_compile_hook
        _TRITON_COMPILE_LISTENER_INSTALLED_PID = current_pid
        if engine_core:
            _record_engine_core_listener_installation()


def _run_vllm_engine_core_with_accurate_triton_listener(
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Install telemetry inside a spawned EngineCore before its runtime starts."""
    _install_triton_compile_listener_in_current_process(engine_core=True)
    core_module = importlib.import_module("vllm.v1.engine.core")
    original_entrypoint = _VLLM_ENGINE_CORE_ORIGINAL_ENTRYPOINT
    if original_entrypoint is None:
        original_entrypoint = core_module.EngineCoreProc.run_engine_core
    if original_entrypoint is _run_vllm_engine_core_with_accurate_triton_listener:
        raise RuntimeError("Unable to recover vLLM's original EngineCore entrypoint.")
    return original_entrypoint(*args, **kwargs)


def _patch_vllm_engine_core_entrypoint() -> None:
    """Make multiprocessing spawn enter EngineCore through the telemetry wrapper."""
    global _VLLM_ENGINE_CORE_ENTRYPOINT_PATCHED_PID
    global _VLLM_ENGINE_CORE_ORIGINAL_ENTRYPOINT
    current_pid = os.getpid()
    if _VLLM_ENGINE_CORE_ENTRYPOINT_PATCHED_PID == current_pid:
        return
    core_module = importlib.import_module("vllm.v1.engine.core")
    current_entrypoint = core_module.EngineCoreProc.run_engine_core
    if current_entrypoint is not _run_vllm_engine_core_with_accurate_triton_listener:
        _VLLM_ENGINE_CORE_ORIGINAL_ENTRYPOINT = current_entrypoint
        core_module.EngineCoreProc.run_engine_core = staticmethod(
            _run_vllm_engine_core_with_accurate_triton_listener
        )
    _VLLM_ENGINE_CORE_ENTRYPOINT_PATCHED_PID = current_pid
    logger.info("Installed cache-aware Triton wrapper on vLLM EngineCore spawn.")


def _install_accurate_triton_compile_listener() -> None:
    """Replace vLLM's cache-blind hook in this process and spawned EngineCore."""
    _install_triton_compile_listener_in_current_process(engine_core=False)
    _patch_vllm_engine_core_entrypoint()


def _normalize_vllm_execution_setting(value: object) -> str:
    """Return a validated auto, eager, or throughput deployment setting."""
    setting = str(value).strip().lower()
    if setting not in _VLLM_EXECUTION_SETTINGS:
        supported = ", ".join(sorted(_VLLM_EXECUTION_SETTINGS))
        raise ValueError(
            "COMFY_MODAL_LLM_VLLM_EXECUTION_MODE must be one of "
            f"{supported}; got {value!r}."
        )
    return setting


def _vllm_execution_setting(profile: LLMModelProfile | None = None) -> str:
    """Return the deployment setting, including a manager-provided override."""
    if profile is not None:
        runtime_setting = profile.backend_option(_VLLM_RUNTIME_SETTING_OPTION)
        if runtime_setting is not None:
            return _normalize_vllm_execution_setting(runtime_setting)
    return _normalize_vllm_execution_setting(
        os.getenv("COMFY_MODAL_LLM_VLLM_EXECUTION_MODE", "auto")
    )


def _vllm_execution_policy(profile: LLMModelProfile) -> tuple[str, bool]:
    """Resolve the effective eager or CUDA-graph mode for one engine load."""
    runtime_mode = profile.backend_option(_VLLM_RUNTIME_MODE_OPTION)
    if runtime_mode is not None:
        mode = str(runtime_mode).strip().lower()
        if mode not in _VLLM_EFFECTIVE_MODES:
            supported = ", ".join(sorted(_VLLM_EFFECTIVE_MODES))
            raise ValueError(
                f"Invalid runtime vLLM mode {runtime_mode!r}; expected {supported}."
            )
        return mode, mode == "eager"
    setting = _vllm_execution_setting(profile)
    mode = "eager" if setting == "auto" else setting
    return mode, mode == "eager"


class VLLMExecutionModeController:
    """Promote one container from eager to throughput after a second workflow."""

    def __init__(self, setting: str | None = None) -> None:
        """Configure one container-local execution-mode state machine."""
        self.setting = _normalize_vllm_execution_setting(
            setting
            if setting is not None
            else os.getenv("COMFY_MODAL_LLM_VLLM_EXECUTION_MODE", "auto")
        )
        self._first_workflow_execution_id: str | None = None
        self._anonymous_execution_count = 0
        self._promoted = False
        self._lock = threading.RLock()

    def observe(self, workflow_execution_id: str | None) -> bool:
        """Record a workflow and return whether this observation promoted auto mode."""
        if self.setting != "auto":
            return False
        with self._lock:
            normalized_id = str(workflow_execution_id or "").strip()
            if not normalized_id:
                self._anonymous_execution_count += 1
                normalized_id = f"anonymous-{self._anonymous_execution_count}"
            if self._first_workflow_execution_id is None:
                self._first_workflow_execution_id = normalized_id
                logger.info(
                    "vLLM auto mode selected eager for this container's first "
                    "workflow execution."
                )
                return False
            if normalized_id == self._first_workflow_execution_id or self._promoted:
                return False
            self._promoted = True
            logger.info(
                "vLLM auto mode observed a second workflow execution; promoting "
                "this container to throughput mode."
            )
            return True

    def effective_mode(self) -> str:
        """Return the effective mode for the next vLLM engine construction."""
        with self._lock:
            if self.setting == "auto":
                return "throughput" if self._promoted else "eager"
            return self.setting

    def force_throughput_after_memory_recovery(
        self,
        workflow_execution_id: str | None,
    ) -> bool:
        """Preserve an auto-mode promotion when retrying on a fresh worker."""
        if self.setting != "auto":
            return False
        with self._lock:
            if self._promoted:
                return False
            normalized_id = str(workflow_execution_id or "memory-recovery").strip()
            self._first_workflow_execution_id = (
                f"pre-recovery:{normalized_id or 'memory-recovery'}"
            )
            self._promoted = True
            logger.info(
                "vLLM auto mode preserved throughput promotion on a fresh "
                "memory-recovery worker."
            )
            return True

    @property
    def promoted(self) -> bool:
        """Return whether auto mode has observed a second workflow."""
        with self._lock:
            return self._promoted

    @property
    def observed_workflow_count(self) -> int:
        """Return the bounded workflow count relevant to auto promotion."""
        with self._lock:
            if self._first_workflow_execution_id is None:
                return 0
            return 2 if self._promoted else 1


_VLLM_MODE_CONTROLLER: VLLMExecutionModeController | None = None
_VLLM_MODE_CONTROLLER_LOCK = threading.Lock()


def get_vllm_execution_mode_controller() -> VLLMExecutionModeController:
    """Return the process-global controller shared by every remote workflow."""
    global _VLLM_MODE_CONTROLLER
    with _VLLM_MODE_CONTROLLER_LOCK:
        if _VLLM_MODE_CONTROLLER is None:
            _VLLM_MODE_CONTROLLER = VLLMExecutionModeController()
        return _VLLM_MODE_CONTROLLER


def observe_modal_workflow_execution(workflow_execution_id: str | None) -> bool:
    """Record one workflow at the container boundary for auto-mode selection."""
    return get_vllm_execution_mode_controller().observe(workflow_execution_id)


def force_modal_vllm_throughput_after_memory_recovery(
    workflow_execution_id: str | None,
) -> bool:
    """Preserve throughput mode when a failed auto worker is replaced."""
    return get_vllm_execution_mode_controller().force_throughput_after_memory_recovery(
        workflow_execution_id
    )


def _profile_for_vllm_execution(
    profile: LLMModelProfile,
    controller: VLLMExecutionModeController,
) -> LLMModelProfile:
    """Attach ephemeral execution policy without changing profile or weight identity."""
    if profile.backend != "vllm":
        return profile
    backend_options = dict(profile.backend_options)
    backend_options[_VLLM_RUNTIME_MODE_OPTION] = controller.effective_mode()
    backend_options[_VLLM_RUNTIME_SETTING_OPTION] = controller.setting
    return replace(profile, backend_options=tuple(sorted(backend_options.items())))



