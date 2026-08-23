"""Resident multimodal Transformers inference inside a Modal ComfyUI worker."""

from __future__ import annotations

import asyncio
import base64
import binascii
import gc
import importlib
import json
import logging
import math
import os
import socket
import subprocess
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Coroutine, Mapping, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image

if __package__:
    from .llm_recovery import (
        LLM_MEMORY_RECOVERY_EXHAUSTED_MARKER,
        LLM_VLLM_THROUGHPUT_FAILURE_MARKER,
    )
    from .llm_profiles import LLMModelProfile, get_llm_profile
    from .llm_reasoning import (
        ReasoningOutputParser,
        create_reasoning_parser,
        reasoning_chat_template_kwargs,
        reasoning_parser_for_request,
    )
    from .llm_staging import is_model_snapshot_staged, model_snapshot_path
else:  # pragma: no cover - remote node bundles may import top-level modules.
    from llm_recovery import (
        LLM_MEMORY_RECOVERY_EXHAUSTED_MARKER,
        LLM_VLLM_THROUGHPUT_FAILURE_MARKER,
    )
    from llm_profiles import LLMModelProfile, get_llm_profile
    from llm_reasoning import (
        ReasoningOutputParser,
        create_reasoning_parser,
        reasoning_chat_template_kwargs,
        reasoning_parser_for_request,
    )
    from llm_staging import is_model_snapshot_staged, model_snapshot_path

logger = logging.getLogger(__name__)

_BYTES_PER_GIB = 1024**3
_DEFAULT_STORAGE_ROOT = "/storage"
_DEFAULT_RESERVE_FREE_VRAM_GB = 24.0
_DEFAULT_MAX_RESIDENT_MODELS = 2
_DEFAULT_MEMORY_RECOVERY_TIMEOUT_SECONDS = 15.0
_DEFAULT_MEMORY_RECOVERY_POLL_INTERVAL_SECONDS = 0.25
_VLLM_EXECUTION_SETTINGS = frozenset({"auto", "eager", "throughput"})
_VLLM_EFFECTIVE_MODES = frozenset({"eager", "throughput"})
_VLLM_SAFETENSORS_LOAD_STRATEGY = "prefetch"
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


@dataclass(frozen=True)
class PreparedVideo:
    """Hold uniformly sampled video frames and their source timestamps."""

    frames: tuple[Image.Image, ...]
    timestamps_seconds: tuple[float, ...]


@dataclass(frozen=True)
class PreparedLLMInputs:
    """Hold normalized user content ready for a multimodal processor."""

    prompt: str
    system_prompt: str
    images: tuple[Image.Image, ...]
    video: PreparedVideo | None
    file_characters: int
    file_count: int


@dataclass(frozen=True)
class LLMGenerationSettings:
    """Hold backend-neutral text generation controls."""

    max_new_tokens: int
    temperature: float
    top_p: float
    seed: int
    enable_reasoning: bool = True


@dataclass(frozen=True)
class LLMProgressEvent:
    """Describe one user-visible phase of resident LLM execution."""

    stage: str
    message: str
    value: float | None = None
    maximum: float | None = None
    unit: str | None = None
    indeterminate: bool = False
    elapsed_seconds: float | None = None
    time_to_first_token_seconds: float | None = None
    tokens_per_second: float | None = None


LLMProgressCallback = Callable[[LLMProgressEvent], None]


@dataclass(frozen=True)
class BackendGenerationResult:
    """Hold text and token counts returned by an inference backend."""

    text: str
    input_tokens: int
    output_tokens: int
    reasoning: str = ""
    reasoning_tokens: int = 0
    reasoning_parser: str = "none"
    time_to_first_token_seconds: float | None = None
    tokens_per_second: float | None = None


@dataclass(frozen=True)
class LLMInferenceResult:
    """Hold the node response and structured runtime telemetry."""

    text: str
    metadata: dict[str, Any]
    reasoning: str = ""


class LLMBackend(Protocol):
    """Define the backend operations managed by the resident model cache."""

    def generate(
        self,
        prepared_inputs: PreparedLLMInputs,
        settings: LLMGenerationSettings,
        progress_callback: LLMProgressCallback,
    ) -> BackendGenerationResult:
        """Generate one response for normalized multimodal content."""

    def unload(self) -> None:
        """Release backend-owned model resources."""


BackendFactory = Callable[[LLMModelProfile, Path, LLMProgressCallback], LLMBackend]


@dataclass
class ResidentModel:
    """Track one loaded backend and its measured device allocation."""

    profile: LLMModelProfile
    backend: LLMBackend
    loaded_at: float
    last_used_at: float
    allocated_bytes: int


def _coerce_positive_int(value: int, name: str, maximum: int) -> int:
    """Validate one positive bounded integer generation setting."""
    resolved = int(value)
    if resolved <= 0 or resolved > maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}, got {resolved}.")
    return resolved


def _tensor_frame_to_pil(frame: Any) -> Image.Image:
    """Convert one ComfyUI HWC float tensor frame into an RGB PIL image."""
    import torch

    if not isinstance(frame, torch.Tensor):
        raise TypeError(
            f"Expected a torch.Tensor image frame, got {type(frame).__name__}."
        )
    if frame.ndim != 3 or frame.shape[-1] not in {1, 3, 4}:
        raise ValueError(
            "Modal LLM images must use ComfyUI's [height, width, channels] frame layout."
        )
    normalized = frame.detach().to(device="cpu", dtype=torch.float32).clamp(0.0, 1.0)
    pixels = (normalized * 255.0).round().to(dtype=torch.uint8).numpy()
    if pixels.shape[-1] == 1:
        pixels = pixels.repeat(3, axis=2)
    return Image.fromarray(pixels).convert("RGB")


def prepare_images(
    images: Any | None, profile: LLMModelProfile
) -> tuple[Image.Image, ...]:
    """Normalize an optional ComfyUI IMAGE batch under the profile limit."""
    if images is None:
        return ()
    if "image" not in profile.modalities:
        raise ValueError(
            f"Model profile {profile.profile_id!r} does not support images."
        )
    if getattr(images, "ndim", None) != 4:
        raise ValueError(
            "Modal LLM images must be a ComfyUI [batch, height, width, channels] tensor."
        )
    image_count = int(images.shape[0])
    if image_count > profile.max_images:
        raise ValueError(
            f"Model profile {profile.profile_id!r} accepts at most {profile.max_images} images; "
            f"received {image_count}."
        )
    return tuple(_tensor_frame_to_pil(images[index]) for index in range(image_count))


def _uniform_sample_indices(frame_count: int, requested_frames: int) -> tuple[int, ...]:
    """Return stable, uniformly spaced frame indices including both endpoints."""
    if frame_count <= 0 or requested_frames <= 0:
        return ()
    sample_count = min(frame_count, requested_frames)
    if sample_count == 1:
        return (0,)
    return tuple(
        min(frame_count - 1, round(index * (frame_count - 1) / (sample_count - 1)))
        for index in range(sample_count)
    )


def prepare_video(
    video: Any | None,
    profile: LLMModelProfile,
    requested_frames: int,
) -> PreparedVideo | None:
    """Decode and uniformly sample a native ComfyUI VIDEO input."""
    if video is None:
        return None
    if "video" not in profile.modalities:
        raise ValueError(
            f"Model profile {profile.profile_id!r} does not support video."
        )
    frame_limit = min(
        _coerce_positive_int(
            requested_frames, "video_frames", profile.max_video_frames
        ),
        profile.max_video_frames,
    )
    components = video.get_components()
    frames = components.images
    if getattr(frames, "ndim", None) != 4:
        raise ValueError("The ComfyUI VIDEO input did not decode to a frame batch.")
    indices = _uniform_sample_indices(int(frames.shape[0]), frame_limit)
    frame_rate = float(components.frame_rate)
    if not math.isfinite(frame_rate) or frame_rate <= 0:
        raise ValueError(
            f"The ComfyUI VIDEO input has invalid frame rate {frame_rate!r}."
        )
    return PreparedVideo(
        frames=tuple(_tensor_frame_to_pil(frames[index]) for index in indices),
        timestamps_seconds=tuple(index / frame_rate for index in indices),
    )


def _file_field(file_value: Any, field_name: str) -> Any:
    """Read one OpenAI input-file field from a mapping or typed object."""
    if isinstance(file_value, Mapping):
        return file_value.get(field_name)
    return getattr(file_value, field_name, None)


def _decode_input_file(file_value: Any, max_file_bytes: int) -> tuple[str, str, bytes]:
    """Decode one built-in OpenAI input-file data URI."""
    filename = str(_file_field(file_value, "filename") or "input.txt").strip()
    file_data = _file_field(file_value, "file_data")
    if not isinstance(file_data, str) or not file_data.startswith("data:"):
        raise ValueError(f"Modal LLM file {filename!r} must contain a base64 data URI.")
    try:
        metadata, encoded_payload = file_data.split(",", maxsplit=1)
    except ValueError as exc:
        raise ValueError(
            f"Modal LLM file {filename!r} has an invalid data URI."
        ) from exc
    if ";base64" not in metadata:
        raise ValueError(f"Modal LLM file {filename!r} must use base64 encoding.")
    mime_type = metadata[5:].split(";", maxsplit=1)[0].lower()
    maximum_encoded_length = math.ceil(max_file_bytes / 3) * 4 + 4
    if len(encoded_payload) > maximum_encoded_length:
        raise ValueError(
            f"Modal LLM file {filename!r} exceeds the profile's {max_file_bytes}-byte limit."
        )
    try:
        raw_bytes = base64.b64decode(encoded_payload, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError(
            f"Modal LLM file {filename!r} has invalid base64 content."
        ) from exc
    if len(raw_bytes) > max_file_bytes:
        raise ValueError(
            f"Modal LLM file {filename!r} is {len(raw_bytes)} bytes; the profile limit is "
            f"{max_file_bytes} bytes."
        )
    return filename, mime_type, raw_bytes


def _extract_pdf_text(filename: str, raw_bytes: bytes) -> str:
    """Extract bounded source text from a PDF file."""
    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except ImportError as exc:
        raise RuntimeError(
            "PDF input requires the pinned pypdf remote dependency."
        ) from exc
    try:
        reader = PdfReader(BytesIO(raw_bytes))
        extracted_text = "\n\n".join(
            (page.extract_text() or "") for page in reader.pages
        )
    except (PdfReadError, OSError, ValueError) as exc:
        raise ValueError(
            f"Unable to extract text from PDF file {filename!r}: {exc}"
        ) from exc
    if not extracted_text.strip():
        raise ValueError(
            f"PDF file {filename!r} contains no extractable text; provide its pages as images."
        )
    return extracted_text


def extract_file_context(
    files: Sequence[Any] | None,
    profile: LLMModelProfile,
) -> tuple[str, int, int]:
    """Turn supported built-in file inputs into bounded, labelled prompt text."""
    if not files:
        return "", 0, 0
    if "file" not in profile.modalities:
        raise ValueError(
            f"Model profile {profile.profile_id!r} does not support files."
        )
    sections: list[str] = []
    total_bytes = 0
    total_characters = 0
    for file_value in files:
        filename, mime_type, raw_bytes = _decode_input_file(
            file_value, profile.max_file_bytes
        )
        total_bytes += len(raw_bytes)
        if total_bytes > profile.max_file_bytes:
            raise ValueError(
                f"Modal LLM files total {total_bytes} bytes; the profile aggregate limit is "
                f"{profile.max_file_bytes} bytes."
            )
        suffix = Path(filename).suffix.lower()
        if suffix == ".pdf" or mime_type == "application/pdf":
            text = _extract_pdf_text(filename, raw_bytes)
        elif suffix in {".txt", ".md", ".csv", ".json"} or mime_type.startswith(
            "text/"
        ):
            try:
                text = raw_bytes.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(
                    f"Modal LLM text file {filename!r} is not valid UTF-8."
                ) from exc
        else:
            raise ValueError(
                f"Modal LLM file {filename!r} has unsupported type {mime_type or suffix!r}; "
                "use UTF-8 text or a text-based PDF."
            )
        total_characters += len(text)
        if total_characters > profile.max_file_characters:
            raise ValueError(
                f"Modal LLM files contain {total_characters} extracted characters; the profile "
                f"limit is {profile.max_file_characters}."
            )
        sections.append(f"<file name={filename!r}>\n{text}\n</file>")
    return "\n\n".join(sections), len(sections), total_characters


def prepare_llm_inputs(
    *,
    prompt: str,
    system_prompt: str,
    images: Any | None,
    video: Any | None,
    files: Sequence[Any] | None,
    video_frames: int,
    profile: LLMModelProfile,
) -> PreparedLLMInputs:
    """Normalize text, image, video, and file inputs under profile limits."""
    prepared_images = prepare_images(images, profile)
    prepared_video = prepare_video(video, profile, video_frames)
    if (
        prepared_images
        and prepared_video is not None
        and not profile.allow_mixed_image_video
    ):
        raise ValueError(
            f"Model profile {profile.profile_id!r} accepts images or video in one request, not both."
        )
    file_context, file_count, file_characters = extract_file_context(files, profile)
    prompt_parts = [prompt]
    if prepared_video is not None:
        timestamps = ", ".join(
            f"{timestamp:.3f}s" for timestamp in prepared_video.timestamps_seconds
        )
        prompt_parts.append(f"Video sample timestamps: {timestamps}")
    if file_context:
        prompt_parts.append("Attached file contents:\n" + file_context)
    combined_prompt = "\n\n".join(part for part in prompt_parts if part)
    if not combined_prompt and not prepared_images and prepared_video is None:
        raise ValueError("Modal LLM requires a prompt, image, video, or file input.")
    return PreparedLLMInputs(
        prompt=combined_prompt,
        system_prompt=system_prompt,
        images=prepared_images,
        video=prepared_video,
        file_characters=file_characters,
        file_count=file_count,
    )


def _dtype_from_name(torch_module: Any, dtype_name: str) -> Any:
    """Resolve a reviewed profile dtype into a torch dtype."""
    dtype = getattr(torch_module, dtype_name, None)
    if dtype is None:
        raise ValueError(f"This PyTorch build does not expose dtype {dtype_name!r}.")
    return dtype


def _move_batch_to_device(batch: Any, device: str) -> Any:
    """Move a Transformers batch encoding to one inference device."""
    if hasattr(batch, "to"):
        return batch.to(device)
    if isinstance(batch, Mapping):
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }
    raise TypeError(
        f"Unsupported Transformers processor output {type(batch).__name__}."
    )


def _stopping_criteria(
    progress_callback: LLMProgressCallback,
    maximum_tokens: int | None = None,
) -> Any:
    """Build a Transformers stopping criterion that reports and checks every token."""
    from transformers import StoppingCriteria, StoppingCriteriaList

    class ComfyProgressStoppingCriteria(StoppingCriteria):
        """Report generated-token progress and surface ComfyUI interruption."""

        def __init__(self) -> None:
            """Initialize the generated-token counter."""
            self.generated_tokens = 0
            self.started_at = time.perf_counter()
            self.first_token_at: float | None = None

        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
            """Report one generation step and continue unless ComfyUI raises."""
            del input_ids, scores, kwargs
            self.generated_tokens += 1
            now = time.perf_counter()
            if self.first_token_at is None:
                self.first_token_at = now
            elapsed_seconds = now - self.started_at
            progress_callback(
                LLMProgressEvent(
                    stage="generating",
                    message="Generating",
                    value=self.generated_tokens,
                    maximum=maximum_tokens,
                    unit="tokens",
                    elapsed_seconds=elapsed_seconds,
                    time_to_first_token_seconds=self.first_token_at - self.started_at,
                    tokens_per_second=(
                        self.generated_tokens / elapsed_seconds
                        if elapsed_seconds > 0
                        else None
                    ),
                )
            )
            return False

    return StoppingCriteriaList([ComfyProgressStoppingCriteria()])


def _apply_multimodal_chat_template(
    processor: Any,
    messages: list[dict[str, Any]],
    *,
    has_predecoded_video: bool,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> Any:
    """Tokenize chat content without resampling video frames prepared by ComfyUI."""
    video_sampling_kwargs: dict[str, Any] = {}
    if has_predecoded_video:
        video_sampling_kwargs["do_sample_frames"] = False
    return processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        **video_sampling_kwargs,
        **dict(chat_template_kwargs or {}),
    )


def _multimodal_messages(prepared_inputs: PreparedLLMInputs) -> list[dict[str, Any]]:
    """Build processor-native multimodal chat messages for either backend."""
    messages: list[dict[str, Any]] = []
    if prepared_inputs.system_prompt:
        messages.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": prepared_inputs.system_prompt}],
            }
        )
    content: list[dict[str, Any]] = [
        {"type": "image", "image": image} for image in prepared_inputs.images
    ]
    if prepared_inputs.video is not None:
        content.append({"type": "video", "video": list(prepared_inputs.video.frames)})
    content.append({"type": "text", "text": prepared_inputs.prompt})
    messages.append({"role": "user", "content": content})
    return messages


def _safetensor_shard_count(snapshot_path: Path) -> int | None:
    """Return the checkpoint shard count exposed by a staged snapshot."""
    index_files = sorted(snapshot_path.glob("*.safetensors.index.json"))
    for index_file in index_files:
        try:
            index_payload = json.loads(index_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Unable to inspect safetensor index %s.", index_file)
            continue
        weight_map = index_payload.get("weight_map")
        if isinstance(weight_map, Mapping):
            shard_names = {
                str(filename)
                for filename in weight_map.values()
                if str(filename).endswith(".safetensors")
            }
            if shard_names:
                return len(shard_names)
    shard_files = tuple(snapshot_path.glob("*.safetensors"))
    return len(shard_files) if shard_files else None


def _weight_progress_message(shard_count: int | None) -> str:
    """Return compact copy for the model-weight startup phase."""
    if shard_count is None:
        return "Loading model weights"
    noun = "shard" if shard_count == 1 else "shards"
    return f"Loading {shard_count} weight {noun}"


class TransformersMultimodalBackend:
    """Run a curated image-text-to-text model through Hugging Face Transformers."""

    def __init__(
        self,
        profile: LLMModelProfile,
        snapshot_path: Path,
        progress_callback: LLMProgressCallback,
    ) -> None:
        """Load processor and model entirely from the staged immutable snapshot."""
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.profile = profile
        self.snapshot_path = snapshot_path
        dtype = _dtype_from_name(torch, profile.dtype)
        progress_callback(
            LLMProgressEvent(
                stage="processor",
                message="Loading processor",
                indeterminate=True,
            )
        )
        logger.info(
            "Loading resident Modal LLM profile=%s path=%s dtype=%s.",
            profile.profile_id,
            snapshot_path,
            profile.dtype,
        )
        self.processor = AutoProcessor.from_pretrained(
            str(snapshot_path),
            local_files_only=True,
            trust_remote_code=False,
        )
        self.reasoning_parser: ReasoningOutputParser = create_reasoning_parser(
            profile,
            self.processor.tokenizer,
        )
        model_options: dict[str, Any] = {
            "local_files_only": True,
            "trust_remote_code": False,
            "dtype": dtype,
            "device_map": "cuda",
        }
        attention_implementation = profile.backend_option(
            "attention_implementation", "sdpa"
        )
        if attention_implementation:
            model_options["attn_implementation"] = attention_implementation
        shard_count = _safetensor_shard_count(snapshot_path)
        progress_callback(
            LLMProgressEvent(
                stage="weights",
                message=_weight_progress_message(shard_count),
                value=0 if shard_count else None,
                maximum=shard_count,
                unit="shards" if shard_count else None,
                indeterminate=shard_count is None,
            )
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(snapshot_path),
            **model_options,
        )
        self.model.eval()
        progress_callback(
            LLMProgressEvent(
                stage="weights",
                message="Model weights loaded",
                value=shard_count,
                maximum=shard_count,
                unit="shards" if shard_count else None,
            )
        )

    def _messages(self, prepared_inputs: PreparedLLMInputs) -> list[dict[str, Any]]:
        """Build processor-native multimodal chat messages."""
        return _multimodal_messages(prepared_inputs)

    def generate(
        self,
        prepared_inputs: PreparedLLMInputs,
        settings: LLMGenerationSettings,
        progress_callback: LLMProgressCallback,
    ) -> BackendGenerationResult:
        """Tokenize multimodal messages and generate only the assistant continuation."""
        import torch

        reasoning_parser = reasoning_parser_for_request(
            self.reasoning_parser,
            settings.enable_reasoning,
        )
        inputs = _apply_multimodal_chat_template(
            self.processor,
            self._messages(prepared_inputs),
            has_predecoded_video=prepared_inputs.video is not None,
            chat_template_kwargs=reasoning_chat_template_kwargs(
                self.profile,
                settings.enable_reasoning,
            ),
        )
        inputs = _move_batch_to_device(inputs, "cuda")
        input_ids = inputs.get("input_ids")
        if input_ids is None:
            raise RuntimeError("The multimodal processor did not return input_ids.")
        input_tokens = int(input_ids.shape[-1])
        if input_tokens + settings.max_new_tokens > self.profile.max_context_tokens:
            raise ValueError(
                f"Modal LLM request requires up to {input_tokens + settings.max_new_tokens} tokens; "
                f"profile {self.profile.profile_id!r} is capped at {self.profile.max_context_tokens}."
            )
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": settings.max_new_tokens,
            "do_sample": settings.temperature > 0,
            "stopping_criteria": _stopping_criteria(
                progress_callback,
                settings.max_new_tokens,
            ),
        }
        if settings.temperature > 0:
            generate_kwargs.update(
                temperature=settings.temperature, top_p=settings.top_p
            )
        pad_token_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            generate_kwargs["pad_token_id"] = pad_token_id
        device_index = torch.cuda.current_device()
        with torch.random.fork_rng(devices=[device_index]), torch.inference_mode():
            torch.manual_seed(settings.seed)
            generated_ids = self.model.generate(**inputs, **generate_kwargs)
        continuation_ids = generated_ids[:, input_tokens:]
        raw_text = self.processor.batch_decode(
            continuation_ids,
            skip_special_tokens=not reasoning_parser.requires_boundary_tokens,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        reasoning_output = reasoning_parser.extract(
            raw_text,
            continuation_ids[0].tolist(),
        )
        return BackendGenerationResult(
            text=reasoning_output.response,
            input_tokens=input_tokens,
            output_tokens=int(continuation_ids.shape[-1]),
            reasoning=reasoning_output.reasoning,
            reasoning_tokens=reasoning_output.reasoning_tokens,
            reasoning_parser=reasoning_output.parser,
        )

    def unload(self) -> None:
        """Move the model off device and release its processor references."""
        import torch

        model = self.model
        self.model = None
        self.processor = None
        del model
        torch.cuda.empty_cache()


class _AsyncLoopRunner:
    """Keep one asyncio loop alive for a resident asynchronous vLLM engine."""

    def __init__(self) -> None:
        """Start a daemon thread and wait until its event loop is available."""
        self.loop = asyncio.new_event_loop()
        self._started = threading.Event()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="modal-llm-vllm-loop",
            daemon=True,
        )
        self._thread.start()
        self._started.wait()

    def _run_loop(self) -> None:
        """Own and run the backend event loop until explicit shutdown."""
        asyncio.set_event_loop(self.loop)
        self._started.set()
        self.loop.run_forever()
        self.loop.close()

    def run(self, coroutine: Coroutine[Any, Any, Any]) -> Any:
        """Run one coroutine on the resident loop and return its result."""
        if not self._thread.is_alive():
            raise RuntimeError("The resident vLLM event loop is not running.")
        future = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
        return future.result()

    def close(self) -> None:
        """Stop and join the resident event-loop thread."""
        if not self._thread.is_alive():
            return
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=10.0)
        if self._thread.is_alive():
            logger.warning("Resident vLLM event-loop thread did not stop promptly.")


@dataclass(frozen=True)
class _VLLMStreamState:
    """Retain the final cumulative output and request timing boundaries."""

    request_output: Any
    started_at: float
    first_token_at: float | None


class LlamaCppServerBackend:
    """Run one curated GGUF model through a resident llama.cpp CUDA server."""

    def __init__(
        self,
        profile: LLMModelProfile,
        snapshot_path: Path,
        progress_callback: LLMProgressCallback,
    ) -> None:
        """Start a private loopback server for one immutable GGUF artifact."""
        from transformers import AutoProcessor

        self.profile = profile
        self.snapshot_path = snapshot_path
        self.model_path = self._model_path()
        self.mmproj_path = self._mmproj_path()
        self.port = self._available_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        log_descriptor, log_path = tempfile.mkstemp(
            prefix="comfy-llama-cpp-", suffix=".log"
        )
        os.close(log_descriptor)
        self._log_path = Path(log_path)
        self._log_file = self._log_path.open("w+b")
        progress_callback(
            LLMProgressEvent(
                stage="processor",
                message="Loading GGUF tokenizer",
                indeterminate=True,
            )
        )
        self.processor = AutoProcessor.from_pretrained(
            str(snapshot_path),
            local_files_only=True,
            trust_remote_code=False,
        )
        self.reasoning_parser: ReasoningOutputParser = create_reasoning_parser(
            profile,
            self.processor.tokenizer,
        )
        progress_callback(
            LLMProgressEvent(
                stage="engine",
                message="Loading GGUF weights into llama.cpp",
                indeterminate=True,
            )
        )
        self.process = self._start_server()
        try:
            self._wait_until_ready()
        except (RuntimeError, TimeoutError):
            self.unload()
            raise
        progress_callback(
            LLMProgressEvent(
                stage="ready",
                message="llama.cpp engine ready",
                value=1,
                maximum=1,
                unit="model",
            )
        )

    def _model_path(self) -> Path:
        """Return the required staged GGUF model path."""
        filename = str(self.profile.backend_option("model_filename", "")).strip()
        if not filename or Path(filename).name != filename:
            raise ValueError(
                f"GGUF profile {self.profile.profile_id!r} has no safe model filename."
            )
        model_path = self.snapshot_path / filename
        if not model_path.is_file():
            raise RuntimeError(f"Staged GGUF model is missing at {model_path}.")
        return model_path

    def _mmproj_path(self) -> Path | None:
        """Return the optional staged multimodal projector path."""
        filename = self.profile.backend_option("mmproj_filename")
        if filename is None:
            return None
        normalized_filename = str(filename).strip()
        if not normalized_filename or Path(normalized_filename).name != normalized_filename:
            raise ValueError(
                f"GGUF profile {self.profile.profile_id!r} has no safe multimodal "
                "projector filename."
            )
        mmproj_path = self.snapshot_path / normalized_filename
        if not mmproj_path.is_file():
            raise RuntimeError(
                f"Staged GGUF multimodal projector is missing at {mmproj_path}."
            )
        return mmproj_path

    @staticmethod
    def _available_port() -> int:
        """Reserve and release one loopback port for the private server."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            return int(listener.getsockname()[1])

    def _server_command(self) -> list[str]:
        """Return the bounded CUDA llama-server command for this profile."""
        binary = str(
            self.profile.backend_option("server_binary", "/app/llama-server")
        ).strip()
        context_size = int(
            self.profile.backend_option("context_size", self.profile.max_context_tokens)
        )
        gpu_layers = int(self.profile.backend_option("gpu_layers", 999))
        command = [
            binary,
            "--model",
            str(self.model_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--ctx-size",
            str(context_size),
            "--parallel",
            "1",
            "--n-gpu-layers",
            str(gpu_layers),
            "--cache-type-k",
            str(self.profile.backend_option("cache_type_k", "q8_0")),
            "--cache-type-v",
            str(self.profile.backend_option("cache_type_v", "q8_0")),
            "--flash-attn",
            "on",
            "--no-webui",
        ]
        if self.mmproj_path is not None:
            command.extend(("--mmproj", str(self.mmproj_path)))
        return command

    def _start_server(self) -> subprocess.Popen[bytes]:
        """Launch llama-server without exposing a network listener."""
        command = self._server_command()
        environment = self._server_environment(command)
        logger.info(
            "Starting llama.cpp profile=%s model=%s context=%s port=%d.",
            self.profile.profile_id,
            self.model_path,
            self.profile.backend_option(
                "context_size", self.profile.max_context_tokens
            ),
            self.port,
        )
        try:
            return subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=self._log_file,
                stderr=subprocess.STDOUT,
                close_fds=True,
                env=environment,
            )
        except OSError as exc:
            raise RuntimeError(
                f"Unable to start llama.cpp for profile {self.profile.profile_id!r}: "
                f"{exc}"
            ) from exc

    @staticmethod
    def _server_environment(command: Sequence[str]) -> dict[str, str]:
        """Expose shared libraries installed beside the llama-server binary."""
        environment = os.environ.copy()
        binary_directory = str(Path(command[0]).resolve().parent)
        existing_path = environment.get("LD_LIBRARY_PATH", "")
        path_entries = [entry for entry in existing_path.split(":") if entry]
        if binary_directory not in path_entries:
            path_entries.insert(0, binary_directory)
        environment["LD_LIBRARY_PATH"] = ":".join(path_entries)
        return environment

    def _log_tail(self, maximum_bytes: int = 8192) -> str:
        """Return the bounded tail of the private server log."""
        try:
            self._log_file.flush()
            with self._log_path.open("rb") as log_file:
                log_file.seek(0, os.SEEK_END)
                size = log_file.tell()
                log_file.seek(max(0, size - maximum_bytes))
                return log_file.read().decode("utf-8", errors="replace").strip()
        except OSError:
            return ""

    def _wait_until_ready(self) -> None:
        """Wait until llama.cpp reports that the model is loaded."""
        timeout_seconds = float(
            self.profile.backend_option("server_startup_timeout_seconds", 900)
        )
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            return_code = self.process.poll()
            if return_code is not None:
                raise RuntimeError(
                    f"llama.cpp exited with code {return_code} while loading "
                    f"{self.profile.profile_id!r}. Log tail:\n{self._log_tail()}"
                )
            try:
                with urlopen(f"{self.base_url}/health", timeout=2.0) as response:
                    if response.status == 200:
                        return
            except HTTPError as exc:
                if exc.code != 503:
                    raise RuntimeError(
                        f"llama.cpp health check failed with HTTP {exc.code}."
                    ) from exc
            except (URLError, TimeoutError, OSError):
                pass
            time.sleep(0.25)
        raise TimeoutError(
            f"llama.cpp did not load profile {self.profile.profile_id!r} within "
            f"{timeout_seconds:.0f} seconds. Log tail:\n{self._log_tail()}"
        )

    def _prompt(
        self,
        prepared_inputs: PreparedLLMInputs,
        settings: LLMGenerationSettings,
    ) -> str:
        """Render one text-only chat prompt with the pinned tokenizer template."""
        if prepared_inputs.images or prepared_inputs.video is not None:
            raise ValueError(f"GGUF profile {self.profile.profile_id!r} is text-only.")
        prompt = self.processor.apply_chat_template(
            _multimodal_messages(prepared_inputs),
            add_generation_prompt=True,
            tokenize=False,
            **reasoning_chat_template_kwargs(
                self.profile,
                settings.enable_reasoning,
            ),
        )
        if not isinstance(prompt, str):
            raise RuntimeError("The GGUF tokenizer did not return a text prompt.")
        return prompt

    @staticmethod
    def _image_data_uri(image: Image.Image) -> str:
        """Encode one normalized image for llama.cpp's private chat endpoint."""
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def _chat_messages(
        self,
        prepared_inputs: PreparedLLMInputs,
    ) -> list[dict[str, Any]]:
        """Return OpenAI-compatible text and image chat messages."""
        messages: list[dict[str, Any]] = []
        if prepared_inputs.system_prompt:
            messages.append(
                {"role": "system", "content": prepared_inputs.system_prompt}
            )
        content: list[dict[str, Any]] = [
            {
                "type": "image_url",
                "image_url": {"url": self._image_data_uri(image)},
            }
            for image in prepared_inputs.images
        ]
        content.append({"type": "text", "text": prepared_inputs.prompt})
        messages.append({"role": "user", "content": content})
        return messages

    def _post_json(
        self,
        endpoint: str,
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        """Submit one JSON request to the private llama.cpp server."""
        request = Request(
            f"{self.base_url}{endpoint}",
            data=json.dumps(dict(payload)).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                decoded = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"llama.cpp request to {endpoint} failed with HTTP {exc.code}: "
                f"{error_body}"
            ) from exc
        except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"llama.cpp request to {endpoint} failed: {exc}"
            ) from exc
        if not isinstance(decoded, dict):
            raise RuntimeError(
                f"llama.cpp request to {endpoint} returned a non-object response."
            )
        return decoded

    def _completion(
        self, payload: Mapping[str, Any], timeout_seconds: float
    ) -> dict[str, Any]:
        """Submit one non-streaming completion request to the private server."""
        return self._post_json("/completion", payload, timeout_seconds)

    def _chat_completion(
        self, payload: Mapping[str, Any], timeout_seconds: float
    ) -> dict[str, Any]:
        """Submit one multimodal OpenAI-compatible chat request."""
        return self._post_json("/v1/chat/completions", payload, timeout_seconds)

    @staticmethod
    def _chat_response_content(response: Mapping[str, Any]) -> tuple[str, int, int]:
        """Extract text and token counts from one chat-completion response."""
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("llama.cpp chat completion returned no choices.")
        first_choice = choices[0]
        message = first_choice.get("message") if isinstance(first_choice, Mapping) else None
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str):
            raise RuntimeError("llama.cpp chat completion omitted message content.")
        usage = response.get("usage")
        usage_mapping = usage if isinstance(usage, Mapping) else {}
        return (
            content,
            int(usage_mapping.get("prompt_tokens", 0)),
            int(usage_mapping.get("completion_tokens", 0)),
        )

    def generate(
        self,
        prepared_inputs: PreparedLLMInputs,
        settings: LLMGenerationSettings,
        progress_callback: LLMProgressCallback,
    ) -> BackendGenerationResult:
        """Generate one response and report bounded llama.cpp telemetry."""
        progress_callback(
            LLMProgressEvent(
                stage="prefill",
                message="Prefill / waiting for llama.cpp",
                value=0,
                maximum=settings.max_new_tokens,
                unit="tokens",
                indeterminate=True,
            )
        )
        started_at = time.perf_counter()
        timeout_seconds = max(900.0, settings.max_new_tokens * 10.0)
        if prepared_inputs.images:
            response = self._chat_completion(
                {
                    "model": self.profile.profile_id,
                    "messages": self._chat_messages(prepared_inputs),
                    "max_tokens": settings.max_new_tokens,
                    "temperature": settings.temperature,
                    "top_p": settings.top_p,
                    "seed": settings.seed,
                    "repeat_penalty": 1.05,
                    "chat_template_kwargs": reasoning_chat_template_kwargs(
                        self.profile,
                        settings.enable_reasoning,
                    ),
                },
                timeout_seconds=timeout_seconds,
            )
            content, input_tokens, output_tokens = self._chat_response_content(response)
            output_token_ids: list[int] = []
        else:
            response = self._completion(
                {
                    "prompt": self._prompt(prepared_inputs, settings),
                    "n_predict": settings.max_new_tokens,
                    "temperature": settings.temperature,
                    "top_p": settings.top_p,
                    "seed": settings.seed,
                    "repeat_penalty": 1.05,
                    "cache_prompt": True,
                    "return_tokens": True,
                },
                timeout_seconds=timeout_seconds,
            )
            content = response.get("content")
            tokens = response.get("tokens")
            if not isinstance(content, str) or not isinstance(tokens, list):
                raise RuntimeError(
                    "llama.cpp completion omitted string content or generated tokens."
                )
            output_token_ids = [int(token) for token in tokens]
            output_tokens = len(output_token_ids)
            input_tokens = int(response.get("tokens_evaluated", 0))
        completed_at = time.perf_counter()
        elapsed_seconds = completed_at - started_at
        timings = response.get("timings")
        timing_mapping = timings if isinstance(timings, Mapping) else {}
        tokens_per_second = timing_mapping.get("predicted_per_second")
        resolved_tokens_per_second = (
            float(tokens_per_second)
            if isinstance(tokens_per_second, int | float)
            else (output_tokens / elapsed_seconds if elapsed_seconds > 0 else None)
        )
        progress_callback(
            LLMProgressEvent(
                stage="generating",
                message="Generated with llama.cpp",
                value=output_tokens,
                maximum=settings.max_new_tokens,
                unit="tokens",
                elapsed_seconds=elapsed_seconds,
                tokens_per_second=resolved_tokens_per_second,
            )
        )
        reasoning_parser = reasoning_parser_for_request(
            self.reasoning_parser,
            settings.enable_reasoning,
        )
        reasoning_output = reasoning_parser.extract(content, output_token_ids)
        return BackendGenerationResult(
            text=reasoning_output.response,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning=reasoning_output.reasoning,
            reasoning_tokens=reasoning_output.reasoning_tokens,
            reasoning_parser=reasoning_output.parser,
            tokens_per_second=resolved_tokens_per_second,
        )

    def runtime_metadata(self) -> dict[str, Any]:
        """Return the GGUF artifact and server configuration."""
        return {
            "llama_cpp_model_filename": self.model_path.name,
            "llama_cpp_mmproj_filename": (
                self.mmproj_path.name if self.mmproj_path is not None else None
            ),
            "llama_cpp_context_size": int(
                self.profile.backend_option(
                    "context_size",
                    self.profile.max_context_tokens,
                )
            ),
            "llama_cpp_cache_type_k": self.profile.backend_option(
                "cache_type_k", "q8_0"
            ),
            "llama_cpp_cache_type_v": self.profile.backend_option(
                "cache_type_v", "q8_0"
            ),
        }

    def unload(self) -> None:
        """Stop the private server and remove its bounded diagnostic log."""
        process = getattr(self, "process", None)
        self.process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)
        log_file = getattr(self, "_log_file", None)
        if log_file is not None:
            log_file.close()
            self._log_file = None
        log_path = getattr(self, "_log_path", None)
        if isinstance(log_path, Path):
            try:
                log_path.unlink()
            except FileNotFoundError:
                pass
        self.processor = None
        gc.collect()


class VLLMMultimodalBackend:
    """Run Qwen multimodal checkpoints through an asynchronous vLLM engine."""

    def __init__(
        self,
        profile: LLMModelProfile,
        snapshot_path: Path,
        progress_callback: LLMProgressCallback,
    ) -> None:
        """Load an immutable local snapshot under explicit co-residency budgets."""
        from transformers import AutoProcessor
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        if os.getenv("COMFY_MODAL_REMOTE_WORKER") == "1":
            _install_accurate_triton_compile_listener()

        self.profile = profile
        self.snapshot_path = snapshot_path
        self.execution_setting = _vllm_execution_setting(profile)
        self.execution_mode, self.enforce_eager = _vllm_execution_policy(profile)
        progress_callback(
            LLMProgressEvent(
                stage="processor",
                message="Loading processor",
                indeterminate=True,
            )
        )
        self.processor = AutoProcessor.from_pretrained(
            str(snapshot_path),
            local_files_only=True,
            trust_remote_code=False,
        )
        self.reasoning_parser: ReasoningOutputParser = create_reasoning_parser(
            profile,
            self.processor.tokenizer,
        )
        quantization = str(profile.backend_option("quantization", "")).strip()
        shard_count = _safetensor_shard_count(snapshot_path)
        self._log_engine_configuration(quantization, shard_count)
        progress_callback(
            LLMProgressEvent(
                stage="engine",
                message=_weight_progress_message(shard_count) + " + engine warmup",
                value=0 if shard_count else None,
                maximum=shard_count,
                unit="shards" if shard_count else None,
                indeterminate=True,
            )
        )
        engine_args = self._engine_arguments(AsyncEngineArgs, quantization)
        self.llm = self._start_engine(AsyncLLMEngine, engine_args)
        progress_callback(
            LLMProgressEvent(
                stage="ready",
                message="vLLM engine ready",
                value=shard_count,
                maximum=shard_count,
                unit="shards" if shard_count else None,
            )
        )

    def _log_engine_configuration(
        self,
        quantization: str,
        shard_count: int | None,
    ) -> None:
        """Log the immutable vLLM configuration before its expensive startup."""
        logger.info(
            "Loading asynchronous vLLM profile=%s path=%s quantization=%s "
            "mode=%s enforce_eager=%s safetensors_load_strategy=%s "
            "max_model_len=%d kv_cache_gib=%.1f shards=%s compile_cache=%s.",
            self.profile.profile_id,
            self.snapshot_path,
            quantization or "auto",
            self.execution_mode,
            self.enforce_eager,
            _VLLM_SAFETENSORS_LOAD_STRATEGY,
            int(
                self.profile.backend_option(
                    "max_model_len",
                    self.profile.max_context_tokens,
                )
            ),
            int(self.profile.backend_option("kv_cache_memory_bytes", 0))
            / _BYTES_PER_GIB,
            shard_count,
            os.getenv("VLLM_CACHE_ROOT", "<ephemeral-default>"),
        )

    def _engine_arguments(self, argument_class: Any, quantization: str) -> Any:
        """Build explicit AsyncLLM co-residency arguments for this profile."""
        return argument_class(
            model=str(self.snapshot_path),
            tokenizer=str(self.snapshot_path),
            trust_remote_code=False,
            dtype=self.profile.dtype,
            quantization=quantization or None,
            max_model_len=int(
                self.profile.backend_option(
                    "max_model_len",
                    self.profile.max_context_tokens,
                )
            ),
            kv_cache_memory_bytes=int(
                self.profile.backend_option(
                    "kv_cache_memory_bytes",
                    12 * _BYTES_PER_GIB,
                )
            ),
            enforce_eager=self.enforce_eager,
            safetensors_load_strategy=_VLLM_SAFETENSORS_LOAD_STRATEGY,
            disable_custom_all_reduce=True,
            attention_config={"backend": "TRITON_ATTN"},
            generation_config="vllm",
            limit_mm_per_prompt={"image": self.profile.max_images, "video": 1},
        )

    def runtime_metadata(self) -> dict[str, Any]:
        """Return the execution and persistent-cache settings used by vLLM."""
        return {
            "vllm_execution_setting": self.execution_setting,
            "vllm_execution_mode": self.execution_mode,
            "vllm_enforce_eager": self.enforce_eager,
            "vllm_safetensors_load_strategy": _VLLM_SAFETENSORS_LOAD_STRATEGY,
            "vllm_compile_cache_root": os.getenv("VLLM_CACHE_ROOT"),
        }

    def _start_engine(self, engine_class: Any, engine_args: Any) -> Any:
        """Start AsyncLLM and clean up its loop if initialization fails."""
        self._loop_runner = _AsyncLoopRunner()
        engine_created = False
        try:
            engine = self._loop_runner.run(
                self._create_engine(engine_class, engine_args)
            )
            engine_created = True
            return engine
        finally:
            if not engine_created:
                self._loop_runner.close()

    @staticmethod
    async def _create_engine(engine_class: Any, engine_args: Any) -> Any:
        """Construct AsyncLLM while its long-lived event loop is current."""
        return engine_class.from_engine_args(engine_args)

    def _request(
        self,
        prepared_inputs: PreparedLLMInputs,
        settings: LLMGenerationSettings,
    ) -> dict[str, Any]:
        """Build one vLLM prompt with direct in-process multimodal data."""
        prompt = self.processor.apply_chat_template(
            _multimodal_messages(prepared_inputs),
            add_generation_prompt=True,
            tokenize=False,
            **reasoning_chat_template_kwargs(
                self.profile,
                settings.enable_reasoning,
            ),
        )
        if not isinstance(prompt, str):
            raise RuntimeError("The multimodal processor did not return a text prompt.")
        multimodal_data: dict[str, Any] = {}
        if prepared_inputs.images:
            multimodal_data["image"] = list(prepared_inputs.images)
        if prepared_inputs.video is not None:
            import numpy as np

            multimodal_data["video"] = np.stack(
                [np.asarray(frame) for frame in prepared_inputs.video.frames]
            )
        request: dict[str, Any] = {"prompt": prompt}
        if multimodal_data:
            request["multi_modal_data"] = multimodal_data
        return request

    def generate(
        self,
        prepared_inputs: PreparedLLMInputs,
        settings: LLMGenerationSettings,
        progress_callback: LLMProgressCallback,
    ) -> BackendGenerationResult:
        """Generate one response while streaming cumulative token telemetry."""
        return self._loop_runner.run(
            self._generate_async(prepared_inputs, settings, progress_callback)
        )

    async def _generate_async(
        self,
        prepared_inputs: PreparedLLMInputs,
        settings: LLMGenerationSettings,
        progress_callback: LLMProgressCallback,
    ) -> BackendGenerationResult:
        """Consume vLLM's async output stream and report every token update."""
        from vllm import SamplingParams
        from vllm.v1.engine.exceptions import EngineDeadError

        self._report_prefill(settings, progress_callback)
        sampling_params = self._sampling_params(SamplingParams, settings)
        request_id = f"modal-llm-{uuid.uuid4().hex}"
        finished = False
        try:
            stream_state = await self._consume_stream(
                prepared_inputs,
                settings,
                sampling_params,
                request_id,
                progress_callback,
            )
            finished = True
        except (EngineDeadError, RuntimeError) as error:
            logger.exception(
                "vLLM generation failed for profile=%s.",
                self.profile.profile_id,
            )
            raise RuntimeError(
                f"vLLM generation failed for profile {self.profile.profile_id!r}: "
                f"{error}"
            ) from None
        finally:
            if not finished:
                await self._abort_request(request_id, EngineDeadError)
        return self._generation_result(stream_state, settings)

    @staticmethod
    def _report_prefill(
        settings: LLMGenerationSettings,
        progress_callback: LLMProgressCallback,
    ) -> None:
        """Publish the indeterminate interval before vLLM yields a token."""
        progress_callback(
            LLMProgressEvent(
                stage="prefill",
                message="Prefill / waiting for first token",
                value=0,
                maximum=settings.max_new_tokens,
                unit="tokens",
                indeterminate=True,
            )
        )

    def _sampling_params(
        self,
        parameter_class: Any,
        settings: LLMGenerationSettings,
    ) -> Any:
        """Translate backend-neutral settings into vLLM sampling parameters."""
        reasoning_parser = reasoning_parser_for_request(
            self.reasoning_parser,
            settings.enable_reasoning,
        )
        return parameter_class(
            max_tokens=settings.max_new_tokens,
            temperature=settings.temperature,
            top_p=settings.top_p,
            seed=settings.seed,
            skip_special_tokens=not reasoning_parser.requires_boundary_tokens,
        )

    async def _consume_stream(
        self,
        prepared_inputs: PreparedLLMInputs,
        settings: LLMGenerationSettings,
        sampling_params: Any,
        request_id: str,
        progress_callback: LLMProgressCallback,
    ) -> _VLLMStreamState:
        """Consume cumulative AsyncLLM outputs and return final stream state."""
        started_at = time.perf_counter()
        first_token_at: float | None = None
        request_output: Any | None = None
        output_stream = self.llm.generate(
            self._request(prepared_inputs, settings),
            sampling_params=sampling_params,
            request_id=request_id,
        )
        async for streamed_output in output_stream:
            if not streamed_output.outputs:
                continue
            request_output = streamed_output
            now = time.perf_counter()
            output_tokens = len(streamed_output.outputs[0].token_ids)
            if output_tokens > 0 and first_token_at is None:
                first_token_at = now
            self._report_token_progress(
                settings,
                output_tokens,
                started_at,
                first_token_at,
                now,
                progress_callback,
            )
        if request_output is None or not request_output.outputs:
            raise RuntimeError("vLLM returned no generation candidate.")
        return _VLLMStreamState(
            request_output=request_output,
            started_at=started_at,
            first_token_at=first_token_at,
        )

    @staticmethod
    def _report_token_progress(
        settings: LLMGenerationSettings,
        output_tokens: int,
        started_at: float,
        first_token_at: float | None,
        now: float,
        progress_callback: LLMProgressCallback,
    ) -> None:
        """Publish one cumulative token count with live timing telemetry."""
        elapsed_seconds = now - started_at
        progress_callback(
            LLMProgressEvent(
                stage="generating",
                message="Generating",
                value=output_tokens,
                maximum=settings.max_new_tokens,
                unit="tokens",
                elapsed_seconds=elapsed_seconds,
                time_to_first_token_seconds=(
                    first_token_at - started_at if first_token_at is not None else None
                ),
                tokens_per_second=(
                    output_tokens / elapsed_seconds
                    if output_tokens > 0 and elapsed_seconds > 0
                    else None
                ),
            )
        )

    async def _abort_request(
        self, request_id: str, engine_error: type[Exception]
    ) -> None:
        """Best-effort abort one request without masking its original failure."""
        try:
            await self.llm.abort(request_id)
        except (engine_error, RuntimeError) as abort_error:
            logger.debug(
                "Unable to abort failed vLLM request %s: %s",
                request_id,
                abort_error,
            )

    def _generation_result(
        self,
        stream_state: _VLLMStreamState,
        settings: LLMGenerationSettings,
    ) -> BackendGenerationResult:
        """Convert the final cumulative vLLM output into the backend contract."""
        request_output = stream_state.request_output
        candidate = request_output.outputs[0]
        output_tokens = len(candidate.token_ids)
        completed_at = time.perf_counter()
        elapsed_seconds = completed_at - stream_state.started_at
        time_to_first_token_seconds = (
            stream_state.first_token_at - stream_state.started_at
            if stream_state.first_token_at is not None
            else None
        )
        tokens_per_second = (
            output_tokens / elapsed_seconds if elapsed_seconds > 0 else None
        )
        native_reasoning = getattr(candidate, "reasoning", None)
        if native_reasoning is None:
            native_reasoning = getattr(candidate, "reasoning_content", None)
        reasoning_parser = reasoning_parser_for_request(
            self.reasoning_parser,
            settings.enable_reasoning,
        )
        reasoning_output = reasoning_parser.extract(
            str(candidate.text),
            candidate.token_ids,
            native_reasoning=(
                str(native_reasoning) if native_reasoning is not None else None
            ),
        )
        return BackendGenerationResult(
            text=reasoning_output.response,
            input_tokens=len(request_output.prompt_token_ids),
            output_tokens=output_tokens,
            reasoning=reasoning_output.reasoning,
            reasoning_tokens=reasoning_output.reasoning_tokens,
            reasoning_parser=reasoning_output.parser,
            time_to_first_token_seconds=time_to_first_token_seconds,
            tokens_per_second=tokens_per_second,
        )

    def unload(self) -> None:
        """Shut down the vLLM engine and release its CUDA allocations."""
        import torch

        llm = self.llm
        self.llm = None
        self.processor = None
        if llm is not None:
            self._loop_runner.run(self._shutdown_engine(llm))
        self._loop_runner.close()
        del llm
        gc.collect()
        torch.cuda.empty_cache()

    @staticmethod
    async def _shutdown_engine(llm: Any) -> None:
        """Shut down AsyncLLM on the event loop that owns its output task."""
        shutdown = getattr(llm, "shutdown", None)
        if callable(shutdown):
            shutdown()
            await asyncio.sleep(0)


def _default_backend_factory(
    profile: LLMModelProfile,
    snapshot_path: Path,
    progress_callback: LLMProgressCallback,
) -> LLMBackend:
    """Create the backend selected by the immutable compatibility profile."""
    if profile.backend == "transformers":
        return TransformersMultimodalBackend(profile, snapshot_path, progress_callback)
    if profile.backend == "vllm":
        return VLLMMultimodalBackend(profile, snapshot_path, progress_callback)
    if profile.backend == "llama_cpp_server":
        return LlamaCppServerBackend(profile, snapshot_path, progress_callback)
    raise ValueError(
        f"Modal LLM profile {profile.profile_id!r} selects unknown backend "
        f"{profile.backend!r}."
    )


def _comfy_loaded_model_names() -> list[str]:
    """Return best-effort class names for models retained by ComfyUI."""
    try:
        import comfy.model_management

        loaded_models = comfy.model_management.loaded_models()
    except (ImportError, AttributeError, RuntimeError) as exc:
        logger.debug(
            "Unable to inspect ComfyUI model residency after LLM inference: %s", exc
        )
        return []
    names: list[str] = []
    for loaded_model in loaded_models:
        model = getattr(loaded_model, "model", loaded_model)
        inner_model = getattr(model, "model", model)
        names.append(type(inner_model).__name__)
    return names


class ResidentLLMManager:
    """Serialize inference and retain multiple LLMs with memory-aware LRU eviction."""

    def __init__(
        self,
        *,
        storage_root: str | Path = _DEFAULT_STORAGE_ROOT,
        backend_factory: BackendFactory = _default_backend_factory,
        max_resident_models: int = _DEFAULT_MAX_RESIDENT_MODELS,
        memory_info: Callable[[], tuple[int, int]] | None = None,
        empty_cache: Callable[[], None] | None = None,
        snapshot_ready: Callable[
            [str | Path, LLMModelProfile], bool
        ] = is_model_snapshot_staged,
        comfy_memory_release: Callable[[int], None] | None = None,
        execution_target: str = "modal",
        device_name: str = "cuda",
        memory_label: str = "GPU memory",
        vllm_mode_controller: VLLMExecutionModeController | None = None,
        memory_recovery_timeout_seconds: float = _DEFAULT_MEMORY_RECOVERY_TIMEOUT_SECONDS,
        memory_recovery_poll_interval_seconds: float = (
            _DEFAULT_MEMORY_RECOVERY_POLL_INTERVAL_SECONDS
        ),
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Configure the shared model cache and injectable hardware operations."""
        if max_resident_models <= 0:
            raise ValueError("max_resident_models must be positive.")
        if memory_recovery_timeout_seconds < 0:
            raise ValueError("memory_recovery_timeout_seconds cannot be negative.")
        if memory_recovery_poll_interval_seconds <= 0:
            raise ValueError("memory_recovery_poll_interval_seconds must be positive.")
        self.storage_root = Path(storage_root).resolve()
        self.backend_factory = backend_factory
        self.max_resident_models = max_resident_models
        self._memory_info = memory_info or self._cuda_memory_info
        self._empty_cache = empty_cache or self._cuda_empty_cache
        self._snapshot_ready = snapshot_ready
        self._comfy_memory_release = comfy_memory_release or self._release_comfy_memory
        self.execution_target = execution_target
        self.device_name = device_name
        self.memory_label = memory_label
        self.memory_recovery_timeout_seconds = memory_recovery_timeout_seconds
        self.memory_recovery_poll_interval_seconds = (
            memory_recovery_poll_interval_seconds
        )
        self._monotonic = monotonic
        self._sleep = sleep
        self._vllm_mode_controller = vllm_mode_controller
        if self._vllm_mode_controller is None and execution_target == "modal":
            self._vllm_mode_controller = get_vllm_execution_mode_controller()
        self._reported_auto_promotion = False
        self._models: OrderedDict[str, ResidentModel] = OrderedDict()
        self._lock = threading.RLock()

    @staticmethod
    def _cuda_memory_info() -> tuple[int, int]:
        """Return CUDA free and total bytes for the current device."""
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("Modal LLM inference requires a CUDA GPU worker.")
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return int(free_bytes), int(total_bytes)

    @staticmethod
    def _cuda_empty_cache() -> None:
        """Release unused allocations from PyTorch's CUDA caching allocator."""
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _release_comfy_memory(required_bytes: int) -> None:
        """Ask ComfyUI to unload idle image/video models before an LLM load."""
        try:
            import comfy.model_management

            comfy.model_management.free_memory(
                required_bytes,
                comfy.model_management.get_torch_device(),
            )
        except (ImportError, AttributeError, RuntimeError) as exc:
            logger.debug(
                "ComfyUI model memory release was unavailable before LLM load: %s", exc
            )

    def _evict(self, profile_id: str) -> None:
        """Unload one resident backend and release cached CUDA allocations."""
        resident = self._models.pop(profile_id)
        logger.info(
            "Evicting resident LLM target=%s profile=%s allocated_gib=%.3f.",
            self.execution_target,
            profile_id,
            resident.allocated_bytes / _BYTES_PER_GIB,
        )
        resident.backend.unload()
        self._empty_cache()

    def _prepare_vllm_profile(
        self,
        profile: LLMModelProfile,
        workflow_execution_id: str | None,
        progress_callback: LLMProgressCallback,
    ) -> tuple[LLMModelProfile, bool]:
        """Apply container auto-mode state and retire incompatible engines."""
        if profile.backend != "vllm":
            return profile, False
        controller = self._vllm_mode_controller
        if controller is None:
            controller = get_vllm_execution_mode_controller()
            self._vllm_mode_controller = controller
        controller.observe(workflow_execution_id)
        runtime_profile = _profile_for_vllm_execution(profile, controller)
        desired_mode = controller.effective_mode()
        mismatched_profile_ids = [
            profile_id
            for profile_id, resident in self._models.items()
            if resident.profile.backend == "vllm"
            and resident.profile.backend_option(_VLLM_RUNTIME_MODE_OPTION)
            != desired_mode
        ]
        should_report_auto_promotion = (
            controller.setting == "auto"
            and controller.promoted
            and not self._reported_auto_promotion
        )
        if mismatched_profile_ids or should_report_auto_promotion:
            progress_callback(
                LLMProgressEvent(
                    stage="engine",
                    message=(
                        "Optimizing vLLM for repeat workflows"
                        if controller.setting == "auto"
                        else f"Switching vLLM to {desired_mode} mode"
                    ),
                    indeterminate=True,
                )
            )
            if should_report_auto_promotion:
                self._reported_auto_promotion = True
            for profile_id in mismatched_profile_ids:
                self._evict(profile_id)
        return runtime_profile, bool(mismatched_profile_ids)

    def _wait_for_memory_recovery(
        self,
        *,
        required_bytes: int,
        free_bytes: int,
        total_bytes: int,
    ) -> tuple[int, int]:
        """Poll device memory until an eviction is visible or the deadline expires."""
        started_at = self._monotonic()
        deadline = started_at + self.memory_recovery_timeout_seconds
        while free_bytes < required_bytes:
            remaining_seconds = deadline - self._monotonic()
            if remaining_seconds <= 0:
                break
            self._sleep(
                min(self.memory_recovery_poll_interval_seconds, remaining_seconds)
            )
            self._empty_cache()
            free_bytes, total_bytes = self._memory_info()
        elapsed_seconds = self._monotonic() - started_at
        logger.info(
            "Post-eviction %s recovery finished recovered=%s elapsed_seconds=%.3f "
            "free_gib=%.3f required_gib=%.3f total_gib=%.3f.",
            self.memory_label,
            free_bytes >= required_bytes,
            elapsed_seconds,
            free_bytes / _BYTES_PER_GIB,
            required_bytes / _BYTES_PER_GIB,
            total_bytes / _BYTES_PER_GIB,
        )
        return free_bytes, total_bytes

    def _make_room(
        self,
        profile: LLMModelProfile,
        reserve_free_vram_gb: float,
        *,
        evicted_before_load: bool = False,
    ) -> None:
        """Evict old LLMs until the new model plus configured reserve can fit."""
        required_bytes = int(
            (profile.estimated_vram_gb + reserve_free_vram_gb) * _BYTES_PER_GIB
        )
        self._comfy_memory_release(required_bytes)
        if evicted_before_load:
            self._empty_cache()
        free_bytes, total_bytes = self._memory_info()
        evicted_any = evicted_before_load
        while free_bytes < required_bytes and self._models:
            oldest_profile_id = next(iter(self._models))
            self._evict(oldest_profile_id)
            evicted_any = True
            self._comfy_memory_release(required_bytes)
            free_bytes, total_bytes = self._memory_info()
        if free_bytes < required_bytes and evicted_any:
            free_bytes, total_bytes = self._wait_for_memory_recovery(
                required_bytes=required_bytes,
                free_bytes=free_bytes,
                total_bytes=total_bytes,
            )
        if free_bytes < required_bytes:
            message = (
                f"LLM profile {profile.profile_id!r} needs approximately "
                f"{profile.estimated_vram_gb:.1f} GiB plus {reserve_free_vram_gb:.1f} GiB reserve, "
                f"but only {free_bytes / _BYTES_PER_GIB:.1f} of "
                f"{total_bytes / _BYTES_PER_GIB:.1f} GiB of {self.memory_label} is "
                "available."
            )
            if evicted_any:
                effective_mode = profile.backend_option(_VLLM_RUNTIME_MODE_OPTION)
                mode_marker = (
                    f" {LLM_VLLM_THROUGHPUT_FAILURE_MARKER}"
                    if effective_mode == "throughput"
                    else ""
                )
                raise RuntimeError(
                    f"[{LLM_MEMORY_RECOVERY_EXHAUSTED_MARKER}]{mode_marker} "
                    f"{message} Post-eviction recovery remained below the admission "
                    f"threshold for {self.memory_recovery_timeout_seconds:.1f} seconds."
                )
            raise RuntimeError(message)

    def _load(
        self,
        profile: LLMModelProfile,
        reserve_free_vram_gb: float,
        progress_callback: LLMProgressCallback,
        *,
        evicted_before_load: bool = False,
    ) -> tuple[ResidentModel, bool]:
        """Return a cached backend or load one after enforcing the VRAM budget."""
        cached = self._models.pop(profile.profile_id, None)
        if cached is not None:
            cached.last_used_at = time.time()
            self._models[profile.profile_id] = cached
            progress_callback(
                LLMProgressEvent(
                    stage="ready",
                    message="Reusing resident model",
                    value=1,
                    maximum=1,
                    unit="model",
                )
            )
            return cached, True
        if not self._snapshot_ready(self.storage_root, profile):
            raise RuntimeError(
                f"LLM profile {profile.profile_id!r} is not staged at "
                f"{model_snapshot_path(self.storage_root, profile)}. Model staging must "
                f"complete before {self.execution_target} inference starts."
            )
        while len(self._models) >= self.max_resident_models:
            self._evict(next(iter(self._models)))
            evicted_before_load = True
        progress_callback(
            LLMProgressEvent(
                stage="memory",
                message=f"Preparing {self.memory_label}",
                indeterminate=True,
            )
        )
        self._make_room(
            profile,
            reserve_free_vram_gb,
            evicted_before_load=evicted_before_load,
        )
        before_free, _ = self._memory_info()
        backend = self.backend_factory(
            profile,
            model_snapshot_path(self.storage_root, profile),
            progress_callback,
        )
        after_free, _ = self._memory_info()
        now = time.time()
        resident = ResidentModel(
            profile=profile,
            backend=backend,
            loaded_at=now,
            last_used_at=now,
            allocated_bytes=max(0, before_free - after_free),
        )
        self._models[profile.profile_id] = resident
        logger.info(
            "Loaded resident LLM target=%s profile=%s measured_allocation_gib=%.3f "
            "residents=%s.",
            self.execution_target,
            profile.profile_id,
            resident.allocated_bytes / _BYTES_PER_GIB,
            list(self._models),
        )
        return resident, False

    def infer(
        self,
        *,
        profile: LLMModelProfile,
        prepared_inputs: PreparedLLMInputs,
        generation_settings: LLMGenerationSettings,
        reserve_free_vram_gb: float,
        keep_model_loaded: bool,
        progress_callback: LLMProgressCallback,
        workflow_execution_id: str | None = None,
    ) -> LLMInferenceResult:
        """Run one inference while protecting shared resident state."""
        if reserve_free_vram_gb < 0:
            raise ValueError("reserve_free_vram_gb cannot be negative.")
        with self._lock:
            runtime_profile, evicted_for_mode_change = self._prepare_vllm_profile(
                profile,
                workflow_execution_id,
                progress_callback,
            )
            before_free, total_bytes = self._memory_info()
            started_at = time.perf_counter()
            resident, cache_hit = self._load(
                runtime_profile,
                reserve_free_vram_gb,
                progress_callback,
                evicted_before_load=evicted_for_mode_change,
            )
            load_finished_at = time.perf_counter()
            generation_result = resident.backend.generate(
                prepared_inputs,
                generation_settings,
                progress_callback,
            )
            generation_finished_at = time.perf_counter()
            after_free, _ = self._memory_info()
            resident.last_used_at = time.time()
            self._models.move_to_end(profile.profile_id)
            resident_ids = list(self._models)
            if not keep_model_loaded:
                self._evict(profile.profile_id)
                resident_ids = list(self._models)
            generation_seconds = generation_finished_at - load_finished_at
            comfy_loaded_model_names = _comfy_loaded_model_names()
            metadata = {
                "backend": profile.backend,
                "execution_target": self.execution_target,
                "device": self.device_name,
                "profile": profile.profile_id,
                "repository": profile.repository,
                "revision": profile.revision,
                "cache_hit": cache_hit,
                "input_tokens": generation_result.input_tokens,
                "output_tokens": generation_result.output_tokens,
                "reasoning_tokens": generation_result.reasoning_tokens,
                "reasoning_parser": generation_result.reasoning_parser,
                "reasoning_enabled": generation_settings.enable_reasoning,
                "time_to_first_token_seconds": (
                    generation_result.time_to_first_token_seconds
                ),
                "tokens_per_second": (
                    generation_result.tokens_per_second
                    if generation_result.tokens_per_second is not None
                    else (
                        generation_result.output_tokens / generation_seconds
                        if generation_seconds > 0
                        else None
                    )
                ),
                "load_seconds": load_finished_at - started_at,
                "generation_seconds": generation_seconds,
                "total_seconds": generation_finished_at - started_at,
                "file_count": prepared_inputs.file_count,
                "file_characters": prepared_inputs.file_characters,
                "image_count": len(prepared_inputs.images),
                "video_frame_count": (
                    len(prepared_inputs.video.frames)
                    if prepared_inputs.video is not None
                    else 0
                ),
                "memory_total_gib": total_bytes / _BYTES_PER_GIB,
                "memory_available_before_gib": before_free / _BYTES_PER_GIB,
                "memory_available_after_gib": after_free / _BYTES_PER_GIB,
                "resident_profiles": resident_ids,
                "comfy_loaded_model_count": len(comfy_loaded_model_names),
                "comfy_loaded_model_names": comfy_loaded_model_names,
            }
            runtime_metadata = getattr(resident.backend, "runtime_metadata", None)
            if callable(runtime_metadata):
                metadata.update(runtime_metadata())
            if profile.backend == "vllm" and self._vllm_mode_controller is not None:
                metadata.update(
                    {
                        "vllm_execution_setting": (self._vllm_mode_controller.setting),
                        "vllm_auto_promoted": (self._vllm_mode_controller.promoted),
                        "vllm_observed_workflow_count": (
                            self._vllm_mode_controller.observed_workflow_count
                        ),
                    }
                )
            if self.execution_target == "modal":
                metadata.update(
                    {
                        "gpu_total_gib": total_bytes / _BYTES_PER_GIB,
                        "gpu_free_before_gib": before_free / _BYTES_PER_GIB,
                        "gpu_free_after_gib": after_free / _BYTES_PER_GIB,
                    }
                )
            logger.info("Completed LLM inference: %s", metadata)
            return LLMInferenceResult(
                text=generation_result.text,
                metadata=metadata,
                reasoning=generation_result.reasoning,
            )

    def prewarm(
        self,
        *,
        profile: LLMModelProfile,
        reserve_free_vram_gb: float,
        representative_request_count: int,
        workflow_execution_id: str | None = None,
    ) -> dict[str, Any]:
        """Load one profile and execute bounded requests that populate compiler caches."""
        if representative_request_count <= 0:
            raise ValueError("representative_request_count must be positive.")

        def report_progress(event: LLMProgressEvent) -> None:
            """Log warmup progress without emitting synthetic ComfyUI node events."""
            logger.info(
                "LLM prewarm profile=%s stage=%s message=%s.",
                profile.profile_id,
                event.stage,
                event.message,
            )

        started_at = time.perf_counter()
        request_timings: list[float] = []
        cache_hits: list[bool] = []
        for request_index in range(representative_request_count):
            request_started_at = time.perf_counter()
            prepared_inputs = _representative_prewarm_inputs(profile, request_index)
            result = self.infer(
                profile=profile,
                prepared_inputs=prepared_inputs,
                generation_settings=LLMGenerationSettings(
                    max_new_tokens=1,
                    temperature=0.0,
                    top_p=1.0,
                    seed=request_index,
                    enable_reasoning=False,
                ),
                reserve_free_vram_gb=reserve_free_vram_gb,
                keep_model_loaded=True,
                progress_callback=report_progress,
                workflow_execution_id=workflow_execution_id,
            )
            request_timings.append(time.perf_counter() - request_started_at)
            cache_hits.append(bool(result.metadata.get("cache_hit")))
        return {
            "profile_id": profile.profile_id,
            "representative_request_count": representative_request_count,
            "request_seconds": request_timings,
            "cache_hits": cache_hits,
            "elapsed_seconds": time.perf_counter() - started_at,
        }

    def resident_profiles(self) -> tuple[str, ...]:
        """Return profile ids in least-to-most-recently-used order."""
        with self._lock:
            return tuple(self._models)

    def unload_all(self) -> None:
        """Release every resident LLM backend."""
        with self._lock:
            for profile_id in list(self._models):
                self._evict(profile_id)


_RESIDENT_MANAGER: ResidentLLMManager | None = None
_RESIDENT_MANAGER_LOCK = threading.Lock()


def _read_float_environment(name: str, default: float) -> float:
    """Read one non-negative float environment setting."""
    raw_value = os.getenv(name)
    value = default if raw_value is None else float(raw_value)
    if value < 0:
        raise ValueError(f"{name} cannot be negative.")
    return value


def _read_positive_int_environment(name: str, default: int) -> int:
    """Read one positive integer environment setting."""
    raw_value = os.getenv(name)
    value = default if raw_value is None else int(raw_value)
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def get_resident_llm_manager() -> ResidentLLMManager:
    """Return the process-global manager that survives warm prompt executions."""
    global _RESIDENT_MANAGER
    with _RESIDENT_MANAGER_LOCK:
        if _RESIDENT_MANAGER is None:
            execution_target = os.getenv(
                "COMFY_MODAL_LLM_EXECUTION_TARGET",
                "modal",
            )
            _RESIDENT_MANAGER = ResidentLLMManager(
                storage_root=os.getenv(
                    "COMFY_MODAL_REMOTE_STORAGE_ROOT", _DEFAULT_STORAGE_ROOT
                ),
                max_resident_models=_read_positive_int_environment(
                    "COMFY_MODAL_LLM_MAX_RESIDENT_MODELS",
                    _DEFAULT_MAX_RESIDENT_MODELS,
                ),
                memory_recovery_timeout_seconds=_read_float_environment(
                    "COMFY_MODAL_LLM_MEMORY_RECOVERY_TIMEOUT_SECONDS",
                    _DEFAULT_MEMORY_RECOVERY_TIMEOUT_SECONDS,
                ),
                execution_target=execution_target,
                device_name=(
                    "cuda (SSH Docker)"
                    if execution_target == "ssh_docker"
                    else "cuda"
                ),
            )
        return _RESIDENT_MANAGER


def _current_workflow_execution_id() -> str | None:
    """Return ComfyUI's current prompt id when inference runs inside a workflow."""
    try:
        from comfy_execution.utils import get_executing_context
    except ImportError:
        return None
    context = get_executing_context()
    if context is None:
        return None
    prompt_id = str(context.prompt_id).strip()
    return prompt_id or None


def _representative_prewarm_inputs(
    profile: LLMModelProfile,
    request_index: int,
) -> PreparedLLMInputs:
    """Build a small deterministic request covering text and supported vision shapes."""
    images: tuple[Image.Image, ...] = ()
    if "image" in profile.modalities and request_index > 0:
        side_length = 512 if request_index == 1 else 1024
        images = (Image.new("RGB", (side_length, side_length), color=(127, 127, 127)),)
    return PreparedLLMInputs(
        prompt="Reply with OK.",
        system_prompt="This is a runtime compilation warmup request.",
        images=images,
        video=None,
        file_characters=0,
        file_count=0,
    )


def prewarm_modal_llm_profile(
    *,
    model_profile: str,
    reserve_free_vram_gb: float | None = None,
    representative_request_count: int = 3,
    workflow_execution_id: str | None = None,
) -> dict[str, Any]:
    """Load one staged Modal LLM and populate its reusable compiler caches."""
    storage_root = os.getenv("COMFY_MODAL_REMOTE_STORAGE_ROOT", _DEFAULT_STORAGE_ROOT)
    profile = get_llm_profile(model_profile, storage_root=storage_root)
    reserve_gb = (
        _read_float_environment(
            "COMFY_MODAL_LLM_RESERVE_FREE_GB",
            _DEFAULT_RESERVE_FREE_VRAM_GB,
        )
        if reserve_free_vram_gb is None
        else float(reserve_free_vram_gb)
    )
    return get_resident_llm_manager().prewarm(
        profile=profile,
        reserve_free_vram_gb=reserve_gb,
        representative_request_count=representative_request_count,
        workflow_execution_id=workflow_execution_id,
    )


def run_modal_llm_inference(
    *,
    prompt: str,
    model_profile: str,
    images: Any | None,
    video: Any | None,
    files: Sequence[Any] | None,
    system_prompt: str,
    enable_reasoning: bool,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    video_frames: int,
    reserve_free_vram_gb: float | None,
    keep_model_loaded: bool,
    progress_callback: LLMProgressCallback,
) -> LLMInferenceResult:
    """Resolve a profile, normalize content, and invoke the resident manager."""
    storage_root = os.getenv("COMFY_MODAL_REMOTE_STORAGE_ROOT", _DEFAULT_STORAGE_ROOT)
    progress_callback(
        LLMProgressEvent(
            stage="profile",
            message="Resolving model profile",
            indeterminate=True,
        )
    )
    profile = get_llm_profile(model_profile, storage_root=storage_root)
    generation_settings = LLMGenerationSettings(
        max_new_tokens=_coerce_positive_int(max_new_tokens, "max_new_tokens", 32768),
        temperature=float(temperature),
        top_p=float(top_p),
        seed=int(seed),
        enable_reasoning=bool(enable_reasoning),
    )
    if not 0.0 <= generation_settings.temperature <= 2.0:
        raise ValueError("temperature must be between 0.0 and 2.0.")
    if not 0.0 < generation_settings.top_p <= 1.0:
        raise ValueError("top_p must be greater than 0.0 and at most 1.0.")
    progress_callback(
        LLMProgressEvent(
            stage="inputs",
            message="Preparing multimodal inputs",
            indeterminate=True,
        )
    )
    prepared_inputs = prepare_llm_inputs(
        prompt=prompt,
        system_prompt=system_prompt,
        images=images,
        video=video,
        files=files,
        video_frames=video_frames,
        profile=profile,
    )
    reserve_gb = (
        _read_float_environment(
            "COMFY_MODAL_LLM_RESERVE_FREE_GB",
            _DEFAULT_RESERVE_FREE_VRAM_GB,
        )
        if reserve_free_vram_gb is None
        else float(reserve_free_vram_gb)
    )
    return get_resident_llm_manager().infer(
        profile=profile,
        prepared_inputs=prepared_inputs,
        generation_settings=generation_settings,
        reserve_free_vram_gb=reserve_gb,
        keep_model_loaded=keep_model_loaded,
        progress_callback=progress_callback,
        workflow_execution_id=_current_workflow_execution_id(),
    )


__all__ = [
    "BackendGenerationResult",
    "LLMGenerationSettings",
    "LLMInferenceResult",
    "LLMProgressEvent",
    "PreparedLLMInputs",
    "PreparedVideo",
    "prewarm_modal_llm_profile",
    "ResidentLLMManager",
    "TransformersMultimodalBackend",
    "VLLMMultimodalBackend",
    "extract_file_context",
    "get_resident_llm_manager",
    "get_vllm_execution_mode_controller",
    "force_modal_vllm_throughput_after_memory_recovery",
    "observe_modal_workflow_execution",
    "prepare_images",
    "prepare_llm_inputs",
    "prepare_video",
    "run_modal_llm_inference",
]
