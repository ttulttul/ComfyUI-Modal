"""Resident multimodal LLM orchestration inside a Modal ComfyUI worker."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Sequence

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
    from .llm_inputs import (
        _decode_input_file,
        _extract_pdf_text,
        _file_field,
        _tensor_frame_to_pil,
        _uniform_sample_indices,
        apply_multimodal_chat_template as _apply_multimodal_chat_template,
        extract_file_context,
        multimodal_messages as _multimodal_messages,
        prepare_images,
        prepare_llm_inputs,
        prepare_video,
    )
    from .llm_backend_llamacpp import LlamaCppServerBackend
    from .llm_backend_transformers import (
        TransformersMultimodalBackend,
        _dtype_from_name,
        _move_batch_to_device,
        _safetensor_shard_count,
        _stopping_criteria,
        _weight_progress_message,
    )
    from .llm_backend_vllm import (
        VLLMMultimodalBackend,
        _AsyncLoopRunner,
        _VLLMStreamState,
    )
    from .llm_types import (
        BackendFactory,
        BackendGenerationResult,
        LLMBackend,
        LLMGenerationSettings,
        LLMInferenceResult,
        LLMProgressCallback,
        LLMProgressEvent,
        PreparedLLMInputs,
        PreparedVideo,
        ResidentModel,
        coerce_positive_int as _coerce_positive_int,
    )
    from .vllm_instrumentation import (
        VLLMExecutionModeController,
        _install_accurate_triton_compile_listener,
        _profile_for_vllm_execution,
        _vllm_execution_policy,
        _vllm_execution_setting,
        _VLLM_RUNTIME_MODE_OPTION,
        force_modal_vllm_throughput_after_memory_recovery,
        get_vllm_execution_mode_controller,
        observe_modal_workflow_execution,
        triton_compile_listener_engine_pids,
        triton_compile_miss_signal_size,
    )
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
    from llm_inputs import (
        _decode_input_file,
        _extract_pdf_text,
        _file_field,
        _tensor_frame_to_pil,
        _uniform_sample_indices,
        apply_multimodal_chat_template as _apply_multimodal_chat_template,
        extract_file_context,
        multimodal_messages as _multimodal_messages,
        prepare_images,
        prepare_llm_inputs,
        prepare_video,
    )
    from llm_backend_llamacpp import LlamaCppServerBackend
    from llm_backend_transformers import (
        TransformersMultimodalBackend,
        _dtype_from_name,
        _move_batch_to_device,
        _safetensor_shard_count,
        _stopping_criteria,
        _weight_progress_message,
    )
    from llm_backend_vllm import (
        VLLMMultimodalBackend,
        _AsyncLoopRunner,
        _VLLMStreamState,
    )
    from llm_types import (
        BackendFactory,
        BackendGenerationResult,
        LLMBackend,
        LLMGenerationSettings,
        LLMInferenceResult,
        LLMProgressCallback,
        LLMProgressEvent,
        PreparedLLMInputs,
        PreparedVideo,
        ResidentModel,
        coerce_positive_int as _coerce_positive_int,
    )
    from vllm_instrumentation import (
        VLLMExecutionModeController,
        _install_accurate_triton_compile_listener,
        _profile_for_vllm_execution,
        _vllm_execution_policy,
        _vllm_execution_setting,
        _VLLM_RUNTIME_MODE_OPTION,
        force_modal_vllm_throughput_after_memory_recovery,
        get_vllm_execution_mode_controller,
        observe_modal_workflow_execution,
        triton_compile_listener_engine_pids,
        triton_compile_miss_signal_size,
    )

logger = logging.getLogger(__name__)

_BYTES_PER_GIB = 1024**3
_DEFAULT_STORAGE_ROOT = "/storage"
_DEFAULT_RESERVE_FREE_VRAM_GB = 24.0
_DEFAULT_MAX_RESIDENT_MODELS = 2
_DEFAULT_MEMORY_RECOVERY_TIMEOUT_SECONDS = 15.0
_DEFAULT_MEMORY_RECOVERY_POLL_INTERVAL_SECONDS = 0.25








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
