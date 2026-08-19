"""Apple Silicon local inference for the Modal LLM ComfyUI node."""

from __future__ import annotations

import gc
from importlib import metadata
import logging
import os
from pathlib import Path
import platform
import sys
import threading
import time
from typing import Any, Callable, Sequence

if __package__:
    from .llm_compatibility import LOCAL_MLX_VLM_VERSION
    from .llm_profiles import (
        LLMModelProfile,
        get_llm_profile,
        profile_for_execution_target,
    )
    from .llm_reasoning import (
        ReasoningOutputParser,
        create_reasoning_parser,
        reasoning_chat_template_kwargs,
    )
    from .llm_resolver import HuggingFaceModelReference, resolve_model_profile
    from .llm_staging import LLMStagingProgress, stage_model_profile
    from .modal_llm_runtime import (
        BackendGenerationResult,
        LLMGenerationSettings,
        LLMInferenceResult,
        LLMProgressCallback,
        LLMProgressEvent,
        PreparedLLMInputs,
        ResidentLLMManager,
        _coerce_positive_int,
        prepare_llm_inputs,
    )
else:  # pragma: no cover - ComfyUI loads this module as part of the package.
    from llm_compatibility import LOCAL_MLX_VLM_VERSION
    from llm_profiles import (
        LLMModelProfile,
        get_llm_profile,
        profile_for_execution_target,
    )
    from llm_reasoning import (
        ReasoningOutputParser,
        create_reasoning_parser,
        reasoning_chat_template_kwargs,
    )
    from llm_resolver import HuggingFaceModelReference, resolve_model_profile
    from llm_staging import LLMStagingProgress, stage_model_profile
    from modal_llm_runtime import (
        BackendGenerationResult,
        LLMGenerationSettings,
        LLMInferenceResult,
        LLMProgressCallback,
        LLMProgressEvent,
        PreparedLLMInputs,
        ResidentLLMManager,
        _coerce_positive_int,
        prepare_llm_inputs,
    )

logger = logging.getLogger(__name__)

_BYTES_PER_GIB = 1024**3
_LOCAL_STORAGE_ENV = "COMFY_MODAL_LOCAL_LLM_STORAGE_ROOT"
_LOCAL_RESERVE_ENV = "COMFY_MODAL_LOCAL_LLM_RESERVE_FREE_GB"
_LOCAL_RESIDENT_MODELS_ENV = "COMFY_MODAL_LOCAL_LLM_MAX_RESIDENT_MODELS"
_DEFAULT_LOCAL_RESERVE_GB = 4.0
_DEFAULT_LOCAL_RESIDENT_MODELS = 1
LOCAL_MLX_VLM_SPEC = f"mlx-vlm=={LOCAL_MLX_VLM_VERSION}"


def _local_install_command() -> str:
    """Return the exact command that installs the validated local backend."""
    return (
        f"uv pip install --python {sys.executable!r} "
        f"{LOCAL_MLX_VLM_SPEC!r} 'psutil>=7,<8'"
    )


def ensure_local_apple_runtime_available() -> None:
    """Fail clearly unless the validated MLX-VLM runtime is usable."""
    system = platform.system()
    machine = platform.machine().lower()
    if system != "Darwin" or machine not in {"arm64", "aarch64"}:
        raise RuntimeError(
            "Local Modal LLM inference currently requires Apple Silicon macOS. "
            f"Detected {system} {machine}. Enable 'Run on Modal' on this node for "
            "the supported remote backend."
        )
    try:
        installed_version = metadata.version("mlx-vlm")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "Apple-local LLM inference requires the optional MLX-VLM backend. "
            f"Install it into ComfyUI's Python environment and restart ComfyUI: "
            f"{_local_install_command()}"
        ) from exc
    if installed_version != LOCAL_MLX_VLM_VERSION:
        raise RuntimeError(
            f"Apple-local LLM inference requires {LOCAL_MLX_VLM_SPEC}, but "
            f"mlx-vlm=={installed_version} is installed. Run: "
            f"{_local_install_command()}"
        )
    try:
        __import__("mlx.core")
        __import__("mlx_vlm")
    except ModuleNotFoundError as exc:
        if exc.name not in {"mlx", "mlx.core", "mlx_vlm"}:
            raise
        raise RuntimeError(
            "The Apple-local MLX-VLM backend is incomplete. Reinstall it and "
            f"restart ComfyUI: {_local_install_command()}"
        ) from exc


def local_llm_storage_root() -> Path:
    """Return the writable ComfyUI model root for immutable local LLM snapshots."""
    configured = os.getenv(_LOCAL_STORAGE_ENV)
    if configured is not None:
        normalized = configured.strip()
        if not normalized:
            raise ValueError(f"{_LOCAL_STORAGE_ENV} must not be blank.")
        return Path(normalized).expanduser().resolve()
    try:
        import folder_paths
    except ModuleNotFoundError as exc:
        if exc.name != "folder_paths":
            raise
        raise RuntimeError(
            f"Unable to locate ComfyUI's model directory. Set {_LOCAL_STORAGE_ENV} "
            "to a writable directory."
        ) from exc
    default_path = Path(folder_paths.models_dir).resolve() / "modal_llm"
    folder_paths.add_model_folder_path(
        "modal_llm",
        str(default_path),
        is_default=True,
    )
    return Path(folder_paths.get_folder_paths("modal_llm")[0]).resolve()


def _read_non_negative_float(name: str, default: float) -> float:
    """Read one non-negative floating-point environment value."""
    raw_value = os.getenv(name)
    value = default if raw_value is None else float(raw_value)
    if value < 0:
        raise ValueError(f"{name} cannot be negative.")
    return value


def _read_positive_int(name: str, default: int) -> int:
    """Read one positive integer environment value."""
    raw_value = os.getenv(name)
    value = default if raw_value is None else int(raw_value)
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def _apple_memory_info() -> tuple[int, int]:
    """Return system-available and total unified memory in bytes."""
    import psutil

    memory = psutil.virtual_memory()
    return int(memory.available), int(memory.total)


def _clear_apple_caches() -> None:
    """Release MLX cache buffers after a resident-model eviction."""
    import mlx.core as mx

    mx.clear_cache()
    gc.collect()


def _stage_progress_callback(
    progress_callback: LLMProgressCallback,
) -> Callable[[LLMStagingProgress], None]:
    """Translate storage progress into the shared node progress contract."""

    def report(progress: LLMStagingProgress) -> None:
        """Forward one local snapshot event."""
        progress_callback(
            LLMProgressEvent(
                stage=progress.stage,
                message=progress.message,
                value=progress.value,
                maximum=progress.maximum,
                unit=progress.unit,
                indeterminate=progress.indeterminate,
            )
        )

    return report


def resolve_and_stage_local_profile(
    model_reference: str,
    storage_root: str | Path,
    *,
    progress_callback: LLMProgressCallback,
) -> LLMModelProfile:
    """Resolve one local model reference and stage its pinned snapshot."""
    normalized_reference = model_reference.strip()
    if not normalized_reference:
        raise ValueError("Modal LLM model_profile must not be blank.")
    try:
        profile = get_llm_profile(normalized_reference, storage_root=storage_root)
    except ValueError:
        if normalized_reference.startswith("hf-"):
            raise
        HuggingFaceModelReference.parse(normalized_reference)
        progress_callback(
            LLMProgressEvent(
                stage="profile",
                message="Inspecting Apple-local model compatibility",
                indeterminate=True,
            )
        )
        profile = resolve_model_profile(
            normalized_reference,
            storage_root,
            execution_target="local_apple",
        ).profile
    else:
        profile = profile_for_execution_target(profile, "local_apple")
    stage_model_profile(
        profile.profile_id,
        storage_root,
        profile=profile,
        progress_callback=_stage_progress_callback(progress_callback),
    )
    return profile


class MLXVLMBackend:
    """Run a pinned multimodal model locally through MLX-VLM."""

    def __init__(
        self,
        profile: LLMModelProfile,
        snapshot_path: Path,
        progress_callback: LLMProgressCallback,
    ) -> None:
        """Load processor and weights from one immutable local snapshot."""
        ensure_local_apple_runtime_available()
        import mlx.core as mx
        from mlx_vlm import load

        if profile.backend != "mlx_vlm":
            raise ValueError(
                f"Apple-local profile {profile.profile_id!r} selects backend "
                f"{profile.backend!r}, not 'mlx_vlm'."
            )
        self.profile = profile
        self.snapshot_path = snapshot_path
        progress_callback(
            LLMProgressEvent(
                stage="processor",
                message="Loading MLX processor",
                indeterminate=True,
            )
        )
        progress_callback(
            LLMProgressEvent(
                stage="weights",
                message="Loading model into unified memory",
                indeterminate=True,
            )
        )
        mx.reset_peak_memory()
        self.model, self.processor = load(
            str(snapshot_path),
            lazy=False,
            strict=True,
        )
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self.reasoning_parser: ReasoningOutputParser = create_reasoning_parser(
            profile,
            tokenizer,
        )
        progress_callback(
            LLMProgressEvent(
                stage="weights",
                message="MLX model ready",
                value=1,
                maximum=1,
                unit="model",
            )
        )

    @staticmethod
    def _media(prepared_inputs: PreparedLLMInputs) -> list[Any]:
        """Return PIL frames that MLX-VLM should treat as image inputs."""
        media = list(prepared_inputs.images)
        if prepared_inputs.video is not None:
            media.extend(prepared_inputs.video.frames)
        return media

    @staticmethod
    def _messages(prepared_inputs: PreparedLLMInputs) -> list[dict[str, str]]:
        """Build system and user messages without embedding binary media."""
        messages: list[dict[str, str]] = []
        if prepared_inputs.system_prompt:
            messages.append(
                {"role": "system", "content": prepared_inputs.system_prompt}
            )
        messages.append({"role": "user", "content": prepared_inputs.prompt})
        return messages

    def generate(
        self,
        prepared_inputs: PreparedLLMInputs,
        settings: LLMGenerationSettings,
        progress_callback: LLMProgressCallback,
    ) -> BackendGenerationResult:
        """Stream one MLX generation while preserving reasoning token IDs."""
        from mlx_vlm.generate import stream_generate
        from mlx_vlm.prompt_utils import apply_chat_template

        media = self._media(prepared_inputs)
        thinking_options = reasoning_chat_template_kwargs(self.profile)
        generation_options: dict[str, bool | int] = dict(thinking_options)
        if thinking_options.get("enable_thinking"):
            generation_options["thinking_budget"] = settings.max_new_tokens // 2
        prompt = apply_chat_template(
            self.processor,
            self.model.config,
            self._messages(prepared_inputs),
            num_images=len(media),
            **thinking_options,
        )
        started_at = time.perf_counter()
        first_token_at: float | None = None
        raw_segments: list[str] = []
        token_ids: list[int] = []
        input_tokens = 0
        output_tokens = 0
        tokens_per_second: float | None = None
        for generation in stream_generate(
            self.model,
            self.processor,
            prompt,
            image=media or None,
            max_tokens=settings.max_new_tokens,
            temperature=settings.temperature,
            top_p=settings.top_p,
            seed=settings.seed,
            skip_special_tokens=False,
            **generation_options,
        ):
            now = time.perf_counter()
            input_tokens = int(generation.prompt_tokens)
            output_tokens = int(generation.generation_tokens)
            if input_tokens + settings.max_new_tokens > self.profile.max_context_tokens:
                raise ValueError(
                    f"Modal LLM request requires up to "
                    f"{input_tokens + settings.max_new_tokens} tokens; profile "
                    f"{self.profile.profile_id!r} is capped at "
                    f"{self.profile.max_context_tokens}."
                )
            if generation.token is not None:
                if first_token_at is None:
                    first_token_at = now
                token = generation.token
                token_ids.append(int(token.item() if hasattr(token, "item") else token))
            raw_segments.append(str(generation.text or ""))
            tokens_per_second = float(generation.generation_tps or 0.0) or None
            progress_callback(
                LLMProgressEvent(
                    stage="generating",
                    message="Generating locally with MLX",
                    value=output_tokens,
                    maximum=settings.max_new_tokens,
                    unit="tokens",
                    elapsed_seconds=now - started_at,
                    time_to_first_token_seconds=(
                        first_token_at - started_at
                        if first_token_at is not None
                        else None
                    ),
                    tokens_per_second=tokens_per_second,
                )
            )
        raw_text = "".join(raw_segments).strip()
        reasoning_output = self.reasoning_parser.extract(raw_text, token_ids)
        return BackendGenerationResult(
            text=reasoning_output.response,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning=reasoning_output.reasoning,
            reasoning_tokens=reasoning_output.reasoning_tokens,
            reasoning_parser=reasoning_output.parser,
            time_to_first_token_seconds=(
                first_token_at - started_at if first_token_at is not None else None
            ),
            tokens_per_second=tokens_per_second,
        )

    def unload(self) -> None:
        """Release model references and reclaim MLX cache buffers."""
        model = self.model
        processor = self.processor
        self.model = None
        self.processor = None
        del model, processor
        _clear_apple_caches()


_LOCAL_MANAGER: ResidentLLMManager | None = None
_LOCAL_MANAGER_ROOT: Path | None = None
_LOCAL_MANAGER_LOCK = threading.Lock()


def get_local_resident_llm_manager(storage_root: str | Path) -> ResidentLLMManager:
    """Return the process-global Apple-local resident model manager."""
    global _LOCAL_MANAGER, _LOCAL_MANAGER_ROOT
    resolved_root = Path(storage_root).resolve()
    with _LOCAL_MANAGER_LOCK:
        if _LOCAL_MANAGER is None or _LOCAL_MANAGER_ROOT != resolved_root:
            if _LOCAL_MANAGER is not None:
                _LOCAL_MANAGER.unload_all()
            _LOCAL_MANAGER = ResidentLLMManager(
                storage_root=resolved_root,
                backend_factory=MLXVLMBackend,
                max_resident_models=_read_positive_int(
                    _LOCAL_RESIDENT_MODELS_ENV,
                    _DEFAULT_LOCAL_RESIDENT_MODELS,
                ),
                memory_info=_apple_memory_info,
                empty_cache=_clear_apple_caches,
                execution_target="local_apple",
                device_name="metal",
                memory_label="unified memory",
            )
            _LOCAL_MANAGER_ROOT = resolved_root
        return _LOCAL_MANAGER


def run_local_llm_inference(
    *,
    prompt: str,
    model_profile: str,
    images: Any | None,
    video: Any | None,
    files: Sequence[Any] | None,
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    video_frames: int,
    reserve_free_vram_gb: float | None,
    keep_model_loaded: bool,
    progress_callback: LLMProgressCallback,
) -> LLMInferenceResult:
    """Resolve, stage, and execute one Apple-local LLM request."""
    ensure_local_apple_runtime_available()
    storage_root = local_llm_storage_root()
    profile = resolve_and_stage_local_profile(
        model_profile,
        storage_root,
        progress_callback=progress_callback,
    )
    settings = LLMGenerationSettings(
        max_new_tokens=_coerce_positive_int(
            max_new_tokens,
            "max_new_tokens",
            32768,
        ),
        temperature=float(temperature),
        top_p=float(top_p),
        seed=int(seed),
    )
    if not 0.0 <= settings.temperature <= 2.0:
        raise ValueError("temperature must be between 0.0 and 2.0.")
    if not 0.0 < settings.top_p <= 1.0:
        raise ValueError("top_p must be greater than 0.0 and at most 1.0.")
    progress_callback(
        LLMProgressEvent(
            stage="inputs",
            message="Preparing local multimodal inputs",
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
        _read_non_negative_float(_LOCAL_RESERVE_ENV, _DEFAULT_LOCAL_RESERVE_GB)
        if reserve_free_vram_gb is None
        else float(reserve_free_vram_gb)
    )
    return get_local_resident_llm_manager(storage_root).infer(
        profile=profile,
        prepared_inputs=prepared_inputs,
        generation_settings=settings,
        reserve_free_vram_gb=reserve_gb,
        keep_model_loaded=keep_model_loaded,
        progress_callback=progress_callback,
    )


__all__ = [
    "LOCAL_MLX_VLM_SPEC",
    "MLXVLMBackend",
    "ensure_local_apple_runtime_available",
    "get_local_resident_llm_manager",
    "local_llm_storage_root",
    "resolve_and_stage_local_profile",
    "run_local_llm_inference",
]
