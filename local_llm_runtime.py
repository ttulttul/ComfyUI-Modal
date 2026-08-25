"""Apple Silicon local inference for the Modal LLM ComfyUI node."""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import platform
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

if __package__:
    from .llm_compatibility import LOCAL_MLX_DSPARK_VERSION, LOCAL_MLX_VLM_VERSION
    from .llm_profiles import (
        LLMModelProfile,
        get_llm_profile,
        profile_for_execution_target,
    )
    from .llm_reasoning import (
        ReasoningOutputParser,
        create_reasoning_parser,
        reasoning_chat_template_kwargs,
        reasoning_parser_for_request,
    )
    from .llm_resolver import (
        HuggingFaceModelReference,
        _field,
        _validate_repository_metadata,
        resolve_model_profile,
    )
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
    from llm_compatibility import LOCAL_MLX_DSPARK_VERSION, LOCAL_MLX_VLM_VERSION
    from llm_profiles import (
        LLMModelProfile,
        get_llm_profile,
        profile_for_execution_target,
    )
    from llm_reasoning import (
        ReasoningOutputParser,
        create_reasoning_parser,
        reasoning_chat_template_kwargs,
        reasoning_parser_for_request,
    )
    from llm_resolver import (
        HuggingFaceModelReference,
        _field,
        _validate_repository_metadata,
        resolve_model_profile,
    )
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
LOCAL_MLX_DSPARK_SPEC = f"mlx-dspark=={LOCAL_MLX_DSPARK_VERSION}"
_LOCAL_MLX_ENGINES = frozenset({"auto", "mlx-vlm", "mlx-dspark"})
_DSPARK_DIRECTORY_NAME = "mlx_dspark_drafters"
_DSPARK_MARKER_FILENAME = ".comfy-modal-mlx-dspark-complete.json"
_DSPARK_SNAPSHOT_ALLOW_PATTERNS = (
    "config.json",
    "*.safetensors",
    "*.safetensors.index.json",
)
_DSPARK_STAGE_LOCK = threading.Lock()


def _local_install_command() -> str:
    """Return the exact command that installs the validated local backend."""
    return (
        f"uv pip install --python {sys.executable!r} "
        f"{LOCAL_MLX_VLM_SPEC!r} {LOCAL_MLX_DSPARK_SPEC!r} 'psutil>=7,<8'"
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


def ensure_local_dspark_runtime_available() -> None:
    """Fail clearly unless the pinned optional mlx-dspark runtime is usable."""
    ensure_local_apple_runtime_available()
    try:
        installed_version = metadata.version("mlx-dspark")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "The selected local MLX engine requires the optional mlx-dspark "
            f"backend. Install it and restart ComfyUI: {_local_install_command()}"
        ) from exc
    if installed_version != LOCAL_MLX_DSPARK_VERSION:
        raise RuntimeError(
            f"The selected local MLX engine requires {LOCAL_MLX_DSPARK_SPEC}, but "
            f"mlx-dspark=={installed_version} is installed. Run: "
            f"{_local_install_command()}"
        )
    try:
        __import__("mlx_dspark")
    except ModuleNotFoundError as exc:
        if exc.name not in {"mlx_dspark", "mlx_lm"}:
            raise
        raise RuntimeError(
            "The local mlx-dspark backend is incomplete. Reinstall it and restart "
            f"ComfyUI: {_local_install_command()}"
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


@dataclass(frozen=True)
class StagedDSparkDrafter:
    """Describe one exact mlx-dspark drafter snapshot stored locally."""

    repository: str
    revision: str
    path: Path


@dataclass(frozen=True)
class LocalMLXEngineSelection:
    """Describe the local engine selected for one prepared request."""

    engine: str
    mode: str | None = None
    drafter_repository: str | None = None


def _dspark_marker_matches(
    snapshot_path: Path,
    repository: str,
    revision: str,
) -> bool:
    """Return whether one complete drafter snapshot has the expected identity."""
    marker_path = snapshot_path / _DSPARK_MARKER_FILENAME
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    return bool(
        isinstance(marker, dict)
        and marker.get("repository") == repository
        and marker.get("revision") == revision
        and (snapshot_path / "config.json").is_file()
    )


def _stage_mlx_dspark_drafter(
    repository: str,
    storage_root: str | Path,
    progress_callback: LLMProgressCallback,
) -> StagedDSparkDrafter:
    """Resolve and stage an exact, security-checked speculative drafter snapshot."""
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.errors import HfHubHTTPError

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    progress_callback(
        LLMProgressEvent(
            stage="profile",
            message="Resolving mlx-dspark drafter revision",
            indeterminate=True,
        )
    )
    try:
        model_info = HfApi().model_info(
            repository,
            files_metadata=True,
            securityStatus=True,
            token=token,
        )
    except (HfHubHTTPError, OSError, ValueError) as exc:
        raise ValueError(
            f"Unable to inspect mlx-dspark drafter {repository!r}. For gated or "
            f"private models, set HF_TOKEN in the local ComfyUI environment: {exc}"
        ) from exc
    revision = str(_field(model_info, "sha", "")).lower()
    if len(revision) != 40 or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError(
            f"Hugging Face did not resolve mlx-dspark drafter {repository!r} "
            "to an exact commit."
        )
    max_download_gb = float(os.getenv("COMFY_MODAL_LLM_MAX_DOWNLOAD_GB", "96"))
    if max_download_gb <= 0:
        raise ValueError("COMFY_MODAL_LLM_MAX_DOWNLOAD_GB must be positive.")
    _validate_repository_metadata(
        model_info,
        max_download_bytes=int(max_download_gb * _BYTES_PER_GIB),
    )
    repository_digest = hashlib.sha256(repository.encode("utf-8")).hexdigest()
    snapshot_path = (
        Path(storage_root).resolve()
        / _DSPARK_DIRECTORY_NAME
        / f"repo-{repository_digest}"
        / revision
    )
    with _DSPARK_STAGE_LOCK:
        if not _dspark_marker_matches(snapshot_path, repository, revision):
            progress_callback(
                LLMProgressEvent(
                    stage="download",
                    message="Downloading immutable mlx-dspark drafter",
                    indeterminate=True,
                )
            )
            snapshot_path.mkdir(parents=True, exist_ok=True)
            try:
                resolved_path = Path(
                    snapshot_download(
                        repo_id=repository,
                        revision=revision,
                        local_dir=str(snapshot_path),
                        token=token,
                        allow_patterns=_DSPARK_SNAPSHOT_ALLOW_PATTERNS,
                    )
                ).resolve()
            except (HfHubHTTPError, OSError, ValueError) as exc:
                raise ValueError(
                    f"Unable to stage mlx-dspark drafter {repository!r} at "
                    f"revision {revision}: {exc}"
                ) from exc
            if resolved_path != snapshot_path.resolve():
                raise RuntimeError(
                    f"Hugging Face staged mlx-dspark drafter {repository!r} at "
                    f"unexpected path {resolved_path}; expected {snapshot_path.resolve()}."
                )
            if not (snapshot_path / "config.json").is_file():
                raise RuntimeError(
                    f"Staged mlx-dspark drafter {repository!r} is missing config.json."
                )
            marker_path = snapshot_path / _DSPARK_MARKER_FILENAME
            temporary_path = marker_path.with_suffix(".tmp")
            temporary_path.write_text(
                json.dumps(
                    {"repository": repository, "revision": revision},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
            os.replace(temporary_path, marker_path)
    return StagedDSparkDrafter(repository, revision, snapshot_path)


def _resolve_mlx_dspark_mode(profile: LLMModelProfile) -> tuple[str, str]:
    """Return a registered accelerated mode and drafter for one target profile."""
    from mlx_dspark import resolve_mode

    mode, _, drafter_repository = resolve_mode(profile.repository, mode="auto")
    if mode not in {"dspark", "dflash"} or not drafter_repository:
        raise ValueError(
            f"mlx-dspark has no registered accelerated drafter for "
            f"{profile.repository!r}."
        )
    return mode, drafter_repository


def select_local_mlx_engine(
    requested_engine: str,
    profile: LLMModelProfile,
    prepared_inputs: PreparedLLMInputs,
) -> LocalMLXEngineSelection:
    """Select MLX-VLM or an accelerated text-only mlx-dspark backend."""
    normalized_engine = requested_engine.strip().lower()
    if normalized_engine not in _LOCAL_MLX_ENGINES:
        supported = ", ".join(sorted(_LOCAL_MLX_ENGINES))
        raise ValueError(
            f"local_mlx_engine must be one of {supported}; got {requested_engine!r}."
        )
    has_media = bool(prepared_inputs.images or prepared_inputs.video is not None)
    if normalized_engine == "mlx-vlm":
        return LocalMLXEngineSelection("mlx-vlm")
    if has_media:
        if normalized_engine == "mlx-dspark":
            raise ValueError(
                "mlx-dspark is text-only and cannot process image or video inputs. "
                "Choose 'auto' or 'mlx-vlm' for this request."
            )
        return LocalMLXEngineSelection("mlx-vlm")
    try:
        ensure_local_dspark_runtime_available()
        mode, drafter_repository = _resolve_mlx_dspark_mode(profile)
    except (RuntimeError, ValueError) as exc:
        if normalized_engine == "mlx-dspark":
            raise
        logger.info(
            "Falling back to MLX-VLM for local profile=%s: %s",
            profile.profile_id,
            exc,
        )
        return LocalMLXEngineSelection("mlx-vlm")
    return LocalMLXEngineSelection(
        "mlx-dspark",
        mode=mode,
        drafter_repository=drafter_repository,
    )


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
        model_reference=normalized_reference,
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
        reasoning_parser = reasoning_parser_for_request(
            self.reasoning_parser,
            settings.enable_reasoning,
        )
        thinking_options = reasoning_chat_template_kwargs(
            self.profile,
            settings.enable_reasoning,
        )
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
            skip_special_tokens=not reasoning_parser.requires_boundary_tokens,
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
        reasoning_output = reasoning_parser.extract(raw_text, token_ids)
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


class MLXDSparkBackend:
    """Run text-only inference through mlx-dspark speculative decoding."""

    def __init__(
        self,
        profile: LLMModelProfile,
        snapshot_path: Path,
        progress_callback: LLMProgressCallback,
    ) -> None:
        """Load an immutable target and its exact registered drafter snapshot."""
        ensure_local_dspark_runtime_available()
        import mlx.core as mx
        from mlx_dspark import load_dflash_pair, load_pair

        self.profile = profile
        self.snapshot_path = snapshot_path
        self.mode, self.drafter_repository = _resolve_mlx_dspark_mode(profile)
        storage_root = snapshot_path.parents[2]
        staged_drafter = _stage_mlx_dspark_drafter(
            self.drafter_repository,
            storage_root,
            progress_callback,
        )
        self.drafter_revision = staged_drafter.revision
        progress_callback(
            LLMProgressEvent(
                stage="weights",
                message=f"Loading MLX target and {self.mode} drafter",
                indeterminate=True,
            )
        )
        mx.reset_peak_memory()
        load_function = load_dflash_pair if self.mode == "dflash" else load_pair
        self.target, self.tokenizer, self.drafter, self.drafter_config = load_function(
            str(snapshot_path),
            drafter=str(staged_drafter.path),
        )
        self.reasoning_parser: ReasoningOutputParser = create_reasoning_parser(
            profile,
            self.tokenizer,
        )
        progress_callback(
            LLMProgressEvent(
                stage="weights",
                message=f"MLX {self.mode} model ready",
                value=1,
                maximum=1,
                unit="model",
            )
        )

    @staticmethod
    def _messages(prepared_inputs: PreparedLLMInputs) -> list[dict[str, str]]:
        """Build the text-only chat transcript for speculative generation."""
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
        """Generate losslessly with DSpark or DFlash while reporting committed tokens."""
        from mlx_dspark import dflash_generate, encode_messages, speculative_generate

        if prepared_inputs.images or prepared_inputs.video is not None:
            raise ValueError(
                "mlx-dspark is text-only and cannot process image or video inputs."
            )
        reasoning_parser = reasoning_parser_for_request(
            self.reasoning_parser,
            settings.enable_reasoning,
        )
        prompt_ids = encode_messages(
            self.tokenizer,
            self._messages(prepared_inputs),
            **reasoning_chat_template_kwargs(
                self.profile,
                settings.enable_reasoning,
            ),
        )
        input_tokens = len(prompt_ids)
        if input_tokens + settings.max_new_tokens > self.profile.max_context_tokens:
            raise ValueError(
                f"Modal LLM request requires up to "
                f"{input_tokens + settings.max_new_tokens} tokens; profile "
                f"{self.profile.profile_id!r} is capped at "
                f"{self.profile.max_context_tokens}."
            )
        started_at = time.perf_counter()
        first_token_at: float | None = None
        committed_tokens = 0

        def on_text(piece: str) -> None:
            """Record time to first decoded text without duplicating final output."""
            nonlocal first_token_at
            if piece and first_token_at is None:
                first_token_at = time.perf_counter()

        def on_round(**round_result: Any) -> None:
            """Translate speculative rounds into the shared token progress contract."""
            nonlocal committed_tokens
            committed_tokens = min(
                settings.max_new_tokens,
                committed_tokens + int(round_result.get("committed", 0)),
            )
            now = time.perf_counter()
            progress_callback(
                LLMProgressEvent(
                    stage="generating",
                    message=f"Generating locally with MLX {self.mode}",
                    value=committed_tokens,
                    maximum=settings.max_new_tokens,
                    unit="tokens",
                    elapsed_seconds=now - started_at,
                    time_to_first_token_seconds=(
                        first_token_at - started_at
                        if first_token_at is not None
                        else None
                    ),
                    tokens_per_second=(
                        committed_tokens / (now - started_at)
                        if committed_tokens and now > started_at
                        else None
                    ),
                )
            )

        generation_options: dict[str, Any] = {
            "prompt_ids": prompt_ids,
            "max_new_tokens": settings.max_new_tokens,
            "temperature": settings.temperature,
            "top_p": settings.top_p,
            "seed": settings.seed,
            "apply_chat_template": False,
            "on_text": on_text,
            "on_round": on_round,
        }
        if self.mode == "dflash":
            generation_options["max_draft_tokens"] = None
            result = dflash_generate(
                self.target,
                self.tokenizer,
                self.drafter,
                **generation_options,
            )
        else:
            result = speculative_generate(
                self.target,
                self.tokenizer,
                self.drafter,
                **generation_options,
            )
        raw_text = str(result.text).strip()
        token_ids = [int(token_id) for token_id in result.token_ids]
        reasoning_output = reasoning_parser.extract(raw_text, token_ids)
        return BackendGenerationResult(
            text=reasoning_output.response,
            input_tokens=input_tokens,
            output_tokens=int(result.num_tokens),
            reasoning=reasoning_output.reasoning,
            reasoning_tokens=reasoning_output.reasoning_tokens,
            reasoning_parser=reasoning_output.parser,
            time_to_first_token_seconds=(
                first_token_at - started_at if first_token_at is not None else None
            ),
            tokens_per_second=float(result.tokens_per_sec),
        )

    def runtime_metadata(self) -> dict[str, Any]:
        """Return the accelerated engine and immutable drafter identity."""
        return {
            "backend": "mlx_dspark",
            "mlx_dspark_version": LOCAL_MLX_DSPARK_VERSION,
            "mlx_dspark_mode": self.mode,
            "mlx_dspark_drafter_repository": self.drafter_repository,
            "mlx_dspark_drafter_revision": self.drafter_revision,
        }

    def unload(self) -> None:
        """Release target and drafter references and reclaim MLX cache buffers."""
        target = self.target
        tokenizer = self.tokenizer
        drafter = self.drafter
        drafter_config = self.drafter_config
        self.target = None
        self.tokenizer = None
        self.drafter = None
        self.drafter_config = None
        del target, tokenizer, drafter, drafter_config
        _clear_apple_caches()


_LOCAL_MANAGERS: dict[tuple[Path, str], ResidentLLMManager] = {}
_LOCAL_MANAGER_LOCK = threading.Lock()


def get_local_resident_llm_manager(
    storage_root: str | Path,
    engine: str = "mlx-vlm",
) -> ResidentLLMManager:
    """Return one process-global Apple-local manager per storage root and engine."""
    resolved_root = Path(storage_root).resolve()
    if engine not in {"mlx-vlm", "mlx-dspark"}:
        raise ValueError(f"Unknown local MLX engine {engine!r}.")
    manager_key = (resolved_root, engine)
    with _LOCAL_MANAGER_LOCK:
        for (manager_root, manager_engine), other_manager in _LOCAL_MANAGERS.items():
            if manager_root == resolved_root and manager_engine != engine:
                other_manager.unload_all()
        manager = _LOCAL_MANAGERS.get(manager_key)
        if manager is None:
            backend_factory = (
                MLXDSparkBackend if engine == "mlx-dspark" else MLXVLMBackend
            )
            manager = ResidentLLMManager(
                storage_root=resolved_root,
                backend_factory=backend_factory,
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
            _LOCAL_MANAGERS[manager_key] = manager
        return manager


def run_local_llm_inference(
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
    local_mlx_engine: str = "auto",
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
        enable_reasoning=bool(enable_reasoning),
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
    selection = select_local_mlx_engine(
        local_mlx_engine,
        profile,
        prepared_inputs,
    )
    progress_callback(
        LLMProgressEvent(
            stage="engine",
            message=(
                f"Selected local MLX {selection.mode} acceleration"
                if selection.mode is not None
                else "Selected local MLX-VLM inference"
            ),
            value=1,
            maximum=1,
            unit="engine",
        )
    )
    reserve_gb = (
        _read_non_negative_float(_LOCAL_RESERVE_ENV, _DEFAULT_LOCAL_RESERVE_GB)
        if reserve_free_vram_gb is None
        else float(reserve_free_vram_gb)
    )
    return get_local_resident_llm_manager(storage_root, selection.engine).infer(
        profile=profile,
        prepared_inputs=prepared_inputs,
        generation_settings=settings,
        reserve_free_vram_gb=reserve_gb,
        keep_model_loaded=keep_model_loaded,
        progress_callback=progress_callback,
    )


__all__ = [
    "LOCAL_MLX_DSPARK_SPEC",
    "LOCAL_MLX_VLM_SPEC",
    "LocalMLXEngineSelection",
    "MLXDSparkBackend",
    "MLXVLMBackend",
    "ensure_local_apple_runtime_available",
    "ensure_local_dspark_runtime_available",
    "get_local_resident_llm_manager",
    "local_llm_storage_root",
    "resolve_and_stage_local_profile",
    "run_local_llm_inference",
    "select_local_mlx_engine",
]
