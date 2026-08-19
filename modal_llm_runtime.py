"""Resident multimodal Transformers inference inside a Modal ComfyUI worker."""

from __future__ import annotations

import asyncio
import base64
import binascii
import gc
import json
import logging
import math
import os
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Coroutine, Mapping, Protocol, Sequence

from PIL import Image

if __package__:
    from .llm_profiles import LLMModelProfile, get_llm_profile
    from .llm_reasoning import (
        ReasoningOutputParser,
        create_reasoning_parser,
        reasoning_chat_template_kwargs,
        reasoning_parser_for_request,
    )
    from .llm_staging import is_model_snapshot_staged, model_snapshot_path
else:  # pragma: no cover - remote node bundles may import top-level modules.
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
        raise TypeError(f"Expected a torch.Tensor image frame, got {type(frame).__name__}.")
    if frame.ndim != 3 or frame.shape[-1] not in {1, 3, 4}:
        raise ValueError(
            "Modal LLM images must use ComfyUI's [height, width, channels] frame layout."
        )
    normalized = frame.detach().to(device="cpu", dtype=torch.float32).clamp(0.0, 1.0)
    pixels = (normalized * 255.0).round().to(dtype=torch.uint8).numpy()
    if pixels.shape[-1] == 1:
        pixels = pixels.repeat(3, axis=2)
    return Image.fromarray(pixels).convert("RGB")


def prepare_images(images: Any | None, profile: LLMModelProfile) -> tuple[Image.Image, ...]:
    """Normalize an optional ComfyUI IMAGE batch under the profile limit."""
    if images is None:
        return ()
    if "image" not in profile.modalities:
        raise ValueError(f"Model profile {profile.profile_id!r} does not support images.")
    if getattr(images, "ndim", None) != 4:
        raise ValueError("Modal LLM images must be a ComfyUI [batch, height, width, channels] tensor.")
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
        raise ValueError(f"Model profile {profile.profile_id!r} does not support video.")
    frame_limit = min(
        _coerce_positive_int(requested_frames, "video_frames", profile.max_video_frames),
        profile.max_video_frames,
    )
    components = video.get_components()
    frames = components.images
    if getattr(frames, "ndim", None) != 4:
        raise ValueError("The ComfyUI VIDEO input did not decode to a frame batch.")
    indices = _uniform_sample_indices(int(frames.shape[0]), frame_limit)
    frame_rate = float(components.frame_rate)
    if not math.isfinite(frame_rate) or frame_rate <= 0:
        raise ValueError(f"The ComfyUI VIDEO input has invalid frame rate {frame_rate!r}.")
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
        raise ValueError(f"Modal LLM file {filename!r} has an invalid data URI.") from exc
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
        raise ValueError(f"Modal LLM file {filename!r} has invalid base64 content.") from exc
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
        raise RuntimeError("PDF input requires the pinned pypdf remote dependency.") from exc
    try:
        reader = PdfReader(BytesIO(raw_bytes))
        extracted_text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
    except (PdfReadError, OSError, ValueError) as exc:
        raise ValueError(f"Unable to extract text from PDF file {filename!r}: {exc}") from exc
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
        raise ValueError(f"Model profile {profile.profile_id!r} does not support files.")
    sections: list[str] = []
    total_bytes = 0
    total_characters = 0
    for file_value in files:
        filename, mime_type, raw_bytes = _decode_input_file(file_value, profile.max_file_bytes)
        total_bytes += len(raw_bytes)
        if total_bytes > profile.max_file_bytes:
            raise ValueError(
                f"Modal LLM files total {total_bytes} bytes; the profile aggregate limit is "
                f"{profile.max_file_bytes} bytes."
            )
        suffix = Path(filename).suffix.lower()
        if suffix == ".pdf" or mime_type == "application/pdf":
            text = _extract_pdf_text(filename, raw_bytes)
        elif suffix in {".txt", ".md", ".csv", ".json"} or mime_type.startswith("text/"):
            try:
                text = raw_bytes.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"Modal LLM text file {filename!r} is not valid UTF-8.") from exc
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
    if prepared_images and prepared_video is not None and not profile.allow_mixed_image_video:
        raise ValueError(
            f"Model profile {profile.profile_id!r} accepts images or video in one request, not both."
        )
    file_context, file_count, file_characters = extract_file_context(files, profile)
    prompt_parts = [prompt]
    if prepared_video is not None:
        timestamps = ", ".join(f"{timestamp:.3f}s" for timestamp in prepared_video.timestamps_seconds)
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
    raise TypeError(f"Unsupported Transformers processor output {type(batch).__name__}.")


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
                "content": [
                    {"type": "text", "text": prepared_inputs.system_prompt}
                ],
            }
        )
    content: list[dict[str, Any]] = [
        {"type": "image", "image": image} for image in prepared_inputs.images
    ]
    if prepared_inputs.video is not None:
        content.append(
            {"type": "video", "video": list(prepared_inputs.video.frames)}
        )
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
            generate_kwargs.update(temperature=settings.temperature, top_p=settings.top_p)
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

        self.profile = profile
        self.snapshot_path = snapshot_path
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
            "max_model_len=%d kv_cache_gib=%.1f shards=%s.",
            self.profile.profile_id,
            self.snapshot_path,
            quantization or "auto",
            int(
                self.profile.backend_option(
                    "max_model_len",
                    self.profile.max_context_tokens,
                )
            ),
            int(self.profile.backend_option("kv_cache_memory_bytes", 0))
            / _BYTES_PER_GIB,
            shard_count,
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
            enforce_eager=bool(
                self.profile.backend_option("enforce_eager", True)
            ),
            disable_custom_all_reduce=True,
            attention_config={"backend": "TRITON_ATTN"},
            generation_config="vllm",
            limit_mm_per_prompt={"image": self.profile.max_images, "video": 1},
        )

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
                    first_token_at - started_at
                    if first_token_at is not None
                    else None
                ),
                tokens_per_second=(
                    output_tokens / elapsed_seconds
                    if output_tokens > 0 and elapsed_seconds > 0
                    else None
                ),
            )
        )

    async def _abort_request(self, request_id: str, engine_error: type[Exception]) -> None:
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
        logger.debug("Unable to inspect ComfyUI model residency after LLM inference: %s", exc)
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
        snapshot_ready: Callable[[str | Path, LLMModelProfile], bool] = is_model_snapshot_staged,
        comfy_memory_release: Callable[[int], None] | None = None,
        execution_target: str = "modal",
        device_name: str = "cuda",
        memory_label: str = "GPU memory",
    ) -> None:
        """Configure the shared model cache and injectable hardware operations."""
        if max_resident_models <= 0:
            raise ValueError("max_resident_models must be positive.")
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
            logger.debug("ComfyUI model memory release was unavailable before LLM load: %s", exc)

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

    def _make_room(self, profile: LLMModelProfile, reserve_free_vram_gb: float) -> None:
        """Evict old LLMs until the new model plus configured reserve can fit."""
        required_bytes = int((profile.estimated_vram_gb + reserve_free_vram_gb) * _BYTES_PER_GIB)
        self._comfy_memory_release(required_bytes)
        free_bytes, total_bytes = self._memory_info()
        while free_bytes < required_bytes and self._models:
            oldest_profile_id = next(iter(self._models))
            self._evict(oldest_profile_id)
            free_bytes, total_bytes = self._memory_info()
        if free_bytes < required_bytes:
            raise RuntimeError(
                f"LLM profile {profile.profile_id!r} needs approximately "
                f"{profile.estimated_vram_gb:.1f} GiB plus {reserve_free_vram_gb:.1f} GiB reserve, "
                f"but only {free_bytes / _BYTES_PER_GIB:.1f} of "
                f"{total_bytes / _BYTES_PER_GIB:.1f} GiB of {self.memory_label} is "
                "available."
            )

    def _load(
        self,
        profile: LLMModelProfile,
        reserve_free_vram_gb: float,
        progress_callback: LLMProgressCallback,
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
        progress_callback(
            LLMProgressEvent(
                stage="memory",
                message=f"Preparing {self.memory_label}",
                indeterminate=True,
            )
        )
        self._make_room(profile, reserve_free_vram_gb)
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
    ) -> LLMInferenceResult:
        """Run one inference while protecting shared resident state."""
        if reserve_free_vram_gb < 0:
            raise ValueError("reserve_free_vram_gb cannot be negative.")
        with self._lock:
            before_free, total_bytes = self._memory_info()
            started_at = time.perf_counter()
            resident, cache_hit = self._load(
                profile,
                reserve_free_vram_gb,
                progress_callback,
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
                    len(prepared_inputs.video.frames) if prepared_inputs.video is not None else 0
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
            _RESIDENT_MANAGER = ResidentLLMManager(
                storage_root=os.getenv("COMFY_MODAL_REMOTE_STORAGE_ROOT", _DEFAULT_STORAGE_ROOT),
                max_resident_models=_read_positive_int_environment(
                    "COMFY_MODAL_LLM_MAX_RESIDENT_MODELS",
                    _DEFAULT_MAX_RESIDENT_MODELS,
                ),
            )
        return _RESIDENT_MANAGER


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
    )


__all__ = [
    "BackendGenerationResult",
    "LLMGenerationSettings",
    "LLMInferenceResult",
    "LLMProgressEvent",
    "PreparedLLMInputs",
    "PreparedVideo",
    "ResidentLLMManager",
    "TransformersMultimodalBackend",
    "VLLMMultimodalBackend",
    "extract_file_context",
    "get_resident_llm_manager",
    "prepare_images",
    "prepare_llm_inputs",
    "prepare_video",
    "run_modal_llm_inference",
]
