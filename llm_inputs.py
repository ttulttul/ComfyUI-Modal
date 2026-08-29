"""Normalize ComfyUI media and files for backend-neutral LLM inference."""

from __future__ import annotations

import base64
import binascii
from io import BytesIO
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

if __package__:
    from .llm_types import PreparedLLMInputs, PreparedVideo, coerce_positive_int
else:  # pragma: no cover - remote node bundles may import top-level modules.
    from llm_types import PreparedLLMInputs, PreparedVideo, coerce_positive_int

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
        coerce_positive_int(
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


def apply_multimodal_chat_template(
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


def multimodal_messages(
    prepared_inputs: PreparedLLMInputs,
) -> list[dict[str, Any]]:
    """Build processor-native multimodal chat messages for any backend."""
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

