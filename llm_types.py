"""Backend-neutral resident LLM value types and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from PIL import Image

if __package__:
    from .llm_profiles import LLMModelProfile
else:  # pragma: no cover - remote node bundles may import top-level modules.
    from llm_profiles import LLMModelProfile


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


def coerce_positive_int(value: int, name: str, maximum: int) -> int:
    """Validate one positive bounded integer generation setting."""
    resolved = int(value)
    if resolved <= 0 or resolved > maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}, got {resolved}.")
    return resolved
