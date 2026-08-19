"""ComfyUI V3 node for resident LLM inference in the Modal worker."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from typing import Any

import comfy.model_management
import comfy.utils
from comfy_api.latest import io

if __package__:
    from .llm_profiles import MODAL_LLM_NODE_ID, llm_profile_options
    from .local_llm_runtime import run_local_llm_inference
    from .modal_llm_runtime import LLMProgressEvent, run_modal_llm_inference
else:  # pragma: no cover - the stable remote runtime imports this module top-level.
    from llm_profiles import MODAL_LLM_NODE_ID, llm_profile_options
    from local_llm_runtime import run_local_llm_inference
    from modal_llm_runtime import LLMProgressEvent, run_modal_llm_inference

logger = logging.getLogger(__name__)


def _modal_llm_primary_inputs() -> list[io.Input]:
    """Return the prompt and multimodal inputs shown by default."""
    return [
        io.String.Input(
            "prompt",
            default="",
            multiline=True,
            tooltip="Text instructions or content for the resident Modal language model.",
        ),
        io.String.Input(
            "model_profile",
            default=llm_profile_options()[0],
            tooltip=(
                "A curated profile or Hugging Face ID such as owner/model. The first run "
                "inspects and pins a compatible model for the selected local or Modal target."
            ),
        ),
        io.Image.Input(
            "images",
            optional=True,
            tooltip="Optional ComfyUI IMAGE batch for visual understanding.",
        ),
        io.Video.Input(
            "video",
            optional=True,
            tooltip="Optional native ComfyUI VIDEO sampled into bounded timestamped frames.",
        ),
        io.Custom("OPENAI_INPUT_FILES").Input(
            "files",
            optional=True,
            tooltip="Optional UTF-8 text or PDF files from OpenAI ChatGPT Input Files.",
        ),
        io.String.Input(
            "system_prompt",
            default="",
            multiline=True,
            optional=True,
            tooltip="Optional system instruction for the model.",
        ),
    ]


def _modal_llm_advanced_inputs() -> list[io.Input]:
    """Return bounded generation and residency controls."""
    return [
        io.Combo.Input(
            "local_mlx_engine",
            options=["auto", "mlx-vlm", "mlx-dspark"],
            default="auto",
            advanced=True,
            tooltip=(
                "Local-only engine selection. Auto uses mlx-dspark for a registered "
                "text-only model and MLX-VLM for image/video requests or unsupported "
                "models. Ignored when Run on Modal is enabled."
            ),
        ),
        io.Int.Input(
            "max_new_tokens",
            default=512,
            min=1,
            max=32768,
            step=1,
            advanced=True,
            tooltip="Maximum combined number of reasoning and response tokens.",
        ),
        io.Boolean.Input(
            "enable_reasoning",
            default=True,
            advanced=True,
            tooltip=(
                "Enable thinking for models whose profile supports it. Disable for a "
                "direct response with no reasoning tokens."
            ),
        ),
        io.Float.Input(
            "temperature",
            default=0.2,
            min=0.0,
            max=2.0,
            step=0.05,
            advanced=True,
            tooltip="Sampling temperature. Zero selects deterministic greedy decoding.",
        ),
        io.Float.Input(
            "top_p",
            default=0.95,
            min=0.01,
            max=1.0,
            step=0.01,
            advanced=True,
            tooltip="Nucleus-sampling probability mass when temperature is non-zero.",
        ),
        io.Int.Input(
            "seed",
            default=0,
            min=0,
            max=0x7FFFFFFFFFFFFFFF,
            step=1,
            advanced=True,
            tooltip="Generation seed.",
        ),
        io.Int.Input(
            "video_frames",
            default=12,
            min=1,
            max=32,
            step=1,
            advanced=True,
            tooltip="Number of uniformly spaced video frames to analyze.",
        ),
        io.Float.Input(
            "reserve_free_vram_gb",
            default=24.0,
            min=0.0,
            max=256.0,
            step=1.0,
            advanced=True,
            tooltip=(
                "Accelerator or unified memory kept free for ComfyUI and the operating "
                "system before loading this LLM."
            ),
        ),
        io.Boolean.Input(
            "keep_model_loaded",
            default=True,
            advanced=True,
            tooltip="Keep the model resident for subsequent requests in this process.",
        ),
        io.Float.Input(
            "local_reserve_free_memory_gb",
            default=4.0,
            min=0.0,
            max=256.0,
            step=1.0,
            advanced=True,
            tooltip=(
                "Apple unified memory kept free for ComfyUI and macOS before loading "
                "a local LLM. Ignored when Run on Modal is enabled."
            ),
        ),
    ]


class ModalLLM(io.ComfyNode):
    """Generate text with a resident local or Modal multimodal model."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        """Expose an OpenAI-like multimodal node contract for in-worker inference."""
        return io.Schema(
            node_id=MODAL_LLM_NODE_ID,
            display_name="Modal LLM",
            category="Modal/text",
            essentials_category="Text Generation",
            description=(
                "Run text, image, video, and bounded file understanding through a "
                "revision-pinned language model. Leave Run on Modal disabled for "
                "Apple-local MLX inference, or enable it for a Modal GPU worker."
            ),
            inputs=[*_modal_llm_primary_inputs(), *_modal_llm_advanced_inputs()],
            outputs=[
                io.String.Output(display_name="response"),
                io.String.Output(display_name="metadata_json"),
                io.String.Output(display_name="reasoning"),
            ],
            hidden=[io.Hidden.unique_id],
            not_idempotent=True,
            is_experimental=True,
        )

    @classmethod
    def execute(
        cls,
        prompt: str,
        model_profile: str,
        images: Any | None = None,
        video: Any | None = None,
        files: Sequence[Any] | None = None,
        system_prompt: str = "",
        enable_reasoning: bool = True,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.95,
        seed: int = 0,
        video_frames: int = 12,
        reserve_free_vram_gb: float = 24.0,
        keep_model_loaded: bool = True,
        local_reserve_free_memory_gb: float = 4.0,
        local_mlx_engine: str = "auto",
        unique_id: str | None = None,
    ) -> io.NodeOutput:
        """Run one cancellation-aware resident inference request."""
        is_remote = os.getenv("COMFY_MODAL_REMOTE_WORKER") == "1"
        execution_target = "modal" if is_remote else "local_apple"
        inference_runner = (
            run_modal_llm_inference if is_remote else run_local_llm_inference
        )
        progress_bar = comfy.utils.ProgressBar(max_new_tokens, node_id=unique_id)

        def report_progress(progress: LLMProgressEvent) -> None:
            """Mirror structured LLM progress into ComfyUI and its remote stream."""
            comfy.model_management.throw_exception_if_processing_interrupted()
            progress_value = float(progress.value or 0.0)
            progress_maximum = float(progress.maximum or 1.0)
            progress_bar.update_absolute(progress_value, progress_maximum)
            payload: dict[str, Any] = {
                "node_id": str(unique_id or ""),
                "stage": progress.stage,
                "message": progress.message,
                "value": progress_value,
                "max": progress_maximum,
                "indeterminate": progress.indeterminate,
            }
            optional_fields = {
                "unit": progress.unit,
                "elapsed_seconds": progress.elapsed_seconds,
                "time_to_first_token_seconds": (progress.time_to_first_token_seconds),
                "tokens_per_second": progress.tokens_per_second,
            }
            payload.update(
                {
                    key: value
                    for key, value in optional_fields.items()
                    if value is not None
                }
            )
            try:
                from server import PromptServer
            except ImportError as error:
                logger.debug("ComfyUI progress server is unavailable: %s", error)
            else:
                PromptServer.instance.send_sync("modal_llm_progress", payload, None)

        logger.info(
            "Starting LLM target=%s profile=%s max_new_tokens=%d images=%s "
            "video=%s files=%d.",
            execution_target,
            model_profile,
            max_new_tokens,
            getattr(images, "shape", None),
            video is not None,
            len(files or ()),
        )
        inference_options: dict[str, Any] = {
            "prompt": prompt,
            "model_profile": model_profile,
            "images": images,
            "video": video,
            "files": files,
            "system_prompt": system_prompt,
            "enable_reasoning": enable_reasoning,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "video_frames": video_frames,
            "reserve_free_vram_gb": (
                reserve_free_vram_gb if is_remote else local_reserve_free_memory_gb
            ),
            "keep_model_loaded": keep_model_loaded,
            "progress_callback": report_progress,
        }
        if not is_remote:
            inference_options["local_mlx_engine"] = local_mlx_engine
        result = inference_runner(**inference_options)
        report_progress(
            LLMProgressEvent(
                stage="complete",
                message="Generation complete",
                value=float(result.metadata["output_tokens"]),
                maximum=float(max_new_tokens),
                unit="tokens",
                elapsed_seconds=result.metadata.get("generation_seconds"),
                time_to_first_token_seconds=result.metadata.get(
                    "time_to_first_token_seconds"
                ),
                tokens_per_second=result.metadata.get("tokens_per_second"),
            )
        )
        return io.NodeOutput(
            result.text,
            json.dumps(result.metadata, sort_keys=True, separators=(",", ":")),
            result.reasoning,
        )


__all__ = ["ModalLLM"]
