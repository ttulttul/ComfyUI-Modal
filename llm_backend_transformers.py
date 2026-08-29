"""Transformers backend for resident multimodal LLM inference."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import time
from typing import Any, Mapping

if __package__:
    from .llm_inputs import (
        apply_multimodal_chat_template as _apply_multimodal_chat_template,
        multimodal_messages as _multimodal_messages,
    )
    from .llm_profiles import LLMModelProfile
    from .llm_reasoning import (
        ReasoningOutputParser,
        create_reasoning_parser,
        reasoning_chat_template_kwargs,
        reasoning_parser_for_request,
    )
    from .llm_types import (
        BackendGenerationResult,
        LLMGenerationSettings,
        LLMProgressCallback,
        LLMProgressEvent,
        PreparedLLMInputs,
    )
else:  # pragma: no cover - remote node bundles may import top-level modules.
    from llm_inputs import (
        apply_multimodal_chat_template as _apply_multimodal_chat_template,
        multimodal_messages as _multimodal_messages,
    )
    from llm_profiles import LLMModelProfile
    from llm_reasoning import (
        ReasoningOutputParser,
        create_reasoning_parser,
        reasoning_chat_template_kwargs,
        reasoning_parser_for_request,
    )
    from llm_types import (
        BackendGenerationResult,
        LLMGenerationSettings,
        LLMProgressCallback,
        LLMProgressEvent,
        PreparedLLMInputs,
    )

logger = logging.getLogger(__name__)

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



