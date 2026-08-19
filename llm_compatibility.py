"""Versioned compatibility policy for generated Modal LLM profiles."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Literal, Mapping

logger = logging.getLogger(__name__)

LLM_PROFILE_SCHEMA_VERSION = 2
LLM_COMPATIBILITY_POLICY_VERSION = 1
TRANSFORMERS_RUNTIME_VERSION = "5.15.0"
VLLM_RUNTIME_VERSION = "0.27.1"
VLLM_TORCH_VERSION = "2.13.0"
LOCAL_MLX_VLM_VERSION = "0.6.15"
LLMExecutionTarget = Literal["modal", "local_apple"]


@dataclass(frozen=True)
class LLMCompatibilityDecision:
    """Describe the reviewed runtime choice for one Hugging Face configuration."""

    backend: str
    architecture: str
    dtype: str
    modalities: frozenset[str]
    quantization_method: str
    advertised_context_tokens: int
    default_context_tokens: int
    estimated_vram_gb: float
    reasoning_parser: str
    backend_options: tuple[tuple[str, str | int | float | bool], ...]
    runtime_requirements: tuple[str, ...]


def _configuration_mapping(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """Return one optional nested configuration mapping."""
    value = config.get(name)
    return value if isinstance(value, Mapping) else {}


def _architecture(config: Mapping[str, Any]) -> str:
    """Return the single model architecture declared by a Hugging Face config."""
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or len(architectures) != 1:
        raise ValueError("The model config must declare exactly one architecture.")
    architecture = str(architectures[0]).strip()
    if not architecture:
        raise ValueError("The model config declares an empty architecture.")
    return architecture


def _advertised_context_tokens(config: Mapping[str, Any]) -> int:
    """Return the advertised text context length from a multimodal config."""
    text_config = _configuration_mapping(config, "text_config")
    raw_value = text_config.get(
        "max_position_embeddings", config.get("max_position_embeddings")
    )
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "The model config does not declare a usable context length."
        ) from exc
    if value <= 0:
        raise ValueError("The model config declares a non-positive context length.")
    return value


def _dtype(config: Mapping[str, Any]) -> str:
    """Normalize a Hugging Face dtype into a supported PyTorch spelling."""
    raw_value = str(
        config.get("dtype") or config.get("torch_dtype") or "bfloat16"
    ).lower()
    aliases = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}
    return aliases.get(raw_value, raw_value)


def _quantization_method(config: Mapping[str, Any]) -> str:
    """Return a normalized checkpoint quantization identifier."""
    quantization = _configuration_mapping(config, "quantization_config")
    method = str(quantization.get("quant_method") or "").lower()
    algorithm = str(quantization.get("quant_algo") or "").upper()
    if method == "modelopt" and algorithm == "NVFP4":
        return "modelopt_fp4"
    if method == "fp8":
        return "fp8"
    mlx_quantization = _configuration_mapping(config, "quantization")
    mlx_bits = mlx_quantization.get("bits")
    mlx_mode = str(mlx_quantization.get("mode") or "").strip().lower()
    if mlx_bits:
        return f"mlx_{mlx_mode or 'quantized'}_{int(mlx_bits)}bit"
    return method or "none"


def _estimated_vram_gb(
    artifact_bytes: int, backend: str, quantization_method: str
) -> float:
    """Estimate model plus runtime allocations before the first measured load."""
    artifact_gib = artifact_bytes / 1024**3
    if backend == "mlx_vlm":
        runtime_overhead_gib = 2.0
    else:
        runtime_overhead_gib = 12.0 if backend == "vllm" else 8.0
    multiplier = 1.12 if quantization_method != "none" else 1.08
    return round(artifact_gib * multiplier + runtime_overhead_gib, 1)


def resolve_compatibility(
    config: Mapping[str, Any],
    *,
    artifact_bytes: int,
    execution_target: LLMExecutionTarget = "modal",
) -> LLMCompatibilityDecision:
    """Select a reviewed backend or reject an unknown model before weight download."""
    architecture = _architecture(config)
    dtype = _dtype(config)
    advertised_context_tokens = _advertised_context_tokens(config)
    default_context_tokens = min(advertised_context_tokens, 32768)
    quantization_method = _quantization_method(config)

    if execution_target == "local_apple":
        if quantization_method != "none" and not quantization_method.startswith(
            "mlx_"
        ):
            raise ValueError(
                f"Quantization {quantization_method!r} is not an MLX checkpoint "
                "format supported by the Apple-local LLM policy. Choose an "
                "unquantized or mlx-community conversion of this model."
            )
        if architecture == "SmolVLMForConditionalGeneration":
            modalities = frozenset({"text", "image", "video", "file"})
            reasoning_parser = "none"
        elif architecture == "MuseGlimmerForConditionalGeneration":
            modalities = frozenset({"text", "image", "file"})
            reasoning_parser = "none"
        elif architecture == "Qwen3_5ForConditionalGeneration":
            modalities = frozenset({"text", "image", "video", "file"})
            reasoning_parser = "qwen3"
        else:
            raise ValueError(
                f"Architecture {architecture!r} is not supported by the Apple-local "
                "LLM compatibility policy "
                f"v{LLM_COMPATIBILITY_POLICY_VERSION}. Add and validate an MLX-VLM "
                "adapter before downloading its weights."
            )
        backend = "mlx_vlm"
        backend_options = ()
        requirements = (f"mlx-vlm=={LOCAL_MLX_VLM_VERSION}",)
    elif architecture == "MuseGlimmerForConditionalGeneration":
        if quantization_method != "none":
            raise ValueError(
                "Muse-Glimmer is currently validated only for its unquantized "
                "Transformers checkpoint."
            )
        backend = "transformers"
        reasoning_parser = "none"
        modalities = frozenset({"text", "image", "file"})
        backend_options: tuple[tuple[str, str | int | float | bool], ...] = (
            ("attention_implementation", "sdpa"),
        )
        requirements = (f"transformers=={TRANSFORMERS_RUNTIME_VERSION}",)
    elif architecture == "Qwen3_5ForConditionalGeneration":
        if quantization_method not in {"fp8", "modelopt_fp4", "none"}:
            raise ValueError(
                f"Qwen3.5 quantization {quantization_method!r} is not in "
                "compatibility policy "
                f"v{LLM_COMPATIBILITY_POLICY_VERSION}."
            )
        backend = "vllm"
        reasoning_parser = "qwen3"
        modalities = frozenset({"text", "image", "video", "file"})
        backend_options = (
            ("enforce_eager", True),
            ("kv_cache_memory_bytes", 12 * 1024**3),
            ("max_model_len", default_context_tokens),
            (
                "quantization",
                quantization_method if quantization_method != "none" else "",
            ),
        )
        requirements = (
            f"torch=={VLLM_TORCH_VERSION}",
            f"transformers=={TRANSFORMERS_RUNTIME_VERSION}",
            f"vllm=={VLLM_RUNTIME_VERSION}",
        )
    else:
        raise ValueError(
            f"Architecture {architecture!r} is not supported by Modal LLM "
            "compatibility policy "
            f"v{LLM_COMPATIBILITY_POLICY_VERSION}. Add and validate an adapter "
            "before downloading "
            "its weights."
        )

    decision = LLMCompatibilityDecision(
        backend=backend,
        architecture=architecture,
        dtype=dtype,
        modalities=modalities,
        quantization_method=quantization_method,
        advertised_context_tokens=advertised_context_tokens,
        default_context_tokens=default_context_tokens,
        estimated_vram_gb=_estimated_vram_gb(
            artifact_bytes, backend, quantization_method
        ),
        reasoning_parser=reasoning_parser,
        backend_options=backend_options,
        runtime_requirements=requirements,
    )
    logger.info(
        "Resolved LLM compatibility target=%s architecture=%s backend=%s "
        "quantization=%s policy=%d.",
        execution_target,
        decision.architecture,
        decision.backend,
        decision.quantization_method,
        LLM_COMPATIBILITY_POLICY_VERSION,
    )
    return decision


__all__ = [
    "LLM_COMPATIBILITY_POLICY_VERSION",
    "LLM_PROFILE_SCHEMA_VERSION",
    "LLMCompatibilityDecision",
    "LLMExecutionTarget",
    "LOCAL_MLX_VLM_VERSION",
    "TRANSFORMERS_RUNTIME_VERSION",
    "VLLM_RUNTIME_VERSION",
    "VLLM_TORCH_VERSION",
    "resolve_compatibility",
]
