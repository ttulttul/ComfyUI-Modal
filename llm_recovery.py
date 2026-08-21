"""Shared protocol markers for recoverable resident-LLM memory failures."""

from __future__ import annotations


LLM_MEMORY_RECOVERY_EXHAUSTED_MARKER = (
    "comfy-modal-llm-memory-recovery-exhausted"
)
LLM_FORCE_VLLM_THROUGHPUT_PAYLOAD_KEY = (
    "force_vllm_throughput_after_memory_recovery"
)
LLM_VLLM_THROUGHPUT_FAILURE_MARKER = "vllm_mode=throughput"


def is_llm_memory_recovery_exhausted(error: BaseException) -> bool:
    """Return whether a remote error reports exhausted post-eviction recovery."""
    return LLM_MEMORY_RECOVERY_EXHAUSTED_MARKER in str(error).lower()


def exhausted_recovery_used_vllm_throughput(error: BaseException) -> bool:
    """Return whether an exhausted recovery was preparing a throughput engine."""
    return LLM_VLLM_THROUGHPUT_FAILURE_MARKER in str(error).lower()
