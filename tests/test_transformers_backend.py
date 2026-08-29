"""Tests for the transformers backend boundary."""

from __future__ import annotations

from modal_llm_test_support import *  # noqa: F401,F403

def test_transformers_stopping_criteria_propagates_cancellation(
    modal_llm_runtime_module: Any,
) -> None:
    """Per-token criteria should surface the same interruption raised by ComfyUI."""
    calls: list[int] = []

    def cancel_on_second_token(progress: Any) -> None:
        """Raise a cancellation marker on the second generation step."""
        calls.append(int(progress.value))
        if progress.value == 2:
            raise InterruptedError("cancelled")

    criterion = modal_llm_runtime_module._stopping_criteria(cancel_on_second_token)[0]

    assert criterion(None, None) is False
    with pytest.raises(InterruptedError, match="cancelled"):
        criterion(None, None)
    assert calls == [1, 2]

def test_transformers_processor_does_not_resample_predecoded_video(
    modal_llm_runtime_module: Any,
) -> None:
    """Native ComfyUI frames should bypass the processor's metadata-based sampler."""
    calls: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []

    class FakeProcessor:
        """Record the chat-template call without importing a real model processor."""

        def apply_chat_template(
            self,
            messages: list[dict[str, Any]],
            **kwargs: Any,
        ) -> str:
            """Capture processor options and return a tokenization sentinel."""
            calls.append((messages, kwargs))
            return "tokenized"

    messages = [{"role": "user", "content": [{"type": "video", "video": [object()]}]}]

    result = modal_llm_runtime_module._apply_multimodal_chat_template(
        FakeProcessor(),
        messages,
        has_predecoded_video=True,
    )

    assert result == "tokenized"
    assert calls[0][1]["do_sample_frames"] is False
    assert "processor_kwargs" not in calls[0][1]

