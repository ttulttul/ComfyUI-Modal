"""Tests for the llm inputs boundary."""

from __future__ import annotations

from modal_llm_test_support import *  # noqa: F401,F403

def test_qwen_reasoning_parser_separates_token_channels(
    llm_reasoning_module: Any,
) -> None:
    """Qwen reasoning must never remain in the primary response string."""
    parser = llm_reasoning_module.Qwen3ReasoningParser(_FakeReasoningTokenizer())

    result = parser.extract(
        "<think>consider carefully</think>final answer",
        [10, 1, 2, 11, 11, 3],
    )

    assert result.response == "final answer"
    assert result.reasoning == "consider carefully"
    assert result.reasoning_tokens == 2
    assert result.parser == "qwen3"

def test_reasoning_parser_uses_architecture_fallback_for_existing_profile(
    llm_reasoning_module: Any,
) -> None:
    """Profiles generated before the parser field existed should remain usable."""
    profile = SimpleNamespace(
        architecture="Qwen3_5ForConditionalGeneration",
        reasoning_parser="",
    )

    parser = llm_reasoning_module.create_reasoning_parser(
        profile,
        _FakeReasoningTokenizer(),
    )

    assert parser.parser_name == "qwen3"

def test_reasoning_chat_template_uses_qwen_hard_switch(
    llm_reasoning_module: Any,
) -> None:
    """Reasoning-capable profiles should receive the exact per-request switch."""
    qwen_profile = SimpleNamespace(
        architecture="Qwen3_5ForConditionalGeneration",
        reasoning_parser="qwen3",
    )
    ordinary_profile = SimpleNamespace(
        architecture="SmolVLMForConditionalGeneration",
        reasoning_parser="none",
    )

    assert llm_reasoning_module.reasoning_chat_template_kwargs(
        qwen_profile,
        True,
    ) == {"enable_thinking": True}
    assert llm_reasoning_module.reasoning_chat_template_kwargs(
        qwen_profile,
        False,
    ) == {"enable_thinking": False}
    assert (
        llm_reasoning_module.reasoning_chat_template_kwargs(
            ordinary_profile,
            False,
        )
        == {}
    )

def test_disabled_reasoning_parser_returns_only_direct_response(
    llm_reasoning_module: Any,
) -> None:
    """Disabling reasoning must not classify a delimiter-free answer as thinking."""
    configured_parser = llm_reasoning_module.Qwen3ReasoningParser(
        _FakeReasoningTokenizer()
    )
    parser = llm_reasoning_module.reasoning_parser_for_request(
        configured_parser,
        False,
    )

    result = parser.extract(
        "final answer",
        [3],
        native_reasoning="unexpected backend reasoning",
    )

    assert result.response == "final answer"
    assert result.reasoning == ""
    assert result.reasoning_tokens == 0
    assert result.parser == "disabled"

def test_qwen_reasoning_parser_keeps_truncated_thinking_out_of_response(
    llm_reasoning_module: Any,
) -> None:
    """A token-limit stop before </think> is reasoning-only, not a final answer."""
    parser = llm_reasoning_module.Qwen3ReasoningParser(_FakeReasoningTokenizer())

    result = parser.extract("consider carefully", [1, 2])

    assert result.response == ""
    assert result.reasoning == "consider carefully"
    assert result.reasoning_tokens == 2

def test_reasoning_parser_prefers_engine_native_channel(
    llm_reasoning_module: Any,
) -> None:
    """A future backend-native split should bypass delimiter parsing."""
    parser = llm_reasoning_module.Qwen3ReasoningParser(_FakeReasoningTokenizer())

    result = parser.extract(
        "final answer",
        [3],
        native_reasoning="engine reasoning",
    )

    assert result.response == "final answer"
    assert result.reasoning == "engine reasoning"
    assert result.parser == "native"

def test_multimodal_preparation_samples_video_and_bounds_files(
    modal_llm_runtime_module: Any,
    llm_profiles_module: Any,
) -> None:
    """Native media and files should normalize without transport encoding."""
    profile = replace(
        llm_profiles_module.get_llm_profile("smolvlm2-2.2b-instruct"),
        allow_mixed_image_video=True,
    )
    images = torch.zeros((2, 4, 5, 3), dtype=torch.float32)
    video_frames = torch.stack(
        [torch.full((4, 5, 3), index / 9.0) for index in range(10)]
    )

    class FakeVideo:
        """Expose the native ComfyUI video component contract."""

        def get_components(self) -> Any:
            """Return ten frames at two frames per second."""
            return SimpleNamespace(images=video_frames, frame_rate=Fraction(2, 1))

    prepared = modal_llm_runtime_module.prepare_llm_inputs(
        prompt="Compare the media.",
        system_prompt="Be concise.",
        images=images,
        video=FakeVideo(),
        files=[_text_file("notes.txt", "important context")],
        video_frames=3,
        profile=profile,
    )

    assert len(prepared.images) == 2
    assert prepared.images[0].size == (5, 4)
    assert prepared.video.timestamps_seconds == (0.0, 2.0, 4.5)
    assert "notes.txt" in prepared.prompt
    assert "important context" in prepared.prompt
    assert prepared.file_count == 1

def test_profile_specific_mixed_media_restriction(
    modal_llm_runtime_module: Any,
    llm_profiles_module: Any,
) -> None:
    """The initial SmolVLM profile should fail clearly for unsupported mixed media."""
    profile = llm_profiles_module.get_llm_profile("smolvlm2-2.2b-instruct")

    class FakeVideo:
        """Expose a single native video frame."""

        def get_components(self) -> Any:
            """Return one frame at one frame per second."""
            return SimpleNamespace(
                images=torch.zeros((1, 2, 2, 3)),
                frame_rate=Fraction(1, 1),
            )

    with pytest.raises(ValueError, match="images or video"):
        modal_llm_runtime_module.prepare_llm_inputs(
            prompt="Describe.",
            system_prompt="",
            images=torch.zeros((1, 2, 2, 3)),
            video=FakeVideo(),
            files=None,
            video_frames=1,
            profile=profile,
        )

def test_pdf_without_extractable_text_is_rejected(
    modal_llm_runtime_module: Any,
    llm_profiles_module: Any,
) -> None:
    """Scanned or blank PDFs should fail clearly instead of adding empty context."""
    from pypdf import PdfWriter

    pdf_buffer = BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.write(pdf_buffer)
    encoded = base64.b64encode(pdf_buffer.getvalue()).decode("ascii")
    with pytest.raises(ValueError, match="no extractable text"):
        modal_llm_runtime_module.extract_file_context(
            [
                {
                    "filename": "blank.pdf",
                    "file_data": f"data:application/pdf;base64,{encoded}",
                }
            ],
            llm_profiles_module.get_llm_profile("smolvlm2-2.2b-instruct"),
        )

def test_representative_llm_prewarm_inputs_cover_text_and_vision_shapes(
    modal_llm_runtime_module: Any,
) -> None:
    """JIT warmup should exercise text plus two bounded image resolutions."""
    profile = SimpleNamespace(modalities=frozenset({"text", "image"}))

    text_inputs = modal_llm_runtime_module._representative_prewarm_inputs(profile, 0)
    small_image_inputs = modal_llm_runtime_module._representative_prewarm_inputs(
        profile, 1
    )
    large_image_inputs = modal_llm_runtime_module._representative_prewarm_inputs(
        profile, 2
    )

    assert text_inputs.images == ()
    assert small_image_inputs.images[0].size == (512, 512)
    assert large_image_inputs.images[0].size == (1024, 1024)

