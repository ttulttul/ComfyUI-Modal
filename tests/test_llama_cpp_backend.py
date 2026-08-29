"""Tests for the llama cpp backend boundary."""

from __future__ import annotations

from modal_llm_test_support import *  # noqa: F401,F403

def test_llama_cpp_backend_generates_with_curated_gguf_profile(
    modal_llm_runtime_module: Any,
    llm_profiles_module: Any,
    llm_reasoning_module: Any,
    tmp_path: Path,
) -> None:
    """The GGUF adapter should preserve reasoning parsing and token telemetry."""
    profile = llm_profiles_module.get_llm_profile(
        "huihui-qwen3.8-27b-abliterated-q2-k-gguf"
    )
    model_path = tmp_path / profile.backend_option("model_filename")
    model_path.write_bytes(b"gguf")
    backend = object.__new__(modal_llm_runtime_module.LlamaCppServerBackend)
    backend.profile = profile
    backend.snapshot_path = tmp_path
    backend.model_path = model_path
    backend.processor = SimpleNamespace(
        tokenizer=_FakeReasoningTokenizer(),
        apply_chat_template=lambda *args, **kwargs: "rendered prompt",
    )
    backend.reasoning_parser = llm_reasoning_module.Qwen3ReasoningParser(
        backend.processor.tokenizer
    )
    backend._completion = lambda payload, timeout_seconds: {
        "content": "<think>consider</think>answer",
        "tokens": [10, 1, 11, 3],
        "tokens_evaluated": 7,
        "timings": {"predicted_per_second": 12.5},
    }
    progress: list[Any] = []

    result = backend.generate(
        modal_llm_runtime_module.PreparedLLMInputs(
            prompt="hello",
            system_prompt="",
            images=(),
            video=None,
            file_characters=0,
            file_count=0,
        ),
        modal_llm_runtime_module.LLMGenerationSettings(
            max_new_tokens=16,
            temperature=0.2,
            top_p=0.95,
            seed=4,
        ),
        progress.append,
    )

    assert result.text == "final answer"
    assert result.reasoning == "consider"
    assert result.input_tokens == 7
    assert result.output_tokens == 4
    assert result.tokens_per_second == 12.5
    assert [event.stage for event in progress] == ["prefill", "generating"]

def test_llama_cpp_backend_adds_binary_directory_to_library_path(
    modal_llm_runtime_module: Any,
    llm_backend_llamacpp_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The private server should find shared libraries shipped beside its binary."""
    backend = object.__new__(modal_llm_runtime_module.LlamaCppServerBackend)
    backend.profile = SimpleNamespace(
        profile_id="gguf-test",
        max_context_tokens=8192,
        backend_option=lambda _name, default=None: default,
    )
    backend.model_path = tmp_path / "model.gguf"
    backend.mmproj_path = None
    backend.port = 18080
    backend._log_file = BytesIO()
    observed: dict[str, Any] = {}

    def popen(command: list[str], **kwargs: Any) -> SimpleNamespace:
        """Capture the subprocess environment for the private server."""
        observed["command"] = command
        observed["kwargs"] = kwargs
        return SimpleNamespace()

    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing/lib")
    monkeypatch.setattr(llm_backend_llamacpp_module.subprocess, "Popen", popen)

    backend._start_server()

    assert observed["command"][0] == "/app/llama-server"
    assert observed["kwargs"]["env"]["LD_LIBRARY_PATH"] == (
        "/app:/existing/lib"
    )

def test_llama_cpp_backend_sends_images_through_chat_completion(
    modal_llm_runtime_module: Any,
    llm_profiles_module: Any,
    llm_reasoning_module: Any,
    tmp_path: Path,
) -> None:
    """The curated projector should receive image data through the OAI route."""
    profile = llm_profiles_module.get_llm_profile(
        "huihui-qwen3.8-27b-abliterated-q2-k-gguf"
    )
    backend = object.__new__(modal_llm_runtime_module.LlamaCppServerBackend)
    backend.profile = profile
    backend.snapshot_path = tmp_path
    backend.model_path = tmp_path / profile.backend_option("model_filename")
    backend.mmproj_path = tmp_path / profile.backend_option("mmproj_filename")
    backend.processor = SimpleNamespace(tokenizer=_FakeReasoningTokenizer())
    backend.reasoning_parser = llm_reasoning_module.Qwen3ReasoningParser(
        backend.processor.tokenizer
    )
    observed: dict[str, Any] = {}

    def fake_chat_completion(
        payload: dict[str, Any], timeout_seconds: float
    ) -> dict[str, Any]:
        """Capture one multimodal request and return OAI-style usage."""
        observed["payload"] = payload
        observed["timeout_seconds"] = timeout_seconds
        return {
            "choices": [{"message": {"content": "image answer"}}],
            "usage": {"prompt_tokens": 101, "completion_tokens": 2},
            "timings": {"predicted_per_second": 8.0},
        }

    backend._chat_completion = fake_chat_completion
    result = backend.generate(
        modal_llm_runtime_module.PreparedLLMInputs(
            prompt="describe",
            system_prompt="be concise",
            images=(Image.new("RGB", (4, 4), color=(255, 0, 0)),),
            video=None,
            file_characters=0,
            file_count=0,
        ),
        modal_llm_runtime_module.LLMGenerationSettings(
            max_new_tokens=16,
            temperature=0.2,
            top_p=0.95,
            seed=4,
            enable_reasoning=False,
        ),
        lambda _event: None,
    )

    messages = observed["payload"]["messages"]
    assert messages[0] == {"role": "system", "content": "be concise"}
    assert messages[1]["content"][0]["type"] == "image_url"
    assert messages[1]["content"][0]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )
    assert messages[1]["content"][1] == {"type": "text", "text": "describe"}
    assert result.text == "image answer"
    assert result.input_tokens == 101
    assert result.output_tokens == 2
    assert result.tokens_per_second == 8.0

