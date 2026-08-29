"""Tests for the vllm backend boundary."""

from __future__ import annotations

from modal_llm_test_support import *  # noqa: F401,F403

def test_vllm_backend_uses_explicit_kv_budget_and_local_multimodal_data(
    modal_llm_runtime_module: Any,
    llm_profiles_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The vLLM adapter should not reserve memory from a whole-GPU fraction."""
    import transformers

    profile = replace(
        llm_profiles_module.get_llm_profile("smolvlm2-2.2b-instruct"),
        profile_id="qwen-vllm-test",
        backend="vllm",
        architecture="Qwen3_5ForConditionalGeneration",
        reasoning_parser="qwen3",
        quantization_method="fp8",
        backend_options=(
            ("enforce_eager", True),
            ("kv_cache_memory_bytes", 12 * 1024**3),
            ("max_model_len", 32768),
            ("quantization", "fp8"),
        ),
    )
    observed: dict[str, Any] = {}

    class FakeProcessor:
        """Render a deterministic prompt from processor-native messages."""

        tokenizer = _FakeReasoningTokenizer()

        def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
            """Record that tokenization stays inside vLLM."""
            observed["messages"] = messages
            observed["chat_kwargs"] = kwargs
            return "rendered prompt"

    class FakeAsyncEngineArgs:
        """Capture explicit asynchronous engine construction arguments."""

        def __init__(self, **kwargs: Any) -> None:
            """Retain the co-residency policy for assertions."""
            self.kwargs = kwargs

    class FakeAsyncLLM:
        """Stream deterministic cumulative vLLM request outputs."""

        @classmethod
        def from_engine_args(cls, engine_args: FakeAsyncEngineArgs) -> Any:
            """Capture engine arguments and return one resident engine."""
            observed["llm_kwargs"] = engine_args.kwargs
            return cls()

        async def generate(self, prompt: Any, **kwargs: Any) -> Any:
            """Yield two cumulative token updates from one request."""
            observed["prompts"] = prompt
            observed["generate_kwargs"] = kwargs
            yield SimpleNamespace(
                prompt_token_ids=[1, 2, 3],
                outputs=[SimpleNamespace(text="<think>consider", token_ids=[10, 1])],
            )
            yield SimpleNamespace(
                prompt_token_ids=[1, 2, 3],
                outputs=[
                    SimpleNamespace(
                        text="<think>consider carefully</think>final answer",
                        token_ids=[10, 1, 2, 11, 3],
                    )
                ],
            )

        async def abort(self, request_id: str) -> None:
            """Record unexpected request cancellation."""
            observed["aborted"] = request_id

        def shutdown(self) -> None:
            """Record deterministic engine cleanup."""
            observed["shutdown"] = True

    class FakeSamplingParams:
        """Capture backend-neutral sampling settings."""

        def __init__(self, **kwargs: Any) -> None:
            """Retain settings for assertions."""
            self.kwargs = kwargs

    class FakeEngineDeadError(Exception):
        """Stand in for vLLM's package-private engine exception."""

    monkeypatch.setattr(
        transformers.AutoProcessor,
        "from_pretrained",
        lambda *args, **kwargs: FakeProcessor(),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(
            AsyncEngineArgs=FakeAsyncEngineArgs,
            AsyncLLMEngine=FakeAsyncLLM,
            SamplingParams=FakeSamplingParams,
            __path__=[],
        ),
    )
    monkeypatch.setitem(sys.modules, "vllm.v1", SimpleNamespace(__path__=[]))
    monkeypatch.setitem(sys.modules, "vllm.v1.engine", SimpleNamespace(__path__=[]))
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.engine.exceptions",
        SimpleNamespace(EngineDeadError=FakeEngineDeadError),
    )
    progress: list[Any] = []
    backend = modal_llm_runtime_module._default_backend_factory(
        profile,
        tmp_path,
        progress.append,
    )
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
            temperature=0.0,
            top_p=1.0,
            seed=4,
        ),
        progress.append,
    )
    backend.unload()

    assert observed["llm_kwargs"]["kv_cache_memory_bytes"] == 12 * 1024**3
    assert "gpu_memory_utilization" not in observed["llm_kwargs"]
    assert observed["llm_kwargs"]["quantization"] == "fp8"
    assert observed["llm_kwargs"]["enforce_eager"] is True
    assert observed["llm_kwargs"]["safetensors_load_strategy"] == "prefetch"
    assert observed["llm_kwargs"]["attention_config"] == {
        "backend": "TRITON_ATTN",
    }
    assert observed["prompts"] == {"prompt": "rendered prompt"}
    assert observed["chat_kwargs"]["tokenize"] is False
    assert observed["chat_kwargs"]["enable_thinking"] is True
    assert [event.value for event in progress if event.stage == "generating"] == [2, 5]
    assert progress[-1].tokens_per_second is not None
    assert result.text == "final answer"
    assert result.reasoning == "consider carefully"
    assert result.reasoning_tokens == 2
    assert result.input_tokens == 3
    assert (
        observed["generate_kwargs"]["sampling_params"].kwargs["skip_special_tokens"]
        is False
    )
    assert observed["generate_kwargs"]["request_id"].startswith("modal-llm-")
    assert observed["shutdown"] is True

def test_vllm_backend_translates_private_runtime_errors(
    modal_llm_runtime_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remote failures should not require vLLM exception classes on the client."""

    class FakeProcessor:
        """Return a deterministic prompt without importing Transformers."""

        def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
            """Ignore the request structure and return a prompt sentinel."""
            del messages, kwargs
            return "rendered prompt"

    class FakeEngineDeadError(Exception):
        """Stand in for vLLM's non-RuntimeError engine exception."""

    class FailingLLM:
        """Raise the RuntimeError shape used by vLLM engine failures."""

        async def generate(self, prompts: Any, **kwargs: Any) -> Any:
            """Fail after accepting a well-formed generation request."""
            del prompts, kwargs
            raise FakeEngineDeadError("engine subprocess stopped")
            yield

        async def abort(self, request_id: str) -> None:
            """Accept cleanup after the failed request."""
            del request_id

    class FakeSamplingParams:
        """Accept backend-neutral generation options without vLLM installed."""

        def __init__(self, **kwargs: Any) -> None:
            """Discard values after proving the adapter constructed them."""
            del kwargs

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(SamplingParams=FakeSamplingParams, __path__=[]),
    )
    monkeypatch.setitem(sys.modules, "vllm.v1", SimpleNamespace(__path__=[]))
    monkeypatch.setitem(sys.modules, "vllm.v1.engine", SimpleNamespace(__path__=[]))
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.engine.exceptions",
        SimpleNamespace(EngineDeadError=FakeEngineDeadError),
    )

    backend = object.__new__(modal_llm_runtime_module.VLLMMultimodalBackend)
    backend.profile = SimpleNamespace(
        profile_id="generated-profile",
        architecture="",
        reasoning_parser="",
        backend_option=lambda name, default=None: default,
    )
    backend.processor = FakeProcessor()
    backend.llm = FailingLLM()
    backend.reasoning_parser = SimpleNamespace(requires_boundary_tokens=False)
    prepared = modal_llm_runtime_module.PreparedLLMInputs(
        prompt="hello",
        system_prompt="",
        images=(),
        video=None,
        file_characters=0,
        file_count=0,
    )
    settings = modal_llm_runtime_module.LLMGenerationSettings(
        max_new_tokens=8,
        temperature=0.0,
        top_p=1.0,
        seed=0,
    )

    with pytest.raises(
        RuntimeError,
        match="vLLM generation failed for profile 'generated-profile'",
    ):
        asyncio.run(backend._generate_async(prepared, settings, lambda progress: None))

def test_vllm_request_disables_thinking_and_boundary_parsing(
    modal_llm_runtime_module: Any,
    llm_reasoning_module: Any,
) -> None:
    """The vLLM request and decoder should share the disabled reasoning state."""
    observed: dict[str, Any] = {}

    class FakeProcessor:
        """Capture the per-request chat-template controls."""

        def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
            """Record template arguments and return a direct-response prompt."""
            del messages
            observed.update(kwargs)
            return "rendered direct prompt"

    class FakeSamplingParams:
        """Retain sampling arguments for assertions."""

        def __init__(self, **kwargs: Any) -> None:
            """Store the backend adapter's keyword arguments."""
            self.kwargs = kwargs

    backend = object.__new__(modal_llm_runtime_module.VLLMMultimodalBackend)
    backend.profile = SimpleNamespace(
        architecture="Qwen3_5ForConditionalGeneration",
        reasoning_parser="qwen3",
    )
    backend.processor = FakeProcessor()
    backend.reasoning_parser = llm_reasoning_module.Qwen3ReasoningParser(
        _FakeReasoningTokenizer()
    )
    prepared = modal_llm_runtime_module.PreparedLLMInputs(
        prompt="hello",
        system_prompt="",
        images=(),
        video=None,
        file_characters=0,
        file_count=0,
    )
    settings = modal_llm_runtime_module.LLMGenerationSettings(
        max_new_tokens=8,
        temperature=0.0,
        top_p=1.0,
        seed=0,
        enable_reasoning=False,
    )

    request = backend._request(prepared, settings)
    sampling_params = backend._sampling_params(FakeSamplingParams, settings)

    assert request == {"prompt": "rendered direct prompt"}
    assert observed["enable_thinking"] is False
    assert sampling_params.kwargs["skip_special_tokens"] is True

def test_vllm_backend_aborts_when_progress_callback_cancels(
    modal_llm_runtime_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ComfyUI cancellation during streaming should abort the vLLM request."""
    aborted_requests: list[str] = []

    class FakeSamplingParams:
        """Accept generation options for the cancellation path."""

        def __init__(self, **kwargs: Any) -> None:
            """Discard validated generation options."""
            del kwargs

    class FakeEngineDeadError(Exception):
        """Stand in for vLLM's private engine exception."""

    class StreamingLLM:
        """Yield one token update and record its subsequent abort."""

        async def generate(self, prompt: Any, **kwargs: Any) -> Any:
            """Yield one cumulative request output."""
            del prompt, kwargs
            yield SimpleNamespace(
                prompt_token_ids=[1],
                outputs=[SimpleNamespace(text="partial", token_ids=[2])],
            )

        async def abort(self, request_id: str) -> None:
            """Record the interrupted request id."""
            aborted_requests.append(request_id)

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(SamplingParams=FakeSamplingParams, __path__=[]),
    )
    monkeypatch.setitem(sys.modules, "vllm.v1", SimpleNamespace(__path__=[]))
    monkeypatch.setitem(sys.modules, "vllm.v1.engine", SimpleNamespace(__path__=[]))
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.engine.exceptions",
        SimpleNamespace(EngineDeadError=FakeEngineDeadError),
    )
    backend = object.__new__(modal_llm_runtime_module.VLLMMultimodalBackend)
    backend.profile = SimpleNamespace(
        profile_id="generated-profile",
        reasoning_parser="",
        architecture="",
    )
    backend.processor = SimpleNamespace(
        apply_chat_template=lambda *args, **kwargs: "rendered prompt"
    )
    backend.llm = StreamingLLM()
    backend.reasoning_parser = SimpleNamespace(requires_boundary_tokens=False)
    prepared = modal_llm_runtime_module.PreparedLLMInputs(
        prompt="hello",
        system_prompt="",
        images=(),
        video=None,
        file_characters=0,
        file_count=0,
    )
    settings = modal_llm_runtime_module.LLMGenerationSettings(
        max_new_tokens=8,
        temperature=0.0,
        top_p=1.0,
        seed=0,
    )

    def cancel_on_token(progress: Any) -> None:
        """Raise the same interruption shape used by the node callback."""
        if progress.stage == "generating":
            raise InterruptedError("cancelled")

    with pytest.raises(InterruptedError, match="cancelled"):
        asyncio.run(backend._generate_async(prepared, settings, cancel_on_token))

    assert len(aborted_requests) == 1
    assert aborted_requests[0].startswith("modal-llm-")

