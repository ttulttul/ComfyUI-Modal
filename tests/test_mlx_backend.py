"""Tests for the mlx backend boundary."""

from __future__ import annotations

from modal_llm_test_support import *  # noqa: F401,F403

def test_local_storage_uses_comfyui_model_directory(
    local_llm_runtime_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Local snapshots should follow ComfyUI's configured model root."""
    paths: list[str] = []
    fake_folder_paths = SimpleNamespace(
        models_dir=str(tmp_path / "models"),
        add_model_folder_path=lambda name, path, is_default=False: paths.append(
            f"{name}:{path}:{is_default}"
        ),
        get_folder_paths=lambda name: [str(tmp_path / "models" / name)],
    )
    monkeypatch.delenv("COMFY_MODAL_LOCAL_LLM_STORAGE_ROOT", raising=False)
    monkeypatch.setitem(sys.modules, "folder_paths", fake_folder_paths)

    storage_root = local_llm_runtime_module.local_llm_storage_root()

    assert storage_root == (tmp_path / "models" / "modal_llm").resolve()
    assert paths == [f"modal_llm:{(tmp_path / 'models' / 'modal_llm').resolve()}:True"]

def test_local_runtime_rejects_non_apple_hardware_actionably(
    local_llm_runtime_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unsupported local hardware should direct the workflow to Modal."""
    monkeypatch.setattr(local_llm_runtime_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(local_llm_runtime_module.platform, "machine", lambda: "x86_64")

    with pytest.raises(RuntimeError, match="Enable 'Run on Modal'"):
        local_llm_runtime_module.ensure_local_apple_runtime_available()

def test_local_runtime_requires_the_pinned_mlx_version(
    local_llm_runtime_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A drifted MLX adapter should fail with an exact repair command."""
    monkeypatch.setattr(local_llm_runtime_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(local_llm_runtime_module.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(
        local_llm_runtime_module.metadata,
        "version",
        lambda distribution: "0.1.0",
    )

    with pytest.raises(RuntimeError, match="mlx-vlm==0.6.15"):
        local_llm_runtime_module.ensure_local_apple_runtime_available()

def test_mlx_backend_streams_multimodal_reasoning_and_progress(
    local_llm_runtime_module: Any,
    modal_llm_runtime_module: Any,
    llm_profiles_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """MLX streaming should retain the shared outputs and reasoning contract."""
    import types

    observed: dict[str, Any] = {}

    class FakeMX:
        """Expose only the MLX lifecycle methods used by the adapter."""

        @staticmethod
        def reset_peak_memory() -> None:
            """Record peak-memory reset."""
            observed["reset_peak_memory"] = True

        @staticmethod
        def clear_cache() -> None:
            """Record cache release."""
            observed["clear_cache"] = True

    processor = SimpleNamespace(tokenizer=_FakeReasoningTokenizer())
    model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_5"))

    def fake_load(path: str, **kwargs: Any) -> tuple[Any, Any]:
        """Return a deterministic resident MLX model and processor."""
        observed["load"] = (path, kwargs)
        return model, processor

    def fake_apply_chat_template(
        selected_processor: Any,
        config: Any,
        messages: Any,
        **kwargs: Any,
    ) -> str:
        """Capture the structured prompt passed to MLX-VLM."""
        observed["chat"] = (selected_processor, config, messages, kwargs)
        return "rendered prompt"

    def fake_stream_generate(*args: Any, **kwargs: Any) -> Any:
        """Yield exact reasoning boundary and response tokens."""
        observed["generate"] = (args, kwargs)
        segments = (
            [
                ("<think>", 10),
                ("consider", 1),
                ("</think>", 11),
                ("final answer", 3),
            ]
            if kwargs["enable_thinking"]
            else [("direct answer", 3)]
        )
        for index, (text, token) in enumerate(segments, start=1):
            yield SimpleNamespace(
                text=text,
                token=token,
                prompt_tokens=5,
                generation_tokens=index,
                generation_tps=10.0 + index,
            )

    mlx_module = types.ModuleType("mlx")
    mlx_module.__path__ = []
    mlx_core_module = types.ModuleType("mlx.core")
    mlx_core_module.reset_peak_memory = FakeMX.reset_peak_memory
    mlx_core_module.clear_cache = FakeMX.clear_cache
    mlx_vlm_module = types.ModuleType("mlx_vlm")
    mlx_vlm_module.__path__ = []
    mlx_vlm_module.load = fake_load
    mlx_generate_module = types.ModuleType("mlx_vlm.generate")
    mlx_generate_module.stream_generate = fake_stream_generate
    mlx_prompt_module = types.ModuleType("mlx_vlm.prompt_utils")
    mlx_prompt_module.apply_chat_template = fake_apply_chat_template
    monkeypatch.setitem(sys.modules, "mlx", mlx_module)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core_module)
    monkeypatch.setitem(sys.modules, "mlx_vlm", mlx_vlm_module)
    monkeypatch.setitem(sys.modules, "mlx_vlm.generate", mlx_generate_module)
    monkeypatch.setitem(sys.modules, "mlx_vlm.prompt_utils", mlx_prompt_module)
    monkeypatch.setattr(
        local_llm_runtime_module,
        "ensure_local_apple_runtime_available",
        lambda: None,
    )
    profile = replace(
        llm_profiles_module.get_llm_profile("smolvlm2-2.2b-instruct"),
        backend="mlx_vlm",
        architecture="Qwen3_5ForConditionalGeneration",
        reasoning_parser="qwen3",
        execution_target="local_apple",
        max_context_tokens=128,
    )
    progress: list[Any] = []
    backend = local_llm_runtime_module.MLXVLMBackend(
        profile,
        tmp_path,
        progress.append,
    )
    prepared = modal_llm_runtime_module.PreparedLLMInputs(
        prompt="describe",
        system_prompt="be concise",
        images=("image-a",),
        video=modal_llm_runtime_module.PreparedVideo(
            frames=("frame-a", "frame-b"),
            timestamps_seconds=(0.0, 1.0),
        ),
        file_characters=0,
        file_count=0,
    )

    result = backend.generate(
        prepared,
        modal_llm_runtime_module.LLMGenerationSettings(
            max_new_tokens=8,
            temperature=0.0,
            top_p=0.95,
            seed=7,
        ),
        progress.append,
    )
    reasoning_chat = observed["chat"]
    reasoning_generate = observed["generate"]
    direct_result = backend.generate(
        prepared,
        modal_llm_runtime_module.LLMGenerationSettings(
            max_new_tokens=8,
            temperature=0.0,
            top_p=0.95,
            seed=7,
            enable_reasoning=False,
        ),
        progress.append,
    )
    direct_chat = observed["chat"]
    direct_generate = observed["generate"]
    backend.unload()

    assert result.text == "final answer"
    assert result.reasoning == "consider"
    assert result.reasoning_tokens == 1
    assert result.input_tokens == 5
    assert result.output_tokens == 4
    assert reasoning_chat[2] == [
        {"role": "system", "content": "be concise"},
        {"role": "user", "content": "describe"},
    ]
    assert reasoning_chat[3]["num_images"] == 3
    assert reasoning_chat[3]["enable_thinking"] is True
    assert reasoning_generate[1]["image"] == [
        "image-a",
        "frame-a",
        "frame-b",
    ]
    assert reasoning_generate[1]["seed"] == 7
    assert reasoning_generate[1]["enable_thinking"] is True
    assert reasoning_generate[1]["thinking_budget"] == 4
    assert direct_result.text == "direct answer"
    assert direct_result.reasoning == ""
    assert direct_result.reasoning_parser == "disabled"
    assert direct_chat[3]["enable_thinking"] is False
    assert direct_generate[1]["enable_thinking"] is False
    assert "thinking_budget" not in direct_generate[1]
    assert direct_generate[1]["skip_special_tokens"] is True
    assert [event.value for event in progress if event.stage == "generating"] == [
        1,
        2,
        3,
        4,
        1,
    ]
    assert observed["clear_cache"] is True

