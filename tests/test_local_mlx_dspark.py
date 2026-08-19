"""Focused coverage for Apple-local mlx-dspark engine selection and inference."""

from __future__ import annotations

import sys
import types
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


def _prepared_inputs(
    modal_llm_runtime_module: Any,
    *,
    images: tuple[Any, ...] = (),
) -> Any:
    """Return a compact prepared-input value for local engine tests."""
    return modal_llm_runtime_module.PreparedLLMInputs(
        prompt="hello",
        system_prompt="be concise",
        images=images,
        video=None,
        file_characters=0,
        file_count=0,
    )


def test_local_engine_auto_uses_registered_dspark_for_text(
    local_llm_runtime_module: Any,
    modal_llm_runtime_module: Any,
    llm_profiles_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto should choose the package's measured mode for registered text targets."""
    profile = llm_profiles_module.get_llm_profile("smolvlm2-2.2b-instruct")
    monkeypatch.setattr(
        local_llm_runtime_module,
        "ensure_local_dspark_runtime_available",
        lambda: None,
    )
    monkeypatch.setattr(
        local_llm_runtime_module,
        "_resolve_mlx_dspark_mode",
        lambda selected_profile: ("dflash", "owner/drafter"),
    )

    selection = local_llm_runtime_module.select_local_mlx_engine(
        "auto",
        profile,
        _prepared_inputs(modal_llm_runtime_module),
    )

    assert selection.engine == "mlx-dspark"
    assert selection.mode == "dflash"
    assert selection.drafter_repository == "owner/drafter"


def test_local_engine_auto_preserves_mlx_vlm_for_media(
    local_llm_runtime_module: Any,
    modal_llm_runtime_module: Any,
    llm_profiles_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Media must never enter mlx-dspark's text-only Qwen loading path."""
    profile = llm_profiles_module.get_llm_profile("smolvlm2-2.2b-instruct")
    monkeypatch.setattr(
        local_llm_runtime_module,
        "ensure_local_dspark_runtime_available",
        lambda: pytest.fail("media selection should not import mlx-dspark"),
    )
    prepared = _prepared_inputs(modal_llm_runtime_module, images=("image",))

    selection = local_llm_runtime_module.select_local_mlx_engine(
        "auto",
        profile,
        prepared,
    )

    assert selection.engine == "mlx-vlm"
    with pytest.raises(ValueError, match="text-only"):
        local_llm_runtime_module.select_local_mlx_engine(
            "mlx-dspark",
            profile,
            prepared,
        )


def test_local_engine_auto_falls_back_when_dspark_is_unavailable(
    local_llm_runtime_module: Any,
    modal_llm_runtime_module: Any,
    llm_profiles_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Existing local installs should retain MLX-VLM behavior until the extra is updated."""
    profile = llm_profiles_module.get_llm_profile("smolvlm2-2.2b-instruct")
    monkeypatch.setattr(
        local_llm_runtime_module,
        "ensure_local_dspark_runtime_available",
        lambda: (_ for _ in ()).throw(RuntimeError("not installed")),
    )

    selection = local_llm_runtime_module.select_local_mlx_engine(
        "auto",
        profile,
        _prepared_inputs(modal_llm_runtime_module),
    )

    assert selection.engine == "mlx-vlm"
    with pytest.raises(RuntimeError, match="not installed"):
        local_llm_runtime_module.select_local_mlx_engine(
            "mlx-dspark",
            profile,
            _prepared_inputs(modal_llm_runtime_module),
        )


def test_switching_local_engines_unloads_other_engine_residents(
    local_llm_runtime_module: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Separate engine caches must not retain two large targets in unified memory."""

    class FakeManager:
        """Record cache-wide unload requests."""

        def __init__(self) -> None:
            """Initialize the unload counter."""
            self.unload_calls = 0

        def unload_all(self) -> None:
            """Record one cache eviction."""
            self.unload_calls += 1

    resolved_root = tmp_path.resolve()
    vlm_manager = FakeManager()
    dspark_manager = FakeManager()
    monkeypatch.setattr(
        local_llm_runtime_module,
        "_LOCAL_MANAGERS",
        {
            (resolved_root, "mlx-vlm"): vlm_manager,
            (resolved_root, "mlx-dspark"): dspark_manager,
        },
    )

    selected = local_llm_runtime_module.get_local_resident_llm_manager(
        resolved_root,
        "mlx-dspark",
    )

    assert selected is dspark_manager
    assert vlm_manager.unload_calls == 1
    assert dspark_manager.unload_calls == 0


def test_resident_manager_accepts_backend_runtime_metadata_override(
    modal_llm_runtime_module: Any,
    llm_profiles_module: Any,
    tmp_path: Path,
) -> None:
    """Accelerated adapters should report their real backend instead of profile storage."""

    class FakeBackend:
        """Return deterministic generation and accelerated runtime metadata."""

        def generate(self, *args: Any, **kwargs: Any) -> Any:
            """Return one token without allocating model memory."""
            return modal_llm_runtime_module.BackendGenerationResult(
                text="done",
                input_tokens=2,
                output_tokens=1,
            )

        def runtime_metadata(self) -> dict[str, Any]:
            """Override the storage profile's MLX-VLM backend label."""
            return {"backend": "mlx_dspark", "mlx_dspark_mode": "dflash"}

        def unload(self) -> None:
            """Release no-op fake state."""

    profile = llm_profiles_module.get_llm_profile("smolvlm2-2.2b-instruct")
    manager = modal_llm_runtime_module.ResidentLLMManager(
        storage_root=tmp_path,
        backend_factory=lambda *args: FakeBackend(),
        memory_info=lambda: (64 * 1024**3, 64 * 1024**3),
        empty_cache=lambda: None,
        snapshot_ready=lambda *args: True,
        comfy_memory_release=lambda required_bytes: None,
        execution_target="local_apple",
        device_name="metal",
        memory_label="unified memory",
    )

    result = manager.infer(
        profile=profile,
        prepared_inputs=_prepared_inputs(modal_llm_runtime_module),
        generation_settings=modal_llm_runtime_module.LLMGenerationSettings(
            max_new_tokens=8,
            temperature=0.0,
            top_p=0.95,
            seed=0,
        ),
        reserve_free_vram_gb=4.0,
        keep_model_loaded=False,
        progress_callback=lambda event: None,
    )

    assert result.metadata["backend"] == "mlx_dspark"
    assert result.metadata["mlx_dspark_mode"] == "dflash"


def test_dspark_drafter_staging_pins_revision_and_skips_other_weight_formats(
    local_llm_runtime_module: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drafter staging should be immutable and avoid unused multi-GB GGUF files."""
    import huggingface_hub

    downloads: list[dict[str, Any]] = []
    model_info = {
        "id": "owner/drafter",
        "sha": "a" * 40,
        "siblings": [
            {"rfilename": "config.json", "size": 100},
            {"rfilename": "model.safetensors", "size": 1024},
            {"rfilename": "unused.gguf", "size": 10 * 1024**3},
        ],
        "securityStatus": {"scansDone": True, "filesWithIssues": []},
    }

    class FakeHfApi:
        """Return one exact revision and reviewed file list."""

        def model_info(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            """Return deterministic repository metadata."""
            return model_info

    def fake_snapshot_download(**kwargs: Any) -> str:
        """Materialize the files selected for the MLX adapter."""
        downloads.append(kwargs)
        snapshot_path = Path(kwargs["local_dir"])
        (snapshot_path / "config.json").write_text("{}", encoding="utf-8")
        (snapshot_path / "model.safetensors").write_bytes(b"weights")
        return str(snapshot_path)

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeHfApi)
    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        fake_snapshot_download,
    )
    progress: list[Any] = []

    first = local_llm_runtime_module._stage_mlx_dspark_drafter(
        "owner/drafter",
        tmp_path,
        progress.append,
    )
    second = local_llm_runtime_module._stage_mlx_dspark_drafter(
        "owner/drafter",
        tmp_path,
        progress.append,
    )

    assert first == second
    assert first.revision == "a" * 40
    assert len(downloads) == 1
    assert downloads[0]["revision"] == "a" * 40
    assert downloads[0]["allow_patterns"] == (
        "config.json",
        "*.safetensors",
        "*.safetensors.index.json",
    )
    assert not (first.path / "unused.gguf").exists()


def test_mlx_dspark_backend_uses_pinned_drafter_and_reports_progress(
    local_llm_runtime_module: Any,
    modal_llm_runtime_module: Any,
    llm_profiles_module: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The adapter should load local target/drafter paths and expose engine metadata."""
    observed: dict[str, Any] = {}

    class FakeMX:
        """Record MLX cache lifecycle calls."""

        reset_calls = 0
        clear_calls = 0

        @classmethod
        def reset_peak_memory(cls) -> None:
            """Record one load reset."""
            cls.reset_calls += 1

        @classmethod
        def clear_cache(cls) -> None:
            """Record one unload cache clear."""
            cls.clear_calls += 1

    class FakeTokenizer:
        """Provide the chat-template method used by mlx-dspark."""

        chat_template = "template"

        def apply_chat_template(self, messages: Any, **kwargs: Any) -> list[int]:
            """Return deterministic input token ids."""
            observed["messages"] = messages
            observed["template_kwargs"] = kwargs
            return [1, 2, 3]

    def fake_load_dflash_pair(model: str, *, drafter: str) -> tuple[Any, ...]:
        """Return a resident fake target/drafter pair."""
        observed["load"] = (model, drafter)
        return object(), FakeTokenizer(), object(), object()

    def fake_generate(*args: Any, **kwargs: Any) -> Any:
        """Emit one speculative round and a deterministic result."""
        observed["generate"] = kwargs
        kwargs["on_text"]("hello")
        kwargs["on_round"](drafted=2, accepted=1, committed=2, cap=2)
        return SimpleNamespace(
            text="hello",
            token_ids=[4, 5],
            num_tokens=2,
            tokens_per_sec=12.5,
        )

    mlx_module = types.ModuleType("mlx")
    mlx_module.__path__ = []
    mlx_core_module = types.ModuleType("mlx.core")
    mlx_core_module.reset_peak_memory = FakeMX.reset_peak_memory
    mlx_core_module.clear_cache = FakeMX.clear_cache
    dspark_module = types.ModuleType("mlx_dspark")
    dspark_module.load_dflash_pair = fake_load_dflash_pair
    dspark_module.load_pair = pytest.fail
    dspark_module.dflash_generate = fake_generate
    dspark_module.speculative_generate = pytest.fail
    dspark_module.encode_messages = lambda tokenizer, messages, **kwargs: (
        tokenizer.apply_chat_template(messages, **kwargs)
    )
    monkeypatch.setitem(sys.modules, "mlx", mlx_module)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core_module)
    monkeypatch.setitem(sys.modules, "mlx_dspark", dspark_module)
    monkeypatch.setattr(
        local_llm_runtime_module,
        "ensure_local_dspark_runtime_available",
        lambda: None,
    )
    monkeypatch.setattr(
        local_llm_runtime_module,
        "_resolve_mlx_dspark_mode",
        lambda profile: ("dflash", "owner/drafter"),
    )
    drafter_path = tmp_path / "drafter"
    monkeypatch.setattr(
        local_llm_runtime_module,
        "_stage_mlx_dspark_drafter",
        lambda repository, storage_root, progress_callback: (
            local_llm_runtime_module.StagedDSparkDrafter(
                repository,
                "a" * 40,
                drafter_path,
            )
        ),
    )
    profile = replace(
        llm_profiles_module.get_llm_profile("smolvlm2-2.2b-instruct"),
        execution_target="local_apple",
        max_context_tokens=128,
    )
    target_path = tmp_path / "llm_models" / "target" / ("b" * 40)
    target_path.mkdir(parents=True)
    progress: list[Any] = []
    backend = local_llm_runtime_module.MLXDSparkBackend(
        profile,
        target_path,
        progress.append,
    )

    result = backend.generate(
        _prepared_inputs(modal_llm_runtime_module),
        modal_llm_runtime_module.LLMGenerationSettings(
            max_new_tokens=8,
            temperature=0.0,
            top_p=0.95,
            seed=7,
        ),
        progress.append,
    )
    metadata = backend.runtime_metadata()
    backend.unload()

    assert observed["load"] == (str(target_path), str(drafter_path))
    assert observed["generate"]["prompt_ids"] == [1, 2, 3]
    assert observed["generate"]["max_draft_tokens"] is None
    assert result.text == "hello"
    assert result.input_tokens == 3
    assert result.output_tokens == 2
    assert result.tokens_per_second == 12.5
    assert metadata["backend"] == "mlx_dspark"
    assert metadata["mlx_dspark_mode"] == "dflash"
    assert metadata["mlx_dspark_drafter_revision"] == "a" * 40
    assert any(event.stage == "generating" and event.value == 2 for event in progress)
    assert FakeMX.reset_calls == 1
    assert FakeMX.clear_calls == 1


def test_modal_llm_schema_exposes_local_engine_without_forwarding_it_remotely(
    modal_llm_node_module: Any,
    modal_llm_runtime_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The selector belongs to local inference and must not alter Modal kwargs."""
    schema = modal_llm_node_module.ModalLLM.define_schema()
    assert "local_mlx_engine" in [input_value.id for input_value in schema.inputs]
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_result(target: str, **kwargs: Any) -> Any:
        """Record one node runner call."""
        calls.append((target, kwargs))
        return modal_llm_runtime_module.LLMInferenceResult(
            text="done",
            metadata={"output_tokens": 1},
        )

    monkeypatch.delenv("COMFY_MODAL_REMOTE_WORKER", raising=False)
    monkeypatch.setattr(
        modal_llm_node_module,
        "run_local_llm_inference",
        lambda **kwargs: fake_result("local", **kwargs),
    )
    modal_llm_node_module.ModalLLM.execute(
        prompt="hello",
        model_profile="smolvlm2-2.2b-instruct",
        local_mlx_engine="mlx-dspark",
        max_new_tokens=2,
    )
    monkeypatch.setenv("COMFY_MODAL_REMOTE_WORKER", "1")
    monkeypatch.setattr(
        modal_llm_node_module,
        "run_modal_llm_inference",
        lambda **kwargs: fake_result("remote", **kwargs),
    )
    modal_llm_node_module.ModalLLM.execute(
        prompt="hello",
        model_profile="smolvlm2-2.2b-instruct",
        local_mlx_engine="mlx-dspark",
        max_new_tokens=2,
    )

    assert calls[0][1]["local_mlx_engine"] == "mlx-dspark"
    assert "local_mlx_engine" not in calls[1][1]
