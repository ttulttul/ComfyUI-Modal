"""Tests for curated, resident multimodal LLM inference."""

from __future__ import annotations

import asyncio
import base64
import json
import sys
from dataclasses import dataclass, replace
from fractions import Fraction
from io import BytesIO, StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest
import torch


def _text_file(filename: str, text: str) -> dict[str, str]:
    """Return one built-in-compatible OpenAI input-file payload."""
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    return {
        "filename": filename,
        "file_data": f"data:text/plain;base64,{encoded}",
        "type": "input_file",
    }


def test_curated_profile_is_revision_pinned_and_found_in_nested_payload(
    llm_profiles_module: Any,
) -> None:
    """The registry should reject drift across split-payload discovery."""
    profiles = llm_profiles_module.load_llm_profiles()
    profile = profiles["smolvlm2-2.2b-instruct"]

    assert profile.revision == "482adb537c021c86670beed01cd58990d01e72e4"
    assert profile.modalities == frozenset({"text", "image", "video", "file"})
    assert profile.trust_remote_code is False
    payload = {
        "split_proxy_payloads": [
            {
                "subgraph_prompt": {
                    "12": {
                        "class_type": llm_profiles_module.MODAL_LLM_NODE_ID,
                        "inputs": {"model_profile": profile.profile_id},
                    }
                }
            }
        ]
    }

    assert llm_profiles_module.llm_profile_ids_from_payload(payload) == (
        profile.profile_id,
    )


def test_profile_registry_rejects_mutable_revision(
    llm_profiles_module: Any,
) -> None:
    """Curated profiles must use an immutable commit rather than main."""
    with pytest.raises(ValueError, match="exact 40-character"):
        llm_profiles_module.LLMModelProfile.from_mapping(
            {
                "id": "unsafe",
                "repository": "owner/model",
                "revision": "main",
                "dtype": "bfloat16",
                "modalities": ["text"],
                "estimated_vram_gb": 1,
                "max_context_tokens": 1024,
                "max_images": 1,
                "max_video_frames": 1,
                "max_file_bytes": 1,
                "max_file_characters": 1,
            }
        )


@pytest.mark.parametrize(
    ("architecture", "quantization_config", "backend", "quantization_method"),
    [
        (
            "Qwen3_5ForConditionalGeneration",
            {"quant_method": "fp8", "fmt": "e4m3"},
            "vllm",
            "fp8",
        ),
        (
            "Qwen3_5ForConditionalGeneration",
            {"quant_method": "modelopt", "quant_algo": "NVFP4"},
            "vllm",
            "modelopt_fp4",
        ),
        (
            "MuseGlimmerForConditionalGeneration",
            {},
            "transformers",
            "none",
        ),
    ],
)
def test_compatibility_policy_selects_requested_model_backends(
    llm_compatibility_module: Any,
    architecture: str,
    quantization_config: dict[str, str],
    backend: str,
    quantization_method: str,
) -> None:
    """The three live-canary model families should resolve before weight download."""
    decision = llm_compatibility_module.resolve_compatibility(
        {
            "architectures": [architecture],
            "dtype": "bfloat16",
            "text_config": {"max_position_embeddings": 262144},
            "quantization_config": quantization_config,
        },
        artifact_bytes=32 * 1024**3,
    )

    assert decision.backend == backend
    assert decision.quantization_method == quantization_method
    assert decision.default_context_tokens == 32768
    assert decision.advertised_context_tokens == 262144
    assert decision.reasoning_parser == ("qwen3" if backend == "vllm" else "none")


class _FakeReasoningTokenizer:
    """Decode deterministic IDs while modelling Qwen's special think tokens."""

    _vocabulary = {"<think>": 10, "</think>": 11}
    _text = {1: "consider", 2: " carefully", 3: "final answer", 10: "", 11: ""}

    def get_vocab(self) -> dict[str, int]:
        """Return exact reasoning boundary token IDs."""
        return dict(self._vocabulary)

    def decode(self, token_ids: Any, **kwargs: Any) -> str:
        """Decode content while respecting special-token cleanup."""
        del kwargs
        return "".join(self._text[token_id] for token_id in token_ids)


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


def test_generated_profile_requires_matching_content_digest(
    llm_profiles_module: Any,
) -> None:
    """A generated manifest must not be mutable under a stable profile identifier."""
    digest = "a" * 64
    with pytest.raises(ValueError, match="does not match its content digest"):
        llm_profiles_module.LLMModelProfile.from_mapping(
            {
                "id": "hf-" + "b" * 64,
                "repository": "owner/model",
                "revision": "1" * 40,
                "dtype": "bfloat16",
                "modalities": ["text"],
                "estimated_vram_gb": 10,
                "max_context_tokens": 1024,
                "max_images": 1,
                "max_video_frames": 1,
                "max_file_bytes": 1,
                "max_file_characters": 1,
                "schema_version": 2,
                "source": "generated",
                "profile_digest": digest,
                "backend": "transformers",
            }
        )


def test_cpu_resolver_pins_and_persists_generated_profile(
    llm_resolver_module: Any,
    llm_profiles_module: Any,
    tmp_path: Path,
) -> None:
    """One model ID should become a stable manifest without downloading weights."""
    revision = "9" * 40
    config_path = tmp_path / "downloaded-config.json"
    config_path.write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "dtype": "bfloat16",
                "text_config": {"max_position_embeddings": 262144},
                "quantization_config": {
                    "quant_method": "modelopt",
                    "quant_algo": "NVFP4",
                },
            }
        ),
        encoding="utf-8",
    )
    model_info = {
        "id": "owner/model",
        "sha": revision,
        "siblings": [
            {"rfilename": "config.json", "size": 1000},
            {"rfilename": "model-00001.safetensors", "size": 20 * 1024**3},
            {"rfilename": "model-00002.safetensors", "size": 10 * 1024**3},
        ],
        "securityStatus": {"scansDone": True, "filesWithIssues": []},
    }

    class FakeApi:
        """Return immutable test metadata from model_info."""

        def model_info(self, repo_id: str, **kwargs: Any) -> dict[str, Any]:
            """Validate the metadata-only resolver request."""
            assert repo_id == "owner/model"
            assert kwargs["revision"] == "release"
            assert kwargs["files_metadata"] is True
            return model_info

    download_calls: list[dict[str, Any]] = []

    def fake_hf_hub_download(**kwargs: Any) -> str:
        """Return only the config file during resolution."""
        download_calls.append(kwargs)
        return str(config_path)

    first = llm_resolver_module.resolve_model_profile(
        "owner/model@release",
        tmp_path,
        api=FakeApi(),
        hf_hub_download=fake_hf_hub_download,
    )
    second = llm_resolver_module.resolve_model_profile(
        "owner/model@release",
        tmp_path,
        api=FakeApi(),
        hf_hub_download=fake_hf_hub_download,
    )

    assert first.profile.profile_id == second.profile.profile_id
    assert first.profile.revision == revision
    assert first.profile.backend == "vllm"
    assert first.profile.reasoning_parser == "qwen3"
    assert first.profile.quantization_method == "modelopt_fp4"
    assert first.profile.artifact_bytes == 30 * 1024**3
    assert first.manifest_created is True
    assert second.manifest_created is False
    assert len(download_calls) == 2
    loaded = llm_profiles_module.get_llm_profile(
        first.profile.profile_id,
        storage_root=tmp_path,
    )
    assert loaded == first.profile


def test_cpu_resolver_rejects_unknown_architecture_before_weights(
    llm_resolver_module: Any,
    tmp_path: Path,
) -> None:
    """Compatibility errors should happen after config-only inspection."""
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "architectures": ["UnreviewedForConditionalGeneration"],
                "max_position_embeddings": 4096,
            }
        ),
        encoding="utf-8",
    )

    class FakeApi:
        """Return a safe but unsupported test repository."""

        def model_info(self, repo_id: str, **kwargs: Any) -> dict[str, Any]:
            """Return metadata without performing network access."""
            del repo_id, kwargs
            return {
                "sha": "8" * 40,
                "siblings": [
                    {"rfilename": "config.json", "size": 1},
                    {"rfilename": "model.safetensors", "size": 1024},
                ],
                "securityStatus": {"scansDone": True},
            }

    with pytest.raises(ValueError, match="not supported"):
        llm_resolver_module.resolve_model_profile(
            "owner/unknown",
            tmp_path,
            api=FakeApi(),
            hf_hub_download=lambda **kwargs: str(config_path),
        )
    assert not (tmp_path / "llm_models").exists()


def test_cpu_resolver_wraps_gated_config_download_error(
    llm_resolver_module: Any,
    tmp_path: Path,
) -> None:
    """A gated config failure should cross Modal as a plain actionable ValueError."""

    class FakeApi:
        """Return metadata visible without gated file access."""

        def model_info(self, repo_id: str, **kwargs: Any) -> dict[str, Any]:
            """Return one otherwise compatible repository."""
            del repo_id, kwargs
            return {
                "sha": "8" * 40,
                "siblings": [
                    {"rfilename": "config.json", "size": 1},
                    {"rfilename": "model.safetensors", "size": 1024},
                ],
                "securityStatus": {"scansDone": True},
            }

    def denied_download(**kwargs: Any) -> str:
        """Simulate gated file access after public metadata resolution."""
        del kwargs
        raise OSError("401 Unauthorized")

    with pytest.raises(
        ValueError,
        match="Unable to download config.json.*HF_TOKEN",
    ):
        llm_resolver_module.resolve_model_profile(
            "owner/gated",
            tmp_path,
            api=FakeApi(),
            hf_hub_download=denied_download,
        )


def test_cpu_stager_writes_completion_marker_and_reuses_snapshot(
    llm_staging_module: Any,
    tmp_path: Path,
) -> None:
    """A completed immutable snapshot should not download twice."""
    calls: list[dict[str, Any]] = []
    progress: list[Any] = []

    def fake_snapshot_download(**kwargs: Any) -> str:
        """Materialize the minimum expected Hugging Face snapshot."""
        calls.append(kwargs)
        snapshot_path = Path(kwargs["local_dir"])
        snapshot_path.mkdir(parents=True, exist_ok=True)
        (snapshot_path / "config.json").write_text("{}", encoding="utf-8")
        progress_bar = kwargs["tqdm_class"](
            total=2,
            unit="files",
            file=StringIO(),
        )
        progress_bar.update(1)
        progress_bar.update(1)
        progress_bar.close()
        return str(snapshot_path)

    first = llm_staging_module.stage_model_profile(
        "smolvlm2-2.2b-instruct",
        tmp_path,
        snapshot_download=fake_snapshot_download,
        progress_callback=progress.append,
    )
    second = llm_staging_module.stage_model_profile(
        "smolvlm2-2.2b-instruct",
        tmp_path,
        snapshot_download=fake_snapshot_download,
    )

    assert first.downloaded is True
    assert second.downloaded is False
    assert len(calls) == 1
    assert calls[0]["revision"] == "482adb537c021c86670beed01cd58990d01e72e4"
    assert "*.safetensors" in calls[0]["allow_patterns"]
    assert "*.bin" not in calls[0]["allow_patterns"]
    assert calls[0]["tqdm_class"].__name__ == "SnapshotProgressTqdm"
    assert [event.value for event in progress if event.stage == "download"] == [
        0.0,
        1.0,
        2.0,
    ]
    assert llm_staging_module.is_model_snapshot_staged(
        tmp_path,
        llm_staging_module.get_llm_profile("smolvlm2-2.2b-instruct"),
    )


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
                outputs=[
                    SimpleNamespace(text="<think>consider", token_ids=[10, 1])
                ],
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
    assert observed["generate_kwargs"]["sampling_params"].kwargs[
        "skip_special_tokens"
    ] is False
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
        asyncio.run(
            backend._generate_async(prepared, settings, lambda progress: None)
        )


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


@dataclass
class _FakeBackend:
    """Record resident backend inference and unload behavior."""

    profile_id: str
    unloaded: bool = False
    generate_calls: int = 0

    def generate(
        self,
        prepared_inputs: Any,
        settings: Any,
        progress_callback: Callable[[Any], None],
    ) -> Any:
        """Return deterministic token counts while exercising progress."""
        del prepared_inputs
        self.generate_calls += 1
        progress_callback(SimpleNamespace(stage="generating", value=1))
        return SimpleNamespace(
            text=f"response:{self.profile_id}:{settings.seed}",
            input_tokens=7,
            output_tokens=1,
            reasoning="",
            reasoning_tokens=0,
            reasoning_parser="none",
            time_to_first_token_seconds=0.25,
            tokens_per_second=4.0,
        )

    def unload(self) -> None:
        """Record that the cache released this backend."""
        self.unloaded = True


def test_resident_manager_reuses_and_lru_evicts_models(
    modal_llm_runtime_module: Any,
    llm_profiles_module: Any,
    tmp_path: Path,
) -> None:
    """Warm requests should hit cache and evict the least-recently used model."""
    base_profile = llm_profiles_module.get_llm_profile("smolvlm2-2.2b-instruct")
    second_profile = replace(
        base_profile,
        profile_id="second-profile",
        revision="1" * 40,
    )
    backends: dict[str, _FakeBackend] = {}

    def backend_factory(
        profile: Any,
        snapshot_path: Path,
        progress_callback: Callable[[Any], None],
    ) -> _FakeBackend:
        """Create one fake backend per load."""
        del snapshot_path, progress_callback
        backend = _FakeBackend(profile.profile_id)
        backends[profile.profile_id] = backend
        return backend

    manager = modal_llm_runtime_module.ResidentLLMManager(
        storage_root=tmp_path,
        backend_factory=backend_factory,
        max_resident_models=1,
        memory_info=lambda: (200 * 1024**3, 256 * 1024**3),
        empty_cache=lambda: None,
        snapshot_ready=lambda storage_root, profile: True,
        comfy_memory_release=lambda required_bytes: None,
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
        seed=3,
    )

    first = manager.infer(
        profile=base_profile,
        prepared_inputs=prepared,
        generation_settings=settings,
        reserve_free_vram_gb=0,
        keep_model_loaded=True,
        progress_callback=lambda value: None,
    )
    second = manager.infer(
        profile=base_profile,
        prepared_inputs=prepared,
        generation_settings=settings,
        reserve_free_vram_gb=0,
        keep_model_loaded=True,
        progress_callback=lambda value: None,
    )
    manager.infer(
        profile=second_profile,
        prepared_inputs=prepared,
        generation_settings=settings,
        reserve_free_vram_gb=0,
        keep_model_loaded=True,
        progress_callback=lambda value: None,
    )

    assert first.metadata["cache_hit"] is False
    assert first.metadata["reasoning_enabled"] is True
    assert second.metadata["cache_hit"] is True
    assert backends[base_profile.profile_id].generate_calls == 2
    assert backends[base_profile.profile_id].unloaded is True
    assert manager.resident_profiles() == (second_profile.profile_id,)


def test_modal_llm_node_is_v3_remote_only_and_returns_metadata(
    modal_llm_node_module: Any,
    modal_llm_runtime_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The V3 node should refuse accidental local model loading."""
    schema = modal_llm_node_module.ModalLLM.define_schema()
    assert schema.node_id == "ModalLLM"
    assert [output.display_name for output in schema.outputs] == [
        "response",
        "metadata_json",
        "reasoning",
    ]
    assert "enable_reasoning" in [input_value.id for input_value in schema.inputs]

    monkeypatch.delenv("COMFY_MODAL_REMOTE_WORKER", raising=False)
    with pytest.raises(RuntimeError, match="Enable 'Run on Modal'"):
        modal_llm_node_module.ModalLLM.execute(
            prompt="hello",
            model_profile="smolvlm2-2.2b-instruct",
        )

    monkeypatch.setenv("COMFY_MODAL_REMOTE_WORKER", "1")
    observed: dict[str, Any] = {}

    def fake_inference(**kwargs: Any) -> Any:
        """Capture node controls and return deterministic inference output."""
        observed.update(kwargs)
        return modal_llm_runtime_module.LLMInferenceResult(
            text="done",
            metadata={"output_tokens": 1, "cache_hit": True},
            reasoning="working",
        )

    monkeypatch.setattr(
        modal_llm_node_module,
        "run_modal_llm_inference",
        fake_inference,
    )
    node_output = modal_llm_node_module.ModalLLM.execute(
        prompt="hello",
        model_profile="smolvlm2-2.2b-instruct",
        enable_reasoning=False,
        max_new_tokens=4,
    )

    assert node_output.result[0] == "done"
    assert json.loads(node_output.result[1])["cache_hit"] is True
    assert node_output.result[2] == "working"
    assert observed["enable_reasoning"] is False


def test_remote_dispatch_stages_llm_once_and_forces_volume_reload(
    remote_modal_app_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The local dispatcher should finish CPU staging before sending work to a GPU."""
    stage_calls: list[list[str]] = []

    class FakeStageMethod:
        """Expose the Modal method remote-call surface."""

        def remote(self, profile_ids: list[str]) -> list[dict[str, str]]:
            """Record and confirm every requested profile."""
            stage_calls.append(profile_ids)
            return [
                {
                    "requested_reference": profile_id,
                    "profile_id": profile_id,
                    "revision": "482adb537c021c86670beed01cd58990d01e72e4",
                }
                for profile_id in profile_ids
            ]

    class FakeStager:
        """Expose the deployed CPU staging method."""

        stage_profiles = FakeStageMethod()

    class FakeRemoteClass:
        """Construct one fake deployed stager instance."""

        def __call__(self) -> FakeStager:
            """Return a fake stager."""
            return FakeStager()

    class FakeCls:
        """Resolve only the expected deployed stager class."""

        @staticmethod
        def from_name(app_name: str, class_name: str) -> FakeRemoteClass:
            """Return a fake class handle after validating lookup identity."""
            assert app_name == "test-b300-app"
            assert class_name == "ModelStager"
            return FakeRemoteClass()

    monkeypatch.setattr(remote_modal_app_module, "modal", SimpleNamespace(Cls=FakeCls))
    with remote_modal_app_module._STAGED_LLM_PROFILES_LOCK:
        remote_modal_app_module._STAGED_LLM_PROFILES.clear()
        remote_modal_app_module._STAGED_LLM_PROFILE_RESULTS.clear()
    payload = {
        "component_id": "llm-component",
        "subgraph_prompt": {
            "1": {
                "class_type": "ModalLLM",
                "inputs": {"model_profile": "smolvlm2-2.2b-instruct"},
            }
        },
    }

    remote_modal_app_module._ensure_llm_profiles_staged(payload, "test-b300-app")
    first_marker = payload["volume_reload_marker"]
    remote_modal_app_module._ensure_llm_profiles_staged(payload, "test-b300-app")

    assert stage_calls == [["smolvlm2-2.2b-instruct"]]
    assert payload["requires_volume_reload"] is True
    assert payload["volume_reload_marker"] == first_marker


def test_remote_dispatch_rewrites_hugging_face_id_to_generated_profile(
    remote_modal_app_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GPU payloads should contain only immutable generated profile IDs."""
    requested_model = "owner/model"
    generated_profile_id = "hf-" + "a" * 64

    class FakeStageMethod:
        """Resolve one user model ID into an immutable profile."""

        def remote(self, model_references: list[str]) -> list[dict[str, Any]]:
            """Return one generated profile result."""
            assert model_references == [requested_model]
            return [
                {
                    "requested_reference": requested_model,
                    "profile_id": generated_profile_id,
                    "revision": "7" * 40,
                    "downloaded": True,
                }
            ]

    class FakeCls:
        """Resolve the CPU ModelStager class."""

        @staticmethod
        def from_name(app_name: str, class_name: str) -> Callable[[], Any]:
            """Return a staging class constructor."""
            assert app_name == "test-b300-app"
            assert class_name == "ModelStager"
            return lambda: SimpleNamespace(stage_profiles=FakeStageMethod())

    monkeypatch.setattr(remote_modal_app_module, "modal", SimpleNamespace(Cls=FakeCls))
    with remote_modal_app_module._STAGED_LLM_PROFILES_LOCK:
        remote_modal_app_module._STAGED_LLM_PROFILES.clear()
        remote_modal_app_module._STAGED_LLM_PROFILE_RESULTS.clear()
    payload = {
        "component_id": "dynamic-llm",
        "subgraph_prompt": {
            "1": {
                "class_type": "ModalLLM",
                "inputs": {"model_profile": requested_model},
            }
        },
    }

    remote_modal_app_module._ensure_llm_profiles_staged(payload, "test-b300-app")
    direct_inputs = remote_modal_app_module.serialize_node_inputs(
        {"model_profile": requested_model, "prompt": "hello"}
    )
    rewritten_inputs = remote_modal_app_module.deserialize_node_inputs(
        remote_modal_app_module._rewrite_staged_llm_kwargs_payload(
            direct_inputs,
            "test-b300-app",
        )
    )

    assert (
        payload["subgraph_prompt"]["1"]["inputs"]["model_profile"]
        == generated_profile_id
    )
    assert rewritten_inputs == {
        "model_profile": generated_profile_id,
        "prompt": "hello",
    }
    assert payload["requires_volume_reload"] is True


def test_remote_dispatch_streams_cpu_llm_staging_progress(
    remote_modal_app_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First-run Hugging Face progress should render before GPU allocation."""
    observed_progress: list[dict[str, Any]] = []

    class FakeStageStream:
        """Yield one download update followed by the immutable profile result."""

        def remote_gen(self, model_references: list[str]) -> Any:
            """Stream deterministic CPU staging envelopes."""
            assert model_references == ["owner/model"]
            yield {
                "kind": "progress",
                "stage": "download",
                "message": "Fetching 8 files",
                "value": 3,
                "max": 8,
                "unit": "files",
            }
            yield {
                "kind": "result",
                "results": [
                    {
                        "requested_reference": "owner/model",
                        "profile_id": "hf-" + "b" * 64,
                        "revision": "8" * 40,
                    }
                ],
            }

    class FakeCls:
        """Resolve the streaming CPU ModelStager."""

        @staticmethod
        def from_name(app_name: str, class_name: str) -> Callable[[], Any]:
            """Return a deterministic streaming stager instance."""
            assert app_name == "test-app"
            assert class_name == "ModelStager"
            return lambda: SimpleNamespace(stage_profiles_stream=FakeStageStream())

    monkeypatch.setattr(remote_modal_app_module, "modal", SimpleNamespace(Cls=FakeCls))
    monkeypatch.setattr(
        remote_modal_app_module,
        "_emit_local_llm_staging_progress",
        lambda payload, event: observed_progress.append(dict(event)),
    )
    with remote_modal_app_module._STAGED_LLM_PROFILES_LOCK:
        remote_modal_app_module._STAGED_LLM_PROFILES.clear()
        remote_modal_app_module._STAGED_LLM_PROFILE_RESULTS.clear()
    payload = {
        "prompt_id": "prompt-1",
        "component_id": "llm-node",
        "extra_data": {"client_id": "client-1"},
        "subgraph_prompt": {
            "llm-node": {
                "class_type": "ModalLLM",
                "inputs": {"model_profile": "owner/model"},
            }
        },
    }

    remote_modal_app_module._ensure_llm_profiles_staged(payload, "test-app")

    assert observed_progress == [
        {
            "kind": "progress",
            "stage": "download",
            "message": "Fetching 8 files",
            "value": 3,
            "max": 8,
            "unit": "files",
        }
    ]
    assert (
        payload["subgraph_prompt"]["llm-node"]["inputs"]["model_profile"]
        == "hf-" + "b" * 64
    )


def test_remote_runtime_registers_deployment_owned_llm_node(
    modal_cloud_module: Any,
) -> None:
    """The remote image should expose ModalLLM even when custom-node sync is absent."""
    nodes_module = SimpleNamespace(
        NODE_CLASS_MAPPINGS={},
        NODE_DISPLAY_NAME_MAPPINGS={},
    )

    modal_cloud_module._register_modal_sync_runtime_nodes(nodes_module)

    assert nodes_module.NODE_CLASS_MAPPINGS["ModalLLM"].__name__ == "ModalLLM"
    assert nodes_module.NODE_DISPLAY_NAME_MAPPINGS["ModalLLM"] == "Modal LLM"
