"""Tests for curated, resident multimodal LLM inference."""

from __future__ import annotations

import base64
from dataclasses import dataclass, replace
from fractions import Fraction
import json
from io import BytesIO
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
    """The registry should reject drift and payload discovery should cross split phases."""
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

    assert llm_profiles_module.llm_profile_ids_from_payload(payload) == (profile.profile_id,)


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


def test_cpu_stager_writes_completion_marker_and_reuses_snapshot(
    llm_staging_module: Any,
    tmp_path: Path,
) -> None:
    """A completed immutable snapshot should not download twice."""
    calls: list[dict[str, Any]] = []

    def fake_snapshot_download(**kwargs: Any) -> str:
        """Materialize the minimum expected Hugging Face snapshot."""
        calls.append(kwargs)
        snapshot_path = Path(kwargs["local_dir"])
        snapshot_path.mkdir(parents=True, exist_ok=True)
        (snapshot_path / "config.json").write_text("{}", encoding="utf-8")
        return str(snapshot_path)

    first = llm_staging_module.stage_model_profile(
        "smolvlm2-2.2b-instruct",
        tmp_path,
        snapshot_download=fake_snapshot_download,
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
    assert llm_staging_module.is_model_snapshot_staged(
        tmp_path,
        llm_staging_module.get_llm_profile("smolvlm2-2.2b-instruct"),
    )


def test_multimodal_preparation_samples_video_and_bounds_files(
    modal_llm_runtime_module: Any,
    llm_profiles_module: Any,
) -> None:
    """Image tensors, native video, and files should normalize without transport encoding."""
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

    def cancel_on_second_token(token_count: int) -> None:
        """Raise a cancellation marker on the second generation step."""
        calls.append(token_count)
        if token_count == 2:
            raise InterruptedError("cancelled")

    criterion = modal_llm_runtime_module._stopping_criteria(cancel_on_second_token)[0]

    assert criterion(None, None) is False
    with pytest.raises(InterruptedError, match="cancelled"):
        criterion(None, None)
    assert calls == [1, 2]


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
        progress_callback: Callable[[int], None],
    ) -> Any:
        """Return deterministic token counts while exercising progress."""
        del prepared_inputs
        self.generate_calls += 1
        progress_callback(1)
        return SimpleNamespace(
            text=f"response:{self.profile_id}:{settings.seed}",
            input_tokens=7,
            output_tokens=1,
        )

    def unload(self) -> None:
        """Record that the cache released this backend."""
        self.unloaded = True


def test_resident_manager_reuses_and_lru_evicts_models(
    modal_llm_runtime_module: Any,
    llm_profiles_module: Any,
    tmp_path: Path,
) -> None:
    """Warm requests should hit cache and a bounded cache should evict least-recently used."""
    base_profile = llm_profiles_module.get_llm_profile("smolvlm2-2.2b-instruct")
    second_profile = replace(
        base_profile,
        profile_id="second-profile",
        revision="1" * 40,
    )
    backends: dict[str, _FakeBackend] = {}

    def backend_factory(profile: Any, snapshot_path: Path) -> _FakeBackend:
        """Create one fake backend per load."""
        del snapshot_path
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
    assert second.metadata["cache_hit"] is True
    assert backends[base_profile.profile_id].generate_calls == 2
    assert backends[base_profile.profile_id].unloaded is True
    assert manager.resident_profiles() == (second_profile.profile_id,)


def test_modal_llm_node_is_v3_remote_only_and_returns_metadata(
    modal_llm_node_module: Any,
    modal_llm_runtime_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The node should register a V3 contract and refuse accidental local model loading."""
    schema = modal_llm_node_module.ModalLLM.define_schema()
    assert schema.node_id == "ModalLLM"
    assert [output.display_name for output in schema.outputs] == ["response", "metadata_json"]

    monkeypatch.delenv("COMFY_MODAL_REMOTE_WORKER", raising=False)
    with pytest.raises(RuntimeError, match="Enable 'Run on Modal'"):
        modal_llm_node_module.ModalLLM.execute(
            prompt="hello",
            model_profile="smolvlm2-2.2b-instruct",
        )

    monkeypatch.setenv("COMFY_MODAL_REMOTE_WORKER", "1")
    monkeypatch.setattr(
        modal_llm_node_module,
        "run_modal_llm_inference",
        lambda **kwargs: modal_llm_runtime_module.LLMInferenceResult(
            text="done",
            metadata={"output_tokens": 1, "cache_hit": True},
        ),
    )
    node_output = modal_llm_node_module.ModalLLM.execute(
        prompt="hello",
        model_profile="smolvlm2-2.2b-instruct",
        max_new_tokens=4,
    )

    assert node_output.result[0] == "done"
    assert json.loads(node_output.result[1])["cache_hit"] is True


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
            return [{"profile_id": profile_id} for profile_id in profile_ids]

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
