"""Opt-in live canaries for the deployed Modal execution path."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import os
import time
from typing import Any, Iterator
import uuid

import pytest


def _environment_flag_enabled(name: str) -> bool:
    """Return whether one opt-in environment flag is truthy."""
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


pytestmark = [
    pytest.mark.live_modal,
    pytest.mark.skipif(
        not _environment_flag_enabled("COMFY_MODAL_RUN_LIVE_CANARIES"),
        reason="set COMFY_MODAL_RUN_LIVE_CANARIES=1 to spend live Modal resources",
    ),
]


@dataclass
class _LiveModalCanaryContext:
    """Track the configured client and shared-state keys created by live canaries."""

    remote_module: Any
    settings: Any
    shared_store_keys: set[str] = field(default_factory=set)

    def payload(
        self,
        name: str,
        *,
        delay_seconds: float = 0.0,
        canary_barrier: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build one isolated dependency-light canary payload."""
        unique_suffix = uuid.uuid4().hex
        invocation_id = f"RIV_CANARY_{unique_suffix}"
        self.shared_store_keys.add(invocation_id)
        payload: dict[str, Any] = {
            "payload_kind": "canary",
            "prompt_id": f"live-canary-{name}-{unique_suffix}",
            "component_id": f"live-canary-{name}",
            "invocation_id": invocation_id,
            "canary_delay_seconds": delay_seconds,
            "requires_volume_reload": False,
            "terminate_container_on_error": False,
        }
        if canary_barrier is not None:
            payload["canary_barrier"] = canary_barrier
            barrier_id = str(canary_barrier["barrier_id"])
            for member_id in canary_barrier["members"]:
                self.shared_store_keys.add(
                    f"CANARY_BARRIER:{barrier_id}:{member_id}"
                )
        return payload

    def invoke(self, payload: dict[str, Any], value: Any) -> tuple[Any, dict[str, Any]]:
        """Invoke one live canary and deserialize its echoed value and metadata."""
        serialized_inputs = self.remote_module.serialize_node_inputs({"value": value})
        response = self.remote_module.invoke_remote_engine(
            payload,
            serialized_inputs,
            allow_implicit_mapping=False,
        )
        outputs = self.remote_module.deserialize_node_outputs(response)
        assert len(outputs) == 2
        assert isinstance(outputs[1], dict)
        return outputs[0], outputs[1]

    def invoke_node(
        self,
        name: str,
        class_type: str,
        inputs: dict[str, Any],
    ) -> tuple[Any, ...]:
        """Invoke one ordinary node through the deployed RemoteEngine."""
        payload = self.payload(name)
        payload.pop("payload_kind", None)
        payload["class_type"] = class_type
        payload["modal_gpu"] = self.settings.modal_gpu
        model_profile = inputs.get("model_profile")
        if isinstance(model_profile, str) and model_profile.strip():
            payload["subgraph_prompt"] = {
                "canary-node": {
                    "class_type": class_type,
                    "inputs": {"model_profile": model_profile},
                }
            }
        response = self.remote_module.invoke_remote_engine(
            payload,
            self.remote_module.serialize_node_inputs(inputs),
            allow_implicit_mapping=False,
        )
        return self.remote_module.deserialize_node_outputs(response)

    def cleanup_shared_state(self) -> None:
        """Remove invocation and barrier metadata created by the live canaries."""
        if not self.shared_store_keys:
            return
        modal_module = self.remote_module.modal
        invocation_store = modal_module.Dict.from_name(
            self.settings.invocation_dict_name,
            environment_name=self.remote_module._modal_environment_name(),
            create_if_missing=True,
        )
        for shared_store_key in self.shared_store_keys:
            invocation_store.pop(shared_store_key, None)


@pytest.fixture
def live_modal_canary(
    remote_modal_app_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[_LiveModalCanaryContext]:
    """Configure a real Modal client while keeping the normal suite local-only."""
    if remote_modal_app_module.modal is None:
        pytest.fail(
            "Live Modal canaries require the remote extra: "
            "uv run --extra remote pytest tests/test_live_modal_canary.py"
        )
    monkeypatch.setenv("COMFY_MODAL_EXECUTION_MODE", "remote")
    monkeypatch.setenv("COMFY_MODAL_AUTO_DEPLOY", "true")
    monkeypatch.setenv("COMFY_MODAL_ALLOW_EPHEMERAL_FALLBACK", "false")
    remote_modal_app_module.get_settings.cache_clear()
    settings = remote_modal_app_module.get_settings()
    context = _LiveModalCanaryContext(
        remote_module=remote_modal_app_module,
        settings=settings,
    )
    try:
        yield context
    finally:
        context.cleanup_shared_state()
        remote_modal_app_module.get_settings.cache_clear()


def test_live_modal_runtime_handshake(live_modal_canary: _LiveModalCanaryContext) -> None:
    """The deployed worker should echo data and match the local runtime fingerprint."""
    payload = live_modal_canary.payload("handshake")

    echoed_value, metadata = live_modal_canary.invoke(payload, "handshake-ok")
    remote_engine = live_modal_canary.remote_module._lookup_deployed_remote_engine(
        dict(payload)
    )
    version_payload = live_modal_canary.remote_module._remote_engine_runtime_version(
        remote_engine
    )

    assert echoed_value == "handshake-ok"
    assert metadata["component_id"] == "live-canary-handshake"
    assert live_modal_canary.remote_module._is_runtime_version_payload_current(
        version_payload
    )
    assert version_payload is not None
    assert version_payload["vllm_version"] == "0.27.1"


def test_live_modal_resident_llm_image_file_video_and_warm_reuse(
    live_modal_canary: _LiveModalCanaryContext,
) -> None:
    """The B300 worker should run every supported modality and reuse resident weights."""
    import base64
    from fractions import Fraction
    import json

    import torch
    from comfy_api.latest._input_impl.video_types import VideoFromComponents
    from comfy_api.latest._util import VideoComponents

    image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    image[:, 8:56, 8:56, 0] = 1.0
    file_payload = {
        "filename": "instruction.txt",
        "file_data": (
            "data:text/plain;base64,"
            + base64.b64encode(b"Reply with one short sentence.").decode("ascii")
        ),
        "type": "input_file",
    }
    first_outputs = live_modal_canary.invoke_node(
        "llm-image-file",
        "ModalLLM",
        {
            "prompt": "Describe the dominant colour and use the attached instruction.",
            "model_profile": "smolvlm2-2.2b-instruct",
            "images": image,
            "files": [file_payload],
            "system_prompt": "Answer plainly.",
            "max_new_tokens": 32,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 0,
            "video_frames": 3,
            "reserve_free_vram_gb": 24.0,
            "keep_model_loaded": True,
        },
    )
    first_metadata = json.loads(first_outputs[1])

    video = VideoFromComponents(
        VideoComponents(
            images=torch.stack(
                [
                    torch.zeros((48, 48, 3), dtype=torch.float32),
                    torch.ones((48, 48, 3), dtype=torch.float32),
                    torch.zeros((48, 48, 3), dtype=torch.float32),
                ]
            ),
            audio=None,
            frame_rate=Fraction(1, 1),
        ),
        bit_depth=8,
    )
    second_outputs = live_modal_canary.invoke_node(
        "llm-video-warm",
        "ModalLLM",
        {
            "prompt": "Briefly describe how the frames change.",
            "model_profile": "smolvlm2-2.2b-instruct",
            "video": video,
            "system_prompt": "",
            "max_new_tokens": 32,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 0,
            "video_frames": 3,
            "reserve_free_vram_gb": 24.0,
            "keep_model_loaded": True,
        },
    )
    second_metadata = json.loads(second_outputs[1])

    assert isinstance(first_outputs[0], str) and first_outputs[0].strip()
    assert first_metadata["image_count"] == 1
    assert first_metadata["file_count"] == 1
    assert first_metadata["output_tokens"] > 0
    assert isinstance(second_outputs[0], str) and second_outputs[0].strip()
    assert second_metadata["video_frame_count"] == 3
    assert second_metadata["cache_hit"] is True
    assert second_metadata["resident_profiles"] == ["smolvlm2-2.2b-instruct"]


@pytest.mark.parametrize(
    ("model_id", "expected_revision", "expected_backend"),
    [
        (
            "orcarouter/Qwen3.8-27B-Uncensored-FP8",
            "9228df5c6c9c509e1019f83b4e085cf643118bac",
            "vllm",
        ),
        (
            "meta-models/Muse-Glimmer-30B",
            "a4e59da52a7bc87ae7251dd5545c0dd437c44b68",
            "transformers",
        ),
        (
            "Blackfrost-AI/Qwen3.8-27B-ABLITERATED-NVFP4",
            "faf7945020c138c8ef864ab1644273f3158f85fa",
            "vllm",
        ),
    ],
)
def test_live_generated_profile_model_inference(
    live_modal_canary: _LiveModalCanaryContext,
    model_id: str,
    expected_revision: str,
    expected_backend: str,
) -> None:
    """Each requested Hub ID should resolve, stage, and understand one image."""
    import json

    import torch

    image = torch.zeros((1, 48, 48, 3), dtype=torch.float32)
    image[:, 8:40, 8:40, 1] = 1.0
    outputs = live_modal_canary.invoke_node(
        "generated-profile-" + expected_backend,
        "ModalLLM",
        {
            "prompt": (
                "State the dominant colour of the central square in one short sentence."
            ),
            "model_profile": model_id,
            "images": image,
            "system_prompt": "Answer plainly.",
            "max_new_tokens": 24,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 0,
            "video_frames": 1,
            "reserve_free_vram_gb": 48.0,
            "keep_model_loaded": False,
        },
    )
    metadata = json.loads(outputs[1])

    assert isinstance(outputs[0], str) and outputs[0].strip()
    assert metadata["backend"] == expected_backend
    assert metadata["repository"] == model_id
    assert metadata["revision"] == expected_revision
    assert metadata["profile"].startswith("hf-")
    assert metadata["image_count"] == 1
    assert metadata["output_tokens"] > 0


def test_live_modal_llm_and_comfy_vae_are_co_resident(
    live_modal_canary: _LiveModalCanaryContext,
    sync_engine_module: Any,
) -> None:
    """A real ComfyUI image VAE and the selected LLM should share one B300 worker."""
    import json
    from pathlib import Path

    vae_path = Path(
        os.getenv(
            "COMFY_MODAL_LLM_CANARY_VAE",
            "/Users/ksimpson/git/Latest_ComfyUI/models/vae/flux2-vae.safetensors",
        )
    ).expanduser()
    if not vae_path.is_file():
        pytest.skip(f"resident LLM co-residency canary VAE is missing: {vae_path}")
    if live_modal_canary.settings.max_containers != 1:
        pytest.skip("co-residency canary requires COMFY_MODAL_MAX_CONTAINERS=1")
    model_profile = os.getenv(
        "COMFY_MODAL_LLM_CANARY_PROFILE",
        "smolvlm2-2.2b-instruct",
    )

    sync_engine = sync_engine_module.ModalAssetSyncEngine.from_environment(
        live_modal_canary.settings
    )
    synced_vae = sync_engine.sync_file(vae_path)

    initial_outputs = live_modal_canary.invoke_node(
        "llm-before-vae",
        "ModalLLM",
        {
            "prompt": "Reply with the word ready.",
            "model_profile": model_profile,
            "max_new_tokens": 8,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 0,
            "video_frames": 1,
            "reserve_free_vram_gb": 24.0,
            "keep_model_loaded": True,
        },
    )
    initial_metadata = json.loads(initial_outputs[1])
    resident_profile = initial_metadata["profile"]
    assert initial_metadata["resident_profiles"] == [resident_profile]

    vae_payload = live_modal_canary.payload("comfy-vae")
    vae_payload.update(
        {
            "payload_kind": "subgraph",
            "modal_gpu": live_modal_canary.settings.modal_gpu,
            "component_node_ids": ["empty-image", "vae-loader", "vae-encode"],
            "subgraph_prompt": {
                "empty-image": {
                    "class_type": "EmptyImage",
                    "inputs": {
                        "width": 64,
                        "height": 64,
                        "batch_size": 1,
                        "color": 0x336699,
                    },
                },
                "vae-loader": {
                    "class_type": "VAELoader",
                    "inputs": {"vae_name": synced_vae.remote_path},
                },
                "vae-encode": {
                    "class_type": "VAEEncode",
                    "inputs": {
                        "pixels": ["empty-image", 0],
                        "vae": ["vae-loader", 0],
                    },
                },
            },
            "boundary_inputs": [],
            "boundary_outputs": [
                {
                    "proxy_output_name": "latent",
                    "node_id": "vae-encode",
                    "output_index": 0,
                    "io_type": "LATENT",
                    "is_list": False,
                }
            ],
            "execute_node_ids": ["vae-encode"],
            "extra_data": {},
            "uploaded_volume_paths": [synced_vae.remote_path],
            "requires_volume_reload": True,
            "volume_reload_marker": f"llm-vae-{synced_vae.sha256}",
        }
    )
    vae_response = live_modal_canary.remote_module.invoke_remote_engine(
        vae_payload,
        live_modal_canary.remote_module.serialize_node_inputs({}),
        allow_implicit_mapping=False,
    )
    vae_outputs = live_modal_canary.remote_module.deserialize_node_outputs(vae_response)
    assert len(vae_outputs) == 1
    assert "samples" in vae_outputs[0]

    final_outputs = live_modal_canary.invoke_node(
        "llm-after-vae",
        "ModalLLM",
        {
            "prompt": "Reply with the word resident.",
            "model_profile": model_profile,
            "max_new_tokens": 8,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 0,
            "video_frames": 1,
            "reserve_free_vram_gb": 24.0,
            "keep_model_loaded": True,
        },
    )
    final_metadata = json.loads(final_outputs[1])

    assert final_metadata["cache_hit"] is True
    assert final_metadata["profile"] == resident_profile
    assert final_metadata["resident_profiles"] == [resident_profile]
    assert final_metadata["gpu_total_gib"] > 200


def test_live_modal_binary_transport_and_durable_replay(
    live_modal_canary: _LiveModalCanaryContext,
) -> None:
    """Tensor RPC should stay binary and a duplicate call should replay exact metadata."""
    import torch

    payload = live_modal_canary.payload("binary-replay")
    value = torch.arange(1024 * 1024, dtype=torch.float32).reshape(1, 1024, 1024)

    first_value, first_metadata = live_modal_canary.invoke(payload, value)
    replayed_value, replayed_metadata = live_modal_canary.invoke(payload, value)

    assert torch.equal(first_value, value)
    assert torch.equal(replayed_value, value)
    assert first_metadata["transport_kind"] == "binary"
    assert replayed_metadata == first_metadata


def test_live_modal_parallel_dispatch_reaches_barrier(
    live_modal_canary: _LiveModalCanaryContext,
) -> None:
    """Two remote calls should be active together instead of serializing on one worker."""
    if live_modal_canary.settings.max_inflight_calls < 2:
        pytest.skip("parallel canary requires COMFY_MODAL_MAX_INFLIGHT_CALLS >= 2")
    if (
        live_modal_canary.settings.max_containers is not None
        and live_modal_canary.settings.max_containers < 2
    ):
        pytest.skip("parallel canary requires COMFY_MODAL_MAX_CONTAINERS >= 2")

    barrier_id = f"parallel-{uuid.uuid4().hex}"
    members = ["member-a", "member-b"]
    payloads = [
        live_modal_canary.payload(
            member_id,
            delay_seconds=0.25,
            canary_barrier={
                "barrier_id": barrier_id,
                "member_id": member_id,
                "members": members,
                "timeout_seconds": 90.0,
            },
        )
        for member_id in members
    ]

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(live_modal_canary.invoke, payload, member_id)
            for payload, member_id in zip(payloads, members, strict=True)
        ]
        results = [future.result(timeout=180.0) for future in futures]

    metadata = [result[1] for result in results]
    assert [result[0] for result in results] == members
    assert all(item["barrier_released_at"] is not None for item in metadata)
    assert len({item["modal_task_id"] for item in metadata}) == 2


def _wait_for_active_prompt(
    context: _LiveModalCanaryContext,
    prompt_id: str,
    *,
    timeout_seconds: float,
) -> None:
    """Wait until the local client has registered a cancellable Modal invocation."""
    deadline = time.monotonic() + timeout_seconds
    while prompt_id not in context.remote_module.active_remote_modal_prompt_ids():
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Live Modal invocation {prompt_id!r} never became cancellable."
            )
        time.sleep(0.05)


def test_live_modal_cancellation_propagates(
    live_modal_canary: _LiveModalCanaryContext,
) -> None:
    """Prompt cancellation should reach and stop a deliberately delayed remote call."""
    import comfy.model_management

    payload = live_modal_canary.payload("cancellation", delay_seconds=30.0)
    executor = ThreadPoolExecutor(max_workers=1)
    future: Future[tuple[Any, dict[str, Any]]] = executor.submit(
        live_modal_canary.invoke,
        payload,
        "must-not-complete",
    )
    try:
        _wait_for_active_prompt(
            live_modal_canary,
            str(payload["prompt_id"]),
            timeout_seconds=120.0,
        )
        assert live_modal_canary.remote_module.request_remote_modal_prompt_interrupt(
            str(payload["prompt_id"])
        )
        with pytest.raises(comfy.model_management.InterruptProcessingException):
            future.result(timeout=15.0)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
