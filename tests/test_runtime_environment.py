"""Tests for pinned remote environment and deployment fingerprints."""

from __future__ import annotations

import shlex
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


def _runtime_settings(**overrides: Any) -> SimpleNamespace:
    """Return minimal settings accepted by the runtime identity builder."""
    values: dict[str, Any] = {
        "app_name": "comfy-modal-sync",
        "buffer_containers": None,
        "enable_gpu_memory_snapshot": True,
        "enable_memory_snapshot": True,
        "execution_timeout_seconds": 3600,
        "invocation_dict_name": "comfy-modal-sync-invocations",
        "invocation_result_inline_max_bytes": 4 * 1024 * 1024,
        "interrupt_dict_name": "comfy-modal-sync-interrupts",
        "max_containers": None,
        "min_containers": 0,
        "modal_gpu": "A100",
        "modal_secret_name": "comfy",
        "node_output_cache_dict_name": "comfy-modal-sync-node-cache",
        "remote_storage_root": "/storage",
        "scaledown_window_seconds": 600,
        "session_bridge_dict_name": "comfy-modal-sync-session-bridges",
        "bridge_inline_max_bytes": 4 * 1024 * 1024,
        "snapshot_profile_dict_name": "comfy-modal-sync-snapshot-profiles",
        "startup_timeout_seconds": 900,
        "stream_event_queue_maxsize": 256,
        "sync_index_dict_name": "comfy-modal-sync-sync-index",
        "volume_name": "comfy-universal-storage",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _runtime_identity(
    runtime_environment_module: Any,
    repo_root: Path,
    comfyui_root: Path,
    custom_nodes_dir: Path | None = None,
    **settings_overrides: Any,
) -> Any:
    """Build one test identity from small temporary source trees."""
    return runtime_environment_module.build_remote_runtime_identity(
        repo_root=repo_root,
        comfyui_root=comfyui_root,
        custom_nodes_dir=custom_nodes_dir,
        settings=_runtime_settings(**settings_overrides),
    )


def test_remote_environment_is_fully_pinned(runtime_environment_module: Any) -> None:
    """The supported Python, ComfyUI, and CUDA environments should not float."""
    apt_packages = runtime_environment_module.remote_apt_packages()
    runtime_packages = runtime_environment_module.remote_runtime_packages()
    torch_packages = runtime_environment_module.remote_torch_packages()

    assert runtime_environment_module.DEFAULT_MODAL_GPU == "RTX-PRO-6000"
    assert runtime_environment_module.REMOTE_APP_PROTOCOL_VERSION == 8
    assert runtime_environment_module.REMOTE_PYTHON_VERSION == "3.13"
    assert apt_packages == ("libgl1", "libglib2.0-0")
    assert all("==" in requirement for requirement in runtime_packages)
    assert all("==" in requirement for requirement in torch_packages)
    assert len(runtime_packages) == len(set(runtime_packages))
    assert "blake3==1.0.9" in runtime_packages
    assert "comfy-angle==0.1.0" in runtime_packages
    assert "comfy-aimdo==0.4.13" in runtime_packages
    assert "comfy-kitchen==0.2.31" in runtime_packages
    assert "pyopengl==3.1.10" in runtime_packages
    assert "pypdf==6.16.1" in runtime_packages
    assert "numpy==2.3.5" in runtime_packages
    assert "safetensors==0.8.0" in runtime_packages
    assert "huggingface-hub==1.28.0" in runtime_packages
    assert "hf-xet==1.6.0" in runtime_packages
    assert "transformers==5.15.0" in runtime_packages
    assert "simpleeval==1.0.7" in runtime_packages


def test_remote_huggingface_stack_is_pinned_and_validated(
    runtime_environment_module: Any,
) -> None:
    """Model staging should install and import the explicit Xet accelerator."""
    assert runtime_environment_module.remote_huggingface_packages() == (
        "huggingface-hub==1.28.0",
        "hf-xet==1.6.0",
    )
    command = runtime_environment_module.remote_huggingface_validation_command()
    assert "import hf_xet, huggingface_hub" in command
    assert "actual_xet=metadata.version" in command
    assert "hf-xet" in command


@pytest.mark.parametrize(
    "modal_gpu",
    (
        "T4",
        "L4",
        "A10",
        "L40S",
        "A100",
        "A100-40GB",
        "A100-80GB",
        "RTX-PRO-6000",
        "H100",
        "H100!",
        "H200",
        "B200",
        "B200+",
        "B300",
        "b300:2",
    ),
)
def test_all_supported_modal_gpus_use_vllm_compatible_cuda_130_build(
    runtime_environment_module: Any,
    modal_gpu: str,
) -> None:
    """Every selectable GPU should use one vLLM-compatible CUDA 13 stack."""
    build = runtime_environment_module.select_remote_torch_build(modal_gpu)

    assert build.cuda_version == "13.0"
    assert build.install_layers == (
        runtime_environment_module.RemoteTorchInstallLayer(
            index_url="https://download.pytorch.org/whl/cu130",
            packages=("torch==2.13.0", "torchvision==0.28.0"),
        ),
        runtime_environment_module.RemoteTorchInstallLayer(
            index_url="https://download.pytorch.org/whl/cpu",
            packages=("torchaudio==2.11.0+cpu",),
            extra_options="--no-deps",
        ),
    )
    assert runtime_environment_module.remote_torch_packages(modal_gpu) == (
        "torch==2.13.0",
        "torchvision==0.28.0",
        "torchaudio==2.11.0+cpu",
    )
    assert runtime_environment_module.remote_accelerator_packages(modal_gpu) == (
        runtime_environment_module.REMOTE_VLLM_WHEEL_URL,
    )


def test_remote_build_validation_imports_complete_torch_and_vllm_stacks(
    runtime_environment_module: Any,
) -> None:
    """The image build should fail before deployment when native packages disagree."""
    build = runtime_environment_module.select_remote_torch_build("B300")

    validation_command = build.validation_command()
    validation_script = shlex.split(validation_command)[2]
    accelerator_command = runtime_environment_module.remote_accelerator_validation_command(
        "RTX-PRO-6000"
    )
    accelerator_script = shlex.split(accelerator_command)[2]

    assert validation_command.startswith("python -c ")
    assert "import torch, torchaudio, torchvision" in validation_script
    assert "expected_cuda='13.0'" in validation_script
    assert "import cv2, numpy, torch, vllm" in accelerator_script
    assert "expected_vllm='0.27.1'" in accelerator_script
    assert "expected_numpy='2.3.5'" in accelerator_script
    assert "expected_opencv='4.13.0.92'" in accelerator_script


def test_empty_modal_gpu_cannot_select_torch_build(runtime_environment_module: Any) -> None:
    """An empty GPU specification should fail before constructing a Modal image."""
    with pytest.raises(ValueError, match="cannot be empty"):
        runtime_environment_module.select_remote_torch_build("  ")


def test_runtime_identity_changes_with_source_and_runtime_options(
    runtime_environment_module: Any,
    tmp_path: Path,
) -> None:
    """Code or deployment-option changes should produce a different fingerprint."""
    repo_root = tmp_path / "repo"
    comfyui_root = tmp_path / "ComfyUI"
    repo_root.mkdir()
    comfyui_root.mkdir()
    repo_source = repo_root / "worker.py"
    comfyui_source = comfyui_root / "execution.py"
    repo_source.write_text("VALUE = 1\n", encoding="utf-8")
    comfyui_source.write_text("VALUE = 1\n", encoding="utf-8")

    baseline = _runtime_identity(runtime_environment_module, repo_root, comfyui_root)
    assert baseline == _runtime_identity(runtime_environment_module, repo_root, comfyui_root)

    repo_source.write_text("VALUE = 2\n", encoding="utf-8")
    source_changed = _runtime_identity(runtime_environment_module, repo_root, comfyui_root)
    option_changed = _runtime_identity(
        runtime_environment_module,
        repo_root,
        comfyui_root,
        modal_gpu="L40S",
    )
    secret_changed = _runtime_identity(
        runtime_environment_module,
        repo_root,
        comfyui_root,
        modal_secret_name="workflow-credentials",
    )

    assert source_changed.fingerprint != baseline.fingerprint
    assert option_changed.fingerprint != source_changed.fingerprint
    assert secret_changed.fingerprint != source_changed.fingerprint
    assert secret_changed.manifest["runtime_options"]["modal_secret_name"] == (
        "workflow-credentials"
    )


def test_runtime_identity_tracks_curated_llm_profile_registry(
    runtime_environment_module: Any,
    tmp_path: Path,
) -> None:
    """Changing a pinned model profile should force a remote deployment rebuild."""
    repo_root = tmp_path / "repo"
    comfyui_root = tmp_path / "ComfyUI"
    repo_root.mkdir()
    comfyui_root.mkdir()
    (repo_root / "worker.py").write_text("VALUE = 1\n", encoding="utf-8")
    profile_path = repo_root / "llm_profiles.json"
    profile_path.write_text('{"profiles":[{"id":"one"}]}\n', encoding="utf-8")
    baseline = _runtime_identity(runtime_environment_module, repo_root, comfyui_root)

    profile_path.write_text('{"profiles":[{"id":"two"}]}\n', encoding="utf-8")
    changed = _runtime_identity(runtime_environment_module, repo_root, comfyui_root)

    assert changed.fingerprint != baseline.fingerprint


def test_runtime_identity_records_system_packages(
    runtime_environment_module: Any,
    tmp_path: Path,
) -> None:
    """The deployment manifest should record system libraries installed in the image."""
    repo_root = tmp_path / "repo"
    comfyui_root = tmp_path / "ComfyUI"
    repo_root.mkdir()
    comfyui_root.mkdir()

    identity = _runtime_identity(runtime_environment_module, repo_root, comfyui_root)

    assert identity.manifest["apt_packages"] == ["libgl1", "libglib2.0-0"]


def test_runtime_identity_records_universal_torch_and_vllm_build(
    runtime_environment_module: Any,
    tmp_path: Path,
) -> None:
    """The deployment manifest should describe the universal native stack."""
    repo_root = tmp_path / "repo"
    comfyui_root = tmp_path / "ComfyUI"
    repo_root.mkdir()
    comfyui_root.mkdir()

    identity = _runtime_identity(
        runtime_environment_module,
        repo_root,
        comfyui_root,
        modal_gpu="B300",
    )

    assert identity.manifest["torch_build"] == {
        "cuda_version": "13.0",
        "install_layers": [
            {
                "index_url": "https://download.pytorch.org/whl/cu130",
                "packages": ["torch==2.13.0", "torchvision==0.28.0"],
                "extra_options": "",
            },
            {
                "index_url": "https://download.pytorch.org/whl/cpu",
                "packages": ["torchaudio==2.11.0+cpu"],
                "extra_options": "--no-deps",
            },
        ],
    }
    assert identity.manifest["torch_packages"] == [
        "torch==2.13.0",
        "torchvision==0.28.0",
        "torchaudio==2.11.0+cpu",
    ]
    assert identity.manifest["accelerator_packages"] == [
        runtime_environment_module.REMOTE_VLLM_WHEEL_URL
    ]


def test_runtime_identity_tracks_custom_node_requirements_but_ignores_payload_source(
    runtime_environment_module: Any,
    tmp_path: Path,
) -> None:
    """Image dependencies should affect identity while request-bundled source should not."""
    repo_root = tmp_path / "repo"
    comfyui_root = tmp_path / "ComfyUI"
    custom_nodes_dir = comfyui_root / "custom_nodes"
    custom_node_dir = custom_nodes_dir / "example"
    repo_root.mkdir()
    comfyui_root.mkdir()
    custom_node_dir.mkdir(parents=True)
    (repo_root / "worker.py").write_text("VALUE = 1\n", encoding="utf-8")
    (comfyui_root / "execution.py").write_text("VALUE = 1\n", encoding="utf-8")
    requirements_path = custom_node_dir / "requirements.txt"
    source_path = custom_node_dir / "node.py"
    requirements_path.write_text("diffusers==0.37.0\n", encoding="utf-8")
    source_path.write_text("VALUE = 1\n", encoding="utf-8")

    baseline = _runtime_identity(
        runtime_environment_module,
        repo_root,
        comfyui_root,
        custom_nodes_dir,
    )
    source_path.write_text("VALUE = 2\n", encoding="utf-8")
    source_changed = _runtime_identity(
        runtime_environment_module,
        repo_root,
        comfyui_root,
        custom_nodes_dir,
    )
    requirements_path.write_text("diffusers==0.38.0\n", encoding="utf-8")
    dependency_changed = _runtime_identity(
        runtime_environment_module,
        repo_root,
        comfyui_root,
        custom_nodes_dir,
    )

    assert source_changed.fingerprint == baseline.fingerprint
    assert dependency_changed.fingerprint != baseline.fingerprint


def test_runtime_identity_ignores_comfyui_directories_outside_image_context(
    runtime_environment_module: Any,
    tmp_path: Path,
) -> None:
    """Unshipped ComfyUI scratch source should not trigger a full app rebuild."""
    repo_root = tmp_path / "repo"
    comfyui_root = tmp_path / "ComfyUI"
    runtime_package = comfyui_root / "comfy"
    scratch_package = comfyui_root / "False"
    repo_root.mkdir()
    runtime_package.mkdir(parents=True)
    scratch_package.mkdir(parents=True)
    (repo_root / "worker.py").write_text("VALUE = 1\n", encoding="utf-8")
    runtime_source = runtime_package / "model_management.py"
    scratch_source = scratch_package / "developer_copy.py"
    runtime_source.write_text("VALUE = 1\n", encoding="utf-8")
    scratch_source.write_text("VALUE = 1\n", encoding="utf-8")

    baseline = _runtime_identity(runtime_environment_module, repo_root, comfyui_root)
    scratch_source.write_text("VALUE = 2\n", encoding="utf-8")
    scratch_changed = _runtime_identity(runtime_environment_module, repo_root, comfyui_root)
    runtime_source.write_text("VALUE = 2\n", encoding="utf-8")
    runtime_changed = _runtime_identity(runtime_environment_module, repo_root, comfyui_root)

    assert scratch_changed.fingerprint == baseline.fingerprint
    assert runtime_changed.fingerprint != scratch_changed.fingerprint
