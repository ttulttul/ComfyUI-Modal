"""Tests for Modal-Sync settings discovery."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def test_settings_discovers_comfyui_root_from_custom_nodes_install_path(
    settings_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """The extension should infer the ComfyUI root from its install location."""
    comfyui_root = tmp_path / "ComfyUI"
    custom_node_repo = comfyui_root / "custom_nodes" / "ComfyUI-Modal"
    custom_node_repo.mkdir(parents=True)
    (comfyui_root / "main.py").write_text("print('main')\n", encoding="utf-8")
    (comfyui_root / "nodes.py").write_text("NODE_CLASS_MAPPINGS = {}\n", encoding="utf-8")

    monkeypatch.setattr(settings_module, "__file__", str(custom_node_repo / "settings.py"))
    monkeypatch.delenv("COMFYUI_ROOT", raising=False)
    monkeypatch.delenv("COMFY_MODAL_COMFYUI_ROOT", raising=False)

    resolved = settings_module._discover_comfyui_root(custom_node_repo)

    assert resolved == comfyui_root.resolve()


def test_settings_prefers_modal_specific_comfyui_root_env(
    settings_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """The Modal-specific ComfyUI root env var should override path inference."""
    env_root = tmp_path / "alt-comfyui"
    env_root.mkdir()

    monkeypatch.setenv("COMFY_MODAL_COMFYUI_ROOT", str(env_root))
    monkeypatch.delenv("COMFYUI_ROOT", raising=False)

    resolved = settings_module._discover_comfyui_root(tmp_path / "repo")

    assert resolved == env_root.resolve()


def test_settings_reads_modal_gpu_override(
    settings_module: Any,
    monkeypatch: Any,
) -> None:
    """The Modal GPU type should be configurable via environment variable."""
    monkeypatch.setenv("COMFY_MODAL_GPU", "L40S")
    settings_module.get_settings.cache_clear()
    try:
        settings = settings_module.get_settings()
    finally:
        settings_module.get_settings.cache_clear()

    assert settings.modal_gpu == "L40S"


def test_settings_reads_modal_container_scaling_overrides(
    settings_module: Any,
    monkeypatch: Any,
) -> None:
    """Modal autoscaling knobs should be configurable via environment variables."""
    monkeypatch.setenv("COMFY_MODAL_MIN_CONTAINERS", "1")
    monkeypatch.setenv("COMFY_MODAL_MAX_CONTAINERS", "6")
    monkeypatch.setenv("COMFY_MODAL_BUFFER_CONTAINERS", "2")
    settings_module.get_settings.cache_clear()
    try:
        settings = settings_module.get_settings()
    finally:
        settings_module.get_settings.cache_clear()

    assert settings.min_containers == 1
    assert settings.max_containers == 6
    assert settings.buffer_containers == 2


def test_settings_defaults_interrupt_dict_name_from_app_name(
    settings_module: Any,
    monkeypatch: Any,
) -> None:
    """The shared interrupt dict name should default to one derived from the app name."""
    monkeypatch.setenv("COMFY_MODAL_APP_NAME", "my-modal-app")
    settings_module.get_settings.cache_clear()
    try:
        settings = settings_module.get_settings()
    finally:
        settings_module.get_settings.cache_clear()

    assert settings.interrupt_dict_name == "my-modal-app-interrupts"


def test_settings_reads_interrupt_dict_name_override(
    settings_module: Any,
    monkeypatch: Any,
) -> None:
    """The shared interrupt dict name should be overridable explicitly."""
    monkeypatch.setenv("COMFY_MODAL_INTERRUPT_DICT_NAME", "custom-interrupt-store")
    settings_module.get_settings.cache_clear()
    try:
        settings = settings_module.get_settings()
    finally:
        settings_module.get_settings.cache_clear()

    assert settings.interrupt_dict_name == "custom-interrupt-store"


def test_settings_reads_terminate_container_on_error_override(
    settings_module: Any,
    monkeypatch: Any,
) -> None:
    """Remote crash teardown should be configurable via environment variable."""
    monkeypatch.setenv("COMFY_MODAL_TERMINATE_CONTAINER_ON_ERROR", "false")
    settings_module.get_settings.cache_clear()
    try:
        settings = settings_module.get_settings()
    finally:
        settings_module.get_settings.cache_clear()

    assert settings.terminate_container_on_error is False
