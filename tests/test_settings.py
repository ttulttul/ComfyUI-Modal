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
