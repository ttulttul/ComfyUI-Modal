"""Tests for Modal-Sync settings discovery."""

from __future__ import annotations

from dataclasses import replace
import re
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


def test_settings_defaults_modal_gpu_to_rtx_pro_6000(
    settings_module: Any,
) -> None:
    """Backend settings should use the same RTX default as the workflow UI."""
    settings_module.get_settings.cache_clear()
    try:
        settings = settings_module.get_settings()
    finally:
        settings_module.get_settings.cache_clear()

    assert settings_module.DEFAULT_MODAL_GPU == "RTX-PRO-6000"
    assert settings.modal_gpu == "RTX-PRO-6000"


def test_modal_gpu_from_workflow_prefers_saved_selection(settings_module: Any) -> None:
    """A saved workflow GPU should override the process-level fallback."""
    workflow = {"extra": {"comfy_modal": {"gpu": "B300"}}}

    selected_gpu = settings_module.modal_gpu_from_workflow(workflow, "L40S")

    assert selected_gpu == "B300"


def test_modal_gpu_from_workflow_uses_fallback_when_unsaved(settings_module: Any) -> None:
    """Older workflows without Modal metadata should retain the configured fallback."""
    assert settings_module.modal_gpu_from_workflow({"nodes": []}, "L40S") == "L40S"


def test_modal_gpu_from_workflow_rejects_unknown_selection(settings_module: Any) -> None:
    """Queue handling should reject edited workflow GPU values outside Modal's supported list."""
    workflow = {"extra": {"comfy_modal": {"gpu": "V100"}}}

    try:
        settings_module.modal_gpu_from_workflow(workflow, "A100")
    except ValueError as exc:
        assert "V100" in str(exc)
    else:
        raise AssertionError("Expected an unsupported workflow GPU to be rejected.")


def test_settings_for_modal_gpu_preserves_other_runtime_settings(settings_module: Any) -> None:
    """A workflow GPU override should change only the deploy-time GPU target."""
    base_settings = settings_module.get_settings()

    overridden_settings = settings_module.settings_for_modal_gpu(base_settings, "h200")

    assert overridden_settings.modal_gpu == "H200"
    assert overridden_settings.app_name == base_settings.app_name
    assert overridden_settings.volume_name == base_settings.volume_name


def test_modal_deployment_app_name_isolates_gpu_targets_from_legacy_a100_app(
    settings_module: Any,
) -> None:
    """The RTX default should not probe the legacy A100 app retained at the base name."""
    base_settings = settings_module.get_settings()

    rtx_settings = settings_module.settings_for_modal_gpu(base_settings, "RTX-PRO-6000")
    a100_settings = settings_module.settings_for_modal_gpu(base_settings, "A100")
    b300_settings = settings_module.settings_for_modal_gpu(base_settings, "B300")
    h100_settings = settings_module.settings_for_modal_gpu(base_settings, "H100")
    priority_h100_settings = settings_module.settings_for_modal_gpu(base_settings, "H100!")

    assert settings_module.modal_deployment_app_name(a100_settings) == base_settings.app_name
    assert settings_module.modal_deployment_app_name(rtx_settings) == (
        f"{base_settings.app_name}-gpu-rtx-pro-6000"
    )
    assert settings_module.modal_deployment_app_name(b300_settings) == (
        f"{base_settings.app_name}-gpu-b300"
    )
    assert settings_module.modal_deployment_app_name(h100_settings).endswith("-gpu-h100")
    assert settings_module.modal_deployment_app_name(priority_h100_settings).endswith(
        "-gpu-h100-priority"
    )


def test_modal_deployment_app_name_stays_within_modal_object_name_limit(
    settings_module: Any,
) -> None:
    """GPU-specific app identities should remain valid for a 64-character base app name."""
    base_settings = settings_module.get_settings()
    long_name_settings = replace(
        base_settings,
        app_name="a" * 64,
        modal_gpu="RTX-PRO-6000",
    )

    deployment_app_name = settings_module.modal_deployment_app_name(long_name_settings)

    assert len(deployment_app_name) <= 64
    assert deployment_app_name.endswith("-gpu-rtx-pro-6000")
    assert re.fullmatch(r"[a-zA-Z0-9-_.]+", deployment_app_name)


def test_settings_generates_stable_per_comfyui_app_name(
    settings_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """Default remote app names should be stable and namespaced by a persisted 64-bit identity."""
    identity_path = tmp_path / "instance-id"
    monkeypatch.delenv("COMFY_MODAL_APP_NAME", raising=False)
    monkeypatch.setenv("COMFY_MODAL_INSTANCE_ID_PATH", str(identity_path))
    settings_module.get_settings.cache_clear()
    try:
        first_settings = settings_module.get_settings()
        settings_module.get_settings.cache_clear()
        second_settings = settings_module.get_settings()
    finally:
        settings_module.get_settings.cache_clear()

    assert first_settings.app_name == second_settings.app_name
    assert re.fullmatch(r"comfy-modal-sync-[A-Za-z0-9_-]{11}", first_settings.app_name)
    assert len(bytes.fromhex(identity_path.read_text(encoding="ascii").strip())) == 8
    assert first_settings.interrupt_dict_name == f"{first_settings.app_name}-interrupts"
    assert first_settings.invocation_dict_name == f"{first_settings.app_name}-invocations"


def test_settings_app_name_override_bypasses_instance_identity(
    settings_module: Any,
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    """An explicit app name should remain authoritative and require no identity file."""
    identity_path = tmp_path / "instance-id"
    monkeypatch.setenv("COMFY_MODAL_APP_NAME", "shared-explicit-app")
    monkeypatch.setenv("COMFY_MODAL_INSTANCE_ID_PATH", str(identity_path))
    settings_module.get_settings.cache_clear()
    try:
        settings = settings_module.get_settings()
    finally:
        settings_module.get_settings.cache_clear()

    assert settings.app_name == "shared-explicit-app"
    assert not identity_path.exists()


def test_settings_cache_tracks_execution_mode_env_changes(
    settings_module: Any,
    monkeypatch: Any,
) -> None:
    """Changing Modal execution mode should not leave callers with stale cached settings."""
    settings_module.get_settings.cache_clear()
    try:
        monkeypatch.setenv("COMFY_MODAL_EXECUTION_MODE", "local")
        local_settings = settings_module.get_settings()

        monkeypatch.setenv("COMFY_MODAL_EXECUTION_MODE", "remote")
        remote_settings = settings_module.get_settings()
    finally:
        settings_module.get_settings.cache_clear()

    assert local_settings.execution_mode == "local"
    assert local_settings.sync_custom_nodes is False
    assert remote_settings.execution_mode == "remote"
    assert remote_settings.sync_custom_nodes is True


def test_settings_enable_gpu_memory_snapshot_defaults_true(
    settings_module: Any,
    monkeypatch: Any,
) -> None:
    """GPU memory snapshots should be enabled by default."""
    monkeypatch.delenv("COMFY_MODAL_ENABLE_GPU_MEMORY_SNAPSHOT", raising=False)
    settings_module.get_settings.cache_clear()
    try:
        settings = settings_module.get_settings()
    finally:
        settings_module.get_settings.cache_clear()

    assert settings.enable_gpu_memory_snapshot is True


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


def test_settings_reads_remote_invocation_runtime_controls(
    settings_module: Any,
    monkeypatch: Any,
) -> None:
    """Remote call concurrency and Modal deadlines should be configurable."""
    monkeypatch.setenv("COMFY_MODAL_MAX_INFLIGHT_CALLS", "7")
    monkeypatch.setenv("COMFY_MODAL_EXECUTION_TIMEOUT_SECONDS", "1800")
    monkeypatch.setenv("COMFY_MODAL_STARTUP_TIMEOUT_SECONDS", "600")
    monkeypatch.setenv("COMFY_MODAL_STREAM_EVENT_QUEUE_MAXSIZE", "32")
    settings_module.get_settings.cache_clear()
    try:
        settings = settings_module.get_settings()
    finally:
        settings_module.get_settings.cache_clear()

    assert settings.max_inflight_calls == 7
    assert settings.execution_timeout_seconds == 1800
    assert settings.startup_timeout_seconds == 600
    assert settings.stream_event_queue_maxsize == 32


def test_settings_reads_proactive_warmup_override(
    settings_module: Any,
    monkeypatch: Any,
) -> None:
    """Proactive remote warmup should be configurable via environment variable."""
    monkeypatch.setenv("COMFY_MODAL_ENABLE_PROACTIVE_WARMUP", "false")
    settings_module.get_settings.cache_clear()
    try:
        settings = settings_module.get_settings()
    finally:
        settings_module.get_settings.cache_clear()

    assert settings.enable_proactive_warmup is False


def test_settings_reads_remote_cancel_grace_override(
    settings_module: Any,
    monkeypatch: Any,
) -> None:
    """The post-interrupt remote wait should be configurable."""
    monkeypatch.setenv("COMFY_MODAL_REMOTE_CANCEL_GRACE_SECONDS", "0.25")
    settings_module.get_settings.cache_clear()
    try:
        settings = settings_module.get_settings()
    finally:
        settings_module.get_settings.cache_clear()

    assert settings.remote_cancel_grace_seconds == 0.25


def test_settings_reads_remote_cancel_restart_override(
    settings_module: Any,
    monkeypatch: Any,
) -> None:
    """The remote post-cancel restart timeout should be configurable."""
    monkeypatch.setenv("COMFY_MODAL_REMOTE_CANCEL_RESTART_SECONDS", "0.75")
    settings_module.get_settings.cache_clear()
    try:
        settings = settings_module.get_settings()
    finally:
        settings_module.get_settings.cache_clear()

    assert settings.remote_cancel_restart_seconds == 0.75


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


def test_settings_defaults_node_cache_dict_name_from_app_name(
    settings_module: Any,
    monkeypatch: Any,
) -> None:
    """The shared node-cache dict should default to one derived from the app name."""
    monkeypatch.setenv("COMFY_MODAL_APP_NAME", "my-modal-app")
    settings_module.get_settings.cache_clear()
    try:
        settings = settings_module.get_settings()
    finally:
        settings_module.get_settings.cache_clear()

    assert settings.node_output_cache_dict_name == "my-modal-app-node-cache"


def test_settings_defaults_session_bridge_dict_name_from_app_name(
    settings_module: Any,
    monkeypatch: Any,
) -> None:
    """The durable session-bridge dict should default to one derived from the app name."""
    monkeypatch.setenv("COMFY_MODAL_APP_NAME", "my-modal-app")
    settings_module.get_settings.cache_clear()
    try:
        settings = settings_module.get_settings()
    finally:
        settings_module.get_settings.cache_clear()

    assert settings.session_bridge_dict_name == "my-modal-app-session-bridges"


def test_settings_defaults_invocation_dict_name_from_app_name(
    settings_module: Any,
    monkeypatch: Any,
) -> None:
    """The durable invocation dict should default to one derived from the app name."""
    monkeypatch.setenv("COMFY_MODAL_APP_NAME", "my-modal-app")
    settings_module.get_settings.cache_clear()
    try:
        settings = settings_module.get_settings()
    finally:
        settings_module.get_settings.cache_clear()

    assert settings.invocation_dict_name == "my-modal-app-invocations"


def test_settings_defaults_node_cache_max_bytes_to_five_mib(
    settings_module: Any,
    monkeypatch: Any,
) -> None:
    """The distributed node-cache size cap should default to 5 MiB."""
    monkeypatch.delenv("COMFY_MODAL_NODE_CACHE_MAX_BYTES", raising=False)
    settings_module.get_settings.cache_clear()
    try:
        settings = settings_module.get_settings()
    finally:
        settings_module.get_settings.cache_clear()

    assert settings.node_output_cache_max_bytes == 5 * 1024 * 1024


def test_settings_reads_node_cache_overrides(
    settings_module: Any,
    monkeypatch: Any,
) -> None:
    """The shared node-cache dict name and size limit should be configurable."""
    monkeypatch.setenv("COMFY_MODAL_NODE_CACHE_DICT_NAME", "custom-node-cache")
    monkeypatch.setenv("COMFY_MODAL_NODE_CACHE_MAX_BYTES", "123456")
    settings_module.get_settings.cache_clear()
    try:
        settings = settings_module.get_settings()
    finally:
        settings_module.get_settings.cache_clear()

    assert settings.node_output_cache_dict_name == "custom-node-cache"
    assert settings.node_output_cache_max_bytes == 123456


def test_settings_reads_session_bridge_dict_name_override(
    settings_module: Any,
    monkeypatch: Any,
) -> None:
    """The durable session-bridge dict name should be configurable."""
    monkeypatch.setenv("COMFY_MODAL_SESSION_BRIDGE_DICT_NAME", "custom-session-bridges")
    settings_module.get_settings.cache_clear()
    try:
        settings = settings_module.get_settings()
    finally:
        settings_module.get_settings.cache_clear()

    assert settings.session_bridge_dict_name == "custom-session-bridges"


def test_settings_reads_durable_invocation_and_bridge_limits(
    settings_module: Any,
    monkeypatch: Any,
) -> None:
    """Durable state metadata stores and inline byte limits should be configurable."""
    monkeypatch.setenv("COMFY_MODAL_INVOCATION_DICT_NAME", "custom-invocations")
    monkeypatch.setenv("COMFY_MODAL_BRIDGE_INLINE_MAX_BYTES", "1234")
    monkeypatch.setenv("COMFY_MODAL_INVOCATION_RESULT_INLINE_MAX_BYTES", "5678")
    settings_module.get_settings.cache_clear()
    try:
        settings = settings_module.get_settings()
    finally:
        settings_module.get_settings.cache_clear()

    assert settings.invocation_dict_name == "custom-invocations"
    assert settings.bridge_inline_max_bytes == 1234
    assert settings.invocation_result_inline_max_bytes == 5678


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
