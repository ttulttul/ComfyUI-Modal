"""Static contract tests for the SSH environment manager extension."""

from __future__ import annotations

from pathlib import Path


def _source() -> str:
    """Return the browser extension source."""
    return (Path(__file__).resolve().parents[1] / "web" / "remote_environments.js").read_text(
        encoding="utf-8"
    )


def test_manager_uses_provider_neutral_routes_and_never_collects_credentials() -> None:
    """SSH credentials should remain exclusively in the user's SSH configuration."""
    source = _source()

    assert 'const ENVIRONMENTS_ROUTE = "/remote/environments";' in source
    assert 'const PROBE_ROUTE = "/remote/environments/probe";' in source
    assert 'const BOOTSTRAP_ROUTE = "/remote/environments/bootstrap";' in source
    assert 'const STATUS_ROUTE = "/remote/environments/status";' in source
    assert 'const STOP_ROUTE = "/remote/environments/stop";' in source
    assert 'field("SSH target or alias", "ssh_target"' in source
    assert 'name="password"' not in source
    assert 'name="private_key"' not in source


def test_manager_exposes_cost_capacity_probe_and_bootstrap_controls() -> None:
    """The UI should cover all scheduler and lifecycle inputs needed for a host."""
    source = _source()

    assert 'field("Cost USD/hour", "cost_per_hour"' in source
    assert 'field("Max workers", "maximum_workers"' in source
    assert 'field("Reserve VRAM GB", "reserve_vram_gb"' in source
    assert 'field("Remote Docker env file", "docker_env_file"' in source
    assert 'actionButton("Save and probe"' in source
    assert 'actionButton("Build / update worker"' in source
    assert 'actionButton("Refresh workers"' in source
    assert 'actionButton("Stop workers"' in source
    assert '"Drain (no new work)"' in source


def test_manager_settings_entry_uses_comfy_custom_renderer() -> None:
    """The settings action must use ComfyUI's supported function-valued control type."""
    source = _source()

    assert 'name: "Remote Execution: SSH environments"' in source
    assert "type: () => {" in source
    assert 'button.textContent = "Manage SSH hosts";' in source
    assert 'type: "button"' not in source
