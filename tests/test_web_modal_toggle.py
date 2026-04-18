"""Regression tests for the frontend Modal queue shim."""

from __future__ import annotations

from pathlib import Path


def _modal_toggle_source() -> str:
    """Return the current frontend extension source."""
    return (Path(__file__).resolve().parents[1] / "web" / "modal_toggle.js").read_text(encoding="utf-8")


def test_synthetic_status_event_matches_comfyui_status_shape() -> None:
    """Synthetic status events should use the same detail shape as websocket status events."""
    source = _modal_toggle_source()

    assert 'dispatchSyntheticApiEvent("status", statusPayload(1));' in source
    assert 'dispatchSyntheticApiEvent("status", statusPayload(0));' in source


def test_synthetic_execution_events_match_comfyui_execution_shapes() -> None:
    """Synthetic execution events should mirror ComfyUI's websocket adapter payloads."""
    source = _modal_toggle_source()

    assert 'dispatchSyntheticApiEvent("execution_start", {' in source
    assert "timestamp: nowMs()," in source
    assert 'dispatchSyntheticApiEvent("executing", displayNode);' in source
    assert 'dispatchSyntheticApiEvent("notification", {' in source
    assert "Waiting for a machine on Modal." in source


def test_global_modal_status_badge_is_installed() -> None:
    """The frontend should expose a dedicated global Modal activity indicator."""
    source = _modal_toggle_source()

    assert 'element.id = "comfy-modal-global-status";' in source
    assert "Modal setup running for" in source
    assert "Modal workflow running on" in source
    assert "installGlobalStatusStyles()" in source


def test_remote_modal_status_tracks_active_node_ids() -> None:
    """The frontend should track and highlight the currently active remote node."""
    source = _modal_toggle_source()

    assert "activeNodeId: null" in source
    assert "hasStreamedProgress: false" in source
    assert "function setPromptActiveNode(promptId, activeNodeId)" in source
    assert "detail.active_node_id" in source
    assert "clearPromptRemoteStates(promptId)" in source


def test_remote_modal_uses_distinct_ready_active_and_complete_colors() -> None:
    """The frontend should distinguish ready, active, and completed remote nodes visually."""
    source = _modal_toggle_source()

    assert 'const READY_BORDER_COLOR = "#22c55e";' in source
    assert 'const ACTIVE_BORDER_COLOR = "#a855f7";' in source
    assert 'const COMPLETE_BORDER_COLOR = "#16a34a";' in source
    assert 'const STATE_READY = "ready";' in source
    assert 'const STATE_ACTIVE = "active";' in source
    assert 'detail.phase === "execution_success"' in source


def test_streamed_modal_progress_takes_precedence_over_proxy_events() -> None:
    """Once streamed node progress starts, coarse proxy execution events should stop overriding it."""
    source = _modal_toggle_source()

    assert "promptState.hasStreamedProgress = true;" in source
    assert "if (promptState.hasStreamedProgress && phase !== STATE_ERROR)" in source
