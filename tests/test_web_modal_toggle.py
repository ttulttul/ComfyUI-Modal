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
    assert "function setPromptActiveNode(promptId, activeNodeId)" in source
    assert "detail.active_node_id" in source
    assert "isActiveRemoteNode" in source
