"""Static safety contracts for the Vast lease manager UI."""

from __future__ import annotations

from pathlib import Path


def _source() -> str:
    """Return the Vast manager browser extension source."""
    return (Path(__file__).resolve().parents[1] / "web" / "vast_manager.js").read_text(
        encoding="utf-8"
    )


def test_vast_manager_exposes_status_verification_and_owned_cleanup() -> None:
    """Operators need visibility and precise lifecycle controls."""
    source = _source()

    assert 'const VAST_STATUS_ROUTE = "/remote/vast/status";' in source
    assert 'const VAST_VERIFY_ROUTE = "/remote/vast/verify";' in source
    assert 'const VAST_REAP_ROUTE = "/remote/vast/reap";' in source
    assert 'const VAST_DESTROY_ROUTE = "/remote/vast/destroy";' in source
    assert 'JSON.stringify({ instance_id: lease.instance_id })' in source


def test_vast_manager_never_collects_or_persists_secrets() -> None:
    """API and SSH credentials remain process-level configuration."""
    source = _source()

    assert 'name="api_key"' not in source
    assert 'name="password"' not in source
    assert "VAST_API_KEY" in source
    assert "COMFY_MODAL_VAST_IMAGE" in source
