"""Static integration checks for the Subrosa Configuration Login UI."""

from __future__ import annotations

from pathlib import Path


def test_subrosa_login_is_direct_nonce_bound_and_non_serialized() -> None:
    """The node button should use the portal without queueing a workflow."""
    source = (Path(__file__).parents[1] / "web" / "subrosa_login.js").read_text(
        encoding="utf-8"
    )

    assert '"SubrosaRemoteConfiguration"' in source
    assert '"Login to Subrosa"' in source
    assert 'serialize: false' in source
    assert 'window.open(' in source
    assert '"about:blank"' in source
    assert '"comfyui_state"' in source
    assert "globalThis.crypto.randomUUID()" in source
    assert "event.origin !== pending.origin" in source
    assert "event.source !== pending.popup" in source
    assert 'hostname.endsWith(".subrosa.red")' in source
    assert "/remote/subrosa/credentials" in source
    assert "/remote/subrosa/status" in source
    assert "token," in source
    assert 'token = ""' in source
    assert "api.queuePrompt" not in source
