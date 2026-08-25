"""Static integration checks for the R2 node Login UI."""

from __future__ import annotations

from pathlib import Path


def test_r2_node_login_is_direct_and_non_serialized() -> None:
    """The R2 node should start OAuth from a button without queue execution."""
    source = (Path(__file__).parents[1] / "web" / "r2_storage.js").read_text(
        encoding="utf-8"
    )

    assert '"R2StorageBackingConfiguration"' in source
    assert '"button"' in source
    assert "startCloudflareLogin" in source
    assert 'serialize: false' in source
    assert "window.open(" in source
    assert '"about:blank"' in source
    assert "/remote/storage/r2/oauth/start" in source
    assert "api.queuePrompt" not in source
