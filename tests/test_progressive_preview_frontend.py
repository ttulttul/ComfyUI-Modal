"""Run the progressive preview frontend checks through pytest."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_progressive_preview_frontend() -> None:
    """Verify slot layout, filling, and native-preview handoff in the browser extension."""
    subprocess.run(
        ["node", "tests/progressive_preview_frontend.test.mjs"],
        cwd=REPO_ROOT,
        check=True,
    )
