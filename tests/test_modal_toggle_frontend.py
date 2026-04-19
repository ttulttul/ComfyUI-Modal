"""Run the frontend Modal toggle regression checks through pytest."""

from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_modal_toggle_frontend_regressions() -> None:
    """Verify mapped-component progress and phase expansion in the browser-side overlay logic."""
    subprocess.run(
        ["node", "tests/modal_toggle_frontend.test.mjs"],
        cwd=REPO_ROOT,
        check=True,
    )
