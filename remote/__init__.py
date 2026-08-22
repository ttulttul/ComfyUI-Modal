"""Remote execution helpers for Modal-Sync."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_MODAL_APP_EXPORTS = frozenset(
    {"execute_node_locally", "execute_subgraph_locally", "invoke_remote_engine"}
)

__all__ = sorted(_MODAL_APP_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load Modal-specific helpers only when a caller requests one."""
    if name not in _MODAL_APP_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    modal_app = import_module(".modal_app", __name__)
    return getattr(modal_app, name)
