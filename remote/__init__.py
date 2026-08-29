"""Remote execution helpers for Modal-Sync."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_REMOTE_EXPORT_MODULES = {
    "execute_node_locally": ".local_execution",
    "execute_subgraph_locally": ".local_execution",
    "invoke_remote_engine": ".modal_app",
}

__all__ = sorted(_REMOTE_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    """Load Modal-specific helpers only when a caller requests one."""
    module_name = _REMOTE_EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    owner_module = import_module(module_name, __name__)
    return getattr(owner_module, name)
