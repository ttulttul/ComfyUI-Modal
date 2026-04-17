"""ComfyUI Modal-Sync extension entrypoint."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

WEB_DIRECTORY = "./web"
try:
    from comfy_api.latest import ComfyExtension, io

    from .api_intercept import setup_modal_queue_route
    from .modal_executor_node import ModalUniversalExecutor
except ModuleNotFoundError:  # pragma: no cover - used during local non-Comfy imports.
    ComfyExtension = object  # type: ignore[assignment]
    io = None  # type: ignore[assignment]
    ModalUniversalExecutor = None  # type: ignore[assignment]

    class ComfyModalSyncExtension:  # type: ignore[no-redef]
        """Fallback placeholder used when ComfyUI is not importable."""

    async def comfy_entrypoint() -> object:  # type: ignore[no-redef]
        """Raise a helpful error when imported outside ComfyUI."""
        raise RuntimeError(
            "ComfyUI Modal-Sync requires the ComfyUI runtime on sys.path to load."
        )

else:

    class ComfyModalSyncExtension(ComfyExtension):
        """Expose Modal-Sync nodes to the ComfyUI extension loader."""

        async def get_node_list(self) -> list[type[io.ComfyNode]]:
            """Return the node list after registering API routes."""
            setup_modal_queue_route()
            return [ModalUniversalExecutor]

    async def comfy_entrypoint() -> ComfyExtension:
        """Create the ComfyUI v3 extension entrypoint."""
        setup_modal_queue_route()
        return ComfyModalSyncExtension()


__all__ = [
    "ComfyModalSyncExtension",
    "ModalUniversalExecutor",
    "WEB_DIRECTORY",
    "comfy_entrypoint",
]
