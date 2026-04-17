"""ComfyUI Modal-Sync extension entrypoint."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)
_EXTENSION_LOGGER_NAME = __name__.split(".")[0]
_EXTENSION_HANDLER_NAME = "comfyui-modal-sync-timestamped"


def _build_extension_log_formatter() -> logging.Formatter:
    """Return the default formatter for Modal-Sync logs with wall-clock and relative timestamps."""
    return logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d +%(relativeCreated)07.0fms %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _configure_extension_logging() -> logging.Logger:
    """Install a dedicated timestamped handler for the Modal-Sync logger hierarchy."""
    extension_logger = logging.getLogger(_EXTENSION_LOGGER_NAME)
    for existing_handler in extension_logger.handlers:
        if getattr(existing_handler, "name", "") == _EXTENSION_HANDLER_NAME:
            return extension_logger

    handler = logging.StreamHandler()
    handler.set_name(_EXTENSION_HANDLER_NAME)
    handler.setFormatter(_build_extension_log_formatter())
    extension_logger.addHandler(handler)
    extension_logger.propagate = False
    return extension_logger


_configure_extension_logging()

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
