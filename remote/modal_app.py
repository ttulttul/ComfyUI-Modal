"""Remote Modal runtime and local execution fallback."""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from ..serialization import (
    deserialize_node_inputs,
    serialize_node_outputs,
)
from ..settings import get_settings

logger = logging.getLogger(__name__)

try:
    import modal  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised by local fallback tests.
    modal = None


def _extract_custom_nodes_bundle(bundle_path: str | None) -> None:
    """Extract a mirrored custom_nodes archive into a temporary import path."""
    if not bundle_path:
        return

    settings = get_settings()
    local_bundle = settings.local_storage_root / bundle_path.lstrip("/")
    if not local_bundle.exists():
        logger.warning("Custom nodes bundle %s was not found in local storage.", local_bundle)
        return

    extraction_root = Path(tempfile.gettempdir()) / "comfy-modal-sync-custom-nodes"
    extraction_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(local_bundle, "r") as archive:
        archive.extractall(extraction_root)

    if str(extraction_root) not in sys.path:
        sys.path.insert(0, str(extraction_root))
    logger.info("Extracted remote custom_nodes bundle to %s", extraction_root)


def _load_nodes_module() -> Any:
    """Import the ComfyUI nodes module lazily."""
    import nodes

    return nodes


def _invoke_original_node(
    node_class: type[Any],
    node_data: dict[str, Any],
    kwargs: dict[str, Any],
) -> tuple[Any, ...]:
    """Execute an original V1 or V3 node class and normalize its outputs."""
    class_type = node_data["class_type"]
    logger.info("Executing remote node %s", class_type)

    if hasattr(node_class, "GET_SCHEMA"):
        node_output = node_class.execute(**kwargs)
        if hasattr(node_output, "result"):
            result = node_output.result
            return tuple(result) if result is not None else tuple()
        return tuple(node_output)

    instance = node_class()
    function_name = getattr(node_class, "FUNCTION", "execute")
    function = getattr(instance, function_name)
    result = function(**kwargs)
    if result is None:
        return tuple()
    if isinstance(result, tuple):
        return result
    if isinstance(result, list):
        return tuple(result)
    return (result,)


def execute_node_locally(
    node_data: dict[str, Any],
    kwargs_payload: bytes | bytearray | str | dict[str, Any],
    node_mapping: dict[str, type[Any]] | None = None,
) -> bytes:
    """Execute the target node in-process and return serialized outputs."""
    _extract_custom_nodes_bundle(node_data.get("custom_nodes_bundle"))
    kwargs = deserialize_node_inputs(kwargs_payload)
    resolved_node_mapping = node_mapping or _load_nodes_module().NODE_CLASS_MAPPINGS
    class_type = node_data["class_type"]
    if class_type not in resolved_node_mapping:
        raise KeyError(f"Remote node class {class_type!r} is not registered.")

    outputs = _invoke_original_node(resolved_node_mapping[class_type], node_data, kwargs)
    return serialize_node_outputs(outputs)


def invoke_remote_engine(node_data: dict[str, Any], kwargs_payload: bytes) -> bytes:
    """Invoke Modal when configured, or fall back to local in-process execution."""
    execution_mode = os.getenv("COMFY_MODAL_EXECUTION_MODE", "local")
    if execution_mode == "local" or modal is None:
        return execute_node_locally(node_data, kwargs_payload)

    engine = RemoteEngine()
    remote_method = getattr(engine.execute_node, "remote", None)
    if callable(remote_method):
        return remote_method(node_data, kwargs_payload)
    return engine.execute_node(node_data, kwargs_payload)


if modal is not None:  # pragma: no branch - simple import-time configuration.
    settings = get_settings()
    app = modal.App(settings.app_name)
    vol = modal.Volume.from_name(settings.volume_name, create_if_missing=True)
    image = modal.Image.debian_slim().pip_install("torch", "safetensors", "pillow", "numpy")

    @app.cls(
        gpu="A100",
        volumes={settings.remote_storage_root: vol},
        scaledown_window=60,
        image=image,
    )
    class RemoteEngine:
        """Modal runtime class that executes proxied ComfyUI nodes."""

        @modal.enter()
        def setup(self) -> None:
            """Prepare the container process for headless node execution."""
            logger.info("RemoteEngine setup complete.")

        @modal.method()
        def execute_node(self, node_data: dict[str, Any], kwargs_payload: bytes) -> bytes:
            """Execute the proxied node inside the Modal container."""
            return execute_node_locally(node_data, kwargs_payload)

else:

    class RemoteEngine:
        """Local fallback runtime used when the Modal SDK is unavailable."""

        def setup(self) -> None:
            """No-op setup for local fallback execution."""

        def execute_node(self, node_data: dict[str, Any], kwargs_payload: bytes) -> bytes:
            """Execute the proxied node locally."""
            return execute_node_locally(node_data, kwargs_payload)
