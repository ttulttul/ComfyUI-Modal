"""Stable Modal cloud entrypoint for ComfyUI Modal-Sync."""

from __future__ import annotations

import asyncio
import copy
import gc
import hashlib
import importlib.util
from io import BytesIO
import json
import inspect
import logging
import os
import queue
import sys
import tempfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

_REPO_ROOT = Path(__file__).resolve().parent
_REMOTE_REPO_ROOT = Path("/root/comfyui_modal_sync_repo")
_LOCAL_COMFYUI_ROOT = (Path.home() / "git" / "ComfyUI").resolve()
_REMOTE_COMFYUI_ROOT = Path("/root/comfyui_src")
_PYTORCH_CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu128"
_COMFYUI_TORCH_VERSION = "2.10.0"
_COMFYUI_TORCHVISION_VERSION = "0.25.0"
_COMFYUI_TORCHAUDIO_VERSION = "2.10.0"
for candidate in (_REPO_ROOT, _REMOTE_REPO_ROOT, _LOCAL_COMFYUI_ROOT, _REMOTE_COMFYUI_ROOT):
    candidate_str = str(candidate)
    try:
        candidate_exists = candidate.exists()
    except PermissionError:
        candidate_exists = False
    if candidate_exists and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from serialization import coerce_serialized_node_outputs, deserialize_node_inputs, serialize_mapping, serialize_node_outputs
from settings import get_settings

logger = logging.getLogger(__name__)
_CLOUD_HANDLER_NAME = "comfyui-modal-sync-cloud-timestamped"
_COMFY_RUNTIME_INIT_LOCK = threading.Lock()
_COMFY_RUNTIME_BASE_INITIALIZED = False
_COMFY_RUNTIME_CUSTOM_NODE_ROOTS: set[str] = set()
_EXTRACTED_CUSTOM_NODE_BUNDLES: dict[str, Path] = {}
_LOADER_CACHE_LOCK = threading.Lock()
_LOADER_CACHE_WRAPPED_CLASSES: set[str] = set()
_LOADER_OUTPUT_CACHE: dict[tuple[str, str], tuple[Any, ...]] = {}
_PROMPT_EXECUTOR_STATES_LOCK = threading.Lock()
_MODAL_VOLUME_RELOAD_MARKERS_LOCK = threading.Lock()

try:
    import modal  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - remote entrypoint only.
    modal = None


class RemoteSubgraphExecutionError(RuntimeError):
    """Raised when remote subgraph execution fails."""


@dataclass
class _ReusablePromptExecutorState:
    """Hold a warm-container PromptExecutor and the lock guarding its reuse."""

    executor: Any
    lock: threading.Lock


_PROMPT_EXECUTOR_STATES: dict[str, _ReusablePromptExecutorState] = {}
_MODAL_VOLUME_RELOAD_OPEN_FILE_RETRY_DELAYS_SECONDS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
_MODAL_VOLUME_RELOAD_MARKER_CACHE_LIMIT = 256
_MODAL_VOLUME_RELOAD_MARKERS: queue.SimpleQueue[str] | None = None
_MODAL_VOLUME_RELOAD_MARKER_SET: set[str] = set()


@dataclass
class _RemoteExecutionControl:
    """Track interruption state for one active remote payload execution."""

    cancellation_event: threading.Event
    interrupt_flag_key: str


class _NullPromptServer:
    """Minimal PromptExecutor server stub for headless subgraph execution."""

    def __init__(self) -> None:
        """Initialize the no-op prompt server state."""
        self.client_id: str | None = None
        self.last_node_id: str | None = None
        self.last_prompt_id: str | None = None

    def send_sync(self, event: str, data: dict[str, Any], client_id: str | None) -> None:
        """Discard PromptExecutor progress and status events."""
        logger.debug("Suppressed remote prompt event %s for client %s.", event, client_id)


class _HeadlessPromptServerInstance:
    """Minimal PromptServer.instance replacement for custom-node import side effects."""

    def __init__(self) -> None:
        """Initialize route registration and no-op websocket state."""
        from aiohttp import web

        self.routes = web.RouteTableDef()
        self.app = web.Application()
        self.supports = ["custom_nodes_from_web"]
        self.client_id: str | None = None
        self.last_node_id: str | None = None
        self.on_prompt_handlers: list[Any] = []

    async def send(self, event: str, data: dict[str, Any], sid: str | None = None) -> None:
        """Discard async websocket sends from import-time custom-node helpers."""
        logger.debug("Suppressed headless remote prompt event %s for client %s.", event, sid)

    def send_sync(self, event: str, data: dict[str, Any], sid: str | None = None) -> None:
        """Discard sync websocket sends from import-time custom-node helpers."""
        logger.debug("Suppressed headless remote prompt event %s for client %s.", event, sid)

    def add_on_prompt_handler(self, handler: Any) -> None:
        """Record prompt handlers registered by custom nodes during import."""
        self.on_prompt_handlers.append(handler)


class _TracingPromptServer(_NullPromptServer):
    """PromptExecutor server stub that records coarse per-node execution timings."""

    def __init__(
        self,
        prompt_id: str,
        prompt: dict[str, Any],
        status_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        """Initialize timing state for a specific prompt execution."""
        super().__init__()
        self.prompt_id = prompt_id
        self.prompt = prompt
        self._status_callback = status_callback
        self._active_node_id: str | None = None
        self._active_node_started_at: float | None = None
        self.last_prompt_id = prompt_id
        self._boundary_outputs_by_node_id: dict[str, list[dict[str, Any]]] = {}
        self._lookup_cache_entry: Callable[[str], Any | None] | None = None
        self._published_boundary_outputs: set[tuple[str, int]] = set()

    def _classify_node_role(self, class_type: str) -> str:
        """Return a coarse role name for a node class."""
        normalized = class_type.lower()
        if "loader" in normalized or normalized in {"clipvisionencode"}:
            return "model_load"
        if "ksampler" in normalized or "sampler" in normalized:
            return "sampling"
        if "encode" in normalized:
            return "conditioning"
        return "node"

    def _log_node_finish(self, reason: str) -> None:
        """Emit a timing line for the currently active node when one is running."""
        if self._active_node_id is None or self._active_node_started_at is None:
            return

        node_id = self._active_node_id
        node_info = self.prompt.get(node_id, {})
        class_type = str(node_info.get("class_type", "<unknown>"))
        role = self._classify_node_role(class_type)
        elapsed_seconds = time.perf_counter() - self._active_node_started_at
        _emit_cloud_info(
            "Remote node %s class_type=%s role=%s finished in %.3fs reason=%s",
            node_id,
            class_type,
            role,
            elapsed_seconds,
            reason,
        )
        self._active_node_id = None
        self._active_node_started_at = None

    def emit_preview_update(
        self,
        *,
        node_id: str,
        preview_image: Any,
    ) -> None:
        """Publish one preview image update through the status callback."""
        if self._status_callback is None:
            return

        try:
            image_type, image, max_size = preview_image
            image_buffer = BytesIO()
            save_kwargs: dict[str, Any] = {"format": image_type}
            if image_type == "JPEG":
                save_kwargs["quality"] = 95
            elif image_type == "PNG":
                save_kwargs["compress_level"] = 1
            image.save(image_buffer, **save_kwargs)
        except Exception:
            logger.exception("Failed to serialize remote preview image for node %s.", node_id)
            return

        try:
            from comfy_execution.progress import get_progress_state

            registry = get_progress_state()
            display_node_id = registry.dynprompt.get_display_node_id(node_id)
            parent_node_id = registry.dynprompt.get_parent_node_id(node_id)
            real_node_id = registry.dynprompt.get_real_node_id(node_id)
        except Exception:
            logger.exception("Failed to resolve preview metadata for remote node %s.", node_id)
            display_node_id = node_id
            parent_node_id = None
            real_node_id = node_id

        self._status_callback(
            {
                "event_type": "preview",
                "node_id": str(node_id),
                "display_node_id": (
                    str(display_node_id) if display_node_id is not None else None
                ),
                "parent_node_id": (
                    str(parent_node_id) if parent_node_id is not None else None
                ),
                "real_node_id": str(real_node_id) if real_node_id is not None else None,
                "image_type": str(image_type),
                "image_bytes": image_buffer.getvalue(),
                "max_size": int(max_size) if max_size is not None else None,
            }
        )

    def configure_boundary_output_stream(
        self,
        *,
        boundary_outputs: list[dict[str, Any]],
        lookup_cache_entry: Callable[[str], Any | None],
    ) -> None:
        """Configure streamed remote boundary-output publication for this execution."""
        outputs_by_node_id: dict[str, list[dict[str, Any]]] = {}
        for boundary_output in boundary_outputs:
            preview_target_node_ids = [
                str(node_id)
                for node_id in boundary_output.get("preview_target_node_ids", [])
                if str(node_id)
            ]
            if not preview_target_node_ids:
                continue
            if str(boundary_output.get("io_type")) != "IMAGE":
                continue
            node_id = str(boundary_output["node_id"])
            outputs_by_node_id.setdefault(node_id, []).append(boundary_output)

        self._boundary_outputs_by_node_id = outputs_by_node_id
        self._lookup_cache_entry = lookup_cache_entry
        self._published_boundary_outputs.clear()

    def _emit_boundary_outputs_for_node(self, node_id: str | None) -> None:
        """Publish configured boundary image outputs for one completed node once."""
        if (
            node_id is None
            or self._status_callback is None
            or self._lookup_cache_entry is None
        ):
            return

        boundary_outputs = self._boundary_outputs_by_node_id.get(str(node_id), [])
        if not boundary_outputs:
            return

        cache_entry = self._lookup_cache_entry(str(node_id))
        if cache_entry is None:
            return

        cache_outputs = getattr(cache_entry, "outputs", None)
        if not isinstance(cache_outputs, (list, tuple)):
            return

        for boundary_output in boundary_outputs:
            output_index = int(boundary_output["output_index"])
            publication_key = (str(node_id), output_index)
            if publication_key in self._published_boundary_outputs:
                continue
            if output_index >= len(cache_outputs):
                continue

            preview_target_node_ids = [
                str(target_node_id)
                for target_node_id in boundary_output.get("preview_target_node_ids", [])
                if str(target_node_id)
            ]
            if not preview_target_node_ids:
                continue

            self._status_callback(
                {
                    "event_type": "boundary_output",
                    "node_id": str(node_id),
                    "output_index": output_index,
                    "io_type": str(boundary_output.get("io_type", "")),
                    "is_list": bool(boundary_output.get("is_list", False)),
                    "preview_target_node_ids": preview_target_node_ids,
                    "value": _collapse_cache_slot(
                        slot_values=cache_outputs[output_index],
                        is_list=bool(boundary_output.get("is_list", False)),
                    ),
                }
            )
            self._published_boundary_outputs.add(publication_key)

    def send_sync(self, event: str, data: dict[str, Any], client_id: str | None) -> None:
        """Track per-node timing transitions from PromptExecutor progress events."""
        if event == "executing":
            next_node_id = data.get("node")
            if next_node_id != self._active_node_id:
                self._emit_boundary_outputs_for_node(self._active_node_id)
                self._log_node_finish(reason="next_node")
            if next_node_id is not None and next_node_id != self._active_node_id:
                node_info = self.prompt.get(str(next_node_id), {})
                class_type = str(node_info.get("class_type", "<unknown>"))
                role = self._classify_node_role(class_type)
                if self._status_callback is not None:
                    self._status_callback(
                        {
                            "phase": "executing",
                            "active_node_id": str(next_node_id),
                            "active_node_class_type": class_type,
                            "active_node_role": role,
                        }
                    )
                self._active_node_id = str(next_node_id)
                self._active_node_started_at = time.perf_counter()
                self.last_node_id = self._active_node_id
                _emit_cloud_info(
                    "Remote node %s class_type=%s role=%s started",
                    self._active_node_id,
                    class_type,
                    role,
                )
            return

        if event == "progress_state":
            if self._status_callback is None:
                return

            nodes_payload = data.get("nodes")
            if not isinstance(nodes_payload, dict):
                return

            tracked_node_id = self._active_node_id
            tracked_node_state: dict[str, Any] | None = None
            if tracked_node_id is not None:
                candidate_state = nodes_payload.get(tracked_node_id)
                if isinstance(candidate_state, dict):
                    tracked_node_state = candidate_state

            if tracked_node_state is None:
                for node_state in nodes_payload.values():
                    if isinstance(node_state, dict) and node_state.get("state") == "running":
                        tracked_node_state = node_state
                        break

            if tracked_node_state is None:
                return

            display_node_id = tracked_node_state.get("display_node_id")
            real_node_id = tracked_node_state.get("real_node_id")
            reported_node_id = display_node_id or real_node_id or tracked_node_state.get("node_id")
            if reported_node_id is None:
                return

            self._status_callback(
                {
                    "event_type": "node_progress",
                    "node_id": str(reported_node_id),
                    "display_node_id": (
                        str(display_node_id) if display_node_id is not None else None
                    ),
                    "value": float(tracked_node_state.get("value", 0.0)),
                    "max": float(tracked_node_state.get("max", 1.0)),
                }
            )
            return

        if event == "executed":
            executed_node_id = data.get("node")
            if executed_node_id is not None and str(executed_node_id) == self._active_node_id:
                self._log_node_finish(reason="executed")
            if self._status_callback is not None and data.get("output") is not None:
                self._status_callback(
                    {
                        "event_type": "executed",
                        "node_id": str(data.get("node")),
                        "display_node_id": (
                            str(data["display_node"])
                            if data.get("display_node") is not None
                            else None
                        ),
                        "output": data.get("output"),
                    }
                )
            return

        if event in {"execution_error", "execution_interrupted", "execution_success"}:
            self._emit_boundary_outputs_for_node(self._active_node_id)
            self._log_node_finish(reason=event)
            if self._status_callback is not None:
                self._status_callback({"phase": event})
            return

        super().send_sync(event, data, client_id)


def _build_cloud_log_formatter() -> logging.Formatter:
    """Return the default formatter for remote Modal-Sync logs with timestamps."""
    return logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d +%(relativeCreated)07.0fms %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _configure_cloud_logging() -> logging.Logger:
    """Install a dedicated timestamped handler for the cloud runtime logger."""
    logger.setLevel(logging.INFO)
    for existing_handler in logger.handlers:
        if getattr(existing_handler, "name", "") == _CLOUD_HANDLER_NAME:
            return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.set_name(_CLOUD_HANDLER_NAME)
    handler.setLevel(logging.INFO)
    handler.setFormatter(_build_cloud_log_formatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def _is_modal_container_runtime() -> bool:
    """Return whether the current process is executing inside a Modal container."""
    return os.getenv("MODAL_IS_REMOTE") == "1" or bool(os.getenv("MODAL_TASK_ID"))


def _cloud_formatter() -> logging.Formatter:
    """Return the configured formatter used for cloud phase trace lines."""
    for existing_handler in logger.handlers:
        if getattr(existing_handler, "name", "") == _CLOUD_HANDLER_NAME:
            formatter = existing_handler.formatter
            if formatter is not None:
                return formatter
    return _build_cloud_log_formatter()


def _emit_cloud_info(message: str, *args: Any) -> None:
    """Emit an info line through logging and mirror it to stdout inside Modal containers."""
    if not _is_modal_container_runtime():
        logger.info(message, *args)
        return

    record = logger.makeRecord(
        logger.name,
        logging.INFO,
        __file__,
        0,
        message,
        args,
        exc_info=None,
    )
    print(_cloud_formatter().format(record), file=sys.stdout, flush=True)


def _remote_execution_key(payload: dict[str, Any]) -> tuple[str, str]:
    """Return the registry key for one active remote execution."""
    prompt_id = str(payload.get("prompt_id") or payload.get("component_id") or "modal-subgraph")
    component_id = str(payload.get("component_id") or "single-node")
    return prompt_id, component_id


def _remote_interrupt_flag_key(prompt_id: str, component_id: str) -> str:
    """Return the shared Modal interrupt-store key for one payload execution."""
    return f"{prompt_id}:{component_id}"


@contextmanager
def _registered_remote_execution(
    payload: dict[str, Any],
) -> Iterator[_RemoteExecutionControl]:
    """Prepare interruption state for one active remote execution."""
    prompt_id, component_id = _remote_execution_key(payload)
    control = _RemoteExecutionControl(
        cancellation_event=threading.Event(),
        interrupt_flag_key=_remote_interrupt_flag_key(prompt_id, component_id),
    )
    if modal is not None and "interrupt_flags" in globals():
        interrupt_flags.pop(control.interrupt_flag_key, None)
    try:
        yield control
    finally:
        if modal is not None and "interrupt_flags" in globals():
            interrupt_flags.pop(control.interrupt_flag_key, None)


@contextmanager
def _timed_phase(phase: str, **fields: Any) -> Iterator[None]:
    """Log a start/finish pair with elapsed time for a named execution phase."""
    field_suffix = ""
    if fields:
        rendered_fields = " ".join(f"{key}={value}" for key, value in fields.items())
        field_suffix = f" {rendered_fields}"
    phase_started_at = time.perf_counter()
    _emit_cloud_info("Starting %s%s", phase, field_suffix)
    try:
        yield
    finally:
        _emit_cloud_info(
            "Finished %s in %.3fs%s",
            phase,
            time.perf_counter() - phase_started_at,
            field_suffix,
        )


_configure_cloud_logging()


@contextmanager
def _temporary_node_mapping(node_mapping: dict[str, type[Any]] | None) -> Iterator[None]:
    """Temporarily overlay node mappings for tests or custom runtimes."""
    if node_mapping is None:
        yield
        return

    import nodes

    original_mappings = dict(nodes.NODE_CLASS_MAPPINGS)
    original_display_mappings = dict(getattr(nodes, "NODE_DISPLAY_NAME_MAPPINGS", {}))
    try:
        nodes.NODE_CLASS_MAPPINGS.update(node_mapping)
        for class_type in node_mapping:
            nodes.NODE_DISPLAY_NAME_MAPPINGS.setdefault(class_type, class_type)
        yield
    finally:
        nodes.NODE_CLASS_MAPPINGS.clear()
        nodes.NODE_CLASS_MAPPINGS.update(original_mappings)
        if hasattr(nodes, "NODE_DISPLAY_NAME_MAPPINGS"):
            nodes.NODE_DISPLAY_NAME_MAPPINGS.clear()
            nodes.NODE_DISPLAY_NAME_MAPPINGS.update(original_display_mappings)


def _extract_custom_nodes_bundle(bundle_path: str | None) -> Path | None:
    """Extract a mirrored custom_nodes archive into a temporary import path."""
    if not bundle_path:
        return None

    settings = get_settings()
    storage_roots = [Path(settings.remote_storage_root)]
    if settings.local_storage_root is not None:
        storage_roots.append(settings.local_storage_root)

    local_bundle: Path | None = None
    for storage_root in storage_roots:
        candidate = storage_root / bundle_path.lstrip("/")
        if candidate.exists():
            local_bundle = candidate
            break

    if local_bundle is None:
        logger.warning("Custom nodes bundle %s was not found in any known storage root.", bundle_path)
        return None

    cached_extraction_root = _EXTRACTED_CUSTOM_NODE_BUNDLES.get(local_bundle.name)
    if cached_extraction_root is not None and cached_extraction_root.exists():
        if str(cached_extraction_root) not in sys.path:
            sys.path.insert(0, str(cached_extraction_root))
        _emit_cloud_info(
            "Reusing extracted remote custom_nodes bundle from %s for %s.",
            cached_extraction_root,
            local_bundle.name,
        )
        return cached_extraction_root

    extraction_root = Path(tempfile.gettempdir()) / "comfy-modal-sync-custom-nodes" / local_bundle.stem
    extraction_root.mkdir(parents=True, exist_ok=True)
    with _timed_phase("extract_custom_nodes_bundle", bundle=local_bundle.name):
        with zipfile.ZipFile(local_bundle, "r") as archive:
            archive.extractall(extraction_root)

    if str(extraction_root) not in sys.path:
        sys.path.insert(0, str(extraction_root))
    _EXTRACTED_CUSTOM_NODE_BUNDLES[local_bundle.name] = extraction_root
    logger.info("Extracted remote custom_nodes bundle to %s", extraction_root)
    return extraction_root


def _register_custom_nodes_root(custom_nodes_root: Path) -> None:
    """Expose an extracted custom_nodes directory to ComfyUI's folder path registry."""
    import folder_paths

    folder_paths.add_model_folder_path("custom_nodes", str(custom_nodes_root), is_default=True)


def _active_comfyui_root() -> Path | None:
    """Return the ComfyUI source root visible to this runtime."""
    for candidate in (_REMOTE_COMFYUI_ROOT, _LOCAL_COMFYUI_ROOT):
        try:
            if candidate.exists():
                return candidate
        except PermissionError:
            continue
    return None


def _force_import_package_from_root(module_name: str, package_root: Path) -> None:
    """Load a top-level package from a specific root, replacing a non-package shadow if needed."""
    existing_module = sys.modules.get(module_name)
    if existing_module is not None and getattr(existing_module, "__path__", None):
        return

    package_dir = package_root / module_name
    init_path = package_dir / "__init__.py"
    if not init_path.exists():
        logger.debug("Package %s does not exist under %s.", module_name, package_root)
        return

    spec = importlib.util.spec_from_file_location(
        module_name,
        init_path,
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create an import spec for package {module_name!r}.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    logger.info("Preloaded ComfyUI package %s from %s.", module_name, package_dir)


def _ensure_comfyui_support_packages() -> None:
    """Preload top-level ComfyUI support packages that are vulnerable to name shadowing."""
    comfyui_root = _active_comfyui_root()
    if comfyui_root is None:
        return

    _force_import_package_from_root("utils", comfyui_root)


def _ensure_headless_prompt_server_instance() -> None:
    """Install a minimal PromptServer.instance for custom-node import-time hooks."""
    try:
        import server
    except ModuleNotFoundError:
        return

    prompt_server_class = getattr(server, "PromptServer", None)
    if prompt_server_class is None:
        return
    if getattr(prompt_server_class, "instance", None) is not None:
        return

    prompt_server_class.instance = _HeadlessPromptServerInstance()
    logger.info("Installed headless PromptServer.instance for remote custom-node initialization.")


@contextmanager
def _temporary_progress_hook(prompt_server: _NullPromptServer) -> Iterator[None]:
    """Install a ComfyUI progress hook so remote samplers emit numeric progress updates."""
    import comfy.utils
    import comfy.model_management
    from comfy_execution.progress import get_progress_state
    from comfy_execution.utils import get_executing_context

    previous_hook = comfy.utils.PROGRESS_BAR_HOOK

    def hook(
        value: float,
        total: float,
        preview_image: Any,
        prompt_id: str | None = None,
        node_id: str | None = None,
    ) -> None:
        """Mirror ComfyUI progress-bar updates into the headless progress registry."""
        executing_context = get_executing_context()
        if prompt_id is None and executing_context is not None:
            prompt_id = executing_context.prompt_id
        if node_id is None and executing_context is not None:
            node_id = executing_context.node_id
        comfy.model_management.throw_exception_if_processing_interrupted()
        if prompt_id is None:
            prompt_id = prompt_server.last_prompt_id
        if node_id is None:
            node_id = prompt_server.last_node_id
        if node_id is None:
            return

        resolved_node_id = str(node_id)
        get_progress_state().update_progress(resolved_node_id, value, total, preview_image)
        preview_emitter = getattr(prompt_server, "emit_preview_update", None)
        if preview_image is not None and callable(preview_emitter):
            preview_emitter(node_id=resolved_node_id, preview_image=preview_image)

    comfy.utils.set_progress_bar_global_hook(hook)
    try:
        yield
    finally:
        comfy.utils.set_progress_bar_global_hook(previous_hook)


@contextmanager
def _temporary_remote_interrupt_monitor(
    component_id: str,
    cancellation_event: threading.Event | None,
    interrupt_store: Any | None = None,
    interrupt_flag_key: str | None = None,
) -> Iterator[None]:
    """Mirror shared cancellation requests into ComfyUI's interrupt flag inside Modal."""
    if cancellation_event is None and (interrupt_store is None or interrupt_flag_key is None):
        yield
        return

    import nodes

    stop_event = threading.Event()

    def monitor_interrupts() -> None:
        """Set ComfyUI's interrupt flag once the caller requests cancellation."""
        while not stop_event.is_set():
            if cancellation_event is not None and cancellation_event.wait(timeout=0.1):
                logger.info("Remote interrupt monitor tripped local event for component=%s.", component_id)
                nodes.interrupt_processing()
                return
            if interrupt_store is None or interrupt_flag_key is None:
                continue
            if not interrupt_store.contains(interrupt_flag_key):
                continue
            logger.info(
                "Remote interrupt monitor observed shared cancel flag for component=%s.",
                component_id,
            )
            interrupt_store.pop(interrupt_flag_key, None)
            if cancellation_event is not None:
                cancellation_event.set()
            nodes.interrupt_processing()
            return

    interrupt_thread = threading.Thread(
        target=monitor_interrupts,
        name=f"modal-interrupt-{component_id}",
        daemon=True,
    )
    interrupt_thread.start()
    try:
        yield
    finally:
        stop_event.set()
        interrupt_thread.join(timeout=1.0)


def _ensure_default_custom_nodes_dir() -> Path | None:
    """Create the default ComfyUI custom_nodes directory when the image omits its contents."""
    comfyui_root = _active_comfyui_root()
    if comfyui_root is None:
        return None

    custom_nodes_dir = comfyui_root / "custom_nodes"
    custom_nodes_dir.mkdir(parents=True, exist_ok=True)
    return custom_nodes_dir


def _materialize_remote_asset_path(value: str) -> str:
    """Resolve a mirrored Modal asset reference to the container-local absolute file path."""
    settings = get_settings()
    remote_storage_root = settings.remote_storage_root.rstrip("/")
    if value.startswith(f"{remote_storage_root}/"):
        return value
    if value.startswith("/assets/"):
        return f"{remote_storage_root}{value}"
    return value


def _clone_loader_cache_value(value: Any) -> Any:
    """Clone a cached loader output when the runtime object supports safe cloning."""
    clone_method = getattr(value, "clone", None)
    if callable(clone_method):
        return clone_method()
    return value


def _clone_loader_cache_outputs(outputs: tuple[Any, ...]) -> tuple[Any, ...]:
    """Return a request-safe copy of cached loader outputs."""
    return tuple(_clone_loader_cache_value(output) for output in outputs)


def _serialize_loader_cache_key(parts: dict[str, Any]) -> str:
    """Serialize a loader cache key into a stable string representation."""
    return json.dumps(parts, sort_keys=True, default=str)


def _build_unet_loader_cache_key(kwargs: dict[str, Any]) -> str:
    """Build a stable cache key for the ComfyUI UNET loader."""
    import folder_paths

    return _serialize_loader_cache_key(
        {
            "unet_path": folder_paths.get_full_path_or_raise(
                "diffusion_models",
                str(kwargs["unet_name"]),
            ),
            "weight_dtype": kwargs.get("weight_dtype", "default"),
        }
    )


def _build_clip_loader_cache_key(kwargs: dict[str, Any]) -> str:
    """Build a stable cache key for the ComfyUI CLIP loader."""
    import folder_paths

    return _serialize_loader_cache_key(
        {
            "clip_path": folder_paths.get_full_path_or_raise(
                "text_encoders",
                str(kwargs["clip_name"]),
            ),
            "type": kwargs.get("type", "stable_diffusion"),
            "device": kwargs.get("device", "default"),
        }
    )


def _build_vae_loader_cache_key(kwargs: dict[str, Any]) -> str:
    """Build a stable cache key for the ComfyUI VAE loader."""
    return _serialize_loader_cache_key({"vae_name": kwargs.get("vae_name")})


def _wrap_loader_method_with_cache(
    class_type: str,
    node_class: type[Any],
    method_name: str,
    cache_key_builder: Any,
) -> None:
    """Install a warm-container cache wrapper around a heavy loader method."""
    if class_type in _LOADER_CACHE_WRAPPED_CLASSES:
        return

    original_method = getattr(node_class, method_name)
    method_signature = inspect.signature(original_method)

    def cached_method(self: Any, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        """Return cached loader outputs when an identical request was already loaded."""
        bound = method_signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        normalized_kwargs = {key: value for key, value in bound.arguments.items() if key != "self"}
        cache_key = (class_type, cache_key_builder(normalized_kwargs))

        with _LOADER_CACHE_LOCK:
            cached_outputs = _LOADER_OUTPUT_CACHE.get(cache_key)
        if cached_outputs is not None:
            _emit_cloud_info("Loader cache hit class_type=%s key=%s", class_type, cache_key[1])
            return _clone_loader_cache_outputs(cached_outputs)

        _emit_cloud_info("Loader cache miss class_type=%s key=%s", class_type, cache_key[1])
        outputs = original_method(self, *args, **kwargs)
        normalized_outputs = tuple(outputs) if isinstance(outputs, (list, tuple)) else (outputs,)
        with _LOADER_CACHE_LOCK:
            _LOADER_OUTPUT_CACHE[cache_key] = normalized_outputs
        return _clone_loader_cache_outputs(normalized_outputs)

    setattr(node_class, method_name, cached_method)
    _LOADER_CACHE_WRAPPED_CLASSES.add(class_type)


def _install_loader_cache_wrappers() -> None:
    """Patch the heavyweight built-in model loaders to reuse warm-container state."""
    nodes_module = _load_nodes_module()
    cacheable_loader_specs = {
        "UNETLoader": ("load_unet", _build_unet_loader_cache_key),
        "CLIPLoader": ("load_clip", _build_clip_loader_cache_key),
        "VAELoader": ("load_vae", _build_vae_loader_cache_key),
    }

    for class_type, (method_name, cache_key_builder) in cacheable_loader_specs.items():
        node_class = nodes_module.NODE_CLASS_MAPPINGS.get(class_type)
        if node_class is None:
            continue
        _wrap_loader_method_with_cache(class_type, node_class, method_name, cache_key_builder)


def _rewrite_modal_asset_references(value: Any) -> Any:
    """Recursively replace mirrored asset markers with container-local absolute file paths."""
    if isinstance(value, str):
        return _materialize_remote_asset_path(value)
    if isinstance(value, list):
        return [_rewrite_modal_asset_references(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _rewrite_modal_asset_references(item) for key, item in value.items()}
    return value


@contextmanager
def _patched_folder_paths_absolute_lookup() -> Iterator[None]:
    """Teach ComfyUI folder lookups to accept already-materialized absolute asset paths."""
    import folder_paths

    original_get_full_path = folder_paths.get_full_path
    original_get_full_path_or_raise = folder_paths.get_full_path_or_raise

    def patched_get_full_path(folder_name: str, filename: str) -> str | None:
        """Return the absolute file when the prompt already points at a materialized asset."""
        resolved_filename = _materialize_remote_asset_path(filename)
        if os.path.isabs(resolved_filename) and Path(resolved_filename).is_file():
            return resolved_filename
        return original_get_full_path(folder_name, resolved_filename)

    def patched_get_full_path_or_raise(folder_name: str, filename: str) -> str:
        """Raise with the original message when no absolute or folder-based match exists."""
        full_path = patched_get_full_path(folder_name, filename)
        if full_path is None:
            raise FileNotFoundError(
                f"Model in folder '{folder_name}' with filename '{filename}' not found."
            )
        return full_path

    folder_paths.get_full_path = patched_get_full_path
    folder_paths.get_full_path_or_raise = patched_get_full_path_or_raise
    try:
        yield
    finally:
        folder_paths.get_full_path = original_get_full_path
        folder_paths.get_full_path_or_raise = original_get_full_path_or_raise


def _ensure_comfy_runtime_initialized(custom_nodes_root: Path | None) -> None:
    """Initialize ComfyUI's built-in and external node registries for remote execution."""
    global _COMFY_RUNTIME_BASE_INITIALIZED

    custom_nodes_root_key = str(custom_nodes_root.resolve()) if custom_nodes_root is not None else None
    with _COMFY_RUNTIME_INIT_LOCK:
        with _timed_phase(
            "ensure_comfy_runtime_initialized",
            custom_nodes=custom_nodes_root_key or "none",
        ):
            _ensure_comfyui_support_packages()
            _ensure_default_custom_nodes_dir()
            _ensure_headless_prompt_server_instance()
            nodes_module = _load_nodes_module()

            if not _COMFY_RUNTIME_BASE_INITIALIZED:
                if custom_nodes_root is not None:
                    _register_custom_nodes_root(custom_nodes_root)
                logger.info(
                    "Initializing remote ComfyUI node registry with built-in extras%s.",
                    " and extracted custom nodes" if custom_nodes_root is not None else "",
                )
                with _timed_phase(
                    "init_extra_nodes",
                    custom_nodes=bool(custom_nodes_root is not None),
                    api_nodes=True,
                ):
                    asyncio.run(
                        nodes_module.init_extra_nodes(
                            init_custom_nodes=custom_nodes_root is not None,
                            init_api_nodes=True,
                        )
                    )
                _install_loader_cache_wrappers()
                _COMFY_RUNTIME_BASE_INITIALIZED = True
                if custom_nodes_root_key is not None:
                    _COMFY_RUNTIME_CUSTOM_NODE_ROOTS.add(custom_nodes_root_key)
                return

            if custom_nodes_root_key is None or custom_nodes_root_key in _COMFY_RUNTIME_CUSTOM_NODE_ROOTS:
                _install_loader_cache_wrappers()
                return

            _register_custom_nodes_root(custom_nodes_root)
            logger.info("Loading extracted remote custom nodes from %s.", custom_nodes_root)
            with _timed_phase("init_external_custom_nodes", custom_nodes=custom_nodes_root_key):
                asyncio.run(nodes_module.init_external_custom_nodes())
            _install_loader_cache_wrappers()
            _COMFY_RUNTIME_CUSTOM_NODE_ROOTS.add(custom_nodes_root_key)


def _load_execution_module() -> Any:
    """Import the ComfyUI execution module lazily."""
    _ensure_comfyui_support_packages()
    import execution

    return execution


def _load_nodes_module() -> Any:
    """Import the ComfyUI nodes module lazily."""
    import nodes

    return nodes


def _prompt_executor_cache_config(execution: Any) -> tuple[Any, dict[str, float]]:
    """Return the cache settings used by ComfyUI's normal prompt worker."""
    from comfy.cli_args import args

    cache_type = execution.CacheType.CLASSIC
    if args.cache_lru > 0:
        cache_type = execution.CacheType.LRU
    elif args.cache_ram > 0:
        cache_type = execution.CacheType.RAM_PRESSURE
    elif args.cache_none:
        cache_type = execution.CacheType.NONE

    return cache_type, {"lru": args.cache_lru, "ram": args.cache_ram}


def _serialize_prompt_executor_cache_scope(
    cache_type: Any,
    cache_args: dict[str, Any],
    custom_nodes_root: Path | None,
) -> str:
    """Return a stable cache scope key for reusable PromptExecutor instances."""
    return json.dumps(
        {
            "cache_type": str(cache_type),
            "cache_args": cache_args,
            "custom_nodes_root": str(custom_nodes_root.resolve()) if custom_nodes_root is not None else None,
        },
        sort_keys=True,
        default=str,
    )


def _reset_prompt_executor_request_state(executor: Any, prompt_server: Any) -> None:
    """Prepare a reusable PromptExecutor for a fresh request without discarding its caches."""
    executor.server = prompt_server
    executor.status_messages = []
    executor.success = True
    executor.history_result = {}
    prompt_server.client_id = None
    prompt_server.last_node_id = None


def _get_or_create_prompt_executor_state(
    execution: Any,
    prompt_server: Any,
    cache_type: Any,
    cache_args: dict[str, Any],
    custom_nodes_root: Path | None,
) -> _ReusablePromptExecutorState:
    """Return the warm-container PromptExecutor state for a cache scope, creating it once."""
    state_key = _serialize_prompt_executor_cache_scope(cache_type, cache_args, custom_nodes_root)
    with _PROMPT_EXECUTOR_STATES_LOCK:
        existing_state = _PROMPT_EXECUTOR_STATES.get(state_key)
        if existing_state is not None:
            _emit_cloud_info("Prompt executor cache hit scope=%s", state_key)
            return existing_state

        _emit_cloud_info("Prompt executor cache miss scope=%s", state_key)
        executor = execution.PromptExecutor(
            prompt_server,
            cache_type=cache_type,
            cache_args=cache_args,
        )
        state = _ReusablePromptExecutorState(executor=executor, lock=threading.Lock())
        _PROMPT_EXECUTOR_STATES[state_key] = state
        return state


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
    cancellation_event: threading.Event | None = None,
    interrupt_store: Any | None = None,
    interrupt_flag_key: str | None = None,
) -> bytes:
    """Execute a single target node in-process and return serialized outputs."""
    custom_nodes_root = _extract_custom_nodes_bundle(node_data.get("custom_nodes_bundle"))
    _ensure_comfy_runtime_initialized(custom_nodes_root)
    kwargs = _rewrite_modal_asset_references(deserialize_node_inputs(kwargs_payload))
    component_id = str(node_data.get("component_id") or node_data.get("class_type") or "single-node")
    if node_mapping is not None:
        class_type = node_data["class_type"]
        if class_type not in node_mapping:
            raise KeyError(f"Remote node class {class_type!r} is not registered.")
        with (
            _patched_folder_paths_absolute_lookup(),
            _temporary_remote_interrupt_monitor(
                component_id,
                cancellation_event,
                interrupt_store=interrupt_store,
                interrupt_flag_key=interrupt_flag_key,
            ),
        ):
            outputs = _invoke_original_node(node_mapping[class_type], node_data, kwargs)
            return serialize_node_outputs(outputs)

    with _temporary_node_mapping(node_mapping):
        resolved_node_mapping = _load_nodes_module().NODE_CLASS_MAPPINGS
        class_type = node_data["class_type"]
        if class_type not in resolved_node_mapping:
            raise KeyError(f"Remote node class {class_type!r} is not registered.")

        with (
            _patched_folder_paths_absolute_lookup(),
            _temporary_remote_interrupt_monitor(
                component_id,
                cancellation_event,
                interrupt_store=interrupt_store,
                interrupt_flag_key=interrupt_flag_key,
            ),
        ):
            outputs = _invoke_original_node(resolved_node_mapping[class_type], node_data, kwargs)
    return serialize_node_outputs(outputs)


def _apply_boundary_inputs(
    prompt: dict[str, Any],
    boundary_input_specs: list[dict[str, Any]],
    hydrated_inputs: dict[str, Any],
) -> None:
    """Inject hydrated local boundary inputs into a remote subgraph prompt."""
    for boundary_input in boundary_input_specs:
        proxy_input_name = str(boundary_input["proxy_input_name"])
        if proxy_input_name not in hydrated_inputs:
            raise KeyError(f"Missing hydrated boundary input {proxy_input_name!r}.")
        value = hydrated_inputs[proxy_input_name]
        for target in boundary_input.get("targets", []):
            node_id = str(target["node_id"])
            input_name = str(target["input_name"])
            prompt[node_id]["inputs"][input_name] = value


def _collapse_cache_slot(slot_values: Any, is_list: bool) -> Any:
    """Convert a PromptExecutor cache slot back into a node-style output value."""
    if is_list:
        return slot_values
    if not isinstance(slot_values, list):
        return slot_values
    if len(slot_values) == 1:
        return slot_values[0]
    return slot_values


def _extract_prompt_executor_error(executor: Any) -> str:
    """Extract a useful failure message from a PromptExecutor run."""
    for event, data in reversed(executor.status_messages):
        if event == "execution_error":
            return str(data.get("exception_message") or "Remote subgraph execution failed.")
        if event == "execution_interrupted":
            return "Remote subgraph execution was interrupted."
    return "Remote subgraph execution failed."


def _normalize_link_output_index(value: Any) -> Any:
    """Unwrap a singleton list around a prompt-link output index when present."""
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], int | float):
        return value[0]
    return value


def _normalize_subgraph_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a subgraph payload with canonical prompt-link and output-index shapes."""
    normalized_payload = copy.deepcopy(payload)

    for node_info in normalized_payload.get("subgraph_prompt", {}).values():
        inputs = node_info.get("inputs") or {}
        for input_name, input_value in list(inputs.items()):
            if not isinstance(input_value, list) or len(input_value) != 2:
                continue
            if not isinstance(input_value[0], str):
                continue
            inputs[input_name] = [
                input_value[0],
                _normalize_link_output_index(input_value[1]),
            ]

    for boundary_output in normalized_payload.get("boundary_outputs", []):
        if "output_index" in boundary_output:
            boundary_output["output_index"] = _normalize_link_output_index(
                boundary_output["output_index"]
            )

    return normalized_payload


def _execute_subgraph_prompt(
    payload: dict[str, Any],
    hydrated_inputs: dict[str, Any],
    custom_nodes_root: Path | None,
    status_callback: Callable[[dict[str, Any]], None] | None = None,
    cancellation_event: threading.Event | None = None,
    interrupt_store: Any | None = None,
    interrupt_flag_key: str | None = None,
) -> tuple[Any, ...]:
    """Execute a remote component prompt and return its exported outputs."""
    component_id = str(payload.get("component_id", "modal-subgraph"))
    normalized_payload = _normalize_subgraph_payload(payload)
    with _timed_phase("prepare_subgraph_prompt", component=component_id):
        prompt = _rewrite_modal_asset_references(copy.deepcopy(normalized_payload["subgraph_prompt"]))
        _apply_boundary_inputs(
            prompt=prompt,
            boundary_input_specs=list(normalized_payload.get("boundary_inputs", [])),
            hydrated_inputs=hydrated_inputs,
        )
    with _timed_phase("load_execution_module", component=component_id):
        execution = _load_execution_module()
        cache_type, cache_args = _prompt_executor_cache_config(execution)

    with (
        _temporary_node_mapping(None),
        _patched_folder_paths_absolute_lookup(),
        _temporary_remote_interrupt_monitor(
            component_id,
            cancellation_event,
            interrupt_store=interrupt_store,
            interrupt_flag_key=interrupt_flag_key,
        ),
        _temporary_progress_hook(
            prompt_server := _TracingPromptServer(
                component_id,
                prompt,
                status_callback=status_callback,
            )
        ),
    ):
        with _timed_phase("create_prompt_executor", component=component_id):
            executor_state = _get_or_create_prompt_executor_state(
                execution=execution,
                prompt_server=prompt_server,
                cache_type=cache_type,
                cache_args=cache_args,
                custom_nodes_root=custom_nodes_root,
            )
        prompt_server.configure_boundary_output_stream(
            boundary_outputs=list(normalized_payload.get("boundary_outputs", [])),
            lookup_cache_entry=lambda node_id: executor_state.executor.caches.outputs.get(node_id),
        )
        with executor_state.lock:
            _reset_prompt_executor_request_state(executor_state.executor, prompt_server)
            with _timed_phase(
                "prompt_executor_execute",
                component=component_id,
                execute_nodes=list(normalized_payload.get("execute_node_ids", [])),
            ):
                executor_state.executor.execute(
                    prompt=prompt,
                    prompt_id=component_id,
                    extra_data=copy.deepcopy(normalized_payload.get("extra_data") or {}),
                    execute_outputs=list(normalized_payload.get("execute_node_ids", [])),
                )
            executor = executor_state.executor
        if not executor.success:
            raise RemoteSubgraphExecutionError(_extract_prompt_executor_error(executor))

        outputs: list[Any] = []
        with _timed_phase(
            "collect_boundary_outputs",
            component=component_id,
            output_count=len(normalized_payload.get("boundary_outputs", [])),
        ):
            for boundary_output in normalized_payload.get("boundary_outputs", []):
                node_id = str(boundary_output["node_id"])
                output_index = int(boundary_output["output_index"])
                cache_entry = executor.caches.outputs.get(node_id)
                if cache_entry is None:
                    raise RemoteSubgraphExecutionError(
                        f"Remote subgraph did not produce cache entry for node {node_id}."
                    )
                if output_index >= len(cache_entry.outputs):
                    raise RemoteSubgraphExecutionError(
                        f"Remote subgraph output index {output_index} is missing for node {node_id}."
                    )
                outputs.append(
                    _collapse_cache_slot(
                        slot_values=cache_entry.outputs[output_index],
                        is_list=bool(boundary_output.get("is_list", False)),
                    )
                )
        return tuple(outputs)


def execute_subgraph_locally(
    payload: dict[str, Any],
    kwargs_payload: bytes | bytearray | str | dict[str, Any],
    status_callback: Callable[[dict[str, Any]], None] | None = None,
    cancellation_event: threading.Event | None = None,
    interrupt_store: Any | None = None,
    interrupt_flag_key: str | None = None,
) -> bytes:
    """Execute a rewritten remote component in-process and return serialized outputs."""
    component_id = str(payload.get("component_id", "modal-subgraph"))
    with _timed_phase("execute_subgraph_locally", component=component_id):
        custom_nodes_root = _extract_custom_nodes_bundle(payload.get("custom_nodes_bundle"))
        _ensure_comfy_runtime_initialized(custom_nodes_root)
        with _timed_phase("deserialize_boundary_inputs", component=component_id):
            hydrated_inputs = deserialize_node_inputs(kwargs_payload)
        with _timed_phase("subgraph_worker_roundtrip", component=component_id):
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    _execute_subgraph_prompt,
                    payload,
                    hydrated_inputs,
                    custom_nodes_root,
                    status_callback,
                    cancellation_event,
                    interrupt_store,
                    interrupt_flag_key,
                )
                outputs = future.result()
        with _timed_phase("serialize_boundary_outputs", component=component_id):
            return serialize_node_outputs(outputs)


def _stream_remote_payload_events(
    payload: dict[str, Any],
    kwargs_payload: bytes | bytearray | str | dict[str, Any],
    cancellation_event: threading.Event | None = None,
    interrupt_store: Any | None = None,
    interrupt_flag_key: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield progress and result events for one remote payload execution."""
    event_queue: queue.Queue[tuple[str, Any]] = queue.Queue()

    def publish_status(progress_state: dict[str, Any]) -> None:
        """Queue a progress envelope for the remote caller."""
        event_queue.put(("progress", serialize_mapping(progress_state)))

    def execute_payload() -> None:
        """Run the payload in a worker thread and enqueue the terminal outcome."""
        try:
            if payload.get("payload_kind") == "subgraph":
                execute_subgraph_kwargs: dict[str, Any] = {"status_callback": publish_status}
                if "cancellation_event" in inspect.signature(execute_subgraph_locally).parameters:
                    execute_subgraph_kwargs["cancellation_event"] = cancellation_event
                if "interrupt_store" in inspect.signature(execute_subgraph_locally).parameters:
                    execute_subgraph_kwargs["interrupt_store"] = interrupt_store
                if "interrupt_flag_key" in inspect.signature(execute_subgraph_locally).parameters:
                    execute_subgraph_kwargs["interrupt_flag_key"] = interrupt_flag_key
                outputs = execute_subgraph_locally(
                    payload,
                    kwargs_payload,
                    **execute_subgraph_kwargs,
                )
            else:
                execute_node_kwargs: dict[str, Any] = {}
                if "cancellation_event" in inspect.signature(execute_node_locally).parameters:
                    execute_node_kwargs["cancellation_event"] = cancellation_event
                if "interrupt_store" in inspect.signature(execute_node_locally).parameters:
                    execute_node_kwargs["interrupt_store"] = interrupt_store
                if "interrupt_flag_key" in inspect.signature(execute_node_locally).parameters:
                    execute_node_kwargs["interrupt_flag_key"] = interrupt_flag_key
                outputs = execute_node_locally(payload, kwargs_payload, **execute_node_kwargs)
        except Exception as exc:  # pragma: no cover - exercised through generator consumer tests.
            event_queue.put(("error", exc))
        else:
            event_queue.put(("result", outputs))
        finally:
            event_queue.put(("done", None))

    worker_thread = threading.Thread(
        target=execute_payload,
        name=f"modal-stream-{payload.get('component_id', 'payload')}",
        daemon=True,
    )
    worker_thread.start()
    try:
        while True:
            event_kind, event_payload = event_queue.get()
            if event_kind == "progress":
                yield {"kind": "progress", **event_payload}
                continue
            if event_kind == "result":
                yield {"kind": "result", "outputs": coerce_serialized_node_outputs(event_payload)}
                continue
            if event_kind == "error":
                raise event_payload
            if event_kind == "done":
                return
    finally:
        worker_thread.join(timeout=1.0)


def _should_ignore_repo_path(path: Path) -> bool:
    """Return whether a local repo path should be omitted from the Modal image mount."""
    parts = set(path.parts)
    if {".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"} & parts:
        return True
    return path.suffix.lower() in {".log", ".pyc", ".pyo", ".swp", ".tmp"}


def _should_ignore_comfyui_path(path: Path) -> bool:
    """Return whether a local ComfyUI path should be omitted from the Modal image mount."""
    parts = path.parts
    if not parts:
        return False

    if {".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"} & set(parts):
        return True

    if parts[0] in {"custom_nodes", "input", "models", "output", "temp", "user"}:
        return True

    return path.suffix.lower() in {".bin", ".ckpt", ".log", ".pt", ".pyc", ".pyo", ".safetensors", ".swp", ".tmp"}


def _comfyui_runtime_packages() -> tuple[str, ...]:
    """Return the Python packages needed to import and execute ComfyUI core inside Modal."""
    return (
        "aiohttp",
        "alembic",
        "av",
        "comfy-kitchen>=0.2.7",
        "einops",
        "kornia",
        "numpy",
        "opencv-python-headless",
        "packaging",
        "pillow",
        "psutil",
        "pydantic",
        "pydantic-settings",
        "pyyaml",
        "requests",
        "safetensors",
        "scipy",
        "sentencepiece",
        "spandrel",
        "sqlalchemy",
        "torchsde",
        "tqdm",
        "transformers",
    )


def _comfyui_torch_packages() -> tuple[str, ...]:
    """Return the pinned CUDA 12.8 PyTorch stack used by the remote Modal image."""
    return (
        f"torch=={_COMFYUI_TORCH_VERSION}",
        f"torchvision=={_COMFYUI_TORCHVISION_VERSION}",
        f"torchaudio=={_COMFYUI_TORCHAUDIO_VERSION}",
    )


def _prewarm_snapshot_state(settings: Any) -> None:
    """Run snapshot-safe initialization before Modal captures a memory snapshot."""
    with _timed_phase("prewarm_snapshot_state", gpu_snapshot=settings.enable_gpu_memory_snapshot):
        _ensure_comfyui_support_packages()
        if settings.enable_gpu_memory_snapshot:
            _ensure_comfy_runtime_initialized(None)
            _load_execution_module()
            _emit_cloud_info("Completed GPU-snapshot ComfyUI prewarm before snapshot capture.")
            return

        _emit_cloud_info(
            "Skipping full ComfyUI runtime prewarm during CPU-only snapshot to avoid accidental CUDA initialization."
        )


def _prewarm_restored_runtime() -> None:
    """Run post-restore initialization that should be ready before serving requests."""
    with _timed_phase("prewarm_restored_runtime"):
        _ensure_comfy_runtime_initialized(None)
        _load_execution_module()


def _remote_engine_cls_options(settings: Any, vol: Any, image: Any) -> dict[str, Any]:
    """Build the Modal class options for the deployed remote execution runtime."""
    options: dict[str, Any] = {
        "gpu": settings.modal_gpu,
        "volumes": {settings.remote_storage_root: vol},
        "scaledown_window": settings.scaledown_window_seconds,
        "min_containers": settings.min_containers,
        "image": image,
        "enable_memory_snapshot": settings.enable_memory_snapshot,
    }
    max_containers = getattr(settings, "max_containers", None)
    buffer_containers = getattr(settings, "buffer_containers", None)
    if max_containers is not None:
        options["max_containers"] = max_containers
    if buffer_containers is not None:
        options["buffer_containers"] = buffer_containers
    if settings.enable_gpu_memory_snapshot:
        options["experimental_options"] = {"enable_gpu_snapshot": True}
    return options


def _should_reload_modal_volume(payload: dict[str, Any]) -> bool:
    """Return whether this request needs the mounted Modal volume reloaded."""
    if not bool(payload.get("requires_volume_reload", True)):
        return False
    reload_marker = _modal_volume_reload_marker(payload)
    if reload_marker is None:
        return True
    return not _has_seen_modal_volume_reload_marker(reload_marker)


def _modal_volume_reload_marker(payload: dict[str, Any]) -> str | None:
    """Return the per-request Modal volume reload marker attached to this payload."""
    marker = payload.get("volume_reload_marker")
    if marker is None:
        return None
    marker_text = str(marker).strip()
    return marker_text or None


def _has_seen_modal_volume_reload_marker(reload_marker: str) -> bool:
    """Return whether this container already reloaded the volume for this marker."""
    with _MODAL_VOLUME_RELOAD_MARKERS_LOCK:
        return reload_marker in _MODAL_VOLUME_RELOAD_MARKER_SET


def _record_modal_volume_reload_marker(reload_marker: str) -> None:
    """Remember that this container has already reloaded the volume for one marker."""
    global _MODAL_VOLUME_RELOAD_MARKERS

    with _MODAL_VOLUME_RELOAD_MARKERS_LOCK:
        if reload_marker in _MODAL_VOLUME_RELOAD_MARKER_SET:
            return
        if _MODAL_VOLUME_RELOAD_MARKERS is None:
            _MODAL_VOLUME_RELOAD_MARKERS = queue.SimpleQueue()
        _MODAL_VOLUME_RELOAD_MARKER_SET.add(reload_marker)
        _MODAL_VOLUME_RELOAD_MARKERS.put(reload_marker)
        while len(_MODAL_VOLUME_RELOAD_MARKER_SET) > _MODAL_VOLUME_RELOAD_MARKER_CACHE_LIMIT:
            expired_marker = _MODAL_VOLUME_RELOAD_MARKERS.get()
            _MODAL_VOLUME_RELOAD_MARKER_SET.discard(expired_marker)


def _clear_warm_remote_caches() -> None:
    """Drop warm-container caches that may retain references to mounted volume files."""
    with _PROMPT_EXECUTOR_STATES_LOCK:
        _PROMPT_EXECUTOR_STATES.clear()
    with _LOADER_CACHE_LOCK:
        _LOADER_OUTPUT_CACHE.clear()


def _prepare_for_modal_volume_reload() -> None:
    """Release warm runtime state so a Modal volume reload can proceed safely."""
    _clear_warm_remote_caches()
    try:
        import comfy.model_management as model_management
    except ModuleNotFoundError:
        gc.collect()
        return

    model_management.unload_all_models()
    model_management.cleanup_models()
    model_management.soft_empty_cache(True)
    gc.collect()


def _is_modal_volume_open_files_error(exc: RuntimeError) -> bool:
    """Return whether a Modal volume reload failed because mounted files are still open."""
    return "open files" in str(exc)


def _sleep_before_modal_volume_reload_retry(delay_seconds: float) -> None:
    """Pause briefly so recently cancelled work can release mounted-volume file handles."""
    if delay_seconds <= 0:
        return
    time.sleep(delay_seconds)


def _reload_modal_volume_for_request(
    volume: Any,
    component_id: str,
    reload_marker: str | None = None,
) -> None:
    """Reload the Modal volume, retrying briefly while warm state releases open files."""
    with _timed_phase("modal_volume_reload", component=component_id):
        for attempt_index, retry_delay_seconds in enumerate(
            _MODAL_VOLUME_RELOAD_OPEN_FILE_RETRY_DELAYS_SECONDS,
            start=1,
        ):
            if attempt_index > 1:
                _sleep_before_modal_volume_reload_retry(retry_delay_seconds)
            try:
                volume.reload()
                if reload_marker is not None:
                    _record_modal_volume_reload_marker(reload_marker)
                if attempt_index > 1:
                    _emit_cloud_info(
                        "Modal volume reload succeeded for component=%s after %d attempt(s).",
                        component_id,
                        attempt_index,
                    )
                return
            except RuntimeError as exc:
                if not _is_modal_volume_open_files_error(exc):
                    raise
                if attempt_index == len(_MODAL_VOLUME_RELOAD_OPEN_FILE_RETRY_DELAYS_SECONDS):
                    raise
                _emit_cloud_info(
                    "Modal volume reload hit open files for component=%s on attempt %d/%d; clearing warm caches and retrying after %.2fs.",
                    component_id,
                    attempt_index,
                    len(_MODAL_VOLUME_RELOAD_OPEN_FILE_RETRY_DELAYS_SECONDS),
                    _MODAL_VOLUME_RELOAD_OPEN_FILE_RETRY_DELAYS_SECONDS[attempt_index],
                )
                _prepare_for_modal_volume_reload()


def _emit_modal_volume_reload_skip(component_id: Any, payload: dict[str, Any]) -> None:
    """Log why a request did not need a Modal volume reload."""
    reload_marker = _modal_volume_reload_marker(payload)
    if reload_marker is not None and _has_seen_modal_volume_reload_marker(reload_marker):
        _emit_cloud_info(
            "Skipping modal_volume_reload for component=%s because this container already reloaded marker=%s.",
            component_id,
            reload_marker,
        )
        return
    _emit_cloud_info(
        "Skipping modal_volume_reload for component=%s because no new assets were uploaded for this request.",
        component_id,
    )


if modal is not None:  # pragma: no branch - remote entrypoint configuration.
    settings = get_settings()
    app = modal.App(settings.app_name)
    vol = modal.Volume.from_name(settings.volume_name, create_if_missing=True)
    interrupt_flags = modal.Dict.from_name(
        settings.interrupt_dict_name,
        create_if_missing=True,
    )
    image = (
        modal.Image.debian_slim()
        .pip_install(*_comfyui_runtime_packages())
        .pip_install(*_comfyui_torch_packages(), index_url=_PYTORCH_CUDA_INDEX_URL)
        .add_local_dir(
            _REPO_ROOT,
            remote_path="/root/comfyui_modal_sync_repo",
            ignore=_should_ignore_repo_path,
        )
    )
    if settings.comfyui_root is not None and settings.comfyui_root.exists():
        image = image.add_local_dir(
            settings.comfyui_root,
            remote_path=str(_REMOTE_COMFYUI_ROOT),
            ignore=_should_ignore_comfyui_path,
        )
        logger.info(
            "Including local ComfyUI checkout %s in Modal image at %s.",
            settings.comfyui_root,
            _REMOTE_COMFYUI_ROOT,
        )
    else:
        logger.warning(
            "No local ComfyUI checkout was discovered; remote Modal execution may fail to import ComfyUI core modules."
        )

    @app.cls(**_remote_engine_cls_options(settings, vol, image))
    @modal.concurrent(max_inputs=1)
    class RemoteEngine:
        """Modal runtime class that executes proxied ComfyUI payloads."""

        @modal.enter(snap=True)
        def setup_snapshot_state(self) -> None:
            """Prepare snapshot-friendly runtime state before Modal captures memory."""
            with _timed_phase("remote_engine_setup_snapshot"):
                _prewarm_snapshot_state(settings)
                logger.info("RemoteEngine snapshot setup complete.")

        @modal.enter(snap=False)
        def setup_restored_runtime(self) -> None:
            """Prepare request-serving runtime state after a fresh boot or snapshot restore."""
            with _timed_phase("remote_engine_setup_restored"):
                _prewarm_restored_runtime()
                logger.info("RemoteEngine restored-runtime setup complete.")

        @modal.method()
        def execute_payload(self, payload: dict[str, Any], kwargs_payload: bytes) -> bytes:
            """Execute a proxied node or subgraph inside the Modal container."""
            component_id = payload.get("component_id", "single-node")
            reload_marker = _modal_volume_reload_marker(payload)
            with _registered_remote_execution(payload) as execution_control:
                with _timed_phase(
                    "remote_engine_execute_payload",
                    component=component_id,
                    payload_kind=payload.get("payload_kind"),
                ):
                    if _should_reload_modal_volume(payload):
                        _reload_modal_volume_for_request(
                            vol,
                            str(component_id),
                            reload_marker=reload_marker,
                        )
                    else:
                        _emit_modal_volume_reload_skip(component_id, payload)
                    if payload.get("payload_kind") == "subgraph":
                        return execute_subgraph_locally(
                            payload,
                            kwargs_payload,
                            cancellation_event=execution_control.cancellation_event,
                            interrupt_store=interrupt_flags,
                            interrupt_flag_key=execution_control.interrupt_flag_key,
                        )
                    return execute_node_locally(
                        payload,
                        kwargs_payload,
                        cancellation_event=execution_control.cancellation_event,
                        interrupt_store=interrupt_flags,
                        interrupt_flag_key=execution_control.interrupt_flag_key,
                    )

        @modal.method()
        def execute_payload_stream(
            self,
            payload: dict[str, Any],
            kwargs_payload: bytes,
        ) -> Iterator[dict[str, Any]]:
            """Stream progress envelopes and a final serialized result for one payload."""
            component_id = payload.get("component_id", "single-node")
            reload_marker = _modal_volume_reload_marker(payload)
            with _registered_remote_execution(payload) as execution_control:
                with _timed_phase(
                    "remote_engine_execute_payload",
                    component=component_id,
                    payload_kind=payload.get("payload_kind"),
                ):
                    if _should_reload_modal_volume(payload):
                        _reload_modal_volume_for_request(
                            vol,
                            str(component_id),
                            reload_marker=reload_marker,
                        )
                    else:
                        _emit_modal_volume_reload_skip(component_id, payload)
                    yield from _stream_remote_payload_events(
                        payload,
                        kwargs_payload,
                        cancellation_event=execution_control.cancellation_event,
                        interrupt_store=interrupt_flags,
                        interrupt_flag_key=execution_control.interrupt_flag_key,
                    )

else:
    app = None
    RemoteEngine = None
