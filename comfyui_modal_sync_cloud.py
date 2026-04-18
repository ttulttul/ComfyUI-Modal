"""Stable Modal cloud entrypoint for ComfyUI Modal-Sync."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import importlib.util
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

from serialization import deserialize_node_inputs, serialize_node_outputs
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


class _NullPromptServer:
    """Minimal PromptExecutor server stub for headless subgraph execution."""

    def __init__(self) -> None:
        """Initialize the no-op prompt server state."""
        self.client_id: str | None = None
        self.last_node_id: str | None = None

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

    def send_sync(self, event: str, data: dict[str, Any], client_id: str | None) -> None:
        """Track per-node timing transitions from PromptExecutor progress events."""
        if event == "executing":
            next_node_id = data.get("node")
            if next_node_id != self._active_node_id:
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
                _emit_cloud_info(
                    "Remote node %s class_type=%s role=%s started",
                    self._active_node_id,
                    class_type,
                    role,
                )
            return

        if event == "executed":
            executed_node_id = data.get("node")
            if executed_node_id is not None and str(executed_node_id) == self._active_node_id:
                self._log_node_finish(reason="executed")
            return

        if event in {"execution_error", "execution_interrupted", "execution_success"}:
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
) -> bytes:
    """Execute a single target node in-process and return serialized outputs."""
    custom_nodes_root = _extract_custom_nodes_bundle(node_data.get("custom_nodes_bundle"))
    _ensure_comfy_runtime_initialized(custom_nodes_root)
    kwargs = _rewrite_modal_asset_references(deserialize_node_inputs(kwargs_payload))
    if node_mapping is not None:
        class_type = node_data["class_type"]
        if class_type not in node_mapping:
            raise KeyError(f"Remote node class {class_type!r} is not registered.")
        with _patched_folder_paths_absolute_lookup():
            outputs = _invoke_original_node(node_mapping[class_type], node_data, kwargs)
            return serialize_node_outputs(outputs)

    with _temporary_node_mapping(node_mapping):
        resolved_node_mapping = _load_nodes_module().NODE_CLASS_MAPPINGS
        class_type = node_data["class_type"]
        if class_type not in resolved_node_mapping:
            raise KeyError(f"Remote node class {class_type!r} is not registered.")

        with _patched_folder_paths_absolute_lookup():
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


def _execute_subgraph_prompt(
    payload: dict[str, Any],
    hydrated_inputs: dict[str, Any],
    custom_nodes_root: Path | None,
    status_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[Any, ...]:
    """Execute a remote component prompt and return its exported outputs."""
    component_id = str(payload.get("component_id", "modal-subgraph"))
    with _timed_phase("prepare_subgraph_prompt", component=component_id):
        prompt = _rewrite_modal_asset_references(copy.deepcopy(payload["subgraph_prompt"]))
        _apply_boundary_inputs(
            prompt=prompt,
            boundary_input_specs=list(payload.get("boundary_inputs", [])),
            hydrated_inputs=hydrated_inputs,
        )
    with _timed_phase("load_execution_module", component=component_id):
        execution = _load_execution_module()
        cache_type, cache_args = _prompt_executor_cache_config(execution)

    with _temporary_node_mapping(None), _patched_folder_paths_absolute_lookup():
        prompt_server = _TracingPromptServer(
            component_id,
            prompt,
            status_callback=status_callback,
        )
        with _timed_phase("create_prompt_executor", component=component_id):
            executor_state = _get_or_create_prompt_executor_state(
                execution=execution,
                prompt_server=prompt_server,
                cache_type=cache_type,
                cache_args=cache_args,
                custom_nodes_root=custom_nodes_root,
            )
        with executor_state.lock:
            _reset_prompt_executor_request_state(executor_state.executor, prompt_server)
            with _timed_phase(
                "prompt_executor_execute",
                component=component_id,
                execute_nodes=list(payload.get("execute_node_ids", [])),
            ):
                executor_state.executor.execute(
                    prompt=prompt,
                    prompt_id=component_id,
                    extra_data=copy.deepcopy(payload.get("extra_data") or {}),
                    execute_outputs=list(payload.get("execute_node_ids", [])),
                )
            executor = executor_state.executor
        if not executor.success:
            raise RemoteSubgraphExecutionError(_extract_prompt_executor_error(executor))

        outputs: list[Any] = []
        with _timed_phase(
            "collect_boundary_outputs",
            component=component_id,
            output_count=len(payload.get("boundary_outputs", [])),
        ):
            for boundary_output in payload.get("boundary_outputs", []):
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
                )
                outputs = future.result()
        with _timed_phase("serialize_boundary_outputs", component=component_id):
            return serialize_node_outputs(outputs)


def _stream_remote_payload_events(
    payload: dict[str, Any],
    kwargs_payload: bytes | bytearray | str | dict[str, Any],
) -> Iterator[dict[str, Any]]:
    """Yield progress and result events for one remote payload execution."""
    event_queue: queue.Queue[tuple[str, Any]] = queue.Queue()

    def publish_status(progress_state: dict[str, Any]) -> None:
        """Queue a progress envelope for the remote caller."""
        event_queue.put(("progress", dict(progress_state)))

    def execute_payload() -> None:
        """Run the payload in a worker thread and enqueue the terminal outcome."""
        try:
            if payload.get("payload_kind") == "subgraph":
                outputs = execute_subgraph_locally(
                    payload,
                    kwargs_payload,
                    status_callback=publish_status,
                )
            else:
                outputs = execute_node_locally(payload, kwargs_payload)
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
                yield {"kind": "result", "outputs": event_payload}
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
    if settings.enable_gpu_memory_snapshot:
        options["experimental_options"] = {"enable_gpu_snapshot": True}
    return options


def _should_reload_modal_volume(payload: dict[str, Any]) -> bool:
    """Return whether this request needs the mounted Modal volume reloaded."""
    return bool(payload.get("requires_volume_reload", True))


if modal is not None:  # pragma: no branch - remote entrypoint configuration.
    settings = get_settings()
    app = modal.App(settings.app_name)
    vol = modal.Volume.from_name(settings.volume_name, create_if_missing=True)
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
            with _timed_phase(
                "remote_engine_execute_payload",
                component=component_id,
                payload_kind=payload.get("payload_kind"),
            ):
                if _should_reload_modal_volume(payload):
                    with _timed_phase("modal_volume_reload", component=component_id):
                        vol.reload()
                else:
                    _emit_cloud_info(
                        "Skipping modal_volume_reload for component=%s because no new assets were uploaded for this request.",
                        component_id,
                    )
                if payload.get("payload_kind") == "subgraph":
                    return execute_subgraph_locally(payload, kwargs_payload)
                return execute_node_locally(payload, kwargs_payload)

        @modal.method()
        def execute_payload_stream(
            self,
            payload: dict[str, Any],
            kwargs_payload: bytes,
        ) -> Iterator[dict[str, Any]]:
            """Stream progress envelopes and a final serialized result for one payload."""
            component_id = payload.get("component_id", "single-node")
            with _timed_phase(
                "remote_engine_execute_payload",
                component=component_id,
                payload_kind=payload.get("payload_kind"),
            ):
                if _should_reload_modal_volume(payload):
                    with _timed_phase("modal_volume_reload", component=component_id):
                        vol.reload()
                else:
                    _emit_cloud_info(
                        "Skipping modal_volume_reload for component=%s because no new assets were uploaded for this request.",
                        component_id,
                    )
                yield from _stream_remote_payload_events(payload, kwargs_payload)

else:
    app = None
    RemoteEngine = None
