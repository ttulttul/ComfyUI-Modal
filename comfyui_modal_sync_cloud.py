"""Stable Modal cloud entrypoint for ComfyUI Modal-Sync."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import importlib.util
import json
import logging
import os
import sys
import tempfile
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

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
_COMFY_RUNTIME_INIT_LOCK = threading.Lock()
_COMFY_RUNTIME_BASE_INITIALIZED = False
_COMFY_RUNTIME_CUSTOM_NODE_ROOTS: set[str] = set()

try:
    import modal  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - remote entrypoint only.
    modal = None


class RemoteSubgraphExecutionError(RuntimeError):
    """Raised when remote subgraph execution fails."""


class _NullPromptServer:
    """Minimal PromptExecutor server stub for headless subgraph execution."""

    def __init__(self) -> None:
        """Initialize the no-op prompt server state."""
        self.client_id: str | None = None
        self.last_node_id: str | None = None

    def send_sync(self, event: str, data: dict[str, Any], client_id: str | None) -> None:
        """Discard PromptExecutor progress and status events."""
        logger.debug("Suppressed remote prompt event %s for client %s.", event, client_id)


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

    extraction_root = Path(tempfile.gettempdir()) / "comfy-modal-sync-custom-nodes"
    extraction_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(local_bundle, "r") as archive:
        archive.extractall(extraction_root)

    if str(extraction_root) not in sys.path:
        sys.path.insert(0, str(extraction_root))
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


def _materialize_remote_asset_path(value: str) -> str:
    """Resolve a mirrored Modal asset reference to the container-local absolute file path."""
    settings = get_settings()
    remote_storage_root = settings.remote_storage_root.rstrip("/")
    if value.startswith(f"{remote_storage_root}/"):
        return value
    if value.startswith("/assets/"):
        return f"{remote_storage_root}{value}"
    return value


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
        _ensure_comfyui_support_packages()
        nodes_module = _load_nodes_module()

        if not _COMFY_RUNTIME_BASE_INITIALIZED:
            if custom_nodes_root is not None:
                _register_custom_nodes_root(custom_nodes_root)
            logger.info(
                "Initializing remote ComfyUI node registry with built-in extras%s.",
                " and extracted custom nodes" if custom_nodes_root is not None else "",
            )
            asyncio.run(
                nodes_module.init_extra_nodes(
                    init_custom_nodes=custom_nodes_root is not None,
                    init_api_nodes=True,
                )
            )
            _COMFY_RUNTIME_BASE_INITIALIZED = True
            if custom_nodes_root_key is not None:
                _COMFY_RUNTIME_CUSTOM_NODE_ROOTS.add(custom_nodes_root_key)
            return

        if custom_nodes_root_key is None or custom_nodes_root_key in _COMFY_RUNTIME_CUSTOM_NODE_ROOTS:
            return

        _register_custom_nodes_root(custom_nodes_root)
        logger.info("Loading extracted remote custom nodes from %s.", custom_nodes_root)
        asyncio.run(nodes_module.init_external_custom_nodes())
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
) -> tuple[Any, ...]:
    """Execute a remote component prompt and return its exported outputs."""
    prompt = _rewrite_modal_asset_references(copy.deepcopy(payload["subgraph_prompt"]))
    _apply_boundary_inputs(
        prompt=prompt,
        boundary_input_specs=list(payload.get("boundary_inputs", [])),
        hydrated_inputs=hydrated_inputs,
    )
    execution = _load_execution_module()
    cache_type, cache_args = _prompt_executor_cache_config(execution)

    with _temporary_node_mapping(None), _patched_folder_paths_absolute_lookup():
        executor = execution.PromptExecutor(
            _NullPromptServer(),
            cache_type=cache_type,
            cache_args=cache_args,
        )
        executor.execute(
            prompt=prompt,
            prompt_id=str(payload.get("component_id", "modal-subgraph")),
            extra_data=copy.deepcopy(payload.get("extra_data") or {}),
            execute_outputs=list(payload.get("execute_node_ids", [])),
        )
        if not executor.success:
            raise RemoteSubgraphExecutionError(_extract_prompt_executor_error(executor))

        outputs: list[Any] = []
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
) -> bytes:
    """Execute a rewritten remote component in-process and return serialized outputs."""
    custom_nodes_root = _extract_custom_nodes_bundle(payload.get("custom_nodes_bundle"))
    _ensure_comfy_runtime_initialized(custom_nodes_root)
    hydrated_inputs = deserialize_node_inputs(kwargs_payload)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_execute_subgraph_prompt, payload, hydrated_inputs)
        outputs = future.result()
    return serialize_node_outputs(outputs)


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

    if parts[0] in {"input", "models", "output", "temp", "user"}:
        return True

    return path.suffix.lower() in {".bin", ".ckpt", ".log", ".pt", ".pyc", ".pyo", ".safetensors", ".swp", ".tmp"}


def _comfyui_runtime_packages() -> tuple[str, ...]:
    """Return the Python packages needed to import and execute ComfyUI core inside Modal."""
    return (
        "aiohttp",
        "av",
        "einops",
        "numpy",
        "opencv-python-headless",
        "packaging",
        "pillow",
        "psutil",
        "pydantic",
        "pyyaml",
        "requests",
        "safetensors",
        "scipy",
        "sentencepiece",
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


def _remote_engine_cls_options(settings: Any, vol: Any, image: Any) -> dict[str, Any]:
    """Build the Modal class options for the deployed remote execution runtime."""
    options: dict[str, Any] = {
        "gpu": "A100",
        "volumes": {settings.remote_storage_root: vol},
        "scaledown_window": 60,
        "image": image,
        "enable_memory_snapshot": settings.enable_memory_snapshot,
    }
    if settings.enable_gpu_memory_snapshot:
        options["experimental_options"] = {"enable_gpu_snapshot": True}
    return options


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

        @modal.enter()
        def setup(self) -> None:
            """Prepare the container process for headless node execution."""
            logger.info("RemoteEngine setup complete.")

        @modal.method()
        def execute_payload(self, payload: dict[str, Any], kwargs_payload: bytes) -> bytes:
            """Execute a proxied node or subgraph inside the Modal container."""
            vol.reload()
            if payload.get("payload_kind") == "subgraph":
                return execute_subgraph_locally(payload, kwargs_payload)
            return execute_node_locally(payload, kwargs_payload)

else:
    app = None
    RemoteEngine = None
