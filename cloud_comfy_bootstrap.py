"""ComfyUI runtime bootstrap, custom-node loading, and warm loader caches."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
import importlib
import inspect
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
from typing import Any, Callable, ContextManager, Iterable, Iterator, Mapping
import zipfile

try:
    from .cloud_prompt_server_shims import _HeadlessPromptServerInstance
    from .settings import get_settings
except ImportError:  # pragma: no cover - flat Modal-container import.
    from cloud_prompt_server_shims import _HeadlessPromptServerInstance
    from settings import get_settings

logger = logging.getLogger(__name__)

_REMOTE_COMFYUI_ROOT = Path("/root/comfyui_src")
_LOCAL_COMFYUI_ROOT = (Path.home() / "git" / "ComfyUI").resolve()
_REMOTE_VOLUME_READTHROUGH_ROOT = (
    Path(tempfile.gettempdir()) / "comfy-modal-volume-readthrough"
)
_COMFY_RUNTIME_INIT_LOCK = threading.Lock()
_COMFY_RUNTIME_BASE_INITIALIZED = False
_COMFY_RUNTIME_CUSTOM_NODE_ROOTS: set[str] = set()
_EXTRACTED_CUSTOM_NODE_BUNDLES: dict[str, Path] = {}
_FOLDER_PATHS_PATCH_LOCK = threading.Lock()
_FOLDER_PATHS_PATCH_DEPTH = 0
_FOLDER_PATHS_ORIGINAL_GET_FULL_PATH: Callable[[str, str], str | None] | None = None
_FOLDER_PATHS_ORIGINAL_GET_FULL_PATH_OR_RAISE: Callable[[str, str], str] | None = None
_LOADER_CACHE_LOCK = threading.Lock()
_LOADER_CACHE_WRAPPED_CLASSES: set[str] = set()
_MODEL_STATE_DICT_COMPAT_WRAPPED = False
_LOADER_OUTPUT_CACHE: dict[tuple[str, str], tuple[Any, ...]] = {}
_LOADER_CACHE_METRICS_LOCK = threading.Lock()
_LOADER_CACHE_METRICS: dict[str, int] = {"hit": 0, "miss": 0}


@dataclass(frozen=True)
class CloudComfyBootstrapHooks:
    """Callbacks and stable errors supplied by the cloud entrypoint."""

    emit_cloud_info: Callable[..., None]
    timed_phase: Callable[..., ContextManager[None]]
    remote_subgraph_error: type[RuntimeError]


_BOOTSTRAP_HOOKS: CloudComfyBootstrapHooks | None = None


def configure_cloud_comfy_bootstrap_hooks(hooks: CloudComfyBootstrapHooks) -> None:
    """Install bootstrap callbacks without importing upward into the entrypoint."""
    global _BOOTSTRAP_HOOKS
    _BOOTSTRAP_HOOKS = hooks


def _bootstrap_hooks() -> CloudComfyBootstrapHooks:
    """Return configured callbacks or fail with a clear import-order error."""
    if _BOOTSTRAP_HOOKS is None:
        raise RuntimeError("Cloud ComfyUI bootstrap hooks have not been configured.")
    return _BOOTSTRAP_HOOKS


def _emit_cloud_info(message: str, *args: Any) -> None:
    """Delegate timestamped cloud logging to the stable entrypoint."""
    _bootstrap_hooks().emit_cloud_info(message, *args)


def _timed_phase(phase: str, **fields: Any) -> ContextManager[None]:
    """Delegate phase timing to the stable entrypoint."""
    return _bootstrap_hooks().timed_phase(phase, **fields)


def clear_warm_caches() -> None:
    """Clear warm loader outputs owned by this module."""
    with _LOADER_CACHE_LOCK:
        _LOADER_OUTPUT_CACHE.clear()


def _record_loader_cache_metric(result: str) -> None:
    """Increment one warm-worker loader-cache metric counter."""
    with _LOADER_CACHE_METRICS_LOCK:
        _LOADER_CACHE_METRICS[result] = _LOADER_CACHE_METRICS.get(result, 0) + 1


def _loader_cache_metric_snapshot() -> dict[str, int]:
    """Return the current cumulative loader-cache metrics."""
    with _LOADER_CACHE_METRICS_LOCK:
        return dict(_LOADER_CACHE_METRICS)


def _extract_custom_nodes_bundle(bundle_path: str | None) -> Path | None:
    """Extract a mirrored custom_nodes bundle ZIP or manifest into a temporary import path."""
    if not bundle_path:
        return None

    settings = get_settings()
    storage_roots = [Path(settings.remote_storage_root)]
    storage_roots.append(_REMOTE_VOLUME_READTHROUGH_ROOT)
    if settings.local_storage_root is not None:
        storage_roots.append(settings.local_storage_root)

    local_bundle: Path | None = None
    for storage_root in storage_roots:
        candidate = storage_root / bundle_path.lstrip("/")
        if candidate.exists():
            local_bundle = candidate
            break

    if local_bundle is None:
        logger.warning(
            "Custom nodes bundle %s was not found in any known storage root.",
            bundle_path,
        )
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

    extraction_root = (
        Path(tempfile.gettempdir())
        / "comfy-modal-sync-custom-nodes"
        / local_bundle.stem
    )
    if extraction_root.exists():
        shutil.rmtree(extraction_root)
    extraction_root.mkdir(parents=True, exist_ok=True)
    with _timed_phase("extract_custom_nodes_bundle", bundle=local_bundle.name):
        archives_to_extract = _resolve_custom_nodes_archives(
            local_bundle, storage_roots
        )
        for archive_path in archives_to_extract:
            with zipfile.ZipFile(archive_path, "r") as archive:
                archive.extractall(extraction_root)
        _materialize_custom_nodes_manifest_assets(
            local_bundle,
            storage_roots,
            extraction_root,
        )

    if str(extraction_root) not in sys.path:
        sys.path.insert(0, str(extraction_root))
    _EXTRACTED_CUSTOM_NODE_BUNDLES[local_bundle.name] = extraction_root
    logger.info("Extracted remote custom_nodes bundle to %s", extraction_root)
    return extraction_root


def _resolve_custom_nodes_archives(
    local_bundle: Path,
    storage_roots: list[Path],
) -> list[Path]:
    """Return the archive paths described by one custom_nodes bundle ZIP or manifest."""
    if local_bundle.suffix.lower() == ".zip":
        return [local_bundle]
    if local_bundle.suffix.lower() != ".json":
        raise RuntimeError(
            f"Unsupported custom_nodes bundle format {local_bundle.suffix!r} for {local_bundle}."
        )
    manifest_payload = _load_custom_nodes_manifest(local_bundle)
    entry_payloads = manifest_payload.get("entries")
    if not isinstance(entry_payloads, list):
        raise RuntimeError(
            f"Custom nodes manifest {local_bundle} did not contain a valid entries list."
        )

    archive_paths: list[Path] = []
    for entry_payload in entry_payloads:
        if not isinstance(entry_payload, dict):
            raise RuntimeError(
                f"Custom nodes manifest {local_bundle} contained a non-object entry."
            )
        remote_path = entry_payload.get("remote_path")
        if not isinstance(remote_path, str) or not remote_path.strip():
            raise RuntimeError(
                f"Custom nodes manifest {local_bundle} contained an entry without remote_path."
            )
        archive_path = _resolve_custom_nodes_bundle_path(remote_path, storage_roots)
        if archive_path is None:
            raise RuntimeError(
                f"Custom nodes archive {remote_path} referenced by {local_bundle} was not found in any storage root."
            )
        archive_paths.append(archive_path)
    return archive_paths


def _load_custom_nodes_manifest(local_bundle: Path) -> dict[str, Any]:
    """Load and validate one versioned custom-node bundle manifest."""
    try:
        manifest_payload = json.loads(local_bundle.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Custom nodes manifest {local_bundle} is unreadable."
        ) from exc
    if not isinstance(manifest_payload, dict):
        raise RuntimeError(
            f"Custom nodes manifest {local_bundle} must be a JSON object."
        )
    manifest_version = manifest_payload.get("version", 1)
    if manifest_version not in {1, 2}:
        raise RuntimeError(
            f"Custom nodes manifest {local_bundle} uses unsupported version {manifest_version!r}."
        )
    return manifest_payload


def _materialize_custom_nodes_manifest_assets(
    local_bundle: Path,
    storage_roots: list[Path],
    extraction_root: Path,
) -> None:
    """Link version-two package assets from mounted storage into extracted code."""
    if local_bundle.suffix.lower() != ".json":
        return
    manifest_payload = _load_custom_nodes_manifest(local_bundle)
    if manifest_payload.get("version", 1) < 2:
        return
    materialized_count = 0
    materialized_bytes = 0
    for asset_payload in _iter_custom_nodes_manifest_assets(
        local_bundle, manifest_payload
    ):
        relative_path = _validated_custom_node_asset_relative_path(
            local_bundle,
            asset_payload,
        )
        remote_path = str(asset_payload["remote_path"])
        asset_path = _resolve_custom_nodes_bundle_path(remote_path, storage_roots)
        if asset_path is None:
            raise RuntimeError(
                f"Custom-node asset {remote_path!r} referenced by {local_bundle} was not found."
            )
        expected_size = int(asset_payload["size_bytes"])
        if asset_path.stat().st_size != expected_size:
            raise RuntimeError(
                f"Custom-node asset {remote_path!r} size did not match its manifest."
            )
        destination = extraction_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            if (
                destination.is_symlink()
                and destination.resolve() == asset_path.resolve()
            ):
                continue
            raise RuntimeError(
                f"Custom-node asset destination {destination} already exists in the code archive."
            )
        destination.symlink_to(asset_path)
        materialized_count += 1
        materialized_bytes += expected_size
    if materialized_count:
        logger.info(
            "Linked %d custom-node model asset(s) totaling %d bytes from mounted storage.",
            materialized_count,
            materialized_bytes,
        )


def _iter_custom_nodes_manifest_assets(
    local_bundle: Path,
    manifest_payload: Mapping[str, Any],
) -> Iterator[dict[str, Any]]:
    """Yield validated asset objects from one version-two manifest."""
    entry_payloads = manifest_payload.get("entries")
    if not isinstance(entry_payloads, list):
        raise RuntimeError(
            f"Custom nodes manifest {local_bundle} has no valid entries list."
        )
    for entry_payload in entry_payloads:
        if not isinstance(entry_payload, dict):
            raise RuntimeError(
                f"Custom nodes manifest {local_bundle} contains an invalid entry."
            )
        asset_payloads = entry_payload.get("assets", [])
        if not isinstance(asset_payloads, list):
            raise RuntimeError(
                f"Custom nodes manifest {local_bundle} contains invalid assets."
            )
        for asset_payload in asset_payloads:
            if not isinstance(asset_payload, dict):
                raise RuntimeError(
                    f"Custom nodes manifest {local_bundle} contains an invalid asset."
                )
            yield asset_payload


def _validated_custom_node_asset_relative_path(
    local_bundle: Path,
    asset_payload: Mapping[str, Any],
) -> Path:
    """Return one safe extraction-relative custom-node asset path."""
    relative_path_value = asset_payload.get("relative_path")
    remote_path = asset_payload.get("remote_path")
    sha256 = asset_payload.get("sha256")
    size_bytes = asset_payload.get("size_bytes")
    if not isinstance(relative_path_value, str) or not relative_path_value:
        raise RuntimeError(
            f"Custom nodes manifest {local_bundle} contains an asset without a path."
        )
    if not isinstance(remote_path, str) or not remote_path:
        raise RuntimeError(
            f"Custom nodes manifest {local_bundle} contains an asset without storage."
        )
    if (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or not Path(remote_path).name.startswith(f"{sha256}_")
    ):
        raise RuntimeError(
            f"Custom nodes manifest {local_bundle} contains an invalid asset digest."
        )
    if (
        isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or size_bytes < 0
    ):
        raise RuntimeError(
            f"Custom nodes manifest {local_bundle} contains an invalid asset size."
        )
    relative_path = Path(relative_path_value)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise RuntimeError(
            f"Custom nodes manifest {local_bundle} contains unsafe asset path {relative_path_value!r}."
        )
    return relative_path


def _resolve_custom_nodes_bundle_path(
    bundle_path: str, storage_roots: list[Path]
) -> Path | None:
    """Resolve one custom_nodes bundle or archive path against the known storage roots."""
    for storage_root in storage_roots:
        candidate = storage_root / bundle_path.lstrip("/")
        if candidate.exists():
            return candidate
    return None


def _register_custom_nodes_root(custom_nodes_root: Path) -> None:
    """Expose an extracted custom_nodes directory to ComfyUI's folder path registry."""
    import folder_paths

    folder_paths.add_model_folder_path(
        "custom_nodes", str(custom_nodes_root), is_default=True
    )


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
        raise ImportError(
            f"Unable to create an import spec for package {module_name!r}."
        )

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

    node_replace_manager_class = getattr(server, "NodeReplaceManager", None)
    node_replace_manager = (
        node_replace_manager_class() if callable(node_replace_manager_class) else None
    )
    prompt_server_class.instance = _HeadlessPromptServerInstance(node_replace_manager)
    logger.info(
        "Installed headless PromptServer.instance for remote custom-node initialization."
    )


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
    if value.startswith("/"):
        volume_relative_roots = (
            "/assets/",
            "/custom_nodes/",
            "/hashes/",
            "/input/",
            "/models/",
            "/output/",
            "/temp/",
            "/user/",
        )
        if any(value.startswith(root) for root in volume_relative_roots):
            return f"{remote_storage_root}{value}"
    if value.startswith("/assets/"):
        return f"{remote_storage_root}{value}"
    return value


def _readthrough_cache_path(volume_path: Path) -> Path | None:
    """Return the safe ephemeral cache path for one mounted-volume path."""
    remote_storage_root = Path(get_settings().remote_storage_root).resolve()
    resolved_volume_path = volume_path.resolve()
    if not resolved_volume_path.is_relative_to(remote_storage_root):
        return None
    relative_path = resolved_volume_path.relative_to(remote_storage_root)
    return _REMOTE_VOLUME_READTHROUGH_ROOT / relative_path


def _resolve_runtime_asset_path(value: str) -> str:
    """Resolve an asset through the mount first and the committed read-through cache second."""
    materialized_path = Path(_materialize_remote_asset_path(value))
    if not materialized_path.is_absolute() or materialized_path.exists():
        return str(materialized_path)
    cache_path = _readthrough_cache_path(materialized_path)
    if cache_path is not None and cache_path.exists():
        return str(cache_path)
    return str(materialized_path)


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


def _build_dual_clip_loader_cache_key(kwargs: dict[str, Any]) -> str:
    """Build a stable cache key for the ComfyUI dual CLIP loader."""
    import folder_paths

    return _serialize_loader_cache_key(
        {
            "clip_path_1": folder_paths.get_full_path_or_raise(
                "text_encoders",
                str(kwargs["clip_name1"]),
            ),
            "clip_path_2": folder_paths.get_full_path_or_raise(
                "text_encoders",
                str(kwargs["clip_name2"]),
            ),
            "type": kwargs.get("type"),
            "device": kwargs.get("device", "default"),
        }
    )


def _build_vae_loader_cache_key(kwargs: dict[str, Any]) -> str:
    """Build a stable cache key for the ComfyUI VAE loader."""
    return _serialize_loader_cache_key({"vae_name": kwargs.get("vae_name")})


def _build_checkpoint_loader_cache_key(kwargs: dict[str, Any]) -> str:
    """Build a stable cache key for checkpoint-style model loaders."""
    import folder_paths

    key_parts: dict[str, Any] = {}
    if "config_name" in kwargs:
        key_parts["config_path"] = folder_paths.get_full_path(
            "configs", str(kwargs["config_name"])
        )
    if "ckpt_name" in kwargs:
        key_parts["ckpt_path"] = folder_paths.get_full_path_or_raise(
            "checkpoints",
            str(kwargs["ckpt_name"]),
        )
    if "model_path" in kwargs:
        key_parts["model_path"] = str(kwargs["model_path"])
    return _serialize_loader_cache_key(key_parts)


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
        normalized_kwargs = {
            key: value for key, value in bound.arguments.items() if key != "self"
        }
        cache_key = (class_type, cache_key_builder(normalized_kwargs))

        with _LOADER_CACHE_LOCK:
            cached_outputs = _LOADER_OUTPUT_CACHE.get(cache_key)
        if cached_outputs is not None:
            _record_loader_cache_metric("hit")
            _emit_cloud_info(
                "Loader cache hit class_type=%s key=%s", class_type, cache_key[1]
            )
            return _clone_loader_cache_outputs(cached_outputs)

        _record_loader_cache_metric("miss")
        _emit_cloud_info(
            "Loader cache miss class_type=%s key=%s", class_type, cache_key[1]
        )
        outputs = original_method(self, *args, **kwargs)
        normalized_outputs = (
            tuple(outputs) if isinstance(outputs, (list, tuple)) else (outputs,)
        )
        with _LOADER_CACHE_LOCK:
            _LOADER_OUTPUT_CACHE[cache_key] = normalized_outputs
        return _clone_loader_cache_outputs(normalized_outputs)

    setattr(node_class, method_name, cached_method)
    _LOADER_CACHE_WRAPPED_CLASSES.add(class_type)


def _install_loader_cache_wrappers() -> None:
    """Patch the heavyweight built-in model loaders to reuse warm-container state."""
    nodes_module = _load_nodes_module()
    cacheable_loader_specs = {
        "CheckpointLoader": ("load_checkpoint", _build_checkpoint_loader_cache_key),
        "CheckpointLoaderSimple": (
            "load_checkpoint",
            _build_checkpoint_loader_cache_key,
        ),
        "UNETLoader": ("load_unet", _build_unet_loader_cache_key),
        "CLIPLoader": ("load_clip", _build_clip_loader_cache_key),
        "DualCLIPLoader": ("load_clip", _build_dual_clip_loader_cache_key),
        "VAELoader": ("load_vae", _build_vae_loader_cache_key),
        "unCLIPCheckpointLoader": (
            "load_checkpoint",
            _build_checkpoint_loader_cache_key,
        ),
        "ImageOnlyCheckpointLoader": (
            "load_checkpoint",
            _build_checkpoint_loader_cache_key,
        ),
    }

    for class_type, (method_name, cache_key_builder) in cacheable_loader_specs.items():
        node_class = nodes_module.NODE_CLASS_MAPPINGS.get(class_type)
        if node_class is None:
            continue
        _wrap_loader_method_with_cache(
            class_type, node_class, method_name, cache_key_builder
        )


def _alias_flux_rms_norm_weight_keys(state_dict: dict[str, Any]) -> int:
    """Add ComfyUI Flux RMSNorm `.scale` aliases for saved files that use `.weight`."""
    alias_count = 0
    replacements = {
        ".norm.key_norm.weight": ".norm.key_norm.scale",
        ".norm.query_norm.weight": ".norm.query_norm.scale",
    }
    for key, value in list(state_dict.items()):
        for source_suffix, target_suffix in replacements.items():
            if not key.endswith(source_suffix):
                continue
            target_key = f"{key[: -len(source_suffix)]}{target_suffix}"
            if target_key in state_dict:
                continue
            state_dict[target_key] = value
            alias_count += 1
    return alias_count


def _install_model_state_dict_compatibility_wrappers() -> None:
    """Patch ComfyUI model loading for known cross-version saved-model key aliases."""
    global _MODEL_STATE_DICT_COMPAT_WRAPPED

    if _MODEL_STATE_DICT_COMPAT_WRAPPED:
        return

    import comfy.sd

    original_load_diffusion_model_state_dict = comfy.sd.load_diffusion_model_state_dict

    def compatible_load_diffusion_model_state_dict(
        sd: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Load diffusion models after adding aliases while preserving ComfyUI's API."""
        alias_count = _alias_flux_rms_norm_weight_keys(sd)
        if alias_count:
            logger.info(
                "Added %d Flux RMSNorm .scale aliases for a diffusion model state_dict saved with .weight keys.",
                alias_count,
            )
        return original_load_diffusion_model_state_dict(sd, *args, **kwargs)

    comfy.sd.load_diffusion_model_state_dict = (
        compatible_load_diffusion_model_state_dict
    )
    _MODEL_STATE_DICT_COMPAT_WRAPPED = True


def _rewrite_modal_asset_references(value: Any) -> Any:
    """Recursively replace mirrored asset markers with container-local absolute file paths."""
    if isinstance(value, str):
        return _materialize_remote_asset_path(value)
    if isinstance(value, list):
        return [_rewrite_modal_asset_references(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _rewrite_modal_asset_references(item)
            for key, item in value.items()
        }
    return value


@contextmanager
def _patched_folder_paths_absolute_lookup() -> Iterator[None]:
    """Teach ComfyUI folder lookups to accept absolute assets across overlapping callers."""
    import folder_paths

    global _FOLDER_PATHS_PATCH_DEPTH
    global _FOLDER_PATHS_ORIGINAL_GET_FULL_PATH
    global _FOLDER_PATHS_ORIGINAL_GET_FULL_PATH_OR_RAISE

    with _FOLDER_PATHS_PATCH_LOCK:
        if _FOLDER_PATHS_PATCH_DEPTH == 0:
            original_get_full_path = folder_paths.get_full_path
            original_get_full_path_or_raise = folder_paths.get_full_path_or_raise
            _FOLDER_PATHS_ORIGINAL_GET_FULL_PATH = original_get_full_path
            _FOLDER_PATHS_ORIGINAL_GET_FULL_PATH_OR_RAISE = (
                original_get_full_path_or_raise
            )

            def patched_get_full_path(folder_name: str, filename: str) -> str | None:
                """Return an absolute file or delegate to ComfyUI's original lookup."""
                resolved_filename = _resolve_runtime_asset_path(filename)
                if os.path.isabs(resolved_filename) and Path(resolved_filename).is_file():
                    return resolved_filename
                return original_get_full_path(folder_name, resolved_filename)

            def patched_get_full_path_or_raise(folder_name: str, filename: str) -> str:
                """Raise with the original message when no absolute or folder match exists."""
                full_path = patched_get_full_path(folder_name, filename)
                if full_path is None:
                    raise FileNotFoundError(
                        f"Model in folder '{folder_name}' with filename '{filename}' not found."
                    )
                return full_path

            folder_paths.get_full_path = patched_get_full_path
            folder_paths.get_full_path_or_raise = patched_get_full_path_or_raise
        _FOLDER_PATHS_PATCH_DEPTH += 1
        logger.debug(
            "Entered absolute folder-path lookup patch depth=%d.",
            _FOLDER_PATHS_PATCH_DEPTH,
        )
    try:
        yield
    finally:
        with _FOLDER_PATHS_PATCH_LOCK:
            _FOLDER_PATHS_PATCH_DEPTH -= 1
            logger.debug(
                "Exited absolute folder-path lookup patch depth=%d.",
                _FOLDER_PATHS_PATCH_DEPTH,
            )
            if _FOLDER_PATHS_PATCH_DEPTH == 0:
                original_get_full_path = _FOLDER_PATHS_ORIGINAL_GET_FULL_PATH
                original_get_full_path_or_raise = (
                    _FOLDER_PATHS_ORIGINAL_GET_FULL_PATH_OR_RAISE
                )
                if original_get_full_path is None or original_get_full_path_or_raise is None:
                    raise RuntimeError("Absolute folder-path lookup patch lost its originals.")
                folder_paths.get_full_path = original_get_full_path
                folder_paths.get_full_path_or_raise = original_get_full_path_or_raise
                _FOLDER_PATHS_ORIGINAL_GET_FULL_PATH = None
                _FOLDER_PATHS_ORIGINAL_GET_FULL_PATH_OR_RAISE = None


def _ensure_comfy_runtime_initialized(custom_nodes_root: Path | None) -> None:
    """Initialize ComfyUI's built-in and external node registries for remote execution."""
    global _COMFY_RUNTIME_BASE_INITIALIZED

    custom_nodes_root_key = (
        str(custom_nodes_root.resolve()) if custom_nodes_root is not None else None
    )
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
                    " and extracted custom nodes"
                    if custom_nodes_root is not None
                    else "",
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
                _install_model_state_dict_compatibility_wrappers()
                _install_loader_cache_wrappers()
                _register_modal_sync_runtime_nodes(nodes_module)
                _COMFY_RUNTIME_BASE_INITIALIZED = True
                if custom_nodes_root_key is not None:
                    _COMFY_RUNTIME_CUSTOM_NODE_ROOTS.add(custom_nodes_root_key)
                return

            if (
                custom_nodes_root_key is None
                or custom_nodes_root_key in _COMFY_RUNTIME_CUSTOM_NODE_ROOTS
            ):
                logger.info(
                    "Reusing initialized remote ComfyUI runtime for custom_nodes=%s without re-running custom node import.",
                    custom_nodes_root_key or "<default>",
                )
                _install_model_state_dict_compatibility_wrappers()
                _install_loader_cache_wrappers()
                _register_modal_sync_runtime_nodes(nodes_module)
                return

            _register_custom_nodes_root(custom_nodes_root)
            logger.info(
                "Loading extracted remote custom nodes from %s.", custom_nodes_root
            )
            with _timed_phase(
                "init_external_custom_nodes", custom_nodes=custom_nodes_root_key
            ):
                asyncio.run(nodes_module.init_external_custom_nodes())
            _install_model_state_dict_compatibility_wrappers()
            _install_loader_cache_wrappers()
            _register_modal_sync_runtime_nodes(nodes_module)
            _COMFY_RUNTIME_CUSTOM_NODE_ROOTS.add(custom_nodes_root_key)


def _register_modal_sync_runtime_nodes(nodes_module: Any) -> None:
    """Register nodes shipped in the deployment image without custom-node sync."""
    from llm_profiles import MODAL_LLM_NODE_ID
    from modal_llm_node import ModalLLM

    existing_node = nodes_module.NODE_CLASS_MAPPINGS.get(MODAL_LLM_NODE_ID)
    if existing_node is not None and existing_node is not ModalLLM:
        logger.info(
            "Preserving custom-node registration for %s instead of replacing it.",
            MODAL_LLM_NODE_ID,
        )
        return
    nodes_module.NODE_CLASS_MAPPINGS[MODAL_LLM_NODE_ID] = ModalLLM
    display_mappings = getattr(nodes_module, "NODE_DISPLAY_NAME_MAPPINGS", None)
    if isinstance(display_mappings, dict):
        display_mappings[MODAL_LLM_NODE_ID] = "Modal LLM"
    logger.info("Registered deployment-owned remote node %s.", MODAL_LLM_NODE_ID)


def _prompt_missing_node_class_types(
    prompt: Mapping[str, Any],
    node_mapping: Mapping[str, type[Any]],
) -> list[str]:
    """Return sorted prompt class types that are absent from the active node registry."""
    missing_class_types: set[str] = set()
    for prompt_node in prompt.values():
        if not isinstance(prompt_node, Mapping):
            continue
        class_type = prompt_node.get("class_type")
        if isinstance(class_type, str) and class_type not in node_mapping:
            missing_class_types.add(class_type)
    return sorted(missing_class_types)


def _reload_external_custom_nodes_for_missing_classes(
    custom_nodes_root: Path | None,
) -> None:
    """Re-run external custom-node import once when a prompt references missing node classes."""
    if custom_nodes_root is None:
        return

    custom_nodes_root_key = str(custom_nodes_root.resolve())
    with _COMFY_RUNTIME_INIT_LOCK:
        _register_custom_nodes_root(custom_nodes_root)
        logger.warning(
            "Re-running external custom-node import for %s because the active prompt referenced unregistered node classes.",
            custom_nodes_root_key,
        )
        nodes_module = _load_nodes_module()
        with _timed_phase(
            "init_external_custom_nodes_retry", custom_nodes=custom_nodes_root_key
        ):
            asyncio.run(nodes_module.init_external_custom_nodes())
        _install_loader_cache_wrappers()
        _COMFY_RUNTIME_CUSTOM_NODE_ROOTS.add(custom_nodes_root_key)


def _iter_missing_class_candidate_files(
    custom_nodes_root: Path,
    class_type: str,
    *,
    max_candidates: int = 5,
) -> Iterator[Path]:
    """Yield Python files in the extracted bundle that mention one missing class type."""
    yielded_count = 0
    for candidate_path in sorted(custom_nodes_root.rglob("*.py")):
        if yielded_count >= max_candidates:
            return
        try:
            if class_type not in candidate_path.read_text(
                encoding="utf-8", errors="ignore"
            ):
                continue
        except OSError:
            continue
        yielded_count += 1
        yield candidate_path


def _custom_node_package_for_candidate_file(
    custom_nodes_root: Path, candidate_file: Path
) -> str:
    """Return the top-level extracted custom-node package name for one candidate file."""
    try:
        relative_path = candidate_file.relative_to(custom_nodes_root)
    except ValueError:
        return str(candidate_file)
    if not relative_path.parts:
        return str(candidate_file)
    return relative_path.parts[0]


def _missing_node_class_diagnostics(
    missing_class_types: Sequence[str],
    custom_nodes_root: Path | None,
    custom_nodes_bundle_path: str | None,
) -> str:
    """Return a concise diagnostic summary for missing prompt node classes."""
    if custom_nodes_root is None:
        if custom_nodes_bundle_path:
            return (
                "Payload requested custom_nodes_bundle="
                f"{custom_nodes_bundle_path!r}, but it was not available in Modal worker storage."
            )
        return "No custom_nodes_bundle path was present in the remote payload."
    if not custom_nodes_root.exists():
        return f"Extracted custom_nodes root does not exist: {custom_nodes_root}."

    diagnostics: list[str] = [f"extracted_custom_nodes_root={custom_nodes_root}"]
    for class_type in missing_class_types:
        candidate_files = list(
            _iter_missing_class_candidate_files(custom_nodes_root, class_type)
        )
        if not candidate_files:
            diagnostics.append(
                f"{class_type}: no matching Python file found in extracted bundle"
            )
            continue
        rendered_candidates = [
            f"{path.relative_to(custom_nodes_root)} (package={_custom_node_package_for_candidate_file(custom_nodes_root, path)})"
            for path in candidate_files
        ]
        diagnostics.append(f"{class_type}: found candidates {rendered_candidates}")
    return "; ".join(diagnostics)


def _ensure_prompt_node_classes_registered(
    *,
    component_id: str,
    prompt: Mapping[str, Any],
    custom_nodes_root: Path | None,
    custom_nodes_bundle_path: str | None = None,
) -> Mapping[str, type[Any]]:
    """Return the active node mapping or raise a clear error for missing prompt node types."""
    nodes_module = _load_nodes_module()
    resolved_node_mapping = nodes_module.NODE_CLASS_MAPPINGS
    missing_class_types = _prompt_missing_node_class_types(
        prompt, resolved_node_mapping
    )
    if missing_class_types and custom_nodes_root is not None:
        logger.warning(
            "Remote prompt component=%s references missing node classes after initial custom-node import: %s",
            component_id,
            missing_class_types,
        )
        _reload_external_custom_nodes_for_missing_classes(custom_nodes_root)
        resolved_node_mapping = nodes_module.NODE_CLASS_MAPPINGS
        missing_class_types = _prompt_missing_node_class_types(
            prompt, resolved_node_mapping
        )
    if missing_class_types:
        raise _bootstrap_hooks().remote_subgraph_error(
            "Remote subgraph references node classes that are not registered in the Modal worker: "
            f"{missing_class_types}. Ensure custom-node sync is enabled and the required custom-node package "
            "imports successfully inside Modal. "
            f"Diagnostics: {_missing_node_class_diagnostics(missing_class_types, custom_nodes_root, custom_nodes_bundle_path)}"
        )
    return resolved_node_mapping


def _load_execution_module() -> Any:
    """Import the ComfyUI execution module lazily."""
    _ensure_comfyui_support_packages()
    import execution

    return execution


def _load_nodes_module() -> Any:
    """Import the ComfyUI nodes module lazily."""
    import nodes

    return nodes
