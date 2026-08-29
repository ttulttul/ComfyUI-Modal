"""Modal volume reload decisions, retries, and committed-file read-through."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
import gc
import logging
import os
from pathlib import Path
import queue
import tempfile
import threading
import time
from typing import Any, Callable, Iterator

try:
    from .cloud_comfy_bootstrap import (
        _load_custom_nodes_manifest,
        _materialize_remote_asset_path,
        _readthrough_cache_path,
        _resolve_runtime_asset_path,
        clear_warm_caches as clear_comfy_bootstrap_warm_caches,
    )
    from .cloud_prompt_execution import (
        clear_warm_caches as clear_cloud_prompt_execution_warm_caches,
    )
    from .cloud_session_bridge import (
        clear_warm_caches as clear_cloud_session_bridge_warm_caches,
    )
    from .settings import get_settings
except ImportError:  # pragma: no cover - flat Modal-container import.
    from cloud_comfy_bootstrap import (
        _load_custom_nodes_manifest,
        _materialize_remote_asset_path,
        _readthrough_cache_path,
        _resolve_runtime_asset_path,
        clear_warm_caches as clear_comfy_bootstrap_warm_caches,
    )
    from cloud_prompt_execution import (
        clear_warm_caches as clear_cloud_prompt_execution_warm_caches,
    )
    from cloud_session_bridge import (
        clear_warm_caches as clear_cloud_session_bridge_warm_caches,
    )
    from settings import get_settings

logger = logging.getLogger(__name__)

_MODAL_VOLUME_RELOAD_OPEN_FILE_RETRY_DELAYS_SECONDS = (
    0.0,
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
    8.0,
)
_MODAL_VOLUME_RELOAD_MARKER_CACHE_LIMIT = 256
_MODAL_VOLUME_RELOAD_MARKERS_LOCK = threading.Lock()
_MODAL_VOLUME_RELOAD_MARKERS: queue.SimpleQueue[str] | None = None
_MODAL_VOLUME_RELOAD_MARKER_SET: set[str] = set()


@dataclass(frozen=True)
class CloudVolumeReloadHooks:
    """Cloud logging callbacks supplied by the stable entrypoint."""

    emit_cloud_info: Callable[..., None]
    timed_phase: Callable[..., AbstractContextManager[None]]


_VOLUME_RELOAD_HOOKS: CloudVolumeReloadHooks | None = None


def configure_cloud_volume_reload_hooks(hooks: CloudVolumeReloadHooks) -> None:
    """Install volume-reload callbacks without importing upward."""
    global _VOLUME_RELOAD_HOOKS
    _VOLUME_RELOAD_HOOKS = hooks


def _volume_reload_hooks() -> CloudVolumeReloadHooks:
    """Return configured callbacks or fail on invalid import order."""
    if _VOLUME_RELOAD_HOOKS is None:
        raise RuntimeError("Cloud volume-reload hooks have not been configured.")
    return _VOLUME_RELOAD_HOOKS


def _emit_cloud_info(message: str, *args: Any) -> None:
    """Delegate timestamped cloud logging to the stable entrypoint."""
    _volume_reload_hooks().emit_cloud_info(message, *args)


def _timed_phase(phase: str, **fields: Any) -> AbstractContextManager[None]:
    """Delegate phase timing to the stable entrypoint."""
    return _volume_reload_hooks().timed_phase(phase, **fields)


def _should_reload_modal_volume(payload: dict[str, Any]) -> bool:
    """Return whether this request needs the mounted Modal volume reloaded."""
    if _payload_volume_paths(payload) and not _payload_volume_paths_visible(payload):
        return True
    if not bool(payload.get("requires_volume_reload", True)):
        return False
    if _payload_uploaded_volume_paths_visible(payload):
        reload_marker = _modal_volume_reload_marker(payload)
        if reload_marker is not None:
            _record_modal_volume_reload_marker(reload_marker)
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
        while (
            len(_MODAL_VOLUME_RELOAD_MARKER_SET)
            > _MODAL_VOLUME_RELOAD_MARKER_CACHE_LIMIT
        ):
            expired_marker = _MODAL_VOLUME_RELOAD_MARKERS.get()
            _MODAL_VOLUME_RELOAD_MARKER_SET.discard(expired_marker)


def _clear_warm_remote_caches() -> None:
    """Drop warm-container caches that may retain references to mounted volume files."""
    clear_cloud_prompt_execution_warm_caches()
    clear_cloud_session_bridge_warm_caches()
    clear_comfy_bootstrap_warm_caches()


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


def _iter_payload_input_strings(value: Any) -> Iterator[str]:
    """Yield string literals nested inside one serialized prompt input value."""
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, list):
        if len(value) == 2 and isinstance(value[0], str):
            return
        for item in value:
            yield from _iter_payload_input_strings(item)
        return
    if isinstance(value, dict):
        for nested_value in value.values():
            yield from _iter_payload_input_strings(nested_value)


def _payload_volume_paths(payload: dict[str, Any]) -> set[Path]:
    """Return mounted-volume paths referenced by this remote payload."""
    remote_storage_root = Path(get_settings().remote_storage_root).resolve()
    referenced_paths: set[Path] = set()

    custom_nodes_bundle = payload.get("custom_nodes_bundle")
    if isinstance(custom_nodes_bundle, str):
        bundle_path = Path(_materialize_remote_asset_path(custom_nodes_bundle))
        if bundle_path.is_absolute() and bundle_path.resolve().is_relative_to(
            remote_storage_root
        ):
            referenced_paths.add(bundle_path)

    prompt = payload.get("subgraph_prompt", {})
    if not isinstance(prompt, dict):
        return referenced_paths

    for prompt_node in prompt.values():
        if not isinstance(prompt_node, dict):
            continue
        inputs = prompt_node.get("inputs", {})
        if not isinstance(inputs, dict):
            continue
        for input_value in inputs.values():
            for candidate_path in _iter_payload_input_strings(input_value):
                materialized_path = _materialize_remote_asset_path(candidate_path)
                materialized_path_obj = Path(materialized_path)
                if (
                    materialized_path_obj.is_absolute()
                    and materialized_path_obj.resolve().is_relative_to(
                        remote_storage_root
                    )
                ):
                    referenced_paths.add(materialized_path_obj)
    return referenced_paths


def _payload_uploaded_volume_paths(payload: dict[str, Any]) -> set[Path]:
    """Return newly uploaded mounted-volume paths relevant to this payload."""
    remote_storage_root = Path(get_settings().remote_storage_root).resolve()
    uploaded_paths: set[Path] = set()
    for candidate_path in payload.get("uploaded_volume_paths", []):
        if isinstance(candidate_path, str) and candidate_path.strip():
            materialized_path = Path(_materialize_remote_asset_path(candidate_path))
            if (
                materialized_path.is_absolute()
                and materialized_path.resolve().is_relative_to(remote_storage_root)
            ):
                uploaded_paths.add(materialized_path)
    return uploaded_paths


def _payload_uploaded_volume_paths_visible(payload: dict[str, Any]) -> bool:
    """Return whether every newly uploaded mounted-volume path is already visible."""
    uploaded_paths = _payload_uploaded_volume_paths(payload)
    if not uploaded_paths:
        return False
    return all(_runtime_volume_path_visible(path) for path in uploaded_paths)


def _runtime_volume_path_visible(volume_path: Path) -> bool:
    """Return whether a mounted path is available directly or through read-through storage."""
    if volume_path.exists():
        return True
    cache_path = _readthrough_cache_path(volume_path)
    return cache_path is not None and cache_path.exists()


def _payload_volume_paths_visible(payload: dict[str, Any]) -> bool:
    """Return whether every mounted-volume path referenced by this payload is already visible."""
    referenced_paths = _payload_volume_paths(payload)
    if not referenced_paths:
        return False
    return all(_runtime_volume_path_visible(path) for path in referenced_paths)


def _download_committed_volume_path(
    volume: Any, volume_path: Path, cache_path: Path
) -> None:
    """Stream one committed Modal Volume file into the worker's ephemeral cache."""
    remote_storage_root = Path(get_settings().remote_storage_root).resolve()
    relative_path = volume_path.resolve().relative_to(remote_storage_root).as_posix()
    read_file_into_fileobj = getattr(volume, "read_file_into_fileobj", None)
    read_file = getattr(volume, "read_file", None)
    if not callable(read_file_into_fileobj) and not callable(read_file):
        raise AttributeError(
            "The configured Modal Volume does not support committed file reads."
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{cache_path.name}.",
        suffix=".tmp",
        dir=cache_path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        with temporary_path.open("wb") as cache_file:
            if callable(read_file_into_fileobj):
                read_file_into_fileobj(relative_path, cache_file)
            else:
                assert callable(read_file)
                for chunk in read_file(relative_path):
                    cache_file.write(chunk)
        os.replace(temporary_path, cache_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _custom_nodes_manifest_dependency_paths(
    volume_path: Path,
    runtime_path: Path,
) -> set[Path]:
    """Return mounted-volume dependencies declared by one custom-node manifest."""
    remote_storage_root = Path(get_settings().remote_storage_root).resolve()
    custom_nodes_root = remote_storage_root / "custom_nodes"
    if (
        volume_path.suffix.lower() != ".json"
        or not volume_path.resolve().is_relative_to(custom_nodes_root)
    ):
        return set()
    try:
        manifest_payload = _load_custom_nodes_manifest(runtime_path)
    except RuntimeError:
        return set()

    dependency_paths: set[Path] = set()
    entry_payloads = manifest_payload.get("entries", [])
    if not isinstance(entry_payloads, list):
        return dependency_paths
    for entry_payload in entry_payloads:
        if not isinstance(entry_payload, dict):
            continue
        candidate_payloads = [entry_payload]
        asset_payloads = entry_payload.get("assets", [])
        if isinstance(asset_payloads, list):
            candidate_payloads.extend(
                asset_payload
                for asset_payload in asset_payloads
                if isinstance(asset_payload, dict)
            )
        for candidate_payload in candidate_payloads:
            remote_path = candidate_payload.get("remote_path")
            if not isinstance(remote_path, str) or not remote_path.strip():
                continue
            materialized_path = Path(_materialize_remote_asset_path(remote_path))
            if _readthrough_cache_path(materialized_path) is not None:
                dependency_paths.add(materialized_path)
    return dependency_paths


def _hydrate_missing_payload_volume_paths(
    volume: Any, payload: dict[str, Any]
) -> list[Path]:
    """Cache committed payload files that are absent from this worker's mounted snapshot."""
    candidate_paths = _payload_volume_paths(payload) | _payload_uploaded_volume_paths(
        payload
    )
    if not candidate_paths:
        return []
    if not callable(getattr(volume, "read_file_into_fileobj", None)) and not callable(
        getattr(volume, "read_file", None)
    ):
        return []

    hydrated_paths: list[Path] = []
    pending_paths = sorted(candidate_paths)
    visited_paths: set[Path] = set()
    component_id = str(payload.get("component_id") or "modal-subgraph")
    while pending_paths:
        volume_path = pending_paths.pop(0)
        if volume_path in visited_paths:
            continue
        visited_paths.add(volume_path)
        runtime_path = Path(_resolve_runtime_asset_path(str(volume_path)))
        if not _runtime_volume_path_visible(volume_path):
            cache_path = _readthrough_cache_path(volume_path)
            if cache_path is None:
                continue
            try:
                with _timed_phase(
                    "committed_volume_readthrough",
                    component=component_id,
                    path=volume_path.name,
                ):
                    _download_committed_volume_path(volume, volume_path, cache_path)
            except FileNotFoundError:
                logger.warning(
                    "Committed Modal Volume path %s was unavailable for component=%s; falling back to mounted-volume reload.",
                    volume_path,
                    component_id,
                )
                continue
            hydrated_paths.append(cache_path)
            runtime_path = cache_path

        pending_paths.extend(
            sorted(
                _custom_nodes_manifest_dependency_paths(volume_path, runtime_path)
                - visited_paths
            )
        )

    if hydrated_paths:
        _emit_cloud_info(
            "Hydrated %d missing committed volume file(s) through read-through storage for component=%s.",
            len(hydrated_paths),
            component_id,
        )
    return hydrated_paths


def _log_payload_volume_reload_diagnostics(
    component_id: str,
    payload: dict[str, Any] | None,
    *,
    context: str,
) -> None:
    """Log the mounted-volume paths relevant to one reload decision or failure."""
    if payload is None:
        return

    uploaded_paths = sorted(
        str(path) for path in _payload_uploaded_volume_paths(payload)
    )
    referenced_paths = sorted(str(path) for path in _payload_volume_paths(payload))
    logger.info(
        "Modal volume reload diagnostics for component=%s context=%s uploaded_paths=%s referenced_paths=%s visible_uploaded=%s visible_referenced=%s.",
        component_id,
        context,
        uploaded_paths,
        referenced_paths,
        _payload_uploaded_volume_paths_visible(payload),
        _payload_volume_paths_visible(payload),
    )


def _reload_modal_volume_for_request(
    volume: Any,
    component_id: str,
    reload_marker: str | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    """Reload the Modal volume, retrying briefly while warm state releases open files."""
    with _timed_phase("modal_volume_reload", component=component_id):
        diagnostics_logged = False
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
                if payload is not None and not diagnostics_logged:
                    _log_payload_volume_reload_diagnostics(
                        component_id,
                        payload,
                        context="open_files_retry",
                    )
                    diagnostics_logged = True
                if attempt_index == len(
                    _MODAL_VOLUME_RELOAD_OPEN_FILE_RETRY_DELAYS_SECONDS
                ):
                    if payload is not None and _payload_volume_paths_visible(payload):
                        _emit_cloud_info(
                            "Modal volume reload still reported open files for component=%s after %d attempt(s), "
                            "but all referenced mounted-volume paths are already visible. Proceeding without reload.",
                            component_id,
                            attempt_index,
                        )
                        if reload_marker is not None:
                            _record_modal_volume_reload_marker(reload_marker)
                        return
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
    if _payload_uploaded_volume_paths_visible(payload):
        _emit_cloud_info(
            "Skipping modal_volume_reload for component=%s because all uploaded mounted-volume paths are already visible in this container.",
            component_id,
        )
        _log_payload_volume_reload_diagnostics(
            str(component_id),
            payload,
            context="skip_visible_uploaded_paths",
        )
        return
    reload_marker = _modal_volume_reload_marker(payload)
    if reload_marker is not None and _has_seen_modal_volume_reload_marker(
        reload_marker
    ):
        _emit_cloud_info(
            "Skipping modal_volume_reload for component=%s because this container already reloaded marker=%s.",
            component_id,
            reload_marker,
        )
        _log_payload_volume_reload_diagnostics(
            str(component_id),
            payload,
            context="skip_reload_marker_seen",
        )
        return
    _emit_cloud_info(
        "Skipping modal_volume_reload for component=%s because no new assets were uploaded for this request.",
        component_id,
    )
    _log_payload_volume_reload_diagnostics(
        str(component_id),
        payload,
        context="skip_no_new_assets",
    )
