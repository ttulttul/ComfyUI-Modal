"""Cloud snapshot, loader, LLM, and warm-container preparation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager
import copy
from dataclasses import dataclass
import hashlib
import importlib
import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Mapping

try:
    from .cloud_comfy_bootstrap import (
        _ensure_comfy_runtime_initialized,
        _ensure_comfyui_support_packages,
        _extract_custom_nodes_bundle,
        _load_execution_module,
        _register_custom_nodes_root,
    )
    from .cloud_image_env import _REMOTE_LLM_COMPILE_CACHE_ROOT
    from .cloud_prompt_execution import _execute_subgraph_prompt, _is_link
    from .cloud_session_bridge import _load_loader_snapshot_profile
    from .cloud_volume_reload import (
        _emit_modal_volume_reload_skip,
        _hydrate_missing_payload_volume_paths,
        _is_modal_volume_open_files_error,
        _modal_volume_reload_marker,
        _reload_modal_volume_for_request,
        _should_reload_modal_volume,
    )
    from .settings import get_settings
except ImportError:  # pragma: no cover - flat Modal-container import.
    from cloud_comfy_bootstrap import (
        _ensure_comfy_runtime_initialized,
        _ensure_comfyui_support_packages,
        _extract_custom_nodes_bundle,
        _load_execution_module,
        _register_custom_nodes_root,
    )
    from cloud_image_env import _REMOTE_LLM_COMPILE_CACHE_ROOT
    from cloud_prompt_execution import _execute_subgraph_prompt, _is_link
    from cloud_session_bridge import _load_loader_snapshot_profile
    from cloud_volume_reload import (
        _emit_modal_volume_reload_skip,
        _hydrate_missing_payload_volume_paths,
        _is_modal_volume_open_files_error,
        _modal_volume_reload_marker,
        _reload_modal_volume_for_request,
        _should_reload_modal_volume,
    )
    from settings import get_settings

logger = logging.getLogger(__name__)

_LOADER_PREWARM_PLAN_KEYS_LOCK = threading.Lock()
_LOADER_PREWARM_PLAN_KEYS: set[str] = set()
_LLM_PREWARM_PLAN_KEYS_LOCK = threading.Lock()
_LLM_PREWARM_PLAN_KEYS: set[str] = set()


@dataclass(frozen=True)
class CloudPrewarmHooks:
    """Cloud logging callbacks supplied by the stable entrypoint."""

    emit_cloud_info: Callable[..., None]
    timed_phase: Callable[..., AbstractContextManager[None]]


_PREWARM_HOOKS: CloudPrewarmHooks | None = None


def configure_cloud_prewarm_hooks(hooks: CloudPrewarmHooks) -> None:
    """Install prewarm callbacks without importing upward."""
    global _PREWARM_HOOKS
    _PREWARM_HOOKS = hooks


def _prewarm_hooks() -> CloudPrewarmHooks:
    """Return configured callbacks or fail on invalid import order."""
    if _PREWARM_HOOKS is None:
        raise RuntimeError("Cloud prewarm hooks have not been configured.")
    return _PREWARM_HOOKS


def _emit_cloud_info(message: str, *args: Any) -> None:
    """Delegate timestamped cloud logging to the stable entrypoint."""
    _prewarm_hooks().emit_cloud_info(message, *args)


def _timed_phase(phase: str, **fields: Any) -> AbstractContextManager[None]:
    """Delegate phase timing to the stable entrypoint."""
    return _prewarm_hooks().timed_phase(phase, **fields)


def _prewarm_snapshot_state(
    *,
    gpu_snapshot_enabled: bool,
    snapshot_profile_key: str = "",
) -> None:
    """Run snapshot-safe initialization before Modal captures a memory snapshot."""
    with _timed_phase(
        "prewarm_snapshot_state",
        gpu_snapshot=gpu_snapshot_enabled,
        snapshot_profile=snapshot_profile_key or None,
    ):
        _ensure_comfyui_support_packages()
        normalized_snapshot_profile_key = snapshot_profile_key.strip()
        if gpu_snapshot_enabled and normalized_snapshot_profile_key:
            _ensure_comfy_runtime_initialized(None)
            _load_execution_module()
            loader_prewarm_plans = _load_loader_snapshot_profile(
                normalized_snapshot_profile_key
            )
            if loader_prewarm_plans:
                _execute_loader_prewarm_plans(
                    component_id=f"snapshot-profile:{normalized_snapshot_profile_key}",
                    loader_prewarm_plans=loader_prewarm_plans,
                    custom_nodes_root=None,
                )
            _emit_cloud_info(
                "Completed GPU-snapshot ComfyUI prewarm before snapshot capture."
            )
            return

        if gpu_snapshot_enabled:
            _emit_cloud_info(
                "Skipping GPU-snapshot ComfyUI prewarm before snapshot capture because no snapshot profile was provided."
            )
        else:
            _emit_cloud_info(
                "Skipping full ComfyUI runtime prewarm during CPU-only snapshot to avoid accidental CUDA initialization."
            )


def _reload_compile_cache_volume(volume: Any | None) -> bool:
    """Refresh persistent compiler artifacts before a runtime opens its caches."""
    if volume is None:
        return False
    reload_method = getattr(volume, "reload", None)
    if not callable(reload_method):
        logger.warning("Modal compile-cache Volume does not expose reload().")
        return False
    with _timed_phase("llm_compile_cache_reload"):
        try:
            reload_method()
        except RuntimeError as exc:
            if _is_modal_volume_open_files_error(exc):
                _log_compile_cache_memory_maps()
            raise
    return True


def _mapped_process_files_under(
    volume_root: Path,
    *,
    proc_root: Path = Path("/proc"),
) -> tuple[tuple[int, str], ...]:
    """Return process ids and files memory-mapped beneath one filesystem root."""
    try:
        resolved_root = volume_root.resolve(strict=True)
        process_directories = tuple(proc_root.iterdir())
    except OSError:
        return ()

    mapped_files: set[tuple[int, str]] = set()
    for process_directory in process_directories:
        if not process_directory.name.isdecimal():
            continue
        try:
            maps_text = (process_directory / "maps").read_text(
                encoding="utf-8",
                errors="replace",
            )
        except OSError:
            continue
        for line in maps_text.splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) != 6 or not fields[5].startswith("/"):
                continue
            mapped_path = fields[5].removesuffix(" (deleted)")
            try:
                Path(mapped_path).relative_to(resolved_root)
            except ValueError:
                continue
            mapped_files.add((int(process_directory.name), mapped_path))
    return tuple(sorted(mapped_files))


def _log_compile_cache_memory_maps() -> None:
    """Log native mappings that explain a busy compile-cache Volume reload."""
    mapped_files = _mapped_process_files_under(_REMOTE_LLM_COMPILE_CACHE_ROOT)
    if not mapped_files:
        logger.warning(
            "Modal compile-cache Volume reload reported open files, but no "
            "memory-mapped cache files were visible in /proc."
        )
        return
    logger.error(
        "Modal compile-cache Volume reload is blocked by %d memory-mapped "
        "native cache file(s): %s",
        len(mapped_files),
        [
            {"pid": process_id, "path": mapped_path}
            for process_id, mapped_path in mapped_files[:8]
        ],
    )


def _prewarm_restored_runtime(compile_cache_volume: Any | None = None) -> None:
    """Run post-restore initialization that should be ready before serving requests."""
    with _timed_phase("prewarm_restored_runtime"):
        _reload_compile_cache_volume(compile_cache_volume)
        _ensure_comfy_runtime_initialized(None)
        _load_execution_module()




def _prepare_warm_container_for_request(
    volume: Any,
    payload: dict[str, Any],
    compile_cache_volume: Any | None = None,
) -> dict[str, Any]:
    """Prime one RemoteEngine container for a request before the first real execution payload arrives."""
    component_id = str(payload.get("component_id") or "modal-warmup")
    reload_marker = _modal_volume_reload_marker(payload)
    _hydrate_missing_payload_volume_paths(volume, payload)
    needs_volume_reload = _should_reload_modal_volume(payload)
    with _timed_phase("remote_engine_warmup", component=component_id):
        if needs_volume_reload:
            _reload_modal_volume_for_request(
                volume,
                component_id,
                reload_marker=reload_marker,
                payload=payload,
            )
        else:
            _emit_modal_volume_reload_skip(component_id, payload)
        custom_nodes_bundle = payload.get("custom_nodes_bundle")
        custom_nodes_root: Path | None = None
        if isinstance(custom_nodes_bundle, str) and custom_nodes_bundle.strip():
            custom_nodes_root = _extract_custom_nodes_bundle(custom_nodes_bundle)
            if custom_nodes_root is not None:
                _register_custom_nodes_root(custom_nodes_root)
        loader_prewarm_plans = payload.get("loader_prewarm_plans")
        if isinstance(loader_prewarm_plans, list) and loader_prewarm_plans:
            _execute_loader_prewarm_plans(
                component_id=component_id,
                loader_prewarm_plans=loader_prewarm_plans,
                custom_nodes_root=custom_nodes_root,
            )
        llm_prewarm_plans = payload.get("llm_prewarm_plans")
        llm_prewarm_results: list[dict[str, Any]] = []
        if isinstance(llm_prewarm_plans, list) and llm_prewarm_plans:
            llm_prewarm_results = _execute_llm_prewarm_plans(
                component_id=component_id,
                prompt_id=(
                    str(payload["prompt_id"])
                    if payload.get("prompt_id") is not None
                    else None
                ),
                llm_prewarm_plans=llm_prewarm_plans,
                compile_cache_volume=compile_cache_volume,
            )
        return {
            "component_id": component_id,
            "task_id": os.getenv("MODAL_TASK_ID"),
            "warmup_slot_index": (
                int(payload["warmup_slot_index"])
                if payload.get("warmup_slot_index") is not None
                else None
            ),
            "reloaded_volume": needs_volume_reload,
            "llm_prewarm_results": llm_prewarm_results,
        }


def _loader_prewarm_plan_key(plan: Mapping[str, Any]) -> str | None:
    """Return the stable worker-local dedupe key for one loader prewarm plan."""
    signature = plan.get("signature")
    if signature is None:
        return None
    normalized_signature = str(signature).strip()
    return normalized_signature or None


def _build_loader_prewarm_payload(
    *,
    component_id: str,
    plan_index: int,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one synthetic single-node subgraph payload for loader warmup."""
    plan_node_id = str(plan.get("node_id") or f"loader-{plan_index}")
    prompt_id = plan.get("prompt_id")
    return {
        "payload_kind": "subgraph",
        "component_id": f"{component_id}::loader-prewarm:{plan_node_id}",
        "prompt_id": (str(prompt_id) if prompt_id is not None else None),
        "component_node_ids": [plan_node_id],
        "subgraph_prompt": copy.deepcopy(dict(plan["subgraph_prompt"])),
        "boundary_inputs": [],
        "boundary_outputs": [],
        "execute_node_ids": list(plan.get("execute_node_ids") or [plan_node_id]),
        "extra_data": {},
    }


def _execute_loader_prewarm_plans(
    *,
    component_id: str,
    loader_prewarm_plans: list[dict[str, Any]],
    custom_nodes_root: Path | None,
) -> None:
    """Execute synthetic one-node loader workflows so fresh workers preload heavyweight models."""
    if not get_settings().enable_loader_prewarm:
        return

    _ensure_comfy_runtime_initialized(custom_nodes_root)
    executable_plans: list[tuple[int, Mapping[str, Any], str | None]] = []
    skipped_plan_count = 0
    for plan_index, plan in enumerate(loader_prewarm_plans):
        if not isinstance(plan, Mapping):
            continue
        plan_key = _loader_prewarm_plan_key(plan)
        if plan_key is not None:
            with _LOADER_PREWARM_PLAN_KEYS_LOCK:
                if plan_key in _LOADER_PREWARM_PLAN_KEYS:
                    skipped_plan_count += 1
                    continue
                _LOADER_PREWARM_PLAN_KEYS.add(plan_key)
        executable_plans.append((plan_index, plan, plan_key))

    def execute_plan(
        plan_entry: tuple[int, Mapping[str, Any], str | None]
    ) -> None:
        """Execute one reserved loader plan and make failures retryable."""
        plan_index, plan, plan_key = plan_entry
        started_at = time.perf_counter()
        try:
            _execute_subgraph_prompt(
                _build_loader_prewarm_payload(
                    component_id=component_id,
                    plan_index=plan_index,
                    plan=plan,
                ),
                hydrated_inputs={},
                custom_nodes_root=custom_nodes_root,
            )
        except Exception:
            if plan_key is not None:
                with _LOADER_PREWARM_PLAN_KEYS_LOCK:
                    _LOADER_PREWARM_PLAN_KEYS.discard(plan_key)
            raise
        logger.info(
            "Completed loader prewarm component=%s class_type=%s plan_index=%d elapsed_seconds=%.3f.",
            component_id,
            plan.get("class_type"),
            plan_index,
            time.perf_counter() - started_at,
        )

    worker_count = min(
        len(executable_plans),
        max(1, int(get_settings().loader_prewarm_workers)),
    )
    if worker_count > 1:
        logger.info(
            "Running %d independent loader prewarms with bounded concurrency=%d component=%s.",
            len(executable_plans),
            worker_count,
            component_id,
        )
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="modal-loader-prewarm",
        ) as executor:
            futures = [
                executor.submit(execute_plan, plan_entry)
                for plan_entry in executable_plans
            ]
            for future in futures:
                future.result()
    else:
        for plan_entry in executable_plans:
            execute_plan(plan_entry)

    executed_plan_count = len(executable_plans)
    if executed_plan_count or skipped_plan_count:
        logger.info(
            "Warm container loader prewarm finished for component=%s executed=%d skipped=%d.",
            component_id,
            executed_plan_count,
            skipped_plan_count,
        )


def _llm_prewarm_model_profile(plan: Mapping[str, Any]) -> str:
    """Return the staged model profile from one rewritten LLM warmup plan."""
    prompt_node = plan.get("prompt_node")
    if isinstance(prompt_node, Mapping):
        inputs = prompt_node.get("inputs")
        if isinstance(inputs, Mapping):
            model_profile = inputs.get("model_profile")
            if isinstance(model_profile, str) and model_profile.strip():
                return model_profile.strip()
    model_profile = plan.get("model_profile")
    if not isinstance(model_profile, str) or not model_profile.strip():
        raise ValueError("LLM prewarm plan requires a fixed model_profile.")
    return model_profile.strip()


def _llm_compile_manifest_path(signature: str) -> Path:
    """Return the content-addressed completion marker for one JIT warmup plan."""
    cache_root = Path(
        os.getenv("TRITON_CACHE_DIR", str(_REMOTE_LLM_COMPILE_CACHE_ROOT))
    ).parent
    return cache_root / "manifests" / f"{signature}.json"


def _write_llm_compile_manifest(
    *,
    signature: str,
    model_profile: str,
    result: Mapping[str, Any],
) -> Path:
    """Atomically publish successful representative-warmup metadata."""
    manifest_path = _llm_compile_manifest_path(signature)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = manifest_path.with_suffix(f".{os.getpid()}.tmp")
    temporary_path.write_text(
        json.dumps(
            {
                "signature": signature,
                "model_profile": model_profile,
                "runtime_fingerprint": os.getenv(
                    "COMFY_MODAL_RUNTIME_FINGERPRINT", ""
                ),
                "completed_at": time.time(),
                "result": dict(result),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    temporary_path.replace(manifest_path)
    return manifest_path


def _execute_llm_prewarm_plans(
    *,
    component_id: str,
    prompt_id: str | None,
    llm_prewarm_plans: list[dict[str, Any]],
    compile_cache_volume: Any | None,
) -> list[dict[str, Any]]:
    """Load resident LLMs, exercise representative shapes, and commit JIT caches."""
    from modal_llm_runtime import prewarm_modal_llm_profile

    results: list[dict[str, Any]] = []
    for plan in llm_prewarm_plans:
        if not isinstance(plan, Mapping):
            continue
        plan_signature = str(plan.get("signature") or "").strip()
        if not plan_signature:
            raise ValueError("LLM prewarm plan requires a stable signature.")
        model_profile = _llm_prewarm_model_profile(plan)
        signature = hashlib.sha256(
            json.dumps(
                {
                    "model_profile": model_profile,
                    "plan_signature": plan_signature,
                    "runtime_fingerprint": os.getenv(
                        "COMFY_MODAL_RUNTIME_FINGERPRINT", ""
                    ),
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        manifest_path = _llm_compile_manifest_path(signature)
        representative_request_count = (
            1
            if manifest_path.exists()
            else max(1, int(plan.get("representative_request_count") or 3))
        )
        with _LLM_PREWARM_PLAN_KEYS_LOCK:
            already_resident = signature in _LLM_PREWARM_PLAN_KEYS
            _LLM_PREWARM_PLAN_KEYS.add(signature)
        if already_resident:
            logger.info(
                "Skipping duplicate resident LLM prewarm profile=%s component=%s.",
                model_profile,
                component_id,
            )
            continue
        try:
            compile_checkpoint = _LLMCompileMissCheckpoint(
                profiles=(model_profile,),
                signal_size=_triton_compile_miss_signal_size(),
                listener_engine_pids=_triton_compile_listener_engine_pids(),
            )
            with _timed_phase(
                "llm_representative_prewarm",
                component=component_id,
                profile=model_profile,
                requests=representative_request_count,
            ):
                result = prewarm_modal_llm_profile(
                    model_profile=model_profile,
                    representative_request_count=representative_request_count,
                    workflow_execution_id=prompt_id,
                )
            manifest_path = _write_llm_compile_manifest(
                signature=signature,
                model_profile=model_profile,
                result=result,
            )
            compile_cache_committed = _commit_actual_llm_compile_cache(
                compile_checkpoint,
                compile_cache_volume,
            )
            results.append(
                {
                    **result,
                    "manifest_path": str(manifest_path),
                    "manifest_cache_hit": representative_request_count == 1,
                    "compile_cache_committed": compile_cache_committed,
                }
            )
        except Exception:
            with _LLM_PREWARM_PLAN_KEYS_LOCK:
                _LLM_PREWARM_PLAN_KEYS.discard(signature)
            raise
    return results


def _llm_profiles_in_payload(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Collect ModalLLM profiles in the executable subgraph dependency closure."""
    if payload.get("payload_kind") not in {"subgraph", "mapped_subgraph"}:
        return ()
    subgraph_prompt = payload.get("subgraph_prompt")
    execute_node_ids = payload.get("execute_node_ids")
    if not isinstance(subgraph_prompt, Mapping) or not isinstance(
        execute_node_ids, (list, tuple)
    ):
        return ()
    prompt = {str(node_id): node for node_id, node in subgraph_prompt.items()}
    profiles: set[str] = set()
    visited: set[str] = set()
    pending = [str(node_id) for node_id in execute_node_ids]
    while pending:
        node_id = pending.pop()
        if node_id in visited:
            continue
        visited.add(node_id)
        prompt_node = prompt.get(node_id)
        if not isinstance(prompt_node, Mapping):
            continue
        inputs = prompt_node.get("inputs")
        if prompt_node.get("class_type") == "ModalLLM" and isinstance(
            inputs, Mapping
        ):
            profile = inputs.get("model_profile")
            if isinstance(profile, str) and profile.strip():
                profiles.add(profile.strip())
        if not isinstance(inputs, Mapping):
            continue
        for input_value in inputs.values():
            if _is_link(input_value):
                pending.append(str(input_value[0]))
    return tuple(sorted(profiles))


@dataclass(frozen=True)
class _LLMCompileMissCheckpoint:
    """Capture the genuine Triton miss signal before one LLM subgraph executes."""

    profiles: tuple[str, ...]
    signal_size: int
    listener_engine_pids: tuple[int, ...]


def _triton_compile_miss_signal_size() -> int:
    """Read the EngineCore compile-miss signal shared through container storage."""
    runtime_module = importlib.import_module("modal_llm_runtime")
    signal_reader = getattr(runtime_module, "triton_compile_miss_signal_size", None)
    if not callable(signal_reader):
        raise RuntimeError(
            "Modal LLM runtime does not expose triton_compile_miss_signal_size()."
        )
    return int(signal_reader())


def _triton_compile_listener_engine_pids() -> tuple[int, ...]:
    """Return live EngineCore processes with cache-aware Triton telemetry."""
    runtime_module = importlib.import_module("modal_llm_runtime")
    listener_reader = getattr(
        runtime_module,
        "triton_compile_listener_engine_pids",
        None,
    )
    if not callable(listener_reader):
        raise RuntimeError(
            "Modal LLM runtime does not expose "
            "triton_compile_listener_engine_pids()."
        )
    return tuple(int(pid) for pid in listener_reader())


def _llm_compile_miss_checkpoint(
    payload: Mapping[str, Any],
) -> _LLMCompileMissCheckpoint | None:
    """Capture the current miss signal for an executable ModalLLM subgraph."""
    profiles = _llm_profiles_in_payload(payload)
    if not profiles:
        return None
    return _LLMCompileMissCheckpoint(
        profiles=profiles,
        signal_size=_triton_compile_miss_signal_size(),
        listener_engine_pids=_triton_compile_listener_engine_pids(),
    )


def _commit_actual_llm_compile_cache(
    checkpoint: _LLMCompileMissCheckpoint | None,
    compile_cache_volume: Any | None,
) -> bool:
    """Commit the compile-cache Volume after a genuine Triton disk-cache miss."""
    if checkpoint is None or compile_cache_volume is None:
        return False
    listener_engine_pids = _triton_compile_listener_engine_pids()
    if not listener_engine_pids:
        logger.warning(
            "Skipping LLM compile-cache commit because no live vLLM EngineCore "
            "reported the cache-aware Triton listener profiles=%s "
            "listener_pids_before=%s.",
            checkpoint.profiles,
            checkpoint.listener_engine_pids,
        )
        return False
    signal_size = _triton_compile_miss_signal_size()
    if signal_size < checkpoint.signal_size:
        raise RuntimeError(
            "Triton compile-miss signal shrank during LLM execution: "
            f"before={checkpoint.signal_size} after={signal_size}."
        )
    if signal_size == checkpoint.signal_size:
        logger.info(
            "Skipping LLM compile-cache commit because every Triton lookup hit "
            "the persistent cache profiles=%s signal_size=%d listener_pids=%s.",
            checkpoint.profiles,
            signal_size,
            listener_engine_pids,
        )
        return False
    commit_method = getattr(compile_cache_volume, "commit", None)
    if not callable(commit_method):
        raise RuntimeError("Modal compile-cache Volume does not expose commit().")
    with _timed_phase(
        "llm_actual_compile_cache_commit",
        profiles=checkpoint.profiles,
        miss_signal_bytes=signal_size - checkpoint.signal_size,
    ):
        commit_method()
    return True

