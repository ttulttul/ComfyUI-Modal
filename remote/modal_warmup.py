"""Modal warmup, snapshot-profile, and local-gap keepalive orchestration."""

from __future__ import annotations

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from contextlib import nullcontext
import copy
from dataclasses import dataclass, field
import hashlib
import json
import logging
import queue
import statistics
import threading
import time
from typing import Any, Callable, Iterator, Mapping

from ..settings import get_settings, modal_deployment_app_name
from .local_execution import _is_link, _resolve_required_subgraph_nodes
from .modal_deployment import (
    ModalRemoteInvocationError,
    _auto_deploy_modal_app,
    _call_modal_method,
    _component_pool_slot_index,
    _is_missing_modal_deployment_error,
    _load_modal_cloud_module,
    _lookup_deployed_remote_engine,
    _lookup_protocol_current_remote_engine,
    _modal_cloud_settings_override,
    _modal_lookup_error_types,
    _remote_worker_affinity_key,
    _settings_for_payload,
)

logger = logging.getLogger(__name__)

try:
    import modal  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - local fallback environments.
    modal = None

_PROMPT_WARMUP_STATES_LOCK = threading.Lock()
_PROMPT_WARMUP_STATES: dict[str, "_PromptWarmupState"] = {}
_PROMPT_WARMUP_STATE_ORDER: queue.SimpleQueue[str] | None = None
_PROMPT_WARMUP_STATE_CACHE_LIMIT = 256
_LOCAL_GAP_KEEPALIVES_LOCK = threading.Lock()
_LOCAL_GAP_KEEPALIVES: dict[tuple[str, int], "_LocalGapKeepaliveState"] = {}
_SNAPSHOT_PROFILE_RECORDS_LOCK = threading.Lock()
_SNAPSHOT_PROFILE_RECORDS: dict[str, dict[str, Any]] = {}
_ROOT_LOADER_PREWARM_CLASS_TYPES = frozenset(
    {"CheckpointLoaderSimple", "UNETLoader", "CLIPLoader", "DualCLIPLoader"}
)
_SPECULATIVE_PREWARM_TARGET_KEY = "speculative_remote_prewarm_target"
_REMOTE_MODAL_WARMUP_EXECUTOR_LOCK = threading.Lock()
_REMOTE_MODAL_WARMUP_EXECUTOR: ThreadPoolExecutor | None = None
_REMOTE_MODAL_KEEPALIVE_EXECUTOR_LOCK = threading.Lock()
_REMOTE_MODAL_KEEPALIVE_EXECUTOR: ThreadPoolExecutor | None = None


@dataclass(frozen=True)
class ModalWarmupHooks:
    """Callbacks supplied by the host orchestrator for adjacent concerns."""

    ensure_llm_profiles_staged: Callable[[dict[str, Any], str], None]
    mapped_execution_parallelism: Callable[[int], int]


_WARMUP_HOOKS: ModalWarmupHooks | None = None


def configure_modal_warmup_hooks(hooks: ModalWarmupHooks) -> None:
    """Install host callbacks without importing upward into the orchestrator."""
    global _WARMUP_HOOKS
    _WARMUP_HOOKS = hooks


def _warmup_hooks() -> ModalWarmupHooks:
    """Return configured host callbacks or fail with a clear import-order error."""
    if _WARMUP_HOOKS is None:
        raise RuntimeError("Modal warmup hooks have not been configured.")
    return _WARMUP_HOOKS


def _ensure_llm_profiles_staged(
    payload: dict[str, Any], deployment_app_name: str
) -> None:
    """Delegate LLM staging through the injected host callback."""
    _warmup_hooks().ensure_llm_profiles_staged(payload, deployment_app_name)


def _remote_modal_call_worker_count() -> int:
    """Return the number of local threads reserved for blocking Modal calls."""
    return max(1, int(get_settings().max_inflight_calls))


def _remote_modal_warmup_executor() -> ThreadPoolExecutor:
    """Return the lazily constructed executor for Modal warmup calls."""
    global _REMOTE_MODAL_WARMUP_EXECUTOR
    with _REMOTE_MODAL_WARMUP_EXECUTOR_LOCK:
        if _REMOTE_MODAL_WARMUP_EXECUTOR is None:
            _REMOTE_MODAL_WARMUP_EXECUTOR = ThreadPoolExecutor(
                max_workers=_remote_modal_call_worker_count()
            )
        return _REMOTE_MODAL_WARMUP_EXECUTOR


def _remote_modal_keepalive_executor() -> ThreadPoolExecutor:
    """Return the lazily constructed executor for Modal keepalive calls."""
    global _REMOTE_MODAL_KEEPALIVE_EXECUTOR
    with _REMOTE_MODAL_KEEPALIVE_EXECUTOR_LOCK:
        if _REMOTE_MODAL_KEEPALIVE_EXECUTOR is None:
            _REMOTE_MODAL_KEEPALIVE_EXECUTOR = ThreadPoolExecutor(
                max_workers=_remote_modal_call_worker_count()
            )
        return _REMOTE_MODAL_KEEPALIVE_EXECUTOR


@dataclass
class _PromptWarmupState:
    """Track proactive warmup state for one local prompt."""

    scheduled_slots: set[int] = field(default_factory=set)
    exact_component_parallelism: dict[str, int] = field(default_factory=dict)
    slot_futures: dict[int, Future[Any]] = field(default_factory=dict)
    scheduled_speculative_affinities: set[str] = field(default_factory=set)
    speculative_affinity_futures: dict[str, Future[Any]] = field(default_factory=dict)


@dataclass
class _LocalGapKeepaliveState:
    """Track one prompt-scoped worker-retention loop during local execution."""

    component_id: str
    stop_event: threading.Event
    future: Future[Any]


def _mapped_execution_parallelism(item_count: int) -> int:
    """Delegate mapped worker-width calculation through the injected callback."""
    return _warmup_hooks().mapped_execution_parallelism(item_count)


def _warmup_prompt_id(warmup_request: dict[str, Any]) -> str | None:
    """Return the prompt id that scopes proactive warmup state."""
    prompt_id = warmup_request.get("prompt_id")
    if prompt_id is None:
        return None
    normalized_prompt_id = str(prompt_id).strip()
    return normalized_prompt_id or None


def _ensure_prompt_warmup_state(prompt_id: str) -> _PromptWarmupState:
    """Return the cached warmup state bucket for one prompt."""
    global _PROMPT_WARMUP_STATE_ORDER

    state = _PROMPT_WARMUP_STATES.get(prompt_id)
    if state is not None:
        return state

    state = _PromptWarmupState()
    _PROMPT_WARMUP_STATES[prompt_id] = state
    if _PROMPT_WARMUP_STATE_ORDER is None:
        _PROMPT_WARMUP_STATE_ORDER = queue.SimpleQueue()
    _PROMPT_WARMUP_STATE_ORDER.put(prompt_id)
    while len(_PROMPT_WARMUP_STATES) > _PROMPT_WARMUP_STATE_CACHE_LIMIT:
        expired_prompt_id = _PROMPT_WARMUP_STATE_ORDER.get()
        _PROMPT_WARMUP_STATES.pop(expired_prompt_id, None)
    return state


def _clamp_prompt_warmup_target(warmup_target: int) -> int:
    """Clamp one proactive warmup target to the configured Modal container cap."""
    normalized_target = max(0, int(warmup_target))
    max_containers = get_settings().max_containers
    if max_containers is not None:
        return min(normalized_target, max_containers)
    return normalized_target


def _iter_loader_prewarm_prompt_payloads(
    payload: dict[str, Any]
) -> Iterator[dict[str, Any]]:
    """Yield subgraph-like payload fragments that may contain root loader nodes."""
    split_proxy_payloads = payload.get("split_proxy_payloads")
    if isinstance(split_proxy_payloads, dict):
        for phase_payload in split_proxy_payloads.values():
            if isinstance(phase_payload, dict):
                yield phase_payload
        return
    if isinstance(split_proxy_payloads, list):
        for phase_payload in split_proxy_payloads:
            if isinstance(phase_payload, dict):
                yield phase_payload
        return
    if isinstance(payload.get("subgraph_prompt"), dict):
        yield payload


def _is_root_literal_loader_node(prompt_node: Mapping[str, Any]) -> bool:
    """Return whether one prompt node is a supported loader with only literal inputs."""
    class_type = str(prompt_node.get("class_type") or "")
    if class_type not in _ROOT_LOADER_PREWARM_CLASS_TYPES:
        return False
    inputs = prompt_node.get("inputs")
    if not isinstance(inputs, Mapping):
        return False
    return not any(_is_link(input_value) for input_value in inputs.values())


def _loader_prewarm_plan_signature(class_type: str, inputs: Mapping[str, Any]) -> str:
    """Return a stable signature for one synthetic loader prewarm plan."""
    return json.dumps(
        {
            "class_type": class_type,
            "inputs": copy.deepcopy(dict(inputs)),
        },
        sort_keys=True,
        default=str,
    )


def _loader_snapshot_profile_key(loader_prewarm_plans: list[dict[str, Any]]) -> str:
    """Return the stable snapshot-profile key for one set of loader prewarm plans."""
    if not loader_prewarm_plans:
        return ""

    profile_payload = {
        "plan_signatures": sorted(
            str(plan.get("signature") or "")
            for plan in loader_prewarm_plans
            if str(plan.get("signature") or "")
        )
    }
    if not profile_payload["plan_signatures"]:
        return ""
    profile_digest = hashlib.sha256(
        json.dumps(profile_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return f"loader-profile:{profile_digest}"


def _normalize_loader_snapshot_profile_record(
    loader_prewarm_plans: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return the normalized snapshot-profile record for one loader-plan set."""
    snapshot_profile_key = _loader_snapshot_profile_key(loader_prewarm_plans)
    if not snapshot_profile_key:
        return None
    return {
        "snapshot_profile_key": snapshot_profile_key,
        "loader_prewarm_plans": copy.deepcopy(loader_prewarm_plans),
        "snapshot_policy": {
            "selected_variant": None,
            "measurements": {"snapshot": [], "direct": []},
        },
    }


def _store_loader_snapshot_profile(loader_prewarm_plans: list[dict[str, Any]]) -> str:
    """Persist one loader snapshot profile so `snap=True` can retrieve it later."""
    settings = get_settings()
    if not settings.enable_gpu_memory_snapshot or modal is None:
        return ""

    normalized_record = _normalize_loader_snapshot_profile_record(loader_prewarm_plans)
    if normalized_record is None:
        return ""

    snapshot_profile_key = str(normalized_record["snapshot_profile_key"])
    with _SNAPSHOT_PROFILE_RECORDS_LOCK:
        cached_record = _SNAPSHOT_PROFILE_RECORDS.get(snapshot_profile_key)
        if cached_record is not None:
            return snapshot_profile_key
        snapshot_profiles = modal.Dict.from_name(
            settings.snapshot_profile_dict_name,
            create_if_missing=True,
        )
        stored_record = snapshot_profiles.get(snapshot_profile_key)
        if isinstance(stored_record, dict):
            merged_record = copy.deepcopy(stored_record)
            merged_record["snapshot_profile_key"] = snapshot_profile_key
            merged_record["loader_prewarm_plans"] = copy.deepcopy(
                loader_prewarm_plans
            )
        else:
            merged_record = normalized_record
        snapshot_profiles[snapshot_profile_key] = merged_record
        _SNAPSHOT_PROFILE_RECORDS[snapshot_profile_key] = merged_record
    return snapshot_profile_key


def _prepare_snapshot_profile_fields(payload: dict[str, Any]) -> str:
    """Derive, persist, and attach the loader profile used by one request."""
    loader_prewarm_plans = _build_loader_prewarm_plans(payload)
    stored_profile_key = _store_loader_snapshot_profile(loader_prewarm_plans)
    existing_profile_key = str(payload.get("snapshot_profile_key") or "").strip()
    snapshot_profile_key = stored_profile_key or existing_profile_key
    if snapshot_profile_key:
        payload["snapshot_profile_key"] = snapshot_profile_key
    return snapshot_profile_key


def _snapshot_profile_store_for_payload(payload: Mapping[str, Any]) -> Any | None:
    """Return the shared profile store when Modal and snapshot policy are enabled."""
    if modal is None:
        return None
    settings = _settings_for_payload(payload)
    return modal.Dict.from_name(
        settings.snapshot_profile_dict_name,
        create_if_missing=True,
    )


def _snapshot_policy_record(
    payload: Mapping[str, Any], snapshot_profile_key: str
) -> dict[str, Any] | None:
    """Load one profile record without discarding locally cached measurements."""
    with _SNAPSHOT_PROFILE_RECORDS_LOCK:
        cached = _SNAPSHOT_PROFILE_RECORDS.get(snapshot_profile_key)
    store = _snapshot_profile_store_for_payload(payload)
    if store is None:
        return copy.deepcopy(cached) if cached is not None else None
    stored = store.get(snapshot_profile_key)
    if not isinstance(stored, dict):
        return copy.deepcopy(cached) if cached is not None else None
    with _SNAPSHOT_PROFILE_RECORDS_LOCK:
        _SNAPSHOT_PROFILE_RECORDS[snapshot_profile_key] = copy.deepcopy(stored)
    return copy.deepcopy(stored)


def _select_gpu_snapshot_for_profile(
    payload: dict[str, Any], snapshot_profile_key: str
) -> bool:
    """Choose snapshot or direct startup from persisted per-profile measurements."""
    settings = _settings_for_payload(payload)
    if not settings.enable_gpu_memory_snapshot or not snapshot_profile_key:
        payload["gpu_snapshot_variant"] = "direct"
        return False
    requested_variant = str(payload.get("gpu_snapshot_variant") or "").strip()
    if requested_variant in {"snapshot", "direct"}:
        return requested_variant == "snapshot"

    record = _snapshot_policy_record(payload, snapshot_profile_key) or {}
    policy = record.get("snapshot_policy")
    if not isinstance(policy, Mapping):
        policy = {}
    selected_variant = policy.get("selected_variant")
    if selected_variant not in {"snapshot", "direct"}:
        measurements = policy.get("measurements")
        if not isinstance(measurements, Mapping):
            measurements = {}
        snapshot_samples = measurements.get("snapshot")
        direct_samples = measurements.get("direct")
        snapshot_count = len(snapshot_samples) if isinstance(snapshot_samples, list) else 0
        direct_count = len(direct_samples) if isinstance(direct_samples, list) else 0
        selected_variant = "snapshot" if snapshot_count <= direct_count else "direct"
    payload["gpu_snapshot_variant"] = selected_variant
    logger.info(
        "Selected Modal startup variant=%s snapshot_profile=%s component=%s.",
        selected_variant,
        snapshot_profile_key,
        payload.get("component_id"),
    )
    return selected_variant == "snapshot"


def _record_snapshot_warmup_measurement(
    payload: Mapping[str, Any], elapsed_seconds: float
) -> None:
    """Persist one warmup latency and select the faster profile after both variants run."""
    snapshot_profile_key = str(payload.get("snapshot_profile_key") or "").strip()
    variant = str(payload.get("gpu_snapshot_variant") or "").strip()
    if not snapshot_profile_key or variant not in {"snapshot", "direct"}:
        return
    store = _snapshot_profile_store_for_payload(payload)
    if store is None:
        return
    record = _snapshot_policy_record(payload, snapshot_profile_key) or {
        "snapshot_profile_key": snapshot_profile_key,
        "loader_prewarm_plans": [],
    }
    policy = record.get("snapshot_policy")
    if not isinstance(policy, dict):
        policy = {}
    measurements = policy.get("measurements")
    if not isinstance(measurements, dict):
        measurements = {"snapshot": [], "direct": []}
    for candidate in ("snapshot", "direct"):
        samples = measurements.get(candidate)
        if not isinstance(samples, list):
            measurements[candidate] = []
    measurements[variant].append(float(elapsed_seconds))
    measurements[variant] = measurements[variant][-5:]
    if measurements["snapshot"] and measurements["direct"]:
        policy["selected_variant"] = min(
            ("snapshot", "direct"),
            key=lambda candidate: statistics.median(measurements[candidate]),
        )
    else:
        policy["selected_variant"] = None
    policy["measurements"] = measurements
    record["snapshot_policy"] = policy
    store[snapshot_profile_key] = record
    with _SNAPSHOT_PROFILE_RECORDS_LOCK:
        _SNAPSHOT_PROFILE_RECORDS[snapshot_profile_key] = copy.deepcopy(record)
    logger.info(
        "Recorded Modal startup measurement snapshot_profile=%s variant=%s elapsed=%.3fs selected=%s.",
        snapshot_profile_key,
        variant,
        elapsed_seconds,
        policy["selected_variant"],
    )


def _build_loader_prewarm_plans(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return synthetic one-node subgraph plans for supported root loader nodes."""
    if not get_settings().enable_loader_prewarm:
        return []

    prompt_id = (
        str(payload.get("prompt_id")) if payload.get("prompt_id") is not None else None
    )
    seen_signatures: set[str] = set()
    plans: list[dict[str, Any]] = []
    for prompt_payload in _iter_loader_prewarm_prompt_payloads(payload):
        subgraph_prompt = prompt_payload.get("subgraph_prompt")
        if not isinstance(subgraph_prompt, Mapping):
            continue
        for node_id, prompt_node in subgraph_prompt.items():
            if not isinstance(prompt_node, Mapping) or not _is_root_literal_loader_node(
                prompt_node
            ):
                continue
            class_type = str(prompt_node.get("class_type") or "")
            inputs = prompt_node.get("inputs")
            if not isinstance(inputs, Mapping):
                continue
            signature = _loader_prewarm_plan_signature(class_type, inputs)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            plans.append(
                {
                    "signature": signature,
                    "node_id": str(node_id),
                    "class_type": class_type,
                    "prompt_id": prompt_id,
                    "subgraph_prompt": {
                        str(node_id): copy.deepcopy(dict(prompt_node)),
                    },
                    "execute_node_ids": [str(node_id)],
                }
            )
    return plans


def _iter_executable_subgraph_nodes(
    payload: Mapping[str, Any],
) -> Iterator[Mapping[str, Any]]:
    """Yield only prompt nodes required by this payload's execution targets."""
    for prompt_payload in _iter_loader_prewarm_prompt_payloads(dict(payload)):
        raw_prompt = prompt_payload.get("subgraph_prompt")
        if not isinstance(raw_prompt, Mapping):
            continue
        prompt = {
            str(node_id): dict(prompt_node)
            for node_id, prompt_node in raw_prompt.items()
            if isinstance(prompt_node, Mapping)
        }
        execute_node_ids: list[str] = []
        for field_name in (
            "execute_node_ids",
            "mapped_execute_node_ids",
            "static_execute_node_ids",
        ):
            field_value = prompt_payload.get(field_name)
            if isinstance(field_value, (list, tuple)):
                execute_node_ids.extend(str(node_id) for node_id in field_value)
        required_node_ids = _resolve_required_subgraph_nodes(
            prompt,
            list(dict.fromkeys(execute_node_ids)),
        )
        for node_id in required_node_ids:
            yield prompt[node_id]


def _build_llm_prewarm_plans(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return LLM warmups found in executable subgraph dependency closures."""
    plans: list[dict[str, Any]] = []
    seen_profiles: set[str] = set()
    for prompt_node in _iter_executable_subgraph_nodes(payload):
        if prompt_node.get("class_type") != "ModalLLM":
            continue
        inputs = prompt_node.get("inputs")
        if not isinstance(inputs, Mapping):
            continue
        model_profile = inputs.get("model_profile")
        if not isinstance(model_profile, str) or not model_profile.strip():
            continue
        normalized_profile = model_profile.strip()
        if normalized_profile in seen_profiles:
            continue
        seen_profiles.add(normalized_profile)
        signature_payload = {
            "model_profile": normalized_profile,
            "representative_request_count": 3,
        }
        plans.append(
            {
                **signature_payload,
                "signature": hashlib.sha256(
                    json.dumps(signature_payload, sort_keys=True).encode("utf-8")
                ).hexdigest(),
                "prompt_node": copy.deepcopy(dict(prompt_node)),
            }
        )
    return plans


def _build_prompt_warmup_request(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Extract the prompt-scoped warmup-relevant fields from one payload."""
    settings = _settings_for_payload(payload)
    loader_prewarm_plans = _build_loader_prewarm_plans(payload)
    llm_prewarm_plans = _build_llm_prewarm_plans(payload)
    snapshot_profile_key = _store_loader_snapshot_profile(loader_prewarm_plans)
    if settings.enable_gpu_memory_snapshot and not snapshot_profile_key and not llm_prewarm_plans:
        logger.info(
            "Skipping proactive Modal warmup request for component=%s because GPU snapshots are enabled and no loader snapshot profile was derived.",
            payload.get("component_id"),
        )
        return None
    return {
        "prompt_id": (
            str(payload.get("prompt_id"))
            if payload.get("prompt_id") is not None
            else None
        ),
        "component_id": str(
            payload.get(
                "mapped_progress_display_node_id",
                payload.get("component_id", "modal-warmup"),
            )
        ),
        "modal_gpu": settings.modal_gpu,
        "remote_worker_affinity_group": payload.get(
            "remote_worker_affinity_group"
        ),
        "remote_local_gap_pool": bool(payload.get("remote_local_gap_pool")),
        "requires_volume_reload": bool(payload.get("requires_volume_reload", True)),
        "volume_reload_marker": payload.get("volume_reload_marker"),
        "uploaded_volume_paths": list(payload.get("uploaded_volume_paths", [])),
        "custom_nodes_bundle": payload.get("custom_nodes_bundle"),
        "loader_prewarm_plans": loader_prewarm_plans,
        "llm_prewarm_plans": llm_prewarm_plans,
        "snapshot_profile_key": snapshot_profile_key,
    }


def _component_parallelism_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the prompt-level parallelism metadata attached by the queue rewrite."""
    extra_data = payload.get("extra_data") or {}
    modal_metadata = extra_data.get("modal") or {}
    if not isinstance(modal_metadata, dict):
        return {}
    return modal_metadata


def _prompt_parallelism_target(
    payload: dict[str, Any],
    *,
    exact_component_parallelism: dict[str, int] | None = None,
) -> int:
    """Return the current best-effort prompt-wide warmup target for one payload."""
    metadata = _component_parallelism_metadata(payload)
    execution_stages = metadata.get("component_execution_stages")
    if not isinstance(execution_stages, list):
        return 1

    mapped_component_ids = {
        str(component_id)
        for component_id in metadata.get("mapped_component_ids", [])
        if str(component_id)
    }
    exact_parallelism = exact_component_parallelism or {}
    stage_parallelism = 0
    for stage in execution_stages:
        if not isinstance(stage, list):
            continue
        current_stage_parallelism = 0
        for component_id_value in stage:
            component_id = str(component_id_value)
            if component_id in exact_parallelism:
                current_stage_parallelism += max(
                    1, int(exact_parallelism[component_id])
                )
            elif component_id in mapped_component_ids:
                current_stage_parallelism += 1
            else:
                current_stage_parallelism += 1
        stage_parallelism = max(stage_parallelism, current_stage_parallelism)

    fallback_parallelism = metadata.get("estimated_max_parallel_requests")
    if stage_parallelism <= 0 and isinstance(fallback_parallelism, int):
        stage_parallelism = max(stage_parallelism, fallback_parallelism)
    return _clamp_prompt_warmup_target(max(1, stage_parallelism))


def _register_exact_component_parallelism(
    payload: dict[str, Any], component_parallelism: int
) -> int:
    """Record exact mapped-component parallelism and return the refined prompt-wide warmup target."""
    prompt_id = (
        str(payload.get("prompt_id")) if payload.get("prompt_id") is not None else None
    )
    component_id = (
        str(payload.get("component_id"))
        if payload.get("component_id") is not None
        else None
    )
    if not prompt_id or not component_id:
        return _clamp_prompt_warmup_target(component_parallelism)

    with _PROMPT_WARMUP_STATES_LOCK:
        warmup_state = _ensure_prompt_warmup_state(prompt_id)
        warmup_state.exact_component_parallelism[component_id] = max(
            1, int(component_parallelism)
        )
        exact_parallelism = dict(warmup_state.exact_component_parallelism)
    return _prompt_parallelism_target(
        payload, exact_component_parallelism=exact_parallelism
    )


def _warmup_slot_payload(
    warmup_request: dict[str, Any], slot_index: int
) -> dict[str, Any]:
    """Return the remote warmup payload for one desired container slot."""
    slot_payload = copy.deepcopy(warmup_request)
    component_id = str(
        slot_payload.get("component_id")
        or slot_payload.get("prompt_id")
        or "modal-warmup"
    )
    slot_payload["component_id"] = f"{component_id}::warmup:{slot_index}"
    slot_payload["warmup_slot_index"] = int(slot_index)
    slot_payload["warmup_only"] = True
    return slot_payload


def _prompt_warmup_head_start_seconds() -> float:
    """Return the bounded delay the local scheduler may wait for exact warmup slots to start."""
    return max(0.0, float(get_settings().proactive_warmup_head_start_seconds))


def _invoke_remote_engine_warmup(
    remote_engine: Any, warmup_request: dict[str, Any]
) -> Any:
    """Ask one prepared remote engine instance to warm a container for a prompt."""
    warmup_method = getattr(remote_engine, "warmup_for_request", None)
    if warmup_method is None:
        logger.warning(
            "Remote warmup method is unavailable for prompt=%s component=%s; skipping proactive warmup.",
            warmup_request.get("prompt_id"),
            warmup_request.get("component_id"),
        )
        return None
    started_at = time.perf_counter()
    if hasattr(warmup_method, "remote"):
        result = warmup_method.remote(warmup_request)
    else:
        result = warmup_method(warmup_request)
    _record_snapshot_warmup_measurement(
        warmup_request,
        time.perf_counter() - started_at,
    )
    return result


def _invoke_remote_engine_warmup_with_recovery(
    remote_engine: Any,
    warmup_request: dict[str, Any],
) -> Any:
    """Retry one warmup call after auto-deploy when a stale deployed handle vanishes."""
    lookup_error_types = _modal_lookup_error_types()
    settings = _settings_for_payload(warmup_request)
    try:
        return _invoke_remote_engine_warmup(remote_engine, warmup_request)
    except lookup_error_types as exc:
        if not settings.auto_deploy or not _is_missing_modal_deployment_error(exc):
            raise
        logger.warning(
            "Modal warmup invocation failed for component=%s because the deployed app was missing at call time: %s. Recreating the app and retrying.",
            warmup_request.get("component_id"),
            exc,
        )
        recovered_remote_engine = _auto_deploy_modal_app(warmup_request, exc)
        _ensure_llm_profiles_staged(
            warmup_request,
            modal_deployment_app_name(settings),
        )
        return _invoke_remote_engine_warmup(recovered_remote_engine, warmup_request)


def _invoke_modal_warmup_blocking(warmup_request: dict[str, Any]) -> Any:
    """Warm one Modal container slot using deployed or ephemeral app state."""
    if modal is None:
        return None

    lookup_error_types = _modal_lookup_error_types()
    settings = _settings_for_payload(warmup_request)
    deployment_app_name = modal_deployment_app_name(settings)
    if lookup_error_types:
        try:
            remote_engine = _lookup_protocol_current_remote_engine(warmup_request)
            _ensure_llm_profiles_staged(warmup_request, deployment_app_name)
            return _invoke_remote_engine_warmup_with_recovery(
                remote_engine, warmup_request
            )
        except lookup_error_types as exc:
            if settings.auto_deploy:
                remote_engine = _auto_deploy_modal_app(warmup_request, exc)
                try:
                    _ensure_llm_profiles_staged(warmup_request, deployment_app_name)
                    return _invoke_remote_engine_warmup_with_recovery(
                        remote_engine,
                        warmup_request,
                    )
                except lookup_error_types as retry_exc:
                    exc = retry_exc
            if not settings.allow_ephemeral_fallback:
                raise ModalRemoteInvocationError(
                    "Proactive Modal warmup requires a deployed Modal app or a successful first-run auto-deploy. "
                    f"Lookup failed for app={deployment_app_name!r}: {exc}."
                ) from exc
    else:
        remote_engine = _lookup_protocol_current_remote_engine(warmup_request)
        _ensure_llm_profiles_staged(warmup_request, deployment_app_name)
        return _invoke_remote_engine_warmup_with_recovery(remote_engine, warmup_request)

    with _modal_cloud_settings_override(settings):
        cloud_module = _load_modal_cloud_module()
    cloud_app = getattr(cloud_module, "app", None)
    cloud_remote_engine = getattr(cloud_module, "RemoteEngine", None)
    if cloud_app is None or cloud_remote_engine is None:
        raise ModalRemoteInvocationError(
            "Stable Modal cloud entry module did not expose app and RemoteEngine."
        )
    run_context = cloud_app.run() if hasattr(cloud_app, "run") else nullcontext()
    with run_context:
        remote_engine = cloud_remote_engine()
        return _invoke_remote_engine_warmup_with_recovery(remote_engine, warmup_request)


def _run_prompt_warmup_slot(
    prompt_id: str,
    slot_index: int,
    warmup_request: dict[str, Any],
    reason: str,
) -> None:
    """Execute one proactive warmup slot and release it for retry on failure."""
    try:
        logger.info(
            "Starting proactive Modal warmup for prompt=%s slot=%d component=%s reason=%s.",
            prompt_id,
            slot_index,
            warmup_request.get("component_id"),
            reason,
        )
        _invoke_modal_warmup_blocking(_warmup_slot_payload(warmup_request, slot_index))
    except Exception:
        with _PROMPT_WARMUP_STATES_LOCK:
            warmup_state = _PROMPT_WARMUP_STATES.get(prompt_id)
            if warmup_state is not None:
                warmup_state.scheduled_slots.discard(slot_index)
        logger.exception(
            "Proactive Modal warmup failed for prompt=%s slot=%d component=%s.",
            prompt_id,
            slot_index,
            warmup_request.get("component_id"),
        )


def _track_prompt_warmup_future(
    prompt_id: str,
    slot_index: int,
    future: Future[Any],
) -> None:
    """Track one in-flight warmup future so mapped execution can await short exact warmup bursts."""
    with _PROMPT_WARMUP_STATES_LOCK:
        warmup_state = _PROMPT_WARMUP_STATES.get(prompt_id)
        if warmup_state is not None:
            warmup_state.slot_futures[slot_index] = future

    def _clear_tracked_future(completed_future: Future[Any]) -> None:
        """Drop one completed future from the prompt warmup state."""
        with _PROMPT_WARMUP_STATES_LOCK:
            warmup_state = _PROMPT_WARMUP_STATES.get(prompt_id)
            if warmup_state is None:
                return
            tracked_future = warmup_state.slot_futures.get(slot_index)
            if tracked_future is completed_future:
                warmup_state.slot_futures.pop(slot_index, None)

    future.add_done_callback(_clear_tracked_future)


async def _await_prompt_warmup_slots(
    prompt_id: str,
    slot_indices: list[int],
    timeout_seconds: float,
) -> int:
    """Wait briefly for any in-flight prompt warmup slots to finish."""
    if timeout_seconds <= 0.0 or not slot_indices:
        return 0

    with _PROMPT_WARMUP_STATES_LOCK:
        warmup_state = _PROMPT_WARMUP_STATES.get(prompt_id)
        if warmup_state is None:
            return 0
        pending_slot_futures = [
            warmup_state.slot_futures[slot_index]
            for slot_index in slot_indices
            if slot_index in warmup_state.slot_futures
            and not warmup_state.slot_futures[slot_index].done()
        ]

    if not pending_slot_futures:
        return 0

    wrapped_futures = [
        asyncio.wrap_future(slot_future) for slot_future in pending_slot_futures
    ]
    done_futures, pending_futures = await asyncio.wait(
        wrapped_futures,
        timeout=timeout_seconds,
        return_when=asyncio.ALL_COMPLETED,
    )
    if done_futures:
        logger.info(
            "Prompt=%s warmup head-start completed %d/%d slot(s) before mapped dispatch.",
            prompt_id,
            len(done_futures),
            len(wrapped_futures),
        )
    elif pending_futures:
        logger.info(
            "Prompt=%s warmup head-start timed out after %.3fs with %d slot(s) still warming.",
            prompt_id,
            timeout_seconds,
            len(pending_futures),
        )
    return len(done_futures)


def ensure_remote_warm_capacity(
    warmup_request: dict[str, Any] | None,
    *,
    warmup_target: int,
    reason: str,
) -> int:
    """Best-effort background warmup so enough Modal containers are ready for one prompt."""
    settings = get_settings()
    if not settings.enable_proactive_warmup:
        return 0
    if settings.execution_mode == "local" or modal is None:
        return 0
    if warmup_request is None:
        return 0

    prompt_id = _warmup_prompt_id(warmup_request)
    if prompt_id is None:
        return 0

    clamped_target = _clamp_prompt_warmup_target(warmup_target)
    if clamped_target <= 0:
        return 0

    with _PROMPT_WARMUP_STATES_LOCK:
        warmup_state = _ensure_prompt_warmup_state(prompt_id)
        missing_slots = [
            slot_index
            for slot_index in range(clamped_target)
            if slot_index not in warmup_state.scheduled_slots
        ]
        for slot_index in missing_slots:
            warmup_state.scheduled_slots.add(slot_index)

    if not missing_slots:
        return clamped_target

    logger.info(
        "Scheduling proactive Modal warmup for prompt=%s target=%d missing_slots=%s component=%s reason=%s.",
        prompt_id,
        clamped_target,
        missing_slots,
        warmup_request.get("component_id"),
        reason,
    )
    for slot_index in missing_slots:
        future = _remote_modal_warmup_executor().submit(
            _run_prompt_warmup_slot,
            prompt_id,
            slot_index,
            copy.deepcopy(warmup_request),
            reason,
        )
        _track_prompt_warmup_future(prompt_id, slot_index, future)
    return clamped_target


def _run_speculative_affinity_prewarm(
    prompt_id: str,
    warmup_identity: str,
    warmup_request: dict[str, Any],
    reason: str,
) -> None:
    """Prepare one future affinity worker and make a failed attempt retryable."""
    affinity_key = _remote_worker_affinity_key(warmup_request)
    try:
        logger.info(
            "Starting speculative Modal prewarm prompt=%s affinity=%s component=%s reason=%s.",
            prompt_id,
            affinity_key,
            warmup_request.get("component_id"),
            reason,
        )
        _invoke_modal_warmup_blocking(_warmup_slot_payload(warmup_request, 0))
        logger.info(
            "Completed speculative Modal prewarm prompt=%s affinity=%s component=%s.",
            prompt_id,
            affinity_key,
            warmup_request.get("component_id"),
        )
    except Exception:
        with _PROMPT_WARMUP_STATES_LOCK:
            warmup_state = _PROMPT_WARMUP_STATES.get(prompt_id)
            if warmup_state is not None:
                warmup_state.scheduled_speculative_affinities.discard(
                    warmup_identity
                )
        logger.exception(
            "Speculative Modal prewarm failed prompt=%s affinity=%s component=%s.",
            prompt_id,
            affinity_key,
            warmup_request.get("component_id"),
        )


def _schedule_speculative_affinity_prewarm(
    payload: Mapping[str, Any],
    *,
    reason: str,
) -> bool:
    """Schedule one planner-selected future worker without blocking current execution."""
    target_payload = payload.get(_SPECULATIVE_PREWARM_TARGET_KEY)
    if not isinstance(target_payload, Mapping):
        return False
    if (
        payload.get("execution_provider") != "modal"
        or target_payload.get("execution_provider") != "modal"
        or payload.get("execution_environment_id")
        != target_payload.get("execution_environment_id")
    ):
        logger.debug(
            "Skipping speculative Modal prewarm across execution environments "
            "source=%s target=%s.",
            payload.get("execution_environment_id"),
            target_payload.get("execution_environment_id"),
        )
        return False

    normalized_target_payload = copy.deepcopy(dict(target_payload))
    settings = _settings_for_payload(normalized_target_payload)
    if (
        not settings.enable_proactive_warmup
        or settings.execution_mode == "local"
        or modal is None
        or (settings.max_containers is not None and settings.max_containers < 2)
    ):
        return False

    warmup_request = _build_prompt_warmup_request(normalized_target_payload)
    if warmup_request is None:
        return False
    prompt_id = _warmup_prompt_id(warmup_request)
    if prompt_id is None:
        return False

    current_affinity_key = _remote_worker_affinity_key(dict(payload))
    target_affinity_key = _remote_worker_affinity_key(warmup_request)
    if target_affinity_key == current_affinity_key:
        return False
    snapshot_profile_key = _prepare_snapshot_profile_fields(warmup_request)
    _select_gpu_snapshot_for_profile(warmup_request, snapshot_profile_key)
    warmup_identity = _speculative_warmup_identity(warmup_request)

    with _PROMPT_WARMUP_STATES_LOCK:
        warmup_state = _ensure_prompt_warmup_state(prompt_id)
        if warmup_identity in warmup_state.scheduled_speculative_affinities:
            return False
        warmup_state.scheduled_speculative_affinities.add(warmup_identity)

    logger.info(
        "Scheduling speculative Modal prewarm prompt=%s current_component=%s current_affinity=%s target_component=%s target_affinity=%s reason=%s.",
        prompt_id,
        payload.get("component_id"),
        current_affinity_key,
        warmup_request.get("component_id"),
        target_affinity_key,
        reason,
    )
    future = _remote_modal_warmup_executor().submit(
        _run_speculative_affinity_prewarm,
        prompt_id,
        warmup_identity,
        copy.deepcopy(warmup_request),
        reason,
    )
    with _PROMPT_WARMUP_STATES_LOCK:
        warmup_state = _PROMPT_WARMUP_STATES.get(prompt_id)
        if warmup_state is not None:
            warmup_state.speculative_affinity_futures[warmup_identity] = future

    def _clear_speculative_future(completed_future: Future[Any]) -> None:
        """Drop a completed speculative future while retaining successful dedupe state."""
        with _PROMPT_WARMUP_STATES_LOCK:
            warmup_state = _PROMPT_WARMUP_STATES.get(prompt_id)
            if warmup_state is None:
                return
            tracked_future = warmup_state.speculative_affinity_futures.get(
                warmup_identity
            )
            if tracked_future is completed_future:
                warmup_state.speculative_affinity_futures.pop(
                    warmup_identity, None
                )

    future.add_done_callback(_clear_speculative_future)
    return True


def _speculative_warmup_identity(payload: Mapping[str, Any]) -> str:
    """Return the exact deployed-worker identity shared by warmup and dispatch."""
    settings = _settings_for_payload(payload)
    identity = {
        "app": modal_deployment_app_name(settings),
        "gpu": settings.modal_gpu,
        "affinity": _remote_worker_affinity_key(dict(payload)),
        "snapshot_profile": str(payload.get("snapshot_profile_key") or ""),
        "snapshot_variant": str(payload.get("gpu_snapshot_variant") or "direct"),
    }
    return json.dumps(identity, sort_keys=True, separators=(",", ":"))


def _await_matching_speculative_prewarm(
    payload: dict[str, Any],
    cancellation_event: threading.Event | None,
) -> None:
    """Join an active speculative warmup instead of racing a duplicate container."""
    prompt_id = _warmup_prompt_id(payload)
    if prompt_id is None:
        return
    snapshot_profile_key = _prepare_snapshot_profile_fields(payload)
    _select_gpu_snapshot_for_profile(payload, snapshot_profile_key)
    warmup_identity = _speculative_warmup_identity(payload)
    with _PROMPT_WARMUP_STATES_LOCK:
        warmup_state = _PROMPT_WARMUP_STATES.get(prompt_id)
        future = (
            warmup_state.speculative_affinity_futures.get(warmup_identity)
            if warmup_state is not None
            else None
        )
    if future is None or future.done():
        return
    started_at = time.perf_counter()
    logger.info(
        "Joining active speculative Modal prewarm before dispatch prompt=%s component=%s identity=%s.",
        prompt_id,
        payload.get("component_id"),
        warmup_identity,
    )
    while not future.done():
        if cancellation_event is not None and cancellation_event.is_set():
            raise ModalRemoteInvocationError(
                "Remote dispatch was cancelled while waiting for speculative warmup."
            )
        try:
            future.result(timeout=0.1)
        except FutureTimeoutError:
            continue
    future.result()
    logger.info(
        "Joined speculative Modal prewarm before dispatch prompt=%s component=%s wait_seconds=%.3f.",
        prompt_id,
        payload.get("component_id"),
        time.perf_counter() - started_at,
    )


def _schedule_post_deploy_runtime_seed(payload: Mapping[str, Any]) -> bool:
    """Seed the just-deployed profile and expose its future to the first dispatch."""
    warmup_request = _build_prompt_warmup_request(copy.deepcopy(dict(payload)))
    if warmup_request is None:
        return False
    prompt_id = _warmup_prompt_id(warmup_request)
    if prompt_id is None:
        return False
    snapshot_profile_key = _prepare_snapshot_profile_fields(warmup_request)
    _select_gpu_snapshot_for_profile(warmup_request, snapshot_profile_key)
    warmup_identity = _speculative_warmup_identity(warmup_request)
    with _PROMPT_WARMUP_STATES_LOCK:
        warmup_state = _ensure_prompt_warmup_state(prompt_id)
        if warmup_identity in warmup_state.scheduled_speculative_affinities:
            return False
        warmup_state.scheduled_speculative_affinities.add(warmup_identity)
        future = _remote_modal_warmup_executor().submit(
            _run_speculative_affinity_prewarm,
            prompt_id,
            warmup_identity,
            copy.deepcopy(warmup_request),
            "post_deploy_runtime_seed",
        )
        warmup_state.speculative_affinity_futures[warmup_identity] = future

    def clear_seed_future(completed_future: Future[Any]) -> None:
        """Forget the completed seed future while retaining successful dedupe state."""
        with _PROMPT_WARMUP_STATES_LOCK:
            current_state = _PROMPT_WARMUP_STATES.get(prompt_id)
            if current_state is None:
                return
            if (
                current_state.speculative_affinity_futures.get(warmup_identity)
                is completed_future
            ):
                current_state.speculative_affinity_futures.pop(
                    warmup_identity, None
                )

    future.add_done_callback(clear_seed_future)
    logger.info(
        "Scheduled automatic post-deployment Modal runtime seed prompt=%s component=%s identity=%s.",
        prompt_id,
        warmup_request.get("component_id"),
        warmup_identity,
    )
    return True


def _local_gap_keepalive_key(payload: Mapping[str, Any]) -> tuple[str, int] | None:
    """Return the prompt and affinity-slot key for a local-gap payload."""
    prompt_id = _warmup_prompt_id(dict(payload))
    if prompt_id is None:
        return None
    return (prompt_id, _component_pool_slot_index(dict(payload)))


def _stop_local_gap_keepalive(payload: Mapping[str, Any], *, reason: str) -> bool:
    """Stop a previously scheduled affinity-slot keepalive, if one exists."""
    keepalive_key = _local_gap_keepalive_key(payload)
    if keepalive_key is None:
        return False
    with _LOCAL_GAP_KEEPALIVES_LOCK:
        keepalive_state = _LOCAL_GAP_KEEPALIVES.pop(keepalive_key, None)
    if keepalive_state is None:
        return False
    keepalive_state.stop_event.set()
    logger.info(
        "Stopping remote local-gap keepalive prompt=%s slot=%d after component=%s reason=%s.",
        keepalive_key[0],
        keepalive_key[1],
        keepalive_state.component_id,
        reason,
    )
    return True


def _invoke_remote_local_gap_keepalive(payload: dict[str, Any]) -> Any:
    """Send one lightweight activity pulse to the payload's affinity slot."""
    remote_engine = _lookup_deployed_remote_engine(payload)
    keepalive_method = getattr(remote_engine, "keepalive_for_local_gap", None)
    if keepalive_method is None:
        raise ModalRemoteInvocationError(
            "The deployed RemoteEngine does not expose keepalive_for_local_gap."
        )
    return _call_modal_method(keepalive_method, payload)


def _run_local_gap_keepalive(
    payload: dict[str, Any],
    stop_event: threading.Event,
    duration_seconds: float,
    interval_seconds: float,
) -> None:
    """Pulse one remote affinity slot until local execution finishes or the budget expires."""
    deadline = time.monotonic() + duration_seconds
    keepalive_count = 0
    while not stop_event.is_set():
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0.0:
            break
        if stop_event.wait(min(interval_seconds, remaining_seconds)):
            break
        keepalive_count += 1
        pulse_started_at = time.perf_counter()
        _invoke_remote_local_gap_keepalive(payload)
        logger.info(
            "Completed remote local-gap keepalive pulse=%d prompt=%s component=%s in %.3fs.",
            keepalive_count,
            payload.get("prompt_id"),
            payload.get("component_id"),
            time.perf_counter() - pulse_started_at,
        )
    logger.info(
        "Remote local-gap keepalive loop finished prompt=%s component=%s pulses=%d stopped=%s.",
        payload.get("prompt_id"),
        payload.get("component_id"),
        keepalive_count,
        stop_event.is_set(),
    )


def _start_local_gap_keepalive(payload: Mapping[str, Any]) -> bool:
    """Retain one remote affinity slot while a downstream local gap executes."""
    if (
        payload.get("execution_provider") != "modal"
        or not bool(payload.get("keepalive_after_remote_component"))
    ):
        return False
    settings = _settings_for_payload(payload)
    duration_seconds = max(0.0, float(settings.local_gap_keepalive_seconds))
    if duration_seconds <= 0.0 or settings.execution_mode == "local" or modal is None:
        return False
    keepalive_key = _local_gap_keepalive_key(payload)
    if keepalive_key is None:
        return False

    _stop_local_gap_keepalive(payload, reason="replacement")
    stop_event = threading.Event()
    keepalive_payload = dict(payload)
    keepalive_payload["warmup_slot_index"] = keepalive_key[1]
    keepalive_payload["component_id"] = (
        f"{payload.get('component_id', 'modal-component')}::local-gap-keepalive"
    )
    interval_seconds = min(
        duration_seconds,
        max(1.0, float(settings.local_gap_keepalive_interval_seconds)),
    )
    future = _remote_modal_keepalive_executor().submit(
        _run_local_gap_keepalive,
        keepalive_payload,
        stop_event,
        duration_seconds,
        interval_seconds,
    )
    keepalive_state = _LocalGapKeepaliveState(
        component_id=str(payload.get("component_id") or "modal-component"),
        stop_event=stop_event,
        future=future,
    )
    with _LOCAL_GAP_KEEPALIVES_LOCK:
        _LOCAL_GAP_KEEPALIVES[keepalive_key] = keepalive_state

    def clear_completed_keepalive(completed_future: Future[Any]) -> None:
        """Forget a finished keepalive and report any best-effort failure."""
        with _LOCAL_GAP_KEEPALIVES_LOCK:
            current_state = _LOCAL_GAP_KEEPALIVES.get(keepalive_key)
            if current_state is keepalive_state:
                _LOCAL_GAP_KEEPALIVES.pop(keepalive_key, None)
        failure = completed_future.exception()
        if failure is not None:
            logger.warning(
                "Remote local-gap keepalive failed prompt=%s slot=%d component=%s: %s",
                keepalive_key[0],
                keepalive_key[1],
                keepalive_state.component_id,
                failure,
            )

    future.add_done_callback(clear_completed_keepalive)
    logger.info(
        "Started remote local-gap keepalive prompt=%s slot=%d component=%s duration=%.1fs interval=%.1fs.",
        keepalive_key[0],
        keepalive_key[1],
        keepalive_state.component_id,
        duration_seconds,
        interval_seconds,
    )
    return True


def boost_mapped_component_warmup(
    payload: dict[str, Any],
    *,
    total_items: int,
    reason: str,
) -> tuple[int, int]:
    """Record exact mapped fan-out and top up prompt warmup for the resulting lane count."""
    parallelism = _mapped_execution_parallelism(total_items)
    refined_prompt_warmup_target = _register_exact_component_parallelism(
        payload, parallelism
    )
    ensure_remote_warm_capacity(
        _build_prompt_warmup_request(payload),
        warmup_target=refined_prompt_warmup_target,
        reason=reason,
    )
    logger.info(
        "Boosted exact mapped warmup for component=%s total_items=%d local_parallelism=%d prompt_warmup_target=%d reason=%s.",
        payload.get("component_id"),
        total_items,
        parallelism,
        refined_prompt_warmup_target,
        reason,
    )
    return parallelism, refined_prompt_warmup_target




