"""Host-side Modal LLM profile staging and staged-reference rewriting."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
import queue
import threading
from typing import Any, Iterable, Iterator, Mapping

from ..llm_profiles import (
    get_llm_profile,
    llm_model_reference_node_ids_from_payload,
    llm_model_references_from_payload,
    resolved_llm_profile_payloads,
    rewrite_llm_model_references,
)
from ..serialization import deserialize_node_inputs, serialize_node_inputs
from ..staging_process import staging_no_progress_timeout_seconds
from .local_ui_events import (
    _emit_local_modal_progress,
    _emit_local_remote_startup_status,
)
from .modal_deployment import ModalRemoteInvocationError

logger = logging.getLogger(__name__)

try:
    import modal  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - local fallback environments.
    modal = None

_STAGED_LLM_PROFILES_LOCK = threading.Lock()
_STAGED_LLM_PROFILES: set[tuple[str, str, str]] = set()
_STAGED_LLM_PROFILE_RESULTS: dict[tuple[str, str], dict[str, Any]] = {}
_MODAL_STAGE_STREAM_END = object()


@dataclass(frozen=True)
class _ModalStageStreamFailure:
    """Carry an arbitrary Modal stream exception across a reader thread."""

    error: Exception


def _emit_local_llm_staging_progress(
    payload: Mapping[str, Any],
    stage_event: Mapping[str, Any],
) -> None:
    """Render one CPU ModelStager update on its actual LLM node bars."""
    prompt_id = (
        str(payload["prompt_id"]) if payload.get("prompt_id") is not None else None
    )
    extra_data = payload.get("extra_data") or {}
    client_id = (
        str(extra_data["client_id"])
        if isinstance(extra_data, Mapping) and extra_data.get("client_id") is not None
        else None
    )
    node_ids_by_reference = llm_model_reference_node_ids_from_payload(payload)
    model_reference = str(stage_event.get("model_reference") or "").strip()
    node_ids = node_ids_by_reference.get(model_reference, ())
    if not node_ids:
        node_ids = tuple(
            sorted(
                {
                    node_id
                    for reference_node_ids in node_ids_by_reference.values()
                    for node_id in reference_node_ids
                }
            )
        )
    if not node_ids:
        component_id = str(payload.get("component_id") or "")
        node_ids = (component_id,) if component_id else ()
    if not node_ids:
        return
    maximum = stage_event.get("max")
    for node_id in node_ids:
        _emit_local_modal_progress(
            prompt_id=prompt_id,
            client_id=client_id,
            node_id=node_id,
            value=float(stage_event.get("value") or 0.0),
            max_value=float(maximum) if maximum is not None else 1.0,
            stage=str(stage_event.get("stage") or "staging"),
            message=str(stage_event.get("message") or "Staging LLM snapshot"),
            unit=(
                str(stage_event["unit"])
                if stage_event.get("unit") is not None
                else None
            ),
            indeterminate=bool(stage_event.get("indeterminate", False)),
            pre_gpu=True,
        )


def _read_modal_stage_events(
    stage_events: Iterable[Any],
    output: queue.Queue[Any],
) -> None:
    """Read a blocking Modal generator while exposing controller timeouts."""
    try:
        for event in stage_events:
            output.put(event)
    except Exception as exc:
        output.put(_ModalStageStreamFailure(exc))
    finally:
        output.put(_MODAL_STAGE_STREAM_END)


def _close_modal_stage_events(stage_events: Iterable[Any]) -> None:
    """Ask Modal to cancel a stage generator that stopped reporting progress."""
    close = getattr(stage_events, "close", None)
    if not callable(close):
        return
    try:
        close()
    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning("Unable to close stalled Modal model staging stream: %s", exc)


def _bounded_modal_stage_events(stage_events: Iterable[Any]) -> Iterator[Any]:
    """Yield Modal staging events with a bounded interval between updates."""
    try:
        timeout_seconds = staging_no_progress_timeout_seconds()
    except ValueError:
        _close_modal_stage_events(stage_events)
        raise
    output: queue.Queue[Any] = queue.Queue()
    reader = threading.Thread(
        target=_read_modal_stage_events,
        args=(stage_events, output),
        name="modal-llm-stage-stream",
        daemon=True,
    )
    reader.start()
    try:
        while True:
            try:
                item = output.get(timeout=timeout_seconds)
            except queue.Empty as exc:
                _close_modal_stage_events(stage_events)
                raise ModalRemoteInvocationError(
                    "Modal model staging produced no progress for "
                    f"{timeout_seconds:.0f} seconds; the staging call was cancelled."
                ) from exc
            if item is _MODAL_STAGE_STREAM_END:
                return
            if isinstance(item, _ModalStageStreamFailure):
                raise item.error
            yield item
    finally:
        reader.join(timeout=1.0)


def _ensure_llm_profiles_staged(
    payload: dict[str, Any],
    deployment_app_name: str,
) -> None:
    """Stage every LLM profile in a payload on a CPU worker before GPU dispatch."""
    if modal is None:
        raise ModalRemoteInvocationError("Modal SDK is unavailable.")
    model_references = llm_model_references_from_payload(payload)
    if not model_references:
        return
    with _STAGED_LLM_PROFILES_LOCK:
        if not _STAGED_LLM_PROFILES:
            _STAGED_LLM_PROFILE_RESULTS.clear()
        missing_model_references = [
            reference
            for reference in model_references
            if (deployment_app_name, reference) not in _STAGED_LLM_PROFILE_RESULTS
        ]
        if missing_model_references:
            resolved_profiles = resolved_llm_profile_payloads(
                payload,
                missing_model_references,
            )
            _emit_local_remote_startup_status(
                payload,
                phase="llm_staging",
                status_message=(
                    "Preparing LLM model snapshots on CPU; no GPU is allocated yet"
                ),
            )
            logger.info(
                "Dispatching CPU model resolution/staging app=%s models=%s "
                "before GPU component=%s.",
                deployment_app_name,
                missing_model_references,
                payload.get("component_id"),
            )
            stager_cls = modal.Cls.from_name(deployment_app_name, "ModelStager")
            stager = stager_cls()
            stage_results: list[dict[str, Any]] = []
            stage_stream = getattr(stager, "stage_profiles_stream", None)
            remote_generator = getattr(stage_stream, "remote_gen", None)
            if callable(remote_generator):
                stage_events = (
                    remote_generator(missing_model_references, resolved_profiles)
                    if resolved_profiles
                    else remote_generator(missing_model_references)
                )
                for stage_event in _bounded_modal_stage_events(stage_events):
                    if not isinstance(stage_event, Mapping):
                        continue
                    if stage_event.get("kind") == "result":
                        candidate_results = stage_event.get("results")
                        if isinstance(candidate_results, list):
                            stage_results = candidate_results
                        continue
                    if stage_event.get("kind") != "progress":
                        continue
                    _emit_local_llm_staging_progress(payload, stage_event)
            else:
                stage_results = (
                    stager.stage_profiles.remote(
                        missing_model_references,
                        resolved_profiles,
                    )
                    if resolved_profiles
                    else stager.stage_profiles.remote(missing_model_references)
                )
            confirmed_references: set[str] = set()
            for stage_result in stage_results:
                if not isinstance(stage_result, Mapping):
                    continue
                requested_reference = str(
                    stage_result.get("requested_reference")
                    or stage_result.get("profile_id")
                    or ""
                )
                profile_id = str(stage_result.get("profile_id") or "")
                revision = str(stage_result.get("revision") or "")
                if not revision and requested_reference == profile_id:
                    try:
                        revision = get_llm_profile(profile_id).revision
                    except ValueError:
                        pass
                if not requested_reference or not profile_id or not revision:
                    continue
                normalized_result = dict(stage_result)
                normalized_result["requested_reference"] = requested_reference
                normalized_result["profile_id"] = profile_id
                normalized_result["revision"] = revision
                _STAGED_LLM_PROFILE_RESULTS[
                    (deployment_app_name, requested_reference)
                ] = normalized_result
                _STAGED_LLM_PROFILE_RESULTS[
                    (deployment_app_name, profile_id)
                ] = normalized_result
                _STAGED_LLM_PROFILES.add((deployment_app_name, profile_id, revision))
                confirmed_references.add(requested_reference)
            missing_results = set(missing_model_references) - confirmed_references
            if missing_results:
                raise ModalRemoteInvocationError(
                    f"Modal ModelStager did not confirm models {sorted(missing_results)}."
                )
            downloaded_gib = sum(
                float(result.get("artifact_bytes") or 0) / 1024**3
                for result in stage_results
                if isinstance(result, Mapping) and result.get("downloaded")
            )
            _emit_local_remote_startup_status(
                payload,
                phase="llm_staged",
                status_message=(
                    f"LLM staging complete ({downloaded_gib:.1f} GiB downloaded); "
                    "starting GPU worker"
                ),
            )
        resolved_results = {
            reference: _STAGED_LLM_PROFILE_RESULTS[(deployment_app_name, reference)]
            for reference in model_references
        }
    profile_ids_by_reference = {
        reference: str(result["profile_id"])
        for reference, result in resolved_results.items()
    }
    rewrite_llm_model_references(payload, profile_ids_by_reference)
    revisions = ",".join(
        f"{result['profile_id']}:{result['revision']}"
        for result in sorted(
            resolved_results.values(),
            key=lambda value: str(value["profile_id"]),
        )
    )
    payload["requires_volume_reload"] = True
    payload["volume_reload_marker"] = hashlib.sha256(
        f"llm-profiles:{revisions}".encode("utf-8")
    ).hexdigest()
    logger.info(
        "Modal LLM models are resolved and staged for component=%s profiles=%s "
        "reload_marker=%s.",
        payload.get("component_id"),
        sorted(profile_ids_by_reference.values()),
        payload["volume_reload_marker"],
    )


def _rewrite_staged_llm_kwargs_payload(
    kwargs_payload: bytes,
    deployment_app_name: str,
) -> bytes:
    """Replace a direct node input model reference with its staged profile ID."""
    hydrated_inputs = deserialize_node_inputs(kwargs_payload)
    if not isinstance(hydrated_inputs, Mapping):
        return kwargs_payload
    model_reference = hydrated_inputs.get("model_profile")
    if not isinstance(model_reference, str) or not model_reference.strip():
        return kwargs_payload
    normalized_reference = model_reference.strip()
    with _STAGED_LLM_PROFILES_LOCK:
        stage_result = _STAGED_LLM_PROFILE_RESULTS.get(
            (deployment_app_name, normalized_reference)
        )
    if stage_result is None:
        return kwargs_payload
    profile_id = str(stage_result.get("profile_id") or "").strip()
    if not profile_id or profile_id == normalized_reference:
        return kwargs_payload
    rewritten_inputs = dict(hydrated_inputs)
    rewritten_inputs["model_profile"] = profile_id
    logger.info(
        "Rewrote direct Modal LLM input model=%s to generated profile=%s.",
        normalized_reference,
        profile_id,
    )
    return serialize_node_inputs(rewritten_inputs)



