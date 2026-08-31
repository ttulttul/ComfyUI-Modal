"""Remote prompt analysis, preparation cancellation, and queue dispatch routes."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from aiohttp import web

if __package__:
    from .modal_executor_node import MODAL_PROMPT_ID_EXTRA_PNGINFO_KEY
    from .queue_bridge import (
        _cancel_remote_preparation,
        _clear_remote_preparation,
        _queue_prompt_json,
        _queued_ssh_environment_ids,
        _set_remote_preparation,
    )
    from .remote_configuration_nodes import compile_remote_configuration_set
    from .remote_graph_analysis import (
        analyze_remote_node_selection,
        requested_remote_node_ids,
    )
    from .remote_plan_types import ModalPromptValidationError, RewriteSummary
    from .route_context import RouteContext
    from .settings import (
        ModalSyncSettings,
        modal_gpu_from_workflow,
        settings_for_modal_gpu,
    )
    from .subrosa_login import (
        SubrosaConfigurationValidationError,
        preflight_subrosa_configurations,
    )
    from .sync_engine import (
        SyncCancelledError,
        begin_r2_writeback_prompt,
        finish_r2_writeback_prompt,
    )
else:  # pragma: no cover - flat import inside the Modal container.
    from modal_executor_node import MODAL_PROMPT_ID_EXTRA_PNGINFO_KEY
    from queue_bridge import (
        _cancel_remote_preparation,
        _clear_remote_preparation,
        _queue_prompt_json,
        _queued_ssh_environment_ids,
        _set_remote_preparation,
    )
    from remote_configuration_nodes import compile_remote_configuration_set
    from remote_graph_analysis import (
        analyze_remote_node_selection,
        requested_remote_node_ids,
    )
    from remote_plan_types import ModalPromptValidationError, RewriteSummary
    from route_context import RouteContext
    from settings import (
        ModalSyncSettings,
        modal_gpu_from_workflow,
        settings_for_modal_gpu,
    )
    from subrosa_login import (
        SubrosaConfigurationValidationError,
        preflight_subrosa_configurations,
    )
    from sync_engine import (
        SyncCancelledError,
        begin_r2_writeback_prompt,
        finish_r2_writeback_prompt,
    )

logger = logging.getLogger(__name__)


@dataclass
class _QueueRequestState:
    """Track one queue request across validation, rewrite, and dispatch stages."""

    json_data: dict[str, Any] | None = None
    workflow: dict[str, Any] | None = None
    remote_node_ids: list[str] = field(default_factory=list)
    request_modal_gpu: str | None = None
    summary: RewriteSummary = field(default_factory=RewriteSummary)
    preparation_prompt_id: str | None = None
    preparation_cancellation: threading.Event = field(default_factory=threading.Event)
    configurator_node_id: str | None = None
    r2_writeback_prompt_id: str | None = None
    prompt_queued: bool = False
    request_settings: ModalSyncSettings | None = None
    client_id: str | None = None
    prompt_id: str | None = None
    status_modal_gpu: str | None = None
    selected_modal_gpus: list[str] = field(default_factory=list)


async def _initialize_queue_request(
    request: web.Request,
    ctx: RouteContext,
    state: _QueueRequestState,
) -> None:
    """Parse request JSON and resolve workflow-scoped selection metadata."""
    json_data = await request.json()
    json_data.setdefault("prompt_id", str(uuid.uuid4()))
    json_data.setdefault("extra_data", {})
    json_data["extra_data"]["prompt_id"] = json_data["prompt_id"]
    if json_data.get("client_id") is not None:
        json_data["extra_data"]["client_id"] = json_data["client_id"]
    state.json_data = json_data
    state.client_id = str(json_data.get("client_id")) if json_data.get("client_id") else None
    state.prompt_id = str(json_data.get("prompt_id")) if json_data.get("prompt_id") else None
    extra_pnginfo = (json_data.get("extra_data") or {}).get("extra_pnginfo") or {}
    state.workflow = extra_pnginfo.get("workflow")
    state.remote_node_ids = sorted(
        requested_remote_node_ids(
            prompt=json_data.get("prompt", {}),
            workflow=state.workflow,
            settings=ctx.settings,
        )
    )
    state.configurator_node_id = ctx.configurator_node_id(json_data.get("prompt", {}))


def _reserve_queue_preparation(
    prompt_server: Any,
    state: _QueueRequestState,
) -> None:
    """Reserve R2 and queue-visible preparation state for one remote prompt."""
    if state.prompt_id is None or state.json_data is None:
        return
    begin_r2_writeback_prompt(state.prompt_id)
    state.r2_writeback_prompt_id = state.prompt_id
    extra_pnginfo = (state.json_data.get("extra_data") or {}).get("extra_pnginfo") or {}
    extra_pnginfo[MODAL_PROMPT_ID_EXTRA_PNGINFO_KEY] = state.prompt_id
    state.json_data["extra_data"]["extra_pnginfo"] = extra_pnginfo
    logger.debug(
        "Attached prompt-scoped Modal execution metadata prompt_id=%s.",
        state.prompt_id,
    )
    if _set_remote_preparation(
        prompt_server,
        prompt_id=state.prompt_id,
        prompt=state.json_data.get("prompt", {}),
        extra_data=state.json_data.get("extra_data", {}),
        cancellation_event=state.preparation_cancellation,
    ):
        state.preparation_prompt_id = state.prompt_id


def _resolve_queue_settings(ctx: RouteContext, state: _QueueRequestState) -> None:
    """Resolve workflow GPU settings and status identity for one request."""
    assert state.json_data is not None
    try:
        state.request_settings = settings_for_modal_gpu(
            ctx.settings,
            modal_gpu_from_workflow(state.workflow, ctx.settings.modal_gpu),
        )
    except ValueError as exc:
        raise ModalPromptValidationError(str(exc)) from exc
    logger.info(
        "Resolved workflow Modal GPU selection gpu=%s prompt_id=%s.",
        state.request_settings.modal_gpu,
        state.prompt_id,
    )
    state.status_modal_gpu = (
        None
        if ctx.prompt_uses_configurator(state.json_data.get("prompt", {}))
        else state.request_settings.modal_gpu
    )
    state.request_modal_gpu = state.status_modal_gpu


async def _preflight_queue_credentials(state: _QueueRequestState) -> None:
    """Validate connected provider credentials before prompt rewriting or execution."""
    assert state.json_data is not None
    try:
        configuration_set = compile_remote_configuration_set(
            state.json_data.get("prompt", {})
        )
    except (TypeError, ValueError) as exc:
        raise ModalPromptValidationError(str(exc)) from exc
    if configuration_set is not None:
        await preflight_subrosa_configurations(configuration_set)


def _emit_setup_status(
    prompt_server: Any,
    ctx: RouteContext,
    state: _QueueRequestState,
    message: str,
    current: int | None = None,
    total: int | None = None,
    *,
    environment_id: str | None = None,
) -> None:
    """Publish one cancellable queue-time setup event."""
    if state.preparation_cancellation.is_set():
        raise SyncCancelledError("Remote workflow preparation was cancelled.")
    ctx.emit_status(
        prompt_server=prompt_server,
        phase="setup",
        client_id=state.client_id,
        prompt_id=state.prompt_id,
        node_ids=state.remote_node_ids,
        configurator_node_id=state.configurator_node_id,
        modal_gpu=state.status_modal_gpu,
        component_node_ids_by_representative=(
            state.summary.component_node_ids_by_representative or None
        ),
        status_message=message,
        status_current=current,
        status_total=total,
        **(
            {"execution_environment_id": environment_id}
            if environment_id is not None
            else {}
        ),
    )


def _emit_execution_plan(
    prompt_server: Any,
    ctx: RouteContext,
    state: _QueueRequestState,
    assignments: dict[str, dict[str, Any]],
    configurations: list[dict[str, Any]],
) -> None:
    """Publish scheduler choices before capacity acquisition can block."""
    component_nodes = {
        component_id: list(assignment.get("node_ids", []))
        for component_id, assignment in assignments.items()
    }
    ctx.emit_status(
        prompt_server=prompt_server,
        phase="setup",
        client_id=state.client_id,
        prompt_id=state.prompt_id,
        node_ids=state.remote_node_ids,
        configurator_node_id=state.configurator_node_id,
        modal_gpu=state.status_modal_gpu,
        component_node_ids_by_representative=component_nodes,
        status_message="Remote execution plan ready",
        remote_execution_assignments=assignments,
        remote_execution_configurations=configurations,
    )


def _store_rewrite_metadata(ctx: RouteContext, state: _QueueRequestState) -> None:
    """Attach rewrite and execution-plan metadata to the queued request."""
    assert state.json_data is not None
    assert state.request_settings is not None
    modal_extra = state.json_data.setdefault("extra_data", {}).setdefault("modal", {})
    modal_extra.update(
        {
            "gpu": state.request_settings.modal_gpu,
            "remote_node_ids": state.summary.remote_node_ids,
            "remote_component_ids": state.summary.remote_component_ids,
            "component_dependency_ids_by_representative": state.summary.component_dependency_ids_by_representative,
            "component_execution_stages": state.summary.component_execution_stages,
            "mapped_component_ids": state.summary.mapped_component_ids,
            "estimated_max_parallel_requests": state.summary.estimated_max_parallel_requests,
            "max_parallel_requests_upper_bound": state.summary.max_parallel_requests_upper_bound,
            "synced_assets": [asset.remote_path for asset in state.summary.synced_assets],
        }
    )
    state.json_data["extra_data"]["remote_execution"] = {
        "assignments": ctx.execution_assignments_payload(
            state.summary,
            state.request_settings,
        ),
        "configurations": list(state.summary.remote_configurations),
    }
    if state.summary.custom_nodes_bundle is not None:
        modal_extra["custom_nodes_bundle"] = state.summary.custom_nodes_bundle.remote_path


async def _rewrite_remote_prompt(
    prompt_server: Any,
    ctx: RouteContext,
    state: _QueueRequestState,
) -> None:
    """Analyze, prepare capacity, rewrite the prompt, and attach dispatch metadata."""
    assert state.json_data is not None
    assert state.request_settings is not None
    _emit_setup_status(prompt_server, ctx, state, "Preparing remote workflow")
    rewrite_started_at = time.perf_counter()
    occupied_environment_ids = _queued_ssh_environment_ids(
        prompt_server,
        excluding_prompt_id=state.prompt_id,
    )
    if occupied_environment_ids:
        logger.info(
            "Queued workflow may reuse SSH capacity after earlier prompts finish environments=%s.",
            sorted(occupied_environment_ids),
        )
    rewritten_prompt, state.summary = await ctx.rewrite_prompt(
        prompt=state.json_data["prompt"],
        workflow=state.workflow,
        sync_engine=ctx.sync_engine,
        settings=state.request_settings,
        extra_data=state.json_data.get("extra_data"),
        status_callback=lambda message, current=None, total=None: _emit_setup_status(
            prompt_server, ctx, state, message, current, total
        ),
        environment_status_callback=lambda environment_id, message, current=None, total=None: _emit_setup_status(
            prompt_server,
            ctx,
            state,
            message,
            current,
            total,
            environment_id=environment_id,
        ),
        plan_callback=lambda assignments, configurations: _emit_execution_plan(
            prompt_server, ctx, state, assignments, configurations
        ),
        cancellation_check=state.preparation_cancellation.is_set,
        occupied_environment_ids=occupied_environment_ids,
    )
    state.selected_modal_gpus = ctx.selected_modal_gpus(
        state.summary,
        state.request_settings.modal_gpu,
    )
    state.status_modal_gpu = (
        state.selected_modal_gpus[0] if len(state.selected_modal_gpus) == 1 else None
    )
    state.request_modal_gpu = state.status_modal_gpu
    logger.info(
        "Modal prompt rewrite finished in %.3fs for %d remote nodes across %d components.",
        time.perf_counter() - rewrite_started_at,
        len(state.summary.remote_node_ids),
        len(state.summary.remote_component_ids),
    )
    state.remote_node_ids = list(state.summary.remote_node_ids)
    state.json_data["prompt"] = rewritten_prompt
    if state.json_data.get("partial_execution_targets"):
        rewritten_targets = {
            state.summary.rewritten_node_id_map.get(str(target), str(target))
            for target in state.json_data["partial_execution_targets"]
        }
        state.json_data["partial_execution_targets"] = sorted(rewritten_targets)
    _store_rewrite_metadata(ctx, state)
    _emit_setup_status(prompt_server, ctx, state, "Submitting remote workflow")


def _modal_queue_response_payload(
    ctx: RouteContext,
    state: _QueueRequestState,
) -> dict[str, Any] | None:
    """Build the Modal-specific successful queue response payload."""
    if not state.summary.remote_node_ids:
        return None
    assert state.request_settings is not None
    return {
        "modal_gpu": state.status_modal_gpu,
        "remote_execution_configurator_node_id": state.configurator_node_id,
        "remote_execution_modal_gpus": state.selected_modal_gpus,
        "modal_remote_node_ids": list(state.summary.remote_node_ids),
        "modal_sandwiched_local_node_ids": list(state.summary.sandwiched_local_node_ids),
        "modal_parallel_local_branch_node_ids": list(
            state.summary.parallel_local_branch_node_ids
        ),
        "remote_execution_assignments": ctx.execution_assignments_payload(
            state.summary,
            state.request_settings,
        ),
        "remote_execution_configurations": list(state.summary.remote_configurations),
        "modal_components": [
            {
                "representative_node_id": representative_node_id,
                "node_ids": list(component_node_ids),
            }
            for representative_node_id, component_node_ids in sorted(
                state.summary.component_node_ids_by_representative.items()
            )
        ],
    }


def _queue_error_response(
    prompt_server: Any,
    ctx: RouteContext,
    state: _QueueRequestState,
    exc: BaseException,
    *,
    phase: str,
    status: int,
) -> web.Response:
    """Emit request-scoped UI failure state and return a JSON error response."""
    failed_node_id = str(
        getattr(exc, "configuration_id", "") or ""
    ).strip()
    if state.json_data is not None:
        kwargs = {
            "status_message": str(exc)
            if phase == "execution_interrupted"
            else None,
            "error_message": str(exc) if phase == "error" else None,
        }
        ctx.emit_status(
            prompt_server=prompt_server,
            phase=phase,
            client_id=state.client_id,
            prompt_id=state.prompt_id,
            node_ids=state.remote_node_ids,
            configurator_node_id=state.configurator_node_id,
            modal_gpu=state.request_modal_gpu,
            failed_node_id=failed_node_id or None,
            error_code=getattr(exc, "code", None),
            **{key: value for key, value in kwargs.items() if value is not None},
        )
    node_errors = _queue_node_errors(state, exc, failed_node_id)
    error: str | dict[str, Any] = str(exc)
    if failed_node_id:
        error = {
            "type": getattr(exc, "code", None) or "provider_configuration_invalid",
            "message": str(exc),
            "details": "",
            "extra_info": {"node_id": failed_node_id},
        }
    return web.json_response(
        {
            "error": error,
            "node_errors": node_errors,
            **({"cancelled": True} if phase == "execution_interrupted" else {}),
        },
        status=status,
    )


def _queue_node_errors(
    state: _QueueRequestState,
    exc: BaseException,
    failed_node_id: str,
) -> dict[str, Any]:
    """Build ComfyUI-compatible validation errors for an attributed queue failure."""
    if not failed_node_id or state.json_data is None:
        return {}
    prompt_node = state.json_data.get("prompt", {}).get(failed_node_id, {})
    class_type = str(prompt_node.get("class_type") or "SubrosaRemoteConfiguration")
    reason = {
        "type": getattr(exc, "code", None) or "provider_configuration_invalid",
        "message": (
            "Subrosa authentication required"
            if getattr(exc, "code", None) == "subrosa_login_required"
            else "Subrosa Configuration failed validation"
        ),
        "details": str(exc),
        "extra_info": {},
    }
    return {
        failed_node_id: {
            "errors": [reason],
            "dependent_outputs": [],
            "class_type": class_type,
        }
    }


async def _handle_modal_queue_prompt(
    request: web.Request,
    prompt_server: Any,
    ctx: RouteContext,
) -> web.Response:
    """Run the validate, analyze, plan, rewrite, and dispatch queue stages."""
    logger.info("Received Modal queue request.")
    state = _QueueRequestState()
    request_started_at = time.perf_counter()
    try:
        await _initialize_queue_request(request, ctx, state)
        assert state.json_data is not None
        if "prompt" in state.json_data and not state.remote_node_ids:
            logger.info(
                "No workflow nodes are marked for Modal execution; forwarding prompt without Modal status or rewrite."
            )
            return await _queue_prompt_json(prompt_server, state.json_data)
        await _preflight_queue_credentials(state)
        _reserve_queue_preparation(prompt_server, state)
        _resolve_queue_settings(ctx, state)
        if "prompt" in state.json_data:
            await _rewrite_remote_prompt(prompt_server, ctx, state)
        response = await _queue_prompt_json(
            prompt_server,
            state.json_data,
            modal_response_payload=_modal_queue_response_payload(ctx, state),
        )
        state.prompt_queued = response.status < 400
        return response
    except SyncCancelledError as exc:
        logger.info("Remote workflow preparation cancelled: %s", exc)
        return _queue_error_response(
            prompt_server, ctx, state, exc, phase="execution_interrupted", status=409
        )
    except FileNotFoundError as exc:
        logger.exception("Modal asset sync failed.")
        return _queue_error_response(
            prompt_server, ctx, state, exc, phase="error", status=400
        )
    except ModalPromptValidationError as exc:
        logger.exception("Modal prompt validation failed.")
        return _queue_error_response(
            prompt_server, ctx, state, exc, phase="error", status=400
        )
    except SubrosaConfigurationValidationError as exc:
        logger.debug(
            "Subrosa configuration credential preflight failed node_id=%s: %s",
            exc.configuration_id,
            exc,
        )
        return _queue_error_response(
            prompt_server, ctx, state, exc, phase="error", status=400
        )
    except Exception as exc:
        logger.exception("Modal queue handler failed.")
        return _queue_error_response(
            prompt_server, ctx, state, exc, phase="error", status=500
        )
    finally:
        logger.info(
            "Modal queue request completed in %.3fs.",
            time.perf_counter() - request_started_at,
        )
        if state.preparation_prompt_id is not None:
            _clear_remote_preparation(prompt_server, state.preparation_prompt_id)
        if state.r2_writeback_prompt_id is not None and not state.prompt_queued:
            finish_r2_writeback_prompt(state.r2_writeback_prompt_id)


def register_queue_routes(prompt_server: Any, ctx: RouteContext) -> None:
    """Register analysis, cancellation, and prompt queue routes."""
    @prompt_server.routes.post(ctx.cancel_preparation_route_path)
    async def cancel_remote_preparation(request: web.Request) -> web.Response:
        """Cancel one prompt while it is still preparing remote execution."""
        payload = await request.json()
        prompt_id = str(payload.get("prompt_id") or "").strip()
        cancelled = _cancel_remote_preparation(prompt_server, prompt_id)
        return web.json_response({"cancelled": cancelled, "prompt_id": prompt_id})

    @prompt_server.routes.post(ctx.analysis_route_path)
    async def modal_analyze_remote_nodes(request: web.Request) -> web.Response:
        """Analyze which workflow nodes should be marked remote for the current graph."""
        logger.info("Received Modal remote-node analysis request.")
        try:
            request_started_at = time.perf_counter()
            json_data = await request.json()
            prompt = json_data.get("prompt")
            if not isinstance(prompt, dict):
                raise ValueError("Modal remote-node analysis requires a 'prompt' object.")
            workflow = json_data.get("workflow")
            seed_node_ids = json_data.get("seed_node_ids") or []
            if not isinstance(seed_node_ids, list):
                raise ValueError(
                    "Modal remote-node analysis requires 'seed_node_ids' to be a list."
                )
            analysis = analyze_remote_node_selection(
                prompt=prompt,
                workflow=workflow if isinstance(workflow, dict) else None,
                seed_workflow_node_paths=[str(node_id) for node_id in seed_node_ids],
                settings=ctx.settings,
            )
            logger.info(
                "Modal remote-node analysis finished in %.3fs with %d requested nodes and %d additions.",
                time.perf_counter() - request_started_at,
                len(analysis.requested_workflow_node_paths),
                len(analysis.added_workflow_node_paths),
            )
            return web.json_response(
                {
                    "requested_node_ids": analysis.requested_node_ids,
                    "requested_workflow_node_paths": analysis.requested_workflow_node_paths,
                    "current_remote_node_ids": analysis.current_remote_node_ids,
                    "current_remote_workflow_node_paths": analysis.current_remote_workflow_node_paths,
                    "resolved_remote_node_ids": analysis.resolved_remote_node_ids,
                    "resolved_workflow_node_paths": analysis.resolved_workflow_node_paths,
                    "added_node_ids": analysis.added_node_ids,
                    "added_workflow_node_paths": analysis.added_workflow_node_paths,
                    "sandwiched_local_node_ids": analysis.sandwiched_local_node_ids,
                    "reasons": [
                        {
                            "node_id": reason.node_id,
                            "class_type": reason.class_type,
                            "required_by_node_id": reason.required_by_node_id,
                            "required_by_class_type": reason.required_by_class_type,
                            "output_index": reason.output_index,
                            "io_type": reason.io_type,
                        }
                        for reason in analysis.reasons
                    ],
                }
            )
        except (TypeError, ValueError) as exc:
            logger.warning("Modal remote-node analysis request was invalid: %s", exc)
            return web.json_response({"error": str(exc), "node_errors": []}, status=400)

    @prompt_server.routes.post(ctx.settings.route_path)
    async def modal_queue_prompt(request: web.Request) -> web.Response:
        """Handle prompt queue requests that include remote execution markers."""
        return await _handle_modal_queue_prompt(request, prompt_server, ctx)
