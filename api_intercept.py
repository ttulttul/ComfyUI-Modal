"""Prompt interception and graph rewriting for Modal-backed execution."""

from __future__ import annotations

import copy
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from aiohttp import web

from .modal_executor_node import ensure_modal_proxy_node_registered
from .settings import ModalSyncSettings, get_settings
from .sync_engine import ModalAssetSyncEngine, SyncedAsset

logger = logging.getLogger(__name__)

_ROUTE_REGISTERED = False
_TRANSPORTABLE_OUTPUT_TYPES = frozenset(
    {
        "*",
        "BOOLEAN",
        "FLOAT",
        "IMAGE",
        "INT",
        "LATENT",
        "MASK",
        "NOISE",
        "SIGMAS",
        "STRING",
    }
)


@dataclass
class RewriteSummary:
    """Summary of the prompt rewrite performed for a queue request."""

    remote_node_ids: list[str] = field(default_factory=list)
    synced_assets: list[SyncedAsset] = field(default_factory=list)
    custom_nodes_bundle: SyncedAsset | None = None


class ModalPromptValidationError(ValueError):
    """Raised when a prompt cannot be executed with the current Modal transport."""


def _get_nodes_module() -> Any:
    """Import the ComfyUI nodes module lazily."""
    import nodes

    return nodes


def _get_server_module() -> Any:
    """Import the ComfyUI server module lazily."""
    import server

    return server


def _get_execution_module() -> Any:
    """Import the ComfyUI execution module lazily."""
    import execution

    return execution


def extract_remote_node_ids(
    workflow: dict[str, Any] | None,
    settings: ModalSyncSettings | None = None,
) -> set[str]:
    """Return the node ids marked for remote execution in the workflow metadata."""
    if workflow is None:
        return set()

    marker = (settings or get_settings()).marker_property
    remote_node_ids: set[str] = set()
    for node in workflow.get("nodes", []):
        properties = node.get("properties") or {}
        if properties.get(marker):
            remote_node_ids.add(str(node.get("id")))
    return remote_node_ids


def _normalize_return_types(node_class: type[Any]) -> tuple[str, ...]:
    """Return normalized output types for a node class."""
    if hasattr(node_class, "GET_SCHEMA"):
        node_class.GET_SCHEMA()
    return tuple(getattr(node_class, "RETURN_TYPES", ("*",))) or ("*",)


def _is_transportable_output_type(io_type: str) -> bool:
    """Return whether a ComfyUI output type can cross the current transport."""
    normalized_parts = [part.strip() for part in io_type.split(",") if part.strip()]
    return bool(normalized_parts) and all(part in _TRANSPORTABLE_OUTPUT_TYPES for part in normalized_parts)


def validate_remote_node_transport_compatibility(
    prompt: dict[str, Any],
    remote_node_ids: set[str],
    nodes_module: Any,
) -> None:
    """Reject remote nodes whose linked inputs require unsupported transport types."""
    validation_errors: list[str] = []

    for remote_node_id in sorted(remote_node_ids):
        prompt_node = prompt.get(remote_node_id)
        if prompt_node is None:
            continue

        remote_class_type = str(prompt_node["class_type"])
        for input_name, input_value in (prompt_node.get("inputs") or {}).items():
            if not (
                isinstance(input_value, list)
                and len(input_value) == 2
                and all(not isinstance(item, dict) for item in input_value)
            ):
                continue

            upstream_node_id = str(input_value[0])
            output_index = int(input_value[1])
            upstream_prompt_node = prompt.get(upstream_node_id)
            if upstream_prompt_node is None:
                continue

            upstream_class_type = str(upstream_prompt_node["class_type"])
            upstream_class = nodes_module.NODE_CLASS_MAPPINGS.get(upstream_class_type)
            if upstream_class is None:
                continue

            upstream_return_types = _normalize_return_types(upstream_class)
            if output_index >= len(upstream_return_types):
                continue

            upstream_output_type = str(upstream_return_types[output_index])
            if _is_transportable_output_type(upstream_output_type):
                continue

            validation_errors.append(
                "Remote node "
                f"{remote_node_id} ({remote_class_type}) input '{input_name}' "
                f"depends on upstream node {upstream_node_id} ({upstream_class_type}) "
                f"output index {output_index} of type '{upstream_output_type}', which "
                "cannot cross the current local/remote boundary. "
                "Current ComfyUI-Modal transport only supports JSON-compatible values, "
                "bytes, and tensor-like outputs such as IMAGE, MASK, LATENT, SIGMAS, "
                "NOISE, INT, FLOAT, BOOLEAN, and STRING."
            )

    if validation_errors:
        raise ModalPromptValidationError("\n".join(validation_errors))


def rewrite_prompt_for_modal(
    prompt: dict[str, Any],
    workflow: dict[str, Any] | None,
    sync_engine: ModalAssetSyncEngine | None = None,
    settings: ModalSyncSettings | None = None,
    nodes_module: Any | None = None,
) -> tuple[dict[str, Any], RewriteSummary]:
    """Rewrite remote-marked nodes into signature-preserving Modal proxy nodes."""
    resolved_settings = settings or get_settings()
    remote_node_ids = extract_remote_node_ids(workflow, resolved_settings)
    summary = RewriteSummary(remote_node_ids=sorted(remote_node_ids))

    if not remote_node_ids:
        return copy.deepcopy(prompt), summary

    resolved_nodes_module = nodes_module or _get_nodes_module()
    resolved_sync_engine = sync_engine or ModalAssetSyncEngine.from_environment(resolved_settings)
    rewritten_prompt = copy.deepcopy(prompt)
    validate_remote_node_transport_compatibility(
        prompt=rewritten_prompt,
        remote_node_ids=remote_node_ids,
        nodes_module=resolved_nodes_module,
    )
    if resolved_settings.sync_custom_nodes:
        summary.custom_nodes_bundle = resolved_sync_engine.sync_custom_nodes_directory()
    else:
        logger.info(
            "Skipping custom_nodes bundle sync because sync is disabled for execution_mode=%s.",
            resolved_settings.execution_mode,
        )

    for node_id in sorted(remote_node_ids):
        prompt_node = rewritten_prompt.get(node_id)
        if prompt_node is None:
            logger.warning("Remote node id %s was not present in the prompt payload.", node_id)
            continue

        original_class_type = str(prompt_node["class_type"])
        original_class = resolved_nodes_module.NODE_CLASS_MAPPINGS[original_class_type]
        proxy_node_id = ensure_modal_proxy_node_registered(
            original_class_type=original_class_type,
            original_class=original_class,
            nodes_module=resolved_nodes_module,
        )

        original_inputs = copy.deepcopy(prompt_node.get("inputs", {}))
        remote_inputs, synced_assets = resolved_sync_engine.sync_prompt_inputs(original_inputs)
        summary.synced_assets.extend(synced_assets)
        original_node_inputs = copy.deepcopy(remote_inputs)

        original_node_data = {
            "node_id": node_id,
            "class_type": original_class_type,
            "inputs": original_node_inputs,
            "meta": copy.deepcopy(prompt_node.get("_meta", {})),
            "custom_nodes_bundle": (
                summary.custom_nodes_bundle.remote_path
                if summary.custom_nodes_bundle is not None
                else None
            ),
        }

        prompt_node["class_type"] = proxy_node_id
        prompt_node["inputs"] = copy.deepcopy(remote_inputs)
        prompt_node["inputs"]["original_node_data"] = original_node_data

        logger.info(
            "Rewrote node %s (%s) to Modal proxy %s.",
            node_id,
            original_class_type,
            proxy_node_id,
        )

    return rewritten_prompt, summary


async def _queue_prompt_json(prompt_server: Any, json_data: dict[str, Any]) -> web.Response:
    """Queue a possibly rewritten prompt using ComfyUI's native semantics."""
    execution = _get_execution_module()
    json_data = prompt_server.trigger_on_prompt(json_data)

    if "number" in json_data:
        number = float(json_data["number"])
    else:
        number = prompt_server.number
        if json_data.get("front"):
            number = -number
        prompt_server.number += 1

    if "prompt" not in json_data:
        return web.json_response(
            {
                "error": {
                    "type": "no_prompt",
                    "message": "No prompt provided",
                    "details": "No prompt provided",
                    "extra_info": {},
                }
            },
            status=400,
        )

    prompt = json_data["prompt"]
    prompt_id = str(json_data.get("prompt_id", uuid.uuid4()))
    partial_execution_targets = json_data.get("partial_execution_targets")
    valid = await execution.validate_prompt(prompt_id, prompt, partial_execution_targets)

    extra_data = dict(json_data.get("extra_data", {}))
    if "client_id" in json_data:
        extra_data["client_id"] = json_data["client_id"]

    if not valid[0]:
        logger.warning("invalid prompt: %s", valid[1])
        return web.json_response({"error": valid[1], "node_errors": valid[3]}, status=400)

    outputs_to_execute = valid[2]
    sensitive: dict[str, Any] = {}
    for sensitive_key in execution.SENSITIVE_EXTRA_DATA_KEYS:
        if sensitive_key in extra_data:
            sensitive[sensitive_key] = extra_data.pop(sensitive_key)

    extra_data["create_time"] = int(time.time() * 1000)
    prompt_server.prompt_queue.put(
        (number, prompt_id, prompt, extra_data, outputs_to_execute, sensitive)
    )
    return web.json_response(
        {"prompt_id": prompt_id, "number": number, "node_errors": valid[3]}
    )


def setup_modal_queue_route(
    prompt_server: Any | None = None,
    sync_engine: ModalAssetSyncEngine | None = None,
    settings: ModalSyncSettings | None = None,
) -> None:
    """Register the `/modal/queue_prompt` route once for the active PromptServer."""
    global _ROUTE_REGISTERED
    if _ROUTE_REGISTERED:
        return

    try:
        resolved_server_module = _get_server_module()
    except ModuleNotFoundError:
        logger.debug("ComfyUI server module is not available; skipping route registration.")
        return

    resolved_settings = settings or get_settings()
    prompt_server = prompt_server or getattr(resolved_server_module.PromptServer, "instance", None)
    if prompt_server is None:
        logger.debug("PromptServer.instance is not available; skipping route registration.")
        return

    resolved_sync_engine = sync_engine or ModalAssetSyncEngine.from_environment(resolved_settings)

    @prompt_server.routes.post(resolved_settings.route_path)
    async def modal_queue_prompt(request: web.Request) -> web.Response:
        """Handle prompt queue requests that include Modal remote markers."""
        logger.info("Received Modal queue request.")
        try:
            request_started_at = time.perf_counter()
            json_data = await request.json()
            extra_pnginfo = ((json_data.get("extra_data") or {}).get("extra_pnginfo") or {})
            workflow = extra_pnginfo.get("workflow")
            if "prompt" in json_data:
                rewrite_started_at = time.perf_counter()
                rewritten_prompt, summary = rewrite_prompt_for_modal(
                    prompt=json_data["prompt"],
                    workflow=workflow,
                    sync_engine=resolved_sync_engine,
                    settings=resolved_settings,
                )
                logger.info(
                    "Modal prompt rewrite finished in %.3fs for %d remote nodes.",
                    time.perf_counter() - rewrite_started_at,
                    len(summary.remote_node_ids),
                )
                json_data["prompt"] = rewritten_prompt
                json_data.setdefault("extra_data", {}).setdefault("modal", {})
                json_data["extra_data"]["modal"]["remote_node_ids"] = summary.remote_node_ids
                json_data["extra_data"]["modal"]["synced_assets"] = [
                    asset.remote_path for asset in summary.synced_assets
                ]
                if summary.custom_nodes_bundle is not None:
                    json_data["extra_data"]["modal"]["custom_nodes_bundle"] = (
                        summary.custom_nodes_bundle.remote_path
                    )
            response = await _queue_prompt_json(prompt_server, json_data)
            logger.info(
                "Modal queue request completed in %.3fs.",
                time.perf_counter() - request_started_at,
            )
            return response
        except FileNotFoundError as exc:
            logger.exception("Modal asset sync failed.")
            return web.json_response({"error": str(exc), "node_errors": []}, status=400)
        except ModalPromptValidationError as exc:
            logger.exception("Modal prompt validation failed.")
            return web.json_response({"error": str(exc), "node_errors": []}, status=400)
        except Exception as exc:
            logger.exception("Modal queue handler failed.")
            return web.json_response({"error": str(exc), "node_errors": []}, status=500)

    _ROUTE_REGISTERED = True
    logger.info("Registered Modal queue route at %s", resolved_settings.route_path)
