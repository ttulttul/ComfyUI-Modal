"""Queue-time grouping of Modal Map Input markers into zipped input channels.

A ``ModalInputChannel`` node links several ``ModalMapInput`` markers. Every
marker on one channel maps the same remote component, and mapped item ``i``
receives item ``i`` of every marker's input.

The channel wiring only matters to the planner. Before planning,
:func:`resolve_modal_map_channels` records each marker's channel in its
``_meta`` and removes the channel links and channel nodes, so no channel value
ever crosses a remote boundary or reaches ComfyUI's executor.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Mapping
from typing import Any

if __package__:
    from .remote_plan_types import ModalPromptValidationError
else:  # pragma: no cover - flat import inside the Modal container.
    from remote_plan_types import ModalPromptValidationError

logger = logging.getLogger(__name__)

MODAL_MAP_INPUT_NODE_ID = "ModalMapInput"
MODAL_INPUT_CHANNEL_NODE_ID = "ModalInputChannel"
MODAL_INPUT_CHANNEL_IO_TYPE = "MODAL_INPUT_CHANNEL"
MODAL_MAP_INPUT_CHANNEL_INPUT_NAME = "channel"
MODAL_MAP_INPUT_NAME_INPUT_NAME = "name"
MODAL_MAP_CHANNEL_META_KEY = "modal_map_channel"


def _is_prompt_link(value: Any) -> bool:
    """Return whether one prompt input value is a ``[node_id, output_index]`` link."""
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], str | int)
        and isinstance(value[1], int)
    )


def _class_type(prompt: Mapping[str, Any], node_id: str) -> str | None:
    """Return the class type of one prompt node, or ``None`` when it is missing."""
    prompt_node = prompt.get(node_id)
    if not isinstance(prompt_node, Mapping):
        return None
    return str(prompt_node.get("class_type"))


def _channel_consumers(prompt: Mapping[str, Any], channel_node_id: str) -> list[str]:
    """Return the ids of every prompt node that links to one channel node."""
    consumers: list[str] = []
    for node_id, prompt_node in prompt.items():
        if not isinstance(prompt_node, Mapping):
            continue
        for input_value in (prompt_node.get("inputs") or {}).values():
            if _is_prompt_link(input_value) and str(input_value[0]) == channel_node_id:
                consumers.append(str(node_id))
                break
    return consumers


def _validate_channel_consumers(prompt: Mapping[str, Any], channel_node_id: str) -> None:
    """Raise when a channel node feeds anything other than Modal Map Input channel sockets."""
    for consumer_id in _channel_consumers(prompt, channel_node_id):
        consumer = prompt[consumer_id]
        for input_name, input_value in (consumer.get("inputs") or {}).items():
            if not (_is_prompt_link(input_value) and str(input_value[0]) == channel_node_id):
                continue
            if (
                _class_type(prompt, consumer_id) != MODAL_MAP_INPUT_NODE_ID
                or input_name != MODAL_MAP_INPUT_CHANNEL_INPUT_NAME
            ):
                raise ModalPromptValidationError(
                    f"Modal Input Channel {channel_node_id} may only connect to the "
                    f"'{MODAL_MAP_INPUT_CHANNEL_INPUT_NAME}' input of Modal Map Input "
                    f"nodes, but it feeds node {consumer_id} "
                    f"({_class_type(prompt, consumer_id)}) input '{input_name}'."
                )


def _bind_marker_channel(
    prompt: dict[str, Any], marker_id: str, channel_link: Any
) -> None:
    """Record one marker's channel in ``_meta`` and drop its channel link."""
    channel_node_id = str(channel_link[0])
    if _class_type(prompt, channel_node_id) != MODAL_INPUT_CHANNEL_NODE_ID:
        raise ModalPromptValidationError(
            f"Modal Map Input {marker_id} channel input must come from a Modal Input "
            f"Channel node, not node {channel_node_id} "
            f"({_class_type(prompt, channel_node_id)})."
        )
    marker = prompt[marker_id]
    marker["inputs"].pop(MODAL_MAP_INPUT_CHANNEL_INPUT_NAME, None)
    marker.setdefault("_meta", {})[MODAL_MAP_CHANNEL_META_KEY] = channel_node_id


def resolve_modal_map_channels(prompt: Mapping[str, Any]) -> dict[str, Any]:
    """Return a prompt copy whose channel wiring is folded into marker metadata.

    Each ``ModalMapInput`` linked to a ``ModalInputChannel`` gets
    ``_meta["modal_map_channel"]`` set to the channel node id; its channel link
    and every channel node are removed.
    """
    resolved = copy.deepcopy(dict(prompt))
    channel_node_ids = sorted(
        node_id
        for node_id in resolved
        if _class_type(resolved, node_id) == MODAL_INPUT_CHANNEL_NODE_ID
    )
    for channel_node_id in channel_node_ids:
        _validate_channel_consumers(resolved, channel_node_id)

    for node_id in sorted(resolved):
        if _class_type(resolved, node_id) != MODAL_MAP_INPUT_NODE_ID:
            continue
        channel_link = (resolved[node_id].get("inputs") or {}).get(
            MODAL_MAP_INPUT_CHANNEL_INPUT_NAME
        )
        if channel_link is None:
            continue
        if not _is_prompt_link(channel_link):
            raise ModalPromptValidationError(
                f"Modal Map Input {node_id} channel input must be linked to a Modal "
                "Input Channel node."
            )
        _bind_marker_channel(resolved, node_id, channel_link)

    for channel_node_id in channel_node_ids:
        resolved.pop(channel_node_id, None)
    if channel_node_ids:
        logger.info(
            "Resolved Modal Input Channels %s into marker metadata: %s.",
            channel_node_ids,
            {
                channel_id: modal_map_channel_members(resolved, channel_id)
                for channel_id in channel_node_ids
            },
        )
    return resolved


def modal_map_channel(prompt_node: Mapping[str, Any] | None) -> str | None:
    """Return the channel id recorded on one ``ModalMapInput`` prompt node."""
    if not isinstance(prompt_node, Mapping):
        return None
    meta = prompt_node.get("_meta")
    if not isinstance(meta, Mapping):
        return None
    channel = meta.get(MODAL_MAP_CHANNEL_META_KEY)
    return str(channel) if channel is not None and str(channel) else None


def modal_map_channel_members(prompt: Mapping[str, Any], channel_id: str) -> list[str]:
    """Return the sorted marker node ids that belong to one channel."""
    return sorted(
        str(node_id)
        for node_id, prompt_node in prompt.items()
        if _class_type(prompt, str(node_id)) == MODAL_MAP_INPUT_NODE_ID
        and modal_map_channel(prompt_node) == channel_id
    )


def modal_map_input_label(prompt: Mapping[str, Any], node_id: str) -> str:
    """Return a human-readable label for one ``ModalMapInput`` marker.

    Uses the optional ``name`` input when set, for example
    ``"seeds" (Modal Map Input 301)``, otherwise ``Modal Map Input 301``.
    """
    prompt_node = prompt.get(node_id)
    name = ""
    if isinstance(prompt_node, Mapping):
        raw_name = (prompt_node.get("inputs") or {}).get(MODAL_MAP_INPUT_NAME_INPUT_NAME)
        if isinstance(raw_name, str):
            name = raw_name.strip()
    base = f"Modal Map Input {node_id}"
    return f'"{name}" ({base})' if name else base
