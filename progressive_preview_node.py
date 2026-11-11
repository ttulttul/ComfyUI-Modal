"""Local ComfyUI node that shows remote images, masks, audio, and text as they stream in."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from typing import Any

from comfy_api.latest import _io as io

if __package__:
    from .progressive_preview import (
        PROGRESSIVE_PREVIEW_NODE_ID,
        PROGRESSIVE_PREVIEW_REGISTRY,
        PreviewEntry,
        ProgressivePreviewState,
        progressive_preview_ui,
        save_temp_preview_entries,
    )
    from .session_state import is_remote_session_bridge_ref_payload
else:  # pragma: no cover - flat import inside the Modal container.
    from progressive_preview import (
        PROGRESSIVE_PREVIEW_NODE_ID,
        PROGRESSIVE_PREVIEW_REGISTRY,
        PreviewEntry,
        ProgressivePreviewState,
        progressive_preview_ui,
        save_temp_preview_entries,
    )
    from session_state import is_remote_session_bridge_ref_payload

logger = logging.getLogger(__name__)


def _executing_prompt_and_node() -> tuple[str | None, str | None]:
    """Return the prompt id and node id ComfyUI is currently executing."""
    from comfy_execution.utils import get_executing_context

    context = get_executing_context()
    if context is None:
        return None, None
    return str(context.prompt_id), str(context.node_id)


def _materialize_bridge_ref(bridge_ref: Mapping[str, Any]) -> Any:
    """Download one remote session bridge value for local preview."""
    if __package__:
        from .remote.modal_app import materialize_remote_session_bridge_ref_locally
    else:  # pragma: no cover - flat import inside the Modal container.
        from remote.modal_app import materialize_remote_session_bridge_ref_locally

    return materialize_remote_session_bridge_ref_locally(bridge_ref)


async def _resolve_item_entries(
    index: int,
    item: Any,
    state: ProgressivePreviewState | None,
) -> list[PreviewEntry]:
    """Return saved preview entries for one input item, reusing streamed slots."""
    if not is_remote_session_bridge_ref_payload(item):
        return save_temp_preview_entries(item)
    streamed_entries = state.slots.get(index) if state is not None else None
    if streamed_entries:
        return list(streamed_entries)
    logger.info(
        "Materializing progressive preview slot %d locally because it was not streamed.",
        index,
    )
    materialized = await asyncio.to_thread(_materialize_bridge_ref, item)
    return save_temp_preview_entries(materialized)


async def resolve_progressive_preview_entries(
    items: list[Any],
    state: ProgressivePreviewState | None,
) -> list[PreviewEntry]:
    """Return the final preview entries, preferring entries already streamed."""
    if state is not None and state.is_complete():
        logger.info(
            "Reusing %d streamed progressive preview slot(s) without re-downloading.",
            state.slot_count,
        )
        return state.ordered_entries()
    resolved: list[PreviewEntry] = []
    for index, item in enumerate(items):
        resolved.extend(await _resolve_item_entries(index, item, state))
    return resolved


class RemoteProgressivePreview(io.ComfyNode):
    """Preview remote results, filling one slot per item as remote work finishes."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        """Declare a preview output node that accepts the whole list of any previewable type."""
        return io.Schema(
            node_id=PROGRESSIVE_PREVIEW_NODE_ID,
            display_name="Remote Progressive Preview",
            category="Modal",
            description=(
                "Preview images, masks, audio, or text produced by remote "
                "execution. When remote work starts, the node shows one empty "
                "slot per mapped item and fills each slot as soon as that item "
                "finishes, without waiting for ComfyUI to receive the whole result."
            ),
            inputs=[
                io.MultiType.Input(
                    "images",
                    types=[io.Image, io.Mask, io.Audio, io.String],
                    display_name="value",
                    tooltip=(
                        "Images, masks, audio, or text from a remote node, usually "
                        "a mapped component. Each item is previewed by its type."
                    ),
                )
            ],
            outputs=[],
            is_output_node=True,
            is_input_list=True,
        )

    @classmethod
    async def execute(cls, images: list[Any]) -> io.NodeOutput:
        """Finalize the preview from streamed slots or the delivered images."""
        prompt_id, node_id = _executing_prompt_and_node()
        state = (
            PROGRESSIVE_PREVIEW_REGISTRY.pop(prompt_id, node_id)
            if prompt_id and node_id
            else None
        )
        entries = await resolve_progressive_preview_entries(list(images), state)
        logger.info(
            "Finalized progressive preview prompt=%s node=%s entries=%d.",
            prompt_id,
            node_id,
            len(entries),
        )
        return io.NodeOutput(ui=progressive_preview_ui(entries))
