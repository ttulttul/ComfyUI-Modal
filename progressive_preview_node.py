"""Local ComfyUI node that shows remote IMAGE results as they stream in."""

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
        PreviewImageRef,
        ProgressivePreviewState,
        save_temp_preview_images,
    )
    from .session_state import is_remote_session_bridge_ref_payload
else:  # pragma: no cover - flat import inside the Modal container.
    from progressive_preview import (
        PROGRESSIVE_PREVIEW_NODE_ID,
        PROGRESSIVE_PREVIEW_REGISTRY,
        PreviewImageRef,
        ProgressivePreviewState,
        save_temp_preview_images,
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


async def _resolve_item_images(
    index: int,
    item: Any,
    state: ProgressivePreviewState | None,
) -> list[PreviewImageRef]:
    """Return saved preview images for one input item, reusing streamed slots."""
    if not is_remote_session_bridge_ref_payload(item):
        return save_temp_preview_images(item)
    streamed_images = state.slots.get(index) if state is not None else None
    if streamed_images:
        return list(streamed_images)
    logger.info(
        "Materializing progressive preview slot %d locally because it was not streamed.",
        index,
    )
    materialized = await asyncio.to_thread(_materialize_bridge_ref, item)
    return save_temp_preview_images(materialized)


async def resolve_progressive_preview_images(
    items: list[Any],
    state: ProgressivePreviewState | None,
) -> list[PreviewImageRef]:
    """Return the final preview images, preferring images already streamed."""
    if state is not None and state.is_complete():
        logger.info(
            "Reusing %d streamed progressive preview slot(s) without re-downloading.",
            state.slot_count,
        )
        return state.ordered_images()
    resolved: list[PreviewImageRef] = []
    for index, item in enumerate(items):
        resolved.extend(await _resolve_item_images(index, item, state))
    return resolved


class RemoteProgressivePreview(io.ComfyNode):
    """Preview remote images, filling one slot per item as remote work finishes."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        """Declare a PreviewImage-style output node that accepts the whole list."""
        return io.Schema(
            node_id=PROGRESSIVE_PREVIEW_NODE_ID,
            display_name="Remote Progressive Preview",
            category="Modal",
            description=(
                "Preview images produced by remote execution. When remote work "
                "starts, the node shows one empty slot per mapped item and fills "
                "each slot as soon as that item finishes, without waiting for "
                "ComfyUI to receive the whole result."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="Images from a remote node, usually a mapped component.",
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
        saved_images = await resolve_progressive_preview_images(list(images), state)
        logger.info(
            "Finalized progressive preview prompt=%s node=%s images=%d.",
            prompt_id,
            node_id,
            len(saved_images),
        )
        return io.NodeOutput(ui={"images": [image.as_ui() for image in saved_images]})
