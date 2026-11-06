"""Shared state for progressive remote image previews.

Remote components stream completed IMAGE boundary values (one per mapped item)
straight to the local host. Each streamed value is saved as temporary preview
images and recorded in a per-prompt slot registry, so the local
``RemoteProgressivePreview`` node can show slots filling in while the remote
work is still running and can reuse those saved images when ComfyUI finally
executes it.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

PROGRESSIVE_PREVIEW_NODE_ID = "RemoteProgressivePreview"
PROGRESSIVE_PREVIEW_EVENT = "modal_progressive_preview"
PROGRESSIVE_PREVIEW_TARGETS_KEY = "progressive_preview_target_node_ids"
_REGISTRY_ENTRY_LIMIT = 128


@dataclass(frozen=True)
class PreviewImageRef:
    """Identify one saved preview image the ComfyUI ``/view`` route can serve."""

    filename: str
    subfolder: str
    type: str

    @classmethod
    def from_ui(cls, ui_image: Mapping[str, Any]) -> PreviewImageRef:
        """Build a reference from one ComfyUI ``ui.images`` entry."""
        return cls(
            filename=str(ui_image["filename"]),
            subfolder=str(ui_image.get("subfolder", "")),
            type=str(ui_image.get("type", "temp")),
        )

    def as_ui(self) -> dict[str, str]:
        """Return the ComfyUI ``ui.images`` entry for this image."""
        return {"filename": self.filename, "subfolder": self.subfolder, "type": self.type}


@dataclass
class ProgressivePreviewState:
    """Track the filled slots of one progressive preview node for one prompt."""

    slot_count: int | None = None
    slots: dict[int, list[PreviewImageRef]] = field(default_factory=dict)

    def effective_slot_count(self) -> int:
        """Return the declared slot count, grown to cover every filled slot."""
        highest_filled = max(self.slots, default=-1) + 1
        return max(self.slot_count or 0, highest_filled)

    def is_complete(self) -> bool:
        """Return whether every declared slot has received its images."""
        if self.slot_count is None or self.slot_count <= 0:
            return False
        return all(index in self.slots for index in range(self.slot_count))

    def ordered_images(self) -> list[PreviewImageRef]:
        """Return every filled image in slot order."""
        return [image for index in sorted(self.slots) for image in self.slots[index]]


class ProgressivePreviewRegistry:
    """Thread-safe registry of progressive preview slots keyed by prompt and node."""

    def __init__(self, entry_limit: int = _REGISTRY_ENTRY_LIMIT) -> None:
        """Create an empty registry that keeps at most ``entry_limit`` entries."""
        self._lock = threading.Lock()
        self._entries: OrderedDict[tuple[str, str], ProgressivePreviewState] = (
            OrderedDict()
        )
        self._entry_limit = entry_limit

    def begin(self, prompt_id: str, node_id: str, slot_count: int) -> None:
        """Reset one node's slots for a new prompt with a known slot count."""
        with self._lock:
            self._drop_other_prompts_locked(prompt_id, node_id)
            self._entries[(prompt_id, node_id)] = ProgressivePreviewState(
                slot_count=max(int(slot_count), 0)
            )
            self._entries.move_to_end((prompt_id, node_id))
            self._evict_locked()

    def fill(
        self,
        prompt_id: str,
        node_id: str,
        slot_index: int,
        images: Sequence[PreviewImageRef],
        slot_count: int | None = None,
    ) -> ProgressivePreviewState:
        """Store one slot's images and return a snapshot of the node's state."""
        with self._lock:
            key = (prompt_id, node_id)
            state = self._entries.get(key)
            if state is None:
                self._drop_other_prompts_locked(prompt_id, node_id)
                state = ProgressivePreviewState()
                self._entries[key] = state
            if slot_count is not None:
                state.slot_count = max(state.slot_count or 0, int(slot_count))
            state.slots[int(slot_index)] = list(images)
            self._entries.move_to_end(key)
            self._evict_locked()
            return _copy_state(state)

    def get(self, prompt_id: str, node_id: str) -> ProgressivePreviewState | None:
        """Return a snapshot of one node's slots, if any were recorded."""
        with self._lock:
            state = self._entries.get((prompt_id, node_id))
            return _copy_state(state) if state is not None else None

    def pop(self, prompt_id: str, node_id: str) -> ProgressivePreviewState | None:
        """Remove and return one node's slots, if any were recorded."""
        with self._lock:
            return self._entries.pop((prompt_id, node_id), None)

    def clear(self) -> None:
        """Forget every recorded slot."""
        with self._lock:
            self._entries.clear()

    def _drop_other_prompts_locked(self, prompt_id: str, node_id: str) -> None:
        """Discard stale slots the same node recorded for earlier prompts."""
        for key in [
            key
            for key in self._entries
            if key[1] == node_id and key[0] != prompt_id
        ]:
            del self._entries[key]

    def _evict_locked(self) -> None:
        """Drop the oldest entries beyond the configured limit."""
        while len(self._entries) > self._entry_limit:
            evicted_key, _state = self._entries.popitem(last=False)
            logger.debug(
                "Evicted progressive preview slots prompt=%s node=%s.",
                evicted_key[0],
                evicted_key[1],
            )


def _copy_state(state: ProgressivePreviewState) -> ProgressivePreviewState:
    """Return a detached copy of one registry state."""
    return ProgressivePreviewState(
        slot_count=state.slot_count,
        slots={index: list(images) for index, images in state.slots.items()},
    )


PROGRESSIVE_PREVIEW_REGISTRY = ProgressivePreviewRegistry()


def progressive_preview_targets(boundary_output: Mapping[str, Any]) -> list[str]:
    """Return the progressive preview node ids attached to one boundary payload."""
    return [
        str(node_id)
        for node_id in boundary_output.get(PROGRESSIVE_PREVIEW_TARGETS_KEY, []) or []
        if str(node_id)
    ]


def iter_image_batches(value: Any) -> Iterator[Any]:
    """Yield every IMAGE tensor batch contained in a streamed boundary value."""
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_image_batches(item)
        return
    if hasattr(value, "shape") and hasattr(value, "cpu"):
        if len(value.shape) == 3:
            yield value.unsqueeze(0)
            return
        yield value
        return
    logger.warning(
        "Ignoring non-image value of type %s in a progressive preview stream.",
        type(value).__name__,
    )


def save_temp_preview_images(value: Any) -> list[PreviewImageRef]:
    """Save every IMAGE batch in ``value`` as temporary ComfyUI preview images."""
    import nodes

    preview_node = nodes.PreviewImage()
    saved: list[PreviewImageRef] = []
    for image_batch in iter_image_batches(value):
        result = preview_node.save_images(images=image_batch)
        ui_images = ((result or {}).get("ui") or {}).get("images") or []
        saved.extend(PreviewImageRef.from_ui(ui_image) for ui_image in ui_images)
    return saved
