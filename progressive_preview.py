"""Shared state for progressive remote previews.

Remote components stream completed boundary values (one per mapped item)
straight to the local host. Each streamed value is sorted into images, masks,
audio clips, or text, saved as temporary preview entries, and recorded in a
per-prompt slot registry, so the local ``RemoteProgressivePreview`` node can
show slots filling in while the remote work is still running and can reuse
those saved entries when ComfyUI finally executes it.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)

PROGRESSIVE_PREVIEW_NODE_ID = "RemoteProgressivePreview"
PROGRESSIVE_PREVIEW_EVENT = "modal_progressive_preview"
PROGRESSIVE_PREVIEW_TARGETS_KEY = "progressive_preview_target_node_ids"
PROGRESSIVE_PREVIEW_UI_KEY = "modal_preview"
_REGISTRY_ENTRY_LIMIT = 128

PreviewKind = Literal["image", "audio", "text"]
PreviewValueKind = Literal["image", "mask", "audio", "text"]


@dataclass(frozen=True)
class PreviewEntry:
    """Describe one previewable result: a saved image or audio file, or text."""

    kind: PreviewKind
    filename: str = ""
    subfolder: str = ""
    type: str = "temp"
    text: str = ""

    @classmethod
    def from_ui(cls, kind: PreviewKind, ui_file: Mapping[str, Any]) -> PreviewEntry:
        """Build a file entry from one ComfyUI ``ui.images``/``ui.audio`` entry."""
        return cls(
            kind=kind,
            filename=str(ui_file["filename"]),
            subfolder=str(ui_file.get("subfolder", "")),
            type=str(ui_file.get("type", "temp")),
        )

    @classmethod
    def from_text(cls, text: str) -> PreviewEntry:
        """Build a text entry."""
        return cls(kind="text", text=str(text))

    def as_file_ui(self) -> dict[str, str]:
        """Return the plain ComfyUI file entry the ``/view`` route can serve."""
        return {"filename": self.filename, "subfolder": self.subfolder, "type": self.type}

    def as_ui(self) -> dict[str, str]:
        """Return the entry as sent to the browser, tagged with its kind."""
        if self.kind == "text":
            return {"kind": "text", "text": self.text}
        return {"kind": self.kind, **self.as_file_ui()}


@dataclass(frozen=True)
class PreviewValue:
    """One raw value sorted into the kind of preview it needs."""

    kind: PreviewValueKind
    value: Any


@dataclass
class ProgressivePreviewState:
    """Track the filled slots of one progressive preview node for one prompt."""

    slot_count: int | None = None
    slots: dict[int, list[PreviewEntry]] = field(default_factory=dict)

    def effective_slot_count(self) -> int:
        """Return the declared slot count, grown to cover every filled slot."""
        highest_filled = max(self.slots, default=-1) + 1
        return max(self.slot_count or 0, highest_filled)

    def is_complete(self) -> bool:
        """Return whether every declared slot has received its entries."""
        if self.slot_count is None or self.slot_count <= 0:
            return False
        return all(index in self.slots for index in range(self.slot_count))

    def ordered_entries(self) -> list[PreviewEntry]:
        """Return every filled entry in slot order."""
        return [entry for index in sorted(self.slots) for entry in self.slots[index]]


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
        entries: Sequence[PreviewEntry],
        slot_count: int | None = None,
    ) -> ProgressivePreviewState:
        """Store one slot's entries and return a snapshot of the node's state."""
        with self._lock:
            key = (prompt_id, node_id)
            state = self._entries.get(key)
            if state is None:
                self._drop_other_prompts_locked(prompt_id, node_id)
                state = ProgressivePreviewState()
                self._entries[key] = state
            if slot_count is not None:
                state.slot_count = max(state.slot_count or 0, int(slot_count))
            state.slots[int(slot_index)] = list(entries)
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
        slots={index: list(entries) for index, entries in state.slots.items()},
    )


PROGRESSIVE_PREVIEW_REGISTRY = ProgressivePreviewRegistry()


def progressive_preview_targets(boundary_output: Mapping[str, Any]) -> list[str]:
    """Return the progressive preview node ids attached to one boundary payload."""
    return [
        str(node_id)
        for node_id in boundary_output.get(PROGRESSIVE_PREVIEW_TARGETS_KEY, []) or []
        if str(node_id)
    ]


def iter_preview_values(value: Any, io_type: str | None = None) -> Iterator[PreviewValue]:
    """Yield every previewable value in ``value``, sorted by the preview it needs.

    ``io_type`` is the ComfyUI type of the value when the caller knows it; it
    settles the one ambiguous case, a 3-D tensor that could be a single image
    frame or a batch of masks.
    """
    if value is None:
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_preview_values(item, io_type)
        return
    if isinstance(value, str):
        yield PreviewValue("text", value)
        return
    if isinstance(value, (bool, int, float)):
        yield PreviewValue("text", str(value))
        return
    if _is_audio(value):
        yield PreviewValue("audio", value)
        return
    if _is_tensor(value):
        yield _classify_tensor(value, io_type)
        return
    logger.warning(
        "Ignoring value of type %s in a progressive preview because it is not "
        "an image, mask, audio clip, or text.",
        type(value).__name__,
    )


def _is_tensor(value: Any) -> bool:
    """Return whether ``value`` looks like a torch tensor."""
    return hasattr(value, "shape") and hasattr(value, "cpu")


def _is_audio(value: Any) -> bool:
    """Return whether ``value`` is a ComfyUI AUDIO dict."""
    return (
        isinstance(value, Mapping)
        and "sample_rate" in value
        and _is_tensor(value.get("waveform"))
    )


def _classify_tensor(tensor: Any, io_type: str | None) -> PreviewValue:
    """Sort one tensor into an IMAGE batch or a MASK."""
    dims = len(tensor.shape)
    if io_type == "MASK" or dims == 2:
        return PreviewValue("mask", tensor)
    if dims == 3 and int(tensor.shape[-1]) in (3, 4):
        return PreviewValue("image", tensor.unsqueeze(0))
    if dims == 3:
        return PreviewValue("mask", tensor)
    return PreviewValue("image", tensor)


def mask_to_image_batch(mask: Any) -> Any:
    """Return a MASK as a grayscale IMAGE batch, the way ComfyUI's PreviewMask does."""
    return (
        mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
        .movedim(1, -1)
        .expand(-1, -1, -1, 3)
    )


def _save_image_batch(image_batch: Any) -> list[PreviewEntry]:
    """Save one IMAGE batch as temporary preview images."""
    import nodes

    result = nodes.PreviewImage().save_images(images=image_batch)
    ui_images = ((result or {}).get("ui") or {}).get("images") or []
    return [PreviewEntry.from_ui("image", ui_image) for ui_image in ui_images]


def _save_audio(audio: Mapping[str, Any]) -> list[PreviewEntry]:
    """Save one AUDIO value as temporary preview clips, one per batch entry."""
    from comfy_api.latest import ui

    saved = ui.PreviewAudio(dict(audio)).values
    return [PreviewEntry.from_ui("audio", saved_audio) for saved_audio in saved]


def save_preview_value(preview_value: PreviewValue) -> list[PreviewEntry]:
    """Save one sorted value and return the entries the browser can show."""
    if preview_value.kind == "text":
        return [PreviewEntry.from_text(preview_value.value)]
    if preview_value.kind == "audio":
        return _save_audio(preview_value.value)
    if preview_value.kind == "mask":
        return _save_image_batch(mask_to_image_batch(preview_value.value))
    return _save_image_batch(preview_value.value)


def save_temp_preview_entries(value: Any, io_type: str | None = None) -> list[PreviewEntry]:
    """Save every image, mask, audio clip, and text in ``value`` for preview."""
    entries: list[PreviewEntry] = []
    for preview_value in iter_preview_values(value, io_type):
        entries.extend(save_preview_value(preview_value))
    return entries


def progressive_preview_ui(entries: Sequence[PreviewEntry]) -> dict[str, list[Any]]:
    """Return the node's final ``ui`` output.

    Every entry goes under ``modal_preview`` for the panel; images are also
    listed under ``images`` so ComfyUI's queue and history show thumbnails.
    """
    ui: dict[str, list[Any]] = {
        PROGRESSIVE_PREVIEW_UI_KEY: [entry.as_ui() for entry in entries]
    }
    images = [entry.as_file_ui() for entry in entries if entry.kind == "image"]
    if images:
        ui["images"] = images
    return ui
