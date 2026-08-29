"""Tests for dynamic Modal proxy nodes and local execution fallback."""

from __future__ import annotations

import asyncio

import copy

from concurrent.futures import Future, ThreadPoolExecutor

from dataclasses import replace

from datetime import datetime, timedelta, timezone

from decimal import Decimal

import hashlib

import importlib

import importlib.util

import json

import pickle

import subprocess

import sys

import threading

import time

import types

from contextlib import nullcontext

from io import BytesIO, StringIO

from pathlib import Path

import logging

from typing import Any, Iterator

import pytest

DEFAULT_TEST_DEPLOYMENT_APP_NAME = "comfy-modal-sync-gpu-rtx-pro-6000"

def _cloud_session_bridge_owner() -> Any:
    """Return the module that owns cloud session-bridge state and replay helpers."""
    return importlib.import_module("cloud_session_bridge")

def _cloud_remote_session_store() -> Any:
    """Return the process-local session store from its extracted owner."""
    return _cloud_session_bridge_owner()._REMOTE_SESSION_STORE

def _cloud_bridge_value_cache() -> dict[str, Any]:
    """Return the warm bridge-value cache from its extracted owner."""
    return _cloud_session_bridge_owner()._REMOTE_SESSION_BRIDGE_VALUE_CACHE

def _cloud_bridge_value_cache_order() -> list[str]:
    """Return the warm bridge-value eviction order from its extracted owner."""
    return _cloud_session_bridge_owner()._REMOTE_SESSION_BRIDGE_VALUE_CACHE_ORDER

def _patch_cloud_session_bridge(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: Any,
) -> None:
    """Patch a session-bridge helper at its extracted owner."""
    monkeypatch.setattr(_cloud_session_bridge_owner(), name, value)

def _cloud_comfy_bootstrap_owner() -> Any:
    """Return the module that owns cloud ComfyUI bootstrap state and helpers."""
    return importlib.import_module("cloud_comfy_bootstrap")

def _cloud_node_output_cache_owner() -> Any:
    """Return the module that owns persisted cloud node-output cache behavior."""
    return importlib.import_module("cloud_node_output_cache")

def _cloud_prompt_execution_owner() -> Any:
    """Return the module that owns cloud prompt execution state and helpers."""
    return importlib.import_module("cloud_prompt_execution")

def _cloud_mapped_execution_owner() -> Any:
    """Return the module that owns cloud mapped execution helpers."""
    return importlib.import_module("cloud_mapped_execution")

def _cloud_execution_control_owner() -> Any:
    """Return the module that owns active cloud execution registration."""
    return importlib.import_module("cloud_execution_control")

def _cloud_streaming_owner() -> Any:
    """Return the module that owns cloud stream buffering and payload dispatch."""
    return importlib.import_module("cloud_streaming")

def _cloud_volume_reload_owner() -> Any:
    """Return the module that owns cloud volume reload state and read-through."""
    return importlib.import_module("cloud_volume_reload")

def _cloud_prewarm_owner() -> Any:
    """Return the module that owns cloud warm-container preparation state."""
    return importlib.import_module("cloud_prewarm")

def _patch_cloud_storage_root(
    monkeypatch: pytest.MonkeyPatch,
    modal_cloud_module: Any,
    storage_root: Path,
) -> None:
    """Patch storage settings across the cloud entrypoint and bootstrap owner."""

    def settings() -> Any:
        """Return storage settings for one isolated cloud test."""
        return types.SimpleNamespace(
            remote_storage_root=str(storage_root),
            local_storage_root=None,
        )

    monkeypatch.setattr(modal_cloud_module, "get_settings", settings)
    monkeypatch.setattr(_cloud_comfy_bootstrap_owner(), "get_settings", settings)
    monkeypatch.setattr(_cloud_volume_reload_owner(), "get_settings", settings)

class _FakeOriginalNode:
    """Simple fake legacy node for proxy signature mirroring."""

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("image", "count")
    OUTPUT_IS_LIST = (False, False)
    FUNCTION = "run"

    def run(self, **kwargs: Any) -> tuple[Any, ...]:
        """Return a tuple that exposes the inputs for verification."""
        return (kwargs["value"], 1)

class _CloneableCacheValue:
    """Simple cloneable object used for loader cache tests."""

    def __init__(self, value: str) -> None:
        """Store an identifying value for later clone assertions."""
        self.value = value

    def clone(self) -> "_CloneableCacheValue":
        """Return a fresh object carrying the same value."""
        return _CloneableCacheValue(self.value)

class _FakeModelValue:
    """Simple cloneable stand-in for a non-transportable MODEL output."""

    def __init__(self, value: str) -> None:
        """Store an identifying value for later assertions."""
        self.value = value

    def clone(self) -> "_FakeModelValue":
        """Return a fresh model value carrying the same identifier."""
        return _FakeModelValue(self.value)

class _FakeModelLoaderNode:
    """Fake self-contained loader node that returns one MODEL-like object."""

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    OUTPUT_IS_LIST = (False,)
    FUNCTION = "load_checkpoint"

    def load_checkpoint(self, ckpt_name: str) -> tuple[_FakeModelValue]:
        """Return a deterministic model value derived from the checkpoint name."""
        return (_FakeModelValue(f"model::{ckpt_name}"),)

class _PromptMetadataSerializationNode:
    """Output node double that JSON-serializes ComfyUI's hidden PROMPT input."""

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("metadata",)
    OUTPUT_IS_LIST = (False,)
    OUTPUT_NODE = True
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, Any]]:
        """Declare one hydrated tensor input and the hidden prompt metadata input."""
        return {
            "required": {"image": ("IMAGE",)},
            "hidden": {"prompt": "PROMPT"},
        }

    def run(self, image: Any, prompt: dict[str, Any]) -> tuple[str]:
        """Serialize the prompt exactly as metadata-writing output nodes do."""
        del image
        return (json.dumps(prompt, sort_keys=True),)

class _FakeClipLoaderNode:
    """Fake self-contained loader node that returns one CLIP-like object."""

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    OUTPUT_IS_LIST = (False,)
    FUNCTION = "load_clip"

    def load_clip(
        self,
        clip_name: str,
        type: str = "stable_diffusion",
        device: str = "default",
    ) -> tuple[str]:
        """Return a deterministic CLIP value derived from the loader inputs."""
        return (f"clip::{clip_name}:{type}:{device}",)

class _FakeSessionValueNode:
    """Fake node that produces one remote-only STRING value."""

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_IS_LIST = (False,)
    FUNCTION = "execute"

    def execute(self) -> tuple[str]:
        """Return a deterministic value that can be stored in session state."""
        return ("shared-session-value",)

class _FakeSessionEchoNode:
    """Fake node that echoes a STRING input back to the caller."""

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_IS_LIST = (False,)
    FUNCTION = "execute"

    def execute(self, text: str) -> tuple[str]:
        """Return the supplied input unchanged for session-ref resolution tests."""
        return (text,)

class _FakeRewriteRemoteModelNode:
    """Fake rewrite-time node that produces a non-transportable MODEL output."""

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    OUTPUT_IS_LIST = (False,)

class _FakeRewriteRemoteSamplerNode:
    """Fake rewrite-time node that produces a transportable LATENT output."""

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    OUTPUT_IS_LIST = (False,)

class _FakeRewriteLatentSourceNode:
    """Fake local source used to feed LATENT values into remote proxies."""

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    OUTPUT_IS_LIST = (False,)

class _FakeRewriteModalMapInputNode:
    """Fake rewrite-time Modal map marker node."""

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("value",)
    OUTPUT_IS_LIST = (False,)

class _FakeRewriteLocalSinkNode:
    """Fake local sink used to model downstream local work in rewrite tests."""

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_IS_LIST = (False,)

class _FakeRewriteLocalFeedbackNode:
    """Fake local node that turns a remote latent into a transportable remote input."""

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_IS_LIST = (False,)

class _FakeRewriteRemoteImageNode:
    """Fake remote node that produces one locally previewable image."""

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_IS_LIST = (False,)

class _FakeRewriteRemoteTextNode:
    """Fake remote node that consumes an image and produces text."""

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_IS_LIST = (False,)

class _FakeRewritePreviewImageNode:
    """Fake local PreviewImage output node."""

    RETURN_TYPES: tuple[str, ...] = ()
    RETURN_NAMES: tuple[str, ...] = ()
    OUTPUT_IS_LIST: tuple[bool, ...] = ()
    OUTPUT_NODE = True

def _current_remote_runtime_payload(remote_modal_app_module: Any) -> dict[str, Any]:
    """Return version metadata for a compatible remote-engine test double."""
    return {
        "protocol_version": remote_modal_app_module._REMOTE_APP_PROTOCOL_VERSION,
        "runtime_fingerprint": remote_modal_app_module._expected_remote_runtime_fingerprint(),
    }

class _FakeImplicitBatchKSamplerNode:
    """Fake sampler node exposing ComfyUI-style LATENT and primitive socket types."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[str]]]:
        """Return the minimal socket schema needed for implicit batch target inspection."""
        return {
            "required": {
                "latent_image": ("LATENT",),
                "seed": ("INT",),
                "positive": ("CONDITIONING",),
            }
        }

def _assert_node_module_identity(node_class: type[Any], expected_module: str) -> None:
    """Assert a registered V3 node exposes its loader-assigned module identity."""
    assert node_class.RELATIVE_PYTHON_MODULE == expected_module
    assert node_class.GET_NODE_INFO_V1()["python_module"] == expected_module

class _PersistentCacheNode:
    """Simple node used to verify persisted node-cache reuse across prompt runs."""

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("value",)
    OUTPUT_IS_LIST = (False,)
    FUNCTION = "run"
    invocation_count = 0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[str]]]:
        """Return the minimal V1 schema needed for cache-key generation."""
        return {"required": {"value": ("INT",)}}

    def run(self, value: int) -> tuple[int]:
        """Count real executions so persisted cache hits are visible to the test."""
        type(self).invocation_count += 1
        return (value + 1,)

class _BoundarySourceNode:
    """Simple source node used for subgraph execution tests."""

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("value",)
    OUTPUT_IS_LIST = (False,)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[str]]]:
        """Return the minimal V1 input schema."""
        return {"required": {"value": ("INT",)}}

    def run(self, value: int) -> tuple[int]:
        """Increment the boundary input."""
        return (value + 1,)

class _BoundarySinkNode:
    """Simple downstream node used for subgraph execution tests."""

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("value",)
    OUTPUT_IS_LIST = (False,)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[str]]]:
        """Return the minimal V1 input schema."""
        return {"required": {"value": ("INT",)}}

    def run(self, value: int) -> tuple[int]:
        """Double the upstream value."""
        return (value * 2,)

class _PrimitiveEchoNode:
    """Simple node used to verify primitive widget coercion."""

    RETURN_TYPES = ("INT", "FLOAT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("steps", "cfg", "enabled", "label")
    OUTPUT_IS_LIST = (False, False, False, False)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[str]]]:
        """Return one primitive input of each V1 widget type."""
        return {
            "required": {
                "steps": ("INT",),
                "cfg": ("FLOAT",),
                "enabled": ("BOOLEAN",),
                "label": ("STRING",),
            }
        }

    def run(
        self,
        steps: int,
        cfg: float,
        enabled: bool,
        label: str,
    ) -> tuple[int, float, bool, str]:
        """Echo primitive inputs after asserting their coerced Python types."""
        assert isinstance(steps, int)
        assert isinstance(cfg, float)
        assert isinstance(enabled, bool)
        assert isinstance(label, str)
        return (steps, cfg, enabled, label)

def _v3_batch_images_node_class() -> type[Any]:
    """Return a dependency-light V3 Batch Images node with Autogrow inputs."""
    class _V3BatchImagesNode:
        """Minimal V3 node matching ComfyUI's Batch Images input schema."""

        @classmethod
        def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[Any, ...]]]:
            """Return the raw V3 Autogrow schema emitted by ComfyUI."""
            return {
                "required": {
                    "images": (
                        "COMFY_AUTOGROW_V3",
                        {
                            "template": {
                                "input": {"required": {"image": ("IMAGE", {})}},
                                "prefix": "image",
                                "min": 1,
                                "max": 50,
                            }
                        },
                    )
                }
            }

    return _V3BatchImagesNode

def _install_fake_v3_input_finalizer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a narrow ComfyUI V3 finalizer double for Autogrow unit tests."""
    comfy_api_module = types.ModuleType("comfy_api")
    comfy_api_module.__path__ = []
    comfy_api_latest_module = types.ModuleType("comfy_api.latest")

    def get_finalized_class_inputs(
        raw_input_types: dict[str, Any],
        live_inputs: dict[str, Any],
    ) -> tuple[dict[str, Any], None, dict[str, Any]]:
        """Expand the test Autogrow schema using ComfyUI's public prompt shape."""
        _, config = raw_input_types["required"]["images"]
        template = config["template"]
        template_input = template["input"]["required"]["image"]
        names = [f"{template['prefix']}{index}" for index in range(template["max"])]
        finalized: dict[str, dict[str, Any]] = {"required": {}, "optional": {}}
        dynamic_paths: dict[str, str] = {}
        for index, name in enumerate(names):
            expanded_name = f"images.{name}"
            section_name = "required" if index < template["min"] else "optional"
            finalized[section_name][expanded_name] = template_input
            if expanded_name in live_inputs:
                dynamic_paths[expanded_name] = expanded_name
        return finalized, None, {"dynamic_paths": dynamic_paths}

    comfy_api_latest_module._io = types.SimpleNamespace(
        get_finalized_class_inputs=get_finalized_class_inputs
    )
    comfy_api_module.latest = comfy_api_latest_module
    monkeypatch.setitem(sys.modules, "comfy_api", comfy_api_module)
    monkeypatch.setitem(sys.modules, "comfy_api.latest", comfy_api_latest_module)

class _ImplicitBatchListSourceNode:
    """Fake node that consumes a whole list once and emits scalar and list outputs."""

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("first_value", "all_values")
    OUTPUT_IS_LIST = (False, True)
    INPUT_IS_LIST = True
    FUNCTION = "run"

class _ImplicitBatchScalarConsumerNode:
    """Fake scalar consumer used for outer implicit batch regressions."""

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_IS_LIST = (False,)
    FUNCTION = "run"

class _ImplicitBatchListConsumerNode:
    """Fake list-aware consumer used for outer implicit batch regressions."""

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_IS_LIST = (True,)
    INPUT_IS_LIST = True
    FUNCTION = "run"


__all__ = tuple(name for name in globals() if not name.startswith("__"))
