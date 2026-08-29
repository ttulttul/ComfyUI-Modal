"""Tests for prompt rewriting and asset sync integration."""

from __future__ import annotations

import asyncio

import json

from dataclasses import replace

from pathlib import Path

import threading

from types import SimpleNamespace

from typing import Any

import pytest

class _FakeRemoteModelNode:
    """Fake node that produces a non-transportable MODEL output."""

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    OUTPUT_IS_LIST = (False,)

class _FakeCheckpointLoaderSimpleNode:
    """Fake root loader node that produces a non-transportable MODEL output."""

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    OUTPUT_IS_LIST = (False,)

class _FakeRemoteSamplerNode:
    """Fake node that consumes a model and produces a transportable latent."""

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    OUTPUT_IS_LIST = (False,)

class _FakeRemoteNoiseNode:
    """Fake node that produces a non-transportable NOISE strategy object."""

    RETURN_TYPES = ("NOISE",)
    RETURN_NAMES = ("noise",)
    OUTPUT_IS_LIST = (False,)

class _FakeVAELoaderNode:
    """Fake VAE loader that produces a non-transportable VAE output."""

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)
    OUTPUT_IS_LIST = (False,)

class _FakeVAEDecodeNode:
    """Fake VAE decoder that consumes latent and VAE inputs and produces an image."""

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_IS_LIST = (False,)

class _FakeVAEEncodeNode:
    """Fake VAE encoder that consumes image and VAE inputs and produces a latent."""

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    OUTPUT_IS_LIST = (False,)

class _FakeLatentSourceNode:
    """Fake node that produces a transportable LATENT output."""

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    OUTPUT_IS_LIST = (False,)

class _FakeRemoteClipNode:
    """Fake node that produces a non-transportable CLIP output."""

    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    OUTPUT_IS_LIST = (False,)

class _FakeLocalSinkNode:
    """Fake local node used to verify downstream rewiring."""

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_IS_LIST = (False,)

class _FakeRemoteConditioningNode:
    """Fake node that produces a non-transportable CONDITIONING output."""

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    OUTPUT_IS_LIST = (False,)

class _FakeRemoteImageNode:
    """Fake remote node that produces a transportable IMAGE output."""

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_IS_LIST = (False,)

class _FakeRemoteVideoNode:
    """Fake remote node that produces a transportable VIDEO output."""

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    OUTPUT_IS_LIST = (False,)

class _FakeSaveVideoNode:
    """Fake terminal SaveVideo node that returns its VIDEO input."""

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    OUTPUT_IS_LIST = (False,)
    OUTPUT_NODE = True

class _FakeRemoteAudioNode:
    """Fake remote node that produces a transportable AUDIO output."""

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    OUTPUT_IS_LIST = (False,)

class _FakeTextNode:
    """Fake text node used for local LLM boundary planning."""

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_IS_LIST = (False,)

class _FakeRemoteImageConsumerNode:
    """Fake remote node that consumes IMAGE and produces IMAGE."""

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_IS_LIST = (False,)

class _FakePreviewImageNode:
    """Fake preview node that behaves like a terminal UI output."""

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    OUTPUT_IS_LIST = ()
    OUTPUT_NODE = True

class _FakeRemoteArtifactWriterNode:
    """Fake terminal remote node that saves artifacts without ComfyUI outputs."""

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    OUTPUT_IS_LIST = ()
    OUTPUT_NODE = False

def _artifact_finalizer_node_id(summary: Any) -> str:
    """Return the finalizer id after asserting that prompt rewrite attached it."""
    finalizer_node_id = summary.artifact_finalizer_node_id
    assert isinstance(finalizer_node_id, str)
    assert finalizer_node_id
    return finalizer_node_id

class _FakeRemoteModelAndImageNode:
    """Fake remote node that produces both MODEL and IMAGE outputs."""

    RETURN_TYPES = ("MODEL", "IMAGE")
    RETURN_NAMES = ("model", "image")
    OUTPUT_IS_LIST = (False, False)

class _FakeRemoteModelAndImageConsumerNode:
    """Fake remote node that consumes MODEL and IMAGE and produces IMAGE."""

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_IS_LIST = (False,)

class _FakePromptListNode:
    """Fake upstream node that represents a prompt-list producer."""

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_IS_LIST = (False,)

class _FakeModalMapInputNode:
    """Fake Modal map marker node that passes a wildcard value through."""

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("value",)
    OUTPUT_IS_LIST = (False,)

class _FakeRemoteStringEchoNode:
    """Fake remote node that echoes a string output."""

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_IS_LIST = (False,)

class _FakeLocalStringSinkNode:
    """Fake local node used to consume remote STRING outputs."""

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_IS_LIST = (False,)


__all__ = tuple(name for name in globals() if not name.startswith("__"))

