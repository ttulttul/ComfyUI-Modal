"""Tests for curated, resident multimodal LLM inference."""

from __future__ import annotations

import asyncio

import base64

import json

import logging

import sys

import threading

from dataclasses import dataclass, replace

from fractions import Fraction

from io import BytesIO, StringIO

from pathlib import Path

from types import SimpleNamespace

from typing import Any, Callable

import pytest

import torch

from PIL import Image

def _text_file(filename: str, text: str) -> dict[str, str]:
    """Return one built-in-compatible OpenAI input-file payload."""
    encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
    return {
        "filename": filename,
        "file_data": f"data:text/plain;base64,{encoded}",
        "type": "input_file",
    }

class _FakeReasoningTokenizer:
    """Decode deterministic IDs while modelling Qwen's special think tokens."""

    _vocabulary = {"<think>": 10, "</think>": 11}
    _text = {1: "consider", 2: " carefully", 3: "final answer", 10: "", 11: ""}

    def get_vocab(self) -> dict[str, int]:
        """Return exact reasoning boundary token IDs."""
        return dict(self._vocabulary)

    def decode(self, token_ids: Any, **kwargs: Any) -> str:
        """Decode content while respecting special-token cleanup."""
        del kwargs
        return "".join(self._text[token_id] for token_id in token_ids)

@dataclass
class _FakeBackend:
    """Record resident backend inference and unload behavior."""

    profile_id: str
    unloaded: bool = False
    generate_calls: int = 0

    def generate(
        self,
        prepared_inputs: Any,
        settings: Any,
        progress_callback: Callable[[Any], None],
    ) -> Any:
        """Return deterministic token counts while exercising progress."""
        del prepared_inputs
        self.generate_calls += 1
        progress_callback(SimpleNamespace(stage="generating", value=1))
        return SimpleNamespace(
            text=f"response:{self.profile_id}:{settings.seed}",
            input_tokens=7,
            output_tokens=1,
            reasoning="",
            reasoning_tokens=0,
            reasoning_parser="none",
            time_to_first_token_seconds=0.25,
            tokens_per_second=4.0,
        )

    def unload(self) -> None:
        """Record that the cache released this backend."""
        self.unloaded = True


__all__ = tuple(name for name in globals() if not name.startswith("__"))

