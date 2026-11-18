"""Run ComfyUI work on helper threads with the same autograd mode as ComfyUI.

ComfyUI executes every prompt inside ``torch.inference_mode()``, which is
thread-local. Modal-Sync hops onto worker threads (``asyncio.to_thread`` and
``ThreadPoolExecutor``) to run nested ``PromptExecutor`` instances, resolve
session bridges, and drop executor caches. Anything that rewrites model weights
on such a thread outside inference mode, most commonly ``ModelPatcher.__del__``
unpatching weights while a nested executor's cache is released, produces
ordinary (non-inference) quantized weight tensors. The next sampler then patches
those weights inside inference mode and PyTorch raises
``Cannot set version_counter for inference tensor``.
"""

from __future__ import annotations

import logging
from typing import Callable, ParamSpec, TypeVar

logger = logging.getLogger(__name__)

_P = ParamSpec("_P")
_R = TypeVar("_R")


def call_in_inference_mode(
    fn: Callable[_P, _R], /, *args: _P.args, **kwargs: _P.kwargs
) -> _R:
    """Call ``fn`` inside ``torch.inference_mode()`` on the current thread.

    The call frame, including every local it releases on return, is torn down
    before inference mode exits, so model patchers garbage-collected there also
    unpatch their weights as inference tensors. Without torch the call runs as-is.
    """
    try:
        import torch
    except ImportError:
        logger.debug("torch is unavailable; calling %r without inference mode.", fn)
        return fn(*args, **kwargs)
    with torch.inference_mode():
        return fn(*args, **kwargs)
