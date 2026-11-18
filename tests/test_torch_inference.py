"""Tests for running ComfyUI work on helper threads under inference mode."""

from __future__ import annotations

import asyncio
from typing import Any, Callable

import pytest
import torch
from torch.utils._pytree import tree_map

_WRAPPER_ALIAS_OPS = (
    torch.ops.aten.detach.default,
    torch.ops.aten.alias.default,
)


class _WrapperWeight(torch.Tensor):
    """Minimal wrapper subclass shaped like ComfyUI's ``QuantizedTensor``."""

    @staticmethod
    def __new__(cls, inner: torch.Tensor) -> "_WrapperWeight":
        """Allocate a wrapper that mirrors the inner tensor's metadata."""
        return torch.Tensor._make_wrapper_subclass(
            cls, inner.shape, dtype=inner.dtype, device=inner.device
        )

    def __init__(self, inner: torch.Tensor) -> None:
        """Keep the wrapped storage tensor."""
        self.inner = inner

    @classmethod
    def __torch_dispatch__(
        cls,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Unwrap arguments and rewrap alias results like ``QuantizedTensor``."""

        def unwrap(value: Any) -> Any:
            """Return the inner tensor for wrapper arguments."""
            return value.inner if isinstance(value, _WrapperWeight) else value

        result = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs or {}))
        if func in _WRAPPER_ALIAS_OPS:
            return _WrapperWeight(result)
        return result


def _patch_weight(weight: torch.Tensor) -> torch.nn.Parameter:
    """Wrap a weight the way ``comfy.utils.set_attr_param`` does."""
    return torch.nn.Parameter(weight, requires_grad=False)


def _unpatch_on_worker_thread(
    run: Callable[[Callable[[], torch.nn.Parameter]], torch.nn.Parameter],
) -> torch.nn.Parameter:
    """Restore a backup weight on a helper thread, like ``ModelPatcher.__del__``."""
    with torch.inference_mode():
        backup = _WrapperWeight(torch.ones(4))

    async def restore() -> torch.nn.Parameter:
        """Hop to a worker thread the way the local executor does."""
        return await asyncio.to_thread(run, lambda: _patch_weight(backup))

    return asyncio.run(restore())


def test_weight_restored_outside_inference_mode_breaks_later_patch() -> None:
    """Reproduce the KSampler failure from a worker thread without inference mode."""
    restored = _unpatch_on_worker_thread(lambda restore: restore())

    with torch.inference_mode():
        with pytest.raises(RuntimeError, match="Cannot set version_counter"):
            _patch_weight(restored)


def test_call_in_inference_mode_keeps_restored_weights_patchable(
    torch_inference_module: Any,
) -> None:
    """Weights restored on a wrapped worker thread stay patchable by samplers."""
    restored = _unpatch_on_worker_thread(
        lambda restore: torch_inference_module.call_in_inference_mode(restore)
    )

    with torch.inference_mode():
        assert _patch_weight(restored).is_inference()


def test_call_in_inference_mode_tears_down_frame_locals_inside_mode(
    torch_inference_module: Any,
) -> None:
    """Objects released when the call returns finalize while inference mode is on."""
    finalizer_modes: list[bool] = []

    class Patcher:
        """Record the autograd mode active when this object is collected."""

        def __del__(self) -> None:
            """Record whether inference mode was enabled during finalization."""
            finalizer_modes.append(torch.is_inference_mode_enabled())

    def run_executor() -> bool:
        """Hold a patcher in a local, like a nested executor cache."""
        patcher = Patcher()
        assert patcher is not None
        return torch.is_inference_mode_enabled()

    assert asyncio.run(
        asyncio.to_thread(torch_inference_module.call_in_inference_mode, run_executor)
    )
    assert finalizer_modes == [True]


def test_execute_subgraph_locally_runs_executor_thread_in_inference_mode(
    local_execution_module: Any,
    serialization_module: Any,
    monkeypatch: Any,
) -> None:
    """The local fallback executor thread must match ComfyUI's inference mode."""
    observed_modes: list[bool] = []

    def fake_execute_subgraph_prompt(
        payload: dict[str, Any],
        hydrated_inputs: dict[str, Any],
        node_mapping: Any = None,
    ) -> tuple[str]:
        """Record the autograd mode of the executor thread."""
        del payload, hydrated_inputs, node_mapping
        observed_modes.append(torch.is_inference_mode_enabled())
        return ("done",)

    monkeypatch.setattr(
        local_execution_module, "_execute_subgraph_prompt", fake_execute_subgraph_prompt
    )
    response = local_execution_module.execute_subgraph_locally(
        {"component_id": "7"}, serialization_module.serialize_node_inputs({})
    )

    assert serialization_module.deserialize_node_outputs(response) == ("done",)
    assert observed_modes == [True]


def test_local_mapped_execution_runs_items_in_inference_mode(
    mapped_execution_module: Any,
    serialization_module: Any,
    monkeypatch: Any,
) -> None:
    """Local mapped lanes run every item, and drop every executor, in inference mode."""
    observed_modes: list[bool] = []

    def fake_execute_mapped_subgraph_payload(
        payload: dict[str, Any],
        hydrated_inputs: dict[str, Any],
        node_mapping: Any = None,
    ) -> tuple[str]:
        """Record the autograd mode of the mapped worker thread."""
        del payload, hydrated_inputs, node_mapping
        observed_modes.append(torch.is_inference_mode_enabled())
        return ("done",)

    monkeypatch.setattr(
        mapped_execution_module,
        "_execute_mapped_subgraph_payload",
        fake_execute_mapped_subgraph_payload,
    )
    response = asyncio.run(
        mapped_execution_module._invoke_mapped_remote_engine_async(
            {"component_id": "100"}, serialization_module.serialize_node_inputs({})
        )
    )

    assert serialization_module.deserialize_node_outputs(response) == ("done",)
    assert observed_modes == [True]
