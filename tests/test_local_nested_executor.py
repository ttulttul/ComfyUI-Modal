"""Tests for the nested PromptExecutor used by local fallback execution."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any


def _install_fake_memory_management(monkeypatch: Any) -> ModuleType:
    """Install a minimal stand-in for ComfyUI's RAM cache release globals."""
    memory_management = ModuleType("comfy.memory_management")
    memory_management.extra_ram_release_callback = None
    memory_management.RAM_CACHE_HEADROOM = 0

    def set_ram_cache_release_state(callback: Any, headroom: int) -> None:
        """Mirror ComfyUI's module-global setter."""
        memory_management.extra_ram_release_callback = callback
        memory_management.RAM_CACHE_HEADROOM = max(0, int(headroom))

    memory_management.set_ram_cache_release_state = set_ram_cache_release_state
    monkeypatch.setitem(sys.modules, "comfy.memory_management", memory_management)
    return memory_management


def _fake_execution_module(memory_management: ModuleType) -> SimpleNamespace:
    """Return an execution module whose executor behaves like current ComfyUI."""

    class FakePromptExecutor:
        """Require RAM cache args and clobber the global release state like ComfyUI."""

        def __init__(
            self,
            server: Any,
            cache_type: Any = False,
            cache_args: Any = None,
        ) -> None:
            """Record construction arguments."""
            self.server = server
            self.cache_type = cache_type
            self.cache_args = cache_args
            self.executed_outputs: list[str] = []

        def execute(
            self,
            *,
            prompt: dict[str, Any],
            prompt_id: str,
            extra_data: dict[str, Any],
            execute_outputs: list[str],
        ) -> None:
            """Fail like ComfyUI on missing cache args, then reset global state."""
            del prompt, prompt_id, extra_data
            headroom = int(self.cache_args["ram"] * (1024**3))
            float(self.cache_args["ram_inactive"])
            memory_management.set_ram_cache_release_state(None, headroom)
            self.executed_outputs = execute_outputs

    return SimpleNamespace(PromptExecutor=FakePromptExecutor)


def test_nested_executor_supplies_ram_cache_args(
    local_execution_module: Any,
    monkeypatch: Any,
) -> None:
    """Current ComfyUI indexes cache_args['ram'] on every execute."""
    memory_management = _install_fake_memory_management(monkeypatch)

    executor = local_execution_module._run_nested_prompt_executor(
        _fake_execution_module(memory_management),
        prompt={},
        prompt_id="prompt-1",
        extra_data={},
        execute_outputs=["9"],
    )

    assert executor.cache_type is False
    assert executor.cache_args == {"lru": 0, "ram": 0.0, "ram_inactive": 0.0}
    assert executor.executed_outputs == ["9"]


def test_nested_executor_restores_host_ram_release_state(
    local_execution_module: Any,
    monkeypatch: Any,
) -> None:
    """The host executor's RAM-pressure eviction must survive a nested run."""
    memory_management = _install_fake_memory_management(monkeypatch)

    def host_release(target: int, free_active: bool = False) -> int:
        """Stand in for the host cache's RAM release callback."""
        del target, free_active
        return 0

    memory_management.set_ram_cache_release_state(host_release, 8 * 1024**3)

    local_execution_module._run_nested_prompt_executor(
        _fake_execution_module(memory_management),
        prompt={},
        prompt_id="prompt-1",
        extra_data={},
        execute_outputs=["9"],
    )

    assert memory_management.extra_ram_release_callback is host_release
    assert memory_management.RAM_CACHE_HEADROOM == 8 * 1024**3
