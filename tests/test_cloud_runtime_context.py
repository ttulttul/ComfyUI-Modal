"""Tests for runtime-store injection shared by cloud subsystems."""

from __future__ import annotations

import importlib
from types import ModuleType

from conftest import PACKAGE_NAME


def _runtime_context_module(extension_package: object) -> ModuleType:
    """Return the registry through the synthetic ComfyUI package import shape."""
    del extension_package
    return importlib.import_module(f"{PACKAGE_NAME}.cloud_runtime_context")


def test_runtime_stores_are_empty_before_registration(
    extension_package: object,
) -> None:
    """Accessors return no store after the registry is explicitly cleared."""
    runtime_context = _runtime_context_module(extension_package)
    runtime_context.clear_cloud_runtime_stores()

    assert runtime_context.session_bridge_store() is None
    assert runtime_context.invocation_record_store() is None
    assert runtime_context.volume_store() is None
    assert runtime_context.snapshot_profile_store() is None
    assert runtime_context.node_output_cache_store() is None
    assert runtime_context.interrupt_flag_store() is None


def test_runtime_store_registration_publishes_one_consistent_set(
    extension_package: object,
) -> None:
    """Registration routes each live Modal object through its typed accessor."""
    runtime_context = _runtime_context_module(extension_package)
    stores = [object() for _ in range(6)]
    try:
        runtime_context.register_cloud_runtime_stores(
            session_bridge_cache=stores[0],
            invocation_records=stores[1],
            volume=stores[2],
            snapshot_profiles=stores[3],
            node_output_cache=stores[4],
            interrupt_flags=stores[5],
        )

        assert runtime_context.session_bridge_store() is stores[0]
        assert runtime_context.invocation_record_store() is stores[1]
        assert runtime_context.volume_store() is stores[2]
        assert runtime_context.snapshot_profile_store() is stores[3]
        assert runtime_context.node_output_cache_store() is stores[4]
        assert runtime_context.interrupt_flag_store() is stores[5]
    finally:
        runtime_context.clear_cloud_runtime_stores()
