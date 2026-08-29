"""Guard the package and flat-container import contracts."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

from conftest import PACKAGE_NAME, REPO_ROOT

_CLOUD_EXCEPTION_NAMES = (
    "RemoteSubgraphExecutionError",
    "RemoteInvocationInProgressError",
    "RemoteInvocationAbandonedError",
    "RemoteCanaryInterruptedError",
    "RemoteCanaryBarrierTimeoutError",
    "ExistingModalAppError",
)


def _root_module_names(repo_root: Path) -> tuple[str, ...]:
    """Return every importable root module except the package initializer."""
    return tuple(
        sorted(path.stem for path in repo_root.glob("*.py") if path.name != "__init__.py")
    )


def test_root_modules_support_package_and_flat_imports(
    extension_package: object,
) -> None:
    """Every shipped root module must load locally and in the flat cloud layout."""
    del extension_package
    module_names = _root_module_names(REPO_ROOT)

    for module_name in module_names:
        importlib.import_module(f"{PACKAGE_NAME}.{module_name}")
    for module_name in module_names:
        importlib.import_module(module_name)


def test_stable_cloud_entrypoint_exports_host_runtime_surface(
    extension_package: object,
) -> None:
    """The stable flat cloud module must retain every name read by the host."""
    del extension_package
    cloud_module = importlib.import_module("comfyui_modal_sync_cloud")

    assert isinstance(cloud_module, ModuleType)
    assert hasattr(cloud_module, "app")
    assert hasattr(cloud_module, "RemoteEngine")
    for exception_name in _CLOUD_EXCEPTION_NAMES:
        exception_type = getattr(cloud_module, exception_name)
        assert exception_type.__module__ == "comfyui_modal_sync_cloud"
