"""Tests for Modal container placement and snapshot-restore diagnostics."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
import subprocess
from typing import Any, Iterator

import pytest

from conftest import PACKAGE_NAME


@pytest.fixture
def placement_module(extension_package: object) -> Iterator[Any]:
    """Return the placement module with no recorded snapshot origin."""
    del extension_package
    module = importlib.import_module(f"{PACKAGE_NAME}.cloud_container_placement")
    module.reset_snapshot_origin_for_tests()
    yield module
    module.reset_snapshot_origin_for_tests()


def _placement(module: Any, *, host: str, gpu_uuid: str, task: str) -> Any:
    """Build a placement for one fake container."""
    return module.ContainerPlacement(
        task_id=task,
        function_id="fu-shared",
        region="us-east-1",
        cloud_provider="CLOUD_PROVIDER_AWS",
        hostname=host,
        gpu=module.GpuIdentity(
            name="NVIDIA RTX PRO 6000", uuid=gpu_uuid, driver_version="580.1"
        ),
    )


def test_parse_nvidia_smi_line_reads_first_gpu(placement_module: Any) -> None:
    """The first CSV row supplies model, UUID, and driver."""
    identity = placement_module._parse_nvidia_smi_line(
        "NVIDIA RTX PRO 6000, GPU-abc, 580.1\nNVIDIA RTX PRO 6000, GPU-def, 580.1\n"
    )

    assert identity == placement_module.GpuIdentity(
        name="NVIDIA RTX PRO 6000", uuid="GPU-abc", driver_version="580.1"
    )


def test_gpu_identity_is_unknown_without_nvidia_smi(
    placement_module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing or failing nvidia-smi must not break container startup."""

    def missing_binary(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        """Simulate a host without nvidia-smi."""
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(placement_module.subprocess, "run", missing_binary)

    assert placement_module.current_gpu_identity() == placement_module.GpuIdentity()


def test_startup_kind_follows_modal_restore_state_file(
    placement_module: Any, tmp_path: Path
) -> None:
    """Modal writes the restore-state file only when it restores a snapshot."""
    restore_state = tmp_path / "restore-state.json"

    assert placement_module.container_startup_kind(restore_state) == "boot"
    restore_state.write_text("{}", encoding="utf-8")
    assert placement_module.container_startup_kind(restore_state) == "restore"


def test_restored_container_reports_its_snapshot_origin(
    placement_module: Any, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A restore reports where the snapshot was taken and whether it moved hosts."""
    origin_placement = _placement(
        placement_module, host="host-a", gpu_uuid="GPU-a", task="ta-origin"
    )
    placement_module.record_snapshot_origin(
        worker_affinity_key="worker-pool:comfy",
        snapshot_profile_key="loader-profile:x",
        placement=origin_placement,
    )
    restore_state = tmp_path / "restore-state.json"
    restore_state.write_text("{}", encoding="utf-8")

    startup = placement_module.describe_container_startup(
        worker_affinity_key="worker-pool:comfy",
        snapshot_profile_key="loader-profile:x",
        gpu_snapshot_enabled=True,
        restore_state_path=restore_state,
        placement=_placement(
            placement_module, host="host-b", gpu_uuid="GPU-b", task="ta-restored"
        ),
    )
    with caplog.at_level(logging.INFO, logger=placement_module.logger.name):
        placement_module.log_container_startup(startup)

    assert startup.kind == "restore"
    assert startup.origin is not None
    assert startup.origin.placement.task_id == "ta-origin"
    message = caplog.records[-1].getMessage()
    assert "kind=restore pool=worker-pool:comfy" in message
    assert "origin_task=ta-origin" in message
    assert "same_host=False same_gpu=False" in message


def test_booted_container_ignores_its_own_snapshot_origin(
    placement_module: Any, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A fresh boot records an origin for later restores but reports none itself."""
    placement = _placement(placement_module, host="host-a", gpu_uuid="GPU-a", task="ta-1")
    placement_module.record_snapshot_origin(
        worker_affinity_key="worker-pool:comfy",
        snapshot_profile_key="",
        placement=placement,
    )

    startup = placement_module.describe_container_startup(
        worker_affinity_key="worker-pool:comfy",
        snapshot_profile_key="",
        gpu_snapshot_enabled=True,
        restore_state_path=tmp_path / "absent.json",
        placement=placement,
    )
    with caplog.at_level(logging.INFO, logger=placement_module.logger.name):
        placement_module.log_container_startup(startup)

    assert startup.kind == "boot"
    assert startup.origin is None
    assert startup.snapshot_age_seconds is None
    assert caplog.records[-1].getMessage().endswith("origin=none.")
