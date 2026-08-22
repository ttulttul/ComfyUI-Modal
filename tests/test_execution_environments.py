"""Tests for provider-neutral execution environment scheduling."""

from __future__ import annotations

from typing import Any

import pytest


def _capabilities(
    module: Any,
    *,
    vram_gb: int,
    ram_gb: int = 64,
    free_vram_gb: int | None = None,
) -> Any:
    """Return deterministic Linux GPU capabilities for scheduler tests."""
    return module.EnvironmentCapabilities(
        architecture="x86_64",
        operating_system="linux",
        cpu_count=16,
        total_ram_bytes=ram_gb * 1024**3,
        available_ram_bytes=ram_gb * 1024**3,
        available_disk_bytes=1024**4,
        docker_version="28.0.0",
        docker_rootless=False,
        nvidia_container_runtime=True,
        gpus=(
            module.GpuCapability(
                uuid=f"GPU-{vram_gb}",
                name=f"Test GPU {vram_gb}",
                total_vram_bytes=vram_gb * 1024**3,
                free_vram_bytes=(
                    free_vram_gb * 1024**3
                    if free_vram_gb is not None
                    else None
                ),
            ),
        ),
    )


def _environment(
    module: Any,
    environment_id: str,
    *,
    vram_gb: int,
    cost: float | None,
    cold_start_seconds: float = 0.0,
) -> Any:
    """Return one ready scheduling candidate."""
    return module.EnvironmentSchedulingState(
        environment_id=environment_id,
        provider=module.ExecutionProvider.SSH_DOCKER,
        enabled=True,
        health=module.EnvironmentHealth.READY,
        cost_usd_per_second=cost,
        capabilities=_capabilities(module, vram_gb=vram_gb),
        cold_start_seconds=cold_start_seconds,
    )


def test_scheduler_chooses_lowest_predicted_cost_compatible_host(
    execution_environments_module: Any,
) -> None:
    """The scheduler should account for runtime and cold-start cost."""
    module = execution_environments_module
    assignment = module.CostAwareEnvironmentScheduler().choose(
        [
            _environment(module, "fast-expensive", vram_gb=80, cost=0.002),
            _environment(module, "warm-cheap", vram_gb=48, cost=0.0005),
        ],
        module.ComponentResourceRequirements(
            minimum_vram_bytes=40 * 1024**3,
            estimated_execution_seconds=60,
        ),
    )

    assert assignment.environment_id == "warm-cheap"
    assert assignment.predicted_cost_usd == pytest.approx(0.03)


def test_scheduler_rejects_hosts_without_required_vram(
    execution_environments_module: Any,
) -> None:
    """Hard VRAM requirements must be satisfied before comparing cost."""
    module = execution_environments_module
    with pytest.raises(module.NoCompatibleExecutionEnvironmentError, match="insufficient GPU VRAM"):
        module.CostAwareEnvironmentScheduler().choose(
            [_environment(module, "small", vram_gb=24, cost=0.0)],
            module.ComponentResourceRequirements(minimum_vram_bytes=48 * 1024**3),
        )


def test_scheduler_uses_probed_free_vram_for_admission(
    execution_environments_module: Any,
) -> None:
    """A busy GPU must not be admitted based only on nameplate VRAM."""
    module = execution_environments_module
    busy_environment = module.EnvironmentSchedulingState(
        environment_id="busy",
        provider=module.ExecutionProvider.SSH_DOCKER,
        enabled=True,
        health=module.EnvironmentHealth.READY,
        cost_usd_per_second=0.0,
        capabilities=_capabilities(
            module,
            vram_gb=80,
            free_vram_gb=16,
        ),
    )

    with pytest.raises(
        module.NoCompatibleExecutionEnvironmentError,
        match=r"16\.00 GiB available, 40\.00 GiB required",
    ):
        module.CostAwareEnvironmentScheduler().choose(
            [busy_environment],
            module.ComponentResourceRequirements(minimum_vram_bytes=40 * 1024**3),
        )


def test_scheduler_places_known_cost_before_unknown_cost(
    execution_environments_module: Any,
) -> None:
    """Unknown prices must not silently win cost optimization."""
    module = execution_environments_module
    assignment = module.CostAwareEnvironmentScheduler().choose(
        [
            _environment(module, "unknown", vram_gb=80, cost=None),
            _environment(module, "known", vram_gb=80, cost=0.001),
        ],
        module.ComponentResourceRequirements(estimated_execution_seconds=10),
    )

    assert assignment.environment_id == "known"


def test_scheduler_honors_explicit_preference_before_cost(
    execution_environments_module: Any,
) -> None:
    """Explicit workflow preferences should constrain deterministic selection."""
    module = execution_environments_module
    assignment = module.CostAwareEnvironmentScheduler().choose(
        [
            _environment(module, "cheap", vram_gb=80, cost=0.0001),
            _environment(module, "preferred", vram_gb=80, cost=0.002),
        ],
        module.ComponentResourceRequirements(
            estimated_execution_seconds=10,
            preferred_environment_ids=("preferred",),
        ),
    )

    assert assignment.environment_id == "preferred"


def test_scheduler_uses_environment_specific_runtime_estimates(
    execution_environments_module: Any,
) -> None:
    """Observed speed can outweigh a higher per-second rate in total cost."""
    module = execution_environments_module
    assignment = module.CostAwareEnvironmentScheduler().choose(
        [
            _environment(module, "slow-cheap", vram_gb=80, cost=0.001),
            _environment(module, "fast-pricey", vram_gb=80, cost=0.002),
        ],
        module.ComponentResourceRequirements(
            estimated_execution_seconds=60,
            estimated_execution_seconds_by_environment={
                "slow-cheap": 100,
                "fast-pricey": 10,
            },
        ),
    )

    assert assignment.environment_id == "fast-pricey"
    assert assignment.predicted_cost_usd == pytest.approx(0.02)
