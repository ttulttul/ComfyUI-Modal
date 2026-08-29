"""Queue-time planning tests for workflow-declared Vast capacity."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest


def _component(api_intercept_module: Any) -> Any:
    """Return one ordinary remote component plan."""
    return api_intercept_module.RemoteComponentPlan(
        node_ids=["1"],
        representative_node_id="1",
        boundary_inputs=[],
        boundary_outputs=[],
        execute_node_ids=["1"],
        contains_output_node=False,
    )


def test_vast_only_policy_requires_configuration_node(
    api_intercept_module: Any,
    execution_scheduling_module: Any,
) -> None:
    """A Vast-only workflow must make its capacity and spending policy explicit."""
    with pytest.raises(
        api_intercept_module.ModalPromptValidationError,
        match="Lease Configuration",
    ):
        execution_scheduling_module._plan_component_execution_assignments(
            components=[_component(api_intercept_module)],
            prompt={"1": {"class_type": "KSampler", "inputs": {}}},
            workflow={"extra": {"remote_execution": {"policy": "vast"}}},
            settings=SimpleNamespace(),
        )


def test_vast_only_policy_quotes_acquires_and_stamps_concrete_environment(
    api_intercept_module: Any,
    execution_scheduling_module: Any,
    execution_environments_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    """The planner must replace a profile quote with an exact rented instance."""
    module = execution_environments_module
    quote = SimpleNamespace(
        profile=SimpleNamespace(profile_name="workflow-vast"),
        predicted_incremental_cost_usd=3.25,
    )
    lease = SimpleNamespace(
        environment_id="vast:90:9001",
        idle_retention_seconds=24 * 3600,
    )
    calls: list[tuple[str, Any]] = []

    class FakeVastService:
        """Provide a deterministic marketplace quote and lease."""

        @classmethod
        def from_environment(cls, settings: Any, *, repo_root: Any) -> "FakeVastService":
            """Record service construction without reading credentials."""
            calls.append(("service", (settings, repo_root)))
            return cls()

        def quote_best_profile_sync(self, profiles: Any, **requirements: Any) -> Any:
            """Return one quoted workflow profile."""
            calls.append(("quote", (profiles, requirements)))
            return quote

        def prefetch_offers_sync(
            self,
            profiles: Any,
            requirements: Any,
        ) -> None:
            """Record the planner's marketplace warmup before selection."""
            calls.append(("prefetch", (profiles, requirements)))

        def scheduling_state(self, selected_quote: Any) -> Any:
            """Expose a compatible zero-cost virtual Vast environment."""
            assert selected_quote is quote
            return module.EnvironmentSchedulingState(
                environment_id="vast:90",
                provider=module.ExecutionProvider.VAST,
                enabled=True,
                health=module.EnvironmentHealth.READY,
                cost_usd_per_second=0.0,
                capabilities=module.EnvironmentCapabilities(
                    architecture="x86_64",
                    operating_system="linux",
                    cpu_count=16,
                    total_ram_bytes=128 * 1024**3,
                    available_ram_bytes=None,
                    available_disk_bytes=1024**4,
                    docker_version="vast-container",
                    docker_rootless=False,
                    nvidia_container_runtime=True,
                    gpus=(
                        module.GpuCapability(
                            "vast-quote",
                            "RTX 6000 Ada",
                            48 * 1024**3,
                        ),
                    ),
                ),
            )

        def acquire_sync(self, selected_quote: Any) -> Any:
            """Return the exact instance selected after the quote wins."""
            assert selected_quote is quote
            calls.append(("acquire", selected_quote))
            return lease

    monkeypatch.setattr(execution_scheduling_module, "VastService", FakeVastService)
    monkeypatch.setattr(
        execution_scheduling_module,
        "_execution_history",
        lambda _settings: None,
    )
    prompt = {
        "1": {"class_type": "KSampler", "inputs": {"steps": 20}},
        "90": {
            "class_type": "VastAILeaseConfiguration",
            "inputs": {
                "profile_name": "workflow-vast",
                "minimum_gpu_vram_gb": 24.0,
                "minimum_cpu_ram_gb": 64.0,
                "maximum_hourly_cost_usd": 1.0,
                "idle_retention_hours": 24.0,
            },
        },
    }

    assignments = execution_scheduling_module._plan_component_execution_assignments(
        components=[_component(api_intercept_module)],
        prompt=prompt,
        workflow={"extra": {"remote_execution": {"policy": "vast"}}},
        settings=SimpleNamespace(
            app_name="test",
            modal_gpu="H200",
            max_containers=1,
            local_storage_root=tmp_path,
        ),
    )

    assignment = assignments["1"]
    assert assignment.provider is module.ExecutionProvider.VAST
    assert assignment.environment_id == "vast:90:9001"
    assert assignment.predicted_cost_usd == 3.25
    assert [call[0] for call in calls] == [
        "service",
        "prefetch",
        "quote",
        "acquire",
    ]


def test_vast_provider_metadata_exposes_safe_markdown_fields(
    api_intercept_module: Any,
    execution_scheduling_module: Any,
) -> None:
    """Proxy payloads should carry the non-secret GPU details used by STRING output."""
    metadata = execution_scheduling_module._vast_provider_metadata(
        SimpleNamespace(
            instance_id=9001,
            profile_id="17",
            profile_name="broad",
            gpu_name="H200 NVL",
            gpu_count=1,
            gpu_ram_mb=143771,
            hourly_cost_usd=3.88,
            idle_retention_seconds=3600,
            runtime_fingerprint="f" * 64,
            worker_image="ghcr.io/example/worker@sha256:" + "a" * 64,
        )
    )

    assert metadata == {
        "vast_instance_id": 9001,
        "vast_profile_id": "17",
        "vast_profile_name": "broad",
        "vast_gpu_name": "H200 NVL",
        "vast_gpu_count": 1,
        "vast_gpu_ram_mb": 143771,
        "vast_hourly_cost_usd": 3.88,
        "vast_idle_retention_seconds": 3600,
        "vast_runtime_fingerprint": "f" * 64,
        "vast_worker_image": "ghcr.io/example/worker@sha256:" + "a" * 64,
    }
