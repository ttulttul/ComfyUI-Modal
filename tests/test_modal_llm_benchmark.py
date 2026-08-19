"""Tests for the reproducible Modal LLM benchmark harness."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import pytest


def _benchmark_module() -> Any:
    """Load the standalone benchmark script as an ordinary module."""
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "benchmark_modal_llm.py"
    )
    spec = importlib.util.spec_from_file_location(
        "modal_llm_benchmark_test",
        script_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load benchmark script {script_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_benchmark_summary_exposes_cold_warm_comparison() -> None:
    """The report summary should retain every latency and throughput dimension."""
    benchmark = _benchmark_module()
    report = {
        "runs": [
            {
                "cycle": 1,
                "kind": "cold",
                "wall_seconds": 10.0,
                "metadata": {
                    "load_seconds": 6.0,
                    "generation_seconds": 4.0,
                    "time_to_first_token_seconds": 2.0,
                    "tokens_per_second": 20.0,
                    "output_tokens": 80,
                    "cache_hit": False,
                },
            },
            {
                "cycle": 1,
                "kind": "warm",
                "wall_seconds": 4.1,
                "metadata": {
                    "load_seconds": 0.0,
                    "generation_seconds": 4.0,
                    "time_to_first_token_seconds": 0.1,
                    "tokens_per_second": 20.0,
                    "output_tokens": 80,
                    "cache_hit": True,
                },
            },
        ]
    }

    summary = benchmark._summarize_report(report)

    assert "cycle kind wall_s load_s gen_s ttft_s tok_s" in summary
    assert "1 cold 10.000 6.000 4.000 2.000 20.000 80 false" in summary
    assert "1 warm 4.100 0.000 4.000 0.100 20.000 80 true" in summary


def test_default_app_name_leaves_room_for_modal_derived_object_suffixes() -> None:
    """The benchmark namespace must fit Modal Dict names derived from the app."""
    benchmark = _benchmark_module()

    app_name = benchmark._default_app_name("RTX-PRO-6000", "throughput")

    assert len(f"{app_name}-session-bridges") < 64
    assert app_name == benchmark._default_app_name("RTX-PRO-6000", "throughput")
    assert app_name != benchmark._default_app_name("B300", "throughput")

    with pytest.raises(ValueError, match="too long"):
        benchmark._validate_app_name("x" * 48)


def test_deployment_name_includes_runtime_gpu_suffix() -> None:
    """Cold resets must target the deployed app rather than its base setting."""
    benchmark = _benchmark_module()
    settings = object()
    remote_module = SimpleNamespace(
        get_settings=lambda: settings,
        modal_deployment_app_name=lambda value: (
            "benchmark-gpu-rtx-pro-6000" if value is settings else "unexpected"
        ),
    )

    assert (
        benchmark._deployment_app_name(remote_module)
        == "benchmark-gpu-rtx-pro-6000"
    )


def test_active_deployment_requires_stopped_state_and_zero_tasks() -> None:
    """A cold cycle must wait for both app state and old tasks to settle."""
    benchmark = _benchmark_module()
    app_name = "benchmark-gpu-rtx-pro-6000"

    assert benchmark._has_active_deployment(
        [{"Description": app_name, "State": "stopping...", "Tasks": "1"}],
        app_name,
    )
    assert not benchmark._has_active_deployment(
        [{"Description": app_name, "State": "stopped", "Tasks": "0"}],
        app_name,
    )
    assert not benchmark._has_active_deployment(
        [{"Description": "another-app", "State": "deployed", "Tasks": "2"}],
        app_name,
    )


def test_invocation_metadata_rejects_false_cold_or_warm_samples() -> None:
    """The harness must fail rather than mislabel an engine-residency sample."""
    benchmark = _benchmark_module()
    config = benchmark.BenchmarkConfig(
        app_name="benchmark",
        cold_cycles=1,
        comfyui_root="/tmp/comfyui",
        enable_reasoning=False,
        gpu="B300",
        keep_app=False,
        max_new_tokens=32,
        mode="throughput",
        model="org/model",
        prompt="prompt",
        seed=1,
        synthetic_image=False,
        warm_runs=1,
    )

    benchmark._validate_invocation_metadata(
        config,
        "cold",
        {"cache_hit": False, "vllm_execution_mode": "throughput"},
    )
    benchmark._validate_invocation_metadata(
        config,
        "warm",
        {"cache_hit": True, "vllm_execution_mode": "throughput"},
    )

    with pytest.raises(RuntimeError, match="Expected 'cold' cache_hit=False"):
        benchmark._validate_invocation_metadata(
            config,
            "cold",
            {"cache_hit": True, "vllm_execution_mode": "throughput"},
        )
    with pytest.raises(RuntimeError, match="Expected vLLM mode 'throughput'"):
        benchmark._validate_invocation_metadata(
            config,
            "warm",
            {"cache_hit": True, "vllm_execution_mode": "eager"},
        )
