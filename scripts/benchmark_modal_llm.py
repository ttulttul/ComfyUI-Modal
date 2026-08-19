"""Run isolated, billable cold/warm Modal LLM benchmarks."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import ModuleType
from typing import Any, Sequence
import uuid

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Blackfrost-AI/Qwen3.8-27B-ABLITERATED-NVFP4"
DEFAULT_PROMPT = (
    "Explain how persistent weight and compilation caches improve repeated LLM "
    "inference. Use approximately 120 words and finish with one practical "
    "recommendation."
)


@dataclass(frozen=True)
class BenchmarkConfig:
    """Describe one reproducible benchmark workload and deployment profile."""

    app_name: str
    cold_cycles: int
    comfyui_root: str
    enable_reasoning: bool
    gpu: str
    keep_app: bool
    max_new_tokens: int
    mode: str
    model: str
    prompt: str
    seed: int
    synthetic_image: bool
    warm_runs: int


@dataclass(frozen=True)
class BenchmarkInvocation:
    """Record one cold-container or resident-engine inference result."""

    cycle: int
    kind: str
    run: int
    wall_seconds: float
    response_characters: int
    response_sha256: str
    reasoning_characters: int
    metadata: dict[str, Any]


def _repository_root() -> Path:
    """Return the source checkout containing this benchmark script."""
    return Path(__file__).resolve().parents[1]


def _git_value(*arguments: str) -> str:
    """Return one best-effort Git value for benchmark provenance."""
    completed = subprocess.run(
        ["git", *arguments],
        cwd=_repository_root(),
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _load_remote_module() -> Any:
    """Load the source package without importing the ComfyUI extension entrypoint."""
    package_name = f"comfyui_modal_benchmark_{uuid.uuid4().hex}"
    package = ModuleType(package_name)
    package.__path__ = [str(_repository_root())]
    package.__package__ = package_name
    sys.modules[package_name] = package
    return importlib.import_module(f"{package_name}.remote.modal_app")


def _configure_environment(config: BenchmarkConfig) -> None:
    """Set isolated deployment controls before importing Modal-Sync modules."""
    environment = {
        "COMFY_MODAL_APP_NAME": config.app_name,
        "COMFY_MODAL_ALLOW_EPHEMERAL_FALLBACK": "false",
        "COMFY_MODAL_AUTO_DEPLOY": "true",
        "COMFY_MODAL_COMFYUI_ROOT": config.comfyui_root,
        "COMFY_MODAL_EXECUTION_MODE": "remote",
        "COMFY_MODAL_GPU": config.gpu,
        "COMFY_MODAL_LLM_MAX_RESIDENT_MODELS": "1",
        "COMFY_MODAL_LLM_VLLM_EXECUTION_MODE": config.mode,
        "COMFY_MODAL_MAX_CONTAINERS": "1",
        "COMFY_MODAL_SCALEDOWN_WINDOW": "600",
        "COMFY_MODAL_SYNC_CUSTOM_NODES": "false",
    }
    os.environ.update(environment)


def _synthetic_image(enabled: bool) -> Any | None:
    """Return a deterministic image tensor when multimodal testing is enabled."""
    if not enabled:
        return None
    import torch

    image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
    image[:, 8:56, 8:56, 0] = 1.0
    image[:, 20:44, 20:44, 1] = 0.5
    return image


def _payload(
    config: BenchmarkConfig,
    cycle: int,
    kind: str,
    run: int,
) -> dict[str, Any]:
    """Build one direct Modal LLM payload with deterministic worker affinity."""
    prompt_id = f"llm-benchmark-{config.mode}-{cycle}-{kind}-{run}-{uuid.uuid4().hex}"
    return {
        "class_type": "ModalLLM",
        "component_id": "modal-llm-benchmark",
        "modal_gpu": config.gpu,
        "prompt_id": prompt_id,
        "requires_volume_reload": False,
        "subgraph_prompt": {
            "modal-llm-benchmark": {
                "class_type": "ModalLLM",
                "inputs": {"model_profile": config.model},
            }
        },
        "terminate_container_on_error": True,
    }


def _node_inputs(config: BenchmarkConfig) -> dict[str, Any]:
    """Return deterministic node inputs shared by every benchmark invocation."""
    inputs: dict[str, Any] = {
        "enable_reasoning": config.enable_reasoning,
        "keep_model_loaded": True,
        "max_new_tokens": config.max_new_tokens,
        "model_profile": config.model,
        "prompt": config.prompt,
        "reserve_free_vram_gb": 24.0,
        "seed": config.seed,
        "system_prompt": "Answer directly and follow the requested length.",
        "temperature": 0.0,
        "top_p": 1.0,
        "video_frames": 1,
    }
    image = _synthetic_image(config.synthetic_image)
    if image is not None:
        inputs["images"] = image
    return inputs


def _invoke(
    remote_module: Any,
    config: BenchmarkConfig,
    *,
    cycle: int,
    kind: str,
    run: int,
) -> BenchmarkInvocation:
    """Invoke one direct node request and retain timing plus backend telemetry."""
    started_at = time.perf_counter()
    response = remote_module.invoke_remote_engine(
        _payload(config, cycle, kind, run),
        remote_module.serialize_node_inputs(_node_inputs(config)),
        allow_implicit_mapping=False,
    )
    wall_seconds = time.perf_counter() - started_at
    outputs = remote_module.deserialize_node_outputs(response)
    if len(outputs) < 3:
        raise RuntimeError(f"Modal LLM benchmark returned {len(outputs)} outputs.")
    response_text = str(outputs[0])
    metadata = json.loads(str(outputs[1]))
    reasoning = str(outputs[2])
    _validate_invocation_metadata(config, kind, metadata)
    return BenchmarkInvocation(
        cycle=cycle,
        kind=kind,
        run=run,
        wall_seconds=wall_seconds,
        response_characters=len(response_text),
        response_sha256=hashlib.sha256(response_text.encode("utf-8")).hexdigest(),
        reasoning_characters=len(reasoning),
        metadata=metadata,
    )


def _validate_invocation_metadata(
    config: BenchmarkConfig,
    kind: str,
    metadata: dict[str, Any],
) -> None:
    """Reject benchmark samples that do not have the requested engine state."""
    observed_mode = metadata.get("vllm_execution_mode")
    if observed_mode != config.mode:
        raise RuntimeError(
            f"Expected vLLM mode {config.mode!r}, observed {observed_mode!r}."
        )
    expected_cache_hit = kind == "warm"
    observed_cache_hit = metadata.get("cache_hit")
    if observed_cache_hit is not expected_cache_hit:
        raise RuntimeError(
            f"Expected {kind!r} cache_hit={expected_cache_hit}, "
            f"observed {observed_cache_hit!r}."
        )


def _reset_deployment_state(remote_module: Any, app_name: str) -> None:
    """Stop only the dedicated benchmark app and clear process-local lookup state."""
    if not remote_module._stop_modal_app_via_sdk(app_name):
        if not remote_module._stop_modal_app_via_cli(app_name):
            raise RuntimeError(f"Unable to stop dedicated benchmark app {app_name!r}.")
    _wait_for_deployment_stopped(app_name)
    with remote_module._MODAL_AUTO_DEPLOY_LOCK:
        remote_module._MODAL_AUTO_DEPLOY_STATES.clear()
        remote_module._MODAL_REMOTE_APP_VERSION_OK.clear()


def _has_active_deployment(rows: Sequence[dict[str, Any]], app_name: str) -> bool:
    """Return whether Modal still reports live tasks for this exact deployment."""
    return any(
        str(row.get("Description")) == app_name
        and (
            str(row.get("State", "")).lower() != "stopped"
            or str(row.get("Tasks", "0")) != "0"
        )
        for row in rows
    )


def _wait_for_deployment_stopped(app_name: str, timeout_seconds: float = 120.0) -> None:
    """Wait until Modal confirms the old deployment has no live containers."""
    modal_cli = shutil.which("modal")
    if modal_cli is None:
        raise RuntimeError("The Modal CLI is required to verify cold benchmark cycles.")
    deadline = time.monotonic() + timeout_seconds
    while True:
        completed = subprocess.run(
            [modal_cli, "app", "list", "--json"],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Unable to list Modal apps while stopping {app_name!r}: "
                f"{completed.stderr.strip()}"
            )
        rows = json.loads(completed.stdout)
        if not _has_active_deployment(rows, app_name):
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Modal deployment {app_name!r} still has live tasks after "
                f"{timeout_seconds:.0f} seconds."
            )
        time.sleep(1.0)


def _deployment_app_name(remote_module: Any) -> str:
    """Resolve the GPU-specific deployed app that owns benchmark containers."""
    return str(
        remote_module.modal_deployment_app_name(remote_module.get_settings())
    )


def _report(
    config: BenchmarkConfig,
    runs: Sequence[BenchmarkInvocation],
) -> dict[str, Any]:
    """Build the stable JSON benchmark artifact."""
    return {
        "benchmark_schema_version": 1,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "runs": [asdict(run) for run in runs],
        "source": {
            "commit": _git_value("rev-parse", "HEAD"),
            "dirty": bool(_git_value("status", "--short")),
        },
    }


def _summarize_report(report: dict[str, Any]) -> str:
    """Render compact comparable cold/warm rows from one report."""
    rows = ["cycle kind wall_s load_s gen_s ttft_s tok_s output_tokens cache_hit"]
    for run in report["runs"]:
        metadata = run["metadata"]
        rows.append(
            " ".join(
                [
                    str(run["cycle"]),
                    str(run["kind"]),
                    f"{run['wall_seconds']:.3f}",
                    f"{float(metadata['load_seconds']):.3f}",
                    f"{float(metadata['generation_seconds']):.3f}",
                    f"{float(metadata['time_to_first_token_seconds']):.3f}",
                    f"{float(metadata['tokens_per_second']):.3f}",
                    str(metadata["output_tokens"]),
                    str(metadata["cache_hit"]).lower(),
                ]
            )
        )
    return "\n".join(rows)


def run_benchmark(config: BenchmarkConfig) -> dict[str, Any]:
    """Run cold-container cycles followed by immediate resident-engine repeats."""
    _configure_environment(config)
    remote_module = _load_remote_module()
    deployment_app_name = _deployment_app_name(remote_module)
    runs: list[BenchmarkInvocation] = []
    try:
        for cycle in range(1, config.cold_cycles + 1):
            _reset_deployment_state(remote_module, deployment_app_name)
            runs.append(
                _invoke(
                    remote_module,
                    config,
                    cycle=cycle,
                    kind="cold",
                    run=1,
                )
            )
            for warm_run in range(1, config.warm_runs + 1):
                runs.append(
                    _invoke(
                        remote_module,
                        config,
                        cycle=cycle,
                        kind="warm",
                        run=warm_run,
                    )
                )
    finally:
        if not config.keep_app:
            _reset_deployment_state(remote_module, deployment_app_name)
    return _report(config, runs)


def _default_comfyui_root() -> Path:
    """Return the live sibling checkout when present, otherwise the canonical source."""
    sibling = _repository_root().parent / "Latest_ComfyUI"
    if sibling.exists():
        return sibling
    return Path.home() / "git" / "ComfyUI"


def _default_app_name(gpu: str, mode: str) -> str:
    """Return a short deterministic app name safe for all derived Modal objects."""
    gpu_digest = hashlib.sha256(gpu.encode("utf-8")).hexdigest()[:8]
    mode_slug = "tp" if mode == "throughput" else "eg"
    return f"cm-llm-bench-{gpu_digest}-{mode_slug}"


def _validate_app_name(app_name: str) -> str:
    """Reject names that cannot accommodate Modal's derived Dict suffixes."""
    if len(f"{app_name}-session-bridges") >= 64:
        raise ValueError(
            "--app-name is too long; it must leave room for Modal's "
            "'-session-bridges' suffix."
        )
    return app_name


def _arguments() -> argparse.Namespace:
    """Parse benchmark controls from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("eager", "throughput"), required=True)
    parser.add_argument("--gpu", default="RTX-PRO-6000")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cold-cycles", type=int, default=2)
    parser.add_argument("--warm-runs", type=int, default=1)
    parser.add_argument("--enable-reasoning", action="store_true")
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--keep-app", action="store_true")
    parser.add_argument("--comfyui-root", type=Path, default=_default_comfyui_root())
    parser.add_argument("--app-name")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    """Execute the benchmark and write its machine-readable artifact."""
    arguments = _arguments()
    if arguments.cold_cycles <= 0 or arguments.warm_runs <= 0:
        raise ValueError("--cold-cycles and --warm-runs must be positive.")
    if arguments.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive.")
    gpu = str(arguments.gpu).strip()
    gpu_slug = gpu.lower().replace("_", "-")
    app_name = _validate_app_name(
        arguments.app_name or _default_app_name(gpu, arguments.mode)
    )
    output_path = arguments.output or Path(
        f"modal-llm-benchmark-{gpu_slug}-{arguments.mode}.json"
    )
    config = BenchmarkConfig(
        app_name=app_name,
        cold_cycles=arguments.cold_cycles,
        comfyui_root=str(arguments.comfyui_root.expanduser().resolve()),
        enable_reasoning=bool(arguments.enable_reasoning),
        gpu=gpu,
        keep_app=bool(arguments.keep_app),
        max_new_tokens=arguments.max_new_tokens,
        mode=arguments.mode,
        model=arguments.model,
        prompt=arguments.prompt,
        seed=arguments.seed,
        synthetic_image=not arguments.text_only,
        warm_runs=arguments.warm_runs,
    )
    report = run_benchmark(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(_summarize_report(report))
    print(f"artifact {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
