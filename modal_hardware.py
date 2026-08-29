"""Credential-free hardware metadata for remote execution targets."""

from __future__ import annotations

from typing import Any

if __package__:
    from .execution_environments import EnvironmentCapabilities
else:  # pragma: no cover - flat import inside the Modal container.
    from execution_environments import EnvironmentCapabilities

_MODAL_GPU_VRAM_GB: dict[str, float] = {
    "T4": 16,
    "L4": 24,
    "A10": 24,
    "L40S": 48,
    "A100": 40,
    "A100-40GB": 40,
    "A100-80GB": 80,
    "RTX-PRO-6000": 96,
    "H100": 80,
    "H100!": 80,
    "H200": 141,
    "B200": 180,
    "B200+": 180,
    "B300": 288,
}
# Modal public pricing verified 2026-08-22; CPU and memory charges are excluded.
_MODAL_GPU_COST_USD_PER_SECOND: dict[str, float] = {
    "T4": 0.000164,
    "L4": 0.000222,
    "A10": 0.000306,
    "L40S": 0.000542,
    "A100": 0.000583,
    "A100-40GB": 0.000583,
    "A100-80GB": 0.000694,
    "RTX-PRO-6000": 0.000842,
    "H100": 0.001097,
    "H100!": 0.001097,
    "H200": 0.001261,
    "B200": 0.001736,
    "B200+": 0.001736,
    "B300": 0.001972,
}
_HBM_GPU_NAME_MARKERS = ("A100", "H100", "H200", "B200", "B300")


def _gpu_memory_kind(machine_type: str) -> str:
    """Return the best-known GPU memory technology for a machine label."""
    normalized = machine_type.upper().replace(" ", "-")
    return (
        "HBM"
        if any(marker in normalized for marker in _HBM_GPU_NAME_MARKERS)
        else "VRAM"
    )


def _hardware_payload(
    *,
    machine_type: str,
    gpu_count: int,
    gpu_memory_bytes_per_device: int,
    gpu_memory_bytes_total: int,
    ram_bytes: int = 0,
    ram_available_bytes: int | None = None,
    ram_capacity_label: str | None = None,
) -> dict[str, Any]:
    """Return compact credential-free target hardware metadata."""
    payload: dict[str, Any] = {
        "machine_type": machine_type or "Unknown GPU",
        "gpu_count": max(0, gpu_count),
        "gpu_memory_kind": _gpu_memory_kind(machine_type),
        "gpu_memory_bytes_per_device": max(0, gpu_memory_bytes_per_device),
        "gpu_memory_bytes_total": max(0, gpu_memory_bytes_total),
    }
    if ram_bytes > 0:
        payload["ram_bytes"] = ram_bytes
    if ram_available_bytes is not None and ram_available_bytes >= 0:
        payload["ram_available_bytes"] = ram_available_bytes
    if ram_capacity_label:
        payload["ram_capacity_label"] = ram_capacity_label
    return payload


def _capabilities_hardware_payload(
    capabilities: EnvironmentCapabilities | None,
) -> dict[str, Any] | None:
    """Summarize a probed host's GPUs and system memory for the UI."""
    if capabilities is None:
        return None
    gpu_names = tuple(dict.fromkeys(gpu.name for gpu in capabilities.gpus))
    machine_type = " + ".join(gpu_names) if gpu_names else "CPU-only"
    total_gpu_memory = sum(gpu.total_vram_bytes for gpu in capabilities.gpus)
    per_device_memory = max(
        (gpu.total_vram_bytes for gpu in capabilities.gpus),
        default=0,
    )
    return _hardware_payload(
        machine_type=machine_type,
        gpu_count=len(capabilities.gpus),
        gpu_memory_bytes_per_device=per_device_memory,
        gpu_memory_bytes_total=total_gpu_memory,
        ram_bytes=capabilities.total_ram_bytes,
        ram_available_bytes=capabilities.available_ram_bytes,
    )


def _modal_hardware_payload(gpu_type: str) -> dict[str, Any]:
    """Summarize known Modal GPU capacity without guessing dynamic host RAM."""
    gpu_memory_bytes = int(_MODAL_GPU_VRAM_GB.get(gpu_type, 0.0) * 1024**3)
    return _hardware_payload(
        machine_type=gpu_type,
        gpu_count=1,
        gpu_memory_bytes_per_device=gpu_memory_bytes,
        gpu_memory_bytes_total=gpu_memory_bytes,
        ram_capacity_label="Provider managed",
    )


def _vast_hardware_payload(resource: Any) -> dict[str, Any]:
    """Summarize a Vast offer or lease using its advertised capacities."""
    gpu_count = max(
        0,
        int(
            getattr(resource, "gpu_count", None)
            or getattr(resource, "num_gpus", 0)
            or 0
        ),
    )
    per_device_memory = max(
        0,
        int(getattr(resource, "gpu_ram_mb", 0) or 0) * 1024**2,
    )
    advertised_total_mb = int(getattr(resource, "gpu_total_ram_mb", 0) or 0)
    total_gpu_memory = (
        advertised_total_mb * 1024**2
        if advertised_total_mb > 0
        else per_device_memory * max(gpu_count, 1)
    )
    return _hardware_payload(
        machine_type=str(
            getattr(resource, "gpu_name", "Unknown GPU") or "Unknown GPU"
        ),
        gpu_count=gpu_count,
        gpu_memory_bytes_per_device=per_device_memory,
        gpu_memory_bytes_total=total_gpu_memory,
        ram_bytes=max(
            0,
            int(getattr(resource, "cpu_ram_mb", 0) or 0) * 1024**2,
        ),
    )
