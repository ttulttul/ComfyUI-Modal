"""Typed Vast.ai marketplace, configuration, and instance models."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

VAST_API_BASE_URL = "https://console.vast.ai"
VAST_CONFIG_NODE_ID = "VastAILeaseConfiguration"
VAST_DEFAULT_IDLE_RETENTION_HOURS = 24.0
VAST_DEFAULT_MINIMUM_OFFER_DURATION_DAYS = 7.0
_PROFILE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$")


class VastRentalType(str, Enum):
    """Identify a supported Vast.ai marketplace rental type."""

    ON_DEMAND = "on-demand"
    INTERRUPTIBLE = "bid"


@dataclass(frozen=True)
class VastResourceProfile:
    """Describe one workflow-declared Vast.ai capacity pool."""

    profile_id: str
    profile_name: str
    gpu_count: int = 1
    minimum_gpu_ram_mb: int = 24 * 1024
    minimum_total_flops: float = 0.0
    minimum_cpu_ram_mb: int = 64 * 1024
    minimum_cpu_cores: float = 8.0
    allocated_disk_gb: float = 200.0
    maximum_hourly_cost_usd: float = 1.0
    idle_retention_seconds: float = VAST_DEFAULT_IDLE_RETENTION_HOURS * 3600
    minimum_offer_duration_seconds: float = (
        VAST_DEFAULT_MINIMUM_OFFER_DURATION_DAYS * 86400
    )
    minimum_reliability: float = 0.99
    minimum_dlperf: float = 0.0
    minimum_download_mb_per_second: float = 100.0
    minimum_cuda_version: float = 13.0
    verified_only: bool = True
    allowed_geolocations: tuple[str, ...] = ()
    maximum_instances: int = 1
    rental_type: VastRentalType = VastRentalType.ON_DEMAND

    def __post_init__(self) -> None:
        """Reject malformed or economically unsafe profile values."""
        if not self.profile_id.strip():
            raise ValueError("Vast profile_id must not be empty.")
        if not _PROFILE_NAME_PATTERN.fullmatch(self.profile_name):
            raise ValueError(
                "Vast profile_name must contain letters, digits, dots, underscores, "
                "or hyphens and be at most 63 characters."
            )
        _require_positive_int(self.gpu_count, "gpu_count")
        _require_non_negative_int(self.minimum_gpu_ram_mb, "minimum_gpu_ram_mb")
        _require_non_negative_int(self.minimum_cpu_ram_mb, "minimum_cpu_ram_mb")
        _require_positive_int(self.maximum_instances, "maximum_instances")
        for field_name, value in (
            ("minimum_total_flops", self.minimum_total_flops),
            ("minimum_cpu_cores", self.minimum_cpu_cores),
            ("allocated_disk_gb", self.allocated_disk_gb),
            ("maximum_hourly_cost_usd", self.maximum_hourly_cost_usd),
            ("idle_retention_seconds", self.idle_retention_seconds),
            ("minimum_offer_duration_seconds", self.minimum_offer_duration_seconds),
            ("minimum_dlperf", self.minimum_dlperf),
            (
                "minimum_download_mb_per_second",
                self.minimum_download_mb_per_second,
            ),
            ("minimum_cuda_version", self.minimum_cuda_version),
        ):
            _require_finite_non_negative(value, field_name)
        if self.allocated_disk_gb <= 0:
            raise ValueError("allocated_disk_gb must be positive.")
        if self.maximum_hourly_cost_usd <= 0:
            raise ValueError("maximum_hourly_cost_usd must be positive.")
        if not math.isfinite(self.minimum_reliability) or not (
            0.0 <= self.minimum_reliability <= 1.0
        ):
            raise ValueError("minimum_reliability must be between 0 and 1.")
        if any(
            not location.strip()
            or len(location.strip()) > 64
            or any(character in location for character in ("\x00", "\n", "\r"))
            for location in self.allowed_geolocations
        ):
            raise ValueError("allowed_geolocations contains an invalid location.")

    @property
    def environment_id(self) -> str:
        """Return the provider-neutral environment identity for this profile."""
        safe_profile_id = re.sub(r"[^A-Za-z0-9._-]+", "-", self.profile_id).strip(
            "-"
        )
        return f"vast:{safe_profile_id or self.profile_name}"

    def search_payload(self, *, limit: int = 25) -> dict[str, Any]:
        """Return the documented Vast marketplace search request."""
        _require_positive_int(limit, "limit")
        payload: dict[str, Any] = {
            "limit": limit,
            "type": self.rental_type.value,
            "rentable": {"eq": True},
            "rented": {"eq": False},
            "gpu_arch": {"eq": "nvidia"},
            "cpu_arch": {"eq": "amd64"},
            "gpu_frac": {"eq": 1.0},
            "gpu_display_active": {"eq": False},
            "num_gpus": {"eq": self.gpu_count},
            "gpu_ram": {"gte": self.minimum_gpu_ram_mb},
            "cpu_ram": {"gte": self.minimum_cpu_ram_mb},
            "cpu_cores_effective": {"gte": self.minimum_cpu_cores},
            "disk_space": {"gte": self.allocated_disk_gb},
            "allocated_storage": self.allocated_disk_gb,
            "duration": {"gte": self.minimum_offer_duration_seconds},
            "reliability": {"gte": self.minimum_reliability},
            "inet_down": {"gte": self.minimum_download_mb_per_second},
            "cuda_max_good": {"gte": self.minimum_cuda_version},
            "direct_port_count": {"gte": 1},
            "dph_total": {"lte": self.maximum_hourly_cost_usd},
            "order": [["dph_total", "asc"]],
        }
        if self.minimum_total_flops:
            payload["total_flops"] = {"gte": self.minimum_total_flops}
        if self.minimum_dlperf:
            payload["dlperf"] = {"gte": self.minimum_dlperf}
        if self.verified_only:
            payload["verified"] = {"eq": True}
        if self.allowed_geolocations:
            payload["geolocation"] = {
                "in": [location.strip() for location in self.allowed_geolocations]
            }
        return payload


@dataclass(frozen=True)
class VastOffer:
    """Describe one normalized Vast marketplace offer."""

    offer_id: int
    gpu_name: str
    num_gpus: int
    gpu_ram_mb: int
    gpu_total_ram_mb: int
    total_flops: float
    cpu_ram_mb: int
    cpu_cores_effective: float
    disk_space_gb: float
    duration_seconds: float
    reliability: float
    dlperf: float
    download_mb_per_second: float
    cuda_max_good: float
    direct_port_count: int
    hourly_cost_usd: float
    storage_cost_usd_per_gb_month: float | None = None
    geolocation: str | None = None
    verification: str | None = None

    @classmethod
    def from_api(cls, payload: Mapping[str, Any]) -> "VastOffer":
        """Normalize one marketplace response record."""
        return cls(
            offer_id=_required_int(payload, "id"),
            gpu_name=str(payload.get("gpu_name") or "Unknown GPU").strip(),
            num_gpus=_non_negative_int(payload.get("num_gpus"), "num_gpus"),
            gpu_ram_mb=_non_negative_int(payload.get("gpu_ram"), "gpu_ram"),
            gpu_total_ram_mb=_non_negative_int(
                payload.get("gpu_total_ram", payload.get("gpu_ram")),
                "gpu_total_ram",
            ),
            total_flops=_non_negative_float(
                payload.get("total_flops"), "total_flops"
            ),
            cpu_ram_mb=_non_negative_int(payload.get("cpu_ram"), "cpu_ram"),
            cpu_cores_effective=_non_negative_float(
                payload.get("cpu_cores_effective", payload.get("cpu_cores")),
                "cpu_cores_effective",
            ),
            disk_space_gb=_non_negative_float(
                payload.get("disk_space"), "disk_space"
            ),
            duration_seconds=_non_negative_float(
                payload.get("duration"), "duration"
            ),
            reliability=_non_negative_float(
                payload.get("reliability"), "reliability"
            ),
            dlperf=_non_negative_float(payload.get("dlperf"), "dlperf"),
            download_mb_per_second=_non_negative_float(
                payload.get("inet_down"), "inet_down"
            ),
            cuda_max_good=_non_negative_float(
                payload.get("cuda_max_good"), "cuda_max_good"
            ),
            direct_port_count=_non_negative_int(
                payload.get("direct_port_count"), "direct_port_count"
            ),
            hourly_cost_usd=_non_negative_float(
                payload.get("dph_total"), "dph_total"
            ),
            storage_cost_usd_per_gb_month=_optional_non_negative_float(
                payload.get("storage_cost"), "storage_cost"
            ),
            geolocation=_optional_string(payload.get("geolocation")),
            verification=_optional_string(payload.get("verification")),
        )

    def incompatibility_reason(self, profile: VastResourceProfile) -> str | None:
        """Return the first unmet hard constraint, if any."""
        checks: Sequence[tuple[bool, str]] = (
            (self.num_gpus == profile.gpu_count, "GPU count does not match"),
            (
                self.gpu_ram_mb >= profile.minimum_gpu_ram_mb,
                "insufficient GPU RAM",
            ),
            (
                self.total_flops >= profile.minimum_total_flops,
                "insufficient total TFLOPS",
            ),
            (
                self.cpu_ram_mb >= profile.minimum_cpu_ram_mb,
                "insufficient CPU RAM",
            ),
            (
                self.cpu_cores_effective >= profile.minimum_cpu_cores,
                "insufficient effective CPU cores",
            ),
            (
                self.disk_space_gb >= profile.allocated_disk_gb,
                "insufficient disk space",
            ),
            (
                self.duration_seconds >= profile.minimum_offer_duration_seconds,
                "insufficient offer duration",
            ),
            (
                self.reliability >= profile.minimum_reliability,
                "insufficient reliability",
            ),
            (self.dlperf >= profile.minimum_dlperf, "insufficient DLPerf"),
            (
                self.download_mb_per_second
                >= profile.minimum_download_mb_per_second,
                "insufficient download bandwidth",
            ),
            (
                self.cuda_max_good >= profile.minimum_cuda_version,
                "insufficient CUDA support",
            ),
            (self.direct_port_count >= 1, "no direct SSH port is available"),
            (
                self.hourly_cost_usd <= profile.maximum_hourly_cost_usd,
                "hourly price exceeds profile maximum",
            ),
        )
        for compatible, reason in checks:
            if not compatible:
                return reason
        if profile.verified_only and self.verification not in {None, "verified"}:
            return "offer is not verified"
        if profile.allowed_geolocations and not _matches_geolocation(
            self.geolocation, profile.allowed_geolocations
        ):
            return "geolocation is not allowed"
        return None

    def ranking_key(self) -> tuple[float, float, float, float, int]:
        """Return deterministic price-first marketplace ordering."""
        return (
            self.hourly_cost_usd,
            -self.reliability,
            -self.dlperf,
            -self.download_mb_per_second,
            self.offer_id,
        )


@dataclass(frozen=True)
class VastInstance:
    """Describe the lifecycle and connection fields used by the controller."""

    instance_id: int
    actual_status: str
    intended_status: str | None
    label: str | None
    ssh_host: str | None
    ssh_port: int | None
    gpu_name: str | None
    num_gpus: int
    gpu_ram_mb: int
    cpu_ram_mb: int
    hourly_cost_usd: float | None

    @classmethod
    def from_api(cls, payload: Mapping[str, Any]) -> "VastInstance":
        """Normalize one Vast instance response record."""
        return cls(
            instance_id=_required_int(payload, "id"),
            actual_status=str(payload.get("actual_status") or "unknown").strip(),
            intended_status=_optional_string(payload.get("intended_status")),
            label=_optional_string(payload.get("label")),
            ssh_host=_optional_string(payload.get("ssh_host")),
            ssh_port=_optional_int(payload.get("ssh_port"), "ssh_port"),
            gpu_name=_optional_string(payload.get("gpu_name")),
            num_gpus=_non_negative_int(payload.get("num_gpus"), "num_gpus"),
            gpu_ram_mb=_non_negative_int(payload.get("gpu_ram"), "gpu_ram"),
            cpu_ram_mb=_non_negative_int(payload.get("cpu_ram"), "cpu_ram"),
            hourly_cost_usd=_optional_non_negative_float(
                payload.get("dph_total"), "dph_total"
            ),
        )

    @property
    def ready_for_ssh(self) -> bool:
        """Return whether the instance reports a usable SSH endpoint."""
        return (
            self.actual_status == "running"
            and bool(self.ssh_host)
            and self.ssh_port is not None
            and self.ssh_port > 0
        )


@dataclass(frozen=True)
class VastInstanceLaunchSpec:
    """Describe the immutable launch settings for one managed Vast instance."""

    image: str
    disk_gb: float
    label: str
    onstart: str
    environment: Mapping[str, str]
    rental_type: VastRentalType = VastRentalType.ON_DEMAND
    bid_price_usd: float | None = None

    def __post_init__(self) -> None:
        """Validate fields before they can trigger a billable API call."""
        if not self.image.strip():
            raise ValueError("Vast launch image must not be empty.")
        if not self.label.strip():
            raise ValueError("Vast launch label must not be empty.")
        _require_finite_non_negative(self.disk_gb, "disk_gb")
        if self.disk_gb <= 0:
            raise ValueError("disk_gb must be positive.")
        if self.rental_type is VastRentalType.INTERRUPTIBLE:
            if self.bid_price_usd is None or self.bid_price_usd <= 0:
                raise ValueError("Interruptible Vast launches require a positive bid.")
        elif self.bid_price_usd is not None:
            raise ValueError("On-demand Vast launches must not include a bid price.")
        for name, value in self.environment.items():
            if not name.strip() or any(
                character in name or character in value
                for character in ("\x00", "\n", "\r")
            ):
                raise ValueError("Vast launch environment contains an unsafe value.")

    def to_api_payload(self) -> dict[str, Any]:
        """Return the documented create-instance request body."""
        payload: dict[str, Any] = {
            "image": self.image,
            "disk": self.disk_gb,
            "label": self.label,
            "runtype": "ssh_direct",
            "target_state": "running",
            "cancel_unavail": True,
            "python_utf8": True,
            "lang_utf8": True,
            "env": dict(self.environment),
            "onstart": self.onstart,
        }
        if self.bid_price_usd is not None:
            payload["price"] = self.bid_price_usd
        return payload


def compatible_offers(
    offers: Sequence[VastOffer], profile: VastResourceProfile
) -> tuple[VastOffer, ...]:
    """Return compatible offers in deterministic best-price order."""
    return tuple(
        sorted(
            (
                offer
                for offer in offers
                if offer.incompatibility_reason(profile) is None
            ),
            key=VastOffer.ranking_key,
        )
    )


def _matches_geolocation(
    value: str | None, allowed_geolocations: Sequence[str]
) -> bool:
    """Return whether an offer's location matches an allowed country or label."""
    normalized_value = (value or "").strip().casefold()
    return any(
        normalized_value == allowed.strip().casefold()
        or normalized_value.endswith(f", {allowed.strip().casefold()}")
        for allowed in allowed_geolocations
    )


def _required_int(payload: Mapping[str, Any], field_name: str) -> int:
    """Return one required integer field."""
    if field_name not in payload:
        raise ValueError(f"Vast response omitted required field {field_name!r}.")
    return _non_negative_int(payload[field_name], field_name)


def _non_negative_int(value: Any, field_name: str) -> int:
    """Return one non-negative integer field."""
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative integer.")
    try:
        normalized = int(value or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a non-negative integer.") from exc
    if normalized < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")
    return normalized


def _optional_int(value: Any, field_name: str) -> int | None:
    """Return one optional integer field."""
    if value is None:
        return None
    return _non_negative_int(value, field_name)


def _non_negative_float(value: Any, field_name: str) -> float:
    """Return one finite non-negative float field."""
    try:
        normalized = float(value or 0.0)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a non-negative number.") from exc
    _require_finite_non_negative(normalized, field_name)
    return normalized


def _optional_non_negative_float(value: Any, field_name: str) -> float | None:
    """Return one optional finite non-negative float field."""
    if value is None:
        return None
    return _non_negative_float(value, field_name)


def _optional_string(value: Any) -> str | None:
    """Return a stripped optional string."""
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _require_positive_int(value: int, field_name: str) -> None:
    """Require a positive non-boolean integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")


def _require_non_negative_int(value: int, field_name: str) -> None:
    """Require a non-negative non-boolean integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")


def _require_finite_non_negative(value: float, field_name: str) -> None:
    """Require a finite non-negative number."""
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{field_name} must be finite and non-negative.")


__all__ = [
    "VAST_API_BASE_URL",
    "VAST_CONFIG_NODE_ID",
    "VAST_DEFAULT_IDLE_RETENTION_HOURS",
    "VAST_DEFAULT_MINIMUM_OFFER_DURATION_DAYS",
    "VastInstance",
    "VastInstanceLaunchSpec",
    "VastOffer",
    "VastRentalType",
    "VastResourceProfile",
    "compatible_offers",
]
