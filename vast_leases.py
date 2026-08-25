"""Persistent Vast.ai lease acquisition, reuse, activity, and idle cleanup."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping

if __package__:
    from .vast_api import (
        VastApiClient,
        VastApiError,
        VastInstanceNotFoundError,
        VastOfferUnavailableError,
    )
    from .vast_models import (
        VastInstance,
        VastInstanceLaunchSpec,
        VastOffer,
        VastResourceProfile,
    )
else:  # pragma: no cover - direct debugging imports.
    from vast_api import (
        VastApiClient,
        VastApiError,
        VastInstanceNotFoundError,
        VastOfferUnavailableError,
    )
    from vast_models import (
        VastInstance,
        VastInstanceLaunchSpec,
        VastOffer,
        VastResourceProfile,
    )

logger = logging.getLogger(__name__)

VAST_LEASE_REGISTRY_VERSION = 1
VAST_LEASE_REGISTRY_FILENAME = "vast-leases.json"
VAST_MANAGED_LABEL_PREFIX = "comfy-modal-vast"
_SAFE_LABEL_PART_PATTERN = re.compile(r"[^a-zA-Z0-9._-]+")
VastInstanceStatusCallback = Callable[[VastInstance], None]


@dataclass(frozen=True)
class VastLeaseRecord:
    """Persist one instance owned by a resource profile and ComfyUI installation."""

    instance_id: int
    offer_id: int
    owner_id: str
    profile_id: str
    profile_name: str
    profile_fingerprint: str
    runtime_fingerprint: str
    label: str
    actual_status: str
    ssh_host: str | None
    ssh_port: int | None
    gpu_name: str
    gpu_count: int
    gpu_ram_mb: int
    cpu_ram_mb: int
    hourly_cost_usd: float
    created_at_epoch: float
    last_activity_at_epoch: float
    idle_deadline_epoch: float
    idle_retention_seconds: float = 24 * 3600
    active_invocations: int = 0
    draining: bool = False
    last_error: str | None = None
    worker_image: str | None = None

    def __post_init__(self) -> None:
        """Validate ownership, lifecycle, and scheduling fields."""
        for field_name, value in (
            ("instance_id", self.instance_id),
            ("offer_id", self.offer_id),
            ("gpu_count", self.gpu_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer.")
        if self.active_invocations < 0:
            raise ValueError("active_invocations must not be negative.")
        if self.gpu_ram_mb < 0 or self.cpu_ram_mb < 0:
            raise ValueError("Vast lease memory fields must not be negative.")
        for field_name, value in (
            ("hourly_cost_usd", self.hourly_cost_usd),
            ("created_at_epoch", self.created_at_epoch),
            ("last_activity_at_epoch", self.last_activity_at_epoch),
            ("idle_deadline_epoch", self.idle_deadline_epoch),
            ("idle_retention_seconds", self.idle_retention_seconds),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{field_name} must be finite and non-negative.")
        if not self.owner_id.strip() or not self.profile_id.strip():
            raise ValueError("Vast lease owner and profile identities must not be empty.")
        if len(self.profile_fingerprint) != 64 or len(self.runtime_fingerprint) != 64:
            raise ValueError("Vast lease fingerprints must be SHA-256 hex digests.")
        expected_prefix = vast_managed_label_prefix(self.owner_id)
        if not self.label.startswith(f"{expected_prefix}:"):
            raise ValueError("Vast lease label does not match its owner identity.")
        if self.ssh_port is not None and self.ssh_port <= 0:
            raise ValueError("ssh_port must be positive when present.")
        if self.worker_image is not None and not self.worker_image.strip():
            raise ValueError("worker_image must be non-empty when present.")

    @property
    def environment_id(self) -> str:
        """Return the provider-neutral identity for this concrete lease."""
        return f"vast:{self.profile_id}:{self.instance_id}"

    @property
    def ready_for_work(self) -> bool:
        """Return whether this lease is running, reachable, and accepting work."""
        return (
            not self.draining
            and self.actual_status == "running"
            and bool(self.ssh_host)
            and self.ssh_port is not None
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a credential-free JSON record."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "VastLeaseRecord":
        """Build one validated record from persisted state."""
        return cls(
            instance_id=int(payload["instance_id"]),
            offer_id=int(payload["offer_id"]),
            owner_id=str(payload["owner_id"]),
            profile_id=str(payload["profile_id"]),
            profile_name=str(payload["profile_name"]),
            profile_fingerprint=str(payload["profile_fingerprint"]),
            runtime_fingerprint=str(payload["runtime_fingerprint"]),
            label=str(payload["label"]),
            actual_status=str(payload.get("actual_status") or "unknown"),
            ssh_host=(str(payload["ssh_host"]) if payload.get("ssh_host") else None),
            ssh_port=(int(payload["ssh_port"]) if payload.get("ssh_port") else None),
            gpu_name=str(payload.get("gpu_name") or "Unknown GPU"),
            gpu_count=int(payload.get("gpu_count", 0)),
            gpu_ram_mb=int(payload.get("gpu_ram_mb", 0)),
            cpu_ram_mb=int(payload.get("cpu_ram_mb", 0)),
            hourly_cost_usd=float(payload.get("hourly_cost_usd", 0.0)),
            created_at_epoch=float(payload.get("created_at_epoch", 0.0)),
            last_activity_at_epoch=float(
                payload.get("last_activity_at_epoch", 0.0)
            ),
            idle_deadline_epoch=float(payload.get("idle_deadline_epoch", 0.0)),
            idle_retention_seconds=float(
                payload.get("idle_retention_seconds", 24 * 3600)
            ),
            active_invocations=int(payload.get("active_invocations", 0)),
            draining=bool(payload.get("draining", False)),
            last_error=(str(payload["last_error"]) if payload.get("last_error") else None),
            worker_image=(
                str(payload["worker_image"]) if payload.get("worker_image") else None
            ),
        )


@dataclass(frozen=True)
class VastLeaseRegistryState:
    """Describe the complete versioned Vast lease registry."""

    leases: tuple[VastLeaseRecord, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the persisted versioned document."""
        return {
            "version": VAST_LEASE_REGISTRY_VERSION,
            "leases": [lease.to_dict() for lease in self.leases],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "VastLeaseRegistryState":
        """Build and validate one registry document."""
        version = int(payload.get("version", 0))
        if version != VAST_LEASE_REGISTRY_VERSION:
            raise ValueError(f"Unsupported Vast lease registry version {version}.")
        raw_leases = payload.get("leases")
        lease_payloads = raw_leases if isinstance(raw_leases, list) else []
        leases = tuple(
            VastLeaseRecord.from_dict(lease)
            for lease in lease_payloads
            if isinstance(lease, Mapping)
        )
        instance_ids = [lease.instance_id for lease in leases]
        if len(instance_ids) != len(set(instance_ids)):
            raise ValueError("Vast lease registry contains duplicate instance IDs.")
        return cls(leases=leases)


@dataclass
class VastLeaseRegistry:
    """Atomically persist managed Vast instances without credentials."""

    config_path: Path
    _lock: threading.RLock = field(default_factory=threading.RLock)

    @classmethod
    def for_user_directory(cls, user_directory: Path) -> "VastLeaseRegistry":
        """Create a registry below one ComfyUI user directory."""
        return cls(
            config_path=(
                user_directory.expanduser().resolve()
                / "comfyui-modal"
                / VAST_LEASE_REGISTRY_FILENAME
            )
        )

    def load(self) -> VastLeaseRegistryState:
        """Load the registry or return empty initial state."""
        with self._lock:
            if not self.config_path.exists():
                return VastLeaseRegistryState()
            try:
                payload = json.loads(self.config_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Vast lease registry {self.config_path} is unreadable."
                ) from exc
            if not isinstance(payload, Mapping):
                raise ValueError("Vast lease registry must be a JSON object.")
            return VastLeaseRegistryState.from_dict(payload)

    def save(self, state: VastLeaseRegistryState) -> None:
        """Validate and atomically persist complete registry state."""
        validated = VastLeaseRegistryState.from_dict(state.to_dict())
        serialized = json.dumps(validated.to_dict(), indent=2, sort_keys=True) + "\n"
        with self._lock:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            file_descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{self.config_path.name}.",
                suffix=".tmp",
                dir=self.config_path.parent,
            )
            temporary_path = Path(temporary_name)
            try:
                with os.fdopen(file_descriptor, "w", encoding="utf-8") as output_file:
                    output_file.write(serialized)
                    output_file.flush()
                    os.fsync(output_file.fileno())
                os.replace(temporary_path, self.config_path)
            finally:
                temporary_path.unlink(missing_ok=True)

    def upsert(self, lease: VastLeaseRecord) -> VastLeaseRecord:
        """Insert or replace one record by instance identity."""
        with self._lock:
            state = self.load()
            leases = [
                existing
                for existing in state.leases
                if existing.instance_id != lease.instance_id
            ]
            leases.append(lease)
            self.save(
                VastLeaseRegistryState(
                    leases=tuple(sorted(leases, key=lambda item: item.instance_id))
                )
            )
        return lease

    def remove(self, instance_id: int) -> VastLeaseRecord | None:
        """Remove and return one record when present."""
        with self._lock:
            state = self.load()
            removed = next(
                (lease for lease in state.leases if lease.instance_id == instance_id),
                None,
            )
            if removed is None:
                return None
            self.save(
                VastLeaseRegistryState(
                    leases=tuple(
                        lease
                        for lease in state.leases
                        if lease.instance_id != instance_id
                    )
                )
            )
        return removed

    def update(
        self,
        instance_id: int,
        transform: Callable[[VastLeaseRecord], VastLeaseRecord],
    ) -> VastLeaseRecord:
        """Atomically replace one record through a deterministic transform."""
        with self._lock:
            state = self.load()
            updated: VastLeaseRecord | None = None
            leases: list[VastLeaseRecord] = []
            for lease in state.leases:
                if lease.instance_id != instance_id:
                    leases.append(lease)
                    continue
                updated = transform(lease)
                if updated.instance_id != instance_id:
                    raise ValueError("Vast lease transforms cannot change instance ID.")
                leases.append(updated)
            if updated is None:
                raise KeyError(f"Unknown Vast lease {instance_id}.")
            self.save(VastLeaseRegistryState(leases=tuple(leases)))
            return updated


class VastLeaseManager:
    """Coordinate reusable leases and destroy them after profile idle retention."""

    def __init__(
        self,
        *,
        api_client: VastApiClient,
        registry: VastLeaseRegistry,
        owner_id: str,
        runtime_fingerprint: str,
        launch_spec_factory: Callable[
            [VastResourceProfile, str], VastInstanceLaunchSpec
        ],
        startup_timeout_seconds: float = 900.0,
        clock: Callable[[], float] = time.time,
        worker_image: str | None = None,
    ) -> None:
        """Configure acquisition, persistence, and lifecycle dependencies."""
        if not owner_id.strip():
            raise ValueError("Vast lease owner_id must not be empty.")
        if len(runtime_fingerprint) != 64:
            raise ValueError("runtime_fingerprint must be a SHA-256 hex digest.")
        if startup_timeout_seconds <= 0:
            raise ValueError("startup_timeout_seconds must be positive.")
        self.api_client = api_client
        self.registry = registry
        self.owner_id = owner_id.strip()
        self.runtime_fingerprint = runtime_fingerprint
        self.launch_spec_factory = launch_spec_factory
        self.startup_timeout_seconds = startup_timeout_seconds
        self.clock = clock
        self.worker_image = worker_image.strip() if worker_image is not None else None
        self._profile_locks_guard = threading.Lock()
        self._profile_locks: dict[str, asyncio.Lock] = {}

    async def ensure_lease(
        self,
        profile: VastResourceProfile,
        *,
        slot: int = 0,
        status_callback: VastInstanceStatusCallback | None = None,
        excluded_offer_ids: frozenset[int] = frozenset(),
    ) -> VastLeaseRecord:
        """Reuse or rent one compatible SSH-ready lease for a profile slot."""
        if slot < 0 or slot >= profile.maximum_instances:
            raise ValueError(
                f"Vast profile {profile.profile_name!r} supports slots 0 through "
                f"{profile.maximum_instances - 1}."
            )
        slot_profile_id = _slot_profile_id(profile.profile_id, slot)
        async with self._profile_lock(slot_profile_id):
            existing = await self._reusable_lease(
                profile,
                slot_profile_id,
                status_callback=status_callback,
            )
            if existing is not None:
                return existing
            return await self._rent_lease(
                profile,
                slot_profile_id,
                status_callback=status_callback,
                excluded_offer_ids=excluded_offer_ids,
            )

    async def _reusable_lease(
        self,
        profile: VastResourceProfile,
        slot_profile_id: str,
        *,
        status_callback: VastInstanceStatusCallback | None,
    ) -> VastLeaseRecord | None:
        """Return one matching live lease after refreshing its API state."""
        fingerprint = vast_profile_fingerprint(profile)
        candidates = [
            lease
            for lease in self.registry.load().leases
            if lease.owner_id == self.owner_id
            and lease.profile_id == slot_profile_id
            and lease.profile_fingerprint == fingerprint
            and (
                (
                    self.worker_image is None
                    and lease.runtime_fingerprint == self.runtime_fingerprint
                )
                or (
                    self.worker_image is not None
                    and lease.worker_image in {None, self.worker_image}
                )
            )
            and not lease.draining
        ]
        for lease in sorted(candidates, key=lambda item: item.created_at_epoch):
            try:
                instance = await self.api_client.show_instance(lease.instance_id)
            except VastInstanceNotFoundError:
                self.registry.remove(lease.instance_id)
                continue
            if instance.actual_status == "stopped":
                await self.api_client.set_instance_state(lease.instance_id, "running")
                instance = await self.api_client.wait_until_ready(
                    lease.instance_id,
                    timeout_seconds=self.startup_timeout_seconds,
                    status_callback=status_callback,
                )
            refreshed = self._refresh_record(lease, instance)
            if refreshed.runtime_fingerprint != self.runtime_fingerprint:
                logger.info(
                    "Adopting compatible Vast lease instance_id=%d from runtime=%s "
                    "to runtime=%s for unchanged worker image.",
                    refreshed.instance_id,
                    refreshed.runtime_fingerprint[:12],
                    self.runtime_fingerprint[:12],
                )
                refreshed = replace(
                    refreshed,
                    runtime_fingerprint=self.runtime_fingerprint,
                    worker_image=self.worker_image,
                )
            self.registry.upsert(refreshed)
            if refreshed.ready_for_work:
                logger.info(
                    "Reusing Vast lease profile=%s instance=%d gpu=%s hourly_cost=%.4f.",
                    profile.profile_name,
                    refreshed.instance_id,
                    refreshed.gpu_name,
                    refreshed.hourly_cost_usd,
                )
                return refreshed
        return None

    async def _rent_lease(
        self,
        profile: VastResourceProfile,
        slot_profile_id: str,
        *,
        status_callback: VastInstanceStatusCallback | None,
        excluded_offer_ids: frozenset[int],
    ) -> VastLeaseRecord:
        """Search, rent, and persist the first still-available compatible offer."""
        offers = await self.api_client.search_offers(profile)
        if not offers:
            raise VastApiError(
                f"No Vast.ai offer satisfies profile {profile.profile_name!r}."
            )
        unavailable_offer_ids = set(excluded_offer_ids)
        for search_attempt in range(2):
            for offer in offers:
                if offer.offer_id in unavailable_offer_ids:
                    continue
                try:
                    return await self._create_from_offer(
                        profile,
                        slot_profile_id,
                        offer,
                        status_callback=status_callback,
                    )
                except VastInstanceNotFoundError:
                    unavailable_offer_ids.add(offer.offer_id)
                    logger.warning(
                        "Vast instance disappeared during provider startup "
                        "profile=%s offer=%d; trying next candidate.",
                        profile.profile_name,
                        offer.offer_id,
                    )
                except VastOfferUnavailableError:
                    unavailable_offer_ids.add(offer.offer_id)
                    logger.info(
                        "Vast offer disappeared before rental profile=%s offer=%d; trying next candidate.",
                        profile.profile_name,
                        offer.offer_id,
                    )
            if search_attempt == 0:
                offers = await self.api_client.search_offers(
                    profile,
                    force_refresh=True,
                )
        raise VastOfferUnavailableError(
            f"Compatible Vast.ai offers disappeared before profile "
            f"{profile.profile_name!r} could be rented."
        )

    async def _create_from_offer(
        self,
        profile: VastResourceProfile,
        slot_profile_id: str,
        offer: VastOffer,
        *,
        status_callback: VastInstanceStatusCallback | None,
    ) -> VastLeaseRecord:
        """Create, wait for, and persist one selected offer."""
        label = vast_managed_label(
            self.owner_id,
            slot_profile_id,
            vast_profile_fingerprint(profile),
            self.runtime_fingerprint,
        )
        launch_spec = self.launch_spec_factory(profile, label)
        created = await self.api_client.create_instance(offer.offer_id, launch_spec)
        now = self.clock()
        provisional = VastLeaseRecord(
            instance_id=created.instance_id,
            offer_id=offer.offer_id,
            owner_id=self.owner_id,
            profile_id=slot_profile_id,
            profile_name=profile.profile_name,
            profile_fingerprint=vast_profile_fingerprint(profile),
            runtime_fingerprint=self.runtime_fingerprint,
            worker_image=self.worker_image,
            label=label,
            actual_status="loading",
            ssh_host=None,
            ssh_port=None,
            gpu_name=offer.gpu_name,
            gpu_count=offer.num_gpus,
            gpu_ram_mb=offer.gpu_ram_mb,
            cpu_ram_mb=offer.cpu_ram_mb,
            hourly_cost_usd=offer.hourly_cost_usd,
            created_at_epoch=now,
            last_activity_at_epoch=now,
            idle_deadline_epoch=now + profile.idle_retention_seconds,
            idle_retention_seconds=profile.idle_retention_seconds,
        )
        self.registry.upsert(provisional)
        try:
            instance = await self.api_client.wait_until_ready(
                created.instance_id,
                timeout_seconds=self.startup_timeout_seconds,
                status_callback=status_callback,
            )
        except VastInstanceNotFoundError:
            self.registry.remove(created.instance_id)
            raise
        except (TimeoutError, VastApiError) as exc:
            failed = replace(provisional, last_error=str(exc))
            self.registry.upsert(failed)
            try:
                await self.api_client.destroy_instance(created.instance_id)
            except (VastApiError, VastInstanceNotFoundError) as cleanup_error:
                logger.warning(
                    "Unable to destroy failed Vast lease instance=%d: %s",
                    created.instance_id,
                    cleanup_error,
                )
            else:
                self.registry.remove(created.instance_id)
            raise
        ready = self._refresh_record(provisional, instance)
        self.registry.upsert(ready)
        logger.info(
            "Rented Vast lease profile=%s instance=%d offer=%d gpu=%s hourly_cost=%.4f idle_hours=%.1f.",
            profile.profile_name,
            ready.instance_id,
            offer.offer_id,
            ready.gpu_name,
            ready.hourly_cost_usd,
            profile.idle_retention_seconds / 3600.0,
        )
        return ready

    def begin_activity(self, instance_id: int) -> VastLeaseRecord:
        """Mark one lease busy before staging or execution begins."""
        now = self.clock()
        return self.registry.update(
            instance_id,
            lambda lease: replace(
                lease,
                active_invocations=lease.active_invocations + 1,
                last_activity_at_epoch=now,
                last_error=None,
            ),
        )

    def finish_activity(
        self,
        instance_id: int,
        *,
        idle_retention_seconds: float,
        error: str | None = None,
    ) -> VastLeaseRecord:
        """Release one activity and extend idle destruction from terminal time."""
        if idle_retention_seconds < 0 or not math.isfinite(idle_retention_seconds):
            raise ValueError("idle_retention_seconds must be finite and non-negative.")
        now = self.clock()

        def finish(lease: VastLeaseRecord) -> VastLeaseRecord:
            """Return terminal activity state without underflowing the count."""
            if lease.active_invocations <= 0:
                raise RuntimeError(
                    f"Vast lease {instance_id} has no active invocation to finish."
                )
            remaining = lease.active_invocations - 1
            return replace(
                lease,
                active_invocations=remaining,
                last_activity_at_epoch=now,
                idle_deadline_epoch=(
                    now + idle_retention_seconds
                    if remaining == 0
                    else lease.idle_deadline_epoch
                ),
                last_error=error,
            )

        return self.registry.update(instance_id, finish)

    async def destroy_expired(self) -> tuple[int, ...]:
        """Destroy owned idle leases whose retention deadline has elapsed."""
        now = self.clock()
        expired = [
            lease
            for lease in self.registry.load().leases
            if lease.owner_id == self.owner_id
            and lease.active_invocations == 0
            and lease.idle_deadline_epoch <= now
        ]
        destroyed: list[int] = []
        for lease in expired:
            try:
                await self.api_client.destroy_instance(lease.instance_id)
            except VastInstanceNotFoundError:
                pass
            except VastApiError as exc:
                error_message = str(exc)
                self.registry.update(
                    lease.instance_id,
                    lambda record: replace(record, last_error=error_message),
                )
                logger.warning(
                    "Unable to destroy expired Vast lease instance=%d: %s",
                    lease.instance_id,
                    exc,
                )
                continue
            self.registry.remove(lease.instance_id)
            destroyed.append(lease.instance_id)
            logger.info(
                "Destroyed expired Vast lease instance=%d profile=%s.",
                lease.instance_id,
                lease.profile_name,
            )
        return tuple(destroyed)

    async def destroy_owned_lease(self, instance_id: int) -> bool:
        """Destroy one exact registry-owned lease after checking its API label."""
        lease = next(
            (
                candidate
                for candidate in self.registry.load().leases
                if candidate.instance_id == instance_id
                and candidate.owner_id == self.owner_id
            ),
            None,
        )
        if lease is None:
            return False
        if lease.active_invocations:
            raise RuntimeError(
                f"Vast lease {instance_id} has active work and cannot be destroyed."
            )
        try:
            instance = await self.api_client.show_instance(instance_id)
        except VastInstanceNotFoundError:
            self.registry.remove(instance_id)
            return True
        if instance.label != lease.label:
            raise RuntimeError(
                f"Vast instance {instance_id} label no longer matches its managed lease."
            )
        await self.api_client.destroy_instance(instance_id)
        self.registry.remove(instance_id)
        return True

    async def reconcile(self) -> tuple[VastLeaseRecord, ...]:
        """Refresh owned registry records and remove instances no longer visible."""
        visible_instances = {
            instance.instance_id: instance
            for instance in await self.api_client.list_instances()
        }
        reconciled: list[VastLeaseRecord] = []
        for lease in self.registry.load().leases:
            if lease.owner_id != self.owner_id:
                reconciled.append(lease)
                continue
            instance = visible_instances.get(lease.instance_id)
            if instance is None:
                self.registry.remove(lease.instance_id)
                continue
            if instance.label != lease.label:
                self.registry.upsert(
                    replace(
                        lease,
                        draining=True,
                        last_error="Vast instance label no longer matches managed state.",
                    )
                )
                continue
            recovered = replace(
                self._refresh_record(lease, instance),
                active_invocations=0,
            )
            self.registry.upsert(recovered)
            reconciled.append(recovered)
        return tuple(reconciled)

    def incremental_retention_cost_usd(
        self,
        profile: VastResourceProfile,
        *,
        existing_lease: VastLeaseRecord | None,
        predicted_execution_seconds: float,
    ) -> float:
        """Estimate compute cost added by a job and its resulting retention deadline."""
        if predicted_execution_seconds < 0:
            raise ValueError("predicted_execution_seconds must not be negative.")
        hourly_cost = (
            existing_lease.hourly_cost_usd
            if existing_lease is not None
            else profile.maximum_hourly_cost_usd
        )
        new_deadline = (
            self.clock()
            + predicted_execution_seconds
            + profile.idle_retention_seconds
        )
        covered_until = (
            existing_lease.idle_deadline_epoch
            if existing_lease is not None
            else self.clock()
        )
        incremental_seconds = max(0.0, new_deadline - covered_until)
        return incremental_seconds / 3600.0 * hourly_cost

    def _refresh_record(
        self,
        lease: VastLeaseRecord,
        instance: VastInstance,
    ) -> VastLeaseRecord:
        """Apply current API status to one persisted record."""
        return replace(
            lease,
            actual_status=instance.actual_status,
            ssh_host=instance.ssh_host,
            ssh_port=instance.ssh_port,
            gpu_name=instance.gpu_name or lease.gpu_name,
            gpu_count=instance.num_gpus or lease.gpu_count,
            gpu_ram_mb=instance.gpu_ram_mb or lease.gpu_ram_mb,
            cpu_ram_mb=instance.cpu_ram_mb or lease.cpu_ram_mb,
            hourly_cost_usd=(
                instance.hourly_cost_usd
                if instance.hourly_cost_usd is not None
                else lease.hourly_cost_usd
            ),
            last_error=None,
        )

    def _profile_lock(self, profile_id: str) -> asyncio.Lock:
        """Return the process-local acquisition lock for one profile slot."""
        with self._profile_locks_guard:
            lock = self._profile_locks.get(profile_id)
            if lock is None:
                lock = asyncio.Lock()
                self._profile_locks[profile_id] = lock
            return lock


def vast_profile_fingerprint(profile: VastResourceProfile) -> str:
    """Return a deterministic fingerprint for all lease-shaping profile fields."""
    payload = asdict(profile)
    payload["rental_type"] = profile.rental_type.value
    serialized = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    return hashlib.sha256(serialized).hexdigest()


def vast_managed_label_prefix(owner_id: str) -> str:
    """Return the exact label prefix owned by one ComfyUI installation."""
    owner_slug = _safe_label_part(owner_id, maximum_length=32)
    return f"{VAST_MANAGED_LABEL_PREFIX}:{owner_slug}"


def vast_managed_label(
    owner_id: str,
    profile_id: str,
    profile_fingerprint: str,
    runtime_fingerprint: str,
) -> str:
    """Return a bounded ownership label for one profile/runtime combination."""
    profile_slug = _safe_label_part(profile_id, maximum_length=24)
    return (
        f"{vast_managed_label_prefix(owner_id)}:{profile_slug}:"
        f"{profile_fingerprint[:10]}:{runtime_fingerprint[:10]}"
    )


def _safe_label_part(value: str, *, maximum_length: int) -> str:
    """Return one non-empty Vast label component."""
    normalized = _SAFE_LABEL_PART_PATTERN.sub("-", value.strip()).strip("-._")
    if not normalized:
        normalized = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return normalized[:maximum_length]


def _slot_profile_id(profile_id: str, slot: int) -> str:
    """Return a deterministic profile identity for one capacity slot."""
    return profile_id if slot == 0 else f"{profile_id}-slot-{slot}"


__all__ = [
    "VAST_LEASE_REGISTRY_FILENAME",
    "VAST_LEASE_REGISTRY_VERSION",
    "VAST_MANAGED_LABEL_PREFIX",
    "VastLeaseManager",
    "VastLeaseRecord",
    "VastLeaseRegistry",
    "VastLeaseRegistryState",
    "vast_managed_label",
    "vast_managed_label_prefix",
    "vast_profile_fingerprint",
]
