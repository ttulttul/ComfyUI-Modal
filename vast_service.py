"""Application service joining Vast marketplace, leases, SSH, sync, and execution."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

if __package__:
    from .execution_environments import (
        EnvironmentCapabilities,
        EnvironmentHealth,
        EnvironmentSchedulingState,
        ExecutionProvider,
        GpuCapability,
    )
    from .runtime_environment import build_remote_runtime_identity
    from .settings import ModalSyncSettings, discover_comfyui_user_directory
    from .sync_engine import ModalAssetSyncEngine
    from .vast_api import VastApiClient, VastApiError
    from .vast_executor import VastExecutorClient
    from .vast_leases import (
        VastLeaseManager,
        VastLeaseRecord,
        VastLeaseRegistry,
        vast_profile_fingerprint,
    )
    from .vast_models import VastInstance, VastOffer, VastResourceProfile
    from .vast_runtime import VastRuntimeConfiguration, VastRuntimeManager
    from .vast_ssh import VastSshRunner, VastSshVolumeBackend, vast_connection_from_lease
else:  # pragma: no cover - direct debugging imports.
    from execution_environments import (
        EnvironmentCapabilities,
        EnvironmentHealth,
        EnvironmentSchedulingState,
        ExecutionProvider,
        GpuCapability,
    )
    from runtime_environment import build_remote_runtime_identity
    from settings import ModalSyncSettings, discover_comfyui_user_directory
    from sync_engine import ModalAssetSyncEngine
    from vast_api import VastApiClient, VastApiError
    from vast_executor import VastExecutorClient
    from vast_leases import (
        VastLeaseManager,
        VastLeaseRecord,
        VastLeaseRegistry,
        vast_profile_fingerprint,
    )
    from vast_models import VastInstance, VastOffer, VastResourceProfile
    from vast_runtime import VastRuntimeConfiguration, VastRuntimeManager
    from vast_ssh import VastSshRunner, VastSshVolumeBackend, vast_connection_from_lease

logger = logging.getLogger(__name__)
VAST_API_KEY_ENV = "VAST_API_KEY"
VAST_API_BASE_URL_ENV = "COMFY_MODAL_VAST_API_BASE_URL"
VAST_SSH_IDENTITY_FILE_ENV = "COMFY_MODAL_VAST_SSH_IDENTITY_FILE"
VAST_OFFER_PREFETCH_CONCURRENCY = 8


def _vast_startup_status_message(instance: VastInstance) -> str:
    """Translate one provider lifecycle record into stable user-facing progress."""
    provider_message = (instance.status_message or "").casefold()
    instance_label = f"Vast.ai instance {instance.instance_id}"
    if "unauthorized" in provider_message or "denied" in provider_message:
        return f"{instance_label} could not download the worker image"
    if "download complete" in provider_message:
        return f"{instance_label} is downloading the worker image (layer complete)"
    if any(word in provider_message for word in ("download", "pull", "extract")):
        return f"{instance_label} is downloading the worker image"
    if instance.actual_status.casefold() == "running":
        return f"{instance_label} is waiting for SSH access"
    return f"{instance_label} is starting ({instance.actual_status})"


@dataclass(frozen=True)
class VastProfileQuote:
    """Describe the least effective-hourly-price profile for one component."""

    profile: VastResourceProfile
    offer: VastOffer | None
    existing_lease: VastLeaseRecord | None
    predicted_incremental_cost_usd: float
    retention_seconds_charged: float
    predicted_execution_seconds: float

    @property
    def hourly_cost_usd(self) -> float:
        """Return the live lease or marketplace hourly price."""
        if self.existing_lease is not None:
            return self.existing_lease.hourly_cost_usd
        if self.offer is None:
            raise RuntimeError("Vast quote has neither an offer nor a lease.")
        return self.offer.hourly_cost_usd

    @property
    def predicted_retention_cost_usd(self) -> float:
        """Return the incremental bill estimate including configured retention."""
        return self.retention_seconds_charged / 3600.0 * self.hourly_cost_usd


@dataclass(frozen=True)
class VastSearchRequirements:
    """Describe component memory floors that affect a marketplace search."""

    minimum_vram_bytes: int
    minimum_ram_bytes: int


class VastService:
    """Provide one credential-safe controller for workflow Vast operations."""

    def __init__(
        self,
        *,
        settings: ModalSyncSettings,
        repo_root: Path,
        user_directory: Path,
        api_client: VastApiClient,
        runtime_configuration: VastRuntimeConfiguration,
        registry: VastLeaseRegistry,
        identity_file: Path | None = None,
    ) -> None:
        """Initialize the controller and its persistent lease manager."""
        self.settings = settings
        self.repo_root = repo_root.resolve()
        self.user_directory = user_directory.resolve()
        self.api_client = api_client
        self.runtime_configuration = runtime_configuration
        self.registry = registry
        self.identity_file = identity_file
        self.lease_manager = VastLeaseManager(
            api_client=api_client,
            registry=registry,
            owner_id=settings.app_name,
            runtime_fingerprint=runtime_configuration.runtime_fingerprint,
            launch_spec_factory=runtime_configuration.launch_spec,
            startup_timeout_seconds=runtime_configuration.startup_timeout_seconds,
        )

    @classmethod
    def from_environment(
        cls,
        settings: ModalSyncSettings,
        *,
        repo_root: Path,
        environment: Mapping[str, str] | None = None,
    ) -> "VastService":
        """Resolve credentials, runtime identity, and local state from the environment."""
        source = os.environ if environment is None else environment
        api_key = str(source.get(VAST_API_KEY_ENV) or "").strip()
        if not api_key:
            raise RuntimeError(
                f"Set {VAST_API_KEY_ENV} before selecting Vast.ai execution."
            )
        user_directory = discover_comfyui_user_directory(settings)
        if user_directory is None:
            raise RuntimeError(
                "Vast.ai execution requires a persistent ComfyUI user directory."
            )
        identity = build_remote_runtime_identity(
            repo_root=repo_root,
            comfyui_root=settings.comfyui_root,
            custom_nodes_dir=settings.custom_nodes_dir,
            settings=settings,
        )
        runtime = VastRuntimeConfiguration.from_environment(
            identity.fingerprint,
            environment=source,
        )
        base_url = str(source.get(VAST_API_BASE_URL_ENV) or "").strip()
        api_client = (
            VastApiClient(api_key, base_url=base_url)
            if base_url
            else VastApiClient(api_key)
        )
        raw_identity_file = str(source.get(VAST_SSH_IDENTITY_FILE_ENV) or "").strip()
        identity_file = (
            Path(raw_identity_file).expanduser().resolve()
            if raw_identity_file
            else None
        )
        return cls(
            settings=settings,
            repo_root=repo_root,
            user_directory=user_directory,
            api_client=api_client,
            runtime_configuration=runtime,
            registry=VastLeaseRegistry.for_user_directory(user_directory),
            identity_file=identity_file,
        )

    async def quote_best_profile(
        self,
        profiles: Sequence[VastResourceProfile],
        *,
        minimum_vram_bytes: int,
        minimum_ram_bytes: int,
        predicted_execution_seconds: float,
    ) -> VastProfileQuote:
        """Return the cheapest compatible existing lease or current offer."""
        if not profiles:
            raise ValueError("At least one Vast resource profile is required.")
        adjusted_profiles = tuple(
            _profile_for_requirements(profile, minimum_vram_bytes, minimum_ram_bytes)
            for profile in profiles
        )
        quotes = await asyncio.gather(
            *(
                self._quote_profile(profile, predicted_execution_seconds)
                for profile in adjusted_profiles
            ),
            return_exceptions=True,
        )
        candidates = [quote for quote in quotes if isinstance(quote, VastProfileQuote)]
        if candidates:
            return min(
                candidates,
                key=lambda quote: (
                    quote.hourly_cost_usd,
                    quote.profile.profile_id,
                ),
            )
        errors = [str(error) for error in quotes if isinstance(error, BaseException)]
        raise VastApiError(
            "No Vast.ai profile currently has a compatible offer: "
            + "; ".join(errors)
        )

    async def prefetch_offers(
        self,
        profiles: Sequence[VastResourceProfile],
        requirements: Sequence[VastSearchRequirements],
    ) -> None:
        """Populate cached effective-profile searches with bounded parallelism."""
        if not profiles or not requirements:
            return
        pending_profiles = self._distinct_marketplace_profiles(
            profiles,
            requirements,
        )
        if not pending_profiles:
            return
        logger.info(
            "Prefetching %d unique Vast marketplace search(es) in parallel for "
            "%d component requirement set(s).",
            len(pending_profiles),
            len(requirements),
        )
        results = await self._search_offer_profiles_in_parallel(pending_profiles)
        cancelled = next(
            (
                result
                for result in results
                if isinstance(result, asyncio.CancelledError)
            ),
            None,
        )
        if cancelled is not None:
            raise cancelled
        failure_count = sum(
            isinstance(result, (OSError, RuntimeError, ValueError))
            for result in results
        )
        logger.info(
            "Completed Vast marketplace prefetch searches=%d successful=%d "
            "failed=%d.",
            len(results),
            len(results) - failure_count,
            failure_count,
        )

    def _distinct_marketplace_profiles(
        self,
        profiles: Sequence[VastResourceProfile],
        requirements: Sequence[VastSearchRequirements],
    ) -> tuple[VastResourceProfile, ...]:
        """Return deduplicated effective profiles without reusable leases."""
        unique_profiles: dict[str, VastResourceProfile] = {}
        for requirement in requirements:
            for profile in profiles:
                adjusted_profile = _profile_for_requirements(
                    profile,
                    requirement.minimum_vram_bytes,
                    requirement.minimum_ram_bytes,
                )
                if self._existing_lease(adjusted_profile) is not None:
                    continue
                query_key = json.dumps(
                    adjusted_profile.search_payload(),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                unique_profiles.setdefault(query_key, adjusted_profile)
        return tuple(unique_profiles.values())

    async def _search_offer_profiles_in_parallel(
        self,
        profiles: Sequence[VastResourceProfile],
    ) -> tuple[object, ...]:
        """Search effective profiles in batches that respect the API bound."""
        results: list[object] = []
        for offset in range(0, len(profiles), VAST_OFFER_PREFETCH_CONCURRENCY):
            batch = profiles[
                offset : offset + VAST_OFFER_PREFETCH_CONCURRENCY
            ]
            results.extend(
                await asyncio.gather(
                    *(self.api_client.search_offers(profile) for profile in batch),
                    return_exceptions=True,
                )
            )
        return tuple(results)

    async def _quote_profile(
        self,
        profile: VastResourceProfile,
        predicted_execution_seconds: float,
    ) -> VastProfileQuote:
        """Quote compute at its hourly rate while tracking retention separately."""
        existing = self._existing_lease(profile)
        if existing is not None:
            retention_cost = self.lease_manager.incremental_retention_cost_usd(
                profile,
                existing_lease=existing,
                predicted_execution_seconds=predicted_execution_seconds,
            )
            charged_seconds = (
                retention_cost / existing.hourly_cost_usd * 3600.0
                if existing.hourly_cost_usd > 0
                else 0.0
            )
            return VastProfileQuote(
                profile=profile,
                offer=None,
                existing_lease=existing,
                predicted_incremental_cost_usd=(
                    predicted_execution_seconds
                    / 3600.0
                    * existing.hourly_cost_usd
                ),
                retention_seconds_charged=charged_seconds,
                predicted_execution_seconds=predicted_execution_seconds,
            )
        offers = await self.api_client.search_offers(profile)
        if not offers:
            raise VastApiError(
                f"No offer satisfies Vast profile {profile.profile_name!r}."
            )
        offer = offers[0]
        charged_seconds = predicted_execution_seconds + profile.idle_retention_seconds
        return VastProfileQuote(
            profile=profile,
            offer=offer,
            existing_lease=None,
            predicted_incremental_cost_usd=(
                predicted_execution_seconds / 3600.0 * offer.hourly_cost_usd
            ),
            retention_seconds_charged=charged_seconds,
            predicted_execution_seconds=predicted_execution_seconds,
        )

    async def acquire(
        self,
        quote: VastProfileQuote,
        *,
        slot: int = 0,
        status_callback: Callable[[str], None] | None = None,
    ) -> VastLeaseRecord:
        """Reuse or rent the selected quoted profile capacity slot."""
        if status_callback is not None:
            status_callback("Requesting Vast.ai capacity")

        def emit_instance_status(instance: VastInstance) -> None:
            """Forward normalized provider progress when a caller is listening."""
            if status_callback is not None:
                status_callback(_vast_startup_status_message(instance))

        lease = await self.lease_manager.ensure_lease(
            quote.profile,
            slot=slot,
            status_callback=(
                emit_instance_status if status_callback is not None else None
            ),
        )
        if status_callback is not None:
            status_callback("Initializing Vast.ai worker")
        await asyncio.to_thread(self._initialize_runtime, lease)
        if status_callback is not None:
            status_callback("Vast.ai worker is ready")
        return lease

    def quote_best_profile_sync(self, *args: object, **kwargs: object) -> VastProfileQuote:
        """Synchronously quote a profile from ComfyUI's queue worker thread."""
        return asyncio.run(self.quote_best_profile(*args, **kwargs))  # type: ignore[arg-type]

    def prefetch_offers_sync(
        self,
        profiles: Sequence[VastResourceProfile],
        requirements: Sequence[VastSearchRequirements],
    ) -> None:
        """Synchronously prefetch searches from ComfyUI's queue worker thread."""
        asyncio.run(self.prefetch_offers(profiles, requirements))

    def acquire_sync(
        self,
        quote: VastProfileQuote,
        *,
        slot: int = 0,
        status_callback: Callable[[str], None] | None = None,
    ) -> VastLeaseRecord:
        """Synchronously acquire a quote slot from ComfyUI's queue worker thread."""
        return asyncio.run(
            self.acquire(
                quote,
                slot=slot,
                status_callback=status_callback,
            )
        )

    def scheduling_state(self, quote: VastProfileQuote) -> EnvironmentSchedulingState:
        """Expose a quote to the provider-neutral cost scheduler."""
        offer = quote.offer
        lease = quote.existing_lease
        gpu_name = lease.gpu_name if lease is not None else offer.gpu_name if offer else "Vast GPU"
        gpu_count = lease.gpu_count if lease is not None else offer.num_gpus if offer else 1
        gpu_ram_mb = lease.gpu_ram_mb if lease is not None else offer.gpu_ram_mb if offer else 0
        cpu_ram_mb = lease.cpu_ram_mb if lease is not None else offer.cpu_ram_mb if offer else 0
        environment_id = lease.environment_id if lease is not None else quote.profile.environment_id
        return EnvironmentSchedulingState(
            environment_id=environment_id,
            provider=ExecutionProvider.VAST,
            enabled=True,
            health=EnvironmentHealth.READY,
            cost_usd_per_second=quote.hourly_cost_usd / 3600.0,
            capabilities=EnvironmentCapabilities(
                architecture="x86_64",
                operating_system="linux",
                cpu_count=max(
                    1,
                    math.ceil(quote.profile.minimum_cpu_cores or 1.0),
                ),
                total_ram_bytes=cpu_ram_mb * 1024**2,
                available_ram_bytes=None,
                available_disk_bytes=int(quote.profile.allocated_disk_gb * 1024**3),
                docker_version="vast-container",
                docker_rootless=False,
                nvidia_container_runtime=True,
                gpus=tuple(
                    GpuCapability(
                        uuid=f"vast-{environment_id}-{index}",
                        name=gpu_name,
                        total_vram_bytes=gpu_ram_mb * 1024**2,
                    )
                    for index in range(gpu_count)
                ),
            ),
            maximum_workers=1,
            cold_start_seconds=0.0,
        )

    def lease_for_environment_id(self, environment_id: str) -> VastLeaseRecord:
        """Resolve a concrete managed lease from an assigned environment ID."""
        lease = next(
            (
                record
                for record in self.registry.load().leases
                if record.environment_id == environment_id
            ),
            None,
        )
        if lease is None:
            raise KeyError(f"Unknown managed Vast environment {environment_id!r}.")
        return lease

    def sync_engine(self, lease: VastLeaseRecord) -> ModalAssetSyncEngine:
        """Build a direct-filesystem content-addressed sync engine for one lease."""
        runner = self._runner(lease)
        vast_settings = replace(
            self.settings,
            execution_mode="vast",
            local_storage_root=(
                self.settings.local_storage_root / "vast" / str(lease.instance_id)
            ).resolve(),
            remote_storage_root=str(self.runtime_configuration.remote_storage_root),
        )
        return ModalAssetSyncEngine(
            volume=VastSshVolumeBackend(
                runner,
                storage_root=self.runtime_configuration.remote_storage_root,
            ),
            settings=vast_settings,
        )

    def executor(self) -> VastExecutorClient:
        """Return the direct worker executor sharing this controller's lease state."""
        return VastExecutorClient(
            registry=self.registry,
            activity_manager=self.lease_manager,
            runtime_configuration=self.runtime_configuration,
            user_directory=self.user_directory,
            settings=self.settings,
            identity_file=self.identity_file,
        )

    def _runner(self, lease: VastLeaseRecord) -> VastSshRunner:
        """Return a direct SSH runner for one ready lease."""
        return VastSshRunner(
            vast_connection_from_lease(
                ssh_host=lease.ssh_host,
                ssh_port=lease.ssh_port,
                user_directory=self.user_directory,
                identity_file=self.identity_file,
            )
        )

    def _initialize_runtime(self, lease: VastLeaseRecord) -> None:
        """Require the pinned worker and publish its initial idle fail-safe state."""
        runtime = VastRuntimeManager(
            runner=self._runner(lease),
            configuration=self.runtime_configuration,
        )
        runtime.ensure_worker()
        runtime.update_watchdog(lease)

    def _existing_lease(self, profile: VastResourceProfile) -> VastLeaseRecord | None:
        """Return the oldest ready lease with an exact profile/runtime identity."""
        fingerprint = vast_profile_fingerprint(profile)
        matches = [
            lease
            for lease in self.registry.load().leases
            if lease.owner_id == self.settings.app_name
            and lease.profile_id == profile.profile_id
            and lease.profile_fingerprint == fingerprint
            and lease.runtime_fingerprint == self.runtime_configuration.runtime_fingerprint
            and lease.ready_for_work
        ]
        return min(matches, key=lambda lease: lease.created_at_epoch, default=None)


def _profile_for_requirements(
    profile: VastResourceProfile,
    minimum_vram_bytes: int,
    minimum_ram_bytes: int,
) -> VastResourceProfile:
    """Raise workflow profile memory floors to inferred component requirements."""
    return replace(
        profile,
        minimum_gpu_ram_mb=_raised_memory_floor_mb(
            profile.minimum_gpu_ram_mb,
            minimum_vram_bytes,
        ),
        minimum_cpu_ram_mb=_raised_memory_floor_mb(
            profile.minimum_cpu_ram_mb,
            minimum_ram_bytes,
        ),
    )


def _raised_memory_floor_mb(
    configured_floor_mb: int | None,
    inferred_floor_bytes: int,
) -> int | None:
    """Apply an inferred requirement while preserving an unconstrained zero."""
    inferred_floor_mb = math.ceil(inferred_floor_bytes / 1024**2)
    if configured_floor_mb is None and inferred_floor_mb == 0:
        return None
    return max(configured_floor_mb or 0, inferred_floor_mb)


__all__ = [
    "VAST_API_BASE_URL_ENV",
    "VAST_API_KEY_ENV",
    "VAST_SSH_IDENTITY_FILE_ENV",
    "VastProfileQuote",
    "VastSearchRequirements",
    "VastService",
]
