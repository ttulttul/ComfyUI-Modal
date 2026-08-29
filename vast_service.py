"""Application service joining Vast marketplace, leases, SSH, sync, and execution."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
from dataclasses import dataclass, replace
from functools import partial
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
    from .huggingface_assets import HuggingFaceAssetRegistry
    from .huggingface_discovery import HuggingFaceAssetDiscovery
    from .runtime_environment import build_remote_runtime_identity
    from .r2_cache import R2CacheClient
    from .settings import ModalSyncSettings, discover_comfyui_user_directory
    from .sync_engine import ModalAssetSyncEngine
    from .vast_api import VastApiClient, VastApiError, VastInstanceNotFoundError
    from .vast_executor import VastExecutorClient
    from .vast_image_build import (
        VAST_IMAGE_BUILD_COMMAND,
        VastWorkerImageBuildError,
        VastWorkerImageBuilder,
    )
    from .vast_leases import (
        VastLeaseManager,
        VastLeaseRecord,
        VastLeaseRegistry,
        vast_profile_fingerprint,
    )
    from .vast_models import VastInstance, VastOffer, VastResourceProfile
    from .vast_runtime import (
        VAST_IMAGE_ENV,
        VastRuntimeConfiguration,
        VastRuntimeFingerprintDriftError,
        VastRuntimeManager,
    )
    from .vast_ssh import (
        VastSshError,
        VastSshRunner,
        VastSshVolumeBackend,
        vast_connection_from_lease,
    )
else:  # pragma: no cover - direct debugging imports.
    from execution_environments import (
        EnvironmentCapabilities,
        EnvironmentHealth,
        EnvironmentSchedulingState,
        ExecutionProvider,
        GpuCapability,
    )
    from huggingface_assets import HuggingFaceAssetRegistry
    from huggingface_discovery import HuggingFaceAssetDiscovery
    from runtime_environment import build_remote_runtime_identity
    from r2_cache import R2CacheClient
    from settings import ModalSyncSettings, discover_comfyui_user_directory
    from sync_engine import ModalAssetSyncEngine
    from vast_api import VastApiClient, VastApiError, VastInstanceNotFoundError
    from vast_executor import VastExecutorClient
    from vast_image_build import (
        VAST_IMAGE_BUILD_COMMAND,
        VastWorkerImageBuildError,
        VastWorkerImageBuilder,
    )
    from vast_leases import (
        VastLeaseManager,
        VastLeaseRecord,
        VastLeaseRegistry,
        vast_profile_fingerprint,
    )
    from vast_models import VastInstance, VastOffer, VastResourceProfile
    from vast_runtime import (
        VAST_IMAGE_ENV,
        VastRuntimeConfiguration,
        VastRuntimeFingerprintDriftError,
        VastRuntimeManager,
    )
    from vast_ssh import (
        VastSshError,
        VastSshRunner,
        VastSshVolumeBackend,
        vast_connection_from_lease,
    )

logger = logging.getLogger(__name__)
VAST_API_KEY_ENV = "VAST_API_KEY"
VAST_API_BASE_URL_ENV = "COMFY_MODAL_VAST_API_BASE_URL"
VAST_SSH_IDENTITY_FILE_ENV = "COMFY_MODAL_VAST_SSH_IDENTITY_FILE"
VAST_OFFER_PREFETCH_CONCURRENCY = 8
VAST_INSTANCE_SETUP_REPLACEMENTS = 1


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
    if instance.lifecycle_status.casefold() == "running":
        return f"{instance_label} is waiting for SSH access"
    return f"{instance_label} is starting ({instance.lifecycle_status})"


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
        r2_cache: R2CacheClient | None = None,
        image_builder: VastWorkerImageBuilder | None = None,
        update_process_environment: bool = False,
    ) -> None:
        """Initialize the controller and its persistent lease manager."""
        self.settings = settings
        self.repo_root = repo_root.resolve()
        self.user_directory = user_directory.resolve()
        self.api_client = api_client
        self.runtime_configuration = runtime_configuration
        self.registry = registry
        self.identity_file = identity_file
        self.r2_cache = r2_cache
        self.image_builder = image_builder or VastWorkerImageBuilder(
            repo_root=self.repo_root,
            comfyui_root=getattr(settings, "comfyui_root", None),
            modal_gpu=settings.modal_gpu,
        )
        self.update_process_environment = update_process_environment
        self.huggingface_asset_registry = (
            HuggingFaceAssetRegistry.for_user_directory(self.user_directory)
        )
        self.huggingface_asset_discovery = HuggingFaceAssetDiscovery(
            registry=self.huggingface_asset_registry,
            user_directory=self.user_directory,
            comfyui_root=getattr(settings, "comfyui_root", None),
        )
        self.lease_manager = self._lease_manager_for(runtime_configuration)

    def _lease_manager_for(
        self,
        runtime_configuration: VastRuntimeConfiguration,
    ) -> VastLeaseManager:
        """Return a lease manager bound to one immutable worker image."""
        return VastLeaseManager(
            api_client=self.api_client,
            registry=self.registry,
            owner_id=self.settings.app_name,
            runtime_fingerprint=runtime_configuration.runtime_fingerprint,
            launch_spec_factory=runtime_configuration.launch_spec,
            startup_timeout_seconds=runtime_configuration.startup_timeout_seconds,
            worker_image=runtime_configuration.image,
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
            r2_cache=R2CacheClient.from_environment(source),
            image_builder=VastWorkerImageBuilder(
                repo_root=repo_root.resolve(),
                comfyui_root=settings.comfyui_root,
                modal_gpu=settings.modal_gpu,
                environment=(source if environment is not None else None),
            ),
            update_process_environment=environment is None,
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

        return await self._acquire_with_replacement(
            quote.profile,
            slot=slot,
            status_callback=status_callback,
            instance_status_callback=(
                emit_instance_status if status_callback is not None else None
            ),
        )

    async def _acquire_with_replacement(
        self,
        profile: VastResourceProfile,
        *,
        slot: int,
        status_callback: Callable[[str], None] | None,
        instance_status_callback: Callable[[VastInstance], None] | None,
        allow_image_rebuild: bool = True,
    ) -> VastLeaseRecord:
        """Initialize capacity and replace one disappeared or unusable contract."""
        excluded_offer_ids: set[int] = set()
        for replacement_attempt in range(
            VAST_INSTANCE_SETUP_REPLACEMENTS + 1
        ):
            lease = await self.lease_manager.ensure_lease(
                profile,
                slot=slot,
                status_callback=instance_status_callback,
                event_callback=status_callback,
                excluded_offer_ids=frozenset(excluded_offer_ids),
            )
            if status_callback is not None:
                status_callback("Initializing Vast.ai worker")
            try:
                initialized_lease = await asyncio.to_thread(
                    self._initialize_runtime,
                    lease,
                )
                if initialized_lease is not None:
                    lease = initialized_lease
            except VastRuntimeFingerprintDriftError as exc:
                cleanup_completed = await self._discard_failed_runtime_lease(
                    lease,
                    str(exc),
                )
                if status_callback is not None:
                    cleanup_status = (
                        "terminated" if cleanup_completed else "marked unusable"
                    )
                    status_callback(
                        "Vast worker source drift detected; stale instance "
                        f"{lease.instance_id} was {cleanup_status}"
                    )
                if not allow_image_rebuild:
                    raise RuntimeError(
                        self._manual_image_build_message(
                            "The automatically published worker image still reported "
                            f"fingerprint {exc.actual_fingerprint[:12]} instead of "
                            f"{exc.expected_fingerprint[:12]}."
                        )
                    ) from exc
                try:
                    image = self.image_builder.build_and_push(
                        exc.expected_fingerprint,
                        status_callback=status_callback,
                    )
                except VastWorkerImageBuildError:
                    if status_callback is not None:
                        status_callback("Automatic Vast worker image build failed")
                    raise
                self._adopt_runtime_image(image)
                if status_callback is not None:
                    status_callback(
                        "Vast worker image updated; requesting fresh capacity"
                    )
                return await self._acquire_with_replacement(
                    profile,
                    slot=slot,
                    status_callback=status_callback,
                    instance_status_callback=instance_status_callback,
                    allow_image_rebuild=False,
                )
            except VastInstanceNotFoundError as exc:
                self.registry.remove(lease.instance_id)
                excluded_offer_ids.add(lease.offer_id)
                if replacement_attempt >= VAST_INSTANCE_SETUP_REPLACEMENTS:
                    if status_callback is not None:
                        status_callback("Vast.ai worker initialization failed")
                    raise RuntimeError(
                        "Vast.ai worker setup failed after the replacement "
                        f"instance also disappeared. Last instance: "
                        f"{lease.instance_id}."
                    ) from exc
                logger.warning(
                    "Vast instance disappeared before worker initialization "
                    "instance_id=%d offer=%d; cold-starting a replacement.",
                    lease.instance_id,
                    lease.offer_id,
                )
                if status_callback is not None:
                    status_callback(
                        "Vast.ai instance disappeared; requesting a replacement"
                    )
                continue
            except (TimeoutError, VastSshError, ValueError) as exc:
                cleanup_completed = await self._discard_failed_runtime_lease(
                    lease,
                    str(exc),
                )
                excluded_offer_ids.add(lease.offer_id)
                if replacement_attempt >= VAST_INSTANCE_SETUP_REPLACEMENTS:
                    if status_callback is not None:
                        status_callback("Vast.ai worker initialization failed")
                    cleanup_summary = (
                        "was destroyed"
                        if cleanup_completed
                        else "could not be confirmed destroyed"
                    )
                    raise RuntimeError(
                        "Vast.ai worker setup failed after a replacement attempt. "
                        f"Last instance {lease.instance_id} {cleanup_summary}. "
                        f"Last error: {exc}"
                    ) from exc
                logger.warning(
                    "Vast instance failed worker initialization instance_id=%d "
                    "offer=%d cleanup_completed=%s; cold-starting a replacement: %s",
                    lease.instance_id,
                    lease.offer_id,
                    cleanup_completed,
                    exc,
                )
                if status_callback is not None:
                    cleanup_status = (
                        "terminated"
                        if cleanup_completed
                        else "marked unusable"
                    )
                    status_callback(
                        f"Vast.ai instance {lease.instance_id} failed worker setup "
                        f"and was {cleanup_status}; requesting a replacement"
                    )
                continue
            if status_callback is not None:
                status_callback("Vast.ai worker is ready")
            return lease
        raise RuntimeError("Vast replacement attempts ended without a lease.")

    def _adopt_runtime_image(self, image: str) -> None:
        """Use one freshly published image for this and later process-local runs."""
        configuration = replace(self.runtime_configuration, image=image)
        self.runtime_configuration = configuration
        self.lease_manager = self._lease_manager_for(configuration)
        if self.update_process_environment:
            os.environ[VAST_IMAGE_ENV] = image
        logger.info(
            "Adopted rebuilt Vast worker image fingerprint=%s.",
            configuration.runtime_fingerprint[:12],
        )

    def _manual_image_build_message(self, cause: str) -> str:
        """Return the manual recovery required after an unusable rebuilt image."""
        return (
            f"{cause} Run `{' '.join(VAST_IMAGE_BUILD_COMMAND)}` from "
            f"{self.repo_root}, set {VAST_IMAGE_ENV} to the printed digest, restart "
            "ComfyUI, and retry the workflow."
        )

    async def _discard_failed_runtime_lease(
        self,
        lease: VastLeaseRecord,
        error_message: str,
    ) -> bool:
        """Drain and destroy capacity that never produced a usable worker."""
        self.registry.update(
            lease.instance_id,
            lambda current: replace(
                current,
                draining=True,
                last_error=error_message,
            ),
        )
        try:
            destroyed = await self.lease_manager.destroy_owned_lease(lease.instance_id)
        except (RuntimeError, ValueError, VastApiError) as cleanup_error:
            logger.warning(
                "Unable to destroy unusable Vast worker instance_id=%d: %s",
                lease.instance_id,
                cleanup_error,
            )
            return False
        if not destroyed:
            logger.warning(
                "Unusable Vast worker was not destroyed because ownership could not "
                "be confirmed instance_id=%d.",
                lease.instance_id,
            )
            return False
        logger.info(
            "Destroyed unusable Vast worker instance_id=%d after initialization failure.",
            lease.instance_id,
        )
        return True

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
            huggingface_asset_registry=self.huggingface_asset_registry,
            huggingface_asset_discovery=self.huggingface_asset_discovery,
            r2_cache=self.r2_cache,
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
        """Return an SSH runner for the lease's selected endpoint."""
        return VastSshRunner(
            vast_connection_from_lease(
                ssh_host=lease.ssh_host,
                ssh_port=lease.ssh_port,
                user_directory=self.user_directory,
                identity_file=self.identity_file,
            )
        )

    def _proxy_runner(self, lease: VastLeaseRecord) -> VastSshRunner | None:
        """Return the distinct Vast proxy endpoint available after direct failure."""
        if (
            lease.ssh_proxy_host is None
            or lease.ssh_proxy_port is None
            or (
                lease.ssh_host == lease.ssh_proxy_host
                and lease.ssh_port == lease.ssh_proxy_port
            )
        ):
            return None
        return VastSshRunner(
            vast_connection_from_lease(
                ssh_host=lease.ssh_proxy_host,
                ssh_port=lease.ssh_proxy_port,
                user_directory=self.user_directory,
                identity_file=self.identity_file,
            )
        )

    def _promote_proxy_endpoint(self, lease: VastLeaseRecord) -> None:
        """Persist the proxy as primary after it proves reachable."""
        if lease.ssh_proxy_host is None or lease.ssh_proxy_port is None:
            raise ValueError("Vast lease does not expose a complete proxy endpoint.")
        self.registry.update(
            lease.instance_id,
            lambda current: replace(
                current,
                ssh_host=lease.ssh_proxy_host,
                ssh_port=lease.ssh_proxy_port,
            ),
        )
        logger.warning(
            "Selected Vast SSH proxy after direct endpoint failure instance_id=%d.",
            lease.instance_id,
        )

    def _current_lease(self, instance_id: int) -> VastLeaseRecord:
        """Return the latest persisted lease after endpoint selection."""
        lease = next(
            (
                candidate
                for candidate in self.registry.load().leases
                if candidate.instance_id == instance_id
            ),
            None,
        )
        if lease is None:
            raise KeyError(f"Vast lease {instance_id} disappeared during setup.")
        return lease

    def _initialize_runtime(self, lease: VastLeaseRecord) -> VastLeaseRecord:
        """Require the pinned worker and publish its initial idle fail-safe state."""
        logger.info(
            "Initializing Vast worker instance_id=%d environment=%s.",
            lease.instance_id,
            lease.environment_id,
        )
        runtime = VastRuntimeManager(
            runner=self._runner(lease),
            configuration=self.runtime_configuration,
            fallback_runner=self._proxy_runner(lease),
            fallback_selected=partial(self._promote_proxy_endpoint, lease),
            instance_validator=partial(self._validate_live_instance, lease),
        )
        try:
            runtime.ensure_worker()
            active_lease = self._current_lease(lease.instance_id)
            runtime.update_watchdog(active_lease)
        except (
            TimeoutError,
            VastInstanceNotFoundError,
            VastSshError,
            ValueError,
        ) as exc:
            logger.error(
                "Vast worker initialization failed instance_id=%d environment=%s: %s",
                lease.instance_id,
                lease.environment_id,
                exc,
            )
            raise
        logger.info(
            "Vast worker initialization completed instance_id=%d environment=%s.",
            lease.instance_id,
            lease.environment_id,
        )
        return active_lease

    def _validate_live_instance(self, lease: VastLeaseRecord) -> None:
        """Require the managed Vast contract to exist before an SSH probe."""
        asyncio.run(self.api_client.show_instance(lease.instance_id))

    def _existing_lease(self, profile: VastResourceProfile) -> VastLeaseRecord | None:
        """Return the oldest ready lease with an exact profile/runtime identity."""
        fingerprint = vast_profile_fingerprint(profile)
        matches = [
            lease
            for lease in self.registry.load().leases
            if lease.owner_id == self.settings.app_name
            and lease.profile_id == profile.profile_id
            and lease.profile_fingerprint == fingerprint
            and lease.worker_image in {None, self.runtime_configuration.image}
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
