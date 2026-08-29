"""Inspect public OCI worker images without downloading their filesystem layers."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

RUNTIME_FINGERPRINT_LABEL = "comfy.remote.runtime-fingerprint"
_MANIFEST_ACCEPT = ", ".join(
    (
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    )
)
_BEARER_PARAMETER_PATTERN = re.compile(r'(\w+)="([^"]*)"')


class VastImageRegistryError(RuntimeError):
    """Report that published worker image metadata could not be inspected."""


class VastImageNotFoundError(VastImageRegistryError):
    """Report that a configured worker image is absent from its registry."""


@dataclass(frozen=True)
class OciImageReference:
    """Identify one registry repository and tag or digest reference."""

    registry: str
    repository: str
    reference: str


def parse_oci_image_reference(image: str) -> OciImageReference:
    """Parse a Docker-compatible image reference for registry HTTP requests."""
    normalized = image.strip()
    if not normalized:
        raise ValueError("Worker image reference must not be empty.")
    name, separator, digest = normalized.partition("@")
    if separator:
        reference = digest
    else:
        final_slash = name.rfind("/")
        final_colon = name.rfind(":")
        if final_colon > final_slash:
            reference = name[final_colon + 1 :]
            name = name[:final_colon]
        else:
            reference = "latest"
    components = name.split("/")
    first = components[0]
    explicit_registry = "." in first or ":" in first or first == "localhost"
    if explicit_registry:
        if len(components) < 2:
            raise ValueError(f"Worker image {image!r} is missing a repository name.")
        registry = first
        repository = "/".join(components[1:])
    else:
        registry = "registry-1.docker.io"
        repository = name if "/" in name else f"library/{name}"
    if registry in {"docker.io", "index.docker.io"}:
        registry = "registry-1.docker.io"
    if not repository or not reference:
        raise ValueError(f"Worker image reference {image!r} is malformed.")
    return OciImageReference(
        registry=registry,
        repository=repository,
        reference=reference,
    )


def published_runtime_fingerprint(
    image: str,
    *,
    timeout_seconds: float = 30.0,
) -> str | None:
    """Return the runtime label from a public image's linux/amd64 config."""
    reference = parse_oci_image_reference(image)
    token: str | None = None
    manifest, token = _registry_json(
        reference,
        resource=f"manifests/{quote(reference.reference, safe=':')}",
        accept=_MANIFEST_ACCEPT,
        token=token,
        timeout_seconds=timeout_seconds,
    )
    selected_digest = _linux_amd64_digest(manifest)
    if selected_digest is not None:
        manifest, token = _registry_json(
            reference,
            resource=f"manifests/{quote(selected_digest, safe=':')}",
            accept=_MANIFEST_ACCEPT,
            token=token,
            timeout_seconds=timeout_seconds,
        )
    config_digest = _config_digest(manifest)
    config, _ = _registry_json(
        reference,
        resource=f"blobs/{quote(config_digest, safe=':')}",
        accept="application/octet-stream",
        token=token,
        timeout_seconds=timeout_seconds,
    )
    labels = _image_labels(config)
    fingerprint = labels.get(RUNTIME_FINGERPRINT_LABEL)
    logger.info(
        "Inspected published Vast worker image image=%s fingerprint=%s.",
        image,
        fingerprint[:12] if isinstance(fingerprint, str) else "missing",
    )
    return fingerprint if isinstance(fingerprint, str) else None


def _registry_json(
    reference: OciImageReference,
    *,
    resource: str,
    accept: str,
    token: str | None,
    timeout_seconds: float,
) -> tuple[Mapping[str, Any], str | None]:
    """Fetch one JSON registry resource, obtaining an anonymous token if needed."""
    url = f"https://{reference.registry}/v2/{reference.repository}/{resource}"
    try:
        return _request_json(url, accept, token, timeout_seconds), token
    except HTTPError as error:
        if error.code != 401:
            if error.code == 404:
                raise VastImageNotFoundError(
                    f"Worker image was not found in {reference.registry}."
                ) from error
            raise VastImageRegistryError(
                f"Registry returned HTTP {error.code} for {reference.registry}."
            ) from error
        challenge = (
            error.headers.get("WWW-Authenticate", "")
            if error.headers is not None
            else ""
        )
    parameters = _bearer_parameters(challenge)
    realm = parameters.get("realm")
    if realm is None:
        raise VastImageRegistryError(
            f"Registry {reference.registry} requires unsupported authentication."
        )
    query = {
        key: value
        for key, value in (
            ("service", parameters.get("service")),
            (
                "scope",
                parameters.get("scope") or f"repository:{reference.repository}:pull",
            ),
        )
        if value is not None
    }
    query_separator = "&" if "?" in realm else "?"
    token_url = f"{realm}{query_separator}{urlencode(query)}" if query else realm
    try:
        token_payload = _request_json(
            token_url,
            "application/json",
            None,
            timeout_seconds,
        )
    except HTTPError as error:
        raise VastImageRegistryError(
            f"Registry token service returned HTTP {error.code}."
        ) from error
    resolved_token = token_payload.get("token") or token_payload.get("access_token")
    if not isinstance(resolved_token, str) or not resolved_token:
        raise VastImageRegistryError(
            f"Registry {reference.registry} did not return an anonymous pull token."
        )
    try:
        document = _request_json(url, accept, resolved_token, timeout_seconds)
    except HTTPError as error:
        if error.code == 404:
            raise VastImageNotFoundError(
                f"Worker image was not found in {reference.registry}."
            ) from error
        raise VastImageRegistryError(
            f"Registry rejected its pull token with HTTP {error.code}."
        ) from error
    return document, resolved_token


def _request_json(
    url: str,
    accept: str,
    token: str | None,
    timeout_seconds: float,
) -> Mapping[str, Any]:
    """Perform one bounded registry request and decode its JSON object."""
    headers = {"Accept": accept, "User-Agent": "ComfyUI-Modal/worker-preflight"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = response.read()
    except HTTPError:
        raise
    except (OSError, URLError) as error:
        raise VastImageRegistryError(
            f"Could not reach worker image registry: {error}"
        ) from error
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VastImageRegistryError(
            "Worker image registry returned invalid JSON."
        ) from error
    if not isinstance(document, Mapping):
        raise VastImageRegistryError(
            "Worker image registry returned a non-object response."
        )
    return document


def _bearer_parameters(challenge: str) -> dict[str, str]:
    """Return quoted parameters from one standard Bearer challenge."""
    if not challenge.casefold().startswith("bearer "):
        return {}
    return dict(_BEARER_PARAMETER_PATTERN.findall(challenge))


def _linux_amd64_digest(manifest: Mapping[str, Any]) -> str | None:
    """Select the linux/amd64 manifest digest from an OCI image index."""
    entries = manifest.get("manifests")
    if entries is None:
        return None
    if not isinstance(entries, list):
        raise VastImageRegistryError(
            "Worker image index has invalid manifests metadata."
        )
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        platform = entry.get("platform")
        if not isinstance(platform, Mapping):
            continue
        if platform.get("os") == "linux" and platform.get("architecture") == "amd64":
            digest = entry.get("digest")
            if isinstance(digest, str) and digest.startswith("sha256:"):
                return digest
    raise VastImageRegistryError("Worker image has no linux/amd64 manifest.")


def _config_digest(manifest: Mapping[str, Any]) -> str:
    """Return the image config digest from one platform manifest."""
    config = manifest.get("config")
    digest = config.get("digest") if isinstance(config, Mapping) else None
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        raise VastImageRegistryError(
            "Worker image manifest has no valid config digest."
        )
    return digest


def _image_labels(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return image config labels or an empty mapping when none are present."""
    configuration = config.get("config")
    labels = configuration.get("Labels") if isinstance(configuration, Mapping) else None
    return labels if isinstance(labels, Mapping) else {}


__all__ = [
    "RUNTIME_FINGERPRINT_LABEL",
    "VastImageNotFoundError",
    "VastImageRegistryError",
    "parse_oci_image_reference",
    "published_runtime_fingerprint",
]
