"""Tests for lightweight OCI worker image fingerprint inspection."""

from __future__ import annotations

import importlib
import json
from email.message import Message
from types import SimpleNamespace
from typing import Any
from urllib.error import HTTPError


def _module() -> Any:
    """Load the registry inspector through the extension package fixture namespace."""
    return importlib.import_module("comfyui_modal_sync_under_test.vast_image_registry")


class FakeResponse:
    """Provide the small context-manager surface used by urllib responses."""

    def __init__(self, payload: dict[str, Any]) -> None:
        """Encode one JSON response payload."""
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "FakeResponse":
        """Return this response from a context manager."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Close the synthetic response without work."""
        del args

    def read(self) -> bytes:
        """Return the encoded payload."""
        return self.payload


def test_parse_registry_reference_supports_ports_tags_and_digests(
    extension_package: Any,
) -> None:
    """Registry parsing must not confuse a registry port with an image tag."""
    del extension_package
    module = _module()

    tagged = module.parse_oci_image_reference("registry.test:5443/team/worker:v1")
    digest = module.parse_oci_image_reference(
        "ghcr.io/team/worker@sha256:" + "a" * 64
    )

    assert tagged == module.OciImageReference(
        registry="registry.test:5443",
        repository="team/worker",
        reference="v1",
    )
    assert digest.registry == "ghcr.io"
    assert digest.repository == "team/worker"
    assert digest.reference == "sha256:" + "a" * 64


def test_published_fingerprint_reads_linux_amd64_config_label(
    extension_package: Any,
    monkeypatch: Any,
) -> None:
    """Preflight should fetch only manifests and config metadata after auth."""
    del extension_package
    module = _module()
    requested_urls: list[str] = []
    expected_fingerprint = "f" * 64
    platform_digest = "sha256:" + "b" * 64
    config_digest = "sha256:" + "c" * 64

    def fake_urlopen(request: Any, *, timeout: float) -> FakeResponse:
        """Serve an anonymous-token OCI index, manifest, and config sequence."""
        assert timeout == 30.0
        url = request.full_url
        requested_urls.append(url)
        authorization = request.get_header("Authorization")
        if "/manifests/v1" in url and authorization is None:
            headers = Message()
            headers["WWW-Authenticate"] = (
                'Bearer realm="https://auth.test/token",service="ghcr.io",'
                'scope="repository:team/worker:pull"'
            )
            raise HTTPError(url, 401, "Unauthorized", headers, None)
        if url.startswith("https://auth.test/token?"):
            return FakeResponse({"token": "pull-token"})
        assert authorization == "Bearer pull-token"
        if "/manifests/v1" in url:
            return FakeResponse(
                {
                    "manifests": [
                        {
                            "digest": "sha256:" + "d" * 64,
                            "platform": {"os": "linux", "architecture": "arm64"},
                        },
                        {
                            "digest": platform_digest,
                            "platform": {"os": "linux", "architecture": "amd64"},
                        },
                    ]
                }
            )
        if f"/manifests/{platform_digest}" in url:
            return FakeResponse({"config": {"digest": config_digest}})
        if f"/blobs/{config_digest}" in url:
            return FakeResponse(
                {
                    "config": {
                        "Labels": {
                            module.RUNTIME_FINGERPRINT_LABEL: expected_fingerprint
                        }
                    }
                }
            )
        raise AssertionError(f"Unexpected registry request {url}")

    monkeypatch.setattr(module, "urlopen", fake_urlopen)

    metadata = module.published_image_metadata("ghcr.io/team/worker:v1")

    assert metadata.runtime_fingerprint == expected_fingerprint
    assert metadata.immutable_image == f"ghcr.io/team/worker@{platform_digest}"
    assert len(requested_urls) == 5


def test_missing_runtime_label_is_reported_as_none(
    extension_package: Any,
    monkeypatch: Any,
) -> None:
    """An otherwise valid legacy image should trigger an automatic rebuild."""
    del extension_package
    module = _module()
    responses = iter(
        (
            FakeResponse({"config": {"digest": "sha256:" + "a" * 64}}),
            FakeResponse({"config": {"Labels": {}}}),
        )
    )
    monkeypatch.setattr(module, "urlopen", lambda *_args, **_kwargs: next(responses))

    assert module.published_runtime_fingerprint("registry.test/team/worker:v1") is None
