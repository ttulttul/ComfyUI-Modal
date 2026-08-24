"""Tests for Hugging Face provenance registration and remote materialization."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pytest


def _source(module: Any, asset_path: Path) -> Any:
    """Return one valid immutable source matching the supplied test asset."""
    return module.HuggingFaceAssetSource(
        repo_id="owner/model",
        revision="a" * 40,
        filename="weights/model.safetensors",
        sha256=module.sha256_file(asset_path),
        size_bytes=asset_path.stat().st_size,
    )


def test_registry_round_trips_validated_sources_atomically(
    huggingface_assets_module: Any,
    tmp_path: Path,
) -> None:
    """The persistent registry should index immutable sources by content digest."""
    asset_path = tmp_path / "model.safetensors"
    asset_path.write_bytes(b"registered-model")
    source = _source(huggingface_assets_module, asset_path)
    registry = huggingface_assets_module.HuggingFaceAssetRegistry(
        tmp_path / "user" / "comfyui-modal" / "huggingface-assets.json"
    )

    assert registry.get(source.sha256) is None
    registry.upsert(source)

    assert registry.get(source.sha256) == source
    payload = json.loads(registry.config_path.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert payload["assets"][source.sha256]["revision"] == "a" * 40
    assert not tuple(registry.config_path.parent.glob("*.tmp"))


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "message"),
    [
        ("repo_id", "missing-owner", "owner/model"),
        ("revision", "main", "exact 40-character"),
        ("filename", "../model.safetensors", "repository-relative"),
        ("sha256", "not-a-digest", "64 lowercase hex"),
        ("size_bytes", 0, "positive integer"),
    ],
)
def test_source_rejects_mutable_or_unsafe_identity(
    huggingface_assets_module: Any,
    field_name: str,
    invalid_value: object,
    message: str,
) -> None:
    """Provenance must not admit mutable revisions or unsafe paths."""
    payload: dict[str, object] = {
        "repo_id": "owner/model",
        "revision": "a" * 40,
        "filename": "model.safetensors",
        "sha256": "b" * 64,
        "size_bytes": 12,
    }
    payload[field_name] = invalid_value

    with pytest.raises(ValueError, match=message):
        huggingface_assets_module.HuggingFaceAssetSource.from_dict(payload)


def test_registration_resolves_commit_and_requires_matching_hub_metadata(
    huggingface_assets_module: Any,
    tmp_path: Path,
) -> None:
    """The registration helper should bind local bytes to exact Hub metadata."""
    registration_module = importlib.import_module("scripts.register_huggingface_asset")
    asset_path = tmp_path / "model.safetensors"
    asset_path.write_bytes(b"hub-model")
    sha256 = huggingface_assets_module.sha256_file(asset_path)

    class FakeApi:
        """Return one exact repository file response."""

        def model_info(self, repo_id: str, **kwargs: object) -> object:
            """Return metadata containing the expected LFS digest and size."""
            assert repo_id == "owner/model"
            assert kwargs == {"revision": "main", "files_metadata": True}
            return {
                "sha": "c" * 40,
                "siblings": [
                    {
                        "rfilename": "weights/model.safetensors",
                        "size": asset_path.stat().st_size,
                        "lfs": {"sha256": sha256},
                    }
                ],
            }

    source = registration_module.resolve_huggingface_asset_source(
        asset_path,
        repo_id="owner/model",
        filename="weights/model.safetensors",
        revision="main",
        api=FakeApi(),
    )

    assert source.revision == "c" * 40
    assert source.sha256 == sha256
    assert source.size_bytes == len(b"hub-model")


def test_registration_rejects_a_different_hub_object(
    tmp_path: Path,
) -> None:
    """A filename match alone must never authorize remote replacement bytes."""
    registration_module = importlib.import_module("scripts.register_huggingface_asset")
    asset_path = tmp_path / "model.safetensors"
    asset_path.write_bytes(b"local-model")

    class FakeApi:
        """Advertise a same-named but content-different file."""

        def model_info(self, repo_id: str, **kwargs: object) -> object:
            """Return deliberately mismatching metadata."""
            del repo_id, kwargs
            return {
                "sha": "d" * 40,
                "siblings": [
                    {
                        "rfilename": "model.safetensors",
                        "size": asset_path.stat().st_size,
                        "lfs": {"sha256": "e" * 64},
                    }
                ],
            }

    with pytest.raises(ValueError, match="does not match Hugging Face metadata"):
        registration_module.resolve_huggingface_asset_source(
            asset_path,
            repo_id="owner/model",
            filename="model.safetensors",
            revision="main",
            api=FakeApi(),
        )


def test_remote_materializer_downloads_verifies_and_atomically_publishes(
    huggingface_assets_module: Any,
    huggingface_materializer_module: Any,
    tmp_path: Path,
) -> None:
    """A correct remote download should become the content-addressed worker asset."""
    hub_file = tmp_path / "hub-cache" / "model.safetensors"
    hub_file.parent.mkdir()
    hub_file.write_bytes(b"remote-hub-model")
    source = _source(huggingface_assets_module, hub_file)
    storage_root = tmp_path / "storage"
    observed_options: dict[str, object] = {}

    def fake_download(**options: object) -> str:
        """Return the prepared Hub cache file and retain immutable request options."""
        observed_options.update(options)
        return str(hub_file)

    request = huggingface_materializer_module.HuggingFaceMaterializationRequest(
        source=source,
        storage_root=storage_root,
        remote_path=f"assets/{source.sha256}_model.safetensors",
        token="secret-token",
    )
    result = huggingface_materializer_module.materialize_huggingface_asset(
        request,
        download_file=fake_download,
    )
    target = storage_root / request.remote_path

    assert result.created is True
    assert target.read_bytes() == b"remote-hub-model"
    assert observed_options["repo_id"] == source.repo_id
    assert observed_options["revision"] == source.revision
    assert observed_options["filename"] == source.filename
    assert observed_options["token"] == "secret-token"


def test_remote_materializer_rejects_wrong_download_before_publication(
    huggingface_assets_module: Any,
    huggingface_materializer_module: Any,
    tmp_path: Path,
) -> None:
    """Downloaded bytes must match local provenance before the target appears."""
    expected_file = tmp_path / "expected.safetensors"
    expected_file.write_bytes(b"expected")
    wrong_file = tmp_path / "wrong.safetensors"
    wrong_file.write_bytes(b"wrong---")
    source = _source(huggingface_assets_module, expected_file)
    storage_root = tmp_path / "storage"
    request = huggingface_materializer_module.HuggingFaceMaterializationRequest(
        source=source,
        storage_root=storage_root,
        remote_path="assets/model.safetensors",
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        huggingface_materializer_module.materialize_huggingface_asset(
            request,
            download_file=lambda **kwargs: str(wrong_file),
        )

    assert not (storage_root / request.remote_path).exists()
