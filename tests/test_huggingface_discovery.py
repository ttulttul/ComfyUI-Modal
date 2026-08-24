"""Tests for zero-touch Hugging Face model provenance discovery."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any


class FakeHubApi:
    """Return deterministic file metadata for candidate verification tests."""

    def __init__(self, responses: dict[tuple[str, str], object]) -> None:
        """Store responses indexed by repository and requested revision."""
        self.responses = responses
        self.calls: list[tuple[str, str]] = []

    def model_info(self, repo_id: str, **kwargs: object) -> object:
        """Return one configured repository response and record the request."""
        revision = str(kwargs["revision"])
        assert kwargs["files_metadata"] is True
        self.calls.append((repo_id, revision))
        return self.responses[(repo_id, revision)]


def _hub_response(
    *,
    filename: str,
    sha256: str,
    size_bytes: int,
    commit: str,
) -> dict[str, object]:
    """Build one Hub model-info payload containing exact file metadata."""
    return {
        "sha": commit,
        "siblings": [
            {
                "rfilename": filename,
                "size": size_bytes,
                "lfs": {"sha256": sha256},
            }
        ],
    }


def _discovery(
    module: Any,
    assets_module: Any,
    *,
    tmp_path: Path,
    user_directory: Path,
    api: FakeHubApi,
    where_from_urls: tuple[str, ...] = (),
) -> Any:
    """Create an isolated discovery service with deterministic local evidence."""
    registry = assets_module.HuggingFaceAssetRegistry(
        tmp_path / "registry" / "huggingface-assets.json"
    )
    return module.HuggingFaceAssetDiscovery(
        registry=registry,
        user_directory=user_directory,
        comfyui_root=tmp_path / "comfyui",
        api=api,
        where_from_reader=lambda path: where_from_urls,
    )


def test_parses_official_resolve_and_cache_urls(
    huggingface_discovery_module: Any,
) -> None:
    """Normal and CDN-backed Hub URLs should yield the same immutable hint shape."""
    resolve_hint = huggingface_discovery_module.huggingface_hint_from_url(
        "https://huggingface.co/owner/model/resolve/main/weights/model.safetensors?download=true",
        evidence="browser",
    )
    cache_hint = huggingface_discovery_module.huggingface_hint_from_url(
        "https://huggingface.co/api/resolve-cache/models/owner/model/"
        f"{'a' * 40}/weights/model.safetensors",
        evidence="browser",
    )

    assert resolve_hint.repo_id == "owner/model"
    assert resolve_hint.revision == "main"
    assert resolve_hint.filename == "weights/model.safetensors"
    assert cache_hint.repo_id == "owner/model"
    assert cache_hint.revision == "a" * 40
    assert cache_hint.filename == "weights/model.safetensors"


def test_discovers_manager_installed_model_and_persists_verified_mapping(
    huggingface_assets_module: Any,
    huggingface_discovery_module: Any,
    tmp_path: Path,
) -> None:
    """Manager's catalog should make ordinary installs zero-touch after hash validation."""
    user_directory = tmp_path / "user"
    catalog = user_directory / "__manager" / "cache" / "123_model-list.json"
    catalog.parent.mkdir(parents=True)
    catalog.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "filename": "manager-model.safetensors",
                        "save_path": "vae",
                        "url": (
                            "https://huggingface.co/wrong/model/resolve/main/"
                            "manager-model.safetensors"
                        ),
                    },
                    {
                        "filename": "manager-model.safetensors",
                        "save_path": "checkpoints",
                        "url": (
                            "https://huggingface.co/owner/model/resolve/main/"
                            "weights/manager-model.safetensors"
                        ),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    asset_path = tmp_path / "models" / "checkpoints" / "manager-model.safetensors"
    asset_path.parent.mkdir(parents=True)
    asset_path.write_bytes(b"manager-model")
    sha256 = huggingface_assets_module.sha256_file(asset_path)
    commit = "b" * 40
    api = FakeHubApi(
        {
            ("owner/model", "main"): _hub_response(
                filename="weights/manager-model.safetensors",
                sha256=sha256,
                size_bytes=asset_path.stat().st_size,
                commit=commit,
            )
        }
    )
    discovery = _discovery(
        huggingface_discovery_module,
        huggingface_assets_module,
        tmp_path=tmp_path,
        user_directory=user_directory,
        api=api,
    )

    source = discovery.discover(asset_path, sha256=sha256)

    assert source.repo_id == "owner/model"
    assert source.revision == commit
    assert source.filename == "weights/manager-model.safetensors"
    assert discovery.registry.get(sha256) == source
    assert discovery.discover(asset_path, sha256=sha256) == source
    assert api.calls == [("owner/model", "main")]


def test_discovers_model_inside_huggingface_git_checkout(
    huggingface_assets_module: Any,
    huggingface_discovery_module: Any,
    tmp_path: Path,
) -> None:
    """An ordinary Hub Git clone should supply its repository, commit, and file path."""
    commit = "8" * 40
    checkout = tmp_path / "checkout"
    git_directory = checkout / ".git"
    git_directory.mkdir(parents=True)
    (git_directory / "config").write_text(
        '[remote "origin"]\n\turl = https://huggingface.co/git/model.git\n',
        encoding="utf-8",
    )
    (git_directory / "HEAD").write_text(f"{commit}\n", encoding="utf-8")
    asset_path = checkout / "weights" / "git.safetensors"
    asset_path.parent.mkdir()
    asset_path.write_bytes(b"git-model")
    sha256 = huggingface_assets_module.sha256_file(asset_path)
    api = FakeHubApi(
        {
            ("git/model", commit): _hub_response(
                filename="weights/git.safetensors",
                sha256=sha256,
                size_bytes=asset_path.stat().st_size,
                commit=commit,
            )
        }
    )
    discovery = _discovery(
        huggingface_discovery_module,
        huggingface_assets_module,
        tmp_path=tmp_path,
        user_directory=tmp_path / "user",
        api=api,
    )

    source = discovery.discover(asset_path, sha256=sha256)

    assert source.repo_id == "git/model"
    assert source.revision == commit
    assert source.filename == "weights/git.safetensors"


def test_discovers_browser_download_source_metadata(
    huggingface_assets_module: Any,
    huggingface_discovery_module: Any,
    tmp_path: Path,
) -> None:
    """Browser WhereFrom metadata should remove the need for manual registration."""
    asset_path = tmp_path / "browser-model.safetensors"
    asset_path.write_bytes(b"browser-model")
    sha256 = huggingface_assets_module.sha256_file(asset_path)
    api = FakeHubApi(
        {
            ("browser/source", "main"): _hub_response(
                filename="browser-model.safetensors",
                sha256=sha256,
                size_bytes=asset_path.stat().st_size,
                commit="c" * 40,
            )
        }
    )
    discovery = _discovery(
        huggingface_discovery_module,
        huggingface_assets_module,
        tmp_path=tmp_path,
        user_directory=tmp_path / "user",
        api=api,
        where_from_urls=(
            "https://huggingface.co/browser/source/resolve/main/browser-model.safetensors",
        ),
    )

    source = discovery.discover(asset_path, sha256=sha256)

    assert source.repo_id == "browser/source"
    assert source.revision == "c" * 40


def test_discovers_standard_huggingface_cache_path(
    huggingface_assets_module: Any,
    huggingface_discovery_module: Any,
    tmp_path: Path,
) -> None:
    """A file or symlink into the standard Hub cache should carry its own source."""
    commit = "d" * 40
    asset_path = (
        tmp_path
        / "hub"
        / "models--cache--model"
        / "snapshots"
        / commit
        / "weights"
        / "cached.safetensors"
    )
    asset_path.parent.mkdir(parents=True)
    asset_path.write_bytes(b"cached-model")
    sha256 = huggingface_assets_module.sha256_file(asset_path)
    api = FakeHubApi(
        {
            ("cache/model", commit): _hub_response(
                filename="weights/cached.safetensors",
                sha256=sha256,
                size_bytes=asset_path.stat().st_size,
                commit=commit,
            )
        }
    )
    discovery = _discovery(
        huggingface_discovery_module,
        huggingface_assets_module,
        tmp_path=tmp_path,
        user_directory=tmp_path / "user",
        api=api,
    )

    source = discovery.discover(asset_path, sha256=sha256)

    assert source.repo_id == "cache/model"
    assert source.filename == "weights/cached.safetensors"


def test_discovers_hub_url_embedded_in_safetensors_metadata(
    huggingface_assets_module: Any,
    huggingface_discovery_module: Any,
    tmp_path: Path,
) -> None:
    """Model metadata containing a Hub file URL should be useful but still verified."""
    header = json.dumps(
        {
            "__metadata__": {
                "modelspec.source": (
                    "https://huggingface.co/embedded/model/resolve/main/"
                    "weights/embedded.safetensors"
                )
            }
        }
    ).encode("utf-8")
    asset_path = tmp_path / "embedded.safetensors"
    asset_path.write_bytes(struct.pack("<Q", len(header)) + header + b"model-bytes")
    sha256 = huggingface_assets_module.sha256_file(asset_path)
    api = FakeHubApi(
        {
            ("embedded/model", "main"): _hub_response(
                filename="weights/embedded.safetensors",
                sha256=sha256,
                size_bytes=asset_path.stat().st_size,
                commit="e" * 40,
            )
        }
    )
    discovery = _discovery(
        huggingface_discovery_module,
        huggingface_assets_module,
        tmp_path=tmp_path,
        user_directory=tmp_path / "user",
        api=api,
    )

    source = discovery.discover(asset_path, sha256=sha256)

    assert source.repo_id == "embedded/model"
    assert source.filename == "weights/embedded.safetensors"


def test_rejects_unverified_hint_and_negative_caches_result(
    huggingface_assets_module: Any,
    huggingface_discovery_module: Any,
    tmp_path: Path,
) -> None:
    """Install hints must never bypass exact content verification or repeat per prompt."""
    asset_path = tmp_path / "mismatch.safetensors"
    asset_path.write_bytes(b"local-content")
    sha256 = huggingface_assets_module.sha256_file(asset_path)
    api = FakeHubApi(
        {
            ("wrong/model", "main"): _hub_response(
                filename="mismatch.safetensors",
                sha256="f" * 64,
                size_bytes=asset_path.stat().st_size,
                commit="f" * 40,
            )
        }
    )
    discovery = _discovery(
        huggingface_discovery_module,
        huggingface_assets_module,
        tmp_path=tmp_path,
        user_directory=tmp_path / "user",
        api=api,
        where_from_urls=(
            "https://huggingface.co/wrong/model/resolve/main/mismatch.safetensors",
        ),
    )

    assert discovery.discover(asset_path, sha256=sha256) is None
    assert discovery.discover(asset_path, sha256=sha256) is None
    assert discovery.registry.get(sha256) is None
    assert api.calls == [("wrong/model", "main")]


def test_retries_unresolved_asset_after_manager_catalog_changes(
    huggingface_assets_module: Any,
    huggingface_discovery_module: Any,
    tmp_path: Path,
) -> None:
    """Installing a model while ComfyUI is open should invalidate negative evidence."""
    user_directory = tmp_path / "user"
    asset_path = tmp_path / "models" / "late.safetensors"
    asset_path.parent.mkdir()
    asset_path.write_bytes(b"late-manager-install")
    sha256 = huggingface_assets_module.sha256_file(asset_path)
    api = FakeHubApi(
        {
            ("late/model", "main"): _hub_response(
                filename="late.safetensors",
                sha256=sha256,
                size_bytes=asset_path.stat().st_size,
                commit="7" * 40,
            )
        }
    )
    discovery = _discovery(
        huggingface_discovery_module,
        huggingface_assets_module,
        tmp_path=tmp_path,
        user_directory=user_directory,
        api=api,
    )

    assert discovery.discover(asset_path, sha256=sha256) is None

    catalog = user_directory / "__manager" / "cache" / "new_model-list.json"
    catalog.parent.mkdir(parents=True)
    catalog.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "filename": "late.safetensors",
                        "url": (
                            "https://huggingface.co/late/model/resolve/main/"
                            "late.safetensors"
                        ),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    source = discovery.discover(asset_path, sha256=sha256)

    assert source.repo_id == "late/model"
    assert source.revision == "7" * 40
    assert api.calls == [("late/model", "main")]
