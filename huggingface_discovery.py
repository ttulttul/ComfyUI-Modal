"""Zero-touch Hugging Face provenance discovery for local ComfyUI model files."""

from __future__ import annotations

import configparser
import json
import logging
import plistlib
import re
import struct
import subprocess
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlparse

import httpx
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

if __package__:
    from .huggingface_assets import (
        HuggingFaceAssetRegistry,
        HuggingFaceAssetSource,
    )
else:  # pragma: no cover - direct debugging imports.
    from huggingface_assets import (
        HuggingFaceAssetRegistry,
        HuggingFaceAssetSource,
    )

logger = logging.getLogger(__name__)

_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SAFE_REPO_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_HUGGINGFACE_URL_PATTERN = re.compile(r"https://huggingface\.co/[^\s\"'<>]+")
_MAX_SAFETENSORS_HEADER_BYTES = 16 * 1024 * 1024
_MAX_XATTR_BYTES = 1024 * 1024
_MAX_MANAGER_FILENAME_FALLBACKS = 8
WhereFromReader = Callable[[Path], Sequence[str]]


@dataclass(frozen=True)
class HuggingFaceAssetHint:
    """Describe a candidate Hub location learned from trustworthy local metadata."""

    repo_id: str
    revision: str
    filename: str
    evidence: str

    def __post_init__(self) -> None:
        """Reject malformed candidates before making a Hub API request."""
        repo_parts = self.repo_id.split("/")
        if len(repo_parts) != 2 or not all(
            _SAFE_REPO_COMPONENT.fullmatch(part) for part in repo_parts
        ):
            raise ValueError("Hugging Face hint repo_id is invalid.")
        revision = self.revision.strip()
        if (
            not revision
            or len(revision) > 200
            or any(character in revision for character in "\x00\n\r")
        ):
            raise ValueError("Hugging Face hint revision is invalid.")
        path = PurePosixPath(self.filename)
        if (
            not self.filename
            or path.is_absolute()
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("Hugging Face hint filename is unsafe.")


@dataclass(frozen=True)
class LocalHuggingFaceDownloadMetadata:
    """Hold immutable metadata written by a Hugging Face local-dir download."""

    commit_hash: str
    etag: str
    filename: str


@dataclass(frozen=True)
class _ManagerHuggingFaceHint:
    """Associate one Manager catalog hint with its expected local model folder."""

    hint: HuggingFaceAssetHint
    save_path: str

    def matches(self, local_path: Path) -> bool:
        """Return whether the local path ends in Manager's declared save location."""
        normalized_save_path = self.save_path.strip().replace("\\", "/")
        save_parts = tuple(
            part
            for part in PurePosixPath(normalized_save_path).parts
            if part not in {"", ".", "default"}
        )
        if not save_parts:
            return False
        expected_parts = (*save_parts, local_path.name)
        return local_path.absolute().parts[-len(expected_parts) :] == expected_parts


def huggingface_hint_from_url(
    url: str,
    *,
    evidence: str,
) -> HuggingFaceAssetHint | None:
    """Parse one official Hugging Face model file URL into a candidate hint."""
    try:
        parsed = urlparse(url.strip())
    except ValueError:
        return None
    if parsed.scheme != "https" or parsed.hostname not in {
        "huggingface.co",
        "www.huggingface.co",
    }:
        return None
    parts = [unquote(part) for part in parsed.path.split("/") if part]
    if len(parts) >= 7 and parts[:3] == ["api", "resolve-cache", "models"]:
        owner, repository, revision = parts[3:6]
        filename_parts = parts[6:]
    elif len(parts) >= 5 and parts[2] in {"blob", "resolve"}:
        owner, repository = parts[:2]
        revision = parts[3]
        filename_parts = parts[4:]
    else:
        return None
    try:
        return HuggingFaceAssetHint(
            repo_id=f"{owner}/{repository.removesuffix('.git')}",
            revision=revision,
            filename=PurePosixPath(*filename_parts).as_posix(),
            evidence=evidence,
        )
    except ValueError:
        return None


def _huggingface_repo_id_from_url(url: str) -> str | None:
    """Return an owner/repository pair from one Hugging Face Git remote URL."""
    normalized = url.strip()
    if normalized.startswith("git@hf.co:"):
        repository_path = normalized.removeprefix("git@hf.co:")
    else:
        try:
            parsed = urlparse(normalized)
        except ValueError:
            return None
        if parsed.scheme != "https" or parsed.hostname not in {
            "huggingface.co",
            "www.huggingface.co",
        }:
            return None
        repository_path = parsed.path.lstrip("/")
    parts = repository_path.removesuffix(".git").split("/")
    if len(parts) != 2 or not all(_SAFE_REPO_COMPONENT.fullmatch(part) for part in parts):
        return None
    return "/".join(parts)


def _metadata_field(value: object, name: str) -> object | None:
    """Read one field from a Hugging Face object or test mapping."""
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _sibling_sha256(sibling: object) -> str | None:
    """Return the raw LFS/Xet-compatible content digest for one Hub file."""
    lfs = _metadata_field(sibling, "lfs")
    if lfs is None:
        return None
    value = _metadata_field(lfs, "sha256") or _metadata_field(lfs, "oid")
    normalized = str(value or "").strip().lower()
    return normalized if _SHA256_PATTERN.fullmatch(normalized) else None


def resolve_huggingface_asset_hint(
    hint: HuggingFaceAssetHint,
    *,
    sha256: str,
    size_bytes: int,
    api: Any,
) -> HuggingFaceAssetSource | None:
    """Resolve a candidate and return it only when Hub metadata matches local bytes."""
    model_info = api.model_info(
        hint.repo_id,
        revision=hint.revision,
        files_metadata=True,
    )
    exact_revision = str(_metadata_field(model_info, "sha") or "").strip().lower()
    if not _COMMIT_PATTERN.fullmatch(exact_revision):
        raise ValueError(
            f"Hugging Face did not resolve {hint.repo_id!r} to an exact commit."
        )
    siblings = _metadata_field(model_info, "siblings")
    if not isinstance(siblings, Sequence) or isinstance(siblings, (str, bytes)):
        return None
    sibling = next(
        (
            item
            for item in siblings
            if str(_metadata_field(item, "rfilename") or "") == hint.filename
        ),
        None,
    )
    if sibling is None or _sibling_sha256(sibling) != sha256:
        return None
    lfs = _metadata_field(sibling, "lfs")
    remote_size_value = _metadata_field(sibling, "size") or _metadata_field(lfs, "size")
    if int(remote_size_value or 0) != size_bytes:
        return None
    return HuggingFaceAssetSource(
        repo_id=hint.repo_id,
        revision=exact_revision,
        filename=hint.filename,
        sha256=sha256,
        size_bytes=size_bytes,
    )


def macos_where_from_urls(path: Path) -> tuple[str, ...]:
    """Read browser source URLs from the macOS download provenance attribute."""
    if sys.platform != "darwin":
        return ()
    try:
        result = subprocess.run(
            (
                "/usr/bin/xattr",
                "-px",
                "com.apple.metadata:kMDItemWhereFroms",
                str(path),
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=2.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ()
    if result.returncode != 0 or len(result.stdout) > _MAX_XATTR_BYTES * 3:
        return ()
    try:
        payload = bytes.fromhex(result.stdout.decode("ascii"))
        urls = plistlib.loads(payload)
    except (UnicodeDecodeError, ValueError, plistlib.InvalidFileException):
        return ()
    if not isinstance(urls, Sequence) or isinstance(urls, (str, bytes)):
        return ()
    return tuple(str(url) for url in urls if isinstance(url, str))


def _safetensors_source_urls(path: Path) -> tuple[str, ...]:
    """Read official Hub URLs embedded in a bounded safetensors metadata header."""
    if path.suffix.casefold() != ".safetensors":
        return ()
    try:
        with path.open("rb") as input_file:
            header_size_payload = input_file.read(8)
            if len(header_size_payload) != 8:
                return ()
            header_size = struct.unpack("<Q", header_size_payload)[0]
            if not (0 < header_size <= _MAX_SAFETENSORS_HEADER_BYTES):
                return ()
            header = json.loads(input_file.read(header_size))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, struct.error):
        return ()
    metadata = header.get("__metadata__") if isinstance(header, Mapping) else None
    if not isinstance(metadata, Mapping):
        return ()
    urls: list[str] = []
    for value in metadata.values():
        if not isinstance(value, str):
            continue
        urls.extend(_HUGGINGFACE_URL_PATTERN.findall(value))
    return tuple(urls)


def _hub_cache_hint(path: Path) -> HuggingFaceAssetHint | None:
    """Recognize the documented models--owner--repo/snapshots/commit cache layout."""
    parts = path.absolute().parts
    for index, part in enumerate(parts):
        if not part.startswith("models--") or index + 2 >= len(parts):
            continue
        repository_parts = part.removeprefix("models--").split("--", maxsplit=1)
        if len(repository_parts) != 2 or parts[index + 1] != "snapshots":
            continue
        revision = parts[index + 2]
        filename_parts = parts[index + 3 :]
        if not filename_parts:
            continue
        try:
            return HuggingFaceAssetHint(
                repo_id="/".join(repository_parts),
                revision=revision,
                filename=PurePosixPath(*filename_parts).as_posix(),
                evidence="Hugging Face cache",
            )
        except ValueError:
            return None
    return None


def _git_head_commit(git_directory: Path) -> str | None:
    """Resolve one ordinary Git HEAD without executing a subprocess."""
    try:
        head = (git_directory / "HEAD").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if _COMMIT_PATTERN.fullmatch(head):
        return head
    if not head.startswith("ref: "):
        return None
    reference = head.removeprefix("ref: ").strip()
    reference_path = git_directory / reference
    try:
        value = reference_path.read_text(encoding="utf-8").strip()
    except OSError:
        value = ""
    if _COMMIT_PATTERN.fullmatch(value):
        return value
    try:
        packed_refs = (git_directory / "packed-refs").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in packed_refs.splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[1] == reference and _COMMIT_PATTERN.fullmatch(fields[0]):
            return fields[0]
    return None


def _git_checkout_hint(path: Path) -> HuggingFaceAssetHint | None:
    """Recognize a model file inside an ordinary Hugging Face Git checkout."""
    for repository_root in path.parents:
        git_directory = repository_root / ".git"
        if not git_directory.is_dir():
            continue
        parser = configparser.ConfigParser(interpolation=None)
        try:
            parser.read(git_directory / "config", encoding="utf-8")
        except (OSError, configparser.Error):
            return None
        repo_id = next(
            (
                repo_id
                for section in parser.sections()
                if section.startswith("remote ")
                for repo_id in [_huggingface_repo_id_from_url(parser.get(section, "url", fallback=""))]
                if repo_id is not None
            ),
            None,
        )
        revision = _git_head_commit(git_directory)
        if repo_id is None or revision is None:
            return None
        try:
            filename = path.absolute().relative_to(repository_root.absolute()).as_posix()
            return HuggingFaceAssetHint(
                repo_id=repo_id,
                revision=revision,
                filename=filename,
                evidence="Hugging Face Git checkout",
            )
        except ValueError:
            return None
    return None


def _local_download_metadata(path: Path, sha256: str) -> LocalHuggingFaceDownloadMetadata | None:
    """Find matching local-dir metadata and its original repository-relative filename."""
    absolute_path = path.absolute()
    for local_directory in absolute_path.parents:
        try:
            filename = absolute_path.relative_to(local_directory).as_posix()
        except ValueError:
            continue
        metadata_path = (
            local_directory
            / ".cache"
            / "huggingface"
            / "download"
            / f"{filename}.metadata"
        )
        if not metadata_path.is_file():
            continue
        try:
            lines = metadata_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return None
        if len(lines) >= 2 and _COMMIT_PATTERN.fullmatch(lines[0]) and lines[1] == sha256:
            return LocalHuggingFaceDownloadMetadata(
                commit_hash=lines[0],
                etag=lines[1],
                filename=filename,
            )
    return None


@dataclass
class HuggingFaceAssetDiscovery:
    """Learn and persist immutable Hub provenance from normal model installation state."""

    registry: HuggingFaceAssetRegistry
    user_directory: Path
    comfyui_root: Path | None
    api: Any = field(default_factory=HfApi)
    where_from_reader: WhereFromReader = macos_where_from_urls
    _manager_hints_by_filename: (
        dict[str, tuple[_ManagerHuggingFaceHint, ...]] | None
    ) = field(init=False, default=None)
    _manager_catalog_signature: tuple[tuple[str, int, int], ...] | None = field(
        init=False,
        default=None,
    )
    _unresolved_evidence_by_sha256: dict[str, tuple[tuple[str, str, str], ...]] = field(
        init=False,
        default_factory=dict,
    )
    _cache_lock: threading.Lock = field(init=False, default_factory=threading.Lock)

    def discover(self, local_path: Path, *, sha256: str) -> HuggingFaceAssetSource | None:
        """Return cached or automatically verified provenance for one local file."""
        registered = self.registry.get(sha256)
        if registered is not None:
            return registered
        source_path = local_path.expanduser().absolute()
        size_bytes = source_path.stat().st_size
        metadata = _local_download_metadata(source_path, sha256)
        hints = self._candidate_hints(source_path)
        if metadata is not None:
            hints = tuple(
                replace(
                    hint,
                    revision=metadata.commit_hash,
                    filename=(
                        metadata.filename
                        if hint.filename == source_path.name
                        else hint.filename
                    ),
                )
                for hint in hints
            )
        evidence_fingerprint = tuple(
            (hint.repo_id, hint.revision, hint.filename) for hint in hints
        )
        with self._cache_lock:
            if self._unresolved_evidence_by_sha256.get(sha256) == evidence_fingerprint:
                return None
        for hint in hints:
            try:
                source = resolve_huggingface_asset_hint(
                    hint,
                    sha256=sha256,
                    size_bytes=size_bytes,
                    api=self.api,
                )
            except (HfHubHTTPError, httpx.HTTPError, OSError, ValueError) as exc:
                logger.debug(
                    "Hugging Face provenance candidate failed evidence=%s repo=%s file=%s: %s",
                    hint.evidence,
                    hint.repo_id,
                    hint.filename,
                    exc,
                )
                continue
            if source is not None:
                logger.info(
                    "Automatically discovered Hugging Face asset source=%s evidence=%s.",
                    source.display_reference,
                    hint.evidence,
                )
                return self.registry.upsert(source)
        with self._cache_lock:
            self._unresolved_evidence_by_sha256[sha256] = evidence_fingerprint
        return None

    def _candidate_hints(self, local_path: Path) -> tuple[HuggingFaceAssetHint, ...]:
        """Collect unique candidate locations from local install provenance."""
        candidates: list[HuggingFaceAssetHint] = []
        for path_candidate in (local_path, local_path.resolve()):
            for hint in (_hub_cache_hint(path_candidate), _git_checkout_hint(path_candidate)):
                if hint is not None:
                    candidates.append(hint)
        for url in (*self.where_from_reader(local_path), *_safetensors_source_urls(local_path)):
            hint = huggingface_hint_from_url(url, evidence="local download metadata")
            if hint is not None:
                candidates.append(hint)
        candidates.extend(self._manager_candidates(local_path))
        unique: dict[tuple[str, str, str], HuggingFaceAssetHint] = {}
        for hint in candidates:
            unique.setdefault((hint.repo_id, hint.revision, hint.filename), hint)
        return tuple(unique.values())

    def _manager_candidates(self, local_path: Path) -> tuple[HuggingFaceAssetHint, ...]:
        """Prefer Manager entries whose configured save path matches this model."""
        records = self._manager_hints().get(local_path.name, ())
        matching = tuple(record.hint for record in records if record.matches(local_path))
        if matching:
            return matching
        return tuple(
            record.hint for record in records[:_MAX_MANAGER_FILENAME_FALLBACKS]
        )

    def _manager_hints(self) -> dict[str, tuple[_ManagerHuggingFaceHint, ...]]:
        """Load Manager's model catalog and refresh it when installation state changes."""
        catalog_paths = self._manager_catalog_paths()
        signature = self._catalog_signature(catalog_paths)
        with self._cache_lock:
            if (
                self._manager_hints_by_filename is not None
                and self._manager_catalog_signature == signature
            ):
                return self._manager_hints_by_filename
            loaded = self._load_manager_hints(catalog_paths)
            self._manager_hints_by_filename = loaded
            self._manager_catalog_signature = signature
            return loaded

    def _load_manager_hints(
        self,
        catalog_paths: tuple[Path, ...],
    ) -> dict[str, tuple[_ManagerHuggingFaceHint, ...]]:
        """Parse every available Manager model-list cache without trusting its bytes."""
        mutable: dict[str, list[_ManagerHuggingFaceHint]] = {}
        for catalog_path in catalog_paths:
            try:
                payload = json.loads(catalog_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            models = payload.get("models") if isinstance(payload, Mapping) else None
            if not isinstance(models, Sequence) or isinstance(models, (str, bytes)):
                continue
            for model in models:
                if not isinstance(model, Mapping):
                    continue
                local_filename = str(model.get("filename") or "").strip()
                url = str(model.get("url") or "").strip()
                hint = huggingface_hint_from_url(url, evidence="ComfyUI Manager catalog")
                if local_filename and hint is not None:
                    mutable.setdefault(local_filename, []).append(
                        _ManagerHuggingFaceHint(
                            hint=hint,
                            save_path=str(model.get("save_path") or ""),
                        )
                    )
        return {
            filename: tuple(hints)
            for filename, hints in mutable.items()
        }

    @staticmethod
    def _catalog_signature(
        catalog_paths: tuple[Path, ...],
    ) -> tuple[tuple[str, int, int], ...]:
        """Return a stable signature that changes when Manager rewrites its catalog."""
        signature: list[tuple[str, int, int]] = []
        for path in catalog_paths:
            try:
                stat_result = path.stat()
            except OSError:
                continue
            signature.append((str(path), stat_result.st_mtime_ns, stat_result.st_size))
        return tuple(signature)

    def _manager_catalog_paths(self) -> tuple[Path, ...]:
        """Return current and legacy ComfyUI Manager model-list cache paths."""
        candidates = list(
            (self.user_directory / "__manager" / "cache").glob("*_model-list.json")
        )
        if self.comfyui_root is not None:
            candidates.extend(
                (
                    self.comfyui_root
                    / "custom_nodes"
                    / "ComfyUI-Manager"
                    / "cache"
                ).glob("*_model-list.json")
            )
        return tuple(sorted(set(path.resolve() for path in candidates if path.is_file())))


__all__ = [
    "HuggingFaceAssetDiscovery",
    "HuggingFaceAssetHint",
    "LocalHuggingFaceDownloadMetadata",
    "huggingface_hint_from_url",
    "macos_where_from_urls",
    "resolve_huggingface_asset_hint",
]
