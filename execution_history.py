"""Persistent runtime observations used by cost-aware environment placement."""

from __future__ import annotations

import logging
import math
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

logger = logging.getLogger(__name__)

EXECUTION_HISTORY_FILENAME = "execution-history.sqlite3"
_RECENT_SAMPLE_LIMIT = 20


@dataclass(frozen=True)
class ExecutionObservation:
    """Describe one completed remote component invocation."""

    component_signature: str
    environment_id: str
    provider: str
    elapsed_seconds: float
    recorded_at_epoch: float


@dataclass(frozen=True)
class ExecutionEstimate:
    """Summarize recent successful runtimes for one component/environment pair."""

    execution_seconds: float
    sample_count: int


@dataclass
class ExecutionHistory:
    """Store bounded component timing samples in a local SQLite database."""

    database_path: Path

    @classmethod
    def for_user_directory(cls, user_directory: Path) -> "ExecutionHistory":
        """Create history storage beneath the node pack's ComfyUI user directory."""
        return cls(
            database_path=(
                user_directory.expanduser().resolve()
                / "comfyui-modal"
                / EXECUTION_HISTORY_FILENAME
            )
        )

    def record(self, observation: ExecutionObservation) -> None:
        """Persist one successful, finite runtime observation."""
        _validate_observation(observation)
        with self._connect() as connection:
            _insert_observation(connection, observation)
            _prune_observations(connection, observation)

    def estimates(
        self,
        component_signature: str,
        environment_ids: Sequence[str],
    ) -> Mapping[str, ExecutionEstimate]:
        """Return robust recent runtime estimates for the requested environments."""
        normalized_ids = tuple(
            dict.fromkeys(
                environment_id.strip()
                for environment_id in environment_ids
                if environment_id.strip()
            )
        )
        if not normalized_ids or not self.database_path.exists():
            return {}
        rows = self._load_estimate_rows(component_signature, normalized_ids)
        samples_by_environment: dict[str, list[float]] = {}
        for environment_id, elapsed_seconds in rows:
            samples = samples_by_environment.setdefault(str(environment_id), [])
            if len(samples) < _RECENT_SAMPLE_LIMIT:
                samples.append(float(elapsed_seconds))
        return {
            environment_id: ExecutionEstimate(
                execution_seconds=_median(samples),
                sample_count=len(samples),
            )
            for environment_id, samples in samples_by_environment.items()
            if samples
        }

    def _load_estimate_rows(
        self,
        component_signature: str,
        environment_ids: tuple[str, ...],
    ) -> list[tuple[str, float]]:
        """Load recent matching observation rows, returning none on index failure."""
        placeholders = ",".join("?" for _ in environment_ids)
        try:
            with self._connect() as connection:
                return connection.execute(
                    f"""
                    SELECT environment_id, elapsed_seconds
                    FROM execution_observations
                    WHERE component_signature = ?
                      AND environment_id IN ({placeholders})
                    ORDER BY recorded_at_epoch DESC, id DESC
                    """,
                    (component_signature, *environment_ids),
                ).fetchall()
        except (OSError, sqlite3.Error) as exc:
            logger.warning("Unable to read remote execution history: %s", exc)
            return []

    def _connect(self) -> sqlite3.Connection:
        """Open an initialized short-lived SQLite connection."""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path, timeout=5.0)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        _initialize_schema(connection)
        return connection


def _validate_observation(observation: ExecutionObservation) -> None:
    """Reject invalid runtime observations before touching persistent state."""
    if not observation.component_signature.strip():
        raise ValueError("component_signature must not be empty.")
    if not observation.environment_id.strip():
        raise ValueError("environment_id must not be empty.")
    if not observation.provider.strip():
        raise ValueError("provider must not be empty.")
    if not math.isfinite(observation.elapsed_seconds) or observation.elapsed_seconds < 0:
        raise ValueError("elapsed_seconds must be finite and non-negative.")
    if (
        not math.isfinite(observation.recorded_at_epoch)
        or observation.recorded_at_epoch < 0
    ):
        raise ValueError("recorded_at_epoch must be finite and non-negative.")


def _insert_observation(
    connection: sqlite3.Connection,
    observation: ExecutionObservation,
) -> None:
    """Insert one observation into an initialized history database."""
    connection.execute(
        """
        INSERT INTO execution_observations (
            component_signature,
            environment_id,
            provider,
            elapsed_seconds,
            recorded_at_epoch
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (
            observation.component_signature,
            observation.environment_id,
            observation.provider,
            observation.elapsed_seconds,
            observation.recorded_at_epoch,
        ),
    )


def _prune_observations(
    connection: sqlite3.Connection,
    observation: ExecutionObservation,
) -> None:
    """Retain only the bounded recent window for one signature and environment."""
    connection.execute(
        """
        DELETE FROM execution_observations
        WHERE id IN (
            SELECT id
            FROM execution_observations
            WHERE component_signature = ? AND environment_id = ?
            ORDER BY recorded_at_epoch DESC, id DESC
            LIMIT -1 OFFSET ?
        )
        """,
        (
            observation.component_signature,
            observation.environment_id,
            _RECENT_SAMPLE_LIMIT,
        ),
    )


def _initialize_schema(connection: sqlite3.Connection) -> None:
    """Create the timing table and lookup index when missing."""
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS execution_observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            component_signature TEXT NOT NULL,
            environment_id TEXT NOT NULL,
            provider TEXT NOT NULL,
            elapsed_seconds REAL NOT NULL,
            recorded_at_epoch REAL NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS execution_observations_lookup
        ON execution_observations (
            component_signature,
            environment_id,
            recorded_at_epoch DESC
        )
        """
    )


def record_completed_execution(
    *,
    history: ExecutionHistory | None,
    component_signature: str | None,
    environment_id: str,
    provider: str,
    elapsed_seconds: float,
) -> None:
    """Best-effort record one completed invocation without failing its result."""
    if history is None or not component_signature:
        return
    try:
        history.record(
            ExecutionObservation(
                component_signature=component_signature,
                environment_id=environment_id,
                provider=provider,
                elapsed_seconds=elapsed_seconds,
                recorded_at_epoch=time.time(),
            )
        )
    except (OSError, sqlite3.Error, ValueError) as exc:
        logger.warning(
            "Unable to persist remote execution timing environment=%s: %s",
            environment_id,
            exc,
        )


def _median(values: Sequence[float]) -> float:
    """Return the median of one non-empty numeric sequence."""
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2
