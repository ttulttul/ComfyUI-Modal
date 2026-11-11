"""Per-prompt accounting of busy remote GPU time, broken down by workflow node.

Every streamed Modal call reports which node is executing. The time between two
``executing`` events is charged to the earlier node. The time before the first
node starts (session restore, input download) counts as setup, and the time after
``execution_success`` (result upload and transfer) counts as returning results.
Time spent waiting for a container is tracked separately, because a cold start
is billed to the container but is not work done for any one node.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Mapping

logger = logging.getLogger(__name__)

_LEDGER_LIMIT = 32
_TERMINAL_PHASES = frozenset(
    {"execution_success", "execution_error", "execution_interrupted"}
)


@dataclass
class NodeGpuUsage:
    """Accumulated GPU time for one workflow node across every call that ran it."""

    node_id: str
    label: str
    class_type: str
    seconds: float = 0.0
    runs: int = 0

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable view for the frontend."""
        return {
            "node_id": self.node_id,
            "label": self.label,
            "class_type": self.class_type,
            "seconds": round(self.seconds, 3),
            "runs": self.runs,
        }


@dataclass
class PromptGpuUsage:
    """Accumulated busy GPU time for every remote call made by one prompt."""

    prompt_id: str
    client_id: str | None = None
    wait_seconds: float = 0.0
    setup_seconds: float = 0.0
    return_seconds: float = 0.0
    invocation_count: int = 0
    failed_invocation_count: int = 0
    seconds_by_gpu: dict[str, float] = field(default_factory=dict)
    nodes: dict[str, NodeGpuUsage] = field(default_factory=dict)

    @property
    def node_seconds(self) -> float:
        """Return the GPU time attributed to individual nodes."""
        return sum(node.seconds for node in self.nodes.values())

    @property
    def busy_seconds(self) -> float:
        """Return all GPU time spent working on this prompt's calls."""
        return sum(self.seconds_by_gpu.values())

    def estimated_cost_usd(self, usd_per_second_by_gpu: Mapping[str, float]) -> float | None:
        """Return busy time priced per GPU type, or None when a price is unknown."""
        total = 0.0
        for gpu, seconds in self.seconds_by_gpu.items():
            rate = usd_per_second_by_gpu.get(gpu)
            if rate is None:
                return None
            total += seconds * rate
        return total

    def to_payload(self, usd_per_second_by_gpu: Mapping[str, float]) -> dict[str, Any]:
        """Return a JSON-serializable snapshot for the configurator panel."""
        cost = self.estimated_cost_usd(usd_per_second_by_gpu)
        return {
            "prompt_id": self.prompt_id,
            "busy_seconds": round(self.busy_seconds, 3),
            "node_seconds": round(self.node_seconds, 3),
            "setup_seconds": round(self.setup_seconds, 3),
            "return_seconds": round(self.return_seconds, 3),
            "wait_seconds": round(self.wait_seconds, 3),
            "invocation_count": self.invocation_count,
            "failed_invocation_count": self.failed_invocation_count,
            "seconds_by_gpu": {
                gpu: round(seconds, 3) for gpu, seconds in sorted(self.seconds_by_gpu.items())
            },
            "estimated_cost_usd": round(cost, 6) if cost is not None else None,
            "nodes": [
                node.to_payload()
                for node in sorted(
                    self.nodes.values(), key=lambda node: (-node.seconds, node.node_id)
                )
            ],
        }


def _node_identity(payload: Mapping[str, Any], node_id: str) -> tuple[str, str]:
    """Return a display label and class type for one node of a remote payload."""
    prompt = payload.get("subgraph_prompt")
    prompt_node = prompt.get(node_id) if isinstance(prompt, Mapping) else None
    if not isinstance(prompt_node, Mapping):
        return f"Node {node_id}", ""
    class_type = str(prompt_node.get("class_type") or "")
    meta = prompt_node.get("_meta")
    title = str(meta.get("title") or "").strip() if isinstance(meta, Mapping) else ""
    name = title or class_type or "Node"
    return f"{name} #{node_id}", class_type


@dataclass
class InvocationGpuUsage:
    """Time one streamed remote call, attributing busy time to its executing nodes."""

    payload: Mapping[str, Any]
    dispatched_at: float
    first_event_at: float | None = None
    active_node_id: str | None = None
    active_node_started_at: float | None = None
    finished_executing_at: float | None = None
    node_seconds: dict[str, float] = field(default_factory=dict)

    def observe(self, received_at: float, stream_event: Mapping[str, Any]) -> None:
        """Record one stream event received at ``received_at`` (monotonic seconds)."""
        if self.first_event_at is None:
            self.first_event_at = received_at
        if str(stream_event.get("kind", "")) != "progress":
            return
        if stream_event.get("event_type"):
            return
        phase = str(stream_event.get("phase", "executing"))
        if phase in _TERMINAL_PHASES:
            self._close_active_node(received_at)
            if self.finished_executing_at is None:
                self.finished_executing_at = received_at
            return
        active_node_id = stream_event.get("active_node_id")
        if phase != "executing" or active_node_id is None:
            return
        self._close_active_node(received_at)
        self.active_node_id = str(active_node_id)
        self.active_node_started_at = received_at

    def _close_active_node(self, now: float) -> None:
        """Charge the running node for the time since it started."""
        if self.active_node_id is None or self.active_node_started_at is None:
            return
        self.node_seconds[self.active_node_id] = self.node_seconds.get(
            self.active_node_id, 0.0
        ) + max(0.0, now - self.active_node_started_at)
        self.active_node_id = None
        self.active_node_started_at = None

    def finish(self, finished_at: float) -> tuple[float, float, float, float]:
        """Close the call and return ``(wait, setup, return, busy)`` seconds."""
        self._close_active_node(self.finished_executing_at or finished_at)
        if self.first_event_at is None:
            return finished_at - self.dispatched_at, 0.0, 0.0, 0.0
        wait = self.first_event_at - self.dispatched_at
        busy = max(0.0, finished_at - self.first_event_at)
        if not self.node_seconds:
            return wait, busy, 0.0, busy
        returning = (
            max(0.0, finished_at - self.finished_executing_at)
            if self.finished_executing_at is not None
            else 0.0
        )
        setup = max(0.0, busy - returning - sum(self.node_seconds.values()))
        return wait, setup, returning, busy


class GpuUsageLedger:
    """Thread-safe, bounded store of per-prompt GPU usage."""

    def __init__(self, limit: int = _LEDGER_LIMIT) -> None:
        """Create an empty ledger that keeps at most ``limit`` prompts."""
        self._lock = threading.Lock()
        self._limit = limit
        self._prompts: OrderedDict[str, PromptGpuUsage] = OrderedDict()

    def record(
        self,
        invocation: InvocationGpuUsage,
        *,
        finished_at: float,
        succeeded: bool,
        usd_per_second_by_gpu: Mapping[str, float],
    ) -> tuple[str | None, dict[str, Any]] | None:
        """Add one finished call to its prompt.

        Returns the prompt's client id and an updated snapshot, or None when the
        call belongs to no prompt.
        """
        prompt_id = invocation.payload.get("prompt_id")
        if prompt_id is None:
            return None
        wait, setup, returning, busy = invocation.finish(finished_at)
        gpu = str(invocation.payload.get("modal_gpu") or "unknown")
        client_id = (invocation.payload.get("extra_data") or {}).get("client_id")
        with self._lock:
            usage = self._prompts.get(str(prompt_id))
            if usage is None:
                usage = PromptGpuUsage(prompt_id=str(prompt_id))
                self._prompts[str(prompt_id)] = usage
                while len(self._prompts) > self._limit:
                    self._prompts.popitem(last=False)
            self._prompts.move_to_end(str(prompt_id))
            if client_id is not None:
                usage.client_id = str(client_id)
            usage.invocation_count += 1
            usage.failed_invocation_count += 0 if succeeded else 1
            usage.wait_seconds += wait
            usage.setup_seconds += setup
            usage.return_seconds += returning
            usage.seconds_by_gpu[gpu] = usage.seconds_by_gpu.get(gpu, 0.0) + busy
            for node_id, seconds in invocation.node_seconds.items():
                node = usage.nodes.get(node_id)
                if node is None:
                    label, class_type = _node_identity(invocation.payload, node_id)
                    node = NodeGpuUsage(node_id=node_id, label=label, class_type=class_type)
                    usage.nodes[node_id] = node
                node.seconds += seconds
                node.runs += 1
            return usage.client_id, usage.to_payload(usd_per_second_by_gpu)

    def clear(self) -> None:
        """Forget every recorded prompt."""
        with self._lock:
            self._prompts.clear()

    def snapshot(
        self, prompt_id: str, usd_per_second_by_gpu: Mapping[str, float]
    ) -> dict[str, Any] | None:
        """Return the usage recorded for one prompt, when any."""
        with self._lock:
            usage = self._prompts.get(str(prompt_id))
            return usage.to_payload(usd_per_second_by_gpu) if usage is not None else None


GPU_USAGE_LEDGER = GpuUsageLedger()
