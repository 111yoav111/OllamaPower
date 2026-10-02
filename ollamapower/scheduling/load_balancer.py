"""Matches queued work to nodes; adapted from DCP's ``LoadBalancer``.

Three loops run on the hub's event loop:

* **dispatch** - every ``DISPATCH_INTERVAL`` take the best few records from the
  ``TaskPool`` and hand the first one that fits to the least loaded node;
  records that keep not fitting are force-dispatched after ``MAX_SKIPS``.
* **heartbeat** - a node silent for ``HEARTBEAT_TIMEOUT`` is kicked and its
  work is requeued.
* **metrics** - a periodic ``LBMetrics`` snapshot for the TUI and CLI.

The load model is still DCP's (CPU percent, +20 while busy). Phase 2/5 replaces
``NodeState.load_score`` with a score over free VRAM, GPU utilisation, link
quality, warm models and queue length; the loops stay.

Network I/O happens through callbacks (``send_task_cb``, ``kick_node_cb``), so
this module never touches a socket and is easy to test.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .task_pool import TaskPool, TaskPriority, TaskRecord, TaskStatus

ACTIVE_STATES = (TaskStatus.ASSIGNED, TaskStatus.RUNNING)

# CPU percent at which a node stops getting work, and below which it recovers.
OVERLOAD_PERCENT = 90.0
RECOVER_PERCENT = 80.0
BUSY_PENALTY = 20.0
# Difficulty -> required free capacity (DCP's scale), capped so nothing is unplaceable.
DIFFICULTY_CAPACITY_SCALE = 6.0
MAX_REQUIRED_CAPACITY = 95.0
COMPLETION_WINDOW = 60.0  # seconds counted in throughput_per_min

SendCallback = Callable[[str, TaskRecord], Awaitable[bool]]
KickCallback = Callable[[str], Awaitable[None]]


class NodeStatus(str, enum.Enum):
    READY = "ready"
    BUSY = "busy"
    OVERLOADED = "overloaded"
    OFFLINE = "offline"


@dataclass
class NodeState:
    """What the hub knows about one connected node."""

    node_id: str
    address: tuple[str, int]
    status: NodeStatus = NodeStatus.READY
    cpu_usage: float = 0.0
    active_task_ids: set[str] = field(default_factory=set)
    completed_tasks: int = 0
    failed_tasks: int = 0
    last_heartbeat: float = field(default_factory=time.monotonic)
    connected_at: float = field(default_factory=time.monotonic)

    @property
    def load_score(self) -> float:
        return self.cpu_usage + (BUSY_PENALTY if self.status is NodeStatus.BUSY else 0.0)

    @property
    def free_capacity(self) -> float:
        return max(0.0, 100.0 - self.load_score)

    @property
    def is_available(self) -> bool:
        return self.status in (NodeStatus.READY, NodeStatus.BUSY) and self.cpu_usage < OVERLOAD_PERCENT

    @property
    def uptime(self) -> float:
        return time.monotonic() - self.connected_at

    def settle(self) -> None:
        """READY when nothing is running any more."""
        if not self.active_task_ids and self.status is NodeStatus.BUSY:
            self.status = NodeStatus.READY


def required_capacity(record: TaskRecord) -> float:
    return min(record.difficulty * DIFFICULTY_CAPACITY_SCALE, MAX_REQUIRED_CAPACITY)


@dataclass
class LBMetrics:
    timestamp: float
    total_nodes: int
    ready_nodes: int
    busy_nodes: int
    offline_nodes: int
    queue_size: int
    tasks_completed: int
    tasks_failed: int
    avg_cpu: float
    throughput_per_min: int


class LoadBalancer:
    HEARTBEAT_TIMEOUT = 15.0
    DISPATCH_INTERVAL = 0.2
    METRICS_INTERVAL = 2.0

    def __init__(self, pool: TaskPool | None = None, logger: logging.Logger | None = None) -> None:
        self.log = logger or logging.getLogger("ollamapower.scheduling")
        self.pool = pool or TaskPool()
        self.nodes: dict[str, NodeState] = {}
        self.send_task_cb: SendCallback | None = None
        self.kick_node_cb: KickCallback | None = None
        self.on_node_change_cbs: list[Callable[[NodeState], None]] = []
        self.on_task_done_cbs: list[Callable[[TaskRecord], None]] = []
        self.on_metrics_cbs: list[Callable[[LBMetrics], None]] = []
        self.running = False
        self.tasks_completed = 0
        self.tasks_failed = 0
        self._completion_times: list[float] = []

    async def start(self) -> None:
        """Run the three loops until ``stop()``."""
        self.running = True
        self.log.info("load balancer started")
        await asyncio.gather(self._dispatch_loop(), self._heartbeat_loop(), self._metrics_loop())

    def stop(self) -> None:
        self.running = False

    def set_send_callback(self, cb: SendCallback) -> None:
        self.send_task_cb = cb

    def set_kick_callback(self, cb: KickCallback) -> None:
        self.kick_node_cb = cb

    # ----- nodes -----------------------------------------------------------------------

    def register_node(self, node_id: str, host: str, port: int) -> NodeState:
        node = self.nodes.get(node_id)
        if node is not None:
            self.log.warning("node %s registered again; refreshing", node_id)
            node.status = NodeStatus.READY
            node.last_heartbeat = time.monotonic()
        else:
            node = NodeState(node_id=node_id, address=(host, port))
            self.nodes[node_id] = node
            self.log.info("node registered: %s @ %s:%d", node_id, host, port)
        self._fire(self.on_node_change_cbs, node)
        return node

    async def unregister_node(self, node_id: str) -> None:
        node = self.nodes.pop(node_id, None)
        if node is None:
            return
        node.status = NodeStatus.OFFLINE
        self._fire(self.on_node_change_cbs, node)
        self.log.info("node unregistered: %s", node_id)
        await self._handle_node_failure(node)

    def update_node_stats(self, node_id: str, cpu: float, status: NodeStatus | None = None) -> None:
        node = self.nodes.get(node_id)
        if node is None:
            self.log.warning("stats update for unknown node %s", node_id)
            return
        node.cpu_usage = cpu
        node.last_heartbeat = time.monotonic()
        if status is not None:
            node.status = status
        if node.cpu_usage >= OVERLOAD_PERCENT:
            if node.status is not NodeStatus.OVERLOADED:
                node.status = NodeStatus.OVERLOADED
                self.log.warning("node %s overloaded (cpu %.0f%%)", node_id, cpu)
        elif node.cpu_usage < RECOVER_PERCENT and node.status is NodeStatus.OVERLOADED:
            node.status = NodeStatus.BUSY if node.active_task_ids else NodeStatus.READY
        self._fire(self.on_node_change_cbs, node)

    async def kick_node(self, node_id: str, reason: str = "manual") -> None:
        self.log.warning("kicking node %s (%s)", node_id, reason)
        if self.kick_node_cb is not None:
            try:
                await self.kick_node_cb(node_id)
            except Exception as exc:  # noqa: BLE001 - the node is dropped either way
                self.log.warning("kick callback for %s failed: %s", node_id, type(exc).__name__)
        await self.unregister_node(node_id)

    # ----- tasks -----------------------------------------------------------------------

    def submit_task(
        self,
        payload: Any = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        difficulty: float = 1.0,
        max_retries: int = 3,
    ) -> str:
        record = self.pool.submit(payload, priority, difficulty, max_retries)
        self.log.debug("task %.8s queued (%s)", record.task_id, record.priority_name)
        return record.task_id

    def _owned(self, node_id: str, task_id: str, what: str) -> TaskRecord | None:
        """The record if ``node_id`` is really running it; stale or foreign reports are ignored."""
        record = self.pool.records.get(task_id)
        if record is None or record.assigned_node != node_id or record.status not in ACTIVE_STATES:
            self.log.warning("ignoring %s for task %.8s from %s (not assigned to it)", what, task_id, node_id)
            return None
        return record

    def _release(self, node_id: str, task_id: str) -> NodeState | None:
        node = self.nodes.get(node_id)
        if node is not None:
            node.active_task_ids.discard(task_id)
            node.settle()
        return node

    def task_completed(self, node_id: str, task_id: str, result: Any = None) -> None:
        record = self._owned(node_id, task_id, "result")
        if record is None:
            return
        self.pool.complete(record, result)
        self.tasks_completed += 1
        self._completion_times.append(time.monotonic())
        node = self._release(node_id, task_id)
        if node is not None:
            node.completed_tasks += 1
        self.log.info("task %.8s done by %s (%.2fs)", task_id, node_id, record.elapsed or 0.0)
        self._fire(self.on_task_done_cbs, record)

    def task_failed(self, node_id: str, task_id: str, error: str) -> None:
        record = self._owned(node_id, task_id, "failure")
        if record is None:
            return
        node = self._release(node_id, task_id)
        if node is not None:
            node.failed_tasks += 1
        if self.pool.fail(record, error):
            self.log.warning(
                "task %.8s failed (%d/%d), requeued: %s", task_id, record.retries, record.max_retries, error
            )
        else:
            self.tasks_failed += 1
            self.log.error("task %.8s failed for good after %d tries", task_id, record.retries)
            self._fire(self.on_task_done_cbs, record)

    # ----- dispatch --------------------------------------------------------------------

    def _select_node(self, record: TaskRecord | None) -> NodeState | None:
        """The least loaded available node; with a record, only nodes with enough free capacity."""
        nodes = [n for n in self.nodes.values() if n.is_available]
        if record is not None:
            nodes = [n for n in nodes if n.free_capacity >= required_capacity(record)]
        return min(nodes, key=lambda n: n.load_score, default=None)

    async def try_dispatch(self) -> TaskRecord | None:
        """One dispatch attempt; returns the record that was sent, if any."""
        candidates = self.pool.candidates()
        if not candidates or self._select_node(None) is None:
            return None  # no node at all is not a "skip": capacity checks must still apply later
        record, node = None, None
        for candidate in candidates:
            node = self._select_node(candidate)
            if node is not None:
                record = candidate
                break
        if record is None:
            due = self.pool.note_skipped(candidates)
            if due:
                record, node = due[0], self._select_node(None)
                self.log.warning("force-dispatching task %.8s after %d skips", record.task_id, record.skipped_count)
        if record is None or node is None or not self.pool.take(record):
            return None

        self.pool.assign(record, node.node_id)
        node.active_task_ids.add(record.task_id)
        node.status = NodeStatus.BUSY
        self.log.info("dispatching task %.8s -> %s", record.task_id, node.node_id)

        sent = False
        if self.send_task_cb is not None:
            try:
                sent = await self.send_task_cb(node.node_id, record)
            except Exception as exc:  # noqa: BLE001 - one bad send must not stop the dispatch loop
                self.log.warning("send of task %.8s to %s failed: %s", record.task_id, node.node_id, type(exc).__name__)
        if sent:
            return record
        # While we awaited, the node may have dropped and its work been requeued already.
        if record.assigned_node == node.node_id and record.status in ACTIVE_STATES:
            self._release(node.node_id, record.task_id)
            self.pool.requeue(record)
        return None

    async def _dispatch_loop(self) -> None:
        while self.running:
            await self.try_dispatch()
            await asyncio.sleep(self.DISPATCH_INTERVAL)

    # ----- heartbeats ------------------------------------------------------------------

    async def check_heartbeats(self, now: float | None = None) -> list[str]:
        """Kick silent nodes (their work is requeued); returns their ids."""
        now = time.monotonic() if now is None else now
        timed_out = [n.node_id for n in self.nodes.values() if now - n.last_heartbeat > self.HEARTBEAT_TIMEOUT]
        for node_id in timed_out:
            self.log.error("node %s heartbeat timeout", node_id)
            await self.kick_node(node_id, "heartbeat timeout")
        return timed_out

    async def _heartbeat_loop(self) -> None:
        while self.running:
            await asyncio.sleep(self.HEARTBEAT_TIMEOUT / 2)
            await self.check_heartbeats()

    async def _handle_node_failure(self, node: NodeState) -> None:
        for task_id in list(node.active_task_ids):
            record = self.pool.records.get(task_id)
            if record is not None and record.assigned_node == node.node_id and record.status in ACTIVE_STATES:
                self.log.warning("requeueing task %.8s after node %s failed", task_id, node.node_id)
                self.task_failed(node.node_id, task_id, "node_offline")
        node.active_task_ids.clear()

    # ----- metrics ---------------------------------------------------------------------

    def compute_metrics(self) -> LBMetrics:
        now = time.monotonic()
        nodes = list(self.nodes.values())
        cpus = [n.cpu_usage for n in nodes if n.status is not NodeStatus.OFFLINE]
        self._completion_times = [t for t in self._completion_times if t > now - COMPLETION_WINDOW]
        return LBMetrics(
            timestamp=now,
            total_nodes=len(nodes),
            ready_nodes=sum(n.status is NodeStatus.READY for n in nodes),
            busy_nodes=sum(n.status is NodeStatus.BUSY for n in nodes),
            offline_nodes=sum(n.status is NodeStatus.OFFLINE for n in nodes),
            queue_size=len(self.pool),
            tasks_completed=self.tasks_completed,
            tasks_failed=self.tasks_failed,
            avg_cpu=sum(cpus) / len(cpus) if cpus else 0.0,
            throughput_per_min=len(self._completion_times),
        )

    async def _metrics_loop(self) -> None:
        while self.running:
            await asyncio.sleep(self.METRICS_INTERVAL)
            self._fire(self.on_metrics_cbs, self.compute_metrics())

    def get_snapshot(self) -> dict[str, Any]:
        """Everything the TUI and CLI show about scheduling, as JSON-safe data."""
        return {
            "nodes": {
                node_id: {
                    "status": n.status.value,
                    "cpu": n.cpu_usage,
                    "active_tasks": sorted(n.active_task_ids),
                    "completed": n.completed_tasks,
                    "failed": n.failed_tasks,
                    "uptime": round(n.uptime, 1),
                    "load_score": round(n.load_score, 1),
                }
                for node_id, n in self.nodes.items()
            },
            "queue": self.pool.snapshot(),
            "stats": {"completed": self.tasks_completed, "failed": self.tasks_failed},
        }

    # ----- events ----------------------------------------------------------------------

    def _fire(self, callbacks: list[Callable[[Any], None]], value: Any) -> None:
        for cb in callbacks:
            try:
                cb(value)
            except Exception as exc:  # noqa: BLE001 - a broken listener must not stop scheduling
                self.log.warning("event callback failed: %s", type(exc).__name__)
