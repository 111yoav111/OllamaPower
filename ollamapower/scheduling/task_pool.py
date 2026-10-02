"""The priority queue, extracted from DCP's load balancer.

Records are ordered by priority, then by enqueue time, so the queue is FIFO
within a priority level. The dispatcher looks at the first few records; a
record that can't be placed is *skipped*, and after ``MAX_SKIPS`` skips it is
force-dispatched to any available target so a demanding request can't starve.
A failed record is requeued up to ``max_retries`` times.

Priorities map onto the ``X-OllamaPower-Priority`` header that OllamaGate will
set later: ``critical | high | normal | low``.

All methods are synchronous: callers share one event loop, so each call is
atomic with respect to the others.
"""

from __future__ import annotations

import enum
import heapq
import itertools
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

MAX_SKIPS = 10
DISPATCH_WINDOW = 3  # records considered per dispatch attempt
MAX_RETRIES = 3


class TaskPriority(enum.IntEnum):
    """Lower value = served first."""

    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4

    @classmethod
    def from_header(cls, value: str | None) -> TaskPriority:
        """Parse an ``X-OllamaPower-Priority`` value; anything unknown is NORMAL."""
        if not value:
            return cls.NORMAL
        try:
            return cls[value.strip().upper()]
        except KeyError:
            return cls.NORMAL


class TaskStatus(str, enum.Enum):
    PENDING = "pending"  # waiting in the queue
    ASSIGNED = "assigned"  # handed to a target
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"  # failed for good
    REQUEUED = "requeued"  # failed, waiting for another try


_sequence = itertools.count()


@dataclass(order=True)
class TaskRecord:
    """One queued unit of work. Only ``priority`` and the enqueue order take part in sorting."""

    priority: int
    enqueue_time: float = field(default_factory=time.monotonic)
    # Tie-breaker: monotonic clocks can return equal values for records queued back to back.
    sequence: int = field(default_factory=lambda: next(_sequence))

    task_id: str = field(compare=False, default_factory=lambda: str(uuid.uuid4()))
    payload: Any = field(compare=False, default=None)
    difficulty: float = field(compare=False, default=1.0)
    status: TaskStatus = field(compare=False, default=TaskStatus.PENDING)
    assigned_node: str | None = field(compare=False, default=None)
    retries: int = field(compare=False, default=0)
    max_retries: int = field(compare=False, default=MAX_RETRIES)
    skipped_count: int = field(compare=False, default=0)
    result: Any = field(compare=False, default=None)
    error: str | None = field(compare=False, default=None)
    started_at: float | None = field(compare=False, default=None)
    finished_at: float | None = field(compare=False, default=None)

    @property
    def elapsed(self) -> float | None:
        if self.started_at is None:
            return None
        return (self.finished_at or time.monotonic()) - self.started_at

    @property
    def priority_name(self) -> str:
        return TaskPriority(self.priority).name


class TaskPool:
    """A heap of ``TaskRecord`` plus an index of every record not yet finished."""

    def __init__(self, max_skips: int = MAX_SKIPS) -> None:
        self.max_skips = max_skips
        self._heap: list[TaskRecord] = []
        self._queued: set[str] = set()  # ids in the heap, so a record is never queued twice
        self.records: dict[str, TaskRecord] = {}

    def __len__(self) -> int:
        return len(self._heap)

    def submit(
        self,
        payload: Any = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        difficulty: float = 1.0,
        max_retries: int = MAX_RETRIES,
    ) -> TaskRecord:
        record = TaskRecord(
            priority=TaskPriority(priority).value, payload=payload, difficulty=difficulty, max_retries=max_retries
        )
        self.push(record)
        self.records[record.task_id] = record
        return record

    def push(self, record: TaskRecord) -> None:
        if record.task_id in self._queued:
            return
        self._queued.add(record.task_id)
        heapq.heappush(self._heap, record)

    def candidates(self, n: int = DISPATCH_WINDOW) -> list[TaskRecord]:
        """The ``n`` records that should be served next, best first."""
        return heapq.nsmallest(n, self._heap)

    def take(self, record: TaskRecord) -> bool:
        """Remove ``record`` from the queue; False if it was no longer there."""
        if record.task_id not in self._queued:
            return False
        self._heap.remove(record)
        self._queued.discard(record.task_id)
        heapq.heapify(self._heap)
        return True

    def note_skipped(self, records: list[TaskRecord]) -> list[TaskRecord]:
        """Count a failed placement attempt; return the records now due for forced dispatch."""
        due = []
        for record in records:
            record.skipped_count += 1
            if record.skipped_count >= self.max_skips:
                due.append(record)
        return due

    def assign(self, record: TaskRecord, node_id: str) -> None:
        record.status = TaskStatus.ASSIGNED
        record.assigned_node = node_id
        record.started_at = time.monotonic()

    def requeue(self, record: TaskRecord) -> None:
        """Put a record back after a failed *send* (not counted as a retry)."""
        record.status = TaskStatus.REQUEUED
        record.assigned_node = None
        record.started_at = None
        record.skipped_count = 0
        self.push(record)

    def fail(self, record: TaskRecord, error: str) -> bool:
        """Record a failed attempt. Requeues and returns True while retries remain."""
        record.error = error
        record.retries += 1
        if record.retries <= record.max_retries:
            self.requeue(record)
            return True
        record.status = TaskStatus.FAILED
        record.finished_at = time.monotonic()
        self.records.pop(record.task_id, None)
        return False

    def complete(self, record: TaskRecord, result: Any = None) -> None:
        record.status = TaskStatus.DONE
        record.result = result
        record.finished_at = time.monotonic()
        self.records.pop(record.task_id, None)

    def snapshot(self) -> list[dict[str, Any]]:
        return [
            {"task_id": r.task_id, "priority": r.priority_name, "status": r.status.value, "retries": r.retries}
            for r in sorted(self._heap)
        ]
