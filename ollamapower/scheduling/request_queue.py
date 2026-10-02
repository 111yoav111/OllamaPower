"""The front door's request queue: DCP's priority heap in front of a fixed number of slots.

At most ``capacity`` inference requests run at once. Waiting requests are
admitted by priority (``X-OllamaPower-Priority: critical|high|normal|low``),
first come first served within a priority. DCP's anti-starvation rule carries
over: every time a waiting request is overtaken by a later one it counts a
*skip*, and after ``MAX_SKIPS`` skips it goes next, so a steady stream of
high-priority work can't park a low-priority request forever.

A waiter that gives up (client gone, timeout) is cancelled and never takes a
slot; a slot granted in the instant before the cancel is tracked by its
``Ticket`` and still released.
"""

from __future__ import annotations

import asyncio
from collections import Counter

from .task_pool import MAX_SKIPS, TaskPool, TaskPriority, TaskRecord


class QueueFull(Exception):
    """Too many requests are already waiting."""


class Ticket:
    """Whether a slot is held, so the holder can always release it exactly once."""

    __slots__ = ("held", "priority")

    def __init__(self, priority: TaskPriority) -> None:
        self.held = False
        self.priority = priority


class RequestGate:
    def __init__(self, capacity: int, max_queue: int, max_skips: int = MAX_SKIPS) -> None:
        self.capacity = capacity
        self.max_queue = max_queue
        self.active = 0
        self._pool = TaskPool(max_skips)
        self._waiters: dict[str, tuple[asyncio.Future[None], Ticket]] = {}

    @property
    def waiting(self) -> int:
        return len(self._pool)

    def waiting_by_priority(self) -> dict[str, int]:
        counts = Counter(r["priority"].lower() for r in self._pool.snapshot())
        return {p.name.lower(): counts.get(p.name.lower(), 0) for p in TaskPriority}

    async def acquire(self, ticket: Ticket) -> None:
        """Wait for a slot. Raises ``QueueFull`` at once if the queue is at ``max_queue``."""
        if self.active < self.capacity and not self.waiting:
            self.active += 1
            ticket.held = True
            return
        if self.waiting >= self.max_queue:
            raise QueueFull()
        record = self._pool.submit(priority=ticket.priority)
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._waiters[record.task_id] = (fut, ticket)
        try:
            await fut
        except asyncio.CancelledError:
            self._forget(record)  # still waiting: just leave the queue
            raise

    def release(self, ticket: Ticket) -> None:
        if not ticket.held:
            return
        ticket.held = False
        self.active -= 1
        self._wake()

    def _forget(self, record: TaskRecord) -> None:
        self._pool.take(record)
        self._pool.records.pop(record.task_id, None)
        self._waiters.pop(record.task_id, None)

    def _next(self) -> TaskRecord:
        waiting = self._pool.candidates(len(self._pool))
        due = [r for r in waiting if r.skipped_count >= self._pool.max_skips]
        chosen = min(due, key=lambda r: r.sequence) if due else waiting[0]
        # Everyone who arrived earlier and is overtaken now counts a skip.
        self._pool.note_skipped([r for r in waiting if r.sequence < chosen.sequence])
        return chosen

    def _wake(self) -> None:
        while self.waiting and self.active < self.capacity:
            record = self._next()
            fut, ticket = self._waiters[record.task_id]
            self._forget(record)
            if fut.done():  # cancelled; its handler runs later and finds nothing to do
                continue
            self.active += 1
            ticket.held = True  # owned now, even if the waiter is cancelled before it resumes
            fut.set_result(None)
