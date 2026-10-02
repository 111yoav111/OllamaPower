"""RequestGate: priorities, FIFO, anti-starvation, cancellation."""

from __future__ import annotations

import asyncio

import pytest

from ollamapower.scheduling.request_queue import QueueFull, RequestGate, Ticket
from ollamapower.scheduling.task_pool import TaskPriority as P


async def _queue(gate: RequestGate, prios: list[P], order: list[str]) -> list[tuple[asyncio.Task, Ticket]]:
    waiters = []
    for i, prio in enumerate(prios):
        ticket = Ticket(prio)

        async def wait(t=ticket, name=f"{prio.name.lower()}{i}"):
            await gate.acquire(t)
            order.append(name)

        waiters.append((asyncio.create_task(wait()), ticket))
        await asyncio.sleep(0)
    return waiters


async def _drain(gate: RequestGate, holder: Ticket, waiters) -> None:
    gate.release(holder)
    for task, ticket in waiters:
        await task
        gate.release(ticket)


async def test_free_slot_is_immediate():
    gate = RequestGate(capacity=2, max_queue=10)
    a, b = Ticket(P.NORMAL), Ticket(P.NORMAL)
    await gate.acquire(a)
    await gate.acquire(b)
    assert gate.active == 2 and a.held and b.held


async def test_priority_then_fifo():
    gate = RequestGate(capacity=1, max_queue=10)
    holder = Ticket(P.NORMAL)
    await gate.acquire(holder)
    order: list[str] = []
    waiters = await _queue(gate, [P.LOW, P.NORMAL, P.CRITICAL, P.NORMAL, P.HIGH], order)
    assert gate.waiting == 5
    assert gate.waiting_by_priority() == {"critical": 1, "high": 1, "normal": 2, "low": 1}
    await _drain(gate, holder, sorted(waiters, key=lambda w: (w[1].priority, waiters.index(w))))
    assert order == ["critical2", "high4", "normal1", "normal3", "low0"]
    assert gate.active == 0


async def test_low_priority_is_not_starved():
    gate = RequestGate(capacity=1, max_queue=100, max_skips=3)
    holder = Ticket(P.NORMAL)
    await gate.acquire(holder)
    order: list[str] = []
    low = await _queue(gate, [P.LOW], order)
    highs = await _queue(gate, [P.HIGH] * 5, order)
    current = holder
    for _ in range(6):
        gate.release(current)
        await asyncio.sleep(0)
        current = next(t for task, t in low + highs if t.held)
    assert order.index("low0") == 3  # overtaken three times, then served


async def test_queue_full():
    gate = RequestGate(capacity=1, max_queue=1)
    await gate.acquire(Ticket(P.NORMAL))
    waiter = asyncio.create_task(gate.acquire(Ticket(P.NORMAL)))
    await asyncio.sleep(0)
    with pytest.raises(QueueFull):
        await gate.acquire(Ticket(P.CRITICAL))
    waiter.cancel()


async def test_cancelled_waiter_leaves_the_queue():
    gate = RequestGate(capacity=1, max_queue=10)
    holder = Ticket(P.NORMAL)
    await gate.acquire(holder)
    ticket = Ticket(P.HIGH)
    task = asyncio.create_task(gate.acquire(ticket))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert gate.waiting == 0 and not ticket.held
    gate.release(holder)
    assert gate.active == 0


async def test_cancel_racing_the_grant_still_releases():
    gate = RequestGate(capacity=1, max_queue=10)
    holder = Ticket(P.NORMAL)
    await gate.acquire(holder)
    ticket = Ticket(P.NORMAL)
    task = asyncio.create_task(gate.acquire(ticket))
    await asyncio.sleep(0)
    gate.release(holder)  # grants the slot...
    task.cancel()  # ...and the waiter is cancelled before it resumes
    with pytest.raises(asyncio.CancelledError):
        await task
    assert ticket.held and gate.active == 1
    gate.release(ticket)
    assert gate.active == 0


async def test_slot_of_already_cancelled_waiter_goes_to_the_next():
    gate = RequestGate(capacity=1, max_queue=10)
    holder = Ticket(P.NORMAL)
    await gate.acquire(holder)
    first, second = Ticket(P.NORMAL), Ticket(P.NORMAL)
    t1 = asyncio.create_task(gate.acquire(first))
    t2 = asyncio.create_task(gate.acquire(second))
    await asyncio.sleep(0)
    t1.cancel()
    gate.release(holder)  # t1's cancel handler hasn't run yet
    await t2
    assert second.held and not first.held and gate.active == 1
    with pytest.raises(asyncio.CancelledError):
        await t1


async def test_release_is_idempotent():
    gate = RequestGate(capacity=1, max_queue=10)
    ticket = Ticket(P.NORMAL)
    await gate.acquire(ticket)
    gate.release(ticket)
    gate.release(ticket)
    gate.release(Ticket(P.NORMAL))  # never held
    assert gate.active == 0
