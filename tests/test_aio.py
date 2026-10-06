"""reap(): stopping a helper task must never swallow the caller's own cancellation."""

from __future__ import annotations

import asyncio

import pytest

from ollamapower.aio import reap


async def test_reap_cancels_and_waits():
    task = asyncio.ensure_future(asyncio.sleep(60))
    await reap(task)
    assert task.cancelled()


async def test_reap_ignores_the_tasks_own_failure():
    async def boom() -> None:
        raise RuntimeError("x")

    task = asyncio.ensure_future(boom())
    await asyncio.sleep(0)
    await reap(task)


async def test_caller_cancelled_while_reaping_stays_cancelled():
    # A task that takes a while to stop (like a watcher waiting for a process).
    async def slow_to_stop() -> None:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            await asyncio.sleep(0.5)
            raise

    inner = asyncio.ensure_future(slow_to_stop())
    await asyncio.sleep(0)

    async def caller() -> str:
        await reap(inner)
        return "kept running"  # the bug: the caller's cancellation was eaten

    outer = asyncio.ensure_future(caller())
    await asyncio.sleep(0.05)
    outer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await outer


async def test_reaping_while_being_cancelled_still_waits_for_every_task():
    async def slow_to_stop(seconds: float) -> None:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            await asyncio.sleep(seconds)  # cleanup, like stopping a process
            raise

    tasks = [asyncio.ensure_future(slow_to_stop(0.02 * i)) for i in range(1, 4)]
    await asyncio.sleep(0)

    async def shutting_down() -> None:
        try:
            await asyncio.sleep(60)
        finally:  # cleanup on the way out of a cancelled task
            await reap(*tasks)

    outer = asyncio.ensure_future(shutting_down())
    await asyncio.sleep(0)
    outer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await outer
    assert all(t.done() for t in tasks)  # not just the first one: the rest were left running


async def test_wait_for_keeps_a_cancellation_that_races_the_result():
    from ollamapower.aio import wait_for

    loop = asyncio.get_running_loop()
    fut = loop.create_future()

    async def caller() -> str:
        await wait_for(fut, 10)
        return "kept running"  # Python 3.11's asyncio.wait_for returns here: the cancellation is lost

    task = asyncio.ensure_future(caller())
    await asyncio.sleep(0)
    fut.set_result(1)  # the work finishes...
    task.cancel()  # ...in the same instant the caller is told to stop
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_wait_for_times_out():
    from ollamapower.aio import wait_for

    with pytest.raises(TimeoutError):
        await wait_for(asyncio.sleep(5), 0.01)
