"""Small asyncio helpers shared by the hub and the agent."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any, TypeVar

T = TypeVar("T")


async def reap(*tasks: asyncio.Future[Any]) -> None:
    """Cancel ``tasks`` and wait for them to finish, ignoring how they end.

    Only *their* cancellation is swallowed: if the caller itself is being
    cancelled (meanwhile, or already - reaping on the way out of a cancelled
    task), that cancellation is re-raised, but only once every task has
    finished: raising at the first one would leave the rest running unawaited.
    (A bare ``suppress(CancelledError)`` around the await would eat it, and the
    caller would never stop - e.g. an agent that can't be shut down.)
    """
    for task in tasks:
        task.cancel()
    cancelled = False
    for task in tasks:
        try:
            await task
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                cancelled = True
        except Exception:  # noqa: BLE001 - the task's own failure doesn't matter once we stop it
            pass
    if cancelled:
        raise asyncio.CancelledError


async def wait_for(aw: Awaitable[T], timeout: float | None) -> T:
    """``asyncio.wait_for`` without Python 3.11's lost-cancellation race.

    On 3.11, if the awaited work finishes at the very moment the caller is
    cancelled, ``asyncio.wait_for`` returns the result and drops the
    cancellation, so a task being shut down carries on (an agent that never
    stops). ``asyncio.timeout`` doesn't have that problem.
    """
    async with asyncio.timeout(timeout):
        return await aw
