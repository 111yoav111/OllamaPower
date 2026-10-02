"""Small asyncio helpers shared by the hub and the agent."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any, TypeVar

T = TypeVar("T")


async def reap(*tasks: asyncio.Future[Any]) -> None:
    """Cancel ``tasks`` and wait for them to finish, ignoring how they end.

    Only *their* cancellation is swallowed: if the caller itself is being
    cancelled meanwhile, that cancellation is re-raised. (A bare
    ``suppress(CancelledError)`` around the await would eat it, and the caller
    would never stop - e.g. an agent that can't be shut down.)
    """
    for task in tasks:
        task.cancel()
    for task in tasks:
        try:
            await task
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise
        except Exception:  # noqa: BLE001 - the task's own failure doesn't matter once we stop it
            pass


async def wait_for(aw: Awaitable[T], timeout: float | None) -> T:
    """``asyncio.wait_for`` without Python 3.11's lost-cancellation race.

    On 3.11, if the awaited work finishes at the very moment the caller is
    cancelled, ``asyncio.wait_for`` returns the result and drops the
    cancellation, so a task being shut down carries on (an agent that never
    stops). ``asyncio.timeout`` doesn't have that problem.
    """
    async with asyncio.timeout(timeout):
        return await aw
