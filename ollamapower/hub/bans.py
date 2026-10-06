"""Temporary bans for clients that keep failing to log in (idea from OllamaGate's bans).

``failures`` failed attempts from one IP within ``window`` seconds ban it for
``ban_seconds``; every repeat offence doubles the ban, up to ``max_seconds``.
A successful login forgets that IP's recent failures (not its ban history), so
a helper that sometimes hangs up mid-handshake never piles up towards a ban.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class _Record:
    failures: deque[float] = field(default_factory=deque)
    strikes: int = 0
    banned_until: float = 0.0


class BanList:
    MAX_TRACKED = 10_000

    def __init__(self, failures: int, window: float, ban_seconds: float, max_seconds: float) -> None:
        self.failures = failures
        self.window = window
        self.ban_seconds = ban_seconds
        self.max_seconds = max_seconds
        self._records: dict[str, _Record] = {}

    def banned(self, ip: str, now: float | None = None) -> bool:
        record = self._records.get(ip)
        return record is not None and record.banned_until > (time.monotonic() if now is None else now)

    def fail(self, ip: str, now: float | None = None) -> float | None:
        """Count a failure; returns the ban length in seconds if this failure triggered a ban."""
        now = time.monotonic() if now is None else now
        if len(self._records) >= self.MAX_TRACKED and ip not in self._records:
            self._forget_stale(now)
            if len(self._records) >= self.MAX_TRACKED:
                return None  # every slot holds an active ban: keep them all
        record = self._records.setdefault(ip, _Record())
        record.failures.append(now)
        while record.failures and now - record.failures[0] > self.window:
            record.failures.popleft()
        if len(record.failures) < self.failures:
            return None
        record.failures.clear()
        seconds = min(self.ban_seconds * 2**record.strikes, self.max_seconds)
        record.strikes += 1
        record.banned_until = now + seconds
        return seconds

    def succeeded(self, ip: str, now: float | None = None) -> None:
        """A login from ``ip`` succeeded: forget its recent failures (an active ban stays)."""
        record = self._records.get(ip)
        if record is None:
            return
        record.failures.clear()
        if record.strikes == 0 and record.banned_until <= (time.monotonic() if now is None else now):
            del self._records[ip]

    def active(self, now: float | None = None) -> dict[str, float]:
        """IP -> seconds left, for status views."""
        now = time.monotonic() if now is None else now
        return {ip: r.banned_until - now for ip, r in self._records.items() if r.banned_until > now}

    def lift(self, ip: str) -> bool:
        return self._records.pop(ip, None) is not None

    def _forget_stale(self, now: float) -> None:
        """Make room: drop expired failures, then records that are neither banned nor recent.

        Active bans are never evicted, so flooding the table from many addresses
        can't un-ban anyone.
        """
        for record in self._records.values():
            while record.failures and now - record.failures[0] > self.window:
                record.failures.popleft()
        for ip in [ip for ip, r in self._records.items() if r.banned_until <= now and not r.failures]:
            del self._records[ip]
        if len(self._records) >= self.MAX_TRACKED:  # still full: drop the oldest records that aren't banned
            unbanned = [ip for ip, r in self._records.items() if r.banned_until <= now]
            for ip in unbanned[: self.MAX_TRACKED // 10]:
                del self._records[ip]
