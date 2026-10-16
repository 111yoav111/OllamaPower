"""Whether this helper lends right now: the owner's pause, the schedule, and auto-yield.

The helper's owner is in charge (spec Section 10.2):

* **Pause** (``P`` / ``ollamapower agent pause``) stops lending at once.
* **Schedule** (``lending.schedule``, e.g. ``["22:00-08:00"]``, local time;
  windows may cross midnight): outside every window the helper doesn't lend.
* **Auto-yield** (``lending.auto_yield``): when the owner starts using the GPU
  - a game, a render - the helper steps back. The signal is the lent GPU
  being busier than ``yield_gpu_percent`` for ``yield_seconds`` while the
  split-mode tunnel is quiet: the hub's own work always comes with tunnel
  traffic, so a busy GPU with a quiet tunnel is somebody else. It lends again
  once the GPU has been calm for the same time. (While the hub is actively
  using the GPU this can't tell the two apart; it notices within
  ``yield_seconds`` of the hub's next pause. GPUs that don't report
  utilisation - Apple, some AMD - can't auto-yield.)

``LendingGuard.evaluate`` turns these into one state for the hub:
``lending``, ``paused`` or ``yielding``, with an owner-readable reason.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Any

from .config import LendingRules

TUNNEL_QUIET = 3.0  # seconds without tunnel traffic before GPU load counts as someone else's


def _minutes(hhmm: str) -> int:
    h, m = hhmm.strip().split(":")
    return int(h) * 60 + int(m)


def in_schedule(schedule: list[str], now: datetime.datetime) -> bool:
    """True if ``now`` (local time) is inside one of the windows; an empty schedule means always."""
    if not schedule:
        return True
    minute = now.hour * 60 + now.minute
    for window in schedule:
        start, _, end = window.partition("-")
        a, b = _minutes(start), _minutes(end)
        if a == b:
            return True  # "00:00-00:00": all day
        if a < b and a <= minute < b:
            return True
        if a > b and (minute >= a or minute < b):  # crosses midnight
            return True
    return False


@dataclass
class Decision:
    state: str  # lending · paused · yielding
    reason: str = ""


class LendingGuard:
    def __init__(self) -> None:
        self._busy_since: float | None = None
        self._calm_since: float | None = None
        self.yielding = False
        self.busy_gpu: dict[str, Any] | None = None
        self.gpu_indexes: set[int] | None = None  # from the last telemetry sample; None = not seen yet

    def _lent_utils(self, rules: LendingRules, sample: dict[str, Any]) -> list[tuple[int, int]]:
        out = []
        for g in sample.get("gpus") or []:
            if not isinstance(g, dict) or not isinstance(g.get("util"), int):
                continue
            if rules.gpus and g.get("index") not in rules.gpus:
                continue
            out.append((int(g.get("index", 0)), int(g["util"])))
        return out

    def update_yield(self, rules: LendingRules, sample: dict[str, Any], tunnel_quiet: bool, now: float) -> bool:
        """Feed one telemetry sample (``now`` = monotonic seconds); returns whether to yield."""
        self.gpu_indexes = {g["index"] for g in sample.get("gpus") or [] if isinstance(g, dict) and "index" in g}
        if not rules.auto_yield:
            self.yielding, self._busy_since, self._calm_since = False, None, None
            return False
        utils = self._lent_utils(rules, sample)
        busy = [(i, u) for i, u in utils if u >= rules.yield_gpu_percent]
        if busy and tunnel_quiet:
            self._calm_since = None
            if self._busy_since is None:
                self._busy_since = now
            if not self.yielding and now - self._busy_since >= rules.yield_seconds:
                self.yielding = True
                self.busy_gpu = {"index": busy[0][0], "util": busy[0][1]}
        else:
            self._busy_since = None
            if self.yielding:
                if self._calm_since is None:
                    self._calm_since = now
                if now - self._calm_since >= rules.yield_seconds:
                    self.yielding, self.busy_gpu, self._calm_since = False, None, None
        return self.yielding

    def evaluate(self, rules: LendingRules, paused: bool, now: datetime.datetime | None = None) -> Decision:
        if paused:
            return Decision("paused", "paused by the helper's owner")
        if not in_schedule(rules.schedule, now or datetime.datetime.now()):
            return Decision("paused", f"outside the lending schedule ({', '.join(rules.schedule)})")
        if rules.gpus and not rules.allow_cpu and self.gpu_indexes is not None:
            if not set(rules.gpus) & self.gpu_indexes:
                have = ", ".join(map(str, sorted(self.gpu_indexes))) or "none"
                lent = ", ".join(map(str, rules.gpus))
                return Decision("paused", f"lending.gpus lists GPU {lent}, but this PC's GPUs are: {have}")
        if self.yielding:
            gpu = self.busy_gpu or {}
            return Decision(
                "yielding",
                f"the helper's owner is using GPU {gpu.get('index', '?')} ({gpu.get('util', '?')} % busy)",
            )
        return Decision("lending")
