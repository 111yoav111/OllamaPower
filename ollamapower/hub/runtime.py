"""The running hub's live state and its one snapshot.

Everything that shows the hub's state - ``GET /ollamapower/status``, the CLI
over the admin channel, the TUI from Phase 2 and later OllamaGate - reads the
same ``Hub.get_snapshot()`` dict (DCP's pattern). All of it runs on one event
loop, so plain method calls are atomic with respect to each other.

Nothing here is persisted; ``state.json`` (via ``StateStore``) is for nodes,
benchmarks and plans only.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import httpx

from .. import __version__
from ..config import Config, is_loopback_host
from ..eventlog import EventLog, NullEventLog
from ..ollama.client import OllamaStatus, human_bytes, probe
from ..scheduling.request_queue import RequestGate
from ..scheduling.task_pool import TaskPriority
from ..state import StateStore

log = logging.getLogger("ollamapower.hub")

OLLAMA_POLL_INTERVAL = 5.0
RECENT_WINDOW = 60.0  # seconds counted in "requests last minute"
MAX_RECENT = 10_000
ENGINE_LOCAL = "local-ollama"

_stream_ids = itertools.count(1)


@dataclass
class ActiveStream:
    """One request being served (or waiting for a slot)."""

    method: str
    path: str
    priority: TaskPriority
    model: str = ""
    engine: str = ENGINE_LOCAL
    id: int = field(default_factory=lambda: next(_stream_ids))
    started: float = field(default_factory=time.monotonic)
    queued_ms: int | None = None  # None while still waiting for a slot
    bytes_out: int = 0
    killed: asyncio.Event = field(default_factory=asyncio.Event)
    kill_reason: str = ""

    def kill(self, reason: str) -> None:
        self.kill_reason = reason
        self.killed.set()


@dataclass
class GenerationStats:
    """Timings Ollama reports in the final chunk of a native-API generation."""

    prompt_tokens: int = 0
    gen_tokens: int = 0
    prompt_tps: float = 0.0
    gen_tps: float = 0.0
    at: float = field(default_factory=time.time)


class Hub:
    def __init__(self, config: Config, state: StateStore | None = None, events: EventLog | None = None) -> None:
        self.config = config
        self.state = state
        self.events = events or NullEventLog()
        self.gate = RequestGate(config.hub.max_concurrency, config.hub.max_queue)
        self.started_at = time.time()
        self.streams: dict[int, ActiveStream] = {}
        self.ollama = OllamaStatus(reachable=False, error="not checked yet")
        self.last_generation: dict[str, GenerationStats] = {}
        self.requests_total = 0
        self.requests_failed = 0
        self._recent: deque[float] = deque(maxlen=MAX_RECENT)
        self.extra_warnings: list[str] = []

    # ----- streams ---------------------------------------------------------------------

    def register_stream(self, method: str, path: str, priority: TaskPriority) -> ActiveStream:
        stream = ActiveStream(method=method, path=path, priority=priority)
        self.streams[stream.id] = stream
        return stream

    def finish_stream(self, stream: ActiveStream, status: int, outcome: str) -> None:
        """Forget the stream, count it and write its metadata line (never content)."""
        if self.streams.pop(stream.id, None) is None:
            return
        self.requests_total += 1
        self._recent.append(time.monotonic())
        if status >= 500 or outcome not in ("complete", "client_disconnect"):
            self.requests_failed += 1
        self.events.request(
            method=stream.method,
            path=stream.path,
            model=stream.model,
            priority=stream.priority.name.lower(),
            engine=stream.engine,
            status=status,
            outcome=outcome,
            queue_ms=stream.queued_ms,
            duration_ms=int((time.monotonic() - stream.started) * 1000),
            bytes_out=stream.bytes_out,
        )

    def kill_all_streams(self, reason: str) -> int:
        streams = list(self.streams.values())
        for stream in streams:
            stream.kill(reason)
        return len(streams)

    def record_generation(self, model: str, stats: GenerationStats) -> None:
        if model:
            self.last_generation[model] = stats

    # ----- Ollama ----------------------------------------------------------------------

    async def poll_ollama(self, client: httpx.AsyncClient) -> None:
        """Keep ``self.ollama`` fresh; logs when Ollama goes away or comes back."""
        while True:
            try:
                status = await probe(client)
            except Exception as exc:  # noqa: BLE001 - whatever Ollama returns, keep polling
                status = OllamaStatus(reachable=False, error=type(exc).__name__)
            if status.reachable != self.ollama.reachable and self.ollama.error != "not checked yet":
                kind = "ollama_up" if status.reachable else "ollama_down"
                self.events.event(kind, url=self.config.ollama.url, error=status.error)
                (log.info if status.reachable else log.warning)(
                    "Ollama %s at %s", "is back" if status.reachable else "unreachable", self.config.ollama.url
                )
            self.ollama = status
            await asyncio.sleep(OLLAMA_POLL_INTERVAL)

    # ----- snapshot --------------------------------------------------------------------

    def warnings(self) -> list[str]:
        warnings = list(self.extra_warnings)
        if self.state is not None:
            warnings.extend(self.state.warnings)
        if not is_loopback_host(self.config.hub.host):
            warnings.append(
                f"The front door listens on {self.config.hub.host}, not loopback: "
                "anyone on that network can use Ollama."
            )
        if not self.ollama.reachable and self.ollama.error != "not checked yet":
            warnings.append(
                f"Ollama is not reachable at {self.config.ollama.url} ({self.ollama.error}); run `ollamapower doctor`."
            )
        for m in self.ollama.models:
            if m.spilled:
                warnings.append(
                    f"{m.name} is only {m.vram_percent}% in VRAM "
                    f"({human_bytes(m.size_vram)} of {human_bytes(m.size)}): the rest runs on the CPU. "
                    "Split mode (Phase 3+) can move it to a helper GPU."
                )
        return warnings

    def get_snapshot(self) -> dict[str, Any]:
        now = time.monotonic()
        while self._recent and now - self._recent[0] > RECENT_WINDOW:
            self._recent.popleft()
        o = self.ollama
        return {
            "version": __version__,
            "uptime_s": round(time.time() - self.started_at, 1),
            "frontdoor": {"url": self.config.frontdoor_url},
            "ollama": {
                "url": self.config.ollama.url,
                "reachable": o.reachable,
                "version": o.version,
                "error": o.error,
                "models": [
                    {
                        "name": m.name,
                        "size": m.size,
                        "size_vram": m.size_vram,
                        "vram_percent": m.vram_percent,
                        "spilled": m.spilled,
                        "engine": ENGINE_LOCAL,
                        "last_gen_tps": round(self.last_generation[m.name].gen_tps, 1)
                        if m.name in self.last_generation
                        else None,
                    }
                    for m in o.models
                ],
            },
            "queue": {
                "capacity": self.gate.capacity,
                "active": self.gate.active,
                "waiting": self.gate.waiting,
                "waiting_by_priority": self.gate.waiting_by_priority(),
                "max_queue": self.gate.max_queue,
            },
            "streams": [
                {
                    "id": s.id,
                    "method": s.method,
                    "path": s.path,
                    "model": s.model,
                    "priority": s.priority.name.lower(),
                    "engine": s.engine,
                    "state": "queued" if s.queued_ms is None else "running",
                    "age_s": round(now - s.started, 1),
                    "bytes_out": s.bytes_out,
                }
                for s in self.streams.values()
            ],
            "requests": {
                "total": self.requests_total,
                "failed": self.requests_failed,
                "last_minute": len(self._recent),
            },
            "generation": {
                model: {
                    "gen_tps": round(g.gen_tps, 1),
                    "prompt_tps": round(g.prompt_tps, 1),
                    "gen_tokens": g.gen_tokens,
                    "prompt_tokens": g.prompt_tokens,
                    "at": g.at,
                }
                for model, g in self.last_generation.items()
            },
            "nodes": [],  # helpers arrive in Phase 2
            "warnings": self.warnings(),
        }
