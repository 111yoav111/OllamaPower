"""Split mode: one model across the hub's GPU and the helpers' GPUs (llama.cpp RPC).

When the plan book (``hub/plans.py``) routes a model here, the first request
starts a split session with the given ``SplitSpec`` (context, helpers):

1. Find the model's GGUF in Ollama's store (no second copy) and read its layer sizes.
2. Make sure the hub has the pinned llama.cpp (downloaded once, hash-checked).
3. If the model fits on the hub's GPU alone, stop here: Ollama serves it, faster.
4. Ask Ollama to unload its models (``keep_alive: 0``) and wait until ``/api/ps``
   is empty, so the GPU memory is really free - Ollama and llama.cpp must not
   fight over VRAM.
5. Ask eligible helpers to start ``rpc-server`` (``rpc_start``); each answers
   with how much memory its owner lends right now.
6. Plan the layers (``planner.py``) and start ``llama-server`` on
   ``127.0.0.1`` with ``--rpc``/``--device``/``--tensor-split`` set to exactly
   that plan; wait for ``/health``. llama-server allows every web origin
   (CORS), so it gets a random API key (passed in its environment, not on the
   command line where other users could read it): a web page that finds its
   port still can't use it. Only the front door knows the key.

The session then serves requests until it has been idle for
``split.keep_alive`` seconds, a helper leaves or fails, or the owner removes
the plan; then llama-server stops, the helpers are released and Ollama gets
the GPU back on its next request.

If anything fails, the request is served by Ollama as before (never worse
than without OllamaPower) and the plan book marks the model degraded; the
reason is shown as a warning and logged with the tail of llama-server's
output.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import secrets
import socket
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import httpx

from ...aio import reap, wait_for
from ...childproc import spawn
from ...core.tunnel import TunnelError, TunnelIdentity, check_cert_pem
from ...llamacpp import binaries as llamacpp_binaries
from ...llamacpp.binaries import LlamaCpp, llamacpp_env, preferred
from ...llamacpp.install import Pins
from ...llamacpp.manager import LlamaCppUnavailable, ensure
from ...ollama.blobs import ModelFiles, ModelUnavailable, resolve
from ...ollama.client import PROBE_HEADERS, human_bytes, parse_ps
from ...textsafe import safe_text
from ..planner import MIB, Device, PlanError, SplitPlan, fits_on_hub, plan_split
from ..plans import AWAY, PERMANENT, TRANSIENT, SplitSpec, layout_problem
from ..tunnel import BULK_BYTES, HubRelay, TunnelClient, measure

if TYPE_CHECKING:
    from ..nodes import Node
    from ..runtime import Hub

log = logging.getLogger("ollamapower.split")

RPC_READY_TIMEOUT = 60.0
OLLAMA_RELEASE_TIMEOUT = 30.0
STOP_GRACE = 10.0
IDLE_CHECK = 5.0
BENCH_TIMEOUT = 120.0  # seconds a whole tunnel test may take
DRAIN_TIMEOUT = 30.0  # seconds running answers get to finish when a helper leaves on purpose
WARNING_TTL = 600.0
ENGINE_SPLIT = "split"


class SplitUnavailable(Exception):
    """Split mode can't serve this request now; Ollama will (the message says why).

    ``kind`` says what the plan book does about it: ``TRANSIENT`` (try again after a growing
    pause), ``PERMANENT`` (not again until the model, its settings or llama.cpp change) or
    ``AWAY`` (not a failure at all).
    """

    kind = TRANSIENT

    def __init__(self, message: str, kind: str | None = None) -> None:
        super().__init__(message)
        if kind is not None:
            self.kind = kind


class HelperAway(SplitUnavailable):
    """A helper stopped lending on purpose (pause, schedule, auto-yield): not a failure of the plan."""

    kind = AWAY


class PlanStale(SplitUnavailable):
    """The setup changed since the model's benchmark: not a failure, the model is benchmarked again."""

    kind = AWAY


# Why llama-server exited, from its last lines. Out of memory first: it also says "failed to load
# model", and memory may be free next time. PERMANENT_SIGNS: this build can never load this model
# (or run here at all), so starting it again changes nothing.
OOM_SIGNS = ("out of memory", "failed to allocate", "unable to allocate", "outofdevicememory", "cudamalloc failed")
PERMANENT_SIGNS = (
    "unknown model architecture",
    "unsupported model",
    "invalid magic",
    "failed to read magic",
    "unknown (magic, version)",
    "error while loading shared libraries",
    "glibc_",
    "illegal instruction",
)
PERMANENT_EXIT_CODES = (126, 127, 132, -4)  # not executable, not found, SIGILL (built for another CPU)


def exit_kind(returncode: int | None, output: list[str]) -> str:
    """Is a llama-server that exited worth starting again later (``TRANSIENT``) or not (``PERMANENT``)?"""
    text = "\n".join(output[-20:]).lower()
    if any(sign in text for sign in OOM_SIGNS):
        return TRANSIENT
    if returncode in PERMANENT_EXIT_CODES or any(sign in text for sign in PERMANENT_SIGNS):
        return PERMANENT
    return TRANSIENT


@dataclass
class HelperLease:
    node: Node
    session_id: str
    mode: str = "secure"
    identity: TunnelIdentity | None = field(default=None, repr=False)
    endpoint: dict[str, Any] = field(default_factory=dict)
    relay: HubRelay | None = None

    @property
    def address(self) -> str:
        """What llama-server's ``--rpc`` gets: the hub's local relay (secure) or the helper's relay (direct)."""
        if self.relay is not None:
            return f"127.0.0.1:{self.relay.port}"
        host = self.node.ip
        return f"[{host}]:{self.endpoint['port']}" if ":" in host else f"{host}:{self.endpoint['port']}"


@dataclass
class SplitSession:
    model: str
    ctx: int
    spec: SplitSpec = field(default_factory=lambda: SplitSpec(ctx=0))
    state: str = "starting"  # starting · ready · stopping · failed
    files: ModelFiles | None = None
    plan: SplitPlan | None = None
    leases: list[HelperLease] = field(default_factory=list)
    process: asyncio.subprocess.Process | None = None
    port: int = 0
    client: httpx.AsyncClient | None = None
    api_key: str = field(default_factory=lambda: secrets.token_urlsafe(32), repr=False)
    started: float = field(default_factory=time.time)
    ready_at: float | None = None
    last_used: float = field(default_factory=time.monotonic)
    active: int = 0
    idle: asyncio.Event = field(default_factory=asyncio.Event)
    output: deque[str] = field(default_factory=lambda: deque(maxlen=60))
    error: str = ""
    draining: bool = False  # a helper stopped lending: finish the current answers, then stop
    proven: bool = False  # an answer came through: the plan book's failure count went back to zero
    # llama-server runs with one slot (-np 1): requests queue here, in the hub, so time spent
    # waiting for the slot never counts as llama-server stalling.
    slot: asyncio.Semaphore = field(default_factory=lambda: asyncio.Semaphore(1))

    def transfer(self) -> dict[str, Any] | None:
        """Model weights sent to the helpers so far (secure mode counts them), for a progress bar."""
        relays = [x.relay for x in self.leases if x.relay is not None]
        if not relays or self.plan is None or self.files is None or self.files.gguf is None:
            return None
        expected = sum(max(0, p.bytes - p.cache) for p in self.plan.helpers)
        return {"sent": sum(r.bytes_out for r in relays), "expected": expected, "loading": self.state == "starting"}

    def layout(self) -> dict[str, Any]:
        """What a benchmark key records of this session's setup (see ``plans.layout_problem``)."""
        plan = self.plan
        return {
            "helpers": [
                {
                    "id": x.node.node_id,
                    "name": x.node.name,
                    "tunnel": x.mode,
                    "device": x.endpoint.get("device", ""),
                    "lent_gib": round(int(x.endpoint.get("free_mib", 0)) / 1024),
                    "ram_gib": round(
                        sum(int(d["free_mib"]) for d in x.endpoint.get("devices") or [] if d["device"] == "CPU") / 1024
                    ),
                }
                for x in self.leases
            ],
            "placement": [
                {"node": p.device.node_id, "device": p.device.device, "layers": p.layers}
                for p in (plan.placements if plan else [])
            ],
            "cpu_layers": plan.cpu_layers if plan else 0,
        }

    def alive(self) -> bool:
        return self.state == "ready" and self.process is not None and self.process.returncode is None

    def snapshot(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "state": self.state,
            "ctx": self.ctx,
            "port": self.port,
            "since": self.started,
            "ready_at": self.ready_at,
            "active": self.active,
            "idle_s": round(time.monotonic() - self.last_used, 1) if not self.active else 0.0,
            "size": self.files.gguf.file_size if self.files and self.files.gguf else 0,
            "plan": self.plan.summary() if self.plan else None,
            "helpers": [{"node": x.node.node_id, "name": x.node.name, "tunnel": x.mode} for x in self.leases],
            "transfer": self.transfer(),
            "error": self.error,
        }


def _count(value: Any) -> bool:
    """A positive int (helpers' answers are untrusted)."""
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


SHARED_RAM_SHARE = 0.75  # at most this much of the RAM counts as an integrated GPU's memory (like Metal's limit)
SHARED_RAM_KEEP = 2 * 1024 * MIB  # and RAM the system keeps for itself and the CPU layers


def _shared_ram(free: bool) -> int:
    import psutil

    vm = psutil.virtual_memory()
    return max(0, int(vm.available) - SHARED_RAM_KEEP) if free else int(vm.total * SHARED_RAM_SHARE)


HUB_RAM_KEEP = 3 * 1024 * MIB  # RAM the hub leaves to the system, the front door and the owner's programs


def _endpoint_devices(endpoint: dict[str, Any]) -> list[dict[str, Any]]:
    """The devices one helper's rpc-server exposes, in its own order (a GPU, then its RAM); [] = just ``device``."""
    raw = endpoint.get("devices")
    if not isinstance(raw, list):
        return []
    out = []
    for d in raw[:4]:
        if isinstance(d, dict) and _count(d.get("free_mib")) and isinstance(d.get("device"), str):
            out.append({"device": safe_text(d["device"], 40), "free_mib": min(d["free_mib"], 1 << 24)})
    return out


def helper_devices(node: Node, endpoint: dict[str, Any]) -> list[Device]:
    """Every device a helper lent for one session, in its rpc-server's order."""
    entries = endpoint.get("devices") or [endpoint]
    return [helper_device(node, e) for e in entries]


def helper_device(node: Node, endpoint: dict[str, Any]) -> Device:
    """What a helper lent for one session, as the planner sees it (its RAM when it lent its CPU)."""
    name = safe_text(str(endpoint.get("device", "")), 40)
    rtt_ms = float(node.link.get("rtt_ms") or 0)
    ram = name.upper() == "CPU"
    return Device(
        f"{node.name} {name}",
        int(endpoint.get("free_mib", 0)) * MIB,
        node_id=node.node_id,
        device=name,
        ram=ram,
        bandwidth=float(node.capabilities.get("ram_bandwidth") or 0) if ram else 0.0,
        rtt=max(0.0, min(rtt_ms, 1000.0)) / 1000,
    )


def _hub_device(d: llamacpp_binaries.Device, free: bool) -> Device:
    """A hub GPU for the planner; an integrated GPU counts no more than the RAM it shares."""
    size = (d.free_mib if free else d.total_mib) * MIB
    if d.integrated:
        size = min(size, _shared_ram(free))
    return Device(f"hub {d.name}", size, device=d.name)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class SplitEngine:
    def __init__(self, hub: Hub) -> None:
        self.hub = hub
        self.config = hub.config
        self.http: httpx.AsyncClient | None = None  # to Ollama; set by the front door
        self.session: SplitSession | None = None
        self.binaries: LlamaCpp | None = None
        self.pins: Pins | None = None
        self._lock = asyncio.Lock()
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._warnings: deque[tuple[float, str]] = deque(maxlen=20)
        self._tasks: set[asyncio.Task[Any]] = set()
        self._acquire_tasks: set[asyncio.Task[Any]] = set()
        self._acquiring = 0  # _acquire calls in progress (waiting for the lock or starting)
        self._pending_nodes: dict[str, Node] = {}  # rpc_start session id -> helper asked
        # model -> (kind, time.monotonic()) of its last failed start, for benchmarks to tell why
        self.last_failure: dict[str, tuple[str, float]] = {}

    def holds(self, model: str) -> bool:
        """Is a session for ``model`` starting or running (holding, or about to hold, the GPU)?"""
        from ..runtime import model_key

        s = self.session
        return s is not None and s.model == model_key(model) and s.state in ("starting", "ready")

    def is_running(self, model: str) -> bool:
        from ..runtime import model_key

        return self.session is not None and self.session.model == model_key(model) and self.session.alive()

    # ----- acquire / release -------------------------------------------------------------

    async def acquire(self, model: str, spec: SplitSpec) -> SplitSession:
        """A ready session for ``model`` started with ``spec`` (started if needed); pair with ``release``."""
        task = asyncio.ensure_future(self._acquire(model, spec))
        self._acquire_tasks.add(task)
        task.add_done_callback(self._acquire_tasks.discard)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # The client left. If others wait too, let a start in progress finish for them;
            # if nobody does, abandon it - a start nobody waits for would keep the GPU from Ollama.
            def done(t: asyncio.Task[SplitSession]) -> None:
                if not t.cancelled() and t.exception() is None:
                    self.release(t.result())

            if not task.done() and self._acquiring <= 1:
                task.cancel()
            task.add_done_callback(done)
            raise

    async def _acquire(self, model: str, spec: SplitSpec) -> SplitSession:
        from ..runtime import model_key

        key = model_key(model)
        self._acquiring += 1
        try:
            return await self._acquire_locked(key, spec)
        finally:
            self._acquiring -= 1

    async def _acquire_locked(self, key: str, spec: SplitSpec) -> SplitSession:
        async with self._lock:
            current = self.session
            reusable = current is not None and current.alive() and not current.draining and not current.error
            if reusable and current.model == key and current.spec == spec:
                current.active += 1
                current.idle.clear()
                return current
            if current is not None:
                if current.active:  # let running answers finish first (another model, or a helper leaving)
                    with contextlib.suppress(TimeoutError):
                        await wait_for(current.idle.wait(), DRAIN_TIMEOUT)
                await self._stop(current, f"switching to {key}" if current.model != key else "restarting")
            session = SplitSession(model=key, ctx=spec.ctx, spec=spec)
            self.session = session
            try:
                await wait_for(self._start(session), self.config.split.load_timeout)
            except BaseException as exc:
                reason = (
                    str(exc)
                    if isinstance(exc, SplitUnavailable)
                    else "timed out loading the model"
                    if isinstance(exc, TimeoutError)
                    else f"internal error ({type(exc).__name__})"
                    if not isinstance(exc, asyncio.CancelledError)
                    else "cancelled"
                )
                if not isinstance(exc, (SplitUnavailable, TimeoutError, asyncio.CancelledError)):
                    log.exception("split start failed")
                session.error = reason
                tail = list(session.output)
                cancelled = isinstance(exc, asyncio.CancelledError)
                await self._stop(session, reason, failed=not cancelled)
                if cancelled:
                    raise  # abandoned (nobody waits, or the hub stops): not a failure of the plan
                # A timeout or an internal error may well not happen again: transient.
                kind = exc.kind if isinstance(exc, SplitUnavailable) else TRANSIENT
                self.last_failure[key] = (kind, time.monotonic())
                if kind != AWAY:
                    self.hub.plans.fail(key, reason, tail, kind)
                raise SplitUnavailable(reason, kind) from None
            session.active += 1
            return session

    def note_success(self, session: SplitSession) -> None:
        """An answer came through ``session``: split mode works again (once per session, so it costs nothing)."""
        if not session.proven:
            session.proven = True
            self.hub.plans.succeeded(session.model)

    def release(self, session: SplitSession) -> None:
        session.active = max(0, session.active - 1)
        session.last_used = time.monotonic()
        if session.active == 0:
            session.idle.set()

    # ----- start -------------------------------------------------------------------------

    async def _start(self, s: SplitSession) -> None:
        if self.http is None:
            raise SplitUnavailable("the front door is not running")
        self.hub.event("split_starting", model=s.model, ctx=s.ctx)
        try:
            s.files = await resolve(self.http, s.model)
        except ModelUnavailable as exc:
            raise SplitUnavailable(str(exc), PERMANENT) from None
        info = s.files.gguf
        assert info is not None
        binaries = await self._ensure_llamacpp()
        if fits_on_hub(info, s.ctx, self.hub_devices(), self.config.planner.memory_margin):
            raise SplitUnavailable(
                f"{s.model} fits on the hub's GPU alone, so Ollama runs it faster without helpers", PERMANENT
            )

        helpers, reasons = self.candidates()
        helpers = [n for n in helpers if n.node_id not in s.spec.exclude]
        if s.spec.helpers is not None:
            # A benchmarked plan: exactly those helpers, in that order, or not at all.
            by_id = {n.node_id: n for n in helpers}
            missing = [h for h in s.spec.helpers if h not in by_id]
            if missing:
                # Paused, outside its schedule or gone since routing: like a helper leaving, not a failure.
                raise HelperAway(f"a helper of the plan is not available ({', '.join(m[:8] for m in missing)})")
            helpers = [by_id[h] for h in s.spec.helpers]
        if not helpers:
            raise HelperAway("no helper can take part" + (f" ({'; '.join(reasons)})" if reasons else ""))

        # Lease the helpers first: if none agrees, Ollama keeps its models loaded.
        await self._lease(s, helpers)
        if not s.leases:
            raise SplitUnavailable("no helper agreed to lend memory")
        await self._release_ollama()
        # Measure free memory now, right before planning and launching (Ollama just let go).
        await asyncio.to_thread(binaries.probe)
        hub_devices = [_hub_device(d, free=True) for d in preferred(binaries.devices)]
        by_node = {x.node.node_id: x for x in s.leases}
        devices = [d for x in s.leases for d in helper_devices(x.node, x.endpoint)]
        try:
            s.plan = plan_split(
                s.model,
                info,
                s.ctx,
                hub_devices,
                devices,
                self.config.planner.memory_margin,
                hub_ram=self.hub_ram(free=True),
            )
        except PlanError as exc:
            raise SplitUnavailable(str(exc)) from None
        # Helpers that got no layers are released now; the rest in llama.cpp's device order (= --rpc order).
        # One lease per helper, even when its GPU and its RAM both got layers.
        used: list[HelperLease] = []
        for p in s.plan.helpers:
            lease = by_node[p.device.node_id] if p.device.node_id else None
            if lease is not None and all(lease is not u for u in used):
                used.append(lease)
        for lease in s.leases:
            if lease not in used:
                await self._send_stop(lease)
        s.leases = used
        if s.spec.layout is not None:
            problem = layout_problem(s.spec.layout, s.layout())
            if problem:
                # Not what was benchmarked (another device, less memory, other layers): it may be slower.
                self.hub.plans.stale(s.model, problem)
                raise PlanStale(f"the benchmark no longer applies: {problem}")
        for lease in s.leases:
            lease.node.split_model = s.model
        await self._launch(s, binaries)
        away = [x.node.name for x in s.leases if x.node.status != "lending"]
        if away:
            # A helper's owner paused it (or its schedule ended) while we were loading.
            raise HelperAway(f"{', '.join(away)} stopped lending while split mode was starting")
        s.state, s.ready_at, s.last_used = "ready", time.time(), time.monotonic()
        self.hub.event(
            "split_ready",
            model=s.model,
            helpers=[x.node.name for x in s.leases],
            cpu_layers=s.plan.cpu_layers,
            layers=s.plan.units,
            load_s=round(time.time() - s.started, 1),
        )

    async def _ensure_llamacpp(self) -> LlamaCpp:
        if self.binaries is not None:
            return self.binaries
        vendors = [g.get("vendor", "") for g in self.hub.local_static.get("gpus", []) if isinstance(g, dict)]
        self.pins = Pins.load()
        try:
            self.binaries = await asyncio.to_thread(
                ensure,
                self.config.config_dir,
                self.config.llamacpp.backend,
                self.config.llamacpp.path,
                [v for v in vendors if v],
                None,
                self.pins,
            )
        except LlamaCppUnavailable as exc:
            raise SplitUnavailable(f"llama.cpp on the hub: {exc}", PERMANENT if exc.permanent else TRANSIENT) from None
        if self.binaries.llama_server is None:
            raise SplitUnavailable("llama-server is missing from the hub's llama.cpp", PERMANENT)
        return self.binaries

    def hub_ram(self, free: bool) -> Device:
        """The hub's own RAM as a place for layers: what is available (``free``) or installed, minus what
        the system and Ollama's own process keep, at the bandwidth the hub measured."""
        import psutil

        vm = psutil.virtual_memory()
        size = int(vm.available if free else vm.total) - HUB_RAM_KEEP
        return Device(
            "hub RAM",
            max(0, size),
            device="CPU",
            ram=True,
            bandwidth=float(self.hub.local_static.get("ram_bandwidth") or 0),
        )

    def hub_devices(self) -> list[Device]:
        """The hub's GPUs at their total memory: what "fits on the hub" is judged against, everywhere.

        llama.cpp's own device list once it is installed (that is what llama-server
        will use), the hub's hardware facts before that.
        """
        if self.binaries is not None and self.binaries.devices:
            return [_hub_device(d, free=False) for d in preferred(self.binaries.devices)]
        return [
            Device(f"hub {g.get('name', 'GPU')}", int(g.get("vram_total") or 0))
            for g in self.hub.local_static.get("gpus", [])
            if isinstance(g, dict) and int(g.get("vram_total") or 0) > 0
        ]

    def candidates(self) -> tuple[list[Node], list[str]]:
        """Helpers that may take part now (best first), and why the others can't."""
        pins = self.pins or self.hub.plans.pins  # cached: this runs on every routed request
        ok, reasons = [], []
        for node in self.hub.registry.connected:
            lc = node.capabilities.get("llama_cpp") or {}
            if node.status != "lending":
                reasons.append(f"{node.name} is {node.status}")
            elif lc.get("status") != "ready":
                reasons.append(f"{node.name}: llama.cpp {lc.get('status') or 'unknown'} {lc.get('error', '')}".strip())
            elif lc.get("build") != pins.build:
                reasons.append(f"{node.name} has llama.cpp build {lc.get('build')}, needs {pins.build}")
            else:  # Wi-Fi too: a split costs a few round trips per token; only the first load is slow
                ok.append(node)

        def preference(n: Node) -> tuple[int, float, float]:
            # A GPU first (fastest memory), then the fastest RAM, then the better link.
            link = float(n.link.get("down_mbps") or n.link.get("speed_mbps") or 0)
            return (n.lendable_vram(), float(n.capabilities.get("ram_bandwidth") or 0), link)

        ok.sort(key=preference, reverse=True)
        return ok, reasons

    async def _release_ollama(self) -> None:
        """Unload everything from Ollama and wait until it has really let go of the GPU."""
        assert self.http is not None
        deadline = time.monotonic() + OLLAMA_RELEASE_TIMEOUT
        asked: set[str] = set()
        while time.monotonic() < deadline:
            try:
                r = await self.http.get("/api/ps", headers=PROBE_HEADERS, timeout=5.0)
                loaded = parse_ps(r.json())
            except (httpx.HTTPError, ValueError) as exc:
                raise SplitUnavailable(f"Ollama is not reachable ({type(exc).__name__})") from None
            if not loaded:
                return
            for m in loaded:
                if m.name in asked:
                    continue
                asked.add(m.name)
                with contextlib.suppress(httpx.HTTPError):
                    await self.http.post(
                        "/api/generate", json={"model": m.name, "keep_alive": 0}, headers=PROBE_HEADERS, timeout=30.0
                    )
            await asyncio.sleep(0.5)
        raise SplitUnavailable("Ollama did not unload its models in time")

    async def _lease(self, s: SplitSession, helpers: list[Node]) -> None:
        """Ask ``helpers`` to start rpc-server; ``s.leases`` ends up holding the ones that are ready.

        The leases are on the session from the first ``rpc_start`` on, so if the start is
        cancelled or times out meanwhile, stopping the session still releases every helper.
        """
        leases = []
        for n in helpers:
            mode = n.tunnel_effective
            identity = TunnelIdentity.generate("hub") if mode == "secure" else None
            leases.append(HelperLease(node=n, session_id=secrets.token_hex(12), mode=mode, identity=identity))
        s.leases = leases
        loop = asyncio.get_running_loop()
        try:
            for lease in leases:
                self._pending[lease.session_id] = loop.create_future()
                self._pending_nodes[lease.session_id] = lease.node
                assert lease.node.channel is not None
                try:
                    await lease.node.channel.send(
                        "rpc_start",
                        session_id=lease.session_id,
                        device="",
                        vram_cap_mb=0,
                        cache=True,
                        ram=True,  # a helper whose owner allows it lends its RAM beside its GPU
                        tunnel={"mode": lease.mode, **({"cert": lease.identity.cert_pem} if lease.identity else {})},
                        model=s.model,
                    )
                except Exception:  # noqa: BLE001 - that helper just left
                    self._pending.pop(lease.session_id).cancel()
            waiting = {x.session_id: x for x in leases if x.session_id in self._pending}
            if waiting:
                await asyncio.wait([self._pending[k] for k in waiting], timeout=RPC_READY_TIMEOUT)
            ready = []
            for session_id, lease in waiting.items():
                fut = self._pending.pop(session_id)
                answer = fut.result() if fut.done() and not fut.cancelled() else None
                if answer is not None and answer.get("type") == "rpc_ready":
                    endpoint = answer.get("endpoint") or {}
                    port, free = endpoint.get("port"), endpoint.get("free_mib")
                    if _count(port) and port < 65536 and _count(free) and endpoint.get("mode") == lease.mode:
                        lease.endpoint = {
                            "port": port,
                            "free_mib": min(free, 1 << 24),
                            "device": safe_text(str(endpoint.get("device", "")), 40),
                            "devices": _endpoint_devices(endpoint),
                        }
                        try:
                            if lease.identity is not None:
                                lease.endpoint["cert"] = check_cert_pem(endpoint.get("cert"))
                                client = TunnelClient(lease.node.ip, port, lease.identity, lease.endpoint["cert"])
                                lease.relay = HubRelay(client)
                                await lease.relay.start()
                            ready.append(lease)
                            continue
                        except TunnelError as exc:
                            self._warnings.append((time.monotonic(), f"{lease.node.name}: secure tunnel: {exc}"))
                elif answer is not None:
                    err = safe_text(str(answer.get("error", "")), 300)
                    self._warnings.append((time.monotonic(), f"{lease.node.name} refused split mode: {err}"))
                    continue
                fut.cancel()
                await self._send_stop(lease)
            s.leases = ready
        finally:
            for lease in leases:
                fut = self._pending.pop(lease.session_id, None)
                if fut is not None:
                    fut.cancel()
                self._pending_nodes.pop(lease.session_id, None)

    async def _launch(self, s: SplitSession, binaries: LlamaCpp) -> None:
        assert s.plan is not None and s.files is not None and binaries.llama_server is not None
        plan = s.plan
        # llama.cpp numbers RPC devices across servers in --rpc order, each server's devices in its own order.
        rpc_names: dict[tuple[str, str], str] = {}
        for lease in s.leases:
            for d in helper_devices(lease.node, lease.endpoint):
                rpc_names[(lease.node.node_id, d.device)] = f"RPC{len(rpc_names)}"
        device_names = [
            rpc_names[(p.device.node_id, p.device.device)] if p.device.node_id else p.device.device
            for p in plan.placements
        ]
        s.port = _free_port()
        args = [
            str(binaries.llama_server),
            "-m",
            str(s.files.path),
            "--host",
            "127.0.0.1",
            "--port",
            str(s.port),
            "--alias",
            s.model,
            "-c",
            str(s.ctx),
            "-np",
            "1",
            "--jinja",
            "--no-ui",
            "--fit",
            "off",
            "-fa",
            "on",  # flash attention keeps compute buffers small and predictable (the planner assumes it)
            "--rpc",
            ",".join(x.address for x in s.leases),
            "--device",
            ",".join(device_names),
            "--tensor-split",
            plan.tensor_split(),
            "-ngl",
            str(plan.gpu_layers),
        ]
        s.process = await spawn(
            *args,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(binaries.llama_server.parent),
            env={**llamacpp_env(), "LLAMA_API_KEY": s.api_key},
        )
        drained = self._spawn(self._drain(s))
        s.client = httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{s.port}",
            headers={"authorization": f"Bearer {s.api_key}"},
            timeout=httpx.Timeout(connect=5.0, read=self.config.split.stall_timeout, write=60.0, pool=30.0),
            trust_env=False,
            follow_redirects=False,
        )
        while True:
            if s.process.returncode is not None:
                with contextlib.suppress(TimeoutError):  # its last words tell why (and whether to try again)
                    await wait_for(asyncio.shield(drained), 2.0)
                last = s.output[-1] if s.output else ""
                raise SplitUnavailable(
                    f"llama-server exited with code {s.process.returncode}: {last}",
                    exit_kind(s.process.returncode, list(s.output)),
                )
            with contextlib.suppress(httpx.HTTPError):
                r = await s.client.get("/health", timeout=2.0)
                if r.status_code == 200:
                    return
            await asyncio.sleep(0.5)

    async def _drain(self, s: SplitSession) -> None:
        proc = s.process
        assert proc is not None and proc.stdout is not None
        while line := await proc.stdout.readline():
            s.output.append(safe_text(line.decode("utf-8", "replace").rstrip(), 300))

    def _spawn(self, coro: Any) -> asyncio.Task[Any]:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    # ----- tunnel test -------------------------------------------------------------------

    async def bench(self, node: Node, mode: str, bulk_bytes: int = BULK_BYTES) -> dict[str, Any]:
        """Time the real tunnel path to ``node`` in ``mode`` (the helper relays to an echo server)."""
        channel = node.channel
        if channel is None or channel.closed:
            raise SplitUnavailable(f"{node.name} is not connected")
        lease = HelperLease(node=node, session_id=secrets.token_hex(12), mode=mode)
        if mode == "secure":
            lease.identity = TunnelIdentity.generate("hub")
        fut: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[lease.session_id] = fut
        self._pending_nodes[lease.session_id] = node
        try:
            await channel.send(
                "rpc_start",
                session_id=lease.session_id,
                device="",
                vram_cap_mb=0,
                cache=False,
                tunnel={"mode": mode, **({"cert": lease.identity.cert_pem} if lease.identity else {})},
                purpose="bench",
            )
            try:
                reply = await wait_for(fut, RPC_READY_TIMEOUT)
            except TimeoutError:
                raise SplitUnavailable(f"{node.name} did not answer the tunnel test") from None
            finally:
                self._pending.pop(lease.session_id, None)
                self._pending_nodes.pop(lease.session_id, None)
            if reply["type"] != "rpc_ready":
                raise SplitUnavailable(f"{node.name}: {safe_text(str(reply.get('error', '')), 300)}")
            endpoint = reply.get("endpoint") or {}
            port = endpoint.get("port")
            if not _count(port) or port >= 65536 or endpoint.get("mode") != mode:
                raise SplitUnavailable(f"{node.name} sent an unusable tunnel endpoint")
            try:
                cert = check_cert_pem(endpoint.get("cert")) if lease.identity else ""
                client = TunnelClient(node.ip, port, lease.identity, cert)
                # A helper that drops off the network mid-test sends no FIN: never wait forever.
                return await wait_for(measure(client, bulk_bytes), BENCH_TIMEOUT)
            except (OSError, TimeoutError, TunnelError, asyncio.IncompleteReadError) as exc:
                raise SplitUnavailable(f"tunnel test to {node.name} failed ({type(exc).__name__})") from None
        finally:
            await self._send_stop(lease)  # also after a timeout: the helper's test relay goes away

    # ----- helper events -----------------------------------------------------------------

    def on_rpc(self, node: Node, message: dict[str, Any] | None) -> None:
        """``rpc_ready``/``rpc_error`` from a helper, or ``None`` when it left."""
        if message is not None:
            fut = self._pending.get(message["session_id"])
            if fut is not None and not fut.done():
                fut.set_result(message)
                return
        else:
            # Gone while we wait for its rpc_ready: don't make the start wait the full minute.
            for session_id, asked in list(self._pending_nodes.items()):
                fut = self._pending.get(session_id)
                if asked is node and fut is not None and not fut.done():
                    fut.set_result({"type": "rpc_error", "error": "the helper disconnected"})
        s = self.session
        if s is None or s.state not in ("ready", "starting"):
            return
        lease = next((x for x in s.leases if x.node is node), None)
        if lease is None or (message is not None and message["session_id"] != lease.session_id):
            return
        if message is None:
            reason = f"{node.name} disconnected"
        else:
            reason = f"{node.name}: {safe_text(str(message.get('error', '')), 200)}"
        self.report_failure(s, reason)

    def report_failure(self, s: SplitSession, reason: str, kind: str = TRANSIENT) -> asyncio.Task[Any] | None:
        """A running session broke (helper gone, stalled, crashed): stop it, Ollama takes over.

        Returns the stopping task, so a caller about to hand the model to Ollama can wait
        until llama-server has given the GPU back.
        """
        if s.state not in ("ready", "starting") or s.error:
            return None
        s.error = reason
        self.hub.plans.fail(s.model, reason, list(s.output), kind)
        task = asyncio.ensure_future(self._stop_and_cut(s, reason))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def _stop_and_cut(self, s: SplitSession, reason: str) -> None:
        async with self._lock:
            if self.session is s:
                await self._stop(s, reason, failed=True)

    # ----- stop --------------------------------------------------------------------------

    async def _send_stop(self, lease: HelperLease) -> None:
        if lease.relay is not None:
            lease.relay.close()
        channel = lease.node.channel
        if channel is not None and not channel.closed:
            with contextlib.suppress(Exception):
                await channel.send("rpc_stop", session_id=lease.session_id)
        if lease.node.split_model:
            lease.node.split_model = ""

    async def _stop(self, s: SplitSession, reason: str, failed: bool = False) -> None:
        """Stop ``s`` completely, even if the caller is cancelled meanwhile.

        A half-stopped session (llama-server killed but helpers never told, the session
        still set) would keep the helpers' VRAM until something else happened to stop it,
        so the cleanup runs as its own task and is waited for before a cancellation goes on.
        """
        task = asyncio.ensure_future(self._stop_now(s, reason, failed))
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if not task.done():
                await task  # bounded by STOP_GRACE; then let the cancellation continue
            raise

    async def _stop_now(self, s: SplitSession, reason: str, failed: bool) -> None:
        was_ready = s.state == "ready"
        s.state = "stopping"
        if failed:
            # Requests still streaming from a broken session end now (with an error), not at the stall timeout.
            from ..runtime import model_key

            for stream in list(self.hub.streams.values()):
                if stream.engine == ENGINE_SPLIT and stream.model and model_key(stream.model) == s.model:
                    stream.kill(f"split mode stopped: {reason}")
        try:
            proc = s.process
            if proc is not None and proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    proc.terminate()
                try:
                    await wait_for(proc.wait(), STOP_GRACE)
                except TimeoutError:
                    with contextlib.suppress(ProcessLookupError):
                        proc.kill()
                    await proc.wait()
        finally:
            for lease in s.leases:
                await self._send_stop(lease)
            if s.client is not None:
                await s.client.aclose()
        s.state = "failed" if failed else "stopped"
        if self.session is s:
            self.session = None
        s.idle.set()
        if was_ready or failed:
            self.hub.event("split_stop", model=s.model, reason=reason)

    async def stop(self, model: str | None = None, reason: str = "stopped", drain: bool = False) -> bool:
        """Stop the session (of ``model``, if given). ``drain``: let running answers finish first."""
        from ..runtime import model_key

        async with self._lock:
            s = self.session
            if s is None or (model is not None and s.model != model_key(model)):
                return False
            if drain and s.active:
                with contextlib.suppress(TimeoutError):
                    await wait_for(s.idle.wait(), DRAIN_TIMEOUT)
                if self.session is not s:
                    return True
            await self._stop(s, reason)
            return True

    def on_node_state(self, node: Node) -> None:
        """A helper stopped lending (schedule, auto-yield, paused): drain the session it is part of.

        This is planned, not a failure: the plan isn't marked degraded, and once the
        helper lends again the same plan applies without a new benchmark.
        """
        s = self.session
        if s is None or s.state not in ("ready", "starting") or s.draining or node.status == "lending":
            return
        if not any(x.node is node for x in s.leases):
            return
        s.draining = True
        reason = f"{node.name} is {node.status}" + (f" ({node.lend_reason})" if node.lend_reason else "")
        self.hub.event("split_drain", model=s.model, node=node.node_id, reason=reason)
        log.info("draining the split session of %s: %s", s.model, reason)
        self._spawn(self._drain_and_stop(s, reason))

    async def _drain_and_stop(self, s: SplitSession, reason: str) -> None:
        if s.active:
            with contextlib.suppress(TimeoutError):
                await wait_for(s.idle.wait(), DRAIN_TIMEOUT)
        async with self._lock:
            if self.session is s:
                await self._stop(s, reason)

    async def run(self) -> None:
        """Stop an idle session after ``split.keep_alive``; notice a crashed llama-server."""
        try:
            while True:
                await asyncio.sleep(IDLE_CHECK)
                s = self.session
                if s is None or s.state != "ready":
                    continue
                if s.process is not None and s.process.returncode is not None:
                    code = s.process.returncode
                    self.report_failure(s, f"llama-server exited with code {code}", exit_kind(code, list(s.output)))
                elif not s.active and time.monotonic() - s.last_used > self.config.split.keep_alive:
                    await self.stop(s.model, "idle")
        finally:
            # Shutdown: abandon starts in progress (not a plan failure), finish background
            # stops and drains, then stop what is left - under the lock, like every stop.
            try:
                await reap(*self._acquire_tasks, *self._tasks)
            finally:
                # reap re-raises our own cancellation once the tasks are done: the stop
                # must still happen, or llama-server and the helpers' VRAM outlive the hub.
                async with self._lock:
                    s = self.session
                    if s is not None:
                        await self._stop(s, "hub shutting down")

    # ----- views -------------------------------------------------------------------------

    def warnings(self) -> list[str]:
        now = time.monotonic()
        return list(dict.fromkeys(m for t, m in self._warnings if now - t < WARNING_TTL))

    def snapshot(self) -> dict[str, Any]:
        s = self.session
        return {
            "session": s.snapshot() if s is not None else None,
            "llama_cpp": self.binaries.summary() if self.binaries is not None else None,
        }

    def ps_entries(self) -> list[dict[str, Any]]:
        """``/api/ps``-shaped entries for the running split model, as Ollama would describe it."""
        from ..runtime import model_key

        s = self.session
        if s is None or not s.alive() or s.files is None or s.files.gguf is None or s.plan is None:
            return []
        size = s.files.gguf.file_size
        on_gpu = sum(p.bytes for p in s.plan.placements)
        installed = next((m for m in self.hub.ollama.installed if model_key(m.name) == s.model), None)
        # When it unloads: keep_alive after the last answer (``ollama ps`` shows "Stopping..." without it).
        idle_for = 0.0 if s.active else time.monotonic() - s.last_used
        expires = datetime.now(UTC) + timedelta(seconds=max(0.0, self.config.split.keep_alive - idle_for))
        return [
            {
                "name": s.model,
                "model": s.model,
                "size": size,
                "size_vram": min(size, on_gpu),
                "digest": (installed.digest if installed and installed.digest else s.files.digest),
                "details": {
                    "parent_model": "",
                    "format": "gguf",
                    "family": s.files.gguf.architecture,
                    "families": [s.files.gguf.architecture],
                    "parameter_size": installed.parameter_size if installed else "",
                    "quantization_level": installed.quantization if installed else "",
                },
                "expires_at": expires.isoformat().replace("+00:00", "Z"),
                "context_length": s.ctx,
                "ollamapower": {"engine": ENGINE_SPLIT, "nodes": [x.node.name for x in s.leases]},
            }
        ]

    def describe(self) -> str:
        s = self.session
        if s is None or s.plan is None:
            return ""
        parts = [f"{p.device.label}: {p.layers} layers ({human_bytes(p.bytes)})" for p in s.plan.placements]
        if s.plan.cpu_layers:
            parts.append(f"hub CPU: {s.plan.cpu_layers} layers")
        return ", ".join(parts)
