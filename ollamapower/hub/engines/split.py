"""Split mode: one model across the hub's GPU and the helpers' GPUs (llama.cpp RPC).

For a model the owner planned with ``ollamapower models plan <model> --split``
the first request starts a split session:

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
than without OllamaPower) and split mode rests for ``RETRY_AFTER`` seconds for
that model; the reason is shown as a warning.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import secrets
import socket
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx

from ...core.tunnel import TunnelError, TunnelIdentity, check_cert_pem
from ...llamacpp.binaries import LlamaCpp
from ...llamacpp.install import Pins
from ...llamacpp.manager import LlamaCppUnavailable, ensure
from ...ollama.blobs import ModelFiles, ModelUnavailable, resolve
from ...ollama.client import PROBE_HEADERS, human_bytes, parse_ps
from ...textsafe import safe_text
from ..planner import MIB, Device, PlanError, SplitPlan, fits_on_hub, plan_split
from ..tunnel import BULK_BYTES, HubRelay, TunnelClient, measure

if TYPE_CHECKING:
    from ..nodes import Node
    from ..runtime import Hub

log = logging.getLogger("ollamapower.split")

RETRY_AFTER = 60.0  # seconds split mode rests for a model after a failed start
RPC_READY_TIMEOUT = 60.0
OLLAMA_RELEASE_TIMEOUT = 30.0
STOP_GRACE = 10.0
IDLE_CHECK = 5.0
WARNING_TTL = 600.0
ENGINE_SPLIT = "split"


class SplitUnavailable(Exception):
    """Split mode can't serve this request now; Ollama will (the message says why)."""


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
            "error": self.error,
        }


def _count(value: Any) -> bool:
    """A positive int (helpers' answers are untrusted)."""
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


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
        self._rest_until: dict[str, float] = {}
        self._warnings: deque[tuple[float, str]] = deque(maxlen=20)
        self._tasks: set[asyncio.Task[Any]] = set()

    # ----- plans -------------------------------------------------------------------------

    def plans(self) -> dict[str, dict[str, Any]]:
        if self.hub.state is None:
            return {}
        return {k: v for k, v in self.hub.state.data.get("plans", {}).items() if isinstance(v, dict)}

    def plan_for(self, model: str) -> dict[str, Any] | None:
        from ..runtime import model_key

        plan = self.plans().get(model_key(model))
        return plan if plan is not None and plan.get("mode") == "split" else None

    def wants(self, model: str) -> bool:
        """Should this model's next request try split mode?"""
        from ..runtime import model_key

        if not model or self.plan_for(model) is None:
            return False
        return time.monotonic() >= self._rest_until.get(model_key(model), 0.0)

    def is_running(self, model: str) -> bool:
        from ..runtime import model_key

        return self.session is not None and self.session.model == model_key(model) and self.session.alive()

    # ----- acquire / release -------------------------------------------------------------

    async def acquire(self, model: str) -> SplitSession:
        """A ready session for ``model`` (started if needed); pair with ``release``."""
        task = asyncio.ensure_future(self._acquire(model))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # The client left: let a start in progress finish (others may wait for it), but don't hold it.
            def done(t: asyncio.Task[SplitSession]) -> None:
                if not t.cancelled() and t.exception() is None:
                    self.release(t.result())

            task.add_done_callback(done)
            raise

    async def _acquire(self, model: str) -> SplitSession:
        from ..runtime import model_key

        key = model_key(model)
        async with self._lock:
            current = self.session
            if current is not None and current.model == key and current.alive():
                current.active += 1
                current.idle.clear()
                return current
            if current is not None:
                if current.model != key and current.active:
                    await current.idle.wait()  # let running requests of the other model finish
                await self._stop(current, f"switching to {key}" if current.model != key else "restarting")
            plan = self.plan_for(key) or {}
            ctx = int(plan.get("ctx") or self.config.split.default_ctx)
            session = SplitSession(model=key, ctx=ctx)
            self.session = session
            try:
                await asyncio.wait_for(self._start(session), self.config.split.load_timeout)
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
                await self._stop(session, reason, failed=True)
                self._rest(key, reason)
                if isinstance(exc, asyncio.CancelledError):
                    raise
                raise SplitUnavailable(reason) from None
            session.active += 1
            return session

    def release(self, session: SplitSession) -> None:
        session.active = max(0, session.active - 1)
        session.last_used = time.monotonic()
        if session.active == 0:
            session.idle.set()

    def _rest(self, model: str, reason: str) -> None:
        self._rest_until[model] = time.monotonic() + RETRY_AFTER
        message = f"split mode for {model} failed: {reason}; Ollama serves it (retrying in {int(RETRY_AFTER)} s)"
        self._warnings.append((time.monotonic(), message))
        self.hub.event("split_failed", model=model, reason=reason)
        log.warning("%s", message)

    # ----- start -------------------------------------------------------------------------

    async def _start(self, s: SplitSession) -> None:
        if self.http is None:
            raise SplitUnavailable("the front door is not running")
        self.hub.event("split_starting", model=s.model, ctx=s.ctx)
        try:
            s.files = await resolve(self.http, s.model)
        except ModelUnavailable as exc:
            raise SplitUnavailable(str(exc)) from None
        info = s.files.gguf
        assert info is not None
        binaries = await self._ensure_llamacpp()
        hub_devices = [Device(f"hub {d.name}", d.total_mib * MIB, device=d.name) for d in binaries.devices]
        if fits_on_hub(info, s.ctx, hub_devices):
            raise SplitUnavailable(f"{s.model} fits on the hub's GPU alone, so Ollama runs it faster without helpers")

        helpers, reasons = self.candidates()
        if not helpers:
            raise SplitUnavailable("no helper can take part" + (f" ({'; '.join(reasons)})" if reasons else ""))

        await self._release_ollama()
        await asyncio.to_thread(binaries.probe)  # free memory now that Ollama let go
        hub_devices = [Device(f"hub {d.name}", d.free_mib * MIB, device=d.name) for d in binaries.devices]

        s.leases = await self._lease(helpers, s.model)
        if not s.leases:
            raise SplitUnavailable("no helper agreed to lend memory")
        by_node = {x.node.node_id: x for x in s.leases}
        devices = [
            Device(
                f"{x.node.name} {x.endpoint.get('device', '')}",
                int(x.endpoint.get("free_mib", 0)) * MIB,
                node_id=x.node.node_id,
                device=safe_text(str(x.endpoint.get("device", "")), 40),
            )
            for x in s.leases
        ]
        try:
            s.plan = plan_split(s.model, info, s.ctx, hub_devices, devices)
        except PlanError as exc:
            raise SplitUnavailable(str(exc)) from None
        # Helpers that got no layers are released now; the rest in llama.cpp's device order (= --rpc order).
        used = [by_node[p.device.node_id] for p in s.plan.helpers if p.device.node_id]
        for lease in s.leases:
            if lease not in used:
                await self._send_stop(lease)
        s.leases = used
        for lease in s.leases:
            lease.node.split_model = s.model
        await self._launch(s, binaries)
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
            raise SplitUnavailable(f"llama.cpp on the hub: {exc}") from None
        if self.binaries.llama_server is None:
            raise SplitUnavailable("llama-server is missing from the hub's llama.cpp")
        return self.binaries

    def candidates(self) -> tuple[list[Node], list[str]]:
        """Helpers that may take part now (best first), and why the others can't."""
        pins = self.pins or Pins.load()
        ok, reasons = [], []
        for node in self.hub.registry.connected:
            lc = node.capabilities.get("llama_cpp") or {}
            if node.status != "lending":
                reasons.append(f"{node.name} is {node.status}")
            elif lc.get("status") != "ready":
                reasons.append(f"{node.name}: llama.cpp {lc.get('status') or 'unknown'} {lc.get('error', '')}".strip())
            elif lc.get("build") != pins.build:
                reasons.append(f"{node.name} has llama.cpp build {lc.get('build')}, needs {pins.build}")
            elif node.link.get("kind") == "wifi":
                reasons.append(f"{node.name} is on Wi-Fi (too slow for split mode)")
            else:
                ok.append(node)

        def preference(n: Node) -> tuple[float, int]:
            return (float(n.link.get("down_mbps") or n.link.get("speed_mbps") or 0), n.lendable_vram())

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

    async def _lease(self, helpers: list[Node], model: str) -> list[HelperLease]:
        leases = []
        for n in helpers:
            mode = n.tunnel_effective
            identity = TunnelIdentity.generate("hub") if mode == "secure" else None
            leases.append(HelperLease(node=n, session_id=secrets.token_hex(12), mode=mode, identity=identity))
        loop = asyncio.get_running_loop()
        for lease in leases:
            self._pending[lease.session_id] = loop.create_future()
            assert lease.node.channel is not None
            try:
                await lease.node.channel.send(
                    "rpc_start",
                    session_id=lease.session_id,
                    device="",
                    vram_cap_mb=0,
                    cache=True,
                    tunnel={"mode": lease.mode, **({"cert": lease.identity.cert_pem} if lease.identity else {})},
                    model=model,
                )
            except Exception:  # noqa: BLE001 - that helper just left
                self._pending.pop(lease.session_id).cancel()
        waiting = {x.session_id: x for x in leases if x.session_id in self._pending}
        if not waiting:
            return []
        done, _ = await asyncio.wait([self._pending[k] for k in waiting], timeout=RPC_READY_TIMEOUT)
        ready = []
        for session_id, lease in waiting.items():
            fut = self._pending.pop(session_id)
            if fut in done and not fut.cancelled() and fut.result().get("type") == "rpc_ready":
                endpoint = fut.result().get("endpoint") or {}
                port, free = endpoint.get("port"), endpoint.get("free_mib")
                if _count(port) and port < 65536 and _count(free) and endpoint.get("mode") == lease.mode:
                    lease.endpoint = {
                        "port": port,
                        "free_mib": min(free, 1 << 24),
                        "device": safe_text(str(endpoint.get("device", "")), 40),
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
            elif fut in done and not fut.cancelled():
                err = safe_text(str(fut.result().get("error", "")), 300)
                self._warnings.append((time.monotonic(), f"{lease.node.name} refused split mode: {err}"))
                continue
            fut.cancel()
            await self._send_stop(lease)
        return ready

    async def _launch(self, s: SplitSession, binaries: LlamaCpp) -> None:
        assert s.plan is not None and s.files is not None and binaries.llama_server is not None
        plan = s.plan
        rpc_names = {x.node.node_id: f"RPC{i}" for i, x in enumerate(s.leases)}
        device_names = [rpc_names[p.device.node_id] if p.device.node_id else p.device.device for p in plan.placements]
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
            "--rpc",
            ",".join(x.address for x in s.leases),
            "--device",
            ",".join(device_names),
            "--tensor-split",
            plan.tensor_split(),
            "-ngl",
            str(plan.gpu_layers),
        ]
        s.process = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(binaries.llama_server.parent),
            env={**os.environ, "LLAMA_API_KEY": s.api_key},
        )
        self._spawn(self._drain(s))
        s.client = httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{s.port}",
            headers={"authorization": f"Bearer {s.api_key}"},
            timeout=httpx.Timeout(connect=5.0, read=self.config.split.stall_timeout, write=60.0, pool=30.0),
            trust_env=False,
            follow_redirects=False,
        )
        while True:
            if s.process.returncode is not None:
                last = s.output[-1] if s.output else ""
                raise SplitUnavailable(f"llama-server exited with code {s.process.returncode}: {last}")
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

    def _spawn(self, coro: Any) -> None:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

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
            reply = await asyncio.wait_for(fut, RPC_READY_TIMEOUT)
        except TimeoutError:
            raise SplitUnavailable(f"{node.name} did not answer the tunnel test") from None
        finally:
            self._pending.pop(lease.session_id, None)
        try:
            if reply["type"] != "rpc_ready":
                raise SplitUnavailable(f"{node.name}: {safe_text(str(reply.get('error', '')), 300)}")
            endpoint = reply.get("endpoint") or {}
            port = endpoint.get("port")
            if not _count(port) or port >= 65536 or endpoint.get("mode") != mode:
                raise SplitUnavailable(f"{node.name} sent an unusable tunnel endpoint")
            try:
                cert = check_cert_pem(endpoint.get("cert")) if lease.identity else ""
                client = TunnelClient(node.ip, port, lease.identity, cert)
                return await measure(client, bulk_bytes)
            except (OSError, TimeoutError, TunnelError, asyncio.IncompleteReadError) as exc:
                raise SplitUnavailable(f"tunnel test to {node.name} failed ({type(exc).__name__})") from None
        finally:
            await self._send_stop(lease)

    # ----- helper events -----------------------------------------------------------------

    def on_rpc(self, node: Node, message: dict[str, Any] | None) -> None:
        """``rpc_ready``/``rpc_error`` from a helper, or ``None`` when it left."""
        if message is not None:
            fut = self._pending.get(message["session_id"])
            if fut is not None and not fut.done():
                fut.set_result(message)
                return
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
        s.error = reason
        self._rest(s.model, reason)
        self._spawn(self._stop_and_cut(s, reason))

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
        was_ready = s.state == "ready"
        s.state = "stopping"
        if failed:
            # Requests still streaming from a broken session end now (with an error), not at the stall timeout.
            from ..runtime import model_key

            for stream in list(self.hub.streams.values()):
                if stream.engine == ENGINE_SPLIT and stream.model and model_key(stream.model) == s.model:
                    stream.kill(f"split mode stopped: {reason}")
        proc = s.process
        if proc is not None and proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), STOP_GRACE)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                await proc.wait()
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

    async def stop(self, model: str | None = None, reason: str = "stopped") -> bool:
        from ..runtime import model_key

        async with self._lock:
            s = self.session
            if s is None or (model is not None and s.model != model_key(model)):
                return False
            await self._stop(s, reason)
            return True

    async def run(self) -> None:
        """Stop an idle session after ``split.keep_alive``; notice a crashed llama-server."""
        try:
            while True:
                await asyncio.sleep(IDLE_CHECK)
                s = self.session
                if s is None or s.state != "ready":
                    continue
                if s.process is not None and s.process.returncode is not None:
                    reason = f"llama-server exited with code {s.process.returncode}"
                    s.error = reason
                    self._rest(s.model, reason)
                    await self._stop_and_cut(s, reason)
                elif not s.active and time.monotonic() - s.last_used > self.config.split.keep_alive:
                    await self.stop(s.model, "idle")
        finally:
            s = self.session
            if s is not None:
                await asyncio.shield(self._stop(s, "hub shutting down"))

    # ----- views -------------------------------------------------------------------------

    def warnings(self) -> list[str]:
        now = time.monotonic()
        return list(dict.fromkeys(m for t, m in self._warnings if now - t < WARNING_TTL))

    def snapshot(self) -> dict[str, Any]:
        s = self.session
        return {
            "session": s.snapshot() if s is not None else None,
            "plans": self.plans(),
            "llama_cpp": self.binaries.summary() if self.binaries is not None else None,
        }

    def ps_entries(self) -> list[dict[str, Any]]:
        """``/api/ps``-shaped entries for the running split model."""
        s = self.session
        if s is None or not s.alive() or s.files is None or s.files.gguf is None or s.plan is None:
            return []
        size = s.files.gguf.file_size
        on_gpu = sum(p.bytes for p in s.plan.placements)
        return [
            {
                "name": s.model,
                "model": s.model,
                "size": size,
                "size_vram": min(size, on_gpu),
                "digest": s.files.digest,
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
