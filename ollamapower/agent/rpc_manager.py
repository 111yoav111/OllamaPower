"""The helper's part of split mode: one llama.cpp ``rpc-server``, run on the hub's behalf.

``rpc-server`` executes whatever a connected client asks for and has no
authentication, so it is only ever bound to ``127.0.0.1``. The hub reaches it
through this agent's relay (``tunnel.py``), which accepts only the hub's
address and only while a session is active:

* **secure mode** (the default): TLS 1.3 with per-session certificates. The
  hub sends its certificate in ``rpc_start``; the agent makes its own key and
  answers with its certificate in ``rpc_ready``. Each side trusts only the
  other's certificate for that session.
* **direct mode** (both owners opted in): plain TCP, unencrypted. A request
  for direct mode is refused unless this helper's owner allowed it, so
  nothing ever falls back to plaintext on its own.

A tunnel test (``purpose: "bench"``) runs the same relay in front of a small
echo server instead of ``rpc-server``, so the hub can measure the path's
latency and throughput, and the cost of encryption, without llama.cpp.

The helper's owner stays in charge: the agent checks every ``rpc_start``
against the lending rules itself (pause, GPU allow-list, CPU, direct mode,
memory cap) instead of trusting the hub, and pausing or changing the rules
ends a running session at once. The program is always the pinned,
hash-checked build; the hub can't choose what runs or with which arguments
beyond picking one of the allowed devices.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import socket
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.tunnel import TunnelError, TunnelIdentity, cert_fingerprint, check_cert_pem
from ..llamacpp.binaries import Device, LlamaCpp
from ..llamacpp.manager import LlamaCppUnavailable, ensure
from ..textsafe import safe_text
from .config import AgentConfig, LendingRules
from .tunnel import EchoTarget, HelperRelay

log = logging.getLogger("ollamapower.agent.rpc")

READY_TIMEOUT = 30.0  # seconds for rpc-server to accept connections
STOP_GRACE = 5.0
BENCH_TTL = 120.0  # a tunnel test's relay closes after this long, even if the hub never says stop
MIB = 1024 * 1024
GIB = 1024 * MIB
# When lending CPU/RAM, keep this much RAM for the helper's owner.
CPU_RAM_RESERVE = 2 * GIB


class RpcRefused(Exception):
    """The helper won't (or can't) start this session; the message goes back to the hub."""


@dataclass
class RpcSession:
    session_id: str
    device: str
    cap_mib: int
    hub_ip: str
    model: str = ""
    started: float = field(default_factory=time.time)
    process: asyncio.subprocess.Process | None = None
    port: int = 0
    relay: HelperRelay | None = None
    output: deque[str] = field(default_factory=lambda: deque(maxlen=40))
    stopping: bool = False
    watcher: asyncio.Task[Any] | None = None

    def info(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "device": self.device,
            "cap_mib": self.cap_mib,
            "model": self.model,
            "since": self.started,
            "connections": len(self.relay.connections) if self.relay else 0,
            "tunnel": self.relay.mode if self.relay else "",
        }

    @property
    def relay_port(self) -> int:
        return self.relay.port if self.relay else 0


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def device_cap_mib(device: Device, index: int, rules: LendingRules) -> int | None:
    """MiB of ``device`` the owner lends right now, or None if it isn't lent at all."""
    if rules.gpus and index not in rules.gpus:
        return None
    cap = device.free_mib
    if rules.max_vram_gb > 0:
        cap = min(cap, int(rules.max_vram_gb * 1024))
    if rules.max_vram_percent > 0:
        cap = min(cap, device.total_mib * rules.max_vram_percent // 100)
    return max(0, cap)


def cpu_cap_mib(rules: LendingRules) -> int:
    try:
        import psutil

        available = psutil.virtual_memory().available
    except Exception:  # noqa: BLE001 - psutil missing or failing: lend nothing
        return 0
    cap = max(0, available - CPU_RAM_RESERVE) // MIB
    if rules.max_vram_gb > 0:
        cap = min(cap, int(rules.max_vram_gb * 1024))
    return int(cap)


class RpcManager:
    def __init__(
        self,
        config: AgentConfig,
        note: Callable[[str], None],
        on_failure: Callable[[str, str], Any] | None = None,
    ) -> None:
        self.config = config
        self.note = note
        self.on_failure = on_failure  # (session_id, error), when rpc-server dies on its own
        self.binaries: LlamaCpp | None = None
        self.status = "not installed"  # not installed · installing · ready · error
        self.error = ""
        self.progress: tuple[int, int] | None = None
        self.session: RpcSession | None = None
        self.benches: dict[str, tuple[HelperRelay, EchoTarget, asyncio.TimerHandle]] = {}
        self._lock = asyncio.Lock()

    @property
    def cache_dir(self) -> Path:
        return self.config.config_dir / "rpc-cache"

    # ----- install -----------------------------------------------------------------------

    async def prepare(self, gpu_vendors: list[str]) -> None:
        """Install (or find) the pinned llama.cpp in the background; never raises."""
        self.status, self.error = "installing", ""

        def progress(name: str, done: int, total: int) -> None:
            self.progress = (done, total)

        try:
            self.binaries = await asyncio.to_thread(
                ensure,
                self.config.config_dir,
                self.config.llamacpp.backend,
                self.config.llamacpp.path,
                gpu_vendors,
                progress,
                None,
                self.config.llamacpp.auto_install,
            )
        except LlamaCppUnavailable as exc:
            self.status, self.error = "error", str(exc)
            self.note(f"llama.cpp unavailable: {exc}")
            return
        except Exception as exc:  # noqa: BLE001 - a failed install must not stop the agent
            self.status, self.error = "error", type(exc).__name__
            log.exception("llama.cpp setup failed")
            return
        finally:
            self.progress = None
        self.status = "ready"
        names = ", ".join(d.name for d in self.binaries.devices) or "CPU only"
        self.note(f"llama.cpp build {self.binaries.build} ready ({names})")

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {"status": self.status, "error": self.error}
        if self.binaries is not None:
            out.update(self.binaries.summary())
        if self.progress is not None:
            out["progress"] = list(self.progress)
        return out

    # ----- sessions ----------------------------------------------------------------------

    async def _fresh_devices(self) -> list[Device]:
        assert self.binaries is not None
        try:
            await asyncio.to_thread(self.binaries.probe)
        except Exception as exc:  # noqa: BLE001 - keep the last known list
            log.warning("re-probing devices failed: %s", type(exc).__name__)
        return self.binaries.devices

    async def _choose(self, wanted: str, rules: LendingRules) -> tuple[str, int, int]:
        """(device name, MiB lent, MiB total) for the requested device, or the best one allowed."""
        devices = await self._fresh_devices()
        lent = [(d, cap) for i, d in enumerate(devices) if (cap := device_cap_mib(d, i, rules)) is not None]
        if wanted and wanted != "CPU":
            match = [(d, cap) for d, cap in lent if d.name == wanted]
            if not match:
                raise RpcRefused(f"device {safe_text(wanted, 40)} is not lent by this helper")
            d, cap = match[0]
            return d.name, cap, d.total_mib
        if wanted != "CPU" and lent:
            d, cap = max(lent, key=lambda x: x[1])
            if cap > 0:
                return d.name, cap, d.total_mib
        if rules.allow_cpu:
            cap = await asyncio.to_thread(cpu_cap_mib, rules)
            return "CPU", cap, cap
        raise RpcRefused("this helper lends no GPU right now and CPU lending is off")

    async def start(
        self,
        message: dict[str, Any],
        hub_ip: str,
        bind_ip: str,
        rules: LendingRules,
        paused: bool,
    ) -> dict[str, Any]:
        """Start a session for ``rpc_start``; returns the endpoint for ``rpc_ready``."""
        async with self._lock:
            if self.status != "ready" or self.binaries is None or self.binaries.rpc_server is None:
                raise RpcRefused(
                    f"llama.cpp is not ready on this helper ({self.status}{': ' + self.error if self.error else ''})"
                )
            if paused:
                raise RpcRefused("lending is paused by the helper's owner")
            tunnel = _Tunnel.from_message(message, rules)
            if self.session is not None:
                await self._stop(self.session, "replaced by a new session")
            device, cap_mib, total_mib = await self._choose(message.get("device", ""), rules)
            if message.get("vram_cap_mb", 0) > 0:
                cap_mib = min(cap_mib, int(message["vram_cap_mb"]))
            session = RpcSession(
                session_id=message["session_id"],
                device=device,
                cap_mib=cap_mib,
                hub_ip=hub_ip,
                model=safe_text(str(message.get("model", "")), 120),
            )
            cache = bool(message.get("cache")) and rules.disk_cache
            try:
                await self._launch(session, cache)
                session.relay = tunnel.relay(session.port, hub_ip)
                await session.relay.start(bind_ip)
            except BaseException:
                await self._stop(session, "failed to start")
                raise
            self.session = session
            self.note(
                f"lending {device} ({cap_mib} MiB) to the hub for {session.model or 'a model'} "
                f"({tunnel.mode} tunnel{', disk cache' if cache else ''})"
            )
            return {
                **tunnel.endpoint(),
                "port": session.relay_port,
                "device": device,
                "free_mib": cap_mib,
                "total_mib": total_mib,
                "build": self.binaries.build,
                "cache": cache,
            }

    async def _launch(self, session: RpcSession, cache: bool) -> None:
        assert self.binaries is not None and self.binaries.rpc_server is not None
        exe = self.binaries.rpc_server
        args = [str(exe), "-H", "127.0.0.1", "-d", session.device]
        if cache:
            self.cache_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            args.append("-c")
        env = {**os.environ, "LLAMA_CACHE": str(self.cache_dir)}
        last_error = ""
        for _ in range(3):  # another program may grab the port between choosing and binding it
            session.port = _free_port()
            proc = await asyncio.create_subprocess_exec(
                *args,
                "-p",
                str(session.port),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(exe.parent),
                env=env,
            )
            session.process = proc
            reader = asyncio.ensure_future(self._drain(session, proc))
            if await self._wait_ready(session, proc):
                session.watcher = asyncio.ensure_future(self._watch(session, proc, reader))
                return
            await reader
            last_error = session.output[-1] if session.output else f"exit code {proc.returncode}"
        raise RpcRefused(f"rpc-server did not start: {last_error}")

    async def _drain(self, session: RpcSession, proc: asyncio.subprocess.Process) -> None:
        assert proc.stdout is not None
        while line := await proc.stdout.readline():
            session.output.append(safe_text(line.decode("utf-8", "replace").rstrip(), 300))

    async def _wait_ready(self, session: RpcSession, proc: asyncio.subprocess.Process) -> bool:
        deadline = time.monotonic() + READY_TIMEOUT
        while time.monotonic() < deadline:
            if proc.returncode is not None:
                return False
            try:
                _, writer = await asyncio.wait_for(asyncio.open_connection("127.0.0.1", session.port), 2.0)
            except (OSError, TimeoutError):
                await asyncio.sleep(0.2)
                continue
            writer.close()
            with contextlib.suppress(OSError):
                await writer.wait_closed()
            return True
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        await proc.wait()
        return False

    async def _watch(self, session: RpcSession, proc: asyncio.subprocess.Process, reader: asyncio.Future[Any]) -> None:
        code = await proc.wait()
        with contextlib.suppress(Exception):
            await reader
        if session.stopping:
            return
        last = session.output[-1] if session.output else ""
        error = f"rpc-server exited with code {code}" + (f": {last}" if last else "")
        self.note(error)
        await self._stop(session, "rpc-server exited")
        if self.session is session:
            self.session = None
        if self.on_failure is not None:
            await self.on_failure(session.session_id, error)

    # ----- tunnel test -------------------------------------------------------------------

    async def start_bench(
        self, message: dict[str, Any], hub_ip: str, bind_ip: str, rules: LendingRules, paused: bool
    ) -> dict[str, Any]:
        """A relay in front of an echo server, so the hub can time the tunnel itself."""
        if paused:
            raise RpcRefused("lending is paused by the helper's owner")
        tunnel = _Tunnel.from_message(message, rules)
        session_id = message["session_id"]
        if len(self.benches) >= 2:
            raise RpcRefused("a tunnel test is already running")
        target = EchoTarget()
        await target.start()
        relay = tunnel.relay(target.port, hub_ip)
        try:
            await relay.start(bind_ip)
        except BaseException:
            target.close()
            raise
        expiry = asyncio.get_running_loop().call_later(BENCH_TTL, self.stop_bench, session_id)
        self.benches[session_id] = (relay, target, expiry)
        return {**tunnel.endpoint(), "port": relay.port}

    def stop_bench(self, session_id: str) -> bool:
        bench = self.benches.pop(session_id, None)
        if bench is None:
            return False
        relay, target, expiry = bench
        expiry.cancel()
        relay.close()
        target.close()
        return True

    # ----- stop --------------------------------------------------------------------------

    async def stop(self, session_id: str | None = None, reason: str = "stopped") -> bool:
        """Stop the session (if ``session_id`` is given, only when it matches)."""
        async with self._lock:
            session = self.session
            if session is None or (session_id is not None and session.session_id != session_id):
                return False
            self.session = None
            await self._stop(session, reason)
            self.note(f"stopped lending to the hub ({reason})")
            return True

    async def _stop(self, session: RpcSession, reason: str) -> None:
        session.stopping = True
        if session.relay is not None:
            session.relay.close()
        proc = session.process
        if proc is not None and proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), STOP_GRACE)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                await proc.wait()
        if session.watcher is not None and session.watcher is not asyncio.current_task():
            session.watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await session.watcher
        log.info("rpc session %s stopped: %s", session.session_id[:8], reason)


class _Tunnel:
    """The tunnel an ``rpc_start`` asks for, checked against this helper's rules."""

    def __init__(self, mode: str, hub_cert: str = "") -> None:
        self.mode = mode
        self.hub_cert = hub_cert
        self.identity = TunnelIdentity.generate("helper") if mode == "secure" else None

    @classmethod
    def from_message(cls, message: dict[str, Any], rules: LendingRules) -> _Tunnel:
        tunnel = message.get("tunnel") or {}
        mode = tunnel.get("mode", "secure")
        if mode == "direct":
            if not rules.allow_direct:
                raise RpcRefused("the helper's owner has not allowed direct (unencrypted) mode")
            return cls("direct")
        if mode == "secure":
            try:
                return cls("secure", check_cert_pem(tunnel.get("cert")))
            except TunnelError as exc:
                raise RpcRefused(f"secure tunnel: {exc}") from None
        raise RpcRefused(f"unknown tunnel mode {safe_text(str(mode), 20)}")

    def relay(self, target_port: int, hub_ip: str) -> HelperRelay:
        if self.identity is None:
            return HelperRelay(target_port, hub_ip)
        tls = self.identity.context(server=True, peer_cert_pem=self.hub_cert)
        return HelperRelay(target_port, hub_ip, tls, cert_fingerprint(self.hub_cert))

    def endpoint(self) -> dict[str, Any]:
        out: dict[str, Any] = {"mode": self.mode}
        if self.identity is not None:
            out["cert"] = self.identity.cert_pem
        return out
