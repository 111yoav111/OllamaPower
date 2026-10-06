"""The agent's connection to the hub (DCP's ``WorkerClient``, adapted).

Lifecycle, retried every ``RECONNECT_DELAY`` seconds like DCP:

1. TCP connect and RSA handshake.
2. **Pin check before anything secret is sent.** A known hub must present the
   pinned key; a new hub is trusted only if it matches ``hub.fingerprint`` from
   ``agent.toml``/``--hub-fingerprint`` or the owner confirms it. A changed key
   stops the agent with an alert - it never retries with the token.
3. ``hello`` with the token -> ``welcome`` (or ``reject``: a wrong token or an
   incompatible version stops the agent; a full hub is retried).
4. ``capabilities``, the lend state, then heartbeats with telemetry every 2 s,
   while answering network tests and split-mode requests (``rpc_start`` /
   ``rpc_stop``, handled by ``RpcManager``, which installs the pinned llama.cpp
   in the background when the agent starts).

``AgentState`` holds what the agent TUI and ``ollamapower agent status`` show.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import secrets
import time
import uuid
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from .. import __version__
from ..aio import reap, wait_for
from ..core import protocol
from ..core.channel import Channel
from ..core.encrypt_layer import CryptoError, SessionCrypto
from ..files import write_private
from ..telemetry import Telemetry, network_info
from ..textsafe import safe_text
from .config import AgentConfig, LendingRules, load_agent_config, save_agent_config
from .lending import TUNNEL_QUIET, LendingGuard
from .pinning import KnownHubs, format_fingerprint, normalize_fingerprint
from .rpc_manager import RpcManager, RpcRefused

log = logging.getLogger("ollamapower.agent")

RECONNECT_DELAY = 5.0  # seconds, as in DCP
DRAIN_GRACE = 45.0  # seconds the hub gets to finish an answer and release us after we stop lending
KICKED_DELAY = 60.0  # after the hub owner kicked us, wait longer before coming back
CONNECT_TIMEOUT = 10.0
MAX_TESTS = 8  # network tests remembered at once

# Confirm callback: (hub address, fingerprint hex) -> trust it?
Confirm = Callable[[str, str], Awaitable[bool]]


class AgentStopped(Exception):
    """A condition retrying can't fix (changed hub key, wrong token, incompatible version)."""


def load_node_id(path: Any) -> str:
    """This helper's stable id, created once."""
    try:
        value = path.read_text().strip()
        if protocol.valid_node_id(value):
            return value
    except FileNotFoundError:
        pass
    value = str(uuid.uuid4())
    write_private(path, value.encode())
    return value


@dataclass
class AgentState:
    connection: str = "starting"  # starting · connecting · connected · retrying · stopped
    hub: str = ""
    fingerprint: str = ""
    pinned: bool = False
    node_id: str = ""
    error: str = ""
    paused: bool = False
    lend_state: str = "lending"  # what the hub is told: lending · paused · yielding
    lend_reason: str = ""
    connected_since: float | None = None
    telemetry: dict[str, Any] = field(default_factory=dict)
    static: dict[str, Any] = field(default_factory=dict)
    network: dict[str, Any] = field(default_factory=dict)
    last_net_test: dict[str, Any] | None = None
    events: deque[tuple[float, str]] = field(default_factory=lambda: deque(maxlen=200))
    version: int = 0

    def note(self, message: str) -> None:
        self.events.append((time.time(), message))
        self.version += 1
        log.info("%s", message)


class AgentClient:
    def __init__(
        self,
        config: AgentConfig,
        token: str,
        address: str,
        port: int,
        confirm: Confirm | None = None,
        telemetry: Telemetry | None = None,
    ) -> None:
        self.config = config
        self.token = token
        self.address = address
        self.port = port
        self.confirm = confirm
        self.telemetry = telemetry or Telemetry()
        self.known = KnownHubs(config.known_hub_path)
        self.state = AgentState(hub=f"{address}:{port}", node_id=load_node_id(config.node_id_path))
        self.channel: Channel | None = None
        self.welcome: dict[str, Any] = {}
        self._tests: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._background: set[asyncio.Task[Any]] = set()
        self._starts: dict[str, asyncio.Task[Any]] = {}  # rpc_start in progress, by session id
        self._requested: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._heartbeat_seq = 0
        self.rpc = RpcManager(config, self.state.note, self._rpc_failed)
        self.guard = LendingGuard()
        self._draining: asyncio.Task[Any] | None = None

    @property
    def hub_key(self) -> str:
        return f"{self.address}:{self.port}"

    # ----- main loop -------------------------------------------------------------------

    async def run(self) -> None:
        """Connect and stay connected until stopped (cancelled) or an unrecoverable error."""
        self.state.static = await asyncio.to_thread(self.telemetry.static)
        self._spawn(self._prepare_llamacpp())
        try:
            await self._run_connected()
        finally:
            # Stopping: nothing the agent started (an rpc_start or rpc_stop in progress,
            # a network test, the llama.cpp setup) may outlive it.
            await self._reap_background()

    async def _reap_background(self) -> None:
        draining = [self._draining] if self._draining is not None else []
        await reap(*self._background, *draining)

    async def _run_connected(self) -> None:
        while True:
            delay = RECONNECT_DELAY
            try:
                delay = await self._connect_and_serve()
            except AgentStopped as exc:
                self.state.connection, self.state.error = "stopped", str(exc)
                self.state.note(f"stopped: {exc}")
                raise
            except (OSError, asyncio.IncompleteReadError, TimeoutError) as exc:
                self.state.error = f"cannot reach the hub at {self.hub_key} ({type(exc).__name__})"
            except (CryptoError, protocol.ProtocolError, ValueError) as exc:
                self.state.error = f"protocol error: {exc}"
            except asyncio.CancelledError:
                # Stopping (Ctrl+C, quitting the TUI): tell the hub, so it isn't logged as a failure.
                if self.channel is not None:
                    await asyncio.shield(self.channel.close("agent stopping"))
                # Before the session is stopped below: an rpc_start in progress would make
                # that wait for it to finish starting (up to a minute and a half).
                await self._reap_background()
                raise
            finally:
                # Without the hub there is no one to lend to: end any split session.
                await asyncio.shield(self.rpc.stop(reason="hub connection lost"))
                if self.channel is not None:
                    await self.channel.close()
                    self.channel = None
                self.state.connected_since = None
            self.state.connection = "retrying"
            self.state.version += 1
            await asyncio.sleep(delay)

    async def _connect_and_serve(self) -> float:
        self.state.connection = "connecting"
        self.state.version += 1
        reader, writer = await wait_for(asyncio.open_connection(self.address, self.port), CONNECT_TIMEOUT)
        try:
            crypto = await wait_for(SessionCrypto.node_handshake(reader, writer), CONNECT_TIMEOUT)
        except BaseException:
            writer.close()
            raise
        channel = Channel(reader, writer, crypto)
        self.channel = channel
        if not await self._check_pin(crypto.peer_fingerprint or ""):
            # The owner must confirm a new key, which can take longer than the hub's
            # handshake timeout: hang up, ask, and come straight back once trusted.
            await channel.close()
            self.channel = None
            await self._confirm_new_key(crypto.peer_fingerprint or "")
            return 0.0
        await channel.send(
            "hello",
            protocol=protocol.PROTOCOL_VERSION,
            agent_version=__version__,
            node_id=self.state.node_id,
            name=self.config.display_name,
            token=self.token,
        )
        reply = await wait_for(channel.recv(), CONNECT_TIMEOUT)
        if reply is None:
            raise ConnectionError("the hub closed the connection")
        if reply["type"] == "reject":
            if reply["code"] in ("bad_token", "protocol"):
                raise AgentStopped(f"the hub refused us: {safe_text(reply['reason'], 200)}")
            self.state.error = f"the hub refused us: {safe_text(reply['reason'], 200)}"
            return RECONNECT_DELAY * 6 if reply["code"] == "banned" else RECONNECT_DELAY
        if reply["type"] != "welcome":
            raise protocol.ProtocolError(f"expected welcome, got {reply['type']}")
        self.welcome = reply
        self.state.connection, self.state.error = "connected", ""
        self.state.connected_since = time.time()
        self.state.note(f"connected to the hub at {self.hub_key}")
        local = channel.local
        self.state.network = await asyncio.to_thread(network_info, local[0] if local else None)
        await channel.send("capabilities", capabilities=await self.capabilities())
        decision = self.guard.evaluate(self.config.lending, self.state.paused)
        self.state.lend_state, self.state.lend_reason = decision.state, decision.reason
        await self._send_lend_state()
        return await self._serve(channel, float(reply["heartbeat_interval"]))

    async def _check_pin(self, fingerprint: str) -> bool:
        """True if the hub key is trusted now; False if the owner has to be asked first."""
        self.state.fingerprint = fingerprint
        pinned = self.known.get(self.hub_key)
        if pinned is not None:
            if pinned != fingerprint:
                raise AgentStopped(
                    f"THE HUB'S KEY CHANGED (was {format_fingerprint(pinned)[:23]}…, now "
                    f"{format_fingerprint(fingerprint)[:23]}…). Someone may be impersonating it. If you replaced "
                    "the hub's key yourself, run `ollamapower agent forget-hub` and connect again."
                )
            self.state.pinned = True
            return True
        expected = normalize_fingerprint(self.config.hub.fingerprint)
        if expected:
            if expected != fingerprint:
                raise AgentStopped(
                    f"the hub's key {format_fingerprint(fingerprint)} doesn't match the configured fingerprint"
                )
            self._pin(fingerprint)
            return True
        return False

    async def _confirm_new_key(self, fingerprint: str) -> None:
        if self.confirm is None or not await self.confirm(self.hub_key, fingerprint):
            raise AgentStopped(
                "the hub's key was not confirmed. Compare it with the fingerprint `ollamapower hub` prints and "
                "pass it with --hub-fingerprint."
            )
        self._pin(fingerprint)

    def _pin(self, fingerprint: str) -> None:
        self.known.pin(self.hub_key, fingerprint)
        self.state.pinned = True
        self.state.note(f"trusted the hub key {format_fingerprint(fingerprint)[:23]}…")

    # ----- session ---------------------------------------------------------------------

    async def _serve(self, channel: Channel, interval: float) -> float:
        heartbeat = asyncio.ensure_future(self._heartbeat_loop(channel, interval))
        # If heartbeats stop (send failed), end the session: the hub would drop us anyway.
        heartbeat.add_done_callback(lambda t: t.cancelled() or t.exception() is None or channel.writer.close())
        try:
            while True:
                message = await channel.recv()
                if message is None:
                    self.state.note("the hub closed the connection")
                    return RECONNECT_DELAY
                if message["type"] == "bye":
                    reason = safe_text(message["reason"], 200)
                    self.state.note(f"the hub disconnected us: {reason}")
                    return KICKED_DELAY if "kicked" in reason else RECONNECT_DELAY
                await self._on_message(channel, message)
        finally:
            await reap(heartbeat)

    async def _heartbeat_loop(self, channel: Channel, interval: float) -> None:
        while True:
            try:
                sample = await asyncio.to_thread(self.telemetry.sample)
            except Exception as exc:  # noqa: BLE001 - a flaky sensor must not stop the heartbeats
                log.warning("telemetry failed: %s", type(exc).__name__)
                sample = {}
            self.state.telemetry = sample
            self.state.version += 1
            await self._check_lending(sample)
            self._heartbeat_seq += 1
            await channel.send("heartbeat", seq=self._heartbeat_seq, telemetry=sample)
            await asyncio.sleep(interval)

    async def _on_message(self, channel: Channel, m: dict[str, Any]) -> None:
        kind = m["type"]
        if kind == "net_test_start":
            self._tests[m["test_id"]] = {"kind": m["kind"], "count": m.get("count", 0), "bytes": 0, "seen": 0}
            while len(self._tests) > MAX_TESTS:
                self._tests.popitem(last=False)
            if m["kind"] == "up":
                self._spawn(self._send_up(channel, m["test_id"], m.get("count", 0), m.get("size", 0)))
        elif kind == "net_test_data":
            await self._on_test_data(channel, m)
        elif kind == "net_test_result":
            fut = self._requested.pop(m["test_id"], None)
            self.state.last_net_test = m["result"]
            self.state.version += 1
            if fut is not None and not fut.done():
                fut.set_result(m["result"])
        elif kind == "rpc_start":
            session_id = m["session_id"]
            task = self._spawn(self._rpc_start(channel, m))  # starting takes a while; keep reading
            self._starts[session_id] = task
            task.add_done_callback(
                lambda t: self._starts.pop(session_id, None) if self._starts.get(session_id) is t else None
            )
        elif kind == "rpc_stop":
            # Stopping waits for that session's start, which can take a minute and a half:
            # not here, where it would hold up everything else the hub sends.
            self._spawn(self._rpc_stop(m["session_id"], self._starts.get(m["session_id"])))

    async def _rpc_stop(self, session_id: str, start: asyncio.Task[Any] | None) -> None:
        if start is not None:
            await asyncio.wait([start])  # stop what that start made, not before it exists
        if not self.rpc.stop_bench(session_id):
            await self.rpc.stop(session_id, "stopped by the hub")
        self.state.version += 1

    def _spawn(self, coro: Any) -> asyncio.Task[Any]:
        task = asyncio.ensure_future(coro)
        self._background.add(task)  # keep a reference: the loop only holds weak ones
        task.add_done_callback(self._background.discard)
        return task

    # ----- split mode ------------------------------------------------------------------

    async def _prepare_llamacpp(self) -> None:
        vendors = [g.get("vendor", "") for g in self.state.static.get("gpus", []) if isinstance(g, dict)]
        await self.rpc.prepare([v for v in vendors if v])
        self.state.version += 1
        channel = self.channel
        if channel is not None and not channel.closed and self.state.connection == "connected":
            with contextlib.suppress(Exception):
                await channel.send("capabilities", capabilities=await self.capabilities())

    async def _rpc_start(self, channel: Channel, m: dict[str, Any]) -> None:
        peer, local = channel.peer, channel.local
        try:
            if peer is None or local is None:
                raise RpcRefused("the connection to the hub is gone")
            start = self.rpc.start_bench if m.get("purpose") == "bench" else self.rpc.start
            rules = self.config.lending
            endpoint = await start(m, peer[0], local[0], rules, lambda: self._not_lending(rules))
        except RpcRefused as exc:
            self.state.note(f"split session refused: {exc}")
            with contextlib.suppress(Exception):
                await channel.send("rpc_error", session_id=m["session_id"], error=str(exc)[:1000])
            return
        except Exception as exc:  # noqa: BLE001 - report it to the hub instead of dying silently
            log.exception("starting rpc-server failed")
            with contextlib.suppress(Exception):
                await channel.send(
                    "rpc_error", session_id=m["session_id"], error=f"internal error ({type(exc).__name__})"
                )
            return
        self.state.version += 1
        try:
            await channel.send("rpc_ready", session_id=m["session_id"], endpoint=endpoint)
        except Exception:  # noqa: BLE001 - the hub is gone: don't keep lending
            await self.rpc.stop(m["session_id"], "hub connection lost")

    async def _rpc_failed(self, session_id: str, error: str) -> None:
        self.state.version += 1
        channel = self.channel
        if channel is not None and not channel.closed:
            with contextlib.suppress(Exception):
                await channel.send("rpc_error", session_id=session_id, error=error[:1000])

    def _not_lending(self, rules: LendingRules) -> str:
        """Why a session started under ``rules`` may not go ahead ("" if it may), asked during the start.

        ``state.paused`` is checked directly: ``set_paused`` sets it before it
        stops anything, while ``lend_state`` only follows once that is done.
        """
        if self.state.paused:
            return "paused by the helper's owner"
        if self.state.lend_state != "lending":
            return self.state.lend_reason
        if self.config.lending is not rules:
            return "the helper's owner changed the lending rules"
        return ""

    async def _end_session(self, reason: str) -> None:
        """The owner changed something: stop lending now and tell the hub why."""
        # stop_current waits for a session that is still starting, so it can't slip through.
        session_id = await self.rpc.stop_current(reason)
        if session_id is not None:
            await self._rpc_failed(session_id, reason)

    async def _on_test_data(self, channel: Channel, m: dict[str, Any]) -> None:
        test = self._tests.get(m["test_id"])
        if test is None:
            return
        if test["kind"] == "rtt":
            await channel.send("net_test_data", test_id=m["test_id"], seq=m["seq"], sent_at=m.get("sent_at", 0.0))
            return
        if test["kind"] != "down":
            return
        now = time.monotonic()
        if test["seen"] == 0:
            test["first"] = now  # the first chunk starts the clock
        else:
            test["bytes"] += len(m.get("data") or b"")
        test["seen"] += 1
        if test["seen"] >= test["count"]:
            seconds = now - test["first"]
            mbps = test["bytes"] * 8 / seconds / 1e6 if seconds > 0 else 0.0
            await channel.send("net_test_result", test_id=m["test_id"], result={"mbps": mbps, "bytes": test["bytes"]})

    async def _send_up(self, channel: Channel, test_id: str, count: int, size: int) -> None:
        with contextlib.suppress(Exception):
            for seq in range(count):
                await channel.send("net_test_data", test_id=test_id, seq=seq, data=os.urandom(size))

    # ----- facts -----------------------------------------------------------------------

    async def capabilities(self) -> dict[str, Any]:
        return {
            **self.state.static,
            "agent_version": __version__,
            "llama_cpp": self.rpc.summary(),
            "network": self.state.network,
            "lending": self.lending_dict(),
        }

    def lending_dict(self) -> dict[str, Any]:
        return {**self.config.lending.to_dict(), "paused": self.state.paused}

    async def _send_lend_state(self) -> None:
        channel = self.channel
        if channel is None or channel.closed or self.state.connection != "connected":
            return
        await channel.send(
            "lend_state", state=self.state.lend_state, reason=self.state.lend_reason, lending=self.lending_dict()
        )

    async def _check_lending(self, sample: dict[str, Any] | None = None) -> None:
        """Re-decide lending (pause, schedule, auto-yield); tell the hub when it changes."""
        rules = self.config.lending
        if sample is not None:
            self.guard.update_yield(rules, sample, self.rpc.tunnel_quiet(TUNNEL_QUIET), time.monotonic())
        decision = self.guard.evaluate(rules, self.state.paused)
        if (decision.state, decision.reason) == (self.state.lend_state, self.state.lend_reason):
            return
        was_lending = self.state.lend_state == "lending"
        self.state.lend_state, self.state.lend_reason = decision.state, decision.reason
        self.state.note(f"lending: {decision.state}" + (f" ({decision.reason})" if decision.reason else ""))
        with contextlib.suppress(Exception):
            await self._send_lend_state()
        if was_lending and decision.state != "lending" and not self.state.paused:
            # Schedule end or auto-yield: give the hub time to finish the current answer
            # and release us (rpc_stop); stop by ourselves only if it doesn't.
            if self._draining is None or self._draining.done():
                self._draining = asyncio.ensure_future(self._drain_session(decision.reason))

    async def _drain_session(self, reason: str) -> None:
        session = self.rpc.session
        if session is None:
            return
        deadline = time.monotonic() + DRAIN_GRACE
        while self.rpc.session is session and time.monotonic() < deadline:
            await asyncio.sleep(0.25)
        if self.rpc.session is session:
            await self._end_session(reason)

    # ----- owner actions ---------------------------------------------------------------

    async def set_paused(self, paused: bool, reason: str = "paused by the helper's owner") -> None:
        self.state.paused = paused
        if paused:  # the owner's pause is instant (spec 10.2)
            await self._end_session(reason)
        await self._check_lending()

    async def update_rules(self, **changes: Any) -> dict[str, Any]:
        """Change lending rules, save agent.toml and tell the hub."""
        from dataclasses import replace

        new = replace(self.config.lending, **changes)
        new.validate()
        await asyncio.to_thread(self._save_rules, new)
        self.config.lending = new
        self.state.note("lending rules changed")
        await self._end_session("the helper's owner changed the lending rules")
        await self._check_lending()
        await self._send_lend_state()  # the hub shows the new rules even if the state didn't change
        return self.lending_dict()

    def _save_rules(self, rules: Any) -> None:
        """Write only the lending rules: one-off CLI overrides (--name, --hub-fingerprint) stay one-off."""
        on_disk = load_agent_config(self.config.config_dir)
        on_disk.lending = rules
        save_agent_config(on_disk)

    async def request_net_test(self, timeout: float = 60.0) -> dict[str, Any]:
        channel = self.channel
        if channel is None or channel.closed or self.state.connection != "connected":
            raise ConnectionError("not connected to the hub")
        test_id = secrets.token_hex(8)
        fut: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._requested[test_id] = fut
        await channel.send("net_test_start", test_id=test_id, kind="request")
        try:
            return await wait_for(fut, timeout)
        finally:
            self._requested.pop(test_id, None)

    async def leave(self) -> None:
        if self.channel is not None:
            await self.channel.close("agent stopping")
