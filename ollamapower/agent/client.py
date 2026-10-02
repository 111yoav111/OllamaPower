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
from ..core import protocol
from ..core.channel import Channel
from ..core.encrypt_layer import CryptoError, SessionCrypto
from ..files import write_private
from ..telemetry import Telemetry, network_info
from ..textsafe import safe_text
from .config import AgentConfig, load_agent_config, save_agent_config
from .pinning import KnownHubs, format_fingerprint, normalize_fingerprint
from .rpc_manager import RpcManager, RpcRefused

log = logging.getLogger("ollamapower.agent")

RECONNECT_DELAY = 5.0  # seconds, as in DCP
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
        self._requested: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._heartbeat_seq = 0
        self.rpc = RpcManager(config, self.state.note, self._rpc_failed)

    @property
    def hub_key(self) -> str:
        return f"{self.address}:{self.port}"

    # ----- main loop -------------------------------------------------------------------

    async def run(self) -> None:
        """Connect and stay connected until stopped (cancelled) or an unrecoverable error."""
        self.state.static = await asyncio.to_thread(self.telemetry.static)
        self._spawn(self._prepare_llamacpp())
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
        reader, writer = await asyncio.wait_for(asyncio.open_connection(self.address, self.port), CONNECT_TIMEOUT)
        try:
            crypto = await asyncio.wait_for(SessionCrypto.node_handshake(reader, writer), CONNECT_TIMEOUT)
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
        reply = await asyncio.wait_for(channel.recv(), CONNECT_TIMEOUT)
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
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await heartbeat

    async def _heartbeat_loop(self, channel: Channel, interval: float) -> None:
        while True:
            try:
                sample = await asyncio.to_thread(self.telemetry.sample)
            except Exception as exc:  # noqa: BLE001 - a flaky sensor must not stop the heartbeats
                log.warning("telemetry failed: %s", type(exc).__name__)
                sample = {}
            self.state.telemetry = sample
            self.state.version += 1
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
            self._spawn(self._rpc_start(channel, m))  # starting takes a while; keep reading
        elif kind == "rpc_stop":
            if not self.rpc.stop_bench(m["session_id"]):
                await self.rpc.stop(m["session_id"], "stopped by the hub")
            self.state.version += 1

    def _spawn(self, coro: Any) -> None:
        task = asyncio.ensure_future(coro)
        self._background.add(task)  # keep a reference: the loop only holds weak ones
        task.add_done_callback(self._background.discard)

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
            endpoint = await start(m, peer[0], local[0], self.config.lending, self.state.paused)
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

    async def _end_session(self, reason: str) -> None:
        """The owner changed something: stop lending now and tell the hub why."""
        session = self.rpc.session
        if session is not None and await self.rpc.stop(session.session_id, reason):
            await self._rpc_failed(session.session_id, reason)

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

    async def _send_lend_state(self, reason: str = "") -> None:
        channel = self.channel
        if channel is None or channel.closed or self.state.connection != "connected":
            return
        state = "paused" if self.state.paused else "lending"
        await channel.send("lend_state", state=state, reason=reason, lending=self.lending_dict())

    # ----- owner actions ---------------------------------------------------------------

    async def set_paused(self, paused: bool, reason: str = "paused by the helper's owner") -> None:
        self.state.paused = paused
        self.state.note("lending paused" if paused else "lending resumed")
        if paused:
            await self._end_session("lending paused by the helper's owner")
        await self._send_lend_state(reason if paused else "")

    async def update_rules(self, **changes: Any) -> dict[str, Any]:
        """Change lending rules, save agent.toml and tell the hub."""
        from dataclasses import replace

        new = replace(self.config.lending, **changes)
        new.validate()
        await asyncio.to_thread(self._save_rules, new)
        self.config.lending = new
        self.state.note("lending rules changed")
        await self._end_session("the helper's owner changed the lending rules")
        await self._send_lend_state()
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
            return await asyncio.wait_for(fut, timeout)
        finally:
            self._requested.pop(test_id, None)

    async def leave(self) -> None:
        if self.channel is not None:
            await self.channel.close("agent stopping")
