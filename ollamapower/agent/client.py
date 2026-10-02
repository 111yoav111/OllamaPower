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
   while answering network tests. Split mode and overflow requests are answered
   with "not available yet" until Phases 3 and 6.

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
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from .. import __version__
from ..core import protocol
from ..core.channel import Channel
from ..core.encrypt_layer import CryptoError, SessionCrypto
from ..files import write_private
from ..telemetry import Telemetry, network_info
from ..textsafe import safe_text
from .config import AgentConfig, save_agent_config
from .pinning import KnownHubs, format_fingerprint, normalize_fingerprint

log = logging.getLogger("ollamapower.agent")

RECONNECT_DELAY = 5.0  # seconds, as in DCP
KICKED_DELAY = 60.0  # after the hub owner kicked us, wait longer before coming back
CONNECT_TIMEOUT = 10.0
MAX_OLLAMA_MODELS = 200

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
        self._tests: dict[str, dict[str, Any]] = {}
        self._requested: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._heartbeat_seq = 0

    @property
    def hub_key(self) -> str:
        return f"{self.address}:{self.port}"

    # ----- main loop -------------------------------------------------------------------

    async def run(self) -> None:
        """Connect and stay connected until stopped (cancelled) or an unrecoverable error."""
        self.state.static = await asyncio.to_thread(self.telemetry.static)
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
            finally:
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
        await self._check_pin(crypto.peer_fingerprint or "")
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

    async def _check_pin(self, fingerprint: str) -> None:
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
            return
        expected = normalize_fingerprint(self.config.hub.fingerprint)
        if expected:
            if expected != fingerprint:
                raise AgentStopped(
                    f"the hub's key {format_fingerprint(fingerprint)} doesn't match the configured fingerprint"
                )
        elif self.confirm is None or not await self.confirm(self.hub_key, fingerprint):
            raise AgentStopped(
                "the hub's key was not confirmed. Compare it with the fingerprint `ollamapower hub` prints and "
                "pass it with --hub-fingerprint."
            )
        self.known.pin(self.hub_key, fingerprint)
        self.state.pinned = True
        self.state.note(f"trusted the hub key {format_fingerprint(fingerprint)[:23]}…")

    # ----- session ---------------------------------------------------------------------

    async def _serve(self, channel: Channel, interval: float) -> float:
        heartbeat = asyncio.ensure_future(self._heartbeat_loop(channel, interval))
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
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat

    async def _heartbeat_loop(self, channel: Channel, interval: float) -> None:
        while True:
            sample = await asyncio.to_thread(self.telemetry.sample)
            self.state.telemetry = sample
            self.state.version += 1
            self._heartbeat_seq += 1
            await channel.send("heartbeat", seq=self._heartbeat_seq, telemetry=sample)
            await asyncio.sleep(interval)

    async def _on_message(self, channel: Channel, m: dict[str, Any]) -> None:
        kind = m["type"]
        if kind == "net_test_start":
            self._tests[m["test_id"]] = {"kind": m["kind"], "count": m.get("count", 0), "bytes": 0, "seen": 0}
            if m["kind"] == "up":
                asyncio.ensure_future(self._send_up(channel, m["test_id"], m.get("count", 0), m.get("size", 0)))
        elif kind == "net_test_data":
            await self._on_test_data(channel, m)
        elif kind == "net_test_result":
            fut = self._requested.pop(m["test_id"], None)
            self.state.last_net_test = m["result"]
            self.state.version += 1
            if fut is not None and not fut.done():
                fut.set_result(m["result"])
        elif kind == "rpc_start":
            await channel.send("rpc_error", session_id=m["session_id"], error="split mode is not available yet")
        elif kind == "overflow_request":
            await channel.send("stream_error", request_id=m["request_id"], error="overflow is not available yet")

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

    async def ollama_info(self) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(base_url=self.config.agent.ollama_url, timeout=3, trust_env=False) as c:
                version = (await c.get("/api/version")).json().get("version", "")
                tags = (await c.get("/api/tags")).json().get("models", []) or []
        except (httpx.HTTPError, ValueError, AttributeError):
            return {"present": False}
        models = [
            {"name": str(m.get("name", "")), "digest": str(m.get("digest", "")), "size": m.get("size", 0)}
            for m in tags[:MAX_OLLAMA_MODELS]
            if isinstance(m, dict)
        ]
        return {"present": True, "version": str(version), "models": models}

    async def capabilities(self) -> dict[str, Any]:
        return {
            **self.state.static,
            "agent_version": __version__,
            "llama_cpp": None,  # Phase 3
            "ollama": await self.ollama_info(),
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
        await self._send_lend_state(reason if paused else "")

    async def update_rules(self, **changes: Any) -> dict[str, Any]:
        """Change lending rules, save agent.toml and tell the hub."""
        from dataclasses import replace

        new = replace(self.config.lending, **changes)
        new.validate()
        self.config.lending = new
        await asyncio.to_thread(save_agent_config, self.config)
        self.state.note("lending rules changed")
        await self._send_lend_state()
        return self.lending_dict()

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
