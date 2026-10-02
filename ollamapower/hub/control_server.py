"""The control plane on the hub: where helpers connect (DCP's ``MasterServer``, adapted).

A connection goes through, in order:

1. **Network allowlist** (``control.allowed_networks``, private ranges by
   default) and the **ban list** - dropped before any byte is read.
2. **RSA handshake** with the hub's persistent key (agents pin its
   fingerprint, so a man in the middle can't pose as the hub).
3. **hello** inside the encrypted session, with frames capped at 64 KB: the
   token is checked in constant time, then the protocol major version.
   Failures count towards a temporary ban of that IP.
4. **welcome**, then the session: capabilities, heartbeats every 2 s, lend
   state, network tests. A node silent for ``heartbeat_timeout`` is dropped.

The handshake must finish within ``handshake_timeout``. A second connection
from the same node id replaces the first (an agent reconnecting after a
network glitch while its old socket is still half-open).
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import logging
import os
import secrets
import time
from collections.abc import Callable
from typing import Any

from cryptography.hazmat.primitives.asymmetric import rsa

from .. import __version__
from ..aio import wait_for
from ..config import ControlConfig
from ..core import protocol
from ..core.auth import check_token
from ..core.channel import PRE_AUTH_MAX_FRAME, Channel
from ..core.encrypt_layer import MAX_FRAME, CryptoError, SessionCrypto
from .bans import BanList
from .nodes import Node, NodeRegistry

log = logging.getLogger("ollamapower.control")

HEARTBEAT_INTERVAL = 2.0
NET_TEST_PINGS = 10
NET_TEST_CHUNKS = 8
NET_TEST_CHUNK_BYTES = 1024 * 1024
NET_TEST_TIMEOUT = 30.0


class NetTestError(Exception):
    pass


class _NetTest:
    """Hub-side bookkeeping of one running test."""

    def __init__(self) -> None:
        loop = asyncio.get_running_loop()
        self.echoes: dict[int, asyncio.Future[float]] = {}
        self.result: asyncio.Future[dict[str, Any]] = loop.create_future()
        self.up_first: float | None = None
        self.up_last: float | None = None
        self.up_bytes = 0
        self.up_chunks = 0
        self.up_done: asyncio.Future[None] = loop.create_future()


class ControlServer:
    def __init__(
        self,
        config: ControlConfig,
        registry: NodeRegistry,
        key: rsa.RSAPrivateKey,
        token: str | None,
        emit: Any,
    ) -> None:
        self.config = config
        self.registry = registry
        self.key = key
        self.token = token
        self.emit = emit
        self.networks = [ipaddress.ip_network(c, strict=False) for c in config.allowed_networks]
        self.bans = BanList(config.ban_failures, config.ban_window, config.ban_seconds, config.max_ban_seconds)
        self.server: asyncio.AbstractServer | None = None
        self._tests: dict[str, _NetTest] = {}
        self._tasks: set[asyncio.Task[Any]] = set()
        self._pending = 0
        # Split mode's handler for rpc_ready/rpc_error; called with None when a helper leaves.
        self.on_rpc: Callable[[Node, dict[str, Any] | None], None] | None = None
        # Split mode's handler for a helper that stopped (or started again) lending.
        self.on_node_state: Callable[[Node], None] | None = None

    # ----- server lifecycle ----------------------------------------------------------------

    async def start(self, sock: Any = None) -> None:
        if sock is not None:
            self.server = await asyncio.start_server(self._handle, sock=sock)
        else:
            self.server = await asyncio.start_server(self._handle, self.config.host, self.config.port)
        self._spawn(self._watchdog())
        log.info("control plane listening on %s:%d", self.config.host, self.port)

    @property
    def port(self) -> int:
        assert self.server is not None
        return int(self.server.sockets[0].getsockname()[1])

    async def close(self) -> None:
        if self.server is not None:
            self.server.close()
        # All at once: each close may wait a few seconds for a slow helper.
        await asyncio.gather(
            *(n.channel.close("hub shutting down") for n in self.registry.connected if n.channel is not None),
            return_exceptions=True,
        )
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        if self.server is not None:
            with contextlib.suppress(Exception):
                await wait_for(self.server.wait_closed(), 3)

    def _spawn(self, coro: Any) -> asyncio.Task[Any]:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def _watchdog(self) -> None:
        while True:
            await asyncio.sleep(1.0)
            for node in self.registry.timed_out():
                log.warning("node %s (%s) missed its heartbeats", node.name, node.node_id[:8])
                channel = node.channel
                self.registry.detach(node, channel, "heartbeat timeout")  # type: ignore[arg-type]
                if channel is not None:
                    self._spawn(channel.close("heartbeat timeout"))

    # ----- connections ---------------------------------------------------------------------

    def allowed(self, ip: str) -> bool:
        try:
            addr = ipaddress.ip_address(ip.split("%")[0])
        except ValueError:
            return False
        if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
            addr = addr.ipv4_mapped
        return any(addr in net for net in self.networks)

    def _fail(self, ip: str, reason: str) -> None:
        self.emit("auth_failed", ip=ip, reason=reason)
        banned_for = self.bans.fail(ip)
        if banned_for is not None:
            log.warning("banned %s for %d s after repeated failed logins", ip, banned_for)
            self.emit("ip_banned", ip=ip, seconds=int(banned_for))

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        peer = writer.get_extra_info("peername")
        ip = str(peer[0]) if isinstance(peer, tuple) else "?"
        if not self.allowed(ip) or self.bans.banned(ip) or self._pending >= self.config.max_nodes:
            writer.close()
            return
        self._pending += 1
        channel: Channel | None = None
        try:
            try:
                hello, channel = await wait_for(self._handshake(reader, writer), self.config.handshake_timeout)
            except TimeoutError:
                self._fail(ip, "handshake timeout")
                return
            except (CryptoError, protocol.ProtocolError, ValueError) as exc:
                self._fail(ip, f"bad handshake ({type(exc).__name__})")
                return
            except (OSError, asyncio.IncompleteReadError):
                return  # closed during the RSA exchange: e.g. an agent refusing our key
            finally:
                self._pending -= 1
            if hello is None:
                return
            node = await self._admit(hello, ip, channel)
            if node is None:
                return
            await self._session(node, channel)
        finally:
            if channel is not None:
                await channel.close()
            else:
                writer.close()

    async def _handshake(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> tuple[dict[str, Any] | None, Channel]:
        crypto = await SessionCrypto.hub_handshake(reader, writer, self.key)
        channel = Channel(reader, writer, crypto)
        channel.limit_frames(PRE_AUTH_MAX_FRAME)
        message = await channel.recv()
        if message is None:
            return None, channel
        if message["type"] != "hello":
            raise protocol.ProtocolError(f"expected hello, got {message['type']}")
        return message, channel

    async def _reject(self, channel: Channel, code: str, reason: str) -> None:
        with contextlib.suppress(Exception):
            await channel.send("reject", code=code, reason=reason)

    async def _admit(self, hello: dict[str, Any], ip: str, channel: Channel) -> Node | None:
        if not check_token(hello["token"], self.token):
            self._fail(ip, "bad token")
            await self._reject(channel, "bad_token", "wrong cluster token (copy OLLAMAPOWER_TOKEN from the hub)")
            return None
        if not protocol.compatible(hello["protocol"]):
            reason = (
                f"protocol {hello['protocol']} doesn't match the hub's {protocol.PROTOCOL_VERSION}: "
                f"install OllamaPower {__version__} on this helper"
            )
            self.registry.mark_incompatible(hello, ip, reason)
            await self._reject(channel, "protocol", reason)
            return None
        known = self.registry.nodes.get(hello["node_id"])
        if len(self.registry.connected) >= self.config.max_nodes and not (known and known.connected):
            await self._reject(channel, "full", f"the hub already has {self.config.max_nodes} helpers")
            return None
        node, replaced = self.registry.attach(hello, ip, channel)
        if replaced is not None:
            self._spawn(replaced.close("replaced by a new connection"))
        channel.limit_frames(MAX_FRAME)
        await channel.send(
            "welcome",
            protocol=protocol.PROTOCOL_VERSION,
            hub_version=__version__,
            node_id=node.node_id,
            heartbeat_interval=HEARTBEAT_INTERVAL,
            heartbeat_timeout=self.config.heartbeat_timeout,
        )
        log.info("helper %s (%s) joined from %s", node.name, node.node_id[:8], ip)
        return node

    async def _session(self, node: Node, channel: Channel) -> None:
        reason = "disconnected"
        try:
            while True:
                message = await channel.recv()
                if message is None:
                    break
                kind = message["type"]
                if kind == "bye":
                    reason = f"left: {message['reason'][:100]}"
                    break
                await self._on_message(node, channel, message)
        except (CryptoError, protocol.ProtocolError, ValueError) as exc:
            reason = f"protocol error ({type(exc).__name__})"
            log.warning("dropping %s: %s", node.name, exc)
        except (OSError, asyncio.IncompleteReadError):
            reason = "connection lost"
        except Exception:  # noqa: BLE001 - a bug must cost one helper's connection, not the hub
            log.exception("unexpected error in the session with %s", node.name)
            reason = "internal error"
        finally:
            self.registry.detach(node, channel, reason)
            if self.on_rpc is not None and not node.connected:
                self.on_rpc(node, None)  # gone: any split session using it is broken

    async def _on_message(self, node: Node, channel: Channel, m: dict[str, Any]) -> None:
        kind = m["type"]
        if kind == "heartbeat":
            self.registry.heartbeat(node, m["telemetry"])
        elif kind == "capabilities":
            self.registry.update_capabilities(node, m["capabilities"])
        elif kind == "lend_state":
            self.registry.set_lend_state(node, m["state"], m.get("reason", ""), m.get("lending"))
            if self.on_node_state is not None:
                self.on_node_state(node)
        elif kind == "net_test_start" and m["kind"] == "request":
            self._spawn(self._agent_requested_test(node, m["test_id"]))
        elif kind == "net_test_data":
            self._on_test_data(m)
        elif kind == "net_test_result":
            test = self._tests.get(m["test_id"])
            if test is not None and not test.result.done():
                test.result.set_result(m["result"])
        elif kind in ("rpc_ready", "rpc_error") and self.on_rpc is not None:
            self.on_rpc(node, m)
        else:
            log.debug("ignoring %s from %s (not used yet)", kind, node.name)

    # ----- network test --------------------------------------------------------------------

    def _on_test_data(self, m: dict[str, Any]) -> None:
        test = self._tests.get(m["test_id"])
        if test is None:
            return
        echo = test.echoes.get(m["seq"])
        if echo is not None and not echo.done() and "data" not in m:
            echo.set_result(time.monotonic())
            return
        data = m.get("data") or b""
        now = time.monotonic()
        if test.up_first is None:
            test.up_first = now
        else:
            test.up_bytes += len(data)  # the first chunk only starts the clock
        test.up_last = now
        test.up_chunks += 1
        if test.up_chunks >= NET_TEST_CHUNKS and not test.up_done.done():
            test.up_done.set_result(None)

    async def net_test(self, node: Node) -> dict[str, Any]:
        """Round-trip time and throughput both ways, over the encrypted control channel.

        Throughput here includes encryption and Python overhead, so it is a
        lower bound for the raw link; the split-mode tunnel is measured on its own.
        """
        channel = node.channel
        if channel is None or channel.closed:
            raise NetTestError(f"{node.name} is not connected")
        test_id = secrets.token_hex(8)
        test = _NetTest()
        self._tests[test_id] = test
        try:
            return await wait_for(self._run_test(node, channel, test_id, test), NET_TEST_TIMEOUT)
        except TimeoutError:
            raise NetTestError(f"{node.name} did not finish the network test in time") from None
        except (OSError, CryptoError) as exc:
            raise NetTestError(f"connection to {node.name} failed: {type(exc).__name__}") from None
        finally:
            self._tests.pop(test_id, None)

    async def _run_test(self, node: Node, channel: Channel, test_id: str, test: _NetTest) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        # 1. Round trips: the agent echoes each packet straight back.
        await channel.send("net_test_start", test_id=test_id, kind="rtt", count=NET_TEST_PINGS)
        rtts = []
        for seq in range(NET_TEST_PINGS):
            test.echoes[seq] = loop.create_future()
            sent = time.monotonic()
            await channel.send("net_test_data", test_id=test_id, seq=seq, sent_at=sent)
            rtts.append((await test.echoes[seq] - sent) * 1000)
        # 2. Hub -> agent: the agent times the chunks and reports.
        await channel.send(
            "net_test_start", test_id=test_id, kind="down", count=NET_TEST_CHUNKS, size=NET_TEST_CHUNK_BYTES
        )
        for seq in range(NET_TEST_CHUNKS):
            await channel.send("net_test_data", test_id=test_id, seq=seq, data=os.urandom(NET_TEST_CHUNK_BYTES))
        down = await test.result
        # 3. Agent -> hub: we time the chunks.
        await channel.send(
            "net_test_start", test_id=test_id, kind="up", count=NET_TEST_CHUNKS, size=NET_TEST_CHUNK_BYTES
        )
        await test.up_done
        up_seconds = (test.up_last or 0) - (test.up_first or 0)
        result = {
            "ts": time.time(),
            "rtt_ms": round(sorted(rtts)[len(rtts) // 2], 2),
            "rtt_min_ms": round(min(rtts), 2),
            "rtt_max_ms": round(max(rtts), 2),
            "down_mbps": round(float(down.get("mbps", 0.0)), 1) if isinstance(down, dict) else None,
            "up_mbps": round(test.up_bytes * 8 / up_seconds / 1e6, 1) if up_seconds > 0 else None,
            "link": node.link.get("kind"),
        }
        self.registry.record_net_test(node, result)
        return result

    async def _agent_requested_test(self, node: Node, request_id: str) -> None:
        try:
            result = await self.net_test(node)
        except NetTestError as exc:
            result = {"error": str(exc)}
        channel = node.channel
        if channel is not None and not channel.closed:
            with contextlib.suppress(Exception):
                await channel.send("net_test_result", test_id=request_id, result=result)

    # ----- owner actions -------------------------------------------------------------------

    async def kick(self, node: Node) -> None:
        channel = node.channel
        if channel is None:
            raise ValueError(f"{node.name} is not connected")
        self.registry.detach(node, channel, "kicked")
        self.emit("node_kicked", node=node.node_id, name=node.name)
        await channel.close("kicked by the hub owner")
