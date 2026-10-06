"""The helper's end of the split-mode tunnel: the only way in to ``rpc-server``.

``rpc-server`` listens on ``127.0.0.1``; this relay listens on the interface
the hub connects from and forwards to it, for one session only:

1. The peer's IP must be the hub's (the address of the control connection).
   Anything else is closed before a single byte is read or any TLS work is
   done, so strangers can't make the helper burn CPU on handshakes.
2. **Secure mode:** TLS 1.3 with the session's pinned certificates
   (``core/tunnel.py``); the peer must present exactly the hub's certificate
   for this session. **Direct mode** (both owners opted in): plain TCP.
3. At most ``MAX_CONNECTIONS`` hub connections at once, counted from the
   moment they are accepted (TLS handshakes in progress included) until they
   are fully closed; all are cut when the session ends.

``EchoTarget`` stands in for ``rpc-server`` during a tunnel test: it sends back
what it receives, so the hub can time the real path (network, relay, TLS)
without loading a model.
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import logging
import socket
import ssl
import time

from ..core.tunnel import TLS_HANDSHAKE_TIMEOUT, TunnelError, check_peer, splice
from ..netsock import listen_socket

log = logging.getLogger("ollamapower.agent.tunnel")

MAX_CONNECTIONS = 8


def same_ip(a: str, b: str) -> bool:
    try:
        x, y = ipaddress.ip_address(a), ipaddress.ip_address(b)
    except ValueError:
        return False
    if isinstance(x, ipaddress.IPv6Address) and x.ipv4_mapped:
        x = x.ipv4_mapped
    if isinstance(y, ipaddress.IPv6Address) and y.ipv4_mapped:
        y = y.ipv4_mapped
    return x == y


class HelperRelay:
    def __init__(
        self,
        target_port: int,
        hub_ip: str,
        tls: ssl.SSLContext | None = None,
        hub_fingerprint: str = "",
    ) -> None:
        self.target_port = target_port
        self.hub_ip = hub_ip
        self.tls = tls
        self.hub_fingerprint = hub_fingerprint
        self.listener: socket.socket | None = None
        self.port = 0
        self.connections: set[asyncio.StreamWriter] = set()  # both ends of each relayed connection
        self.active = 0  # hub connections accepted and not finished yet (what MAX_CONNECTIONS limits)
        self.closed = False
        self.refused = 0
        self.last_active = 0.0  # monotonic time of the last byte from the hub (auto-yield needs it)
        self._accepting: asyncio.Task[None] | None = None
        self._handlers: set[asyncio.Task[None]] = set()

    @property
    def mode(self) -> str:
        return "secure" if self.tls is not None else "direct"

    async def start(self, bind_ip: str) -> int:
        # Our own accept loop instead of asyncio.start_server: the peer's address is
        # checked on the raw socket, and TLS starts on a socket nobody has read from yet.
        listener = listen_socket(bind_ip, 0, backlog=MAX_CONNECTIONS)  # exclusive on Windows
        listener.setblocking(False)
        self.listener = listener
        self.port = listener.getsockname()[1]
        self._accepting = asyncio.ensure_future(self._accept_loop())
        return self.port

    async def _accept_loop(self) -> None:
        loop = asyncio.get_running_loop()
        assert self.listener is not None
        while not self.closed:
            try:
                conn, peer = await loop.sock_accept(self.listener)
            except OSError:
                if self.closed:
                    return
                await asyncio.sleep(0.05)
                continue
            if not same_ip(peer[0], self.hub_ip) or self.closed or self.active >= MAX_CONNECTIONS:
                self.refused += 1
                log.warning("relay: refused a connection from %s", peer[0])
                conn.close()
                continue
            self.active += 1
            task = asyncio.ensure_future(self._serve(conn))
            self._handlers.add(task)
            task.add_done_callback(lambda t, conn=conn: self._finished(t, conn))

    def _finished(self, task: asyncio.Task[None], conn: socket.socket) -> None:
        """Runs however ``_serve`` ended, even if it was cancelled before it started."""
        self._handlers.discard(task)
        self.active -= 1
        if task.cancelled():
            conn.close()  # idempotent; covers a task cancelled before it ever ran

    async def _serve(self, conn: socket.socket) -> None:
        loop = asyncio.get_running_loop()
        writer: asyncio.StreamWriter | None = None
        upstream: asyncio.StreamWriter | None = None
        try:
            reader = asyncio.StreamReader(limit=2**20)
            protocol = asyncio.StreamReaderProtocol(reader)
            transport, _ = await loop.connect_accepted_socket(
                lambda: protocol,
                conn,
                ssl=self.tls,
                ssl_handshake_timeout=TLS_HANDSHAKE_TIMEOUT if self.tls is not None else None,
            )
            writer = asyncio.StreamWriter(transport, protocol, reader, loop)
            self.connections.add(writer)
            if self.tls is not None:
                check_peer(writer, self.hub_fingerprint)
            up_reader, upstream = await asyncio.open_connection("127.0.0.1", self.target_port)
            self.connections.add(upstream)
            await splice((reader, writer), (up_reader, upstream), self._count)
        except (OSError, TimeoutError, ssl.SSLError, TunnelError) as exc:
            log.warning("relay: dropped a connection from the hub's address: %s", type(exc).__name__)
        finally:
            if writer is None:
                conn.close()
            for w in (writer, upstream):
                if w is not None:
                    self.connections.discard(w)
                    w.close()

    def _count(self, n: int) -> None:
        self.last_active = time.monotonic()

    def close(self) -> None:
        self.closed = True
        if self._accepting is not None:
            self._accepting.cancel()
        if self.listener is not None:
            self.listener.close()
        for w in list(self.connections):
            w.close()
        for task in list(self._handlers):
            task.cancel()


class EchoTarget:
    """A loopback echo server: the far end of a tunnel test."""

    def __init__(self) -> None:
        self.server: asyncio.Server | None = None
        self.port = 0

    async def start(self) -> int:
        async def echo(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            with contextlib.suppress(OSError, ConnectionError):
                while data := await reader.read(256 * 1024):
                    writer.write(data)
                    await writer.drain()
            writer.close()

        self.server = await asyncio.start_server(echo, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]
        return self.port

    def close(self) -> None:
        if self.server is not None:
            self.server.close()
