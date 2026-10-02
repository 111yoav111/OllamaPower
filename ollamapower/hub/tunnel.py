"""The hub's end of the split-mode tunnel, and the tunnel test.

``llama-server --rpc`` speaks plain TCP, so in secure mode each helper gets a
``HubRelay``: a listener on the hub's ``127.0.0.1`` that llama-server connects
to, which opens a TLS 1.3 connection to the helper's relay for every
connection it accepts (llama.cpp opens several short ones, then one long one).
The hub trusts only the certificate the helper sent for this session. In
direct mode no hub relay is needed: llama-server connects to the helper's
relay itself.

``measure`` times the same path for ``ollamapower nodes tunnel-test``: round
trips of small messages (what RPC does once or more per token) and bulk
throughput (what loading a model's weights does), through the helper's relay
to an echo server. Running it in both modes shows what encryption costs.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import ssl
import statistics
import time
from typing import Any

from ..core.tunnel import (
    TLS_HANDSHAKE_TIMEOUT,
    TunnelError,
    TunnelIdentity,
    cert_fingerprint,
    check_peer,
    nodelay,
    same_user_on_loopback,
    splice,
)

log = logging.getLogger("ollamapower.tunnel")

CONNECT_TIMEOUT = 10.0
MAX_LOCAL_CONNECTIONS = 16
RTT_ROUNDS = 200
RTT_SIZE = 64
BULK_BYTES = 64 * 1024 * 1024
BULK_CHUNK = 256 * 1024


class TunnelClient:
    """How the hub connects to one helper's relay: TLS with pinned certificates, or plain TCP (direct)."""

    def __init__(self, host: str, port: int, identity: TunnelIdentity | None = None, helper_cert: str = "") -> None:
        self.host, self.port = host, port
        self.ctx = identity.context(server=False, peer_cert_pem=helper_cert) if identity else None
        self.fingerprint = cert_fingerprint(helper_cert) if identity else ""

    @property
    def mode(self) -> str:
        return "direct" if self.ctx is None else "secure"

    async def open(self) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        return await open_tunnel(self.host, self.port, self.ctx, self.fingerprint)


async def open_tunnel(
    host: str, port: int, ctx: ssl.SSLContext | None, fingerprint: str = ""
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """One connection to a helper's relay."""
    if ctx is None:
        reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), CONNECT_TIMEOUT)
        nodelay(writer)
        return reader, writer
    reader, writer = await asyncio.wait_for(
        asyncio.open_connection(host, port, ssl=ctx, server_hostname="", ssl_handshake_timeout=TLS_HANDSHAKE_TIMEOUT),
        CONNECT_TIMEOUT + TLS_HANDSHAKE_TIMEOUT,
    )
    try:
        check_peer(writer, fingerprint)
    except TunnelError:
        writer.close()
        raise
    nodelay(writer)
    return reader, writer


class HubRelay:
    """``127.0.0.1:<port>`` -> TLS -> the helper's relay, for llama-server's ``--rpc``."""

    def __init__(self, client: TunnelClient) -> None:
        self.client = client
        self.server: asyncio.Server | None = None
        self.port = 0
        self.connections: set[asyncio.StreamWriter] = set()
        self.failures = 0
        self.refused = 0
        self.bytes_out = 0  # hub -> helper: mostly model weights while loading
        self.last_error = ""

    async def start(self) -> int:
        self.server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]
        return self.port

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        # Only llama-server, i.e. our own user, may use the tunnel: another local user
        # would otherwise reach the helper's unauthenticated rpc-server through it.
        if len(self.connections) >= MAX_LOCAL_CONNECTIONS or not same_user_on_loopback(writer):
            self.refused += 1
            log.warning("tunnel: refused a local connection that isn't from llama-server's user")
            writer.close()
            return
        self.connections.add(writer)
        remote: asyncio.StreamWriter | None = None
        try:
            r_reader, remote = await self.client.open()
            self.connections.add(remote)
            await splice((reader, writer), (r_reader, remote), self._count)
        except (OSError, TimeoutError, ssl.SSLError, TunnelError) as exc:
            self.failures += 1
            self.last_error = type(exc).__name__
            log.warning("tunnel to %s failed: %s", self.client.host, type(exc).__name__)
        finally:
            for w in (writer, remote):
                if w is not None:
                    self.connections.discard(w)
                    w.close()

    def _count(self, n: int) -> None:
        self.bytes_out += n

    def close(self) -> None:
        if self.server is not None:
            self.server.close()
        for w in list(self.connections):
            w.close()


async def measure(client: TunnelClient, bulk_bytes: int = BULK_BYTES) -> dict[str, Any]:
    """Round-trip time and echo throughput through a helper's relay (a ``bench`` session)."""
    started = time.monotonic()
    reader, writer = await client.open()
    connect_ms = (time.monotonic() - started) * 1000
    try:
        payload = os.urandom(RTT_SIZE)
        rtts = []
        for _ in range(RTT_ROUNDS):
            t = time.perf_counter()
            writer.write(payload)
            await writer.drain()
            await reader.readexactly(RTT_SIZE)
            rtts.append((time.perf_counter() - t) * 1000)

        chunk = os.urandom(BULK_CHUNK)

        async def send() -> None:
            for _ in range(max(1, bulk_bytes // BULK_CHUNK)):
                writer.write(chunk)
                await writer.drain()

        async def receive() -> None:
            left = max(1, bulk_bytes // BULK_CHUNK) * BULK_CHUNK
            while left:
                data = await reader.read(min(left, 1024 * 1024))
                if not data:
                    raise ConnectionError("the tunnel closed during the test")
                left -= len(data)

        t = time.perf_counter()
        await asyncio.gather(send(), receive())
        seconds = time.perf_counter() - t
    finally:
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()
    rtts.sort()
    return {
        "mode": client.mode,
        "connect_ms": round(connect_ms, 2),
        "rtt_ms": round(statistics.median(rtts), 3),
        "rtt_p95_ms": round(rtts[int(len(rtts) * 0.95) - 1], 3),
        # The echo carries every byte both ways at once; this is the one-way rate.
        "mbps": round(max(1, bulk_bytes // BULK_CHUNK) * BULK_CHUNK * 8 / seconds / 1e6, 1),
        "bytes": max(1, bulk_bytes // BULK_CHUNK) * BULK_CHUNK,
    }
