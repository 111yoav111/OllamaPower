"""An authenticated, encrypted message pipe between the hub and one agent.

Wraps a stream pair and a ``SessionCrypto``: ``send`` validates and encrypts a
protocol message, ``recv`` decrypts and validates the next one. Writes are
serialised by one lock so frames stay whole and in counter order.

What a peer can make us spend on one message is bounded by ``limit_frames``:
before the peer has authenticated, both the encrypted frame and the
*decompressed* payload are capped at 64 KB (a few hundred compressed bytes
could otherwise expand into millions of objects on the hub's event loop).
Afterwards payloads may be larger, so they are decoded in a worker thread and
a slow decode can't stall every other connection.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from ..aio import wait_for
from . import protocol
from .encrypt_layer import SessionCrypto
from .network_io import read_one_packet, send_packet

# Until the peer has proven itself, frames (and decompressed payloads) bigger than this are refused.
PRE_AUTH_MAX_FRAME = 64 * 1024
# Decompressed size of one message from an authenticated peer: 4x the largest legitimate one
# (a 4 MiB network-test chunk), far below what would hurt the hub.
MAX_PAYLOAD = 16 * 1024 * 1024
BYE_GRACE = 1.0  # seconds a peer gets to hang up after our bye


class Channel:
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, crypto: SessionCrypto) -> None:
        self.reader = reader
        self.writer = writer
        self.crypto = crypto
        self.role = crypto.role
        self.peer_role = crypto.peer_role
        self._lock = asyncio.Lock()
        self.closed = False
        self.max_payload = MAX_PAYLOAD

    @property
    def peer(self) -> tuple[str, int] | None:
        info = self.writer.get_extra_info("peername")
        return (str(info[0]), int(info[1])) if isinstance(info, tuple) and len(info) >= 2 else None

    @property
    def local(self) -> tuple[str, int] | None:
        info = self.writer.get_extra_info("sockname")
        return (str(info[0]), int(info[1])) if isinstance(info, tuple) and len(info) >= 2 else None

    def limit_frames(self, max_frame: int) -> None:
        """Cap incoming frames at ``max_frame`` bytes and their decompressed payloads likewise (at most MAX_PAYLOAD)."""
        self.crypto.max_frame = max_frame
        self.max_payload = min(max_frame, MAX_PAYLOAD)

    async def send(self, msg_type: str, /, **fields: Any) -> None:
        message = protocol.make(msg_type, self.role, **fields)
        await send_packet(self.writer, self._lock, protocol.encode(message), self.crypto)

    async def recv(self) -> dict[str, Any] | None:
        """The next message, or ``None`` when the peer closed the connection cleanly."""
        packet = await read_one_packet(self.reader, self.crypto)
        if packet is None:
            return None
        if self.max_payload <= PRE_AUTH_MAX_FRAME:
            return protocol.decode(packet, self.peer_role, self.max_payload)  # small enough to do inline
        return await asyncio.to_thread(protocol.decode, packet, self.peer_role, self.max_payload)

    async def close(self, reason: str | None = None) -> None:
        if self.closed:
            return
        self.closed = True
        if reason is not None:
            with contextlib.suppress(Exception):
                await wait_for(self.send("bye", reason=reason[: protocol.MAX_STRING]), 2)
                # Half-close and give the peer a moment to read the bye and hang up: closing
                # at once can turn its next write into a reset that discards the bye unread.
                if self.writer.can_write_eof():
                    self.writer.write_eof()
                deadline = asyncio.get_running_loop().time() + BYE_GRACE
                while not self.reader.at_eof() and asyncio.get_running_loop().time() < deadline:
                    await asyncio.sleep(0.02)
        self.writer.close()
        with contextlib.suppress(Exception):
            await wait_for(self.writer.wait_closed(), 2)
