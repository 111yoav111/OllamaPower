"""An authenticated, encrypted message pipe between the hub and one agent.

Wraps a stream pair and a ``SessionCrypto``: ``send`` validates and encrypts a
protocol message, ``recv`` decrypts and validates the next one. Writes are
serialised by one lock so frames stay whole and in counter order.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from . import protocol
from .encrypt_layer import SessionCrypto
from .network_io import read_one_packet, send_packet

# Until the peer has proven itself, frames bigger than this are refused.
PRE_AUTH_MAX_FRAME = 64 * 1024


class Channel:
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, crypto: SessionCrypto) -> None:
        self.reader = reader
        self.writer = writer
        self.crypto = crypto
        self.role = crypto.role
        self.peer_role = crypto.peer_role
        self._lock = asyncio.Lock()
        self.closed = False

    @property
    def peer(self) -> tuple[str, int] | None:
        info = self.writer.get_extra_info("peername")
        return (str(info[0]), int(info[1])) if isinstance(info, tuple) and len(info) >= 2 else None

    @property
    def local(self) -> tuple[str, int] | None:
        info = self.writer.get_extra_info("sockname")
        return (str(info[0]), int(info[1])) if isinstance(info, tuple) and len(info) >= 2 else None

    def limit_frames(self, max_frame: int) -> None:
        self.crypto.max_frame = max_frame

    async def send(self, msg_type: str, /, **fields: Any) -> None:
        message = protocol.make(msg_type, self.role, **fields)
        await send_packet(self.writer, self._lock, protocol.encode(message), self.crypto)

    async def recv(self) -> dict[str, Any] | None:
        """The next message, or ``None`` when the peer closed the connection cleanly."""
        packet = await read_one_packet(self.reader, self.crypto)
        if packet is None:
            return None
        return protocol.decode(packet, self.peer_role)

    async def close(self, reason: str | None = None) -> None:
        if self.closed:
            return
        self.closed = True
        if reason is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self.send("bye", reason=reason[: protocol.MAX_STRING]), 2)
        self.writer.close()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(self.writer.wait_closed(), 2)
