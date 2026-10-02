"""Reading and writing packets on an asyncio stream, inherited from DCP.

With a ``SessionCrypto`` every packet is exactly one encrypted frame. Without
one (only the hello/welcome exchange during the handshake, as in DCP) packets
are read incrementally: the first byte tells the packet type and therefore how
many more bytes to read.

All writes to one connection go through a shared ``asyncio.Lock`` so frames
written from different tasks never interleave.
"""

from __future__ import annotations

import asyncio
import logging
import struct

from .encrypt_layer import SessionCrypto
from .packets import (
    CONTROL_HEADER_SIZE,
    CTRL_VALUES_SET,
    DISCONNECT_FLAGS,
    TASK_HEADER_SIZE,
    TASK_VALUES_SET,
    ControlPacket,
    Packet,
    TaskPacket,
    build_disconnect,
    build_heartbeat,
    build_hello,
    build_status,
    build_welcome,
    parse_packet,
)

log = logging.getLogger("ollamapower.net")

MAX_PAYLOAD_BYTES = 64 * 1024 * 1024  # plaintext path sanity cap


async def read_one_packet(reader: asyncio.StreamReader, crypto: SessionCrypto | None = None) -> Packet | None:
    """Read exactly one packet. ``None`` on a clean end of stream; ``ValueError`` on bad data."""
    if crypto is not None:
        plaintext = await crypto.decrypt_from_reader(reader)
        if plaintext is None:
            return None
        pkt = parse_packet(plaintext)
        log.debug("recv %r", pkt)
        return pkt

    try:
        first = await reader.readexactly(1)
    except asyncio.IncompleteReadError:
        return None
    flag = first[0]
    try:
        if flag in CTRL_VALUES_SET:
            rest = await reader.readexactly(CONTROL_HEADER_SIZE - 1)
            pkt: Packet = ControlPacket.from_bytes(first + rest)
        elif flag in TASK_VALUES_SET:
            header = first + await reader.readexactly(TASK_HEADER_SIZE - 1)
            payload_size = struct.unpack("!I", header[4:8])[0]
            if payload_size > MAX_PAYLOAD_BYTES:
                raise ValueError(f"TaskPacket payload too large ({payload_size} bytes)")
            pkt = TaskPacket.from_bytes(header + await reader.readexactly(payload_size))
        else:
            raise ValueError(f"unknown packet flag 0x{flag:02X}")
    except asyncio.IncompleteReadError:
        raise ValueError("connection closed inside a packet") from None
    log.debug("recv %r", pkt)
    return pkt


async def send_packet(
    writer: asyncio.StreamWriter, lock: asyncio.Lock, data: bytes, crypto: SessionCrypto | None = None
) -> None:
    """Write one already-built packet, encrypted when the session has a key."""
    if crypto is not None:
        data = crypto.encrypt(data)
    async with lock:
        writer.write(data)
        await writer.drain()


# ----- control helpers ------------------------------------------------------------------


async def net_send_hello(writer: asyncio.StreamWriter, lock: asyncio.Lock) -> None:
    """Node -> hub. Plaintext: it carries nothing and precedes the token."""
    await send_packet(writer, lock, build_hello())


async def net_send_welcome(writer: asyncio.StreamWriter, lock: asyncio.Lock, node_id: int) -> None:
    """Hub -> node, after the token checked out. Plaintext: carries only the id."""
    await send_packet(writer, lock, build_welcome(node_id))


async def net_send_heartbeat(
    writer: asyncio.StreamWriter, lock: asyncio.Lock, node_id: int, crypto: SessionCrypto | None = None
) -> None:
    await send_packet(writer, lock, build_heartbeat(node_id), crypto)


async def net_send_status(
    writer: asyncio.StreamWriter, lock: asyncio.Lock, node_id: int, cpu: int, crypto: SessionCrypto | None = None
) -> None:
    await send_packet(writer, lock, build_status(node_id, cpu), crypto)


async def net_send_disconnect(
    writer: asyncio.StreamWriter,
    lock: asyncio.Lock,
    node_id: int,
    reason: DISCONNECT_FLAGS = DISCONNECT_FLAGS.clean,
    crypto: SessionCrypto | None = None,
) -> None:
    await send_packet(writer, lock, build_disconnect(node_id, reason), crypto)
    log.info("sent disconnect to node %d (%s)", node_id, reason.name)
