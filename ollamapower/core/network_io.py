"""Reading and writing packets on an asyncio stream, inherited from DCP.

With a ``SessionCrypto`` every packet is exactly one encrypted frame. Without
one - only the hello/welcome exchange before the token is checked, as in DCP -
just the fixed-size 12-byte control packets are accepted, so an unauthenticated
peer can't make us buffer a large payload.

All writes to one connection go through a shared ``asyncio.Lock`` so frames
written from different tasks never interleave.
"""

from __future__ import annotations

import asyncio
import logging

from .encrypt_layer import SessionCrypto
from .packets import (
    CONTROL_HEADER_SIZE,
    CTRL_VALUES_SET,
    DISCONNECT_FLAGS,
    ControlPacket,
    Packet,
    build_disconnect,
    build_heartbeat,
    build_hello,
    build_status,
    build_welcome,
    parse_packet,
)

log = logging.getLogger("ollamapower.net")


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
    if first[0] not in CTRL_VALUES_SET:
        raise ValueError(f"only control packets are allowed before encryption (got flag 0x{first[0]:02X})")
    try:
        pkt = ControlPacket.from_bytes(first + await reader.readexactly(CONTROL_HEADER_SIZE - 1))
    except asyncio.IncompleteReadError:
        raise ValueError("connection closed inside a packet") from None
    log.debug("recv %r", pkt)
    return pkt


async def send_packet(
    writer: asyncio.StreamWriter, lock: asyncio.Lock, data: bytes, crypto: SessionCrypto | None = None
) -> None:
    """Write one already-built packet, encrypted when the session has a key.

    Encryption happens under the lock so frames hit the wire in counter order.
    """
    async with lock:
        writer.write(crypto.encrypt(data) if crypto is not None else data)
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
