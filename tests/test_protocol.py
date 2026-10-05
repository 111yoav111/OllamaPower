"""Every Section 11 message type, direction rules and validation."""

from __future__ import annotations

import asyncio

import pytest

from ollamapower.core import protocol as p
from ollamapower.core.channel import Channel
from ollamapower.core.encrypt_layer import SessionCrypto
from ollamapower.core.packets import parse_packet

NODE_ID = "a1b2c3d4-e5f6"

# One valid example per message type: (sender, fields).
EXAMPLES: dict[str, tuple[str, dict]] = {
    "hello": (
        "node",
        {"protocol": "1.0", "agent_version": "0.1.0", "node_id": NODE_ID, "name": "desk", "token": "op_x"},
    ),
    "welcome": (
        "hub",
        {
            "protocol": "1.0",
            "hub_version": "0.1.0",
            "node_id": NODE_ID,
            "heartbeat_interval": 2.0,
            "heartbeat_timeout": 10,
        },
    ),
    "reject": ("hub", {"code": "bad_token", "reason": "wrong token"}),
    "bye": ("hub", {"reason": "kicked"}),
    "capabilities": ("node", {"capabilities": {"os": "Linux", "gpus": [{"name": "RTX 4090", "vram_total": 24}]}}),
    "heartbeat": ("node", {"seq": 3, "telemetry": {"cpu_percent": 12.5}}),
    "lend_state": ("node", {"state": "paused", "reason": "owner", "lending": {"max_vram_gb": 8}}),
    "net_test_start": ("hub", {"test_id": "t1", "kind": "rtt", "count": 10, "size": 0}),
    "net_test_data": ("node", {"test_id": "t1", "seq": 1, "sent_at": 1.5, "data": b"\x00" * 10}),
    "net_test_result": ("node", {"test_id": "t1", "result": {"mbps": 940.0}}),
    "rpc_start": (
        "hub",
        {
            "session_id": "session-1",
            "device": "CUDA0",
            "vram_cap_mb": 8000,
            "cache": True,
            "tunnel": {"mode": "direct"},
        },
    ),
    "rpc_ready": ("node", {"session_id": "session-1", "endpoint": {"port": 50052}}),
    "rpc_error": ("node", {"session_id": "session-1", "error": "out of memory"}),
    "rpc_stop": ("hub", {"session_id": "session-1"}),
}


def test_every_type_has_an_example():
    assert set(EXAMPLES) == set(p.SCHEMAS)


@pytest.mark.parametrize("kind", sorted(EXAMPLES))
def test_round_trip(kind):
    sender, fields = EXAMPLES[kind]
    message = p.make(kind, sender, **fields)
    assert p.decode(parse_packet(p.encode(message)), sender) == {"type": kind, **fields}


@pytest.mark.parametrize("kind", sorted(k for k, (d, _) in p.SCHEMAS.items() if d is not p.Direction.BOTH))
def test_wrong_direction_is_refused(kind):
    sender, fields = EXAMPLES[kind]
    other = "hub" if sender == "node" else "node"
    with pytest.raises(p.ProtocolError, match="may not be sent"):
        p.validate({"type": kind, **fields}, other)


@pytest.mark.parametrize(
    "message",
    [
        "not a map",
        {"type": "nope"},
        {"type": 5},
        {"type": "heartbeat", "seq": 1},  # missing telemetry
        {"type": "heartbeat", "seq": True, "telemetry": {}},  # bool is not an int
        {"type": "heartbeat", "seq": -1, "telemetry": {}},
        {"type": "heartbeat", "seq": "1", "telemetry": {}},
        {"type": "lend_state", "state": "gone"},
        {"type": "net_test_start", "test_id": "t", "kind": "rtt", "count": 100000},
        {"type": "net_test_start", "test_id": "t", "kind": "flood"},
        {"type": "hello", **{**EXAMPLES["hello"][1], "node_id": "../../etc"}},
        {"type": "hello", **{**EXAMPLES["hello"][1], "token": "x" * 1000}},
        {"type": "capabilities", "capabilities": {}, "extra": 1} | {"capabilities": "x"},
    ],
)
def test_invalid_messages(message):
    with pytest.raises(p.ProtocolError):
        p.validate(message, "node")


def test_unknown_fields_are_dropped():
    m = p.validate({"type": "rpc_stop", "session_id": "session-1", "future_field": 1}, "hub")
    assert m == {"type": "rpc_stop", "session_id": "session-1"}


def test_version_compatibility():
    assert p.compatible("1.0") and p.compatible("1.7")
    assert not p.compatible("2.0") and not p.compatible("garbage")


def test_non_message_packet_is_refused():
    from ollamapower.core.packets import build_task_packet

    with pytest.raises(p.ProtocolError):
        p.decode(parse_packet(build_task_packet({"type": "rpc_stop", "session_id": "session-1"})), "hub")


async def test_channel_round_trip_and_eof():
    key = b"k" * 32
    a_reader, b_reader = asyncio.StreamReader(), asyncio.StreamReader()

    class Pipe:
        def __init__(self, target):
            self.target = target

        def write(self, data):
            self.target.feed_data(data)

        async def drain(self):
            pass

        def close(self):
            self.target.feed_eof()

        async def wait_closed(self):
            pass

        def get_extra_info(self, name):
            return ("127.0.0.1", 1)

    hub = Channel(a_reader, Pipe(b_reader), SessionCrypto(key, "hub"))
    node = Channel(b_reader, Pipe(a_reader), SessionCrypto(key, "node"))
    await node.send("heartbeat", seq=1, telemetry={"cpu_percent": 1.0})
    assert (await hub.recv())["seq"] == 1
    await hub.send("rpc_stop", session_id="session-1")
    assert (await node.recv())["type"] == "rpc_stop"
    with pytest.raises(p.ProtocolError):
        await node.send("rpc_stop", session_id="session-1")  # agents may not send rpc_stop
    await hub.close("kicked")
    assert (await node.recv()) == {"type": "bye", "reason": "kicked"}
    assert await node.recv() is None


async def test_pre_auth_frame_limit():
    from ollamapower.core.encrypt_layer import CryptoError

    key = b"k" * 32
    reader = asyncio.StreamReader()
    big = SessionCrypto(key, "node").encrypt(b"x" * 100_000)
    reader.feed_data(big)
    hub = Channel(reader, None, SessionCrypto(key, "hub"))  # type: ignore[arg-type]
    hub.limit_frames(64 * 1024)
    with pytest.raises(CryptoError):
        await hub.recv()


async def test_bomb_message_before_auth_is_refused_by_the_channel():
    import zstandard as zstd

    from ollamapower.core.packets import PACKET_FLAGS, PayloadKind, TaskPacket

    key = b"k" * 32
    pkt = TaskPacket.from_payload({}, PACKET_FLAGS.message, kind=PayloadKind.message)
    pkt.raw_payload = zstd.ZstdCompressor(level=19).compress(b"\xdd" + (10**7).to_bytes(4, "big") + b"\x80" * 10**7)
    reader = asyncio.StreamReader()
    reader.feed_data(SessionCrypto(key, "node").encrypt(pkt.to_bytes()))
    hub = Channel(reader, None, SessionCrypto(key, "hub"))  # type: ignore[arg-type]
    hub.limit_frames(64 * 1024)
    assert hub.max_payload == 64 * 1024
    with pytest.raises(p.ProtocolError):
        await hub.recv()


async def test_large_messages_after_auth_still_decode():
    key = b"k" * 32
    reader = asyncio.StreamReader()
    message = p.make("net_test_data", "node", test_id="t1", seq=0, data=b"\x01" * (4 * 1024 * 1024))
    reader.feed_data(SessionCrypto(key, "node").encrypt(p.encode(message)))
    hub = Channel(reader, None, SessionCrypto(key, "hub"))  # type: ignore[arg-type]
    hub.limit_frames(30 * 1024 * 1024)
    assert (await hub.recv())["data"] == message["data"]


@pytest.mark.parametrize("purpose,ok", [("rpc", True), ("bench", True), ("shell", False)])
def test_rpc_start_purpose(purpose, ok):
    m = {
        "type": "rpc_start",
        "session_id": "session-1",
        "device": "",
        "vram_cap_mb": 0,
        "cache": False,
        "tunnel": {"mode": "secure", "cert": "x"},
        "purpose": purpose,
    }
    if ok:
        assert p.validate(m, "hub")["purpose"] == purpose
    else:
        with pytest.raises(p.ProtocolError):
            p.validate(m, "hub")


def test_protocol_minor():
    assert (p.protocol_minor("1.1"), p.protocol_minor("1.0"), p.protocol_minor("2"), p.protocol_minor("x.y")) == (
        1,
        0,
        -1,
        -1,
    )
