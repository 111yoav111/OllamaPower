"""Packet format (ported from DCP's test_packets.py, generic payloads instead of tasks)."""

from __future__ import annotations

import struct

import msgpack
import pytest
import zstandard as zstd

from ollamapower.core import packets as p


def test_task_request_round_trip():
    payload = {"model": "llama3:8b", "n": 3, "blob": b"\x00\x01"}
    [pkt] = p.parse_packets(p.build_task_packet(payload, priority=3))
    assert isinstance(pkt, p.TaskPacket)
    assert pkt.packet_type is p.PACKET_FLAGS.task_request
    assert pkt.priority == 3
    assert pkt.kind is p.PayloadKind.generic
    assert not pkt.is_subtask
    assert pkt.unpack_payload() == payload


def test_result_packets_name_their_request():
    request = p.parse_packet(p.build_task_packet({"q": 1}))
    [pkt] = p.parse_packets(p.build_result_packet({"ok": True}, request_id=request.task_id))
    assert pkt.packet_type is p.PACKET_FLAGS.task_result
    assert pkt.status is p.STATUS_FLAGS.DONE
    assert pkt.parent_id == request.task_id
    raw = p.build_result_packet({"ok": False}, request_id=7, status=p.STATUS_FLAGS.FAILED, subtask_index=1)
    [pkt] = p.parse_packets(raw)
    assert pkt.packet_type is p.PACKET_FLAGS.subtask_result
    assert pkt.status is p.STATUS_FLAGS.FAILED
    assert (pkt.parent_id, pkt.subtask_index) == (7, 1)


def test_subtask_header_fields():
    raw = p.build_subtask_packet({"part": 1}, parent_id=7, subtask_index=1, total_subtasks=2)
    header = p.read_header(raw)
    assert header["packet_type"] is p.PACKET_FLAGS.subtask_request
    assert header["parent_id"] == 7
    assert header["subtask_index"] == 1
    assert header["total_subtasks"] == 2
    assert header["is_subtask"]


def test_wire_ids_are_unique_and_nonzero():
    ids = {p.parse_packet(p.build_task_packet({})).task_id for _ in range(50)}
    assert len(ids) == 50 and p.NO_PARENT not in ids


@pytest.mark.parametrize(
    "raw,flag",
    [
        (p.build_hello(), p.PACKET_FLAGS.ctrl_hello),
        (p.build_welcome(1), p.PACKET_FLAGS.ctrl_welcome),
        (p.build_heartbeat(1), p.PACKET_FLAGS.ctrl_heartbeat),
        (p.build_heartbeat_ack(1), p.PACKET_FLAGS.ctrl_heartbeat_ack),
        (p.build_disconnect(1), p.PACKET_FLAGS.ctrl_disconnect),
        (p.build_status(1, 55), p.PACKET_FLAGS.ctrl_status),
    ],
)
def test_every_control_packet_type(raw, flag):
    assert len(raw) == p.CONTROL_HEADER_SIZE == 12
    [pkt] = p.parse_packets(raw)
    assert pkt.packet_type is flag


def test_control_packet_fields():
    assert p.parse_packet(p.build_welcome(42)).worker_id == 42
    assert p.parse_packet(p.build_status(3, 55)).cpu_percent == 55
    assert p.parse_packet(p.build_status(3, 250)).cpu_percent == 100  # clamped, never overflows the byte
    pkt = p.parse_packet(p.build_disconnect(3, p.DISCONNECT_FLAGS.overloaded))
    assert pkt.disconnect_reason is p.DISCONNECT_FLAGS.overloaded


def test_unknown_disconnect_reason_is_not_fatal():
    raw = struct.pack(p.CONTROL_HEADER_FORMAT, p.PACKET_FLAGS.ctrl_disconnect.value, 0, 99, 1)
    assert p.parse_packet(raw).disconnect_reason is p.DISCONNECT_FLAGS.unknown


def test_stream_of_mixed_packets():
    raw = p.build_hello() + p.build_task_packet({"a": 1}) + p.build_status(1, 10)
    kinds = [type(pkt).__name__ for pkt in p.parse_packets(raw)]
    assert kinds == ["ControlPacket", "TaskPacket", "ControlPacket"]


@pytest.mark.parametrize(
    "raw",
    [
        b"\xff" + b"\x00" * 11,  # unknown flag
        p.build_hello()[:5],  # truncated control packet
        p.build_task_packet({"a": 1})[:10],  # truncated header
        p.build_task_packet({"a": 1})[:-1],  # truncated payload
    ],
)
def test_malformed_input_raises_instead_of_returning_none(raw):
    with pytest.raises(ValueError):
        p.parse_packets(raw)


def test_parse_packet_wants_exactly_one():
    with pytest.raises(ValueError):
        p.parse_packet(p.build_hello() + p.build_hello())


def _packet_with_raw_payload(raw_payload: bytes) -> p.TaskPacket:
    pkt = p.TaskPacket.from_payload({}, p.PACKET_FLAGS.task_request)
    pkt.raw_payload = raw_payload
    return p.parse_packet(pkt.to_bytes())


def test_payload_must_be_a_dict():
    pkt = _packet_with_raw_payload(zstd.ZstdCompressor().compress(msgpack.packb([1, 2, 3])))
    with pytest.raises(ValueError):
        pkt.unpack_payload()


def test_garbage_payload_raises_value_error():
    pkt = _packet_with_raw_payload(b"not zstd at all")
    with pytest.raises(ValueError):
        pkt.unpack_payload()


def test_decompression_bomb_is_refused():
    bomb = zstd.ZstdCompressor().compress(msgpack.packb({"x": b"\x00" * (p.MAX_DECOMPRESSED_BYTES + 1)}))
    assert len(bomb) < 100_000
    pkt = _packet_with_raw_payload(bomb)
    with pytest.raises(ValueError):
        pkt.unpack_payload()


def test_payload_cannot_carry_objects():
    """msgpack ext types (DCP replaced pickle to stop RCE) must not decode into objects."""
    pkt = _packet_with_raw_payload(zstd.ZstdCompressor().compress(msgpack.packb({"x": msgpack.ExtType(5, b"evil")})))
    value = pkt.unpack_payload()
    assert isinstance(value["x"], msgpack.ExtType)  # inert data, never instantiated


def test_control_type_is_not_a_task_packet():
    with pytest.raises(ValueError):
        p.TaskPacket.from_payload({}, p.PACKET_FLAGS.ctrl_hello)


async def test_plaintext_path_accepts_only_control_packets():
    """Before the token is checked, a peer must not be able to make us buffer a payload."""
    import asyncio

    from ollamapower.core.network_io import read_one_packet

    reader = asyncio.StreamReader()
    reader.feed_data(p.build_hello() + p.build_task_packet({"x": 1}))
    reader.feed_eof()
    assert (await read_one_packet(reader)).packet_type is p.PACKET_FLAGS.ctrl_hello
    with pytest.raises(ValueError):
        await read_one_packet(reader)
