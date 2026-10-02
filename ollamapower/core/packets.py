"""Binary packet format of the control plane, inherited from DCP.

Two packet shapes share one stream, told apart by their first byte:

* ``ControlPacket`` - always 12 bytes (``!B B H Q``: type, flags, extra, node id).
  Connection housekeeping: hello, welcome, heartbeat, status, disconnect.
* ``TaskPacket`` - a 28-byte header (``!B B B B I Q Q H H``) followed by a
  ``zstd(msgpack(dict))`` payload. The header can be read without touching
  the payload, so routing decisions stay cheap.

Payloads are plain msgpack dicts - never pickled objects - so a peer can't
make us run code by sending a crafted packet. On an encrypted session each
packet travels inside one AES-GCM frame (see ``encrypt_layer.py``).

Phase 2 adds a protocol version and the packet types of spec Section 11.
"""

from __future__ import annotations

import enum
import itertools
import struct
from typing import Any

import msgpack
import zstandard as zstd


class PACKET_FLAGS(enum.Enum):  # noqa: N801 - DCP's wire-level name, kept recognisable
    task_request = 1  # hub asks a node to do something
    task_result = 2  # node answers
    task_ack = 3
    task_fail = 4
    subtask_request = 5
    subtask_result = 6

    ctrl_hello = 10  # node connecting
    ctrl_welcome = 11  # hub accepted the node and assigns its id
    ctrl_heartbeat = 12
    ctrl_heartbeat_ack = 13
    ctrl_disconnect = 14
    ctrl_status = 15  # node load report


class PayloadKind(enum.Enum):
    """What a TaskPacket payload carries. DCP's render/simulation kinds are gone."""

    generic = 1


class STATUS_FLAGS(enum.Enum):  # noqa: N801 - wire-level name
    CREATED = 0
    RUNNING = 1
    DONE = 2
    FAILED = 3


class DISCONNECT_FLAGS(enum.Enum):  # noqa: N801 - wire-level name
    clean = 0
    timeout = 1  # heartbeats stopped
    task_crash = 2
    overloaded = 3  # the node is leaving because it is busy
    unknown = 10


TASK_HEADER_FORMAT = "!B B B B I Q Q H H"
TASK_HEADER_SIZE = struct.calcsize(TASK_HEADER_FORMAT)

CONTROL_HEADER_FORMAT = "!B B H Q"
CONTROL_HEADER_SIZE = struct.calcsize(CONTROL_HEADER_FORMAT)

ZSTD_COMPRESS_LEVEL = 3
# A decompressed payload larger than this is refused (zstd bomb protection).
MAX_DECOMPRESSED_BYTES = 64 * 1024 * 1024

NO_PARENT = 0
NO_SUBTASK = 0xFFFF

CTRL_VALUES_SET = frozenset(f.value for f in PACKET_FLAGS if f.name.startswith("ctrl_"))
TASK_VALUES_SET = frozenset(f.value for f in PACKET_FLAGS if not f.name.startswith("ctrl_"))

# Wire-level packet ids; 0 is reserved for "no parent".
_id_counter = itertools.count(1)


def next_id() -> int:
    return next(_id_counter)


def _u8(value: int, name: str) -> int:
    if not 0 <= value <= 0xFF:
        raise ValueError(f"{name} must fit in one byte (got {value})")
    return value


class TaskPacket:
    """A header plus a compressed msgpack payload."""

    __slots__ = (
        "packet_type",
        "kind",
        "status",
        "priority",
        "task_id",
        "parent_id",
        "subtask_index",
        "total_subtasks",
        "raw_payload",
    )

    packet_type: PACKET_FLAGS
    kind: PayloadKind
    status: STATUS_FLAGS
    priority: int
    task_id: int
    parent_id: int | None
    subtask_index: int | None
    total_subtasks: int | None
    raw_payload: bytes

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        packet_type: PACKET_FLAGS,
        kind: PayloadKind = PayloadKind.generic,
        status: STATUS_FLAGS = STATUS_FLAGS.CREATED,
        priority: int = 1,
        parent_id: int | None = None,
        subtask_index: int | None = None,
        total_subtasks: int | None = None,
    ) -> TaskPacket:
        """Build a packet on the sending side; assigns a fresh wire id."""
        if packet_type.value not in TASK_VALUES_SET:
            raise ValueError(f"{packet_type.name} is not a task packet type")
        if not isinstance(payload, dict):
            raise TypeError("payload must be a dict")
        pkt = cls.__new__(cls)
        pkt.packet_type = packet_type
        pkt.kind = kind
        pkt.status = status
        pkt.priority = max(0, min(255, int(priority)))
        pkt.task_id = next_id()
        pkt.parent_id = parent_id
        pkt.subtask_index = subtask_index
        pkt.total_subtasks = total_subtasks
        raw = msgpack.packb(payload, use_bin_type=True)
        pkt.raw_payload = zstd.ZstdCompressor(level=ZSTD_COMPRESS_LEVEL).compress(raw)
        return pkt

    @classmethod
    def from_bytes(cls, data: bytes) -> TaskPacket:
        """Parse one packet on the receiving side. The payload stays compressed."""
        if len(data) < TASK_HEADER_SIZE:
            raise ValueError("data too short for a TaskPacket header")
        (
            packet_type,
            kind,
            status,
            priority,
            payload_size,
            task_id,
            parent_id,
            subtask_index,
            total_subtasks,
        ) = struct.unpack(TASK_HEADER_FORMAT, data[:TASK_HEADER_SIZE])
        if packet_type not in TASK_VALUES_SET:
            raise ValueError(f"not a task packet type: {packet_type}")
        end = TASK_HEADER_SIZE + payload_size
        if len(data) < end:
            raise ValueError("data too short for the TaskPacket payload")
        pkt = cls.__new__(cls)
        try:
            pkt.packet_type = PACKET_FLAGS(packet_type)
            pkt.kind = PayloadKind(kind)
            pkt.status = STATUS_FLAGS(status)
        except ValueError as exc:
            raise ValueError(f"invalid TaskPacket header: {exc}") from None
        pkt.priority = priority
        pkt.task_id = task_id
        pkt.parent_id = parent_id if parent_id != NO_PARENT else None
        pkt.subtask_index = subtask_index if subtask_index != NO_SUBTASK else None
        pkt.total_subtasks = total_subtasks if total_subtasks != NO_SUBTASK else None
        pkt.raw_payload = data[TASK_HEADER_SIZE:end]
        return pkt

    def unpack_payload(self) -> dict[str, Any]:
        """Decompress and decode the payload; refuses anything but a dict."""
        try:
            # Stream with a hard cap: ``decompress(max_output_size=...)`` trusts the
            # size declared in the frame header, which an attacker controls.
            with zstd.ZstdDecompressor().stream_reader(self.raw_payload) as reader:
                raw = reader.read(MAX_DECOMPRESSED_BYTES + 1)
            if len(raw) > MAX_DECOMPRESSED_BYTES:
                raise ValueError("payload too large")
            value = msgpack.unpackb(raw, raw=False, strict_map_key=True)
        except (zstd.ZstdError, ValueError, msgpack.ExtraData, msgpack.FormatError, msgpack.StackError) as exc:
            raise ValueError(f"invalid TaskPacket payload: {type(exc).__name__}") from None
        if not isinstance(value, dict):
            raise ValueError("TaskPacket payload must be a dict")
        return value

    def to_bytes(self) -> bytes:
        header = struct.pack(
            TASK_HEADER_FORMAT,
            self.packet_type.value,
            self.kind.value,
            self.status.value,
            _u8(self.priority, "priority"),
            len(self.raw_payload),
            self.task_id,
            self.parent_id if self.parent_id is not None else NO_PARENT,
            self.subtask_index if self.subtask_index is not None else NO_SUBTASK,
            self.total_subtasks if self.total_subtasks is not None else NO_SUBTASK,
        )
        return header + self.raw_payload

    @property
    def is_subtask(self) -> bool:
        return self.subtask_index is not None

    @property
    def payload_size(self) -> int:
        return len(self.raw_payload)

    def __repr__(self) -> str:
        return (
            f"TaskPacket(type={self.packet_type.name}, kind={self.kind.name}, status={self.status.name}, "
            f"priority={self.priority}, id={self.task_id})"
        )


class ControlPacket:
    """Fixed 12-byte connection-control packet."""

    __slots__ = ("packet_type", "packet_flags", "extra_info", "worker_id")

    def __init__(self, packet_type: PACKET_FLAGS, packet_flags: int, extra_info: int, worker_id: int) -> None:
        self.packet_type = packet_type
        self.packet_flags = packet_flags
        self.extra_info = extra_info
        self.worker_id = worker_id  # the node id; DCP's field name is kept on the wire

    @classmethod
    def hello(cls) -> ControlPacket:
        """Node -> hub: start of a connection, asking for an id."""
        return cls(PACKET_FLAGS.ctrl_hello, 0, 0, 0)

    @classmethod
    def welcome(cls, assigned_id: int) -> ControlPacket:
        """Hub -> node: accepted, here is your id."""
        return cls(PACKET_FLAGS.ctrl_welcome, 0, 0, assigned_id)

    @classmethod
    def heartbeat(cls, node_id: int) -> ControlPacket:
        return cls(PACKET_FLAGS.ctrl_heartbeat, 0, 0, node_id)

    @classmethod
    def heartbeat_ack(cls, node_id: int) -> ControlPacket:
        return cls(PACKET_FLAGS.ctrl_heartbeat_ack, 0, 0, node_id)

    @classmethod
    def disconnect(cls, node_id: int, reason: DISCONNECT_FLAGS = DISCONNECT_FLAGS.clean) -> ControlPacket:
        return cls(PACKET_FLAGS.ctrl_disconnect, 0, reason.value, node_id)

    @classmethod
    def status(cls, node_id: int, cpu_percent: int) -> ControlPacket:
        """Node -> hub: current CPU load in percent (0-100)."""
        return cls(PACKET_FLAGS.ctrl_status, max(0, min(100, int(cpu_percent))), 0, node_id)

    def to_bytes(self) -> bytes:
        return struct.pack(
            CONTROL_HEADER_FORMAT, self.packet_type.value, self.packet_flags, self.extra_info, self.worker_id
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> ControlPacket:
        if len(data) != CONTROL_HEADER_SIZE:
            raise ValueError(f"a ControlPacket is exactly {CONTROL_HEADER_SIZE} bytes (got {len(data)})")
        packet_type, packet_flags, extra_info, worker_id = struct.unpack(CONTROL_HEADER_FORMAT, data)
        if packet_type not in CTRL_VALUES_SET:
            raise ValueError(f"not a control packet type: {packet_type}")
        return cls(PACKET_FLAGS(packet_type), packet_flags, extra_info, worker_id)

    @property
    def cpu_percent(self) -> int:
        return self.packet_flags

    @property
    def disconnect_reason(self) -> DISCONNECT_FLAGS:
        try:
            return DISCONNECT_FLAGS(self.extra_info)
        except ValueError:
            return DISCONNECT_FLAGS.unknown

    def __repr__(self) -> str:
        if self.packet_type is PACKET_FLAGS.ctrl_disconnect:
            info = f"reason={self.disconnect_reason.name}"
        elif self.packet_type is PACKET_FLAGS.ctrl_status:
            info = f"cpu={self.cpu_percent}"
        else:
            info = f"flags={self.packet_flags}, extra={self.extra_info}"
        return f"ControlPacket(type={self.packet_type.name}, id={self.worker_id}, {info})"


Packet = ControlPacket | TaskPacket


# ----- builders ------------------------------------------------------------------------


def build_task_packet(payload: dict[str, Any], priority: int = 1, kind: PayloadKind = PayloadKind.generic) -> bytes:
    """Hub -> node: a request."""
    return TaskPacket.from_payload(payload, PACKET_FLAGS.task_request, kind=kind, priority=priority).to_bytes()


def build_subtask_packet(
    payload: dict[str, Any],
    parent_id: int,
    subtask_index: int,
    total_subtasks: int,
    priority: int = 1,
    kind: PayloadKind = PayloadKind.generic,
) -> bytes:
    """Hub -> node: one part of a larger request."""
    return TaskPacket.from_payload(
        payload,
        PACKET_FLAGS.subtask_request,
        kind=kind,
        priority=priority,
        parent_id=parent_id,
        subtask_index=subtask_index,
        total_subtasks=total_subtasks,
    ).to_bytes()


def build_result_packet(
    payload: dict[str, Any],
    request_id: int,
    status: STATUS_FLAGS = STATUS_FLAGS.DONE,
    subtask_index: int | None = None,
    kind: PayloadKind = PayloadKind.generic,
) -> bytes:
    """Node -> hub: the answer to the (sub)task request with wire id ``request_id``.

    The request id travels in the ``parent_id`` header field, so the hub can
    match a result to its request without opening the payload.
    """
    packet_type = PACKET_FLAGS.subtask_result if subtask_index is not None else PACKET_FLAGS.task_result
    return TaskPacket.from_payload(
        payload, packet_type, kind=kind, status=status, parent_id=request_id, subtask_index=subtask_index
    ).to_bytes()


def build_hello() -> bytes:
    return ControlPacket.hello().to_bytes()


def build_welcome(node_id: int) -> bytes:
    return ControlPacket.welcome(node_id).to_bytes()


def build_heartbeat(node_id: int) -> bytes:
    return ControlPacket.heartbeat(node_id).to_bytes()


def build_heartbeat_ack(node_id: int) -> bytes:
    return ControlPacket.heartbeat_ack(node_id).to_bytes()


def build_disconnect(node_id: int, reason: DISCONNECT_FLAGS = DISCONNECT_FLAGS.clean) -> bytes:
    return ControlPacket.disconnect(node_id, reason).to_bytes()


def build_status(node_id: int, cpu_percent: int) -> bytes:
    return ControlPacket.status(node_id, cpu_percent).to_bytes()


# ----- parsing -------------------------------------------------------------------------


def parse_packet(data: bytes) -> Packet:
    """Parse exactly one complete packet (e.g. the plaintext of one encrypted frame)."""
    packets = parse_packets(data)
    if len(packets) != 1:
        raise ValueError(f"expected one packet, found {len(packets)}")
    return packets[0]


def parse_packets(data: bytes) -> list[Packet]:
    """Split a byte buffer into packets. Raises ``ValueError`` on anything malformed."""
    packets: list[Packet] = []
    cursor = 0
    while cursor < len(data):
        flag = data[cursor]
        if flag in CTRL_VALUES_SET:
            end = cursor + CONTROL_HEADER_SIZE
            if len(data) < end:
                raise ValueError(f"truncated ControlPacket at byte {cursor}")
            packets.append(ControlPacket.from_bytes(data[cursor:end]))
        elif flag in TASK_VALUES_SET:
            if len(data) - cursor < TASK_HEADER_SIZE:
                raise ValueError(f"truncated TaskPacket header at byte {cursor}")
            payload_size = struct.unpack("!I", data[cursor + 4 : cursor + 8])[0]
            end = cursor + TASK_HEADER_SIZE + payload_size
            if len(data) < end:
                raise ValueError(f"truncated TaskPacket payload at byte {cursor}")
            packets.append(TaskPacket.from_bytes(data[cursor:end]))
        else:
            raise ValueError(f"unknown packet flag 0x{flag:02X} at byte {cursor}")
        cursor = end
    return packets


def read_header(data: bytes) -> dict[str, Any]:
    """Decode only a TaskPacket's header (no decompression)."""
    pkt = TaskPacket.from_bytes(data)
    return {
        "packet_type": pkt.packet_type,
        "kind": pkt.kind,
        "status": pkt.status,
        "priority": pkt.priority,
        "task_id": pkt.task_id,
        "parent_id": pkt.parent_id,
        "subtask_index": pkt.subtask_index,
        "total_subtasks": pkt.total_subtasks,
        "payload_size": pkt.payload_size,
        "is_subtask": pkt.is_subtask,
    }
