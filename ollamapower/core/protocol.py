"""Control-plane messages (spec Section 11), carried in DCP's framing.

Every message is a msgpack dict ``{"type": ..., <fields>}`` inside a
``TaskPacket`` (``PACKET_FLAGS.message``), inside one AES-GCM frame. This
module is the single source of truth for which messages exist, which way they
may travel and what their fields look like; nothing else builds or parses them.

Rules:

* ``PROTOCOL_VERSION`` is ``"MAJOR.MINOR"``. Peers with different majors
  refuse each other with a clear message; minors only add optional fields.
  Major 2 changed the handshake (a hub nonce in the session key, see
  ``encrypt_layer``); the messages themselves are those of 1.1.
* A message of the wrong direction, of an unknown type, missing a required
  field or with a field of the wrong type is a ``ProtocolError`` - the
  connection is dropped. Unknown *fields* are dropped silently (a newer minor
  may send them).
* Strings are capped; free-form dict fields (capabilities, telemetry, ...) are
  bounded by the frame size and sanitised by whoever displays them.
"""

from __future__ import annotations

import enum
from typing import Any

from .packets import MAX_DECOMPRESSED_BYTES, PACKET_FLAGS, PayloadKind, TaskPacket

PROTOCOL_VERSION = "2.0"
MAX_STRING = 4096
MAX_TOKEN = 512


class ProtocolError(ValueError):
    """A peer sent something the protocol doesn't allow."""


class Direction(enum.Enum):
    TO_HUB = "agent->hub"
    TO_AGENT = "hub->agent"
    BOTH = "both"


def protocol_major(version: str) -> int:
    try:
        return int(str(version).split(".", 1)[0])
    except ValueError:
        return -1


def protocol_minor(version: str) -> int:
    try:
        return int(str(version).split(".", 1)[1])
    except (IndexError, ValueError):
        return -1


def at_least(version: str, major: int, minor: int) -> bool:
    """Is ``version`` ``major.minor`` or newer?"""
    return (protocol_major(version), protocol_minor(version)) >= (major, minor)


def compatible(version: str) -> bool:
    return protocol_major(version) == protocol_major(PROTOCOL_VERSION)


# field name -> (accepted python types, required)
Field = tuple[tuple[type, ...], bool]
STR, INT, NUM, BOOL, DICT, BYTES = (str,), (int,), (int, float), (bool,), (dict,), (bytes,)

SCHEMAS: dict[str, tuple[Direction, dict[str, Field]]] = {
    # --- connection ---
    "hello": (
        Direction.TO_HUB,
        {
            "protocol": (STR, True),
            "agent_version": (STR, True),
            "node_id": (STR, True),
            "name": (STR, True),
            "token": (STR, True),
        },
    ),
    "welcome": (
        Direction.TO_AGENT,
        {
            "protocol": (STR, True),
            "hub_version": (STR, True),
            "node_id": (STR, True),
            "heartbeat_interval": (NUM, True),
            "heartbeat_timeout": (NUM, True),
        },
    ),
    "reject": (Direction.TO_AGENT, {"code": (STR, True), "reason": (STR, True)}),
    "bye": (Direction.BOTH, {"reason": (STR, True)}),
    # --- state ---
    "capabilities": (Direction.TO_HUB, {"capabilities": (DICT, True)}),
    "heartbeat": (Direction.TO_HUB, {"seq": (INT, True), "telemetry": (DICT, True)}),
    "lend_state": (
        Direction.TO_HUB,
        {"state": (STR, True), "reason": (STR, False), "lending": (DICT, False)},
    ),
    # --- network test ---
    "net_test_start": (
        Direction.BOTH,
        {"test_id": (STR, True), "kind": (STR, True), "count": (INT, False), "size": (INT, False)},
    ),
    "net_test_data": (
        Direction.BOTH,
        {"test_id": (STR, True), "seq": (INT, True), "sent_at": (NUM, False), "data": (BYTES, False)},
    ),
    "net_test_result": (Direction.BOTH, {"test_id": (STR, True), "result": (DICT, True)}),
    # --- split mode (Phase 3) ---
    "rpc_start": (
        Direction.TO_AGENT,
        {
            "session_id": (STR, True),
            "device": (STR, True),
            "vram_cap_mb": (INT, True),
            "cache": (BOOL, True),
            "tunnel": (DICT, True),
            "model": (STR, False),
            "purpose": (STR, False),  # "rpc" (default) or "bench" (a tunnel test; added in 1.1)
            "ram": (BOOL, False),  # lend RAM beside the GPU when the owner allows it (added in 2.1)
        },
    ),
    "rpc_ready": (Direction.TO_HUB, {"session_id": (STR, True), "endpoint": (DICT, False)}),
    "rpc_error": (Direction.TO_HUB, {"session_id": (STR, True), "error": (STR, True)}),
    "rpc_stop": (Direction.TO_AGENT, {"session_id": (STR, True)}),
}

LEND_STATES = ("lending", "paused", "yielding")
NET_TEST_KINDS = ("request", "rtt", "down", "up")
TUNNEL_MODES = ("secure", "direct")
REJECT_CODES = ("bad_token", "protocol", "banned", "not_allowed", "duplicate", "full", "malformed")

# Messages hub -> agent and agent -> hub, for direction checks.
_ROLE_SENDS = {"hub": (Direction.TO_AGENT, Direction.BOTH), "node": (Direction.TO_HUB, Direction.BOTH)}


def _check_type(value: Any, types: tuple[type, ...]) -> bool:
    if isinstance(value, bool) and bool not in types:
        return False  # True is an int in Python; never accept it as a number
    return isinstance(value, types)


def validate(message: Any, sender: str) -> dict[str, Any]:
    """Check ``message`` as sent by ``sender`` (``"hub"`` or ``"node"``); return a clean copy."""
    if not isinstance(message, dict):
        raise ProtocolError("message must be a map")
    kind = message.get("type")
    if not isinstance(kind, str) or kind not in SCHEMAS:
        raise ProtocolError(f"unknown message type {str(kind)[:40]!r}")
    direction, fields = SCHEMAS[kind]
    if direction not in _ROLE_SENDS[sender]:
        raise ProtocolError(f"{kind} may not be sent by the {sender}")
    clean: dict[str, Any] = {"type": kind}
    for name, (types, required) in fields.items():
        if name not in message or message[name] is None:
            if required:
                raise ProtocolError(f"{kind}: missing {name}")
            continue
        value = message[name]
        if not _check_type(value, types):
            raise ProtocolError(f"{kind}: {name} has the wrong type")
        if isinstance(value, str) and len(value) > (MAX_TOKEN if name == "token" else MAX_STRING):
            raise ProtocolError(f"{kind}: {name} is too long")
        clean[name] = value
    _check_values(clean)
    return clean


def _check_values(m: dict[str, Any]) -> None:
    kind = m["type"]
    if kind == "lend_state" and m["state"] not in LEND_STATES:
        raise ProtocolError(f"lend_state: unknown state {m['state'][:20]!r}")
    if kind == "net_test_start":
        if m["kind"] not in NET_TEST_KINDS:
            raise ProtocolError(f"net_test_start: unknown kind {m['kind'][:20]!r}")
        if not 0 <= m.get("count", 0) <= 1000 or not 0 <= m.get("size", 0) <= 4 * 1024 * 1024:
            raise ProtocolError("net_test_start: count or size out of range")
    if kind == "reject" and m["code"] not in REJECT_CODES:
        raise ProtocolError("reject: unknown code")
    if kind == "hello" and not valid_node_id(m["node_id"]):
        raise ProtocolError("hello: invalid node_id")
    if kind in ("heartbeat", "net_test_data") and m["seq"] < 0:
        raise ProtocolError(f"{kind}: negative seq")
    if kind.startswith("rpc_") and not valid_node_id(m["session_id"]):
        raise ProtocolError(f"{kind}: invalid session_id")
    if kind == "rpc_start":
        if not 0 <= m["vram_cap_mb"] <= 16 * 1024 * 1024 or len(m["device"]) > 64:
            raise ProtocolError("rpc_start: vram_cap_mb or device out of range")
        if m["tunnel"].get("mode") not in TUNNEL_MODES:
            raise ProtocolError("rpc_start: unknown tunnel mode")
        if m.get("purpose", "rpc") not in ("rpc", "bench"):
            raise ProtocolError("rpc_start: unknown purpose")


def valid_node_id(value: str) -> bool:
    """Node ids are generated by the agent: 8-64 characters of ``[A-Za-z0-9-]``."""
    return 8 <= len(value) <= 64 and all(c.isascii() and (c.isalnum() or c == "-") for c in value)


def make(msg_type: str, sender: str, /, **fields: Any) -> dict[str, Any]:
    """Build and validate an outgoing message."""
    return validate({**fields, "type": msg_type}, sender)


def encode(message: dict[str, Any]) -> bytes:
    return TaskPacket.from_payload(message, PACKET_FLAGS.message, kind=PayloadKind.message).to_bytes()


def decode(packet: Any, sender: str, max_payload: int = MAX_DECOMPRESSED_BYTES) -> dict[str, Any]:
    """A received packet as a validated message from ``sender``, decompressing at most ``max_payload`` bytes."""
    if not isinstance(packet, TaskPacket) or packet.packet_type is not PACKET_FLAGS.message:
        raise ProtocolError("expected a message packet")
    try:
        payload = packet.unpack_payload(max_payload)
    except ValueError as exc:
        raise ProtocolError(str(exc)) from None
    return validate(payload, sender)
