"""The hub's view of its helpers (DCP's worker table, grown up).

Each helper is a ``Node``, keyed by the stable id its agent generated. A node
is known once it has authenticated; it stays known (offline) after it leaves,
and its name, capabilities, lending rules and the owner's per-node settings
(hub-side pause, tunnel mode) are kept in ``state.json``.

Everything an agent reports is untrusted: ``sanitize`` bounds its size and
shape and makes strings terminal-safe before anything stores or shows it.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..core.channel import Channel
from ..state import StateStore
from ..textsafe import safe_text

HISTORY_SAMPLES = 300  # 10 minutes of 2-second heartbeats, for sparklines
NET_TEST_HISTORY = 20
TUNNEL_MODES = ("secure", "direct")
GIB = 1024**3

Emit = Callable[..., None]


def sanitize(value: Any, depth: int = 0) -> Any:
    """Bound an untrusted JSON-ish value: depth 6, 64 keys/items, 200-char strings, finite numbers."""
    if depth > 6:
        return None
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value if -(2**63) <= value < 2**63 else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return safe_text(value, 200)
    if isinstance(value, dict):
        return {safe_text(str(k), 64): sanitize(v, depth + 1) for k, v in list(value.items())[:64]}
    if isinstance(value, (list, tuple)):
        return [sanitize(v, depth + 1) for v in list(value)[:64]]
    return None


def _num(value: Any, default: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else default


@dataclass
class Node:
    node_id: str
    name: str
    ip: str = ""
    agent_version: str = ""
    protocol: str = ""
    capabilities: dict[str, Any] = field(default_factory=dict)
    telemetry: dict[str, Any] = field(default_factory=dict)
    lending: dict[str, Any] = field(default_factory=dict)
    lend_state: str = "lending"
    lend_reason: str = ""
    hub_paused: bool = False
    tunnel_mode: str = "secure"
    incompatible: str = ""
    first_seen: float = field(default_factory=time.time)
    last_seen: float | None = None
    connected_at: float | None = None
    last_heartbeat: float = field(default_factory=time.monotonic)
    history: deque[tuple[float, float | None, int, float]] = field(
        default_factory=lambda: deque(maxlen=HISTORY_SAMPLES)
    )
    net_tests: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=NET_TEST_HISTORY))
    channel: Channel | None = None

    @property
    def connected(self) -> bool:
        return self.channel is not None and not self.channel.closed

    @property
    def status(self) -> str:
        """``lending`` · ``paused`` · ``yielding`` · ``offline`` · ``incompatible``."""
        if self.incompatible:
            return "incompatible"
        if not self.connected:
            return "offline"
        if self.hub_paused:
            return "paused"
        return self.lend_state

    @property
    def link(self) -> dict[str, Any]:
        net = self.capabilities.get("network") if isinstance(self.capabilities.get("network"), dict) else {}
        link = {"kind": net.get("kind", "unknown"), "speed_mbps": net.get("speed_mbps")}
        if self.net_tests:
            last = self.net_tests[-1]
            link.update(rtt_ms=last.get("rtt_ms"), down_mbps=last.get("down_mbps"), up_mbps=last.get("up_mbps"))
        return link

    def gpus(self) -> list[dict[str, Any]]:
        """Static GPU facts merged with the latest telemetry."""
        static = self.capabilities.get("gpus") if isinstance(self.capabilities.get("gpus"), list) else []
        live = {g.get("index"): g for g in self.telemetry.get("gpus", []) if isinstance(g, dict)}
        out = []
        for g in static:
            if not isinstance(g, dict):
                continue
            merged = {**g, **live.get(g.get("index"), {})}
            total = int(_num(merged.get("vram_total")))
            used = int(_num(merged.get("vram_used")))
            merged["vram_free"] = max(0, total - used) if merged.get("vram_used") is not None else total
            out.append(merged)
        return out

    def lendable_vram(self) -> int:
        """Bytes of VRAM the helper's owner allows us to use right now."""
        if self.status != "lending":
            return 0
        allowed = self.lending.get("gpus") or []
        gpus = [g for g in self.gpus() if not allowed or g.get("index") in allowed]
        free = sum(g["vram_free"] for g in gpus)
        total = sum(int(_num(g.get("vram_total"))) for g in gpus)
        cap_gb, cap_pct = _num(self.lending.get("max_vram_gb")), _num(self.lending.get("max_vram_percent"))
        cap = free
        if cap_gb > 0:
            cap = min(cap, int(cap_gb * GIB))
        if 0 < cap_pct <= 100:
            cap = min(cap, int(total * cap_pct / 100))
        return max(0, cap)

    @property
    def tunnel_effective(self) -> str:
        """Direct mode needs both the hub owner and the helper owner to opt in."""
        if self.tunnel_mode == "direct" and self.lending.get("allow_direct") is True:
            return "direct"
        return "secure"

    def warnings(self) -> list[str]:
        out = []
        if self.incompatible:
            out.append(self.incompatible)
        if self.link["kind"] == "wifi":
            out.append(f"{self.name} is on Wi-Fi: too slow for split mode; use a cable.")
        if self.tunnel_effective == "direct":
            out.append(f"{self.name} uses trusted-LAN direct mode: split-mode traffic is unencrypted.")
        elif self.tunnel_mode == "direct":
            out.append(f"{self.name}: direct mode requested, but the helper's owner hasn't allowed it (secure used).")
        return out

    def summary(self) -> dict[str, Any]:
        tel = self.telemetry
        gpus = self.gpus()
        return {
            "id": self.node_id,
            "name": self.name,
            "ip": self.ip,
            "status": self.status,
            "connected": self.connected,
            "agent_version": self.agent_version,
            "protocol": self.protocol,
            "os": self.capabilities.get("os", ""),
            "gpus": [
                {
                    k: g.get(k)
                    for k in ("index", "name", "backend", "vram_total", "vram_used", "vram_free", "util", "temp")
                }
                for g in gpus
            ],
            "vram_total": sum(int(_num(g.get("vram_total"))) for g in gpus),
            "vram_free": sum(g["vram_free"] for g in gpus),
            "vram_lendable": self.lendable_vram(),
            "cpu_percent": tel.get("cpu_percent"),
            "ram_total": tel.get("ram_total") or self.capabilities.get("ram_total"),
            "ram_used": tel.get("ram_used"),
            "link": self.link,
            "lend_state": self.lend_state,
            "lend_reason": self.lend_reason,
            "lending": self.lending,
            "hub_paused": self.hub_paused,
            "tunnel_mode": self.tunnel_mode,
            "tunnel_effective": self.tunnel_effective,
            "role": "idle",  # split / overflow roles arrive with Phases 3 and 6
            "first_seen": self.first_seen,
            "last_seen": self.last_seen if not self.connected else time.time(),
            "connected_at": self.connected_at,
            "warnings": self.warnings(),
        }

    def detail(self) -> dict[str, Any]:
        return {
            **self.summary(),
            "capabilities": self.capabilities,
            "history": [list(h) for h in self.history],
            "net_tests": list(self.net_tests),
        }


class NodeRegistry:
    def __init__(self, state: StateStore | None, emit: Emit, heartbeat_timeout: float) -> None:
        self.state = state
        self.emit = emit
        self.heartbeat_timeout = heartbeat_timeout
        self.nodes: dict[str, Node] = {}
        self.version = 0  # bumped on every change, for cheap TUI refreshes
        if state is not None:
            self._load(state.data.get("nodes", {}))

    # ----- persistence ---------------------------------------------------------------------

    def _load(self, saved: dict[str, Any]) -> None:
        for node_id, d in saved.items():
            if not isinstance(d, dict):
                continue
            node = Node(node_id=node_id, name=safe_text(str(d.get("name", node_id[:8])), 64))
            node.ip = safe_text(str(d.get("last_ip", "")), 64)
            node.capabilities = sanitize(d.get("capabilities") or {}) or {}
            node.lending = sanitize(d.get("lending") or {}) or {}
            node.hub_paused = d.get("hub_paused") is True
            node.tunnel_mode = d.get("tunnel_mode") if d.get("tunnel_mode") in TUNNEL_MODES else "secure"
            node.first_seen = _num(d.get("first_seen"), time.time())
            node.last_seen = _num(d.get("last_seen")) or None
            if isinstance(d.get("last_net_test"), dict):
                node.net_tests.append(sanitize(d["last_net_test"]))
            self.nodes[node_id] = node

    def _save(self, node: Node) -> None:
        self.version += 1
        if self.state is None:
            return
        self.state.upsert_node(
            node.node_id,
            name=node.name,
            first_seen=node.first_seen,
            last_seen=node.last_seen,
            last_ip=node.ip,
            capabilities=node.capabilities,
            lending=node.lending,
            hub_paused=node.hub_paused,
            tunnel_mode=node.tunnel_mode,
            last_net_test=node.net_tests[-1] if node.net_tests else None,
        )

    # ----- lookup --------------------------------------------------------------------------

    def find(self, ref: str) -> Node | None:
        """By id, unique id prefix (≥ 4 chars) or exact name (case-insensitive)."""
        if ref in self.nodes:
            return self.nodes[ref]
        by_name = [n for n in self.nodes.values() if n.name.lower() == ref.lower()]
        if len(by_name) == 1:
            return by_name[0]
        by_prefix = [n for n in self.nodes.values() if len(ref) >= 4 and n.node_id.startswith(ref)]
        return by_prefix[0] if len(by_prefix) == 1 else None

    @property
    def connected(self) -> list[Node]:
        return [n for n in self.nodes.values() if n.connected]

    # ----- lifecycle -----------------------------------------------------------------------

    def attach(self, hello: dict[str, Any], ip: str, channel: Channel) -> tuple[Node, Channel | None]:
        """A node authenticated. Returns it and the connection it replaces (to be closed), if any."""
        node_id = hello["node_id"]
        node = self.nodes.get(node_id) or Node(node_id=node_id, name="")
        old = node.channel if node.connected else None
        node.name = safe_text(hello["name"], 64) or node_id[:8]
        node.ip = ip
        node.agent_version = safe_text(hello["agent_version"], 32)
        node.protocol = safe_text(hello["protocol"], 16)
        node.incompatible = ""
        node.channel = channel
        node.connected_at = time.time()
        node.last_seen = time.time()
        node.last_heartbeat = time.monotonic()
        node.lend_state, node.lend_reason = "lending", ""
        self.nodes[node_id] = node
        self._save(node)
        self.emit("node_join", node=node_id, name=node.name, ip=ip, agent_version=node.agent_version)
        return node, old

    def mark_incompatible(self, hello: dict[str, Any], ip: str, reason: str) -> None:
        node = self.nodes.get(hello["node_id"]) or Node(node_id=hello["node_id"], name="")
        node.name = safe_text(hello["name"], 64) or node.node_id[:8]
        node.ip, node.agent_version = ip, safe_text(hello["agent_version"], 32)
        node.protocol, node.incompatible = safe_text(hello["protocol"], 16), reason
        node.last_seen = time.time()
        self.nodes[node.node_id] = node
        self._save(node)
        self.emit("node_incompatible", node=node.node_id, name=node.name, ip=ip, reason=reason)

    def detach(self, node: Node, channel: Channel, reason: str) -> None:
        if node.channel is not channel:
            return  # already replaced by a newer connection
        node.channel = None
        node.last_seen = time.time()
        self._save(node)
        self.emit("node_leave", node=node.node_id, name=node.name, reason=reason)

    def update_capabilities(self, node: Node, caps: dict[str, Any]) -> None:
        node.capabilities = sanitize(caps) or {}
        lending = node.capabilities.pop("lending", None)
        if isinstance(lending, dict):
            node.lending = lending
        self._save(node)

    def heartbeat(self, node: Node, telemetry: dict[str, Any]) -> None:
        node.telemetry = sanitize(telemetry) or {}
        node.last_heartbeat = time.monotonic()
        node.last_seen = time.time()
        gpus = node.gpus()
        utils = [_num(g.get("util"), -1) for g in gpus if g.get("util") is not None]
        node.history.append(
            (
                time.time(),
                max(utils) if utils else None,
                sum(int(_num(g.get("vram_used"))) for g in gpus),
                _num(node.telemetry.get("cpu_percent")),
            )
        )
        self.version += 1

    def set_lend_state(self, node: Node, state: str, reason: str, lending: dict[str, Any] | None) -> None:
        changed = state != node.lend_state
        node.lend_state, node.lend_reason = state, safe_text(reason, 200)
        if lending is not None:
            node.lending = sanitize(lending) or {}
        self._save(node)
        if changed:
            self.emit("lend_state", node=node.node_id, name=node.name, state=state, reason=node.lend_reason)

    def timed_out(self, now: float | None = None) -> list[Node]:
        now = time.monotonic() if now is None else now
        return [n for n in self.connected if now - n.last_heartbeat > self.heartbeat_timeout]

    # ----- owner actions -------------------------------------------------------------------

    def set_hub_paused(self, node: Node, paused: bool) -> None:
        node.hub_paused = paused
        self._save(node)
        self.emit("node_paused" if paused else "node_resumed", node=node.node_id, name=node.name)

    def set_tunnel_mode(self, node: Node, mode: str) -> None:
        if mode not in TUNNEL_MODES:
            raise ValueError(f"tunnel mode must be one of {', '.join(TUNNEL_MODES)}")
        node.tunnel_mode = mode
        self._save(node)
        self.emit("node_tunnel", node=node.node_id, name=node.name, mode=mode)

    def record_net_test(self, node: Node, result: dict[str, Any]) -> None:
        node.net_tests.append(sanitize(result))
        self._save(node)
        self.emit(
            "net_test",
            node=node.node_id,
            name=node.name,
            **{k: result.get(k) for k in ("rtt_ms", "down_mbps", "up_mbps")},
        )

    def forget(self, node: Node) -> None:
        if node.connected:
            raise ValueError("disconnect the node first")
        del self.nodes[node.node_id]
        self.version += 1
        if self.state is not None:
            self.state.data["nodes"].pop(node.node_id, None)
            self.state.save_soon()
        self.emit("node_forgotten", node=node.node_id, name=node.name)

    def snapshot(self) -> list[dict[str, Any]]:
        order = {"lending": 0, "yielding": 1, "paused": 2, "incompatible": 3, "offline": 4}
        nodes = sorted(self.nodes.values(), key=lambda n: (order.get(n.status, 9), n.name.lower()))
        return [n.summary() for n in nodes]


async def wait_closed(channel: Channel | None, reason: str) -> None:
    if channel is not None:
        await asyncio.shield(channel.close(reason))
