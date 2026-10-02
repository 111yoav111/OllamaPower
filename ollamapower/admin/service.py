"""Hub admin commands, shared by the CLI (over the admin channel) and the hub TUI.

Every owner action goes through ``dispatch`` so there is one code path to
audit; every action that changes something is recorded as an event.
Commands are dotted ``noun.verb`` strings, like OllamaGate's.
"""

from __future__ import annotations

from typing import Any

from ..hub.control_server import NetTestError
from ..hub.nodes import TUNNEL_MODES, Node
from ..hub.runtime import Hub
from .ipc import AdminError

# Estimated overhead over the model file when deciding whether it fits in VRAM
# (KV cache and buffers). Phase 5 replaces this with a real estimate from GGUF metadata.
FIT_MARGIN = 1.2


class Admin:
    def __init__(self, hub: Hub) -> None:
        self.hub = hub

    def _node(self, ref: Any) -> Node:
        node = self.hub.registry.find(str(ref)) if isinstance(ref, str) and ref else None
        if node is None:
            raise AdminError(f"no helper {ref!r} (use its name, id or the first 8 characters of the id)")
        return node

    # ----- views -----------------------------------------------------------------------

    def models(self) -> list[dict[str, Any]]:
        """Installed models with a rough "fits in the hub's VRAM?" verdict."""
        hub_vram = sum(g.get("vram_total", 0) for g in self.hub.local_static.get("gpus", []))
        loaded = {m.name: m for m in self.hub.ollama.models}
        out = []
        for m in self.hub.ollama.installed:
            needed = int(m.size * FIT_MARGIN)
            live = loaded.get(m.name)
            gen = self.hub.last_generation.get(m.name)
            out.append(
                {
                    "name": m.name,
                    "size": m.size,
                    "parameters": m.parameter_size,
                    "quantization": m.quantization,
                    "digest": m.digest,
                    "loaded": live is not None,
                    "vram_percent": live.vram_percent if live else None,
                    "fits": None if not hub_vram else needed <= hub_vram,
                    "plan": "local-ollama",
                    "last_gen_tps": round(gen.gen_tps, 1) if gen else None,
                }
            )
        return out

    # ----- dispatch --------------------------------------------------------------------

    async def dispatch(self, cmd: str, args: dict[str, Any]) -> Any:
        hub, registry = self.hub, self.hub.registry
        if cmd == "status":
            return hub.get_snapshot()
        if cmd == "nodes.list":
            return registry.snapshot()
        if cmd == "nodes.show":
            return self._node(args.get("ref")).detail()
        if cmd in ("nodes.pause", "nodes.resume"):
            node = self._node(args.get("ref"))
            registry.set_hub_paused(node, cmd == "nodes.pause")
            return node.summary()
        if cmd == "nodes.kick":
            node = self._node(args.get("ref"))
            if hub.control is None or not node.connected:
                raise AdminError(f"{node.name} is not connected")
            await hub.control.kick(node)
            return node.summary()
        if cmd == "nodes.net_test":
            node = self._node(args.get("ref"))
            if hub.control is None:
                raise AdminError("the control plane is not running")
            try:
                return await hub.control.net_test(node)
            except NetTestError as exc:
                raise AdminError(str(exc)) from None
        if cmd == "nodes.tunnel":
            node = self._node(args.get("ref"))
            mode = args.get("mode")
            if mode not in TUNNEL_MODES:
                raise AdminError(f"tunnel mode must be {' or '.join(TUNNEL_MODES)}")
            registry.set_tunnel_mode(node, mode)
            return node.summary()
        if cmd == "nodes.forget":
            node = self._node(args.get("ref"))
            try:
                registry.forget(node)
            except ValueError as exc:
                raise AdminError(str(exc)) from None
            return {"forgotten": node.node_id}
        if cmd == "bans.lift":
            if hub.control is None or not hub.control.bans.lift(str(args.get("ip", ""))):
                raise AdminError("no such ban")
            hub.event("ban_lifted", ip=str(args.get("ip")))
            return {"lifted": args.get("ip")}
        if cmd == "models.list":
            return self.models()
        if cmd == "events.list":
            limit = args.get("limit", 100)
            return list(hub.recent_events)[-int(limit) :] if isinstance(limit, int) and limit > 0 else []
        raise AdminError(f"unknown command {cmd!r}")
