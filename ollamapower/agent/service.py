"""Agent admin commands, used by ``ollamapower agent pause|resume|status`` and the agent TUI."""

from __future__ import annotations

from typing import Any

from ..admin.ipc import AdminError
from ..config import ConfigError
from .client import AgentClient

RULE_KEYS = ("max_vram_gb", "max_vram_percent", "gpus", "allow_cpu", "allow_overflow", "allow_direct", "schedule")


def agent_snapshot(client: AgentClient) -> dict[str, Any]:
    s = client.state
    return {
        "connection": s.connection,
        "hub": s.hub,
        "fingerprint": s.fingerprint,
        "pinned": s.pinned,
        "node_id": s.node_id,
        "name": client.config.display_name,
        "error": s.error,
        "paused": s.paused,
        "connected_since": s.connected_since,
        "lending": client.lending_dict(),
        "lent": None,  # what the hub uses right now: split sessions arrive in Phase 3
        "hardware": s.static,
        "telemetry": s.telemetry,
        "network": s.network,
        "last_net_test": s.last_net_test,
        "events": [{"ts": ts, "message": msg} for ts, msg in list(s.events)[-50:]],
    }


class AgentAdmin:
    def __init__(self, client: AgentClient) -> None:
        self.client = client

    async def dispatch(self, cmd: str, args: dict[str, Any]) -> Any:
        if cmd == "agent.status":
            return agent_snapshot(self.client)
        if cmd == "agent.pause":
            await self.client.set_paused(True)
            return agent_snapshot(self.client)
        if cmd == "agent.resume":
            await self.client.set_paused(False)
            return agent_snapshot(self.client)
        if cmd == "agent.net_test":
            try:
                return await self.client.request_net_test()
            except (ConnectionError, TimeoutError) as exc:
                raise AdminError(str(exc) or "the network test timed out") from None
        if cmd == "agent.rules":
            changes = {k: v for k, v in args.items() if k in RULE_KEYS}
            if len(changes) != len(args):
                raise AdminError(f"unknown rule; use {', '.join(RULE_KEYS)}")
            try:
                return await self.client.update_rules(**changes)
            except (ConfigError, TypeError) as exc:
                raise AdminError(str(exc)) from None
        raise AdminError(f"unknown command {cmd!r}")
