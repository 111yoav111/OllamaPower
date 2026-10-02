"""Admin commands, shared by the CLI (over the admin channel) and, from Phase 2, the TUI.

Every owner action goes through ``dispatch`` so there is one code path to
audit. Commands are dotted ``noun.verb`` strings, like OllamaGate's.
"""

from __future__ import annotations

from typing import Any

from ..hub.runtime import Hub
from .ipc import AdminError


class Admin:
    def __init__(self, hub: Hub) -> None:
        self.hub = hub

    async def dispatch(self, cmd: str, args: dict[str, Any]) -> Any:
        if cmd == "status":
            return self.hub.get_snapshot()
        raise AdminError(f"unknown command {cmd!r}")
