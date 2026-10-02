"""``agent.toml``: where the hub is and what the helper's owner lends.

The helper's owner is in control: everything under ``[lending]`` is decided
here (or in the agent TUI / CLI), never by the hub. The agent can write the
file back when the owner edits the rules, so it only holds known keys.
"""

from __future__ import annotations

import json
import math
import platform
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..config import ConfigError, _apply, default_config_dir, is_loopback_host
from ..files import write_private

AGENT_FILE = "agent.toml"
DEFAULT_HUB_PORT = 9000


@dataclass
class HubAddress:
    # The hub's LAN address, e.g. "192.168.1.10". Can also be given with --hub.
    address: str = ""
    port: int = DEFAULT_HUB_PORT
    # Optional: the hub key fingerprint to trust on first connect, as printed by
    # `ollamapower hub`. Without it the agent asks before trusting a new hub.
    fingerprint: str = ""


@dataclass
class AgentSettings:
    # How this helper appears on the hub (default: the computer's name).
    name: str = ""
    # This helper's own Ollama (for overflow, Phase 6).
    ollama_url: str = "http://127.0.0.1:11434"


@dataclass
class LendingRules:
    """What the helper's owner allows. 0 means "no cap"."""

    max_vram_gb: float = 0.0
    max_vram_percent: int = 0
    gpus: list[int] = field(default_factory=list)  # empty = all GPUs
    allow_cpu: bool = False  # lend CPU/RAM when there is no GPU
    allow_overflow: bool = False  # run whole requests on this helper's Ollama (Phase 6)
    allow_direct: bool = False  # allow unencrypted trusted-LAN mode for split traffic (Phase 4)
    schedule: list[str] = field(default_factory=list)  # e.g. ["22:00-08:00"] (Phase 7)

    def validate(self) -> None:
        if not math.isfinite(self.max_vram_gb) or self.max_vram_gb < 0:
            raise ConfigError("lending.max_vram_gb must be 0 or more")
        if not 0 <= self.max_vram_percent <= 100:
            raise ConfigError("lending.max_vram_percent must be between 0 and 100")
        if any(g < 0 for g in self.gpus):
            raise ConfigError("lending.gpus must be GPU indexes (0, 1, ...)")
        for window in self.schedule:
            start, sep, end = window.partition("-")
            if not sep or not all(_is_hhmm(t) for t in (start, end)):
                raise ConfigError(f"lending.schedule entry {window!r} must look like 22:00-08:00")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_hhmm(value: str) -> bool:
    h, sep, m = value.strip().partition(":")
    return bool(sep) and h.isdigit() and m.isdigit() and 0 <= int(h) < 24 and 0 <= int(m) < 60 and len(m) == 2


@dataclass
class AgentConfig:
    hub: HubAddress = field(default_factory=HubAddress)
    agent: AgentSettings = field(default_factory=AgentSettings)
    lending: LendingRules = field(default_factory=LendingRules)
    config_dir: Path = field(default_factory=default_config_dir)

    @property
    def path(self) -> Path:
        return self.config_dir / AGENT_FILE

    @property
    def known_hub_path(self) -> Path:
        return self.config_dir / "known_hub"

    @property
    def node_id_path(self) -> Path:
        return self.config_dir / "node_id"

    @property
    def env_path(self) -> Path:
        return self.config_dir / ".env"

    @property
    def display_name(self) -> str:
        return self.agent.name or platform.node() or "helper"

    def validate(self) -> None:
        if not (1 <= self.hub.port <= 65535):
            raise ConfigError(f"hub.port out of range: {self.hub.port}")
        if len(self.agent.name) > 64:
            raise ConfigError("agent.name is too long (64 characters at most)")
        from urllib.parse import urlsplit

        url = urlsplit(self.agent.ollama_url)
        if url.scheme not in ("http", "https") or not url.hostname or not is_loopback_host(url.hostname):
            raise ConfigError("agent.ollama_url must be a loopback http(s) URL")
        self.lending.validate()


SECTIONS = ("hub", "agent", "lending")


def load_agent_config(config_dir: Path | None = None, path: Path | None = None) -> AgentConfig:
    cfg = AgentConfig()
    if config_dir is not None:
        cfg.config_dir = config_dir
    path = path or cfg.path
    if path.exists():
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{path}: {exc}") from None
        for section, values in data.items():
            if section not in SECTIONS or not isinstance(values, dict):
                raise ConfigError(f"unknown section [{section}] in {path.name}")
            _apply(getattr(cfg, section), values, section)
    cfg.validate()
    return cfg


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)  # JSON string escapes are valid TOML basic-string escapes
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise TypeError(type(value).__name__)


def save_agent_config(cfg: AgentConfig) -> None:
    """Write agent.toml (owner-only), e.g. after the owner edited the lending rules."""
    cfg.validate()
    lines = ["# OllamaPower agent settings. Edit here, in the agent TUI (E) or with the CLI.", ""]
    for section in SECTIONS:
        lines.append(f"[{section}]")
        for key, value in asdict(getattr(cfg, section)).items():
            lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")
    write_private(cfg.path, "\n".join(lines).encode("utf-8"))
