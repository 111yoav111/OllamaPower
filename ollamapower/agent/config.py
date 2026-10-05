"""``agent.toml``: where the hub is and what the helper's owner lends.

The helper's owner is in control: everything under ``[lending]`` is decided
here (or in the agent TUI / CLI), never by the hub. The agent can write the
file back when the owner edits the rules, so it only holds known keys.
"""

from __future__ import annotations

import json
import math
import os
import platform
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..config import ConfigError, _apply, apply_env, default_config_dir, section_keys
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


@dataclass
class LendingRules:
    """What the helper's owner allows. 0 means "no cap"."""

    max_vram_gb: float = 0.0
    max_vram_percent: int = 0
    gpus: list[int] = field(default_factory=list)  # empty = all GPUs
    allow_cpu: bool = False  # lend CPU/RAM when there is no GPU
    allow_direct: bool = False  # allow unencrypted trusted-LAN mode for split traffic
    disk_cache: bool = True  # keep received model weights on disk so later loads skip the network
    schedule: list[str] = field(default_factory=list)  # e.g. ["22:00-08:00"], local time; empty = always
    # Step back when you use the GPU yourself: it is busier than yield_gpu_percent for
    # yield_seconds while the hub isn't using it (see agent/lending.py).
    auto_yield: bool = True
    yield_gpu_percent: int = 60
    yield_seconds: int = 20

    def validate(self) -> None:
        if not math.isfinite(self.max_vram_gb) or self.max_vram_gb < 0:
            raise ConfigError("lending.max_vram_gb must be 0 or more")
        if not 0 <= self.max_vram_percent <= 100:
            raise ConfigError("lending.max_vram_percent must be between 0 and 100")
        if not 1 <= self.yield_gpu_percent <= 100:
            raise ConfigError("lending.yield_gpu_percent must be between 1 and 100")
        if not 2 <= self.yield_seconds <= 3600:
            raise ConfigError("lending.yield_seconds must be between 2 and 3600")
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
class AgentLlamaCpp:
    # auto | cpu | metal | vulkan | cuda12 | cuda13 | rocm (see the hub's config).
    backend: str = "auto"
    path: str = ""  # your own build instead of the download (must be the pinned build)
    # Download the pinned build automatically when the agent starts (12-33 MB; CUDA/ROCm much more).
    auto_install: bool = True


@dataclass
class AgentConfig:
    hub: HubAddress = field(default_factory=HubAddress)
    agent: AgentSettings = field(default_factory=AgentSettings)
    lending: LendingRules = field(default_factory=LendingRules)
    llamacpp: AgentLlamaCpp = field(default_factory=AgentLlamaCpp)
    config_dir: Path = field(default_factory=default_config_dir)
    # (section, key) pairs taken from OLLAMAPOWER_<SECTION>__<KEY>: saving keeps the file's own value for them.
    env_overrides: list[tuple[str, str]] = field(default_factory=list, compare=False, repr=False)

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
        self.lending.validate()
        from ..llamacpp.variants import BACKENDS

        if self.llamacpp.backend not in BACKENDS:
            raise ConfigError(f"llamacpp.backend must be one of {', '.join(BACKENDS)}")


SECTIONS = ("hub", "agent", "lending", "llamacpp")


def _read_toml(path: Path) -> dict[str, Any] | None:
    """The parsed file, or None if there is none. Unreadable or broken files are config errors."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"cannot read {path}: {getattr(exc, 'strerror', None) or exc}") from None
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path}: {exc}") from None


def load_agent_config(
    config_dir: Path | None = None, path: Path | None = None, env: Mapping[str, str] | None = None
) -> AgentConfig:
    """Load agent.toml, then apply ``OLLAMAPOWER_<SECTION>__<KEY>`` env overrides, then validate."""
    from ..config import SECTIONS as HUB_SECTIONS
    from ..config import Config

    env = os.environ if env is None else env
    cfg = AgentConfig()
    if config_dir is not None:
        cfg.config_dir = config_dir
    path = path or cfg.path
    data = _read_toml(path)
    for section, values in (data or {}).items():
        if section not in SECTIONS or not isinstance(values, dict):
            raise ConfigError(f"unknown section [{section}] in {path.name}")
        _apply(getattr(cfg, section), values, section)
    cfg.env_overrides = apply_env(cfg, SECTIONS, section_keys(Config(), HUB_SECTIONS), env)
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
    # A value that came from an env var is not the owner's file setting: keep what the file says.
    on_disk = AgentConfig()
    if cfg.env_overrides:
        for section, values in (_read_toml(cfg.path) or {}).items():
            if section in SECTIONS and isinstance(values, dict):
                _apply(getattr(on_disk, section), values, section)
    lines = ["# OllamaPower agent settings. Edit here, in the agent TUI (E) or with the CLI.", ""]
    for section in SECTIONS:
        lines.append(f"[{section}]")
        for key, value in asdict(getattr(cfg, section)).items():
            if (section, key) in cfg.env_overrides:
                value = getattr(getattr(on_disk, section), key)
            lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")
    write_private(cfg.path, "\n".join(lines).encode("utf-8"))
