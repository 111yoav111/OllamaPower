"""Configuration: a TOML file plus ``OLLAMAPOWER_<SECTION>__<KEY>`` env overrides.

Everything has a safe default, so running without a config file works. The
shared cluster token is a secret and never lives here: it is read from the
``OLLAMAPOWER_TOKEN`` env var or the owner-only ``.env`` next to this file.
"""

from __future__ import annotations

import ipaddress
import math
import os
import sys
import tomllib
import typing
from dataclasses import dataclass, field, fields
from pathlib import Path
from urllib.parse import urlsplit

ENV_PREFIX = "OLLAMAPOWER_"


class ConfigError(ValueError):
    """Raised when the configuration is invalid or insecure."""


def default_config_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "ollamapower"


@dataclass
class HubConfig:
    # The front door. Clients that used Ollama's default port keep working unchanged.
    host: str = "127.0.0.1"
    port: int = 11434
    # Inference requests forwarded at once; the rest wait in the priority queue.
    # Match it to Ollama's OLLAMA_NUM_PARALLEL: higher lets Ollama queue requests
    # itself (first come first served, priorities lost), lower leaves Ollama idle.
    max_concurrency: int = 4
    # Waiting requests beyond this get 503 at once instead of queueing.
    max_queue: int = 256
    # Seconds a request may wait for a slot before getting 503 "busy".
    queue_timeout: float = 300.0
    # Inference request bodies are read whole (to see the model); images make them big.
    max_body_bytes: int = 100 * 1024 * 1024
    max_connections: int = 512


@dataclass
class OllamaConfig:
    # Where the real Ollama listens after `ollamapower doctor`'s port move.
    url: str = "http://127.0.0.1:11435"
    connect_timeout: float = 5.0
    # Max seconds to wait for the next chunk from Ollama (model loads can be slow).
    read_timeout: float = 600.0


@dataclass
class LoggingConfig:
    level: str = "info"  # debug | info | warning | error
    # Rotating JSON-lines files in <config dir>/logs: size of each file, files kept.
    file_max_bytes: int = 10 * 1024 * 1024
    file_backups: int = 5


@dataclass
class Config:
    hub: HubConfig = field(default_factory=HubConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    config_dir: Path = field(default_factory=default_config_dir)

    def validate(self) -> None:
        h = self.hub
        if not (1 <= h.port <= 65535):
            raise ConfigError(f"hub.port out of range: {h.port}")
        for name in ("max_concurrency", "max_queue", "max_body_bytes", "max_connections"):
            if getattr(h, name) <= 0:
                raise ConfigError(f"hub.{name} must be positive")
        for name, value in (
            ("hub.queue_timeout", h.queue_timeout),
            ("ollama.connect_timeout", self.ollama.connect_timeout),
            ("ollama.read_timeout", self.ollama.read_timeout),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ConfigError(f"{name} must be a positive number")
        up = urlsplit(self.ollama.url)
        try:
            up_port = up.port
        except ValueError:
            raise ConfigError(f"invalid ollama.url {self.ollama.url!r}") from None
        if up.scheme not in ("http", "https") or not up.hostname:
            raise ConfigError(f"invalid ollama.url {self.ollama.url!r}")
        if not is_loopback_host(up.hostname):
            raise ConfigError("ollama.url must point at a loopback address (Ollama must stay local)")
        default_port = 443 if up.scheme == "https" else 80
        serves_loopback = is_loopback_host(h.host) or h.host in ("0.0.0.0", "::", "")
        if serves_loopback and (up_port or default_port) == h.port:
            raise ConfigError(
                f"ollama.url uses port {h.port}, which is the front door's own port; move Ollama "
                "to another port (default 11435, see `ollamapower doctor`)"
            )
        if self.logging.level not in ("debug", "info", "warning", "error"):
            raise ConfigError("logging.level must be debug, info, warning or error")
        if self.logging.file_max_bytes < 64 * 1024 or self.logging.file_backups < 1:
            raise ConfigError("logging.file_max_bytes must be at least 65536 and logging.file_backups at least 1")

    @property
    def frontdoor_url(self) -> str:
        host = self.hub.host
        if host in ("0.0.0.0", "::"):
            host = "127.0.0.1"
        if ":" in host:
            host = f"[{host}]"
        return f"http://{host}:{self.hub.port}"

    @property
    def env_path(self) -> Path:
        return self.config_dir / ".env"

    @property
    def state_path(self) -> Path:
        return self.config_dir / "state.json"

    @property
    def logs_dir(self) -> Path:
        return self.config_dir / "logs"


def is_loopback_host(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def _coerce(value: typing.Any, hint: typing.Any, where: str) -> typing.Any:
    """Coerce a TOML or env-string value to the dataclass field type."""
    origin = typing.get_origin(hint)
    if origin is list:
        if isinstance(value, str):
            value = [v.strip() for v in value.split(",") if v.strip()]
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise ConfigError(f"{where} must be a list of strings")
        return list(value)
    if hint is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in ("1", "true", "yes", "on"):
            return True
        if isinstance(value, str) and value.lower() in ("0", "false", "no", "off"):
            return False
        raise ConfigError(f"{where} must be a boolean")
    if hint is int:
        if isinstance(value, bool):
            raise ConfigError(f"{where} must be an integer")
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{where} must be an integer") from exc
    if hint is float:
        if isinstance(value, bool):
            raise ConfigError(f"{where} must be a number")
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{where} must be a number") from exc
    if hint is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where} must be a string")
        return value
    raise ConfigError(f"unsupported config type for {where}")


def _apply(section_obj: typing.Any, data: dict, section: str) -> None:
    hints = typing.get_type_hints(type(section_obj))
    known = {f.name for f in fields(section_obj)}
    for key, value in data.items():
        if key not in known:
            raise ConfigError(f"unknown config key {section}.{key}")
        setattr(section_obj, key, _coerce(value, hints[key], f"{section}.{key}"))


SECTIONS = ("hub", "ollama", "logging")
# Env vars with the prefix that are not config overrides.
NON_CONFIG_ENV = {"OLLAMAPOWER_TOKEN", "OLLAMAPOWER_CONFIG_DIR"}


def load_config(path: Path | None = None, env: typing.Mapping[str, str] | None = None) -> Config:
    """Load config from ``path`` (optional) and apply env overrides, then validate."""
    env = os.environ if env is None else env
    cfg = Config()
    if env.get(ENV_PREFIX + "CONFIG_DIR"):
        cfg.config_dir = Path(env[ENV_PREFIX + "CONFIG_DIR"])
    if path is None:
        candidate = cfg.config_dir / "config.toml"
        path = candidate if candidate.exists() else None
    if path is not None:
        try:
            with open(path, "rb") as fh:
                data = tomllib.load(fh)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{path}: {exc}") from None
        for section, values in data.items():
            if section not in SECTIONS or not isinstance(values, dict):
                raise ConfigError(f"unknown config section [{section}]")
            _apply(getattr(cfg, section), values, section)
    for name, value in env.items():
        if not name.startswith(ENV_PREFIX) or name in NON_CONFIG_ENV or "__" not in name:
            continue
        section, _, key = name[len(ENV_PREFIX) :].lower().partition("__")
        if section not in SECTIONS:
            raise ConfigError(f"unknown config section in env var {name}")
        _apply(getattr(cfg, section), {key: value}, section)
    cfg.validate()
    return cfg
