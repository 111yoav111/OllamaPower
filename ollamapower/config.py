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
    # Max seconds to wait for Ollama's next bytes; 0 = no limit (the default, like
    # Ollama itself: a big pull or a long "stream": false answer sends nothing for
    # minutes). A client that hangs up still stops the request at once.
    read_timeout: float = 0.0


DEFAULT_AGENT_NETWORKS = [
    "127.0.0.0/8",
    "::1/128",
    "10.0.0.0/8",
    "172.16.0.0/12",
    "192.168.0.0/16",
    "100.64.0.0/10",  # CGNAT, incl. Tailscale
    "fc00::/7",
    "fe80::/10",
]


@dataclass
class ControlConfig:
    """The control plane helpers connect to (DCP's protocol, encrypted)."""

    enabled: bool = True
    # Helpers on the LAN must reach it, so it listens on all interfaces by default;
    # connections from outside allowed_networks are dropped before the handshake.
    host: str = "0.0.0.0"
    port: int = 9000
    allowed_networks: list[str] = field(default_factory=lambda: list(DEFAULT_AGENT_NETWORKS))
    # An agent silent for this long is treated as gone (it beats every 2 s).
    heartbeat_timeout: float = 10.0
    # Everything before the token is checked (RSA exchange + hello) must finish in this many seconds.
    handshake_timeout: float = 10.0
    max_nodes: int = 32
    # Unfinished handshakes one IP may hold open at once (a helper needs one).
    max_pending_per_ip: int = 4
    # Failed logins from one IP within ban_window seconds -> banned, doubling on repeat.
    ban_failures: int = 5
    ban_window: int = 600
    ban_seconds: int = 600
    max_ban_seconds: int = 86400


@dataclass
class LlamaCppConfig:
    # auto | cpu | metal | vulkan | cuda12 | cuda13 | rocm. auto = Metal on Apple
    # Silicon, Vulkan with a GPU, CPU otherwise. CUDA is faster on NVIDIA but a
    # 580-770 MB download.
    backend: str = "auto"
    # Use your own build instead of downloading (it must be the pinned build).
    path: str = ""


@dataclass
class SplitConfig:
    """Split mode: one model across the hub's GPU and helpers' GPUs (llama.cpp RPC)."""

    # Seconds an idle split model stays loaded before the GPU goes back to Ollama.
    keep_alive: float = 300.0
    # Context size when the plan doesn't set one.
    default_ctx: int = 4096
    # Seconds to wait for llama-server to load the model (first loads copy the
    # weights to the helpers over the network).
    load_timeout: float = 900.0
    # No token for this long counts as a failure.
    stall_timeout: float = 120.0


@dataclass
class PlannerConfig:
    """When the hub uses split mode by itself (Golden rule: never slower than plain Ollama)."""

    # Split mode is chosen only if its generation speed beats plain Ollama by this much (0.10 = 10 %).
    min_gain: float = 0.10
    # Added to weights + KV cache when estimating whether a model fits the hub's VRAM.
    memory_margin: float = 0.10
    # Benchmark models that would benefit, by itself, while the hub is idle. A benchmark
    # unloads Ollama's models and is abandoned as soon as a request arrives.
    auto_benchmark: bool = True
    # Seconds without any request before an automatic benchmark may start.
    idle_seconds: float = 300.0
    # After a split failure, Ollama serves the model for at least this long, then split
    # mode is tried again once all of the plan's helpers are back. The pause doubles with
    # every failure in a row (up to an hour) and starts over once split mode answers again.
    # Failures that can't go away by themselves (llama.cpp can't load the model, llama.cpp
    # missing or the wrong build) wait for the model, its context or llama.cpp to change.
    retry_after: float = 120.0
    # The benchmark workload (same for every engine): prompt size and tokens generated.
    prompt_tokens: int = 512
    gen_tokens: int = 128


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
    control: ControlConfig = field(default_factory=ControlConfig)
    llamacpp: LlamaCppConfig = field(default_factory=LlamaCppConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
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
        ):
            if not math.isfinite(value) or value <= 0:
                raise ConfigError(f"{name} must be a positive number")
        if not math.isfinite(self.ollama.read_timeout) or self.ollama.read_timeout < 0:
            raise ConfigError("ollama.read_timeout must be 0 (no limit) or a positive number")
        up = urlsplit(self.ollama.url)
        try:
            up_port = up.port
        except ValueError:
            raise ConfigError(f"invalid ollama.url {self.ollama.url!r}") from None
        if up.scheme not in ("http", "https") or not up.hostname:
            raise ConfigError(f"invalid ollama.url {self.ollama.url!r}")
        if not is_loopback_host(up.hostname):
            raise ConfigError("ollama.url must point at a loopback address (Ollama must stay local)")
        if up_port is None:
            # Ollama's own default is 11434 but a URL's is 80/443; an explicit port leaves no doubt.
            raise ConfigError(f"ollama.url needs an explicit port, e.g. http://{up.netloc}:11435")
        serves_loopback = is_loopback_host(h.host) or h.host in ("0.0.0.0", "::", "")
        if serves_loopback and up_port == h.port:
            raise ConfigError(
                f"ollama.url uses port {h.port}, which is the front door's own port; move Ollama "
                "to another port (default 11435, see `ollamapower doctor`)"
            )
        c = self.control
        if not (1 <= c.port <= 65535):
            raise ConfigError(f"control.port out of range: {c.port}")
        if c.enabled and c.port == h.port:
            raise ConfigError("control.port must differ from hub.port")
        for name in ("heartbeat_timeout", "handshake_timeout"):
            value = getattr(c, name)
            if not math.isfinite(value) or value <= 0:
                raise ConfigError(f"control.{name} must be a positive number")
        if c.max_nodes < 1 or c.ban_failures < 1 or c.ban_window < 1 or c.ban_seconds < 1:
            raise ConfigError("control.max_nodes, ban_failures, ban_window and ban_seconds must be positive")
        if c.max_pending_per_ip < 1:
            raise ConfigError("control.max_pending_per_ip must be positive")
        if c.max_ban_seconds < c.ban_seconds:
            raise ConfigError("control.max_ban_seconds must be at least control.ban_seconds")
        if c.enabled and not c.allowed_networks:
            raise ConfigError(
                "control.allowed_networks is empty, so no helper could connect: list your LAN ranges "
                "(see config.example.toml) or set control.enabled = false"
            )
        for cidr in c.allowed_networks:
            try:
                ipaddress.ip_network(cidr, strict=False)
            except ValueError:
                raise ConfigError(f"invalid control.allowed_networks entry {cidr!r}") from None
        from .llamacpp.variants import BACKENDS

        if self.llamacpp.backend not in BACKENDS:
            raise ConfigError(f"llamacpp.backend must be one of {', '.join(BACKENDS)}")
        sp = self.split
        for name in ("keep_alive", "load_timeout", "stall_timeout"):
            value = getattr(sp, name)
            if not math.isfinite(value) or value <= 0:
                raise ConfigError(f"split.{name} must be a positive number")
        if not 256 <= sp.default_ctx <= 1_048_576:
            raise ConfigError("split.default_ctx must be between 256 and 1048576")
        pl = self.planner
        if not (math.isfinite(pl.min_gain) and 0 <= pl.min_gain <= 10):
            raise ConfigError("planner.min_gain must be between 0 and 10 (0.10 = 10 %)")
        if not (math.isfinite(pl.memory_margin) and 0 <= pl.memory_margin <= 2):
            raise ConfigError("planner.memory_margin must be between 0 and 2")
        for name in ("idle_seconds", "retry_after"):
            value = getattr(pl, name)
            if not math.isfinite(value) or value <= 0:
                raise ConfigError(f"planner.{name} must be a positive number")
        if not 16 <= pl.prompt_tokens <= 32768 or not 8 <= pl.gen_tokens <= 4096:
            raise ConfigError("planner.prompt_tokens must be 16-32768 and planner.gen_tokens 8-4096")
        self.logging.level = self.logging.level.lower()
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
    def hub_key_path(self) -> Path:
        return self.config_dir / "hub_key.pem"

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


def _coerce(value: typing.Any, hint: typing.Any, where: str, from_env: bool = False) -> typing.Any:
    """Coerce a TOML value or an env string to the dataclass field type.

    TOML values must already have the right type (``port = "9000"`` is a mistake worth reporting); only
    env strings are parsed. Floats are never truncated into integer fields.
    """
    origin = typing.get_origin(hint)
    if origin is list:
        item = (typing.get_args(hint) or (str,))[0]
        if from_env and isinstance(value, str):
            value = [v.strip() for v in value.split(",") if v.strip()]
        if not isinstance(value, list):
            raise ConfigError(f"{where} must be a list")
        return [_coerce(v, item, where, from_env) for v in value]
    if hint is bool:
        if isinstance(value, bool):
            return value
        if from_env and isinstance(value, str) and value.lower() in ("1", "true", "yes", "on"):
            return True
        if from_env and isinstance(value, str) and value.lower() in ("0", "false", "no", "off"):
            return False
        raise ConfigError(f"{where} must be a boolean (true or false)")
    if hint is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if from_env and isinstance(value, str):
            try:
                return int(value.strip())
            except ValueError:
                pass
        raise ConfigError(f"{where} must be a whole number")
    if hint is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if from_env and isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                pass
        raise ConfigError(f"{where} must be a number")
    if hint is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where} must be a string")
        return value
    raise ConfigError(f"unsupported config type for {where}")


def _apply(section_obj: typing.Any, data: dict, section: str, from_env: bool = False) -> None:
    hints = typing.get_type_hints(type(section_obj))
    known = {f.name for f in fields(section_obj)}
    for key, value in data.items():
        if key not in known:
            raise ConfigError(f"unknown config key {section}.{key}")
        setattr(section_obj, key, _coerce(value, hints[key], f"{section}.{key}", from_env))


SECTIONS = ("hub", "ollama", "control", "llamacpp", "split", "planner", "logging")
# Env vars with the prefix that are not config overrides.
NON_CONFIG_ENV = {"OLLAMAPOWER_TOKEN", "OLLAMAPOWER_CONFIG_DIR"}


def load_config(path: Path | None = None, env: typing.Mapping[str, str] | None = None) -> Config:
    """Load config from ``path`` (optional) and apply env overrides, then validate."""
    env = os.environ if env is None else env
    cfg = Config()
    if env.get(ENV_PREFIX + "CONFIG_DIR"):
        cfg.config_dir = Path(env[ENV_PREFIX + "CONFIG_DIR"])
    explicit = path is not None
    path = path or cfg.config_dir / "config.toml"
    try:
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
    except FileNotFoundError:
        if explicit:
            raise ConfigError(f"{path} does not exist") from None
        data = None  # no config file: the defaults apply
    except OSError as exc:  # a directory, no permission, ...
        raise ConfigError(f"cannot read {path}: {exc.strerror or exc}") from None
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise ConfigError(f"{path}: {exc}") from None
    if data is not None:
        for section, values in data.items():
            if section not in SECTIONS or not isinstance(values, dict):
                raise ConfigError(f"unknown config section [{section}]")
            _apply(getattr(cfg, section), values, section)
    from .agent.config import SECTIONS as AGENT_SECTIONS
    from .agent.config import AgentConfig

    apply_env(cfg, SECTIONS, section_keys(AgentConfig(), AGENT_SECTIONS), env)
    cfg.validate()
    return cfg


def section_keys(cfg: typing.Any, sections: typing.Iterable[str]) -> dict[str, set[str]]:
    """The keys each section of a config object knows."""
    return {s: {f.name for f in fields(getattr(cfg, s))} for s in sections}


def apply_env(
    cfg: typing.Any,
    sections: typing.Iterable[str],
    other: dict[str, set[str]],
    env: typing.Mapping[str, str],
) -> list[tuple[str, str]]:
    """Apply the ``OLLAMAPOWER_<SECTION>__<KEY>`` overrides meant for ``cfg``; returns the (section, key)s set.

    The hub's config.toml and a helper's agent.toml read the same environment (and both have a ``[hub]``
    and a ``[llamacpp]`` section, with different keys), so a variable that belongs to the ``other`` file is
    skipped instead of failing every command on a machine that runs both. A name neither file knows is
    still an error: a typo must not silently do nothing.
    """
    own = section_keys(cfg, sections)
    applied = []
    for name, value in env.items():
        if not name.startswith(ENV_PREFIX) or name in NON_CONFIG_ENV or "__" not in name:
            continue
        section, _, key = name[len(ENV_PREFIX) :].lower().partition("__")
        if key in own.get(section, ()):
            _apply(getattr(cfg, section), {key: value}, section, from_env=True)
            applied.append((section, key))
        elif key in other.get(section, ()) or (section in other and section not in own):
            continue
        elif section in own:
            raise ConfigError(f"unknown config key {section}.{key} in env var {name}")
        else:
            raise ConfigError(f"unknown config section in env var {name}")
    return applied
