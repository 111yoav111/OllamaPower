"""Setup checks, and the walk-through for moving Ollama off port 11434.

OllamaPower takes Ollama's usual address (``127.0.0.1:11434``) so clients
need no change, and the real Ollama moves to ``127.0.0.1:11435``. ``doctor``
finds out where things stand - who answers on which port - and prints the
exact steps for this OS when the move hasn't happened yet.
"""

from __future__ import annotations

import ipaddress
import os
import socket
import sys
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

from .config import Config, is_loopback_host
from .core.auth import TOKEN_ENV, TokenError, validate_token
from .files import load_token, permission_problem

PROBE_TIMEOUT = 2.0


@dataclass
class Finding:
    level: str  # ok | warn | error
    title: str
    fix: str = ""


# ----- probing -------------------------------------------------------------------------


def local_ip_addresses() -> list[str]:
    """Best-effort list of this machine's non-loopback IP addresses."""
    addrs: set[str] = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None):
            addrs.add(str(info[4][0]))
    except OSError:
        pass
    # The UDP "connect" trick reveals the primary outbound address; no packet is sent.
    for family, target in ((socket.AF_INET, "192.0.2.1"), (socket.AF_INET6, "2001:db8::1")):
        try:
            with socket.socket(family, socket.SOCK_DGRAM) as s:
                s.connect((target, 9))
                addrs.add(s.getsockname()[0])
        except OSError:
            pass
    result = []
    for a in addrs:
        try:
            ip = ipaddress.ip_address(a.split("%")[0])
        except ValueError:
            continue
        if not ip.is_loopback and not ip.is_unspecified and not ip.is_link_local:
            result.append(str(ip))
    return sorted(result)


def reachable_on_lan(port: int, timeout: float = 0.5) -> list[str]:
    """The non-loopback addresses on which something accepts connections on ``port``."""
    exposed = []
    for addr in local_ip_addresses():
        try:
            with socket.create_connection((addr, port), timeout=timeout):
                exposed.append(addr)
        except OSError:
            continue
    return exposed


def who_answers(base_url: str) -> str:
    """``"ollamapower"``, ``"ollama"``, ``"other"`` or ``"nothing"`` at ``base_url``."""
    try:
        with httpx.Client(base_url=base_url, timeout=PROBE_TIMEOUT, trust_env=False) as client:
            r = client.get("/ollamapower/status")
            if r.status_code == 200 and isinstance(r.json(), dict) and "queue" in r.json():
                return "ollamapower"
            r = client.get("/api/version")
            if r.status_code == 200 and isinstance(r.json(), dict) and "version" in r.json():
                return "ollama"
            return "other"
    except (httpx.HTTPError, ValueError):
        return "nothing"
    except OSError:
        return "nothing"


def ollama_version(base_url: str) -> str | None:
    try:
        with httpx.Client(base_url=base_url, timeout=PROBE_TIMEOUT, trust_env=False) as client:
            value = client.get("/api/version").json().get("version")
            return str(value) if value is not None else None
    except (httpx.HTTPError, ValueError, AttributeError, OSError):
        return None


# ----- the port move -------------------------------------------------------------------


def migration_steps(ollama_url: str, platform: str = sys.platform) -> str:
    """How to make Ollama listen on ``ollama_url``'s host:port, for this OS."""
    parts = urlsplit(ollama_url)
    target = f"{parts.hostname}:{parts.port or 11434}"
    if platform == "win32":
        steps = [
            "Quit Ollama (tray icon -> Quit Ollama).",
            f"Run:  setx OLLAMA_HOST {target}",
            "Start Ollama again from the Start menu.",
            "Note: the `ollama` CLI reads OLLAMA_HOST too and will now talk to Ollama directly,",
            "      bypassing OllamaPower. That is harmless; to go through OllamaPower use",
            "      `set OLLAMA_HOST=127.0.0.1:11434` in that terminal.",
        ]
    elif platform == "darwin":
        steps = [
            f"Run:  launchctl setenv OLLAMA_HOST {target}",
            "Quit Ollama (menu bar icon -> Quit Ollama) and open it again.",
            "launchctl settings are lost at reboot; to keep it, add the same line to a",
            "LaunchAgent or your login items script.",
        ]
    else:
        steps = [
            "If Ollama runs as a systemd service (the default Linux install):",
            "  sudo systemctl edit ollama",
            "  add these two lines, save and close:",
            "    [Service]",
            f'    Environment="OLLAMA_HOST={target}"',
            "  sudo systemctl restart ollama",
            f"If you start it by hand instead:  OLLAMA_HOST={target} ollama serve",
        ]
    steps += ["Then start OllamaPower (`ollamapower hub`) and run `ollamapower doctor` again."]
    return "\n".join(steps)


# ----- checks --------------------------------------------------------------------------


def check_ports(config: Config) -> list[Finding]:
    findings = []
    front = config.frontdoor_url
    ollama_at_target = ollama_version(config.ollama.url)
    at_front = who_answers(front)

    if ollama_at_target is not None:
        findings.append(Finding("ok", f"Ollama {ollama_at_target} answers at {config.ollama.url}"))
    elif at_front == "ollama":
        findings.append(
            Finding(
                "error",
                f"Ollama still listens on {front}, the port OllamaPower needs; nothing at {config.ollama.url}",
                migration_steps(config.ollama.url),
            )
        )
    else:
        findings.append(
            Finding(
                "error",
                f"Ollama is not reachable at {config.ollama.url}",
                "Start Ollama with OLLAMA_HOST set to that address:\n" + migration_steps(config.ollama.url),
            )
        )

    if at_front == "ollamapower":
        findings.append(Finding("ok", f"OllamaPower answers at {front}"))
    elif at_front == "nothing":
        findings.append(Finding("warn", f"Nothing listens on {front} yet", "Start it with `ollamapower hub`."))
    elif at_front == "other":
        findings.append(
            Finding("error", f"Something other than Ollama or OllamaPower uses {front}", "Stop it or change hub.port.")
        )
    return findings


def check_exposure(config: Config) -> list[Finding]:
    findings = []
    fix = "Bind Ollama to loopback with OLLAMA_HOST:\n" + migration_steps(config.ollama.url)
    env_host = os.environ.get("OLLAMA_HOST", "")
    if env_host:
        host = urlsplit(env_host if "://" in env_host else f"http://{env_host}").hostname or ""
        if host and not is_loopback_host(host):
            findings.append(Finding("error", f"OLLAMA_HOST is set to {env_host!r} (not loopback)", fix))
    port = urlsplit(config.ollama.url).port or 11434
    exposed = reachable_on_lan(port)
    if exposed:
        findings.append(
            Finding(
                "error",
                f"Ollama's port {port} is reachable on {', '.join(exposed)}",
                "Anyone on that network can use Ollama directly. "
                f"Set OLLAMA_HOST={urlsplit(config.ollama.url).hostname}:{port}.",
            )
        )
    else:
        findings.append(Finding("ok", f"Ollama's port {port} is not reachable from the network"))
    if not is_loopback_host(config.hub.host):
        findings.append(
            Finding(
                "warn",
                f"hub.host is {config.hub.host}: the front door is open to the network",
                'Remote access is OllamaGate\'s job; set hub.host = "127.0.0.1".',
            )
        )
    return findings


def check_token(config: Config) -> list[Finding]:
    token = load_token(config.env_path)
    if token is None:
        return [Finding("warn", f"No {TOKEN_ENV} yet (needed once helpers join)", "Run `ollamapower init`.")]
    try:
        validate_token(token)
    except TokenError as exc:
        return [Finding("error", str(exc), "Run `ollamapower init --rotate-token`.")]
    return [Finding("ok", f"{TOKEN_ENV} is set and strong")]


def check_permissions(config: Config) -> list[Finding]:
    if sys.platform == "win32":
        return [Finding("ok", "File permissions: per-user profile folder (not checked on Windows)")]
    findings = []
    d = config.config_dir
    paths = [
        d,
        config.env_path,
        config.state_path,
        config.state_path.with_name("state.json.bak"),
        config.hub_key_path,
        d / "agent.toml",
        d / "known_hub",
        d / "node_id",
    ]
    if config.logs_dir.exists():
        paths += [config.logs_dir, *sorted(config.logs_dir.iterdir())]
    for path in paths:
        problem = permission_problem(path)
        if problem:
            findings.append(
                Finding("error", f"{path} {problem}", f"chmod {'700' if path.is_dir() else '600'} '{path}'")
            )
    if not findings:
        findings.append(Finding("ok", "Token, keys, state and log files are owner-only"))
    return findings


def check_control(config: Config) -> list[Finding]:
    c = config.control
    if not c.enabled:
        return [Finding("ok", "Control plane disabled (no helpers can join)")]
    findings = []
    public = []
    for cidr in c.allowed_networks:
        net = ipaddress.ip_network(cidr, strict=False)
        if net.prefixlen == 0 or (net.is_global and not net.is_private):
            public.append(cidr)
    if public:
        findings.append(
            Finding(
                "warn",
                f"control.allowed_networks includes public addresses ({', '.join(public)})",
                "Helpers belong on your LAN (or a tailnet). Keep the private ranges only.",
            )
        )
    else:
        findings.append(
            Finding(
                "ok", f"Control plane on port {c.port} accepts only {len(c.allowed_networks)} private network ranges"
            )
        )
    if config.hub_key_path.exists():
        findings.append(Finding("ok", "Hub key exists (helpers pin its fingerprint)"))
    else:
        findings.append(Finding("warn", "No hub key yet", "Run `ollamapower init` (or start the hub once)."))
    return findings


def check_user() -> list[Finding]:
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        return [Finding("warn", "Running as root", "Run OllamaPower as your normal user.")]
    return []


def run_all(config: Config) -> list[Finding]:
    return [
        *check_ports(config),
        *check_exposure(config),
        *check_token(config),
        *check_control(config),
        *check_permissions(config),
        *check_user(),
    ]
