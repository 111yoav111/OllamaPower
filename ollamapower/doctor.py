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
import stat
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from urllib.parse import urlsplit

import httpx

from .config import Config, is_loopback_host
from .core.auth import TOKEN_ENV
from .files import load_token, token_fix, token_problem

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
            "Run `ollamapower setup`: it does this for you, asking for your password, and",
            "`ollamapower uninstall` undoes it. Or by hand, if Ollama is a systemd service:",
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
        findings.append(
            Finding(
                "warn",
                f"Nothing listens on {front}: apps using it find no Ollama",
                "Start the hub with `ollamapower hub`, or give the port back to Ollama with `ollamapower uninstall`.",
            )
        )
    elif at_front == "other":
        findings.append(
            Finding("error", f"Something other than Ollama or OllamaPower uses {front}", "Stop it or change hub.port.")
        )
    return findings


def parse_ollama_host(value: str) -> tuple[str, int]:
    """``OLLAMA_HOST`` as Ollama reads it: ``[scheme://]host[:port]``; an empty host means all interfaces.

    Raises ``ValueError`` for what Ollama could only guess at (an unclosed ``[``, a bad port, ...).
    """
    text = value.strip()
    scheme, sep, rest = text.partition("://")
    if not sep:
        rest, default = text, 11434
    elif scheme.lower() in ("http", "https"):
        default = 80 if scheme.lower() == "http" else 443
    else:
        raise ValueError(f"unknown scheme {scheme!r} (use http)")
    hostport = rest.split("/", 1)[0]
    port_text: str | None = None
    if hostport.startswith("["):
        host, closed, tail = hostport[1:].partition("]")
        if not closed:
            raise ValueError("an IPv6 address is missing its closing ']'")
        if tail and not tail.startswith(":"):
            raise ValueError(f"unexpected {tail!r} after the IPv6 address")
        port_text = tail[1:] if tail else None
    elif hostport.count(":") == 1:
        host, _, port_text = hostport.partition(":")
    else:
        host = hostport  # a name, an IPv4 address or a bare IPv6 address
    if "[" in host or "]" in host:
        raise ValueError("misplaced '[' or ']'")
    if port_text is None:
        return host, default
    if not (port_text.isascii() and port_text.isdigit()) or not 1 <= int(port_text) <= 65535:
        raise ValueError(f"bad port {port_text!r} (1-65535)")
    return host, int(port_text)


def check_exposure(config: Config) -> list[Finding]:
    findings = []
    target = urlsplit(config.ollama.url)
    port = target.port or 11434
    fix = "Bind Ollama to loopback with OLLAMA_HOST:\n" + migration_steps(config.ollama.url)
    env_host = os.environ.get("OLLAMA_HOST", "")
    if env_host.strip():
        try:
            host, _ = parse_ollama_host(env_host)
        except ValueError as exc:
            findings.append(
                Finding(
                    "error",
                    f"OLLAMA_HOST is set to {env_host!r}, which is not a valid address ({exc})",
                    f"Set it to host:port, e.g. OLLAMA_HOST={target.hostname}:{port}",
                )
            )
        else:
            if not host:
                findings.append(Finding("error", f"OLLAMA_HOST is set to {env_host!r} (all interfaces)", fix))
            elif not is_loopback_host(host):
                findings.append(Finding("error", f"OLLAMA_HOST is set to {env_host!r} (not loopback)", fix))
    exposed = reachable_on_lan(port)
    if exposed:
        findings.append(
            Finding(
                "error",
                f"Ollama's port {port} is reachable on {', '.join(exposed)}",
                f"Anyone on that network can use Ollama directly. Set OLLAMA_HOST={target.hostname}:{port}.",
            )
        )
    else:
        findings.append(Finding("ok", f"Ollama's port {port} is not reachable from the network"))
    findings.extend(check_frontdoor_exposure(config))
    return findings


def check_frontdoor_exposure(config: Config) -> list[Finding]:
    """The front door (11434) has no authentication: it belongs on loopback unless the owner chose otherwise."""
    front = config.hub.port
    deliberate = not is_loopback_host(config.hub.host)
    gate = 'Remote access is OllamaGate\'s job; set hub.host = "127.0.0.1".'
    exposed = reachable_on_lan(front)
    if exposed and deliberate:
        where = ", ".join(exposed)
        return [Finding("warn", f"hub.host is {config.hub.host}: port {front} is open on {where} (no auth)", gate)]
    if exposed:
        return [
            Finding(
                "error",
                f"Port {front} is reachable on {', '.join(exposed)}, although hub.host is {config.hub.host}",
                "Something else listens there on all interfaces - often Ollama itself with OLLAMA_HOST=0.0.0.0 - "
                "and anyone on that network can use it without authentication. Bind it to loopback:\n"
                + migration_steps(config.ollama.url),
            )
        ]
    if deliberate:
        return [Finding("warn", f"hub.host is {config.hub.host}: the front door is open to the network", gate)]
    return [Finding("ok", f"The front door's port {front} is not reachable from the network")]


def check_token(config: Config) -> list[Finding]:
    if load_token(config.env_path) is None:
        return [Finding("warn", f"No {TOKEN_ENV} yet (needed once helpers join)", "Run `ollamapower init`.")]
    _, problem = token_problem(config.env_path)
    if problem:
        fix = token_fix(config.env_path)
        return [Finding("error", problem, fix[0].upper() + fix[1:] + ".")]
    return [Finding("ok", f"{TOKEN_ENV} is set and strong")]


def windows_dir_is_private(config_dir: Path | str, home: Path | str) -> bool:
    """Is ``config_dir`` inside the user's profile (``C:\\Users\\<name>``)?

    Windows doesn't use the POSIX modes ``write_private`` sets. Files are private
    because the profile folder's ACL admits only its owner (and admins), and new
    files inherit that. A config dir moved elsewhere with ``OLLAMAPOWER_CONFIG_DIR``
    (``D:\\OllamaPower``, ``C:\\OllamaPower``) inherits that drive's ACL, which
    usually lets every local user read the token and the hub key.
    """
    d, h = PureWindowsPath(config_dir), PureWindowsPath(home)  # compared case-insensitively
    return d == h or h in d.parents


def _mode_problem(path: Path, secret: bool) -> tuple[str, str] | None:
    """(level, why) when ``path`` is not private enough, else None. POSIX only; a missing path is fine."""
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return None
    except OSError as exc:
        return "warn", f"cannot be checked ({exc.strerror or exc})"
    if stat.S_ISLNK(st.st_mode):
        return "error", "is a symlink"
    if hasattr(os, "getuid") and st.st_uid != os.getuid():
        return "error", "is owned by another user"
    mode = stat.S_IMODE(st.st_mode)
    if mode & 0o022:
        return "error", f"is writable by group or others (mode {mode:o})"
    if mode & 0o044:
        return ("error" if secret else "warn"), f"is readable by group or others (mode {mode:o})"
    if mode & 0o011:
        return ("error" if secret else "warn"), f"is accessible by group or others (mode {mode:o})"
    return None


def check_permissions(config: Config, platform: str = sys.platform, home: Path | None = None) -> list[Finding]:
    """The config dir, token, keys and admin secrets must be owner-only (error); other files should be (warn).

    Anything writable by others is an error: ``known_hub`` decides which hub a helper trusts and
    ``llamacpp/`` holds programs that run.
    """
    if platform == "win32":
        d = config.config_dir
        full = d if PureWindowsPath(d).is_absolute() else os.path.abspath(d)
        if windows_dir_is_private(full, home or Path.home()):
            return [Finding("ok", "File permissions: inside your user profile, which only you can read")]
        return [
            Finding(
                "warn",
                f"{d} is outside your user profile: other users of this PC may be able to read the token and keys",
                "Use the default config dir (unset OLLAMAPOWER_CONFIG_DIR), or restrict this one: "
                f'icacls "{d}" /inheritance:r /grant:r "%USERNAME%:(OI)(CI)F"',
            )
        ]
    d = config.config_dir
    secret = [d, config.env_path, config.hub_key_path, d / "run" / "admin.json", d / "agent-run" / "admin.json"]
    other = [
        d / "config.toml",
        d / "agent.toml",
        config.state_path,
        config.state_path.with_name("state.json.bak"),
        d / "known_hub",
        d / "node_id",
        d / "run",
        d / "agent-run",
        d / "llamacpp",
    ]
    try:
        if config.logs_dir.is_dir() and not config.logs_dir.is_symlink():
            other += [config.logs_dir, *sorted(config.logs_dir.iterdir())]
    except OSError:
        other.append(config.logs_dir)
    findings = []
    for path, is_secret in [*((p, True) for p in secret), *((p, False) for p in other)]:
        problem = _mode_problem(path, is_secret)
        if problem:
            level, why = problem
            findings.append(Finding(level, f"{path} {why}", f"chmod {'700' if path.is_dir() else '600'} '{path}'"))
    if not findings:
        findings.append(Finding("ok", "Config directory, token, keys, state and log files are owner-only"))
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


def check_llamacpp(config: Config, vendors: list[str] | None = None) -> list[Finding]:
    """The pinned llama.cpp build split mode uses: installed, verified, and it runs (``--version``)."""
    from .llamacpp.install import Pins
    from .llamacpp.manager import LlamaCppNotInstalled, LlamaCppUnavailable, ensure, gpu_vendors

    lc = config.llamacpp
    pins = Pins.load()
    if vendors is None:
        vendors = gpu_vendors()
    try:
        found = ensure(config.config_dir, lc.backend, lc.path, vendors, download=False)
    except LlamaCppNotInstalled:
        return [
            Finding(
                "ok",
                f"llama.cpp {pins.tag} (split mode) is not installed yet: it downloads on first use "
                "(or run `ollamapower llamacpp install`)",
            )
        ]
    except LlamaCppUnavailable as exc:
        fix = (
            f"llamacpp.path is set: it must hold build {pins.build} ({pins.tag}), or remove it to use the download."
            if lc.path
            else "Run `ollamapower llamacpp install` to download and verify it again."
        )
        return [Finding("error", f"llama.cpp: {exc}", fix)]
    return [Finding("ok", f"llama.cpp build {found.build} ({pins.tag}) is installed, verified and runs")]


def check_user() -> list[Finding]:
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        return [Finding("warn", "Running as root", "Run OllamaPower as your normal user.")]
    return []


def run_all(config: Config) -> list[Finding]:
    checks: list[tuple[str, Callable[[], list[Finding]]]] = [
        ("ports", lambda: check_ports(config)),
        ("exposure", lambda: check_exposure(config)),
        ("token", lambda: check_token(config)),
        ("control plane", lambda: check_control(config)),
        ("permissions", lambda: check_permissions(config)),
        ("llama.cpp", lambda: check_llamacpp(config)),
        ("user", check_user),
    ]
    findings: list[Finding] = []
    for name, check in checks:
        try:
            findings.extend(check())
        except Exception as exc:  # noqa: BLE001 - one broken check must not hide the others
            findings.append(Finding("error", f"The {name} check failed: {type(exc).__name__}: {exc}"))
    return findings
