"""``ollamapower setup`` / ``uninstall``: the system changes OllamaPower needs, done for you and undone exactly.

On the hub: Ollama moves to ``ollama.url`` (127.0.0.1:11435) so the front door can take its usual
address, and the firewall lets LAN helpers reach the control port. On a helper: the firewall lets
the hub reach the split-mode relay port. ``uninstall`` puts back what ``setup`` changed, nothing else.

Security: OllamaPower never runs as root, and these commands refuse to. They run as the user and
call ``sudo`` only for fixed commands - argument lists, never a shell - whose only variable parts
are validated port numbers, a loopback IP and private LAN networks. Every command is shown before
it runs and the user confirms; ``--print`` only shows them. Ollama's port goes into a drop-in file
of our own (the user's ``override.conf`` is never edited), so undoing it is deleting that file.
Only firewall rules that were actually added are recorded and removed: a rule the user had made
already is left alone.

Linux with systemd (and ufw or firewalld) only; elsewhere the steps are printed to do by hand.
"""

from __future__ import annotations

import ipaddress
import json
import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

DROPIN_DIR = "/etc/systemd/system/ollama.service.d"
DROPIN = f"{DROPIN_DIR}/ollamapower.conf"
UNIT = "ollama.service"
RECORD = "setup.json"  # what `setup` changed on the hub; the helper's is AGENT_RECORD
AGENT_RECORD = "agent-setup.json"
RULE_TAG = "ollamapower"
DEFAULT_OLLAMA_PORT = 11434


class SetupError(Exception):
    """setup/uninstall can't (or must not) go ahead; the message says why and what to do."""


@dataclass(frozen=True)
class Step:
    why: str
    argv: tuple[str, ...]  # run as ``sudo -- argv``
    stdin: str | None = None
    rule: tuple[str, int] | None = None  # the firewall rule this step adds or removes
    kind: str = ""  # "query" / "ufw-list" (read-only checks), "restore-done" (Ollama back on its port)

    def shown(self) -> str:
        line = "sudo " + shlex.join(self.argv)
        if self.stdin is None:
            return line
        return f"{line} > /dev/null <<'EOF'\n{self.stdin}EOF"


@dataclass
class Plan:
    steps: list[Step] = field(default_factory=list)
    rules: list[tuple[str, int]] = field(default_factory=list)  # (network, port) the firewall steps add or remove
    firewall: str | None = None  # ufw · firewalld · None
    notes: list[str] = field(default_factory=list)
    summary: str = ""  # what it does, in plain words, shown before the commands


# ----- validation: the only values that ever reach a command ----------------------------


def valid_port(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
        raise SetupError(f"not a valid port: {value!r}")
    return value


def ollama_target(url: str) -> tuple[str, int]:
    """``ollama.url`` as (loopback IP, port): what goes into OLLAMA_HOST."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    if host == "localhost":
        host = "127.0.0.1"
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        raise SetupError(f"ollama.url must use a loopback IP address (e.g. 127.0.0.1), not {host!r}") from None
    if not ip.is_loopback:
        raise SetupError(f"ollama.url must be on this PC only (loopback), not {host}")
    try:
        port = parts.port
    except ValueError:
        port = None
    if port is None:
        raise SetupError("ollama.url needs an explicit port, e.g. http://127.0.0.1:11435")
    return (f"[{ip}]" if ip.version == 6 else str(ip)), valid_port(port)


# Container, VM and VPN bridges: never opened to (best effort; the /16 cap below is the real limit).
VIRTUAL_IFACES = ("docker", "br-", "veth", "virbr", "vnet", "podman", "cni", "flannel", "lxc", "lxd", "vmnet")
MIN_PREFIX = 16  # a firewall rule never covers more than a /16 (65 536 addresses)


def lan_networks(addrs: dict[str, list[object]] | None = None) -> dict[str, str]:
    """The private IPv4 networks this PC is on, each with its interfaces (``{"192.168.1.0/24": "eth0"}``):
    who the firewall rules let in. Never wider than a /16, never loopback, link-local or a container bridge."""
    import socket

    if addrs is None:
        import psutil

        addrs = psutil.net_if_addrs()  # type: ignore[assignment]
    nets: dict[ipaddress.IPv4Network, list[str]] = {}
    for name, entries in (addrs or {}).items():
        if name.startswith(VIRTUAL_IFACES):
            continue
        for a in entries:
            if getattr(a, "family", None) != socket.AF_INET or not getattr(a, "netmask", None):
                continue
            try:
                net = ipaddress.IPv4Interface(f"{a.address}/{a.netmask}").network  # type: ignore[attr-defined]
            except ValueError:
                continue
            if lan_network_ok(net):
                nets.setdefault(net, []).append(name)
    return {str(n): ", ".join(sorted(nets[n])) for n in sorted(nets)}


def lan_network_ok(net: ipaddress.IPv4Network) -> bool:
    return net.is_private and not net.is_loopback and not net.is_link_local and net.prefixlen >= MIN_PREFIX


# ----- what is on this machine (read-only, no sudo) -------------------------------------


def _quiet(argv: list[str]) -> str:
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=10, check=False).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def has_systemd_ollama() -> bool:
    return sys.platform.startswith("linux") and _quiet(["systemctl", "show", "-p", "LoadState", "--value", UNIT]) == (
        "loaded"
    )


def active_firewall() -> str | None:
    for name in ("ufw", "firewalld"):
        if _quiet(["systemctl", "is-active", name]) == "active":
            return name
    return None


def ollama_host_set_elsewhere() -> list[str]:
    """Unit files (other than ours) that set OLLAMA_HOST, or that we can't read (so can't rule out):
    setup and uninstall can't own Ollama's port then."""
    paths = _quiet(["systemctl", "show", "-p", "FragmentPath", "-p", "DropInPaths", "--value", UNIT]).split()
    found = []
    for path in paths:
        if path == DROPIN:
            continue
        try:
            text = Path(path).read_text(errors="replace")
        except OSError:
            found.append(f"{path} (unreadable)")
            continue
        lines = [line.strip() for line in text.splitlines() if not line.lstrip().startswith(("#", ";"))]
        if any("OLLAMA_HOST" in line or line.startswith("EnvironmentFile") for line in lines):
            found.append(path)
    return found


def dropin_text(host: str, port: int) -> str:
    return (
        "# Written by `ollamapower setup`; `ollamapower uninstall` removes this file.\n"
        "[Service]\n"
        f'Environment="OLLAMA_HOST={host}:{valid_port(port)}"\n'
    )


# ----- plans -----------------------------------------------------------------------------


def _rule(net: str, port: object) -> tuple[str, int]:
    """A (network, port) that may reach a firewall command: a private LAN network and a real port."""
    parsed = ipaddress.IPv4Network(net)  # raises ValueError on anything that isn't one
    if not lan_network_ok(parsed):
        raise ValueError(f"{net} is not a private LAN network of at most a /{MIN_PREFIX}")
    return str(parsed), valid_port(port)


def ufw_spec(net: str, port: int) -> tuple[str, ...]:
    return ("from", net, "to", "any", "port", str(port), "proto", "tcp")


def firewalld_rule(net: str, port: int) -> str:
    return f'rule family="ipv4" source address="{net}" port port="{port}" protocol="tcp" accept'


def firewall_steps(backend: str | None, rules: list[tuple[str, int]], remove: bool = False) -> list[Step]:
    """Adding: first look at the existing rules (read-only), so a rule the owner already had is neither
    added again nor recorded as ours - it then survives uninstall. That check reads exit codes and ufw's
    own rule syntax, which don't change with the language. Removing: only the rules we recorded."""
    steps: list[Step] = []
    checked = [_rule(net, port) for net, port in rules]  # nothing else may reach a command
    if backend == "ufw" and checked and not remove:
        steps.append(Step("see which rules exist already (read-only)", ("ufw", "show", "added"), kind="ufw-list"))
    for net, port in checked:
        verb = "close" if remove else "open"
        if backend == "ufw":
            spec = ufw_spec(net, port)
            argv = ("ufw", "delete", "allow", *spec) if remove else ("ufw", "allow", *spec, "comment", RULE_TAG)
            steps.append(Step(f"{verb} TCP {port} for {net}", argv, rule=(net, port)))
        elif backend == "firewalld":
            if not remove:
                query = ("firewall-cmd", "--permanent", "--query-rich-rule", firewalld_rule(net, port))
                steps.append(Step(f"is TCP {port} for {net} open already? (read-only)", query, kind="query"))
            flag = "--remove-rich-rule" if remove else "--add-rich-rule"
            argv = ("firewall-cmd", "--permanent", flag, firewalld_rule(net, port))
            steps.append(Step(f"{verb} TCP {port} for {net}", argv, rule=(net, port)))
    if backend == "firewalld" and checked:
        steps.append(Step("apply the firewall change", ("firewall-cmd", "--reload")))
    return steps


def move_ollama_steps(host: str, port: int) -> list[Step]:
    return [
        Step("create Ollama's drop-in folder", ("install", "-d", "-m", "0755", DROPIN_DIR)),
        Step(f"tell Ollama to listen on {host}:{port}", ("tee", DROPIN), stdin=dropin_text(host, port)),
        Step("make it readable like the rest of /etc", ("chmod", "0644", DROPIN)),
        Step("reload systemd", ("systemctl", "daemon-reload")),
        Step("restart Ollama on its new port", ("systemctl", "restart", UNIT)),
    ]


def restore_ollama_steps() -> list[Step]:
    return [
        Step("remove OllamaPower's drop-in (Ollama goes back to 11434)", ("rm", "-f", "--", DROPIN)),
        Step("reload systemd", ("systemctl", "daemon-reload")),
        Step("restart Ollama on its usual port", ("systemctl", "restart", UNIT), kind="restore-done"),
    ]


def _who(nets: dict[str, str]) -> str:
    return ", ".join(f"{n} ({ifaces})" for n, ifaces in nets.items()) or "no network"


def plan_hub_setup(ollama_url: str, frontdoor_port: int, control_port: int, control_open: bool) -> Plan:
    host, port = ollama_target(ollama_url)
    if port == valid_port(frontdoor_port):
        raise SetupError(f"ollama.url and the front door both use port {port}; give Ollama another one")
    if port == DEFAULT_OLLAMA_PORT:
        raise SetupError("ollama.url is Ollama's usual port: nothing to move (set it to e.g. 127.0.0.1:11435)")
    if not has_systemd_ollama():
        raise SetupError("Ollama doesn't run as a systemd service here")
    elsewhere = ollama_host_set_elsewhere()
    if elsewhere:
        raise SetupError(
            f"Ollama's address is already set in {', '.join(elsewhere)}. Remove the OLLAMA_HOST (or "
            "EnvironmentFile) line there first - `sudo systemctl edit ollama` opens it - so that setup and "
            "uninstall alone own Ollama's port, then run setup again"
        )
    plan = Plan(steps=move_ollama_steps(host, port))
    plan.summary = (
        f"Ollama moves to {host}:{port} (private to this PC) and restarts: anything it is generating or "
        "downloading right now stops."
    )
    if control_open:
        valid_port(control_port)
        plan.firewall = active_firewall()
        if plan.firewall:
            nets = lan_networks()
            plan.rules = [(net, control_port) for net in nets]
            plan.steps += firewall_steps(plan.firewall, plan.rules)
            plan.summary += (
                f" Helpers on {_who(nets)} may reach port {control_port}, which only lets in helpers that hold"
                " your cluster token. Ollama's own API is not opened to anyone."
            )
        else:
            plan.notes.append(f"No active firewall (ufw/firewalld) found: port {control_port} needs nothing.")
    return plan


def plan_helper_setup(relay_port: int) -> Plan:
    if relay_port == 0:
        raise SetupError("agent.relay_port is 0 (any free port), which no firewall rule can allow: set it to e.g. 9001")
    plan = Plan(firewall=active_firewall())
    if plan.firewall:
        nets = lan_networks()
        plan.rules = [(net, valid_port(relay_port)) for net in nets]
        plan.steps = firewall_steps(plan.firewall, plan.rules)
        plan.summary = (
            f"The hub, from {_who(nets)}, may reach port {relay_port}. It only listens during a split session"
            " and only lets in your hub (its address and its pinned key)."
        )
    else:
        plan.notes.append(f"No active firewall (ufw/firewalld) found: port {relay_port} needs nothing.")
    return plan


def plan_uninstall(record: dict[str, object], hub: bool, port: int) -> Plan:
    """Undo what the record says was done. A record that isn't one we wrote (tampered, hand-edited) is
    refused rather than half-trusted: only ufw/firewalld rules on private LAN networks for ``port``."""
    plan = Plan()
    if hub and (record.get("dropin") or Path(DROPIN).exists()):
        plan.steps += restore_ollama_steps()
    backend, raw = record.get("firewall"), record.get("rules") or []
    try:
        if not isinstance(raw, list) or (raw and backend not in ("ufw", "firewalld")):
            raise ValueError("unknown firewall")
        rules = [_rule(str(n), p) for n, p in raw]
        if any(p != port for _, p in rules):
            raise ValueError(f"a rule for another port than {port}")
    except (ValueError, TypeError) as exc:
        raise SetupError(
            f"the setup record doesn't look like one OllamaPower wrote ({exc}); remove it and check the firewall "
            "by hand (`sudo ufw status` or `sudo firewall-cmd --list-rich-rules`)"
        ) from None
    if rules:
        plan.firewall, plan.rules = str(backend), rules
        plan.steps += firewall_steps(plan.firewall, rules, remove=True)
    return plan


# ----- running -----------------------------------------------------------------------------

Runner = Callable[[Step], subprocess.CompletedProcess[str]]
SUDO_TIMEOUT = 600  # includes typing the password


def sudo_runner(step: Step) -> subprocess.CompletedProcess[str]:
    sudo = "/usr/bin/sudo" if os.path.exists("/usr/bin/sudo") else shutil.which("sudo")
    if not sudo:
        raise SetupError("sudo is not installed")
    try:
        return subprocess.run(
            [sudo, "--", *step.argv],
            input=step.stdin,
            capture_output=True,
            text=True,
            check=False,
            timeout=SUDO_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        raise SetupError(f"`{step.shown().splitlines()[0]}` didn't finish in {SUDO_TIMEOUT} s") from None


def _ufw_has(listing: str, net: str, port: int) -> bool:
    spec = " ".join(("allow", *ufw_spec(net, port)))
    return any(spec in line for line in listing.splitlines())


def execute(plan: Plan, run: Runner, adding: bool, changed: list[tuple[str, int]]) -> bool:
    """Run the steps in order, stopping at the first failure. ``changed`` collects the firewall rules
    really added (or removed) as it goes, so the caller records them even when a later step fails.
    Returns whether Ollama's port was restored (uninstall's last Ollama step succeeded)."""
    listing, exists, restored = "", False, False
    for step in plan.steps:
        if adding and step.rule is not None and (exists or _ufw_has(listing, *step.rule)):
            exists = False
            continue  # the owner had this rule already: leave it, and never remove it later
        result = run(step)
        if step.kind == "query":
            exists = result.returncode == 0  # firewall-cmd --query-*: 0 = yes, 1 = no
            continue
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()[:300]
            raise SetupError(f"`{step.shown().splitlines()[0]}` failed: {detail}")
        if step.kind == "ufw-list":
            listing = result.stdout
        elif step.kind == "restore-done":
            restored = True
        elif step.rule is not None:
            changed.append(step.rule)
    return restored


def read_record(config_dir: Path, name: str = RECORD) -> dict[str, object]:
    try:
        data = json.loads((config_dir / name).read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def write_record(config_dir: Path, record: dict[str, object], name: str = RECORD) -> None:
    from .files import write_private

    if not record.get("dropin") and not record.get("rules"):
        (config_dir / name).unlink(missing_ok=True)  # nothing left to undo
        return
    write_private(config_dir / name, json.dumps(record, indent=2).encode())
