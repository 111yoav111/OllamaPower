"""``ollamapower agent ...``: run a helper, and control a running one.

``ollamapower agent --hub 192.168.1.10`` runs it (with its TUI, or headless
with ``--no-tui``). ``pause``, ``resume``, ``status``, ``rules`` and
``net-test`` talk to the running agent over its own local admin channel
(``agent-run`` in the config directory), so a hub and an agent can share one
machine.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections import deque
from pathlib import Path
from typing import Any

import typer

from ..cli_common import Lines, call, configure_logging, gb, refuse_root, show, stop_like_ctrl_c, ts
from ..config import ConfigError, default_config_dir
from ..textsafe import safe

agent_cli = typer.Typer(help="Run this PC as a helper, or control the running helper")
RUN_DIR = "agent-run"

ConfigDirOption = typer.Option(
    None,
    "--config-dir",
    "-c",
    file_okay=False,
    help="Directory with agent.toml, .env and known_hub (default: the OllamaPower config dir)",
)


def _lent_text(lent: dict[str, Any] | None) -> str:
    if not lent:
        return "nothing"
    return (
        f"{lent['device']} ({lent['cap_mib']} MiB) for {lent['model'] or 'a model'}, "
        f"{lent.get('tunnel') or '?'} tunnel, since {ts(lent['since'])}"
    )


def _llama_text(lc: dict[str, Any]) -> str:
    if lc.get("status") == "ready":
        return f"build {lc.get('build')}, " + (", ".join(d["name"] for d in lc.get("devices", [])) or "CPU only")
    return f"{lc.get('status', '?')} {lc.get('error', '')}".strip()


def _config_dir(value: Path | None) -> Path:
    import os

    if value is not None:
        return value
    env = os.environ.get("OLLAMAPOWER_CONFIG_DIR")
    return Path(env) if env else default_config_dir()


def _load_agent(config_dir: Path) -> Any:
    from .config import load_agent_config

    try:
        return load_agent_config(config_dir)
    except ConfigError as exc:
        typer.secho(f"Config error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from None


def parse_hub_address(value: str, default_port: int) -> tuple[str, int]:
    """``192.168.1.10``, ``192.168.1.10:9000``, ``[fd00::1]:9000`` or ``hub.lan``."""
    value = value.strip()
    port: str | None = None
    if value.startswith("["):
        host, closed, rest = value[1:].partition("]")
        if not closed:
            raise typer.BadParameter(f"{value!r}: an IPv6 address needs its closing ']'")
        if rest and not rest.startswith(":"):
            raise typer.BadParameter(f"{value!r}: expected [address]:port")
        port = rest[1:] if rest else None
    elif value.count(":") == 1:
        host, _, port = value.partition(":")
    else:
        host = value  # a name, an IPv4 address or a bare IPv6 address
    if not host:
        raise typer.BadParameter("missing host")
    if "[" in host or "]" in host or any(ch.isspace() for ch in host):
        raise typer.BadParameter(f"bad host {host!r}")
    if port is None:
        return host, default_port
    if not (port.isascii() and port.isdigit()) or not 1 <= int(port) <= 65535:
        raise typer.BadParameter(f"bad port {port!r} (1-65535)")
    return host, int(port)


@agent_cli.callback(invoke_without_command=True)
def agent(
    ctx: typer.Context,
    hub: str = typer.Option(
        "", "--hub", help="The hub's address: ip[:port] or [ipv6]:port (or set hub.address in agent.toml)"
    ),
    hub_fingerprint: str = typer.Option("", "--hub-fingerprint", help="Trust this hub key on first connect"),
    name: str = typer.Option("", "--name", help="How this helper appears on the hub"),
    no_tui: bool = typer.Option(False, "--no-tui", help="Run headless with plain logs"),
    allow_root: bool = typer.Option(False, "--allow-root", help="Run even as root/Administrator (not recommended)"),
    config_dir: Path | None = ConfigDirOption,
) -> None:
    """Lend this PC's GPU/CPU to the hub. Subcommands control a running agent."""
    if ctx.invoked_subcommand is not None:
        return
    from ..admin.ipc import start_control_server
    from ..files import token_problem
    from .client import AgentClient, AgentStopped
    from .pinning import format_fingerprint
    from .service import AgentAdmin

    cfg = _load_agent(_config_dir(config_dir))
    refuse_root(allow_root, "agent")
    address, port = cfg.hub.address, cfg.hub.port
    if hub:
        address, port = parse_hub_address(hub, cfg.hub.port)
    if not address:
        typer.secho("Where is the hub? Use --hub <ip>[:port] or set hub.address in agent.toml.", fg="red", err=True)
        raise typer.Exit(2)
    if hub_fingerprint:
        cfg.hub.fingerprint = hub_fingerprint
    if name:
        cfg.agent.name = name
    token, problem = token_problem(cfg.env_path)
    if problem or token is None:
        typer.secho(
            f"{problem}. On the hub run `ollamapower token show` and put that line in {cfg.env_path} "
            "(or set OLLAMAPOWER_TOKEN to it).",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(2)

    log_lines: deque[str] = deque(maxlen=200)
    configure_logging("info", None if no_tui else log_lines)
    log = logging.getLogger("ollamapower.agent")

    async def ask_in_terminal(addr: str, fp: str) -> bool:
        if not sys.stdin.isatty():
            return False
        typer.echo(
            f"\nFirst connection to the hub at {addr}. Its key fingerprint (SHA-256):\n  {format_fingerprint(fp)}"
        )
        typer.echo("Compare it with what `ollamapower hub` shows on the hub PC.")
        return await asyncio.to_thread(typer.confirm, "Trust this hub?", default=False)

    client = AgentClient(cfg, token, address, port, confirm=ask_in_terminal if no_tui else None)

    async def main() -> int:
        try:
            control = await start_control_server(cfg.config_dir, AgentAdmin(client).dispatch, name=RUN_DIR)
        except RuntimeError as exc:
            typer.secho(str(exc), fg=typer.colors.RED, err=True)
            return 1
        try:
            if not no_tui and await _run_agent_tui(client, log_lines):
                return 0
            if no_tui:
                log.info("helper %s connecting to the hub at %s:%d", cfg.display_name, address, port)
            client.confirm = ask_in_terminal
            try:
                await client.run()
            except AgentStopped as exc:
                typer.secho(f"Stopped: {exc}", fg=typer.colors.RED, err=True)
                return 1
        except _TUIStopped as exc:
            typer.secho(f"Stopped: {exc}", fg=typer.colors.RED, err=True)
            return 1
        finally:
            await control.close()
        return 0

    try:
        with stop_like_ctrl_c():  # kill / systemd / closing the window: stop rpc-server too
            code = asyncio.run(main())
    except KeyboardInterrupt:
        code = 0
    raise typer.Exit(code)


class _TUIStopped(Exception):
    """The agent stopped for good while its TUI ran (wrong token, changed hub key, ...)."""


async def _run_agent_tui(client: Any, log_lines: deque[str]) -> bool:
    """Run the TUI. True when the owner quit; False when it crashed and the agent should go on headless."""
    from .tui.app import AgentTUI

    app = AgentTUI(client)
    crashed = None
    try:
        await app.run_async()
    except Exception as exc:  # noqa: BLE001 - the helper must survive a TUI bug
        crashed = exc
    if app.stopped_reason:
        raise _TUIStopped(app.stopped_reason)
    if app.quit_confirmed:
        return True
    configure_logging("info")
    for line in log_lines:
        typer.echo(line, err=True)
    typer.secho(
        f"The TUI stopped unexpectedly ({crashed!r}); the agent keeps running headless. Press Ctrl+C to stop it.",
        fg=typer.colors.RED,
        err=True,
    )
    return False


# ----- control a running agent ---------------------------------------------------------


def render_agent(s: dict[str, Any]) -> Lines:
    from .pinning import format_fingerprint

    r = s["lending"]
    tel = s["telemetry"]
    rows = [
        ("Hub", s["hub"]),
        ("Connection", s["connection"] + (f" ({s['error']})" if s["error"] else "")),
        (
            "Hub key",
            (format_fingerprint(s["fingerprint"]) if s["fingerprint"] else "-") + (" (pinned)" if s["pinned"] else ""),
        ),
        ("Helper", f"{s['name']} ({s['node_id']})"),
        (
            "Lending",
            "on"
            if s.get("lend_state", "lending") == "lending"
            else f"{s.get('lend_state', '').upper()} ({s.get('lend_reason', '')})",
        ),
        ("Lent now", _lent_text(s.get("lent"))),
        ("llama.cpp", _llama_text(s.get("llama_cpp") or {})),
        ("Rules", json.dumps({k: v for k, v in r.items() if k != "paused"})),
        ("Link", f"{s['network'].get('kind', '?')} via {s['network'].get('interface', '?')}"),
    ]
    lines: Lines = [
        (
            f"{label + ':':12} {safe(value)}",
            typer.colors.RED if label == "Connection" and s["connection"] == "stopped" else None,
        )
        for label, value in rows
    ]
    for g in tel.get("gpus", []):
        lines.append(
            (
                f"{'GPU ' + str(g.get('index')) + ':':12} {gb(g.get('vram_used'))}/{gb(g.get('vram_total'))} GB, "
                f"{g.get('util', '-')}% busy",
                None,
            )
        )
    if tel.get("cpu_percent") is not None:
        lines.append(
            (
                f"{'CPU/RAM:':12} {tel['cpu_percent']:.0f}%, {gb(tel.get('ram_used'))}/{gb(tel.get('ram_total'))} GB",
                None,
            )
        )
    for e in s["events"][-5:]:
        lines.append((f"  {ts(e['ts'])}  {safe(e['message'])}", None))
    return lines


@agent_cli.command("status")
def agent_status(
    config_dir: Path | None = ConfigDirOption,
    as_json: bool = typer.Option(False, "--json", help="Print JSON"),
    watch: bool = typer.Option(False, "--watch", "-w", help="Refresh every 2 seconds until Ctrl+C"),
) -> None:
    """Show the running agent: connection, hub key, lending, GPU use."""
    d = _config_dir(config_dir)
    show(lambda: call(d, "agent.status", name=RUN_DIR), render_agent, as_json, watch)


@agent_cli.command("pause")
def agent_pause(config_dir: Path | None = ConfigDirOption) -> None:
    """Stop lending right now (the hub moves work elsewhere)."""
    call(_config_dir(config_dir), "agent.pause", name=RUN_DIR)
    typer.echo("Lending paused.")


@agent_cli.command("resume")
def agent_resume(config_dir: Path | None = ConfigDirOption) -> None:
    """Lend again."""
    call(_config_dir(config_dir), "agent.resume", name=RUN_DIR)
    typer.echo("Lending resumed.")


@agent_cli.command("net-test")
def agent_net_test(config_dir: Path | None = ConfigDirOption) -> None:
    """Ask the hub to measure the link to this PC."""
    r = call(_config_dir(config_dir), "agent.net_test", name=RUN_DIR, timeout=70)
    if "error" in r:
        typer.secho(safe(r["error"]), fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    typer.echo(f"RTT {r['rtt_ms']} ms, hub -> here {r['down_mbps']} Mbps, here -> hub {r['up_mbps']} Mbps")


@agent_cli.command("rules")
def agent_rules(
    config_dir: Path | None = ConfigDirOption,
    max_vram_gb: float | None = typer.Option(None, help="0 = no cap"),
    max_vram_percent: int | None = typer.Option(None, help="0 = no cap"),
    gpus: str | None = typer.Option(None, help='GPU indexes to lend, e.g. "0,1"; "" = all'),
    allow_cpu: bool | None = typer.Option(None, "--allow-cpu/--no-cpu"),
    allow_direct: bool | None = typer.Option(
        None, "--allow-direct/--no-direct", help="Allow unencrypted trusted-LAN mode"
    ),
    disk_cache: bool | None = typer.Option(
        None, "--disk-cache/--no-disk-cache", help="Keep received model weights on disk for faster reloads"
    ),
    schedule: str | None = typer.Option(
        None, help='When to lend, local time, e.g. "22:00-08:00" or "09:00-12:00,20:00-23:00"; "" = always'
    ),
    auto_yield: bool | None = typer.Option(
        None, "--auto-yield/--no-auto-yield", help="Step back when you use the GPU yourself"
    ),
    yield_gpu_percent: int | None = typer.Option(None, help="GPU load that counts as you using it (1-100)"),
    yield_seconds: int | None = typer.Option(None, help="For how long, before stepping back (and back in)"),
) -> None:
    """Show or change the lending rules of the running agent (saved to agent.toml)."""
    changes: dict[str, Any] = {
        k: v
        for k, v in {
            "max_vram_gb": max_vram_gb,
            "max_vram_percent": max_vram_percent,
            "allow_cpu": allow_cpu,
            "allow_direct": allow_direct,
            "disk_cache": disk_cache,
            "auto_yield": auto_yield,
            "yield_gpu_percent": yield_gpu_percent,
            "yield_seconds": yield_seconds,
        }.items()
        if v is not None
    }
    if gpus is not None:
        parts = [g.strip() for g in gpus.split(",") if g.strip()]
        if not all(p.isdigit() for p in parts):
            raise typer.BadParameter("GPU indexes like 0,1")
        changes["gpus"] = [int(p) for p in parts]
    if schedule is not None:
        changes["schedule"] = [w.strip() for w in schedule.split(",") if w.strip()]
    from .config import LendingRules

    try:  # a bad value is a usage error (exit 2), caught before the running agent is asked
        LendingRules(**changes).validate()
    except ConfigError as exc:
        raise typer.BadParameter(str(exc)) from None
    d = _config_dir(config_dir)
    rules = (
        call(d, "agent.rules", name=RUN_DIR, **changes) if changes else call(d, "agent.status", name=RUN_DIR)["lending"]
    )
    for key, value in rules.items():
        typer.echo(f"{key:18} {safe(value)}")


@agent_cli.command("example-config")
def agent_example_config() -> None:
    """Print an agent.toml listing every setting (save it with `> agent.toml` in the config dir)."""
    from importlib import resources

    typer.echo(resources.files("ollamapower").joinpath("agent.example.toml").read_text(encoding="utf-8"), nl=False)


@agent_cli.command("forget-hub")
def agent_forget_hub(
    hub: str = typer.Option(
        "", "--hub", help="The hub to forget: ip[:port] or [ipv6]:port, as given to --hub (default: all)"
    ),
    config_dir: Path | None = ConfigDirOption,
    yes: bool = typer.Option(False, "--yes", "-y", help="Don't ask for confirmation"),
) -> None:
    """Forget a pinned hub key (only after you replaced the hub's key yourself)."""
    from .pinning import KnownHubs, hub_entry

    cfg = _load_agent(_config_dir(config_dir))
    entry = hub_entry(*parse_hub_address(hub, cfg.hub.port)) if hub else None
    known = KnownHubs(cfg.known_hub_path)
    if entry is not None and known.get(entry) is None:
        pinned = ", ".join(sorted(known.entries())) or "none"
        typer.secho(f"No pinned key for {entry} (pinned: {pinned}).", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    if not yes:
        what = f"the pinned key of {entry}" if entry else "every pinned hub key"
        typer.confirm(f"Forget {what}? The next connection will ask you to trust it again.", abort=True)
    removed = known.forget(entry)
    typer.echo(f"Forgot {removed} pinned hub key(s).")
