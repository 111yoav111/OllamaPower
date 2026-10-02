"""Command-line entry point.

``hub`` runs the hub and ``agent`` runs a helper. Commands that show or change
a running process (``status``, ``nodes``, ``models``, ``agent pause|resume|
status``) talk to it over the local-only admin channel (see ``admin/ipc.py``);
``init``, ``doctor``, ``logs`` and ``token`` work on the files directly, so
they also work while nothing runs.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from importlib import resources
from pathlib import Path
from typing import Any

import typer

from . import __version__
from .agent.cli import agent_cli
from .cli_common import (
    ConfigOption,
    Lines,
    call,
    configure_logging,
    duration,
    gb,
    load,
    refuse_root,
    show,
    ts,
)
from .textsafe import safe

cli = typer.Typer(add_completion=False, no_args_is_help=True, help="OllamaPower - cluster-boosted Ollama")
nodes_cli = typer.Typer(no_args_is_help=True, help="Helpers known to the hub")
models_cli = typer.Typer(no_args_is_help=True, help="Models and their plans")
token_cli = typer.Typer(no_args_is_help=True, help="The cluster token helpers need")
cli.add_typer(agent_cli, name="agent")
cli.add_typer(nodes_cli, name="nodes")
cli.add_typer(models_cli, name="models")
cli.add_typer(token_cli, name="token")

STATUS_ICON = {"lending": "🟢", "paused": "🟡", "yielding": "🟡", "offline": "🔴", "incompatible": "⚠ "}


@cli.callback()
def _root() -> None:
    """OllamaPower - cluster-boosted Ollama."""


@cli.command()
def version() -> None:
    """Show the OllamaPower version."""
    typer.echo(f"ollamapower {__version__}")


# ----- init / token --------------------------------------------------------------------


@cli.command()
def init(
    config_path: Path | None = ConfigOption,
    rotate_token: bool = typer.Option(
        False, "--rotate-token", help="Replace an existing token (helpers need the new one)"
    ),
) -> None:
    """Create the config directory, a cluster token in .env, config.toml and an empty state.json."""
    from .core.auth import TOKEN_ENV, new_token
    from .files import load_or_create_hub_key, load_token, private_dir, write_private, write_token
    from .state import init_state_file

    config = load(config_path)
    private_dir(config.config_dir)
    typer.echo(f"Config directory: {config.config_dir}")

    existing = load_token(config.env_path, environ={})
    if existing and not rotate_token:
        typer.echo(f"  {config.env_path.name}: {TOKEN_ENV} already set (use --rotate-token to replace it)")
    else:
        write_token(config.env_path, new_token())
        typer.echo(f"  {config.env_path.name}: {'new' if not existing else 'rotated'} {TOKEN_ENV} written (owner-only)")
        if existing:
            typer.secho("  Every helper needs the new token before it can reconnect.", fg=typer.colors.YELLOW)

    target = config.config_dir / "config.toml"
    if target.exists():
        typer.echo("  config.toml: kept")
    else:
        example = resources.files("ollamapower").joinpath("config.example.toml").read_bytes()
        write_private(target, example)
        typer.echo("  config.toml: created from the example")

    typer.echo(f"  state.json: {'created' if init_state_file(config.state_path) else 'kept'}")
    created = not config.hub_key_path.exists()
    key = load_or_create_hub_key(config.hub_key_path)
    from .agent.pinning import format_fingerprint
    from .core.encrypt_layer import public_key_fingerprint

    typer.echo(f"  hub_key.pem: {'created' if created else 'kept'} (owner-only)")
    typer.echo(
        "\nHub key fingerprint (helpers confirm this on first connect):\n  "
        + format_fingerprint(public_key_fingerprint(key.public_key()))
    )
    typer.echo("\nNext: `ollamapower doctor` checks Ollama and walks you through moving it to port 11435.")


@token_cli.command("show")
def token_show(config_path: Path | None = ConfigOption) -> None:
    """Print the cluster token, to put in each helper's .env (keep it secret)."""
    from .core.auth import TOKEN_ENV
    from .files import load_token

    config = load(config_path)
    token = load_token(config.env_path)
    if not token:
        typer.secho("No token yet: run `ollamapower init`.", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    typer.secho(
        "Anyone with this token can join your cluster as a helper. Copy it only to your helpers.",
        err=True,
        fg=typer.colors.YELLOW,
    )
    typer.echo(f'{TOKEN_ENV}="{token}"')


# ----- hub -----------------------------------------------------------------------------


@cli.command()
def hub(
    config_path: Path | None = ConfigOption,
    no_tui: bool = typer.Option(False, "--no-tui", help="Run headless with plain logs"),
    allow_root: bool = typer.Option(False, "--allow-root", help="Run even as root/Administrator (not recommended)"),
    i_know: bool = typer.Option(
        False, "--i-know-what-im-doing", help="Allow hub.host to be a non-loopback address", hidden=True
    ),
) -> None:
    """Run the hub: the Ollama-compatible front door, the queue, the control plane and the TUI."""
    import socket

    from .admin.ipc import start_control_server
    from .admin.service import Admin
    from .agent.pinning import format_fingerprint
    from .config import is_loopback_host
    from .core.auth import TokenError, validate_token
    from .core.encrypt_layer import public_key_fingerprint
    from .eventlog import EventLog, NullEventLog
    from .files import load_or_create_hub_key, load_token, private_dir
    from .hub.frontdoor.app import create_app
    from .hub.frontdoor.server import build_server
    from .hub.runtime import Hub
    from .state import StateError, StateStore

    config = load(config_path)
    refuse_root(allow_root, "hub")
    if not is_loopback_host(config.hub.host):
        if not i_know:
            typer.secho(
                f"Refusing to listen on {config.hub.host}: the front door has no authentication, so anyone on that "
                'network could use your models. Keep hub.host = "127.0.0.1" (use OllamaGate for remote access), '
                "or pass --i-know-what-im-doing.",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(2)
        typer.secho(f"WARNING: the front door listens on {config.hub.host} without authentication.", fg="red", err=True)

    log_lines: deque[str] = deque(maxlen=200)
    configure_logging(config.logging.level, None if no_tui else log_lines)
    log = logging.getLogger("ollamapower")

    try:
        private_dir(config.config_dir)
        state = StateStore(config.state_path)
        state.load()
        key = load_or_create_hub_key(config.hub_key_path)
    except (StateError, OSError, ValueError) as exc:
        typer.secho(f"Startup error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from None
    try:
        events: EventLog = EventLog(config.logs_dir, config.logging.file_max_bytes, config.logging.file_backups)
    except OSError as exc:
        log.warning("cannot write log files in %s (%s); request/event logs disabled", config.logs_dir, exc)
        events = NullEventLog()

    the_hub = Hub(config, state, events)
    token = load_token(config.env_path)
    try:
        validate_token(token)
    except TokenError as exc:
        token = None
        the_hub.extra_warnings.append(f"{exc}. Helpers can't join until then.")
    app = create_app(the_hub)
    server = build_server(the_hub, app, own_logging=False)
    try:
        sock = server.config.bind_socket()  # bind now so "address in use" is reported clearly
    except SystemExit:
        typer.secho(
            f"Cannot listen on {config.frontdoor_url}: the port is in use. If Ollama still runs there, "
            "run `ollamapower doctor` for the steps to move it to port 11435.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1) from None
    control_sock = None
    if config.control.enabled and token:
        try:
            control_sock = socket.create_server((config.control.host, config.control.port), reuse_port=False)
        except OSError as exc:
            typer.secho(f"Cannot open the control plane on port {config.control.port}: {exc}", fg="red", err=True)
            raise typer.Exit(1) from None
    fingerprint = format_fingerprint(public_key_fingerprint(key.public_key()))
    if no_tui:
        typer.echo(f"OllamaPower {__version__}: front door {config.frontdoor_url} -> Ollama {config.ollama.url}")
        if control_sock is not None:
            typer.echo(f"Helpers connect to port {config.control.port}; hub key fingerprint:\n  {fingerprint}")

    async def main() -> None:
        try:
            control = await start_control_server(config.config_dir, Admin(the_hub).dispatch)
        except RuntimeError as exc:
            typer.secho(str(exc), fg=typer.colors.RED, err=True)
            raise typer.Exit(1) from None
        if control_sock is not None:
            await the_hub.start_control(key, token, sock=control_sock)
        else:
            the_hub.fingerprint = public_key_fingerprint(key.public_key())
        the_hub.event("hub_start", version=__version__, frontdoor=config.frontdoor_url, ollama=config.ollama.url)
        log.info("front door on %s -> Ollama at %s", config.frontdoor_url, config.ollama.url)
        for warning in the_hub.warnings():
            log.warning("%s", warning)
        server_task = asyncio.ensure_future(server.serve(sockets=[sock]))
        try:
            if not no_tui:
                await _run_hub_tui(the_hub, server, server_task, log_lines, config.logging.level)
            await server_task
        finally:
            await the_hub.stop_control()
            await control.close()
            try:
                await state.flush()
            except OSError as exc:
                log.error("could not save state: %s", exc)
            the_hub.event("hub_stop")
            events.close()

    asyncio.run(main())


async def _run_hub_tui(the_hub: Any, server: Any, server_task: Any, log_lines: deque[str], level: str) -> None:
    from .admin.service import Admin
    from .hub.tui.app import HubTUI

    tui = HubTUI(Admin(the_hub), services=server_task, log_lines=log_lines)
    crashed = None
    try:
        await tui.run_async()
    except Exception as exc:  # noqa: BLE001 - the hub must survive a TUI bug
        crashed = exc
    if tui.quit_confirmed or server_task.done():
        server.should_exit = True
        return
    # The TUI died on its own: keep serving clients and helpers headless.
    configure_logging(level)
    for line in log_lines:
        typer.echo(line, err=True)
    typer.secho(
        f"The TUI stopped unexpectedly ({crashed!r}); the hub keeps running headless. Use the CLI or Ctrl+C to stop.",
        fg=typer.colors.RED,
        err=True,
    )


# ----- status --------------------------------------------------------------------------


def _human(n: int) -> str:
    from .ollama.client import human_bytes

    return human_bytes(n)


def _link(link: dict[str, Any]) -> str:
    parts = [str(link.get("kind") or "?")]
    if link.get("rtt_ms") is not None:
        parts.append(f"{link['rtt_ms']} ms")
    if link.get("down_mbps") is not None:
        parts.append(f"{link['down_mbps']} Mbps")
    return ", ".join(parts)


def render_nodes(nodes: list[dict[str, Any]]) -> Lines:
    if not nodes:
        return [("No helpers yet. Run `ollamapower agent --hub <this PC's IP>` on a helper.", None)]
    lines: Lines = [
        (f"   {'ID':9} {'NAME':18} {'IP':16} {'STATUS':13} {'GPU':22} {'VRAM FREE/LEND':15} {'LINK':24} TUNNEL", None)
    ]
    for n in nodes:
        gpus = n["gpus"]
        gpu = (
            "CPU only" if not gpus else (gpus[0].get("name") or "GPU") + (f" +{len(gpus) - 1}" if len(gpus) > 1 else "")
        )
        color = typer.colors.YELLOW if n["warnings"] else None
        lines.append(
            (
                f"{STATUS_ICON.get(n['status'], '? ')} {n['id'][:8]:9} {safe(n['name'])[:18]:18} "
                f"{safe(n['ip'])[:16]:16} "
                f"{n['status']:13} {safe(gpu)[:22]:22} {gb(n['vram_free']) + '/' + gb(n['vram_lendable']) + ' GB':15} "
                f"{safe(_link(n['link']))[:24]:24} {n['tunnel_effective']}",
                color,
            )
        )
    return lines


def render_status(s: dict[str, Any]) -> Lines:
    """The status dict as (text, color) lines; shared by the plain and --watch views."""
    from .agent.pinning import format_fingerprint

    o, q, r, c = s["ollama"], s["queue"], s["requests"], s["control"]
    lines: Lines = []
    waiting = ", ".join(f"{n} {p}" for p, n in q["waiting_by_priority"].items() if n) or "none"
    nodes = s["nodes"]
    lending = [n for n in nodes if n["status"] == "lending"]
    rows = [
        ("Front door", s["frontdoor"]["url"]),
        ("Ollama", f"{o['version']} at {o['url']}" if o["reachable"] else f"UNREACHABLE at {o['url']} ({o['error']})"),
        ("Uptime", duration(s["uptime_s"])),
        ("Queue", f"{q['active']}/{q['capacity']} running, waiting: {waiting}"),
        ("Requests", f"{r['last_minute']}/min, {r['total']} total, {r['failed']} failed"),
        (
            "Helpers",
            f"{len(lending)} lending, {len(nodes)} known, "
            f"{gb(sum(n['vram_lendable'] for n in lending))} GB VRAM lendable",
        ),
        ("Control", f"port {c['port']}" if c["enabled"] else "off (no cluster token? run `ollamapower init`)"),
        ("Hub key", format_fingerprint(c["fingerprint"]) if c["fingerprint"] else "-"),
    ]
    for label, value in rows:
        lines.append((f"{label + ':':12} {safe(value)}", None))
    if o["models"]:
        lines.append(("", None))
        lines.append((f"{'MODEL':40} {'SIZE':>10} {'VRAM':>5} {'ENGINE':14} TOK/S", None))
        for m in o["models"]:
            tps = "-" if m["last_gen_tps"] is None else f"{m['last_gen_tps']:.1f}"
            color = typer.colors.YELLOW if m["spilled"] else None
            lines.append(
                (
                    f"{safe(m['name'])[:40]:40} {_human(m['size']):>10} {m['vram_percent']:>4}% {m['engine']:14} {tps}",
                    color,
                )
            )
    if nodes:
        lines.append(("", None))
        lines.extend(render_nodes(nodes))
    if s["streams"]:
        lines.append(("", None))
        lines.append((f"{'ID':>5} {'STATE':8} {'PRIORITY':9} {'AGE':>7} {'MODEL':30} PATH", None))
        for st in s["streams"]:
            lines.append(
                (
                    f"{st['id']:>5} {st['state']:8} {st['priority']:9} {st['age_s']:>6.0f}s "
                    f"{safe(st['model'])[:30]:30} {safe(st['path'])}",
                    None,
                )
            )
    if s["warnings"]:
        lines.append(("", None))
        for w in s["warnings"]:
            lines.append((f"⚠ {safe(w)}", typer.colors.RED))
    return lines


@cli.command()
def status(
    config_path: Path | None = ConfigOption,
    as_json: bool = typer.Option(False, "--json", help="Print the raw status JSON"),
    watch: bool = typer.Option(False, "--watch", "-w", help="Refresh every 2 seconds until Ctrl+C"),
) -> None:
    """Show the hub: front door, Ollama, queue, models, helpers, active requests, warnings."""
    config = load(config_path)
    show(lambda: call(config.config_dir, "status"), render_status, as_json, watch)


# ----- nodes ---------------------------------------------------------------------------


@nodes_cli.command("list")
def nodes_list(
    config_path: Path | None = ConfigOption,
    as_json: bool = typer.Option(False, "--json", help="Print JSON"),
    watch: bool = typer.Option(False, "--watch", "-w", help="Refresh every 2 seconds until Ctrl+C"),
) -> None:
    """List helpers: status, GPU, free and lendable VRAM, link, tunnel mode."""
    config = load(config_path)
    show(lambda: call(config.config_dir, "nodes.list"), render_nodes, as_json, watch)


def render_node(d: dict[str, Any]) -> Lines:
    caps = d.get("capabilities", {})
    cpu = caps.get("cpu") or {}
    ollama = caps.get("ollama") or {}
    rows = [
        ("ID", d["id"]),
        ("Name", d["name"]),
        ("Status", d["status"] + (f" ({d['lend_reason']})" if d["lend_reason"] else "")),
        ("Address", d["ip"]),
        ("Agent", f"{d['agent_version']} (protocol {d['protocol']})"),
        ("System", f"{caps.get('os', '?')} {caps.get('os_release', '')} {caps.get('arch', '')}".strip()),
        ("CPU", f"{cpu.get('model', '?')}, {cpu.get('cores', '?')} cores"),
        ("RAM", f"{gb(d['ram_used'])} / {gb(d['ram_total'])} GB"),
        ("VRAM", f"{gb(d['vram_free'])} free of {gb(d['vram_total'])} GB, {gb(d['vram_lendable'])} GB lendable"),
        ("Link", _link(d["link"])),
        ("Tunnel", f"{d['tunnel_effective']} (requested {d['tunnel_mode']})"),
        (
            "Ollama",
            f"{ollama.get('version', '')}, {len(ollama.get('models') or [])} models"
            if ollama.get("present")
            else "not found",
        ),
        ("Lending", json.dumps(d["lending"])),
        ("Connected", ts(d["connected_at"]) if d["connected"] else f"no (last seen {ts(d['last_seen'])})"),
    ]
    lines: Lines = [(f"{label + ':':12} {safe(value)}", None) for label, value in rows]
    for g in d["gpus"]:
        util = f", {g['util']}% busy" if g.get("util") is not None else ""
        lines.append(
            (
                f"{'GPU ' + str(g.get('index')) + ':':12} {safe(g.get('name'))} ({g.get('backend')}), "
                f"{gb(g.get('vram_used'))}/{gb(g.get('vram_total'))} GB{util}",
                None,
            )
        )
    for test in d.get("net_tests", [])[-5:]:
        lines.append(
            (
                f"{'Net test:':12} {ts(test.get('ts'))}  RTT {test.get('rtt_ms')} ms, "
                f"↓ {test.get('down_mbps')} Mbps, ↑ {test.get('up_mbps')} Mbps",
                None,
            )
        )
    lines.extend((f"⚠ {safe(w)}", typer.colors.YELLOW) for w in d["warnings"])
    return lines


@nodes_cli.command("show")
def nodes_show(
    node: str = typer.Argument(..., help="Name, id or id prefix"),
    config_path: Path | None = ConfigOption,
    as_json: bool = typer.Option(False, "--json", help="Print JSON"),
    watch: bool = typer.Option(False, "--watch", "-w", help="Refresh every 2 seconds until Ctrl+C"),
) -> None:
    """Everything known about one helper."""
    config = load(config_path)
    show(lambda: call(config.config_dir, "nodes.show", ref=node), render_node, as_json, watch)


@nodes_cli.command("pause")
def nodes_pause(node: str, config_path: Path | None = ConfigOption) -> None:
    """Stop using a helper (it stays connected; the helper's owner can't override this)."""
    d = call(load(config_path).config_dir, "nodes.pause", ref=node)
    typer.echo(f"Paused {safe(d['name'])}: the hub won't use it until `ollamapower nodes resume`.")


@nodes_cli.command("resume")
def nodes_resume(node: str, config_path: Path | None = ConfigOption) -> None:
    """Use a hub-paused helper again."""
    d = call(load(config_path).config_dir, "nodes.resume", ref=node)
    typer.echo(f"Resumed {safe(d['name'])}.")


@nodes_cli.command("kick")
def nodes_kick(
    node: str,
    config_path: Path | None = ConfigOption,
    yes: bool = typer.Option(False, "--yes", "-y", help="Don't ask for confirmation"),
) -> None:
    """Disconnect a helper now (it reconnects after a minute; pause it to stop using it)."""
    if not yes:
        typer.confirm(f"Disconnect {node}?", abort=True)
    d = call(load(config_path).config_dir, "nodes.kick", ref=node)
    typer.echo(f"Disconnected {safe(d['name'])}.")


@nodes_cli.command("net-test")
def nodes_net_test(
    node: str, config_path: Path | None = ConfigOption, as_json: bool = typer.Option(False, "--json")
) -> None:
    """Measure round-trip time and throughput both ways to a helper."""
    r = call(load(config_path).config_dir, "nodes.net_test", ref=node, timeout=60)
    if as_json:
        typer.echo(json.dumps(r, indent=2))
        return
    typer.echo(f"RTT {r['rtt_ms']} ms (min {r['rtt_min_ms']}, max {r['rtt_max_ms']})")
    typer.echo(
        f"Hub -> helper {r['down_mbps']} Mbps, helper -> hub {r['up_mbps']} Mbps (over the encrypted control channel)"
    )
    if r.get("link") == "wifi":
        typer.secho("The helper is on Wi-Fi: too slow for split mode. Use a cable.", fg=typer.colors.YELLOW)


@nodes_cli.command("tunnel")
def nodes_tunnel(
    node: str,
    mode: str = typer.Argument(..., help="secure (mTLS, default) or direct (trusted LAN, unencrypted)"),
    config_path: Path | None = ConfigOption,
    yes: bool = typer.Option(False, "--yes", "-y", help="Don't ask for confirmation"),
) -> None:
    """Choose how split-mode traffic to a helper travels (used from Phase 4)."""
    if mode == "direct" and not yes:
        typer.secho(
            "Direct mode sends split-mode traffic (weights, activations) UNENCRYPTED over your LAN. Only use it on a "
            "wired network you fully trust. The helper's owner must also allow it (allow_direct).",
            fg=typer.colors.YELLOW,
        )
        typer.confirm("Use direct mode?", abort=True)
    d = call(load(config_path).config_dir, "nodes.tunnel", ref=node, mode=mode)
    typer.echo(f"{safe(d['name'])}: tunnel {d['tunnel_mode']} (in effect: {d['tunnel_effective']})")
    for w in d["warnings"]:
        typer.secho(f"⚠ {safe(w)}", fg=typer.colors.YELLOW)


@nodes_cli.command("forget")
def nodes_forget(
    node: str,
    config_path: Path | None = ConfigOption,
    yes: bool = typer.Option(False, "--yes", "-y", help="Don't ask for confirmation"),
) -> None:
    """Remove an offline helper and its settings."""
    if not yes:
        typer.confirm(f"Forget {node}?", abort=True)
    call(load(config_path).config_dir, "nodes.forget", ref=node)
    typer.echo("Forgotten.")


# ----- models --------------------------------------------------------------------------


def render_models(models: list[dict[str, Any]]) -> Lines:
    if not models:
        return [("No models (is Ollama reachable?)", None)]
    lines: Lines = [
        (f"{'MODEL':40} {'SIZE':>10} {'PARAMS':>7} {'QUANT':8} {'LOADED':8} {'FITS':6} {'PLAN':13} TOK/S", None)
    ]
    for m in models:
        fits = "?" if m["fits"] is None else ("yes" if m["fits"] else "no")
        loaded = f"{m['vram_percent']}%" if m["loaded"] else "-"
        tps = "-" if m["last_gen_tps"] is None else f"{m['last_gen_tps']:.1f}"
        color = typer.colors.YELLOW if m["fits"] is False else None
        lines.append(
            (
                f"{safe(m['name'])[:40]:40} {_human(m['size']):>10} {safe(m['parameters']):>7} "
                f"{safe(m['quantization']):8} {loaded:8} {fits:6} {m['plan']:13} {tps}",
                color,
            )
        )
    lines.append(("Plans, benchmarks and pinning arrive with the planner (Phase 5). FITS is a rough estimate.", None))
    return lines


@models_cli.command("list")
def models_list(
    config_path: Path | None = ConfigOption,
    as_json: bool = typer.Option(False, "--json", help="Print JSON"),
    watch: bool = typer.Option(False, "--watch", "-w", help="Refresh every 2 seconds until Ctrl+C"),
) -> None:
    """Installed models: size, whether they fit the hub's VRAM, plan, last speed."""
    config = load(config_path)
    show(lambda: call(config.config_dir, "models.list"), render_models, as_json, watch)


# ----- doctor --------------------------------------------------------------------------


@cli.command()
def doctor(config_path: Path | None = ConfigOption) -> None:
    """Check the setup (Ollama's port move, exposure, token, keys, file permissions)."""
    from . import doctor as doc

    config = load(config_path)
    colors = {"ok": typer.colors.GREEN, "warn": typer.colors.YELLOW, "error": typer.colors.RED}
    worst = "ok"
    for f in doc.run_all(config):
        typer.secho(f"[{f.level.upper():5}] {safe(f.title)}", fg=colors[f.level])
        for i, line in enumerate(f.fix.splitlines()):
            typer.echo(f"        {'fix: ' if i == 0 else '     '}{line}")
        if f.level == "error" or (f.level == "warn" and worst == "ok"):
            worst = f.level
    raise typer.Exit(1 if worst == "error" else 0)


# ----- logs ----------------------------------------------------------------------------


def _format_entry(e: dict[str, Any]) -> str:
    stamp = e.get("ts")
    when = ts(stamp) if isinstance(stamp, (int, float)) else "-"
    if e.get("log") == "requests":
        detail = (
            f"{e.get('status', '-')} {e.get('method', '')} {e.get('path', '')} model={e.get('model') or '-'} "
            f"prio={e.get('priority', '-')} queue={e.get('queue_ms', '-')}ms took={e.get('duration_ms', '-')}ms "
            f"{e.get('outcome', '')}"
        )
    else:
        detail = " ".join(f"{k}={v}" for k, v in e.items() if k not in ("ts", "kind", "log"))
    return f"{when}  {safe(e.get('kind', '?')):14} {safe(detail)}"


@cli.command()
def logs(
    config_path: Path | None = ConfigOption,
    follow: bool = typer.Option(False, "--follow", "-f", help="Keep printing new entries"),
    kind: list[str] = typer.Option([], "--kind", "-k", help="Only these kinds (repeatable), e.g. request, node_join"),
    which: str = typer.Option("all", "--file", help="events, requests or all"),
    limit: int = typer.Option(50, "--limit", "-n", help="How many recent entries to show"),
    as_json: bool = typer.Option(False, "--json", help="Print raw JSON lines"),
) -> None:
    """Show the event and request logs (metadata only, never prompts or responses)."""
    from .eventlog import follow_entries, read_entries

    config = load(config_path)
    if which not in ("events", "requests", "all"):
        typer.secho("--file must be events, requests or all", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)
    files = ["events", "requests"] if which == "all" else [which]
    kinds = kind or None

    def emit(entry: dict[str, Any]) -> None:
        typer.echo(json.dumps(entry) if as_json else _format_entry(entry))

    for entry in read_entries(config.logs_dir, files, kinds, limit):
        emit(entry)
    if follow:
        try:
            for entry in follow_entries(config.logs_dir, files, kinds):
                emit(entry)
        except KeyboardInterrupt:
            pass


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
