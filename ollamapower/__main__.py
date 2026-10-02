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
    tunnel_test_lines,
)
from .textsafe import safe

cli = typer.Typer(add_completion=False, no_args_is_help=True, help="OllamaPower - cluster-boosted Ollama")
nodes_cli = typer.Typer(no_args_is_help=True, help="Helpers known to the hub")
models_cli = typer.Typer(no_args_is_help=True, help="Models and their plans")
token_cli = typer.Typer(no_args_is_help=True, help="The cluster token helpers need")
llamacpp_cli = typer.Typer(no_args_is_help=True, help="The pinned llama.cpp build split mode uses")
cli.add_typer(agent_cli, name="agent")
cli.add_typer(nodes_cli, name="nodes")
cli.add_typer(models_cli, name="models")
cli.add_typer(token_cli, name="token")
cli.add_typer(llamacpp_cli, name="llamacpp")

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
            control_sock = _control_socket(config.control.host, config.control.port)
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


def _control_socket(host: str, port: int) -> Any:
    """Listen for helpers. "All interfaces" means IPv4 *and* IPv6 where the OS allows one dual-stack socket."""
    import socket

    if host in ("0.0.0.0", "::", "") and socket.has_dualstack_ipv6():
        return socket.create_server(("::", port), family=socket.AF_INET6, dualstack_ipv6=True)
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    return socket.create_server((host, port), family=family)


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
    session = (s.get("split") or {}).get("session")
    if session:
        plan = session.get("plan") or {}
        where = ", ".join(f"{d['label']} {d['layers']}" for d in plan.get("devices", []))
        if plan.get("cpu_layers"):
            where += f", hub CPU {plan['cpu_layers']}"
        rows.append(("Split", f"{session['model']} {session['state']} (layers: {where or '-'})"))
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
    if d.get("tunnel_test"):
        for line in tunnel_test_lines(d["tunnel_test"]):
            lines.append((f"{'Tunnel:':12} {safe(line)}", None))
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


@nodes_cli.command("tunnel-test")
def nodes_tunnel_test(
    node: str, config_path: Path | None = ConfigOption, as_json: bool = typer.Option(False, "--json")
) -> None:
    """Measure the split-mode tunnel to a helper: secure (TLS), and direct if its owner allows it."""
    r = call(load(config_path).config_dir, "nodes.tunnel_test", ref=node, timeout=180)
    if as_json:
        typer.echo(json.dumps(r, indent=2))
        return
    for line in tunnel_test_lines(r):
        typer.echo(line)
    if r.get("direct") is None:
        typer.echo("(direct mode not measured: the helper's owner hasn't allowed it)")


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
    """Choose how split-mode traffic to a helper travels: secure (TLS, default) or direct."""
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


def _tps(value: Any) -> str:
    return "-" if value is None else f"{value:.1f}"


def render_models(models: list[dict[str, Any]]) -> Lines:
    if not models:
        return [("No models (is Ollama reachable?)", None)]
    lines: Lines = [
        (
            f"{'MODEL':34} {'SIZE':>10} {'QUANT':8} {'LOADED':7} {'FITS':5} {'ENGINE':13} {'MODE':6} "
            f"{'LOCAL':>6} {'SPLIT':>6} WHY",
            None,
        )
    ]
    for m in models:
        fits = "?" if m["fits"] is None else ("yes" if m["fits"] else "no")
        if not m.get("fits_estimated") and m["fits"] is not None:
            fits += "~"
        loaded = f"{m['vram_percent']}%" if m["loaded"] else "-"
        engine = m["plan"] + (" ●" if m.get("running_split") else "")
        color = typer.colors.YELLOW if m.get("plan_degraded") or m.get("spilled") else None
        lines.append(
            (
                f"{safe(m['name'])[:34]:34} {_human(m['size']):>10} {safe(m['quantization']):8} {loaded:7} {fits:5} "
                f"{engine:13} {m['plan_mode']:6} {_tps(m['plan_local_tps']):>6} {_tps(m['plan_split_tps']):>6} "
                f"{safe(m['plan_why'])[:70]}",
                color,
            )
        )
    lines.append(("LOCAL/SPLIT: generation tok/s from the last benchmark. FITS ~ = rough (size + 20 %).", None))
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


@models_cli.command("show")
def models_show(
    model: str, config_path: Path | None = ConfigOption, as_json: bool = typer.Option(False, "--json")
) -> None:
    """A model's plan, why it runs where it runs, its memory estimate and benchmark history."""
    d = call(load(config_path).config_dir, "models.show", model=model, timeout=60)
    if as_json:
        typer.echo(json.dumps(d, indent=2))
        return
    f = d["facts"]
    typer.echo(f"{safe(d['model'])}: served by {d['engine']} ({d['mode']}) - {safe(d['why'])}")
    if f.get("splittable"):
        verdict = "fits" if f["fits"] else "does NOT fit" if f["fits"] is False else "?"
        typer.echo(
            f"  Memory:     {_human(f['needed'])} needed at context {d['ctx']} (weights + KV cache + margin); "
            f"hub VRAM {_human(f['hub_vram'])}: {verdict}" + (" · Ollama spilled it to the CPU" if f["spilled"] else "")
        )
        typer.echo(f"  Model:      {f['layers']} layers, {safe(f['architecture'])}")
    else:
        typer.echo(f"  Split mode: not possible ({safe(f.get('reason', ''))})")
    typer.echo(f"  Helpers now: {', '.join(safe(h) for h in d['helpers_now']) or 'none eligible'}")
    if d["record"]["exclude"]:
        typer.echo(f"  Excluded:   {', '.join(x[:8] for x in d['record']['exclude'])}")
    if d["benchmarks"]:
        typer.echo(
            f"\n  {'WHEN':19} {'ENGINE':6} {'GEN TOK/S':>9} {'PROMPT':>8} {'TTFT':>8} {'LOAD':>7}  HELPERS / ERROR"
        )
        for b in d["benchmarks"]:
            helpers = ", ".join(f"{h['name']} ({h['tunnel']})" for h in b.get("helpers") or [])
            detail = helpers if b.get("ok") else f"failed: {b.get('error', '')}"
            typer.echo(
                f"  {ts(b['ts'])} {b['engine']:6} {_tps(b.get('gen_tps')):>9} {_tps(b.get('prompt_tps')):>8} "
                f"{(str(b.get('ttft_ms')) + ' ms') if b.get('ttft_ms') is not None else '-':>8} "
                f"{(str(b.get('load_s')) + ' s') if b.get('load_s') is not None else '-':>7}  {safe(detail)}"
            )


@models_cli.command("benchmark")
def models_benchmark(
    model: str,
    ctx: int | None = typer.Option(None, "--ctx", help="Context size (default: the plan's, else split.default_ctx)"),
    config_path: Path | None = ConfigOption,
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Benchmark a model on plain Ollama and with helpers, then pick the faster (never slower)."""
    config = load(config_path)
    typer.echo(f"Benchmarking {model}: Ollama first, then split mode. Loading can take minutes the first time…")
    d = call(config.config_dir, "models.benchmark", model=model, ctx=ctx, timeout=config.split.load_timeout * 4 + 600)
    if as_json:
        typer.echo(json.dumps(d, indent=2))
        return
    for r in [d["local"], *d["splits"]]:
        name = (
            "Ollama" if r["engine"] == "local" else "split (" + ", ".join(h["name"] for h in r.get("helpers", [])) + ")"
        )
        if r["ok"]:
            typer.echo(
                f"  {safe(name):40} {r['gen_tps']:7.1f} tok/s · prompt {_tps(r.get('prompt_tps'))} tok/s · "
                f"first token {r['ttft_ms']} ms · load {r['load_s']} s"
            )
        else:
            typer.secho(f"  {safe(name):40} failed: {safe(r['error'])}", fg=typer.colors.YELLOW)
    typer.echo(f"Result: {d.get('choice') or '-'} - {safe(d.get('reason', ''))}")


@models_cli.command("pin")
def models_pin(
    model: str,
    engine: str = typer.Argument(..., help="split or local"),
    ctx: int | None = typer.Option(None, "--ctx", help="Context size for split mode"),
    config_path: Path | None = ConfigOption,
) -> None:
    """Always use one engine for a model (overrides the benchmark; `unpin` goes back to automatic)."""
    d = call(load(config_path).config_dir, "models.pin", model=model, engine=engine, ctx=ctx, timeout=60)
    typer.echo(f"{safe(d['model'])}: pinned to {engine} - now served by {d['engine']}.")


@models_cli.command("unpin")
def models_unpin(model: str, config_path: Path | None = ConfigOption) -> None:
    """Let the benchmark decide again (automatic mode)."""
    d = call(load(config_path).config_dir, "models.unpin", model=model)
    typer.echo(f"{safe(d['model'])}: automatic - served by {d['engine']} ({safe(d['why'])}).")


@models_cli.command("force-local")
def models_force_local(model: str, config_path: Path | None = ConfigOption) -> None:
    """Always serve a model with plain Ollama, whatever the benchmark says."""
    d = call(load(config_path).config_dir, "models.force_local", model=model)
    typer.echo(f"{safe(d['model'])}: forced to local Ollama.")


@models_cli.command("exclude")
def models_exclude(
    model: str,
    helper: str,
    undo: bool = typer.Option(False, "--undo", help="Allow the helper again"),
    config_path: Path | None = ConfigOption,
) -> None:
    """Don't use a helper for a model (--undo allows it again)."""
    d = call(load(config_path).config_dir, "models.exclude", model=model, node=helper, exclude=not undo)
    typer.echo(f"{safe(d['model'])}: {len(d['exclude'])} helper(s) excluded.")


@models_cli.command("plan")
def models_plan(
    model: str = typer.Argument(..., help="An installed Ollama model, e.g. llama3.1:70b"),
    split: bool = typer.Option(False, "--split", help="Run it in split mode across helper GPUs"),
    ctx: int | None = typer.Option(None, "--ctx", help="Context size for split mode (default: split.default_ctx)"),
    config_path: Path | None = ConfigOption,
    as_json: bool = typer.Option(False, "--json", help="Print JSON"),
) -> None:
    """Pin a model to split mode (same as `models pin <model> split`, with a preview)."""
    if not split:
        typer.secho("Say how: --split (the only plan so far; `models unplan` goes back to Ollama).", fg="red", err=True)
        raise typer.Exit(2)
    config = load(config_path)
    d = call(config.config_dir, "models.plan", model=model, ctx=ctx)
    if as_json:
        typer.echo(json.dumps(d, indent=2))
        return
    typer.echo(f"{safe(d['model'])} will run in split mode (context {d['ctx']}).")
    typer.echo(
        f"  Weights:    {_human(d['size'])}, {d['layers']} layers ({safe(d['architecture'])}), "
        f"largest layer {_human(d['layer_bytes_max'])} + {_human(d['kv_per_layer'])} KV cache"
    )
    for h in d["helpers"]:
        typer.echo(f"  Helper:     {safe(h['name'])} ({_human(h['lendable'])} lendable now)")
    for reason in d["unavailable"]:
        typer.secho(f"  Not used:   {safe(reason)}", fg=typer.colors.YELLOW)
    for note in d["notes"]:
        typer.secho(f"  Note:       {safe(note)}", fg=typer.colors.YELLOW)
    typer.echo("The layers are placed when the model is first used; `ollamapower status` shows where they went.")


@models_cli.command("unplan")
def models_unplan(model: str, config_path: Path | None = ConfigOption) -> None:
    """Forget a model's plan and benchmark decision (automatic mode, nothing decided yet)."""
    config = load(config_path)
    d = call(config.config_dir, "models.unplan", model=model)
    typer.echo(f"Removed the plan for {safe(d['model'])}{' and stopped its split session' if d['stopped'] else ''}.")


# ----- llama.cpp -----------------------------------------------------------------------


def _agent_or_hub_llamacpp(config_path: Path | None) -> tuple[Path, str, str]:
    """(config dir, backend, custom path): a helper's agent.toml wins on a machine without a hub config."""
    from .agent.config import load_agent_config

    config = load(config_path)
    hub_file = config.config_dir / "config.toml"
    if config_path is None and not hub_file.exists() and (config.config_dir / "agent.toml").exists():
        agent = load_agent_config(config.config_dir).llamacpp
        return config.config_dir, agent.backend, agent.path
    return config.config_dir, config.llamacpp.backend, config.llamacpp.path


@llamacpp_cli.command("install")
def llamacpp_install(
    backend: str | None = typer.Option(None, "--backend", help="auto, cpu, metal, vulkan, cuda12, cuda13 or rocm"),
    config_path: Path | None = ConfigOption,
) -> None:
    """Download and verify the pinned llama.cpp build for this machine (hub or helper)."""
    from .llamacpp.install import Pins
    from .llamacpp.manager import LlamaCppUnavailable, ensure
    from .llamacpp.variants import BACKENDS
    from .telemetry import Telemetry

    base, configured, custom = _agent_or_hub_llamacpp(config_path)
    backend = backend or configured
    if backend not in BACKENDS:
        typer.secho(f"--backend must be one of {', '.join(BACKENDS)}", fg="red", err=True)
        raise typer.Exit(2)
    vendors = [g.get("vendor", "") for g in Telemetry().static().get("gpus", []) if g.get("vendor")]
    pins = Pins.load()
    last = [0]

    def progress(name: str, done: int, total: int) -> None:
        pct = done * 100 // total if total else 100
        if pct >= last[0] + 5 or done == total:
            last[0] = pct
            typer.echo(f"\r  {name}: {pct:3d}% of {_human(total)}", nl=done == total)

    typer.echo(f"llama.cpp {pins.tag} ({backend}) -> {base / 'llamacpp'}")
    try:
        found = ensure(base, backend, custom, vendors, progress, pins)
    except LlamaCppUnavailable as exc:
        typer.secho(str(exc), fg="red", err=True)
        raise typer.Exit(1) from None
    typer.echo(f"Ready: build {found.build} in {found.root}")
    for dev in found.devices:
        typer.echo(f"  {safe(dev.name):10} {safe(dev.description)} ({dev.free_mib} of {dev.total_mib} MiB free)")
    if not found.devices:
        typer.echo("  No GPU visible to llama.cpp: only CPU can be used.")


@llamacpp_cli.command("status")
def llamacpp_status(config_path: Path | None = ConfigOption, as_json: bool = typer.Option(False, "--json")) -> None:
    """Show the installed llama.cpp build and the devices it sees (installs nothing)."""
    from .llamacpp.manager import LlamaCppUnavailable, ensure
    from .telemetry import Telemetry

    base, backend, custom = _agent_or_hub_llamacpp(config_path)
    vendors = [g.get("vendor", "") for g in Telemetry().static().get("gpus", []) if g.get("vendor")]
    try:
        found = ensure(base, backend, custom, vendors, download=False)
    except LlamaCppUnavailable as exc:
        if as_json:
            typer.echo(json.dumps({"installed": False, "error": str(exc)}))
        else:
            typer.secho(str(exc), fg=typer.colors.YELLOW)
        raise typer.Exit(1) from None
    if as_json:
        typer.echo(json.dumps({"installed": True, "root": str(found.root), **found.summary()}))
        return
    typer.echo(f"llama.cpp build {found.build} in {found.root}")
    for dev in found.devices:
        typer.echo(f"  {safe(dev.name):10} {safe(dev.description)} ({dev.free_mib} of {dev.total_mib} MiB free)")


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
