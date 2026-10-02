"""Command-line entry point.

``hub`` runs the hub. ``status`` talks to the running hub over the local-only
admin channel (see ``admin/ipc.py``); ``init``, ``doctor`` and ``logs`` work
on the files directly, so they also work while the hub is stopped.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import time
from importlib import resources
from pathlib import Path
from typing import Any

import typer

from . import __version__
from .config import Config, ConfigError, load_config
from .textsafe import safe

cli = typer.Typer(add_completion=False, no_args_is_help=True, help="OllamaPower - cluster-boosted Ollama")

ConfigOption = typer.Option(None, "--config", "-c", help="Path to config.toml", exists=True, dir_okay=False)
WATCH_INTERVAL = 2.0


def _load(config_path: Path | None) -> Config:
    try:
        return load_config(config_path)
    except ConfigError as exc:
        typer.secho(f"Config error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from None


def _call(config: Config, cmd: str, **args: Any) -> Any:
    from .admin.ipc import AdminError, ControlClient, NotRunning

    try:
        with ControlClient(config.config_dir) as client:
            return client.call(cmd, **args)
    except NotRunning as exc:
        typer.secho(f"{exc}. Start it with `ollamapower hub`.", fg=typer.colors.RED, err=True)
        raise typer.Exit(3) from None
    except AdminError as exc:
        typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from None


def _ts(value: float | None) -> str:
    if value is None:
        return "-"
    return dt.datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def _duration(secs: float) -> str:
    secs = int(secs)
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


@cli.callback()
def _root() -> None:
    """OllamaPower - cluster-boosted Ollama."""


@cli.command()
def version() -> None:
    """Show the OllamaPower version."""
    typer.echo(f"ollamapower {__version__}")


# ----- init ----------------------------------------------------------------------------


@cli.command()
def init(
    config_path: Path | None = ConfigOption,
    rotate_token: bool = typer.Option(
        False, "--rotate-token", help="Replace an existing token (helpers need the new one)"
    ),
) -> None:
    """Create the config directory, a cluster token in .env, config.toml and an empty state.json."""
    from .core.auth import TOKEN_ENV, new_token
    from .files import load_token, private_dir, write_private, write_token
    from .state import init_state_file

    config = _load(config_path)
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
    typer.echo("\nNext: `ollamapower doctor` checks Ollama and walks you through moving it to port 11435.")


# ----- hub -----------------------------------------------------------------------------


def configure_logging(level: str) -> None:
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=level.upper(), format=fmt, force=True)
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@cli.command()
def hub(
    config_path: Path | None = ConfigOption,
    no_tui: bool = typer.Option(False, "--no-tui", help="Run headless with plain logs"),
    allow_root: bool = typer.Option(False, "--allow-root", help="Run even as root/Administrator (not recommended)"),
    i_know: bool = typer.Option(
        False, "--i-know-what-im-doing", help="Allow hub.host to be a non-loopback address", hidden=True
    ),
) -> None:
    """Run the hub: the Ollama-compatible front door and the request queue."""
    from .admin.ipc import start_control_server
    from .admin.service import Admin
    from .config import is_loopback_host
    from .core.auth import TokenError, validate_token
    from .eventlog import EventLog, NullEventLog
    from .files import load_token, private_dir
    from .hub.frontdoor.app import create_app
    from .hub.frontdoor.server import build_server
    from .hub.runtime import Hub
    from .state import StateError, StateStore

    config = _load(config_path)
    if _running_as_admin() and not allow_root:
        typer.secho(
            "Refusing to run as root/Administrator: run OllamaPower as a normal user "
            "(or pass --allow-root if you really must, e.g. inside a container).",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(2)
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

    configure_logging(config.logging.level)
    log = logging.getLogger("ollamapower")
    if not no_tui:
        log.info("the terminal UI arrives in Phase 2; running headless")

    try:
        private_dir(config.config_dir)
        state = StateStore(config.state_path)
        state.load()
    except (StateError, OSError) as exc:
        typer.secho(f"State error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from None
    try:
        events: EventLog = EventLog(config.logs_dir, config.logging.file_max_bytes, config.logging.file_backups)
    except OSError as exc:
        log.warning("cannot write log files in %s (%s); request/event logs disabled", config.logs_dir, exc)
        events = NullEventLog()

    the_hub = Hub(config, state, events)
    try:
        validate_token(load_token(config.env_path))
    except TokenError as exc:
        the_hub.extra_warnings.append(f"{exc} (only needed once helpers join).")
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

    async def main() -> None:
        try:
            control = await start_control_server(config.config_dir, Admin(the_hub).dispatch)
        except RuntimeError as exc:
            typer.secho(str(exc), fg=typer.colors.RED, err=True)
            raise typer.Exit(1) from None
        events.event("hub_start", version=__version__, frontdoor=config.frontdoor_url, ollama=config.ollama.url)
        log.info(
            "OllamaPower %s front door on %s -> Ollama at %s", __version__, config.frontdoor_url, config.ollama.url
        )
        for warning in the_hub.warnings():
            log.warning("%s", warning)
        try:
            await server.serve(sockets=[sock])
        finally:
            await control.close()
            try:
                await state.flush()
            except OSError as exc:
                log.error("could not save state: %s", exc)
            events.event("hub_stop")
            events.close()

    asyncio.run(main())


def _running_as_admin() -> bool:
    import os

    if hasattr(os, "geteuid"):
        return os.geteuid() == 0
    try:
        import ctypes

        return bool(ctypes.windll.shell32.IsUserAnAdmin())  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return False


# ----- status --------------------------------------------------------------------------


def _human(n: int) -> str:
    from .ollama.client import human_bytes

    return human_bytes(n)


def render_status(s: dict[str, Any]) -> list[tuple[str, Any]]:
    """The status dict as (text, color) lines; shared by the plain and --watch views."""
    o, q, r = s["ollama"], s["queue"], s["requests"]
    lines: list[tuple[str, Any]] = []
    waiting = ", ".join(f"{n} {p}" for p, n in q["waiting_by_priority"].items() if n) or "none"
    rows = [
        ("Front door", s["frontdoor"]["url"]),
        ("Ollama", f"{o['version']} at {o['url']}" if o["reachable"] else f"UNREACHABLE at {o['url']} ({o['error']})"),
        ("Uptime", _duration(s["uptime_s"])),
        ("Queue", f"{q['active']}/{q['capacity']} running, waiting: {waiting}"),
        ("Requests", f"{r['last_minute']}/min, {r['total']} total, {r['failed']} failed"),
        ("Helpers", ", ".join(n.get("name", "?") for n in s["nodes"]) or "none"),
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
    """Show the hub: front door, Ollama, queue, loaded models, active requests, warnings."""
    config = _load(config_path)
    try:
        while True:
            s = _call(config, "status")
            if watch:
                typer.clear()
            if as_json:
                typer.echo(json.dumps(s, indent=2))
            else:
                for text, color in render_status(s):
                    typer.secho(text, fg=color)
            if not watch:
                return
            time.sleep(WATCH_INTERVAL)
    except KeyboardInterrupt:
        pass


# ----- doctor --------------------------------------------------------------------------


@cli.command()
def doctor(config_path: Path | None = ConfigOption) -> None:
    """Check the setup (Ollama's port move, exposure, token, file permissions)."""
    from . import doctor as doc

    config = _load(config_path)
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
    ts = e.get("ts")
    when = _ts(ts) if isinstance(ts, (int, float)) else "-"
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
    kind: list[str] = typer.Option([], "--kind", "-k", help="Only these kinds (repeatable), e.g. request, hub_start"),
    which: str = typer.Option("all", "--file", help="events, requests or all"),
    limit: int = typer.Option(50, "--limit", "-n", help="How many recent entries to show"),
    as_json: bool = typer.Option(False, "--json", help="Print raw JSON lines"),
) -> None:
    """Show the event and request logs (metadata only, never prompts or responses)."""
    from .eventlog import follow_entries, read_entries

    config = _load(config_path)
    if which not in ("events", "requests", "all"):
        typer.secho("--file must be events, requests or all", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)
    files = ["events", "requests"] if which == "all" else [which]
    kinds = kind or None

    def show(entry: dict[str, Any]) -> None:
        typer.echo(json.dumps(entry) if as_json else _format_entry(entry))

    for entry in read_entries(config.logs_dir, files, kinds, limit):
        show(entry)
    if follow:
        try:
            for entry in follow_entries(config.logs_dir, files, kinds):
                show(entry)
        except KeyboardInterrupt:
            pass


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
