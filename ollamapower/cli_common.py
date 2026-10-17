"""Helpers shared by the hub and agent CLI commands."""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import logging
import os
import signal
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import typer

from .config import Config, ConfigError, load_config

ConfigOption = typer.Option(None, "--config", "-c", help="Path to config.toml", exists=True, dir_okay=False)
WATCH_INTERVAL = 2.0
Lines = list[tuple[str, Any]]


# Characters the CLI prints (status icons, arrows, ellipses, warnings).
_CLI_CHARS = "…→↓↑✓⚠●·🟢"


def tolerant_stdio() -> None:
    """Replace, instead of crash on, characters stdout/stderr can't encode.

    Redirected output on Windows uses the ANSI code page (cp1252 and the like),
    which has no "→" or "✓": the first one printed raised ``UnicodeEncodeError``
    and the command died half-way. Streams that can encode them (UTF-8) are
    left alone.
    """
    for stream in (sys.stdout, sys.stderr):
        encoding = getattr(stream, "encoding", None)
        try:
            _CLI_CHARS.encode(encoding or "ascii")
            continue
        except (UnicodeEncodeError, LookupError):
            pass
        with contextlib.suppress(AttributeError, ValueError, OSError):
            stream.reconfigure(errors="replace")  # type: ignore[union-attr]


@contextlib.contextmanager
def stop_like_ctrl_c() -> Iterator[None]:
    """Make SIGTERM (and, on Windows, Ctrl+Break or closing the console window) stop like Ctrl+C.

    Their default action ends the process at once, skipping every ``finally``: an
    agent stopped by ``kill``, systemd or a closed window left its ``rpc-server``
    running (holding VRAM) and its admin files behind. Inside ``asyncio.run`` the
    SIGINT handler cancels the main task, so cleanup runs and ``KeyboardInterrupt``
    comes out at the end. Uses ``signal.signal``, which works with Windows'
    Proactor event loop (``loop.add_signal_handler`` does not).
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def forward(signum: int, frame: Any) -> None:
        handler = signal.getsignal(signal.SIGINT)
        if callable(handler):
            handler(signal.SIGINT, frame)
        else:
            raise KeyboardInterrupt

    signals = [signal.SIGTERM]
    if hasattr(signal, "SIGBREAK"):
        signals.append(signal.SIGBREAK)
    original = {sig: signal.signal(sig, forward) for sig in signals}
    try:
        yield
    finally:
        for sig, handler in original.items():
            signal.signal(sig, handler)


def load(config_path: Path | None, config_dir: Path | None = None) -> Config:
    """The hub config; ``config_dir`` stands in for ``OLLAMAPOWER_CONFIG_DIR`` (its config.toml, .env, ...)."""
    env = None if config_dir is None else {**os.environ, "OLLAMAPOWER_CONFIG_DIR": str(config_dir)}
    try:
        return load_config(config_path, env=env)
    except ConfigError as exc:
        typer.secho(f"Config error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from None


def call(config_dir: Path, cmd: str, *, name: str = "run", timeout: float = 10.0, **args: Any) -> Any:
    """Run an admin command on the running hub (``name="run"``) or agent (``"agent-run"``)."""
    from .admin.ipc import AdminError, ControlClient, NotRunning

    what = "hub" if name == "run" else "agent"
    try:
        with ControlClient(config_dir, timeout=timeout, name=name) as client:
            return client.call(cmd, **args)
    except NotRunning as exc:
        message = str(exc).replace("the OllamaPower hub", f"the OllamaPower {what}")
        typer.secho(f"{message}. Start it with `ollamapower {what}`.", fg=typer.colors.RED, err=True)
        raise typer.Exit(3) from None
    except AdminError as exc:
        message = str(exc).replace("the OllamaPower hub", f"the OllamaPower {what}")
        typer.secho(f"Error: {message}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from None


def ts(value: float | None) -> str:
    if value is None:
        return "-"
    return dt.datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def duration(secs: float) -> str:
    secs = int(secs)
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


def gb(n: float | None) -> str:
    return "-" if n is None else f"{float(n) / 1024**3:.1f}"


def running_as_admin() -> bool:
    if hasattr(os, "geteuid"):
        return os.geteuid() == 0
    try:
        import ctypes

        return bool(ctypes.windll.shell32.IsUserAnAdmin())  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return False


def refuse_root(allow_root: bool, what: str) -> None:
    if running_as_admin() and not allow_root:
        typer.secho(
            f"Refusing to run the {what} as root/Administrator: run OllamaPower as a normal user "
            "(or pass --allow-root if you really must, e.g. inside a container).",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(2)


class BufferHandler(logging.Handler):
    """Keeps log lines in memory while a TUI owns the terminal (writing would corrupt it)."""

    def __init__(self, lines: Any) -> None:
        super().__init__()
        self.lines = lines

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.lines.append(self.format(record))
        except Exception:  # noqa: BLE001 - logging must never raise
            self.handleError(record)


def configure_logging(level: str, buffer: Any = None) -> None:
    """Headless: plain lines on stderr. With a TUI: into ``buffer`` instead."""
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    if buffer is None:
        logging.basicConfig(level=level.upper(), format=fmt, force=True)
    else:
        handler = BufferHandler(buffer)
        handler.setFormatter(logging.Formatter(fmt, "%H:%M:%S"))
        logging.basicConfig(level=level.upper(), handlers=[handler], force=True)
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def show(fetch: Callable[[], Any], render: Callable[[Any], Lines], as_json: bool, watch: bool) -> None:
    """Print once, or (``--watch``) refresh in place until Ctrl+C."""
    try:
        while True:
            data = fetch()
            if watch:
                typer.clear()
            if as_json:
                typer.echo(json.dumps(data, indent=2))
            else:
                for text, color in render(data):
                    typer.secho(text, fg=color)
            if not watch:
                return
            time.sleep(WATCH_INTERVAL)
    except KeyboardInterrupt:
        pass


def control_text(c: dict[str, Any]) -> str:
    """The control plane's state, and why it is off: turned off on purpose, or on but without a usable token."""
    if c["enabled"]:
        return f"port {c['port']}"
    if not c.get("configured", True):
        return "off (control.enabled = false in config.toml; helpers can't join)"
    return "not running: no usable cluster token, see the warnings (helpers can't join)"


def tunnel_test_lines(r: dict[str, Any]) -> list[str]:
    lines = []
    for mode in ("secure", "direct"):
        m = r.get(mode)
        if isinstance(m, dict):
            lines.append(
                f"{mode:7} RTT {m.get('rtt_ms')} ms (p95 {m.get('rtt_p95_ms')}), {m.get('mbps')} Mbps, "
                f"connect {m.get('connect_ms')} ms"
            )
    if r.get("overhead_pct") is not None:
        lines.append(f"Encryption costs {r['overhead_pct']}% throughput and {r['rtt_overhead_ms']} ms per round trip.")
    return lines


def apply_setup_plan(
    plan: Any, *, yes: bool, print_only: bool, adding: bool, record_dir: Path, record_name: str
) -> bool:
    """Show ``plan``, ask, run it with sudo and record what changed (see ``ollamapower.setup``).

    Returns False if there was nothing to do or the user said no.
    """
    from . import setup as s

    if running_as_admin():
        typer.secho(
            "Run this as your normal user, not with sudo: it asks for your password itself, only for the "
            "steps that need it.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(2)
    for note in plan.notes:
        typer.echo(note)
    if not plan.steps:
        typer.echo("Nothing to change.")
        return False
    if plan.summary:
        typer.echo(plan.summary + "\n")
    typer.echo("These commands will run (with sudo; the read-only checks change nothing):\n")
    for step in plan.steps:
        typer.echo(f"  # {step.why}")
        typer.echo("  " + step.shown().replace("\n", "\n  "))
    typer.echo()
    if print_only:
        typer.echo("(--print: nothing was run)")
        return False
    if not yes and not typer.confirm("Run them now?", default=False):
        return False
    changed: list[tuple[str, int]] = []
    restored = False
    hub = record_name == s.RECORD
    try:
        restored = s.execute(plan, s.sudo_runner, adding, changed)
    finally:
        record = s.read_record(record_dir, record_name)
        rules = {tuple(r) for r in record.get("rules", []) or []}  # type: ignore[union-attr]
        rules = rules | set(changed) if adding else rules - set(changed)
        record["rules"] = sorted(rules)
        if changed and adding:
            record["firewall"] = plan.firewall
        if hub:
            # Ollama's port counts as given back only once Ollama restarted without the drop-in: if a step
            # failed half-way, the record still says so and `uninstall` tries again.
            record["dropin"] = Path(s.DROPIN).exists() or (not adding and bool(record.get("dropin")) and not restored)
        try:
            s.write_record(record_dir, record, record_name)
        except OSError as exc:  # never hide the error that brought us here
            typer.secho(f"Couldn't save {record_dir / record_name}: {exc}", fg=typer.colors.RED, err=True)
    return True
