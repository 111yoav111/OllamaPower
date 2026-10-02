"""The hub TUI: Overview, Nodes, Node detail, Models & plans, Log.

It runs in the hub's process and on its event loop, reads ``Hub.get_snapshot()``
and acts only through ``Admin`` - the same code path as the CLI. It is never
reachable over the network. Visible data is redrawn once a second; tables are
updated in place so the cursor stays where the owner put it.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import Any

from rich.console import Group
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import DataTable, Footer, Header, Input, Sparkline, Static, TabbedContent, TabPane

from ...admin.ipc import AdminError
from ...admin.service import Admin
from ...agent.pinning import format_fingerprint
from ...cli_common import tunnel_test_lines
from ...textsafe import safe
from ...tui.widgets import STATUS_DOT, ConfirmScreen, ago, bar, duration, gib, human_bytes, t, ts

log = logging.getLogger("ollamapower.tui")

REFRESH_INTERVAL = 1.0
ALERT_EVENTS = {"auth_failed", "ip_banned", "node_incompatible", "node_kicked"}
WARN_EVENTS = {"node_leave", "lend_state", "ollama_down", "node_paused", "split_failed"}
TOAST_EVENTS = {"node_join", "node_leave", "ip_banned", "node_incompatible"}


def _link(link: dict[str, Any]) -> Text:
    kind = link.get("kind") or "?"
    parts = []
    if link.get("rtt_ms") is not None:
        parts.append(f"{link['rtt_ms']:.1f} ms")
    if link.get("down_mbps") is not None:
        parts.append(f"{link['down_mbps']:.0f} Mbps")
    elif link.get("speed_mbps"):
        parts.append(f"{link['speed_mbps']} Mbps link")
    parts.append(kind)
    return t(" · ".join(parts), "bold yellow" if kind == "wifi" else "")


def _rules_text(r: dict[str, Any]) -> str:
    caps = [f"{r['max_vram_gb']:g} GB" for _ in [0] if r.get("max_vram_gb")]
    caps += [f"{r['max_vram_percent']}%" for _ in [0] if r.get("max_vram_percent")]
    allowed = [name for key, name in (("allow_cpu", "CPU"), ("allow_direct", "direct mode")) if r.get(key)]
    gpus = ", ".join(str(g) for g in r.get("gpus") or []) or "all"
    return f"VRAM cap {' and '.join(caps) or 'none'} · GPUs {gpus} · also allows {', '.join(allowed) or 'nothing else'}"


def _gpu_names(gpus: list[dict[str, Any]]) -> str:
    if not gpus:
        return "CPU only"
    names = [str(g.get("name") or "GPU") for g in gpus]
    return names[0] if len(names) == 1 else f"{len(names)}× {names[0]}"


class NodeTable(DataTable):
    BINDINGS = [Binding("enter", "app.node_detail", "Details")]


# Actions only offered on some tabs (hidden from the footer elsewhere).
NODE_ACTIONS = {"node_pause", "node_kick", "node_net_test", "node_tunnel", "node_tunnel_test", "node_forget"}
MODEL_ACTIONS = {"phase5"}


class HubTUI(App[None]):
    TITLE = "OllamaPower"
    CSS_PATH = "../../tui/power.tcss"
    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [
        Binding("1", "tab('overview')", "Overview", show=False),
        Binding("2", "tab('nodes')", "Nodes", show=False),
        Binding("3", "tab('detail')", "Detail", show=False),
        Binding("4", "tab('models')", "Models", show=False),
        Binding("5", "tab('log')", "Log", show=False),
        Binding("p", "node_pause", "Pause/resume"),
        Binding("k", "node_kick", "Kick"),
        Binding("n", "node_net_test", "Net test"),
        Binding("t", "node_tunnel", "Tunnel"),
        Binding("m", "node_tunnel_test", "Tunnel test"),
        Binding("x", "node_forget", "Forget"),
        Binding("b", "phase5('benchmark')", "Benchmark"),
        Binding("l", "phase5('pin a plan')", "Pin plan"),
        Binding("f", "phase5('force local')", "Force local"),
        Binding("/", "filter", "Filter"),
        Binding("q", "quit_hub", "Quit"),
        Binding("ctrl+q", "quit_hub", "Quit", show=False, priority=True),
    ]

    def __init__(
        self,
        admin: Admin,
        *,
        services: asyncio.Future[Any] | None = None,
        log_lines: deque[str] | None = None,
    ) -> None:
        super().__init__()
        self.admin = admin
        self.hub = admin.hub
        self.services = services
        self.log_lines = log_lines if log_lines is not None else deque(maxlen=200)
        self.selected_node: str | None = None
        self._filter = ""
        self._filter_log = ""
        self._seen_event = self.hub.event_seq
        self._log_rendered = -1
        self.quit_confirmed = False

    # ----- layout --------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with TabbedContent(initial="overview"):
            with TabPane("1 Overview", id="overview"):
                with VerticalScroll():
                    yield Static(id="warnings", classes="warnings")
                    yield Static(id="status", classes="panel")
                    yield Static(id="hub-gpu", classes="panel")
                    yield Static(id="models-live", classes="panel")
                    yield Static(id="streams", classes="panel")
                    yield Static(id="recent", classes="panel")
            with TabPane("2 Nodes", id="nodes"):
                yield Input(
                    placeholder="Filter by name, IP, status, GPU…  (Esc to clear)", id="node-filter", classes="filter"
                )
                yield NodeTable(id="node-table", cursor_type="row", zebra_stripes=True, classes="grow")
            with TabPane("3 Node detail", id="detail"):
                with VerticalScroll():
                    yield Static(id="detail-info", classes="panel")
                    yield Static("GPU utilisation", id="label-gpu", classes="spark-label")
                    yield Sparkline([], id="spark-gpu")
                    yield Static("VRAM used", id="label-vram", classes="spark-label")
                    yield Sparkline([], id="spark-vram")
                    yield Static("CPU", id="label-cpu", classes="spark-label")
                    yield Sparkline([], id="spark-cpu")
                    yield Static(id="detail-net", classes="panel")
            with TabPane("4 Models", id="models"):
                yield Static(
                    Text(
                        "Plan 'split' spreads a model over helper GPUs (`ollamapower models plan <model> --split`); "
                        "others run on local Ollama. 'Fits' is a rough estimate (file size + 20%).",
                        style="dim",
                    )
                )
                yield DataTable(id="model-table", cursor_type="row", zebra_stripes=True, classes="grow")
            with TabPane("5 Log", id="log"):
                yield Input(
                    placeholder="Filter events, e.g. node_join, auth_failed, 192.168.",
                    id="log-filter",
                    classes="filter",
                )
                yield DataTable(id="log-table", cursor_type="row", zebra_stripes=True, classes="grow")
        yield Footer()

    def on_mount(self) -> None:
        nodes = self.query_one("#node-table", DataTable)
        self._node_columns = nodes.add_columns(
            "", "Name", "IP", "Status", "GPU", "VRAM tot/free/lend GB", "GPU %", "CPU/RAM", "Link", "Role", "Tunnel"
        )
        self.query_one("#model-table", DataTable).add_columns(
            "Model", "Size", "Params", "Quant", "Loaded", "Fits locally", "Plan", "Last tok/s"
        )
        self.query_one("#log-table", DataTable).add_columns("Time", "Event", "Detail")
        self.set_interval(REFRESH_INTERVAL, self.refresh_visible)
        if self.services is not None:
            self.run_worker(self._run_services(), exclusive=True, group="services")
        self.refresh_visible()

    async def _run_services(self) -> None:
        assert self.services is not None
        try:
            await asyncio.shield(self.services)
        except asyncio.CancelledError:
            raise
        except BaseException:  # noqa: BLE001 - the hub stopped; nothing more to show
            pass
        self.exit()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action in NODE_ACTIONS or action in MODEL_ACTIONS:
            try:
                tab = self.query_one(TabbedContent).active
            except Exception:  # noqa: BLE001 - not mounted yet
                return False
            if action in NODE_ACTIONS:
                return tab in ("nodes", "detail")
            return tab == "models"
        return True

    # ----- refresh -------------------------------------------------------------------

    def _live(self) -> bool:
        return self.is_running and bool(self.screen_stack)

    def refresh_visible(self) -> None:
        if not self._live():
            return
        try:
            snap = self.hub.get_snapshot()
            tab = self.query_one(TabbedContent).active
            if tab == "overview":
                self._update_overview(snap)
            elif tab == "nodes":
                self._update_nodes(snap)
            elif tab == "detail":
                self._update_detail()
            elif tab == "models":
                self._update_models()
            elif tab == "log":
                self._update_log()
            self._toast_new_events()
        except Exception:  # noqa: BLE001 - a drawing bug must never take the hub down
            log.exception("TUI refresh failed")

    def _toast_new_events(self) -> None:
        new = [e for e in self.hub.recent_events if e["seq"] > self._seen_event][-5:]
        self._seen_event = self.hub.event_seq
        for e in new:
            if e["kind"] not in TOAST_EVENTS:
                continue
            who = e.get("name") or e.get("ip") or ""
            text = {
                "node_join": f"Helper joined: {who} ({e.get('ip', '')})",
                "node_leave": f"Helper left: {who} ({e.get('reason', '')})",
                "ip_banned": f"Banned {e.get('ip')} after repeated failed logins",
                "node_incompatible": f"Incompatible helper {who}: {e.get('reason', '')}",
            }[e["kind"]]
            severity = "information" if e["kind"] == "node_join" else "warning"
            self.say(text, title="Helpers", severity=severity, timeout=8)

    def _update_overview(self, s: dict[str, Any]) -> None:
        warn = self.query_one("#warnings", Static)
        warnings = s["warnings"]
        warn.update(Text("\n".join("⚠ " + safe(w) for w in warnings), style="bold red") if warnings else "")
        warn.display = bool(warnings)

        o, q, r, c = s["ollama"], s["queue"], s["requests"], s["control"]
        nodes = s["nodes"]
        lending = [n for n in nodes if n["status"] == "lending"]
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="bold")
        grid.add_column()
        waiting = ", ".join(f"{n} {p}" for p, n in q["waiting_by_priority"].items() if n) or "none"
        rows = [
            ("Front door", s["frontdoor"]["url"]),
            ("Ollama", f"{o['version']} at {o['url']}" if o["reachable"] else f"UNREACHABLE at {o['url']}"),
            ("Uptime", duration(s["uptime_s"])),
            ("Queue", f"{q['active']}/{q['capacity']} running · waiting: {waiting}"),
            ("Requests", f"{r['last_minute']}/min · {r['total']} total · {r['failed']} failed"),
            (
                "Helpers",
                f"{len(lending)} lending · {sum(1 for n in nodes if n['connected'])} connected · {len(nodes)} known · "
                f"{gib(sum(n['vram_lendable'] for n in lending))} GB VRAM lendable",
            ),
            (
                "Control plane",
                f"port {c['port']}  ·  key {format_fingerprint(c['fingerprint'])[:47]}…" if c["enabled"] else "off",
            ),
        ]
        session = (s.get("split") or {}).get("session")
        if session:
            plan = session.get("plan") or {}
            where = " · ".join(f"{d['label']} {d['layers']}" for d in plan.get("devices", []))
            if plan.get("cpu_layers"):
                where += f" · hub CPU {plan['cpu_layers']}"
            rows.append(("Split mode", f"{session['model']} {session['state']} · layers: {where or '-'}"))
        if c.get("bans"):
            rows.append(("Banned IPs", ", ".join(f"{ip} ({left}s)" for ip, left in c["bans"].items())))
        for label, value in rows:
            grid.add_row(label, t(value, "bold red" if str(value).startswith("UNREACHABLE") else ""))
        self.query_one("#status", Static).update(grid)

        hw = s.get("hub_hardware") or {}
        live = {g.get("index"): g for g in (hw.get("live") or {}).get("gpus", [])}
        lines: list[Any] = [Text("Hub hardware", style="bold")]
        for g in hw.get("gpus", []):
            lg = live.get(g.get("index"), {})
            total = lg.get("vram_total", g.get("vram_total", 0))
            used = lg.get("vram_used", 0)
            util = f" · GPU {lg['util']}%" if lg.get("util") is not None else ""
            lines.append(
                Text.assemble(
                    f"{safe(g.get('name', 'GPU'))[:28]:28} VRAM ",
                    bar(used, total),
                    f"  {gib(used)}/{gib(total)} GB{util}",
                )
            )
        if not hw.get("gpus"):
            lines.append(Text("No GPU detected on the hub (CPU only)", style="yellow"))
        cpu = (hw.get("live") or {}).get("cpu_percent")
        if cpu is not None:
            lines.append(Text.assemble(f"{'CPU':28}      ", bar(cpu, 100)))
        self.query_one("#hub-gpu", Static).update(Group(*lines))

        models = Table("Loaded model", "Size", "VRAM", "Engine", "Last tok/s", box=None, header_style="bold")
        for m in o["models"]:
            style = "yellow" if m["spilled"] else ""
            tps = "-" if m["last_gen_tps"] is None else f"{m['last_gen_tps']:.1f}"
            models.add_row(
                t(m["name"], style), human_bytes(m["size"]), t(f"{m['vram_percent']}%", style), m["engine"], tps
            )
        self.query_one("#models-live", Static).update(models if o["models"] else Text("No model loaded", style="dim"))

        streams = Table("State", "Priority", "Age", "Model", "Path", box=None, header_style="bold")
        for st in s["streams"]:
            streams.add_row(st["state"], st["priority"], f"{st['age_s']:.0f}s", t(st["model"]), t(st["path"]))
        self.query_one("#streams", Static).update(
            Group(Text("Active requests", style="bold"), streams)
            if s["streams"]
            else Text("No active requests", style="dim")
        )
        recent = [t(f"{ts(e['ts'])[11:]}  {e['kind']}  {self._detail(e)}", "dim") for e in s["events"][-6:]]
        lines_log = [t(line, "dim") for line in list(self.log_lines)[-4:]]
        self.query_one("#recent", Static).update(Group(Text("Recent events", style="bold"), *(recent + lines_log)))

    @staticmethod
    def _detail(e: dict[str, Any]) -> str:
        return " ".join(f"{k}={v}" for k, v in e.items() if k not in ("seq", "ts", "kind") and v not in (None, ""))

    def _node_rows(self, nodes: list[dict[str, Any]]) -> list[tuple[str, list[Text]]]:
        needle = self._filter.lower()
        rows = []
        for n in nodes:
            haystack = f"{n['name']} {n['ip']} {n['status']} {_gpu_names(n['gpus'])} {n['id']}".lower()
            if needle and needle not in haystack:
                continue
            utils = [g["util"] for g in n["gpus"] if g.get("util") is not None]
            cpu = n["cpu_percent"]
            ram = f"{gib(n['ram_used'])}/{gib(n['ram_total'])} GB" if n["ram_used"] is not None else "-"
            tunnel_style = "bold yellow" if n["tunnel_effective"] == "direct" else ""
            cells = [
                STATUS_DOT.get(n["status"], Text("?")),
                t(n["name"]),
                t(n["ip"]),
                t(n["status"], "bold red" if n["status"] == "incompatible" else ""),
                t(_gpu_names(n["gpus"])),
                t(f"{gib(n['vram_total'])}/{gib(n['vram_free'])}/{gib(n['vram_lendable'])}"),
                t(f"{max(utils)}%" if utils else "-"),
                t(f"{cpu:.0f}% · {ram}" if cpu is not None else "-"),
                _link(n["link"]),
                t(n["role"]),
                t(n["tunnel_effective"], tunnel_style),
            ]
            rows.append((n["id"], cells))
        return rows

    def _update_nodes(self, s: dict[str, Any]) -> None:
        table = self.query_one("#node-table", DataTable)
        rows = self._node_rows(s["nodes"])
        if [row.key.value for row in table.ordered_rows] == [key for key, _ in rows]:
            for key, cells in rows:
                for column, cell in zip(self._node_columns, cells, strict=True):
                    table.update_cell(key, column, cell)
            return
        selected = self._cursor_key(table)
        table.clear()
        for key, cells in rows:
            table.add_row(*cells, key=key)
        if selected in [key for key, _ in rows]:
            table.move_cursor(row=table.get_row_index(selected))

    def _update_detail(self) -> None:
        info = self.query_one("#detail-info", Static)
        node = self.hub.registry.nodes.get(self.selected_node or "")
        if node is None:
            info.update(Text("Select a helper on the Nodes tab (2) and press Enter.", style="dim"))
            self.query_one("#detail-net", Static).update("")
            return
        d = node.detail()
        caps = d["capabilities"]
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="bold")
        grid.add_column()
        cpu = caps.get("cpu") or {}
        lending = d["lending"]
        rows = [
            ("Name", d["name"]),
            ("Status", d["status"] + (f" ({d['lend_reason']})" if d["lend_reason"] else "")),
            ("ID", d["id"]),
            ("Address", d["ip"]),
            ("Agent", f"{d['agent_version']} (protocol {d['protocol']})"),
            ("System", f"{caps.get('os', '?')} {caps.get('os_release', '')} {caps.get('arch', '')}"),
            ("CPU", f"{cpu.get('model', '?')} · {cpu.get('cores', '?')} cores / {cpu.get('threads', '?')} threads"),
            ("RAM", f"{gib(d['ram_used'])} / {gib(d['ram_total'])} GB"),
            ("Backends", ", ".join(caps.get("backends") or []) or "-"),
            ("Link", _link(d["link"]).plain),
            ("Tunnel", f"{d['tunnel_effective']} (requested {d['tunnel_mode']})"),
            ("Lending rules", _rules_text(lending)),
            ("Lendable now", f"{gib(d['vram_lendable'])} GB VRAM"),
            ("Connected", ago(d["connected_at"]) if d["connected"] else f"offline, last seen {ago(d['last_seen'])}"),
        ]
        for label, value in rows:
            grid.add_row(label, t(value))
        gpus = Table("GPU", "Backend", "VRAM used/total", "Util", "Temp", box=None, header_style="bold")
        for g in d["gpus"]:
            gpus.add_row(
                t(g.get("name")),
                t(g.get("backend")),
                f"{gib(g.get('vram_used'))}/{gib(g.get('vram_total'))} GB",
                f"{g['util']}%" if g.get("util") is not None else "-",
                f"{g['temp']}°C" if g.get("temp") is not None else "-",
            )
        parts: list[Any] = [grid, gpus]
        for w in d["warnings"]:
            parts.append(Text("⚠ " + safe(w), style="bold yellow"))
        parts.append(Text("Keys: P pause/resume · K kick · N net test · T tunnel mode · X forget", style="dim"))
        info.update(Group(*parts))
        history = d["history"][-120:]
        minutes = len(history) * 2 // 60
        for key, label, values, unit in (
            ("gpu", "GPU utilisation", [h[1] or 0 for h in history], "%"),
            ("vram", "VRAM used", [h[2] / 1024**3 for h in history], " GB"),
            ("cpu", "CPU", [h[3] for h in history], "%"),
        ):
            # Pin the scale at 0 so a flat line doesn't look like a full bar.
            self.query_one(f"#spark-{key}", Sparkline).data = [0.0, *values] if values else []
            now = f"now {values[-1]:.0f}{unit}, peak {max(values):.0f}{unit}" if values else "no data yet"
            self.query_one(f"#label-{key}", Static).update(t(f"{label} - {now} (last {minutes or '<1'} min)"))
        net = Table("When", "RTT", "Hub→helper", "Helper→hub", "Link", box=None, header_style="bold")
        for r in reversed(d["net_tests"][-10:]):
            net.add_row(
                ts(r.get("ts")),
                f"{r.get('rtt_ms', '-')} ms",
                f"{r.get('down_mbps', '-')} Mbps",
                f"{r.get('up_mbps', '-')} Mbps",
                t(r.get("link")),
            )
        parts = [
            Group(Text("Network tests (N to run one)", style="bold"), net)
            if d["net_tests"]
            else Text("No network test yet - press N", style="dim")
        ]
        if d.get("tunnel_test"):
            parts.append(Text("\nSplit-mode tunnel (M to measure again)", style="bold"))
            parts.extend(t(line) for line in tunnel_test_lines(d["tunnel_test"]))
        else:
            parts.append(Text("\nNo tunnel test yet - press M", style="dim"))
        self.query_one("#detail-net", Static).update(Group(*parts))

    def _update_models(self) -> None:
        table = self.query_one("#model-table", DataTable)
        selected = self._cursor_key(table)
        table.clear()
        for m in self.admin.models():
            fits = "?" if m["fits"] is None else ("yes" if m["fits"] else "no → helpers")
            table.add_row(
                t(m["name"]),
                human_bytes(m["size"]),
                t(m["parameters"]),
                t(m["quantization"]),
                t(
                    f"{m['vram_percent']}% VRAM" if m["loaded"] else "-",
                    "yellow" if m["loaded"] and m["vram_percent"] < 100 else "",
                ),
                t(fits, "yellow" if m["fits"] is False else ""),
                t(m["plan"]),
                t("-" if m["last_gen_tps"] is None else f"{m['last_gen_tps']:.1f}"),
                key=m["name"],
            )
        if selected is not None:
            try:
                table.move_cursor(row=table.get_row_index(selected), scroll=False)
            except Exception:  # noqa: BLE001 - the model is gone
                pass

    def _update_log(self) -> None:
        if self.hub.event_seq == self._log_rendered:
            return
        self._log_rendered = self.hub.event_seq
        table = self.query_one("#log-table", DataTable)
        table.clear()
        needle = self._filter_log.lower()
        for e in reversed(list(self.hub.recent_events)):
            detail = self._detail(e)
            if needle and needle not in f"{e['kind']} {detail}".lower():
                continue
            style = "red" if e["kind"] in ALERT_EVENTS else "yellow" if e["kind"] in WARN_EVENTS else ""
            table.add_row(t(ts(e["ts"])), t(e["kind"], style), t(detail), key=str(e["seq"]))

    # ----- input ---------------------------------------------------------------------

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "node-filter":
            self._filter = event.value
            self._update_nodes(self.hub.get_snapshot())
        elif event.input.id == "log-filter":
            self._filter_log = event.value
            self._log_rendered = -1
            self._update_log()

    def on_key(self, event: Any) -> None:
        if event.key == "escape" and isinstance(self.focused, Input):
            self.focused.value = ""
            target = "#node-table" if self.focused.id == "node-filter" else "#log-table"
            self.query_one(target).focus()

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        self._log_rendered = -1
        self.refresh_bindings()
        self.refresh_visible()
        self._focus_tab(event.pane.id or "")

    def _focus_tab(self, tab: str) -> None:
        target = {"nodes": "#node-table", "models": "#model-table", "log": "#log-table"}.get(tab)
        if target:
            self.query_one(target).focus()
        else:
            self.screen.set_focus(None)

    def action_tab(self, tab: str) -> None:
        while len(self.screen_stack) > 1:
            self.pop_screen()
        self.query_one(TabbedContent).active = tab
        self.call_after_refresh(self._focus_tab, tab)

    def action_filter(self) -> None:
        tabs = self.query_one(TabbedContent)
        if tabs.active not in ("nodes", "log"):
            tabs.active = "nodes"
        self.query_one("#node-filter" if tabs.active == "nodes" else "#log-filter", Input).focus()

    # ----- node actions --------------------------------------------------------------

    def say(self, message: str, **kwargs: Any) -> None:
        """A toast. Never markup: messages contain helper names and other remote text."""
        self.notify(safe(message), markup=False, **kwargs)

    @staticmethod
    def _cursor_key(table: DataTable) -> str | None:
        if table.row_count == 0:
            return None
        try:
            return table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value
        except Exception:  # noqa: BLE001
            return None

    def _target(self) -> str | None:
        if self.query_one(TabbedContent).active == "nodes":
            key = self._cursor_key(self.query_one("#node-table", DataTable))
            if key is not None:
                self.selected_node = key
        if self.selected_node is None or self.selected_node not in self.hub.registry.nodes:
            self.say("Select a helper first (Nodes tab)", severity="warning")
            return None
        return self.selected_node

    async def _admin(self, cmd: str, **args: Any) -> Any:
        try:
            result = await self.admin.dispatch(cmd, args)
        except AdminError as exc:
            self.say(str(exc), severity="error")
            return None
        except Exception:  # noqa: BLE001 - a TUI bug must never take the hub down
            log.exception("admin action %s failed in the TUI", cmd)
            self.say("That action failed unexpectedly (see the log).", severity="error")
            return None
        self.refresh_visible()
        return result

    def action_node_detail(self) -> None:
        if self._target() is not None:
            self.query_one(TabbedContent).active = "detail"

    async def action_node_pause(self) -> None:
        node_id = self._target()
        if node_id is None:
            return
        node = self.hub.registry.nodes[node_id]
        cmd = "nodes.resume" if node.hub_paused else "nodes.pause"
        if await self._admin(cmd, ref=node_id) is not None:
            self.say(f"{node.name}: {'resumed' if cmd == 'nodes.resume' else 'paused (the hub will not use it)'}")

    def action_node_kick(self) -> None:
        node_id = self._target()
        if node_id is None:
            return
        name = self.hub.registry.nodes[node_id].name

        async def done(yes: bool | None) -> None:
            if yes and await self._admin("nodes.kick", ref=node_id) is not None:
                self.say(f"Kicked {name}; it will reconnect in a minute (pause it to stop using it).")

        self.push_screen(ConfirmScreen(f"Disconnect {safe(name)} now?"), done)

    def action_node_net_test(self) -> None:
        node_id = self._target()
        if node_id is None:
            return
        name = self.hub.registry.nodes[node_id].name
        self.say(f"Testing the network to {name}…", timeout=4)

        async def run() -> None:
            r = await self._admin("nodes.net_test", ref=node_id)
            if r is not None:
                self.say(
                    f"{name}: RTT {r['rtt_ms']} ms · hub→helper {r['down_mbps']} Mbps · helper→hub {r['up_mbps']} Mbps",
                    timeout=10,
                )

        self.run_worker(run(), group="net-test")

    def action_node_tunnel_test(self) -> None:
        node_id = self._target()
        if node_id is None:
            return
        name = self.hub.registry.nodes[node_id].name
        self.say(f"Measuring the split-mode tunnel to {name}…", timeout=4)

        async def run() -> None:
            r = await self._admin("nodes.tunnel_test", ref=node_id)
            if r is not None:
                secure = r["secure"]
                extra = f" · encryption costs {r['overhead_pct']}%" if r.get("overhead_pct") is not None else ""
                self.say(f"{name}: secure {secure['mbps']} Mbps, RTT {secure['rtt_ms']} ms{extra}", timeout=10)

        self.run_worker(run(), group="tunnel-test")

    def action_node_tunnel(self) -> None:
        node_id = self._target()
        if node_id is None:
            return
        node = self.hub.registry.nodes[node_id]
        new = "secure" if node.tunnel_mode == "direct" else "direct"

        async def done(yes: bool | None) -> None:
            if yes and await self._admin("nodes.tunnel", ref=node_id, mode=new) is not None:
                self.say(f"{node.name}: tunnel mode {new}")

        if new == "direct":
            question = (
                f"Use trusted-LAN direct mode for {safe(node.name)}?\n\nSplit-mode traffic (model weights and "
                "activations) will cross your network UNENCRYPTED. Only for a cable LAN you fully trust. The helper's "
                "owner must also allow it (allow_direct)."
            )
            self.push_screen(ConfirmScreen(question), done)
        else:
            self.run_worker(done(True))

    def action_node_forget(self) -> None:
        node_id = self._target()
        if node_id is None:
            return
        name = self.hub.registry.nodes[node_id].name

        async def done(yes: bool | None) -> None:
            if yes and await self._admin("nodes.forget", ref=node_id) is not None:
                self.selected_node = None
                self.say(f"Forgot {name}")

        self.push_screen(ConfirmScreen(f"Forget {safe(name)} (its settings and history)?"), done)

    def action_phase5(self, what: str) -> None:
        self.say(f"'{what}' arrives with the planner (Phase 5).", severity="warning")

    def action_quit_hub(self) -> None:
        def done(yes: bool | None) -> None:
            if yes:
                self.quit_confirmed = True
                self.exit()

        self.push_screen(ConfirmScreen("Quit? This stops the hub: the front door and all helpers."), done)
