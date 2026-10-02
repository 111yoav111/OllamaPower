"""The agent TUI: connection and hub key, what is lent, live GPU use, lending rules.

Keys: ``P`` pause/resume lending · ``E`` edit the lending rules · ``N`` run a
network test · ``Q`` quit. The first connection to a new hub asks the owner
to confirm the hub's key fingerprint here.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from rich.console import Group
from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Footer, Header, Input, Label, Static

from ...admin.ipc import AdminError
from ...textsafe import safe
from ...tui.widgets import ConfirmScreen, ago, bar, gib, t, ts
from ..client import AgentClient, AgentStopped
from ..pinning import format_fingerprint
from ..service import AgentAdmin, agent_snapshot

log = logging.getLogger("ollamapower.agent.tui")

REFRESH_INTERVAL = 1.0
CONNECTION_STYLE = {"connected": "bold green", "connecting": "yellow", "retrying": "yellow", "stopped": "bold red"}


class RulesScreen(ModalScreen[dict[str, Any] | None]):
    """Edit the lending rules. Dismisses with the changed values, or None."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, rules: dict[str, Any]) -> None:
        super().__init__()
        self.rules = rules

    def compose(self) -> ComposeResult:
        r = self.rules
        with Vertical(classes="dialog wide"):
            yield Label(Text("Lending rules (you decide; the hub can't change them)", style="bold"))
            for key, label, value in (
                ("max_vram_gb", "Max VRAM to lend (GB, 0 = no cap)", r["max_vram_gb"]),
                ("max_vram_percent", "Max VRAM to lend (%, 0 = no cap)", r["max_vram_percent"]),
                ("gpus", "GPUs to lend (e.g. 0,1; empty = all)", ",".join(str(g) for g in r["gpus"])),
                ("schedule", "Schedule (e.g. 22:00-08:00; Phase 6)", ",".join(r["schedule"])),
            ):
                with Horizontal(classes="field"):
                    yield Label(label, classes="field-key")
                    yield Input(str(value), id=f"rule-{key}", classes="field-value")
            yield Checkbox("Lend CPU/RAM when there is no GPU", r["allow_cpu"], id="rule-allow_cpu")
            yield Checkbox(
                "Allow trusted-LAN direct mode (split traffic UNENCRYPTED; the hub owner must also choose it)",
                r["allow_direct"],
                id="rule-allow_direct",
            )
            yield Static(id="rule-error")
            with Horizontal(classes="buttons"):
                yield Button("Save", id="save", variant="primary")
                yield Button("Cancel", id="cancel")

    def _value(self, key: str) -> str:
        return self.query_one(f"#rule-{key}", Input).value.strip()

    def collect(self) -> dict[str, Any]:
        def number(key: str, kind: type) -> Any:
            raw = self._value(key) or "0"
            try:
                return kind(raw)
            except ValueError:
                raise ValueError(f"{key} must be a number") from None

        gpus = [g.strip() for g in self._value("gpus").split(",") if g.strip()]
        if not all(g.isdigit() for g in gpus):
            raise ValueError("GPUs must be indexes like 0,1")
        return {
            "max_vram_gb": number("max_vram_gb", float),
            "max_vram_percent": number("max_vram_percent", int),
            "gpus": [int(g) for g in gpus],
            "schedule": [w.strip() for w in self._value("schedule").split(",") if w.strip()],
            "allow_cpu": self.query_one("#rule-allow_cpu", Checkbox).value,
            "allow_direct": self.query_one("#rule-allow_direct", Checkbox).value,
        }

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(None)
            return
        try:
            self.dismiss(self.collect())
        except ValueError as exc:
            self.query_one("#rule-error", Static).update(Text(str(exc), style="bold red"))

    def action_cancel(self) -> None:
        self.dismiss(None)


class AgentTUI(App[None]):
    TITLE = "OllamaPower agent"
    CSS_PATH = "../../tui/power.tcss"
    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [
        Binding("p", "pause", "Pause/resume lending"),
        Binding("e", "edit_rules", "Edit rules"),
        Binding("n", "net_test", "Net test"),
        Binding("q", "quit_agent", "Quit"),
        Binding("ctrl+q", "quit_agent", "Quit", show=False, priority=True),
    ]

    def __init__(self, client: AgentClient, *, run_client: bool = True) -> None:
        super().__init__()
        self.client = client
        self.admin = AgentAdmin(client)
        self.run_client = run_client
        self.stopped_reason = ""

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with VerticalScroll():
            yield Static(id="banner", classes="warnings")
            yield Static(id="connection", classes="panel")
            yield Static(id="lending", classes="panel")
            yield Static(id="gpu", classes="panel")
            yield Static(id="events", classes="panel")
        yield Footer()

    def on_mount(self) -> None:
        if self.client.confirm is None:
            self.client.confirm = self.confirm_hub_key
        if self.run_client:
            self.run_worker(self._run_client(), exclusive=True, group="client")
        self.set_interval(REFRESH_INTERVAL, self.refresh_view)
        self.refresh_view()

    async def _run_client(self) -> None:
        try:
            await self.client.run()
        except AgentStopped as exc:
            self.stopped_reason = str(exc)
            self.refresh_view()

    async def confirm_hub_key(self, address: str, fingerprint: str) -> bool:
        """Trust on first use: show the hub's fingerprint and let the owner decide."""
        fut: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        question = (
            f"First connection to the hub at {address}.\n\nIts key fingerprint (SHA-256) is:\n\n"
            f"{format_fingerprint(fingerprint)}\n\nCompare it with what `ollamapower hub` (or `ollamapower status`) "
            "shows on the hub PC. Trust this hub?"
        )
        self.push_screen(ConfirmScreen(question, danger=False), lambda yes: fut.set_result(bool(yes)))
        return await fut

    # ----- view ----------------------------------------------------------------------

    def refresh_view(self) -> None:
        if not (self.is_running and self.screen_stack):
            return
        try:
            self._draw(agent_snapshot(self.client))
        except Exception:  # noqa: BLE001 - drawing must never stop the agent
            log.exception("agent TUI refresh failed")

    def _draw(self, s: dict[str, Any]) -> None:
        banner = self.query_one("#banner", Static)
        alerts = []
        if s["connection"] == "stopped":
            alerts.append(s["error"])
        if s["network"].get("kind") == "wifi":
            alerts.append("This PC reaches the hub over Wi-Fi: too slow for split mode. Use a cable.")
        if s["lending"]["allow_direct"]:
            alerts.append("Direct mode allowed: if the hub owner turns it on, split traffic is unencrypted.")
        banner.update(Text("\n".join("⚠ " + safe(a) for a in alerts), style="bold red") if alerts else "")
        banner.display = bool(alerts)

        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="bold")
        grid.add_column()
        state = s["connection"]
        net = s["network"]
        rows: list[tuple[str, Any]] = [
            ("Hub", t(s["hub"])),
            (
                "Connection",
                t(
                    state + (f" ({s['error']})" if s["error"] and state != "connected" else ""),
                    CONNECTION_STYLE.get(state, ""),
                ),
            ),
            ("Hub key", t(format_fingerprint(s["fingerprint"])[:47] + "…" if s["fingerprint"] else "-")),
            ("Key pinned", t("yes (known_hub)" if s["pinned"] else "no")),
            ("This helper", t(f"{s['name']} · {s['node_id']}")),
            ("Connected", t(ago(s["connected_since"]) if s["connected_since"] else "-")),
            (
                "Link",
                t(
                    f"{net.get('kind', '?')} via {net.get('interface', '?')}"
                    + (f" · {net['speed_mbps']} Mbps" if net.get("speed_mbps") else "")
                ),
            ),
        ]
        for label, value in rows:
            grid.add_row(label, value)
        self.query_one("#connection", Static).update(grid)

        r = s["lending"]
        lend = Table.grid(padding=(0, 2))
        lend.add_column(style="bold")
        lend.add_column()
        lend.add_row("Lending", t("PAUSED" if s["paused"] else "on", "bold yellow" if s["paused"] else "bold green"))
        lend.add_row("Lent right now", t("nothing yet (split mode arrives in Phase 3)", "dim"))
        caps = []
        if r["max_vram_gb"]:
            caps.append(f"{r['max_vram_gb']} GB")
        if r["max_vram_percent"]:
            caps.append(f"{r['max_vram_percent']}%")
        lend.add_row("Max VRAM", t(" and ".join(caps) or "no cap"))
        lend.add_row("GPUs", t(", ".join(str(g) for g in r["gpus"]) or "all"))
        lend.add_row(
            "Also allowed",
            t(
                ", ".join(
                    k
                    for k, on in (
                        ("CPU", r["allow_cpu"]),
                        ("direct mode", r["allow_direct"]),
                    )
                    if on
                )
                or "nothing else"
            ),
        )
        if s["last_net_test"]:
            n = s["last_net_test"]
            if "error" in n:
                lend.add_row("Last net test", t(n["error"], "red"))
            else:
                lend.add_row(
                    "Last net test",
                    t(f"RTT {n.get('rtt_ms')} ms · ↓ {n.get('down_mbps')} Mbps · ↑ {n.get('up_mbps')} Mbps"),
                )
        self.query_one("#lending", Static).update(
            Group(Text("Lending (P pause · E edit · N net test)", style="bold"), lend)
        )

        hw, tel = s["hardware"], s["telemetry"]
        live = {g.get("index"): g for g in tel.get("gpus", [])}
        lines: list[Any] = [Text("This PC", style="bold")]
        for g in hw.get("gpus", []):
            lg = live.get(g.get("index"), {})
            total, used = lg.get("vram_total", g.get("vram_total", 0)), lg.get("vram_used", 0)
            extra = "".join(f" · {k} {lg[k]}{u}" for k, u in (("util", "%"), ("temp", "°C")) if lg.get(k) is not None)
            lines.append(
                Text.assemble(
                    f"GPU {g.get('index')} {safe(g.get('name', ''))[:24]:24} ",
                    bar(used, total),
                    f" {gib(used)}/{gib(total)} GB{extra}",
                )
            )
        if not hw.get("gpus"):
            lines.append(Text("No GPU found: only CPU can be lent (if allowed)", style="yellow"))
        if tel.get("cpu_percent") is not None:
            lines.append(Text.assemble(f"{'CPU':30} ", bar(tel["cpu_percent"], 100)))
            lines.append(
                Text.assemble(
                    f"{'RAM':30} ",
                    bar(tel.get("ram_used", 0), tel.get("ram_total", 1)),
                    f" {gib(tel.get('ram_used'))}/{gib(tel.get('ram_total'))} GB",
                )
            )
        self.query_one("#gpu", Static).update(Group(*lines))

        events = [t(f"{ts(e['ts'])[11:]}  {e['message']}", "dim") for e in s["events"][-8:]]
        self.query_one("#events", Static).update(Group(Text("Recent", style="bold"), *events))

    # ----- actions -------------------------------------------------------------------

    def say(self, message: str, **kwargs: Any) -> None:
        self.notify(safe(message), markup=False, **kwargs)

    async def _admin(self, cmd: str, **args: Any) -> Any:
        try:
            result = await self.admin.dispatch(cmd, args)
        except AdminError as exc:
            self.say(str(exc), severity="error")
            return None
        self.refresh_view()
        return result

    async def action_pause(self) -> None:
        paused = self.client.state.paused
        if await self._admin("agent.resume" if paused else "agent.pause") is not None:
            self.say("Lending resumed" if paused else "Lending paused: the hub won't use this PC")

    def action_edit_rules(self) -> None:
        async def done(changes: dict[str, Any] | None) -> None:
            if changes is not None and await self._admin("agent.rules", **changes) is not None:
                self.say("Lending rules saved to agent.toml")

        self.push_screen(RulesScreen(self.client.config.lending.to_dict()), done)

    def action_net_test(self) -> None:
        self.say("Asking the hub for a network test…", timeout=4)

        async def run() -> None:
            r = await self._admin("agent.net_test")
            if r is not None and "error" not in r:
                self.say(
                    f"RTT {r['rtt_ms']} ms · hub→here {r['down_mbps']} Mbps · here→hub {r['up_mbps']} Mbps", timeout=10
                )
            elif r is not None:
                self.say(r["error"], severity="error")

        self.run_worker(run(), group="net-test")

    def action_quit_agent(self) -> None:
        def done(yes: bool | None) -> None:
            if yes:
                self.exit()

        self.push_screen(ConfirmScreen("Quit? The hub stops using this PC."), done)
