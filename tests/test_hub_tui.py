"""Drive the hub TUI headlessly with Textual's pilot, against a real hub and agent."""

from __future__ import annotations

import asyncio
import os

from rich.console import Console
from textual.widgets import DataTable, TabbedContent

from ollamapower.admin.service import Admin
from ollamapower.agent.pinning import format_fingerprint
from ollamapower.core.encrypt_layer import public_key_fingerprint
from ollamapower.hub.tui.app import HubTUI
from ollamapower.tui.widgets import ConfirmScreen

from .test_control import Setup, hub_key, until  # noqa: F401 - fixture

SHOTS = os.environ.get("OLLAMAPOWER_TUI_SHOTS")  # a directory to save SVG screenshots into


def plain(app, selector: str) -> str:
    console = Console(width=220, record=True, file=open(os.devnull, "w", encoding="utf-8"))
    console.print(app.query_one(selector).content)
    return console.export_text()


async def shot(app, name: str) -> None:
    if SHOTS:
        app.save_screenshot(filename=f"{name}.svg", path=SHOTS)


async def wait(pilot, condition, timeout: float = 5.0) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if condition():
            return True
        await pilot.pause(0.1)
    return condition()


async def test_overview_nodes_detail_log(tmp_path, hub_key):  # noqa: F811
    s = Setup(tmp_path, hub_key)
    await s.start()
    agent = s.agent(name="gamer-pc", max_vram_gb=10.0)
    s.run(agent)
    assert await until(lambda: s.node("gamer-pc") is not None and s.node("gamer-pc").telemetry)
    app = HubTUI(Admin(s.hub))
    try:
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause()
            assert "Control plane" in plain(app, "#status") and "1 lending" in plain(app, "#status")
            fp = format_fingerprint(public_key_fingerprint(hub_key.public_key()))  # whole, to compare
            assert fp[:47] in plain(app, "#status") and fp[48:] in plain(app, "#status")
            await shot(app, "overview")

            await pilot.press("2")
            table = app.query_one("#node-table", DataTable)
            assert await wait(pilot, lambda: table.row_count == 1)
            row = table.get_row_at(0)
            assert row[1].plain == "gamer-pc" and row[3].plain == "lending" and row[4].plain == "RTX 4090"
            assert row[5].plain == "24.0/20.0/10.0"
            await shot(app, "nodes")

            await pilot.press("enter")
            assert app.query_one(TabbedContent).active == "detail"
            assert await wait(pilot, lambda: "Lendable now" in plain(app, "#detail-info"))
            assert "10.0 GB VRAM" in plain(app, "#detail-info")
            await shot(app, "detail")

            await pilot.press("n")  # network test from the detail tab
            assert await wait(pilot, lambda: bool(s.node("gamer-pc").net_tests), 10)

            await pilot.press("p")  # hub-side pause
            assert s.node("gamer-pc").hub_paused

            await pilot.press("5")
            log_table = app.query_one("#log-table", DataTable)
            assert await wait(pilot, lambda: log_table.row_count >= 2)
            kinds = {log_table.get_row_at(i)[1].plain for i in range(log_table.row_count)}
            assert {"node_join", "net_test", "node_paused"} <= kinds
    finally:
        await s.close()


async def test_kick_asks_first_and_tunnel_warns(tmp_path, hub_key):  # noqa: F811
    s = Setup(tmp_path, hub_key)
    await s.start()
    agent = s.agent(name="desk")
    s.run(agent)
    assert await until(lambda: s.node("desk") is not None and s.node("desk").connected)
    app = HubTUI(Admin(s.hub))
    try:
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.press("2")
            table = app.query_one("#node-table", DataTable)
            assert await wait(pilot, lambda: table.row_count == 1)
            await pilot.press("k")
            assert isinstance(app.screen, ConfirmScreen)
            await pilot.press("n")  # no
            assert s.node("desk").connected
            await pilot.press("t")
            assert isinstance(app.screen, ConfirmScreen) and "UNENCRYPTED" in app.screen.question
            await pilot.press("y")
            assert await wait(pilot, lambda: s.node("desk").tunnel_mode == "direct")
            # The helper's owner hasn't allowed it, so it stays secure, with a warning.
            assert s.node("desk").tunnel_effective == "secure"
            assert any("hasn't allowed it" in w for w in s.hub.warnings())
            await pilot.press("k")
            await pilot.press("y")
            assert await wait(pilot, lambda: not s.node("desk").connected)
    finally:
        await s.close()


async def test_quit_asks_and_models_tab_explains(tmp_path, hub_key):  # noqa: F811
    s = Setup(tmp_path, hub_key)
    app = HubTUI(Admin(s.hub))
    async with app.run_test(size=(160, 40)) as pilot:
        await pilot.press("4")
        await pilot.press("b")
        await pilot.pause()
        await pilot.press("q")
        assert isinstance(app.screen, ConfirmScreen)
        await pilot.press("y")
        await pilot.pause()
    assert app.quit_confirmed


async def test_slash_from_overview_types_into_the_filter(tmp_path, hub_key):  # noqa: F811
    """`/` on another tab opens Nodes with the filter focused; letters typed there are text, not node actions."""
    from textual.widgets import Input

    s = Setup(tmp_path, hub_key)
    await s.start()
    agent = s.agent(name="desk")
    s.run(agent)
    assert await until(lambda: s.node("desk") is not None and s.node("desk").connected)
    app = HubTUI(Admin(s.hub))
    try:
        async with app.run_test(size=(180, 50)) as pilot:
            await pilot.pause()
            await pilot.press("/")
            await pilot.pause()
            await pilot.pause()
            assert app.query_one(TabbedContent).active == "nodes"
            assert isinstance(app.focused, Input) and app.focused.id == "node-filter"
            await pilot.press("p", "t")
            await pilot.pause()
            assert app.query_one("#node-filter", Input).value == "pt"
            assert not s.node("desk").hub_paused
            assert not isinstance(app.screen, ConfirmScreen)
            # A plain tab switch still focuses the table.
            await pilot.press("escape", "1", "2")
            assert await wait(pilot, lambda: app.focused is app.query_one("#node-table"))
    finally:
        await s.close()


async def test_models_tab_pins_and_shows_why(tmp_path, hub_key):  # noqa: F811
    from ollamapower.ollama.client import InstalledModel

    s = Setup(tmp_path, hub_key)
    await s.start()
    s.hub.ollama.installed = [InstalledModel(name="big:70b", size=40 << 30, digest="d1", quantization="Q4_K_M")]
    app = HubTUI(Admin(s.hub))
    try:
        async with app.run_test(size=(200, 50)) as pilot:
            await pilot.press("4")
            table = app.query_one("#model-table", DataTable)
            assert await wait(pilot, lambda: table.row_count == 1)
            assert table.get_row_at(0)[6].plain == "auto" and "no benchmark yet" in table.get_row_at(0)[9].plain
            table.focus()
            await pilot.press("f")
            assert await wait(pilot, lambda: s.hub.plans.record("big:70b")["pin"] == "local")
            assert await wait(pilot, lambda: "forced local" in table.get_row_at(0)[9].plain)
            await pilot.press("u")
            assert await wait(pilot, lambda: s.hub.plans.record("big:70b")["pin"] is None)
            assert "No benchmark" in plain(app, "#model-progress")
            await shot(app, "models")
    finally:
        await s.close()
