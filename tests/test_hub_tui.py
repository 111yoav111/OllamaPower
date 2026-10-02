"""Drive the hub TUI headlessly with Textual's pilot, against a real hub and agent."""

from __future__ import annotations

import asyncio
import os

from rich.console import Console
from textual.widgets import DataTable, TabbedContent

from ollamapower.admin.service import Admin
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
