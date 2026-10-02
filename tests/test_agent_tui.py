"""Drive the agent TUI against a real hub."""

from __future__ import annotations

from textual.widgets import Checkbox, Input

from ollamapower.agent.rpc_manager import RpcSession
from ollamapower.agent.tui.app import AgentTUI, RulesScreen
from ollamapower.tui.widgets import ConfirmScreen

from .test_control import Setup, hub_key, until  # noqa: F401 - fixture
from .test_hub_tui import plain, shot, wait


async def test_first_connect_asks_to_trust_the_key(tmp_path, hub_key):  # noqa: F811
    s = Setup(tmp_path, hub_key)
    await s.start()
    agent = s.agent(name="laptop", confirm=None)
    app = AgentTUI(agent)
    try:
        async with app.run_test(size=(150, 45)) as pilot:
            assert await wait(pilot, lambda: isinstance(app.screen, ConfirmScreen))
            assert "Trust this hub?" in app.screen.question
            await shot(app, "agent-trust")
            await pilot.press("y")
            assert await wait(pilot, lambda: agent.state.connection == "connected")
            assert await wait(pilot, lambda: "connected" in plain(app, "#connection"))
            assert "yes (known_hub)" in plain(app, "#connection")
            assert await wait(pilot, lambda: "not installed" in plain(app, "#lending"))  # llama.cpp row
            agent.rpc.session = RpcSession("session-1", "Vulkan0", 6000, "127.0.0.1", model="big:70b")
            agent.state.version += 1
            assert await wait(pilot, lambda: "Vulkan0 (6000 MiB) for big:70b" in plain(app, "#lending"))
            await shot(app, "agent")
            agent.rpc.session = None

            await pilot.press("p")
            assert await wait(pilot, lambda: s.node("laptop").status == "paused")
            assert await wait(pilot, lambda: "PAUSED" in plain(app, "#lending"))
            await pilot.press("p")
            assert await wait(pilot, lambda: s.node("laptop").status == "lending")

            await pilot.press("e")
            assert isinstance(app.screen, RulesScreen)
            app.screen.query_one("#rule-max_vram_gb", Input).value = "6"
            app.screen.query_one("#rule-allow_cpu", Checkbox).value = True
            await pilot.click("#save")
            assert await wait(pilot, lambda: s.node("laptop").lending.get("max_vram_gb") == 6.0)
            assert "allow_cpu = true" in agent.config.path.read_text()

            await pilot.press("n")
            assert await wait(pilot, lambda: "Last net test" in plain(app, "#lending"), 15)
    finally:
        await s.close()


async def test_refusing_the_key_stops_with_a_banner(tmp_path, hub_key):  # noqa: F811
    s = Setup(tmp_path, hub_key)
    await s.start()
    agent = s.agent(name="laptop", confirm=None)
    app = AgentTUI(agent)
    try:
        async with app.run_test(size=(150, 45)) as pilot:
            assert await wait(pilot, lambda: isinstance(app.screen, ConfirmScreen))
            await pilot.press("n")
            assert await wait(pilot, lambda: "not confirmed" in plain(app, "#banner"))
            assert s.node("laptop") is None
    finally:
        await s.close()


async def test_invalid_rules_are_explained(tmp_path, hub_key):  # noqa: F811
    s = Setup(tmp_path, hub_key)
    await s.start()
    agent = s.agent(name="laptop")
    app = AgentTUI(agent)
    try:
        async with app.run_test(size=(150, 45)) as pilot:
            await pilot.press("e")
            app.screen.query_one("#rule-gpus", Input).value = "zero"
            await pilot.click("#save")
            assert isinstance(app.screen, RulesScreen)
            assert "indexes" in str(app.screen.query_one("#rule-error").content)
            await pilot.press("escape")
            app.screen  # noqa: B018
            await pilot.press("e")
            app.screen.query_one("#rule-max_vram_percent", Input).value = "150"
            await pilot.click("#save")
            assert await wait(pilot, lambda: not isinstance(app.screen, RulesScreen))
            assert agent.config.lending.max_vram_percent == 0  # refused by validation, toast shown
    finally:
        await s.close()
