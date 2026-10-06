"""The agent's split-mode sessions under concurrency: pauses, stops and replacements while a session starts.

Starting ``rpc-server`` can take up to a minute and a half, so the owner (or the
hub) acting in the middle of a start is a normal case, not a corner case. These
tests hold ``_launch`` on an event to put that moment exactly where they want it.
"""

from __future__ import annotations

import asyncio
import types
from pathlib import Path
from typing import Any

import pytest

from ollamapower.agent.client import AgentClient
from ollamapower.agent.config import AgentConfig
from ollamapower.agent.lending import Decision
from ollamapower.agent.rpc_manager import RpcRefused
from ollamapower.aio import wait_for
from ollamapower.core.tunnel import TunnelIdentity

HUB_CERT = TunnelIdentity.generate("hub").cert_pem


class FakeTelemetry:
    def static(self) -> dict[str, Any]:
        return {}

    def sample(self) -> dict[str, Any]:
        return {}


class FakeChannel:
    def __init__(self) -> None:
        self.closed = False
        self.peer = ("127.0.0.1", 1)
        self.local = ("127.0.0.1", 2)
        self.sent: list[tuple[str, dict[str, Any]]] = []

    async def send(self, kind: str, **fields: Any) -> None:
        if self.closed:
            raise ConnectionError("closed")
        self.sent.append((kind, fields))

    def kinds(self, session_id: str) -> list[str]:
        return [k for k, f in self.sent if f.get("session_id") == session_id]


class Agent:
    """An AgentClient with a fake hub channel and an ``rpc-server`` launch the test controls."""

    def __init__(self, tmp_path: Path) -> None:
        self.client = AgentClient(AgentConfig(config_dir=tmp_path), "token", "127.0.0.1", 1, telemetry=FakeTelemetry())
        self.client.state.connection = "connected"
        self.client._save_rules = lambda rules: None  # don't write agent.toml
        self.channel = FakeChannel()
        self.client.channel = self.channel
        self.rpc = self.client.rpc
        self.rpc.status = "ready"
        self.rpc.binaries = types.SimpleNamespace(rpc_server=Path("/bin/true"), build="b1")
        self.launching = asyncio.Event()  # set when a launch begins
        self.release = asyncio.Event()  # set by the test to let it finish
        self.launch_error: Exception | None = None

        async def choose(wanted: str, rules: Any) -> tuple[str, int, int]:
            return "GPU0", 4000, 8000

        async def launch(session: Any, cache: bool) -> None:
            self.launching.set()
            await self.release.wait()
            if self.launch_error is not None:
                raise self.launch_error

        self.rpc._choose = choose
        self.rpc._launch = launch

    def start(self, session_id: str = "s1", **extra: Any) -> asyncio.Task[None]:
        message = {
            "type": "rpc_start",
            "session_id": session_id,
            "tunnel": {"mode": "secure", "cert": HUB_CERT},
            "model": "m",
            **extra,
        }
        return asyncio.ensure_future(self.client._rpc_start(self.channel, message))

    async def started(self, session_id: str = "s1") -> None:
        """Start a session and let it finish starting."""
        self.release.set()
        await self.start(session_id)
        assert self.rpc.session is not None and self.rpc.session.session_id == session_id
        self.launching.clear()
        self.release.clear()

    def close(self) -> None:
        session = self.rpc.session
        if session is not None and session.relay is not None:
            session.relay.close()


@pytest.fixture
async def agent(tmp_path):
    a = Agent(tmp_path)
    yield a
    a.close()


async def test_pause_while_a_session_is_starting_refuses_it(agent):
    start = agent.start()
    await agent.launching.wait()
    pause = asyncio.ensure_future(agent.client.set_paused(True))
    await asyncio.sleep(0.05)
    agent.release.set()
    await wait_for(asyncio.gather(start, pause), 5)
    assert agent.rpc.session is None
    assert agent.channel.kinds("s1") == ["rpc_error"]  # never rpc_ready: the hub doesn't get a paused helper
    assert "paused" in next(f["error"] for k, f in agent.channel.sent if k == "rpc_error")
    assert agent.client.state.lend_state == "paused"


async def test_schedule_end_or_auto_yield_while_starting_refuses_it(agent, monkeypatch):
    start = agent.start()
    await agent.launching.wait()
    monkeypatch.setattr(agent.client.guard, "evaluate", lambda *a, **k: Decision("yielding", "the owner is gaming"))
    await agent.client._check_lending()
    agent.release.set()
    await wait_for(start, 5)
    assert agent.rpc.session is None
    assert agent.channel.kinds("s1") == ["rpc_error"]
    assert "the owner is gaming" in next(f["error"] for k, f in agent.channel.sent if k == "rpc_error")


async def test_changing_the_rules_while_starting_refuses_it(agent):
    start = agent.start()
    await agent.launching.wait()
    update = asyncio.ensure_future(agent.client.update_rules(allow_cpu=True))
    await asyncio.sleep(0.05)
    agent.release.set()
    await wait_for(asyncio.gather(start, update), 5)
    assert agent.rpc.session is None
    assert agent.channel.kinds("s1") == ["rpc_error"]


async def test_pause_stops_a_running_session_and_tells_the_hub(agent):
    await agent.started()
    relay = agent.rpc.session.relay
    await agent.client.set_paused(True)
    assert agent.rpc.session is None and relay.closed
    assert agent.channel.kinds("s1") == ["rpc_ready", "rpc_error"]


async def test_stop_current_waits_for_a_session_that_is_starting(agent):
    message = {"session_id": "s1", "tunnel": {"mode": "secure", "cert": HUB_CERT}}
    rules = agent.client.config.lending
    start = asyncio.ensure_future(agent.rpc.start(message, "127.0.0.1", "127.0.0.1", rules, lambda: ""))
    await agent.launching.wait()
    stop = asyncio.ensure_future(agent.rpc.stop_current("paused"))
    await asyncio.sleep(0.05)
    assert not stop.done()  # it waits for the start instead of missing the session
    agent.release.set()
    endpoint = await wait_for(start, 5)
    assert await wait_for(stop, 5) == "s1"
    assert endpoint["port"] and agent.rpc.session is None


# ----- which device a session gets -----------------------------------------------------

IGPU = "Vulkan0", "Intel(R) UHD Graphics 770", 32768, 30000  # llama.cpp lists the iGPU first, with shared RAM
DGPU = "Vulkan1", "NVIDIA GeForce RTX 3060", 12288, 11000
TELEMETRY = [{"index": 0, "name": "NVIDIA GeForce RTX 3060", "vendor": "nvidia", "vram_total": 12288 * 1024 * 1024}]


def manager(tmp_path, devices, gpus=TELEMETRY, ram_mib=6000, monkeypatch=None):
    from ollamapower.agent import rpc_manager
    from ollamapower.agent.rpc_manager import RpcManager
    from ollamapower.llamacpp.binaries import Device

    rpc = RpcManager(AgentConfig(config_dir=tmp_path), lambda text: None)
    rpc.binaries = types.SimpleNamespace(probe=lambda: None, devices=[Device(*d) for d in devices])
    rpc.gpus = gpus
    if monkeypatch is not None:
        monkeypatch.setattr(rpc_manager, "cpu_cap_mib", lambda rules: ram_mib)
    return rpc


async def test_the_gpu_allow_list_uses_the_owners_gpu_numbers(tmp_path, monkeypatch):
    from ollamapower.agent.config import LendingRules

    rpc = manager(tmp_path, [IGPU, DGPU], monkeypatch=monkeypatch)
    # "GPU 0" in the agent's views is the RTX 3060, even though llama.cpp calls it device 1.
    assert (await rpc._choose("", LendingRules(gpus=[0])))[0] == "Vulkan1"
    with pytest.raises(RpcRefused):
        await rpc._choose("", LendingRules(gpus=[1]))  # no GPU 1 here, and the iGPU isn't one the owner listed


async def test_a_dedicated_gpu_wins_over_an_integrated_one_with_more_shared_memory(tmp_path, monkeypatch):
    from ollamapower.agent.config import LendingRules

    rpc = manager(tmp_path, [IGPU, DGPU], monkeypatch=monkeypatch)
    assert await rpc._choose("", LendingRules()) == ("Vulkan1", 11000, 12288)


async def test_an_integrated_gpu_lends_no_more_than_the_ram_cpu_lending_would(tmp_path, monkeypatch):
    from ollamapower.agent.config import LendingRules

    rpc = manager(tmp_path, [IGPU], gpus=[], ram_mib=6000, monkeypatch=monkeypatch)
    assert await rpc._choose("", LendingRules()) == ("Vulkan0", 6000, 32768)


async def test_start_rechecks_lending_after_launching(agent):
    reasons = iter(["", "outside the lending schedule"])
    message = {"session_id": "s1", "tunnel": {"mode": "secure", "cert": HUB_CERT}}
    agent.release.set()
    with pytest.raises(RpcRefused, match="outside the lending schedule"):
        await agent.rpc.start(message, "127.0.0.1", "127.0.0.1", agent.client.config.lending, lambda: next(reasons))
    assert agent.rpc.session is None


async def test_a_failed_replacement_leaves_no_stale_session(agent):
    await agent.started("s1")
    agent.launch_error = RpcRefused("no luck")
    agent.release.set()
    await wait_for(agent.start("s2"), 5)
    assert agent.rpc.session is None  # s1 was stopped for s2: it must not still be "the session"
    await agent.client.set_paused(True)
    assert agent.channel.kinds("s1") == ["rpc_ready"]  # not reported (or stopped) a second time
    assert agent.channel.kinds("s2") == ["rpc_error"]


class FakeProcess:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self.exited = asyncio.Event()

    def exit(self, code: int) -> None:
        if self.returncode is None:
            self.returncode = code
        self.exited.set()

    async def wait(self) -> int:
        await self.exited.wait()
        assert self.returncode is not None
        return self.returncode

    def terminate(self) -> None:
        self.exit(-15)

    kill = terminate


async def test_rpc_server_dying_while_the_session_starts_is_reported_once(agent):
    proc = FakeProcess()

    async def launch(session: Any, cache: bool) -> None:
        session.process = proc
        reader = asyncio.get_running_loop().create_future()
        reader.set_result(None)
        session.watcher = asyncio.ensure_future(agent.rpc._watch(session, proc, reader))
        agent.launching.set()
        await agent.release.wait()

    agent.rpc._launch = launch
    start = agent.start()
    await agent.launching.wait()
    proc.exit(1)  # crashes right after it came up, before the relay is ready
    await asyncio.sleep(0.05)
    agent.release.set()
    await wait_for(start, 5)
    assert agent.rpc.session is None  # a dead session must not be handed to the hub
    assert agent.channel.kinds("s1") == ["rpc_error"]


async def test_a_tunnel_test_started_as_the_hub_leaves_is_stopped(agent):
    agent.client.config.lending.allow_direct = True
    agent.channel.closed = True  # gone before rpc_ready can be sent
    message = {"type": "rpc_start", "session_id": "b1", "purpose": "bench", "tunnel": {"mode": "direct"}}
    await wait_for(agent.client._rpc_start(agent.channel, message), 5)
    assert agent.rpc.benches == {}


async def test_a_slow_rpc_stop_does_not_hold_up_the_receive_loop(agent):
    start = agent.start()
    await agent.launching.wait()
    # The hub gives up on the start and says stop: stopping waits for the start to finish,
    # which can take a minute and a half. Heartbeat replies and other messages can't.
    await wait_for(agent.client._on_message(agent.channel, {"type": "rpc_stop", "session_id": "s1"}), 1)
    agent.release.set()
    await wait_for(start, 5)
    await wait_for(asyncio.gather(*agent.client._background), 5)
    assert agent.rpc.session is None  # the stop still came after the start: nothing is left running
    assert agent.channel.kinds("s1") == ["rpc_ready"]


async def test_rpc_stop_right_after_a_tunnel_test_start_stops_it(agent):
    agent.client.config.lending.allow_direct = True
    start = {"type": "rpc_start", "session_id": "b1", "purpose": "bench", "tunnel": {"mode": "direct"}}
    await agent.client._on_message(agent.channel, start)
    await agent.client._on_message(agent.channel, {"type": "rpc_stop", "session_id": "b1"})
    await wait_for(asyncio.gather(*agent.client._background), 5)
    assert agent.rpc.benches == {}  # was left running until its two-minute expiry
