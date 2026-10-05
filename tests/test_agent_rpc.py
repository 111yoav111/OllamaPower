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


async def test_start_rechecks_lending_after_launching(agent):
    reasons = iter(["", "outside the lending schedule"])
    message = {"session_id": "s1", "tunnel": {"mode": "secure", "cert": HUB_CERT}}
    agent.release.set()
    with pytest.raises(RpcRefused, match="outside the lending schedule"):
        await agent.rpc.start(message, "127.0.0.1", "127.0.0.1", agent.client.config.lending, lambda: next(reasons))
    assert agent.rpc.session is None
