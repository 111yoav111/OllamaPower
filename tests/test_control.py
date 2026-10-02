"""Hub control plane <-> agent, for real, over localhost."""

from __future__ import annotations

import asyncio
import json
import socket
import time
from typing import Any

import pytest

from ollamapower.agent import client as agent_client
from ollamapower.agent.client import AgentClient, AgentStopped
from ollamapower.agent.config import AgentConfig
from ollamapower.config import Config
from ollamapower.core import protocol
from ollamapower.core.auth import new_token
from ollamapower.core.channel import Channel
from ollamapower.core.encrypt_layer import SessionCrypto, generate_rsa_keypair, public_key_fingerprint
from ollamapower.hub.nodes import GIB, NodeRegistry, sanitize
from ollamapower.hub.runtime import Hub
from ollamapower.state import StateStore

TOKEN = new_token()


class FakeTelemetry:
    def __init__(self, vram_used: int = 4 * GIB) -> None:
        self.vram_used = vram_used

    def static(self) -> dict[str, Any]:
        return {
            "os": "Linux",
            "cpu": {"model": "Test CPU", "cores": 8, "threads": 16},
            "ram_total": 64 * GIB,
            "gpus": [{"index": 0, "name": "RTX 4090", "vendor": "nvidia", "backend": "cuda", "vram_total": 24 * GIB}],
            "backends": ["cuda", "cpu"],
        }

    def sample(self) -> dict[str, Any]:
        return {
            "cpu_percent": 10.0,
            "ram_total": 64 * GIB,
            "ram_used": 8 * GIB,
            "gpus": [{"index": 0, "vram_total": 24 * GIB, "vram_used": self.vram_used, "util": 30, "temp": 55}],
        }


@pytest.fixture(scope="module")
def hub_key():
    return generate_rsa_keypair()


async def until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return predicate()


class Setup:
    def __init__(self, tmp_path, hub_key, **control) -> None:
        cfg = Config()
        cfg.config_dir = tmp_path / "hub"
        for key, value in control.items():
            setattr(cfg.control, key, value)
        self.state = StateStore(tmp_path / "hub" / "state.json")
        self.hub = Hub(cfg, self.state)
        self.key = hub_key
        self.tmp = tmp_path
        self.agents: list[asyncio.Task] = []

    async def start(self, token: str = TOKEN) -> int:
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        await self.hub.start_control(self.key, token, sock=sock)
        return self.hub.control.port

    def agent(self, name: str = "desk", token: str = TOKEN, confirm: Any = "yes", **lending) -> AgentClient:
        cfg = AgentConfig(config_dir=self.tmp / name)
        cfg.agent.name = name
        for key, value in lending.items():
            setattr(cfg.lending, key, value)

        async def yes(addr, fp):
            return True

        confirm_cb = yes if confirm == "yes" else confirm
        return AgentClient(
            cfg, token, "127.0.0.1", self.hub.control.port, confirm=confirm_cb, telemetry=FakeTelemetry()
        )

    def run(self, agent: AgentClient) -> asyncio.Task:
        task = asyncio.ensure_future(agent.run())
        self.agents.append(task)
        return task

    def node(self, name: str = "desk"):
        return self.hub.registry.find(name)

    async def close(self) -> None:
        for t in self.agents:
            t.cancel()
        await asyncio.gather(*self.agents, return_exceptions=True)
        await self.hub.stop_control()


@pytest.fixture
async def setup(tmp_path, hub_key):
    s = Setup(tmp_path, hub_key)
    await s.start()
    yield s
    await s.close()


async def test_agent_joins_reports_and_heartbeats(setup):
    agent = setup.agent(max_vram_gb=8.0)
    setup.run(agent)
    assert await until(lambda: setup.node() is not None and setup.node().telemetry)
    node = setup.node()
    snap = node.summary()
    assert snap["status"] == "lending" and snap["connected"] and snap["ip"] == "127.0.0.1"
    assert snap["gpus"][0]["name"] == "RTX 4090" and snap["gpus"][0]["util"] == 30
    assert snap["vram_total"] == 24 * GIB and snap["vram_free"] == 20 * GIB
    assert snap["vram_lendable"] == 8 * GIB  # capped by the helper owner's rule
    assert snap["link"]["kind"] in ("loopback", "unknown")
    assert agent.state.connection == "connected" and agent.state.pinned
    assert any(e["kind"] == "node_join" for e in setup.hub.recent_events)
    assert await until(lambda: len(node.history) >= 2, 6)


async def test_key_is_pinned_on_first_use(setup):
    asked = []

    async def confirm(addr, fp):
        asked.append(fp)
        return True

    agent = setup.agent(confirm=confirm)
    setup.run(agent)
    assert await until(lambda: agent.state.connection == "connected")
    fp = public_key_fingerprint(setup.key.public_key())
    assert asked == [fp]
    assert agent.known.get(agent.hub_key) == fp
    # The next agent run with the same files doesn't ask again.
    again = setup.agent(confirm=None)
    setup.run(again)
    assert await until(lambda: again.state.connection == "connected")


async def test_changed_hub_key_is_refused_without_sending_the_token(setup):
    agent = setup.agent()
    agent.known.pin(agent.hub_key, "ab" * 32)
    with pytest.raises(AgentStopped, match="KEY CHANGED"):
        await asyncio.wait_for(agent.run(), 5)
    await asyncio.sleep(0.1)
    assert setup.node() is None
    assert not any(e["kind"] == "auth_failed" for e in setup.hub.recent_events)  # the hub never saw a token


async def test_unconfirmed_or_mismatched_key_stops(setup):
    async def no(addr, fp):
        return False

    with pytest.raises(AgentStopped, match="not confirmed"):
        await asyncio.wait_for(setup.agent(confirm=no).run(), 5)
    other = setup.agent(name="other", confirm=None)
    other.config.hub.fingerprint = "12:34"
    with pytest.raises(AgentStopped, match="doesn't match"):
        await asyncio.wait_for(other.run(), 5)
    good = setup.agent(name="third", confirm=None)
    good.config.hub.fingerprint = public_key_fingerprint(setup.key.public_key()).upper()
    setup.run(good)
    assert await until(lambda: good.state.connection == "connected")


async def test_wrong_token_stops_and_repeat_offenders_are_banned(tmp_path, hub_key):
    s = Setup(tmp_path, hub_key, ban_failures=3)
    await s.start()
    try:
        for _ in range(3):
            with pytest.raises(AgentStopped, match="wrong cluster token"):
                await asyncio.wait_for(s.agent(token=new_token()).run(), 5)
        assert "127.0.0.1" in s.hub.control.bans.active()
        assert any(e["kind"] == "ip_banned" for e in s.hub.recent_events)
        # Even the right token is turned away while banned (closed before the handshake).
        agent = s.agent(name="good")
        task = s.run(agent)
        await asyncio.sleep(0.5)
        assert agent.state.connection != "connected" and s.node("good") is None
        task.cancel()
    finally:
        await s.close()


async def _raw_connect(port: int, hub_fp: str | None = None) -> Channel:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    crypto = await SessionCrypto.node_handshake(reader, writer)
    return Channel(reader, writer, crypto)


def _hello(**over) -> dict:
    return {
        "protocol": protocol.PROTOCOL_VERSION,
        "agent_version": "0.1.0",
        "node_id": "raw-node-0001",
        "name": "raw",
        "token": TOKEN,
        **over,
    }


async def test_incompatible_protocol_is_rejected_and_shown(setup):
    ch = await _raw_connect(setup.hub.control.port)
    await ch.send("hello", **_hello(protocol="2.0"))
    reply = await ch.recv()
    assert reply["type"] == "reject" and reply["code"] == "protocol"
    node = setup.node("raw")
    assert node.status == "incompatible" and "protocol 2.0" in node.summary()["warnings"][0]
    await ch.close()


async def test_big_frame_before_auth_is_refused(setup):
    ch = await _raw_connect(setup.hub.control.port)
    ch.writer.write(ch.crypto.encrypt(b"x" * 200_000))
    await ch.writer.drain()
    try:
        assert await ch.reader.read() == b""  # connection dropped...
    except ConnectionResetError:
        pass  # ...or reset, since our unread bytes were still in flight
    assert await until(lambda: any(e["kind"] == "auth_failed" for e in setup.hub.recent_events))


async def test_silent_node_times_out(tmp_path, hub_key):
    s = Setup(tmp_path, hub_key, heartbeat_timeout=1.0)
    await s.start()
    try:
        ch = await _raw_connect(s.hub.control.port)
        await ch.send("hello", **_hello())
        assert (await ch.recv())["type"] == "welcome"
        assert s.node("raw").connected
        assert await until(lambda: not s.node("raw").connected, 4)
        leave = [e for e in s.hub.recent_events if e["kind"] == "node_leave"]
        assert leave and leave[-1]["reason"] == "heartbeat timeout"
        await ch.close()
    finally:
        await s.close()


async def test_reconnect_replaces_the_old_connection(setup):
    first = await _raw_connect(setup.hub.control.port)
    await first.send("hello", **_hello())
    await first.recv()
    second = await _raw_connect(setup.hub.control.port)
    await second.send("hello", **_hello())
    assert (await second.recv())["type"] == "welcome"
    assert await first.recv() == {"type": "bye", "reason": "replaced by a new connection"}
    assert setup.node("raw").channel.closed is False
    await second.close()


async def test_pause_resume_and_hub_pause(setup):
    agent = setup.agent()
    setup.run(agent)
    assert await until(lambda: setup.node() is not None and setup.node().connected)
    await agent.set_paused(True)
    assert await until(lambda: setup.node().status == "paused")
    assert setup.node().summary()["vram_lendable"] == 0
    await agent.set_paused(False)
    assert await until(lambda: setup.node().status == "lending")
    setup.hub.registry.set_hub_paused(setup.node(), True)
    assert setup.node().status == "paused"


async def test_rule_changes_reach_the_hub_and_disk(setup):
    agent = setup.agent()
    setup.run(agent)
    assert await until(lambda: setup.node() is not None and setup.node().connected)
    await agent.update_rules(max_vram_percent=50, allow_direct=True)
    assert await until(lambda: setup.node().lending.get("max_vram_percent") == 50)
    assert "max_vram_percent = 50" in agent.config.path.read_text()
    assert await until(lambda: setup.node().summary()["vram_lendable"] == 12 * GIB)


async def test_kick_disconnects_and_agent_waits(setup, monkeypatch):
    monkeypatch.setattr(agent_client, "KICKED_DELAY", 0.5)
    agent = setup.agent()
    setup.run(agent)
    assert await until(lambda: setup.node() is not None and setup.node().connected)
    await setup.hub.control.kick(setup.node())
    assert not setup.node().connected
    assert await until(lambda: any("kicked" in m for _, m in agent.state.events))
    assert await until(lambda: setup.node().connected, 5)  # back after the delay


async def test_network_test_both_ways(setup):
    agent = setup.agent()
    setup.run(agent)
    assert await until(lambda: setup.node() is not None and setup.node().connected)
    result = await setup.hub.control.net_test(setup.node())
    assert result["rtt_ms"] > 0 and result["down_mbps"] > 0 and result["up_mbps"] > 0
    assert setup.node().summary()["link"]["rtt_ms"] == result["rtt_ms"]
    from_agent = await agent.request_net_test()
    assert from_agent["rtt_ms"] > 0 and agent.state.last_net_test == from_agent


async def test_split_requests_are_declined_for_now(setup):
    agent = setup.agent()
    setup.run(agent)
    assert await until(lambda: setup.node() is not None and setup.node().connected)
    await setup.node().channel.send("rpc_start", session_id="s", device="CUDA0", vram_cap_mb=1, cache=True, tunnel={})
    await asyncio.sleep(0.2)
    assert setup.node().connected  # the agent answered rpc_error and stayed


async def test_outside_allowed_networks_is_dropped(tmp_path, hub_key):
    s = Setup(tmp_path, hub_key, allowed_networks=["10.0.0.0/8"])
    await s.start()
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", s.hub.control.port)
        assert await asyncio.wait_for(reader.read(), 2) == b""  # not even the hub key is sent
        writer.close()
    finally:
        await s.close()


async def test_nodes_persist_in_state(setup):
    agent = setup.agent()
    setup.run(agent)
    assert await until(lambda: setup.node() is not None and setup.node().capabilities)
    setup.hub.registry.set_tunnel_mode(setup.node(), "direct")
    await setup.state.flush()
    saved = json.loads(setup.state.path.read_text())["nodes"][agent.state.node_id]
    assert saved["name"] == "desk" and saved["tunnel_mode"] == "direct"
    fresh = NodeRegistry(setup.state, lambda *a, **k: None, 10)
    node = fresh.find("desk")
    assert node.status == "offline" and node.capabilities["gpus"][0]["name"] == "RTX 4090"
    assert node.tunnel_effective == "secure"  # the helper hasn't allowed direct mode


def test_sanitize_bounds_untrusted_reports():
    nasty = {"name": "\x1b]52;c;evil\x07" + "x" * 500, "deep": [[[[[[[[1]]]]]]]], "n": float("nan"), "big": 2**70}
    clean = sanitize({**nasty, **{str(i): i for i in range(100)}})
    assert "\x1b" not in clean["name"] and len(clean["name"]) <= 201
    assert clean["n"] is None and clean["big"] is None
    assert len(clean) == 64


# ----- review regressions ----------------------------------------------------------------


async def test_odd_shaped_reports_cannot_break_status(setup):
    ch = await _raw_connect(setup.hub.control.port)
    await ch.send("hello", **_hello())
    await ch.recv()
    await ch.send(
        "capabilities", capabilities={"gpus": [{"index": [1]}, {"index": 0, "vram_total": "lots"}, 7], "network": 3}
    )
    await ch.send("heartbeat", seq=1, telemetry={"gpus": 5, "cpu_percent": "high"})
    await ch.send("lend_state", state="lending", lending={"gpus": 3, "max_vram_gb": "5", "allow_direct": "yes"})
    await asyncio.sleep(0.2)
    node = setup.node("raw")
    assert node.connected  # still served
    snap = setup.hub.get_snapshot()["nodes"][0]
    assert snap["gpus"] == [
        {
            "index": 0,
            "name": None,
            "backend": None,
            "vram_total": None,
            "vram_used": None,
            "vram_free": 0,
            "util": None,
            "temp": None,
        }
    ]
    assert snap["cpu_percent"] is None and snap["lending"]["max_vram_gb"] == 0.0
    assert node.tunnel_effective == "secure"  # "yes" is not True
    await ch.close()


async def test_slow_key_confirmation_is_not_a_failed_login(tmp_path, hub_key):
    s = Setup(tmp_path, hub_key, handshake_timeout=0.5)
    await s.start()
    try:

        async def slow_owner(addr, fp):
            await asyncio.sleep(1.0)  # longer than the hub's handshake timeout
            return True

        agent = s.agent(confirm=slow_owner)
        s.run(agent)
        assert await until(lambda: agent.state.connection == "connected", 6)
        assert not any(e["kind"] == "auth_failed" for e in s.hub.recent_events)
    finally:
        await s.close()


async def test_stopping_agent_says_goodbye(setup):
    agent = setup.agent()
    task = setup.run(agent)
    assert await until(lambda: setup.node() is not None and setup.node().connected)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert await until(lambda: any(e["kind"] == "node_leave" for e in setup.hub.recent_events))
    leave = [e for e in setup.hub.recent_events if e["kind"] == "node_leave"][-1]
    assert leave["reason"] == "left: agent stopping"


async def test_failing_sensor_keeps_heartbeats_going(setup):
    agent = setup.agent()

    def broken():
        raise RuntimeError("sensor")

    agent.telemetry.sample = broken
    setup.run(agent)
    assert await until(lambda: setup.node() is not None and len(setup.node().history) >= 2, 6)
    assert setup.node().connected


async def test_saving_rules_keeps_one_off_overrides_out_of_agent_toml(setup):
    agent = setup.agent()
    agent.config.agent.name = "temp\n[evil]"
    agent.config.hub.fingerprint = "ab" * 32
    await agent.update_rules(max_vram_gb=4.0)
    text = agent.config.path.read_text()
    assert "max_vram_gb = 4.0" in text and "temp" not in text and "abab" not in text
    from ollamapower.agent.config import AgentConfig, load_agent_config, save_agent_config

    cfg = AgentConfig(config_dir=setup.tmp / "toml")
    cfg.agent.name = 'line\nbreak "quoted"'
    save_agent_config(cfg)
    assert load_agent_config(setup.tmp / "toml").agent.name == 'line\nbreak "quoted"'


def test_active_bans_survive_a_full_table():
    from ollamapower.hub.bans import BanList

    bans = BanList(failures=1, window=60, ban_seconds=600, max_seconds=600)
    bans.MAX_TRACKED = 10
    for i in range(10):
        bans.fail(f"10.0.0.{i}", now=0)
    for i in range(50):
        bans.fail(f"10.0.1.{i}", now=1)  # flood from other addresses
    assert all(bans.banned(f"10.0.0.{i}", now=2) for i in range(10))


def test_control_socket_is_dual_stack():
    if not socket.has_dualstack_ipv6():
        pytest.skip("no dual-stack sockets here")
    from ollamapower.__main__ import _control_socket

    sock = _control_socket("0.0.0.0", 0)
    port = sock.getsockname()[1]
    try:
        for family, host in ((socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")):
            with socket.socket(family) as c:
                c.connect((host, port))
    finally:
        sock.close()
