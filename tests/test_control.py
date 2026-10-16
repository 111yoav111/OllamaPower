"""Hub control plane <-> agent, for real, over localhost."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
import time
from typing import Any

import pytest

from ollamapower.agent import client as agent_client
from ollamapower.agent.client import AgentClient, AgentStopped
from ollamapower.agent.config import AgentConfig
from ollamapower.agent.pinning import fingerprint_lines, format_fingerprint
from ollamapower.config import Config
from ollamapower.core import protocol
from ollamapower.core.auth import new_token
from ollamapower.core.channel import Channel
from ollamapower.core.encrypt_layer import (
    SessionCrypto,
    frame_bytes,
    generate_rsa_keypair,
    public_key_fingerprint,
    read_frame,
    rsa_encrypt_aes_key,
    rsa_key_from_pem,
    rsa_public_key_to_pem,
)
from ollamapower.hub.nodes import GIB, NodeRegistry, sanitize
from ollamapower.hub.runtime import Hub
from ollamapower.state import StateStore

TOKEN = new_token()


class FakeTelemetry:
    def __init__(self, vram_used: int = 4 * GIB, util: int = 30) -> None:
        self.vram_used = vram_used
        self.util = util

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
            "gpus": [{"index": 0, "vram_total": 24 * GIB, "vram_used": self.vram_used, "util": self.util, "temp": 55}],
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
        cfg.llamacpp.auto_install = False  # never download in tests
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
    with pytest.raises(AgentStopped, match="KEY CHANGED") as stopped:
        await asyncio.wait_for(agent.run(), 5)
    # Both keys in full, so the owner can compare them with what the hub prints.
    assert format_fingerprint("ab" * 32) in str(stopped.value)
    assert format_fingerprint(public_key_fingerprint(setup.key.public_key())) in str(stopped.value)
    await asyncio.sleep(0.1)
    assert setup.node() is None
    assert not any(e["kind"] == "auth_failed" for e in setup.hub.recent_events)  # the hub never saw a token


def test_fingerprint_lines_keep_every_pair():
    fp = "0123456789abcdef" * 4
    lines = fingerprint_lines(fp)
    assert len(lines) == 2 and all(len(line) == 47 for line in lines)
    assert ":".join(lines) == format_fingerprint(fp)
    assert fingerprint_lines("ab") == ["AB"] and fingerprint_lines("") == []


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
            bad = s.agent(token=new_token())
            # Known key: no hang-up to ask the owner first, which would count as a (quiet) failure too.
            bad.config.hub.fingerprint = public_key_fingerprint(hub_key.public_key())
            with pytest.raises(AgentStopped, match="wrong cluster token"):
                await asyncio.wait_for(bad.run(), 5)
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


async def _raw_connect(port: int, hub_fp: str | None = None, source: str = "127.0.0.1") -> Channel:
    reader, writer = await asyncio.open_connection("127.0.0.1", port, local_addr=(source, 0))
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
    await ch.send("hello", **_hello(protocol="3.0"))
    reply = await ch.recv()
    assert reply["type"] == "reject" and reply["code"] == "protocol"
    node = setup.node("raw")
    assert node.status == "incompatible" and "protocol 3.0" in node.summary()["warnings"][0]
    await ch.close()


class _Recorder:
    """A writer wrapper that keeps a copy of everything sent: what a sniffer on the LAN sees."""

    def __init__(self, writer: asyncio.StreamWriter) -> None:
        self.writer, self.sent = writer, bytearray()

    def write(self, data: bytes) -> None:
        self.sent += data
        self.writer.write(data)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.writer, name)


async def test_a_recorded_session_cannot_be_replayed(setup):
    port = setup.hub.control.port
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    recorder = _Recorder(writer)
    crypto = await SessionCrypto.node_handshake(reader, recorder)  # type: ignore[arg-type]
    ch = Channel(reader, recorder, crypto)  # type: ignore[arg-type]
    await ch.send("hello", **_hello())
    assert (await ch.recv())["type"] == "welcome"
    await ch.send("lend_state", state="paused", reason="owner")
    await ch.close("done")
    assert await until(lambda: setup.node("raw") is not None and not setup.node("raw").connected)
    joins = sum(e["kind"] == "node_join" for e in setup.hub.recent_events)

    # Replay every byte the helper sent, after reading the hub's (fresh) key blob.
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    await read_frame(reader)
    writer.write(bytes(recorder.sent))
    await writer.drain()
    try:
        assert await asyncio.wait_for(reader.read(), 3) == b""  # dropped, nothing answered
    except ConnectionResetError:
        pass
    writer.close()
    assert await until(lambda: any(e["kind"] == "auth_failed" for e in setup.hub.recent_events))
    failed = [e for e in setup.hub.recent_events if e["kind"] == "auth_failed"][-1]
    assert failed["reason"] == "bad handshake (CryptoError)"
    assert sum(e["kind"] == "node_join" for e in setup.hub.recent_events) == joins
    assert not setup.node("raw").connected


async def _protocol1_connect(port: int) -> Channel:
    """What a protocol-1 agent does: read the PEM key, send its key RSA-wrapped, use it as is."""
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    hub_key = rsa_key_from_pem(await read_frame(reader))  # the PEM parser skips the v2 nonce after the key
    aes_key = os.urandom(32)
    writer.write(frame_bytes(rsa_encrypt_aes_key(hub_key, aes_key)))
    await writer.drain()
    return Channel(reader, writer, SessionCrypto(aes_key, "node"))


async def test_protocol1_agent_is_told_to_update_and_never_admitted(setup):
    ch = await _protocol1_connect(setup.hub.control.port)
    await ch.send("hello", **_hello(protocol="1.1"))
    reply = await asyncio.wait_for(ch.recv(), 3)
    assert reply["type"] == "reject" and reply["code"] == "protocol"
    assert "install OllamaPower" in reply["reason"]
    await ch.close()
    assert setup.node("raw").status == "incompatible"
    # Even claiming the current protocol, the old handshake is never enough for a session.
    ch = await _protocol1_connect(setup.hub.control.port)
    await ch.send("hello", **_hello(node_id="raw-node-0002", name="raw2"))
    reply = await asyncio.wait_for(ch.recv(), 3)
    assert reply["type"] == "reject" and reply["code"] == "protocol"
    await ch.close()
    assert not setup.node("raw2").connected
    # A wrong token over the old handshake is still a failed login.
    ch = await _protocol1_connect(setup.hub.control.port)
    await ch.send("hello", **_hello(protocol="1.1", token=new_token()))
    assert (await asyncio.wait_for(ch.recv(), 3))["code"] == "bad_token"
    await ch.close()
    assert any(e["kind"] == "auth_failed" for e in setup.hub.recent_events)


async def test_protocol1_session_cannot_flag_a_connected_helper(setup):
    agent = setup.agent()
    setup.run(agent)
    assert await until(lambda: setup.node() is not None and setup.node().connected)
    ch = await _protocol1_connect(setup.hub.control.port)  # e.g. a session recorded before the upgrade
    await ch.send("hello", **_hello(protocol="1.1", node_id=agent.state.node_id, name="desk"))
    assert (await asyncio.wait_for(ch.recv(), 3))["code"] == "protocol"
    await ch.close()
    assert setup.node().status == "lending" and setup.node().connected


async def test_agent_explains_a_protocol1_hub(tmp_path, hub_key):
    async def old_hub(reader, writer):  # sends the bare PEM key, as protocol 1 did
        writer.write(frame_bytes(rsa_public_key_to_pem(hub_key)))
        await writer.drain()
        await reader.read()

    server = await asyncio.start_server(old_hub, "127.0.0.1", 0)
    try:
        cfg = AgentConfig(config_dir=tmp_path / "a")
        cfg.llamacpp.auto_install = False
        agent = AgentClient(cfg, TOKEN, "127.0.0.1", server.sockets[0].getsockname()[1], telemetry=FakeTelemetry())
        task = asyncio.ensure_future(agent.run())
        assert await until(lambda: "update OllamaPower on the hub" in agent.state.error)
        assert agent.state.connection != "connected"
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    finally:
        server.close()


async def test_big_frame_before_auth_is_refused(setup):
    ch = await _raw_connect(setup.hub.control.port)
    ch.writer.write(ch.crypto.encrypt(b"x" * 200_000))
    await ch.writer.drain()
    try:
        assert await ch.reader.read() == b""  # connection dropped...
    except ConnectionResetError:
        pass  # ...or reset, since our unread bytes were still in flight
    assert await until(lambda: any(e["kind"] == "auth_failed" for e in setup.hub.recent_events))


async def test_bomb_hello_is_refused_quickly_and_counted(setup):
    import zstandard as zstd

    from ollamapower.core.packets import PACKET_FLAGS, PayloadKind, TaskPacket

    pkt = TaskPacket.from_payload({}, PACKET_FLAGS.message, kind=PayloadKind.message)
    pkt.raw_payload = zstd.ZstdCompressor(level=19).compress(b"\xdd" + (10**7).to_bytes(4, "big") + b"\x80" * 10**7)
    ch = await _raw_connect(setup.hub.control.port)
    start = time.monotonic()
    ch.writer.write(ch.crypto.encrypt(pkt.to_bytes()))
    await ch.writer.drain()
    try:
        assert await asyncio.wait_for(ch.reader.read(), 2) == b""
    except ConnectionResetError:
        pass
    assert time.monotonic() - start < 1.0  # was ~3 s of a blocked event loop and 720 MB
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


async def test_a_close_cancelled_during_its_goodbye_still_closes_the_socket(setup):
    peer = await _raw_connect(setup.hub.control.port)
    await peer.send("hello", **_hello())
    assert (await peer.recv())["type"] == "welcome"
    channel = setup.node("raw").channel
    # The peer doesn't hang up after the bye, so close() waits; the hub shutting down
    # cancels that wait (ControlServer.close cancels its pending close tasks).
    closing = asyncio.ensure_future(channel.close("hub shutting down"))
    await asyncio.sleep(0.1)
    closing.cancel()
    await asyncio.gather(closing, return_exceptions=True)
    assert channel.writer.is_closing()  # was left open: closed=True, so nobody closed it after
    await peer.close()


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


async def test_split_request_without_llamacpp_is_refused_with_a_reason(setup):
    agent = setup.agent()
    setup.run(agent)
    assert await until(lambda: setup.node() is not None and setup.node().connected)
    got = []
    setup.hub.control.on_rpc = lambda node, m: got.append(m)
    await setup.node().channel.send(
        "rpc_start", session_id="session-1", device="", vram_cap_mb=0, cache=True, tunnel={"mode": "direct"}
    )
    assert await until(lambda: got)
    assert got[0]["type"] == "rpc_error" and "not ready" in got[0]["error"]
    assert setup.node().connected  # the agent answered and stayed


async def test_stopping_the_agent_ends_its_background_work(setup):
    agent = setup.agent()
    task = setup.run(agent)
    assert await until(lambda: setup.node() is not None and setup.node().connected)
    stopping = asyncio.Event()

    async def slow_stop(session_id=None, reason=""):
        if session_id is not None:
            stopping.set()
            await asyncio.Event().wait()  # a stop that never finishes on its own
        return False

    agent.rpc.stop = slow_stop
    await setup.node().channel.send("rpc_stop", session_id="session-1")
    await asyncio.wait_for(stopping.wait(), 5)
    background = set(agent._background)
    assert background
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert all(t.done() for t in background)  # nothing the hub asked for outlives the agent


async def _tunnel_test_started(setup, agent) -> tuple:
    got = []
    setup.hub.control.on_rpc = lambda node, m: got.append(m)
    await setup.node().channel.send(
        "rpc_start",
        session_id="bench-0001",
        device="",
        vram_cap_mb=0,
        cache=False,
        tunnel={"mode": "direct"},
        purpose="bench",
    )
    assert await until(lambda: got)
    assert got[0]["type"] == "rpc_ready"
    relay, target, _ = agent.rpc.benches["bench-0001"]
    return relay, target


@pytest.mark.parametrize("how", ["agent stops", "hub connection lost"])
async def test_tunnel_tests_end_with_the_agent_or_the_connection(setup, how):
    agent = setup.agent(allow_direct=True)
    task = setup.run(agent)
    assert await until(lambda: setup.node() is not None and setup.node().connected)
    relay, target = await _tunnel_test_started(setup, agent)
    if how == "agent stops":
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    else:
        await setup.hub.control.kick(setup.node())
        assert await until(lambda: agent.state.connection != "connected")
    assert await until(lambda: not agent.rpc.benches, 2)  # not left listening until the two-minute expiry
    assert relay.closed and not target.server.is_serving()


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


async def _stall(port: int, source: str = "127.0.0.1") -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """A peer that connects and then never answers the hub's key."""
    return await asyncio.open_connection("127.0.0.1", port, local_addr=(source, 0))


async def _closed_at_once(reader: asyncio.StreamReader) -> bool:
    """Did the hub hang up without sending anything (not even its key)?"""
    try:
        return await asyncio.wait_for(reader.read(), 2) == b""
    except ConnectionResetError:
        return True


async def test_one_ip_cannot_hold_every_handshake_slot(tmp_path, hub_key):
    s = Setup(tmp_path, hub_key, max_pending_per_ip=2, ban_failures=100, handshake_timeout=5.0)
    port = await s.start()
    try:
        stalled = [await _stall(port) for _ in range(2)]
        assert await until(lambda: s.hub.control._pending == 2)
        reader, writer = await _stall(port)
        assert await _closed_at_once(reader)  # over the per-IP limit: dropped before any work
        writer.close()
        assert await until(lambda: any(e["kind"] == "auth_failed" for e in s.hub.recent_events))
        if sys.platform.startswith("linux"):  # all of 127/8 is local there: a second source address
            ch = await _raw_connect(port, source="127.0.0.2")
            await ch.send("hello", **_hello())
            assert (await asyncio.wait_for(ch.recv(), 3))["type"] == "welcome"
            await ch.close()
        for _, w in stalled:
            w.close()
        assert await until(lambda: s.hub.control._pending == 0 and not s.hub.control._pending_by_ip)
        agent = s.agent()
        s.run(agent)
        assert await until(lambda: agent.state.connection == "connected")
    finally:
        await s.close()


async def test_abandoned_and_timed_out_handshakes_count_towards_a_ban(tmp_path, hub_key):
    s = Setup(tmp_path, hub_key, ban_failures=3, handshake_timeout=0.3)
    port = await s.start()
    try:
        for _ in range(2):  # connect and hang up: no token was ever tried
            _, writer = await _stall(port)
            writer.close()
            await asyncio.sleep(0.05)
        assert await until(lambda: s.hub.control._pending == 0)
        assert not s.hub.control.bans.active()
        assert not any(e["kind"] == "auth_failed" for e in s.hub.recent_events)  # quiet, but counted
        reader, writer = await _stall(port)  # hold the slot past the deadline
        assert await until(lambda: "127.0.0.1" in s.hub.control.bans.active(), 3)
        writer.close()
        reader, writer = await _stall(port)
        assert await _closed_at_once(reader)  # banned: dropped before the handshake
        writer.close()
    finally:
        await s.close()


async def test_a_successful_login_clears_abandoned_handshakes(tmp_path, hub_key):
    s = Setup(tmp_path, hub_key, ban_failures=2)
    port = await s.start()
    try:
        _, writer = await _stall(port)
        writer.close()
        assert await until(lambda: s.hub.control._pending == 0)
        agent = s.agent()
        agent.config.hub.fingerprint = public_key_fingerprint(hub_key.public_key())
        task = s.run(agent)
        assert await until(lambda: agent.state.connection == "connected")
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        _, writer = await _stall(port)  # without the reset this would be the second failure
        writer.close()
        assert await until(lambda: s.hub.control._pending == 0)
        await asyncio.sleep(0.1)
        assert not s.hub.control.bans.active()
    finally:
        await s.close()


def test_success_forgets_failures_but_not_bans():
    from ollamapower.hub.bans import BanList

    bans = BanList(failures=2, window=60, ban_seconds=10, max_seconds=100)
    bans.fail("10.0.0.1", now=0)
    bans.succeeded("10.0.0.1", now=1)
    assert bans.fail("10.0.0.1", now=2) is None  # the count started again
    assert bans.fail("10.0.0.1", now=3) == 10
    bans.succeeded("10.0.0.1", now=4)
    assert bans.banned("10.0.0.1", now=5)  # an active ban stays
    bans.fail("10.0.0.1", now=20)
    assert bans.fail("10.0.0.1", now=21) == 20  # and so does the repeat-offence doubling


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


async def test_reconnect_fails_what_waited_on_the_old_connection(setup):
    from ollamapower.hub.engines.split import SplitUnavailable

    first = await _raw_connect(setup.hub.control.port)
    await first.send("hello", **_hello())
    assert (await first.recv())["type"] == "welcome"
    node = setup.node("raw")
    bench = asyncio.ensure_future(setup.hub.split.bench(node, "direct"))
    assert (await first.recv())["type"] == "rpc_start"
    # The helper reconnects (its old socket half-open): it stopped everything it had
    # started on the old connection, so nothing may keep waiting for an answer there.
    second = await _raw_connect(setup.hub.control.port)
    await second.send("hello", **_hello())
    assert (await second.recv())["type"] == "welcome"
    with pytest.raises(SplitUnavailable, match="disconnected"):
        await asyncio.wait_for(bench, 3)
    await first.close()
    await second.close()


def test_ipv4_helpers_on_the_dual_stack_listener_get_their_plain_address():
    from ollamapower.hub.control_server import plain_ip

    assert plain_ip("::ffff:192.168.1.36") == "192.168.1.36"
    assert plain_ip("192.168.1.36") == "192.168.1.36"
    assert plain_ip("fe80::1%eth0") == "fe80::1%eth0"
    assert plain_ip("fd00::1") == "fd00::1"
