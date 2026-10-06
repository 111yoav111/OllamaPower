"""Split mode end to end, hermetically: hub + agent over localhost with fake llama.cpp programs.

The real control plane, relay, planner, Ollama handoff and API translation
run; only ``llama-server``/``ggml-rpc-server`` are small fakes (see
``fake_llamacpp.py``) and Ollama is the usual fake. ``test_split_real.py``
repeats the core of this with the real llama.cpp build.
"""

from __future__ import annotations

import asyncio
import json
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest

from ollamapower.admin.ipc import AdminError
from ollamapower.admin.service import Admin
from ollamapower.agent.client import AgentClient
from ollamapower.agent.config import AgentConfig
from ollamapower.core.auth import new_token
from ollamapower.core.encrypt_layer import generate_rsa_keypair
from ollamapower.eventlog import EventLog
from ollamapower.hub.frontdoor.app import create_app
from ollamapower.hub.frontdoor.server import build_server
from ollamapower.hub.runtime import Hub
from ollamapower.llamacpp.install import Pins
from ollamapower.state import StateStore

from .conftest import ThreadedServer, wait_for
from .fake_llamacpp import install_fakes
from .test_control import FakeTelemetry
from .tiny_gguf import write_tiny_llama

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the fake llama.cpp programs are POSIX scripts")

TOKEN = new_token()
AGENT_GPU = "FakeGPU0: Fake GPU (8192 MiB, 8000 MiB free)"


@dataclass
class Cluster:
    url: str
    hub: Hub
    app: Any
    rec: Any
    hub_bin: Path
    agent: AgentClient
    agent_task: Any
    model_path: Path

    def run(self, coro: Any, timeout: float = 10) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self.app.state.loop).result(timeout=timeout)

    def call(self, fn: Any) -> Any:
        async def run() -> Any:
            return fn()

        return self.run(run())

    def admin(self, cmd: str, **args: Any) -> Any:
        return self.run(Admin(self.hub).dispatch(cmd, args))

    @property
    def node(self) -> Any:
        return next(iter(self.hub.registry.nodes.values()), None)

    def server_args(self) -> list[str]:
        return json.loads((self.hub_bin / "args.json").read_text())


def make_cluster(
    tmp_path: Path,
    fake_ollama: Any,
    make_config: Any,
    hub_devices: list[str],
    real_binaries: Path | None = None,
    tunnel: str = "secure",
    **lending: Any,
) -> Any:
    upstream, rec = fake_ollama
    build = Pins.load().build
    if real_binaries is not None:
        hub_bin = agent_bin = real_binaries
    else:
        hub_bin = install_fakes(tmp_path / "hub-bin", hub_devices, build)
        agent_bin = install_fakes(tmp_path / "agent-bin", [AGENT_GPU], build)
    (tmp_path / "models" / "blobs").mkdir(parents=True)
    model_path = write_tiny_llama(tmp_path / "models" / "blobs" / ("sha256-" + "ab" * 32))
    rec.show_models["tiny:latest"] = str(model_path)

    cfg = make_config(upstream.url, llamacpp__path=str(hub_bin))
    cfg.split.keep_alive = 300
    state = StateStore(tmp_path / "state.json")
    hub = Hub(cfg, state, EventLog(cfg.logs_dir, cfg.logging.file_max_bytes, cfg.logging.file_backups))
    app = create_app(hub)
    server = ThreadedServer(server=build_server(hub, app, own_logging=False)).start()
    cfg.hub.port = server.port  # the harness binds its own port; benchmarks call the front door by config

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    key = generate_rsa_keypair()
    loop = app.state.loop
    asyncio.run_coroutine_threadsafe(hub.start_control(key, TOKEN, sock=sock), loop).result(5)

    acfg = AgentConfig(config_dir=tmp_path / "agent")
    acfg.agent.name = "desk"
    acfg.llamacpp.path = str(agent_bin)
    acfg.lending.allow_direct = True
    for k, v in lending.items():
        setattr(acfg.lending, k, v)

    async def yes(addr: str, fp: str) -> bool:
        return True

    agent = AgentClient(acfg, TOKEN, "127.0.0.1", hub.control.port, confirm=yes, telemetry=FakeTelemetry())

    async def start_agent() -> Any:
        return asyncio.ensure_future(agent.run())

    task = asyncio.run_coroutine_threadsafe(start_agent(), loop).result(5)
    c = Cluster(server.url, hub, app, rec, hub_bin, agent, task, model_path)

    def ready() -> bool:
        node = c.node
        return bool(node and node.connected and (node.capabilities.get("llama_cpp") or {}).get("status") == "ready")

    assert wait_for(ready, 15), "the agent never reported a ready llama.cpp"
    if tunnel == "direct":
        c.call(lambda: hub.registry.set_tunnel_mode(c.node, "direct"))

    def close() -> None:
        async def stop() -> None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await hub.stop_control()

        asyncio.run_coroutine_threadsafe(stop(), loop).result(15)
        server.stop()

    return c, close


@pytest.fixture
def cluster(tmp_path, fake_ollama, make_config):
    c, close = make_cluster(tmp_path, fake_ollama, make_config, hub_devices=[])
    yield c
    close()


@pytest.fixture
def client():
    with httpx.Client(timeout=30, trust_env=False) as c:
        yield c


def chat(client: httpx.Client, c: Cluster, **body: Any) -> httpx.Response:
    return client.post(
        c.url + "/api/chat", json={"model": "tiny", "messages": [{"role": "user", "content": "hi"}], **body}
    )


def test_planned_model_runs_split_with_the_helper(cluster, client):
    c = cluster
    preview = c.admin("models.plan", model="tiny", ctx=512)
    assert preview["layers"] == 4 and preview["helpers"][0]["name"] == "desk"

    r = chat(client, c)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-ndjson")
    lines = [json.loads(x) for x in r.text.splitlines()]
    assert "".join(x["message"]["content"] for x in lines) == "Hello from split"
    final = lines[-1]
    assert final["done"] is True and final["done_reason"] == "stop"
    assert final["eval_count"] == 3 and final["eval_duration"] == 150_000_000
    assert final["prompt_eval_count"] == 7 and final["total_duration"] > 0

    # llama-server got exactly the plan: everything on the helper (the hub has no GPU).
    args = c.server_args()
    assert args[args.index("--host") + 1] == "127.0.0.1"
    assert args[args.index("--device") + 1] == "RPC0"
    assert args[args.index("--tensor-split") + 1] == "5"  # 4 blocks + the output layer
    assert args[args.index("-ngl") + 1] == "5"
    assert args[args.index("-c") + 1] == "512"
    assert args[args.index("--fit") + 1] == "off"
    assert args[args.index("-m") + 1] == str(c.model_path)
    # Secure by default: llama-server talks to the hub's local relay, which speaks TLS to the helper's relay.
    rpc_host, rpc_port = args[args.index("--rpc") + 1].rsplit(":", 1)
    lease = c.call(lambda: c.hub.split.session.leases[0])
    assert rpc_host == "127.0.0.1" and int(rpc_port) == lease.relay.port
    assert lease.mode == "secure" and lease.relay.client.ctx is not None
    assert lease.relay.client.port == c.agent.rpc.session.relay_port  # the helper's relay, never rpc-server
    assert c.agent.rpc.session.relay.mode == "secure"
    assert c.agent.rpc.session.port != c.agent.rpc.session.relay_port

    # Ollama gave its GPU up first and never saw the chat.
    assert c.rec.ps_models == []
    assert not [x for x in c.rec.client_requests if x["path"] == "/api/chat"]
    snap = c.call(c.hub.get_snapshot)
    assert snap["split"]["session"]["state"] == "ready"
    assert any(m["engine"] == "split" and m["name"] == "tiny:latest" for m in snap["ollama"]["models"])
    assert snap["nodes"][0]["role"] == "split"
    # The speed tap read the translated answer (recorded just after the last byte went out).
    assert wait_for(lambda: "tiny:latest" in c.call(lambda: c.hub.last_generation))
    assert c.call(lambda: c.hub.last_generation["tiny:latest"].gen_tokens) == 3
    # llama-server allows any web origin, so it must refuse anyone without the hub's key.
    port = c.call(lambda: c.hub.split.session.port)
    assert client.post(f"http://127.0.0.1:{port}/v1/chat/completions", json={"messages": []}).status_code == 401
    lent = c.agent.rpc.session.info()
    assert lent["device"] == "FakeGPU0" and lent["model"] == "tiny:latest"


def test_non_streaming_generate_tools_and_ps(cluster, client):
    c = cluster
    c.admin("models.plan", model="tiny", ctx=512)
    r = chat(client, c, stream=False)
    body = r.json()
    assert r.headers["content-type"].startswith("application/json")
    assert body["message"]["content"] == "Hello from split" and body["done"] is True

    r = client.post(c.url + "/api/generate", json={"model": "tiny", "prompt": "hi", "system": "be brief"})
    lines = [json.loads(x) for x in r.text.splitlines()]
    assert "".join(x.get("response", "") for x in lines) == "Hello from split"
    sent = json.loads((c.hub_bin / "last_request.json").read_text())
    assert sent["path"] == "/v1/chat/completions"
    assert [m["role"] for m in sent["body"]["messages"]] == ["system", "user"]

    tools = [{"type": "function", "function": {"name": "get", "parameters": {"type": "object"}}}]
    r = chat(client, c, stream=False, tools=tools, options={"temperature": 0.1, "num_predict": 9})
    assert r.json()["message"]["tool_calls"] == [
        {"id": "fake1", "function": {"index": 0, "name": "get", "arguments": {"a": 1}}}
    ]
    sent = json.loads((c.hub_bin / "last_request.json").read_text())["body"]
    assert sent["temperature"] == 0.1 and sent["max_tokens"] == 9 and sent["tools"] == tools

    r = chat(client, c, logprobs=True, top_logprobs=2)
    lines = [json.loads(x) for x in r.text.splitlines()]
    assert [x["logprobs"][0]["token"] for x in lines if "logprobs" in x] == ["Hello", " from", " split"]
    assert len(lines[0]["logprobs"][0]["top_logprobs"]) == 2
    sent = json.loads((c.hub_bin / "last_request.json").read_text())["body"]
    assert sent["logprobs"] is True and sent["top_logprobs"] == 2

    ps = client.get(c.url + "/api/ps").json()["models"]
    assert [(m["name"], m["ollamapower"]["engine"]) for m in ps] == [("tiny:latest", "split")]


def test_modelfile_system_parameters_and_messages_apply(cluster, client):
    c = cluster
    c.rec.show_extra["tiny:latest"] = {
        "system": "be terse",
        "parameters": 'stop                           "<end>"\ntemperature                    0.6',
        "messages": [{"role": "user", "content": "m1"}, {"role": "assistant", "content": "m2"}],
    }
    c.admin("models.plan", model="tiny", ctx=512)
    assert chat(client, c, options={"temperature": 0.2}).status_code == 200
    sent = json.loads((c.hub_bin / "last_request.json").read_text())["body"]
    assert [m["content"] for m in sent["messages"]] == ["be terse", "m1", "m2", "hi"]
    assert sent["temperature"] == 0.2 and sent["stop"] == ["<end>"]


def test_a_prompt_longer_than_the_context_drops_the_oldest_messages(cluster, client):
    c = cluster
    c.admin("models.plan", model="tiny", ctx=512)
    assert chat(client, c).status_code == 200
    assert not (c.hub_bin / "calls.json").exists()  # a prompt that fits costs no extra request

    old = [{"role": r, "content": f"{i}" * 100} for i, r in enumerate(["user", "assistant"] * 4)]
    messages = [{"role": "system", "content": "sys"}, *old, {"role": "user", "content": "latest"}]
    r = chat(client, c, messages=messages)
    assert r.status_code == 200, r.text
    assert "".join(json.loads(x)["message"]["content"] for x in r.text.splitlines()) == "Hello from split"
    sent = json.loads((c.hub_bin / "last_request.json").read_text())["body"]["messages"]
    # The fake template makes ~110 tokens of each old message: the system prompt, the last four
    # old messages and the latest one fit in 512 (474 tokens), one more wouldn't.
    assert [m["content"] for m in sent] == ["sys", *(f"{i}" * 100 for i in range(4, 8)), "latest"]

    r = chat(client, c, messages=messages, truncate=False)
    assert r.status_code == 400 and "exceeds the available context size" in r.json()["error"]
    r = chat(client, c, messages=[{"role": "user", "content": "x" * 600}, {"role": "user", "content": "y" * 600}])
    assert (
        r.status_code == 400 and "exceeds the available context size" in r.json()["error"]
    )  # even the last is too long


def test_generate_context_goes_out_and_comes_back(cluster, client):
    c = cluster
    c.admin("models.plan", model="tiny", ctx=512)
    r = client.post(c.url + "/api/generate", json={"model": "tiny", "prompt": "hi", "stream": False})
    first = r.json()
    prompt = "<user>hi\n<assistant>"  # the fake's chat template; one token per character
    assert first["response"] == "Hello from split"
    assert first["context"] == [ord(ch) for ch in prompt + "Hello from split"]
    assert json.loads((c.hub_bin / "calls.json").read_text()) == ["/apply-template", "/tokenize"]

    r = client.post(
        c.url + "/api/generate", json={"model": "tiny", "prompt": "more", "context": first["context"], "stream": True}
    )
    lines = [json.loads(x) for x in r.text.splitlines()]
    assert "".join(x["response"] for x in lines) == "Hello from split"
    sent = json.loads((c.hub_bin / "last_request.json").read_text())
    assert sent["path"] == "/v1/completions" and "messages" not in sent["body"]
    second = "<user>more\n<assistant>"
    assert sent["body"]["prompt"] == prompt + "Hello from split" + second
    assert lines[-1]["done"] is True
    assert lines[-1]["context"] == [ord(ch) for ch in prompt + "Hello from split" + second + "Hello from split"]

    r = client.post(c.url + "/api/generate", json={"model": "tiny", "prompt": "x", "context": ["a"]})
    assert r.status_code == 400 and "context must be a list of token ids" in r.json()["error"]


def test_openai_requests_pass_through_to_llama_server(cluster, client):
    c = cluster
    c.admin("models.plan", model="tiny", ctx=512)
    r = client.post(
        c.url + "/v1/chat/completions", json={"model": "tiny", "messages": [{"role": "user", "content": "x"}]}
    )
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.text.rstrip().endswith("data: [DONE]")
    assert json.loads((c.hub_bin / "last_request.json").read_text())["body"]["model"] == "tiny"


def test_images_and_upstream_errors_are_clear(cluster, client):
    c = cluster
    c.admin("models.plan", model="tiny", ctx=512)
    r = chat(client, c, messages=[{"role": "user", "content": "x", "images": ["aGk="]}])
    assert r.status_code == 400 and "images are not supported in split mode" in r.json()["error"]
    r = chat(client, c, messages=[{"role": "user", "content": "fail"}])
    assert r.status_code == 400 and r.json() == {"error": "bad request from fake"}


def test_load_and_unload_requests(cluster, client):
    c = cluster
    c.admin("models.plan", model="tiny", ctx=512)
    r = client.post(c.url + "/api/generate", json={"model": "tiny"})
    assert r.json()["done_reason"] == "load"
    assert c.call(lambda: c.hub.split.is_running("tiny"))
    r = client.post(c.url + "/api/generate", json={"model": "tiny", "keep_alive": 0})
    assert r.json()["done_reason"] == "unload"
    assert not c.call(lambda: c.hub.split.is_running("tiny"))
    assert wait_for(lambda: c.agent.rpc.session is None, 5)  # the helper was released


def test_unplan_stops_the_session_and_ollama_serves_again(cluster, client):
    c = cluster
    c.admin("models.plan", model="tiny", ctx=512)
    assert chat(client, c).status_code == 200
    assert c.admin("models.unplan", model="tiny:latest")["stopped"] is True
    assert wait_for(lambda: c.agent.rpc.session is None, 5)
    assert c.node.split_model == ""
    chat(client, c)
    assert c.rec.last["path"] == "/api/chat"  # Ollama again
    with pytest.raises(AdminError):
        c.admin("models.unplan", model="tiny")


def test_paused_helper_means_ollama_serves_and_says_why(cluster, client):
    c = cluster
    c.admin("models.plan", model="tiny", ctx=512)
    c.run(c.agent.set_paused(True))
    assert wait_for(lambda: c.node.status == "paused", 5)
    r = chat(client, c)
    assert r.status_code == 200 and c.rec.last["path"] == "/api/chat"
    why = c.admin("models.show", model="tiny")["why"]
    assert "no helper can take part now" in why and "desk is paused" in why
    # A helper being away on purpose is not a failure: nothing is degraded.
    assert c.call(lambda: c.hub.plans.record("tiny")["degraded"]) is None
    assert c.call(lambda: c.hub.plans.route("tiny")) is None


def test_helper_owner_pausing_mid_session_ends_it(cluster, client):
    c = cluster
    c.admin("models.plan", model="tiny", ctx=512)
    assert chat(client, c).status_code == 200
    c.run(c.agent.set_paused(True))
    assert wait_for(lambda: not c.call(lambda: c.hub.split.is_running("tiny")), 10)
    assert c.agent.rpc.session is None
    assert any("paused by the helper's owner" in w for w in c.call(c.hub.warnings))


def test_helper_leaving_mid_session_ends_it(cluster, client):
    c = cluster
    c.admin("models.plan", model="tiny", ctx=512)
    assert chat(client, c).status_code == 200
    c.run(c.hub.control.kick(c.node))
    assert wait_for(lambda: not c.call(lambda: c.hub.split.is_running("tiny")), 10)
    assert any("desk disconnected" in w for w in c.call(c.hub.warnings))


def test_relay_only_accepts_the_hub(cluster, client):
    c = cluster
    c.admin("models.plan", model="tiny", ctx=512)
    assert chat(client, c).status_code == 200
    session = c.agent.rpc.session
    session.relay.hub_ip = "10.9.9.9"  # pretend the hub is elsewhere: a local connection must now be refused

    async def probe() -> bytes:
        reader, writer = await asyncio.open_connection("127.0.0.1", session.relay_port)
        writer.write(b"ping")
        try:
            return await asyncio.wait_for(reader.read(4), 2)
        except ConnectionResetError:
            return b""
        finally:
            writer.close()

    assert c.run(probe()) == b""


def test_secure_relay_refuses_a_wrong_certificate_and_plain_tcp(cluster, client):
    from ollamapower.core.tunnel import TunnelIdentity
    from ollamapower.hub.tunnel import TunnelClient

    c = cluster
    c.admin("models.plan", model="tiny", ctx=512)
    assert chat(client, c).status_code == 200
    session = c.agent.rpc.session
    helper_cert_pem = c.call(lambda: c.hub.split.session.leases[0].endpoint["cert"])

    async def impostor() -> bytes:
        # Right address, a certificate the hub never sent: the helper must not relay anything.
        evil = TunnelClient("127.0.0.1", session.relay_port, TunnelIdentity.generate("evil"), helper_cert_pem)
        reader, writer = await evil.open()
        writer.write(b"ping")
        try:
            return await asyncio.wait_for(reader.read(4), 2)
        except (ConnectionResetError, OSError):
            return b""
        finally:
            writer.close()

    async def plaintext() -> bytes:
        reader, writer = await asyncio.open_connection("127.0.0.1", session.relay_port)
        writer.write(b"ping")
        try:
            return await asyncio.wait_for(reader.read(4), 3)
        except (ConnectionResetError, OSError, TimeoutError):
            return b""
        finally:
            writer.close()

    assert c.run(impostor()) == b""
    assert c.run(plaintext()) != b"ping"
    assert chat(client, c).status_code == 200  # the real hub still gets through


def test_direct_mode_needs_both_owners(tmp_path, fake_ollama, make_config, client):
    c, close = make_cluster(tmp_path, fake_ollama, make_config, hub_devices=[], tunnel="direct", allow_direct=False)
    try:
        c.admin("models.plan", model="tiny", ctx=512)
        assert chat(client, c).status_code == 200
        assert (
            c.agent.rpc.session.relay.mode == "secure"
        )  # the hub asked for direct; the helper's owner didn't allow it
        assert any("direct mode requested" in w for w in c.call(c.hub.warnings))
    finally:
        close()


def test_direct_mode_end_to_end(tmp_path, fake_ollama, make_config, client):
    c, close = make_cluster(tmp_path, fake_ollama, make_config, hub_devices=[], tunnel="direct")
    try:
        c.admin("models.plan", model="tiny", ctx=512)
        assert chat(client, c).status_code == 200
        lease = c.call(lambda: c.hub.split.session.leases[0])
        assert lease.mode == "direct" and lease.relay is None
        rpc = c.server_args()[c.server_args().index("--rpc") + 1]
        assert rpc == f"127.0.0.1:{c.agent.rpc.session.relay_port}"  # straight to the helper's relay
        assert c.agent.rpc.session.relay.mode == "direct"
        assert any("unencrypted" in w for w in c.call(c.hub.warnings))
    finally:
        close()


def test_tunnel_test_measures_both_modes(cluster):
    c = cluster
    r = c.admin("nodes.tunnel_test", ref="desk")
    assert r["secure"]["mode"] == "secure" and r["secure"]["mbps"] > 0 and r["secure"]["rtt_ms"] > 0
    assert r["direct"]["mode"] == "direct" and "overhead_pct" in r
    assert c.node.tunnel_test["secure"]["mbps"] == r["secure"]["mbps"]
    assert wait_for(lambda: not c.agent.rpc.benches, 5)  # the helper closed the test relay


def test_tunnel_test_without_direct_permission_measures_secure_only(tmp_path, fake_ollama, make_config):
    c, close = make_cluster(tmp_path, fake_ollama, make_config, hub_devices=[], allow_direct=False)
    try:
        r = c.admin("nodes.tunnel_test", ref="desk")
        assert "secure" in r and "direct" not in r and "overhead_pct" not in r
    finally:
        close()


def test_model_that_fits_on_the_hub_stays_with_ollama(tmp_path, fake_ollama, make_config, client):
    c, close = make_cluster(
        tmp_path, fake_ollama, make_config, hub_devices=["Vulkan0: Big GPU (24576 MiB, 24000 MiB free)"]
    )
    try:
        c.admin("models.plan", model="tiny", ctx=512)
        chat(client, c)
        assert c.rec.last["path"] == "/api/chat"
        assert any("fits on the hub's GPU alone" in w for w in c.call(c.hub.warnings))
        assert c.agent.rpc.session is None  # no helper was even asked
    finally:
        close()


def test_failed_llama_server_start_falls_back_and_releases_helpers(cluster, client):
    c = cluster
    (c.hub_bin / "fail_start").write_text("1")
    c.admin("models.plan", model="tiny", ctx=512)
    r = chat(client, c)
    assert r.status_code == 200 and c.rec.last["path"] == "/api/chat"
    assert any("llama-server exited with code 3" in w for w in c.call(c.hub.warnings))
    assert wait_for(lambda: c.agent.rpc.session is None, 5)


def test_plan_refuses_unknown_and_bad_input(cluster):
    with pytest.raises(AdminError, match="not installed"):
        cluster.admin("models.plan", model="missing", ctx=512)
    with pytest.raises(AdminError, match="--ctx"):
        cluster.admin("models.plan", model="tiny", ctx=10)


def test_idle_session_stops_after_keep_alive(cluster, client, monkeypatch):
    from ollamapower.hub.engines import split as split_mod

    c = cluster
    c.hub.config.split.keep_alive = 0.2
    monkeypatch.setattr(split_mod, "IDLE_CHECK", 0.1)
    # The idle loop started before the patch; restart it so it picks up the short interval.
    c.admin("models.plan", model="tiny", ctx=512)
    assert chat(client, c).status_code == 200

    async def kick_idle_loop() -> None:
        asyncio.ensure_future(c.hub.split.run())

    c.run(kick_idle_loop())
    assert wait_for(lambda: not c.call(lambda: c.hub.split.is_running("tiny")), 5)
    assert wait_for(lambda: c.agent.rpc.session is None, 5)
    time.sleep(0.05)


def test_tunnel_test_refuses_an_older_agent(cluster):
    c = cluster
    c.node.protocol = "1.0"
    with pytest.raises(AdminError, match="older agent"):
        c.admin("nodes.tunnel_test", ref="desk")


def test_web_pages_ollama_would_refuse_cannot_use_split_mode(cluster, client):
    c = cluster
    c.admin("models.plan", model="tiny", ctx=512)
    body = {"model": "tiny", "messages": [{"role": "user", "content": "hi"}]}
    r = client.post(c.url + "/api/chat", json=body, headers={"Origin": "https://evil.example"})
    assert r.status_code == 403 and r.json() == {"error": "origin not allowed"}
    assert not c.call(lambda: c.hub.split.holds("tiny"))  # nothing was started for it
    r = client.post(c.url + "/api/chat", json=body, headers={"Origin": "http://localhost:3000"})
    assert r.status_code == 200 and r.headers["access-control-allow-origin"] == "http://localhost:3000"
    r = client.post(c.url + "/v1/chat/completions", json=body)
    assert "access-control-allow-origin" not in r.headers  # llama-server's "allow everyone" never leaks out


def test_any_inference_for_a_split_model_hands_the_gpu_back_first(cluster, client):
    c = cluster
    c.admin("models.plan", model="tiny", ctx=512)
    assert chat(client, c).status_code == 200 and c.call(lambda: c.hub.split.holds("tiny"))
    r = client.post(c.url + "/api/embed", json={"model": "tiny", "input": "x"})
    assert r.status_code == 200 and c.rec.last["path"] == "/api/embed"
    assert not c.call(lambda: c.hub.split.holds("tiny"))  # Ollama didn't load it next to llama-server


async def test_hub_shutdown_stops_the_session_even_with_background_work_pending():
    import types

    from ollamapower.config import Config
    from ollamapower.hub.engines.split import SplitEngine, SplitSession

    hub = types.SimpleNamespace(config=Config(), streams={}, event=lambda *a, **k: None)
    engine = SplitEngine(hub)
    session = SplitSession(model="m", ctx=0, state="ready")
    engine.session = session
    engine._spawn(asyncio.sleep(3600))  # e.g. a drain waiting for an answer to finish
    run = asyncio.ensure_future(engine.run())
    await asyncio.sleep(0)
    run.cancel()
    await asyncio.gather(run, return_exceptions=True)
    assert session.state == "stopped" and engine.session is None  # llama-server and helpers released
