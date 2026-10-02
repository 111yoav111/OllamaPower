"""The front door against a fake Ollama: passthrough, streaming, disconnects, errors."""

from __future__ import annotations

import json
import threading
import time

import httpx
import pytest

from .conftest import free_port, wait_for


def chat(fd, client, **body):
    body.setdefault("model", "small:1b")
    body.setdefault("messages", [{"role": "user", "content": "hi"}])
    return client.post(fd.url + "/api/chat", json=body)


# ----- passthrough ---------------------------------------------------------------------


def test_chat_streams_through(frontdoor, client):
    r = chat(frontdoor, client, fake_chunks=4)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-ndjson")
    lines = [json.loads(line) for line in r.text.splitlines()]
    assert [ln["message"]["content"] for ln in lines[:-1]] == ["tok0 ", "tok1 ", "tok2 ", "tok3 "]
    assert lines[-1]["done"] is True


def test_streaming_is_not_buffered(frontdoor, client):
    """The first chunk reaches the client long before Ollama finishes."""
    body = {"model": "small:1b", "messages": [], "fake_chunks": 5, "fake_delay": 0.3}
    start = time.monotonic()
    with client.stream("POST", frontdoor.url + "/api/chat", json=body) as r:
        lines = r.iter_lines()
        first = next(lines)
        first_at = time.monotonic() - start
        rest = list(lines)
    total = time.monotonic() - start
    assert json.loads(first)["message"]["content"] == "tok0 "
    assert first_at < 0.5 < total
    assert len(rest) == 5


def test_non_streaming_chat_and_timings_recorded(frontdoor, client):
    r = chat(frontdoor, client, stream=False, fake_chunks=10)
    assert r.json()["message"]["content"] == "hello " * 10
    assert wait_for(lambda: "small:1b" in frontdoor.call(frontdoor.hub.get_snapshot)["generation"])
    gen = frontdoor.call(frontdoor.hub.get_snapshot)["generation"]["small:1b"]
    assert gen["gen_tps"] == 20.0 and gen["prompt_tps"] == 80.0 and gen["gen_tokens"] == 10


def test_streamed_generate_timings_recorded(frontdoor, client):
    r = client.post(frontdoor.url + "/api/generate", json={"model": "big:70b", "prompt": "x", "fake_chunks": 2})
    assert r.status_code == 200
    assert wait_for(lambda: "big:70b" in frontdoor.call(frontdoor.hub.get_snapshot)["generation"])
    assert frontdoor.call(frontdoor.hub.get_snapshot)["generation"]["big:70b"]["gen_tokens"] == 2


def test_openai_sse_streams_through(frontdoor, client):
    r = client.post(frontdoor.url + "/v1/chat/completions", json={"model": "small:1b", "messages": [], "stream": True})
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.text.rstrip().endswith("data: [DONE]")


@pytest.mark.parametrize(
    "method,path,body",
    [
        ("GET", "/api/tags", None),
        ("GET", "/api/version", None),
        ("POST", "/api/show", {"model": "small:1b"}),
        ("POST", "/api/embed", {"model": "e", "input": "x"}),
        ("DELETE", "/api/delete", {"model": "x"}),
        ("GET", "/", None),
    ],
)
def test_other_endpoints_pass_through(frontdoor, client, method, path, body):
    r = client.request(method, frontdoor.url + path, json=body)
    assert r.status_code == 200
    assert frontdoor.recorder.last["path"] == path
    assert frontdoor.recorder.last["body"] == body


def test_pull_progress_streams(frontdoor, client):
    r = client.post(frontdoor.url + "/api/pull", json={"model": "x"})
    assert [json.loads(x)["status"] for x in r.text.splitlines()] == ["pulling manifest", "verifying", "success"]


def test_blob_upload_is_streamed_and_not_size_capped(frontdoor_factory, client):
    fd = frontdoor_factory(hub__max_body_bytes=1024)
    data = b"\x00" * (3 * 1024 * 1024)
    r = client.post(fd.url + "/api/blobs/sha256-abc", content=data)
    assert r.status_code == 201
    assert fd.recorder.blob_bytes == len(data)
    assert client.head(fd.url + "/api/blobs/sha256-abc").status_code == 404
    # Inference bodies are capped, though.
    big = chat(fd, client, messages=[{"role": "user", "content": "x" * 2000}])
    assert big.status_code == 413 and big.json() == {"error": "request body too large"}


def test_query_string_passes_through(frontdoor, client):
    client.get(frontdoor.url + "/api/tags?foo=bar")
    assert frontdoor.recorder.last["query"] == "foo=bar"


def test_only_safe_headers_cross(frontdoor, client):
    r = client.get(
        frontdoor.url + "/api/version",
        headers={"Authorization": "Bearer secret", "Cookie": "a=b", "X-OllamaPower-Priority": "high"},
    )
    sent = frontdoor.recorder.last["headers"]
    assert "authorization" not in sent and "cookie" not in sent and "x-ollamapower-priority" not in sent
    assert "x-upstream-secret" not in r.headers
    assert "server" not in r.headers


def test_api_ps_is_annotated(frontdoor, client):
    models = client.get(frontdoor.url + "/api/ps").json()["models"]
    assert [m["ollamapower"] for m in models] == [{"engine": "local-ollama", "nodes": []}] * 2
    assert models[1]["size_vram"] == 20 * 1024**3  # Ollama's own fields untouched


def test_unknown_paths_404(frontdoor, client):
    for path in ("/docs", "/openapi.json", "/admin", "/ollamapower"):
        r = client.get(frontdoor.url + path)
        assert r.status_code == 404 and r.json() == {"error": "not found"}


# ----- status --------------------------------------------------------------------------


def test_status_endpoint(frontdoor, client):
    assert wait_for(lambda: client.get(frontdoor.url + "/ollamapower/status").json()["ollama"]["reachable"])
    s = client.get(frontdoor.url + "/ollamapower/status").json()
    assert s["ollama"]["version"] == "0.0.0-fake"
    assert s["queue"]["capacity"] == 4
    big = next(m for m in s["ollama"]["models"] if m["name"] == "big:70b")
    assert big["spilled"] and big["vram_percent"] == 50
    assert any("big:70b is only 50% in VRAM" in w for w in s["warnings"])
    assert s["nodes"] == []


# ----- errors --------------------------------------------------------------------------


def test_ollama_down_gives_502_in_each_api_shape(frontdoor_factory, client):
    fd = frontdoor_factory(ollama_url=f"http://127.0.0.1:{free_port()}")
    r = chat(fd, client)
    assert r.status_code == 502 and "not reachable" in r.json()["error"]
    r = client.post(fd.url + "/v1/chat/completions", json={"model": "x", "messages": []})
    assert r.status_code == 502 and "not reachable" in r.json()["error"]["message"]
    assert fd.call(lambda: fd.hub.gate.active) == 0  # the slot was given back
    assert wait_for(lambda: any("not reachable" in w for w in fd.call(fd.hub.warnings)))


def test_ollama_errors_pass_through(frontdoor, client):
    r = chat(frontdoor, client, model="missing")
    assert r.status_code == 404 and r.json() == {"error": "model 'missing' not found"}


def test_ollama_crash_mid_stream_ends_with_error_line(frontdoor, client):
    r = chat(frontdoor, client, fake_crash=True, fake_chunks=5)
    lines = r.text.splitlines()
    assert json.loads(lines[0])["message"]["content"] == "tok0 "
    assert json.loads(lines[-1]) == {"error": "upstream error"}


def test_client_disconnect_stops_ollama(frontdoor, client):
    body = {"model": "small:1b", "messages": [], "fake_chunks": 50, "fake_delay": 0.05}
    with client.stream("POST", frontdoor.url + "/api/chat", json=body) as r:
        next(r.iter_lines())
    assert wait_for(lambda: frontdoor.recorder.streams_aborted == 1)
    assert wait_for(lambda: frontdoor.call(lambda: not frontdoor.hub.streams))
    assert frontdoor.call(lambda: frontdoor.hub.gate.active) == 0


def test_shutdown_cuts_streams(frontdoor, client):
    body = {"model": "small:1b", "messages": [], "fake_chunks": 1000, "fake_delay": 0.05}
    lines: list[str] = []

    def consume():
        try:
            with (
                httpx.Client(timeout=10, trust_env=False) as c,
                c.stream("POST", frontdoor.url + "/api/chat", json=body) as r,
            ):
                for line in r.iter_lines():
                    lines.append(line)
        except httpx.HTTPError:
            pass

    t = threading.Thread(target=consume)
    t.start()
    assert wait_for(lambda: len(lines) >= 1)
    started = time.monotonic()
    frontdoor.server.stop()
    t.join(5)
    assert time.monotonic() - started < 4
    assert json.loads(lines[-1]) == {"error": "stream terminated: hub shutting down"}


# ----- request log ---------------------------------------------------------------------


def test_request_log_has_metadata_but_no_content(frontdoor, client):
    secret = "my secret prompt 1234"
    chat(frontdoor, client, messages=[{"role": "user", "content": secret}], fake_chunks=2)
    path = frontdoor.config.logs_dir / "requests.jsonl"
    assert wait_for(lambda: path.exists() and path.read_text().strip())
    text = path.read_text()
    assert secret not in text and "tok0" not in text
    entry = json.loads(text.splitlines()[-1])
    assert entry["kind"] == "request" and entry["path"] == "/api/chat" and entry["model"] == "small:1b"
    assert entry["status"] == 200 and entry["outcome"] == "complete" and entry["priority"] == "normal"


# ----- browser protections ---------------------------------------------------------------


@pytest.mark.parametrize("host", ["evil.example", "evil.example:11434", "8.8.8.8", "[2606:4700::1111]:11434"])
def test_foreign_host_header_is_refused(frontdoor, client, host):
    """DNS rebinding: a page under evil.example pointed at 127.0.0.1 must get nothing."""
    for path in ("/api/tags", "/ollamapower/status", "/"):
        r = client.get(frontdoor.url + path, headers={"Host": host})
        assert r.status_code == 403 and r.json() == {"error": "forbidden host"}
    assert all(r["path"] != "/api/tags" for r in frontdoor.recorder.requests)


@pytest.mark.parametrize("host", ["localhost:11434", "127.0.0.1", "[::1]:11434", "my-pc.local", "192.168.1.10:11434"])
def test_local_host_headers_pass(frontdoor, client, host):
    assert client.get(frontdoor.url + "/api/tags", headers={"Host": host}).status_code == 200


def test_ollama_still_decides_on_origins(frontdoor, client):
    bad = client.post(frontdoor.url + "/api/pull", json={"model": "x"}, headers={"Origin": "https://evil.example"})
    assert bad.status_code == 403 and bad.json() == {"error": "origin not allowed"}  # Ollama's verdict
    ok = client.get(frontdoor.url + "/api/tags", headers={"Origin": "http://localhost:3000"})
    assert ok.status_code == 200 and ok.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_cors_preflight_and_streamed_cors_headers(frontdoor, client):
    pre = client.options(
        frontdoor.url + "/api/chat",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"},
    )
    assert pre.status_code == 204 and pre.headers["access-control-allow-methods"] == "GET, POST"
    r = client.post(
        frontdoor.url + "/api/chat",
        json={"model": "small:1b", "messages": []},
        headers={"Origin": "http://localhost:3000"},
    )
    assert r.status_code == 200 and r.headers["access-control-allow-origin"] == "http://localhost:3000"
    ps = client.get(frontdoor.url + "/api/ps", headers={"Origin": "http://localhost:3000"})
    assert ps.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_host_parsing():
    from ollamapower.hub.frontdoor.hostcheck import host_allowed, split_host

    assert split_host("LocalHost:11434") == "localhost"
    assert split_host("[::1]:11434") == "::1"
    assert split_host("::1") == "::1"
    assert host_allowed("", "pc") and host_allowed("pc", "pc") and host_allowed("0.0.0.0", "pc")
    assert not host_allowed("localhost.evil.example", "pc")
    assert not host_allowed("evil.example", "pc")


# ----- review regressions ----------------------------------------------------------------


def test_inference_keeps_query_and_raw_path(frontdoor, client):
    client.post(frontdoor.url + "/v1/chat/completions?foo=bar", json={"model": "m", "messages": []})
    assert frontdoor.recorder.last["query"] == "foo=bar"
    client.post(frontdoor.url + "/api/blobs/sha256-a%2Bb", content=b"x")
    assert frontdoor.recorder.last["raw_path"] == "/api/blobs/sha256-a%2Bb"


def test_untagged_model_name_matches_latest(frontdoor, client):
    frontdoor.recorder.ps_models.append({"name": "plain:latest", "size": 1, "size_vram": 1})
    chat(frontdoor, client, model="plain", stream=False)

    def tps():
        models = frontdoor.call(frontdoor.hub.get_snapshot)["ollama"]["models"]
        return next((m["last_gen_tps"] for m in models if m["name"] == "plain:latest"), None)

    assert wait_for(lambda: tps() == 20.0, 8)
