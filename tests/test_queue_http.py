"""The priority queue as seen through the front door."""

from __future__ import annotations

import threading

import httpx

from .conftest import wait_for


def _post(url: str, results: dict, key: str, priority: str | None, **body) -> None:
    headers = {"X-OllamaPower-Priority": priority} if priority else {}
    body = {"model": "small:1b", "messages": [], "tag": key, **body}
    try:
        with httpx.Client(timeout=20, trust_env=False) as c:
            results[key] = c.post(url + "/api/chat", json=body, headers=headers)
    except httpx.HTTPError as exc:
        results[key] = exc


def _start(url, results, key, priority=None, **body) -> threading.Thread:
    t = threading.Thread(target=_post, args=(url, results, key, priority), kwargs=body)
    t.start()
    return t


def test_higher_priority_goes_first(frontdoor_factory):
    fd = frontdoor_factory(hub__max_concurrency=1)
    results: dict = {}
    blocker = _start(fd.url, results, "blocker", fake_first_delay=0.6)
    assert wait_for(lambda: fd.call(lambda: fd.hub.gate.active) == 1)
    threads = []
    for key, prio in (("low", "low"), ("normal", None), ("critical", "critical"), ("high", "high")):
        threads.append(_start(fd.url, results, key, prio))
        assert wait_for(lambda n=len(threads): fd.call(lambda: fd.hub.gate.waiting) == n)
    snap = fd.call(fd.hub.get_snapshot)
    assert snap["queue"]["waiting_by_priority"] == {"critical": 1, "high": 1, "normal": 1, "low": 1}
    assert {s["state"] for s in snap["streams"]} == {"running", "queued"}
    for t in [blocker, *threads]:
        t.join(10)
    order = [r["body"]["tag"] for r in fd.recorder.requests if r["path"] == "/api/chat"]
    assert order == ["blocker", "critical", "high", "normal", "low"]
    assert all(r.status_code == 200 for r in results.values())


def test_queue_timeout_gives_503(frontdoor_factory):
    fd = frontdoor_factory(hub__max_concurrency=1, hub__queue_timeout=0.3)
    results: dict = {}
    blocker = _start(fd.url, results, "blocker", fake_first_delay=1.0)
    assert wait_for(lambda: fd.call(lambda: fd.hub.gate.active) == 1)
    _post(fd.url, results, "late", None)
    assert results["late"].status_code == 503
    assert results["late"].headers["retry-after"] == "10"
    blocker.join(5)
    assert [r["body"]["tag"] for r in fd.recorder.requests if r["path"] == "/api/chat"] == ["blocker"]


def test_full_queue_rejects_at_once(frontdoor_factory):
    fd = frontdoor_factory(hub__max_concurrency=1, hub__max_queue=1)
    results: dict = {}
    blocker = _start(fd.url, results, "blocker", fake_first_delay=0.8)
    assert wait_for(lambda: fd.call(lambda: fd.hub.gate.active) == 1)
    waiter = _start(fd.url, results, "waiter")
    assert wait_for(lambda: fd.call(lambda: fd.hub.gate.waiting) == 1)
    _post(fd.url, results, "third", None)
    assert results["third"].status_code == 503 and "queue full" in results["third"].json()["error"]
    blocker.join(5)
    waiter.join(5)
    assert results["waiter"].status_code == 200


def test_client_leaving_the_queue_never_reaches_ollama(frontdoor_factory):
    fd = frontdoor_factory(hub__max_concurrency=1)
    results: dict = {}
    blocker = _start(fd.url, results, "blocker", fake_first_delay=0.8)
    assert wait_for(lambda: fd.call(lambda: fd.hub.gate.active) == 1)

    def impatient():
        try:
            with httpx.Client(timeout=0.3, trust_env=False) as c:
                c.post(fd.url + "/api/chat", json={"model": "small:1b", "messages": [], "tag": "impatient"})
        except httpx.HTTPError:
            pass

    impatient()
    assert wait_for(lambda: fd.call(lambda: fd.hub.gate.waiting) == 0)
    blocker.join(5)
    assert [r["body"]["tag"] for r in fd.recorder.requests if r["path"] == "/api/chat"] == ["blocker"]
    assert wait_for(lambda: fd.call(lambda: fd.hub.gate.active) == 0)


def test_passthrough_is_not_queued(frontdoor_factory, client):
    fd = frontdoor_factory(hub__max_concurrency=1)
    results: dict = {}
    blocker = _start(fd.url, results, "blocker", fake_first_delay=0.8)
    assert wait_for(lambda: fd.call(lambda: fd.hub.gate.active) == 1)
    assert client.get(fd.url + "/api/tags", timeout=0.5).status_code == 200
    blocker.join(5)
