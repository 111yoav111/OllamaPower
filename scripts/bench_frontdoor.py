"""Front-door overhead: the same requests straight to a fake Ollama and through the front door.

Development tool (needs the repo's tests/ for the fake Ollama). Each side runs in
its own process so they don't share a GIL. Run from the repo root:

    python scripts/bench_frontdoor.py            # ROUNDS=2 by default
    PROFILE=hub.prof python scripts/bench_frontdoor.py   # also profile the front door

Prints the median and p95 latency of small non-streaming /api/chat requests and
the NDJSON chunks per second of streaming ones.
"""

import multiprocessing as mp
import os
import signal
import socket
import statistics
import sys
import tempfile
import time
from pathlib import Path

import httpx


def port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def run_ollama(p):
    import uvicorn

    from tests.fake_ollama import make_fake_ollama

    app, _ = make_fake_ollama()
    uvicorn.run(app, host="127.0.0.1", port=p, log_level="error")


def run_hub(p, op):
    from ollamapower.config import Config
    from ollamapower.eventlog import EventLog
    from ollamapower.hub.frontdoor.app import create_app
    from ollamapower.hub.frontdoor.server import build_server
    from ollamapower.hub.runtime import Hub

    cfg = Config()
    cfg.config_dir = Path(tempfile.mkdtemp())
    cfg.ollama.url = f"http://127.0.0.1:{op}"
    cfg.hub.port = p
    cfg.validate()
    hub = Hub(cfg, events=EventLog(cfg.logs_dir, cfg.logging.file_max_bytes, cfg.logging.file_backups))
    app = create_app(hub)
    import cProfile
    import pstats

    pr = cProfile.Profile() if os.environ.get("PROFILE") else None
    if pr:
        pr.enable()
    try:
        build_server(hub, app, own_logging=False).run()
    finally:
        if pr:
            pr.disable()
            pstats.Stats(pr).sort_stats("tottime").dump_stats(os.environ["PROFILE"])


def wait(url):
    for _ in range(200):
        try:
            httpx.get(url + "/api/version", timeout=1)
            return
        except httpx.HTTPError:
            time.sleep(0.05)
    raise SystemExit("not up " + url)


def bench(url, n=400, chunks=200):
    c = httpx.Client(timeout=30, trust_env=False)
    body = {"model": "small:1b", "messages": [{"role": "user", "content": "hi"}], "stream": False, "fake_chunks": 1}
    for _ in range(30):
        c.post(url + "/api/chat", json=body)
    lat = []
    for _ in range(n):
        t = time.perf_counter()
        r = c.post(url + "/api/chat", json=body)
        r.read()
        lat.append((time.perf_counter() - t) * 1000)
    sbody = {
        "model": "small:1b",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
        "fake_chunks": chunks,
    }
    t = time.perf_counter()
    got = 0
    for _ in range(20):
        with c.stream("POST", url + "/api/chat", json=sbody) as r:
            for _line in r.iter_lines():
                got += 1
    secs = time.perf_counter() - t
    lat.sort()
    return {
        "p50_ms": round(statistics.median(lat), 3),
        "p95_ms": round(lat[int(n * 0.95)], 3),
        "chunks_per_s": round(got / secs),
    }


if __name__ == "__main__":
    sys.path.insert(0, ".")
    op, hp = port(), port()
    a = mp.Process(target=run_ollama, args=(op,), daemon=True)
    a.start()
    b = mp.Process(target=run_hub, args=(hp, op), daemon=True)
    b.start()
    wait(f"http://127.0.0.1:{op}")
    wait(f"http://127.0.0.1:{hp}")
    for _ in range(int(os.environ.get("ROUNDS", "2"))):
        print("direct ", bench(f"http://127.0.0.1:{op}"))
        print("front  ", bench(f"http://127.0.0.1:{hp}"))
    os.kill(b.pid, signal.SIGINT)
    b.join(10)
    a.terminate()
