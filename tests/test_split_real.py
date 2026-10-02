"""Split mode with the real, pinned llama.cpp build: a tiny generated model, one helper on CPU.

Runs only when ``OLLAMAPOWER_TEST_LLAMACPP`` points to an unpacked build of the
pinned release (CI doesn't download it). Everything is real except Ollama:
the hub starts ``llama-server`` with ``--rpc`` through the tunnel (TLS, or
direct) and the helper's relay to a real ``ggml-rpc-server``, and the layers
are computed on the helper.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import pytest

from .conftest import wait_for
from .test_split import make_cluster

BUILD_DIR = os.environ.get("OLLAMAPOWER_TEST_LLAMACPP", "")
pytestmark = pytest.mark.skipif(
    not BUILD_DIR or not Path(BUILD_DIR).is_dir(), reason="set OLLAMAPOWER_TEST_LLAMACPP to a llama.cpp build"
)


@pytest.mark.parametrize("tunnel", ["secure", "direct"])
def test_real_llama_cpp_split_over_the_relay(tmp_path, fake_ollama, make_config, tunnel):
    c, close = make_cluster(
        tmp_path,
        fake_ollama,
        make_config,
        hub_devices=[],
        real_binaries=Path(BUILD_DIR),
        tunnel=tunnel,
        allow_cpu=True,
    )
    try:
        c.admin("models.plan", model="tiny", ctx=256)
        with httpx.Client(timeout=120, trust_env=False) as client:
            r = client.post(
                c.url + "/api/chat",
                json={"model": "tiny", "messages": [{"role": "user", "content": "hi"}], "options": {"num_predict": 4}},
            )
            assert r.status_code == 200, r.text
            final = [json.loads(x) for x in r.text.splitlines()][-1]
            assert final["done"] is True and final["eval_count"] == 4 and final["eval_duration"] > 0

            r = client.post(
                c.url + "/api/generate",
                json={"model": "tiny", "prompt": "x", "stream": False, "options": {"num_predict": 2}},
            )
            assert r.json()["done"] is True and r.json()["eval_count"] == 2

        session = c.call(lambda: c.hub.split.session)
        plan = session.plan
        assert plan.cpu_layers == 0 and [p.device.device for p in plan.placements] == ["CPU"]
        assert plan.placements[0].layers == 5  # every layer on the helper
        rpc = c.agent.rpc.session
        assert rpc.device == "CPU" and rpc.relay.mode == tunnel
        assert any("Accepted client connection" in line for line in rpc.output)  # llama-server really used it
        assert c.admin("models.unplan", model="tiny")["stopped"]
        assert wait_for(lambda: c.agent.rpc.session is None, 10)
    finally:
        close()
