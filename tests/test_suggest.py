"""`models suggest`: the pool the helpers could lend now, and every model's verdict against it."""

from __future__ import annotations

import types
from typing import Any

import httpx
import pytest

from ollamapower.admin.ipc import AdminError
from ollamapower.admin.service import Admin
from ollamapower.config import Config
from ollamapower.hub.nodes import Node, clean_capabilities, clean_lending, clean_telemetry
from ollamapower.hub.planner import GIB, Device
from ollamapower.hub.runtime import Hub
from ollamapower.hub.suggest import catalog_info, load_catalog, pool_devices, suggest, verdict
from ollamapower.llamacpp.install import Pins
from ollamapower.ollama.client import InstalledModel
from ollamapower.state import StateStore

from .tiny_gguf import write_tiny_llama

BUILD = Pins.load().build


def helper(
    name: str,
    gpus: list[dict[str, Any]] | None = None,
    ram_used: int = 8 * GIB,
    rtt_ms: float = 1.0,
    **lending: Any,
) -> Node:
    """A connected helper with a ready llama.cpp, as its agent would have reported it."""
    gpus = gpus or []
    node = Node(f"node-{name}", name)
    node.channel = types.SimpleNamespace(closed=False)  # type: ignore[assignment]
    node.capabilities = clean_capabilities(
        {
            "ram_total": 32 * GIB,
            "ram_bandwidth": 30e9,
            "gpus": [{"index": i, "name": g["name"], "vram_total": g["total"]} for i, g in enumerate(gpus)],
            "llama_cpp": {"status": "ready", "build": BUILD},
        }
    )
    node.telemetry = clean_telemetry(
        {
            "ram_total": 32 * GIB,
            "ram_used": ram_used,
            "gpus": [{"index": i, "vram_total": g["total"], "vram_used": g.get("used", 0)} for i, g in enumerate(gpus)],
        }
    )
    node.lending = clean_lending(lending)
    node.net_tests.append({"rtt_ms": rtt_ms})
    return node


def make_hub(tmp_path, ollama_url: str = "http://127.0.0.1:9") -> Hub:
    cfg = Config()
    cfg.config_dir = tmp_path
    cfg.ollama.url = ollama_url
    return Hub(cfg, StateStore(tmp_path / "state.json"))


def add(hub: Hub, *nodes: Node) -> None:
    for n in nodes:
        hub.registry.nodes[n.node_id] = n


def test_pool_devices_follow_each_owners_rules(tmp_path):
    hub = make_hub(tmp_path)
    gpu = helper("gpu", [{"name": "RTX 4090", "total": 24 * GIB, "used": 4 * GIB}], max_vram_gb=16)
    laptop = helper("laptop", allow_cpu=True, rtt_ms=4.0)  # Wi-Fi, no GPU
    capped = helper("capped", allow_cpu=True, max_vram_gb=10)
    closed = helper("closed")  # no GPU, CPU lending off
    paused = helper("paused", [{"name": "RTX 3060", "total": 12 * GIB}])
    paused.hub_paused = True
    full = helper("full", [{"name": "GTX 1060", "total": 6 * GIB, "used": 6 * GIB}], allow_cpu=True)
    add(hub, gpu, laptop, capped, closed, paused, full)

    devices = {d.label: d for d in pool_devices(hub)}
    assert set(devices) == {"gpu RTX 4090", "laptop CPU", "capped CPU", "full CPU"}
    d = devices["gpu RTX 4090"]
    assert d.free == 16 * GIB and not d.ram and d.node_id == "node-gpu"  # the owner's cap, below the 20 GiB free
    d = devices["laptop CPU"]
    assert d.ram and d.free == (32 - 8 - 3) * GIB and d.bandwidth == 30e9 and d.rtt == 0.004
    assert devices["capped CPU"].free == 10 * GIB
    assert devices["full CPU"].ram  # its only GPU is full: like its agent, it lends RAM instead
    assert next(iter(devices)) == "gpu RTX 4090"  # a GPU first


@pytest.fixture
def tiny(tmp_path, fake_ollama):
    """A hub whose Ollama (the fake) has one tiny installed GGUF model, with no GPU of its own."""
    upstream, rec = fake_ollama
    (tmp_path / "blobs").mkdir()
    path = write_tiny_llama(tmp_path / "blobs" / ("sha256-" + "cd" * 32), n_layer=8)
    rec.show_models["tiny:latest"] = str(path)
    hub = make_hub(tmp_path, upstream.url)
    hub.local_static = {"gpus": []}
    hub.ollama.installed = [
        InstalledModel(name="tiny", size=path.stat().st_size, digest="d-tiny"),
        InstalledModel(name="cloudy:1b", size=GIB, digest="d-cloudy"),  # no GGUF Ollama will point to
    ]
    ram = {"bytes": 16 * GIB}
    hub.split.hub_ram = lambda free: Device("hub RAM", ram["bytes"], device="CPU", ram=True)  # type: ignore[method-assign]
    return hub, ram


async def run_suggest(hub: Hub, **kwargs: Any) -> dict[str, Any]:
    async with httpx.AsyncClient(base_url=hub.config.ollama.url, trust_env=False) as http:
        hub.split.http = http
        try:
            return await suggest(hub, **kwargs)
        finally:
            hub.split.http = None


async def test_installed_models_get_a_verdict(tiny):
    hub, _ = tiny
    hub.local_static = {"gpus": [{"name": "G", "vram_total": 8 * GIB}]}
    d = await run_suggest(hub)
    rows = {r["model"]: r for r in d["installed"]}
    r = rows["tiny:latest"]
    assert r["type"] == "dense" and r["fits_hub_gpu"] and r["verdict"] == "runs fine on the hub"
    assert r["alone_tps"] > 0 and r["pooled_tps"] is None and r["ctx"] == hub.config.split.default_ctx
    assert rows["cloudy:1b"]["verdict"].startswith("can't estimate:")
    assert d["pool"] == [] and d["not_counted"] == []
    assert all(not c["installed"] and c["verdict"].startswith("estimate, not installed: ") for c in d["catalog"])


async def test_a_model_too_big_for_the_hub_runs_pooled_and_shows_measured_speeds(tiny):
    hub, ram = tiny
    ram["bytes"] = 64 * 1024  # the hub can't hold even the token embeddings: Ollama would page from disk
    alone = await run_suggest(hub, ctx=512)
    r = next(r for r in alone["installed"] if r["model"] == "tiny:latest")
    assert not r["fits_hub"] and r["pooled_tps"] is None and r["missing_bytes"] > 0
    assert r["verdict"].startswith("doesn't fit even pooled: ") and r["verdict"].endswith(" GB missing")

    add(hub, helper("desk", [{"name": "RTX 4090", "total": 24 * GIB}]), helper("closed"))
    pooled = await run_suggest(hub, ctx=512)
    r = next(r for r in pooled["installed"] if r["model"] == "tiny:latest")
    assert r["ctx"] == 512 and r["pooled_tps"] > 0 and r["pooled_on"] == ["desk RTX 4090"]
    assert r["verdict"].startswith("pooled: ~") and "alone)" in r["verdict"]
    assert [p["label"] for p in pooled["pool"]] == ["desk RTX 4090"]
    assert any("closed" in x for x in pooled["not_counted"])

    hub.state.data["benchmarks"] = [{"model": "tiny:latest", "engine": "split", "ok": True, "gen_tps": 42.0}]
    r = next(r for r in (await run_suggest(hub, ctx=512))["installed"] if r["model"] == "tiny:latest")
    assert r["measured_pooled_tps"] == 42.0 and r["verdict"].startswith("pooled: 42 tok/s")


async def test_admin_command_and_table(tiny):
    from ollamapower.__main__ import render_suggest

    hub, _ = tiny
    async with httpx.AsyncClient(base_url=hub.config.ollama.url, trust_env=False) as http:
        hub.split.http = http
        d = await Admin(hub).dispatch("models.suggest", {"ctx": 2048})
        with pytest.raises(AdminError, match="--ctx"):
            await Admin(hub).dispatch("models.suggest", {"ctx": 10})
    text = "\n".join(line for line, _ in render_suggest(d))
    assert "tiny:latest" in text and "VERDICT" in text and "Pool now: no helper lending" in text
    installed = {r["model"] for r in d["installed"]}
    assert not installed & {r["model"] for r in d["catalog"]}


async def test_installed_catalog_models_are_not_suggested_again(tiny):
    hub, _ = tiny
    hub.ollama.installed.append(InstalledModel(name="gpt-oss:20b", size=14 * 10**9, digest="d-oss"))
    d = await run_suggest(hub)
    assert "gpt-oss:20b" in {r["model"] for r in d["installed"]}
    assert "gpt-oss:20b" not in {r["model"] for r in d["catalog"]}


def test_catalog_entries_are_sane():
    entries = load_catalog()
    names = [e["name"] for e in entries]
    assert len(names) == len(set(names)) >= 8 and all(":" in n for n in names)
    for e in entries:
        per_weight = e["download_gb"] / e["params_b"]
        assert 0.4 <= per_weight <= 1.2, e  # 4-8 bits per weight: the default tags are Q4-Q8
        assert 10 <= e["layers"] <= 200 and e["kv_heads"] > 0 and e["head_dim"] in (64, 128, 256), e
        info = catalog_info(e)
        size = e["download_gb"] * 1e9
        assert info.weights_bytes == pytest.approx(size, rel=0.01), e
        active = sum(info.active_layer_bytes()) + info.output_bytes
        if e["active_params_b"] is None:
            assert not info.is_moe and active == pytest.approx(size - info.input_bytes, rel=0.01)
        else:
            assert 0 < e["active_params_b"] < e["params_b"] and info.is_moe, e
            assert active == pytest.approx(size * e["active_params_b"] / e["params_b"], rel=0.05), e


def test_a_model_spilling_into_the_hubs_ram_is_not_called_fine():
    row = {"fits_hub": True, "fits_hub_gpu": False, "alone_tps": 2.4, "pooled_tps": None, "installed": True}
    assert verdict(row, 0.1) == "runs on the hub, partly from RAM: ~2.4 tok/s"
