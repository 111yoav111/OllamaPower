"""The planner: the never-slower decision, routing under a benchmark key, benchmarks, fault tolerance."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass
from typing import Any

import httpx
import pytest

from ollamapower.config import Config
from ollamapower.hub import plans as plans_mod
from ollamapower.hub.benchmark import make_prompt
from ollamapower.hub.runtime import Hub
from ollamapower.ollama.client import InstalledModel
from ollamapower.state import StateStore

from .conftest import wait_for
from .test_split import chat, make_cluster

# ----- the decision rule and routing (no network) ----------------------------------------


@dataclass
class StubNode:
    node_id: str
    name: str
    tunnel_effective: str = "secure"

    def lendable_vram(self) -> int:
        return 8 << 30


@pytest.fixture
def book(tmp_path):
    cfg = Config()
    cfg.config_dir = tmp_path
    hub = Hub(cfg, StateStore(tmp_path / "state.json"))
    helpers = [StubNode("node-aaaa", "desk"), StubNode("node-bbbb", "attic")]
    hub.split.candidates = lambda: (list(helpers), [])  # type: ignore[method-assign]
    hub.ollama.installed = [InstalledModel(name="big:70b", size=40 << 30, digest="d1")]
    return hub.plans, hub, helpers


def run(tps: float, ok: bool = True, helpers: list[StubNode] | None = None, ctx: int = 4096) -> dict[str, Any]:
    r: dict[str, Any] = {"ok": ok, "gen_tps": tps, "error": "" if ok else "boom"}
    if helpers is not None:
        r["key"] = {
            "v": plans_mod.KEY_VERSION,
            "digest": "d1",
            "ctx": ctx,
            "build": plans_mod.Pins.load().build,
            **layout(helpers),
        }
    return r


def layout(helpers: list[StubNode], lent: int = 8, layers: tuple[int, ...] = ()) -> dict[str, Any]:
    """A ``SplitSession.layout()``: each helper lends Vulkan0, the hub's CUDA0 holds the rest."""
    counts = list(layers) or [10] * len(helpers) + [31]
    return {
        "helpers": [
            {"id": n.node_id, "name": n.name, "tunnel": n.tunnel_effective, "device": "Vulkan0", "lent_gib": lent}
            for n in helpers
        ],
        "placement": [
            *({"node": n.node_id, "device": "Vulkan0", "layers": c} for n, c in zip(helpers, counts, strict=False)),
            {"node": None, "device": "CUDA0", "layers": counts[-1]},
        ],
        "cpu_layers": 0,
    }


@pytest.mark.parametrize(
    "local,split,choice",
    [
        (10.0, 10.99, "local"),  # +9.9 %: not enough (min_gain 10 %)
        (10.0, 11.0, "split"),  # +10 %: enough
        (10.0, 30.0, "split"),
        (30.0, 10.0, "local"),
    ],
)
def test_split_only_when_it_wins_by_min_gain(book, local, split, choice):
    plans, _, helpers = book
    rec = plans.decide("big:70b", run(local), [run(split, helpers=helpers)])
    assert rec["choice"] == choice


def test_local_failure_lets_split_run_models_ollama_cannot(book):
    plans, _, helpers = book
    assert plans.decide("big:70b", run(0, ok=False), [run(5.0, helpers=helpers)])["choice"] == "split"
    assert plans.decide("big:70b", run(5.0), [run(0, ok=False)])["choice"] == "local"


def test_best_candidate_wins_and_its_helpers_are_the_plan(book):
    plans, _, helpers = book
    plans.decide("big:70b", run(10.0), [run(12.0, helpers=helpers[:1]), run(20.0, helpers=helpers)])
    spec = plans.route("big:70b")
    assert spec is not None and spec.helpers == ("node-aaaa", "node-bbbb") and spec.ctx == 4096


def test_route_requires_the_benchmark_key_to_still_hold(book):
    plans, hub, helpers = book
    plans.decide("big:70b", run(10.0), [run(20.0, helpers=helpers)])
    assert plans.route("big:70b") is not None
    helpers[1].tunnel_effective = "direct"
    assert plans.route("big:70b") is None and "tunnel mode changed" in plans.describe("big:70b")["why"]
    helpers[1].tunnel_effective = "secure"
    gone = helpers.pop()
    assert plans.route("big:70b") is None and "attic is not available" in plans.describe("big:70b")["why"]
    helpers.append(gone)
    hub.ollama.installed[0].digest = "d2"
    assert plans.route("big:70b") is None  # the model was re-pulled: benchmark again
    hub.ollama.installed[0].digest = "d1"
    plans.pin("big:70b", None, ctx=8192)
    assert plans.route("big:70b") is None  # another context size
    plans.pin("big:70b", None, ctx=4096)
    assert plans.route("big:70b") is not None


def test_layout_must_match_the_benchmark_up_to_small_jitter(book):
    _, _, helpers = book
    key = run(20.0, helpers=helpers)["key"]
    assert plans_mod.layout_problem(key, layout(helpers)) == ""
    # Free memory moved a little: lent memory within 1 GiB, a layer more or less here and there.
    assert plans_mod.layout_problem(key, layout(helpers, lent=9, layers=(11, 9, 31))) == ""
    assert plans_mod.layout_problem(key, layout(helpers, lent=7, layers=(10, 10, 32))) == ""
    assert "lends 5 GiB now, 8 GiB" in plans_mod.layout_problem(key, layout(helpers, lent=5))
    assert "split differently" in plans_mod.layout_problem(key, layout(helpers, layers=(4, 10, 37)))
    other = layout(helpers)
    other["helpers"][0]["device"] = other["placement"][0]["device"] = "Vulkan1"
    assert "desk lends Vulkan1 now, not Vulkan0" in plans_mod.layout_problem(key, other)
    other = layout(helpers)
    other["placement"][-1]["device"] = "CUDA1"
    assert "other devices" in plans_mod.layout_problem(key, other)
    assert "not part of the benchmark" in plans_mod.layout_problem(key, layout([*helpers, StubNode("node-c", "c")]))
    spilled = layout(helpers)
    spilled["cpu_layers"] = 5
    assert "split differently" in plans_mod.layout_problem(key, spilled)


def test_routed_spec_carries_the_key_and_a_stale_key_goes_back_to_ollama(book):
    plans, hub, helpers = book
    plans.decide("big:70b", run(10.0), [run(20.0, helpers=helpers)])
    spec = plans.route("big:70b")
    assert spec is not None and spec.layout == plans.record("big:70b")["key"]
    plans.stale("big:70b", "helper desk lends 2 GiB now, 8 GiB at the benchmark")
    assert plans.route("big:70b") is None
    assert "lends 2 GiB now" in plans.describe("big:70b")["why"]
    assert plans.record("big:70b")["degraded"] is None  # nothing failed
    assert [e for e in hub.recent_events if e["kind"] == "plan_stale"]


def test_old_benchmark_keys_stay_readable_but_never_match(book):
    plans, hub, helpers = book
    old_key = {
        "digest": "d1",
        "ctx": 4096,
        "build": plans_mod.Pins.load().build,
        "helpers": [{"id": n.node_id, "name": n.name, "tunnel": n.tunnel_effective} for n in helpers],
    }
    hub.state.data["plans"]["big:70b"] = {"choice": "split", "reason": "old", "key": old_key}
    assert plans.record("big:70b")["key"] == old_key
    assert plans.route("big:70b") is None
    assert "predates device and placement tracking" in plans.describe("big:70b")["why"]


def test_pins_and_exclusions(book):
    plans, _, helpers = book
    plans.decide("big:70b", run(10.0), [run(20.0, helpers=helpers)])
    plans.pin("big:70b", "local")
    assert plans.route("big:70b") is None and plans.describe("big:70b")["why"] == "forced local by the owner"
    plans.pin("big:70b", None)
    assert plans.route("big:70b") is not None
    plans.set_excluded("big:70b", "node-bbbb", True)
    assert plans.route("big:70b") is None  # the benchmarked set needs the excluded helper
    plans.pin("big:70b", "split")
    spec = plans.route("big:70b")
    assert spec is not None and spec.helpers is None and spec.exclude == ("node-bbbb",)
    with pytest.raises(ValueError):
        plans.pin("big:70b", "turbo")


def test_degraded_until_retry_after_and_helpers_back(book, monkeypatch):
    plans, hub, helpers = book
    hub.config.planner.retry_after = 0.2
    now = [time.time()]  # a fake clock: a loaded test machine must not end the cool-down early
    monkeypatch.setattr("ollamapower.hub.plans.time.time", lambda: now[0])
    plans.decide("big:70b", run(10.0), [run(20.0, helpers=helpers)])
    plans.fail("big:70b", "desk disconnected", ["line 1", "GGML_ASSERT x"])
    assert plans.route("big:70b") is None
    assert any("split mode for big:70b failed" in w for w in plans.warnings())
    event = [e for e in hub.recent_events if e["kind"] == "split_failed"][-1]
    assert event["tail"][-1] == "GGML_ASSERT x"
    gone = helpers.pop(0)
    now[0] += 0.25
    assert plans.route("big:70b") is None  # cooled down, but desk is still missing
    helpers.insert(0, gone)
    assert plans.route("big:70b") is not None  # recovered: split again
    assert plans.record("big:70b")["degraded"] is None


@pytest.fixture
def auto(book, monkeypatch):
    """Automatic benchmarks on a fake clock, with a fake benchmark whose split runs give ``outcome``."""
    plans, hub, helpers = book
    now = [1_000_000.0]
    monkeypatch.setattr("ollamapower.hub.plans.time.time", lambda: now[0])
    outcome: dict[str, Any] = {"ok": False, "error": "llama-server exited with code 1"}
    calls: list[bool] = []

    async def fake_benchmark(hub_: Any, model: str, ctx: int | None = None, automatic: bool = False) -> dict:
        calls.append(automatic)
        failed = {"ok": False, "error": outcome["error"], "failure": outcome.get("failure", "transient")}
        split = run(30.0, helpers=helpers) if outcome["ok"] else failed
        plans.decide(model, run(10.0), [split])
        if split["ok"]:
            plans.bench_succeeded(model)
        return {"splits": [split], "decided": True}

    async def facts(model: str, refresh: bool = False) -> dict[str, Any]:
        return {"worth_splitting": True}

    monkeypatch.setattr("ollamapower.hub.benchmark.benchmark_model", fake_benchmark)
    monkeypatch.setattr(plans, "facts", facts)

    def cycle() -> list[str]:
        """One look of the planner loop (minus the idle wait): what was due, after benchmarking it."""

        async def go() -> list[str]:
            plans.last_request["big:70b"] = now[0]
            due = await plans.pending()
            if due:
                await plans.auto_benchmark(due[0])
            return due

        return asyncio.run(go())

    return plans, hub, helpers, now, outcome, calls, cycle


def test_auto_benchmark_gives_up_after_repeated_split_failures(auto):
    plans, hub, helpers, now, outcome, calls, cycle = auto
    for attempt in range(plans_mod.AUTO_BENCH_MAX_FAILURES):
        assert cycle() == ["big:70b"], attempt
        assert plans.record("big:70b")["choice"] == "local"
        assert cycle() == []  # backing off: not again right away
        now[0] += plans_mod.FAILED_BENCH_BACKOFF + 1
    assert cycle() == [] and len(calls) == plans_mod.AUTO_BENCH_MAX_FAILURES  # given up: no endless loop
    ab = plans.record("big:70b")["auto_bench"]
    assert ab["stopped"] and ab["failures"] == 3 and "exited with code 1" in ab["error"]
    stopped = [w for w in plans.warnings() if "automatic benchmarks of big:70b stopped after 3" in w]
    assert stopped and "models benchmark big:70b" in stopped[0]
    assert "stopped after 3" in plans.describe("big:70b")["why"]
    assert [e for e in hub.recent_events if e["kind"] == "auto_benchmark_stopped"]
    now[0] += 10 * plans_mod.FAILED_BENCH_BACKOFF
    assert cycle() == []  # time alone doesn't bring it back

    # Something in the key changes (here: a helper fewer): worth another try.
    gone = helpers.pop()
    assert cycle() == ["big:70b"]
    helpers.append(gone)

    # A split run that works (e.g. a manual `models benchmark`) clears the count.
    outcome["ok"] = True
    plans.bench_succeeded("big:70b")
    assert plans.record("big:70b")["auto_bench"] is None
    assert not [w for w in plans.warnings() if "stopped after" in w]
    assert cycle() == ["big:70b"] and plans.record("big:70b")["choice"] == "split"
    assert cycle() == []  # decided under the current conditions


def test_auto_benchmark_failures_only_count_in_a_row(auto):
    plans, _, _, now, outcome, _, cycle = auto
    for _ in range(plans_mod.AUTO_BENCH_MAX_FAILURES - 1):
        cycle()
        now[0] += plans_mod.FAILED_BENCH_BACKOFF + 1
    outcome["ok"] = True
    cycle()
    assert plans.record("big:70b")["auto_bench"] is None


def test_transient_failures_back_off_exponentially_up_to_a_cap(book, monkeypatch):
    plans, hub, helpers = book
    hub.config.planner.retry_after = 120.0
    now = [1_000_000.0]
    monkeypatch.setattr("ollamapower.hub.plans.time.time", lambda: now[0])
    plans.decide("big:70b", run(10.0), [run(20.0, helpers=helpers)])
    waits = []
    for _ in range(8):
        plans.fail("big:70b", "desk disconnected")
        rec = plans.record("big:70b")
        wait = rec["degraded"]["until"] - now[0]
        waits.append(wait)
        assert rec["degraded"]["kind"] == "transient" and rec["failures"] == len(waits)
        now[0] += wait - 1
        assert plans.route("big:70b") is None  # still cooling down
        now[0] += 2
        assert plans.route("big:70b") is not None  # tried again (and failed again, below)
    assert waits == [120, 240, 480, 960, 1920, 3600, 3600, 3600]
    assert any("retried after 3600 s" in w for w in plans.warnings())
    plans.succeeded("big:70b")  # an answer came through split mode
    assert plans.record("big:70b")["failures"] == 0
    plans.fail("big:70b", "timed out loading the model")
    assert plans.record("big:70b")["degraded"]["until"] - now[0] == 120


def test_permanent_failures_wait_for_a_change(book, monkeypatch):
    plans, hub, helpers = book
    now = [1_000_000.0]
    monkeypatch.setattr("ollamapower.hub.plans.time.time", lambda: now[0])
    plans.decide("big:70b", run(10.0), [run(20.0, helpers=helpers)])
    plans.fail("big:70b", "llama-server exited with code 1: unknown model architecture", kind="permanent")
    rec = plans.record("big:70b")
    assert rec["degraded"]["kind"] == "permanent" and rec["failures"] == 0
    assert any("isn't tried again until" in w for w in plans.warnings())
    now[0] += 365 * 24 * 3600
    assert plans.route("big:70b") is None  # time alone changes nothing
    hub.config.llamacpp.path = "/opt/llama.cpp"  # the owner points the hub at another llama.cpp
    assert plans.route("big:70b") is not None and plans.record("big:70b")["degraded"] is None
    plans.fail("big:70b", "unsupported model", kind="permanent")
    hub.ollama.installed[0].digest = "d2"  # the model was pulled again
    plans.pin("big:70b", "split")
    assert plans.route("big:70b") is not None


def test_old_degraded_records_still_cool_down(book, monkeypatch):
    plans, hub, helpers = book
    now = [1_000_000.0]
    monkeypatch.setattr("ollamapower.hub.plans.time.time", lambda: now[0])
    plans.decide("big:70b", run(10.0), [run(20.0, helpers=helpers)])
    hub.state.data["plans"]["big:70b"]["degraded"] = {"reason": "x", "at": now[0], "until": now[0] + 60}
    assert plans.route("big:70b") is None
    now[0] += 61
    assert plans.route("big:70b") is not None


@pytest.mark.parametrize(
    "code,output,kind",
    [
        (1, ["llama_model_load: error loading model: unknown model architecture: 'mamba9'"], "permanent"),
        (127, ["sh: llama-server: not found"], "permanent"),
        (-4, [], "permanent"),  # SIGILL: built for another CPU
        (1, ["error while loading shared libraries: libvulkan.so.1"], "permanent"),
        (
            1,
            ["ggml_backend_cuda_buffer_type_alloc_buffer: allocating 9000 MiB: cudaMalloc failed: out of memory"],
            "transient",
        ),
        (1, ["unknown model architecture", "ErrorOutOfDeviceMemory"], "transient"),  # memory may be free next time
        (3, ["error: failed to load model"], "transient"),
        (-9, ["GGML_ASSERT failed"], "transient"),
    ],
)
def test_llama_server_exits_are_classified(code, output, kind):
    from ollamapower.hub.engines.split import exit_kind

    assert exit_kind(code, output) == kind


def test_helper_away_is_never_a_failure_and_llamacpp_problems_are_classified():
    from ollamapower.hub.engines.split import HelperAway, PlanStale, SplitUnavailable
    from ollamapower.llamacpp.manager import LlamaCppUnavailable

    assert HelperAway("x").kind == PlanStale("x").kind == "away"
    assert SplitUnavailable("x").kind == "transient" and SplitUnavailable("x", "permanent").kind == "permanent"
    assert LlamaCppUnavailable("downloading x failed").permanent is False


def test_auto_benchmark_stops_at_once_on_a_permanent_failure(auto):
    plans, _, _, _, outcome, calls, cycle = auto
    outcome.update(error="llama-server exited with code 1: unknown model architecture", failure="permanent")
    assert cycle() == ["big:70b"]
    ab = plans.record("big:70b")["auto_bench"]
    assert ab["stopped"] and ab["permanent"] and ab["failures"] == 1
    assert any("can't work under these conditions" in w for w in plans.warnings())


def test_auto_benchmark_does_not_count_helpers_stepping_away(auto):
    plans, _, _, now, outcome, calls, cycle = auto
    outcome.update(error="desk stopped lending while split mode was starting", failure="away")
    for _ in range(plans_mod.AUTO_BENCH_MAX_FAILURES + 1):
        assert cycle() == ["big:70b"]  # no backoff, no count
    assert plans.record("big:70b")["auto_bench"] is None


def test_phase3_plans_still_mean_split(book):
    plans, hub, _ = book
    hub.state.data["plans"]["old:latest"] = {"mode": "split", "ctx": 2048}
    spec = plans.route("old")
    assert spec is not None and spec.ctx == 2048 and spec.helpers is None


def test_the_facts_and_split_mode_use_one_fits_on_hub_check(book):
    """The facts' verdict (worth benchmarking?) and split mode's start check (Ollama is faster?) must agree."""
    from ollamapower.hub.planner import GIB, fits_on_hub
    from ollamapower.llamacpp.binaries import Device as LlamaDevice
    from ollamapower.llamacpp.binaries import LlamaCpp
    from ollamapower.ollama.gguf import GGUFInfo

    plans, hub, _ = book
    info = GGUFInfo(
        path=None,  # type: ignore[arg-type]
        file_size=0,
        version=3,
        architecture="llama",
        block_count=40,
        embedding_length=8192,
        head_count=64,
        head_count_kv=8,
        feed_forward_length=28672,
        layer_bytes=[GIB // 4] * 40,
        output_bytes=GIB // 2,
    )
    base = {"splittable": True, "weights": info.weights_bytes, "reason": ""}
    plans.gguf_cache["d1"] = info
    margin, ctx = hub.config.planner.memory_margin, plans.ctx_for("big:70b")
    verdicts = set()
    for vram_gib in [x / 4 for x in range(36, 64)]:
        hub.local_static = {"gpus": [{"name": "G", "vram_total": int(vram_gib * GIB)}]}
        facts = plans._with_vram(base, "big:70b")
        assert facts["fits"] == fits_on_hub(info, ctx, hub.split.hub_devices(), margin), vram_gib
        verdicts.add(facts["fits"])
    assert verdicts == {True, False}  # the sweep crosses the boundary
    # Once llama.cpp is installed its device list is what both checks look at.
    hub.split.binaries = LlamaCpp(root=None, devices=[LlamaDevice("CUDA0", "big", 80 * 1024, 70 * 1024)])  # type: ignore[arg-type]
    assert plans._with_vram(base, "big:70b")["fits"] is True
    assert [d.device for d in hub.split.hub_devices()] == ["CUDA0"]


def test_benchmark_prompt_is_unique_and_sized():
    a, b = make_prompt(512), make_prompt(512)
    assert a != b and 300 < len(a.split()) < 450


# ----- end to end, on the hermetic cluster -----------------------------------------------

pytestmark_e2e = pytest.mark.skipif(sys.platform == "win32", reason="fake llama.cpp programs are POSIX scripts")


@pytest.fixture
def cluster(tmp_path, fake_ollama, make_config):
    c, close = make_cluster(tmp_path, fake_ollama, make_config, hub_devices=[])
    c.hub.config.planner.auto_benchmark = False
    yield c
    close()


@pytest.fixture
def client():
    with httpx.Client(timeout=30, trust_env=False) as cl:
        yield cl


def split_speed(c, tps: float) -> None:
    """Make the fake llama-server report ``tps`` generation tokens/s (fake Ollama does 20)."""
    (c.hub_bin / "timings.json").write_text(json.dumps({"predicted_n": 3, "predicted_ms": 3000 / tps}))


@pytestmark_e2e
def test_benchmark_picks_split_when_it_is_faster_and_routes_to_it(cluster, client):
    c = cluster
    split_speed(c, 30.0)
    r = c.admin("models.benchmark", model="tiny")
    assert r["local"]["ok"], r
    assert r["local"]["gen_tps"] == pytest.approx(20.0, rel=0.01)
    assert r["splits"][0]["ok"] and r["splits"][0]["gen_tps"] == pytest.approx(30.0, rel=0.01)
    assert r["choice"] == "split" and "+50 %" in r["reason"]
    assert r["splits"][0]["helpers"][0]["name"] == "desk"
    stored = c.call(lambda: c.hub.plans.benchmarks_for("tiny"))
    assert {b["engine"] for b in stored} == {"local", "split"} and all(b["ttft_ms"] >= 0 for b in stored)

    # Automatic mode now sends the model to split, no manual plan needed.
    assert chat(client, c).status_code == 200
    assert c.call(lambda: c.hub.split.is_running("tiny"))
    assert not [x for x in c.rec.client_requests if x["path"] == "/api/chat"]
    shown = c.admin("models.show", model="tiny")
    assert shown["engine"] == "split" and shown["mode"] == "auto" and shown["split_tps"] == pytest.approx(30.0, 0.01)

    # The helper leaves: the benchmark no longer applies, Ollama serves.
    c.run(c.hub.control.kick(c.node))
    assert wait_for(lambda: c.call(lambda: c.hub.plans.route("tiny")) is None, 10)


@pytestmark_e2e
def test_a_setup_that_changed_since_the_benchmark_is_not_reused(cluster, client):
    c = cluster
    split_speed(c, 30.0)
    assert c.admin("models.benchmark", model="tiny")["choice"] == "split"
    key = c.call(lambda: c.hub.plans.record("tiny")["key"])
    assert key["v"] == plans_mod.KEY_VERSION
    assert key["helpers"][0]["device"] == "FakeGPU0" and key["helpers"][0]["lent_gib"] == 8  # 8000 MiB lent
    assert key["placement"][0]["node"] == c.node.node_id and key["placement"][0]["layers"] > 0
    assert chat(client, c).status_code == 200 and c.call(lambda: c.hub.split.is_running("tiny"))
    c.run(c.hub.split.stop())

    # The helper's owner lends much less memory now than when the benchmark ran.
    c.call(lambda: c.hub.plans._update("tiny", key={**key, "helpers": [{**key["helpers"][0], "lent_gib": 20}]}))
    r = chat(client, c)
    assert r.status_code == 200 and c.rec.last["path"] == "/api/chat"  # Ollama served, transparently
    rec = c.call(lambda: c.hub.plans.record("tiny"))
    assert "lends 8 GiB now, 20 GiB at the benchmark" in rec["key"]["stale"] and rec["degraded"] is None
    assert c.call(lambda: c.hub.plans.route("tiny")) is None  # until it is benchmarked again
    assert c.admin("models.benchmark", model="tiny")["choice"] == "split"
    assert c.call(lambda: c.hub.plans.route("tiny")) is not None


@pytestmark_e2e
def test_benchmark_keeps_ollama_when_split_is_not_faster_enough(cluster, client):
    c = cluster
    split_speed(c, 21.0)  # +5 %
    r = c.admin("models.benchmark", model="tiny")
    assert r["choice"] == "local" and "isn't at least 10 % faster" in r["reason"]
    assert c.call(lambda: c.hub.split.session) is None  # the benchmark cleaned up after itself
    chat(client, c)
    assert c.rec.last["path"] == "/api/chat"


@pytestmark_e2e
def test_pinned_model_keeps_its_pin_after_a_benchmark(cluster):
    c = cluster
    split_speed(c, 30.0)
    c.admin("models.force_local", model="tiny")
    r = c.admin("models.benchmark", model="tiny")
    assert r["decided"] is False and "pinned to local" in r["reason"]
    assert c.call(lambda: c.hub.plans.route("tiny")) is None


@pytestmark_e2e
def test_automatic_benchmark_runs_when_idle(cluster, client, monkeypatch):
    c = cluster
    split_speed(c, 40.0)
    chat(client, c)  # a request marks the model as used (Ollama serves it: nothing decided yet)
    assert c.rec.last["path"] == "/api/chat"
    c.hub.config.planner.auto_benchmark = True
    c.hub.config.planner.idle_seconds = 0.1
    monkeypatch.setattr(plans_mod, "AUTO_CHECK", 0.1)

    async def start() -> None:
        asyncio.ensure_future(c.hub.plans.run())

    c.run(start())
    assert wait_for(lambda: c.call(lambda: c.hub.plans.record("tiny")["choice"]) == "split", 20)
    events = [e["kind"] for e in c.call(lambda: list(c.hub.recent_events))]
    assert "benchmark_start" in events and "plan_chosen" in events


@pytestmark_e2e
def test_client_request_preempts_an_automatic_benchmark(cluster, client):
    c = cluster

    async def slow_bench() -> None:
        c.hub.plans.auto_task = asyncio.ensure_future(asyncio.sleep(60))

    c.run(slow_bench())
    chat(client, c)
    assert wait_for(lambda: c.call(lambda: c.hub.plans.auto_task.cancelled()), 5)


@pytestmark_e2e
def test_failure_before_the_first_token_falls_back_transparently(cluster, client):
    c = cluster
    c.hub.config.split.stall_timeout = 1.0
    c.admin("models.pin", model="tiny", engine="split")
    assert chat(client, c).status_code == 200  # split works
    r = chat(client, c, messages=[{"role": "user", "content": "hang"}])
    assert r.status_code == 200 and c.rec.last["path"] == "/api/chat"  # Ollama answered instead
    assert "tok0" in r.text
    assert c.call(lambda: c.hub.plans.record("tiny")["degraded"]) is not None
    chat(client, c)
    assert c.rec.last["path"] == "/api/chat"  # still local while degraded


@pytestmark_e2e
def test_crash_mid_stream_ends_with_an_error_and_degrades(cluster, client):
    c = cluster
    c.hub.config.planner.retry_after = 0.5
    c.admin("models.pin", model="tiny", engine="split")
    r = chat(client, c, messages=[{"role": "user", "content": "die"}])
    lines = [json.loads(x) for x in r.text.splitlines()]
    assert lines[0]["message"]["content"] == "Hello" and "error" in lines[-1]
    assert wait_for(lambda: c.call(lambda: c.hub.plans.record("tiny")["degraded"]) is not None, 10)
    events = c.call(lambda: [e for e in c.hub.recent_events if e["kind"] == "split_failed"])
    assert any("GGML_ASSERT failed: fake crash" in line for e in events for line in e.get("tail", []))
    chat(client, c)
    assert c.rec.last["path"] == "/api/chat"
    time.sleep(0.6)
    assert chat(client, c).status_code == 200  # cooled down, helper still there: split again
    assert c.call(lambda: c.hub.split.is_running("tiny"))


@pytestmark_e2e
@pytest.mark.parametrize("secret", [None, "wrong"])
def test_clients_cannot_force_an_engine(cluster, client, secret):
    c = cluster
    headers = {"x-ollamapower-engine": "split", "x-ollamapower-ctx": "512"}
    if secret:
        headers["x-ollamapower-bench"] = secret
    r = client.post(
        c.url + "/api/chat",
        json={"model": "tiny", "messages": [{"role": "user", "content": "x"}]},
        headers=headers,
    )
    assert r.status_code == 200 and c.rec.last["path"] == "/api/chat"
    assert c.call(lambda: c.hub.split.session) is None
    assert "x-ollamapower-engine" not in c.rec.last["headers"]


@pytestmark_e2e
def test_benchmark_refuses_models_split_mode_cannot_run(cluster):
    from ollamapower.admin.ipc import AdminError

    with pytest.raises(AdminError, match="could not find|not installed"):
        cluster.admin("models.benchmark", model="small:1b")


@pytestmark_e2e
def test_models_list_shows_engine_mode_and_speeds(cluster):
    from ollamapower.__main__ import render_models

    c = cluster
    split_speed(c, 30.0)
    c.admin("models.benchmark", model="tiny")
    c.rec.ps_models[:] = []
    c.call(lambda: c.hub.ollama.installed.append(InstalledModel(name="tiny:latest", size=1 << 20, digest="")))
    rows = {m["name"]: m for m in c.admin("models.list")}
    assert rows["tiny:latest"]["plan"] == "split" and rows["tiny:latest"]["plan_split_tps"] == pytest.approx(30, 0.01)
    text = "\n".join(line for line, _ in render_models(list(rows.values())))
    assert "tiny:latest" in text and "auto" in text


@pytestmark_e2e
def test_benchmark_waits_for_a_busy_split_session(cluster, monkeypatch):
    from ollamapower.admin.ipc import AdminError
    from ollamapower.hub import benchmark as bench_mod

    c = cluster
    monkeypatch.setattr(bench_mod, "IDLE_WAIT", 0.3)
    c.admin("models.pin", model="tiny", engine="split")
    with httpx.Client(timeout=30, trust_env=False) as cl:
        assert chat(cl, c).status_code == 200
    session = c.call(lambda: c.hub.split.session)
    c.call(lambda: setattr(session, "active", 1) or session.idle.clear())  # a request is streaming
    with pytest.raises(AdminError, match="busy"):
        c.admin("models.benchmark", model="tiny")
    assert c.call(lambda: c.hub.split.is_running("tiny"))  # nobody's answer was cut off


@pytestmark_e2e
def test_manual_benchmark_runs_after_automatic_ones_gave_up(cluster):
    c = cluster
    split_speed(c, 30.0)

    def give_up() -> None:
        c.hub.plans._update(
            "tiny:latest",
            auto_bench={"key": c.hub.plans._attempt_key("tiny:latest"), "failures": 3, "error": "x", "stopped": True},
        )

    c.call(give_up)
    assert c.call(lambda: c.hub.plans.describe("tiny")["auto_benchmark_stopped"])
    r = c.admin("models.benchmark", model="tiny")
    assert r["choice"] == "split"
    assert c.call(lambda: c.hub.plans.record("tiny")["auto_bench"]) is None


@pytestmark_e2e
def test_benchmark_report_for_the_docs(cluster):
    from ollamapower.__main__ import benchmark_markdown

    c = cluster
    split_speed(c, 30.0)
    c.admin("models.benchmark", model="tiny")
    report = benchmark_markdown(c.admin("benchmarks.list"))
    assert "| Date | Model |" in report and "**Helper desk:** RTX 4090" in report
    rows = [line for line in report.splitlines() if line.startswith("| 20")]
    assert len(rows) == 2 and any("| split | desk (secure) |" in r for r in rows)
