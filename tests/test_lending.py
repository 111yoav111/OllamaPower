"""The helper owner's lending controls: schedule, auto-yield, and the hub draining gracefully."""

from __future__ import annotations

import datetime
import json
import sys
import threading

import httpx
import pytest

from ollamapower.agent.config import LendingRules
from ollamapower.agent.lending import LendingGuard, in_schedule
from ollamapower.config import ConfigError

from .conftest import wait_for
from .test_split import chat, make_cluster


def at(hhmm: str) -> datetime.datetime:
    h, m = map(int, hhmm.split(":"))
    return datetime.datetime(2026, 10, 2, h, m)


@pytest.mark.parametrize(
    "schedule,time,inside",
    [
        ([], "13:00", True),
        (["22:00-08:00"], "23:30", True),
        (["22:00-08:00"], "03:00", True),
        (["22:00-08:00"], "08:00", False),  # the end is exclusive
        (["22:00-08:00"], "12:00", False),
        (["09:00-12:00", "20:00-23:00"], "21:15", True),
        (["09:00-12:00", "20:00-23:00"], "13:00", False),
        (["00:00-00:00"], "17:42", True),
    ],
)
def test_schedule_windows(schedule, time, inside):
    assert in_schedule(schedule, at(time)) is inside


def sample(util: int | None, index: int = 0) -> dict:
    gpu = {"index": index}
    if util is not None:
        gpu["util"] = util
    return {"gpus": [gpu]}


def test_yields_after_sustained_foreign_load_and_comes_back():
    rules, guard = LendingRules(yield_gpu_percent=60, yield_seconds=20), LendingGuard()
    assert not guard.update_yield(rules, sample(90), tunnel_quiet=True, now=0)
    assert not guard.update_yield(rules, sample(90), tunnel_quiet=True, now=19)
    assert guard.update_yield(rules, sample(90), tunnel_quiet=True, now=20)
    d = guard.evaluate(rules, paused=False, now=at("12:00"))
    assert d.state == "yielding" and "GPU 0 (90 % busy)" in d.reason
    assert guard.update_yield(rules, sample(10), tunnel_quiet=True, now=30)  # calm, but not for long enough
    assert not guard.update_yield(rules, sample(10), tunnel_quiet=True, now=50)
    assert guard.evaluate(rules, paused=False, now=at("12:00")).state == "lending"


def test_load_while_the_hub_uses_the_tunnel_is_ours():
    rules, guard = LendingRules(yield_seconds=5), LendingGuard()
    for t in range(0, 60, 2):
        assert not guard.update_yield(rules, sample(100), tunnel_quiet=False, now=t)


def test_a_short_spike_resets():
    rules, guard = LendingRules(yield_seconds=10), LendingGuard()
    guard.update_yield(rules, sample(90), True, now=0)
    guard.update_yield(rules, sample(5), True, now=8)
    assert not guard.update_yield(rules, sample(90), True, now=12)
    assert guard.update_yield(rules, sample(90), True, now=22)


def test_only_lent_gpus_count_and_unknown_util_never_yields():
    rules, guard = LendingRules(gpus=[1], yield_seconds=2), LendingGuard()
    for t in range(10):
        assert not guard.update_yield(rules, sample(100, index=0), True, now=t)  # GPU 0 isn't lent
        assert not guard.update_yield(rules, sample(None, index=1), True, now=t)  # no utilisation reported


def test_lent_gpus_missing_on_this_pc_is_not_lending():
    rules, guard = LendingRules(gpus=[5], auto_yield=False), LendingGuard()
    assert guard.evaluate(rules, paused=False).state == "lending"  # no telemetry yet: can't tell
    guard.update_yield(rules, sample(0, index=0), True, now=0)
    d = guard.evaluate(rules, paused=False)
    assert d.state == "paused" and "GPU 5" in d.reason and "GPUs are: 0" in d.reason
    assert guard.evaluate(LendingRules(gpus=[5], allow_cpu=True), paused=False).state == "lending"  # RAM still lent
    assert guard.evaluate(LendingRules(gpus=[0, 5]), paused=False).state == "lending"


def test_auto_yield_off():
    rules, guard = LendingRules(auto_yield=False, yield_seconds=2), LendingGuard()
    for t in range(10):
        assert not guard.update_yield(rules, sample(100), True, now=t)


def test_precedence_pause_then_schedule_then_yield():
    rules, guard = LendingRules(schedule=["22:00-08:00"], yield_seconds=2), LendingGuard()
    guard.update_yield(rules, sample(100), True, now=0)
    guard.update_yield(rules, sample(100), True, now=5)
    assert guard.evaluate(rules, paused=True, now=at("23:00")).reason == "paused by the helper's owner"
    d = guard.evaluate(rules, paused=False, now=at("12:00"))
    assert d.state == "paused" and "outside the lending schedule (22:00-08:00)" in d.reason
    assert guard.evaluate(rules, paused=False, now=at("23:00")).state == "yielding"


@pytest.mark.parametrize("field,value", [("yield_gpu_percent", 0), ("yield_gpu_percent", 101), ("yield_seconds", 1)])
def test_yield_rules_are_validated(field, value):
    with pytest.raises(ConfigError):
        LendingRules(**{field: value}).validate()


# ----- with a hub and a helper ---------------------------------------------------------

e2e = pytest.mark.skipif(sys.platform == "win32", reason="fake llama.cpp programs are POSIX scripts")


@pytest.fixture
def cluster(tmp_path, fake_ollama, make_config):
    c, close = make_cluster(tmp_path, fake_ollama, make_config, hub_devices=[], yield_seconds=2)
    c.hub.config.planner.auto_benchmark = False
    yield c
    close()


@e2e
def test_outside_the_schedule_the_hub_does_not_use_the_helper(cluster):
    c = cluster
    now = datetime.datetime.now()
    later = (now + datetime.timedelta(hours=2)).strftime("%H:%M")
    much_later = (now + datetime.timedelta(hours=3)).strftime("%H:%M")
    c.run(c.agent.update_rules(schedule=[f"{later}-{much_later}"]))
    assert wait_for(lambda: c.node.status == "paused", 10)
    assert "outside the lending schedule" in c.node.lend_reason
    c.admin("models.pin", model="tiny", engine="split")
    with httpx.Client(timeout=30, trust_env=False) as cl:
        chat(cl, c)
    assert c.rec.last["path"] == "/api/chat"  # Ollama
    c.run(c.agent.update_rules(schedule=[]))
    assert wait_for(lambda: c.node.status == "lending", 10)


@e2e
def test_auto_yield_drains_the_answer_then_steps_back_and_returns(cluster):
    c = cluster
    c.admin("models.pin", model="tiny", engine="split")
    with httpx.Client(timeout=60, trust_env=False) as cl:
        assert chat(cl, c).status_code == 200  # the split session is up
        result = {}

        def slow_answer() -> None:
            # /v1 passes straight through, so the fake llama-server sees fake_delay: ~6 s of streaming.
            r = cl.post(
                c.url + "/v1/chat/completions",
                json={"model": "tiny", "messages": [{"role": "user", "content": "x"}], "stream": True, "fake_delay": 2},
            )
            result["text"] = r.text

        worker = threading.Thread(target=slow_answer)
        worker.start()
        assert wait_for(lambda: c.call(lambda: c.hub.split.session.active) == 1, 5)
        c.agent.telemetry.util = 95  # the helper's owner starts a game
        assert wait_for(lambda: c.node.status == "yielding", 15)
        assert "the helper's owner is using GPU 0" in c.node.lend_reason
        worker.join(30)
        assert result["text"].rstrip().endswith("data: [DONE]")  # the answer in flight finished
        assert wait_for(lambda: not c.call(lambda: c.hub.split.is_running("tiny")), 15)
        assert wait_for(lambda: c.agent.rpc.session is None, 10)  # released by the hub, not killed
        events = c.call(lambda: [e["kind"] for e in c.hub.recent_events])
        assert "split_drain" in events and "split_failed" not in events
        assert c.call(lambda: c.hub.plans.record("tiny")["degraded"]) is None  # planned, not a failure

        chat(cl, c)
        assert c.rec.last["path"] == "/api/chat"  # Ollama while the helper is busy

        c.agent.telemetry.util = 5  # the game is over
        assert wait_for(lambda: c.node.status == "lending", 15)
        assert chat(cl, c).status_code == 200
        assert c.call(lambda: c.hub.split.is_running("tiny"))  # split again, no new benchmark needed


@e2e
def test_hub_pause_drains_too(cluster):
    c = cluster
    c.admin("models.pin", model="tiny", engine="split")
    with httpx.Client(timeout=30, trust_env=False) as cl:
        assert chat(cl, c).status_code == 200
    c.admin("nodes.pause", ref="desk")
    assert wait_for(lambda: not c.call(lambda: c.hub.split.is_running("tiny")), 10)
    assert c.call(lambda: c.hub.plans.record("tiny")["degraded"]) is None


@e2e
def test_agent_status_shows_why_it_is_not_lending(cluster):
    from ollamapower.agent.service import agent_snapshot

    c = cluster
    c.agent.telemetry.util = 99
    assert wait_for(lambda: c.agent.state.lend_state == "yielding", 15)
    snap = agent_snapshot(c.agent)
    assert snap["lend_state"] == "yielding" and "using GPU 0" in snap["lend_reason"]
    assert json.dumps(snap)
