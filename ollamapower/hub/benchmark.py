"""Benchmark a model on plain Ollama and on split candidates (spec Section 8.2).

The workload is the same for every engine: a ~``planner.prompt_tokens``
prompt (with a random first line so no prompt cache helps), then
``planner.gen_tokens`` tokens at temperature 0 with a fixed seed. Measured:
load time, time to first token, prompt and generation tokens/s.

Requests go through the hub's own front door, so they take the same path as
real traffic (queue, translation, tunnel). Two headers steer them, and only
work together with ``X-OllamaPower-Bench``, a random secret the hub makes at
start-up: no client can force an engine.

Candidates: every eligible helper (best first); when some helpers lend only
CPU, the GPU helpers alone are tried first (CPU helpers last, spec 8.2).
"""

from __future__ import annotations

import json
import secrets
import time
from typing import TYPE_CHECKING, Any

import httpx

from ..aio import wait_for
from ..textsafe import safe_text
from .plans import KEY_VERSION, TRANSIENT

if TYPE_CHECKING:
    from .runtime import Hub

BENCH_HEADER = "x-ollamapower-bench"
ENGINE_HEADER = "x-ollamapower-engine"
HELPERS_HEADER = "x-ollamapower-helpers"
CTX_HEADER = "x-ollamapower-ctx"
SEED = 42
IDLE_WAIT = 120.0  # seconds a benchmark waits for a busy split session
WORDS = (
    "The history of computing is a story of machines that learned to share work. Early mainframes "
    "divided their time between many users, networks later joined distant computers, and clusters "
    "spread one large problem over many small machines. Each step asked the same questions: how to "
    "split the work, how to move the data, and how to keep the parts in step."
).split()


class BenchmarkError(Exception):
    """The model can't be benchmarked (the message says why)."""


def make_prompt(tokens: int) -> str:
    """About ``tokens`` tokens of plain English (≈0.75 words per token), with a unique first line."""
    words = [WORDS[i % len(WORDS)] for i in range(int(tokens * 0.75))]
    return f"Benchmark {secrets.token_hex(6)}.\n" + " ".join(words) + "\n\nSummarize the text above."


def headers(hub: Hub, engine: str, ctx: int, helpers: tuple[str, ...] = ()) -> dict[str, str]:
    out = {BENCH_HEADER: hub.bench_secret, ENGINE_HEADER: engine, CTX_HEADER: str(ctx)}
    if helpers:
        out[HELPERS_HEADER] = ",".join(helpers)
    return out


async def run_workload(hub: Hub, model: str, ctx: int, engine: str, helpers: tuple[str, ...] = ()) -> dict[str, Any]:
    """One measured run on ``engine`` ("local" or "split"). Never raises for engine failures."""
    cfg = hub.config
    result: dict[str, Any] = {"engine": engine, "ok": False, "error": ""}
    h = headers(hub, engine, ctx, helpers)
    timeout = httpx.Timeout(connect=5.0, read=cfg.split.load_timeout, write=60.0, pool=cfg.hub.queue_timeout)
    async with httpx.AsyncClient(base_url=cfg.frontdoor_url, timeout=timeout, trust_env=False) as client:
        try:
            t = time.perf_counter()
            r = await client.post(
                "/api/generate", json={"model": model, "keep_alive": "10m", "options": {"num_ctx": ctx}}, headers=h
            )
            if r.status_code != 200:
                result["error"] = _error_text(r.content, r.status_code)
                return result
            result["load_s"] = round(time.perf_counter() - t, 2)
            body = {
                "model": model,
                "prompt": make_prompt(cfg.planner.prompt_tokens),
                "stream": True,
                "keep_alive": "10m",
                "options": {"temperature": 0, "seed": SEED, "num_predict": cfg.planner.gen_tokens, "num_ctx": ctx},
            }
            t = time.perf_counter()
            ttft = None
            final: dict[str, Any] | None = None
            async with client.stream("POST", "/api/generate", json=body, headers=h) as r:
                if r.status_code != 200:
                    result["error"] = _error_text(await r.aread(), r.status_code)
                    return result
                async for line in r.aiter_lines():
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    if "error" in obj:
                        result["error"] = safe_text(str(obj["error"]), 300)
                        return result
                    if ttft is None and (obj.get("response") or obj.get("thinking")):
                        ttft = time.perf_counter() - t
                    if obj.get("done"):
                        final = obj
        except (httpx.HTTPError, ValueError) as exc:
            result["error"] = f"{type(exc).__name__}"
            return result
    if final is None:
        result["error"] = "the answer ended without a final object"
        return result
    prompt_n, prompt_ns = int(final.get("prompt_eval_count") or 0), int(final.get("prompt_eval_duration") or 0)
    gen_n, gen_ns = int(final.get("eval_count") or 0), int(final.get("eval_duration") or 0)
    if not gen_n or not gen_ns:
        result["error"] = "no tokens were generated"
        return result
    result.update(
        ok=True,
        ttft_ms=round((ttft if ttft is not None else time.perf_counter() - t) * 1000, 1),
        prompt_tokens=prompt_n,
        gen_tokens=gen_n,
        prompt_tps=round(prompt_n / (prompt_ns / 1e9), 1) if prompt_ns else None,
        gen_tps=round(gen_n / (gen_ns / 1e9), 2),
    )
    return result


def _error_text(raw: bytes, status: int) -> str:
    try:
        err = json.loads(raw).get("error")
    except (ValueError, AttributeError):
        err = None
    if isinstance(err, dict):
        err = err.get("message")
    return safe_text(str(err or f"HTTP {status}"), 300)


def candidate_sets(hub: Hub, model: str) -> list[tuple[str, ...]]:
    helpers = hub.plans.current_helpers(model)
    if not helpers:
        return []
    everyone = tuple(n.node_id for n in helpers)
    gpu = tuple(n.node_id for n in helpers if n.lendable_vram() > 0)
    return [gpu, everyone] if gpu and gpu != everyone else [everyone]


async def benchmark_model(hub: Hub, model: str, ctx: int | None = None, automatic: bool = False) -> dict[str, Any]:
    """Benchmark ``model`` locally and on each split candidate, then decide (unless pinned)."""
    from .runtime import model_key

    plans = hub.plans
    model = model_key(model)
    async with plans.bench_lock:
        facts = await plans.facts(model, refresh=True)
        if not facts.get("splittable"):
            raise BenchmarkError(facts.get("reason") or f"{model} can't run in split mode")
        sets = candidate_sets(hub, model)
        if not sets:
            raise BenchmarkError("no helper can take part right now")
        ctx = ctx or plans.ctx_for(model)
        digest = plans.digest_of(model)
        started = time.time()
        plans.benchmarking = {"model": model, "stage": "local", "started": started, "automatic": automatic}
        hub.event("benchmark_start", model=model, ctx=ctx, automatic=automatic)
        try:
            session = hub.split.session
            if session is not None and session.active:
                try:  # don't cut off anyone's answer: wait for the running split session to go idle
                    await wait_for(session.idle.wait(), IDLE_WAIT)
                except TimeoutError:
                    raise BenchmarkError("split mode is busy serving requests; try again later") from None
            await hub.split.stop(reason="benchmark")
            if facts.get("too_big"):
                # Plain Ollama would page the model from disk for every token: minutes per answer,
                # nothing worth measuring. Recorded as Ollama not running it, so split decides alone.
                local = {
                    "engine": "local",
                    "ok": False,
                    "skipped": True,
                    "error": "too big for this PC alone (Ollama would read it from disk, about "
                    f"{facts.get('predicted_local_tps', 0)} tok/s)",
                }
            else:
                local = await run_workload(hub, model, ctx, "local")
            _store(hub, model, digest, ctx, local, None)
            splits = []
            for i, ids in enumerate(sets, 1):
                plans.benchmarking = {**plans.benchmarking, "stage": f"split {i}/{len(sets)}"}
                began = time.monotonic()
                result = await run_workload(hub, model, ctx, "split", ids)
                if not result["ok"]:
                    # Why split mode didn't start (a helper stepping away is no failure); else: transient.
                    kind, at = hub.split.last_failure.get(model, (TRANSIENT, 0.0))
                    result["failure"] = kind if at >= began else TRANSIENT
                session = hub.split.session
                key = None
                if result["ok"] and session is not None:
                    layout = session.layout()
                    result["helpers"] = layout["helpers"]
                    key = {"v": KEY_VERSION, "digest": digest, "ctx": ctx, "build": plans.pins.build, **layout}
                    result["key"] = key
                    if session.plan is not None:
                        result["placement"] = session.plan.summary()
                _store(hub, model, digest, ctx, result, key)
                splits.append(result)
                await hub.split.stop(reason="benchmark")
        finally:
            plans.benchmarking = None
        if any(s.get("ok") for s in splits):
            plans.bench_succeeded(model)  # automatic benchmarks may run again (after a manual one, too)
        out: dict[str, Any] = {"model": model, "ctx": ctx, "local": local, "splits": splits, "decided": False}
        if ctx != plans.ctx_for(model):
            plans._update(model, ctx=ctx)  # measured at this context: that's the context the plan runs at
        if plans.record(model)["pin"] is None:
            rec = plans.decide(model, local, splits)
            out.update(decided=True, choice=rec["choice"], reason=rec["reason"])
        else:
            out["reason"] = f"pinned to {plans.record(model)['pin']}: the result is kept, the pin still decides"
        hub.event("benchmark_done", model=model, seconds=round(time.time() - started, 1), choice=out.get("choice"))
        return out


def _store(hub: Hub, model: str, digest: str, ctx: int, result: dict[str, Any], key: dict[str, Any] | None) -> None:
    if hub.state is None:
        return
    entry = {
        "model": model,
        "model_digest": digest or model,
        "ctx": ctx,
        "llama_cpp": hub.plans.pins.build if result["engine"] == "split" else None,
        "ts": time.time(),
        **{k: v for k, v in result.items() if k != "key"},
    }
    if key is not None:
        entry["tunnel_modes"] = sorted({h["tunnel"] for h in key["helpers"]})
    hub.state.add_benchmark(entry)
