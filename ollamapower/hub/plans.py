"""Which engine serves each model: the plan book (spec Section 8).

Every model has a record in ``state.json["plans"]`` (keyed by the model's
canonical name):

* ``pin``: ``None`` (automatic, the default), ``"split"`` or ``"local"`` -
  the owner's override (``models pin`` / ``force-local`` / ``unpin``).
* ``choice``: what the last benchmark decided for automatic mode, with the
  ``reason`` and the ``key`` it was measured under.
* ``exclude``: helpers the owner doesn't want used for this model.
* ``degraded``: set when split mode failed; Ollama serves the model until
  ``planner.retry_after`` has passed *and* every helper of the plan is back.

**Golden rule - never slower.** In automatic mode a model runs split only if
a benchmark showed split generating at least ``planner.min_gain`` faster than
plain Ollama, and only while nothing that benchmark depended on has changed:
the model's digest, the context size, the llama.cpp build and the exact set
of helpers with their tunnel modes (the benchmark *key*). If anything differs
the model goes back to Ollama and is benchmarked again (automatically while
the hub is idle, if ``planner.auto_benchmark`` is on).

Memory facts (does the model fit the hub's VRAM?) come from the GGUF header
plus a margin, and from Ollama's own ``/api/ps`` (a loaded model with
``size_vram < size`` spilled to the CPU). Only models that don't fit, or
spilled, are worth benchmarking.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..aio import reap
from ..llamacpp.install import Pins
from ..ollama.blobs import ModelUnavailable, resolve
from ..ollama.gguf import GGUFInfo
from ..textsafe import safe_text
from .planner import fits_on_hub, hub_need

if TYPE_CHECKING:
    from .runtime import Hub

log = logging.getLogger("ollamapower.planner")

PINS = ("split", "local")
AUTO_CHECK = 30.0  # seconds between looks for something to benchmark
RECENT_USE = 24 * 3600.0  # only models used in the last day are benchmarked automatically
FAILED_BENCH_BACKOFF = 3600.0  # a benchmark that failed isn't retried with the same key for an hour
WARNING_TTL = 600.0


@dataclass(frozen=True)
class SplitSpec:
    """How to start a split session: the context and which helpers (None = every eligible one)."""

    ctx: int
    helpers: tuple[str, ...] | None = None
    exclude: tuple[str, ...] = ()


def _default_record() -> dict[str, Any]:
    return {
        "pin": None,
        "ctx": None,
        "exclude": [],
        "choice": None,
        "reason": "",
        "chosen_at": None,
        "key": None,
        "degraded": None,
    }


class PlanBook:
    def __init__(self, hub: Hub) -> None:
        self.hub = hub
        self.config = hub.config
        self.facts_cache: dict[str, dict[str, Any]] = {}  # digest -> facts
        self.gguf_cache: dict[str, GGUFInfo] = {}  # digest -> header (the KV cache depends on the context)
        self.last_request: dict[str, float] = {}  # model -> time.time() of its last client request
        self.last_activity = time.monotonic()
        self.bench_lock = asyncio.Lock()
        self.auto_task: asyncio.Task[Any] | None = None
        self.benchmarking: dict[str, Any] | None = None  # what runs now, for the views
        self._failed_keys: dict[str, float] = {}
        self._warnings: deque[tuple[float, str]] = deque(maxlen=20)
        self._pins: Pins | None = None

    @property
    def pins(self) -> Pins:
        if self._pins is None:
            self._pins = Pins.load()
        return self._pins

    # ----- records ---------------------------------------------------------------------

    def _plans(self) -> dict[str, Any]:
        if self.hub.state is None:
            return {}
        return self.hub.state.data.setdefault("plans", {})

    def record(self, model: str) -> dict[str, Any]:
        """A copy of ``model``'s record with every field present."""
        from .runtime import model_key

        rec = self._plans().get(model_key(model))
        out = _default_record()
        if isinstance(rec, dict):
            out.update({k: v for k, v in rec.items() if k in out})
            if rec.get("mode") == "split" and out["pin"] is None:  # a Phase 3 `models plan --split`
                out["pin"] = "split"
        return out

    def _update(self, model: str, **changes: Any) -> dict[str, Any]:
        from .runtime import model_key

        if self.hub.state is None:
            raise RuntimeError("the hub has no state file")
        key = model_key(model)
        rec = self.record(key)
        rec.update(changes)
        self._plans()[key] = rec
        self.hub.state.save_soon()
        return rec

    def records(self) -> dict[str, dict[str, Any]]:
        return {name: self.record(name) for name in self._plans()}

    def pin(self, model: str, engine: str | None, ctx: int | None = None) -> dict[str, Any]:
        if engine is not None and engine not in PINS:
            raise ValueError(f"pin must be one of {', '.join(PINS)}")
        changes: dict[str, Any] = {"pin": engine, "degraded": None}
        if ctx is not None:
            changes["ctx"] = ctx
        rec = self._update(model, **changes)
        self.hub.event("plan_pinned", model=model, pin=engine or "auto", ctx=rec["ctx"])
        return rec

    def set_excluded(self, model: str, node_id: str, excluded: bool) -> dict[str, Any]:
        current = list(self.record(model)["exclude"])
        if excluded and node_id not in current:
            current.append(node_id)
        if not excluded and node_id in current:
            current.remove(node_id)
        rec = self._update(model, exclude=current)
        self.hub.event("plan_exclude", model=model, node=node_id, excluded=excluded)
        return rec

    def remove(self, model: str) -> bool:
        from .runtime import model_key

        if self._plans().pop(model_key(model), None) is None:
            return False
        if self.hub.state is not None:
            self.hub.state.save_soon()
        self.hub.event("plan_removed", model=model_key(model))
        return True

    def ctx_for(self, model: str) -> int:
        return int(self.record(model)["ctx"] or self.config.split.default_ctx)

    # ----- routing -----------------------------------------------------------------------

    def digest_of(self, model: str) -> str:
        from .runtime import model_key

        key = model_key(model)
        for m in self.hub.ollama.installed:
            if model_key(m.name) == key:
                return m.digest
        return ""

    def current_helpers(self, model: str) -> list[Any]:
        """Eligible helpers for ``model`` now (best first), the owner's exclusions removed."""
        excluded = set(self.record(model)["exclude"])
        helpers, _ = self.hub.split.candidates()
        return [n for n in helpers if n.node_id not in excluded]

    def key_problem(self, model: str, rec: dict[str, Any]) -> str:
        """Why the benchmark behind ``rec`` no longer applies ("" if it still does)."""
        key = rec.get("key")
        if not isinstance(key, dict):
            return "no benchmark yet"
        digest = self.digest_of(model)
        if digest and key.get("digest") and digest != key["digest"]:
            return "the model changed since its benchmark"
        if key.get("ctx") != self.ctx_for(model):
            return "the context size changed"
        if key.get("build") != self.pins.build:
            return "llama.cpp changed"
        eligible = {n.node_id: n for n in self.current_helpers(model)}
        for h in key.get("helpers") or []:
            node = eligible.get(h.get("id"))
            if node is None:
                return f"helper {h.get('name', h.get('id', '?'))} is not available"
            if node.tunnel_effective != h.get("tunnel"):
                return f"helper {node.name}'s tunnel mode changed"
        return ""

    def route(self, model: str) -> SplitSpec | None:
        """The split session to use for ``model``'s next request, or None for plain Ollama."""
        if not model or self.hub.state is None:
            return None
        rec = self.record(model)
        if rec["pin"] == "local":
            return None
        exclude = tuple(rec["exclude"])
        ctx = self.ctx_for(model)
        degraded = rec["degraded"]
        if isinstance(degraded, dict):
            if time.time() < float(degraded.get("until", 0)):
                return None
            if rec["pin"] != "split" and self.key_problem(model, rec):
                return None  # a plan's helper is still missing; keep serving locally
            self._update(model, degraded=None)
            self.hub.event("plan_recovered", model=model)
        if rec["pin"] == "split":
            if not self.current_helpers(model):
                return None  # every helper is away (paused, yielding, outside its schedule): not a failure
            return SplitSpec(ctx=ctx, exclude=exclude)
        if rec["choice"] == "split" and not self.key_problem(model, rec):
            ids = tuple(h["id"] for h in rec["key"]["helpers"])
            return SplitSpec(ctx=ctx, helpers=ids, exclude=exclude)
        return None

    def fail(self, model: str, reason: str, tail: list[str] | None = None) -> None:
        """Split mode failed for ``model``: Ollama serves it until the plan can recover."""
        until = time.time() + self.config.planner.retry_after
        if self.hub.state is not None:
            self._update(model, degraded={"reason": safe_text(reason, 300), "at": time.time(), "until": until})
        message = (
            f"split mode for {model} failed: {reason}; Ollama serves it "
            f"(split is retried after {int(self.config.planner.retry_after)} s, once its helpers are back)"
        )
        self._warn(message)
        fields: dict[str, Any] = {"model": model, "reason": safe_text(reason, 300)}
        if tail:
            fields["tail"] = [safe_text(line, 200) for line in tail[-5:]]
        self.hub.event("split_failed", **fields)
        log.warning("%s", message)

    def _warn(self, message: str) -> None:
        self._warnings.append((time.monotonic(), message))

    # ----- facts ---------------------------------------------------------------------------

    async def facts(self, model: str, refresh: bool = False) -> dict[str, Any]:
        """Size, layers and whether ``model`` fits the hub's VRAM (cached per model digest)."""
        digest = self.digest_of(model)
        cache_key = digest or model
        if not refresh and cache_key in self.facts_cache:
            return self._with_vram(self.facts_cache[cache_key], model)
        base: dict[str, Any] = {"splittable": False, "reason": ""}
        http = self.hub.split.http
        if http is None:
            base["reason"] = "the front door is not running"
            return self._with_vram(base, model)
        try:
            files = await resolve(http, model)
        except ModelUnavailable as exc:
            base["reason"] = str(exc)
        else:
            info = files.gguf
            assert info is not None
            base.update(
                splittable=True,
                size=info.file_size,
                weights=info.weights_bytes,
                layers=info.block_count,
                architecture=info.architecture,
                vision=files.vision,
            )
            self.gguf_cache[cache_key] = info
        self.facts_cache[cache_key] = base
        return self._with_vram(base, model)

    def _with_vram(self, base: dict[str, Any], model: str) -> dict[str, Any]:
        """Add the live verdict: needed memory at the plan's context vs the hub's VRAM, and spill."""
        from .runtime import model_key

        out = dict(base)
        hub = self.hub.split.hub_devices()
        out["hub_vram"] = sum(d.free for d in hub)
        ctx = self.ctx_for(model)
        info = self.gguf_cache.get(self.digest_of(model) or model)
        if base.get("splittable") and info is not None:
            margin = self.config.planner.memory_margin
            out["kv"] = info.kv_bytes_total(ctx)
            out["needed"] = hub_need(info, ctx, margin)
            out["fits"] = fits_on_hub(info, ctx, hub, margin)  # the same check split mode starts with
        else:
            out["fits"] = None
        loaded = next((m for m in self.hub.ollama.models if model_key(m.name) == model_key(model)), None)
        out["spilled"] = bool(loaded and loaded.spilled)
        out["worth_splitting"] = bool(base.get("splittable")) and (out["fits"] is False or out["spilled"])
        return out

    # ----- decision ------------------------------------------------------------------------

    def decide(self, model: str, local: dict[str, Any], splits: list[dict[str, Any]]) -> dict[str, Any]:
        """Pick local or split from benchmark results; store it. Never slower: split must win by min_gain."""
        ok_splits = [s for s in splits if s.get("ok")]
        best = max(ok_splits, key=lambda s: s["gen_tps"], default=None)
        gain = self.config.planner.min_gain
        if best is None:
            choice, reason = "local", "split mode didn't work" + (f": {splits[-1].get('error')}" if splits else "")
        elif not local.get("ok"):
            choice, reason = "split", f"Ollama couldn't run it ({local.get('error', '?')}); split runs it"
        elif best["gen_tps"] >= local["gen_tps"] * (1 + gain):
            choice = "split"
            reason = (
                f"split {best['gen_tps']:.1f} tok/s vs Ollama {local['gen_tps']:.1f} tok/s "
                f"(+{(best['gen_tps'] / max(local['gen_tps'], 1e-9) - 1) * 100:.0f} %)"
            )
        else:
            choice = "local"
            reason = (
                f"Ollama {local['gen_tps']:.1f} tok/s vs split {best['gen_tps']:.1f} tok/s: "
                f"split isn't at least {gain * 100:.0f} % faster"
            )
        key = best["key"] if best is not None else None
        rec = self._update(model, choice=choice, reason=reason, chosen_at=time.time(), key=key, degraded=None)
        self.hub.event("plan_chosen", model=model, choice=choice, reason=reason)
        return rec

    # ----- automatic benchmarks --------------------------------------------------------------

    def note_request(self, model: str) -> None:
        from .runtime import model_key

        self.last_activity = time.monotonic()
        if model:
            self.last_request[model_key(model)] = time.time()
        task = self.auto_task
        if task is not None and not task.done():
            task.cancel()  # a client wants the hub: an automatic benchmark gives way at once

    async def pending(self) -> list[str]:
        """Models an automatic benchmark should look at now."""
        out = []
        now = time.time()
        for model, used in sorted(self.last_request.items(), key=lambda x: -x[1]):
            if now - used > RECENT_USE:
                continue
            rec = self.record(model)
            if rec["pin"] is not None:
                continue
            problem = self.key_problem(model, rec)
            if rec["choice"] is not None and not problem:
                continue  # decided under the current conditions
            if not self.current_helpers(model):
                continue
            facts = await self.facts(model)
            if not facts["worth_splitting"]:
                continue
            if now - self._failed_keys.get(self._attempt_key(model), 0.0) < FAILED_BENCH_BACKOFF:
                continue
            out.append(model)
        return out

    def _attempt_key(self, model: str) -> str:
        ids = ",".join(sorted(f"{n.node_id}/{n.tunnel_effective}" for n in self.current_helpers(model)))
        return f"{model}|{self.digest_of(model)}|{self.ctx_for(model)}|{ids}"

    async def run(self) -> None:
        """Benchmark models that would benefit, while the hub is idle (if ``planner.auto_benchmark``)."""
        from .benchmark import benchmark_model

        while True:
            await asyncio.sleep(AUTO_CHECK)
            if not self.config.planner.auto_benchmark or self.bench_lock.locked():
                continue
            if self.hub.streams or time.monotonic() - self.last_activity < self.config.planner.idle_seconds:
                continue
            try:
                models = await self.pending()
            except Exception as exc:  # noqa: BLE001 - the planner must never stop the hub
                log.warning("planner check failed: %s", type(exc).__name__)
                continue
            if not models:
                continue
            model = models[0]
            attempt = self._attempt_key(model)
            self.auto_task = asyncio.ensure_future(benchmark_model(self.hub, model, automatic=True))
            try:
                result = await self.auto_task
                if not result.get("decided"):
                    self._failed_keys[attempt] = time.time()
            except asyncio.CancelledError:
                current = asyncio.current_task()
                if current is not None and current.cancelling():
                    raise
                self.hub.event("benchmark_abandoned", model=model, reason="a request arrived")
            except Exception as exc:  # noqa: BLE001 - one failed benchmark must not stop the planner
                log.warning("automatic benchmark of %s failed: %s", model, type(exc).__name__)
                self._failed_keys[attempt] = time.time()
            finally:
                self.auto_task = None

    async def stop(self) -> None:
        if self.auto_task is not None:
            await reap(self.auto_task)

    # ----- views -----------------------------------------------------------------------------

    def benchmarks_for(self, model: str) -> list[dict[str, Any]]:
        from .runtime import model_key

        if self.hub.state is None:
            return []
        return [b for b in self.hub.state.data.get("benchmarks", []) if b.get("model") == model_key(model)]

    def latest(self, model: str) -> dict[str, dict[str, Any] | None]:
        """The newest successful local and split benchmark of ``model``."""
        out: dict[str, dict[str, Any] | None] = {"local": None, "split": None}
        for b in self.benchmarks_for(model):
            if b.get("ok") and b.get("engine") in out:
                out[b["engine"]] = b
        return out

    def describe(self, model: str) -> dict[str, Any]:
        """What the views show for one model: the effective engine and why."""
        rec = self.record(model)
        spec = self.route(model)
        problem = self.key_problem(model, rec) if rec["choice"] == "split" else ""
        if rec["pin"] == "local":
            why = "forced local by the owner"
        elif isinstance(rec["degraded"], dict):
            why = f"degraded: {rec['degraded'].get('reason', '')}"
        elif rec["pin"] == "split" and spec is None:
            _, away = self.hub.split.candidates()
            why = "pinned to split, but no helper can take part now" + (f" ({'; '.join(away)})" if away else "")
        elif rec["pin"] == "split":
            why = "pinned to split by the owner"
        elif rec["choice"] == "split" and problem:
            why = f"benchmark no longer applies ({problem})"
        else:
            why = rec["reason"] or "no benchmark yet"
        latest = self.latest(model)
        return {
            "engine": "split" if spec is not None else "local-ollama",
            "mode": rec["pin"] or "auto",
            "choice": rec["choice"],
            "why": why,
            "ctx": self.ctx_for(model),
            "exclude": rec["exclude"],
            "degraded": rec["degraded"],
            "local_tps": latest["local"]["gen_tps"] if latest["local"] else None,
            "split_tps": latest["split"]["gen_tps"] if latest["split"] else None,
        }

    def warnings(self) -> list[str]:
        now = time.monotonic()
        return list(dict.fromkeys(m for t, m in self._warnings if now - t < WARNING_TTL))

    def snapshot(self) -> dict[str, Any]:
        return {
            "records": self.records(),
            "benchmarking": self.benchmarking,
            "auto_benchmark": self.config.planner.auto_benchmark,
        }
