"""Hub admin commands, shared by the CLI (over the admin channel) and the hub TUI.

Every owner action goes through ``dispatch`` so there is one code path to
audit; every action that changes something is recorded as an event.
Commands are dotted ``noun.verb`` strings, like OllamaGate's.
"""

from __future__ import annotations

import time
from typing import Any

from ..hub.control_server import NetTestError
from ..hub.nodes import TUNNEL_MODES, Node
from ..hub.planner import MIB, Device, fits_on_hub
from ..hub.runtime import Hub, model_key
from ..ollama.blobs import ModelUnavailable, resolve
from .ipc import AdminError

MAX_CTX = 1 << 20

# Estimated overhead over the model file when deciding whether it fits in VRAM
# (KV cache and buffers). Phase 5 replaces this with a real estimate from GGUF metadata.
FIT_MARGIN = 1.2


class Admin:
    def __init__(self, hub: Hub) -> None:
        self.hub = hub

    def _node(self, ref: Any) -> Node:
        node = self.hub.registry.find(str(ref)) if isinstance(ref, str) and ref else None
        if node is None:
            raise AdminError(f"no helper {ref!r} (use its name, id or the first 8 characters of the id)")
        return node

    # ----- views -----------------------------------------------------------------------

    def models(self) -> list[dict[str, Any]]:
        """Installed models: size, whether they fit the hub's VRAM, the engine that serves them and why."""
        hub_vram = sum(g.get("vram_total", 0) for g in self.hub.local_static.get("gpus", []))
        loaded = {model_key(m.name): m for m in self.hub.ollama.models}
        plans = self.hub.plans
        out = []
        for m in self.hub.ollama.installed:
            key = model_key(m.name)
            live = loaded.get(key)
            gen = self.hub.last_generation.get(key)
            facts = plans.facts_cache.get(m.digest)
            if facts is not None:
                fits = plans._with_vram(facts, key)["fits"]
            else:
                fits = None if not hub_vram else int(m.size * FIT_MARGIN) <= hub_vram
            out.append(
                {
                    "name": m.name,
                    "size": m.size,
                    "parameters": m.parameter_size,
                    "quantization": m.quantization,
                    "digest": m.digest,
                    "loaded": live is not None,
                    "vram_percent": live.vram_percent if live else None,
                    "spilled": bool(live and live.spilled),
                    "fits": fits,
                    "fits_estimated": facts is not None,
                    **{f"plan_{k}": v for k, v in plans.describe(key).items()},
                    "plan": plans.describe(key)["engine"],
                    "running_split": self.hub.split.is_running(key),
                    "last_gen_tps": round(gen.gen_tps, 1) if gen else None,
                }
            )
        return out

    def _model(self, value: Any) -> str:
        if not isinstance(value, str) or not value or len(value) > 200:
            raise AdminError("give a model name, e.g. llama3.1:70b")
        return model_key(value)

    def _ctx(self, value: Any) -> int | None:
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or not 256 <= value <= MAX_CTX):
            raise AdminError(f"--ctx must be between 256 and {MAX_CTX}")
        return value

    async def show_model(self, model: str) -> dict[str, Any]:
        plans = self.hub.plans
        return {
            "model": model,
            **plans.describe(model),
            "record": plans.record(model),
            "facts": await plans.facts(model),
            "benchmarks": plans.benchmarks_for(model)[-10:],
            "helpers_now": [n.name for n in plans.current_helpers(model)],
        }

    async def tunnel_test(self, node: Node) -> dict[str, Any]:
        """Measure the split-mode path to ``node``: secure always, direct too if its owner allows it."""
        from ..core.protocol import protocol_minor
        from ..hub.engines.split import SplitUnavailable

        if protocol_minor(node.protocol) < 1:
            raise AdminError(f"{node.name} runs an older agent without tunnel tests; update OllamaPower there")
        modes = ["secure"] + (["direct"] if node.lending.get("allow_direct") else [])
        results = {}
        try:
            for mode in modes:
                results[mode] = await self.hub.split.bench(node, mode)
        except SplitUnavailable as exc:
            raise AdminError(str(exc)) from None
        out: dict[str, Any] = {"ts": time.time(), "node": node.name, **results}
        secure, direct = results.get("secure"), results.get("direct")
        if secure and direct and direct["mbps"] > 0:
            out["overhead_pct"] = round((1 - secure["mbps"] / direct["mbps"]) * 100, 1)
            out["rtt_overhead_ms"] = round(secure["rtt_ms"] - direct["rtt_ms"], 3)
        self.hub.registry.record_tunnel_test(node, out)
        return out

    async def plan_split(self, model: Any, ctx: Any) -> dict[str, Any]:
        """Check that ``model`` can run split, save the plan and preview it."""
        hub, split = self.hub, self.hub.split
        if not isinstance(model, str) or not model or len(model) > 200:
            raise AdminError("give a model name, e.g. llama3.1:70b")
        if ctx is not None and (not isinstance(ctx, int) or isinstance(ctx, bool) or not 256 <= ctx <= MAX_CTX):
            raise AdminError(f"--ctx must be between 256 and {MAX_CTX}")
        if hub.state is None or split.http is None:
            raise AdminError("the hub is not fully running")
        try:
            files = await resolve(split.http, model)
        except ModelUnavailable as exc:
            raise AdminError(str(exc)) from None
        info = files.gguf
        assert info is not None
        ctx = ctx or hub.config.split.default_ctx
        key = model_key(model)
        hub.plans.pin(key, "split", ctx)
        hub_gpus = [
            Device(f"hub {g.get('name', 'GPU')}", int(g.get("vram_total", 0)))
            for g in hub.local_static.get("gpus", [])
            if isinstance(g, dict)
        ]
        helpers, reasons = split.candidates()
        notes = []
        if hub_gpus and fits_on_hub(info, ctx, hub_gpus):
            notes.append("it fits on the hub's GPU alone, so Ollama keeps serving it (split mode would be slower)")
        if files.vision:
            notes.append("images won't work while it runs in split mode (text only)")
        if not helpers:
            notes.append("no helper can take part right now; Ollama serves it until one can")
        return {
            "model": key,
            "ctx": ctx,
            "path": str(files.path),
            "size": info.file_size,
            "layers": info.block_count,
            "architecture": info.architecture,
            "kv_per_layer": info.kv_bytes_per_layer(ctx),
            "layer_bytes_max": max(info.layer_bytes, default=0),
            "helpers": [{"name": n.name, "lendable": n.lendable_vram()} for n in helpers],
            "unavailable": reasons,
            "notes": notes,
            "hub_vram": sum(d.free for d in hub_gpus) // MIB * MIB,
        }

    # ----- dispatch --------------------------------------------------------------------

    async def dispatch(self, cmd: str, args: dict[str, Any]) -> Any:
        hub, registry = self.hub, self.hub.registry
        if cmd == "status":
            return hub.get_snapshot()
        if cmd == "nodes.list":
            return registry.snapshot()
        if cmd == "nodes.show":
            return self._node(args.get("ref")).detail()
        if cmd in ("nodes.pause", "nodes.resume"):
            node = self._node(args.get("ref"))
            registry.set_hub_paused(node, cmd == "nodes.pause")
            return node.summary()
        if cmd == "nodes.kick":
            node = self._node(args.get("ref"))
            if hub.control is None or not node.connected:
                raise AdminError(f"{node.name} is not connected")
            await hub.control.kick(node)
            return node.summary()
        if cmd == "nodes.net_test":
            node = self._node(args.get("ref"))
            if hub.control is None:
                raise AdminError("the control plane is not running")
            try:
                return await hub.control.net_test(node)
            except NetTestError as exc:
                raise AdminError(str(exc)) from None
        if cmd == "nodes.tunnel":
            node = self._node(args.get("ref"))
            mode = args.get("mode")
            if mode not in TUNNEL_MODES:
                raise AdminError(f"tunnel mode must be {' or '.join(TUNNEL_MODES)}")
            registry.set_tunnel_mode(node, mode)
            return node.summary()
        if cmd == "nodes.tunnel_test":
            return await self.tunnel_test(self._node(args.get("ref")))
        if cmd == "nodes.forget":
            node = self._node(args.get("ref"))
            try:
                registry.forget(node)
            except ValueError as exc:
                raise AdminError(str(exc)) from None
            return {"forgotten": node.node_id}
        if cmd == "bans.lift":
            if hub.control is None or not hub.control.bans.lift(str(args.get("ip", ""))):
                raise AdminError("no such ban")
            hub.event("ban_lifted", ip=str(args.get("ip")))
            return {"lifted": args.get("ip")}
        if cmd == "models.list":
            return self.models()
        if cmd == "models.plan":
            return await self.plan_split(args.get("model"), args.get("ctx"))
        if cmd == "models.unplan":
            key = self._model(args.get("model"))
            if not hub.plans.remove(key):
                raise AdminError(f"{key!r} has no plan")
            stopped = await hub.split.stop(key, "plan removed")
            return {"model": key, "stopped": stopped}
        if cmd in ("models.pin", "models.unpin", "models.force_local"):
            key = self._model(args.get("model"))
            engine = {"models.unpin": None, "models.force_local": "local"}.get(cmd, args.get("engine"))
            if cmd == "models.pin" and engine not in ("split", "local"):
                raise AdminError("pin to split or local")
            ctx = self._ctx(args.get("ctx"))
            if engine == "split":
                await self.plan_split(key, ctx)  # checks the model can run split at all
            else:
                hub.plans.pin(key, engine, ctx)
            stopped = False
            if hub.plans.route(key) is None:
                stopped = await hub.split.stop(key, "Ollama serves this model now")
            return {"model": key, **hub.plans.describe(key), "stopped": stopped}
        if cmd == "models.benchmark":
            from ..hub.benchmark import BenchmarkError, benchmark_model

            key = self._model(args.get("model"))
            if hub.plans.bench_lock.locked():
                raise AdminError("a benchmark is already running")
            try:
                return await benchmark_model(hub, key, self._ctx(args.get("ctx")))
            except BenchmarkError as exc:
                raise AdminError(str(exc)) from None
        if cmd == "models.exclude":
            key = self._model(args.get("model"))
            node = self._node(args.get("node"))
            hub.plans.set_excluded(key, node.node_id, args.get("exclude") is not False)
            return {"model": key, **hub.plans.describe(key)}
        if cmd == "models.show":
            return await self.show_model(self._model(args.get("model")))
        if cmd == "split.stop":
            return {"stopped": await hub.split.stop(reason="stopped by the owner")}
        if cmd == "events.list":
            limit = args.get("limit", 100)
            return list(hub.recent_events)[-int(limit) :] if isinstance(limit, int) and limit > 0 else []
        raise AdminError(f"unknown command {cmd!r}")
