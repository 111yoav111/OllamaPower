"""``ollamapower models suggest``: with the PCs connected now, which models run, and how fast.

For every model installed in Ollama: its size, dense or mixture-of-experts,
whether the hub holds it alone, the planner's predicted speed on plain Ollama
and pooled over the helpers that could lend right now, and the measured
numbers once a benchmark exists. Then a small curated catalog of popular
large models that are not installed (``catalog.json``), estimated from
approximate sizes, for "what could I pull?".

Read-only and on demand: nothing is started, no helper is asked anything. The
pool is what each lending helper's last report says it could lend (its best
GPU within its owner's caps, else its RAM if the owner allows CPU lending),
which is what ``rpc-server`` would offer if a split session started now.
Speeds are the planner's estimates (bytes read per token over bandwidth plus
network round trips); the benchmark is what decides.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..ollama.gguf import GGUFInfo
from .engines.split import helper_device
from .planner import (
    GIB,
    MIB,
    Device,
    PlanError,
    device_reserve,
    fits_hub_memory,
    fits_on_hub,
    plan_split,
    predict_local,
    unit_sizes,
    with_reserve,
)

if TYPE_CHECKING:
    from .nodes import Node
    from .runtime import Hub

HELPER_RAM_KEEP = 3 * GIB  # RAM a helper lending its CPU keeps for its owner (estimate; the agent measures)
CATALOG_FILE = "catalog.json"
# A catalog model's experts are synthetic: only the bytes a token reads matter to the estimate.
SYNTHETIC_EXPERTS = 64


def _num(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _helper_ram(node: Node) -> int | None:
    """Bytes of RAM ``node`` could lend now (from its last heartbeat), None if it hasn't said."""
    total = _num(node.telemetry.get("ram_total")) or _num(node.capabilities.get("ram_total"))
    used = _num(node.telemetry.get("ram_used"))
    if total is None or used is None:
        return None
    size = max(0, int(total - used) - HELPER_RAM_KEEP)
    cap_gb = _num(node.lending.get("max_vram_gb")) or 0.0
    return min(size, int(cap_gb * GIB)) if cap_gb > 0 else size


def _helper_gpu(node: Node) -> tuple[str, int] | None:
    """The GPU ``node``'s agent would lend now and how much of it, like its ``rpc-server`` choice:
    the largest allowed dedicated GPU within the owner's caps, an integrated one only failing that
    (and no more than its free RAM)."""
    rules = node.lending
    allowed = rules.get("gpus") or []
    cap_gb, cap_pct = _num(rules.get("max_vram_gb")) or 0.0, _num(rules.get("max_vram_percent")) or 0.0
    dedicated: list[tuple[int, str]] = []
    integrated: list[tuple[int, str]] = []
    for g in node.gpus():
        if allowed and g.get("index") not in allowed:
            continue
        cap = int(g.get("vram_free") or 0)
        if cap_gb > 0:
            cap = min(cap, int(cap_gb * GIB))
        if 0 < cap_pct <= 100:
            cap = min(cap, int((_num(g.get("vram_total")) or 0) * cap_pct / 100))
        name = str(g.get("name") or f"GPU{g.get('index', '')}")
        if g.get("unified"):
            integrated.append((min(cap, _helper_ram(node) or 0), name))
        else:
            dedicated.append((cap, name))
    usable = [x for x in dedicated if x[0] > 0] or [x for x in integrated if x[0] > 0]
    if not usable:
        return None
    cap, name = max(usable)
    return name, cap


def _pool(nodes: list[Node]) -> tuple[list[Device], list[str]]:
    devices, reasons = [], []
    for node in nodes:
        gpu = _helper_gpu(node)
        if gpu is not None:
            devices.append(helper_device(node, {"device": gpu[0], "free_mib": gpu[1] // MIB}))
        elif node.lending.get("allow_cpu"):
            ram = _helper_ram(node)
            if ram:
                devices.append(helper_device(node, {"device": "CPU", "free_mib": ram // MIB}))
            else:
                reasons.append(f"{node.name} has no RAM to spare right now")
        else:
            reasons.append(f"{node.name} lends no GPU now and its owner hasn't allowed CPU lending")
    return devices, reasons


def pool_devices(hub: Hub) -> list[Device]:
    """What the helpers that could take part now could lend, best first, without starting anything."""
    helpers, _ = hub.split.candidates()
    return _pool(helpers)[0]


# ----- the catalog ------------------------------------------------------------------------------------------


def load_catalog() -> list[dict[str, Any]]:
    data = json.loads(resources.files("ollamapower.hub").joinpath(CATALOG_FILE).read_text(encoding="utf-8"))
    return list(data["models"])


def catalog_info(entry: dict[str, Any]) -> GGUFInfo:
    """An approximate GGUF header for a model that isn't installed: equal layers, the token embeddings
    and output layer at the file's average bytes per weight, and for an MoE model expert tensors sized
    so a token reads ``size * active / params`` bytes."""
    size = int(entry["download_gb"] * 1e9)
    layers = int(entry["layers"])
    per_weight = size / (entry["params_b"] * 1e9)
    emb, vocab = int(entry["embedding"]), int(entry["vocab"])
    embeddings = int(emb * vocab * per_weight)
    layer = max(0, size - 2 * embeddings) // layers
    info = GGUFInfo(
        path=Path(entry["name"]),
        file_size=size,
        version=3,
        architecture="estimate",
        block_count=layers,
        embedding_length=emb,
        head_count=int(entry["kv_heads"]),
        head_count_kv=int(entry["kv_heads"]),
        key_length=int(entry["head_dim"]),
        value_length=int(entry["head_dim"]),
        feed_forward_length=4 * emb,
        vocab_size=vocab,
        layer_bytes=[layer] * layers,
        output_bytes=embeddings,
        input_bytes=embeddings,
    )
    active = entry.get("active_params_b")
    if active:
        # Per block a token reads (layer - experts) + experts / SYNTHETIC_EXPERTS; solve for the experts.
        target = (size * active / entry["params_b"] - embeddings) / layers
        target = min(layer, max(layer / SYNTHETIC_EXPERTS, target))
        experts = int((layer - target) / (1 - 1 / SYNTHETIC_EXPERTS))
        info.expert_bytes = [min(layer, experts)] * layers
        info.expert_count, info.expert_used_count = SYNTHETIC_EXPERTS, 1
    return info


# ----- estimates --------------------------------------------------------------------------------------------


def _missing(info: GGUFInfo, ctx: int, devices: list[Device], ram: Device, margin: float) -> int:
    """Roughly how many bytes the whole pool lacks for ``info`` (at least one layer: it didn't fit)."""
    units = [int(u * (1 + margin)) for u in unit_sizes(info, ctx)]
    need = sum(units) + info.input_bytes + info.other_bytes
    have = sum(d.budget for d in with_reserve(devices, info, ctx)) + max(0, ram.budget - device_reserve(info, ctx))
    return max(need - have, min(units))


def estimate(
    model: str, info: GGUFInfo, ctx: int, gpus: list[Device], ram: Device, pool: list[Device], margin: float
) -> dict[str, Any]:
    """The planner's view of ``model``: fits the hub?, tok/s alone and pooled, or what is missing."""
    active = sum(info.active_layer_bytes()) + info.output_bytes
    fits_gpu = fits_on_hub(info, ctx, gpus, margin)
    out: dict[str, Any] = {
        "type": "MoE" if info.is_moe else "dense",
        "weights": info.weights_bytes,
        "active_bytes": active,
        "ctx": ctx,
        "fits_hub_gpu": fits_gpu,
        "fits_hub": fits_gpu or fits_hub_memory(info, ctx, gpus, ram, margin),
        "alone_tps": round(1 / max(predict_local(info, ctx, gpus, ram), 1e-9), 2),
        "pooled_tps": None,
        "pooled_on": [],
        "missing_bytes": None,
        "plan_error": "",
    }
    if fits_gpu:
        return out  # split mode is never used for it: nothing to pool
    try:
        plan = plan_split(model, info, ctx, gpus, pool, margin, hub_ram=ram)
    except PlanError as exc:
        out["plan_error"] = str(exc)
        if not out["fits_hub"]:
            out["missing_bytes"] = _missing(info, ctx, [*gpus, *pool], ram, margin)
    else:
        out["pooled_tps"] = round(1 / max(plan.seconds_per_token, 1e-9), 2)
        out["pooled_on"] = [p.device.label for p in plan.helpers]
    return out


def tps_text(value: float | None, measured: bool) -> str:
    if value is None:
        return "-"
    text = "<0.1" if value < 0.05 else f"{value:.0f}" if value >= 10 else f"{value:.1f}"
    return text if measured else f"~{text}"


def verdict(row: dict[str, Any], min_gain: float) -> str:
    """One line: runs on the hub, pooled speed vs alone, or how much memory is missing."""
    if "fits_hub" not in row:
        return f"can't estimate: {row.get('reason') or 'no GGUF header'}"
    alone = row.get("measured_alone_tps") or row["alone_tps"]
    pooled = row.get("measured_pooled_tps") or row["pooled_tps"]
    alone_text = tps_text(alone, bool(row.get("measured_alone_tps")))
    pooled_text = tps_text(pooled, bool(row.get("measured_pooled_tps")))
    if row["fits_hub_gpu"]:
        text = "runs fine on the hub"
    elif pooled is not None and (not row["fits_hub"] or pooled > alone * (1 + min_gain)):
        text = f"pooled: {pooled_text} tok/s (vs {alone_text} alone)"
    elif row["fits_hub"]:
        text = "runs fine on the hub"
    elif row["missing_bytes"] is not None:
        text = f"doesn't fit even pooled: {row['missing_bytes'] / GIB:.1f} GB missing"
    else:
        text = row["plan_error"] or "doesn't fit"
    return text if row["installed"] else f"estimate, not installed: {text}"


async def suggest(hub: Hub, ctx: int | None = None) -> dict[str, Any]:
    """Every installed model, then the catalog's models that aren't installed, against the pool now."""
    from .runtime import model_key

    plans, split, config = hub.plans, hub.split, hub.config
    margin, gain = config.planner.memory_margin, config.planner.min_gain
    gpus = split.hub_devices()
    ram = split.hub_ram(free=False)
    helpers, away = split.candidates()
    pool, idle = _pool(helpers)

    installed: list[dict[str, Any]] = []
    for m in list(hub.ollama.installed):
        key = model_key(m.name)
        facts = await plans.facts(key)  # reads the GGUF header once per digest (cached)
        info = plans.gguf_cache.get(plans.digest_of(key) or key)
        row: dict[str, Any] = {"model": key, "installed": True, "size": m.size or None}
        if facts.get("splittable") and info is not None:
            row["size"] = row["size"] or info.file_size
            row.update(estimate(key, info, ctx or plans.ctx_for(key), gpus, ram, pool, margin))
        else:
            row["reason"] = str(facts.get("reason") or "")
        latest = plans.latest(key)
        row["measured_alone_tps"] = latest["local"]["gen_tps"] if latest["local"] else None
        row["measured_pooled_tps"] = latest["split"]["gen_tps"] if latest["split"] else None
        row["verdict"] = verdict(row, gain)
        installed.append(row)

    have = {r["model"] for r in installed}
    catalog: list[dict[str, Any]] = []
    for entry in load_catalog():
        key = model_key(entry["name"])
        if key in have:
            continue
        row = {"model": key, "installed": False, "estimate": True, "size": int(entry["download_gb"] * 1e9)}
        row.update(estimate(key, catalog_info(entry), ctx or config.split.default_ctx, gpus, ram, pool, margin))
        row["verdict"] = verdict(row, gain)
        catalog.append(row)

    return {
        "hub": {"gpus": [{"label": d.label, "bytes": d.free} for d in gpus], "ram": ram.free},
        "pool": [{"label": d.label, "node": d.node_id, "bytes": d.free, "ram": d.ram} for d in pool],
        "not_counted": [*away, *idle],
        "installed": installed,
        "catalog": catalog,
    }
