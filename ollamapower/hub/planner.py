"""Decide how many layers of a model go on each GPU (split mode).

llama.cpp assigns layers to devices in one contiguous run each, in its device
order: the helpers' RPC devices first (in ``--rpc`` order), then the hub's own
GPUs, and the first layers that fit nowhere stay on the hub's CPU. The output
layer is "layer N" and lands on the last device. ``--tensor-split`` with whole
layer counts reproduces our placement exactly (see ``llama-model.cpp``), so
the plan below is what really happens.

Placement fills from the top: the hub's GPUs first (no network hop), then the
helpers, best first. A helper gets only as many layers as fit under the cap its
owner set - ``rpc-server`` itself has no memory limit, so this *is* the cap.
Every device keeps a reserve free for llama.cpp's compute buffer, which grows
with the context (``device_reserve``), and the output layer's device also
for the logits.

Rule "never slower": if the model fits on the hub's GPUs alone, split mode is
not used; Ollama serves it as before.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from ..ollama.gguf import GGUFInfo

MIB = 1024 * 1024
GIB = 1024 * MIB
# Kept free on every GPU besides the compute buffer: the backend's own context, driver and allocator slack.
DEVICE_OVERHEAD = 512 * MIB
# Generation reads every weight a token uses once per token, so time = bytes / bandwidth.
# Defaults until a machine's own measurement (or a benchmark) says otherwise:
GPU_BANDWIDTH = 250e9  # a dedicated GPU we know nothing more about (mid-range cards do 250-450 GB/s)
RAM_BANDWIDTH = 25e9  # desktop dual-channel RAM, all cores
DISK_BANDWIDTH = 1e9  # weights paged in from an SSD because they don't fit in RAM
HOP_SECONDS = 0.0006  # llama.cpp's own cost of handing a token to another machine (measured, loopback)
HUB_RAM_PREFERRED = 1.2  # the hub's RAM goes before helpers' only when this much faster
UBATCH = 512  # llama-server's default -ub: the compute graph is reserved for this many tokens at once
COMPUTE_SLACK = 16 * MIB


class PlanError(Exception):
    """No useful split exists (the message says why)."""


@dataclass
class Device:
    """One place layers can go."""

    label: str  # shown to the owner, e.g. "hub Vulkan0" or "desk-pc Vulkan0"
    free: int  # bytes free right now
    cap: int | None = None  # the helper owner's limit in bytes, None = no limit
    node_id: str | None = None  # None for the hub's own GPUs
    device: str = ""  # llama.cpp device name on its own machine ("Vulkan0", "CUDA1", "CPU")
    reserve: int = 0  # kept free for compute buffers and overhead (set by the planner for a model and context)
    ram: bool = False  # system RAM (a CPU device), not a GPU's own memory
    bandwidth: float = 0.0  # bytes/s; 0 = the default for its kind
    rtt: float = 0.0  # seconds for a round trip between the hub and this device's machine

    @property
    def speed(self) -> float:
        return self.bandwidth or (RAM_BANDWIDTH if self.ram else GPU_BANDWIDTH)

    @property
    def budget(self) -> int:
        usable = self.free if self.cap is None else min(self.free, self.cap)
        return max(0, usable - self.reserve)


@dataclass
class Placement:
    device: Device
    layers: int
    bytes: int  # weights + cache
    cache: int = 0  # the KV cache / recurrent state part of ``bytes``


@dataclass
class SplitPlan:
    model: str
    ctx: int
    units: int  # block_count + 1 (the output layer)
    cpu_layers: int
    placements: list[Placement] = field(default_factory=list)  # llama.cpp device order
    weights_bytes: int = 0
    cpu_bytes: int = 0
    seconds_per_token: float = 0.0  # predicted generation time (see ``predict``)
    hub_expert_layers: list[int] = field(default_factory=list)  # blocks on the hub's GPUs whose experts stay in RAM

    def expert_override(self) -> str:
        """llama.cpp's ``-ot`` rule keeping those experts in the hub's RAM ("" if none)."""
        if not self.hub_expert_layers:
            return ""
        layers = "|".join(str(i) for i in self.hub_expert_layers)
        return rf"blk\.({layers})\.ffn_[a-z_]*_exps\.=CPU"

    @property
    def gpu_layers(self) -> int:
        return self.units - self.cpu_layers

    @property
    def helpers(self) -> list[Placement]:
        return [p for p in self.placements if p.device.node_id is not None]

    @property
    def hub_devices(self) -> list[Placement]:
        return [p for p in self.placements if p.device.node_id is None]

    def tensor_split(self) -> str:
        return ",".join(str(p.layers) for p in self.placements)

    def summary(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "ctx": self.ctx,
            "layers": self.units,
            "cpu_layers": self.cpu_layers,
            "cpu_bytes": self.cpu_bytes,
            "weights_bytes": self.weights_bytes,
            "predicted_tps": round(1 / self.seconds_per_token, 2) if self.seconds_per_token else None,
            "devices": [
                {
                    "label": p.device.label,
                    "node": p.device.node_id,
                    "device": p.device.device,
                    "layers": p.layers,
                    "bytes": p.bytes,
                    "cache": p.cache,
                    "budget": p.device.budget,
                }
                for p in self.placements
            ],
        }


def compute_buffer(info: GGUFInfo, ctx: int) -> int:
    """llama.cpp's compute buffer on each device, with flash attention on (``-fa on``).

    Flash attention never materialises the attention scores, so what grows with
    the context is the f16 KQ mask (one row per ubatch token, one column per cached
    token); the rest are the widest activations of one ubatch, reused between ops,
    plus a little for small tensors. Measured on b11347: the mask term is exact.
    """
    cells = -(-ctx // 256) * 256
    return 2 * UBATCH * cells + _activations(info) + COMPUTE_SLACK


def _activations(info: GGUFInfo) -> int:
    return 4 * UBATCH * max(6 * info.embedding_length, 3 * info.feed_forward_length + 2 * info.embedding_length)


def device_reserve(info: GGUFInfo, ctx: int) -> int:
    """What every device keeps free besides its layers for ``info`` at ``ctx``: grows with the context."""
    return DEVICE_OVERHEAD + compute_buffer(info, ctx)


def logits_bytes(info: GGUFInfo) -> int:
    """Extra compute buffer on the output layer's device: the f32 logits of a whole ubatch (llama.cpp
    reserves for the worst case), beyond the activations every device already reserves for."""
    return max(0, 4 * UBATCH * info.vocab_size - _activations(info))


def unit_sizes(info: GGUFInfo, ctx: int) -> list[int]:
    """Bytes each of llama.cpp's layer slots needs: blocks (weights + KV cache), then the output layer."""
    blocks = [b + kv for b, kv in zip(info.layer_bytes, info.kv_bytes(ctx), strict=True)]
    return [*blocks, info.output_bytes + logits_bytes(info)]


def with_reserve(devices: list[Device], info: GGUFInfo, ctx: int) -> list[Device]:
    reserve = device_reserve(info, ctx)
    return [replace(d, reserve=reserve) for d in devices]


def _fill(units: list[int], devices: list[Device]) -> tuple[list[tuple[Device, int, int]], int]:
    """Place units from the top on ``devices`` (last device first). Returns placements and the CPU count."""
    top = len(units)
    placed: list[tuple[Device, int, int]] = []
    for device in reversed(devices):
        budget, count, used = device.budget, 0, 0
        while top - count > 0 and used + units[top - count - 1] <= budget:
            used += units[top - count - 1]
            count += 1
        placed.append((device, count, used))
        top -= count
    placed.reverse()
    return placed, top


def hub_need(info: GGUFInfo, ctx: int, margin: float = 0.0) -> int:
    """GPU memory ``info`` needs at ``ctx`` on one device: layers, caches, reserve, with ``margin`` on top."""
    return int(sum(unit_sizes(info, ctx)) * (1 + margin)) + device_reserve(info, ctx)


def fits_on_hub(info: GGUFInfo, ctx: int, hub: list[Device], margin: float = 0.0) -> bool:
    """THE check "does the model fit on the hub's GPUs alone?" - the facts, the split start and the
    planner all use it, so they never disagree (a model one says fits and another says doesn't would be
    benchmarked, refused, and benchmarked again). ``margin`` (planner.memory_margin) is added to every
    layer, as safety against the estimate being low."""
    units = [int(u * (1 + margin)) for u in unit_sizes(info, ctx)]
    _, cpu = _fill(units, with_reserve(hub, info, ctx))
    return bool(hub) and cpu == 0


def plan_split(
    model: str,
    info: GGUFInfo,
    ctx: int,
    hub: list[Device],
    helpers: list[Device],
    margin: float = 0.0,
    hub_ram: Device | None = None,
) -> SplitPlan:
    """Place ``model`` on the hub's GPUs, the helpers' devices and the hub's RAM, fastest memory first.

    llama.cpp gives each device one contiguous run of layers: the hub's CPU takes the first
    layers, then the devices in ``--device`` order, the hub's GPUs last (with the output layer).
    So: GPUs (the hub's, then the helpers', fastest first) fill from the top; the hub's RAM
    takes the bottom layers (no network hop); helpers lending RAM take what is left in the
    middle. Without ``hub_ram`` the hub's CPU takes whatever is left, however much (v1).
    """
    if ctx <= 0:
        raise PlanError("the context size must be positive")
    units = [int(u * (1 + margin)) for u in unit_sizes(info, ctx)]
    if fits_on_hub(info, ctx, hub, margin):
        raise PlanError(f"{model} fits on the hub's GPU alone; Ollama runs it faster without helpers")
    # ``helpers`` come best first; _fill takes the last device first, so reverse, then order by speed
    # (stable: equally fast helpers keep the caller's preference).
    gpus = sorted((d for d in reversed(helpers) if not d.ram), key=lambda d: d.speed)
    rams = sorted((d for d in reversed(helpers) if d.ram), key=lambda d: d.speed)
    ram_budget = 0
    if hub_ram is not None:
        ram_budget = max(0, hub_ram.budget - device_reserve(info, ctx) - info.input_bytes - info.other_bytes)
    # MoE on the hub: its GPUs hold the hot part of their layers and its RAM their experts
    # (llama.cpp's --n-cpu-moe; GPU and RAM are on the same machine, so no network per layer).
    offload = hub_ram is not None and bool(hub) and info.is_moe
    experts = [int(e * (1 + margin)) for e in info.expert_bytes] + [0] if offload else [0] * len(units)
    hub_units = [u - e for u, e in zip(units, experts, strict=True)]
    hub_placed, below_hub = _fill(hub_units, with_reserve(hub, info, ctx))
    on_hub = range(below_hub, len(units) - 1)
    if sum(experts[i] for i in on_hub) > ram_budget:
        # Not all those experts fit in the hub's RAM: keep the top layers whose experts do.
        keep, used = 0, 0
        for i in reversed(on_hub):
            if used + experts[i] > ram_budget:
                break
            used, keep = used + experts[i], keep + 1
        start = len(units) - 1 - keep
        hub_placed, left = _fill(hub_units[start:], with_reserve(hub, info, ctx))
        below_hub = start + left
        on_hub = range(below_hub, len(units) - 1)
    hub_experts = sum(experts[i] for i in on_hub)
    gpu_placed, rest = _fill(units[:below_hub], with_reserve(gpus, info, ctx))
    top_placed = [*gpu_placed, *hub_placed]
    cpu = rest
    if hub_ram is not None:
        # The hub's RAM holds the bottom layers plus the token embeddings and small tensors. Helpers
        # lending RAM go first (a few ms per token, and the hub's RAM stays free for its owner),
        # unless the hub's RAM is clearly faster.
        budget = max(0, ram_budget - hub_experts)
        hub_first = bool(rams) and hub_ram.speed > HUB_RAM_PREFERRED * max(d.speed for d in rams)
        bottom = 0
        if hub_first or not rams:
            used = 0
            while bottom < rest and used + units[bottom] <= budget:
                used += units[bottom]
                bottom += 1
        middle_placed, left = _fill(units[bottom:rest], with_reserve(rams, info, ctx))
        if not hub_first and rams:
            # What the helpers couldn't take stays at the bottom, on the hub's RAM.
            bottom = left
            left = 0 if sum(units[:bottom]) <= budget else left
        if left:
            short = sum(units[:bottom]) - budget if rams and not hub_first else sum(units[bottom : bottom + left])
            raise PlanError(
                f"{model} doesn't fit even with the helpers: about {max(short, 0) / GIB:.1f} GB more memory is needed"
            )
        cpu = bottom
    else:
        middle_placed = []
    placed = [*[x for x in middle_placed if x[1]], *[x for x in top_placed if x[1]]]
    caches = [*info.kv_bytes(ctx), 0]
    placements, first = [], cpu
    for d, n, b in placed:
        placements.append(Placement(d, n, b, sum(caches[first : first + n])))
        first += n
    if not any(p.device.node_id for p in placements):
        raise PlanError(f"no helper has room for even one layer of {model} (each layer needs ~{max(units) // MIB} MB)")
    plan = SplitPlan(
        model=model,
        ctx=ctx,
        units=len(units),
        cpu_layers=cpu,
        placements=placements,
        weights_bytes=info.weights_bytes,
        cpu_bytes=sum(units[:cpu]) + info.input_bytes + info.other_bytes + hub_experts,
        hub_expert_layers=list(on_hub) if offload else [],
    )
    plan.seconds_per_token = predict(plan, info, hub_ram)
    return plan


def _active_units(info: GGUFInfo) -> list[int]:
    """Bytes each layer slot reads per token: active weights of every block, then the output layer."""
    return [*info.active_layer_bytes(), info.output_bytes]


def predict(plan: SplitPlan, info: GGUFInfo, hub_ram: Device | None = None) -> float:
    """Seconds per generated token: each device reads its layers' active weights, plus the network.

    A token crosses to every helper and back (about two round trips per helper per token,
    measured), plus llama.cpp's own handoff cost. An estimate to rank plans and skip hopeless
    benchmarks; the benchmark is what decides.
    """
    active = _active_units(info)
    ram_speed = hub_ram.speed if hub_ram is not None else RAM_BANDWIDTH
    seconds = sum(active[: plan.cpu_layers]) / ram_speed
    if plan.hub_expert_layers:
        share = min(1.0, info.expert_used_count / max(info.expert_count, 1))
        for i in plan.hub_expert_layers:  # read from the hub's RAM, not its GPU
            moved = info.expert_bytes[i] * share
            active[i] -= moved
            seconds += moved / ram_speed
    first = plan.cpu_layers
    nodes: dict[str, float] = {}
    for p in plan.placements:
        seconds += sum(active[first : first + p.layers]) / p.device.speed
        first += p.layers
        if p.device.node_id:
            nodes[p.device.node_id] = max(nodes.get(p.device.node_id, 0.0), p.device.rtt)
    return seconds + sum(2 * rtt + HOP_SECONDS for rtt in nodes.values())


def fits_hub_memory(info: GGUFInfo, ctx: int, hub: list[Device], hub_ram: Device, margin: float = 0.0) -> bool:
    """Can the hub hold ``info`` in its GPUs and RAM together (so Ollama runs it without paging from disk)?"""
    units = [int(u * (1 + margin)) for u in unit_sizes(info, ctx)]
    _, rest = _fill(units, with_reserve(hub, info, ctx))
    return sum(units[:rest]) + info.input_bytes + info.other_bytes <= hub_ram.budget


def predict_local(info: GGUFInfo, ctx: int, hub: list[Device], hub_ram: Device | None) -> float:
    """Seconds per token for plain Ollama on the hub alone: its GPUs from the top, its RAM, then the disk.

    What doesn't fit in RAM is paged in from disk for every token, which is what makes a
    too-big model crawl on one PC.
    """
    units = unit_sizes(info, ctx)
    active = _active_units(info)
    placed, rest = _fill(units, with_reserve(hub, info, ctx))
    seconds, top = 0.0, len(units)
    for d, n, _ in reversed(placed):
        seconds += sum(active[top - n : top]) / d.speed
        top -= n
    ram_budget = hub_ram.budget - info.input_bytes - info.other_bytes if hub_ram is not None else None
    used = 0
    for i in range(rest):
        in_ram = ram_budget is None or used + units[i] <= ram_budget
        used += units[i] if in_ram else 0
        speed = (hub_ram.speed if hub_ram is not None else RAM_BANDWIDTH) if in_ram else DISK_BANDWIDTH
        seconds += active[i] / speed
    return seconds
