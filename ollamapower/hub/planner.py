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
    model: str, info: GGUFInfo, ctx: int, hub: list[Device], helpers: list[Device], margin: float = 0.0
) -> SplitPlan:
    """Place ``model``: the hub's GPUs first, then ``helpers`` in the given order of preference."""
    if ctx <= 0:
        raise PlanError("the context size must be positive")
    units = unit_sizes(info, ctx)
    if fits_on_hub(info, ctx, hub, margin):
        raise PlanError(f"{model} fits on the hub's GPU alone; Ollama runs it faster without helpers")
    order = with_reserve([*reversed(helpers), *hub], info, ctx)  # the best helper sits next to the hub's GPUs
    placed, cpu = _fill(units, order)
    caches = [*info.kv_bytes(ctx), 0]
    placements, first = [], cpu
    for d, n, b in placed:
        if n > 0:
            placements.append(Placement(d, n, b, sum(caches[first : first + n])))
        first += n
    if not any(p.device.node_id for p in placements):
        raise PlanError(f"no helper has room for even one layer of {model} (each layer needs ~{max(units) // MIB} MB)")
    # Whatever didn't fit (the first ``cpu`` layers) stays on the hub's CPU: -ngl is set below the layer count.
    return SplitPlan(
        model=model,
        ctx=ctx,
        units=len(units),
        cpu_layers=cpu,
        placements=placements,
        weights_bytes=info.weights_bytes,
        cpu_bytes=sum(units[:cpu]) + info.input_bytes + info.other_bytes,
    )
