"""Layer placement for split mode, checked against llama.cpp's own assignment rule."""

from __future__ import annotations

import bisect

import pytest

from ollamapower.hub.planner import DEVICE_RESERVE, MIB, Device, PlanError, plan_split, unit_sizes
from ollamapower.ollama.gguf import GGUFInfo

GIB = 1024 * MIB


def model(layers: int = 10, layer_gb: float = 1.0, output_gb: float = 0.5) -> GGUFInfo:
    return GGUFInfo(
        path=None,  # type: ignore[arg-type]
        file_size=0,
        version=3,
        architecture="llama",
        block_count=layers,
        embedding_length=4096,
        head_count=32,
        head_count_kv=8,
        layer_bytes=[int(layer_gb * GIB)] * layers,
        output_bytes=int(output_gb * GIB),
        input_bytes=int(0.5 * GIB),
    )


def gpu(label: str, gb: float, node: str | None = None, cap_gb: float | None = None) -> Device:
    return Device(
        label,
        int(gb * GIB) + DEVICE_RESERVE,
        cap=None if cap_gb is None else int(cap_gb * GIB) + DEVICE_RESERVE,
        node_id=node,
        device="X",
    )


def llama_cpp_assignment(units: int, ngl: int, split: list[float]) -> list[int]:
    """llama-model.cpp's rule: -1 = CPU, else the device index, for layers 0..units-1."""
    sums, total = [], 0.0
    for s in split:
        total += s
        sums.append(total)
    sums = [s / total for s in sums]
    start = max(units - ngl, 0)
    out = []
    for il in range(units):
        if il < start:
            out.append(-1)
        else:
            out.append(bisect.bisect_right(sums, (il - start) / ngl))
    return out


def our_assignment(plan) -> list[int]:
    out = [-1] * plan.cpu_layers
    for i, p in enumerate(plan.placements):
        out += [i] * p.layers
    return out


@pytest.mark.parametrize(
    "hub_gb,helpers_gb",
    [(0, [4.0]), (3.2, [4.1]), (2.0, [3.0, 5.5]), (0, [1.2, 1.2, 1.2]), (5.0, [20.0]), (1.0, [0.5, 30.0])],
)
def test_tensor_split_reproduces_the_plan_in_llama_cpp(hub_gb, helpers_gb):
    info = model(layers=12, layer_gb=0.7)
    hub = [gpu("hub", hub_gb)] if hub_gb else []
    helpers = [gpu(f"h{i}", gb, node=f"n{i}") for i, gb in enumerate(helpers_gb)]
    plan = plan_split("m", info, 1024, hub, helpers)
    split = [float(p.layers) for p in plan.placements]
    assert llama_cpp_assignment(plan.units, plan.gpu_layers, split) == our_assignment(plan)


def test_hub_gpu_first_then_best_helper_then_cpu():
    info = model(layers=10, layer_gb=1.0, output_gb=0.5)
    kv = info.kv_bytes_per_layer(2048)
    hub = [gpu("hub", 3.0)]
    helpers = [gpu("fast", 4.5, node="a"), gpu("slow", 2.5, node="b")]
    plan = plan_split("m", info, 2048, hub, helpers)
    by_label = {p.device.label: p for p in plan.placements}
    # The hub takes the output layer and as many blocks as fit; then "fast", then "slow".
    assert by_label["hub"].layers == 1 + int((3.0 * GIB - 0.5 * GIB) // (GIB + kv))
    assert [p.device.label for p in plan.placements] == ["slow", "fast", "hub"]  # llama.cpp order: RPC first
    assert plan.cpu_layers == plan.units - sum(p.layers for p in plan.placements)
    for p in plan.placements:
        assert p.bytes <= p.device.budget  # never more than the device (or its owner) allows


def test_owner_cap_limits_a_helper():
    info = model(layers=10)
    plan = plan_split("m", info, 1024, [], [gpu("big", 40.0, node="a", cap_gb=2.2)])
    assert plan.placements[0].layers == 2 and plan.placements[0].bytes <= 2.2 * GIB
    assert plan.cpu_layers == 9


def test_fits_on_hub_is_left_to_ollama():
    with pytest.raises(PlanError, match="fits on the hub"):
        plan_split("m", model(layers=4), 1024, [gpu("hub", 24.0)], [gpu("h", 24.0, node="a")])


def test_helpers_too_small_for_a_layer():
    with pytest.raises(PlanError, match="no helper has room"):
        plan_split("m", model(layers=4, layer_gb=2.0), 1024, [gpu("hub", 1.0)], [gpu("h", 1.0, node="a")])


def test_kv_cache_grows_with_context():
    info = model(layers=4)
    small, big = unit_sizes(info, 1024), unit_sizes(info, 32768)
    assert big[0] - small[0] == info.kv_bytes_per_layer(32768) - info.kv_bytes_per_layer(1024)
    assert big[-1] == small[-1]  # the output layer has no KV cache


def test_per_layer_cache_places_more_layers_and_is_counted_per_device():
    from ollamapower.ollama.gguf import KVLayer

    ctx = 32768
    full = model(layers=10, layer_gb=0.5)
    per_token = full.kv_bytes_per_layer(1)
    windowed = model(layers=10, layer_gb=0.5)
    windowed.kv_layers = [KVLayer(per_token, window=1536 if i % 2 == 0 else 0) for i in range(10)]
    helpers = [gpu("h", 4.0, node="a")]
    a, b = plan_split("m", full, ctx, [], helpers), plan_split("m", windowed, ctx, [], helpers)
    assert b.gpu_layers > a.gpu_layers  # a smaller (correct) cache estimate fits more layers
    kv = [*windowed.kv_bytes(ctx), 0]
    first = b.cpu_layers
    for p in b.placements:
        assert p.cache == sum(kv[first : first + p.layers])
        first += p.layers


def test_summary_is_json_friendly():
    import json

    plan = plan_split("m", model(), 1024, [], [gpu("h", 5.0, node="a")])
    json.dumps(plan.summary())
    assert plan.tensor_split() == str(plan.placements[0].layers)
