"""Layer placement for split mode, checked against llama.cpp's own assignment rule."""

from __future__ import annotations

import bisect
import re

import pytest

from ollamapower.hub.planner import MIB, Device, PlanError, device_reserve, plan_split, unit_sizes
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


def gpu(label: str, gb: float, node: str | None = None, cap_gb: float | None = None, ctx: int = 1024) -> Device:
    """A device with ``gb`` (or ``cap_gb``) for layers once the reserve for ``model()`` at ``ctx`` is set aside."""
    reserve = device_reserve(model(), ctx)
    return Device(
        label,
        int(gb * GIB) + reserve,
        cap=None if cap_gb is None else int(cap_gb * GIB) + reserve,
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
    hub = [gpu("hub", 3.0, ctx=2048)]
    helpers = [gpu("fast", 4.5, node="a", ctx=2048), gpu("slow", 2.5, node="b", ctx=2048)]
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


def test_device_reserve_grows_with_the_context():
    from ollamapower.hub.planner import DEVICE_OVERHEAD, UBATCH, compute_buffer

    info = model()
    info.feed_forward_length = 14336  # Llama-3-8B widths
    small, big = device_reserve(info, 4096), device_reserve(info, 131072)
    # With flash attention the growing part is the f16 KQ mask: ubatch x context.
    assert big - small == 2 * UBATCH * (131072 - 4096)
    assert compute_buffer(info, 4096) < 128 * MIB  # an 8B at 4k: well under the old flat 768 MiB in total
    assert DEVICE_OVERHEAD < small < big < 768 * MIB  # at most the old flat reserve for an 8B, even at 128k
    # The same helper takes fewer layers at a long context (KV cache and compute buffer both grow).
    helpers = [gpu("h", 6.0, node="a")]
    short, long = plan_split("m", info, 4096, [], helpers), plan_split("m", info, 131072, [], helpers)
    assert long.gpu_layers < short.gpu_layers
    assert all(p.device.reserve == device_reserve(info, 131072) for p in long.placements)


def test_compute_buffer_matches_llama_cpp_on_a_tiny_model(tmp_path):
    """b11347 on CPU, tiny model (n_embd 256, n_ff 512), -fa on: 8.0 / 36.0 / 132.0 MiB plus small change."""
    from ollamapower.hub.planner import compute_buffer
    from ollamapower.ollama.gguf import read_gguf

    from .tiny_gguf import write_tiny_llama

    info = read_gguf(write_tiny_llama(tmp_path / "m.gguf", n_embd=256, n_ff=512, ctx=131072))
    for ctx, measured_mib in ((4096, 8.51), (32768, 36.51), (131072, 132.51)):
        estimate = compute_buffer(info, ctx) / MIB
        assert measured_mib <= estimate <= measured_mib + 20


def test_output_device_reserves_the_logits_of_a_ubatch():
    from ollamapower.hub.planner import UBATCH

    info = model(layers=4)
    info.vocab_size = 152064  # Qwen2.5
    assert unit_sizes(info, 1024)[-1] > info.output_bytes + 3 * UBATCH * info.vocab_size


def test_summary_is_json_friendly():
    import json

    plan = plan_split("m", model(), 1024, [], [gpu("h", 5.0, node="a")])
    json.dumps(plan.summary())
    assert plan.tensor_split() == str(plan.placements[0].layers)


# ----- v2: RAM pooling and the cost model ---------------------------------------------------


def ram(label: str, gb: float, node: str | None = None, gbps: float = 25.0, rtt_ms: float = 0.3) -> Device:
    """System RAM with ``gb`` usable for layers once the reserve for ``model()`` is set aside."""
    reserve = device_reserve(model(), 1024)
    return Device(
        label,
        int(gb * GIB) + reserve,
        node_id=node,
        device="CPU",
        ram=True,
        bandwidth=gbps * 1e9,
        rtt=rtt_ms / 1000,
    )


def moe(layers: int = 10, layer_gb: float = 1.0, experts_share: float = 0.9) -> GGUFInfo:
    info = model(layers=layers, layer_gb=layer_gb)
    info.expert_bytes = [int(layer_gb * GIB * experts_share)] * layers
    info.expert_count, info.expert_used_count = 32, 4
    return info


def test_a_hub_without_a_gpu_pools_the_helpers_ram():
    info = model(layers=12, layer_gb=1.0)
    hub_ram = ram("hub RAM", 4.1 + 0.5 + device_reserve(info, 1024) / GIB)  # 4 layers beside embeddings and reserve
    helpers = [ram("laptop", 5.0, node="a"), ram("desktop", 5.0, node="b")]
    plan = plan_split("m", info, 1024, [], helpers, hub_ram=hub_ram)
    assert plan.cpu_layers == 4  # the hub's own RAM first: no network hop
    assert sum(p.layers for p in plan.placements) == plan.units - 4
    assert {p.device.node_id for p in plan.placements} == {"a", "b"}
    split = [float(p.layers) for p in plan.placements]
    assert llama_cpp_assignment(plan.units, plan.gpu_layers, split) == our_assignment(plan)
    assert plan.summary()["predicted_tps"] > 0


def test_a_model_that_fits_nowhere_says_how_much_is_missing():
    with pytest.raises(PlanError, match=r"doesn't fit even with the helpers: about \d+\.\d GB more"):
        plan_split("m", model(layers=12), 1024, [], [ram("laptop", 3.0, node="a")], hub_ram=ram("hub RAM", 2.0))


def test_missing_memory_is_counted_when_no_helper_lends_ram():
    with pytest.raises(PlanError, match=r"about (\d+\.\d) GB more") as exc:
        plan_split("m", model(layers=12), 1024, [], [gpu("g", 3.0, node="g")], hub_ram=ram("hub RAM", 2.0))
    assert float(re.search(r"about (\d+\.\d) GB", str(exc.value)).group(1)) >= 5  # not "about 0.0 GB"


def test_gpus_take_the_top_layers_then_helpers_ram_then_the_hubs():
    info = model(layers=12, layer_gb=1.0)
    plan = plan_split(
        "m", info, 1024, [gpu("hub", 3.0)], [gpu("g", 4.0, node="g"), ram("r", 20.0, node="r")], hub_ram=ram("hub", 4.0)
    )
    labels = [p.device.label for p in plan.placements]
    assert labels == ["r", "g", "hub"]  # GPUs nearest the output, the helper's RAM below them
    assert plan.cpu_layers == 0  # the hub's RAM stays free for its owner while a helper has room


def test_the_hubs_ram_goes_first_when_it_is_clearly_faster():
    info = model(layers=12, layer_gb=1.0)
    fast_hub = ram("hub", 6.0, gbps=60.0)
    plan = plan_split("m", info, 1024, [], [ram("laptop", 20.0, node="a", gbps=15.0)], hub_ram=fast_hub)
    assert plan.cpu_layers == 5 and plan.placements[0].device.node_id == "a"  # 6 GB less the embeddings


def test_moe_models_are_predicted_much_faster_than_dense_ones_of_the_same_size():
    helpers = [ram("laptop", 10.0, node="a")]
    dense = plan_split("m", model(layers=12), 1024, [], helpers, hub_ram=ram("hub", 5.0))
    sparse = plan_split("m", moe(layers=12), 1024, [], helpers, hub_ram=ram("hub", 5.0))
    assert sparse.seconds_per_token < dense.seconds_per_token / 3


def test_paging_from_disk_is_what_makes_a_big_model_crawl_on_one_pc():
    from ollamapower.hub.planner import predict_local

    info = model(layers=12)
    pooled = plan_split("m", info, 1024, [], [ram("laptop", 10.0, node="a")], hub_ram=ram("hub", 5.0))
    alone = predict_local(info, 1024, [], ram("hub", 5.0))
    assert alone > 5 * pooled.seconds_per_token
    # A slow link costs per token, but far less than the disk.
    wifi = plan_split("m", info, 1024, [], [ram("laptop", 10.0, node="a", rtt_ms=5)], hub_ram=ram("hub", 5.0))
    assert pooled.seconds_per_token < wifi.seconds_per_token < alone


def test_moe_on_the_hub_keeps_hot_parts_on_its_gpu_and_experts_in_its_ram():
    info = moe(layers=12, layer_gb=1.0, experts_share=0.9)  # 0.1 GB hot + 0.9 GB experts per layer
    hub, helpers = [gpu("hub", 2.0)], [ram("laptop", 20.0, node="a")]
    dense = plan_split("m", model(layers=12), 1024, hub, helpers, hub_ram=ram("hub RAM", 6.0))
    sparse = plan_split("m", info, 1024, hub, helpers, hub_ram=ram("hub RAM", 6.0))
    on_gpu = next(p for p in sparse.placements if p.device.label == "hub")
    assert on_gpu.layers > 2 * next(p for p in dense.placements if p.device.label == "hub").layers
    assert sparse.hub_expert_layers == list(range(sparse.units - on_gpu.layers, sparse.units - 1))
    assert sparse.seconds_per_token < dense.seconds_per_token
    assert sparse.expert_override().startswith(r"blk\.(") and sparse.expert_override().endswith(r"_exps\.=CPU")
    assert not dense.hub_expert_layers and dense.expert_override() == ""


def test_hub_experts_only_go_where_the_hubs_ram_can_hold_them():
    info = moe(layers=12, layer_gb=1.0, experts_share=0.9)
    plan = plan_split("m", info, 1024, [gpu("hub", 2.0)], [ram("laptop", 20.0, node="a")], hub_ram=ram("hub RAM", 3.5))
    assert len(plan.hub_expert_layers) == 3  # 3 x 0.9 GB fit in the hub's RAM (beside the embeddings)
