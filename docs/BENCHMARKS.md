# Benchmarks

Real numbers only. Each table says where it was measured; numbers from the
owner's own machines are added as they come in (Phase 7). Measure your setup
with `ollamapower nodes tunnel-test <helper>` and `ollamapower status`
(tokens/s per model).

## Front-door overhead (requests Ollama serves)

What the front door adds to a request it passes to Ollama
(a benchmark script, `scripts/bench_frontdoor.py` in the git history: a fake Ollama and the front door in separate
processes, small non-streaming `/api/chat` requests, then 200-chunk streams).

**Loopback, development container (4 vCPUs, Python 3.11, uvicorn with h11).**

| Path | Latency (median) | Latency (p95) | Stream chunks/s |
|---|---|---|---|
| straight to Ollama | 1.0 ms | 1.3 ms | ~40 000 |
| through the front door | 2.7 ms | 3.3 ms | ~15 000 |

About 1.7 ms per request, against a model that takes tens of milliseconds
or more per token; the stream ceiling is ~100x what Ollama generates. Without
`sniffio` installed, httpcore repeats a failed import on every request, which
cost another 0.35 ms (it is now a dependency).

## Split-mode tunnel overhead

What encryption costs on the split-mode path (`nodes tunnel-test`: round
trips of 64-byte messages, then 64 MB echoed through the helper's relay).

**Loopback, development container (4 vCPUs, Python 3.11, OpenSSL 3.0).** The
hub and the helper are on one machine, so this shows the *relay's own cost*
(its CPU ceiling), not a network.

| Mode | Round trip (median) | Throughput |
|---|---|---|
| direct (plain TCP relay) | 0.05 ms | 3.3-3.5 Gbit/s |
| secure (TLS 1.3 relays) | 0.08 ms | 1.8-2.1 Gbit/s |

The same machine running a real split session (pinned llama.cpp b11347, a
tiny generated 8-layer model, every layer on the "helper's" `rpc-server`,
64 tokens, 4 runs):

| Mode | Generation |
|---|---|
| direct | 437-456 tok/s |
| secure | 394-409 tok/s |

Reading it: the TLS relays cost about 0.03 ms per round trip and ~0.25 ms
per token here. A tiny model makes that look big (each token takes only
~2.3 ms); a real model takes 30-100+ ms per token over a LAN, where the same
0.25 ms is under 1 %. The secure relay's ceiling (~2 Gbit/s on 4 vCPUs) is
above what a gigabit LAN carries, so on gigabit Ethernet the network, not the
encryption, is the limit. On 10 GbE the Python relay could become the
bottleneck; measure with `tunnel-test` first. If it is, the options are a
small compiled relay or direct mode inside WireGuard/Tailscale - a decision
for the owner, never switched silently.

**LAN numbers:** over Wi-Fi (~94 Mbit/s), the secure tunnel carried 90.8
Mbit/s at 4.8 ms RTT: the link, not the encryption, is the limit (see the
two-PC run below).

## Split mode vs plain Ollama (the planner's own benchmarks)

The hub benchmarks every model that doesn't fit its GPU (`ollamapower models
benchmark <model>`, or automatically when idle) and keeps the results in
`state.json`; `ollamapower models show <model>` lists them (load time, time
to first token, prompt and generation tokens/s, which helpers and tunnel
modes). Split mode is used only when it wins by `planner.min_gain`.

### First real two-PC run (2026-10-17)

**Hub:** RTX 3060 12 GB, i7-11700F, 15.3 GB RAM, wired. **Helper:** a laptop
with an i3-1115G4 (2 cores), 7.4 GB RAM, no GPU, on **Wi-Fi**, lending ~2 GB of
RAM. llama.cpp b11347 (Vulkan on the hub, CPU on the helper), secure tunnel.

| Model | Engine | Load s | TTFT ms | Prompt tok/s | Gen tok/s |
|---|---|---|---|---|---|
| qwen3:14b | Ollama (hub alone) | 15.1 | 713 | 801 | 31.8 |
| qwen3:14b | split, hub GPU + laptop RAM | 131.6 (first load), 16.9 (from the laptop's disk cache) | 11 606 | 41.9 | 4.3 |
| huihui_ai/qwen3-abliterated:30b-a3b (MoE, 18.6 GB) | Ollama (hub GPU + RAM) | 58-62 | 10 538-31 418 | 15.0-45.0 | 21.4-26.4 (3 runs) |
| same, 4K context | split: hub GPU 44 layers + hub RAM 8.0 GB of experts + laptop 5 layers (2.3 GB) | 238 (first), 63-70 (cached) | 17 495-18 052 | 26.7-27.4 | 8.4-10.4 (2 runs; predicted 27.1) |
| same, 16K context | Ollama | 58.2 | 16 086 | 29.3 | 20.4 |
| same, 16K context | split | - | - | - | doesn't fit (0.4 GB short) |

`qwen3:14b` fits this hub's GPU, so the run forced it over with
`planner.memory_margin = 0.4` (13.3 GB "needed") to put layers on the laptop.
Wi-Fi link: 3.3 ms RTT, ~94 Mbit/s both ways; secure tunnel 4.8 ms RTT,
90.8 Mbit/s (the encryption costs nothing measurable at Wi-Fi speeds).

Reading it:

- **Everything worked across two real PCs**: joining, key pinning, the
  secure tunnel over Wi-Fi, weights sent to the helper and cached on its disk
  (8x faster second load), and the fallback: cutting the helper off
  mid-answer ended the stream with an `{"error": ...}`, the next request was
  answered by Ollama 5 s later, and the model was marked degraded.
- **Never slower held**: split lost (4.3 vs 31.8 tok/s), so the hub kept
  Ollama. A 2-core laptop CPU on Wi-Fi can't speed up a hub with a 12 GB GPU;
  a helper earns its place with a GPU, or with RAM for a model the hub can't
  hold at all.
- **The planner's estimate was 2x optimistic** for this helper (8.7 predicted
  vs 4.3 measured tok/s): the benchmark, not the estimate, decides.
- **The 30B MoE fits this hub alone** (GPU + RAM), and Ollama runs it at
  21-26 tok/s; split with the laptop ran at 8-10 tok/s and the hub kept
  Ollama. Ollama's own prompt speed varies a lot between runs (15-45 tok/s),
  so compare generation speeds over several runs.
- What was tried and what broke: [`FIELD_TEST.md`](FIELD_TEST.md).

### Roles swapped: a model too big for the hub (2026-10-17)

**Hub:** the laptop above (7.4 GB RAM, no GPU, Wi-Fi). **Helper:** the desktop
above (RTX 3060 12 GB + RAM), lending its GPU.

| Model | Engine | Load s | TTFT ms | Prompt tok/s | Gen tok/s |
|---|---|---|---|---|---|
| huihui_ai/gpt-oss-abliterated:20b (MoE, 13.8 GB) | Ollama (laptop alone) | - | - | - | ~0.33 (estimated; too big, read from disk) |
| same | split, laptop + desktop GPU | 960.6 (first load, 13.3 GB over Wi-Fi) | 12 572 | 45.0 | **5.7** (predicted 3.5) |

The model the laptop can't run on its own runs at a usable 5.7 tok/s, about
17x faster than it could alone.

### Adding your numbers

1. On the hub, with the helpers connected: `ollamapower models benchmark <model>`
   for each model you care about (or let the hub do it while idle), and
   `ollamapower nodes tunnel-test <helper>` for each helper.
2. `ollamapower models benchmarks --markdown` prints the hub's GPUs, each
   helper's GPUs and link, and a table of every successful run (date, model,
   context, engine, helpers with tunnel mode, load time, time to first token,
   prompt and generation tokens/s). Paste it below with a line on the setup
   (switch, cables), and `nodes show <helper>` for the tunnel numbers.
