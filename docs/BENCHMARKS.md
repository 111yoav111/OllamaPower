# Benchmarks

Real numbers only. Each table says where it was measured; numbers from the
owner's own machines are added as they come in (Phase 7). Measure your setup
with `ollamapower nodes tunnel-test <helper>` and `ollamapower status`
(tokens/s per model).

## Front-door overhead (requests Ollama serves)

What the front door adds to a request it passes to Ollama
(`scripts/bench_frontdoor.py`: a fake Ollama and the front door in separate
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

**LAN numbers:** not measured yet (no multi-machine setup in development).

## Split mode vs plain Ollama (the planner's own benchmarks)

The hub benchmarks every model that doesn't fit its GPU (`ollamapower models
benchmark <model>`, or automatically when idle) and keeps the results in
`state.json`; `ollamapower models show <model>` lists them (load time, time
to first token, prompt and generation tokens/s, which helpers and tunnel
modes). Split mode is used only when it wins by `planner.min_gain`.

**Real model numbers:** not measured yet. The development container has no
GPU and no second machine, so its only split runs use a tiny generated test
model, which says nothing about real speed-ups. Numbers from the owner's
hub and helpers go here.

### Adding your numbers

1. On the hub, with the helpers connected: `ollamapower models benchmark <model>`
   for each model you care about (or let the hub do it while idle), and
   `ollamapower nodes tunnel-test <helper>` for each helper.
2. `ollamapower models benchmarks --markdown` prints the hub's GPUs, each
   helper's GPUs and link, and a table of every successful run (date, model,
   context, engine, helpers with tunnel mode, load time, time to first token,
   prompt and generation tokens/s). Paste it below with a line on the setup
   (switch, cables), and `nodes show <helper>` for the tunnel numbers.
