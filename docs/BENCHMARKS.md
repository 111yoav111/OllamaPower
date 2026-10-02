# Benchmarks

Real numbers only. Each table says where it was measured; numbers from the
owner's own machines are added as they come in (Phase 7). Measure your setup
with `ollamapower nodes tunnel-test <helper>` and `ollamapower status`
(tokens/s per model).

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
