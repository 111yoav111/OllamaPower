# OllamaPower v2: run models bigger than any PC you own

**Goal.** Use every PC you have (gaming PC, old desktop, laptop without a
GPU) as one machine, so models that don't fit on any single one run, and the
big mixture-of-experts (MoE) models run at a usable speed. Never slower than
plain Ollama, and plain Ollama stays the fallback.

**Any PC.** Nothing may require a strong machine: the hub can be a laptop
without a GPU running Ollama on the CPU, and a helper can be any Windows,
macOS or Linux PC with spare RAM. GPUs anywhere are used when present, never
required. Every number the planner needs (RAM, bandwidth, link) is measured
on the machine itself, so the plan fits what each PC really has.

v1 treated VRAM as the only resource worth borrowing, so it only helped an
owner who already had a second GPU PC. v2 pools **RAM first, GPUs as a
bonus**, and is built around what the measurements below show.

## What decides speed (measured or published, not guessed)

1. **Generation is memory-bound.** Each token reads every weight it uses, so
   time per token ≈ bytes read ÷ memory bandwidth. A dense 70B model reads
   ~40 GB per token. An MoE model reads only its shared part plus the few
   experts the router picks: gpt-oss-120b reads ~3 GB of its ~65 GB.
   Published: gpt-oss-120b does 10-25 tok/s on one PC with 64 GB RAM and an
   8-12 GB GPU (llama.cpp discussion 15396). Large MoE models are the
   models a pool of ordinary PCs can actually run well.
2. **Cut between machines only at layer boundaries.** With the pinned
   llama.cpp (b11347), measured here (loopback, tiny model, strace):

   | Placement | Network round trips per token | Extra time per token |
   |---|---|---|
   | contiguous layer blocks (pipeline), 2 helpers | ~4 in total | ~0.6 ms per hop |
   | parts of each layer on another machine (`-ot`) | 2 per layer | 2-4 ms per layer, even on loopback |
   | tensor split (`-sm tensor`, new, experimental) | ~4 per layer per helper | ~1.5 ms per layer on loopback |

   On a LAN each round trip adds its RTT (~0.3 ms wired, 2-5 ms Wi-Fi). So:
   pipeline costs a few ms per token (Wi-Fi is fine once the model is
   loaded), per-layer splitting across machines is too expensive, and tensor
   split needs a cable.
3. **Inside one machine, split by temperature.** Attention, shared tensors
   and the KV cache are read for every token: they go on that machine's
   GPU. Routed experts are read sparsely: they go in its RAM. rpc-server
   can expose several devices (`-d CUDA0,CPU`) and llama.cpp's `-ot`
   places tensors on any of them (names like `RPC0[host:port]`; verified).
   This is llama.cpp's `--n-cpu-moe` trick, applied on every machine.
4. **Tensor split scales bandwidth for dense models.** Every machine works
   on every token, so N CPU-only PCs read weights N times faster, at ~4
   round trips per layer. Worth it for a dense model on wired, similar
   machines; the benchmark decides.
5. **Speculative decoding hides the network.** One pass through the
   pipeline checks several drafted tokens. llama.cpp has MTP heads (models
   that ship them, ~2x reported) and draft models. Ollama only does this on
   Apple MLX (checked in its source, Oct 2026).
6. **Load once.** Weights cross the network on the first load only:
   rpc-server's cache (`-c`) skips tensors it already has (content hash).
   A 65 GB model over Wi-Fi takes 30+ minutes the first time and seconds
   after that.

Prior art that confirms the direction: prima.cpp (ICLR 2026) runs Llama 3
70B on a Mac, two gaming laptops/PCs and a phone at ~1.5 tok/s where plain
llama.cpp takes 10 s per token; distributed-llama runs Qwen3-30B-A3B on 4
Raspberry Pi 5 at ~13 tok/s with tensor parallelism.

## Expected results (estimates from point 1, to be replaced by benchmarks)

| Setup | Model | Alone | Pooled (estimate) |
|---|---|---|---|
| 3 PCs, 32 GB RAM each, no GPU | gpt-oss-120b (65 GB, MoE) | doesn't fit | ~8-10 tok/s |
| hub with 8 GB GPU + 3 PCs, 48 GB RAM each | Qwen3-235B-A22B Q4 (~140 GB, MoE) | doesn't fit | ~2.5-3.5 tok/s |
| 2 PCs, 32 GB RAM each, wired | dense 70B Q4 (40 GB) | doesn't fit (disk paging, <0.2 tok/s) | ~0.9 tok/s; ~1.5 with tensor split or speculation |

## Design

1. **Devices.** Every helper reports its GPUs (VRAM), its RAM, cores and,
   from a short self-test, its memory bandwidth; the link (RTT, Mbit/s,
   Wi-Fi or cable) is measured already. RAM lending is an explicit owner
   choice (`allow_cpu`); the first run asks what to lend.
2. **Planner (the core).** Per layer: hot bytes, expert bytes, active
   expert bytes (`n_expert_used / n_expert`), KV bytes for the context.
   Machines get contiguous layer blocks; inside each, hot tensors go to its
   GPU and experts to its RAM. Predicted time per token = Σ machines
   (hot bytes ÷ GPU or RAM bandwidth + active expert bytes ÷ RAM bandwidth)
   + hops × (RTT + 0.6 ms). It tries helper subsets (a slow helper can make
   things worse) and keeps the fastest plan that fits. Output: `--rpc`,
   `-ot` rules, `--tensor-split`, threads, context and speculation flags.
3. **Never slower, without wasting an hour.** A model Ollama can serve is
   still benchmarked both ways. A model that can't fit on the hub (Ollama
   would page from disk) skips the Ollama run when the prediction says
   split is 10x faster.
4. **`ollamapower models suggest`.** For the current machines: which
   installed models fit, how fast (predicted, then measured), and which
   popular models would fit if pulled (curated table of sizes and active
   parameters; reading GGUF headers straight from the registry with range
   requests is the upgrade, unverified from this container).
5. **Wi-Fi.** Allowed for pipeline mode with a note about the first load;
   tensor split refused on Wi-Fi.
6. **Kept from v1:** front door on 11434, fallback to Ollama, pinned mTLS
   tunnels, owner rules (caps, schedule, auto-yield), benchmarks and
   backoff.

## Phases

- **A: RAM pooling and MoE-aware placement** (biggest win).
  Done: rpc-server on all physical cores (it defaulted to 2 threads); RAM
  bandwidth probe on every machine; MoE-aware cost model; the hub's RAM and
  helpers' RAM as devices (helpers' first, a model that fits nowhere refused
  with what is missing); a model too big for the hub routed to split at once
  and benchmarked without the Ollama run; Wi-Fi helpers. Next: per-machine
  GPU/RAM split (`-d GPU,CPU` + `-ot` for experts; needs verifying on real
  GPUs, which the development container lacks).
- **B: `models suggest`**, and calibrating the cost model from real
  benchmarks.
- **C: speculative decoding** in split mode (MTP when the model has it, a
  draft model on the hub otherwise).
- **D: tensor split** for dense models on wired machines, benchmark-gated.
- **E: later**: helpers fetch their slice from the registry themselves
  (hub uplink not the bottleneck); keep the most-used experts on GPUs.

## Limits

Dense 70B stays around 1-2 tok/s on ordinary PCs. Phones are clients, not
helpers. Speed can't exceed what the pooled memory bandwidth allows; the
win is running what didn't fit at all, and MoE models at usable speed.
