# Field test: two real PCs (2026-10-17)

The first run of OllamaPower on real hardware instead of the development
container: what was tried, what broke, what was fixed, and what is still open.
The numbers are in [`BENCHMARKS.md`](BENCHMARKS.md).

**Hub:** RTX 3060 12 GB (Vulkan), i7-11700F, 15.3 GB RAM, wired, Ollama 0.34.4.
**Helper:** laptop, i3-1115G4 (2 cores), 7.4 GB RAM, no GPU, Wi-Fi (3.3 ms
RTT, ~94 Mbit/s), lending 2-2.8 GB of RAM. llama.cpp b11347, secure tunnel.
**Models:** Qwen3-30B-A3B (MoE, Q4_K_M, 18.6 GB, an abliterated build) and
qwen3:14b.

## What was run

| Scenario | Result |
|---|---|
| Benchmark, default settings | Split ran with the laptop (8-10 tok/s), Ollama won (21-26 tok/s), the hub kept Ollama. |
| 16K context | Split doesn't fit (0.4 GB short); Ollama 20.4 tok/s. |
| Every API path through split (pinned) | Chat (stream / not), generate with context, thinking, JSON schema, tool calls, OpenAI `/v1` chat + stream, a 13-message history truncated to fit: all fine. Raw prompts and images: a clear 400. |
| Six concurrent requests, mixed priorities | After the fix below: critical and high go first, all six done in 21 s. |
| Helper kicked mid-answer | `{"error": "stream terminated: split mode stopped: <helper> disconnected"}`, next request on Ollama 5 s later, model degraded, recovers after the cooldown. |
| Helper paused (hub side) mid-answer | Answers get 30 s to finish, then end with "split mode stopped: <helper> is paused"; not counted as a failure. |
| Client hangs up mid-answer | llama-server stops at once; the next request is answered in 1.1 s. |
| Automatic idle benchmark | Starts after `planner.idle_seconds`; gives way the moment a client request arrives (that request waited 12.7 s for the GPU to be handed back and Ollama to load its model). |
| Disk cache on the helper | First load over Wi-Fi 132-238 s, later loads 17-70 s. |

Not testable on this hardware: a helper with a GPU, direct tunnel mode (the
helper's owner must allow it), auto-yield (the laptop has no GPU load to
watch), schedules, the MTP and tensor-split variants (wrong model types).

## Bugs found and fixed (each with a regression test)

1. The helper's split-mode relay listened on a random port, which the
   laptop's firewall dropped: llama-server aborted with a gdb backtrace as the
   only "reason". Now a fixed port (`agent.relay_port`, 9001), a clear "is its
   firewall blocking it?" error, and `GGML_NO_BACKTRACE` for llama.cpp.
2. The planner put every MoE expert in RAM and left most of the VRAM empty,
   then found the model "doesn't fit even with the helpers".
3. Requests waiting for llama-server's single slot went first come first
   served: priorities only applied in front of it.
4. A stale keep-alive connection to llama-server failed a request before the
   first byte; it counted as split mode failing, degraded the model, and sent
   four requests to a 60 s Ollama load. No keep-alive to llama-server now.
5. A planned stop (helper paused) ended answers with a bare "upstream error".
6. A benchmark at a context split mode can't run at switched the model's plan
   to that context, breaking a pinned model.
7. After unpinning, the reason shown was from an older benchmark.
8. An embedding request for a chat model served by split mode stopped split
   mode so Ollama could load 18.6 GB just to refuse it. Now refused at once,
   split keeps running.
9. A CPU-only helper showed "0.0/0.0 GB" in the Nodes table even while lending
   RAM. Smaller ones: the address showed as `::ffff:a.b.c.d`; `models suggest`
   called a model spilling into RAM "runs fine"; Wi-Fi was called "too slow
   for split mode" in two places.

## Review of the results (LLM council: five advisors, peer review, chairman)

Agreed: fix the user-facing bugs (embeddings, the CPU-helper display), stop
the docs and `models suggest` from claiming more than was measured, and build
no new split features until split wins somewhere. Not done on purpose:

- **Fitting a correction to the speed predictions.** They were 2-2.6x too
  optimistic, but two runs on one Wi-Fi laptop would overfit. The benchmark
  output now shows the prediction next to the measurement, and `models
  suggest` says predictions can be optimistic.
- **Rejecting a per-request `num_ctx` in split mode** (suggested by the
  council): many clients send it on every request, so a 400 would make split
  models unusable for them. It stays a documented limit.
- Retrying a pre-first-token error, a longer drain, prompt speed in the
  decision: wait for real cases.

Open questions for the owner:

- **Where does split beat Ollama?** Not on this hardware: the 30B fits the
  hub's GPU plus RAM. The claimed case (a model too big for any one PC, a
  helper with a GPU or a wired link) is still untested.
- **Where does a split token's time go** (helper compute, Wi-Fi round trips,
  llama-server)? A profile would show whether better predictions are possible
  or the Wi-Fi link sets a hard limit.
- **Measurement noise:** Ollama's prompt speed swung 15-45 tok/s between runs;
  the 10 % switch rule needs several runs per side to be trusted.
