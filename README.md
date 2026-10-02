# OllamaPower

Make the PC that runs Ollama (the **Hub**) stronger by borrowing GPU, CPU and
VRAM from other machines on your network (**Helpers**). Clients keep talking to
the Hub exactly as they talk to Ollama today; the Hub decides, per model, whether
a helper can make it faster or let it run a model that doesn't fit, and only
uses a helper when a benchmark proves it wins.

> **Status: Phase 0 (bootstrap).** This repository currently contains the
> project skeleton and the parts of DCP it builds on (encrypted control-plane
> protocol, token auth, priority queue, load balancer), with tests. Nothing is
> runnable for end users yet; see the roadmap below.

## What it can and can't do

LLM generation is limited by **memory bandwidth**: every token reads the whole
model. VRAM moves hundreds of GB/s, system RAM tens of GB/s, a gigabit LAN about
0.12 GB/s. That sets hard rules, and OllamaPower follows them:

| Situation | Does a helper help? |
|---|---|
| The model fits in the Hub's VRAM | **No.** Splitting it across machines makes it slower. OllamaPower leaves it on plain Ollama. |
| The model doesn't fit, so Ollama spills layers to the Hub's CPU/RAM | **Often yes**, over a wired LAN: those layers run on a helper's GPU instead (llama.cpp RPC). It also lets you load models that don't fit at all. |
| Several requests at once (agents, batch jobs, several users) | **Yes**: a helper with its own Ollama can take whole requests (overflow). |
| Helper on Wi-Fi or across the internet | **No** for split mode. Wi-Fi helpers get a loud warning. |

**Golden rule: OllamaPower must never make things slower.** Every non-local plan
is benchmarked against plain local Ollama and is only used if it beats it (by
10 % by default).

## How it works

```
clients ──► OllamaPower Hub (127.0.0.1:11434, Ollama-compatible API)
              ├─ local-ollama → Ollama on 127.0.0.1:11435
              ├─ split        → llama-server --rpc → helpers' rpc-server (over an mTLS tunnel)
              └─ overflow     → a helper's own Ollama, over the encrypted control plane
```

- **Ollama doesn't do multi-machine inference.** Split mode uses llama.cpp's RPC
  backend (`rpc-server` on helpers, `llama-server --rpc` on the Hub), loading the
  GGUF blob Ollama already has on disk, so there is no second copy of the model.
- llama.cpp RPC has no authentication or encryption, so `rpc-server` only ever
  listens on the helper's loopback and the Hub reaches it through a mutually
  authenticated TLS 1.3 tunnel.
- No GUI, ever: a terminal UI (Textual) plus a full CLI, and a headless mode.

## Roadmap

| Phase | Delivers |
|---|---|
| **0 Bootstrap** (done) | Style notes, DCP reuse map, DCP networking/crypto/auth/scheduling ported with tests, lint + test + audit CI |
| 1 Front door | Ollama-compatible passthrough with true streaming, priority queue, `/ollamapower/status`, `doctor` port migration |
| 2 Agents & telemetry | Control plane with protocol version and Hub key pinning, GPU telemetry, net tests, Hub and agent TUIs, CLI |
| 3 Split MVP | llama.cpp pin/download/verify, Ollama blob resolver, `rpc-server`/`llama-server` lifecycle, API translation |
| 4 Secure tunnel | mTLS relays, IP allowlist, opt-in trusted-LAN direct mode, overhead measured |
| 5 Planner | Memory estimate, spill detection, benchmarks, automatic "never slower" plan choice, fault tolerance |
| 6 Overflow | Whole requests on helpers' own Ollama when the Hub is busy |
| 7 Lending & polish | Schedules, auto-yield, real benchmark numbers, start scripts |

Later ideas, not planned yet: speculative decoding with helpers running the
draft model, MoE expert offload, several Hubs.

OllamaPower only makes the Hub stronger. Remote access, device pairing and
per-device quotas are [OllamaGate](https://github.com/111yoav111/OllamaGate)'s
job; the two will be merged later (OllamaGate in front, OllamaPower behind it).

## Development

Python 3.11+.

```bash
python -m venv .venv && .venv/bin/pip install --require-hashes -r requirements-dev.lock
.venv/bin/pip install --no-deps -e .
.venv/bin/python -m pytest -q
.venv/bin/ruff check . && .venv/bin/ruff format --check .
```

Lock files are generated with
`uv pip compile pyproject.toml [--extra dev] --universal --generate-hashes --python-version 3.11`.

Project docs: [`docs/STYLE_NOTES.md`](docs/STYLE_NOTES.md) (conventions shared
with OllamaGate), [`docs/DCP_REUSE.md`](docs/DCP_REUSE.md) (what came from DCP
and why), [`SECURITY.md`](SECURITY.md).

## Credits

OllamaPower is the continuation of
[DCP - Dynamic Compute Power](https://github.com/111yoav111/Dynamic-Compute-Power),
whose encrypted protocol, token auth, priority queue and load balancer it reuses
(MIT). Split mode is built on [llama.cpp](https://github.com/ggml-org/llama.cpp)'s
RPC backend; models come from your local [Ollama](https://ollama.com).

## License

MIT, see [LICENSE](LICENSE).
