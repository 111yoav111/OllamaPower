# OllamaPower

Make the PC that runs Ollama (the **Hub**) stronger by borrowing GPU, CPU and
VRAM from other machines on your network (**Helpers**). Clients keep talking to
the Hub exactly as they talk to Ollama today; the Hub decides, per model, whether
a helper can make it faster or let it run a model that doesn't fit, and only
uses a helper when a benchmark proves it wins.

> **Status: Phase 3 (split mode MVP).** The hub runs in front of your Ollama
> with a priority queue and a terminal UI; helpers join and report their GPUs,
> and a model you plan for split mode runs across the hub's GPU and the
> helpers' GPUs with llama.cpp. For now split mode needs **direct mode**
> (plain TCP on a trusted, wired LAN, opted in by both owners); the encrypted
> tunnel arrives in Phase 4, and automatic plans with benchmarks in Phase 5.
> Everything else is served by your local Ollama, exactly as before.

## What it can and can't do

LLM generation is limited by **memory bandwidth**: every token reads the whole
model. VRAM moves hundreds of GB/s, system RAM tens of GB/s, a gigabit LAN about
0.12 GB/s. That sets hard rules, and OllamaPower follows them:

| Situation | Does a helper help? |
|---|---|
| The model fits in the Hub's VRAM | **No.** Splitting it across machines makes it slower. OllamaPower leaves it on plain Ollama. |
| The model doesn't fit, so Ollama spills layers to the Hub's CPU/RAM | **Often yes**, over a wired LAN: those layers run on a helper's GPU instead (llama.cpp RPC). It also lets you load models that don't fit at all. |
| Several requests at once (agents, batch jobs, several users) | **A little**: helper VRAM leaves the Hub room for a larger context and more parallel requests, but every request still runs through the Hub. Helpers never run Ollama or their own copy of a model. |
| Helper on Wi-Fi or across the internet | **No** for split mode. Wi-Fi helpers get a loud warning. |

**Golden rule: OllamaPower must never make things slower.** Today that means:
a model that fits on the hub's GPU is never split, and if split mode can't
start, Ollama serves the request as before. From Phase 5 every non-local plan
is also benchmarked against plain local Ollama and only used if it beats it
(by 10 % by default).

## Requirements

- Python 3.11+
- Ollama on the same PC

## Install

```bash
pipx install git+https://github.com/111yoav111/OllamaPower    # or, for development: see below
```

## Set up the Hub

```bash
ollamapower init      # config dir, a cluster token in .env, config.toml, state.json
ollamapower doctor    # checks everything and shows how to move Ollama to port 11435
ollamapower hub       # run it with the terminal UI (--no-tui: headless, plain logs)
```

OllamaPower takes Ollama's usual address, `127.0.0.1:11434`, so your clients
(Open WebUI, scripts, the `ollama` CLI) keep working without any change. The
real Ollama moves to `127.0.0.1:11435`. `ollamapower doctor` detects where
Ollama listens and prints the exact steps for your OS, e.g. on Linux:

```bash
sudo systemctl edit ollama        # add: [Service]  Environment="OLLAMA_HOST=127.0.0.1:11435"
sudo systemctl restart ollama
```

`hub` refuses to run as root/Administrator (`--allow-root` overrides) and to
listen on anything but loopback (remote access is OllamaGate's job).

### The hub TUI

| Tab | Shows |
|---|---|
| **1 Overview** | front door, Ollama, queue, requests, helpers and lendable VRAM, control-plane port and key fingerprint, the hub's GPU/VRAM bars, loaded models (VRAM %, last tok/s, spill warning), active requests, recent events, warnings |
| **2 Nodes** | every helper: status (🟢 lending · 🟡 paused · 🔴 offline · ⚠ incompatible), name, IP, GPU, VRAM total/free/lendable, GPU %, CPU/RAM, link (RTT, Mbps, wired/Wi-Fi), role, tunnel mode |
| **3 Node detail** | full capabilities, GPU/VRAM/CPU sparklines, network-test history |
| **4 Models** | installed models: size, quantisation, loaded VRAM %, whether they fit the hub's VRAM, plan, last tok/s |
| **5 Log** | filterable event feed: joins, leaves, pauses, bans, failed logins, net tests |

Keys: `1`–`5` tabs · `Enter` helper details · `P` pause/resume a helper (the
hub stops using it) · `K` kick · `N` network test · `T` tunnel mode · `X`
forget an offline helper · `/` filter · `Q` quit. Destructive actions ask
first. `B`/`L`/`F` (benchmark, pin plan, force local) arrive with the planner.

### The front door

| Route | What happens |
|---|---|
| `POST /api/chat`, `/api/generate`, `/api/embed`, `/api/embeddings`, `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings` | queued by priority, then streamed through (NDJSON / SSE, never buffered) |
| `GET /api/ps` | Ollama's answer plus an `ollamapower` field per model (`engine`, `nodes`) that normal clients ignore |
| everything else under `/`, `/api/*`, `/v1/*` (tags, show, pull, create, copy, delete, push, blobs, ...) | passed straight through, not queued |
| `GET /ollamapower/status` | the hub's status as JSON |

At most `hub.max_concurrency` inference requests (default 4) go to Ollama at
once; the rest wait, highest priority first. Set the priority per request with
a header:

```
X-OllamaPower-Priority: critical | high | normal | low      (default: normal)
```

A waiting request that keeps being overtaken goes next after 10 overtakes, so
low priority work is delayed, never starved. A full queue (`hub.max_queue`) or
a wait longer than `hub.queue_timeout` gets `503` with `Retry-After`. If a
client disconnects, its generation is stopped in Ollama right away.

> Set `hub.max_concurrency` to your `OLLAMA_NUM_PARALLEL`. Higher, and Ollama
> queues the extra requests itself (first come first served, priorities lost);
> lower, and Ollama sits partly idle.

### Status and logs

```bash
ollamapower status            # front door, Ollama, queue, loaded models (VRAM %, last tok/s), active requests, warnings
ollamapower status --watch    # refreshes every 2 s
ollamapower status --json     # the same data as /ollamapower/status
ollamapower logs [--follow] [--kind request] [--file events|requests] [-n 50] [--json]
```

`status` warns when a loaded model doesn't fully fit in VRAM (Ollama spilled
part of it to the CPU): that is the case helpers will speed up.

Logs are JSON lines in `<config dir>/logs/` (`events.jsonl`, `requests.jsonl`),
rotated at 10 MB x 5. They hold metadata only (path, model, priority, status,
queue and total time) - never prompts or responses.

`status` talks to the running hub over a local-only channel (a `0600` Unix
socket, or `127.0.0.1` on Windows, plus a secret in an owner-only file).

## Adding helpers

A helper is any other PC on your LAN with a GPU (or spare CPU/RAM). Install
OllamaPower there too, then:

1. On the **hub**: `ollamapower token show` prints the cluster token line.
   Put it in the helper's `.env` (in the helper's config dir, see below), or
   set `OLLAMAPOWER_TOKEN` there. Anyone with the token can join, so treat it
   like a password.
2. On the **helper**:

   ```bash
   ollamapower agent --hub 192.168.1.10          # the hub PC's LAN address (control port 9000)
   ```

   On first contact the agent shows the hub's **key fingerprint**; compare it
   with the one the hub prints (`ollamapower init`, `ollamapower status` or the
   Overview tab) and confirm. Headless helpers pass it up front:
   `--hub-fingerprint AB:CD:...`. The key is then pinned in `known_hub`; if
   the hub's key ever changes, the agent refuses to connect and says so.

The helper appears on the hub's Nodes tab. Helpers on Wi-Fi get a warning:
split mode needs a cable.

### What a helper lends (its owner decides)

The helper's owner controls lending in the agent TUI (`E`), with
`ollamapower agent rules`, or in `agent.toml`:

| Rule | Default | Effect |
|---|---|---|
| `max_vram_gb`, `max_vram_percent` | no cap | most VRAM the hub may use |
| `gpus` | all | which GPUs to lend |
| `allow_cpu` | off | lend CPU/RAM when there is no GPU |
| `allow_direct` | off | allow trusted-LAN direct mode (below) |
| `disk_cache` | on | keep received model weights on disk, so the next load skips the network |
| `schedule` | always | e.g. `22:00-08:00` (enforced from Phase 6) |

`P` in the agent TUI (or `ollamapower agent pause`) stops lending at once.

```bash
ollamapower agent status [--json] [--watch]
ollamapower agent pause | resume
ollamapower agent rules --max-vram-gb 8 --gpus 0
ollamapower agent net-test
ollamapower agent forget-hub          # only after you replaced the hub's key yourself
```

### Managing helpers from the hub

```bash
ollamapower nodes list [--json] [--watch]
ollamapower nodes show <name|id> [--json]
ollamapower nodes pause|resume <name|id>      # the hub stops/starts using it
ollamapower nodes kick <name|id>              # disconnect now (it reconnects after a minute)
ollamapower nodes net-test <name|id>          # round-trip time and throughput both ways
ollamapower nodes tunnel <name|id> secure|direct
ollamapower nodes forget <name|id>            # an offline helper and its settings
ollamapower models list [--json]
```

### Tunnel modes (how split-mode traffic travels)

- **secure** (default): mutually authenticated TLS 1.3 between the hub and
  the helper. **Arrives in Phase 4**; until then a helper in secure mode is
  simply not used for split mode (it never falls back to plaintext on its own).
- **direct**: plain TCP on a trusted, wired LAN, a bit faster. It is used only
  if **both** owners opt in - the hub's owner with `nodes tunnel <helper>
  direct` (or `T`), the helper's owner with `allow_direct`. While it is on,
  the TUI and `status` show a warning. The helper's relay accepts connections
  only from the hub's address and only while a session runs.

## Split mode

One model across the hub's GPU and the helpers' GPUs, for models that don't
fit on the hub alone:

```bash
ollamapower models plan llama3.1:70b --split [--ctx 8192]   # checks the model, previews, saves the plan
ollamapower models unplan llama3.1:70b                      # back to plain Ollama
ollamapower llamacpp install | status                       # the pinned llama.cpp (also done automatically)
```

The first request for a planned model starts a split session:

1. The hub finds the model's GGUF file in Ollama's store (via `/api/show`; no
   second copy) and reads its layer sizes.
2. If it fits on the hub's GPU, nothing changes: Ollama serves it.
3. Ollama is asked to unload its models (`keep_alive: 0`) and the hub waits
   until `/api/ps` is empty, so the two never fight over VRAM.
4. Each eligible helper starts `rpc-server` (bound to its loopback) and
   reports how much memory its owner lends right now.
5. The hub places the layers: its own GPU first, then the helpers, the rest on
   its CPU; a helper never gets more than its owner's cap. `llama-server`
   starts on `127.0.0.1` with exactly that placement.

Clients don't notice: `/api/chat` and `/api/generate` are translated to and
from llama-server (streaming NDJSON, the final `done` object with token counts
and durations, tool calls, `format`, `think`, sampling options), `/v1/*`
requests pass straight through, and `/api/ps` lists the split model with
`"ollamapower": {"engine": "split", "nodes": [...]}`. The session stops after
`split.keep_alive` seconds idle (Ollama gets the GPU back on its next
request), when a helper leaves or its owner pauses, or with `models unplan`.

If anything fails (no eligible helper, a helper refuses, llama-server doesn't
start), the request is served by Ollama and the reason shows as a warning;
split mode retries a minute later.

Limits today: images (vision models' projector isn't loaded), raw/suffix
`generate` prompts and per-request `num_ctx` are not supported in split mode
(images get a clear 400; the context is fixed per plan); sharded models,
adapters (LoRA) and safetensors imports can't be split; embeddings always run
on Ollama; one split model at a time; one GPU per helper.

### The llama.cpp download

Every machine runs the same pinned llama.cpp release (**b11347**), downloaded
from llama.cpp's GitHub releases on first use and checked against SHA-256
hashes stored in OllamaPower (llama.cpp publishes none). Helpers download it
when the agent starts (`auto_install = false` under `[llamacpp]` in
`agent.toml` turns that off). Sizes:

| Backend (`llamacpp.backend`) | Linux | Windows | macOS |
|---|---|---|---|
| `cpu` | 17.6 MB | 19.3 MB | 11.4 MB (Intel) |
| `metal` (Apple Silicon) | - | - | 11.8 MB |
| `vulkan` (any GPU) | 31.5 MB | 33.2 MB | - |
| `cuda12` (NVIDIA) | 765 MB | 654 MB | - |
| `cuda13` (NVIDIA) | 592 MB | 577 MB | - |
| `rocm` (AMD) | 243 MB | 256 MB | - |

`auto` picks Metal on Apple Silicon, Vulkan when there is a GPU and CPU
otherwise, so a typical helper downloads about 30 MB once. CUDA is usually
faster on NVIDIA but much bigger (it includes the CUDA runtime), so it is an
opt-in. Machines in one cluster may use different backends; only the build
must match (a helper with another build is shown as incompatible). Nothing is
kept in RAM only: the build lives in `<config dir>/llamacpp/`, and received
model weights are cached in `<config dir>/rpc-cache/` when `disk_cache` is on.

## Configuration

`ollamapower init` copies [`config.example.toml`](config.example.toml) to the
config directory:

| OS | Config dir |
|---|---|
| Linux | `~/.config/ollamapower/` |
| macOS | `~/Library/Application Support/ollamapower/` |
| Windows | `%APPDATA%\ollamapower\` |

Every value can be overridden with `OLLAMAPOWER_<SECTION>__<KEY>` env vars,
e.g. `OLLAMAPOWER_HUB__MAX_CONCURRENCY=2`; `OLLAMAPOWER_CONFIG_DIR` moves the
directory. The cluster token lives only in `.env` (or the `OLLAMAPOWER_TOKEN`
env var). On disk, all owner-only: `config.toml`, `.env`, `hub_key.pem`,
`state.json` (+ `.bak`), `logs/`, and while running `run/`. A helper keeps
`agent.toml`, `.env`, `known_hub` and `node_id` in the same kind of directory
(`--config-dir` to choose another), and `agent-run/` while running.

## How it works

```
clients ──► OllamaPower Hub (127.0.0.1:11434, Ollama-compatible API)
              ├─ local-ollama → Ollama on 127.0.0.1:11435
              └─ split        → llama-server --rpc → helpers' rpc-server (direct relay; mTLS tunnel in Phase 4)
```

- **Ollama doesn't do multi-machine inference.** Split mode uses llama.cpp's RPC
  backend (`rpc-server` on helpers, `llama-server --rpc` on the Hub), loading the
  GGUF blob Ollama already has on disk, so there is no second copy of the model.
- llama.cpp RPC has no authentication or encryption, so `rpc-server` only ever
  listens on the helper's loopback. The Hub reaches it through the agent's
  relay: plain TCP limited to the hub's address in direct mode, a mutually
  authenticated TLS 1.3 tunnel from Phase 4.
- No GUI, ever: a terminal UI (Textual) plus a full CLI, and a headless mode.

## Roadmap

| Phase | Delivers |
|---|---|
| 0 Bootstrap (done) | Style notes, DCP reuse map, DCP networking/crypto/auth/scheduling ported with tests, lint + test + audit CI |
| 1 Front door (done) | Ollama-compatible passthrough with true streaming, priority queue, `/ollamapower/status`, `doctor` port migration |
| 2 Agents & telemetry (done) | Control plane with protocol version and Hub key pinning, GPU telemetry, net tests, Hub and agent TUIs, CLI |
| **3 Split MVP** (done) | llama.cpp pin/download/verify, Ollama blob resolver, `rpc-server`/`llama-server` lifecycle, layer placement under the lending caps, API translation, direct-mode relay |
| 4 Secure tunnel | mTLS relays, IP allowlist, opt-in trusted-LAN direct mode, overhead measured |
| 5 Planner | Memory estimate, spill detection, benchmarks, automatic "never slower" plan choice, fault tolerance |
| 6 Lending & polish | Schedules, auto-yield, real benchmark numbers, start scripts |

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
OLLAMAPOWER_TEST_LLAMACPP=/path/to/llama-b11347 .venv/bin/python -m pytest -q tests/test_split_real.py   # real llama.cpp
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
