# OllamaPower

Run models bigger than any PC you own, by pooling the PCs you already have.
Every machine counts: the PC you use Ollama on (the **Hub**) can be an old
desktop or a laptop without a GPU, and so can every machine that helps it
(**Helpers**). Together their RAM, CPUs and any GPUs hold a model none of them
could load alone, and the big mixture-of-experts models (gpt-oss-120b,
Qwen3-235B, ...) run at a usable speed. Clients keep talking to the Hub exactly
as they talk to Ollama today, and plain Ollama stays the fallback: OllamaPower
never makes anything slower.

> **Status: v2 built (phases A-D).** A front door for your Ollama with a
> queue and a terminal UI; helpers that join over encrypted tunnels and lend
> on their owners' terms (caps, schedule, pause, auto-yield). Any PC counts:
> the hub's and helpers' RAM are pooled (a GPU-less hub or helper is fine), a
> helper lends its GPU and its RAM together, MoE models keep their experts in
> RAM and their hot parts on the GPUs, a model too big for the hub goes to
> split mode on its first request, `models suggest` tells what your PCs can
> run and how fast, and the benchmark also tries MTP speculation and tensor
> split, keeping whatever is fastest. Never slower than plain Ollama, which
> stays the fallback. Tested on two real PCs (a GPU hub and a laptop helper on
> Wi-Fi); no speed-up measured yet with a helper that has a GPU: see
> [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) and
> [`docs/DESIGN_V2.md`](docs/DESIGN_V2.md).

## What it can and can't do

LLM generation is limited by **memory**: every token reads the weights it
uses, so a model must fit somewhere, and its speed is roughly the bytes read
per token divided by the memory bandwidth. VRAM moves hundreds of GB/s, RAM
tens of GB/s, a gigabit LAN about 0.12 GB/s. Splitting a model between
machines at layer boundaries costs only a few network round trips per token,
so the network matters for loading the model, much less for running it.
No machine has to be strong: a hub without a GPU and helpers without GPUs
all count.

| Situation | Does pooling help? |
|---|---|
| The model fits in the Hub's own memory, at full speed | **No.** OllamaPower leaves it on plain Ollama. |
| The model doesn't fit in the Hub's RAM/VRAM (Ollama pages it from disk, or refuses) | **Yes, the main case** (designed for; not yet measured on real PCs). Spread over the Hub's and helpers' RAM (and any GPUs), it runs instead of crawling; the hub sends it to split mode on the first request. Helpers without a GPU lend RAM once their owner allows it (`ollamapower agent rules --allow-cpu`). |
| Mixture-of-experts models (gpt-oss-120b, Qwen3-235B-A22B, ...) too big for the Hub | **Yes, the best case** (estimated). Each token reads only a few experts, so the pooled RAM of ordinary PCs runs them at usable speed (estimates in [`docs/DESIGN_V2.md`](docs/DESIGN_V2.md)). *(v2)* |
| The model fits in the Hub's GPU **plus** its RAM, and the only helper is a CPU on Wi-Fi | **No** (measured): a 30B MoE ran at 21-26 tok/s on Ollama alone and 8-10 tok/s split with a 2-core laptop helper. The hub measured this and kept Ollama ([`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)). |
| Dense 70B on ordinary PCs | **It runs** (~1-2 tok/s estimated) where a single PC pages from disk. *(v2)* |
| Ollama spills part of a model to the Hub's CPU and a helper has a GPU | **Often faster**: those layers run on the helper's GPU. *(v1, works now)* |
| Helper on Wi-Fi | **Allowed.** A few round trips per token, and the first load of a big model is slow (4 min for 2.3 GB of layers here; later loads come from the helper's disk cache). |
| Phones, tablets | Clients only, not helpers. |

**Golden rule: OllamaPower must never make things slower.** A model that fits
on the hub's GPU is never split. Any other model runs split only after a
benchmark showed split generating at least 10 % (`planner.min_gain`) faster
than plain Ollama, and only while nothing that benchmark depended on has
changed. If split mode fails, Ollama serves the request.

## Quick start

Two roles, one program:

| PC | Role | Needs |
|---|---|---|
| The PC where you already use Ollama | **Hub** (exactly one) | Python 3.11+, Ollama |
| Every other PC that should help | **Helper** (any number) | Python 3.11+ (no Ollama needed) |

All PCs must be on the same LAN. Two firewall ports: helpers reach the hub on
TCP **9000**, and the hub reaches each helper on TCP **9001** for split mode.

### 1. Install (on every PC)

OllamaPower is installed with [pipx](https://pipx.pypa.io), which keeps it in
its own environment and puts `ollamapower` on your PATH.

**Linux**

```bash
sudo apt install pipx          # Fedora: sudo dnf install pipx · Arch: sudo pacman -S python-pipx
pipx ensurepath                # then open a new terminal
pipx install git+https://github.com/111yoav111/OllamaPower
```

**macOS** (with [Homebrew](https://brew.sh))

```bash
brew install pipx git
pipx ensurepath                # then open a new terminal
pipx install git+https://github.com/111yoav111/OllamaPower
```

**Windows** (PowerShell; install [Python](https://www.python.org/downloads/)
with "Add python.exe to PATH" ticked, and [Git](https://git-scm.com/download/win))

```powershell
py -m pip install --user pipx
py -m pipx ensurepath          # then open a new terminal
pipx install git+https://github.com/111yoav111/OllamaPower
```

Check it worked with `ollamapower version`. To update later:
`pipx upgrade ollamapower`.

### 2. The Hub (the PC with Ollama)

```bash
ollamapower init       # creates the config dir, a cluster token, config.toml
ollamapower doctor     # tells you exactly what to fix, step by step
```

`doctor` will ask you to move Ollama from port 11434 to **11435**, so
OllamaPower can take 11434 and your apps keep working unchanged:

| OS | Move Ollama to 11435 |
|---|---|
| Linux (systemd) | `sudo systemctl edit ollama`, add `[Service]` and `Environment="OLLAMA_HOST=127.0.0.1:11435"`, then `sudo systemctl restart ollama` |
| macOS | `launchctl setenv OLLAMA_HOST 127.0.0.1:11435`, then quit and reopen the Ollama app (lost at reboot; `doctor` shows how to keep it) |
| Windows | `setx OLLAMA_HOST 127.0.0.1:11435`, then quit Ollama from the tray icon and start it again |

Run `ollamapower doctor` again until it shows no `[ERROR]`, then start the hub:

```bash
ollamapower hub        # terminal UI; --no-tui for plain logs (servers, services)
```

Allow incoming TCP port **9000** in the hub's firewall so helpers can connect
(Windows asks on the first run: allow it for **private** networks;
Linux with ufw: `sudo ufw allow 9000/tcp`). Find the hub's LAN address with
`ip -4 addr` (Linux), `ipconfig getifaddr en0` (macOS) or `ipconfig`
(Windows), e.g. `192.168.1.10`.

Finally, print the cluster token for the helpers:

```bash
ollamapower token show     # prints OLLAMAPOWER_TOKEN="..." - treat it like a password
```

### 3. Each Helper

Give the helper the token, either in a `.env` file in its config dir (create
the folder if needed and paste the whole line `token show` printed):

| OS | Token file |
|---|---|
| Linux | `~/.config/ollamapower/.env` |
| macOS | `~/Library/Application Support/ollamapower/.env` |
| Windows | `%APPDATA%\ollamapower\.env` |

or as an environment variable for the current terminal:

```bash
export OLLAMAPOWER_TOKEN="..."             # Linux / macOS
```

```powershell
$env:OLLAMAPOWER_TOKEN = "..."             # Windows PowerShell
```

Then start it, with the hub's LAN address:

```bash
ollamapower agent --hub 192.168.1.10
```

On first connect it shows the hub's **key fingerprint**: check it matches the
one in the hub's Overview tab (or `ollamapower status` on the hub) and
confirm. The helper appears on the hub's Nodes tab. It downloads its
llama.cpp build once (about 30 MB) and then waits to be used.

Allow incoming TCP port **9001** in the helper's firewall: the hub connects
there to run a model split across the PCs (Linux with ufw:
`sudo ufw allow from 192.168.1.0/24 to any port 9001 proto tcp`; Windows asks
on the first split run: allow it for **private** networks). Another port:
`relay_port` in `agent.toml`.

A helper with **no GPU** lends only after its owner allows it:

```bash
ollamapower agent rules --allow-cpu
```

### 4. Use it

Nothing changes for your apps: Open WebUI, scripts and the `ollama` CLI keep
using `http://127.0.0.1:11434`. To see what your PCs can now run together:

```bash
ollamapower models suggest
```

A model too big for the hub alone is split across the PCs on its first
request. `ollamapower status` on the hub shows what is happening.

### Starting it from a source checkout

Instead of pipx you can clone the repository and use the start scripts,
which use the checkout's `.venv` if there is one (see
[Development](#development)), else the installed `ollamapower`. Extra
arguments are passed on:

| | Linux / macOS | Windows |
|---|---|---|
| Hub | `scripts/start_hub.sh` | `scripts\start_hub.bat` |
| Helper | `scripts/start_agent.sh --hub 192.168.1.10` | `scripts\start_agent.bat --hub 192.168.1.10` |

### Troubleshooting

| Problem | Fix |
|---|---|
| `ollamapower: command not found` | Open a new terminal after `pipx ensurepath`. |
| Helper says the token is missing or wrong | Run `ollamapower token show` on the hub again and copy the whole line into the helper's `.env`. |
| Helper can't reach the hub | Is `ollamapower hub` running? Is port 9000 open in the hub's firewall? Are both PCs on the same network? Try `ping <hub address>` from the helper. |
| Helper refuses: the hub's key changed | Only if you re-created the hub's key yourself: `ollamapower agent forget-hub` on the helper, then connect again. |
| A model you pulled doesn't show up | It went to another Ollama: don't run `ollama serve` yourself (it ignores the service's settings, e.g. `OLLAMA_MODELS`). With the hub running, a plain `ollama pull <model>` reaches the right Ollama. |
| Split mode fails with "couldn't reach <helper> ... is its firewall blocking it?" | Allow incoming TCP 9001 (`relay_port`) in that helper's firewall. |
| Hub says port 11434 is in use | Ollama still listens there: do step 2's "Move Ollama" again and restart Ollama. |
| Anything else | `ollamapower doctor` on the hub, `ollamapower logs` for the event log. |

## Set up the Hub (details)

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

`doctor` also checks that neither Ollama's port nor the front door's (11434)
answers on a network address, that `OLLAMA_HOST` is a valid loopback address,
the cluster token, the hub key, the permissions of the config directory and
its secrets, and the pinned llama.cpp build (installed, verified, runs). It
exits 1 if anything is an `[ERROR]`.

`hub` refuses to run as root/Administrator (`--allow-root` overrides) and to
listen on anything but loopback (remote access is OllamaGate's job).

### The hub TUI

| Tab | Shows |
|---|---|
| **1 Overview** | front door, Ollama, queue, the running split session (layers per device), requests, helpers and lendable VRAM, control-plane port and key fingerprint, the hub's GPU/VRAM bars, loaded models (VRAM %, last tok/s, spill warning), active requests, recent events, warnings |
| **2 Nodes** | every helper: status (🟢 lending · 🟡 paused or stepping back · 🔴 offline · ⚠ incompatible), name, IP, GPU, VRAM total/free/lendable, GPU %, CPU/RAM, link (RTT, Mbps, wired/Wi-Fi), role, tunnel mode |
| **3 Node detail** | full capabilities, GPU/VRAM/CPU sparklines, network-test history, last tunnel test (secure vs direct) |
| **4 Models** | installed models: size, loaded VRAM %, whether they fit the hub's VRAM, engine and mode (auto/pinned), Ollama vs split tok/s from the last benchmark, why; benchmark and first-load progress |
| **5 Log** | filterable event feed: joins, leaves, pauses, bans, failed logins, net tests |

Keys: `1`–`5` tabs · `Enter` helper details · `P` pause/resume a helper (the
hub stops using it) · `K` kick · `N` network test · `T` tunnel mode · `M` tunnel test · `X`
forget an offline helper · `/` filter · `Q` quit. Destructive actions ask
first. Models tab: `B` benchmark · `L` pin split · `F` force local · `U` automatic.

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
part of it to the CPU): the case where split mode may help. The hub benchmarks
such a model with helpers while it is idle (`planner.auto_benchmark`) and switches
only if split mode is measurably faster.

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
   Overview tab, where it is shown whole on two lines) and confirm; compare
   every pair, not just the first few. Headless helpers pass it up front:
   `--hub-fingerprint AB:CD:...`. The key is then pinned in `known_hub`; if
   the hub's key ever changes, the agent refuses to connect and says so.

The helper appears on the hub's Nodes tab. Helpers on Wi-Fi take part too:
a split model costs only a few round trips per token, but the first load of
a big model over Wi-Fi is slow (the helper's disk cache makes later loads
fast), so the hub notes it.

### What a helper lends (its owner decides)

The helper's owner controls lending in the agent TUI (`E`), with
`ollamapower agent rules`, or in `agent.toml`
([`agent.example.toml`](agent.example.toml) lists every setting;
`ollamapower agent example-config > agent.toml` in the config directory
writes it on a machine that installed OllamaPower without this repository):

| Rule | Default | Effect |
|---|---|---|
| `max_vram_gb`, `max_vram_percent` | no cap | most VRAM the hub may use |
| `gpus` | all | which GPUs to lend |
| `allow_cpu` | off | lend CPU/RAM when there is no GPU |
| `allow_direct` | off | allow trusted-LAN direct mode (below) |
| `disk_cache` | on | keep received model weights on disk, so the next load skips the network |
| `schedule` | always | when to lend, local time, e.g. `22:00-08:00` or `09:00-12:00,20:00-23:00` (windows may cross midnight) |
| `auto_yield` | on | step back when you use the GPU yourself (below) |
| `yield_gpu_percent`, `yield_seconds` | 60 %, 20 s | what counts as "you are using it" |

`P` in the agent TUI (or `ollamapower agent pause`) stops lending at once:
any split session using the helper ends immediately.

**Schedule and auto-yield are gentler.** When the schedule ends, or the
helper's GPU has been busier than `yield_gpu_percent` for `yield_seconds`
while the hub wasn't using it (you started a game or a render), the helper
tells the hub it is stepping back. The hub lets the answers it is generating
finish (up to 30 s), stops split mode, and serves the model from Ollama;
nothing counts as a failure. When the helper lends again (the schedule
starts, or the GPU has been calm for `yield_seconds`), split mode resumes
without a new benchmark. The agent TUI and `agent status` say why the helper
isn't lending right now. Limits: while the hub is actively using the GPU,
your own load can't be told apart from it, so the helper steps back within
`yield_seconds` of the hub's next pause; GPUs that don't report their load
(Apple, some AMD) can't auto-yield - use the schedule or `P`.

```bash
ollamapower agent status [--json] [--watch]
ollamapower agent pause | resume
ollamapower agent rules --max-vram-gb 8 --gpus 0
ollamapower agent net-test
ollamapower agent example-config      # every agent.toml setting, commented
ollamapower agent forget-hub [--hub 192.168.1.10[:9000] | --hub [fd00::1]:9000]
                                      # only after you replaced the hub's key yourself (default: all)
```

### Managing helpers from the hub

```bash
ollamapower nodes list [--json] [--watch]
ollamapower nodes show <name|id> [--json]
ollamapower nodes pause|resume <name|id>      # the hub stops/starts using it
ollamapower nodes kick <name|id>              # disconnect now (it reconnects after a minute)
ollamapower nodes net-test <name|id>          # round-trip time and throughput both ways
ollamapower nodes tunnel <name|id> secure|direct
ollamapower nodes tunnel-test <name|id>       # split-mode tunnel speed: secure, and direct if allowed
ollamapower nodes forget <name|id>            # an offline helper and its settings
ollamapower models list [--json]
```

### Tunnel modes (how split-mode traffic travels)

- **secure** (default): mutually authenticated TLS 1.3 between the hub and
  the helper. For every split session both sides make a fresh key and
  certificate and exchange only the certificates, over the already
  authenticated control channel; each side then trusts exactly the other's
  certificate for that session. `llama-server` on the hub talks to a local
  relay, which carries the traffic through TLS to the helper's relay, which
  hands it to `rpc-server` on the helper's loopback.
- **direct**: plain TCP on a trusted, wired LAN, a bit faster. It is used only
  if **both** owners opt in - the hub's owner with `nodes tunnel <helper>
  direct` (or `T`), the helper's owner with `allow_direct`; if only one did,
  secure is used. While it is on, the TUI and `status` show a warning.

In both modes the helper's relay listens only while a session runs, and
drops any connection that doesn't come from the hub's address before reading
a byte (so strangers can't even start a TLS handshake).

**What does encryption cost?** `ollamapower nodes tunnel-test <helper>` (or
`M` on the Nodes tab) sends traffic through the real path - the helper's
relay to a small echo server - and reports round-trip time and throughput for
secure mode, and for direct mode too if the helper's owner allows it, plus the
difference. The result is kept and shown in `nodes show`. On our test machine
(loopback, 4 CPUs) the Python TLS relay carried about 1.8-2 Gbit/s and added
about 0.03 ms per round trip: more than a gigabit LAN can carry, and under 1 %
of a typical token's time for real models. Measure your own LAN; see
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

## Split mode and the planner

One model across the hub's GPU and the helpers' GPUs, for models that don't
fit on the hub alone. The hub decides by itself, by measuring:

```bash
ollamapower models list                      # engine per model, why, last benchmark (Ollama vs split tok/s)
ollamapower models suggest [--ctx N] [--json] # what the PCs connected now can run, and how fast
ollamapower models show <model>              # memory estimate, plan, benchmark history
ollamapower models benchmark <model> [--ctx N]
ollamapower models pin <model> split|local   # override the benchmark   (TUI: L / F)
ollamapower models force-local <model>       # same as `pin <model> local`
ollamapower models unpin <model>             # automatic again           (TUI: U)
ollamapower models exclude <model> <helper> [--undo]
ollamapower models benchmarks [--markdown]   # every result; --markdown for docs/BENCHMARKS.md
ollamapower llamacpp install | status [--config-dir DIR]   # the pinned llama.cpp (also done automatically)
```

### Which models can I run? (`models suggest`)

For every installed model: dense or MoE, the planner's predicted speed on the
hub alone and pooled over what the helpers lend right now (each helper's best
GPU within its owner's caps, else its RAM if CPU lending is allowed), the
measured speeds once benchmarked, or how much memory is missing. Then popular
models you could pull, estimated from a small built-in table of approximate
sizes. Nothing is started or loaded; `~` marks a prediction. A hub with a
12 GB GPU and 32 GB RAM, a helper with a 12 GB GPU, a laptop lending RAM:

```text
MODEL                             SIZE TYPE   ALONE POOLED  VERDICT
gpt-oss:120b                   60.5 GB MoE     ~1.0    ~13  pooled: ~13 tok/s (vs ~1.0 alone)
qwen3:8b                        4.8 GB dense    ~52      -  runs fine on the hub
Could pull (estimates from approximate sizes, not installed):
qwen3:30b-a3b                  17.7 GB MoE      ~29    ~92  estimate, not installed: pooled: ~92 tok/s (vs ~29 alone)
llama3.3:70b                   40.0 GB dense   ~0.4   ~1.2  estimate, not installed: pooled: ~1.2 tok/s (vs ~0.4 alone)
1 more popular model(s) wouldn't fit even pooled (--json lists them).
Pool now: desk RTX 4070 11.0 GB, laptop CPU 21.0 GB · hub RAM 29.0 GB
```

### How the hub decides

1. **Does it fit?** From the model's GGUF file: weights + KV cache at the
   plan's context + `planner.memory_margin` (10 %), against the hub's VRAM.
   Ollama's `/api/ps` showing the model partly on the CPU (a *spill*) counts
   as "doesn't fit" too. A model that fits stays on Ollama.
2. **Benchmark.** The same workload on plain Ollama and on each split
   candidate (all eligible helpers, best link first; GPU-only helpers before
   adding CPU-only ones): a ~512-token prompt, then 128 tokens at temperature
   0 with a fixed seed, through the real front door, tunnel and all. Measured:
   load time, time to first token, prompt and generation tokens/s. Results
   are kept in `state.json` (last 20 per model).
3. **Choose.** Split only if its generation speed beats Ollama by
   `planner.min_gain`, or if Ollama couldn't run the model at all.
4. **Keep it honest.** The choice is tied to the model's digest, the context,
   the llama.cpp build and the exact helpers with their tunnel modes. If any
   of these changes (a helper leaves, the model is re-pulled), the model goes
   back to Ollama until it is benchmarked again.

Benchmarks run when you ask (`models benchmark`, `B` in the Models tab) and,
with `planner.auto_benchmark` (on by default), by themselves for models that
were used in the last day and don't fit - but only after the hub has been idle
for `planner.idle_seconds` (5 minutes). An automatic benchmark unloads
Ollama's models while it runs and gives way the moment a request arrives.

Your pins always win over the benchmark (`models pin`, `force-local`); a
pinned model is still never split if it fits on the hub's GPU alone.

### What a split session does

The first request for a model routed to split mode starts a session:

1. The hub finds the model's GGUF file in Ollama's store (via `/api/show`; no
   second copy).
2. Ollama is asked to unload its models (`keep_alive: 0`) and the hub waits
   until `/api/ps` is empty, so the two never fight over VRAM.
3. Each helper of the plan starts `rpc-server` (bound to its loopback) and
   reports how much memory its owner lends right now.
4. The hub places the layers: its own GPU first, then the helpers, the rest on
   its CPU; a helper never gets more than its owner's cap. `llama-server`
   starts on `127.0.0.1` with exactly that placement. The first load sends
   the weights to the helpers (the Models tab shows progress and an ETA in
   secure mode); helpers cache them on disk, so later loads are quick.

Clients don't notice: `/api/chat` and `/api/generate` are translated to and
from llama-server (streaming NDJSON, the final `done` object with token counts
and durations, tool calls, `format`, `think`, sampling options), `/v1/*`
requests pass straight through, and `/api/ps` lists the split model with
`"ollamapower": {"engine": "split", "nodes": [...]}`. The session stops after
`split.keep_alive` seconds idle (Ollama gets the GPU back on its next
request), when a helper leaves or its owner pauses, or when the plan changes.

`rpc-server` and `llama-server` die with the process that started them, even
when it is killed outright (`kill -9`, Task Manager): on Linux through
`PR_SET_PDEATHSIG`, on Windows through a kill-on-close Job Object. macOS has no
equivalent; after a hard kill there, end leftovers with `pkill rpc-server` or
`pkill llama-server` (a normal stop, Ctrl+C or `kill`, cleans up everywhere).

### When something breaks

- **Before the first token** (a helper vanished, llama-server didn't start or
  stalled): the request is answered by Ollama instead, transparently.
- **Mid-answer:** the stream ends with an Ollama-style `{"error": ...}`
  object (never a fake `done`), and the model is marked *degraded*.
- **Stalls:** no token for `split.stall_timeout` seconds counts as a failure.
- A degraded model runs on Ollama for at least `planner.retry_after` seconds
  and until every helper of its plan is back; then split mode is used again.
  Each failure in a row doubles the wait (up to an hour); an answer through
  split mode starts it over. Failures that won't go away by themselves
  (llama.cpp can't load the model, llama.cpp missing or the wrong build) wait
  until the model, its context or llama.cpp change. A helper pausing or
  leaving its schedule is never a failure. The reason, and the last lines of
  llama-server's output on a crash, are in the warnings and the event log.

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

Every value in `config.toml` and a helper's `agent.toml` can be overridden
with `OLLAMAPOWER_<SECTION>__<KEY>` env vars, e.g.
`OLLAMAPOWER_HUB__MAX_CONCURRENCY=2` on the hub or
`OLLAMAPOWER_LENDING__MAX_VRAM_GB=8` on a helper; `OLLAMAPOWER_CONFIG_DIR`
moves the directory. Both files read the same environment, so each one skips
the variables that belong to the other (`[hub]` and `[llamacpp]` exist in both
with different keys); a name neither file knows is an error, as is an unknown
key in a file. A value set by an env var wins over the file while it is set:
editing that rule in the agent TUI or with `agent rules` changes the running
agent but is not written to `agent.toml`. Values are checked strictly: whole
numbers stay whole (`2.5` is refused, not truncated), TOML strings are not read
as numbers, and `ollama.url` needs an explicit port. The cluster token lives only in `.env` (or the `OLLAMAPOWER_TOKEN`
env var). On disk, all owner-only: `config.toml`, `.env`, `hub_key.pem`,
`state.json` (+ `.bak`), `logs/`, and while running `run/`. A helper keeps
`agent.toml`, `.env`, `known_hub` and `node_id` in the same kind of directory
(`--config-dir` on `agent` commands to choose another; `llamacpp
install|status --config-dir` too), and `agent-run/` while running.

## How it works

```
clients ──► OllamaPower Hub (127.0.0.1:11434, Ollama-compatible API)
              ├─ local-ollama → Ollama on 127.0.0.1:11435
              └─ split        → llama-server --rpc → local relay ══ TLS 1.3 ══► helper relay → rpc-server (127.0.0.1)
```

- **Ollama doesn't do multi-machine inference.** Split mode uses llama.cpp's RPC
  backend (`rpc-server` on helpers, `llama-server --rpc` on the Hub), loading the
  GGUF blob Ollama already has on disk, so there is no second copy of the model.
- llama.cpp RPC has no authentication or encryption, so `rpc-server` only ever
  listens on the helper's loopback. The Hub reaches it through a mutually
  authenticated TLS 1.3 tunnel (or, if both owners opt in, plain TCP limited
  to the hub's address).
- No GUI, ever: a terminal UI (Textual) plus a full CLI, and a headless mode.

## Roadmap

| Phase | Delivers |
|---|---|
| 0 Bootstrap (done) | Style notes, DCP reuse map, DCP networking/crypto/auth/scheduling ported with tests, lint + test + audit CI |
| 1 Front door (done) | Ollama-compatible passthrough with true streaming, priority queue, `/ollamapower/status`, `doctor` port migration |
| 2 Agents & telemetry (done) | Control plane with protocol version and Hub key pinning, GPU telemetry, net tests, Hub and agent TUIs, CLI |
| 3 Split MVP (done) | llama.cpp pin/download/verify, Ollama blob resolver, `rpc-server`/`llama-server` lifecycle, layer placement under the lending caps, API translation, direct-mode relay |
| 4 Secure tunnel (done) | Per-session pinned TLS 1.3 relays, hub-address allowlist checked before TLS, two-owner direct mode, `nodes tunnel-test` overhead measurement |
| 5 Planner (done) | Memory estimate, spill detection, benchmarks through the real path, automatic "never slower" choice tied to a benchmark key, pins and exclusions, idle auto-benchmarks, transparent fallback, degraded plans and recovery |
| **6 Lending & polish** (done) | Lending schedules, auto-yield with graceful drain, `agent.example.toml`, start scripts, `models benchmarks --markdown` for real numbers |

**v2: any PC helps** ([`docs/DESIGN_V2.md`](docs/DESIGN_V2.md)):

| Phase | Delivers |
|---|---|
| A RAM pooling (done) | The hub's and helpers' RAM pooled (GPU-less hubs and helpers first-class); a helper lends its GPU and its RAM together; MoE experts in each machine's RAM, hot parts on its GPU (hub); an MoE-aware cost model predicting tok/s; a model too big for the hub routed to split at once and benchmarked without the hopeless Ollama run; Wi-Fi helpers |
| B `models suggest` (done) | "With these PCs you can run X at about N tok/s", for installed models and a catalog of popular ones |
| C Speculative decoding (done) | Models with MTP heads: drafted tokens checked in one trip through every machine; benchmarked, kept only if faster |
| D Tensor split (done) | Dense models on wired machines, every machine working on every token (llama.cpp's experimental `-sm tensor`); benchmarked, kept only if faster |
| E Later | Helpers fetch their slice of the model themselves; most-used experts pinned to GPUs |

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

Project docs: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) (real measurements), [`docs/FIELD_TEST.md`](docs/FIELD_TEST.md) (the first two-PC run: what broke, what is open), [`docs/STYLE_NOTES.md`](docs/STYLE_NOTES.md) (conventions shared
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
