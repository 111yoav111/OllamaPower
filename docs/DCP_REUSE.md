# DCP reuse map

OllamaPower continues [DCP - Dynamic Compute Power](https://github.com/111yoav111/Dynamic-Compute-Power)
(MIT, by Yoav). This file maps every DCP file to its fate here, read at DCP
commit `20facc2` ("UI change + test file fix").

Fates: **reuse** (ported with renames and cleanups only), **adapt** (the
structure stays, the behaviour changes), **replace** (the idea stays, the code
is new), **drop** (not carried over).

| DCP file | Fate | Where it goes | Phase | Notes |
|---|---|---|---|---|
| `src/networking/packets.py` | reuse | `ollamapower/core/packets.py` | 0 (2 for new types) | Binary `ControlPacket` (12 bytes) and `TaskPacket` (28-byte header + `zstd(msgpack(...))` payload) kept. The payload is now a plain msgpack dict instead of a render/simulation task object, so `task_from_dict`, `detect_task_type` and `TASK_FLAGS.render/simulation` are gone; `TASK_FLAGS` becomes `PayloadKind`. `parse_packets` raises `ValueError` instead of printing and returning `None`. Phase 2 adds `protocol_version` and the Section 11 packet types. |
| `src/networking/network_io.py` | reuse | `ollamapower/core/network_io.py` | 0 | `read_one_packet` and the locked/encrypted write path kept. The task-specific senders (`net_send_task`, `net_send_subtask`, `net_send_result`) collapse into one `send_packet`; control senders (`hello`, `welcome`, `status`, `disconnect`, `heartbeat`) kept. A clean EOF between frames returns `None`; EOF inside a frame raises. |
| `src/networking/encrypt_layer.py` | reuse | `ollamapower/core/encrypt_layer.py` | 0 (2 for pinning) | RSA-2048 OAEP-SHA256 key transport + AES-256-GCM frames with a fresh random 96-bit nonce per message, kept; frames now also bind sender role + counter (see below). `master_handshake`/`worker_handshake` → `hub_handshake`/`node_handshake`. Adds `public_key_fingerprint()`; Phase 2 uses it for trust-on-first-use pinning of the Hub key (the plain DCP handshake can be MITM'd). |
| token auth (`DCP_TOKEN` in `.env`, checked in `server.py`) | reuse | `ollamapower/core/auth.py` | 0 | Renamed `OLLAMAPOWER_TOKEN`. Comparison is now constant time (`hmac.compare_digest`; DCP used `!=`), tokens are generated with 256 bits from `secrets`, and short/empty tokens are rejected. Rate-limiting failed auths per IP comes with the control server in Phase 2. |
| `src/loadbalancer/load_balancer.py` — `TaskPriority`, `TaskRecord`, heap, skip counter, forced dispatch, requeue | reuse | `ollamapower/scheduling/task_pool.py` | 0 | Extracted into a `TaskPool` class (priority + FIFO heap, `MAX_SKIPS = 10` forced dispatch, retry/requeue). Priorities map 1:1 onto the `X-OllamaPower-Priority: critical\|high\|normal\|low` header (Phase 1). |
| `src/loadbalancer/load_balancer.py` — `WorkerState`, dispatch/heartbeat/metrics loops, callbacks, `get_snapshot()` | adapt | `ollamapower/scheduling/load_balancer.py` | 0, scoring in 2/5 | Worker → node. Dispatch, heartbeat timeout (→ requeue), metrics and the `get_snapshot()` dict are kept. The CPU-only `load_score` stays for now; Phase 2/5 scores by VRAM free, GPU util, link quality, warm model and queue length. |
| `src/loadbalancer/load_balancer.py` — `SubtaskGroup`, `register_subtask_group`, `subtask_*` merge path | drop | — | — | Result merging of divisible CPU tasks has no LLM equivalent; a split model is one `llama-server` process, not N partial results. |
| `src/loadbalancer/task_pool.py` | drop | — | — | Despite the name it is DCP's *random demo task generator* (difficulty tables, `generate_*_task`, `task_pool_loop`), not the queue. The real queue lived in `load_balancer.py` and is reused above. |
| `src/tasks/divisible_task.py` | replace | `ollamapower/hub/planner/` | 3, 5 | The idea of splitting work by each node's free capacity becomes a *split plan*: layers/tensor-split proportions across Hub GPU and helper devices. `_compute_weights` (capacity-proportional weights with a floor) is the model for `--tensor-split`. |
| `src/tasks/task.py` | drop | — | — | Abstract executable task; agents run processes (`rpc-server`, Ollama), not Python tasks. |
| `src/tasks/simul_task.py`, `src/tasks/rend_task.py` | drop | — | — | Simulation/render demo workloads; kept out of the repo per spec. |
| `src/master/server.py` | adapt | `ollamapower/hub/control_server.py` | 2 | Kept: accept loop, handshake → hello → token → welcome, per-node read loop, kick, disconnect cleanup. Changes: protocol version, key pinning, failed-auth rate limit, `CAPABILITIES`/`HEARTBEAT` telemetry, `RPC_*` and overflow packets. Bound address comes from config (DCP hard-coded `0.0.0.0`). |
| `src/worker/client.py` | adapt | `ollamapower/agent/client.py` | 2 | Kept: reconnect every 5 s, handshake + token, receive loop, status heartbeat. Replaced: `ProcessPoolExecutor` task execution → `rpc_manager.py` (manages `rpc-server`) and `overflow.py` (calls local Ollama). |
| `src/master/master_ui.py` | drop (logic kept) | `ollamapower/hub/tui/` | 2 | tkinter removed entirely. The logic behind it (worker table from `on_worker_status_change`, kick action, metrics panel, event popups) is rebuilt as the Textual Hub TUI reading `get_snapshot()`. |
| `src/worker/worker_ui.py` | drop (logic kept) | `ollamapower/agent/tui/` | 2 | Same: connection state, assigned id and current work become the Textual agent TUI. |
| `src/master/main_master.py`, `src/worker/main_worker.py` | replace | `ollamapower/__main__.py` | 1, 2 | `ollamapower hub` / `ollamapower agent` (Typer). No UI thread + backend thread split: Textual and the servers share one event loop, like OllamaGate. |
| `tests/test_packets.py` | reuse | `tests/test_packets.py` | 0 | Rewritten as pytest functions over generic payloads; covers every control packet type. |
| `tests/test_auth.py` | reuse | `tests/test_auth.py` | 0 | Same handshake → hello → token → welcome flow over a real local socket, as async pytest tests; adds empty-token and constant-time checks. |
| `tests/test_lb.py` | reuse | `tests/test_task_pool.py`, `tests/test_load_balancer.py` | 0 | Worker-state, priority, dispatch and requeue assertions kept; the difficulty-table asserts are dropped with the generator. |
| `tests/test_tasks.py` | drop | — | — | Tests the render/simulation tasks. |
| `start.bat` | replace | `scripts/start_hub.*`, `scripts/start_agent.*` | 7 | |
| `requirements.txt` | replace | `pyproject.toml` + hashed lock files | 0 | DCP's file was UTF-16 and pinned `asyncio`/`logger`/`uuid` from PyPI (stdlib modules; the PyPI `asyncio` package is an obsolete backport). Only real dependencies are kept. |
| `assets/*.png` | drop | — | — | GUI images. |
| `LICENSE` (MIT) | reuse | `LICENSE` | 0 | DCP's copyright line is kept alongside OllamaPower's. |
| `README.md` | replace | `README.md` | 0, 7 | Credits DCP. |

## Issues found in DCP while porting

Fixed in the port (with tests where it makes sense):

- Token compared with `!=` (timing side channel) → `hmac.compare_digest`.
- No minimum token strength → `ollamapower init` will generate 256-bit tokens;
  `check_token` refuses empty ones.
- `parse_packets` printed to stdout and returned `None` on bad data; callers
  could crash on `None` → raises `ValueError`.
- `TaskPacket.from_bytes` built the object with `cls().__new__(cls)` (runs
  `__init__` for nothing) → `cls.__new__(cls)`.
- `decrypt_from_reader` raised on EOF before a frame but returned `None` on EOF
  inside one (the opposite of what callers expect) → clean EOF between frames is
  `None`, a truncated frame raises.
- AES-GCM frames had no sequence number and one key served both directions,
  so frames could be replayed, reordered or reflected within a session → each
  frame's associated data binds the sender role and a per-direction counter;
  encryption happens under the write lock so frames stay in counter order.
- The plaintext (pre-auth) read path accepted 64 MB task packets → only the
  12-byte control packets are read before encryption.
- Result packets got a fresh wire id and no link to their request (DCP matched
  them through the task object) → `build_result_packet` takes `request_id`,
  carried in the `parent_id` header field.
- Late or duplicate results/failures from a node that no longer owned a task
  were applied (double execution, nodes stuck BUSY); a node dropping during a
  send queued the task twice; any non-`OSError` from the send callback killed the
  dispatch loop; skips were counted while no node existed, so capacity checks
  were bypassed ~2 s after start; timed-out nodes stayed OFFLINE forever →
  ownership checks, an idempotent queue, a broad catch around the send,
  skips only counted when a node exists, and timed-out nodes are kicked.
- Task ids came from a module-global `itertools.count`; kept, but the payload
  carries its own ids from Phase 2 on.

Left for the phase that rewrites the code (noted so they are not forgotten):

- The handshake trusts whatever RSA key the server sends (no pinning) — Phase 2,
  TOFU fingerprint pinning.
- `ctrl_hello` / `ctrl_welcome` travel unencrypted after the key exchange and
  carry no protocol version — Phase 2.
- The master bound `0.0.0.0` unconditionally and dropped workers at ≥ 90 % CPU
  — Phase 2 (configurable bind; GPU-aware lending rules instead).
- No rate limit or ban on failed auth — Phase 2.
