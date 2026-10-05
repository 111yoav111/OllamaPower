# Security

OllamaPower lets other machines on your network take part in running your
models, so it is built to fail closed. This file describes the threat model and
what is (and isn't) protected. It is updated after every build phase; this is
the **Phase 6** version: the front door, the control plane between the hub
and its helpers, split mode (llama.cpp RPC) with its encrypted tunnel, the
planner that decides when to use it, and the helper owner's lending controls.

## Threat model

**Attacker:** anyone on the same network as the Hub or a helper - another
device on the LAN, a guest on the Wi-Fi, a compromised IoT gadget - who can
connect to open ports, sniff traffic or sit in the middle of a connection.

**Assets:** the Hub and helper machines themselves (no remote code execution),
the prompts and responses, the models and GPU time, and the shared cluster
token.

Out of scope: exposing the Hub to the internet (that is OllamaGate's job), and
an attacker who already runs code as the owner on the Hub or a helper.

## What's protected

### Front door (Phase 1)

- **Loopback only.** The Ollama-compatible API listens on `127.0.0.1`; any
  other address is refused unless started with `--i-know-what-im-doing`, and
  then a warning stays in `status`. It has no authentication by design: it
  replaces Ollama's own local port, and remote access belongs to OllamaGate.
- **Web pages can't use it.** Requests whose `Host` isn't a local name are
  refused with `403` (DNS rebinding: a page under `evil.example` re-pointed at
  127.0.0.1), using Ollama's own rules (and, like Ollama, only while the front
  door listens on loopback). `Origin` is forwarded, so Ollama's CORS policy
  (`OLLAMA_ORIGINS`) still decides which web origins may call it, and its
  CORS answers are passed back unchanged. Requests served in split mode never
  reach Ollama, so the front door applies Ollama's origin policy itself
  (defaults plus `OLLAMA_ORIGINS` from the hub's environment): other origins
  get `403`, and llama-server's own "allow every origin" headers are never
  passed on.
- **Ollama stays local.** `ollama.url` must be a loopback address and may not
  be the front door's own port. `doctor` flags Ollama's port answering on any
  non-loopback interface and a non-loopback `OLLAMA_HOST`.
- **Header hygiene.** Requests reach Ollama with the client's headers except
  hop-by-hop ones, `Host`, `Content-Length`, cookies and every
  `X-OllamaPower-*` header; Ollama needs the rest (Origin for its CORS check,
  Authorization and Content-Encoding for its Codex and cloud proxies).
  Responses come back with Ollama's headers except hop-by-hop ones,
  `Set-Cookie`, `Server` and `Date`. No docs/OpenAPI routes, unknown paths `404`.
- **Bounded resources.** Inference bodies are capped (100 MB), the queue is
  capped (256) with a wait timeout, connections are capped (512), upstream
  connect/read timeouts apply, and a client that disconnects stops its
  generation in Ollama. Active streams are cut as soon as shutdown begins.
- **No content in logs.** Request logs hold metadata only; the generation-speed
  tap reads Ollama's final timing object and keeps nothing else.
- **Owner-only files.** `.env`, `state.json`, its backup and the logs are
  `0600` in a `0700` directory, written atomically; symlinked or foreign-owned
  directories are refused. `doctor` checks the permissions. `hub` refuses to
  run as root.
- **Admin channel.** `status` reaches the hub only over a local channel (Unix
  socket `0600`, or `127.0.0.1` on Windows) keyed by a 256-bit secret in an
  owner-only file. The secret never crosses the channel: the hub sends a fresh
  challenge per connection, every request answers it with an HMAC over a
  single-use nonce, and every reply carries the hub's own HMAC. A process that
  grabs a stale port learns nothing and can't forge answers.

Out of scope here: any local process running as you can use the front door,
as it could use Ollama directly before.

### Control plane (Phase 2)

- **Who may connect.** The control port (9000) drops connections from outside
  `control.allowed_networks` (private ranges, loopback and CGNAT/Tailscale by
  default) before sending a single byte; `doctor` warns if public ranges are
  added. At most `max_nodes` handshakes run at once and each must finish
  within 10 s.
- **The hub can't be impersonated.** The hub's RSA key is persistent
  (`hub_key.pem`, owner-only) and agents pin its SHA-256 fingerprint on first
  use (`known_hub`), after the owner compared it with what the hub prints, or
  given up front with `--hub-fingerprint`. A headless agent with no way to ask
  refuses unknown keys. The pin is checked **before the token is sent**; a
  changed key stops the agent with an alert, and it does not retry.
- **Token.** Checked in constant time inside the encrypted session; a wrong
  token stops the agent (no retry loop).
- **Failed logins are banned.** 5 failures (bad token, malformed or oversized
  handshake, timeout) from one IP in 10 minutes ban it for 10 minutes,
  doubling on repeat up to a day. Banned IPs are dropped before the handshake.
- **Nothing large before authentication.** Frames are capped at 64 KB until
  the token checks out.
- **Strict protocol.** One module defines every message, its direction and
  field types; anything else drops the connection. Protocol majors must match
  (an outdated helper is shown as incompatible, with the fix).
- **Agent reports are untrusted.** Capabilities, telemetry and test results
  are bounded in size and depth and made terminal-safe before the hub stores
  or shows them. Agents can't make the hub run anything.
- **The helper's owner decides.** Lending rules live on the helper and only its
  owner changes them; the hub can pause or kick a helper but never widen what
  it lends. Unencrypted direct mode needs **both** owners to opt in and keeps a
  warning on screen.
- **Helpers never run models on their own.** They only lend GPU/CPU to the
  Hub's engine (split mode). There is no way for the Hub to make a helper run
  Ollama or any other program than the pinned, hash-checked `rpc-server`, and
  the Hub chooses none of its arguments beyond one of the devices the owner
  lends.

### Split mode (Phase 3)

- **Pinned, verified llama.cpp.** Every machine runs one pinned release. Each
  archive's size and SHA-256 are stored in OllamaPower (llama.cpp publishes no
  checksums); a download that doesn't match is deleted and never unpacked or
  run. Unpacking refuses absolute paths, `..`, links pointing outside the
  install directory and device files. The build number is checked before use;
  a helper with another build is shown as incompatible.
- **`rpc-server` only on loopback.** llama.cpp RPC has no authentication or
  encryption and executes whatever its client asks, so it is always started
  with `-H 127.0.0.1` on a random port and only while the Hub uses it.
- **The helper's relay.** The agent forwards to `rpc-server` through a relay
  that listens only on the interface the hub connects from and only during a
  session, accepts at most 8 connections, and drops any connection that
  doesn't come from the hub's address **on the raw socket, before reading a
  byte or doing any TLS work**.
- **The helper's owner keeps control.** The agent checks every `rpc_start`
  itself against its own rules (paused, GPU allow-list, CPU lending, direct
  mode, VRAM cap) instead of trusting the Hub. Pausing or changing the rules
  ends a running session at once. `rpc-server` has no memory limit of its own,
  so the cap is enforced by the Hub's layer placement (`--tensor-split`); a
  compromised Hub could ignore it, as it could ask for anything the owner
  lends.
- **`llama-server` only on the hub's loopback, with a key.** It listens on
  `127.0.0.1` on a random port. It allows every web origin, so it is started
  with a random API key (passed in its environment, not on the command line)
  that only the front door sends; a web page that finds the port gets `401`.
- **Untrusted helper answers.** Ports, memory figures and device names in
  `rpc_ready` are type- and range-checked; the Hub connects to the address it
  sees for the helper's control connection, never to an address the helper
  names.
- **Model files are only read.** The Hub reads Ollama's GGUF blob (header
  parsed by a bounded, dependency-free reader) and llama-server loads it
  read-only. Request content is translated in memory and never logged.

### Split-mode tunnel (Phase 4)

- **Encrypted by default.** Split-mode traffic (model weights, activations,
  RPC commands) travels inside **TLS 1.3 only**, with mutual authentication.
- **Per-session pinned certificates, no CA.** For every split session (and
  every tunnel test) the hub and the helper each generate a fresh EC P-256
  key and a self-signed certificate. Only the certificates are exchanged,
  inside `rpc_start`/`rpc_ready` on the control channel, which is itself
  encrypted and authenticated (cluster token + pinned hub key). Private keys
  never leave the machine that made them and are never written anywhere but
  a short-lived owner-only temp file while `ssl` loads them. Each side trusts
  exactly the other's certificate for that session; after the handshake the
  certificate's SHA-256 and the TLS version are checked again. A certificate
  from an earlier session, or from anyone else, is refused.
- **Direct mode needs both owners.** The hub's owner (`nodes tunnel <helper>
  direct`) *and* the helper's owner (`allow_direct`). If only one opted in,
  secure is used; the helper itself refuses a direct-mode request its owner
  didn't allow. Nothing ever falls back to plaintext on its own. While direct
  mode is in effect, the TUI and `status` show a warning.
- **The hub's local relay serves only llama-server's user.** `llama-server`
  speaks plain TCP to a relay on the hub's `127.0.0.1`, which carries it into
  the tunnel. On Linux the relay checks (via `/proc/net/tcp`) that each
  connection comes from a process of the user running the hub and refuses
  anyone else, so other local users can't reach a helper's `rpc-server`
  through it.
- **Overhead is measured, not guessed.** `nodes tunnel-test` measures the real
  path in both modes so the owner can decide; see `docs/BENCHMARKS.md`.

### Planner (Phase 5)

- **No client can choose the engine.** Benchmarks steer their own requests
  with `X-OllamaPower-Engine` / `-Helpers` / `-Ctx` headers, which are
  honoured only together with `X-OllamaPower-Bench` carrying a random secret
  the hub makes at every start (compared in constant time). Without it they
  are ignored, and like every `X-OllamaPower-*` header they never reach
  Ollama.
- **Benchmarks use a synthetic prompt** (fixed English text with a random
  first line), never anything a client sent; only timings are stored.
- **Answers are never faked.** If llama-server dies mid-answer, the stream
  ends with an error object, never with a `done` object that would make a
  cut-off answer look complete. A failure before the first token is retried
  on Ollama instead, so the client gets a real answer.
- **Decisions can't go stale silently.** A split decision is tied to the
  model digest, context, llama.cpp build and the exact helpers with their
  tunnel modes; if a helper's tunnel mode changes (e.g. to direct), the old
  decision no longer applies.
- **Availability, honestly:** an automatic benchmark unloads Ollama's models
  while it runs (only when the hub has been idle for `planner.idle_seconds`,
  and it stops as soon as a request arrives). Turn it off with
  `planner.auto_benchmark = false`.

### Lending controls (Phase 6)

- **The schedule and auto-yield are decided on the helper**, from its own
  clock and its own GPU readings; the hub is only told the result
  (`lending`, `paused` or `yielding`, with a reason) and can't override it.
  The agent refuses `rpc_start` whenever it isn't lending.
- **Pause is instant:** any split session using the helper ends at once.
  Schedule ends and auto-yield are graceful: the hub gets up to 45 s to finish
  its current answers and release the helper, then the agent stops
  `rpc-server` by itself anyway.
- **Auto-yield looks at nothing new:** only the GPU load percentage the hub
  already sees in the heartbeats. No process names or lists are read or
  reported; the hub learns "yielding" and which GPU was busy.

### Control-plane encryption (Phase 0 building blocks)

- **Encrypted control plane.** Every packet after the handshake is one
  AES-256-GCM frame with a fresh random 96-bit nonce (tested for uniqueness);
  tampered or truncated frames, and frames from another session, fail
  authentication and drop the connection. The session key is sent to the Hub
  with RSA-2048 OAEP-SHA256.
- **No replay, reordering or reflection.** Each frame is bound (as GCM
  associated data) to its sender's role and a per-direction counter, so a
  frame re-sent, reordered or bounced back to its sender is rejected. Frames
  are encrypted under the connection's write lock, so concurrent senders can't
  put them on the wire out of order.
- **Nothing large before authentication.** Until the token is checked only the
  fixed 12-byte control packets are read in plaintext.
- **Safe deserialisation.** Payloads are msgpack dicts (never pickle);
  extension types stay inert data. Decompressed payloads are capped at 64 MB,
  so a small zstd "bomb" can't exhaust memory, and every size field is
  bounded before anything is allocated.
- **Token auth.** Helpers prove membership with a 256-bit token from
  `secrets`, sent only inside the encrypted session and compared in constant
  time. Empty or short tokens are refused.

## Known gaps (closed in later phases)

- **Trust on first use.** If someone already impersonates the hub at the
  moment a helper connects for the very first time *and* the owner confirms
  without comparing fingerprints, the helper trusts the wrong hub. Compare the
  fingerprint, or pass it with `--hub-fingerprint`.
- **Network-test throughput** runs over the encrypted control channel, so it
  understates a fast link; `nodes tunnel-test` measures the split-mode path.
- **Direct mode is unencrypted by design.** When both owners opt in, model
  weights, activations (which reveal prompts and answers to someone who can
  decode them) and RPC commands cross the LAN in plain TCP. Anyone who can
  sniff the LAN, or spoof the hub's IP, could read it or reach `rpc-server`
  through the relay while a session runs. Use it only on a wired LAN you
  trust; secure mode has none of these problems.
- **Other local users on the hub (macOS, Windows)** could connect to the
  hub's local tunnel relay while a session runs and so reach the helper's
  `rpc-server`; the owner check above is Linux-only for now. Don't run the
  hub on a machine shared with people you don't trust.
- **Other local users on a helper** can connect to `rpc-server` on its
  loopback port while a session runs (it has no authentication). Don't lend a
  machine shared with people you don't trust.
- **Other local users on the hub** can reach `llama-server`'s port, but not
  use it without the key; `/health` is open.
