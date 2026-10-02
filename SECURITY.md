# Security

OllamaPower lets other machines on your network take part in running your
models, so it is built to fail closed. This file describes the threat model and
what is (and isn't) protected. It is updated after every build phase; this is
the **Phase 2** version: the front door and the control plane between the hub
and its helpers.

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
  127.0.0.1), using Ollama's own rules. `Origin` is forwarded, so Ollama's CORS
  policy (`OLLAMA_ORIGINS`) still decides which web origins may call it, and
  its CORS answers are passed back unchanged.
- **Ollama stays local.** `ollama.url` must be a loopback address and may not
  be the front door's own port. `doctor` flags Ollama's port answering on any
  non-loopback interface and a non-loopback `OLLAMA_HOST`.
- **Header hygiene.** Only `Content-Type` and `Accept` are forwarded to Ollama
  (never `Authorization`, cookies or `X-OllamaPower-*`); only `Content-Type`
  comes back. No `Server` header, no docs/OpenAPI routes, unknown paths `404`.
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
- **Agent requests are refused for now.** Split-mode and overflow requests are
  answered "not available" until Phases 3 and 6.

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
  understates a fast link; the split-mode tunnel is measured separately later.
- **llama.cpp RPC is unauthenticated and unencrypted (Phases 3-4).** It will
  only ever listen on a helper's loopback, reached through a mutually
  authenticated TLS 1.3 tunnel; plain TCP only as an explicit per-helper opt-in
  with a permanent warning.
