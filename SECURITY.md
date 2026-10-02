# Security

OllamaPower lets other machines on your network take part in running your
models, so it is built to fail closed. This file describes the threat model and
what is (and isn't) protected. It is updated after every build phase; this is
the **Phase 1** version: the inherited building blocks and the front door.

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
  socket `0600`, or `127.0.0.1` on Windows) that requires a 256-bit secret from
  an owner-only file, and the hub proves itself with an HMAC on every reply
  (OllamaGate's design).

Out of scope here: any local process running as you can use the front door,
as it could use Ollama directly before.

### Control plane (Phase 0 building blocks)

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

- **Hub key pinning (Phase 2).** The DCP handshake trusts whatever RSA key the
  Hub presents, so an active man-in-the-middle could read the token. Phase 2
  pins the Hub key fingerprint on each agent (trust on first use, confirmed by
  the owner) and refuses a changed key.
- **Failed-auth rate limiting (Phase 2).** Per-IP limits and temporary bans
  arrive with the control-plane server.
- **llama.cpp RPC is unauthenticated and unencrypted (Phases 3-4).** It will
  only ever listen on a helper's loopback, reached through a mutually
  authenticated TLS 1.3 tunnel; plain TCP only as an explicit per-helper opt-in
  with a permanent warning.
