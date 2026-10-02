# Security

OllamaPower lets other machines on your network take part in running your
models, so it is built to fail closed. This file describes the threat model and
what is (and isn't) protected. It is updated after every build phase; this is
the **Phase 0** version, which covers only the inherited building blocks.

## Threat model

**Attacker:** anyone on the same network as the Hub or a helper - another
device on the LAN, a guest on the Wi-Fi, a compromised IoT gadget - who can
connect to open ports, sniff traffic or sit in the middle of a connection.

**Assets:** the Hub and helper machines themselves (no remote code execution),
the prompts and responses, the models and GPU time, and the shared cluster
token.

Out of scope: exposing the Hub to the internet (that is OllamaGate's job), and
an attacker who already runs code as the owner on the Hub or a helper.

## What's protected (Phase 0)

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
- **Front door, file permissions, `doctor` (Phase 1).** The Ollama-compatible
  API binds `127.0.0.1` only; secrets and state files are owner-only.
