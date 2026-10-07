# Notes for the merge with OllamaGate

OllamaPower started from OllamaGate's code for the parts both need. This file
lists what is shared, where it has diverged on purpose, and the conventions
already aligned, so the merge is a diff review rather than archaeology.

## Shared modules

| OllamaGate | OllamaPower | State |
|---|---|---|
| `textsafe.py` | `textsafe.py` | identical |
| `server/proxy.py` | `hub/frontdoor/proxy.py` | diverged, see below |
| `bans.py` | `hub/bans.py` | diverged: also counts abandoned handshakes, `succeeded()` clears an IP's recent failures |
| `admin/ipc.py` | `admin/ipc.py` | diverged: both use challenge-response; commands differ |
| `admin/service.py` | `admin/service.py` | diverged: different command set (nodes, models, plans) |
| `ollama_status.py` | `ollama/client.py` | diverged: also `/api/show`, blob lookup |

## Deliberate divergences in the proxy

- **Header forwarding is a blocklist, not an allowlist.** Ollama needs
  `Origin` (its own CORS check), `Authorization` and `Content-Encoding` (its
  Codex and cloud proxies) and `User-Agent`. Dropped: hop-by-hop headers,
  `Host`, `Content-Length`, `Cookie`, `Accept-Encoding` (forced to
  `identity` so bodies stream verbatim), `Expect`, and every
  `X-OllamaPower-*` header. Responses drop hop-by-hop headers, `Set-Cookie`,
  `Server` and `Date`. An internet-facing build must keep stripping its own
  auth header before forwarding (OllamaGate's token), which the allowlist
  did implicitly.
- **No read timeout on Ollama by default** (`ollama.read_timeout = 0`): a
  big model loading or a long prompt can take minutes before the first byte.
  Connect/pool timeouts give 502, a read timeout (if set) gives 504.
- **`ProxyStreamResponse` takes `transform`, `first`, `chunks` and
  `cors_origin`** for split mode (llama-server answers translated to
  Ollama's format, CORS headers set the way Ollama would).
- **`GenerationTap`** reads Ollama's final `done` object for tokens/s; it
  keeps no content.

## Kept on purpose (council decision)

- **h11, not httptools.** httptools would save ~0.3 ms per request but
  uvicorn's header-size cap (`h11_max_incomplete_event_size`, 16 KB) only
  exists for h11. The merged, internet-facing build needs that cap even
  more. Revisit only with an equivalent cap.
- **`sniffio` is a dependency**: without it httpcore repeats a failed import
  on every request (0.35 ms).

## Conventions already aligned

- `-c` means `--config` (a `config.toml`) on every command, as in
  OllamaGate; `--config-dir` is long-only.
- Same dependency ranges for fastapi, starlette, uvicorn, httpx, textual,
  typer and cryptography.
- Same log rule: metadata only, never prompt or response content.
- Commits: Conventional Commits, owner as author.

## To decide at merge time

- One package and CLI (`ollamapower` with OllamaGate's pairing as a
  subsystem, or the reverse), and one config file with both sections.
- The front door would then face the internet through OllamaGate's TLS and
  pairing; OllamaPower's host check and origin policy assume loopback
  clients (`hub.host` must stay loopback unless OllamaGate's auth is in front).
