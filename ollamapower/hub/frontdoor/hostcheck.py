"""Browser protections Ollama has, applied where we answer instead of Ollama.

**Host check (DNS rebinding).** A web page can make the browser send requests
to ``127.0.0.1:11434`` under a hostname the attacker controls (``evil.example``
re-pointed at 127.0.0.1) and read the answers. Ollama blocks this on a
loopback listener by checking the ``Host`` header; since we now own that port,
we do the same, with Ollama's rules: loopback, private and unspecified IPs,
``localhost``, this machine's hostname, and names under ``.localhost``,
``.local`` and ``.internal``. Like Ollama, the check applies only when the
front door listens on loopback (a LAN listener is reached by LAN names).

**Origin check (CORS).** Requests Ollama serves keep their ``Origin`` header,
so Ollama's own policy decides. Requests served in split mode never reach
Ollama, so ``origin_allowed`` applies Ollama's default policy itself:
``localhost``/``127.0.0.1``/``0.0.0.0`` on any port, ``app://``, ``file://``,
``tauri://``, ``vscode-webview://``, ``vscode-file://``, plus whatever
``OLLAMA_ORIGINS`` (in the hub's environment) adds.
"""

from __future__ import annotations

import fnmatch
import ipaddress
import json
import os
import socket

from starlette.types import ASGIApp, Receive, Scope, Send

LOCAL_SUFFIXES = (".localhost", ".local", ".internal")


def split_host(header: str) -> str:
    """The hostname part of a ``Host`` header (``[::1]:11434`` -> ``::1``)."""
    header = header.strip().lower()
    if header.startswith("["):
        return header[1 : header.find("]")] if "]" in header else header
    if header.count(":") == 1:
        return header.partition(":")[0]
    return header  # bare host, or a bare IPv6 address


def host_allowed(host: str, hostname: str | None = None) -> bool:
    if host in ("", "localhost"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        return ip.is_loopback or ip.is_private or ip.is_unspecified
    if host == (hostname if hostname is not None else socket.gethostname()).lower():
        return True
    return host.endswith(LOCAL_SUFFIXES)


def allowed_origins(env: dict[str, str] | None = None) -> list[str]:
    """Ollama's ``envconfig.AllowedOrigins``: ``OLLAMA_ORIGINS`` first, then the defaults."""
    env = os.environ if env is None else env
    out = [o.strip() for o in env.get("OLLAMA_ORIGINS", "").split(",") if o.strip()]
    for host in ("localhost", "127.0.0.1", "0.0.0.0"):
        out += [f"http://{host}", f"https://{host}", f"http://{host}:*", f"https://{host}:*"]
    out += ["app://*", "file://*", "tauri://*", "vscode-webview://*", "vscode-file://*"]
    return out


def origin_allowed(origin: str, patterns: list[str]) -> bool:
    """Does ``origin`` match one of ``patterns`` (``*`` wildcards, case-insensitive, like gin-contrib/cors)?"""
    origin = origin.strip().lower()
    return any(fnmatch.fnmatchcase(origin, p.lower()) for p in patterns)


class LocalHostOnlyMiddleware:
    def __init__(self, app: ASGIApp, enforce: bool = True) -> None:
        self.app = app
        self.enforce = enforce  # False when the front door listens beyond loopback (as Ollama does)
        self.hostname = socket.gethostname().lower()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and self.enforce:
            host = ""
            for name, value in scope.get("headers", []):
                if name == b"host":
                    host = value.decode("latin-1")
                    break
            if not host_allowed(split_host(host), self.hostname):
                body = json.dumps({"error": "forbidden host"}).encode()
                await send(
                    {
                        "type": "http.response.start",
                        "status": 403,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode()),
                        ],
                    }
                )
                await send({"type": "http.response.body", "body": body})
                return
        await self.app(scope, receive, send)
