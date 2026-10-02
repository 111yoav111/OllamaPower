"""Refuse requests whose ``Host`` header isn't a local name (DNS rebinding protection).

A web page can make the browser send requests to ``127.0.0.1:11434`` under a
hostname the attacker controls (``evil.example`` re-pointed at 127.0.0.1) and
read the answers. Ollama blocks this on its loopback port by checking the
``Host`` header; since we now own that port, we do the same check, using
Ollama's rules: loopback, private and unspecified IPs, ``localhost``, this
machine's hostname, and names under ``.localhost``, ``.local`` and
``.internal``.
"""

from __future__ import annotations

import ipaddress
import json
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


class LocalHostOnlyMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self.hostname = socket.gethostname().lower()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
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
