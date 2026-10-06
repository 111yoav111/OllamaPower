"""Listening TCP sockets that no other program can share or take over.

Windows handles port reuse differently from POSIX:

* ``SO_REUSEADDR`` there lets a socket bind a port that another socket is
  already *listening* on. uvicorn sets it on every platform, so on Windows
  the hub started "fine" while Ollama still listened on port 11434, and
  connections went to either program at random.
* A listening socket without ``SO_EXCLUSIVEADDRUSE`` can have its address
  taken over by another program that binds it with ``SO_REUSEADDR``, which
  then receives the connections meant for us.

So every port OllamaPower listens on is opened here: with
``SO_EXCLUSIVEADDRUSE`` (and never ``SO_REUSEADDR``) on Windows, and with
``SO_REUSEADDR`` on POSIX, where it only lets a restart bind past old
``TIME_WAIT`` connections and never lets two sockets listen on one address.
"""

from __future__ import annotations

import errno
import socket
import sys

WINDOWS = sys.platform == "win32"
BACKLOG = 128
# Binding a port that is in use: EADDRINUSE, or WSAEADDRINUSE / WSAEACCES on
# Windows (the latter when the other socket holds it exclusively).
_IN_USE = {errno.EADDRINUSE, 10048, 10013}


def listen_socket(host: str, port: int, *, dualstack: bool = False, backlog: int = BACKLOG) -> socket.socket:
    """A bound, listening TCP socket on ``host:port`` (``dualstack``: IPv4 and IPv6 on ``::``)."""
    family = socket.AF_INET6 if dualstack or ":" in host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        if WINDOWS:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)  # type: ignore[attr-defined]
        else:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if family == socket.AF_INET6 and hasattr(socket, "IPV6_V6ONLY"):
            sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0 if dualstack else 1)
        sock.bind((host, port))
        sock.listen(backlog)
    except BaseException:
        sock.close()
        raise
    return sock


def address_in_use(exc: OSError) -> bool:
    return exc.errno in _IN_USE or getattr(exc, "winerror", None) in _IN_USE
