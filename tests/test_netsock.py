"""Listening sockets can't be shared or taken over (the Windows SO_REUSEADDR trap)."""

from __future__ import annotations

import errno
import socket
import sys

import pytest
from typer.testing import CliRunner

from ollamapower.__main__ import cli
from ollamapower.netsock import address_in_use, listen_socket


def test_listen_socket_accepts_connections():
    sock = listen_socket("127.0.0.1", 0)
    try:
        with socket.create_connection(sock.getsockname()):
            pass
    finally:
        sock.close()


def test_a_port_we_listen_on_cannot_be_bound_again():
    """With SO_REUSEADDR on the intruder (what uvicorn and port thieves use) the bind still fails."""
    sock = listen_socket("127.0.0.1", 0)
    try:
        with socket.socket() as intruder:
            intruder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            with pytest.raises(OSError) as exc:
                intruder.bind(sock.getsockname())
            assert address_in_use(exc.value)
    finally:
        sock.close()


def test_we_cannot_bind_a_port_someone_else_listens_on():
    """Ollama still on the port: the hub must notice, not share it."""
    with socket.socket() as other:
        other.bind(("127.0.0.1", 0))
        other.listen()
        with pytest.raises(OSError) as exc:
            listen_socket("127.0.0.1", other.getsockname()[1])
        assert address_in_use(exc.value)


@pytest.mark.skipif(sys.platform != "win32", reason="SO_EXCLUSIVEADDRUSE exists only on Windows")
def test_windows_sockets_are_exclusive():
    sock = listen_socket("127.0.0.1", 0)
    try:
        assert sock.getsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE)
        assert not sock.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR)
    finally:
        sock.close()


def test_address_in_use_ignores_other_errors():
    assert not address_in_use(OSError(errno.EACCES, "permission denied"))


def test_hub_reports_a_busy_front_door(tmp_path, monkeypatch):
    with socket.socket() as other:
        other.bind(("127.0.0.1", 0))
        other.listen()
        port = other.getsockname()[1]
        (tmp_path / "config.toml").write_text(f"[hub]\nport = {port}\n[control]\nenabled = false\n")
        monkeypatch.setenv("OLLAMAPOWER_CONFIG_DIR", str(tmp_path))
        monkeypatch.delenv("OLLAMAPOWER_TOKEN", raising=False)
        result = CliRunner().invoke(cli, ["hub", "--no-tui", "--allow-root"])
    assert result.exit_code == 1, result.output
    assert "the port is in use" in result.output
