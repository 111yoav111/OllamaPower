"""The admin channel: challenge-response, the secret never on the wire, impostors refused."""

from __future__ import annotations

import asyncio
import json
import socket
import sys
import threading

import pytest

from ollamapower.admin import ipc


async def handler(cmd, args):
    if cmd == "boom":
        raise ipc.AdminError("bad thing")
    return {"cmd": cmd, "args": args}


async def test_round_trip_and_errors(tmp_path):
    control = await ipc.start_control_server(tmp_path, handler)
    try:

        def use():
            with ipc.ControlClient(tmp_path) as c:
                assert c.call("ping") == "pong"
                assert c.call("status", x=1) == {"cmd": "status", "args": {"x": 1}}
                with pytest.raises(ipc.AdminError, match="bad thing"):
                    c.call("boom")

        await asyncio.to_thread(use)
    finally:
        await control.close()
    assert not (tmp_path / "run" / "admin.json").exists()


async def test_second_hub_refused(tmp_path):
    control = await ipc.start_control_server(tmp_path, handler)
    try:
        with pytest.raises(RuntimeError, match="already running"):
            await ipc.start_control_server(tmp_path, handler)
    finally:
        await control.close()


async def _raw_session(info: dict) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    if info["transport"] == "unix":
        return await asyncio.open_unix_connection(info["path"])
    return await asyncio.open_connection("127.0.0.1", info["port"])


async def test_wrong_secret_and_replayed_nonce_are_refused(tmp_path):
    control = await ipc.start_control_server(tmp_path, handler)
    info = json.loads((tmp_path / "run" / "admin.json").read_text())
    try:
        reader, writer = await _raw_session(info)
        challenge = json.loads(await reader.readline())["challenge"]
        nonce = "n" * 32
        good = ipc._mac(info["secret"], "client", challenge, nonce)
        writer.write(json.dumps({"nonce": nonce, "auth": good, "cmd": "ping"}).encode() + b"\n")
        assert json.loads(await reader.readline())["ok"]
        writer.write(json.dumps({"nonce": nonce, "auth": good, "cmd": "ping"}).encode() + b"\n")  # replay
        assert json.loads(await reader.readline()) == {"ok": False, "error": "unauthorized"}
        writer.close()

        reader, writer = await _raw_session(info)
        await reader.readline()
        # The old style (secret in the request) no longer works.
        writer.write(json.dumps({"secret": info["secret"], "nonce": nonce, "cmd": "ping"}).encode() + b"\n")
        assert json.loads(await reader.readline())["error"] == "unauthorized"
        writer.close()
    finally:
        await control.close()


@pytest.mark.skipif(sys.platform == "win32", reason="uses a Unix socket impostor")
def test_impostor_never_sees_the_secret(tmp_path):
    """A process squatting on the channel learns nothing and can't forge an answer."""
    secret = "s" * 43
    sock_path = tmp_path / "impostor.sock"
    info = tmp_path / "admin.json"
    info.write_text(json.dumps({"transport": "unix", "path": str(sock_path), "secret": secret}))
    received: list[bytes] = []
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(sock_path))
    server.listen(1)

    def impostor():
        conn, _ = server.accept()
        conn.sendall(b'{"challenge": "' + b"c" * 32 + b'"}\n')
        data = conn.recv(65536)
        received.append(data)
        nonce = json.loads(data)["nonce"]
        forged = ipc._mac("guess", "server", "c" * 32, nonce)
        conn.sendall(json.dumps({"ok": True, "result": "forged", "proof": forged}).encode() + b"\n")
        conn.close()

    t = threading.Thread(target=impostor)
    t.start()
    client = ipc.ControlClient.from_info_file(info, timeout=5)
    with pytest.raises(ipc.NotRunning, match="did not prove"):
        client.call("status")
    client.close()
    t.join(5)
    server.close()
    assert secret.encode() not in received[0]


async def test_slow_hub_is_busy_not_down(tmp_path):
    """A reply slower than the timeout means "busy", not "not running" (the CLI must not say "start it")."""

    async def slow(cmd, args):
        await asyncio.sleep(1.0)
        return "late"

    control = await ipc.start_control_server(tmp_path, slow)
    try:

        def use():
            with ipc.ControlClient(tmp_path, timeout=0.2) as c:
                with pytest.raises(ipc.AdminError, match="didn't answer within 0.2 s"):
                    c.call("models.unpin")

        await asyncio.to_thread(use)
    finally:
        await control.close()


async def test_cli_reports_a_busy_hub_with_exit_1(tmp_path, monkeypatch, capsys):
    import typer

    from ollamapower.cli_common import call

    async def slow(cmd, args):
        await asyncio.sleep(1.0)
        return "late"

    control = await ipc.start_control_server(tmp_path, slow)
    try:
        with pytest.raises(typer.Exit) as exc:
            await asyncio.to_thread(call, tmp_path, "models.unplan", timeout=0.2)
    finally:
        await control.close()
    assert exc.value.exit_code == 1
    err = capsys.readouterr().err
    assert "busy" in err and "Start it" not in err
