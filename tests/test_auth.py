"""Token auth over a real local socket (ported from DCP's test_auth.py).

The flow is DCP's: RSA handshake -> plaintext hello -> encrypted token ->
plaintext welcome (accepted) or closed connection (rejected).
"""

from __future__ import annotations

import asyncio

import pytest

from ollamapower.core import auth
from ollamapower.core.encrypt_layer import SessionCrypto, generate_rsa_keypair, receive_token, send_token
from ollamapower.core.network_io import net_send_hello, net_send_welcome, read_one_packet
from ollamapower.core.packets import PACKET_FLAGS, ControlPacket

TOKEN = auth.new_token()


@pytest.fixture(scope="module")
def hub_key():
    return generate_rsa_keypair()


async def _run(hub_key, token_sent: str, token_expected: str) -> tuple[str, str]:
    hub_result: asyncio.Future[str] = asyncio.get_running_loop().create_future()

    async def handle(reader, writer):
        try:
            crypto = await SessionCrypto.hub_handshake(reader, writer, hub_key)
            hello = await read_one_packet(reader)
            assert isinstance(hello, ControlPacket) and hello.packet_type is PACKET_FLAGS.ctrl_hello
            token = await receive_token(reader, crypto)
            if not auth.check_token(token, token_expected):
                hub_result.set_result("rejected")
                return
            await net_send_welcome(writer, asyncio.Lock(), 1)
            hub_result.set_result("accepted")
        finally:
            writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    async with server:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        try:
            crypto = await SessionCrypto.node_handshake(reader, writer)
            await net_send_hello(writer, asyncio.Lock())
            await send_token(writer, token_sent, crypto)
            pkt = await asyncio.wait_for(read_one_packet(reader), 5)
        finally:
            writer.close()
        node_result = (
            "connected"
            if isinstance(pkt, ControlPacket) and pkt.packet_type is PACKET_FLAGS.ctrl_welcome and pkt.worker_id == 1
            else "rejected"
        )
        return node_result, await asyncio.wait_for(hub_result, 5)


async def test_correct_token(hub_key):
    assert await _run(hub_key, TOKEN, TOKEN) == ("connected", "accepted")


async def test_wrong_token(hub_key):
    assert await _run(hub_key, auth.new_token(), TOKEN) == ("rejected", "rejected")


async def test_empty_expected_token_never_matches(hub_key):
    assert await _run(hub_key, "", "") == ("rejected", "rejected")


def test_new_tokens_are_256_bit_and_unique():
    tokens = {auth.new_token() for _ in range(100)}
    assert len(tokens) == 100
    for token in tokens:
        assert token.startswith(auth.TOKEN_PREFIX)
        assert len(token) - len(auth.TOKEN_PREFIX) >= 43  # 32 bytes, base64url


def test_check_token():
    assert auth.check_token(TOKEN, TOKEN)
    assert not auth.check_token(TOKEN[:-1], TOKEN)
    assert not auth.check_token(None, TOKEN)
    assert not auth.check_token(TOKEN, None)
    assert not auth.check_token("", "")


@pytest.mark.parametrize("bad", [None, "", "short-token"])
def test_weak_tokens_are_refused(bad):
    with pytest.raises(auth.TokenError):
        auth.validate_token(bad)


def test_generated_token_validates():
    assert auth.validate_token(TOKEN) == TOKEN
