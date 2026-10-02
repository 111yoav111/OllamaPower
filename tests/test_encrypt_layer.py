"""AES-GCM framing and the RSA handshake."""

from __future__ import annotations

import asyncio

import pytest

from ollamapower.core import encrypt_layer as e

KEY = b"k" * e.AES_KEY_SIZE


def test_round_trip():
    frame = e.encrypt_packet(KEY, b"hello")
    assert e.decrypt_packet(KEY, frame[e.FRAME_LAYER_SIZE :]) == b"hello"


def test_fresh_nonce_per_message():
    nonces = set()
    for _ in range(2000):
        frame = e.encrypt_packet(KEY, b"same plaintext")
        nonces.add(frame[e.FRAME_LAYER_SIZE : e.FRAME_LAYER_SIZE + e.AES_NONCE_SIZE])
    assert len(nonces) == 2000


def test_tampered_frame_is_rejected():
    body = bytearray(e.encrypt_packet(KEY, b"hello")[e.FRAME_LAYER_SIZE :])
    body[-1] ^= 1
    with pytest.raises(e.CryptoError):
        e.decrypt_packet(KEY, bytes(body))


def test_wrong_key_is_rejected():
    body = e.encrypt_packet(KEY, b"hello")[e.FRAME_LAYER_SIZE :]
    with pytest.raises(e.CryptoError):
        e.decrypt_packet(b"x" * e.AES_KEY_SIZE, body)


def test_short_frame_is_rejected():
    with pytest.raises(e.CryptoError):
        e.decrypt_packet(KEY, b"\x00" * 20)


def test_key_size_enforced():
    with pytest.raises(e.CryptoError):
        e.SessionCrypto(b"short")


def _reader(data: bytes) -> asyncio.StreamReader:
    reader = asyncio.StreamReader()
    reader.feed_data(data)
    reader.feed_eof()
    return reader


async def test_reader_eof_between_frames_is_clean():
    crypto = e.SessionCrypto(KEY)
    reader = _reader(crypto.encrypt(b"one"))
    assert await crypto.decrypt_from_reader(reader) == b"one"
    assert await crypto.decrypt_from_reader(reader) is None


@pytest.mark.parametrize("cut", [2, 10])
async def test_reader_eof_inside_a_frame_raises(cut):
    crypto = e.SessionCrypto(KEY)
    with pytest.raises(e.CryptoError):
        await crypto.decrypt_from_reader(_reader(crypto.encrypt(b"payload")[:cut]))


async def test_oversized_frame_length_is_refused():
    crypto = e.SessionCrypto(KEY)
    with pytest.raises(e.CryptoError):
        await crypto.decrypt_from_reader(_reader((e.MAX_FRAME + 1).to_bytes(4, "big")))


async def test_handshake_agrees_on_key_and_reports_fingerprint():
    hub_key = e.generate_rsa_keypair()
    done: asyncio.Future[e.SessionCrypto] = asyncio.get_running_loop().create_future()

    async def on_connect(reader, writer):
        done.set_result(await e.SessionCrypto.hub_handshake(reader, writer, hub_key))
        writer.close()

    server = await asyncio.start_server(on_connect, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    async with server:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        node = await e.SessionCrypto.node_handshake(reader, writer)
        hub = await asyncio.wait_for(done, 5)
        writer.close()
    assert node.aes_key == hub.aes_key
    assert node.peer_fingerprint == e.public_key_fingerprint(hub_key.public_key())
    assert len(node.peer_fingerprint) == 64


async def test_handshake_rejects_oversized_key_frame():
    with pytest.raises(e.CryptoError):
        await e.read_frame(_reader((e.MAX_HANDSHAKE_FRAME + 1).to_bytes(4, "big")))
