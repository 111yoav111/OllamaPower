"""Session encryption for the control plane, inherited from DCP.

Handshake (once per connection):

1. The hub sends its RSA-2048 public key (PEM, length-prefixed).
2. The node picks a random 256-bit AES key, encrypts it to that public key
   with RSA-OAEP-SHA256 and sends it back (length-prefixed).

After that every packet is one frame ``[4-byte length][12-byte nonce][AES-256-GCM
ciphertext + tag]``. The nonce is fresh random for every message, so a nonce is
never reused under the same key (2^-48 collision odds only after ~2^32 frames).

The plain handshake trusts whatever key the server sends, so on its own it can
be man-in-the-middled. Phase 2 pins the hub key's fingerprint on the agent
(trust on first use, like SSH ``known_hosts``) using ``public_key_fingerprint``.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import struct

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

log = logging.getLogger("ollamapower.crypto")

RSA_KEY_SIZE = 2048
AES_KEY_SIZE = 32  # 256 bits
AES_NONCE_SIZE = 12  # 96 bits, the GCM standard
AES_TAG_SIZE = 16
FRAME_LAYER_FORMAT = "!I"
FRAME_LAYER_SIZE = struct.calcsize(FRAME_LAYER_FORMAT)
# PEM keys and RSA blobs are well under this; anything bigger is garbage or an attack.
MAX_HANDSHAKE_FRAME = 10 * 1024
# One encrypted packet. Large enough for streamed chunks and capability reports.
MAX_FRAME = 30 * 1024 * 1024


class CryptoError(ValueError):
    """A frame failed to decrypt or authenticate; the connection must be dropped."""


def generate_rsa_keypair() -> rsa.RSAPrivateKey:
    """The hub's key pair; one per hub, shared by every node connection."""
    return rsa.generate_private_key(public_exponent=65537, key_size=RSA_KEY_SIZE)


def rsa_public_key_to_pem(private_key: rsa.RSAPrivateKey) -> bytes:
    return private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def rsa_key_from_pem(pem_data: bytes) -> rsa.RSAPublicKey:
    key = serialization.load_pem_public_key(pem_data)
    if not isinstance(key, rsa.RSAPublicKey) or key.key_size < RSA_KEY_SIZE:
        raise CryptoError("hub key must be an RSA key of at least 2048 bits")
    return key


def public_key_fingerprint(public_key: rsa.RSAPublicKey) -> str:
    """SHA-256 over the DER SubjectPublicKeyInfo, as hex - what the owner compares."""
    der = public_key.public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
    return hashlib.sha256(der).hexdigest()


def _oaep() -> padding.OAEP:
    return padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None)


def rsa_encrypt_aes_key(public_key: rsa.RSAPublicKey, aes_key: bytes) -> bytes:
    return public_key.encrypt(aes_key, _oaep())


def rsa_decrypt_aes_key(private_key: rsa.RSAPrivateKey, ciphertext: bytes) -> bytes:
    try:
        return private_key.decrypt(ciphertext, _oaep())
    except ValueError as exc:
        raise CryptoError("could not decrypt the session key") from exc


# ----- AES-256-GCM frames ---------------------------------------------------------------


def encrypt_packet(aes_key: bytes, plaintext: bytes) -> bytes:
    """Return a frame ready for the socket: ``[length][nonce][ciphertext+tag]``."""
    nonce = os.urandom(AES_NONCE_SIZE)
    payload = nonce + AESGCM(aes_key).encrypt(nonce, plaintext, None)
    return struct.pack(FRAME_LAYER_FORMAT, len(payload)) + payload


def decrypt_packet(aes_key: bytes, frame: bytes) -> bytes:
    """Decrypt a frame *without* its length prefix (the reader already consumed it)."""
    if len(frame) < AES_NONCE_SIZE + AES_TAG_SIZE:
        raise CryptoError("frame too short for AES-GCM")
    try:
        return AESGCM(aes_key).decrypt(frame[:AES_NONCE_SIZE], frame[AES_NONCE_SIZE:], None)
    except InvalidTag:
        raise CryptoError("frame failed authentication") from None


# ----- handshake framing ----------------------------------------------------------------


def frame_bytes(data: bytes) -> bytes:
    """Length-prefix ``data`` (used for the handshake blobs)."""
    return struct.pack(FRAME_LAYER_FORMAT, len(data)) + data


async def read_frame(reader: asyncio.StreamReader) -> bytes:
    """Read one length-prefixed handshake blob (PEM key or encrypted session key)."""
    raw_len = await reader.readexactly(FRAME_LAYER_SIZE)
    length = struct.unpack(FRAME_LAYER_FORMAT, raw_len)[0]
    if length == 0 or length > MAX_HANDSHAKE_FRAME:
        raise CryptoError(f"handshake frame length out of range ({length} bytes)")
    return await reader.readexactly(length)


# ----- session --------------------------------------------------------------------------


class SessionCrypto:
    """The AES key of one hub <-> node connection."""

    def __init__(self, aes_key: bytes) -> None:
        if len(aes_key) != AES_KEY_SIZE:
            raise CryptoError(f"AES key must be {AES_KEY_SIZE} bytes")
        self.aes_key = aes_key
        # Set on the node side: the key the hub presented, for pinning.
        self.peer_fingerprint: str | None = None

    @classmethod
    async def hub_handshake(
        cls, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, rsa_private_key: rsa.RSAPrivateKey
    ) -> SessionCrypto:
        """Hub side: send our public key, receive and unwrap the node's session key."""
        writer.write(frame_bytes(rsa_public_key_to_pem(rsa_private_key)))
        await writer.drain()
        encrypted_key = await read_frame(reader)
        aes_key = rsa_decrypt_aes_key(rsa_private_key, encrypted_key)
        log.debug("handshake complete (hub side)")
        return cls(aes_key)

    @classmethod
    async def node_handshake(cls, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> SessionCrypto:
        """Node side: receive the hub key, send it a fresh session key.

        The caller must check ``peer_fingerprint`` against the pinned hub key
        before sending anything secret (the token).
        """
        hub_key = rsa_key_from_pem(await read_frame(reader))
        aes_key = os.urandom(AES_KEY_SIZE)
        writer.write(frame_bytes(rsa_encrypt_aes_key(hub_key, aes_key)))
        await writer.drain()
        session = cls(aes_key)
        session.peer_fingerprint = public_key_fingerprint(hub_key)
        log.debug("handshake complete (node side)")
        return session

    def encrypt(self, plaintext: bytes) -> bytes:
        return encrypt_packet(self.aes_key, plaintext)

    def decrypt(self, frame: bytes) -> bytes:
        return decrypt_packet(self.aes_key, frame)

    async def decrypt_from_reader(self, reader: asyncio.StreamReader) -> bytes | None:
        """Read and decrypt one frame.

        Returns ``None`` on a clean end of stream between frames; raises if the
        stream ends inside a frame or the frame is invalid.
        """
        try:
            raw_len = await reader.readexactly(FRAME_LAYER_SIZE)
        except asyncio.IncompleteReadError as exc:
            if not exc.partial:
                return None
            raise CryptoError("connection closed inside a frame length") from None
        length = struct.unpack(FRAME_LAYER_FORMAT, raw_len)[0]
        if length == 0 or length > MAX_FRAME:
            raise CryptoError(f"encrypted frame length out of range ({length} bytes)")
        try:
            frame = await reader.readexactly(length)
        except asyncio.IncompleteReadError:
            raise CryptoError("connection closed inside a frame") from None
        return self.decrypt(frame)

    async def encrypt_to_writer(self, writer: asyncio.StreamWriter, lock: asyncio.Lock, plaintext: bytes) -> None:
        """Encrypt and write one frame; ``lock`` keeps concurrent frames from interleaving."""
        frame = self.encrypt(plaintext)
        async with lock:
            writer.write(frame)
            await writer.drain()


# ----- token exchange -------------------------------------------------------------------


async def send_token(writer: asyncio.StreamWriter, token: str, crypto: SessionCrypto) -> None:
    """Node -> hub: the shared token, inside an encrypted frame."""
    writer.write(crypto.encrypt(token.encode("utf-8")))
    await writer.drain()


async def receive_token(reader: asyncio.StreamReader, crypto: SessionCrypto) -> str:
    """Hub side: read the node's token."""
    plain = await crypto.decrypt_from_reader(reader)
    if plain is None:
        raise CryptoError("connection closed before the token arrived")
    try:
        return plain.decode("utf-8")
    except UnicodeDecodeError:
        raise CryptoError("token is not valid UTF-8") from None
