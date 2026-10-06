"""Session encryption for the control plane, inherited from DCP.

Handshake (once per connection, protocol 2):

1. The hub sends its RSA-2048 public key (PEM) followed by ``HUB_NONCE_TAG``
   and a fresh random 256-bit **hub nonce**, as one length-prefixed blob.
2. The node picks a random 256-bit secret, encrypts it to that public key with
   RSA-OAEP-SHA256 and sends back ``NODE_TAG``, its own random 256-bit **node
   nonce** and the encrypted secret (one length-prefixed blob).
3. Both sides derive the AES-256 session key with HKDF-SHA256 from the secret,
   salted with both nonces and bound to the hub key's fingerprint.

The secret only ever crosses the wire RSA-encrypted, so the session key is as
secret as it was when the RSA-wrapped value was used directly; the nonces are
public. What the hub nonce adds is freshness: every connection has a key the
hub helped choose, so a recorded session replayed to the hub (the same RSA
blob, then the same frames) derives a different key and its first frame fails
authentication. Protocol 1 used the node's secret as the key directly, which
let anyone who recorded a session replay all of it.

After that every packet is one frame ``[4-byte length][12-byte nonce][AES-256-GCM
ciphertext + tag]``. The nonce is fresh random for every message, so a nonce is
never reused under the same key (2^-48 collision odds only after ~2^32 frames).

Each frame is also bound (as GCM associated data) to its sender's role and a
per-direction counter. A frame that is replayed, reordered, dropped-then-skipped
or reflected back to its sender fails authentication. DCP had no such binding.

Older peers: the hub's first blob still starts with a plain PEM key, which a
protocol-1 agent reads and ignores the rest of; the hub recognises that
agent's reply (a bare RSA blob), finishes a ``legacy`` session only so it can
send a readable "update this helper" reject, and never admits it. A protocol-2
agent that meets a protocol-1 hub (a bare PEM key) stops with
``HandshakeVersionError`` before sending anything.

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
from typing import Literal

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

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
AAD_LABEL = b"ollamapower-frame-v1"
HANDSHAKE_NONCE_SIZE = 32
# After the hub's PEM key: marks a protocol-2 hub (a protocol-1 agent's PEM parser ignores it).
HUB_NONCE_TAG = b"\nollamapower-handshake-v2 nonce:"
# Starts a protocol-2 node's reply; a protocol-1 node sends the bare RSA blob.
NODE_TAG = b"OPnode2:"
KDF_INFO = b"ollamapower-control-session-v2"

Role = Literal["hub", "node"]
_ROLE_BYTE: dict[str, bytes] = {"hub": b"H", "node": b"N"}


class CryptoError(ValueError):
    """A frame failed to decrypt or authenticate; the connection must be dropped."""


class HandshakeVersionError(CryptoError):
    """The hub speaks an older handshake (protocol 1) that this agent no longer accepts."""


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


def derive_session_key(secret: bytes, hub_nonce: bytes, node_nonce: bytes, hub_fingerprint: str) -> bytes:
    """The AES-256 key of one connection: HKDF-SHA256 over the node's secret, salted with both nonces."""
    if len(secret) != AES_KEY_SIZE or len(hub_nonce) != HANDSHAKE_NONCE_SIZE or len(node_nonce) != HANDSHAKE_NONCE_SIZE:
        raise CryptoError("handshake values have the wrong size")
    info = KDF_INFO + bytes.fromhex(hub_fingerprint)
    return HKDF(algorithm=hashes.SHA256(), length=AES_KEY_SIZE, salt=hub_nonce + node_nonce, info=info).derive(secret)


# ----- AES-256-GCM frames ---------------------------------------------------------------


def frame_aad(sender: Role, counter: int) -> bytes:
    return AAD_LABEL + _ROLE_BYTE[sender] + counter.to_bytes(8, "big")


def encrypt_packet(aes_key: bytes, plaintext: bytes, aad: bytes | None = None) -> bytes:
    """Return a frame ready for the socket: ``[length][nonce][ciphertext+tag]``."""
    nonce = os.urandom(AES_NONCE_SIZE)
    payload = nonce + AESGCM(aes_key).encrypt(nonce, plaintext, aad)
    return struct.pack(FRAME_LAYER_FORMAT, len(payload)) + payload


def decrypt_packet(aes_key: bytes, frame: bytes, aad: bytes | None = None) -> bytes:
    """Decrypt a frame *without* its length prefix (the reader already consumed it)."""
    if len(frame) < AES_NONCE_SIZE + AES_TAG_SIZE:
        raise CryptoError("frame too short for AES-GCM")
    try:
        return AESGCM(aes_key).decrypt(frame[:AES_NONCE_SIZE], frame[AES_NONCE_SIZE:], aad)
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
    """The AES key and frame counters of one hub <-> node connection, seen from ``role``.

    Not safe to share between concurrent writers without a lock: frames must
    reach the wire in the order they were encrypted (``encrypt_to_writer`` and
    ``network_io.send_packet`` encrypt while holding the connection's lock).
    """

    def __init__(self, aes_key: bytes, role: Role, legacy: bool = False) -> None:
        if len(aes_key) != AES_KEY_SIZE:
            raise CryptoError(f"AES key must be {AES_KEY_SIZE} bytes")
        if role not in _ROLE_BYTE:
            raise ValueError(f"role must be 'hub' or 'node' (got {role!r})")
        self.aes_key = aes_key
        self.role: Role = role
        self.peer_role: Role = "node" if role == "hub" else "hub"
        self.send_counter = 0
        self.recv_counter = 0
        # Largest frame we accept; kept small until the peer has authenticated.
        self.max_frame = MAX_FRAME
        # Set on the node side: the key the hub presented, for pinning.
        self.peer_fingerprint: str | None = None
        # Hub side: a protocol-1 node's session (no hub nonce, so replayable). It may only be told to update.
        self.legacy = legacy

    @classmethod
    async def hub_handshake(
        cls, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, rsa_private_key: rsa.RSAPrivateKey
    ) -> SessionCrypto:
        """Hub side: send our public key and a fresh nonce, unwrap the node's secret, derive the key."""
        hub_nonce = os.urandom(HANDSHAKE_NONCE_SIZE)
        writer.write(frame_bytes(rsa_public_key_to_pem(rsa_private_key) + HUB_NONCE_TAG + hub_nonce))
        await writer.drain()
        reply = await read_frame(reader)
        rsa_size = rsa_private_key.key_size // 8
        prefix = len(NODE_TAG) + HANDSHAKE_NONCE_SIZE
        if len(reply) == prefix + rsa_size and reply.startswith(NODE_TAG):
            node_nonce = reply[len(NODE_TAG) : prefix]
            secret = rsa_decrypt_aes_key(rsa_private_key, reply[prefix:])
            fingerprint = public_key_fingerprint(rsa_private_key.public_key())
            log.debug("handshake complete (hub side)")
            return cls(derive_session_key(secret, hub_nonce, node_nonce, fingerprint), "hub")
        if len(reply) == rsa_size:
            # A protocol-1 node: its secret is the key. Good for one reject, never for a session.
            log.debug("protocol-1 handshake (hub side)")
            return cls(rsa_decrypt_aes_key(rsa_private_key, reply), "hub", legacy=True)
        raise CryptoError("unrecognised handshake reply")

    @classmethod
    async def node_handshake(cls, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> SessionCrypto:
        """Node side: receive the hub key and nonce, send our nonce and a fresh secret, derive the key.

        The caller must check ``peer_fingerprint`` against the pinned hub key
        before sending anything secret (the token).
        """
        blob = await read_frame(reader)
        tail = len(HUB_NONCE_TAG) + HANDSHAKE_NONCE_SIZE
        if len(blob) <= tail or blob[-tail:-HANDSHAKE_NONCE_SIZE] != HUB_NONCE_TAG:
            if blob.lstrip().startswith(b"-----BEGIN"):
                raise HandshakeVersionError(
                    "the hub runs an older OllamaPower (control protocol 1) that this helper can't use: "
                    "update OllamaPower on the hub"
                )
            raise CryptoError("unrecognised handshake from the hub")
        hub_key = rsa_key_from_pem(blob[:-tail])
        hub_nonce = blob[-HANDSHAKE_NONCE_SIZE:]
        secret, node_nonce = os.urandom(AES_KEY_SIZE), os.urandom(HANDSHAKE_NONCE_SIZE)
        writer.write(frame_bytes(NODE_TAG + node_nonce + rsa_encrypt_aes_key(hub_key, secret)))
        await writer.drain()
        fingerprint = public_key_fingerprint(hub_key)
        session = cls(derive_session_key(secret, hub_nonce, node_nonce, fingerprint), "node")
        session.peer_fingerprint = fingerprint
        log.debug("handshake complete (node side)")
        return session

    def encrypt(self, plaintext: bytes) -> bytes:
        """Encrypt the next outgoing frame (advances the send counter)."""
        frame = encrypt_packet(self.aes_key, plaintext, frame_aad(self.role, self.send_counter))
        self.send_counter += 1
        return frame

    def decrypt(self, frame: bytes) -> bytes:
        """Decrypt the next incoming frame; anything but the expected next frame fails."""
        plaintext = decrypt_packet(self.aes_key, frame, frame_aad(self.peer_role, self.recv_counter))
        self.recv_counter += 1
        return plaintext

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
        if length == 0 or length > self.max_frame:
            raise CryptoError(f"encrypted frame length out of range ({length} bytes)")
        try:
            frame = await reader.readexactly(length)
        except asyncio.IncompleteReadError:
            raise CryptoError("connection closed inside a frame") from None
        return self.decrypt(frame)

    async def encrypt_to_writer(self, writer: asyncio.StreamWriter, lock: asyncio.Lock, plaintext: bytes) -> None:
        """Encrypt and write one frame; ``lock`` keeps frames whole and in counter order."""
        async with lock:
            writer.write(self.encrypt(plaintext))
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
