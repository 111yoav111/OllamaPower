"""Building blocks of the split-mode data plane: pinned mutual TLS and byte relays.

llama.cpp's RPC protocol has no authentication or encryption, so between the
hub and a helper it travels inside **mutually authenticated TLS 1.3**:

* Each side makes a fresh EC P-256 key and a self-signed certificate for every
  split session (``TunnelIdentity``). Private keys never leave the machine that
  made them; only certificates are exchanged, inside the control channel,
  which is already encrypted and authenticated (token + pinned hub key).
* Each side trusts exactly one certificate: the one its peer sent for this
  session. There is no CA, no hostname and nothing to reuse after the session.
  After the handshake the peer certificate's SHA-256 is compared again with
  the pinned one (defence in depth).
* TLS 1.3 only.

``pipe``/``splice`` copy bytes both ways with ``TCP_NODELAY`` set: RPC sends
many small messages per token, so Nagle's delay would cost more than TLS.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import hashlib
import logging
import os
import shutil
import socket
import ssl
import sys
import tempfile
from dataclasses import dataclass, field

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

log = logging.getLogger("ollamapower.tunnel")

RELAY_CHUNK = 256 * 1024
MAX_CERT_PEM = 4096  # a P-256 self-signed certificate is ~600 bytes
TLS_HANDSHAKE_TIMEOUT = 10.0
CERT_LIFETIME = datetime.timedelta(days=30)


class TunnelError(Exception):
    """The peer isn't who the session says it should be, or TLS failed."""


def cert_fingerprint(cert_pem: str) -> str:
    cert = x509.load_pem_x509_certificate(cert_pem.encode())
    return cert.fingerprint(hashes.SHA256()).hex()


def der_fingerprint(der: bytes) -> str:
    return hashlib.sha256(der).hexdigest()


def check_cert_pem(value: object) -> str:
    """A peer's certificate from the network: bounded, parseable PEM, or ``TunnelError``."""
    if not isinstance(value, str) or not 0 < len(value) <= MAX_CERT_PEM:
        raise TunnelError("missing or oversized tunnel certificate")
    try:
        cert_fingerprint(value)
    except ValueError:
        raise TunnelError("unreadable tunnel certificate") from None
    return value


@dataclass
class TunnelIdentity:
    """A fresh key and self-signed certificate for one split session."""

    cert_pem: str
    key_pem: bytes = field(repr=False)

    @classmethod
    def generate(cls, role: str) -> TunnelIdentity:
        key = ec.generate_private_key(ec.SECP256R1())
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, f"ollamapower-{role}")])
        now = datetime.datetime.now(datetime.UTC)
        cert = (
            x509.CertificateBuilder()
            .subject_name(name)
            .issuer_name(name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(minutes=5))
            .not_valid_after(now + CERT_LIFETIME)
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .sign(key, hashes.SHA256())
        )
        key_pem = key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
        )
        return cls(cert.public_bytes(serialization.Encoding.PEM).decode(), key_pem)

    @property
    def fingerprint(self) -> str:
        return cert_fingerprint(self.cert_pem)

    def context(self, server: bool, peer_cert_pem: str) -> ssl.SSLContext:
        """A TLS 1.3 context that presents this identity and trusts only ``peer_cert_pem``."""
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER if server else ssl.PROTOCOL_TLS_CLIENT)
        ctx.minimum_version = ssl.TLSVersion.TLSv1_3
        ctx.check_hostname = False  # identity is the pinned certificate, not a name
        ctx.verify_mode = ssl.CERT_REQUIRED
        ctx.load_verify_locations(cadata=peer_cert_pem)
        # ssl can only load a key from a file: write it to a fresh owner-only
        # directory and remove it again as soon as it is loaded.
        tmp = tempfile.mkdtemp(prefix="ollamapower-tls-")
        try:
            cert_path, key_path = os.path.join(tmp, "cert.pem"), os.path.join(tmp, "key.pem")
            for path, data in ((cert_path, self.cert_pem.encode()), (key_path, self.key_pem)):
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, "wb") as fh:
                    fh.write(data)
            ctx.load_cert_chain(cert_path, key_path)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return ctx


def check_peer(writer: asyncio.StreamWriter, expected_fingerprint: str) -> None:
    """After the handshake: the peer presented exactly the pinned certificate."""
    ssl_object = writer.get_extra_info("ssl_object")
    der = ssl_object.getpeercert(binary_form=True) if ssl_object is not None else None
    if not der or der_fingerprint(der) != expected_fingerprint:
        raise TunnelError("the tunnel peer presented an unexpected certificate")
    if ssl_object.version() != "TLSv1.3":
        raise TunnelError("the tunnel did not negotiate TLS 1.3")


def _proc_port(field: str) -> int:
    return int(field.rsplit(":", 1)[1], 16)


UNKNOWN = -1  # the connection tables can't be read at all


def loopback_peer_uid(
    peer_port: int, local_port: int, tables: tuple[str, ...] = ("/proc/net/tcp", "/proc/net/tcp6")
) -> int | None:
    """Linux: the user owning the other end of a loopback TCP connection.

    The client's socket is the row whose local port is ``peer_port`` and whose
    remote port is ``local_port`` (ours). None: no such row; ``UNKNOWN``: no
    table could be read (e.g. a hardened /proc).
    """
    readable = False
    for table in tables:
        try:
            with open(table, encoding="ascii") as fh:
                readable = True
                next(fh, None)
                for line in fh:
                    cols = line.split()
                    if len(cols) > 7 and _proc_port(cols[1]) == peer_port and _proc_port(cols[2]) == local_port:
                        return int(cols[7])
        except (OSError, ValueError):
            continue
    return None if readable else UNKNOWN


def same_user_on_loopback(writer: asyncio.StreamWriter) -> bool:
    """Is this loopback connection from a process of the user running us?

    True where it can't be told (not Linux, or /proc unreadable): the caller
    documents that gap rather than breaking split mode.
    """
    if not sys.platform.startswith("linux"):
        return True
    peer, local = writer.get_extra_info("peername"), writer.get_extra_info("sockname")
    if not peer or not local:
        return False
    uid = loopback_peer_uid(peer[1], local[1])
    if uid == UNKNOWN:
        log.warning("cannot read /proc/net/tcp: the tunnel can't check which local user connects")
        return True
    return uid is not None and uid == os.getuid()


def nodelay(writer: asyncio.StreamWriter) -> None:
    sock = writer.get_extra_info("socket")
    if sock is not None:
        with contextlib.suppress(OSError):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


async def pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> bool:
    """Copy until EOF; True if the EOF could be passed on (half-close), so the other way may go on."""
    try:
        while data := await reader.read(RELAY_CHUNK):
            writer.write(data)
            await writer.drain()
    except (OSError, asyncio.IncompleteReadError, ssl.SSLError):
        return False
    try:
        if writer.can_write_eof():  # TLS transports can't half-close
            writer.write_eof()
            return True
    except (OSError, RuntimeError, NotImplementedError):
        pass
    return False


async def splice(
    a: tuple[asyncio.StreamReader, asyncio.StreamWriter], b: tuple[asyncio.StreamReader, asyncio.StreamWriter]
) -> None:
    """Relay two connections into each other until both directions are done, then close both.

    When one direction ends and its EOF can't be passed on (TLS), both
    connections close at once: otherwise the far side would wait forever.
    """
    for _, w in (a, b):
        nodelay(w)
    tasks = {asyncio.ensure_future(pipe(a[0], b[1])), asyncio.ensure_future(pipe(b[0], a[1]))}
    try:
        while tasks:
            done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            if not all(t.result() for t in done):
                break
    finally:
        for t in tasks:
            t.cancel()
        for _, w in (a, b):
            w.close()
        await asyncio.gather(*tasks, return_exceptions=True)
