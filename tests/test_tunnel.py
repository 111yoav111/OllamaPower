"""The split-mode tunnel: pinned TLS 1.3 identities, the helper's relay, the hub's relay."""

from __future__ import annotations

import asyncio
import ssl
import sys

import pytest

from ollamapower.agent.tunnel import EchoTarget, HelperRelay, same_ip
from ollamapower.core.tunnel import TunnelError, TunnelIdentity, check_cert_pem, splice
from ollamapower.hub.tunnel import HubRelay, TunnelClient, measure


@pytest.fixture(scope="module")
def ids():
    return {role: TunnelIdentity.generate(role) for role in ("hub", "helper", "evil")}


async def echo_through(relay: HelperRelay) -> EchoTarget:
    target = EchoTarget()
    await target.start()
    relay.target_port = target.port
    await relay.start("127.0.0.1")
    return target


def secure_relay(ids, hub_ip: str = "127.0.0.1") -> HelperRelay:
    tls = ids["helper"].context(server=True, peer_cert_pem=ids["hub"].cert_pem)
    return HelperRelay(0, hub_ip, tls, ids["hub"].fingerprint)


async def until(predicate, timeout: float = 5.0) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return predicate()


async def roundtrip(client: TunnelClient, data: bytes = b"ping") -> bytes:
    reader, writer = await client.open()
    try:
        writer.write(data)
        await writer.drain()
        return await asyncio.wait_for(reader.readexactly(len(data)), 3)
    finally:
        writer.close()


async def test_pinned_tls_round_trip(ids):
    relay = secure_relay(ids)
    target = await echo_through(relay)
    try:
        client = TunnelClient("127.0.0.1", relay.port, ids["hub"], ids["helper"].cert_pem)
        assert await roundtrip(client, b"x" * 100_000) == b"x" * 100_000
        reader, writer = await client.open()
        assert writer.get_extra_info("ssl_object").version() == "TLSv1.3"
        writer.close()
    finally:
        relay.close()
        target.close()


async def test_helper_refuses_a_certificate_the_hub_did_not_send(ids):
    relay = secure_relay(ids)
    target = await echo_through(relay)
    try:
        impostor = TunnelClient("127.0.0.1", relay.port, ids["evil"], ids["helper"].cert_pem)
        with pytest.raises((ssl.SSLError, OSError, asyncio.IncompleteReadError, TimeoutError)):
            await roundtrip(impostor)
    finally:
        relay.close()
        target.close()


async def test_hub_refuses_a_helper_with_the_wrong_certificate(ids):
    relay = secure_relay(ids)
    target = await echo_through(relay)
    try:
        expecting_someone_else = TunnelClient("127.0.0.1", relay.port, ids["hub"], ids["evil"].cert_pem)
        with pytest.raises((ssl.SSLError, TunnelError)):
            await expecting_someone_else.open()
    finally:
        relay.close()
        target.close()


async def test_tls_12_is_refused(ids):
    relay = secure_relay(ids)
    target = await echo_through(relay)
    try:
        ctx = ids["hub"].context(server=False, peer_cert_pem=ids["helper"].cert_pem)
        ctx.minimum_version = ctx.maximum_version = ssl.TLSVersion.TLSv1_2
        with pytest.raises((ssl.SSLError, OSError)):
            r, w = await asyncio.open_connection("127.0.0.1", relay.port, ssl=ctx, server_hostname="")
            w.close()
    finally:
        relay.close()
        target.close()


async def test_relay_drops_other_addresses_before_any_tls(ids):
    relay = secure_relay(ids, hub_ip="10.1.2.3")
    target = await echo_through(relay)
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", relay.port)
        assert await asyncio.wait_for(reader.read(10), 3) == b""  # closed, no ServerHello
        writer.close()
        assert relay.refused == 1
    finally:
        relay.close()
        target.close()


async def test_closing_the_relay_cuts_live_connections(ids):
    relay = secure_relay(ids)
    target = await echo_through(relay)
    client = TunnelClient("127.0.0.1", relay.port, ids["hub"], ids["helper"].cert_pem)
    reader, writer = await client.open()
    writer.write(b"a")
    assert await reader.readexactly(1) == b"a"
    relay.close()
    assert await asyncio.wait_for(reader.read(10), 3) == b""
    target.close()


async def test_hub_relay_gives_llama_server_a_plain_local_port(ids):
    relay = secure_relay(ids)
    target = await echo_through(relay)
    hub = HubRelay(TunnelClient("127.0.0.1", relay.port, ids["hub"], ids["helper"].cert_pem))
    await hub.start()
    try:
        # What llama-server does: plain TCP to 127.0.0.1, several short connections.
        for _ in range(3):
            r, w = await asyncio.open_connection("127.0.0.1", hub.port)
            w.write(b"hello")
            assert await asyncio.wait_for(r.readexactly(5), 3) == b"hello"
            w.close()
        # The far side closing ends the local connection too (TLS can't half-close; nothing may hang).
        r, w = await asyncio.open_connection("127.0.0.1", hub.port)
        w.write(b"x")
        assert await r.readexactly(1) == b"x"
        relay.close()
        assert await asyncio.wait_for(r.read(10), 3) == b""
        w.close()
    finally:
        hub.close()
        target.close()


async def _refused(reader: asyncio.StreamReader) -> bool:
    try:
        return await asyncio.wait_for(reader.read(10), 3) == b""
    except (ConnectionResetError, ssl.SSLError):
        return True


async def test_helper_relay_counts_each_hub_connection_once(ids, monkeypatch):
    from ollamapower.agent import tunnel as agent_tunnel

    monkeypatch.setattr(agent_tunnel, "MAX_CONNECTIONS", 3)
    relay = secure_relay(ids)
    target = await echo_through(relay)
    client = TunnelClient("127.0.0.1", relay.port, ids["hub"], ids["helper"].cert_pem)
    try:
        # As many live relayed connections as the limit says (each is two writers inside).
        opened = []
        for i in range(3):
            r, w = await client.open()
            w.write(b"%d" % i)
            assert await asyncio.wait_for(r.readexactly(1), 3) == b"%d" % i
            opened.append(w)
        assert relay.active == 3
        reader, writer = await asyncio.open_connection("127.0.0.1", relay.port)
        assert await _refused(reader)
        writer.close()
        assert relay.refused == 1
        for w in opened:
            w.close()
        assert await until(lambda: relay.active == 0)
        # Unfinished TLS handshakes hold slots too, and give them back when they fail.
        stalled = [await asyncio.open_connection("127.0.0.1", relay.port) for _ in range(3)]
        await until(lambda: relay.active == 3)
        reader, writer = await asyncio.open_connection("127.0.0.1", relay.port)
        assert await _refused(reader)
        writer.close()
        for _, w in stalled:
            w.close()
        assert await until(lambda: relay.active == 0)
        assert await roundtrip(client) == b"ping"
        # Closing the relay mid-handshake releases everything too.
        stalled = [await asyncio.open_connection("127.0.0.1", relay.port) for _ in range(2)]
        await until(lambda: relay.active == 2)
        relay.close()
        assert await until(lambda: relay.active == 0 and not relay.connections)
        for _, w in stalled:
            w.close()
    finally:
        relay.close()
        target.close()


async def test_hub_relay_counts_each_local_connection_once(ids, monkeypatch):
    from ollamapower.hub import tunnel as hub_tunnel

    monkeypatch.setattr(hub_tunnel, "MAX_LOCAL_CONNECTIONS", 3)
    relay = secure_relay(ids)
    target = await echo_through(relay)
    hub = HubRelay(TunnelClient("127.0.0.1", relay.port, ids["hub"], ids["helper"].cert_pem))
    await hub.start()
    try:
        opened = []
        for i in range(3):  # used to stop at 2: each one put two writers in the count
            r, w = await asyncio.open_connection("127.0.0.1", hub.port)
            w.write(b"%d" % i)
            assert await asyncio.wait_for(r.readexactly(1), 3) == b"%d" % i
            opened.append(w)
        assert hub.active == 3
        r, w = await asyncio.open_connection("127.0.0.1", hub.port)
        assert await _refused(r)
        w.close()
        assert hub.refused == 1
        for w in opened:
            w.close()
        assert await until(lambda: hub.active == 0 and not hub.connections)
    finally:
        hub.close()
        relay.close()
        target.close()


async def test_closing_the_hub_relay_releases_connections_still_opening(ids):
    async def never_answer(reader, writer):  # a "helper" that accepts and never does TLS
        await reader.read()

    silent = await asyncio.start_server(never_answer, "127.0.0.1", 0)
    port = silent.sockets[0].getsockname()[1]
    hub = HubRelay(TunnelClient("127.0.0.1", port, ids["hub"], ids["helper"].cert_pem))
    await hub.start()
    try:
        r, w = await asyncio.open_connection("127.0.0.1", hub.port)
        assert await until(lambda: hub.active == 1)
        hub.close()
        assert await until(lambda: hub.active == 0 and not hub.connections)
        assert await _refused(r)
        w.close()
    finally:
        hub.close()
        silent.close()


async def test_plain_splice_passes_half_close():
    async def server(reader, writer):
        data = await reader.read()  # until EOF
        writer.write(data.upper())
        await writer.drain()
        writer.close()

    far = await asyncio.start_server(server, "127.0.0.1", 0)
    far_port = far.sockets[0].getsockname()[1]

    async def relay(reader, writer):
        await splice((reader, writer), await asyncio.open_connection("127.0.0.1", far_port))

    mid = await asyncio.start_server(relay, "127.0.0.1", 0)
    r, w = await asyncio.open_connection("127.0.0.1", mid.sockets[0].getsockname()[1])
    w.write(b"abc")
    w.write_eof()
    assert await asyncio.wait_for(r.read(), 3) == b"ABC"
    w.close()
    mid.close()
    far.close()


async def test_measure_reports_rtt_and_throughput(ids):
    relay = secure_relay(ids)
    target = await echo_through(relay)
    try:
        client = TunnelClient("127.0.0.1", relay.port, ids["hub"], ids["helper"].cert_pem)
        r = await measure(client, bulk_bytes=2 * 1024 * 1024)
        assert r["mode"] == "secure" and r["rtt_ms"] > 0 and r["mbps"] > 0 and r["bytes"] == 2 * 1024 * 1024
    finally:
        relay.close()
        target.close()


def test_certificates_from_the_network_are_checked(ids):
    assert check_cert_pem(ids["hub"].cert_pem) == ids["hub"].cert_pem
    for bad in (None, "", "x" * 5000, "-----BEGIN CERTIFICATE-----\nnope\n-----END CERTIFICATE-----\n", 7):
        with pytest.raises(TunnelError):
            check_cert_pem(bad)


def test_identities_are_fresh_and_keys_stay_private(ids):
    again = TunnelIdentity.generate("hub")
    assert again.fingerprint != ids["hub"].fingerprint
    assert "PRIVATE" not in again.cert_pem and "key_pem" not in repr(again)


@pytest.mark.parametrize(
    "a,b,same",
    [("127.0.0.1", "127.0.0.1", True), ("::ffff:10.0.0.5", "10.0.0.5", True), ("10.0.0.5", "10.0.0.6", False)],
)
def test_same_ip(a, b, same):
    assert same_ip(a, b) is same


def test_loopback_peer_uid_reads_the_client_row(tmp_path):
    from ollamapower.core.tunnel import loopback_peer_uid

    table = tmp_path / "tcp"
    table.write_text(
        "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n"
        "   0: 0100007F:1F90 0100007F:D431 01 00000000:00000000 00:00000000 00000000  1000        0 1 1\n"
        "   1: 0100007F:D431 0100007F:1F90 01 00000000:00000000 00:00000000 00000000  1234        0 2 1\n"
    )
    # client port 0xD431 connected to our port 0x1F90: the second row is the client's socket
    assert loopback_peer_uid(0xD431, 0x1F90, (str(table),)) == 1234
    assert loopback_peer_uid(1, 2, (str(table),)) is None


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the owner check reads /proc/net/tcp")
async def test_hub_relay_refuses_other_local_users(ids, monkeypatch):
    from ollamapower.core import tunnel as core_tunnel

    relay = secure_relay(ids)
    target = await echo_through(relay)
    hub = HubRelay(TunnelClient("127.0.0.1", relay.port, ids["hub"], ids["helper"].cert_pem))
    await hub.start()
    try:
        r, w = await asyncio.open_connection("127.0.0.1", hub.port)  # us: allowed
        w.write(b"a")
        assert await asyncio.wait_for(r.readexactly(1), 3) == b"a"
        w.close()
        monkeypatch.setattr(core_tunnel.os, "getuid", lambda: 4242)  # now the connecting user looks different
        r, w = await asyncio.open_connection("127.0.0.1", hub.port)
        w.write(b"a")
        try:
            assert await asyncio.wait_for(r.read(1), 3) == b""
        except ConnectionResetError:
            pass  # closed with unread data: also a refusal
        assert hub.refused == 1
        w.close()
    finally:
        hub.close()
        relay.close()
        target.close()


def test_unreadable_proc_is_unknown_not_a_refusal(tmp_path):
    from ollamapower.core.tunnel import UNKNOWN, loopback_peer_uid

    assert loopback_peer_uid(1, 2, (str(tmp_path / "missing"),)) == UNKNOWN
