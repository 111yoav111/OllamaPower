"""``known_hub``: the hub keys this agent trusts, like SSH's ``known_hosts``.

The control-plane handshake starts with the hub's RSA key. On first contact
the agent shows its fingerprint and the owner confirms it against what the hub
prints (trust on first use); from then on a *different* key for that hub is
refused loudly - it means someone is impersonating the hub, or the hub's key
was replaced (then: ``ollamapower agent forget-hub``).

One line per hub: ``<host>:<port> <sha256 hex>``.
"""

from __future__ import annotations

from pathlib import Path

from ..files import write_private


def hub_entry(host: str, port: int) -> str:
    """The ``known_hub`` key for a hub. IPv6 hosts are stored unbracketed (``fd00::1:9000``), as always."""
    return f"{host}:{port}"


def normalize_fingerprint(value: str) -> str:
    """Accept ``AB:CD:...``, spaces or plain hex; return lower-case hex."""
    return "".join(ch for ch in value.lower() if ch in "0123456789abcdef")


def format_fingerprint(fp_hex: str) -> str:
    """OllamaGate's format: colon-separated upper-case pairs."""
    return ":".join(fp_hex[i : i + 2] for i in range(0, len(fp_hex), 2)).upper()


def fingerprint_lines(fp_hex: str) -> list[str]:
    """The whole fingerprint in two halves of 16 pairs, for narrow views (a TUI row).

    Never shorten a fingerprint the owner compares: a prefix is easy to forge.
    """
    pairs = [p for p in format_fingerprint(fp_hex).split(":") if p]
    half = (len(pairs) + 1) // 2
    return [":".join(part) for part in (pairs[:half], pairs[half:]) if part]


class KnownHubs:
    def __init__(self, path: Path) -> None:
        self.path = path

    def _read(self) -> dict[str, str]:
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        entries = {}
        for line in text.splitlines():
            parts = line.split()
            if len(parts) == 2 and not line.startswith("#"):
                entries[parts[0]] = normalize_fingerprint(parts[1])
        return entries

    def _write(self, entries: dict[str, str]) -> None:
        body = "".join(f"{addr} {fp}\n" for addr, fp in sorted(entries.items()))
        write_private(self.path, b"# hub address -> trusted key fingerprint (SHA-256)\n" + body.encode())

    def get(self, address: str) -> str | None:
        return self._read().get(address)

    def entries(self) -> list[str]:
        """The pinned hub addresses."""
        return list(self._read())

    def pin(self, address: str, fingerprint: str) -> None:
        entries = self._read()
        entries[address] = normalize_fingerprint(fingerprint)
        self._write(entries)

    def forget(self, address: str | None = None) -> int:
        """Forget one hub, or all of them. Returns how many entries were removed."""
        entries = self._read()
        if address is None:
            removed = len(entries)
            entries = {}
        else:
            removed = 1 if entries.pop(address, None) else 0
        self._write(entries)
        return removed
