"""The installed llama.cpp programs: where they are, which build, which devices.

Names changed across releases (``rpc-server`` became ``ggml-rpc-server``), so
both are looked up. The build number must equal the pinned one on every
machine; ``check()`` says what is wrong in owner-readable words.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

EXE = ".exe" if sys.platform == "win32" else ""
RPC_NAMES = ("ggml-rpc-server", "rpc-server")
SERVER_NAMES = ("llama-server",)
VERSION_RE = re.compile(r"build (\d+)")
# "  Vulkan0: NVIDIA GeForce RTX 4090 (24564 MiB, 23012 MiB free)"
DEVICE_RE = re.compile(r"^\s+(\S+): (.*) \((\d+) MiB, (\d+) MiB free\)\s*$")
TIMEOUT = 30.0


class BinaryError(Exception):
    """The llama.cpp programs are missing, broken or the wrong build."""


@dataclass
class Device:
    name: str  # what --device / -dev takes, e.g. "CUDA0", "Vulkan1"
    description: str
    total_mib: int
    free_mib: int


@dataclass
class LlamaCpp:
    root: Path
    rpc_server: Path | None = None
    llama_server: Path | None = None
    build: int | None = None
    devices: list[Device] = field(default_factory=list)

    @classmethod
    def find(cls, root: Path) -> LlamaCpp:
        def first(names: tuple[str, ...]) -> Path | None:
            for name in names:
                for candidate in (root / (name + EXE), root / "bin" / (name + EXE)):
                    if candidate.is_file():
                        return candidate
            return None

        return cls(root=root, rpc_server=first(RPC_NAMES), llama_server=first(SERVER_NAMES))

    def _run(self, *args: str) -> str:
        assert self.llama_server is not None
        try:
            result = subprocess.run(
                [str(self.llama_server), *args],
                capture_output=True,
                text=True,
                timeout=TIMEOUT,
                cwd=self.llama_server.parent,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise BinaryError(f"cannot run {self.llama_server.name}: {exc}") from None
        return result.stdout + result.stderr

    def probe(self) -> LlamaCpp:
        """Read the build number and the GPU devices llama.cpp can see."""
        if self.llama_server is None or self.rpc_server is None:
            missing = [
                n for n, p in (("llama-server", self.llama_server), ("ggml-rpc-server", self.rpc_server)) if p is None
            ]
            raise BinaryError(f"{', '.join(missing)} not found in {self.root}")
        match = VERSION_RE.search(self._run("--version"))
        if match is None:
            raise BinaryError(f"{self.llama_server} did not report a build number")
        self.build = int(match.group(1))
        self.devices = [
            Device(m.group(1), m.group(2), int(m.group(3)), int(m.group(4)))
            for line in self._run("--list-devices").splitlines()
            if (m := DEVICE_RE.match(line))
        ]
        return self

    def check(self, build: int) -> None:
        if self.build != build:
            raise BinaryError(
                f"llama.cpp build {self.build} at {self.root}, but this OllamaPower needs build {build} "
                "(every machine must run the same build)"
            )

    def summary(self) -> dict[str, object]:
        return {
            "build": self.build,
            "devices": [d.__dict__ for d in self.devices],
            "rpc_server": self.rpc_server is not None,
        }
