"""The installed llama.cpp programs: where they are, which build, which devices.

Names changed across releases (``rpc-server`` became ``ggml-rpc-server``), so
both are looked up. The build number must equal the pinned one on every
machine; ``check()`` says what is wrong in owner-readable words.

Device numbering: the owner's GPU allow-list (``lending.gpus``) uses the
indexes the agent's own views show (NVML's order for NVIDIA, the DRM cards
for AMD), not llama.cpp's device list, which can differ (CUDA orders the
fastest GPU first, Vulkan also lists integrated GPUs). Every llama.cpp program
runs with ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` so CUDA follows NVML's order, and
``telemetry_indexes`` matches the rest by name and memory size.

Integrated GPUs share the system's RAM: llama.cpp reports that RAM (or a
large part of it) as their memory. They are told apart (``integrated``) and
used only where no dedicated GPU is, never counted as dedicated VRAM.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

EXE = ".exe" if sys.platform == "win32" else ""
RPC_NAMES = ("ggml-rpc-server", "rpc-server")
SERVER_NAMES = ("llama-server",)
VERSION_RE = re.compile(r"build (\d+)")
# "  Vulkan0: NVIDIA GeForce RTX 4090 (24564 MiB, 23012 MiB free)"
DEVICE_RE = re.compile(r"^\s+(\S+): (.*) \((\d+) MiB, (\d+) MiB free\)\s*$")
TIMEOUT = 30.0
MIB = 1024 * 1024
# Environment for every llama.cpp program: CUDA numbers devices like NVML (and the owner's views) do.
ENV = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
# Shared-memory GPUs, from the device description llama.cpp prints (Vulkan, SYCL, CUDA on Tegra/GB10, Metal).
_INTEGRATED = re.compile(
    r"^Apple\b"
    r"|\bIntel\b.*\b(UHD|HD|Iris|Xe)\b"
    r"|\bIntel(\(R\))? (Arc(\(TM\))? )?Graphics\b"
    r"|\bAMD Radeon(\(TM\))? Graphics\b"
    r"|\bRadeon(\(TM\))? (Vega \d+ )?Graphics\b"
    r"|\bRadeon(\(TM\))? \d{3,4}[MS]\b"
    r"|\b(RENOIR|REMBRANDT|PHOENIX|CEZANNE|LUCIENNE|VANGOGH|RAVEN|PICASSO|BARCELO|STRIX|KRACKAN|GFX(90C|103[5-7]|1103|115\d))\b"
    r"|\b(Tegra|Orin|Xavier|Thor|GB10)\b",
    re.IGNORECASE,
)


def llamacpp_env() -> dict[str, str]:
    """The environment llama.cpp programs run with (device order consistent with the probe)."""
    return {**os.environ, **ENV}


class BinaryError(Exception):
    """The llama.cpp programs are missing, broken or the wrong build."""


@dataclass
class Device:
    name: str  # what --device / -dev takes, e.g. "CUDA0", "Vulkan1"
    description: str
    total_mib: int
    free_mib: int

    @property
    def integrated(self) -> bool:
        """A GPU that shares the system's RAM (its "memory" is not dedicated VRAM)."""
        return bool(_INTEGRATED.search(self.description))


def preferred(devices: list[Device]) -> list[Device]:
    """The devices worth using: the dedicated GPUs, or the integrated ones if there is no dedicated GPU."""
    dedicated = [d for d in devices if not d.integrated]
    return dedicated or [d for d in devices if d.integrated]


def _words(text: str) -> str:
    return " ".join(re.sub(r"\((r|tm)\)", " ", text.lower()).replace("-", " ").split())


def telemetry_indexes(devices: list[Device], gpus: list[dict[str, Any]]) -> list[int | None]:
    """The owner-facing GPU index (telemetry's ``index``) of each llama.cpp device, None if it has none.

    Matched by name and memory size, in order. Without telemetry GPUs (no NVML,
    no sysfs) the dedicated devices are numbered in llama.cpp's order.
    """
    known = [g for g in gpus if isinstance(g, dict) and isinstance(g.get("index"), int)]
    out: list[int | None] = [None] * len(devices)
    if not known:
        dedicated = [i for i, d in enumerate(devices) if not d.integrated]
        for n, i in enumerate(dedicated):
            out[i] = n
        return out
    for i, d in enumerate(devices):
        best, best_score = None, 0
        for g in known:
            name = _words(str(g.get("name", "")))
            desc = _words(d.description)
            named = bool(name) and (name == desc or name in desc or desc in name)
            total = int(g.get("vram_total") or 0) // MIB
            sized = total > 0 and abs(total - d.total_mib) <= max(512, d.total_mib // 10)
            if d.integrated and not g.get("unified") and not named:
                continue  # telemetry doesn't list integrated GPUs (Apple's unified GPU aside)
            score = 2 * named + sized
            if score > best_score:
                best, best_score = g, score
        if best is not None:
            out[i] = best["index"]
            known.remove(best)
    return out


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
                env=llamacpp_env(),
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
            "devices": [{**d.__dict__, "integrated": d.integrated} for d in self.devices],
            "rpc_server": self.rpc_server is not None,
        }
