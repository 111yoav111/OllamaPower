"""Which official llama.cpp archive fits this machine.

Backends, smallest download first (sizes for the pinned build, measured):

* ``cpu``    - 11-20 MB. No GPU; also what macOS Intel gets.
* ``metal``  - 12 MB. Apple Silicon (the macOS arm64 build includes Metal).
* ``vulkan`` - 25-33 MB. Any GPU with a Vulkan driver (NVIDIA, AMD, Intel).
  The default for GPUs on Windows and Linux: small and vendor-neutral.
* ``cuda12`` / ``cuda13`` - NVIDIA only, usually faster than Vulkan, but
  580-770 MB with the CUDA runtime libraries. Opt in per machine.
* ``rocm``   - AMD, ~250 MB; needs the ROCm runtime installed on Linux.

Machines in one cluster may use different backends: the RPC protocol only
cares that the *build* matches.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from dataclasses import dataclass

BACKENDS = ("auto", "cpu", "metal", "vulkan", "cuda12", "cuda13", "rocm")


class VariantError(ValueError):
    """No official build exists for this combination."""


@dataclass(frozen=True)
class Variant:
    backend: str
    archives: tuple[str, ...]  # the build first, then extra runtime archives (CUDA libraries)

    @property
    def name(self) -> str:
        return self.archives[0]


def _rosetta() -> bool:
    """Is this (x86_64) Python running under Rosetta 2 on Apple Silicon?"""
    try:
        out = subprocess.run(
            ["sysctl", "-n", "sysctl.proc_translated"], capture_output=True, text=True, timeout=2, check=False
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return out.strip() == "1"


def native_machine() -> str:
    """The CPU architecture of the machine, not of this Python.

    An Intel Python on an Apple Silicon Mac (an old Homebrew or pyenv install)
    runs under Rosetta and reports ``x86_64``; taking that at face value would
    install the Intel CPU build and lend no GPU at all, though the native arm64
    build with Metal runs fine as a separate program.
    """
    machine = platform.machine().lower()
    if sys.platform == "darwin" and machine == "x86_64" and _rosetta():
        return "arm64"
    return machine


def current_platform() -> tuple[str, str]:
    """(``linux``|``windows``|``macos``, ``x64``|``arm64``)."""
    system = {"linux": "linux", "win32": "windows", "darwin": "macos"}.get(sys.platform, sys.platform)
    machine = native_machine()
    arch = "arm64" if machine in ("arm64", "aarch64") else "x64" if machine in ("x86_64", "amd64") else machine
    return system, arch


def available(system: str, arch: str, backend: str) -> bool:
    """Is there an official build of ``backend`` for this OS and architecture?"""
    return (system, arch, backend) in _table("")


def resolve_backend(backend: str, system: str, arch: str, gpu_vendors: list[str]) -> str:
    """Turn ``auto`` into a concrete backend for this machine."""
    if backend != "auto":
        return backend
    if system == "macos":
        return "metal" if arch == "arm64" else "cpu"
    # Vulkan for any GPU, where llama.cpp publishes it (not for Windows on ARM).
    return "vulkan" if gpu_vendors and available(system, arch, "vulkan") else "cpu"


def _table(t: str) -> dict[tuple[str, str, str], tuple[str, ...]]:
    """(system, arch, backend) -> the archives of release ``t``."""
    return {
        ("macos", "arm64", "metal"): (f"llama-{t}-bin-macos-arm64.tar.gz",),
        ("macos", "arm64", "cpu"): (f"llama-{t}-bin-macos-arm64.tar.gz",),
        ("macos", "x64", "cpu"): (f"llama-{t}-bin-macos-x64.tar.gz",),
        ("linux", "x64", "cpu"): (f"llama-{t}-bin-ubuntu-x64.tar.gz",),
        ("linux", "arm64", "cpu"): (f"llama-{t}-bin-ubuntu-arm64.tar.gz",),
        ("linux", "x64", "vulkan"): (f"llama-{t}-bin-ubuntu-vulkan-x64.tar.gz",),
        ("linux", "arm64", "vulkan"): (f"llama-{t}-bin-ubuntu-vulkan-arm64.tar.gz",),
        ("linux", "x64", "cuda12"): (
            f"llama-{t}-bin-ubuntu-cuda-12.8-x64.tar.gz",
            f"cudart-llama-{t}-bin-ubuntu-cuda-12.8-x64.tar.gz",
        ),
        ("linux", "x64", "cuda13"): (
            f"llama-{t}-bin-ubuntu-cuda-13.4-x64.tar.gz",
            f"cudart-llama-{t}-bin-ubuntu-cuda-13.4-x64.tar.gz",
        ),
        ("linux", "x64", "rocm"): (f"llama-{t}-bin-ubuntu-rocm-10.0-x64.tar.gz",),
        ("windows", "x64", "cpu"): (f"llama-{t}-bin-win-cpu-x64.zip",),
        ("windows", "arm64", "cpu"): (f"llama-{t}-bin-win-cpu-arm64.zip",),
        ("windows", "x64", "vulkan"): (f"llama-{t}-bin-win-vulkan-x64.zip",),
        ("windows", "x64", "cuda12"): (
            f"llama-{t}-bin-win-cuda-12.4-x64.zip",
            "cudart-llama-bin-win-cuda-12.4-x64.zip",
        ),
        ("windows", "x64", "cuda13"): (
            f"llama-{t}-bin-win-cuda-13.4-x64.zip",
            "cudart-llama-bin-win-cuda-13.4-x64.zip",
        ),
        ("windows", "x64", "rocm"): (f"llama-{t}-bin-win-rocm-10.0-x64.zip",),
    }


def archives_for(tag: str, system: str, arch: str, backend: str) -> Variant:
    archives = _table(tag).get((system, arch, backend))
    if archives is None:
        raise VariantError(
            f"no official llama.cpp {backend} build for {system} {arch}; choose another backend "
            "(llamacpp.backend) or point llamacpp.path at your own build"
        )
    return Variant(backend, archives)
