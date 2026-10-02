"""Get a working, verified llama.cpp for this machine (install it if needed)."""

from __future__ import annotations

from pathlib import Path

from .binaries import BinaryError, LlamaCpp
from .install import InstallError, Pins, Progress, install
from .variants import VariantError, archives_for, current_platform, resolve_backend


class LlamaCppUnavailable(Exception):
    """llama.cpp can't be used here; the message says why and how to fix it."""


def ensure(
    base: Path,
    backend: str = "auto",
    custom_path: str = "",
    gpu_vendors: list[str] | None = None,
    progress: Progress | None = None,
    pins: Pins | None = None,
) -> LlamaCpp:
    """The pinned build: from ``custom_path`` if set, else downloaded and verified into ``base``."""
    pins = pins or Pins.load()
    try:
        if custom_path:
            root = Path(custom_path).expanduser()
        else:
            system, arch = current_platform()
            concrete = resolve_backend(backend, system, arch, gpu_vendors or [])
            root = install(archives_for(pins.tag, system, arch, concrete), base, pins, progress)
        binaries = LlamaCpp.find(root).probe()
        binaries.check(pins.build)
    except (InstallError, VariantError, BinaryError) as exc:
        raise LlamaCppUnavailable(str(exc)) from None
    return binaries
