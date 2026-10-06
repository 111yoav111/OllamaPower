"""Get a working, verified llama.cpp for this machine (install it if needed)."""

from __future__ import annotations

from pathlib import Path

from .binaries import BinaryError, LlamaCpp
from .install import InstallError, Pins, Progress, install, install_dir, is_installed
from .variants import VariantError, archives_for, current_platform, resolve_backend


class LlamaCppUnavailable(Exception):
    """llama.cpp can't be used here; the message says why and how to fix it.

    ``permanent``: trying again won't help until something changes (no build for this
    platform, programs missing or the wrong build), unlike a failed download.
    """

    def __init__(self, message: str, permanent: bool = False) -> None:
        super().__init__(message)
        self.permanent = permanent


class LlamaCppNotInstalled(LlamaCppUnavailable):
    """``download=False`` and the pinned build is not installed (nothing is broken)."""


def gpu_vendors() -> list[str]:
    """The GPU vendors this machine has, for ``backend = "auto"``."""
    from ..telemetry import Telemetry

    return [g.get("vendor", "") for g in Telemetry().static().get("gpus", []) if g.get("vendor")]


def ensure(
    base: Path,
    backend: str = "auto",
    custom_path: str = "",
    gpu_vendors: list[str] | None = None,
    progress: Progress | None = None,
    pins: Pins | None = None,
    download: bool = True,
) -> LlamaCpp:
    """The pinned build: from ``custom_path`` if set, else downloaded and verified into ``base``.

    With ``download=False`` only an existing install is used.
    """
    pins = pins or Pins.load()
    try:
        if custom_path:
            root = Path(custom_path).expanduser()
        else:
            system, arch = current_platform()
            concrete = resolve_backend(backend, system, arch, gpu_vendors or [])
            variant = archives_for(pins.tag, system, arch, concrete)
            if not download and not is_installed(install_dir(base, pins.tag, concrete), variant, pins):
                raise LlamaCppNotInstalled(
                    f"llama.cpp {pins.tag} ({concrete}) is not installed; run `ollamapower llamacpp install`"
                )
            root = install(variant, base, pins, progress)
        binaries = LlamaCpp.find(root).probe()
        binaries.check(pins.build)
    except (VariantError, BinaryError) as exc:
        raise LlamaCppUnavailable(str(exc), permanent=True) from None
    except InstallError as exc:  # download, verification or unpacking: may work next time
        raise LlamaCppUnavailable(str(exc)) from None
    except OSError as exc:  # e.g. the config dir isn't writable: a message, not a traceback
        raise LlamaCppUnavailable(f"llama.cpp setup failed: {exc}") from None
    return binaries
