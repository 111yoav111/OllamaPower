"""Download, verify and unpack the pinned llama.cpp release.

* The release tag and the SHA-256 and size of every archive we may use are
  stored in ``pins.json`` inside this package (computed when the pin was made;
  llama.cpp publishes no checksums). A download that doesn't match is deleted
  and never unpacked or run.
* Downloads resume (``Range``) after an interruption and report progress.
* Unpacking refuses absolute paths, ``..`` and links that point outside the
  install directory.
* Installs live in ``<config dir>/llamacpp/<tag>/<backend>/`` and are marked
  complete only after everything was verified and unpacked.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import stat
import tarfile
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from importlib import resources
from pathlib import Path, PurePosixPath
from typing import Any

import httpx

from ..files import private_dir
from .variants import Variant

log = logging.getLogger("ollamapower.llamacpp")

RELEASE_URL = "https://github.com/ggml-org/llama.cpp/releases/download/{tag}/{name}"
MARKER = ".ollamapower-installed.json"
CHUNK = 1024 * 1024

Progress = Callable[[str, int, int], None]  # (archive, bytes done, bytes total)


class InstallError(Exception):
    """Download, verification or unpacking failed."""


@dataclass(frozen=True)
class Pins:
    tag: str
    build: int
    archives: dict[str, dict[str, Any]]  # name -> {"size": int, "sha256": str}

    @classmethod
    def load(cls) -> Pins:
        data = json.loads(resources.files("ollamapower.llamacpp").joinpath("pins.json").read_text())
        return cls(tag=data["tag"], build=int(data["build"]), archives=data["archives"])

    def expect(self, name: str) -> dict[str, Any]:
        if name not in self.archives:
            raise InstallError(f"{name} is not in the pinned release {self.tag}")
        return self.archives[name]


def install_dir(base: Path, tag: str, backend: str) -> Path:
    return base / "llamacpp" / tag / backend


def is_installed(target: Path, variant: Variant, pins: Pins) -> bool:
    try:
        marker = json.loads((target / MARKER).read_text())
    except (OSError, ValueError):
        return False
    return marker.get("tag") == pins.tag and marker.get("archives") == {
        name: pins.expect(name)["sha256"] for name in variant.archives
    }


# ----- download --------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def download(
    name: str, dest_dir: Path, pins: Pins, progress: Progress | None = None, client: httpx.Client | None = None
) -> Path:
    """Fetch one archive into ``dest_dir`` and verify it; resumes a partial download."""
    expected = pins.expect(name)
    size, digest = int(expected["size"]), str(expected["sha256"])
    final = dest_dir / name
    if final.exists() and final.stat().st_size == size and _sha256(final) == digest:
        return final
    part = dest_dir / (name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if have > size:
        part.unlink()
        have = 0
    url = RELEASE_URL.format(tag=pins.tag, name=name)
    own = client is None
    client = client or httpx.Client(follow_redirects=True, timeout=httpx.Timeout(30.0, read=120.0))
    try:
        headers = {"Range": f"bytes={have}-"} if have else {}
        with client.stream("GET", url, headers=headers) as r:
            if r.status_code == 200 and have:
                have = 0  # the server ignored Range: start over
            elif r.status_code not in (200, 206):
                raise InstallError(f"downloading {name} failed: HTTP {r.status_code}")
            with open(part, "ab" if have else "wb") as fh:
                done = have
                for chunk in r.iter_bytes(CHUNK):
                    done += len(chunk)
                    if done > size:
                        raise InstallError(f"{name} is larger than the pinned size; refusing it")
                    fh.write(chunk)
                    if progress is not None:
                        progress(name, done, size)
    except httpx.HTTPError as exc:
        raise InstallError(f"downloading {name} failed: {type(exc).__name__} (run again to resume)") from None
    finally:
        if own:
            client.close()
    if part.stat().st_size != size:
        raise InstallError(f"{name} is incomplete ({part.stat().st_size} of {size} bytes); run again to resume")
    actual = _sha256(part)
    if actual != digest:
        part.unlink()
        raise InstallError(f"{name} does not match its pinned SHA-256 (got {actual[:16]}…); deleted, not used")
    os.replace(part, final)
    return final


# ----- unpack ----------------------------------------------------------------------------


def _strip_top(name: str) -> PurePosixPath | None:
    """Archive member path without its top-level folder, or None to skip it."""
    parts = PurePosixPath(name.replace("\\", "/")).parts
    if not parts:
        return None
    if parts[0].startswith(("llama-", "cudart-llama-")) and len(parts) > 1:
        parts = parts[1:]
    return PurePosixPath(*parts)


def _safe_target(root: Path, rel: PurePosixPath) -> Path:
    if rel.is_absolute() or ".." in rel.parts or any(":" in p for p in rel.parts):
        raise InstallError(f"unsafe path in archive: {rel}")
    target = (root / Path(*rel.parts)).resolve()
    if not target.is_relative_to(root.resolve()):
        raise InstallError(f"unsafe path in archive: {rel}")
    return target


def unpack(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                rel = _strip_top(info.filename)
                if rel is None or info.is_dir():
                    continue
                target = _safe_target(dest, rel)
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out)
        return
    with tarfile.open(archive, "r:gz") as tf:
        links = []
        for member in tf.getmembers():
            rel = _strip_top(member.name)
            if rel is None or str(rel) == "." or member.isdir():
                continue
            target = _safe_target(dest, rel)
            if member.issym():
                link_target = (target.parent / member.linkname).resolve()
                if os.path.isabs(member.linkname) or not link_target.is_relative_to(dest.resolve()):
                    raise InstallError(f"unsafe link in archive: {rel} -> {member.linkname}")
                links.append((target, member.linkname))
                continue
            if not member.isfile():
                continue  # no devices, fifos or hard links
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tf.extractfile(member)
            assert src is not None
            with src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            if member.mode & 0o111:
                target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        for target, linkname in links:
            if target.exists() or target.is_symlink():
                target.unlink()
            target.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(linkname, target)


def install(
    variant: Variant, base: Path, pins: Pins, progress: Progress | None = None, client: httpx.Client | None = None
) -> Path:
    """Make sure ``variant`` of the pinned release is installed; return its directory."""
    target = install_dir(base, pins.tag, variant.backend)
    if is_installed(target, variant, pins):
        return target
    downloads = base / "llamacpp" / "downloads"
    private_dir(downloads)
    archives = [download(name, downloads, pins, progress, client) for name in variant.archives]
    staging = target.with_name(target.name + ".staging")
    shutil.rmtree(staging, ignore_errors=True)
    for archive in archives:
        unpack(archive, staging)
    marker = {"tag": pins.tag, "archives": {name: pins.expect(name)["sha256"] for name in variant.archives}}
    (staging / MARKER).write_text(json.dumps(marker))
    shutil.rmtree(target, ignore_errors=True)
    os.replace(staging, target)
    for archive in archives:
        archive.unlink(missing_ok=True)  # unpacked and verified: the archive is no longer needed
    log.info("installed llama.cpp %s (%s) in %s", pins.tag, variant.backend, target)
    return target
