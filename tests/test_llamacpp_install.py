"""Pinned llama.cpp downloads: verification, resume, safe unpacking, variant choice."""

from __future__ import annotations

import hashlib
import io
import os
import sys
import tarfile
import zipfile

import httpx
import pytest

from ollamapower.llamacpp import install as inst
from ollamapower.llamacpp import variants
from ollamapower.llamacpp.variants import Variant, VariantError, archives_for, resolve_backend


def make_tgz(files: dict[str, bytes], links: dict[str, str] | None = None, top: str = "llama-b1") -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(f"{top}/{name}" if top else name)
            info.size, info.mode = len(data), 0o755
            tf.addfile(info, io.BytesIO(data))
        for name, target in (links or {}).items():
            info = tarfile.TarInfo(f"{top}/{name}")
            info.type, info.linkname = tarfile.SYMTYPE, target
            tf.addfile(info)
    return buf.getvalue()


def make_zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


class FakeRelease:
    """Serves archives like GitHub's release downloads, with Range support."""

    def __init__(self, archives: dict[str, bytes]) -> None:
        self.archives = archives
        self.requests: list[httpx.Request] = []
        self.cut_after: int | None = None  # simulate a dropped connection

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        data = self.archives.get(request.url.path.rsplit("/", 1)[-1])
        if data is None:
            return httpx.Response(404)
        start = 0
        if "range" in request.headers:
            start = int(request.headers["range"].split("=")[1].split("-")[0])
        body = data[start:]
        if self.cut_after is not None:
            body, self.cut_after = body[: self.cut_after], None
        return httpx.Response(206 if start else 200, content=body)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler))

    def pins(self, tag: str = "b1", **wrong: str) -> inst.Pins:
        return inst.Pins(
            tag=tag,
            build=1,
            archives={
                name: {"size": len(data), "sha256": wrong.get(name, hashlib.sha256(data).hexdigest())}
                for name, data in self.archives.items()
            },
        )


def test_download_and_install(tmp_path):
    # Creating symlinks needs a privilege on Windows (which only gets .zip builds anyway).
    links = {} if sys.platform == "win32" else {"libggml.so": "libggml.so.0"}
    rel = FakeRelease({"a.tar.gz": make_tgz({"llama-server": b"#!srv", "libggml.so.0": b"lib"}, links)})
    pins, seen = rel.pins(), []
    target = inst.install(
        Variant("cpu", ("a.tar.gz",)), tmp_path, pins, progress=lambda n, d, t: seen.append(d), client=rel.client()
    )
    assert (target / "llama-server").read_bytes() == b"#!srv"
    if sys.platform != "win32":
        assert os.readlink(target / "libggml.so") == "libggml.so.0"
        assert os.access(target / "llama-server", os.X_OK)
    assert seen[-1] == len(rel.archives["a.tar.gz"])
    assert not (tmp_path / "llamacpp" / "downloads" / "a.tar.gz").exists()  # archive removed after unpacking
    # Installed: nothing is fetched again.
    count = len(rel.requests)
    inst.install(Variant("cpu", ("a.tar.gz",)), tmp_path, pins, client=rel.client())
    assert len(rel.requests) == count


def test_hash_mismatch_is_deleted_and_never_unpacked(tmp_path):
    rel = FakeRelease({"a.tar.gz": make_tgz({"llama-server": b"evil"})})
    pins = rel.pins(**{"a.tar.gz": "0" * 64})
    with pytest.raises(inst.InstallError, match="does not match its pinned SHA-256"):
        inst.install(Variant("cpu", ("a.tar.gz",)), tmp_path, pins, client=rel.client())
    assert not list((tmp_path / "llamacpp" / "downloads").iterdir())
    assert not (tmp_path / "llamacpp" / "b1").exists()


def test_interrupted_download_resumes(tmp_path):
    data = make_tgz({"llama-server": os.urandom(200_000)})
    rel = FakeRelease({"a.tar.gz": data})
    rel.cut_after = 50_000
    pins = rel.pins()
    downloads = tmp_path / "dl"
    downloads.mkdir()
    with pytest.raises(inst.InstallError, match="incomplete"):
        inst.download("a.tar.gz", downloads, pins, client=rel.client())
    path = inst.download("a.tar.gz", downloads, pins, client=rel.client())
    assert path.read_bytes() == data
    assert rel.requests[-1].headers["range"] == "bytes=50000-"


def test_oversized_download_is_refused(tmp_path):
    rel = FakeRelease({"a.tar.gz": b"x" * 100})
    pins = rel.pins()
    pins.archives["a.tar.gz"]["size"] = 10
    with pytest.raises(inst.InstallError, match="larger than the pinned size"):
        inst.download("a.tar.gz", tmp_path, pins, client=rel.client())


def test_unknown_archive_is_refused(tmp_path):
    with pytest.raises(inst.InstallError, match="not in the pinned release"):
        inst.download("other.zip", tmp_path, FakeRelease({}).pins(), client=FakeRelease({}).client())


@pytest.mark.parametrize(
    "archive",
    [
        make_tgz({"../escape": b"x"}, top=""),
        make_tgz({"/etc/passwd": b"x"}, top=""),
        make_tgz({}, {"evil": "../../outside"}),
        make_tgz({}, {"evil": "/etc/passwd"}),
    ],
)
def test_unsafe_tar_members_are_refused(tmp_path, archive):
    path = tmp_path / "a.tar.gz"
    path.write_bytes(archive)
    with pytest.raises(inst.InstallError, match="unsafe"):
        inst.unpack(path, tmp_path / "out")
    assert not (tmp_path / "escape").exists()


@pytest.mark.parametrize("name", ["../escape.dll", "C:/windows/evil.dll", "/abs.dll"])
def test_unsafe_zip_members_are_refused(tmp_path, name):
    path = tmp_path / "a.zip"
    path.write_bytes(make_zip({name: b"x"}))
    with pytest.raises(inst.InstallError, match="unsafe"):
        inst.unpack(path, tmp_path / "out")


def test_windows_zip_layout_and_cuda_runtime_merge(tmp_path):
    rel = FakeRelease(
        {
            "llama-b1-bin-win-cuda.zip": make_zip({"llama-server.exe": b"srv", "ggml-cuda.dll": b"cuda"}),
            "cudart-llama-bin-win-cuda.zip": make_zip({"cudart64_12.dll": b"rt"}),
        }
    )
    target = inst.install(
        Variant("cuda12", ("llama-b1-bin-win-cuda.zip", "cudart-llama-bin-win-cuda.zip")),
        tmp_path,
        rel.pins(),
        client=rel.client(),
    )
    assert {p.name for p in target.iterdir()} >= {"llama-server.exe", "ggml-cuda.dll", "cudart64_12.dll"}


def test_pins_file_covers_every_variant():
    """Every build we may pick has a pinned size and SHA-256, and every pin is used."""
    pins = inst.Pins.load()
    assert pins.tag == f"b{pins.build}"
    used = set()
    for system, arch, backend in variants._table(pins.tag):
        for name in archives_for(pins.tag, system, arch, backend).archives:
            entry = pins.expect(name)
            assert len(entry["sha256"]) == 64 and entry["size"] > 1_000_000
            used.add(name)
    assert used == set(pins.archives)


def test_backend_choice():
    assert resolve_backend("auto", "macos", "arm64", []) == "metal"
    assert resolve_backend("auto", "macos", "x64", []) == "cpu"
    assert resolve_backend("auto", "linux", "x64", ["nvidia"]) == "vulkan"
    assert resolve_backend("auto", "windows", "x64", []) == "cpu"
    assert resolve_backend("cuda13", "linux", "x64", ["nvidia"]) == "cuda13"
    # No Vulkan build for Windows on ARM: auto must not pick a build that doesn't exist.
    assert resolve_backend("auto", "windows", "arm64", ["qualcomm"]) == "cpu"
    assert resolve_backend("auto", "linux", "arm64", ["nvidia"]) == "vulkan"
    for system, arch in [("linux", "x64"), ("linux", "arm64"), ("windows", "x64"), ("windows", "arm64")]:
        for vendors in ([], ["nvidia"]):
            archives_for("b1", system, arch, resolve_backend("auto", system, arch, vendors))
    with pytest.raises(VariantError, match="no official llama.cpp cuda12 build for macos"):
        archives_for("b1", "macos", "arm64", "cuda12")


@pytest.mark.parametrize(
    ("platform_name", "machine", "translated", "expected"),
    [
        ("darwin", "x86_64", "1", ("macos", "arm64")),  # Intel Python under Rosetta on Apple Silicon
        ("darwin", "x86_64", "0", ("macos", "x64")),  # a real Intel Mac
        ("darwin", "arm64", "", ("macos", "arm64")),
        ("win32", "AMD64", "", ("windows", "x64")),
        ("win32", "ARM64", "", ("windows", "arm64")),
        ("linux", "aarch64", "", ("linux", "arm64")),
    ],
)
def test_current_platform(monkeypatch, platform_name, machine, translated, expected):
    monkeypatch.setattr(variants.sys, "platform", platform_name)
    monkeypatch.setattr(variants.platform, "machine", lambda: machine)
    monkeypatch.setattr(
        variants.subprocess, "run", lambda *a, **k: variants.subprocess.CompletedProcess(a, 0, translated, "")
    )
    assert variants.current_platform() == expected


def test_unpack_failure_is_an_install_error_and_leaves_nothing(tmp_path, monkeypatch):
    """E.g. Windows without the symlink privilege, or a full disk: a message, no half install."""
    rel = FakeRelease({"a.tar.gz": make_tgz({"llama-server": b"#!srv"}, {"libggml.so": "llama-server"})})

    def no_symlinks(*_a, **_k):
        raise OSError(1314, "A required privilege is not held by the client")

    monkeypatch.setattr(inst.os, "symlink", no_symlinks)
    with pytest.raises(inst.InstallError, match="cannot install llama.cpp"):
        inst.install(Variant("cpu", ("a.tar.gz",)), tmp_path, rel.pins(), client=rel.client())
    tag_dir = tmp_path / "llamacpp" / "b1"
    assert not tag_dir.exists() or not list(tag_dir.iterdir())


def test_ensure_turns_os_errors_into_a_message(tmp_path, monkeypatch):
    from ollamapower.llamacpp import manager

    def locked(*_a, **_k):
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(manager, "install", locked)
    with pytest.raises(manager.LlamaCppUnavailable, match="Access is denied"):
        manager.ensure(tmp_path, "cpu", pins=FakeRelease({}).pins())
