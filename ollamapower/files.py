"""Owner-only files: atomic writes, permission checks and the ``.env`` secret file.

Everything OllamaPower writes to disk (token, state, logs) may only be readable
by the owner. Writes go to a temp file in the same directory, are ``fsync``'d
and then ``os.replace``'d over the target, so a crash never leaves a
half-written file behind.
"""

from __future__ import annotations

import contextlib
import os
import stat
import sys
import tempfile
from pathlib import Path

from .core.auth import TOKEN_ENV

POSIX = sys.platform != "win32"


def private_dir(path: Path) -> None:
    """Create ``path`` as a directory only we can use; refuse anything suspicious."""
    path.mkdir(parents=True, exist_ok=True)
    st = os.lstat(path)
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
        raise OSError(f"{path} must be a real directory, not a symlink")
    if POSIX:
        if st.st_uid != os.getuid():
            raise OSError(f"{path} is not owned by the current user")
        os.chmod(path, 0o700)


def write_private(path: Path, data: bytes) -> None:
    """Atomically replace ``path`` with ``data``, readable by the owner only."""
    private_dir(path.parent)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        if POSIX:
            os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    if POSIX:
        with contextlib.suppress(OSError):  # make the rename itself durable
            dir_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)


def permission_problem(path: Path) -> str | None:
    """Why ``path`` is not owner-only, or None if it is (or doesn't exist). POSIX only."""
    if not POSIX or not path.exists():
        return None
    st = os.lstat(path)
    if stat.S_ISLNK(st.st_mode):
        return "is a symlink"
    if st.st_uid != os.getuid():
        return "is owned by another user"
    if st.st_mode & 0o077:
        return f"is accessible by others (mode {stat.S_IMODE(st.st_mode):o})"
    return None


# ----- .env ----------------------------------------------------------------------------


def read_env_file(path: Path) -> dict[str, str]:
    """Parse ``KEY=VALUE`` lines (optional quotes, ``#`` comments). Missing file = empty."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    values: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip().removeprefix("export ").strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def load_token(env_path: Path, environ: dict[str, str] | None = None) -> str | None:
    """The cluster token: the environment wins over the ``.env`` file."""
    environ = dict(os.environ) if environ is None else environ
    return environ.get(TOKEN_ENV) or read_env_file(env_path).get(TOKEN_ENV) or None


def write_token(env_path: Path, token: str) -> None:
    """Write (or replace) the token line in ``.env``, keeping any other lines."""
    lines = []
    try:
        lines = [
            ln
            for ln in env_path.read_text(encoding="utf-8").splitlines()
            if ln.partition("=")[0].strip().removeprefix("export ").strip() != TOKEN_ENV
        ]
    except FileNotFoundError:
        pass
    lines.append(f'{TOKEN_ENV}="{token}"')
    write_private(env_path, ("\n".join(lines) + "\n").encode("utf-8"))
