from __future__ import annotations

import os
import stat
import sys

import pytest

from ollamapower import files
from ollamapower.core.auth import TOKEN_ENV

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")


def test_env_file_parsing(tmp_path):
    p = tmp_path / ".env"
    p.write_text(f'# comment\nexport A=1\nB = "two"\nC=\'3\'\n{TOKEN_ENV}="op_abc"\nbroken line\n')
    assert files.read_env_file(p) == {"A": "1", "B": "two", "C": "3", TOKEN_ENV: "op_abc"}
    assert files.read_env_file(tmp_path / "missing") == {}


def test_token_env_wins_over_file(tmp_path):
    p = tmp_path / ".env"
    files.write_token(p, "from-file")
    assert files.load_token(p, environ={}) == "from-file"
    assert files.load_token(p, environ={TOKEN_ENV: "from-env"}) == "from-env"
    assert files.load_token(tmp_path / "none", environ={}) is None


def test_write_token_keeps_other_lines(tmp_path):
    p = tmp_path / ".env"
    p.write_text(f"OTHER=1\n{TOKEN_ENV}=old\n{TOKEN_ENV}_EXTRA=keep\n")
    files.write_token(p, "new")
    values = files.read_env_file(p)
    assert values == {"OTHER": "1", f"{TOKEN_ENV}_EXTRA": "keep", TOKEN_ENV: "new"}


@posix_only
def test_written_files_are_owner_only(tmp_path):
    target = tmp_path / "sub" / "secret"
    files.write_private(target, b"x")
    assert stat.S_IMODE(os.stat(target).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(target.parent).st_mode) == 0o700
    assert files.permission_problem(target) is None
    os.chmod(target, 0o644)
    assert "accessible by others" in files.permission_problem(target)


@posix_only
def test_symlinked_dir_is_refused(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    with pytest.raises(OSError):
        files.private_dir(link)
