from __future__ import annotations

from typer.testing import CliRunner

from ollamapower import __version__
from ollamapower.__main__ import cli


def test_version():
    result = CliRunner().invoke(cli, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_no_args_shows_help():
    result = CliRunner().invoke(cli, [])
    assert "Usage" in result.output
