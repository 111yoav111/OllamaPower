from __future__ import annotations

import json
import os
import signal
import subprocess
import sys

import httpx
import pytest
from typer.testing import CliRunner

from ollamapower import __version__
from ollamapower.__main__ import cli
from ollamapower.admin import ipc
from ollamapower.admin.service import Admin
from ollamapower.core.auth import TOKEN_ENV
from ollamapower.files import read_env_file

from .conftest import free_port, wait_for


def run(*args, env=None):
    return CliRunner().invoke(cli, list(args), env=env)


@pytest.fixture
def cfg_env(tmp_path, monkeypatch):
    monkeypatch.setenv("OLLAMAPOWER_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("OLLAMAPOWER_TOKEN", raising=False)
    return tmp_path


def test_version():
    result = run("version")
    assert result.exit_code == 0 and __version__ in result.output


def test_no_args_shows_help():
    assert "Usage" in run().output


def test_init_creates_everything_once(cfg_env):
    result = run("init")
    assert result.exit_code == 0, result.output
    token = read_env_file(cfg_env / ".env")[TOKEN_ENV]
    assert token.startswith("op_") and len(token) > 40
    assert (cfg_env / "config.toml").exists() and json.loads((cfg_env / "state.json").read_text())["version"] == 1
    if sys.platform != "win32":
        assert oct(os.stat(cfg_env / ".env").st_mode)[-3:] == "600"
    again = run("init")
    assert "already set" in again.output and read_env_file(cfg_env / ".env")[TOKEN_ENV] == token
    rotated = run("init", "--rotate-token")
    assert "rotated" in rotated.output and read_env_file(cfg_env / ".env")[TOKEN_ENV] != token


def test_a_weak_token_from_the_environment_is_not_accepted(cfg_env):
    assert run("init").exit_code == 0
    strong = read_env_file(cfg_env / ".env")[TOKEN_ENV]
    shown = run("token", "show", env={TOKEN_ENV: "short-and-weak"})
    assert shown.exit_code == 1 and "short-and-weak" not in shown.output
    assert f"{TOKEN_ENV} in the environment is too short" in shown.output and f"unset {TOKEN_ENV}" in shown.output
    assert strong in run("token", "show").output

    init = run("init", env={TOKEN_ENV: "short-and-weak"})
    assert init.exit_code == 0 and "in the environment is too short" in init.output
    assert read_env_file(cfg_env / ".env")[TOKEN_ENV] == strong  # the file's good token is kept

    agent = run("agent", "--no-tui", "--allow-root", "--hub", "127.0.0.1", env={TOKEN_ENV: "short-and-weak"})
    assert agent.exit_code == 2 and f"{TOKEN_ENV} in the environment is too short" in agent.output


def test_init_replaces_a_weak_token_in_the_file(cfg_env):
    (cfg_env / ".env").write_text(f'{TOKEN_ENV}="weak"\n')
    result = run("init")
    assert result.exit_code == 0 and "The old token is too short" in result.output
    assert len(read_env_file(cfg_env / ".env")[TOKEN_ENV]) > 40


@pytest.mark.parametrize(
    "args",
    [
        ["nodes", "tunnel", "desk", "plaintext", "--yes"],
        ["models", "pin", "llama3:8b", "fast"],
        ["models", "pin", "llama3:8b", "split", "--ctx", "10"],
        ["models", "benchmark", "llama3:8b", "--ctx", "99999999"],
        ["models", "plan", "llama3:8b", "--split", "--ctx", "-1"],
        ["models", "plan", "llama3:8b"],
        ["logs", "--limit", "0"],
        ["logs", "--file", "bogus"],
        ["llamacpp", "install", "--backend", "gpu"],
        ["agent", "rules", "--max-vram-percent", "150"],
        ["agent", "rules", "--schedule", "late"],
        ["agent", "rules", "--gpus", "a,b"],
        ["agent", "--no-tui", "--allow-root", "--hub", "[fd00::1"],
        ["agent", "forget-hub", "--hub", "10.0.0.1:99999", "--yes"],
    ],
)
def test_usage_errors_exit_2(cfg_env, args):
    # None of these reach a running hub or agent (none runs): the arguments are refused first.
    result = run(*args)
    assert result.exit_code == 2, result.output


def test_forget_hub_parses_the_address_like_agent_hub(cfg_env):
    from ollamapower.agent.pinning import KnownHubs

    known = KnownHubs(cfg_env / "known_hub")
    known.pin("fd00::1:9000", "ab" * 32)
    known.pin("192.168.1.10:9000", "cd" * 32)
    known.pin("192.168.1.10:9100", "ef" * 32)

    missing = run("agent", "forget-hub", "--hub", "192.168.1.11", "--yes")
    assert missing.exit_code == 1 and "No pinned key for 192.168.1.11:9000" in missing.output
    assert len(known.entries()) == 3

    assert "Forgot 1" in run("agent", "forget-hub", "--hub", "[fd00::1]:9000", "--yes").output
    by_dir = run("agent", "forget-hub", "--config-dir", str(cfg_env), "--hub", "192.168.1.10", "--yes")
    assert "Forgot 1" in by_dir.output
    assert known.entries() == ["192.168.1.10:9100"]  # the default port (9000) was meant, not this one
    assert "Forgot 1" in run("agent", "forget-hub", "--yes").output and known.entries() == []


def test_config_dir_short_option_and_llamacpp_config_dir(cfg_env, tmp_path):
    from ollamapower.agent.pinning import KnownHubs
    from ollamapower.llamacpp.install import Pins

    from .fake_llamacpp import install_fakes

    helper = tmp_path / "helper"
    helper.mkdir()
    KnownHubs(helper / "known_hub").pin("10.0.0.2:9000", "ab" * 32)
    forgot = run("agent", "forget-hub", "-c", str(helper), "--hub", "10.0.0.2", "--yes")
    assert forgot.exit_code == 0 and "Forgot 1" in forgot.output
    assert run("agent", "status", "-c", str(helper)).exit_code == 3  # -c reached the agent's run dir

    build = Pins.load().build
    fakes = install_fakes(tmp_path / "bin", [], build)
    (helper / "agent.toml").write_text(f'[llamacpp]\npath = "{fakes}"\n')
    status = run("llamacpp", "status", "--config-dir", str(helper), "--json")
    assert status.exit_code == 0, status.output
    assert json.loads(status.output)["build"] == build
    assert run("llamacpp", "status").exit_code == 1  # the default config dir (cfg_env) has no llama.cpp


def test_status_when_not_running(cfg_env):
    result = run("status")
    assert result.exit_code == 3 and "not running" in result.output


@pytest.fixture
def admin_frontdoor(frontdoor, cfg_env):
    control = frontdoor.run(ipc.start_control_server(frontdoor.config.config_dir, Admin(frontdoor.hub).dispatch))
    yield frontdoor
    frontdoor.run(control.close())


def test_status_over_the_admin_channel(admin_frontdoor, client):
    fd = admin_frontdoor
    assert wait_for(lambda: fd.call(lambda: fd.hub.ollama.reachable))
    result = run("status", "--json")
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["ollama"]["version"] == "0.0.0-fake"
    plain = run("status").output
    assert "Front door:" in plain and "big:70b" in plain and "⚠ big:70b is only 50% in VRAM" in plain


def test_logs_command(admin_frontdoor, client):
    fd = admin_frontdoor
    client.post(fd.url + "/api/chat", json={"model": "small:1b", "messages": []})
    path = fd.config.logs_dir / "requests.jsonl"
    assert wait_for(lambda: path.exists() and path.read_text().strip())
    out = run("logs", "--file", "requests").output
    assert "/api/chat" in out and "model=small:1b" in out
    raw = run("logs", "--json", "--kind", "request").output.splitlines()
    assert json.loads(raw[-1])["path"] == "/api/chat"
    assert run("logs", "--file", "bogus").exit_code == 2


def test_doctor_exit_code(cfg_env):
    (cfg_env / "config.toml").write_text(f'[ollama]\nurl = "http://127.0.0.1:{free_port()}"\n')
    result = run("doctor")
    assert result.exit_code == 1 and "[ERROR]" in result.output and "fix:" in result.output


def test_doctor_survives_a_malformed_ollama_host(cfg_env):
    result = run("doctor", env={"OLLAMA_HOST": "http://[::1"})
    assert result.exit_code == 1 and isinstance(result.exception, SystemExit)
    assert "OLLAMA_HOST is set to 'http://[::1', which is not a valid address" in result.output


def test_hub_refuses_non_loopback(cfg_env):
    (cfg_env / "config.toml").write_text('[hub]\nhost = "0.0.0.0"\n')
    result = run("hub", "--allow-root")
    assert result.exit_code == 2 and "Refusing to listen on 0.0.0.0" in result.output


@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() != 0, reason="needs root to test the root check")
def test_hub_refuses_root(cfg_env):
    result = run("hub")
    assert result.exit_code == 2 and "root" in result.output


@pytest.mark.skipif(sys.platform == "win32", reason="SIGINT handling differs on Windows")
def test_hub_end_to_end(fake_ollama, tmp_path):
    """The real `ollamapower hub` process: serves, answers `status`, shuts down cleanly on Ctrl+C."""
    upstream, rec = fake_ollama
    port = free_port()
    (tmp_path / "config.toml").write_text(f'[hub]\nport = {port}\n[ollama]\nurl = "{upstream.url}"\n')
    env = {**os.environ, "OLLAMAPOWER_CONFIG_DIR": str(tmp_path)}
    env.pop("OLLAMAPOWER_TOKEN", None)
    proc = subprocess.Popen(
        [sys.executable, "-m", "ollamapower", "hub", "--no-tui", "--allow-root"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        url = f"http://127.0.0.1:{port}"

        def up():
            try:
                return httpx.get(url + "/ollamapower/status", timeout=0.5, trust_env=False).status_code == 200
            except httpx.HTTPError:
                return False

        assert wait_for(up, 15), proc.stdout.read() if proc.poll() is not None else "hub did not start"
        r = httpx.post(url + "/api/chat", json={"model": "small:1b", "messages": []}, trust_env=False)
        assert r.status_code == 200
        status = subprocess.run(
            [sys.executable, "-m", "ollamapower", "status", "--json"], env=env, capture_output=True, text=True
        )
        assert status.returncode == 0, status.stderr
        assert json.loads(status.stdout)["requests"]["total"] >= 1
        assert (tmp_path / "run" / "admin.json").exists()
    finally:
        proc.send_signal(signal.SIGINT)
        out, _ = proc.communicate(timeout=15)
    assert proc.returncode == 0, out
    assert "front door on" in out and "OLLAMAPOWER_TOKEN is not set" in out
    assert not (tmp_path / "run" / "admin.json").exists()
    assert (tmp_path / "state.json").exists()
    kinds = [json.loads(line)["kind"] for line in (tmp_path / "logs" / "events.jsonl").read_text().splitlines()]
    assert kinds[0] == "hub_start" and kinds[-1] == "hub_stop"
