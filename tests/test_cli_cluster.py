"""The real `ollamapower hub` and `ollamapower agent` processes, driven through the CLI."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys

import pytest
import typer

from ollamapower.agent.cli import parse_hub_address

from .conftest import free_port, wait_for

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="SIGINT handling differs on Windows")


def test_parse_hub_address():
    assert parse_hub_address("192.168.1.10", 9000) == ("192.168.1.10", 9000)
    assert parse_hub_address("192.168.1.10:9100", 9000) == ("192.168.1.10", 9100)
    assert parse_hub_address("[fd00::1]:9100", 9000) == ("fd00::1", 9100)
    assert parse_hub_address("fd00::1", 9000) == ("fd00::1", 9000)
    assert parse_hub_address("hub.lan", 9000) == ("hub.lan", 9000)
    for bad in ("", ":9000", "host:http", "host:70000"):
        with pytest.raises(typer.BadParameter):
            parse_hub_address(bad, 9000)


def cli(*args: str, env: dict, timeout: float = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "ollamapower", *args], env=env, capture_output=True, text=True, timeout=timeout
    )


def test_hub_and_agent_end_to_end(fake_ollama, tmp_path):
    upstream, _ = fake_ollama
    hub_dir, agent_dir = tmp_path / "hub", tmp_path / "agent"
    base = {k: v for k, v in os.environ.items() if not k.startswith("OLLAMAPOWER_")}
    hub_env = {**base, "OLLAMAPOWER_CONFIG_DIR": str(hub_dir)}
    agent_env = {**base, "OLLAMAPOWER_CONFIG_DIR": str(agent_dir)}

    init = cli("init", env=hub_env)
    assert init.returncode == 0, init.stderr
    fingerprint = re.search(r"((?:[0-9A-F]{2}:){31}[0-9A-F]{2})", init.stdout).group(1)
    control_port = free_port()
    (hub_dir / "config.toml").write_text(
        f'[hub]\nport = {free_port()}\n[ollama]\nurl = "{upstream.url}"\n[control]\nport = {control_port}\n'
        "heartbeat_timeout = 10.0\n"
    )
    token_line = cli("token", "show", env=hub_env).stdout.strip()
    agent_dir.mkdir()
    (agent_dir / ".env").write_text(token_line + "\n")
    os.chmod(agent_dir / ".env", 0o600)

    hub = subprocess.Popen(
        [sys.executable, "-m", "ollamapower", "hub", "--no-tui", "--allow-root"],
        env=hub_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    agent = None
    try:
        assert wait_for(lambda: (hub_dir / "run" / "admin.json").exists(), 15)
        agent = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "ollamapower",
                "agent",
                "--no-tui",
                "--allow-root",
                "--name",
                "desk",
                "--hub",
                f"127.0.0.1:{control_port}",
                "--hub-fingerprint",
                fingerprint,
            ],
            env=agent_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        def node_status():
            r = cli("nodes", "list", "--json", env=hub_env)
            nodes = json.loads(r.stdout) if r.returncode == 0 else []
            return nodes[0]["status"] if nodes else None

        assert wait_for(lambda: node_status() == "lending", 20)
        assert "desk" in cli("nodes", "list", env=hub_env).stdout
        assert "pinned" in cli("agent", "status", env=agent_env).stdout

        assert cli("agent", "pause", env=agent_env).returncode == 0
        assert wait_for(lambda: node_status() == "paused", 10)
        assert cli("agent", "resume", env=agent_env).returncode == 0
        assert wait_for(lambda: node_status() == "lending", 10)

        rules = cli("agent", "rules", "--max-vram-percent", "40", env=agent_env)
        assert rules.returncode == 0 and "max_vram_percent   40" in rules.stdout
        assert "max_vram_percent = 40" in (agent_dir / "agent.toml").read_text()

        test = cli("nodes", "net-test", "desk", env=hub_env, timeout=70)
        assert test.returncode == 0 and "RTT" in test.stdout, test.stderr
        show = json.loads(cli("nodes", "show", "desk", "--json", env=hub_env).stdout)
        assert show["net_tests"] and show["lending"]["max_vram_percent"] == 40

        status = cli("status", env=hub_env).stdout
        assert "1 lending" in status and fingerprint in status
        assert cli("nodes", "show", "nobody", env=hub_env).returncode == 1
        assert cli("nodes", "pause", "desk", env=hub_env).returncode == 0
        assert wait_for(lambda: node_status() == "paused", 5)
    finally:
        for proc in (agent, hub):
            if proc is not None:
                proc.send_signal(signal.SIGINT)
        outputs = [proc.communicate(timeout=20)[0] for proc in (agent, hub) if proc is not None]
    assert hub.returncode == 0, outputs[-1]
    assert agent.returncode == 0, outputs[0]
    assert "front door on" in outputs[-1] and fingerprint in outputs[-1]
    saved = json.loads((hub_dir / "state.json").read_text())["nodes"]
    assert [n["name"] for n in saved.values()] == ["desk"] and next(iter(saved.values()))["hub_paused"]
    assert (agent_dir / "known_hub").exists()


def test_agent_without_token_explains(tmp_path):
    env = {
        **{k: v for k, v in os.environ.items() if not k.startswith("OLLAMAPOWER_")},
        "OLLAMAPOWER_CONFIG_DIR": str(tmp_path),
    }
    r = cli("agent", "--no-tui", "--allow-root", "--hub", "127.0.0.1", env=env)
    assert r.returncode == 2 and "token show" in r.stderr


def test_agent_refuses_unknown_key_headless(tmp_path):
    """No terminal to ask in and no --hub-fingerprint: the agent stops before sending the token."""
    from ollamapower.core.auth import new_token

    port = free_port()
    hub_dir, agent_dir = tmp_path / "hub", tmp_path / "agent"
    env = {k: v for k, v in os.environ.items() if not k.startswith("OLLAMAPOWER_")}
    hub_env, agent_env = (
        {**env, "OLLAMAPOWER_CONFIG_DIR": str(hub_dir)},
        {**env, "OLLAMAPOWER_CONFIG_DIR": str(agent_dir)},
    )
    assert cli("init", env=hub_env).returncode == 0
    (hub_dir / "config.toml").write_text(f"[hub]\nport = {free_port()}\n[control]\nport = {port}\n")
    hub = subprocess.Popen(
        [sys.executable, "-m", "ollamapower", "hub", "--no-tui", "--allow-root"],
        env=hub_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        assert wait_for(lambda: (hub_dir / "run" / "admin.json").exists(), 15)
        agent_dir.mkdir()
        (agent_dir / ".env").write_text(f'OLLAMAPOWER_TOKEN="{new_token()}"\n')
        r = subprocess.run(
            [sys.executable, "-m", "ollamapower", "agent", "--no-tui", "--allow-root", "--hub", f"127.0.0.1:{port}"],
            env=agent_env,
            capture_output=True,
            text=True,
            timeout=30,
            stdin=subprocess.DEVNULL,
        )
        assert r.returncode == 4 and "--hub-fingerprint" in r.stderr
        events = cli("logs", "--kind", "auth_failed", env=hub_env).stdout
        assert "auth_failed" not in events  # the wrong token was never sent
    finally:
        hub.send_signal(signal.SIGINT)
        hub.communicate(timeout=20)
