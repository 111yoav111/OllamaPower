from __future__ import annotations

import os
import sys

import pytest

from ollamapower import doctor
from ollamapower.core.auth import new_token
from ollamapower.files import write_token

from .conftest import free_port


def levels(findings):
    return [(f.level, f.title) for f in findings]


@pytest.mark.parametrize("platform", ["linux", "darwin", "win32"])
def test_migration_steps_name_the_target(platform):
    steps = doctor.migration_steps("http://127.0.0.1:11435", platform)
    assert "127.0.0.1:11435" in steps
    assert "ollamapower doctor" in steps


def test_linux_steps_use_a_systemd_override():
    steps = doctor.migration_steps("http://127.0.0.1:11435", "linux")
    assert "sudo systemctl edit ollama" in steps and 'Environment="OLLAMA_HOST=127.0.0.1:11435"' in steps


def test_ollama_still_on_the_front_door_port(fake_ollama, make_config):
    upstream, _ = fake_ollama
    cfg = make_config(f"http://127.0.0.1:{free_port()}", hub__port=upstream.port)
    findings = doctor.check_ports(cfg)
    first = findings[0]
    assert first.level == "error" and "Ollama still listens on" in first.title
    assert "OLLAMA_HOST" in first.fix
    assert all("OllamaPower answers" not in f.title for f in findings)


def test_everything_moved(frontdoor, make_config):
    cfg = make_config(frontdoor.config.ollama.url, hub__port=frontdoor.server.port)
    assert [f.level for f in doctor.check_ports(cfg)] == ["ok", "ok"]


def test_nothing_running(make_config):
    cfg = make_config(f"http://127.0.0.1:{free_port()}")
    found = levels(doctor.check_ports(cfg))
    assert found[0][0] == "error" and "not reachable" in found[0][1]
    assert found[1][0] == "warn" and "Nothing listens" in found[1][1]


def test_lan_exposure_is_an_error(fake_ollama, make_config, monkeypatch):
    upstream, _ = fake_ollama
    cfg = make_config(upstream.url)
    monkeypatch.setattr(doctor, "local_ip_addresses", lambda: ["127.0.0.1"])  # pretend loopback is the LAN
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert doctor.check_exposure(cfg)[0].level == "error"
    monkeypatch.setattr(doctor, "local_ip_addresses", lambda: [])
    assert doctor.check_exposure(cfg)[0].level == "ok"


def test_ollama_host_env_on_all_interfaces(make_config, monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "0.0.0.0:11435")
    monkeypatch.setattr(doctor, "local_ip_addresses", lambda: [])
    cfg = make_config(f"http://127.0.0.1:{free_port()}")
    assert doctor.check_exposure(cfg)[0].level == "error"


def test_token_checks(make_config, monkeypatch):
    monkeypatch.delenv("OLLAMAPOWER_TOKEN", raising=False)
    cfg = make_config("http://127.0.0.1:11435")
    assert doctor.check_token(cfg)[0].level == "warn"
    write_token(cfg.env_path, "short")
    assert doctor.check_token(cfg)[0].level == "error"
    write_token(cfg.env_path, new_token())
    assert doctor.check_token(cfg)[0].level == "ok"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
def test_permission_checks(make_config):
    cfg = make_config("http://127.0.0.1:11435")
    write_token(cfg.env_path, new_token())
    assert doctor.check_permissions(cfg)[0].level == "ok"
    os.chmod(cfg.env_path, 0o644)
    bad = doctor.check_permissions(cfg)
    assert bad[0].level == "error" and "chmod 600" in bad[0].fix
