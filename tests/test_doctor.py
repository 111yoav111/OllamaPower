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
    finding = doctor.check_exposure(cfg)[0]
    assert finding.level == "error"
    assert f"OLLAMA_HOST=127.0.0.1:{cfg.ollama.url.rsplit(':', 1)[1]}" in finding.fix


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


@pytest.mark.parametrize(
    ("config_dir", "private"),
    [
        (r"C:\Users\Ann\AppData\Roaming\ollamapower", True),
        (r"c:\users\ann\AppData\Roaming\ollamapower", True),  # NTFS paths are case-insensitive
        (r"C:\Users\Ann", True),
        (r"C:\Users\Annabel\ollamapower", False),  # another user's profile, not a subfolder of ours
        (r"D:\OllamaPower", False),
        (r"C:\OllamaPower", False),
    ],
)
def test_windows_config_dir_must_stay_in_the_profile(make_config, config_dir, private):
    from pathlib import Path

    assert doctor.windows_dir_is_private(config_dir, r"C:\Users\Ann") is private
    cfg = make_config("http://127.0.0.1:11435")
    cfg.config_dir = Path(config_dir)
    [finding] = doctor.check_permissions(cfg, "win32", Path(r"C:\Users\Ann"))
    assert finding.level == ("ok" if private else "warn")
    assert private or "icacls" in finding.fix


def test_control_checks(make_config):
    cfg = make_config("http://127.0.0.1:11435")
    assert [f.level for f in doctor.check_control(cfg)] == ["ok", "warn"]  # no hub key yet
    cfg.control.allowed_networks = ["0.0.0.0/0"]
    assert doctor.check_control(cfg)[0].level == "warn"
    cfg.control.enabled = False
    assert doctor.check_control(cfg)[0].level == "ok"
