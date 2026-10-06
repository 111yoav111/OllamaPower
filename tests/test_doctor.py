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


@pytest.mark.parametrize(
    "value, expected",
    [
        ("127.0.0.1:11435", ("127.0.0.1", 11435)),
        ("0.0.0.0", ("0.0.0.0", 11434)),
        ("http://localhost", ("localhost", 80)),
        ("https://127.0.0.1/", ("127.0.0.1", 443)),
        ("[::1]:11435", ("::1", 11435)),
        ("::1", ("::1", 11434)),
        (":11435", ("", 11435)),
    ],
)
def test_parse_ollama_host(value, expected):
    assert doctor.parse_ollama_host(value) == expected


@pytest.mark.parametrize("value", ["http://[::1", ":abc", "host:99999", "host:", "[::1]x", "ftp://h:1", "h:١٢"])
def test_malformed_ollama_host_is_a_failed_check_not_a_crash(make_config, monkeypatch, value):
    with pytest.raises(ValueError):
        doctor.parse_ollama_host(value)
    monkeypatch.setenv("OLLAMA_HOST", value)
    monkeypatch.setattr(doctor, "local_ip_addresses", lambda: [])
    finding = doctor.check_exposure(make_config(f"http://127.0.0.1:{free_port()}"))[0]
    assert finding.level == "error" and "not a valid address" in finding.title and "OLLAMA_HOST=" in finding.fix


def test_ollama_host_without_a_host_means_all_interfaces(make_config, monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", ":11435")
    monkeypatch.setattr(doctor, "local_ip_addresses", lambda: [])
    finding = doctor.check_exposure(make_config(f"http://127.0.0.1:{free_port()}"))[0]
    assert finding.level == "error" and "all interfaces" in finding.title


def test_front_door_port_exposure(make_config, monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    cfg = make_config(f"http://127.0.0.1:{free_port()}")
    front = cfg.hub.port
    monkeypatch.setattr(doctor, "reachable_on_lan", lambda port, timeout=0.5: ["192.168.1.5"] if port == front else [])
    exposure = doctor.check_exposure(cfg)
    assert exposure[0].level == "ok"  # Ollama's own port
    assert exposure[1].level == "error" and f"Port {front} is reachable on 192.168.1.5" in exposure[1].title
    assert "OLLAMA_HOST" in exposure[1].fix
    cfg.hub.host = "0.0.0.0"  # the owner's choice (--i-know-what-im-doing): a warning
    assert doctor.check_frontdoor_exposure(cfg)[0].level == "warn"
    monkeypatch.setattr(doctor, "reachable_on_lan", lambda port, timeout=0.5: [])
    assert doctor.check_frontdoor_exposure(cfg)[0].level == "warn"
    cfg.hub.host = "127.0.0.1"
    assert [f.level for f in doctor.check_frontdoor_exposure(cfg)] == ["ok"]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
def test_extended_permission_checks(make_config):
    cfg = make_config("http://127.0.0.1:11435")
    d = cfg.config_dir
    os.chmod(d, 0o700)
    write_token(cfg.env_path, new_token())
    for name in ("known_hub", "state.json"):
        (d / name).write_text("{}")
        os.chmod(d / name, 0o600)
    assert [f.level for f in doctor.check_permissions(cfg)] == ["ok"]

    os.chmod(cfg.state_path, 0o644)  # readable, not a secret: a warning
    os.chmod(d / "known_hub", 0o620)  # writable by others: they could pin their own "hub"
    os.chmod(d, 0o755)  # the config dir guards the secrets: an error
    found = {f.title.split(" ")[0]: f for f in doctor.check_permissions(cfg)}
    assert found[str(d)].level == "error" and "chmod 700" in found[str(d)].fix
    assert found[str(cfg.state_path)].level == "warn" and "readable by group" in found[str(cfg.state_path)].title
    assert found[str(d / "known_hub")].level == "error" and "writable" in found[str(d / "known_hub")].title
    os.chmod(d, 0o700)

    run = d / "run"
    run.mkdir(mode=0o700)
    (run / "admin.json").write_text("{}")
    os.chmod(run / "admin.json", 0o640)  # holds the admin channel's secret
    assert [f.level for f in doctor.check_permissions(cfg) if "admin.json" in f.title] == ["error"]


def test_permission_checks_skip_windows(make_config):
    findings = doctor.check_permissions(make_config("http://127.0.0.1:11435"), platform="win32")
    assert [f.level for f in findings] == ["ok"] and "Windows" in findings[0].title


def test_llamacpp_check(make_config, tmp_path):
    from ollamapower.llamacpp.install import Pins

    from .fake_llamacpp import install_fakes

    cfg = make_config("http://127.0.0.1:11435")
    missing = doctor.check_llamacpp(cfg, vendors=[])
    assert missing[0].level == "ok" and "not installed yet" in missing[0].title

    build = Pins.load().build
    cfg.llamacpp.path = str(install_fakes(tmp_path / "good", [], build))
    good = doctor.check_llamacpp(cfg, vendors=[])
    assert good[0].level == "ok" and f"build {build}" in good[0].title and "runs" in good[0].title

    cfg.llamacpp.path = str(install_fakes(tmp_path / "old", [], build - 1))
    wrong = doctor.check_llamacpp(cfg, vendors=[])
    assert wrong[0].level == "error" and f"needs build {build}" in wrong[0].title and "llamacpp.path" in wrong[0].fix

    cfg.llamacpp.path = str(tmp_path / "nothing-here")
    assert doctor.check_llamacpp(cfg, vendors=[])[0].level == "error"


def test_a_crashing_check_is_reported_not_raised(make_config, monkeypatch):
    def boom(config):
        raise RuntimeError("kaputt")

    monkeypatch.setattr(doctor, "check_ports", boom)
    monkeypatch.setattr(doctor, "check_llamacpp", lambda config: [])
    monkeypatch.setattr(doctor, "local_ip_addresses", lambda: [])
    findings = doctor.run_all(make_config("http://127.0.0.1:11435"))
    assert findings[0].level == "error" and "ports check failed" in findings[0].title and "kaputt" in findings[0].title
    assert len(findings) > 1
