"""`ollamapower setup` / `uninstall`: only validated values reach sudo, and undo removes exactly what was added."""

from __future__ import annotations

import socket
import subprocess
import types

import pytest
import typer

from ollamapower import setup as s
from ollamapower.cli_common import apply_setup_plan

LAN = "192.168.1.0/24"


def addr(ip: str, mask: str, family: int = socket.AF_INET) -> types.SimpleNamespace:
    return types.SimpleNamespace(family=family, address=ip, netmask=mask)


def done(returncode: int = 0, out: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], returncode, out, "")


class FakeSudo:
    """Answers like ufw / firewalld would, and remembers what ran."""

    def __init__(self, existing: str = "", fail: tuple[str, ...] = ()) -> None:
        self.ran: list[tuple[str, ...]] = []
        self.existing, self.fail = existing, fail

    def __call__(self, step: s.Step) -> subprocess.CompletedProcess[str]:
        self.ran.append(step.argv)
        if any(word in step.argv for word in self.fail):
            return done(1, "boom")
        if step.argv[:3] == ("ufw", "show", "added"):
            return done(out=f"Added user rules (see 'ufw status' for running firewall):\n{self.existing}")
        if "--query-rich-rule" in step.argv:
            return done(0 if self.existing and self.existing in step.argv[-1] else 1)
        return done(out="Rule added")


@pytest.mark.parametrize("value", [0, 65536, -1, True, "9000", 9000.0, None])
def test_only_real_ports_pass(value):
    with pytest.raises(s.SetupError):
        s.valid_port(value)


@pytest.mark.parametrize(
    "url,why",
    [
        ("http://0.0.0.0:11435", "loopback"),
        ("http://192.168.1.5:11435", "loopback"),
        ("http://ollama.example:11435", "loopback IP"),
        ("http://127.0.0.1", "explicit port"),
        ('http://127.0.0.1:11435"\nExecStart=/bin/sh', "port"),  # an attempt to smuggle a line into the unit file
    ],
)
def test_ollama_target_accepts_only_a_loopback_ip_and_a_port(url, why):
    with pytest.raises(s.SetupError, match=why):
        s.ollama_target(url)


def test_ollama_target_normalises():
    assert s.ollama_target("http://localhost:11435") == ("127.0.0.1", 11435)
    assert s.ollama_target("http://[::1]:11435") == ("[::1]", 11435)


def test_lan_networks_are_private_narrow_and_never_container_bridges():
    nets = s.lan_networks(
        {
            "enp3s0": [addr("192.168.1.156", "255.255.255.0")],
            "wlan0": [addr("192.168.1.32", "255.255.255.0")],  # same network twice: one rule
            "lo": [addr("127.0.0.1", "255.0.0.0")],
            "pub": [addr("8.8.8.8", "255.255.255.0")],
            "ll": [addr("169.254.3.3", "255.255.0.0")],
            "corp": [addr("10.1.2.3", "255.0.0.0")],  # a whole /8: too wide
            "docker0": [addr("172.17.0.1", "255.255.0.0")],
            "br-1a2b": [addr("172.18.0.1", "255.255.0.0")],
            "v6": [addr("fd00::1", "ffff::", family=socket.AF_INET6)],
            "wg0": [addr("10.8.0.2", "255.255.255.0")],
        }
    )
    assert nets == {"10.8.0.0/24": "wg0", LAN: "enp3s0, wlan0"}


def test_firewall_steps_are_fixed_argument_lists():
    add = s.firewall_steps("ufw", [(LAN, 9000)])
    assert [st.argv for st in add] == [
        ("ufw", "show", "added"),  # read-only check first
        ("ufw", "allow", "from", LAN, "to", "any", "port", "9000", "proto", "tcp", "comment", "ollamapower"),
    ]
    remove = s.firewall_steps("ufw", [(LAN, 9000)], remove=True)
    assert [st.argv[:3] for st in remove] == [("ufw", "delete", "allow")] and remove[0].rule == (LAN, 9000)
    fw = s.firewall_steps("firewalld", [(LAN, 9001)])
    assert [st.argv[2] for st in fw[:2]] == ["--query-rich-rule", "--add-rich-rule"]
    assert f'source address="{LAN}" port port="9001"' in fw[1].argv[3]
    assert fw[-1].argv == ("firewall-cmd", "--reload")
    assert s.firewall_steps(None, [(LAN, 9000)]) == []


@pytest.mark.parametrize(
    "net", [f"{LAN}; rm -rf /", f"{LAN}\" accept'", "any", "0.0.0.0/0", "10.0.0.0/8", "8.8.8.0/24"]
)
def test_a_network_that_isnt_a_private_lan_never_reaches_a_command(net):
    with pytest.raises(ValueError):
        s.firewall_steps("ufw", [(net, 9000)])


def test_the_dropin_holds_one_validated_line():
    assert s.dropin_text("127.0.0.1", 11435).splitlines()[1:] == [
        "[Service]",
        'Environment="OLLAMA_HOST=127.0.0.1:11435"',
    ]
    with pytest.raises(s.SetupError):
        s.dropin_text("127.0.0.1", 70000)


@pytest.fixture
def systemd(monkeypatch):
    monkeypatch.setattr(s, "has_systemd_ollama", lambda: True)
    monkeypatch.setattr(s, "ollama_host_set_elsewhere", lambda: [])
    monkeypatch.setattr(s, "active_firewall", lambda: "ufw")
    monkeypatch.setattr(s, "lan_networks", lambda: {LAN: "enp3s0"})


def test_hub_plan_moves_ollama_then_opens_the_control_port_and_says_so(systemd):
    plan = s.plan_hub_setup("http://127.0.0.1:11435", 11434, 9000, control_open=True)
    assert plan.steps[1].argv == ("tee", s.DROPIN) and "127.0.0.1:11435" in (plan.steps[1].stdin or "")
    assert ("systemctl", "restart", "ollama.service") in [st.argv for st in plan.steps]
    assert plan.rules == [(LAN, 9000)]
    assert "restarts" in plan.summary and "cluster token" in plan.summary and f"{LAN} (enp3s0)" in plan.summary
    assert s.plan_hub_setup("http://127.0.0.1:11435", 11434, 9000, control_open=False).rules == []


@pytest.mark.parametrize(
    "url,front,why",
    [("http://127.0.0.1:11434", 11000, "usual port"), ("http://127.0.0.1:11435", 11435, "both use port")],
)
def test_hub_plan_refuses_ports_that_make_no_sense(systemd, url, front, why):
    with pytest.raises(s.SetupError, match=why):
        s.plan_hub_setup(url, front, 9000, control_open=True)


def test_hub_plan_refuses_when_someone_else_sets_ollamas_address(systemd, monkeypatch):
    monkeypatch.setattr(s, "ollama_host_set_elsewhere", lambda: ["/etc/systemd/system/ollama.service.d/override.conf"])
    with pytest.raises(s.SetupError, match="override.conf.*systemctl edit ollama"):
        s.plan_hub_setup("http://127.0.0.1:11435", 11434, 9000, control_open=True)


@pytest.mark.parametrize("backend", ["ufw", "firewalld"])
def test_a_rule_the_owner_already_had_is_left_alone_and_never_recorded(backend):
    other = "10.8.0.0/24"
    plan = s.Plan(steps=s.firewall_steps(backend, [(LAN, 9000), (other, 9000)]))
    # ufw lists it; firewalld's query answers yes (exit 0) - whatever language the tools speak.
    existing = f"ufw allow from {LAN} to any port 9000 proto tcp" if backend == "ufw" else f'"{LAN}"'
    sudo, changed = FakeSudo(existing), []
    s.execute(plan, sudo, adding=True, changed=changed)
    assert changed == [(other, 9000)]
    added = [a for a in sudo.ran if "allow" in a[:2] or "--add-rich-rule" in a]
    assert len(added) == 1 and other in " ".join(added[0])


def test_a_failure_stops_but_keeps_what_was_done():
    plan = s.Plan(steps=[*s.firewall_steps("ufw", [(LAN, 9000)]), s.Step("x", ("systemctl", "restart", "x"))])
    changed: list = []
    with pytest.raises(s.SetupError, match="boom"):
        s.execute(plan, FakeSudo(fail=("restart",)), adding=True, changed=changed)
    assert changed == [(LAN, 9000)]


def test_uninstall_removes_only_recorded_rules():
    plan = s.plan_uninstall({"firewall": "ufw", "rules": [[LAN, 9001]]}, hub=False, port=9001)
    assert [st.argv[:3] for st in plan.steps] == [("ufw", "delete", "allow")]
    assert s.plan_uninstall({}, hub=False, port=9001).steps == []


@pytest.mark.parametrize(
    "record",
    [
        {"firewall": "ufw", "rules": [["0.0.0.0/0", 22]]},  # malware asking uninstall to close SSH
        {"firewall": "ufw", "rules": [[LAN, 22]]},  # a port that was never ours
        {"firewall": "iptables", "rules": [[LAN, 9001]]},
        {"firewall": "ufw", "rules": [["x", "y"]]},
        {"firewall": "ufw", "rules": "nope"},
        {"firewall": "ufw", "rules": [[LAN]]},
    ],
)
def test_uninstall_refuses_a_record_it_didnt_write(record):
    with pytest.raises(s.SetupError, match="doesn't look like one OllamaPower wrote"):
        s.plan_uninstall(record, hub=False, port=9001)


@pytest.fixture
def user(monkeypatch):
    monkeypatch.setattr("ollamapower.cli_common.running_as_admin", lambda: False)


def test_apply_refuses_root(tmp_path, monkeypatch):
    monkeypatch.setattr("ollamapower.cli_common.running_as_admin", lambda: True)
    plan = s.Plan(steps=s.firewall_steps("ufw", [(LAN, 9001)]), firewall="ufw")
    with pytest.raises(typer.Exit):
        apply_setup_plan(plan, yes=True, print_only=False, adding=True, record_dir=tmp_path, record_name=s.AGENT_RECORD)


def test_apply_records_exactly_and_print_runs_nothing(tmp_path, monkeypatch, user, capsys):
    plan = s.Plan(steps=s.firewall_steps("ufw", [(LAN, 9001)]), firewall="ufw", rules=[(LAN, 9001)])
    sudo = FakeSudo()
    monkeypatch.setattr(s, "sudo_runner", sudo)
    assert not apply_setup_plan(
        plan, yes=True, print_only=True, adding=True, record_dir=tmp_path, record_name=s.AGENT_RECORD
    )
    assert sudo.ran == []
    sudo.existing = f"ufw allow from {LAN} to any port 9001 proto tcp"  # the owner opened it already
    assert apply_setup_plan(
        plan, yes=True, print_only=False, adding=True, record_dir=tmp_path, record_name=s.AGENT_RECORD
    )
    assert "was open already" in capsys.readouterr().out
    assert not (tmp_path / s.AGENT_RECORD).exists()  # nothing of ours to undo
    sudo.existing = ""
    assert apply_setup_plan(
        plan, yes=True, print_only=False, adding=True, record_dir=tmp_path, record_name=s.AGENT_RECORD
    )
    record = s.read_record(tmp_path, s.AGENT_RECORD)
    assert record["rules"] == [[LAN, 9001]] and record["firewall"] == "ufw"
    assert (tmp_path / s.AGENT_RECORD).stat().st_mode & 0o077 == 0
    undo = s.plan_uninstall(record, hub=False, port=9001)
    assert apply_setup_plan(
        undo, yes=True, print_only=False, adding=False, record_dir=tmp_path, record_name=s.AGENT_RECORD
    )
    assert not (tmp_path / s.AGENT_RECORD).exists()  # nothing left to undo


def test_an_uninstall_that_fails_half_way_can_be_run_again(tmp_path, monkeypatch, user):
    # The drop-in is removed, then restarting Ollama fails: the record must still say "drop-in",
    # or the next uninstall would find nothing to do and leave Ollama on 11435.
    monkeypatch.setattr(s, "DROPIN", str(tmp_path / "missing.conf"))  # already deleted
    s.write_record(tmp_path, {"dropin": True, "rules": []}, s.RECORD)
    monkeypatch.setattr(s, "sudo_runner", FakeSudo(fail=("restart",)))
    plan = s.plan_uninstall(s.read_record(tmp_path, s.RECORD), hub=True, port=9000)
    with pytest.raises(s.SetupError):
        apply_setup_plan(plan, yes=True, print_only=False, adding=False, record_dir=tmp_path, record_name=s.RECORD)
    assert s.read_record(tmp_path, s.RECORD)["dropin"] is True
    retry = s.plan_uninstall(s.read_record(tmp_path, s.RECORD), hub=True, port=9000)
    assert ("systemctl", "restart", "ollama.service") in [st.argv for st in retry.steps]
    monkeypatch.setattr(s, "sudo_runner", FakeSudo())
    assert apply_setup_plan(retry, yes=True, print_only=False, adding=False, record_dir=tmp_path, record_name=s.RECORD)
    assert not (tmp_path / s.RECORD).exists()


def test_sudo_timing_out_is_a_clean_error(monkeypatch):
    def slow(*args, **kwargs):
        raise subprocess.TimeoutExpired(args, 1)

    monkeypatch.setattr(subprocess, "run", slow)
    with pytest.raises(s.SetupError, match="didn't finish"):
        s.sudo_runner(s.Step("x", ("systemctl", "daemon-reload")))
