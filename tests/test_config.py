from __future__ import annotations

import pytest

from ollamapower.config import Config, ConfigError, load_config


def test_defaults_are_valid_and_safe(tmp_path):
    cfg = load_config(env={"OLLAMAPOWER_CONFIG_DIR": str(tmp_path)})
    assert (cfg.hub.host, cfg.hub.port) == ("127.0.0.1", 11434)
    assert cfg.ollama.url == "http://127.0.0.1:11435"
    assert cfg.frontdoor_url == "http://127.0.0.1:11434"


def test_file_and_env_overrides(tmp_path):
    (tmp_path / "config.toml").write_text('[hub]\nmax_concurrency = 2\n[ollama]\nurl = "http://localhost:11500"\n')
    env = {
        "OLLAMAPOWER_CONFIG_DIR": str(tmp_path),
        "OLLAMAPOWER_HUB__QUEUE_TIMEOUT": "12.5",
        "OLLAMAPOWER_TOKEN": "not a config key",
    }
    cfg = load_config(env=env)
    assert cfg.hub.max_concurrency == 2 and cfg.hub.queue_timeout == 12.5
    assert cfg.ollama.url == "http://localhost:11500"


@pytest.mark.parametrize(
    "toml",
    [
        "[hub]\nnope = 1\n",
        "[nope]\nx = 1\n",
        '[hub]\nport = "x"\n',
        "[hub]\nmax_concurrency = 0\n",
        '[ollama]\nurl = "http://192.168.1.5:11435"\n',  # Ollama must stay local
        '[ollama]\nurl = "http://127.0.0.1:11434"\n',  # the front door's own port
        '[hub]\nhost = "0.0.0.0"\n[ollama]\nurl = "http://127.0.0.1:11434"\n',  # 0.0.0.0 includes loopback
        '[hub]\nport = 80\n[ollama]\nurl = "http://127.0.0.1"\n',  # no port: ambiguous (80 or 11434?)
        '[ollama]\nurl = "http://127.0.0.1"\n',
        "[hub]\nmax_concurrency = 2.7\n",  # never truncated to 2
        '[hub]\nmax_concurrency = "4"\n',  # TOML has real numbers; a string is a mistake
        '[hub]\nqueue_timeout = "30"\n',
        '[control]\nenabled = "yes"\n',
        "[control]\nallowed_networks = []\n",  # control on, but no helper could ever connect
        '[ollama]\nurl = "ftp://127.0.0.1"\n',
        '[logging]\nlevel = "loud"\n',
        "this is not toml",
    ],
)
def test_invalid_configs(tmp_path, toml):
    (tmp_path / "config.toml").write_text(toml)
    with pytest.raises(ConfigError):
        load_config(env={"OLLAMAPOWER_CONFIG_DIR": str(tmp_path)})


def test_unknown_env_section(tmp_path):
    with pytest.raises(ConfigError):
        load_config(env={"OLLAMAPOWER_CONFIG_DIR": str(tmp_path), "OLLAMAPOWER_NOPE__X": "1"})


def test_example_config_is_the_default(tmp_path):
    from pathlib import Path

    from ollamapower import config as c

    root = Path(__file__).parent.parent
    packaged = Path(c.__file__).parent / "config.example.toml"
    assert (root / "config.example.toml").read_text() == packaged.read_text()
    cfg = load_config(packaged, env={"OLLAMAPOWER_CONFIG_DIR": str(tmp_path)})
    default = Config()
    assert (cfg.hub, cfg.ollama, cfg.control, cfg.logging) == (
        default.hub,
        default.ollama,
        default.control,
        default.logging,
    )


def test_url_without_port_is_refused(tmp_path):
    """config.py read a port-less URL as port 80 while doctor probed 11434: an explicit port avoids both."""
    (tmp_path / "config.toml").write_text('[ollama]\nurl = "http://127.0.0.1"\n')
    with pytest.raises(ConfigError, match="explicit port"):
        load_config(env={"OLLAMAPOWER_CONFIG_DIR": str(tmp_path)})


def test_valid_odd_values(tmp_path):
    (tmp_path / "config.toml").write_text(
        '[logging]\nlevel = "INFO"\n[hub]\nqueue_timeout = 30\n[control]\nenabled = false\nallowed_networks = []\n'
    )
    cfg = load_config(env={"OLLAMAPOWER_CONFIG_DIR": str(tmp_path)})
    assert cfg.logging.level == "info" and cfg.hub.queue_timeout == 30.0 and not cfg.control.enabled


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("OLLAMAPOWER_HUB__MAX_CONCURRENCY", "2.5"),
        ("OLLAMAPOWER_HUB__PORT", "x"),
        ("OLLAMAPOWER_CONTROL__ENABLED", "2"),
    ],
)
def test_bad_env_values(tmp_path, name, value):
    with pytest.raises(ConfigError):
        load_config(env={"OLLAMAPOWER_CONFIG_DIR": str(tmp_path), name: value})


def test_unreadable_config_is_a_config_error(tmp_path):
    from ollamapower.agent.config import load_agent_config

    (tmp_path / "config.toml").mkdir()
    (tmp_path / "agent.toml").mkdir()
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(env={"OLLAMAPOWER_CONFIG_DIR": str(tmp_path)})
    with pytest.raises(ConfigError, match="cannot read"):
        load_agent_config(tmp_path, env={})


def test_env_overrides_for_the_other_file_are_ignored(tmp_path):
    """Hub and agent read one environment: a helper's setting must not break hub commands, and vice versa."""
    from ollamapower.agent.config import load_agent_config

    env = {
        "OLLAMAPOWER_CONFIG_DIR": str(tmp_path),
        "OLLAMAPOWER_LENDING__MAX_VRAM_GB": "8",  # agent only
        "OLLAMAPOWER_HUB__ADDRESS": "192.168.1.10",  # [hub] in agent.toml
        "OLLAMAPOWER_HUB__MAX_CONCURRENCY": "2",  # [hub] in config.toml
        "OLLAMAPOWER_LLAMACPP__AUTO_INSTALL": "false",  # agent's [llamacpp] only
        "OLLAMAPOWER_LLAMACPP__BACKEND": "cpu",  # both
        "OLLAMAPOWER_SPLIT__KEEP_ALIVE": "60",  # hub only
    }
    hub = load_config(env=env)
    assert hub.hub.max_concurrency == 2 and hub.llamacpp.backend == "cpu" and hub.split.keep_alive == 60.0
    agent = load_agent_config(tmp_path, env=env)
    assert agent.lending.max_vram_gb == 8.0 and agent.hub.address == "192.168.1.10"
    assert not agent.llamacpp.auto_install and agent.llamacpp.backend == "cpu"


@pytest.mark.parametrize("name", ["OLLAMAPOWER_HUB__NOPE", "OLLAMAPOWER_LENDINGG__MAX_VRAM_GB"])
def test_env_typos_are_still_errors(tmp_path, name):
    from ollamapower.agent.config import load_agent_config

    with pytest.raises(ConfigError):
        load_config(env={"OLLAMAPOWER_CONFIG_DIR": str(tmp_path), name: "1"})
    with pytest.raises(ConfigError):
        load_agent_config(tmp_path, env={name: "1"})


def test_saving_agent_rules_keeps_the_file_value_of_env_overrides(tmp_path):
    from ollamapower.agent.config import load_agent_config, save_agent_config

    (tmp_path / "agent.toml").write_text("[lending]\nmax_vram_gb = 4.0\n")
    cfg = load_agent_config(tmp_path, env={"OLLAMAPOWER_LENDING__MAX_VRAM_GB": "8"})
    assert cfg.lending.max_vram_gb == 8.0
    cfg.lending.allow_cpu = True
    save_agent_config(cfg)
    on_disk = load_agent_config(tmp_path, env={})
    assert on_disk.lending.max_vram_gb == 4.0 and on_disk.lending.allow_cpu


def test_agent_example_is_the_default(tmp_path):
    from dataclasses import asdict
    from pathlib import Path

    from ollamapower import config as c
    from ollamapower.agent.config import AgentConfig, load_agent_config

    root = Path(__file__).resolve().parent.parent
    packaged = Path(c.__file__).parent / "agent.example.toml"
    assert (root / "agent.example.toml").read_text() == packaged.read_text()
    loaded = load_agent_config(tmp_path, path=packaged)
    default = AgentConfig(config_dir=tmp_path)
    for section in ("hub", "agent", "lending", "llamacpp"):
        assert asdict(getattr(loaded, section)) == asdict(getattr(default, section)), section
