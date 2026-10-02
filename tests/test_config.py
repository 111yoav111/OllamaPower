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
        '[hub]\nport = 80\n[ollama]\nurl = "http://127.0.0.1"\n',  # no port means 80
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
    assert (cfg.hub, cfg.ollama, cfg.logging) == (default.hub, default.ollama, default.logging)


def test_url_without_port_is_port_80_not_11434(tmp_path):
    (tmp_path / "config.toml").write_text('[ollama]\nurl = "http://127.0.0.1"\n')
    assert load_config(env={"OLLAMAPOWER_CONFIG_DIR": str(tmp_path)}).ollama.url == "http://127.0.0.1"
