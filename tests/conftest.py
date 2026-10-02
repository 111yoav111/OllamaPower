from __future__ import annotations

import asyncio
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any

import httpx
import pytest
import uvicorn

from ollamapower.config import Config
from ollamapower.eventlog import EventLog
from ollamapower.hub.frontdoor.app import create_app
from ollamapower.hub.frontdoor.server import build_server
from ollamapower.hub.runtime import Hub

from .fake_ollama import Recorder, make_fake_ollama


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


class ThreadedServer:
    """Run an ASGI app under a real uvicorn server in a background thread."""

    def __init__(self, app: Any = None, server: uvicorn.Server | None = None) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(("127.0.0.1", 0))
        self.port = self.sock.getsockname()[1]
        if server is None:
            config = uvicorn.Config(app, log_level="warning", lifespan="on")
            config.load()
            server = uvicorn.Server(config)
        self.server = server
        self.thread = threading.Thread(target=self.server.run, kwargs={"sockets": [self.sock]}, daemon=True)

    def start(self) -> ThreadedServer:
        self.thread.start()
        deadline = time.time() + 10
        while not self.server.started:
            if time.time() > deadline or not self.thread.is_alive():
                raise RuntimeError("server failed to start")
            time.sleep(0.01)
        return self

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


@dataclass
class FrontDoor:
    url: str
    hub: Hub
    app: Any
    recorder: Recorder
    server: ThreadedServer
    config: Config

    def run(self, coro: Any) -> Any:
        """Run a coroutine on the hub's event loop and return its result."""
        return asyncio.run_coroutine_threadsafe(coro, self.app.state.loop).result(timeout=5)

    def call(self, fn: Any) -> Any:
        async def run() -> Any:
            return fn()

        return self.run(run())


@pytest.fixture
def fake_ollama():
    app, rec = make_fake_ollama()
    server = ThreadedServer(app).start()
    yield server, rec
    server.stop()


@pytest.fixture
def make_config(tmp_path):
    def factory(ollama_url: str, **overrides: Any) -> Config:
        cfg = Config()
        cfg.config_dir = tmp_path
        cfg.ollama.url = ollama_url
        cfg.ollama.read_timeout = 10
        cfg.hub.port = free_port()
        for dotted, value in overrides.items():
            section, key = dotted.split("__")
            setattr(getattr(cfg, section), key, value)
        cfg.validate()
        return cfg

    return factory


@pytest.fixture
def frontdoor_factory(fake_ollama, make_config):
    servers: list[ThreadedServer] = []

    def factory(ollama_url: str | None = None, **overrides: Any) -> FrontDoor:
        upstream, rec = fake_ollama
        cfg = make_config(ollama_url or upstream.url, **overrides)
        events = EventLog(cfg.logs_dir, cfg.logging.file_max_bytes, cfg.logging.file_backups)
        hub = Hub(cfg, events=events)
        app = create_app(hub)
        hub_server = build_server(hub, app, own_logging=False)
        server = ThreadedServer(server=hub_server).start()
        servers.append(server)
        return FrontDoor(server.url, hub, app, rec, server, cfg)

    yield factory
    for s in servers:
        s.stop()
        assert not s.thread.is_alive(), "front door did not shut down"


@pytest.fixture
def frontdoor(frontdoor_factory) -> FrontDoor:
    return frontdoor_factory()


@pytest.fixture
def client():
    with httpx.Client(timeout=10, trust_env=False) as c:
        yield c
