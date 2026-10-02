"""uvicorn wiring for the front door (adapted from OllamaGate's runner)."""

from __future__ import annotations

import contextlib
import signal
import threading
from collections.abc import Iterator

import uvicorn
from fastapi import FastAPI

from ..runtime import Hub

GRACEFUL_SHUTDOWN_SECONDS = 5


class HubServer(uvicorn.Server):
    """uvicorn server that cuts active streams as soon as shutdown begins.

    uvicorn waits for open connections before running the app's lifespan
    shutdown, so long generations would otherwise block exit.
    """

    def __init__(self, config: uvicorn.Config, hub: Hub) -> None:
        super().__init__(config)
        self.hub = hub

    async def shutdown(self, sockets: list | None = None) -> None:  # type: ignore[override]
        self.hub.kill_all_streams("hub shutting down")
        await super().shutdown(sockets)

    @contextlib.contextmanager
    def capture_signals(self) -> Iterator[None]:
        """Like uvicorn's, but don't re-raise the signal afterwards, so our own
        cleanup (admin socket, state flush) still runs."""
        if threading.current_thread() is not threading.main_thread():
            yield
            return
        signals = [signal.SIGINT, signal.SIGTERM]
        if hasattr(signal, "SIGBREAK"):  # Windows Ctrl+Break
            signals.append(signal.SIGBREAK)
        original = {sig: signal.signal(sig, self.handle_exit) for sig in signals}
        try:
            yield
        finally:
            for sig, handler in original.items():
                signal.signal(sig, handler)


def build_server(hub: Hub, app: FastAPI, own_logging: bool = True) -> HubServer:
    config = hub.config
    uv_config = uvicorn.Config(
        app,
        host=config.hub.host,
        port=config.hub.port,
        proxy_headers=False,
        server_header=False,
        date_header=False,
        access_log=False,  # requests.jsonl records metadata; never content
        log_level="warning",
        log_config=uvicorn.config.LOGGING_CONFIG if own_logging else None,
        timeout_keep_alive=30,
        limit_concurrency=config.hub.max_connections,
        h11_max_incomplete_event_size=16 * 1024,
        timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_SECONDS,
    )
    uv_config.load()
    return HubServer(uv_config, hub)
