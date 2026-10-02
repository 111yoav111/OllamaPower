"""The front door: an Ollama-compatible HTTP API on ``127.0.0.1:11434``.

Clients keep talking to "Ollama"; in Phase 1 every request is served by the
``local-ollama`` engine (the real Ollama, moved to ``127.0.0.1:11435``).

* **Inference** (``/api/chat``, ``/api/generate``, ``/api/embed``,
  ``/api/embeddings``, ``/v1/chat/completions``, ``/v1/completions``,
  ``/v1/embeddings``): the body is read (to learn the model), then the request
  waits in the priority queue for one of ``hub.max_concurrency`` slots and is
  streamed through. ``X-OllamaPower-Priority: critical|high|normal|low`` sets
  its priority (OllamaGate will set it per device later).
* **``GET /api/ps``**: Ollama's answer plus an ``ollamapower`` field per model
  (engine, nodes) that normal clients ignore.
* **Everything else** under ``/``, ``/api/*`` and ``/v1/*`` (tags, show, pull,
  create, blobs, ...) is passed straight through, request body streamed too,
  without queueing.
* **``GET /ollamapower/status``**: the hub snapshot as JSON.

Browser protections are kept: requests whose ``Host`` isn't a local name are
refused (DNS rebinding), exactly like Ollama does on its loopback port, and
``Origin`` is forwarded so Ollama's own CORS check still decides.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.requests import ClientDisconnect

from ...scheduling.request_queue import QueueFull, Ticket
from ...scheduling.task_pool import TaskPriority
from ...textsafe import safe_text
from ..runtime import ENGINE_LOCAL, Hub
from .hostcheck import LocalHostOnlyMiddleware
from .proxy import (
    ClientGone,
    GenerationTap,
    ProxyStreamResponse,
    StreamKilled,
    build_upstream_headers,
    error_body,
    race_kill,
    response_headers,
)

log = logging.getLogger("ollamapower.frontdoor")

PRIORITY_HEADER = "x-ollamapower-priority"
INFERENCE_ENDPOINTS = frozenset(
    {
        "/api/chat",
        "/api/generate",
        "/api/embed",
        "/api/embeddings",
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/embeddings",
    }
)
# Native endpoints whose final chunk carries Ollama's timings.
TIMED_ENDPOINTS = frozenset({"/api/chat", "/api/generate"})
PROXY_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]
MAX_PS_BYTES = 4 * 1024 * 1024
# Bodies bigger than this are parsed off the event loop (base64 images can be huge).
INLINE_PARSE_BYTES = 256 * 1024


class BodyTooLarge(Exception):
    pass


def error(path: str, status: int, message: str, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse(error_body(path, message), status_code=status, headers=headers)


async def read_body_limited(request: Request, limit: int) -> bytes:
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > limit:
        raise BodyTooLarge()
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise BodyTooLarge()
        chunks.append(chunk)
    return b"".join(chunks)


def requested_model(body: bytes) -> str:
    try:
        value = json.loads(body).get("model", "")
    except (ValueError, AttributeError):
        return ""
    return safe_text(value, 120) if isinstance(value, str) else ""


def annotate_ps(raw: bytes, engines: dict[str, dict[str, Any]]) -> bytes:
    """Add an ``ollamapower`` field to each model of an ``/api/ps`` answer."""
    try:
        data = json.loads(raw)
    except ValueError:
        return raw
    models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(models, list):
        return raw
    for m in models:
        if isinstance(m, dict):
            name = m.get("name") or m.get("model") or ""
            m["ollamapower"] = engines.get(name, {"engine": ENGINE_LOCAL, "nodes": []})
    return json.dumps(data).encode()


def create_app(hub: Hub, *, upstream_transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    config = hub.config

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        timeout = httpx.Timeout(
            connect=config.ollama.connect_timeout, read=config.ollama.read_timeout, write=60.0, pool=30.0
        )
        client = httpx.AsyncClient(
            base_url=config.ollama.url,
            timeout=timeout,
            transport=upstream_transport,
            limits=httpx.Limits(max_connections=128, max_keepalive_connections=16),
            trust_env=False,  # never route to Ollama through an env-configured proxy
            follow_redirects=False,
        )
        app.state.http = client
        app.state.loop = asyncio.get_running_loop()
        pollers = [asyncio.ensure_future(hub.poll_ollama(client)), asyncio.ensure_future(hub.poll_local())]
        try:
            yield
        finally:
            for poller in pollers:
                poller.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.gather(*pollers, return_exceptions=True)
            hub.kill_all_streams("hub shutting down")
            await client.aclose()

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.router.redirect_slashes = False
    app.state.hub = hub

    @app.get("/ollamapower/status", include_in_schema=False)
    async def status() -> Response:
        return JSONResponse(hub.get_snapshot())

    async def forward(request: Request) -> Response:
        method, path = request.method, request.url.path
        # Forward exactly what the client asked for: the raw (still percent-encoded) path and query.
        target = request.scope.get("raw_path", path.encode()).decode("latin-1")
        if request.url.query:
            target += "?" + request.url.query
        priority = TaskPriority.from_header(request.headers.get(PRIORITY_HEADER))
        stream = hub.register_stream(method, path, priority)
        ticket = Ticket(priority)
        http: httpx.AsyncClient = request.app.state.http
        inference = method == "POST" and path in INFERENCE_ENDPOINTS
        tap = GenerationTap() if path in TIMED_ENDPOINTS else None

        def finish(outcome: str, status: int) -> None:
            hub.gate.release(ticket)  # no-op if no slot was granted
            if tap is not None and status < 400:
                stats = tap.close()
                if stats is not None:
                    hub.record_generation(stream.model, stats)
            hub.finish_stream(stream, status, outcome)

        handed_off = False
        try:
            if inference:
                # 1. Read the body to learn the model.
                try:
                    body = await race_kill(read_body_limited(request, config.hub.max_body_bytes), stream)
                except BodyTooLarge:
                    finish("rejected", 413)
                    return error(path, 413, "request body too large")
                except ClientDisconnect:
                    finish("client_disconnect", 499)
                    return Response(status_code=499)
                if len(body) > INLINE_PARSE_BYTES:
                    stream.model = await asyncio.to_thread(requested_model, body)
                else:
                    stream.model = requested_model(body)

                # 2. Wait for a slot, highest priority first, giving up if the client leaves.
                queued_at = time.monotonic()
                try:
                    await asyncio.wait_for(
                        race_kill(hub.gate.acquire(ticket), stream, request.receive), config.hub.queue_timeout
                    )
                except QueueFull:
                    finish("rejected", 503)
                    return error(path, 503, "server busy: queue full", {"Retry-After": "5"})
                except TimeoutError:
                    finish("rejected", 503)
                    return error(path, 503, "server busy, try again shortly", {"Retry-After": "10"})
                stream.queued_ms = int((time.monotonic() - queued_at) * 1000)
                upstream_request = http.build_request(
                    method, target, content=body, headers=build_upstream_headers(request.headers)
                )
                upstream = await race_kill(http.send(upstream_request, stream=True), stream, request.receive)
            else:
                # Passthrough: stream the request body too (blob uploads can be gigabytes).
                stream.queued_ms = 0
                has_body = method not in ("GET", "HEAD", "OPTIONS")
                upstream_request = http.build_request(
                    method,
                    target,
                    content=request.stream() if has_body else None,
                    headers=build_upstream_headers(request.headers),
                )
                upstream = await race_kill(http.send(upstream_request, stream=True), stream)

            if method == "GET" and path == "/api/ps":
                try:
                    raw = await race_kill(upstream.aread(), stream, request.receive)
                finally:
                    await upstream.aclose()
                ok = upstream.status_code == 200 and len(raw) <= MAX_PS_BYTES
                content = annotate_ps(raw, {}) if ok else raw
                stream.bytes_out = len(content)
                finish("complete", upstream.status_code)
                headers = response_headers(upstream)
                headers.pop("content-type", None)
                media = upstream.headers.get("content-type", "application/json")
                return Response(content, status_code=upstream.status_code, media_type=media, headers=headers)

            handed_off = True
            return ProxyStreamResponse(upstream, stream, finish, tap=tap)
        except httpx.HTTPError as exc:
            log.warning("Ollama unavailable for %s %s: %s", method, path, type(exc).__name__)
            finish("upstream_error", 502)
            return error(path, 502, f"Ollama is not reachable at {config.ollama.url}")
        except ClientDisconnect:
            finish("client_disconnect", 499)
            return Response(status_code=499)
        except StreamKilled as exc:
            finish("killed", 503)
            return error(path, 503, f"request terminated: {exc.reason}")
        except ClientGone:
            finish("client_disconnect", 499)
            return Response(status_code=499)
        finally:
            # Any other exit (e.g. cancellation) must not leave the stream registered or the slot held.
            if not handed_off and stream.id in hub.streams:
                finish("error", 500)

    for prefix in ("/api/{rest:path}", "/v1/{rest:path}"):
        app.add_api_route(prefix, forward, methods=PROXY_METHODS, include_in_schema=False)
    app.add_api_route("/", forward, methods=["GET", "HEAD"], include_in_schema=False)

    @app.exception_handler(404)
    async def not_found(request: Request, exc: Exception) -> Response:
        return error(request.url.path, 404, "not found")

    app.add_middleware(LocalHostOnlyMiddleware)
    return app
