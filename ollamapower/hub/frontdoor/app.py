"""The front door: an Ollama-compatible HTTP API on ``127.0.0.1:11434``.

Clients keep talking to "Ollama". Requests are served by the ``local-ollama``
engine (the real Ollama, moved to ``127.0.0.1:11435``), except chat and
generate requests for a model the owner planned for split mode: those go to
the split engine (llama-server with helper GPUs, see ``engines/split.py``),
translated to and from Ollama's API. If split mode can't start, Ollama serves
the request as before.

* **Inference** (``/api/chat``, ``/api/generate``, ``/api/embed``,
  ``/api/embeddings``, ``/v1/chat/completions``, ``/v1/completions``,
  ``/v1/embeddings``): the body is read (to learn the model), then the request
  waits in the priority queue for one of ``hub.max_concurrency`` slots and is
  streamed through. ``X-OllamaPower-Priority: critical|high|normal|low`` sets
  its priority (OllamaGate will set it per device later).
* **``GET /api/ps``**: Ollama's answer plus the running split model, with an
  ``ollamapower`` field per model (engine, nodes) that normal clients ignore.
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
import hmac
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

from ...aio import reap, wait_for
from ...config import is_loopback_host
from ...scheduling.request_queue import QueueFull, Ticket
from ...scheduling.task_pool import TaskPriority
from ...textsafe import safe_text
from ..benchmark import BENCH_HEADER, CTX_HEADER, ENGINE_HEADER, HELPERS_HEADER, SPLIT_VARIANTS, VARIANT_HEADER
from ..engines.split import ENGINE_SPLIT, SplitUnavailable
from ..engines.translate import (
    OllamaStream,
    UnsupportedRequest,
    apply_model_defaults,
    chat_request,
    completion_request,
    context_overflow,
    context_prompt,
    fit_messages,
    generate_request,
    is_load_request,
    load_reply,
    output_context,
)
from ..plans import SplitSpec
from ..runtime import ENGINE_LOCAL, Hub
from .hostcheck import LocalHostOnlyMiddleware, allowed_origins, origin_allowed
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
        "/v1/messages",  # Anthropic API (e.g. Claude Code through Ollama)
        "/v1/responses",
        "/v1/responses/compact",
    }
)
# Native endpoints whose final chunk carries Ollama's timings.
TIMED_ENDPOINTS = frozenset({"/api/chat", "/api/generate"})
# What split mode serves; embeddings always go to Ollama.
SPLIT_ENDPOINTS = frozenset({"/api/chat", "/api/generate", "/v1/chat/completions", "/v1/completions"})
PROXY_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]
MAX_PS_BYTES = 4 * 1024 * 1024
MAX_ERROR_BYTES = 1024 * 1024  # a llama-server error answer read whole (to see if the prompt was too long)
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


def annotate_ps(raw: bytes, extra: list[dict[str, Any]] | None = None) -> bytes:
    """Add an ``ollamapower`` field to each model of an ``/api/ps`` answer, plus ``extra`` models."""
    try:
        data = json.loads(raw)
    except ValueError:
        return raw
    models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(models, list):
        return raw
    for m in models:
        if isinstance(m, dict):
            m["ollamapower"] = {"engine": ENGINE_LOCAL, "nodes": []}
    models.extend(extra or [])
    return json.dumps(data).encode()


SPLIT_STOPPED = "split mode stopped"


def bench_spec(headers: Any) -> SplitSpec:
    """The split session a benchmark request asks for (only honoured with the hub's bench secret)."""
    try:
        ctx = int(headers.get(CTX_HEADER, ""))
    except ValueError:
        ctx = 0
    helpers = tuple(h for h in headers.get(HELPERS_HEADER, "").split(",") if h)
    variant = headers.get(VARIANT_HEADER, "")
    return SplitSpec(
        ctx=ctx if 256 <= ctx <= 1 << 20 else 4096,
        helpers=helpers or None,
        variant=variant if variant in SPLIT_VARIANTS else "",
    )


def keep_alive_zero(value: Any) -> bool:
    """Ollama's "unload now": ``keep_alive`` of 0, "0", "0s", "0m", ..."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value == 0
    return isinstance(value, str) and value.strip().rstrip("smh") in ("0", "0.0")


async def read_rest(chunks: AsyncIterator[bytes], limit: int = MAX_ERROR_BYTES) -> bytes:
    """The rest of a (small) error answer; anything past ``limit`` is dropped."""
    out = bytearray()
    async for chunk in chunks:
        if len(out) < limit:
            out += chunk[: limit - len(out)]
    return bytes(out)


async def no_more() -> AsyncIterator[bytes]:
    return
    yield b""  # an async generator that yields nothing


def cors(origin: str | None) -> dict[str, str] | None:
    """The CORS headers Ollama would add for an allowed ``origin`` (split-mode answers)."""
    return {"access-control-allow-origin": origin, "vary": "Origin"} if origin else None


def create_app(hub: Hub, *, upstream_transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    config = hub.config
    origins = allowed_origins()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        timeout = httpx.Timeout(
            connect=config.ollama.connect_timeout, read=config.ollama.read_timeout or None, write=60.0, pool=30.0
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
        hub.split.http = client
        pollers = [
            asyncio.ensure_future(hub.poll_ollama(client)),
            asyncio.ensure_future(hub.poll_local()),
            asyncio.ensure_future(hub.split.run()),
            asyncio.ensure_future(hub.plans.run()),
        ]
        try:
            yield
        finally:
            await reap(*pollers)
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
        bench = hmac.compare_digest(request.headers.get(BENCH_HEADER, "").encode(), hub.bench_secret.encode())
        forced = request.headers.get(ENGINE_HEADER) if bench else None
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

        async def serve_split(body: bytes, spec: SplitSpec, strict: bool, origin: str | None) -> Response | None:
            """Serve from the split engine. None means "let Ollama serve it" (never for a benchmark: ``strict``)."""

            def unavailable(reason: str) -> Response | None:
                if not strict:
                    return None
                finish("rejected", 503)
                return error(path, 503, f"split mode unavailable: {reason}")

            try:
                parsed = (
                    json.loads(body) if len(body) <= INLINE_PARSE_BYTES else await asyncio.to_thread(json.loads, body)
                )
            except ValueError:
                return None  # Ollama answers malformed requests in its own words
            if not isinstance(parsed, dict):
                return None
            native = path.startswith("/api/")
            if native and is_load_request(path, parsed):
                unload = keep_alive_zero(parsed.get("keep_alive"))
                if unload:
                    await hub.split.stop(stream.model, "unloaded by a client")
                else:
                    try:
                        session = await race_kill(hub.split.acquire(stream.model, spec), stream, request.receive)
                    except SplitUnavailable as exc:
                        return unavailable(str(exc))
                    hub.split.release(session)
                finish("complete", 200)
                return JSONResponse(load_reply(path, str(parsed.get("model", "")), unload), headers=cors(origin))
            try:
                translated = (
                    chat_request(parsed) if path == "/api/chat" else generate_request(parsed) if native else None
                )
            except UnsupportedRequest as exc:
                finish("rejected", 400)
                return error(path, 400, str(exc), cors(origin))
            try:
                session = await race_kill(hub.split.acquire(stream.model, spec), stream, request.receive)
            except SplitUnavailable as exc:
                return unavailable(str(exc))
            stream.engine = ENGINE_SPLIT
            if translated is not None and session.files is not None:
                # Ollama applies the Modelfile's SYSTEM/PARAMETER/MESSAGE; llama-server knows nothing of them.
                apply_model_defaults(translated, parsed, path == "/api/chat", session.files.defaults)
            released = False
            has_slot = False

            def release() -> None:
                nonlocal released, has_slot
                if has_slot:
                    has_slot = False
                    session.slot.release()
                if not released:
                    released = True
                    hub.split.release(session)

            def done(outcome: str, status: int) -> None:
                release()
                cut_off = transform is not None and transform.truncated
                if (outcome == "upstream_error" or cut_off) and not strict:
                    # llama-server stopped answering mid-stream (stalled past split.stall_timeout, or died).
                    hub.split.report_failure(session, "llama-server stopped answering mid-answer")
                elif outcome == "complete" and status < 400 and not cut_off:
                    hub.split.note_success(session)  # resets the failure backoff (once per session)
                finish(outcome, status)

            upstream: httpx.Response | None = None
            chunks: AsyncIterator[bytes] | None = None
            first = b""
            endpoint = "/v1/chat/completions"
            chat_body = translated  # what the chat template is applied to (for /api/generate's context)
            prompt_text: str | None = None  # the whole prompt, when it is built here (generate with a context)
            context = parsed.get("context") if path == "/api/generate" else None

            async def start(first_byte: float, read: float | None) -> None:
                """Send the request; read the first bytes of the answer."""
                nonlocal upstream, chunks, first
                assert session.client is not None
                upstream_request = session.client.build_request(
                    "POST",
                    endpoint if native else target,
                    content=json.dumps(translated).encode() if native else body,
                    headers={"content-type": "application/json", "accept-encoding": "identity"},
                    timeout=httpx.Timeout(connect=5.0, read=read, write=60.0, pool=30.0),
                )
                upstream = await race_kill(session.client.send(upstream_request, stream=True), stream, request.receive)
                chunks = upstream.aiter_raw()
                try:
                    first = await race_kill(wait_for(anext(chunks), first_byte), stream, request.receive)
                except StopAsyncIteration:
                    first = b""

            async def truncate_and_retry(first_byte: float, read: float | None) -> None:
                """llama-server refuses a prompt longer than the context; Ollama drops the oldest messages."""
                nonlocal upstream, chunks, first
                assert upstream is not None and chunks is not None and translated is not None
                raw = first + await race_kill(read_rest(chunks), stream, request.receive)
                first, chunks = raw, no_more()  # unless we retry, this error goes to the client as before
                n_ctx = context_overflow(raw)
                if n_ctx is None:
                    return
                assert session.client is not None
                fitted = await race_kill(
                    fit_messages(session.client, translated, n_ctx or session.ctx), stream, request.receive
                )
                if fitted is None:
                    return
                log.info(
                    "split: %d of %d messages dropped to fit the context",
                    len(translated["messages"]) - len(fitted),
                    len(translated["messages"]),
                )
                translated["messages"] = fitted
                await upstream.aclose()
                upstream = None
                await start(first_byte, read)

            # Send the request and wait for the first bytes of the answer. Until then nothing has
            # reached the client, so if split mode breaks now, Ollama answers instead - transparently.
            try:
                # One request at a time per llama-server slot; waiting here isn't llama-server's time.
                await race_kill(session.slot.acquire(), stream, request.receive)
                has_slot = True
                assert session.client is not None
                if native:
                    # Progress chunks while the prompt is processed: a long prompt is alive, not stalled.
                    translated["return_progress"] = True
                    first_byte, read = config.split.stall_timeout, config.split.stall_timeout
                    if context:
                        # Ollama puts the earlier tokens, as text, before the templated prompt.
                        prompt_text = await race_kill(
                            context_prompt(session.client, translated, context), stream, request.receive
                        )
                        if prompt_text is None:
                            release()
                            finish("rejected", 400)
                            return error(path, 400, "split mode could not read the request's context", cors(origin))
                        translated = completion_request(translated, prompt_text)
                        endpoint = "/v1/completions"
                else:
                    # /v1 passes through as is: no progress, and "stream": false sends nothing until done.
                    first_byte, read = config.split.load_timeout, None
                await start(first_byte, read)
                if path == "/api/chat" and upstream.status_code == 400 and parsed.get("truncate") is not False:
                    await truncate_and_retry(first_byte, read)
            except (httpx.HTTPError, TimeoutError, StreamKilled) as exc:
                if upstream is not None:
                    await upstream.aclose()
                release()
                if isinstance(exc, StreamKilled) and not exc.reason.startswith(SPLIT_STOPPED):
                    raise  # e.g. the hub is shutting down: don't retry anywhere
                reason = (
                    exc.reason
                    if isinstance(exc, StreamKilled)
                    else f"no answer from llama-server ({type(exc).__name__})"
                )
                if not strict:
                    stopping = hub.split.report_failure(session, reason)
                    if stopping is not None:
                        # Let llama-server give the GPU back before Ollama loads the model.
                        with contextlib.suppress(Exception):
                            await wait_for(asyncio.shield(stopping), 30)
                    stream.killed.clear()
                    stream.kill_reason = ""
                    stream.engine = ENGINE_LOCAL
                    log.warning("split mode failed before the first token for %s; retrying on Ollama", stream.model)
                return unavailable(reason)
            except BaseException:
                if upstream is not None:
                    await upstream.aclose()
                release()
                raise
            assert upstream is not None

            async def answer_context(response: str) -> list[int] | None:
                assert session.client is not None and chat_body is not None
                return await output_context(session.client, chat_body, prompt_text, response)

            transform = (
                OllamaStream(
                    str(parsed.get("model", "")),
                    path,
                    parsed.get("stream", True) is not False,
                    upstream.status_code,
                    (stream.queued_ms or 0) * 1_000_000,
                    top_logprobs=parsed.get("top_logprobs") or 0,
                    context=answer_context if path == "/api/generate" else None,
                )
                if native
                else None
            )
            return ProxyStreamResponse(
                upstream,
                stream,
                done,
                tap=tap if native else None,
                transform=transform,
                first=first,
                chunks=chunks,
                cors_origin=origin or "",  # "" = drop llama-server's CORS headers, add none
            )

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
                if not bench:
                    hub.plans.note_request(stream.model)  # also makes an automatic benchmark give way

                # 2. Wait for a slot, highest priority first, giving up if the client leaves.
                queued_at = time.monotonic()
                try:
                    await wait_for(
                        race_kill(hub.gate.acquire(ticket), stream, request.receive), config.hub.queue_timeout
                    )
                except QueueFull:
                    finish("rejected", 503)
                    return error(path, 503, "server busy: queue full", {"Retry-After": "5"})
                except TimeoutError:
                    finish("rejected", 503)
                    return error(path, 503, "server busy, try again shortly", {"Retry-After": "10"})
                stream.queued_ms = int((time.monotonic() - queued_at) * 1000)
                spec = None
                if path in SPLIT_ENDPOINTS:
                    if forced == "split":
                        spec = bench_spec(request.headers)
                    elif forced != "local":
                        if hub.plans.unknown(stream.model) and hub.registry.connected:
                            # First request for this model with helpers around: learn its size once
                            # (Ollama's /api/show and the GGUF header, a few ms), so a model too big
                            # for this PC goes to split mode instead of crawling from disk.
                            with contextlib.suppress(Exception):
                                await wait_for(hub.plans.facts(stream.model), 5.0)
                        spec = hub.plans.route(stream.model)
                if spec is not None:
                    origin = request.headers.get("origin")
                    if origin and not origin_allowed(origin, origins):
                        # Ollama would refuse this web page; split mode never asks Ollama, so we refuse it.
                        finish("rejected", 403)
                        return error(path, 403, "origin not allowed")
                    served = await serve_split(body, spec, strict=bench, origin=origin)
                    if served is not None:
                        handed_off = isinstance(served, ProxyStreamResponse)
                        return served
                if hub.split.holds(stream.model):
                    # Ollama serves this model now (or an endpoint split mode doesn't handle, like
                    # embeddings): llama-server must give the GPU back first.
                    await hub.split.stop(stream.model, "Ollama serves this model now", drain=True)
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
                content = annotate_ps(raw, hub.split.ps_entries()) if ok else raw
                stream.bytes_out = len(content)
                finish("complete", upstream.status_code)
                headers = response_headers(upstream)
                headers.pop("content-type", None)
                media = upstream.headers.get("content-type", "application/json")
                return Response(content, status_code=upstream.status_code, media_type=media, headers=headers)

            handed_off = True
            return ProxyStreamResponse(upstream, stream, finish, tap=tap)
        except httpx.TimeoutException as exc:
            if isinstance(exc, httpx.ConnectTimeout | httpx.PoolTimeout):
                log.warning("Ollama unavailable for %s %s: %s", method, path, type(exc).__name__)
                finish("upstream_error", 502)
                return error(path, 502, f"Ollama is not reachable at {config.ollama.url}")
            log.warning("Ollama didn't answer %s %s in time: %s", method, path, type(exc).__name__)
            finish("upstream_error", 504)
            return error(
                path, 504, f"Ollama did not answer within ollama.read_timeout ({config.ollama.read_timeout:g} s)"
            )
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

    app.add_middleware(LocalHostOnlyMiddleware, enforce=is_loopback_host(config.hub.host))
    return app
