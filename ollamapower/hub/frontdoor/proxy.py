"""Streaming forwarder to Ollama (adapted from OllamaGate's proxy).

Responses are passed through chunk by chunk, never buffered: NDJSON for
``/api/*``, SSE for ``/v1/*``, raw bytes for everything else. A stream ends as
soon as any of these happens:

* Ollama finishes the response,
* the client disconnects - the upstream request is closed, so Ollama stops
  generating, or
* the stream is killed (hub shutdown; fallbacks in later phases) - within one
  event-loop tick, even if the client has stopped reading.

``GenerationTap`` watches the bytes going past for Ollama's final ``done``
chunk to record generation speed. It never stores content.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Protocol, TypeVar

import httpx
from starlette.responses import Response
from starlette.types import Receive, Scope, Send

from ..runtime import ActiveStream, GenerationStats

log = logging.getLogger("ollamapower.proxy")

T = TypeVar("T")

# Only these request headers reach Ollama; everything else (cookies, auth,
# X-OllamaPower-*, hop-by-hop headers) is dropped. Origin and the preflight
# headers are kept so Ollama still applies its own CORS policy (OLLAMA_ORIGINS):
# without them a web page could reach Ollama through us that Ollama would refuse.
FORWARD_REQUEST_HEADERS = (
    "content-type",
    "accept",
    "origin",
    "access-control-request-method",
    "access-control-request-headers",
)
# Only these response headers go back to the client (CORS ones so browser-based
# clients keep working).
FORWARD_RESPONSE_HEADERS = (
    "content-type",
    "access-control-allow-origin",
    "access-control-allow-methods",
    "access-control-allow-headers",
    "access-control-allow-credentials",
    "access-control-expose-headers",
    "access-control-max-age",
    "vary",
)
# How long we try to deliver an in-band error before giving up on a client.
TRAILER_TIMEOUT = 1.0
# The tap only looks at lines up to this size. /api/generate's final chunk
# carries the whole token context (~7 bytes per token), so allow for 128k+ tokens.
MAX_TAP_LINE = 16 * 1024 * 1024


class Transform(Protocol):
    """Rewrites a response body on the way through (split mode's Ollama translation)."""

    content_type: str

    def feed(self, chunk: bytes) -> bytes: ...

    def finish(self) -> bytes: ...


class StreamKilled(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class ClientGone(Exception):
    pass


def response_headers(upstream: httpx.Response) -> dict[str, str]:
    return {name: upstream.headers[name] for name in FORWARD_RESPONSE_HEADERS if name in upstream.headers}


def build_upstream_headers(incoming: Any) -> dict[str, str]:
    headers = {name: incoming[name] for name in FORWARD_REQUEST_HEADERS if name in incoming}
    # An uncompressed body can be streamed through verbatim.
    headers["accept-encoding"] = "identity"
    return headers


async def wait_for_disconnect(receive: Receive) -> None:
    """Return once the client has disconnected. The request body must already be consumed."""
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            return


async def _reap(task: asyncio.Future[Any]) -> None:
    """Await a task we just cancelled, closing any response it still produced."""
    try:
        result = await task
    except asyncio.CancelledError:
        current = asyncio.current_task()
        if current is not None and current.cancelling():
            raise  # we are being cancelled ourselves; don't swallow it
    except Exception:  # noqa: BLE001 - the task's own failure is irrelevant now
        pass
    else:
        if isinstance(result, httpx.Response):
            await result.aclose()


async def race_kill(work: Awaitable[T], stream: ActiveStream, receive: Receive | None = None) -> T:
    """Run ``work`` but abort it if the stream is killed or (optionally) the client leaves."""
    work_task = asyncio.ensure_future(work)
    watchers = [asyncio.ensure_future(stream.killed.wait())]
    if receive is not None:
        watchers.append(asyncio.ensure_future(wait_for_disconnect(receive)))
    try:
        await asyncio.wait({work_task, *watchers}, return_when=asyncio.FIRST_COMPLETED)
    except BaseException:
        work_task.cancel()
        await _reap(work_task)
        raise
    finally:
        for w in watchers:
            w.cancel()
    if work_task.done():
        return work_task.result()
    work_task.cancel()
    await _reap(work_task)
    if stream.killed.is_set():
        raise StreamKilled(stream.kill_reason)
    raise ClientGone()


def error_body(path: str, message: str) -> dict[str, Any]:
    """Ollama's error shape for ``/api``, OpenAI's for ``/v1``."""
    if path.startswith("/v1/"):
        return {"error": {"message": message, "type": "ollamapower_error", "param": None, "code": None}}
    return {"error": message}


def _error_trailer(content_type: str, path: str, message: str) -> bytes:
    """An in-band error for a stream whose status line has already been sent."""
    if "text/event-stream" in content_type:
        return b"data: " + json.dumps(error_body("/v1/", message)).encode() + b"\n\n"
    if "ndjson" in content_type:
        return json.dumps(error_body(path, message)).encode() + b"\n"
    return b""


class GenerationTap:
    """Finds the ``"done": true`` object of a native-API response and reads its timings."""

    def __init__(self) -> None:
        self._buf = bytearray()
        self._overflow = False
        self.stats: GenerationStats | None = None

    def feed(self, chunk: bytes) -> None:
        start = 0
        while (end := chunk.find(b"\n", start)) != -1:
            if not self._overflow:
                self._buf += chunk[start:end]
                self._consider(bytes(self._buf))
            self._buf.clear()
            self._overflow = False
            start = end + 1
        if not self._overflow:
            self._buf += chunk[start:]
            if len(self._buf) > MAX_TAP_LINE:  # absurdly long line: skip it, don't hold it
                self._buf.clear()
                self._overflow = True

    def close(self) -> GenerationStats | None:
        if self._buf and not self._overflow:
            self._consider(bytes(self._buf))  # non-streaming replies have no trailing newline
        self._buf.clear()
        return self.stats

    def _consider(self, line: bytes) -> None:
        if b'"done"' not in line:
            return
        try:
            obj = json.loads(line)
        except ValueError:
            return
        if not isinstance(obj, dict) or obj.get("done") is not True:
            return
        stats = GenerationStats(
            prompt_tokens=_count(obj.get("prompt_eval_count")), gen_tokens=_count(obj.get("eval_count"))
        )
        prompt_ns, gen_ns = _count(obj.get("prompt_eval_duration")), _count(obj.get("eval_duration"))
        if prompt_ns:
            stats.prompt_tps = stats.prompt_tokens / (prompt_ns / 1e9)
        if gen_ns:
            stats.gen_tps = stats.gen_tokens / (gen_ns / 1e9)
        self.stats = stats


def _count(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


class ProxyStreamResponse(Response):
    """ASGI response that pipes an ``httpx`` streaming response to the client.

    Subclasses ``Response`` only so FastAPI passes it through untouched; the
    base initialiser is intentionally not used.
    """

    def __init__(
        self,
        upstream: httpx.Response,
        stream: ActiveStream,
        on_done: Callable[[str, int], None],
        tap: GenerationTap | None = None,
        transform: Transform | None = None,
        first: bytes = b"",
        chunks: AsyncIterator[bytes] | None = None,
    ) -> None:
        self.upstream = upstream
        self.stream = stream
        self.on_done = on_done
        self.tap = tap
        self.transform = transform
        self.first = first  # already read from upstream (split mode waits for it before answering)
        self.chunks = chunks
        self.background = None
        self.status_code = upstream.status_code
        self.content_type = transform.content_type if transform else upstream.headers.get("content-type", "")

    def _headers(self) -> list[tuple[bytes, bytes]]:
        headers = response_headers(self.upstream)
        if self.transform is not None:
            headers["content-type"] = self.content_type
        return [(k.encode("latin-1"), v.encode("latin-1")) for k, v in headers.items()]

    async def _send_body(self, chunk: bytes, send: Send) -> None:
        if not chunk:
            return
        if self.tap is not None:
            self.tap.feed(chunk)
        self.stream.bytes_out += len(chunk)
        await send({"type": "http.response.body", "body": chunk, "more_body": True})

    async def _pump(self, send: Send) -> str:
        """Copy the body. Returns "complete" or "upstream_error"."""
        try:
            source = self.chunks if self.chunks is not None else self.upstream.aiter_raw()
            if self.first:
                await self._send_body(self.transform.feed(self.first) if self.transform else self.first, send)
            async for chunk in source:
                await self._send_body(self.transform.feed(chunk) if self.transform else chunk, send)
            if self.transform is not None:
                await self._send_body(self.transform.finish(), send)
        except httpx.HTTPError as exc:
            log.warning("upstream stream error: %s", type(exc).__name__)
            await self._end_with_error("upstream error", send)
            return "upstream_error"
        await send({"type": "http.response.body", "body": b"", "more_body": False})
        return "complete"

    async def _end_with_error(self, message: str, send: Send) -> None:
        """Tell a streaming client why the stream ended, then end it.

        Bodies that are not line-delimited streams get no terminator, so the
        connection is aborted and the client sees a truncated response rather
        than a clean-looking but incomplete one.
        """
        trailer = _error_trailer(self.content_type, self.stream.path, message)
        if not trailer:
            return

        async def deliver() -> None:
            await send({"type": "http.response.body", "body": trailer, "more_body": True})
            await send({"type": "http.response.body", "body": b"", "more_body": False})

        try:
            await asyncio.wait_for(deliver(), TRAILER_TIMEOUT)
        except (TimeoutError, OSError):
            pass

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        outcome = "client_disconnect"
        try:
            await send({"type": "http.response.start", "status": self.status_code, "headers": self._headers()})
            pump = asyncio.ensure_future(self._pump(send))
            gone = asyncio.ensure_future(wait_for_disconnect(receive))
            kill = asyncio.ensure_future(self.stream.killed.wait())
            try:
                await asyncio.wait({pump, gone, kill}, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for task in (pump, gone, kill):
                    task.cancel()
                await asyncio.gather(pump, gone, kill, return_exceptions=True)

            if pump.done() and not pump.cancelled():
                exc = pump.exception()
                if exc is None:
                    outcome = pump.result()
                else:
                    log.error("stream pump failed: %s", type(exc).__name__)
                    outcome = "error"
            elif kill.done() and not kill.cancelled():
                outcome = "killed"
                # Stop Ollama first, then (best effort) tell the client why.
                await self.upstream.aclose()
                await self._end_with_error(f"stream terminated: {self.stream.kill_reason}", send)
        finally:
            try:
                await self.upstream.aclose()
            finally:
                self.on_done(outcome, self.status_code)
