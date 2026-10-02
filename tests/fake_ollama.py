"""A small fake Ollama for tests (extended from OllamaGate's).

It records what it receives (so tests can check headers and arrival order)
and whether streams finished or were cut short (so tests can check that a
client disconnect reaches Ollama). Request bodies can steer it:
``fake_chunks``, ``fake_delay`` (between chunks), ``fake_first_delay``
(e.g. model loading) and ``fake_crash`` (die mid-stream).
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from starlette.routing import Route

GIB = 1024**3


@dataclass
class Recorder:
    requests: list[dict[str, Any]] = field(default_factory=list)
    streams_started: int = 0
    streams_completed: int = 0
    streams_aborted: int = 0
    blob_bytes: int = 0
    ps_models: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {"name": "small:1b", "model": "small:1b", "size": 2 * GIB, "size_vram": 2 * GIB, "digest": "aa"},
            {"name": "big:70b", "model": "big:70b", "size": 40 * GIB, "size_vram": 20 * GIB, "digest": "bb"},
        ]
    )

    @property
    def last(self) -> dict[str, Any]:
        return self.requests[-1]

    def paths(self) -> list[str]:
        return [r["path"] for r in self.requests]


def make_fake_ollama() -> tuple[Starlette, Recorder]:
    rec = Recorder()

    async def record(request: Request, body: bool = True) -> dict[str, Any]:
        raw = await request.body() if body else b""
        try:
            parsed = json.loads(raw) if raw else None
        except ValueError:
            parsed = None
        entry = {
            "method": request.method,
            "path": request.url.path,
            "query": request.url.query,
            "raw_path": request.scope.get("raw_path", b"").decode("latin-1"),
            "headers": {k.lower(): v for k, v in request.headers.items()},
            "body": parsed,
        }
        rec.requests.append(entry)
        return entry

    async def stream_lines(lines: list[str], delay: float, crash: bool = False):
        rec.streams_started += 1
        finished = False
        try:
            for i, line in enumerate(lines):
                if crash and i == 1:
                    raise RuntimeError("ollama crashed")
                yield line
                await asyncio.sleep(delay)
            finished = True
        finally:
            if finished:
                rec.streams_completed += 1
            else:
                rec.streams_aborted += 1

    def final(model: str, chunks: int) -> dict[str, Any]:
        return {
            "model": model,
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 8,
            "prompt_eval_duration": 100_000_000,  # 0.1 s -> 80 tok/s
            "eval_count": chunks,
            "eval_duration": 50_000_000 * chunks,  # 20 tok/s
        }

    async def chat(request: Request):
        entry = await record(request)
        body = entry["body"] or {}
        model = body.get("model", "test")
        if model == "missing":
            return JSONResponse({"error": f"model '{model}' not found"}, status_code=404)
        chunks = int(body.get("fake_chunks", 3))
        delay = float(body.get("fake_delay", 0.0))
        await asyncio.sleep(float(body.get("fake_first_delay", 0.0)))
        key = "message" if request.url.path == "/api/chat" else "response"
        if body.get("stream") is False:
            text = "hello " * chunks
            content = {"role": "assistant", "content": text} if key == "message" else text
            return JSONResponse({**final(model, chunks), key: content})
        lines = []
        for i in range(chunks):
            content = {"role": "assistant", "content": f"tok{i} "} if key == "message" else f"tok{i} "
            lines.append(json.dumps({"model": model, key: content, "done": False}) + "\n")
        lines.append(json.dumps(final(model, chunks)) + "\n")
        return StreamingResponse(
            stream_lines(lines, delay, bool(body.get("fake_crash"))), media_type="application/x-ndjson"
        )

    async def openai_chat(request: Request):
        entry = await record(request)
        body = entry["body"] or {}
        chunks = int(body.get("fake_chunks", 3))
        delay = float(body.get("fake_delay", 0.0))
        await asyncio.sleep(float(body.get("fake_first_delay", 0.0)))
        lines = [
            "data: " + json.dumps({"choices": [{"delta": {"content": f"tok{i}"}}]}) + "\n\n" for i in range(chunks)
        ]
        lines.append("data: [DONE]\n\n")
        return StreamingResponse(stream_lines(lines, delay), media_type="text/event-stream")

    async def embed(request: Request):
        await record(request)
        return JSONResponse({"model": "e", "embeddings": [[0.1, 0.2]]})

    async def tags(request: Request):
        await record(request)
        return JSONResponse({"models": [{"name": "small:1b"}, {"name": "big:70b"}]})

    async def ps(request: Request):
        await record(request)
        return JSONResponse({"models": rec.ps_models})

    async def version(request: Request):
        await record(request)
        return JSONResponse({"version": "0.0.0-fake"}, headers={"x-upstream-secret": "leak"})

    async def pull(request: Request):
        await record(request)
        lines = [json.dumps({"status": s}) + "\n" for s in ("pulling manifest", "verifying", "success")]
        return StreamingResponse(stream_lines(lines, 0.0), media_type="application/x-ndjson")

    async def blobs(request: Request):
        entry = await record(request, body=False)
        if request.method == "HEAD":
            return Response(status_code=404)
        size = 0
        async for chunk in request.stream():
            size += len(chunk)
        rec.blob_bytes += size
        entry["size"] = size
        return Response(status_code=201)

    async def root(request: Request):
        await record(request, body=False)
        return PlainTextResponse("Ollama is running")

    async def generic(request: Request):
        await record(request)
        return JSONResponse({"status": "success"})

    routes = [
        Route("/", root, methods=["GET", "HEAD"]),
        Route("/api/chat", chat, methods=["POST"]),
        Route("/api/generate", chat, methods=["POST"]),
        Route("/v1/chat/completions", openai_chat, methods=["POST"]),
        Route("/api/embed", embed, methods=["POST"]),
        Route("/api/tags", tags, methods=["GET"]),
        Route("/api/ps", ps, methods=["GET"]),
        Route("/api/version", version, methods=["GET"]),
        Route("/api/pull", pull, methods=["POST"]),
        Route("/api/blobs/{digest}", blobs, methods=["HEAD", "POST"]),
        Route("/api/delete", generic, methods=["DELETE"]),
        Route("/api/show", generic, methods=["POST"]),
    ]
    return Starlette(routes=routes, middleware=[Middleware(FakeCors)]), rec


class FakeCors:
    """Ollama's CORS behaviour in miniature: local origins allowed, others 403."""

    ALLOWED = ("http://localhost", "http://127.0.0.1", "app://")

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        origin = dict(scope["headers"]).get(b"origin", b"").decode()
        if origin and not origin.startswith(self.ALLOWED):
            return await JSONResponse({"error": "origin not allowed"}, status_code=403)(scope, receive, send)
        cors = {"access-control-allow-origin": origin} if origin else {}
        if scope["method"] == "OPTIONS" and origin:
            cors["access-control-allow-methods"] = "GET, POST"
            return await Response(status_code=204, headers=cors)(scope, receive, send)

        async def send_with_cors(message):
            if message["type"] == "http.response.start" and cors:
                message["headers"] = [*message["headers"], *((k.encode(), v.encode()) for k, v in cors.items())]
            await send(message)

        await self.app(scope, receive, send_with_cors)
