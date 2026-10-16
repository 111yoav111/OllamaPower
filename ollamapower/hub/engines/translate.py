"""Ollama's native API on top of llama-server's OpenAI API (split mode).

Clients keep calling ``/api/chat`` and ``/api/generate``; when the model runs
in split mode the request is rewritten for llama-server's
``/v1/chat/completions`` and the answer is turned back into Ollama's NDJSON
(or one JSON object for ``"stream": false``), including the final ``done``
object with Ollama's counts and durations, so tools that read them (and the
hub's own speed tap) keep working.

What doesn't translate is refused with a clear error rather than half-done:
images (the vision projector isn't loaded in split mode yet) and per-request
context sizes (split mode loads the model once with a fixed context).
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import string
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

import httpx

from ...ollama.blobs import ModelDefaults

log = logging.getLogger("ollamapower.split")

# Ollama option -> llama-server field (llama-server accepts its native sampling fields on /v1 too).
OPTION_MAP = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "min_p": "min_p",
    "typical_p": "typical_p",
    "seed": "seed",
    "stop": "stop",
    "repeat_penalty": "repeat_penalty",
    "repeat_last_n": "repeat_last_n",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
    "mirostat": "mirostat",
    "mirostat_tau": "mirostat_tau",
    "mirostat_eta": "mirostat_eta",
}
THINK_LEVELS = frozenset({"high", "medium", "low"})  # llama-server's reasoning_effort takes the same words
FINISH = {"stop": "stop", "length": "length", "tool_calls": "stop", "content_filter": "stop"}
NO_IMAGES = (
    "images are not supported in split mode yet (the vision part of the model isn't loaded); "
    "remove the split plan to use this model with images"
)
MAX_TOP_LOGPROBS = 20  # Ollama's limit
ID_CHARS = string.ascii_lowercase + string.digits
MAX_PENDING = 16 * 1024 * 1024  # an SSE line longer than this is dropped


class UnsupportedRequest(ValueError):
    """The request uses something split mode can't do; answered with 400."""


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sampling(options: dict[str, Any]) -> dict[str, Any]:
    """Ollama options -> llama-server sampling fields."""
    out: dict[str, Any] = {}
    for key, target in OPTION_MAP.items():
        if key in options and options[key] is not None:
            out[target] = options[key]
    num_predict = options.get("num_predict")
    if isinstance(num_predict, int) and not isinstance(num_predict, bool) and num_predict > 0:
        out["max_tokens"] = num_predict
    return out


def _options(body: dict[str, Any]) -> dict[str, Any]:
    options = body.get("options") if isinstance(body.get("options"), dict) else {}
    out = _sampling(options)
    fmt = body.get("format")
    if fmt == "json":
        out["response_format"] = {"type": "json_object"}
    elif isinstance(fmt, dict):
        out["response_format"] = {"type": "json_schema", "json_schema": {"name": "response", "schema": fmt}}
    out.update(_think(body.get("think")))
    top = body.get("top_logprobs")
    if top is not None and (isinstance(top, bool) or not isinstance(top, int) or not 0 <= top <= MAX_TOP_LOGPROBS):
        raise UnsupportedRequest(f"top_logprobs must be between 0 and {MAX_TOP_LOGPROBS}")
    if body.get("logprobs") is True:
        out["logprobs"] = True
        if top:
            out["top_logprobs"] = top  # llama-server, like OpenAI, only takes it together with logprobs
    return out


def _think(think: Any) -> dict[str, Any]:
    """Ollama's ``think``: true/false, or a level ("high", "medium", "low") that also enables thinking."""
    if think is None:
        return {}
    if isinstance(think, bool):
        return {"chat_template_kwargs": {"enable_thinking": think}}
    if isinstance(think, str) and think in THINK_LEVELS:
        return {"chat_template_kwargs": {"enable_thinking": True}, "reasoning_effort": think}
    # Ollama refuses anything else while decoding the request, with this message.
    shown = json.dumps(think[:100]) if isinstance(think, str) else str(think)[:100]
    raise UnsupportedRequest(f'invalid think value: {shown} (must be "high", "medium", "low", true, or false)')


def _message(m: Any, index: int) -> dict[str, Any]:
    if not isinstance(m, dict):
        raise UnsupportedRequest(f"messages[{index}] must be an object")
    if m.get("images"):
        raise UnsupportedRequest(NO_IMAGES)
    out: dict[str, Any] = {"role": m.get("role", "user"), "content": m.get("content") or ""}
    if m.get("thinking"):
        out["reasoning_content"] = m["thinking"]
    calls = m.get("tool_calls")
    if isinstance(calls, list) and calls:
        out["tool_calls"] = []
        for i, call in enumerate(calls):
            fn = call.get("function", {}) if isinstance(call, dict) else {}
            args = fn.get("arguments", {})
            out["tool_calls"].append(
                {
                    "id": str(call.get("id") or f"call_{index}_{i}"),
                    "type": "function",
                    "function": {
                        "name": fn.get("name", ""),
                        "arguments": args if isinstance(args, str) else json.dumps(args),
                    },
                }
            )
    if m.get("role") == "tool":
        if m.get("tool_call_id"):
            out["tool_call_id"] = m["tool_call_id"]
        if m.get("tool_name"):
            out["name"] = m["tool_name"]
    return out


def chat_request(body: dict[str, Any]) -> dict[str, Any]:
    """``/api/chat`` body -> ``/v1/chat/completions`` body (always streamed upstream)."""
    messages = body.get("messages") or []
    if not isinstance(messages, list):
        raise UnsupportedRequest("messages must be a list")
    out = {
        "model": body.get("model", ""),
        "messages": [_message(m, i) for i, m in enumerate(messages)],
        "stream": True,
        "stream_options": {"include_usage": True},
        **_options(body),
    }
    if isinstance(body.get("tools"), list) and body["tools"]:
        out["tools"] = body["tools"]
    return out


def generate_request(body: dict[str, Any]) -> dict[str, Any]:
    """``/api/generate`` body -> a chat request (Ollama applies the chat template to the prompt too)."""
    if body.get("images"):
        raise UnsupportedRequest(NO_IMAGES)
    if body.get("raw"):
        raise UnsupportedRequest("raw prompts are not supported in split mode; use /api/chat or /v1/completions")
    if body.get("suffix"):
        raise UnsupportedRequest("suffix (fill-in-the-middle) is not supported in split mode")
    context = body.get("context")
    if context is not None and (
        not isinstance(context, list) or not all(isinstance(t, int) and not isinstance(t, bool) for t in context)
    ):
        raise UnsupportedRequest("context must be a list of token ids")
    messages = []
    if body.get("system"):
        messages.append({"role": "system", "content": body["system"]})
    messages.append({"role": "user", "content": body.get("prompt") or ""})
    return chat_request({**body, "messages": messages})


def apply_model_defaults(out: dict[str, Any], body: dict[str, Any], chat: bool, defaults: ModelDefaults) -> None:
    """Apply the Modelfile's SYSTEM, PARAMETER and MESSAGE to a translated request, the way Ollama does.

    ``out`` is what ``chat_request``/``generate_request`` made of ``body``.
    Parameters are defaults under the request's own options; the model's
    messages go before the request's; its system prompt is used when the
    request brings none (``/api/chat``: its first message isn't a system
    message; ``/api/generate``: no ``system``). ``/api/generate`` with a
    ``context`` leaves the model's messages out: they are in the context.
    """
    if not defaults:
        return
    requested = body.get("options") if isinstance(body.get("options"), dict) else {}
    model_options = {k: v for k, v in defaults.options.items() if requested.get(k) is None}
    for key, value in _sampling(model_options).items():
        out.setdefault(key, value)
    model_messages = [_message(m, i) for i, m in enumerate(defaults.messages)]
    system = [{"role": "system", "content": defaults.system}] if defaults.system else []
    messages = out["messages"]
    if chat:
        first = body["messages"][0] if body.get("messages") else None
        own_system = isinstance(first, dict) and first.get("role") == "system"
        out["messages"] = [*([] if own_system else system), *model_messages, *messages]
        return
    head, prompt = messages[:-1], messages[-1:]  # generate: [its own system], user prompt
    out["messages"] = [*(head or system), *([] if body.get("context") else model_messages), *prompt]


def context_overflow(raw: bytes) -> int | None:
    """llama-server's "prompt longer than the context" error -> the context size (0 if not said); else None."""
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    err = data.get("error") if isinstance(data, dict) else None
    if not isinstance(err, dict) or err.get("type") != "exceed_context_size_error":
        return None
    n_ctx = err.get("n_ctx")
    return n_ctx if isinstance(n_ctx, int) and not isinstance(n_ctx, bool) and n_ctx > 0 else 0


async def _prompt_tokens(client: httpx.AsyncClient, body: dict[str, Any]) -> int | None:
    """How many tokens llama-server makes of ``body``'s messages (None: it couldn't render them)."""
    prompt = await _render(client, body)
    if prompt is None:
        return None
    tokenize = {"content": prompt, "add_special": True, "parse_special": True}  # as llama-server reads prompts
    tokens = _field(await _post(client, "/tokenize", tokenize), "tokens")
    return len(tokens) if isinstance(tokens, list) else None


async def _render(client: httpx.AsyncClient, body: dict[str, Any]) -> str | None:
    """The prompt llama-server's chat template makes of a ``/v1/chat/completions`` body."""
    probe = {k: v for k, v in body.items() if k not in ("stream", "stream_options", "return_progress")}
    prompt = _field(await _post(client, "/apply-template", probe), "prompt")
    return prompt if isinstance(prompt, str) else None


async def _post(client: httpx.AsyncClient, path: str, body: dict[str, Any]) -> httpx.Response:
    """POST to one of llama-server's side endpoints (template, tokenize: no side effects).

    Tried twice: a kept-alive connection llama-server has meanwhile closed
    fails on first use, and these requests are safe to repeat.
    """
    try:
        return await client.post(path, json=body)
    except httpx.TimeoutException:
        raise
    except httpx.TransportError:
        return await client.post(path, json=body)


def _field(r: httpx.Response, key: str) -> Any:
    """``key`` of a llama-server JSON answer; None if it failed or isn't readable."""
    if r.status_code != 200:
        return None
    try:
        data = r.json()
    except ValueError:
        return None
    return data.get(key) if isinstance(data, dict) else None


async def fit_messages(client: httpx.AsyncClient, body: dict[str, Any], n_ctx: int) -> list[dict[str, Any]] | None:
    """The longest tail of ``body``'s messages that fits in ``n_ctx`` tokens, like Ollama's chat truncation.

    The oldest messages are dropped first; system messages and the last
    message are always kept. None if even that doesn't fit. Only called
    after llama-server refused the whole prompt, so the extra requests
    (render and count, a binary search) never cost a normal request anything.
    """
    messages = body["messages"]
    last = len(messages) - 1
    if last < 1:
        return None

    def keep_from(i: int) -> list[dict[str, Any]]:
        return [m for m in messages[:i] if m.get("role") == "system"] + messages[i:]

    async def fits(i: int) -> bool:
        n = await _prompt_tokens(client, {**body, "messages": keep_from(i)})
        return n is not None and n < n_ctx  # llama-server needs room for at least one new token

    if not await fits(last):
        return None
    lo, hi = 1, last
    while lo < hi:
        mid = (lo + hi) // 2
        if await fits(mid):
            hi = mid
        else:
            lo = mid + 1
    return keep_from(lo)


async def context_prompt(client: httpx.AsyncClient, chat: dict[str, Any], context: list[int]) -> str | None:
    """``/api/generate`` with a ``context``: the earlier tokens as text, then the templated prompt (as Ollama does).

    Both are asked of llama-server at the same time, so this costs one
    round trip, and only for requests that send a context.
    """
    detok = _post(client, "/detokenize", {"tokens": context})
    text, prompt = await asyncio.gather(detok, _render(client, chat))
    earlier = _field(text, "content")
    if not isinstance(earlier, str) or prompt is None:
        return None
    return earlier + prompt


def completion_request(chat: dict[str, Any], prompt: str) -> dict[str, Any]:
    """A translated chat request -> a ``/v1/completions`` request for a prompt that is already templated."""
    chat_only = ("messages", "tools", "chat_template_kwargs", "reasoning_effort", "response_format", "top_logprobs")
    out = {k: v for k, v in chat.items() if k not in chat_only}
    out["prompt"] = prompt
    fmt = chat.get("response_format") or {}
    if fmt.get("type") == "json_object":
        out["json_schema"] = {"type": "object"}
    elif fmt.get("type") == "json_schema":
        out["json_schema"] = fmt["json_schema"]["schema"]
    if chat.get("logprobs"):
        out["logprobs"] = chat.get("top_logprobs") or 1  # /v1/completions takes how many alternatives to list
    return out


async def output_context(
    client: httpx.AsyncClient, chat: dict[str, Any], prompt: str | None, response: str
) -> list[int] | None:
    """The ``context`` Ollama's ``/api/generate`` ends with: the tokens of the templated prompt and the answer.

    No BOS token is added: the next request's prompt gets its own when
    llama-server reads it, after this context.
    """
    if prompt is None:
        prompt = await _render(client, chat)
        if prompt is None:
            return None
    tokenize = {"content": prompt + response, "add_special": False, "parse_special": True}
    tokens = _field(await _post(client, "/tokenize", tokenize), "tokens")
    if not isinstance(tokens, list) or not all(isinstance(t, int) for t in tokens):
        return None
    return tokens


def is_load_request(path: str, body: dict[str, Any]) -> bool:
    """An empty request: Ollama's way to load (or, with keep_alive 0, unload) a model."""
    if path == "/api/chat":
        return not body.get("messages")
    return not body.get("prompt") and not body.get("images")


def load_reply(path: str, model: str, unload: bool) -> dict[str, Any]:
    reply: dict[str, Any] = {"model": model, "created_at": _now(), "done": True}
    reply["done_reason"] = "unload" if unload else "load"
    if path == "/api/chat":
        reply["message"] = {"role": "assistant", "content": ""}
    else:
        reply["response"] = ""
    return reply


class OllamaStream:
    """Turns llama-server's SSE chat stream into Ollama's answer, chunk by chunk.

    ``feed`` takes raw upstream bytes and returns bytes for the client;
    ``finish`` returns the rest (the final ``done`` object, or the whole
    answer when the client asked for ``"stream": false``). Content is passed
    on, never stored, except while assembling a non-streamed answer or the
    ``context`` of an ``/api/generate`` answer: ``context`` turns the answer
    into those tokens, and ``settle`` (awaited before ``finish``) calls it.
    """

    def __init__(
        self,
        model: str,
        path: str,
        stream: bool,
        status: int = 200,
        queued_ns: int = 0,
        top_logprobs: int = MAX_TOP_LOGPROBS,
        context: Callable[[str], Awaitable[list[int] | None]] | None = None,
        began: int | None = None,
        load_ns: int = 0,
    ) -> None:
        self.model = model
        self.key = "message" if path == "/api/chat" else "response"
        self.stream = stream
        self.status = status
        # Like Ollama: total_duration counts the load and the prompt, not just the streaming.
        self.started = time.monotonic_ns() if began is None else began
        self.queued_ns = queued_ns
        self.load_ns = load_ns
        self._pending = bytearray()
        self._content: list[str] = []
        self._thinking: list[str] = []
        self._logprobs: list[dict[str, Any]] = []
        self._tools: dict[int, dict[str, str]] = {}
        self._finish = "stop"
        self._timings: dict[str, Any] = {}
        self._usage: dict[str, Any] = {}
        self._error: str | None = None
        self._finished = False  # llama-server said the answer is complete (finish_reason or [DONE])
        self.truncated = False  # the answer ended without that: llama-server died mid-answer
        self._raw: list[bytes] = []  # error bodies only
        self.top_logprobs = top_logprobs
        self._context_fn = context
        self._context: list[int] | None = None
        self._tail = b""  # output of the last, unterminated line (read by settle, sent by finish)

    @property
    def content_type(self) -> str:
        if self.status != 200 or not self.stream:
            return "application/json; charset=utf-8"
        return "application/x-ndjson"

    def feed(self, chunk: bytes) -> bytes:
        if self.status != 200:
            if sum(len(r) for r in self._raw) < 1024 * 1024:
                self._raw.append(chunk)
            return b""
        self._pending += chunk
        out = bytearray()
        while (end := self._pending.find(b"\n")) != -1:
            line = bytes(self._pending[:end]).strip()
            del self._pending[: end + 1]
            out += self._line(line)
        if len(self._pending) > MAX_PENDING:
            self._pending.clear()
        return bytes(out)

    def _line(self, line: bytes) -> bytes:
        if not line.startswith(b"data:"):
            return b""
        data = line[5:].strip()
        if data == b"[DONE]":
            self._finished = True
            return b""
        if not data:
            return b""
        try:
            event = json.loads(data)
        except ValueError:
            return b""
        if not isinstance(event, dict):
            return b""
        if "error" in event:
            err = event["error"]
            self._error = str(err.get("message") if isinstance(err, dict) else err)
            return b""
        if isinstance(event.get("timings"), dict):
            self._timings = event["timings"]
        if isinstance(event.get("usage"), dict):
            self._usage = event["usage"]
        choices = event.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            return b""
        choice = choices[0]
        if choice.get("finish_reason"):
            self._finished = True
            self._finish = FINISH.get(str(choice["finish_reason"]), "stop")
        delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
        for call in delta.get("tool_calls") or []:
            if isinstance(call, dict):
                slot = self._tools.setdefault(_int(call.get("index")), {"id": "", "name": "", "arguments": ""})
                if call.get("id") and not slot["id"]:
                    slot["id"] = str(call["id"])[:128]
                fn = call.get("function") if isinstance(call.get("function"), dict) else {}
                slot["name"] += str(fn.get("name") or "")
                slot["arguments"] += str(fn.get("arguments") or "")
        content = delta.get("content") or choice.get("text") or ""  # "text": /v1/completions
        thinking = delta.get("reasoning_content") or ""
        logprobs = _logprobs(choice.get("logprobs"), self.top_logprobs)
        if not content and not thinking and not logprobs:
            return b""
        if not self.stream:
            self._content.append(content)
            self._thinking.append(thinking)
            self._logprobs.extend(logprobs)
            return b""
        if self._context_fn is not None and content:
            self._content.append(content)
        piece = self._piece(content, thinking, done=False)
        if logprobs:
            piece["logprobs"] = logprobs
        return self._dump(piece)

    def _piece(self, content: str, thinking: str, done: bool) -> dict[str, Any]:
        obj: dict[str, Any] = {"model": self.model, "created_at": _now()}
        if self.key == "message":
            msg: dict[str, Any] = {"role": "assistant", "content": content}
            if thinking:
                msg["thinking"] = thinking
            obj["message"] = msg
        else:
            obj["response"] = content
            if thinking:
                obj["thinking"] = thinking
        obj["done"] = done
        return obj

    def _tool_calls(self) -> list[dict[str, Any]]:
        """The finished tool calls, shaped like Ollama's: an id, the call's index, arguments as an object."""
        calls = []
        for index, (_, slot) in enumerate(sorted(self._tools.items())):
            try:
                args = json.loads(slot["arguments"]) if slot["arguments"].strip() else {}
            except ValueError:
                args = {}
            if not isinstance(args, dict):
                args = {}  # Ollama's arguments are always an object; it never passes on a broken one
            calls.append(
                {
                    "id": slot["id"] or _call_id(),
                    "function": {"index": index, "name": slot["name"], "arguments": args},
                }
            )
        return calls

    def _final(self) -> dict[str, Any]:
        t = self._timings
        prompt_n = t.get("prompt_n", self._usage.get("prompt_tokens", 0))
        gen_n = t.get("predicted_n", self._usage.get("completion_tokens", 0))
        context = {"context": self._context} if self._context is not None else {}
        return {
            "done_reason": self._finish,
            **context,
            "total_duration": time.monotonic_ns() - self.started + self.queued_ns,
            "load_duration": self.load_ns,
            "prompt_eval_count": int(prompt_n or 0),
            "prompt_eval_duration": int(float(t.get("prompt_ms", 0) or 0) * 1e6),
            "eval_count": int(gen_n or 0),
            "eval_duration": int(float(t.get("predicted_ms", 0) or 0) * 1e6),
        }

    @staticmethod
    def _dump(obj: dict[str, Any]) -> bytes:
        return json.dumps(obj, ensure_ascii=False).encode() + b"\n"

    def _read_tail(self) -> None:
        if self._pending:
            self._tail += self._line(bytes(self._pending).strip())
            self._pending.clear()

    async def settle(self) -> None:
        """Ask for the answer's ``context`` tokens, once the answer is complete."""
        if self._context_fn is None or self.status != 200:
            return
        self._read_tail()
        if self._error is not None or not self._finished:
            return
        try:
            self._context = await self._context_fn("".join(self._content))
        except httpx.HTTPError as exc:
            log.warning("split: could not tokenize the answer's context: %s", type(exc).__name__)

    def finish(self) -> bytes:
        if self.status != 200:
            return self._dump({"error": _upstream_error(b"".join(self._raw), self.status)})
        self._read_tail()
        tail, self._tail = self._tail, b""
        if self._error is not None:
            return tail + self._dump({"error": self._error})
        if not self._finished:
            # Never report a cut-off answer as complete.
            self.truncated = True
            return tail + self._dump({"error": "the split engine stopped in the middle of the answer"})
        calls = self._tool_calls()
        if self.stream:
            out = bytearray(tail)
            if calls:
                piece = self._piece("", "", done=False)
                if self.key == "message":
                    piece["message"]["tool_calls"] = calls
                out += self._dump(piece)
            out += self._dump({**self._piece("", "", done=True), **self._final()})
            return bytes(out)
        final = self._piece("".join(self._content), "".join(self._thinking), done=True)
        if calls and self.key == "message":
            final["message"]["tool_calls"] = calls
        if self._logprobs:
            final["logprobs"] = self._logprobs
        return json.dumps({**final, **self._final()}, ensure_ascii=False).encode()


def _token_logprob(entry: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"token": str(entry.get("token", ""))}
    try:
        out["logprob"] = float(entry.get("logprob", 0.0))
    except (TypeError, ValueError):
        out["logprob"] = 0.0
    raw = entry.get("bytes")
    if isinstance(raw, list) and raw and all(isinstance(b, int) for b in raw):
        out["bytes"] = raw
    return out


def _logprobs(value: Any, top_n: int = MAX_TOP_LOGPROBS) -> list[dict[str, Any]]:
    """llama-server's (OpenAI's) ``logprobs.content`` -> Ollama's ``logprobs`` list (``top_n`` alternatives at most)."""
    entries = value.get("content") if isinstance(value, dict) else None
    if not isinstance(entries, list):
        return []
    out = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        item = _token_logprob(entry)
        top = entry.get("top_logprobs")
        if isinstance(top, list) and top[:top_n]:
            item["top_logprobs"] = [_token_logprob(t) for t in top[:top_n] if isinstance(t, dict)]
        out.append(item)
    return out


def _int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _call_id() -> str:
    """A tool call id like Ollama's own (``call_`` and 8 random characters)."""
    return "call_" + "".join(secrets.choice(ID_CHARS) for _ in range(8))


def _upstream_error(raw: bytes, status: int) -> str:
    try:
        data = json.loads(raw)
    except ValueError:
        return f"split engine error (HTTP {status})"
    err = data.get("error") if isinstance(data, dict) else None
    if isinstance(err, dict):
        return str(err.get("message", f"split engine error (HTTP {status})"))[:2000]
    return str(err or f"split engine error (HTTP {status})")[:2000]
