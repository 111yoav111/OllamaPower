"""Ollama's native API <-> llama-server's OpenAI API (split mode)."""

from __future__ import annotations

import json

import pytest

from ollamapower.hub.engines.translate import (
    OllamaStream,
    UnsupportedRequest,
    chat_request,
    generate_request,
    is_load_request,
)


def sse(*events: dict) -> bytes:
    return b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events) + b"data: [DONE]\n\n"


def delta(**d) -> dict:
    return {"choices": [{"index": 0, "delta": d, "finish_reason": None}]}


FINAL = {"choices": [], "timings": {"prompt_n": 5, "prompt_ms": 50.0, "predicted_n": 2, "predicted_ms": 40.0}}


def test_chat_request_maps_options_format_and_think():
    body = chat_request(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "options": {"temperature": 0.3, "num_predict": 50, "top_k": 20, "num_ctx": 9999, "stop": ["x"]},
            "format": {"type": "object"},
            "think": False,
            "keep_alive": "5m",
        }
    )
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
    assert body["temperature"] == 0.3 and body["max_tokens"] == 50 and body["top_k"] == 20 and body["stop"] == ["x"]
    assert "num_ctx" not in body and "keep_alive" not in body
    assert body["response_format"]["json_schema"]["schema"] == {"type": "object"}
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert chat_request({"messages": [], "format": "json"})["response_format"] == {"type": "json_object"}


def test_tool_history_is_converted():
    body = chat_request(
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": "f", "arguments": {"a": 1}}}],
                },
                {"role": "tool", "content": "42", "tool_name": "f"},
            ]
        }
    )
    call = body["messages"][0]["tool_calls"][0]
    assert call["type"] == "function" and json.loads(call["function"]["arguments"]) == {"a": 1}
    assert body["messages"][1] == {"role": "tool", "content": "42", "name": "f"}


@pytest.mark.parametrize(
    "fn,body",
    [
        (chat_request, {"messages": [{"role": "user", "content": "x", "images": ["aGk="]}]}),
        (generate_request, {"prompt": "x", "images": ["aGk="]}),
        (generate_request, {"prompt": "x", "raw": True}),
        (generate_request, {"prompt": "x", "suffix": "y"}),
        (chat_request, {"messages": "nope"}),
    ],
)
def test_unsupported_requests_are_refused(fn, body):
    with pytest.raises(UnsupportedRequest):
        fn(body)


def test_load_requests():
    assert is_load_request("/api/chat", {"model": "m"})
    assert not is_load_request("/api/chat", {"messages": [{"role": "user", "content": "x"}]})
    assert is_load_request("/api/generate", {"model": "m", "keep_alive": 0})


def test_stream_translation_across_odd_chunk_boundaries():
    raw = sse(delta(content="Hel"), delta(reasoning_content="hmm"), delta(content="lo"), FINAL)
    t = OllamaStream("m", "/api/chat", stream=True)
    out = b"".join(t.feed(raw[i : i + 7]) for i in range(0, len(raw), 7)) + t.finish()
    lines = [json.loads(x) for x in out.splitlines()]
    assert [x["message"]["content"] for x in lines] == ["Hel", "", "lo", ""]
    assert lines[1]["message"]["thinking"] == "hmm"
    final = lines[-1]
    assert final["done"] is True and final["done_reason"] == "stop"
    assert (final["prompt_eval_count"], final["prompt_eval_duration"]) == (5, 50_000_000)
    assert (final["eval_count"], final["eval_duration"]) == (2, 40_000_000)


def test_generate_non_streamed_and_length_finish():
    t = OllamaStream("m", "/api/generate", stream=False)
    body = t.feed(sse(delta(content="a"), delta(content="b"), {"choices": [{"delta": {}, "finish_reason": "length"}]}))
    assert body == b""
    obj = json.loads(t.finish())
    assert obj["response"] == "ab" and obj["done_reason"] == "length" and t.content_type.startswith("application/json")


def test_upstream_error_status_becomes_ollama_error():
    t = OllamaStream("m", "/api/chat", stream=True, status=500)
    t.feed(json.dumps({"error": {"code": 500, "message": "context too long"}}).encode())
    assert json.loads(t.finish()) == {"error": "context too long"}
    assert t.content_type.startswith("application/json")


def test_error_event_mid_stream():
    t = OllamaStream("m", "/api/chat", stream=True)
    out = t.feed(sse(delta(content="x"), {"error": {"message": "boom"}}))
    out += t.finish()
    assert json.loads(out.splitlines()[-1]) == {"error": "boom"}


def test_garbage_lines_are_ignored():
    t = OllamaStream("m", "/api/chat", stream=True)
    out = t.feed(b": keep-alive\n\ndata: {not json\n\ndata: [1,2]\n\n" + sse(delta(content="ok"), FINAL)) + t.finish()
    lines = [json.loads(x) for x in out.splitlines()]
    assert lines[0]["message"]["content"] == "ok" and lines[-1]["done"] is True


def test_an_answer_cut_off_is_never_reported_as_done():
    t = OllamaStream("m", "/api/chat", stream=True)
    raw = b"".join(b"data: " + json.dumps(delta(content=w)).encode() + b"\n\n" for w in ("a", "b"))
    out = t.feed(raw) + t.finish()  # no finish_reason, no [DONE]: the connection just ended
    lines = [json.loads(x) for x in out.splitlines()]
    assert all(not x.get("done") for x in lines) and "stopped in the middle" in lines[-1]["error"]
    assert t.truncated


@pytest.mark.parametrize(
    "think,expected",
    [
        (True, {"chat_template_kwargs": {"enable_thinking": True}}),
        (False, {"chat_template_kwargs": {"enable_thinking": False}}),
        ("high", {"chat_template_kwargs": {"enable_thinking": True}, "reasoning_effort": "high"}),
        ("medium", {"chat_template_kwargs": {"enable_thinking": True}, "reasoning_effort": "medium"}),
        ("low", {"chat_template_kwargs": {"enable_thinking": True}, "reasoning_effort": "low"}),
    ],
)
def test_think_bools_and_levels(think, expected):
    body = chat_request({"messages": [], "think": think})
    assert {k: body[k] for k in ("chat_template_kwargs", "reasoning_effort") if k in body} == expected
    assert "reasoning_effort" not in chat_request({"messages": []})


def tool_delta(index: int, fn: dict, **extra) -> dict:
    return delta(tool_calls=[{"index": index, **extra, "function": fn}])


def test_tool_calls_get_ids_indexes_and_object_arguments():
    t = OllamaStream("m", "/api/chat", stream=True)
    raw = sse(
        tool_delta(0, {"name": "a", "arguments": '{"x":'}, id="up1"),
        tool_delta(0, {"arguments": " 1}"}),
        tool_delta(1, {"name": "b", "arguments": "{broken"}),
        tool_delta(2, {"name": "c", "arguments": "[1, 2]"}),
        tool_delta(3, {"name": "d", "arguments": ""}),
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        FINAL,
    )
    lines = [json.loads(x) for x in (t.feed(raw) + t.finish()).splitlines()]
    calls = lines[0]["message"]["tool_calls"]
    assert calls[0] == {"id": "up1", "function": {"index": 0, "name": "a", "arguments": {"x": 1}}}
    assert [c["function"]["index"] for c in calls] == [0, 1, 2, 3]
    assert [c["function"]["arguments"] for c in calls[1:]] == [{}, {}, {}]  # broken JSON never reaches the client
    generated = [c["id"] for c in calls[1:]]
    assert all(i.startswith("call_") and len(i) == 13 for i in generated) and len(set(generated)) == 3
    assert lines[-1]["done"] is True and lines[-1]["done_reason"] == "stop"


@pytest.mark.parametrize("think", ["max", "true", "", 1])
def test_unknown_think_values_are_refused_like_ollama(think):
    with pytest.raises(UnsupportedRequest, match="invalid think value"):
        generate_request({"prompt": "x", "think": think})
