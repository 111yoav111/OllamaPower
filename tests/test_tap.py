from __future__ import annotations

import json

from ollamapower.hub.frontdoor.proxy import MAX_TAP_LINE, GenerationTap

FINAL = {
    "done": True,
    "eval_count": 40,
    "eval_duration": 2_000_000_000,
    "prompt_eval_count": 10,
    "prompt_eval_duration": 500_000_000,
}


def test_final_chunk_split_across_reads():
    tap = GenerationTap()
    data = (json.dumps({"done": False, "message": {"content": "x"}}) + "\n" + json.dumps(FINAL) + "\n").encode()
    for i in range(0, len(data), 7):
        tap.feed(data[i : i + 7])
    stats = tap.close()
    assert stats.gen_tps == 20.0 and stats.prompt_tps == 20.0 and stats.gen_tokens == 40


def test_non_streamed_reply_without_newline():
    tap = GenerationTap()
    tap.feed(json.dumps({**FINAL, "response": "hi"}).encode())
    assert tap.close().gen_tokens == 40


def test_huge_lines_are_skipped_not_buffered():
    tap = GenerationTap()
    tap.feed(b'{"embedding": [' + b"1," * MAX_TAP_LINE)
    assert len(tap._buf) == 0
    tap.feed(b'1], "done": true}\n')
    assert tap.close() is None


def test_no_done_no_stats():
    tap = GenerationTap()
    tap.feed(b'{"done": false}\nnot json "done"\n')
    assert tap.close() is None
