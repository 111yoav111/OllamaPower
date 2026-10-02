from __future__ import annotations

import json
import threading

from ollamapower.eventlog import EventLog, follow_entries, read_entries

from .conftest import wait_for


def test_write_and_read_back_merged_by_time(tmp_path):
    log = EventLog(tmp_path, 1024 * 1024, 2)
    log.event("hub_start", version="x")
    log.request(path="/api/chat", status=200)
    log.event("hub_stop")
    log.close()
    entries = list(read_entries(tmp_path, ["events", "requests"]))
    assert [e["kind"] for e in entries] == ["hub_start", "request", "hub_stop"]
    assert [e["log"] for e in entries] == ["events", "requests", "events"]
    assert [e["kind"] for e in read_entries(tmp_path, ["events"], kinds=["hub_stop"])] == ["hub_stop"]


def test_rotation_keeps_old_entries_readable(tmp_path):
    log = EventLog(tmp_path, 2000, 3)
    for i in range(200):
        log.request(n=i, pad="x" * 50)
    log.close()
    assert (tmp_path / "requests.jsonl.3").exists() and not (tmp_path / "requests.jsonl.4").exists()
    ns = [e["n"] for e in read_entries(tmp_path, ["requests"])]
    assert ns == list(range(ns[0], 200)) and ns[0] > 0  # contiguous, oldest rotated away


def test_follow_sees_new_entries(tmp_path):
    log = EventLog(tmp_path, 1024 * 1024, 2)
    log.event("before")
    seen: list[str] = []

    def run():
        for e in follow_entries(tmp_path, ["events"], poll=0.05):
            seen.append(e["kind"])
            if len(seen) == 2:
                return

    t = threading.Thread(target=run, daemon=True)
    t.start()
    wait_for(lambda: False, 0.2)
    log.event("after1")
    log.event("after2")
    t.join(3)
    assert seen == ["after1", "after2"]


def test_lines_are_valid_json(tmp_path):
    log = EventLog(tmp_path, 1024 * 1024, 2)
    log.event("weird", text='quote " newline \n')
    log.close()
    line = (tmp_path / "events.jsonl").read_text().splitlines()[0]
    assert json.loads(line)["text"] == 'quote " newline \n'


def test_read_entries_limit_reads_only_what_it_needs(tmp_path):
    log = EventLog(tmp_path, 2000, 3)
    for i in range(200):
        log.request(n=i, pad="x" * 50)
        log.event("tick", n=i)
    log.close()
    last = read_entries(tmp_path, ["events", "requests"], limit=5)
    assert [(e["kind"], e["n"]) for e in last] == [
        ("tick", 197),
        ("request", 198),
        ("tick", 198),
        ("request", 199),
        ("tick", 199),
    ]
    assert [e["n"] for e in read_entries(tmp_path, ["requests"], kinds=["request"], limit=3)] == [197, 198, 199]
