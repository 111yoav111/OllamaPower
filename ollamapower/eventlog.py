"""Rotating JSON-lines logs: ``logs/events.jsonl`` and ``logs/requests.jsonl``.

These are logs, not state: joins, leaves, plan changes and failures go to the
events file, one line per front-door request to the requests file. Only
metadata is written - never prompts, responses or secrets. Files rotate at
``logging.file_max_bytes`` and are owner-only.

Writing happens on a background thread (``QueueHandler`` -> ``QueueListener``)
so disk I/O and rotation never stall the event loop that serves streams.

``ollamapower logs`` reads them back (they are also plain ``grep`` material).
"""

from __future__ import annotations

import heapq
import json
import logging
import logging.handlers
import os
import queue
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .files import POSIX, private_dir

LOG_FILES = {"events": "events.jsonl", "requests": "requests.jsonl"}


class _PrivateRotatingHandler(logging.handlers.RotatingFileHandler):
    def _open(self):  # type: ignore[no-untyped-def]
        old = os.umask(0o077) if POSIX else None
        try:
            return super()._open()
        finally:
            if old is not None:
                os.umask(old)


# Lines waiting for the writer thread; beyond this they are dropped (and counted).
MAX_PENDING = 10_000


class JsonlLog:
    """One rotating JSON-lines file, written by a background thread."""

    def __init__(self, path: Path, max_bytes: int, backups: int) -> None:
        private_dir(path.parent)
        self.path = path
        self.dropped = 0
        self._file = _PrivateRotatingHandler(path, maxBytes=max_bytes, backupCount=backups, encoding="utf-8")
        self._file.setFormatter(logging.Formatter("%(message)s"))
        self._queue: queue.Queue[logging.LogRecord] = queue.Queue(MAX_PENDING)
        self._listener = logging.handlers.QueueListener(self._queue, self._file)
        self._listener.start()

    def write(self, kind: str, **fields: Any) -> None:
        line = json.dumps({"ts": round(time.time(), 6), "kind": kind, **fields}, default=str, ensure_ascii=False)
        try:
            self._queue.put_nowait(logging.LogRecord("ollamapower.jsonl", logging.INFO, "", 0, line, None, None))
        except queue.Full:
            self.dropped += 1  # never block the event loop on a stuck disk

    def close(self) -> None:
        """Write everything still queued, then close the file."""
        self._listener.stop()
        self._file.close()


class EventLog:
    """The hub's two log files."""

    def __init__(self, logs_dir: Path, max_bytes: int, backups: int) -> None:
        self.events = JsonlLog(logs_dir / LOG_FILES["events"], max_bytes, backups)
        self.requests = JsonlLog(logs_dir / LOG_FILES["requests"], max_bytes, backups)

    def event(self, kind: str, **fields: Any) -> None:
        self.events.write(kind, **fields)

    def request(self, **fields: Any) -> None:
        self.requests.write("request", **fields)

    def close(self) -> None:
        self.events.close()
        self.requests.close()


class NullEventLog(EventLog):
    """Discards everything (tests, or when the log directory is unusable)."""

    def __init__(self) -> None:  # nothing to open
        pass

    def event(self, kind: str, **fields: Any) -> None:
        pass

    def request(self, **fields: Any) -> None:
        pass

    def close(self) -> None:
        pass


# ----- reading -------------------------------------------------------------------------


def _files_oldest_first(path: Path) -> list[Path]:
    rotated = [p for p in path.parent.glob(path.name + ".*") if p.suffix[1:].isdigit()]
    rotated.sort(key=lambda p: -int(p.suffix[1:]))  # .5 is the oldest, .1 the newest
    return rotated + ([path] if path.exists() else [])


def _parse(line: str) -> dict[str, Any] | None:
    try:
        value = json.loads(line)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _ts(entry: dict[str, Any]) -> float:
    ts = entry.get("ts")
    return float(ts) if isinstance(ts, (int, float)) else 0.0


def _tail(path: Path, name: str, kinds: list[str] | None, limit: int) -> list[dict[str, Any]]:
    """The last ``limit`` matching entries of one log, reading rotated files only as far back as needed."""
    newest_first: list[dict[str, Any]] = []
    for file in reversed(_files_oldest_first(path)):
        with open(file, encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()  # one file at most file_max_bytes
        while lines and len(newest_first) < limit:
            entry = _parse(lines.pop())
            if entry is not None and (not kinds or entry.get("kind") in kinds):
                entry.setdefault("log", name)
                newest_first.append(entry)
        if len(newest_first) >= limit:
            break
    return newest_first[::-1]


def read_entries(
    logs_dir: Path, files: list[str], kinds: list[str] | None = None, limit: int = 1000
) -> list[dict[str, Any]]:
    """The last ``limit`` entries across the given logs (``events``/``requests``), oldest first."""
    if limit <= 0:
        return []
    per_log = [_tail(logs_dir / LOG_FILES[name], name, kinds, limit) for name in files]
    merged = list(heapq.merge(*per_log, key=_ts))
    return merged[-limit:]


def follow_entries(
    logs_dir: Path, files: list[str], kinds: list[str] | None = None, poll: float = 0.5
) -> Iterator[dict[str, Any]]:
    """Yield entries appended from now on (survives rotation). Runs until interrupted."""
    positions: dict[Path, tuple[int, int]] = {}
    for name in files:
        path = logs_dir / LOG_FILES[name]
        if path.exists():
            st = path.stat()
            positions[path] = (st.st_ino, st.st_size)
    while True:
        for name in files:
            path = logs_dir / LOG_FILES[name]
            if not path.exists():
                continue
            st = path.stat()
            inode, offset = positions.get(path, (st.st_ino, 0))
            if inode != st.st_ino or st.st_size < offset:
                offset = 0  # rotated: start the new file from the top
            with open(path, "rb") as fh:
                fh.seek(offset)
                for raw in fh:
                    if not raw.endswith(b"\n"):
                        break  # half-written line; read it next round
                    offset += len(raw)
                    entry = _parse(raw.decode("utf-8", errors="replace"))
                    if entry is not None and (not kinds or entry.get("kind") in kinds):
                        entry.setdefault("log", name)
                        yield entry
            positions[path] = (st.st_ino, offset)
        time.sleep(poll)
