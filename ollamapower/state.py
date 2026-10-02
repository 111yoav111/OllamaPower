"""``state.json``: the hub's only persistent state (spec Section 16).

Nodes seen, benchmark results and chosen plans (keyed by model name) live in one small JSON file.
**Nothing else reads or writes it** - keeping all access here makes a later
move into OllamaGate's database a contained job.

* Writes are atomic (temp file, ``fsync``, ``os.replace``) and owner-only.
* Before each write the previous good file is kept as ``state.json.bak``; if
  ``state.json`` doesn't parse on start, the backup is loaded and a warning
  logged.
* Writes are serialised and debounced (``save_soon``: at most one write per
  ``DEBOUNCE_SECONDS``); ``flush`` writes immediately (shutdown, ``init``).
* ``version`` plus small upgrade functions handle format changes.
* Benchmarks are capped per model so the file stays small.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .files import write_private

log = logging.getLogger("ollamapower.state")

STATE_VERSION = 1
DEBOUNCE_SECONDS = 1.0
MAX_BENCHMARKS_PER_MODEL = 20
MAX_STATE_BYTES = 16 * 1024 * 1024  # far above anything real; refuse to parse bigger


class StateError(ValueError):
    """``state.json`` and its backup are both unusable."""


def empty_state() -> dict[str, Any]:
    return {"version": STATE_VERSION, "nodes": {}, "benchmarks": [], "plans": {}}


# Upgrade functions: UPGRADES[n] turns a version-n document into version n+1.
UPGRADES: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {}


def _upgrade(doc: dict[str, Any]) -> dict[str, Any]:
    version = doc.get("version")
    if not isinstance(version, int) or version < 1:
        raise StateError(f"state has no valid version (got {version!r})")
    if version > STATE_VERSION:
        raise StateError(f"state version {version} is newer than this OllamaPower ({STATE_VERSION}); upgrade it")
    while version < STATE_VERSION:
        doc = UPGRADES[version](doc)
        version = doc["version"]
    return doc


def _validate(doc: Any) -> dict[str, Any]:
    if not isinstance(doc, dict):
        raise StateError("state is not a JSON object")
    doc = _upgrade(doc)
    for key, kind in (("nodes", dict), ("benchmarks", list), ("plans", dict)):
        if not isinstance(doc.setdefault(key, kind()), kind):
            raise StateError(f"state.{key} must be a {kind.__name__}")
    return doc


def _read(path: Path) -> dict[str, Any]:
    if path.stat().st_size > MAX_STATE_BYTES:
        raise StateError(f"{path.name} is unreasonably large")
    try:
        return _validate(json.loads(path.read_text(encoding="utf-8")))
    except (ValueError, UnicodeDecodeError) as exc:
        raise StateError(f"{path.name} is not valid: {exc}") from None


class StateStore:
    """Load, query and save ``state.json``. Methods are synchronous except the savers."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.backup_path = path.with_name(path.name + ".bak")
        self.data = empty_state()
        self.warnings: list[str] = []
        self._lock = asyncio.Lock()
        self._pending: asyncio.Task[None] | None = None

    # ----- load / save -------------------------------------------------------------------

    def load(self) -> None:
        """Load the file, falling back to the backup, or start empty if neither exists."""
        if not self.path.exists():
            if self.backup_path.exists():
                self._load_backup(f"{self.path.name} is missing")
            return
        try:
            self.data = _read(self.path)
        except (StateError, OSError) as exc:
            if not self.backup_path.exists():
                raise StateError(f"{exc}; no backup to fall back to - fix or delete {self.path}") from None
            self._load_backup(str(exc))

    def _load_backup(self, why: str) -> None:
        try:
            self.data = _read(self.backup_path)
        except (StateError, OSError) as exc:
            raise StateError(f"{why}, and the backup is unusable too ({exc})") from None
        message = f"{why}; loaded {self.backup_path.name} instead"
        self.warnings.append(message)
        log.warning("state: %s", message)

    def write_now(self) -> None:
        """Synchronous atomic write (used by ``init`` and at shutdown)."""
        payload = json.dumps(self.data, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        if self.path.exists():
            with contextlib.suppress(StateError, OSError):
                _read(self.path)  # only a good file becomes the backup
                write_private(self.backup_path, self.path.read_bytes())
        write_private(self.path, payload)

    async def flush(self) -> None:
        """Write now. A pending debounced write still runs later; it rewrites the same data."""
        async with self._lock:
            await asyncio.to_thread(self.write_now)

    def save_soon(self) -> None:
        """Schedule a write; many calls within ``DEBOUNCE_SECONDS`` become one write.

        Without a running event loop (CLI tools) nothing is scheduled; the
        caller writes with ``write_now``.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        if self._pending is None or self._pending.done():
            self._pending = asyncio.ensure_future(self._debounced())

    async def _debounced(self) -> None:
        await asyncio.sleep(DEBOUNCE_SECONDS)
        async with self._lock:
            try:
                await asyncio.to_thread(self.write_now)
            except OSError as exc:
                log.error("could not write %s: %s", self.path.name, exc)

    # ----- accessors (more arrive with the phases that need them) --------------------------

    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)

    def upsert_node(self, node_id: str, **fields: Any) -> None:
        self.data["nodes"].setdefault(node_id, {}).update(fields)
        self.save_soon()

    def add_benchmark(self, result: dict[str, Any]) -> None:
        """Append a result, keeping only the newest ``MAX_BENCHMARKS_PER_MODEL`` per model."""
        benchmarks: list[dict[str, Any]] = self.data["benchmarks"]
        benchmarks.append(result)
        digest = result.get("model_digest")
        same = [i for i, b in enumerate(benchmarks) if b.get("model_digest") == digest]
        for i in reversed(same[:-MAX_BENCHMARKS_PER_MODEL]):
            del benchmarks[i]
        self.save_soon()

    def set_plan(self, model: str, plan: dict[str, Any]) -> None:
        self.data["plans"][model] = plan
        self.save_soon()


def init_state_file(path: Path) -> bool:
    """Create an empty, owner-only ``state.json`` if there is none. True if created."""
    if path.exists():
        return False
    StateStore(path).write_now()
    return True
