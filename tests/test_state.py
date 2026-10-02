"""state.json: atomic, owner-only writes, backup fallback, versioning, caps, debounce."""

from __future__ import annotations

import asyncio
import json
import os
import stat
import sys

import pytest

from ollamapower import state as st

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")


def test_init_creates_empty_owner_only_file(tmp_path):
    path = tmp_path / "state.json"
    assert st.init_state_file(path)
    assert not st.init_state_file(path)
    assert json.loads(path.read_text()) == st.empty_state()
    if sys.platform != "win32":
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


def test_round_trip_and_backup(tmp_path):
    store = st.StateStore(tmp_path / "state.json")
    store.upsert_node("n1", name="desk")
    store.write_now()
    store.set_plan("digest", {"engine": "split"})
    store.write_now()
    again = st.StateStore(tmp_path / "state.json")
    again.load()
    assert again.data["nodes"]["n1"]["name"] == "desk"
    assert again.data["plans"]["digest"]["engine"] == "split"
    backup = json.loads((tmp_path / "state.json.bak").read_text())
    assert backup["plans"] == {}  # the previous good version


def test_corrupt_file_falls_back_to_backup(tmp_path):
    store = st.StateStore(tmp_path / "state.json")
    store.upsert_node("n1", name="desk")
    store.write_now()
    store.write_now()  # now the backup holds n1 too
    (tmp_path / "state.json").write_text("{ half written")
    again = st.StateStore(tmp_path / "state.json")
    again.load()
    assert again.data["nodes"]["n1"]["name"] == "desk"
    assert "loaded state.json.bak instead" in again.warnings[0]


def test_corrupt_file_never_becomes_the_backup(tmp_path):
    store = st.StateStore(tmp_path / "state.json")
    store.write_now()
    store.write_now()
    good_backup = (tmp_path / "state.json.bak").read_text()
    (tmp_path / "state.json").write_text("garbage")
    store.write_now()
    assert (tmp_path / "state.json.bak").read_text() == good_backup


def test_corrupt_without_backup_is_an_error(tmp_path):
    (tmp_path / "state.json").write_text("[]")
    with pytest.raises(st.StateError):
        st.StateStore(tmp_path / "state.json").load()


def test_newer_version_is_refused(tmp_path):
    (tmp_path / "state.json").write_text(json.dumps({**st.empty_state(), "version": 99}))
    with pytest.raises(st.StateError, match="newer"):
        st.StateStore(tmp_path / "state.json").load()


def test_upgrade_functions_run(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "STATE_VERSION", 2)
    monkeypatch.setitem(st.UPGRADES, 1, lambda d: {**d, "version": 2, "upgraded": True})
    (tmp_path / "state.json").write_text(json.dumps({"version": 1, "nodes": {}, "benchmarks": [], "plans": {}}))
    store = st.StateStore(tmp_path / "state.json")
    store.load()
    assert store.data["version"] == 2 and store.data["upgraded"]


def test_missing_file_starts_empty(tmp_path):
    store = st.StateStore(tmp_path / "state.json")
    store.load()
    assert store.data == st.empty_state()


def test_benchmarks_are_capped_per_model(tmp_path):
    store = st.StateStore(tmp_path / "state.json")
    for i in range(st.MAX_BENCHMARKS_PER_MODEL + 5):
        store.add_benchmark({"model_digest": "a", "n": i})
    store.add_benchmark({"model_digest": "b", "n": 0})
    a = [b["n"] for b in store.data["benchmarks"] if b["model_digest"] == "a"]
    assert len(a) == st.MAX_BENCHMARKS_PER_MODEL and a[-1] == st.MAX_BENCHMARKS_PER_MODEL + 4
    assert any(b["model_digest"] == "b" for b in store.data["benchmarks"])


async def test_saves_are_debounced(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "DEBOUNCE_SECONDS", 0.05)
    store = st.StateStore(tmp_path / "state.json")
    writes = []
    real = store.write_now
    monkeypatch.setattr(store, "write_now", lambda: (writes.append(1), real()))
    for i in range(20):
        store.upsert_node(f"n{i}", name=str(i))
    await asyncio.sleep(0.2)
    assert len(writes) == 1
    assert len(json.loads((tmp_path / "state.json").read_text())["nodes"]) == 20


@posix_only
def test_no_temp_files_left_behind(tmp_path):
    store = st.StateStore(tmp_path / "state.json")
    store.write_now()
    store.write_now()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["state.json", "state.json.bak"]
    assert stat.S_IMODE(os.stat(tmp_path).st_mode) == 0o700
