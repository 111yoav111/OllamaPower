"""The priority queue (ported from the queue parts of DCP's test_lb.py)."""

from __future__ import annotations

import heapq

import pytest

from ollamapower.scheduling.task_pool import MAX_SKIPS, TaskPool, TaskPriority, TaskRecord, TaskStatus


def test_heap_serves_critical_first():
    heap: list[TaskRecord] = []
    for prio in (TaskPriority.LOW, TaskPriority.CRITICAL, TaskPriority.NORMAL):
        heapq.heappush(heap, TaskRecord(priority=prio))
    assert heapq.heappop(heap).priority == TaskPriority.CRITICAL


def test_fifo_within_a_priority():
    pool = TaskPool()
    first = pool.submit("a", TaskPriority.HIGH)
    second = pool.submit("b", TaskPriority.HIGH)
    low = pool.submit("c", TaskPriority.LOW)
    assert pool.candidates() == [first, second, low]


@pytest.mark.parametrize(
    "header,expected",
    [
        ("critical", TaskPriority.CRITICAL),
        ("HIGH", TaskPriority.HIGH),
        (" low ", TaskPriority.LOW),
        ("normal", TaskPriority.NORMAL),
        (None, TaskPriority.NORMAL),
        ("urgent!!", TaskPriority.NORMAL),
    ],
)
def test_priority_header(header, expected):
    assert TaskPriority.from_header(header) is expected


def test_skips_lead_to_forced_dispatch():
    pool = TaskPool()
    record = pool.submit("job")
    for _ in range(MAX_SKIPS - 1):
        assert pool.note_skipped([record]) == []
    assert pool.note_skipped([record]) == [record]


def test_fail_requeues_until_retries_run_out():
    pool = TaskPool()
    record = pool.submit("job", max_retries=2)
    assert pool.take(record)
    for _ in range(2):
        assert pool.fail(record, "crash")
        assert record.status is TaskStatus.REQUEUED
        assert pool.take(record)
    assert not pool.fail(record, "crash")
    assert record.status is TaskStatus.FAILED
    assert record.task_id not in pool.records
    assert len(pool) == 0


def test_take_twice():
    pool = TaskPool()
    record = pool.submit("job")
    assert pool.take(record)
    assert not pool.take(record)


def test_complete_forgets_the_record():
    pool = TaskPool()
    record = pool.submit("job")
    pool.take(record)
    pool.assign(record, "node-1")
    pool.complete(record, result=42)
    assert record.status is TaskStatus.DONE and record.result == 42
    assert record.task_id not in pool.records
    assert record.elapsed is not None


def test_snapshot():
    pool = TaskPool()
    pool.submit("a", TaskPriority.LOW)
    pool.submit("b", TaskPriority.CRITICAL)
    assert [s["priority"] for s in pool.snapshot()] == ["CRITICAL", "LOW"]
