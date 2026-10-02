"""Dispatch, failure handling and snapshots (ported from DCP's test_lb.py)."""

from __future__ import annotations

import time

from ollamapower.scheduling.load_balancer import LoadBalancer, NodeState, NodeStatus
from ollamapower.scheduling.task_pool import MAX_SKIPS, TaskPriority, TaskStatus


def make_lb(result: bool = True):
    lb = LoadBalancer()
    sent: list[tuple[str, str]] = []

    async def send(node_id, record):
        sent.append((node_id, record.task_id))
        return result

    lb.set_send_callback(send)
    return lb, sent


def test_node_state_load_model():
    node = NodeState(node_id="n1", address=("127.0.0.1", 9000))
    node.cpu_usage, node.status = 50.0, NodeStatus.BUSY
    assert node.load_score == 70.0
    assert node.free_capacity == 30.0
    assert node.is_available
    node.cpu_usage = 95.0
    assert not node.is_available


async def test_register_and_unregister():
    lb, _ = make_lb()
    lb.register_node("n1", "127.0.0.1", 9001)
    assert "n1" in lb.nodes
    await lb.unregister_node("n1")
    assert "n1" not in lb.nodes


async def test_dispatch_to_least_loaded_node():
    lb, sent = make_lb()
    lb.register_node("busy", "10.0.0.1", 1)
    lb.register_node("idle", "10.0.0.2", 1)
    lb.update_node_stats("busy", cpu=60.0)
    lb.update_node_stats("idle", cpu=5.0)
    task_id = lb.submit_task("job", TaskPriority.NORMAL)
    record = await lb.try_dispatch()
    assert record is not None and record.task_id == task_id
    assert sent == [("idle", task_id)]
    assert lb.nodes["idle"].status is NodeStatus.BUSY
    assert record.status is TaskStatus.ASSIGNED


async def test_nothing_to_dispatch_without_nodes():
    lb, sent = make_lb()
    lb.submit_task("job")
    assert await lb.try_dispatch() is None
    assert sent == [] and len(lb.pool) == 1


async def test_hard_task_is_force_dispatched_after_skips():
    lb, sent = make_lb()
    lb.register_node("n1", "127.0.0.1", 1)
    lb.update_node_stats("n1", cpu=50.0)  # 50% free, the task wants 60
    lb.submit_task("heavy", difficulty=10.0)
    for _ in range(MAX_SKIPS - 1):
        assert await lb.try_dispatch() is None
    assert await lb.try_dispatch() is not None
    assert len(sent) == 1


async def test_failed_send_requeues():
    lb, _ = make_lb(result=False)
    lb.register_node("n1", "127.0.0.1", 1)
    task_id = lb.submit_task("job")
    assert await lb.try_dispatch() is None
    record = lb.pool.records[task_id]
    assert record.status is TaskStatus.REQUEUED
    assert len(lb.pool) == 1
    assert lb.nodes["n1"].status is NodeStatus.READY
    assert not lb.nodes["n1"].active_task_ids


async def test_task_failure_requeues():
    lb, _ = make_lb()
    lb.register_node("n1", "127.0.0.1", 1)
    task_id = lb.submit_task("job", max_retries=2)
    await lb.try_dispatch()
    lb.task_failed("n1", task_id, "crash")
    assert lb.pool.records[task_id].status is TaskStatus.REQUEUED
    assert lb.nodes["n1"].failed_tasks == 1


async def test_completion_fires_callback_and_frees_node():
    lb, _ = make_lb()
    done = []
    lb.on_task_done_cbs.append(done.append)
    lb.register_node("n1", "127.0.0.1", 1)
    task_id = lb.submit_task("job")
    await lb.try_dispatch()
    lb.task_completed("n1", task_id, result="ok")
    assert [r.result for r in done] == ["ok"]
    assert lb.nodes["n1"].status is NodeStatus.READY
    assert lb.compute_metrics().tasks_completed == 1


async def test_heartbeat_timeout_kicks_node_and_requeues_work():
    lb, _ = make_lb()
    kicked = []

    async def kick(node_id):
        kicked.append(node_id)

    lb.set_kick_callback(kick)
    lb.register_node("n1", "127.0.0.1", 1)
    task_id = lb.submit_task("job")
    await lb.try_dispatch()
    timed_out = await lb.check_heartbeats(now=time.monotonic() + lb.HEARTBEAT_TIMEOUT + 1)
    assert timed_out == kicked == ["n1"]
    assert "n1" not in lb.nodes
    assert lb.pool.records[task_id].status is TaskStatus.REQUEUED
    assert len(lb.pool) == 1


async def test_overload_and_recovery():
    lb, _ = make_lb()
    lb.register_node("n1", "127.0.0.1", 1)
    lb.update_node_stats("n1", cpu=95.0)
    assert lb.nodes["n1"].status is NodeStatus.OVERLOADED
    lb.update_node_stats("n1", cpu=85.0)  # hysteresis: still overloaded
    assert lb.nodes["n1"].status is NodeStatus.OVERLOADED
    lb.update_node_stats("n1", cpu=10.0)
    assert lb.nodes["n1"].status is NodeStatus.READY


async def test_kick_calls_back_and_requeues():
    lb, _ = make_lb()
    kicked = []

    async def kick(node_id):
        kicked.append(node_id)

    lb.set_kick_callback(kick)
    lb.register_node("n1", "127.0.0.1", 1)
    task_id = lb.submit_task("job")
    await lb.try_dispatch()
    await lb.kick_node("n1")
    assert kicked == ["n1"]
    assert "n1" not in lb.nodes
    assert lb.pool.records[task_id].status is TaskStatus.REQUEUED


async def test_broken_listener_does_not_break_scheduling():
    lb, _ = make_lb()

    def broken(_):
        raise RuntimeError("boom")

    lb.on_node_change_cbs.append(broken)
    lb.register_node("n1", "127.0.0.1", 1)
    assert "n1" in lb.nodes


async def test_snapshot_is_json_safe():
    import json

    lb, _ = make_lb()
    lb.register_node("n1", "127.0.0.1", 1)
    lb.submit_task("job", TaskPriority.HIGH)
    snap = lb.get_snapshot()
    json.dumps(snap)
    assert snap["nodes"]["n1"]["status"] == "ready"
    assert snap["queue"][0]["priority"] == "HIGH"


async def test_late_result_from_previous_owner_is_ignored():
    lb, sent = make_lb()
    lb.register_node("a", "10.0.0.1", 1)
    task_id = lb.submit_task("job")
    await lb.try_dispatch()
    await lb.unregister_node("a")  # requeued
    lb.register_node("b", "10.0.0.2", 1)
    await lb.try_dispatch()
    lb.task_completed("a", task_id, "stale")  # A's late answer
    lb.task_failed("a", task_id, "stale")
    record = lb.pool.records[task_id]
    assert record.status is TaskStatus.ASSIGNED and record.assigned_node == "b"
    assert len(lb.pool) == 0 and record.retries == 1
    lb.task_completed("b", task_id, "fresh")
    assert record.result == "fresh" and lb.nodes["b"].status is NodeStatus.READY


async def test_result_for_queued_task_is_ignored():
    lb, _ = make_lb()
    lb.register_node("n1", "127.0.0.1", 1)
    task_id = lb.submit_task("job")
    lb.task_completed("n1", task_id, "never dispatched")
    assert lb.pool.records[task_id].status is TaskStatus.PENDING
    assert len(lb.pool) == 1


async def test_node_dropping_during_send_does_not_queue_twice():
    lb = LoadBalancer()
    lb.register_node("n1", "127.0.0.1", 1)

    async def send(node_id, record):
        await lb.unregister_node(node_id)  # the node disconnects while we send
        return False

    lb.set_send_callback(send)
    task_id = lb.submit_task("job")
    assert await lb.try_dispatch() is None
    assert len(lb.pool) == 1
    assert lb.pool.records[task_id].status is TaskStatus.REQUEUED


async def test_raising_send_callback_does_not_stop_dispatch():
    lb = LoadBalancer()
    lb.register_node("n1", "127.0.0.1", 1)

    async def send(node_id, record):
        raise RuntimeError("transport closing")

    lb.set_send_callback(send)
    task_id = lb.submit_task("job")
    assert await lb.try_dispatch() is None
    assert lb.pool.records[task_id].status is TaskStatus.REQUEUED
    assert lb.nodes["n1"].status is NodeStatus.READY


async def test_waiting_for_nodes_does_not_count_as_skips():
    lb, sent = make_lb()
    lb.submit_task("heavy", difficulty=10.0)
    for _ in range(MAX_SKIPS * 2):
        await lb.try_dispatch()  # no nodes yet
    lb.register_node("n1", "127.0.0.1", 1)
    lb.update_node_stats("n1", cpu=50.0)
    assert await lb.try_dispatch() is None  # still subject to the capacity check
    assert sent == []
