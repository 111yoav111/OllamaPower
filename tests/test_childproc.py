"""Children started through ``childproc.spawn`` die when their parent is hard-killed."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import textwrap
import time

import psutil
import pytest

from ollamapower import childproc

PARENT = textwrap.dedent(
    """
    import asyncio, sys
    from ollamapower.childproc import spawn

    async def main():
        child = await spawn(sys.executable, "-c", "import time; time.sleep(60)")
        print(child.pid, flush=True)
        await asyncio.sleep(60)

    asyncio.run(main())
    """
)


def _gone(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="PR_SET_PDEATHSIG is Linux-only")
def test_child_dies_when_parent_is_sigkilled():
    parent = subprocess.Popen([sys.executable, "-c", PARENT], stdout=subprocess.PIPE, text=True)
    child_pid = None
    try:
        assert parent.stdout is not None
        child_pid = int(parent.stdout.readline())
        assert not _gone(child_pid)
        os.kill(parent.pid, signal.SIGKILL)
        parent.wait(10)
        deadline = time.monotonic() + 10
        while not _gone(child_pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert _gone(child_pid)
    finally:
        parent.kill()
        if child_pid is not None and not _gone(child_pid):
            os.kill(child_pid, signal.SIGKILL)


async def test_spawn_works_without_prctl(monkeypatch):
    """A failed lookup must not stop the launch."""
    monkeypatch.setattr(childproc, "_prctl", lambda: None)
    proc = await childproc.spawn(sys.executable, "-c", "print('ok')", stdout=asyncio.subprocess.PIPE)
    out, _ = await proc.communicate()
    assert proc.returncode == 0 and out.strip() == b"ok"


def test_failing_job_object_is_not_an_error(monkeypatch):
    job = childproc._WindowsJob()

    def fail() -> None:
        raise OSError("no job objects here")

    monkeypatch.setattr(job, "_create", fail)
    job.assign(os.getpid())
    job.assign(os.getpid())  # tried once, then skipped
    assert job.tried and job.handle is None
