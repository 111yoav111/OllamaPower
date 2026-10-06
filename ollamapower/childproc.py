"""Start child processes (``rpc-server``, ``llama-server``) that die with their parent.

Cleanup on a normal stop (Ctrl+C, SIGTERM, a closed window) already ends them,
but a hard kill of OllamaPower (``kill -9``, Task Manager's "End task", the OOM
killer) runs no Python code at all, and the child kept running: an
``rpc-server`` holding a helper's VRAM, a ``llama-server`` holding the hub GPU.

- Linux: the child asks the kernel for ``SIGKILL`` when its parent dies
  (``prctl(PR_SET_PDEATHSIG)``, set between fork and exec). Strictly, the
  signal fires when the *thread* that started the child ends; children are
  started from the event loop thread, which lives as long as the process.
- Windows: every child joins one Job Object per process, created with
  ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``; when the process dies, Windows closes
  its last handle to the job and ends everything in it.
- macOS (and other systems): there is no equivalent, so a hard-killed hub or
  agent can still leave its child behind (``pkill rpc-server`` /
  ``pkill llama-server`` cleans up).

None of this may stop a launch: any failure is logged at debug level and the
child runs as before.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import signal
import sys
from collections.abc import Callable
from typing import Any

log = logging.getLogger(__name__)

PR_SET_PDEATHSIG = 1


@functools.cache
def _prctl() -> Any:
    """libc's ``prctl``, or None (looked up once)."""
    try:
        import ctypes

        prctl = ctypes.CDLL(None, use_errno=True).prctl
        prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
        prctl.restype = ctypes.c_int
        return prctl
    except (OSError, AttributeError) as e:
        log.debug("prctl unavailable, children won't die with this process: %s", e)
        return None


def _linux_preexec() -> Callable[[], None] | None:
    """A ``preexec_fn`` that sets PR_SET_PDEATHSIG(SIGKILL), or None if prctl isn't reachable.

    Everything is looked up here, before the fork: the child only makes the
    foreign call and a couple of syscalls, nothing that takes a lock another
    thread of the parent could have held at fork time.
    """
    prctl = _prctl()
    if prctl is None:
        return None
    parent = os.getpid()
    kill, getppid, getpid, sigkill = os.kill, os.getppid, os.getpid, signal.SIGKILL

    def die_with_parent() -> None:
        try:
            prctl(PR_SET_PDEATHSIG, sigkill, 0, 0, 0)
            if getppid() != parent:  # the parent died before prctl took effect
                kill(getpid(), sigkill)
        except Exception:  # noqa: BLE001 - never fail the launch; nothing can be logged in the child
            pass

    return die_with_parent


class _WindowsJob:
    """One kill-on-close Job Object per process; ``None`` handle if it couldn't be made."""

    PROCESS_TERMINATE = 0x0001
    PROCESS_SET_QUOTA = 0x0100
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
    JobObjectExtendedLimitInformation = 9

    def __init__(self) -> None:
        self.handle: int | None = None
        self.tried = False
        self.k32: Any = None

    def _create(self) -> None:
        import ctypes
        from ctypes import wintypes

        class IoCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in ("r", "w", "o", "rb", "wb", "ob")]

        class BasicLimits(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimits),
                ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k32.SetInformationJobObject.restype = wintypes.BOOL
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        k32.AssignProcessToJobObject.restype = wintypes.BOOL
        k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        k32.CloseHandle.restype = wintypes.BOOL
        self.k32 = k32

        job = k32.CreateJobObjectW(None, None)  # not inheritable: only this process holds it
        if not job:
            raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")
        info = ExtendedLimits()
        info.BasicLimitInformation.LimitFlags = self.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        size = ctypes.sizeof(info)
        if not k32.SetInformationJobObject(job, self.JobObjectExtendedLimitInformation, ctypes.byref(info), size):
            err = ctypes.get_last_error()
            k32.CloseHandle(job)
            raise OSError(err, "SetInformationJobObject failed")
        self.handle = job  # kept open for the life of the process, on purpose

    def assign(self, pid: int) -> None:
        if not self.tried:
            self.tried = True
            try:
                self._create()
            except Exception as e:  # noqa: BLE001 - never fail a launch over this
                log.debug("no kill-on-close job object, children won't die with this process: %s", e)
        if self.handle is None:
            return
        import ctypes

        try:
            proc = self.k32.OpenProcess(self.PROCESS_SET_QUOTA | self.PROCESS_TERMINATE, False, pid)
            if not proc:
                raise OSError(ctypes.get_last_error(), "OpenProcess failed")
            try:
                if not self.k32.AssignProcessToJobObject(self.handle, proc):
                    raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject failed")
            finally:
                self.k32.CloseHandle(proc)
        except Exception as e:  # noqa: BLE001 - never fail a launch over this
            log.debug("could not tie child %s to this process: %s", pid, e)


_job = _WindowsJob()


def spawn_kwargs() -> dict[str, Any]:
    """Extra ``create_subprocess_exec``/``Popen`` arguments that tie the child to this process."""
    if sys.platform.startswith("linux"):
        preexec = _linux_preexec()
        if preexec is not None:
            return {"preexec_fn": preexec}
    return {}


def adopt(pid: int) -> None:
    """After the child started: on Windows, put it in this process's kill-on-close job."""
    if sys.platform == "win32":
        _job.assign(pid)


async def spawn(*args: str, **kwargs: Any) -> asyncio.subprocess.Process:
    """``asyncio.create_subprocess_exec`` for a child that must not outlive this process."""
    proc = await asyncio.create_subprocess_exec(*args, **spawn_kwargs(), **kwargs)
    adopt(proc.pid)
    return proc
