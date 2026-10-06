"""SIGTERM / SIGBREAK stop a running asyncio program the way Ctrl+C does: cleanup runs."""

from __future__ import annotations

import asyncio
import signal

import pytest

from ollamapower.cli_common import stop_like_ctrl_c

STOP_SIGNALS = [signal.SIGTERM] + ([signal.SIGBREAK] if hasattr(signal, "SIGBREAK") else [])


@pytest.mark.parametrize("sig", STOP_SIGNALS)
def test_stop_signal_cancels_main_and_runs_cleanup(sig):
    cleaned = []

    async def main():
        try:
            signal.raise_signal(sig)
            await asyncio.sleep(30)
        finally:
            cleaned.append(True)

    before = signal.getsignal(sig)
    with stop_like_ctrl_c(), pytest.raises(KeyboardInterrupt):
        asyncio.run(main())
    assert cleaned == [True]
    assert signal.getsignal(sig) == before  # restored afterwards


def test_stop_signal_outside_asyncio_is_a_keyboard_interrupt(monkeypatch):
    monkeypatch.setattr(signal, "getsignal", lambda _sig: signal.SIG_DFL)
    with stop_like_ctrl_c(), pytest.raises(KeyboardInterrupt):
        signal.raise_signal(signal.SIGTERM)
