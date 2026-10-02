"""Small building blocks for the TUI: safe text, time formatting and modal dialogs.

Everything that came from a client (names, user agents, paths, IPs) must be
rendered through ``t()``: it escapes control characters and returns a Rich
``Text``, which - unlike a plain ``str`` - is never parsed as markup, so a
device named ``[blink]...`` can't restyle the admin screen.
"""

from __future__ import annotations

import time
from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static

from ..cli_common import duration, ts  # noqa: F401 - re-exported for the TUIs
from ..cli_common import gb as gib  # noqa: F401
from ..ollama.client import human_bytes  # noqa: F401
from ..textsafe import safe


def t(value: Any, style: str = "") -> Text:
    return Text("-" if value is None or value == "" else safe(value), style=style)


def ago(value: float | None, now: float | None = None) -> str:
    if value is None:
        return "never"
    secs = int((time.time() if now is None else now) - value)
    if secs < 60:
        return f"{secs}s ago"
    if secs < 3600:
        return f"{secs // 60}m ago"
    if secs < 86400:
        return f"{secs // 3600}h ago"
    return f"{secs // 86400}d ago"


# Helper status dots (README legend: 🟢 lending · 🟡 paused · 🔴 offline · ⚠ incompatible).
STATUS_DOT = {
    "lending": Text("●", style="bold green"),
    "yielding": Text("●", style="yellow"),
    "paused": Text("●", style="yellow"),
    "offline": Text("●", style="bold red"),
    "incompatible": Text("⚠", style="bold red"),
}


def bar(used: float, total: float, width: int = 24) -> Text:
    """A text meter: green, yellow from 75 %, red from 90 %."""
    frac = 0.0 if total <= 0 else min(1.0, max(0.0, used / total))
    filled = round(frac * width)
    style = "green" if frac < 0.75 else "yellow" if frac < 0.9 else "red"
    return Text.assemble(("█" * filled, style), ("░" * (width - filled), "dim"), f" {frac * 100:3.0f}%")


class ConfirmScreen(ModalScreen[bool]):
    """Yes/No question. Dismisses with True only on an explicit "Yes"."""

    BINDINGS = [Binding("escape", "no", "Cancel"), Binding("y", "yes", "Yes"), Binding("n", "no", "No")]

    def __init__(self, question: str, danger: bool = True) -> None:
        super().__init__()
        self.question = question
        self.danger = danger

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Static(Text(self.question))
            with Horizontal(classes="buttons"):
                yield Button("Yes", id="yes", variant="error" if self.danger else "primary")
                yield Button("No", id="no")

    def on_mount(self) -> None:
        self.query_one("#no", Button).focus()  # the safe choice is the default

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")

    def action_yes(self) -> None:
        self.dismiss(True)

    def action_no(self) -> None:
        self.dismiss(False)


class InputScreen(ModalScreen[str | None]):
    """Ask for one line of text. Dismisses with None on cancel."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, prompt: str, value: str = "", placeholder: str = "") -> None:
        super().__init__()
        self.prompt = prompt
        self.value = value
        self.placeholder = placeholder

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Label(Text(self.prompt))
            yield Input(value=self.value, placeholder=self.placeholder, id="value")
            with Horizontal(classes="buttons"):
                yield Button("OK", id="ok", variant="primary")
                yield Button("Cancel", id="cancel")

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(self.query_one(Input).value if event.button.id == "ok" else None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class MessageScreen(ModalScreen[None]):
    """Show a message (e.g. a freshly rotated token) until dismissed."""

    BINDINGS = [Binding("escape", "close", "Close"), Binding("enter", "close", "Close")]

    def __init__(self, title: str, body: Text | str) -> None:
        super().__init__()
        self.title_text = title
        self.body = body if isinstance(body, Text) else Text(body)

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog wide"):
            yield Label(Text(self.title_text, style="bold"))
            yield Static(self.body, classes="selectable")
            with Horizontal(classes="buttons"):
                yield Button("Close", id="close", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(None)

    def action_close(self) -> None:
        self.dismiss(None)
