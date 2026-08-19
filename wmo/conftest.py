"""Suite-wide fixtures for explicitly interactive CLI tests."""

from __future__ import annotations

import errno
import fcntl
import os
import pty
import re
import select
import struct
import subprocess
import sys
import termios
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

_CONTROL = re.compile(r"\x1b\[([0-9;?]*)([A-Za-z])")
_SETTLE_SECONDS = 0.3


@pytest.fixture
def interactive_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Present a terminal stdin for a test that must reach an interactive prompt."""
    from wmo.cli.shared import consent

    monkeypatch.setattr(consent, "_stdin_is_terminal", lambda: True)


@dataclass(frozen=True)
class TerminalRun:
    """Everything one keyboard-driven terminal child process produced.

    Attributes:
        transcript: Every byte the child wrote, decoded, including control sequences.
        screen: The lines a terminal would still display after the child exited.
        returncode: Exit status of the child process.
    """

    transcript: str
    screen: tuple[str, ...]
    returncode: int

    def screen_text(self) -> str:
        """Return the visible screen as one newline-joined block of text."""
        return "\n".join(self.screen)


class _Screen:
    """A minimal terminal that applies the cursor control a redrawn region emits.

    Only the sequences rich uses to refresh a live region are honored: line erase, cursor up and
    down, carriage return, and cursor visibility. Anything else is ignored, which is safe because
    an ignored sequence can only leave extra text behind and therefore cannot hide duplication.
    """

    def __init__(self) -> None:
        """Start with one empty line and the cursor at its origin."""
        self._lines: list[list[str]] = [[]]
        self._row = 0
        self._column = 0

    def feed(self, text: str) -> None:
        """Apply one chunk of terminal output to the screen.

        Args:
            text: Decoded output, which may mix printable text and control sequences.
        """
        position = 0
        while position < len(text):
            match = _CONTROL.search(text, position)
            if match is None:
                self._write(text[position:])
                return
            self._write(text[position : match.start()])
            self._control(match.group(1), match.group(2))
            position = match.end()

    def lines(self) -> tuple[str, ...]:
        """Return every screen line with trailing blank lines and spaces removed."""
        rendered = ["".join(line).rstrip() for line in self._lines]
        while rendered and not rendered[-1]:
            rendered.pop()
        return tuple(rendered)

    def _write(self, text: str) -> None:
        """Place printable text at the cursor, honoring newlines and carriage returns.

        Args:
            text: Text without control sequences.
        """
        for character in text:
            if character == "\n":
                self._row += 1
                self._column = 0
                self._ensure_row()
                continue
            if character == "\r":
                self._column = 0
                continue
            line = self._lines[self._row]
            while len(line) <= self._column:
                line.append(" ")
            line[self._column] = character
            self._column += 1

    def _control(self, parameters: str, final: str) -> None:
        """Apply one control sequence to the cursor or the current line.

        Args:
            parameters: Numeric parameters of the sequence, possibly empty.
            final: Final character naming the sequence.
        """
        count = int(parameters) if parameters.isdigit() else 1
        if final == "K":
            self._erase_line(parameters)
        elif final == "A":
            self._row = max(0, self._row - count)
        elif final == "B":
            self._row += count
            self._ensure_row()
        elif final == "G":
            self._column = max(0, count - 1)
        elif final == "J" and parameters in {"", "0"}:
            del self._lines[self._row + 1 :]
            self._erase_line("0")

    def _erase_line(self, parameters: str) -> None:
        """Erase part or all of the current line.

        Args:
            parameters: ``2`` erases the whole line, ``1`` its start, anything else its remainder.
        """
        line = self._lines[self._row]
        if parameters == "2":
            line.clear()
        elif parameters == "1":
            for index in range(min(self._column + 1, len(line))):
                line[index] = " "
        else:
            del line[self._column :]

    def _ensure_row(self) -> None:
        """Grow the screen so the cursor row exists."""
        while len(self._lines) <= self._row:
            self._lines.append([])


def _render(transcript: str) -> tuple[str, ...]:
    """Replay a terminal transcript and return the lines it leaves visible.

    Args:
        transcript: Decoded child output including cursor control sequences.

    Returns:
        The visible screen lines, so an in-place redraw shows one copy of each row.
    """
    screen = _Screen()
    screen.feed(transcript)
    return screen.lines()


def _run_terminal_child(
    command: Sequence[str],
    *,
    steps: Sequence[tuple[str | None, str]],
    size: tuple[int, int] = (24, 100),
    cwd: Path | None = None,
    environment: dict[str, str] | None = None,
    returncode: int = 0,
    timeout: float = 30,
) -> TerminalRun:
    """Drive one interactive child process through a pseudo-terminal.

    Args:
        command: Argument vector of the child process.
        steps: Ordered pairs of the output marker to wait for and the keys to send once it
            appears. A marker of ``None`` sends its keys once the previous write settled, which
            drives two separate key batches into the same screen.
        size: Terminal rows and columns reported to the child.
        cwd: Working directory of the child.
        environment: Complete environment of the child, or ``None`` to inherit this one. Any
            terminal-size override is dropped so the requested size decides the layout.
        returncode: Exit status the child is expected to report.
        timeout: Seconds allowed for the whole session.

    Returns:
        The transcript, the still-visible screen, and the child exit status.
    """
    master, slave = pty.openpty()
    fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", size[0], size[1], 0, 0))
    child_environment = dict(os.environ if environment is None else environment)
    for override in ("COLUMNS", "LINES"):
        child_environment.pop(override, None)
    process = subprocess.Popen(
        list(command),
        stdin=slave,
        stdout=slave,
        stderr=slave,
        cwd=None if cwd is None else str(cwd),
        env=child_environment,
        close_fds=True,
    )
    os.close(slave)
    output = bytearray()
    pending = list(steps)
    searched = 0
    settle_at = 0.0
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            readable, _, _ = select.select([master], [], [], 0.05)
            if readable:
                try:
                    output.extend(os.read(master, 65_536))
                except OSError as error:
                    if error.errno != errno.EIO:
                        raise
                    break
            if pending:
                marker, keys = pending[0]
                ready = (
                    time.monotonic() >= settle_at
                    if marker is None
                    else marker.encode() in output[searched:]
                )
                if ready:
                    searched = len(output)
                    settle_at = time.monotonic() + _SETTLE_SECONDS
                    os.write(master, keys.encode())
                    pending.pop(0)
                    continue
            if process.poll() is not None:
                break
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
    finally:
        os.close(master)
    transcript = output.decode(errors="replace")
    waiting = pending[0][0] if pending else None
    assert not pending, f"child never rendered {waiting!r}:\n{transcript}"
    assert process.returncode == returncode, (
        f"child exited {process.returncode}, expected {returncode}:\n{transcript}"
    )
    return TerminalRun(
        transcript=transcript,
        screen=_render(transcript),
        returncode=process.returncode,
    )


@pytest.fixture
def rendered_screen() -> Callable[[str], tuple[str, ...]]:
    """Return a replay of terminal output into the lines a terminal would still display.

    Returns:
        The callable used to prove that a redrawn region leaves one copy of each row.
    """
    return _render


@pytest.fixture
def terminal_child() -> Callable[..., TerminalRun]:
    """Return a runner that drives one interactive child through a pseudo-terminal.

    Returns:
        The callable used by keyboard picker and installed distribution tests.
    """
    return _run_terminal_child


@pytest.fixture
def python_terminal_child(
    terminal_child: Callable[..., TerminalRun],
) -> Callable[..., TerminalRun]:
    """Return a runner that executes one inline script under a pseudo-terminal.

    Args:
        terminal_child: Generic pseudo-terminal runner.

    Returns:
        A callable taking the script source plus the arguments of ``terminal_child``.
    """

    def run(script: str, **keywords: object) -> TerminalRun:
        """Run the script in a child interpreter attached to a pseudo-terminal.

        Args:
            script: Python source executed by the child.
            keywords: Arguments forwarded to the pseudo-terminal runner.

        Returns:
            The transcript, visible screen, and exit status of the child.
        """
        return terminal_child([sys.executable, "-c", script], **keywords)

    return run
