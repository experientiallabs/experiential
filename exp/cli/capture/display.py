"""Compact foreground Capture status without model content or certificate details."""

from __future__ import annotations

import time

from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

from exp.runtime.capture.proxy import CaptureBypassReason
from exp.runtime.capture.upload import UploadStats


def _format_tokens(count: int) -> str:
    """Compact token totals to two decimal places, promoting rounded unit boundaries."""
    if count < 1_000:
        return str(count)
    divisor = 1_000
    for suffix in ("K", "M", "B", "T"):
        hundredths = (count * 100 + divisor // 2) // divisor
        if hundredths < 100_000 or suffix == "T":
            break
        divisor *= 1_000
    number = f"{hundredths // 100}.{hundredths % 100:02d}".rstrip("0").rstrip(".")
    return f"{number}{suffix}"


class CaptureDisplay:
    """Separate pending startup from active collection and visibly changing counters."""

    def __init__(self, console: Console, *, verbose: bool) -> None:
        """Configure one transient terminal line or change-only output when redirected."""
        self._console = console
        self._verbose = verbose
        self._live = Live(console=console, refresh_per_second=4, transient=True)
        self._live_started = False
        self._last_stats: UploadStats | None = None
        self._last_capture_at: float | None = None
        self._bypasses: set[tuple[str, str]] = set()

    def phase(self, message: str) -> None:
        """Show work immediately without claiming interception has started."""
        if self._console.is_terminal and not self._console.is_dumb_terminal:
            self._live.update(Spinner("dots", text=Text(message)), refresh=True)
            if not self._live_started:
                self._live.start(refresh=True)
                self._live_started = True
        else:
            self._console.print(message, markup=False)

    def started(self, *, organization: str, telemetry_url: str) -> None:
        """Announce collection only after native readiness, then make room for counters."""
        self.close()
        self._console.print(
            Text.assemble(("Capturing", "green"), f" to {organization} · Ctrl+C to stop")
        )
        self._console.print(f"Telemetry: {telemetry_url}", markup=False)

    def waiting(self) -> None:
        """Expose the actionable native approval location without verbose setup prose."""
        self._console.print(
            "Still waiting for Mitmproxy Redirector. Enable it in System Settings > "
            "General > Login Items & Extensions > Network Extensions.",
            markup=False,
        )

    def progress(self, stats: UploadStats) -> None:
        """Refresh terminal activity; redirected output changes only when counters change."""
        now = time.monotonic()
        previous_count = self._last_stats.captured_exchanges if self._last_stats else 0
        if stats.captured_exchanges > previous_count:
            self._last_capture_at = now
        line = self._stats_text(stats, now=now)
        if self._console.is_terminal and not self._console.is_dumb_terminal:
            self._live.update(line, refresh=True)
            if not self._live_started:
                self._live.start(refresh=True)
                self._live_started = True
        elif stats != self._last_stats:
            self._console.print(line)
        self._last_stats = stats

    def diagnostic(self, message: str) -> None:
        """Show timestamped content-free connection events only when explicitly enabled."""
        if self._verbose:
            self._console.print(f"{time.strftime('%H:%M:%S')} {message}", markup=False, style="dim")

    def bypassed(self, application: str, host: str, reason: CaptureBypassReason) -> None:
        """Expose reduced coverage once and retain it beside the live request counters."""
        target = (application, host)
        if target in self._bypasses:
            return
        self._bypasses.add(target)
        self._console.print(
            f"Not capturing {application} on {host}: certificate rejected. "
            "Retry the request; restart the app and Capture to retry tracing.",
            markup=False,
            style="yellow",
        )
        if self._last_stats is not None:
            line = self._stats_text(self._last_stats, now=time.monotonic())
            if self._live_started:
                self._live.update(line, refresh=True)
            else:
                self._console.print(line)

    def _stats_text(self, stats: UploadStats, *, now: float) -> Text:
        """Keep request activity prominent and show delivery problems even in quiet mode."""
        count = stats.captured_exchanges
        parts = [f"{count} {'request' if count == 1 else 'requests'} captured"]
        if self._bypasses:
            parts.append(self._coverage_summary())
        if tokens := self._token_summary(stats):
            parts.append(tokens)
        if self._last_capture_at is not None:
            elapsed = max(0, int(now - self._last_capture_at))
            parts.append("last just now" if elapsed < 2 else f"last {elapsed}s ago")
        elif count == 0:
            parts.append("waiting for first completed request")
        if self._verbose:
            parts.append(f"{stats.uploaded_batches} batches uploaded")
        for count, label in (
            (stats.pending_batches, "pending"),
            (stats.dropped_exchanges, "dropped"),
            (stats.upload_errors, "upload errors"),
        ):
            if count or self._verbose:
                parts.append(f"{count} {label}")
        style = "yellow" if self._bypasses else "green" if stats.captured_exchanges else ""
        return Text(" · ".join(parts), style=style)

    def _coverage_summary(self) -> str:
        """Keep exclusions visible without repeating per-app warnings on every update."""
        count = len(self._bypasses)
        return f"partial capture: {count} {'target' if count == 1 else 'targets'} not captured"

    @staticmethod
    def _token_summary(stats: UploadStats) -> str:
        """Show compact reported totals while distinguishing missing or partial usage."""
        if not stats.captured_exchanges:
            return ""
        if not stats.usage_exchanges:
            return "tokens unavailable"
        tokens = (
            f"{_format_tokens(stats.input_tokens)} in / "
            f"{_format_tokens(stats.output_tokens)} out tokens"
        )
        if stats.usage_exchanges < stats.captured_exchanges:
            tokens += " (partial)"
        return tokens

    def stopped(self, stats: UploadStats) -> None:
        """Retain a concise final receipt after interception shutdown was confirmed."""
        self.close()
        count = stats.captured_exchanges
        receipt = f"{count} {'request' if count == 1 else 'requests'} captured"
        if tokens := self._token_summary(stats):
            receipt += f" · {tokens}"
        if self._bypasses:
            receipt += f" · {self._coverage_summary()}"
        self._console.print(
            Text.assemble(
                ("Capture stopped.", "green"),
                f" {receipt}.",
            )
        )
        if self._verbose or stats.dropped_exchanges or stats.upload_errors:
            self._console.print(self._stats_text(stats, now=time.monotonic()))
        if stats.pending_batches:
            self._console.print(
                f"{stats.pending_batches} batches pending; the next exp capture retries them."
            )

    def close(self) -> None:
        """Release terminal rendering on successful completion, cancellation, or error."""
        if self._live_started:
            self._live.stop()
            self._live_started = False
