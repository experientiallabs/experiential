"""Capture status makes readiness, recent collection, and delivery trouble visible."""

import io
from dataclasses import replace

import pytest
from rich.console import Console
from rich.live import Live
from rich.text import Text

from exp.cli.capture import display as display_module
from exp.cli.capture.display import CaptureDisplay
from exp.runtime.capture.proxy import CaptureBypassReason
from exp.runtime.capture.upload import UploadStats


def test_startup_does_not_claim_collection_before_readiness() -> None:
    """Pending setup gives an actionable hint without a premature capture announcement."""
    output = io.StringIO()
    display = CaptureDisplay(Console(file=output, width=200), verbose=False)

    display.phase("Connecting to Experiential…")
    display.phase("Starting network extension… Ctrl+C to cancel")
    display.waiting()

    pending = output.getvalue()
    assert "Connecting to Experiential" in pending
    assert "Starting network extension" in pending
    assert "System Settings > General > Login Items & Extensions > Network Extensions" in pending
    assert "Capturing to" not in pending
    assert "requests captured" not in pending

    display.started(organization="Team [demo]", telemetry_url="https://example.com/capture")
    ready = output.getvalue()[len(pending) :]
    assert "Capturing to Team [demo] · Ctrl+C to stop" in ready
    assert "Telemetry: https://example.com/capture" in ready
    assert len(ready.splitlines()) == 2


def test_redirected_output_reports_changes_without_heartbeat_spam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Piped output reports the first state and changed counts without repeating idle ticks."""
    now = [100.0]
    monkeypatch.setattr(display_module.time, "monotonic", lambda: now[0])
    output = io.StringIO()
    display = CaptureDisplay(Console(file=output, width=200), verbose=False)
    empty = UploadStats(0, 0, 0, 0, 0)

    display.progress(empty)
    initial = output.getvalue()
    assert "0 requests captured · waiting for first completed request" in initial
    display.progress(replace(empty))
    assert output.getvalue() == initial

    captured = replace(
        empty,
        captured_exchanges=1,
        pending_batches=1,
        input_tokens=12_400,
        output_tokens=3_100,
        usage_exchanges=1,
    )
    display.progress(captured)
    first = output.getvalue()
    assert "1 request captured · 12.4K in / 3.1K out tokens · last just now · 1 pending" in first
    now[0] = 108.0
    display.progress(captured)
    assert output.getvalue() == first

    display.progress(replace(captured, pending_batches=0, uploaded_batches=1))
    delivered = output.getvalue()[len(first) :]
    assert "1 request captured · 12.4K in / 3.1K out tokens · last 8s ago" in delivered
    assert "batches uploaded" not in delivered
    assert "pending" not in delivered
    assert len(output.getvalue().splitlines()) == 3


def test_terminal_refreshes_activity_and_releases_rendering_between_phases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The actual Rich renderer can stop after startup, restart for counts, and close twice."""
    monkeypatch.setenv("TERM", "xterm-256color")
    now = [100.0]
    monkeypatch.setattr(display_module.time, "monotonic", lambda: now[0])
    lives: list[Live] = []

    def live(*, console: Console, refresh_per_second: int, transient: bool) -> Live:
        """Use a real terminal renderer with deterministic explicit refreshes."""
        renderer = Live(
            console=console,
            refresh_per_second=refresh_per_second,
            transient=transient,
            auto_refresh=False,
        )
        lives.append(renderer)
        return renderer

    monkeypatch.setattr(display_module, "Live", live)
    output = io.StringIO()
    console = Console(file=output, width=100, force_terminal=True, color_system=None)
    display = CaptureDisplay(console, verbose=False)
    renderer = lives[0]
    try:
        display.phase("Checking certificate…")
        assert renderer.is_started
        display.phase("Starting network extension… Ctrl+C to cancel")
        assert "Starting network extension" in Text.from_ansi(output.getvalue()).plain
        display.started(organization="Test", telemetry_url="https://example.com/capture")
        assert not renderer.is_started

        stats = UploadStats(0, 0, 0, 1, 1, input_tokens=800, output_tokens=20, usage_exchanges=1)
        display.progress(stats)
        assert renderer.is_started
        now[0] = 109.0
        prior_length = len(output.getvalue())
        display.progress(stats)
        refreshed = Text.from_ansi(output.getvalue()[prior_length:]).plain
        assert "1 request captured · 800 in / 20 out tokens · last 9s ago" in refreshed

        prior_length = len(output.getvalue())
        display.progress(
            replace(
                stats,
                captured_exchanges=2,
                input_tokens=1_100,
                output_tokens=30,
                usage_exchanges=2,
            )
        )
        refreshed = Text.from_ansi(output.getvalue()[prior_length:]).plain
        assert "2 requests captured · 1.1K in / 30 out tokens · last just now" in refreshed
    finally:
        display.close()
        display.close()

    assert not renderer.is_started
    assert "\x1b[?25h" in output.getvalue()


def test_dumb_terminal_keeps_startup_and_collection_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A terminal without cursor control uses immediate text instead of an invisible Live view."""
    monkeypatch.setenv("TERM", "dumb")
    output = io.StringIO()
    console = Console(file=output, width=200, force_terminal=True, color_system=None)
    assert console.is_terminal
    assert console.is_dumb_terminal
    display = CaptureDisplay(console, verbose=False)
    try:
        display.phase("Starting network extension… Ctrl+C to cancel")
        assert "Starting network extension" in output.getvalue()

        empty = UploadStats(0, 0, 0, 0, 0)
        display.progress(empty)
        assert "waiting for first completed request" in output.getvalue()
        prior = output.getvalue()
        display.progress(empty)
        assert output.getvalue() == prior

        display.progress(replace(empty, captured_exchanges=1))
        assert (
            "1 request captured · tokens unavailable · last just now"
            in output.getvalue()[len(prior) :]
        )
        assert "\x1b[" not in output.getvalue()
    finally:
        display.close()


@pytest.mark.parametrize(
    ("stats", "message"),
    [
        (UploadStats(2, 0, 0, 3, 1), "2 pending"),
        (UploadStats(0, 2, 0, 3, 1), "2 upload errors"),
        (UploadStats(0, 0, 2, 3, 1), "2 dropped"),
    ],
)
def test_delivery_problems_remain_visible_without_verbose(stats: UploadStats, message: str) -> None:
    """Concise output retains upload trouble even when setup details are hidden."""
    output = io.StringIO()
    display = CaptureDisplay(Console(file=output, width=200), verbose=False)
    display.progress(stats)
    assert message in output.getvalue()


@pytest.mark.parametrize("verbose", [False, True])
def test_verbose_controls_delivery_details(verbose: bool) -> None:
    """Quiet counters omit healthy delivery fields while verbose exposes the complete state."""
    output = io.StringIO()
    display = CaptureDisplay(Console(file=output, width=200), verbose=verbose)
    display.progress(UploadStats(0, 0, 0, 3, 2))
    text = output.getvalue()
    assert "3 requests captured" in text
    for detail in ("2 batches uploaded", "0 pending", "0 dropped", "0 upload errors"):
        assert (detail in text) is verbose


def test_stop_receipt_keeps_pending_retries_and_failures_visible() -> None:
    """The final receipt preserves actionable delivery state after transient status disappears."""
    output = io.StringIO()
    display = CaptureDisplay(Console(file=output, width=200), verbose=False)
    display.stopped(UploadStats(2, 1, 3, 97, 95))
    receipt = output.getvalue()
    assert "Capture stopped. 97 requests captured · tokens unavailable." in receipt
    assert "1 upload errors" in receipt
    assert "3 dropped" in receipt
    assert "2 batches pending; the next exp capture retries them." in receipt
    assert "95 batches uploaded" not in receipt


def test_successful_stop_is_one_short_receipt() -> None:
    """A healthy quiet shutdown retains request and token totals on one line."""
    output = io.StringIO()
    display = CaptureDisplay(Console(file=output, width=200), verbose=False)
    display.stopped(
        UploadStats(0, 0, 0, 97, 95, input_tokens=12_400, output_tokens=3_100, usage_exchanges=97)
    )
    assert output.getvalue().splitlines() == [
        "Capture stopped. 97 requests captured · 12.4K in / 3.1K out tokens."
    ]


@pytest.mark.parametrize(
    ("captured", "usage_exchanges", "input_tokens", "output_tokens", "expected"),
    [
        (3, 3, 12_400, 3_100, "12.4K in / 3.1K out tokens"),
        (3, 1, 12_400, 3_100, "12.4K in / 3.1K out tokens (partial)"),
        (3, 3, 1_752_000, 2_340_000_000, "1.75M in / 2.34B out tokens"),
        (3, 1, 999_995, 999_995_000, "1M in / 1B out tokens (partial)"),
        (3, 0, 0, 0, "tokens unavailable"),
        (3, 3, 0, 0, "0 in / 0 out tokens"),
        (0, 0, 0, 0, ""),
    ],
)
def test_token_totals_distinguish_complete_partial_and_missing_usage(
    captured: int,
    usage_exchanges: int,
    input_tokens: int,
    output_tokens: int,
    expected: str,
) -> None:
    """Live and final summaries distinguish real zero usage from absent or partial reporting."""
    output = io.StringIO()
    display = CaptureDisplay(Console(file=output, width=200), verbose=False)
    stats = UploadStats(
        0,
        0,
        0,
        captured,
        0,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        usage_exchanges=usage_exchanges,
    )
    display.progress(stats)
    progress = output.getvalue()
    display.stopped(stats)
    receipt = output.getvalue()[len(progress) :]
    for text in (progress, receipt):
        if expected:
            assert expected in text
        else:
            assert "tokens" not in text
        assert ("(partial)" in text) is (0 < usage_exchanges < captured)
        assert ("tokens unavailable" in text) is (captured > 0 and usage_exchanges == 0)


def test_redirected_output_reports_token_updates_without_resetting_activity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Usage updates appear even without a new request and retain the last request age."""
    now = [100.0]
    monkeypatch.setattr(display_module.time, "monotonic", lambda: now[0])
    output = io.StringIO()
    display = CaptureDisplay(Console(file=output, width=200), verbose=False)
    stats = UploadStats(0, 0, 0, 1, 0)
    display.progress(stats)
    before = output.getvalue()
    now[0] = 107.0
    display.progress(replace(stats, input_tokens=800, output_tokens=20, usage_exchanges=1))
    assert (
        "1 request captured · 800 in / 20 out tokens · last 7s ago"
        in output.getvalue()[len(before) :]
    )


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (0, "0"),
        (999, "999"),
        (1_000, "1K"),
        (1_005, "1.01K"),
        (12_400, "12.4K"),
        (999_994, "999.99K"),
        (999_995, "1M"),
        (1_000_000, "1M"),
        (1_752_000, "1.75M"),
        (999_994_999, "999.99M"),
        (999_995_000, "1B"),
        (1_000_000_000, "1B"),
        (2_345_000_000, "2.35B"),
        (999_995_000_000, "1T"),
        (1_000_000_000_000, "1T"),
        (123_456_789_012_345_678_901, "123456789.01T"),
    ],
)
def test_token_units_round_without_precision_loss_or_scientific_notation(
    count: int, expected: str
) -> None:
    """Readable token totals trim zeros and promote rounded thousands and millions."""
    assert display_module._format_tokens(count) == expected


@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize("reason", ["certificate"])
def test_bypassed_targets_remain_visible_as_partial_capture(
    monkeypatch: pytest.MonkeyPatch, terminal: bool, reason: CaptureBypassReason
) -> None:
    """Trust recovery names the uncaptured target once and persists beside changing usage."""
    monkeypatch.setenv("TERM", "xterm-256color")
    output = io.StringIO()
    display = CaptureDisplay(
        Console(file=output, width=200, force_terminal=terminal, color_system=None), verbose=False
    )
    stats = UploadStats(
        0, 0, 0, 8, 8, input_tokens=1_752_000, output_tokens=3_100, usage_exchanges=8
    )
    try:
        display.progress(stats)
        display.bypassed("Codex [Helper]", "chatgpt.com", reason)
        before = output.getvalue()
        display.bypassed("Codex [Helper]", "chatgpt.com", reason)
        assert output.getvalue() == before
        warning = "Not capturing Codex [Helper] on chatgpt.com: certificate rejected."
        assert warning in Text.from_ansi(before).plain

        display.progress(replace(stats, captured_exchanges=9, usage_exchanges=9))
        progress = Text.from_ansi(output.getvalue()[len(before) :]).plain
        assert "9 requests captured" in progress
        assert "partial capture: 1 target not captured" in progress
        assert "1.75M in / 3.1K out tokens" in progress

        before = output.getvalue()
        display.stopped(stats)
        receipt = Text.from_ansi(output.getvalue()[len(before) :]).plain
        assert "partial capture: 1 target not captured" in receipt
    finally:
        display.close()


def test_host_wide_bypass_is_explicit_before_any_requests() -> None:
    """Unidentified client recovery cannot leave zero-request capture looking complete."""
    output = io.StringIO()
    display = CaptureDisplay(Console(file=output, width=200), verbose=False)
    display.bypassed("All apps", "chatgpt.com", "certificate")
    display.progress(UploadStats(0, 0, 0, 0, 0))
    text = output.getvalue()
    assert "Not capturing All apps on chatgpt.com" in text
    assert "0 requests captured · partial capture: 1 target not captured" in text


@pytest.mark.parametrize("verbose", [False, True])
def test_diagnostics_are_timestamped_and_opt_in(verbose: bool) -> None:
    """Quiet Capture never gains connection noise and verbose output remains literal."""
    output = io.StringIO()
    display = CaptureDisplay(Console(file=output), verbose=verbose)
    display.diagnostic("tls_client_ready · chatgpt.com · connection abc123")
    if verbose:
        assert "tls_client_ready · chatgpt.com · connection abc123" in output.getvalue()
        assert output.getvalue()[:8].count(":") == 2
    else:
        assert not output.getvalue()
