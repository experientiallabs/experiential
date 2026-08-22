"""Tests for the mock-isolated gateway latency report."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from exp.runtime.gateway.latency_measure import (
    MockOpenAIServer,
    RequestSample,
    chat_payload,
    percentile,
)
from exp.runtime.gateway.latency_report import (
    CAVEAT,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    GatewayAddedLatency,
    LatencyArmStats,
    LatencyMeasuredRun,
    LatencyReport,
    LatencyRunConfig,
    RunnerContext,
    _assert_functional_success,
    default_config,
    gateway_added,
    measure_arm,
    parse_args,
    render_markdown,
    run_latency_report,
    select_representative_run,
    summarize_samples,
)


def _stats(
    *,
    p50_ms: float,
    p95_ms: float = 0.0,
    p99_ms: float = 0.0,
    failures: int = 0,
    requests: int = 4,
) -> LatencyArmStats:
    """Build one arm summary for table and selection tests.

    Args:
        p50_ms: Median latency.
        p95_ms: 95th-percentile latency.
        p99_ms: 99th-percentile latency.
        failures: Failed request count.
        requests: Total request count.

    Returns:
        Synthetic arm statistics.
    """
    return LatencyArmStats(
        requests=requests,
        failures=failures,
        failure_rate=failures / requests,
        rps=10.0,
        mean_ms=p50_ms,
        p50_ms=p50_ms,
        p95_ms=p95_ms or p50_ms,
        p99_ms=p99_ms or p50_ms,
    )


def _run(
    run_index: int,
    *,
    experiential_p50: float,
    mock_p50: float = 1.0,
    litellm_p50: float | None = None,
    failures: int = 0,
    litellm_failures: int = 0,
) -> LatencyMeasuredRun:
    """Build one measured run for representative-run tests.

    Args:
        run_index: 1-based repeat number.
        experiential_p50: Experiential median used for median-run selection.
        mock_p50: Mock-direct median.
        litellm_p50: Optional LiteLLM median.
        failures: Failures applied to the Experiential arm.
        litellm_failures: Failures applied to the LiteLLM arm.

    Returns:
        Synthetic measured run.
    """
    mock = _stats(p50_ms=mock_p50)
    experiential = _stats(p50_ms=experiential_p50, failures=failures)
    litellm = None if litellm_p50 is None else _stats(p50_ms=litellm_p50, failures=litellm_failures)
    return LatencyMeasuredRun(
        run_index=run_index,
        gateway_order=("experiential", "litellm") if litellm is not None else ("experiential",),
        mock_direct=mock,
        experiential=experiential,
        experiential_added=gateway_added(experiential, mock),
        litellm=litellm,
        litellm_added=None if litellm is None else gateway_added(litellm, mock),
    )


def test_percentile_matches_litellm_nearest_rank() -> None:
    """Nearest-rank p50/p95/p99 follow the LiteLLM bench convention."""
    values = tuple(float(index) for index in range(1, 11))
    assert percentile((), 50) == 0.0
    assert percentile(values, 50) == 6.0
    assert percentile(values, 95) == 10.0
    assert percentile(values, 99) == 10.0
    with pytest.raises(ValueError, match="between 0 and 100"):
        percentile(values, 120)


def test_summarize_samples_reports_failures_and_throughput() -> None:
    """A mixed arm reports failure rate, RPS, and success-only percentiles."""
    samples = (
        RequestSample(success=True, latency_ms=10.0),
        RequestSample(success=True, latency_ms=20.0),
        RequestSample(success=False, latency_ms=30.0, error="timeout"),
    )
    stats = summarize_samples(samples, wall_time_s=2.0)
    assert stats.requests == 3
    assert stats.failures == 1
    assert stats.failure_rate == pytest.approx(1 / 3)
    assert stats.rps == pytest.approx(1.0)
    assert stats.p50_ms == 20.0
    assert stats.p95_ms == 20.0
    assert stats.p99_ms == 20.0


def test_gateway_added_is_client_observed_difference() -> None:
    """Overhead is gateway minus mock, including a signed noisy p99."""
    added = gateway_added(
        _stats(p50_ms=40.0, p95_ms=50.0, p99_ms=8.0),
        _stats(p50_ms=10.0, p95_ms=12.0, p99_ms=10.0),
    )
    assert added == GatewayAddedLatency(p50_ms=30.0, p95_ms=38.0, p99_ms=-2.0)


def test_select_representative_run_keeps_one_whole_run() -> None:
    """The median Experiential p50 selects one internally consistent run."""
    runs = (
        _run(1, experiential_p50=50.0, litellm_p50=80.0),
        _run(2, experiential_p50=20.0, litellm_p50=90.0),
        _run(3, experiential_p50=30.0, litellm_p50=70.0),
    )
    chosen = select_representative_run(runs)
    assert chosen.run_index == 3
    assert chosen.experiential_added.p50_ms == 29.0
    assert chosen.litellm is not None
    assert chosen.litellm.p50_ms == 70.0
    with pytest.raises(ValueError, match="at least one"):
        select_representative_run(())


def test_functional_failures_fail_closed() -> None:
    """Any measured request failure is a hard report error."""
    with pytest.raises(RuntimeError, match="run 1 experiential"):
        _assert_functional_success((_run(1, experiential_p50=20.0, failures=1),))
    with pytest.raises(RuntimeError, match="run 1 litellm"):
        _assert_functional_success(
            (_run(1, experiential_p50=20.0, litellm_p50=30.0, litellm_failures=2),)
        )
    _assert_functional_success((_run(1, experiential_p50=20.0, litellm_p50=30.0),))


def test_render_markdown_states_same_host_comparison() -> None:
    """The job summary names hardware, both proxies, and the mock caveat."""
    mock = _stats(p50_ms=2.0, p95_ms=3.0, p99_ms=4.0, requests=40)
    experiential = _stats(p50_ms=32.0, p95_ms=40.0, p99_ms=50.0, requests=40)
    litellm = _stats(p50_ms=80.0, p95_ms=90.0, p99_ms=100.0, requests=40)
    report = LatencyReport(
        measured_at=datetime(2026, 8, 22, 18, 0, tzinfo=UTC),
        config=default_config(compare_litellm=True),
        runner=RunnerContext(
            commit_sha="abc123def456",
            python_version="3.12.11",
            platform_name="Linux",
            runner_name="GitHub Actions ubuntu-latest",
            runner_os="Linux",
            cpu_count=4,
            cpu_model="AMD EPYC",
            gateway_engine="rust",
            litellm_version="1.97.0",
            litellm_startup="litellm --config",
        ),
        representative_run=LatencyMeasuredRun(
            run_index=2,
            gateway_order=("litellm", "experiential"),
            mock_direct=mock,
            experiential=experiential,
            experiential_added=gateway_added(experiential, mock),
            litellm=litellm,
            litellm_added=gateway_added(litellm, mock),
            mock_direct_ttft=mock,
            experiential_ttft=experiential,
            experiential_added_ttft=gateway_added(experiential, mock),
            litellm_ttft=litellm,
            litellm_added_ttft=gateway_added(litellm, mock),
        ),
        runs=(
            _run(1, experiential_p50=32.0, litellm_p50=80.0),
            _run(2, experiential_p50=32.0, litellm_p50=80.0),
            _run(3, experiential_p50=40.0, litellm_p50=90.0),
        ),
    )
    markdown = render_markdown(report)
    assert CAVEAT in markdown
    assert "`abc123def456`" in markdown
    assert "rust" in markdown
    assert "experiential-added" in markdown
    assert "litellm-added" in markdown
    assert "4 x AMD EPYC" in markdown
    assert "litellm then experiential" in markdown
    assert "Streaming time to first token" in markdown
    assert "1K-RPS" in markdown
    parsed = LatencyReport.model_validate_json(report.model_dump_json())
    assert parsed.schema_name == SCHEMA_NAME
    assert parsed.schema_version == 2


def test_mock_server_answers_json_and_first_token() -> None:
    """The loopback mock serves non-streaming JSON and an immediate first token."""
    server = MockOpenAIServer()
    server.start()
    try:
        url = f"{server.base_url}/chat/completions"
        complete = httpx.post(url, json=chat_payload("latency-mock", stream=False), timeout=2)
        assert complete.status_code == 200
        assert complete.json()["choices"][0]["message"]["content"] == "hello"
        streamed = measure_arm(
            url=url,
            headers={"Content-Type": "application/json"},
            payload=chat_payload("latency-mock", stream=True),
            warmup=1,
            requests=4,
            concurrency=2,
            timeout_s=2.0,
            stream=True,
        )
        assert streamed.failures == 0
        assert streamed.requests == 4
        assert streamed.p50_ms >= 0.0
        non_stream = measure_arm(
            url=url,
            headers={"Content-Type": "application/json"},
            payload=chat_payload("latency-mock", stream=False),
            warmup=1,
            requests=4,
            concurrency=2,
            timeout_s=2.0,
            stream=False,
        )
        assert non_stream.failures == 0
        assert non_stream.rps > 0
    finally:
        server.stop()


def test_parse_args_keeps_ci_defaults() -> None:
    """The module CLI default schedule matches the CI preset."""
    args = parse_args([])
    config = default_config()
    assert args.warmup == config.warmup_requests
    assert args.requests == config.measured_requests
    assert args.concurrency == config.concurrency
    assert args.repeats == config.repeats
    assert args.stream_requests == config.stream_measured_requests
    assert args.no_stream_ttft is False
    assert args.compare_litellm is False
    compare = parse_args(["--compare-litellm"])
    assert compare.compare_litellm is True


def test_run_latency_report_against_local_mock(tmp_path: Path) -> None:
    """The product gateway and mock are measured in one process-local report."""
    report = run_latency_report(
        work_root=tmp_path,
        config=LatencyRunConfig(
            warmup_requests=1,
            measured_requests=2,
            concurrency=1,
            repeats=1,
            stream_warmup_requests=1,
            stream_measured_requests=2,
            stream_concurrency=1,
            timeout_s=10.0,
            measure_streaming_ttft=True,
            compare_litellm=False,
        ),
    )
    payload = json.loads(report.model_dump_json())
    assert payload["schema_name"] == SCHEMA_NAME
    assert payload["schema_version"] == SCHEMA_VERSION
    assert report.representative_run.mock_direct.failures == 0
    assert report.representative_run.experiential.failures == 0
    assert report.representative_run.experiential_ttft is not None
    assert report.representative_run.experiential_ttft.failures == 0
    assert report.representative_run.litellm is None
    assert report.runner.gateway_engine in {"rust", "python", "unknown"}
    assert "raw_key" not in payload
    assert "EXP_LATENCY_MOCK_KEY" not in json.dumps(payload)
