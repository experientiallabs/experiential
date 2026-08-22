"""Measure Experiential and LiteLLM overhead against one local mock.

The runner starts one loopback OpenAI-compatible mock, then measures that mock
directly, the Experiential native gateway, and (when requested) a pinned
LiteLLM config-file proxy. All three arms use the same payload, schedule,
warmup, concurrency, and sampler. Gateway measurement order rotates across
repeats so neither proxy systematically benefits from first position.

Reported added latency is the client-observed difference versus the mock
direct baseline. This is a same-host mock-upstream comparison, not end-to-end
model latency and not LiteLLM's public 1K-RPS topology.

Methodology follows the official LiteLLM mock-isolated scripts: warmup
requests are discarded, measured runs use fixed request and concurrency
counts, arms run sequentially, and the representative result is the median
run by Experiential p50.
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field
from rich.console import Console

from exp.common.core.artifacts import ContractModel, JsonObject
from exp.runtime.gateway.latency_litellm import (
    LITELLM_MASTER_KEY,
    LITELLM_PIN,
    LITELLM_STARTUP,
    installed_litellm_version,
    start_litellm_process,
    stop_litellm_process,
    write_litellm_config,
)
from exp.runtime.gateway.latency_measure import (
    ALIAS_ID,
    CHAT_PATH,
    MOCK_MODEL,
    MockOpenAIServer,
    RequestSample,
    chat_payload,
    collect_arm_samples,
    configure_gateway,
    percentile,
    start_gateway_process,
    stop_gateway_process,
    unused_loopback_port,
)

SCHEMA_NAME = "exp.gateway.latency_report"
SCHEMA_VERSION = 2
CAVEAT = (
    "This report is a same-host mock-upstream comparison of Experiential and "
    "LiteLLM against one local OpenAI-compatible mock. It is not end-to-end "
    "model latency and is not LiteLLM's public 1K-RPS multi-worker topology."
)
_DEFAULT_WARMUP = 10
_DEFAULT_REQUESTS = 40
_DEFAULT_CONCURRENCY = 8
_DEFAULT_REPEATS = 3
_DEFAULT_STREAM_WARMUP = 5
_DEFAULT_STREAM_REQUESTS = 20
_DEFAULT_STREAM_CONCURRENCY = 4
_DEFAULT_TIMEOUT_S = 10.0
_ARM_TABLE_HEADER = (
    "| Arm | Requests | Failures | Failure rate | RPS | p50 (ms) | p95 (ms) | p99 (ms) |"
)
_ARM_TABLE_DIVIDER = "|---|---:|---:|---:|---:|---:|---:|---:|"
_EXPERIENTIAL = "experiential"
_LITELLM = "litellm"


class LatencyArmStats(ContractModel):
    """Aggregated latency, throughput, and failure counts for one measured arm."""

    requests: int = Field(ge=0)
    failures: int = Field(ge=0)
    failure_rate: float = Field(ge=0.0, le=1.0)
    rps: float = Field(ge=0.0)
    mean_ms: float = Field(ge=0.0)
    p50_ms: float = Field(ge=0.0)
    p95_ms: float = Field(ge=0.0)
    p99_ms: float = Field(ge=0.0)


class GatewayAddedLatency(ContractModel):
    """Client-observed gateway overhead as gateway minus mock-direct percentiles."""

    p50_ms: float
    p95_ms: float
    p99_ms: float


class LatencyRunConfig(ContractModel):
    """Fixed request schedule used by every arm in one report."""

    warmup_requests: int = Field(ge=0)
    measured_requests: int = Field(ge=1)
    concurrency: int = Field(ge=1)
    repeats: int = Field(ge=1)
    stream_warmup_requests: int = Field(ge=0)
    stream_measured_requests: int = Field(ge=1)
    stream_concurrency: int = Field(ge=1)
    timeout_s: float = Field(gt=0.0)
    measure_streaming_ttft: bool
    compare_litellm: bool = False


class RunnerContext(ContractModel):
    """Host and runtime facts recorded beside the measurements."""

    commit_sha: str = Field(min_length=1)
    python_version: str = Field(min_length=1)
    platform_name: str = Field(min_length=1)
    runner_name: str = Field(min_length=1)
    gateway_engine: str = Field(min_length=1)
    runner_os: str = Field(min_length=1, default="unknown")
    cpu_count: int = Field(ge=0, default=0)
    cpu_model: str = Field(min_length=1, default="unknown")
    litellm_version: str | None = None
    litellm_startup: str | None = None


class LatencyMeasuredRun(ContractModel):
    """One sequential pass of mock-direct and rotated gateway arms."""

    run_index: int = Field(ge=1)
    gateway_order: tuple[str, ...]
    mock_direct: LatencyArmStats
    experiential: LatencyArmStats
    experiential_added: GatewayAddedLatency
    litellm: LatencyArmStats | None = None
    litellm_added: GatewayAddedLatency | None = None
    mock_direct_ttft: LatencyArmStats | None = None
    experiential_ttft: LatencyArmStats | None = None
    experiential_added_ttft: GatewayAddedLatency | None = None
    litellm_ttft: LatencyArmStats | None = None
    litellm_added_ttft: GatewayAddedLatency | None = None


class LatencyReport(ContractModel):
    """Versioned machine-readable same-run gateway comparison report."""

    schema_name: Literal["exp.gateway.latency_report"] = SCHEMA_NAME
    schema_version: Literal[2] = SCHEMA_VERSION
    measured_at: datetime
    caveat: str = CAVEAT
    config: LatencyRunConfig
    runner: RunnerContext
    representative_run: LatencyMeasuredRun
    runs: tuple[LatencyMeasuredRun, ...]


def summarize_samples(samples: tuple[RequestSample, ...], wall_time_s: float) -> LatencyArmStats:
    """Aggregate one arm from completed samples and the measured wall time.

    Args:
        samples: Completed request outcomes, including failures.
        wall_time_s: Elapsed seconds covering only the measured requests.

    Returns:
        Throughput, failure rate, and latency percentiles over successes.
    """
    successes = tuple(sample.latency_ms for sample in samples if sample.success)
    failures = len(samples) - len(successes)
    return LatencyArmStats(
        requests=len(samples),
        failures=failures,
        failure_rate=(failures / len(samples)) if samples else 1.0,
        rps=(len(successes) / wall_time_s) if wall_time_s > 0 else 0.0,
        mean_ms=(sum(successes) / len(successes)) if successes else 0.0,
        p50_ms=percentile(successes, 50),
        p95_ms=percentile(successes, 95),
        p99_ms=percentile(successes, 99),
    )


def gateway_added(gateway: LatencyArmStats, mock_direct: LatencyArmStats) -> GatewayAddedLatency:
    """Subtract mock-direct percentiles from gateway percentiles.

    Args:
        gateway: Client-observed gateway arm.
        mock_direct: Client-observed mock arm.

    Returns:
        Signed overhead. A negative value means measurement noise dominated.
    """
    return GatewayAddedLatency(
        p50_ms=gateway.p50_ms - mock_direct.p50_ms,
        p95_ms=gateway.p95_ms - mock_direct.p95_ms,
        p99_ms=gateway.p99_ms - mock_direct.p99_ms,
    )


def comparison_order(run_index: int, *, compare_litellm: bool) -> tuple[str, ...]:
    """Return the gateway measurement order for one 1-based repeat.

    Odd runs measure Experiential first. Even runs measure LiteLLM first when
    the comparison is enabled, so position does not systematically favor one
    proxy.

    Args:
        run_index: 1-based repeat number.
        compare_litellm: Whether LiteLLM is part of this report.

    Returns:
        Ordered gateway arm names.
    """
    if not compare_litellm:
        return (_EXPERIENTIAL,)
    if run_index % 2 == 1:
        return (_EXPERIENTIAL, _LITELLM)
    return (_LITELLM, _EXPERIENTIAL)


def resolve_commit_sha() -> str:
    """Return ``GITHUB_SHA`` or the current checkout SHA.

    Returns:
        Full or abbreviated revision, or ``unknown`` when Git is unavailable.
    """
    env_sha = os.environ.get("GITHUB_SHA", "").strip()
    if env_sha:
        return env_sha
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return "unknown"
    sha = result.stdout.strip()
    return sha if result.returncode == 0 and sha else "unknown"


def runner_cpu_model() -> str:
    """Return the host CPU model from ``/proc/cpuinfo`` when present.

    Returns:
        CPU model string, or a platform fallback.
    """
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                model = line.split(":", 1)[1].strip()
                if model:
                    return model
    return platform.processor() or "unknown"


def render_markdown(report: LatencyReport) -> str:
    """Render the GitHub Actions job summary for one completed report.

    Args:
        report: Versioned latency report.

    Returns:
        Compact Markdown with context, the representative table, and the caveat.
    """
    run = report.representative_run
    config = report.config
    runner = report.runner
    lines = [
        "# Gateway latency report",
        "",
        report.caveat,
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Commit | `{runner.commit_sha}` |",
        f"| Runner | {runner.runner_name} |",
        f"| OS | {runner.runner_os} |",
        f"| Platform | {runner.platform_name} |",
        f"| CPU | {runner.cpu_count} x {runner.cpu_model} |",
        f"| Python | {runner.python_version} |",
        f"| Experiential engine | {runner.gateway_engine} |",
        f"| Schema | `{report.schema_name}` v{report.schema_version} |",
        (
            f"| Non-stream schedule | warmup {config.warmup_requests}, "
            f"{config.measured_requests} requests, concurrency "
            f"{config.concurrency}, {config.repeats} runs |"
        ),
        f"| Gateway order | {' then '.join(run.gateway_order)} |",
        f"| Representative run | {run.run_index} of {len(report.runs)} (median experiential p50) |",
        f"| Measured at | {report.measured_at.isoformat()} |",
    ]
    if runner.litellm_version:
        lines.append(f"| LiteLLM | {runner.litellm_version} |")
    if runner.litellm_startup:
        lines.append(f"| LiteLLM startup | {runner.litellm_startup} |")
    lines.extend(
        [
            "",
            "## Non-streaming chat completions",
            "",
            _ARM_TABLE_HEADER,
            _ARM_TABLE_DIVIDER,
            _arm_row("mock direct", run.mock_direct),
            _arm_row("experiential", run.experiential),
            _added_row("experiential-added", run.experiential_added),
        ]
    )
    if run.litellm is not None and run.litellm_added is not None:
        lines.append(_arm_row("litellm", run.litellm))
        lines.append(_added_row("litellm-added", run.litellm_added))
    if run.mock_direct_ttft is not None and run.experiential_ttft is not None:
        lines.extend(
            [
                "",
                "## Streaming time to first token",
                "",
                (
                    f"Schedule: warmup {config.stream_warmup_requests}, "
                    f"{config.stream_measured_requests} requests, concurrency "
                    f"{config.stream_concurrency}."
                ),
                "",
                _ARM_TABLE_HEADER,
                _ARM_TABLE_DIVIDER,
                _arm_row("mock direct TTFT", run.mock_direct_ttft),
                _arm_row("experiential TTFT", run.experiential_ttft),
            ]
        )
        if run.experiential_added_ttft is not None:
            lines.append(_added_row("experiential-added TTFT", run.experiential_added_ttft))
        if run.litellm_ttft is not None:
            lines.append(_arm_row("litellm TTFT", run.litellm_ttft))
        if run.litellm_added_ttft is not None:
            lines.append(_added_row("litellm-added TTFT", run.litellm_added_ttft))
    return "\n".join(lines) + "\n"


def _arm_row(name: str, stats: LatencyArmStats) -> str:
    """Format one Markdown table row for a measured arm.

    Args:
        name: Arm label.
        stats: Aggregated samples.

    Returns:
        One Markdown table row.
    """
    return (
        f"| {name} | {stats.requests} | {stats.failures} | "
        f"{stats.failure_rate:.1%} | {stats.rps:.2f} | {stats.p50_ms:.2f} | "
        f"{stats.p95_ms:.2f} | {stats.p99_ms:.2f} |"
    )


def _added_row(name: str, added: GatewayAddedLatency) -> str:
    """Format one Markdown row for gateway-added percentiles.

    Args:
        name: Row label.
        added: Signed overhead versus the mock-direct baseline.

    Returns:
        One Markdown table row.
    """
    return f"| {name} |  |  |  |  | {added.p50_ms:.2f} | {added.p95_ms:.2f} | {added.p99_ms:.2f} |"


def measure_arm(
    *,
    url: str,
    headers: dict[str, str],
    payload: JsonObject,
    warmup: int,
    requests: int,
    concurrency: int,
    timeout_s: float,
    stream: bool,
) -> LatencyArmStats:
    """Warm up, then measure one sequential arm at a fixed concurrency.

    Args:
        url: Absolute chat-completions URL.
        headers: Authorization and content-type headers.
        payload: Shared JSON body.
        warmup: Discarded requests run before the timed window.
        requests: Timed request count.
        concurrency: Maximum in-flight requests.
        timeout_s: Per-request timeout.
        stream: Whether to stop at the first content token.

    Returns:
        Aggregated arm statistics.
    """
    samples, wall_time_s = collect_arm_samples(
        url=url,
        headers=headers,
        payload=payload,
        warmup=warmup,
        requests=requests,
        concurrency=concurrency,
        timeout_s=timeout_s,
        stream=stream,
    )
    return summarize_samples(samples, wall_time_s)


def _measure_named_arm(
    *,
    name: str,
    targets: dict[str, tuple[str, dict[str, str]]],
    config: LatencyRunConfig,
    stream: bool,
) -> LatencyArmStats:
    """Measure one named gateway or skip-safe target.

    Args:
        name: Arm name from :func:`comparison_order`.
        targets: Mapping of arm name to URL and headers.
        config: Fixed request schedule.
        stream: Whether to stop at the first content token.

    Returns:
        Aggregated arm statistics.
    """
    url, headers = targets[name]
    return measure_arm(
        url=url,
        headers=headers,
        payload=chat_payload(ALIAS_ID if name != "mock" else MOCK_MODEL, stream=stream),
        warmup=(config.stream_warmup_requests if stream else config.warmup_requests),
        requests=(config.stream_measured_requests if stream else config.measured_requests),
        concurrency=(config.stream_concurrency if stream else config.concurrency),
        timeout_s=config.timeout_s,
        stream=stream,
    )


def _measure_run(
    *,
    run_index: int,
    config: LatencyRunConfig,
    mock_url: str,
    mock_headers: dict[str, str],
    targets: dict[str, tuple[str, dict[str, str]]],
) -> LatencyMeasuredRun:
    """Run mock-direct, then the rotated gateway arms, for one repeat.

    Args:
        run_index: 1-based repeat number.
        config: Fixed request schedule.
        mock_url: Direct mock chat-completions URL.
        mock_headers: Headers for the mock.
        targets: Experiential and optional LiteLLM URL/header pairs.

    Returns:
        One complete measured run.
    """
    order = comparison_order(run_index, compare_litellm=config.compare_litellm)
    mock_direct = measure_arm(
        url=mock_url,
        headers=mock_headers,
        payload=chat_payload(MOCK_MODEL, stream=False),
        warmup=config.warmup_requests,
        requests=config.measured_requests,
        concurrency=config.concurrency,
        timeout_s=config.timeout_s,
        stream=False,
    )
    measured: dict[str, LatencyArmStats] = {}
    for name in order:
        measured[name] = _measure_named_arm(
            name=name,
            targets=targets,
            config=config,
            stream=False,
        )
    experiential = measured[_EXPERIENTIAL]
    litellm_stats = measured.get(_LITELLM)
    mock_ttft: LatencyArmStats | None = None
    ttft: dict[str, LatencyArmStats] = {}
    if config.measure_streaming_ttft:
        mock_ttft = measure_arm(
            url=mock_url,
            headers=mock_headers,
            payload=chat_payload(MOCK_MODEL, stream=True),
            warmup=config.stream_warmup_requests,
            requests=config.stream_measured_requests,
            concurrency=config.stream_concurrency,
            timeout_s=config.timeout_s,
            stream=True,
        )
        for name in order:
            ttft[name] = _measure_named_arm(
                name=name,
                targets=targets,
                config=config,
                stream=True,
            )
    experiential_ttft = ttft.get(_EXPERIENTIAL)
    litellm_ttft = ttft.get(_LITELLM)
    return LatencyMeasuredRun(
        run_index=run_index,
        gateway_order=order,
        mock_direct=mock_direct,
        experiential=experiential,
        experiential_added=gateway_added(experiential, mock_direct),
        litellm=litellm_stats,
        litellm_added=(
            gateway_added(litellm_stats, mock_direct) if litellm_stats is not None else None
        ),
        mock_direct_ttft=mock_ttft,
        experiential_ttft=experiential_ttft,
        experiential_added_ttft=(
            gateway_added(experiential_ttft, mock_ttft)
            if experiential_ttft is not None and mock_ttft is not None
            else None
        ),
        litellm_ttft=litellm_ttft,
        litellm_added_ttft=(
            gateway_added(litellm_ttft, mock_ttft)
            if litellm_ttft is not None and mock_ttft is not None
            else None
        ),
    )


def select_representative_run(runs: tuple[LatencyMeasuredRun, ...]) -> LatencyMeasuredRun:
    """Return the run whose Experiential non-stream p50 is the median.

    Choosing one whole run keeps related arms in the same execution context.

    Args:
        runs: Completed repeats. Must not be empty.

    Returns:
        Median run by Experiential p50.

    Raises:
        ValueError: ``runs`` is empty.
    """
    if not runs:
        raise ValueError("at least one measured run is required")
    ordered = sorted(runs, key=lambda item: item.experiential.p50_ms)
    return ordered[len(ordered) // 2]


def run_latency_report(
    *,
    work_root: Path,
    config: LatencyRunConfig,
    mock_credential: str = "latency-mock-credential",
) -> LatencyReport:
    """Configure, serve, measure, and return one versioned latency report.

    Args:
        work_root: Directory that receives the temporary EXP root.
        config: Fixed request schedule.
        mock_credential: Value placed in the mock provider environment.

    Returns:
        Completed report for the representative (median) run plus every repeat.

    Raises:
        RuntimeError: A gateway fails to start or a measured arm has failures.
    """
    mock = MockOpenAIServer()
    mock.start()
    process: subprocess.Popen[str] | None = None
    litellm_process: subprocess.Popen[str] | None = None
    try:
        raw_key = configure_gateway(work_root, provider_base_url=mock.base_url)
        port = unused_loopback_port()
        process, engine = start_gateway_process(
            root=work_root,
            port=port,
            credential=mock_credential,
        )
        mock_url = f"{mock.base_url}/chat/completions"
        mock_headers = {
            "Authorization": f"Bearer {mock_credential}",
            "Content-Type": "application/json",
        }
        targets: dict[str, tuple[str, dict[str, str]]] = {
            _EXPERIENTIAL: (
                f"http://127.0.0.1:{port}{CHAT_PATH}",
                {
                    "Authorization": f"Bearer {raw_key}",
                    "Content-Type": "application/json",
                },
            )
        }
        litellm_version: str | None = None
        litellm_startup: str | None = None
        if config.compare_litellm:
            config_path = work_root / "litellm.yaml"
            write_litellm_config(
                config_path,
                api_base=mock.base_url,
                api_key=mock_credential,
            )
            litellm_process, litellm_port = start_litellm_process(config_path=config_path)
            targets[_LITELLM] = (
                f"http://127.0.0.1:{litellm_port}{CHAT_PATH}",
                {
                    "Authorization": f"Bearer {LITELLM_MASTER_KEY}",
                    "Content-Type": "application/json",
                },
            )
            litellm_version = installed_litellm_version()
            litellm_startup = LITELLM_STARTUP
        runs = tuple(
            _measure_run(
                run_index=index,
                config=config,
                mock_url=mock_url,
                mock_headers=mock_headers,
                targets=targets,
            )
            for index in range(1, config.repeats + 1)
        )
        representative = select_representative_run(runs)
        _assert_functional_success(runs)
        return LatencyReport(
            measured_at=datetime.now(UTC),
            config=config,
            runner=RunnerContext(
                commit_sha=resolve_commit_sha(),
                python_version=platform.python_version(),
                platform_name=platform.platform(),
                runner_name=os.environ.get("RUNNER_NAME") or platform.node() or "unknown",
                runner_os=os.environ.get("RUNNER_OS") or platform.system() or "unknown",
                cpu_count=os.cpu_count() or 0,
                cpu_model=runner_cpu_model(),
                gateway_engine=engine,
                litellm_version=litellm_version,
                litellm_startup=litellm_startup,
            ),
            representative_run=representative,
            runs=runs,
        )
    finally:
        if litellm_process is not None:
            stop_litellm_process(litellm_process)
        if process is not None:
            stop_gateway_process(process)
        mock.stop()


def _assert_functional_success(runs: tuple[LatencyMeasuredRun, ...]) -> None:
    """Fail closed when any measured request failed.

    Latency differences never fail the report. Only HTTP or parse failures do.

    Args:
        runs: Completed repeats.

    Raises:
        RuntimeError: Any arm recorded a failure.
    """
    failures: list[str] = []
    for run in runs:
        arms = (
            ("mock_direct", run.mock_direct),
            ("experiential", run.experiential),
            ("litellm", run.litellm),
            ("mock_direct_ttft", run.mock_direct_ttft),
            ("experiential_ttft", run.experiential_ttft),
            ("litellm_ttft", run.litellm_ttft),
        )
        for name, stats in arms:
            if stats is not None and stats.failures:
                failures.append(f"run {run.run_index} {name}: {stats.failures} failed")
    if failures:
        raise RuntimeError("latency report functional failures: " + "; ".join(failures))


def default_config(
    *,
    measure_streaming_ttft: bool = True,
    compare_litellm: bool = False,
) -> LatencyRunConfig:
    """Return the CI schedule used by the GitHub Actions workflow.

    Args:
        measure_streaming_ttft: Whether to include the streaming TTFT arms.
        compare_litellm: Whether to start and measure the pinned LiteLLM proxy.

    Returns:
        Fixed small request schedule.
    """
    return LatencyRunConfig(
        warmup_requests=_DEFAULT_WARMUP,
        measured_requests=_DEFAULT_REQUESTS,
        concurrency=_DEFAULT_CONCURRENCY,
        repeats=_DEFAULT_REPEATS,
        stream_warmup_requests=_DEFAULT_STREAM_WARMUP,
        stream_measured_requests=_DEFAULT_STREAM_REQUESTS,
        stream_concurrency=_DEFAULT_STREAM_CONCURRENCY,
        timeout_s=_DEFAULT_TIMEOUT_S,
        measure_streaming_ttft=measure_streaming_ttft,
        compare_litellm=compare_litellm,
    )


def write_report_outputs(
    report: LatencyReport,
    *,
    output_json: Path | None,
    github_summary: Path | None,
    console: Console,
) -> None:
    """Write the JSON artifact, optional job summary, and console Markdown.

    Args:
        report: Completed report.
        output_json: Optional destination for the versioned JSON artifact.
        github_summary: Optional GitHub Actions step-summary path.
        console: User-facing console for the Markdown table.
    """
    markdown = render_markdown(report)
    console.print(markdown, markup=False)
    if output_json is not None:
        output_json.write_text(
            report.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
    if github_summary is not None:
        github_summary.write_text(markdown, encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the module CLI used by CI and local operators.

    Args:
        argv: Optional argument vector. ``None`` reads ``sys.argv``.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Write the versioned JSON report to this path.",
    )
    parser.add_argument(
        "--github-summary",
        type=Path,
        help="Write the Markdown job summary to this path.",
    )
    parser.add_argument("--warmup", type=int, default=_DEFAULT_WARMUP)
    parser.add_argument("--requests", type=int, default=_DEFAULT_REQUESTS)
    parser.add_argument("--concurrency", type=int, default=_DEFAULT_CONCURRENCY)
    parser.add_argument("--repeats", type=int, default=_DEFAULT_REPEATS)
    parser.add_argument("--stream-warmup", type=int, default=_DEFAULT_STREAM_WARMUP)
    parser.add_argument("--stream-requests", type=int, default=_DEFAULT_STREAM_REQUESTS)
    parser.add_argument("--stream-concurrency", type=int, default=_DEFAULT_STREAM_CONCURRENCY)
    parser.add_argument("--timeout", type=float, default=_DEFAULT_TIMEOUT_S)
    parser.add_argument(
        "--no-stream-ttft",
        action="store_true",
        help="Skip the streaming time-to-first-token arms.",
    )
    parser.add_argument(
        "--compare-litellm",
        action="store_true",
        help=f"Measure pinned LiteLLM[proxy] {LITELLM_PIN} in the same job.",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        help="EXP root used for the temporary gateway. Defaults to a temp dir.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the report and write artifacts.

    Args:
        argv: Optional argument vector.

    Returns:
        Process exit status. Functional request failures return 1.
    """
    args = parse_args(argv)
    console = Console()
    config = LatencyRunConfig(
        warmup_requests=args.warmup,
        measured_requests=args.requests,
        concurrency=args.concurrency,
        repeats=args.repeats,
        stream_warmup_requests=args.stream_warmup,
        stream_measured_requests=args.stream_requests,
        stream_concurrency=args.stream_concurrency,
        timeout_s=args.timeout,
        measure_streaming_ttft=not args.no_stream_ttft,
        compare_litellm=args.compare_litellm,
    )
    summary = args.github_summary
    if summary is None:
        env_summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if env_summary:
            summary = Path(env_summary)
    if args.work_root is not None:
        return _run_and_write(args.work_root, config, args.output_json, summary, console)
    with tempfile.TemporaryDirectory(prefix="exp-gateway-latency-") as tmp_dir:
        return _run_and_write(Path(tmp_dir), config, args.output_json, summary, console)


def _run_and_write(
    work_root: Path,
    config: LatencyRunConfig,
    output_json: Path | None,
    github_summary: Path | None,
    console: Console,
) -> int:
    """Execute one report and persist operator-facing outputs.

    Args:
        work_root: EXP root for the temporary gateway.
        config: Fixed request schedule.
        output_json: Optional JSON artifact path.
        github_summary: Optional GitHub Actions summary path.
        console: User-facing console.

    Returns:
        Process exit status.
    """
    try:
        report = run_latency_report(work_root=work_root, config=config)
    except RuntimeError as exc:
        console.print(str(exc), markup=False)
        return 1
    write_report_outputs(
        report,
        output_json=output_json,
        github_summary=github_summary,
        console=console,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
