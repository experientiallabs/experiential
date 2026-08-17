"""Build-cost ceiling command tests."""

from pathlib import Path

from wmo.cli.build_cost import over_ceiling_message, sufficient_ceiling_usd


def test_over_ceiling_command_quotes_paths_and_covers_full_precision() -> None:
    """The suggested command quotes paths and covers the exact estimate."""
    estimate = 1.2345674
    message = over_ceiling_message(
        estimate=estimate,
        ceiling=0.01,
        project="support",
        trace_file=Path("/tmp/my traces/export.jsonl"),
        source="otlp",
        root=Path("/tmp/wmo root"),
        world_model=None,
        judge=None,
        embedder=None,
        top_k=5,
    )

    assert "conservative embedding estimate $1.234567 exceeds" in message
    assert "--traces '/tmp/my traces/export.jsonl'" in message
    assert "--root '/tmp/wmo root'" in message
    sufficient = sufficient_ceiling_usd(estimate)
    assert float(sufficient) >= estimate
    assert f"--max-build-cost-usd {sufficient}" in message
