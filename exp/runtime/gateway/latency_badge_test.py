"""Tests for the Shields gateway-overhead badge payload."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from exp.runtime.gateway.latency_badge import (
    BADGE_FILENAME,
    RAW_ENDPOINT_URL,
    SHIELDS_COLOR,
    SHIELDS_IMAGE_URL,
    SHIELDS_LABEL,
    format_overhead_ms,
    overhead_p50_ms_from_report_json,
    shields_endpoint,
    write_shields_endpoint,
)


def test_format_overhead_ms_uses_one_decimal() -> None:
    """The badge message rounds the representative p50 to one decimal place."""
    assert format_overhead_ms(14.58) == "14.6 ms"
    assert format_overhead_ms(3.0) == "3.0 ms"
    assert format_overhead_ms(-1.24) == "-1.2 ms"


def test_shields_endpoint_is_schema_version_one() -> None:
    """The payload is a Shields endpoint with a numeric millisecond message."""
    payload = shields_endpoint(p50_ms=14.58)
    assert payload == {
        "schemaVersion": 1,
        "label": SHIELDS_LABEL,
        "message": "14.6 ms",
        "color": SHIELDS_COLOR,
        "cacheSeconds": 300,
    }
    assert SHIELDS_LABEL == "gateway overhead"
    message = payload["message"]
    assert isinstance(message, str)
    assert message.endswith(" ms")
    assert "badge.svg" not in SHIELDS_IMAGE_URL
    assert BADGE_FILENAME in RAW_ENDPOINT_URL
    assert "img.shields.io/endpoint" in SHIELDS_IMAGE_URL


def test_write_shields_endpoint_round_trips(tmp_path: Path) -> None:
    """The written file is pretty-printed Shields JSON sourced from p50_ms."""
    path = tmp_path / "nested" / BADGE_FILENAME
    written = write_shields_endpoint(p50_ms=22.204, path=path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == written
    assert loaded["message"] == "22.2 ms"


def test_overhead_p50_ms_from_report_json_reads_representative_run(tmp_path: Path) -> None:
    """Badge generation reads p50 from the versioned report, not a constant."""
    report = tmp_path / "gateway-latency.json"
    report.write_text(
        json.dumps(
            {
                "representative_run": {"gateway_added": {"p50_ms": 8.44}},
            }
        ),
        encoding="utf-8",
    )
    assert overhead_p50_ms_from_report_json(report) == pytest.approx(8.44)
    payload = write_shields_endpoint(
        p50_ms=overhead_p50_ms_from_report_json(report),
        path=tmp_path / BADGE_FILENAME,
    )
    assert payload["message"] == "8.4 ms"


def test_overhead_p50_ms_from_report_json_rejects_missing_fields(tmp_path: Path) -> None:
    """A report without a numeric representative p50 fails closed."""
    report = tmp_path / "bad.json"
    report.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        overhead_p50_ms_from_report_json(report)
    report.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="representative_run"):
        overhead_p50_ms_from_report_json(report)
    report.write_text(
        json.dumps({"representative_run": {"gateway_added": {"p50_ms": "fast"}}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must be a number"):
        overhead_p50_ms_from_report_json(report)


def test_readme_uses_shields_endpoint_not_actions_status() -> None:
    """The root README renders the numeric Shields badge, not a workflow status."""
    readme = Path(__file__).resolve().parents[3] / "README.md"
    text = readme.read_text(encoding="utf-8")
    assert SHIELDS_IMAGE_URL in text
    assert RAW_ENDPOINT_URL.split("https://")[1] in text or "endpoint?url=" in text
    assert "actions/workflows/gateway-latency.yml/badge.svg" not in text
