"""Reject content-bearing or ambiguous observations before a metrics sink sees them."""

import json
import os
import subprocess
import sys

import pytest
from pydantic import ValidationError

from exp.common.observability.metrics import MetricRecord


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "1", True, {"text": "prompt"}])
def test_metrics_require_explicit_finite_numeric_values(value: object) -> None:
    """Reject invalid numeric payloads before any provider adapter can observe them."""
    with pytest.raises(ValidationError):
        MetricRecord.model_validate({"event_id": "sample-1", "values": {"train/loss": value}})


def test_domain_axes_do_not_require_a_shared_history_step() -> None:
    """Keep optimizer and evaluation axes independent of global history ordering."""
    record = MetricRecord(
        event_id="update-1", values={"train/loss": -0.5, "train/optimizer_step": 1}
    )
    assert record.step is None
    with pytest.raises(ValidationError):
        MetricRecord(event_id="bad", values={"loss": 1}, step=-1)
    with pytest.raises(ValidationError, match="cannot contain"):
        MetricRecord(event_id="bad", values={"api_key": 123})


def test_default_metric_and_controller_imports_do_not_load_wandb() -> None:
    """Keep provider dependencies absent from the default learner import path."""
    command = (
        "import json,sys; "
        "import exp.common.observability.metrics; "
        "import exp.optimize.claas.service.controller; "
        "print(json.dumps([name for name in sys.modules if name.split('.')[0]=='wandb']))"
    )
    result = subprocess.run(
        [sys.executable, "-c", command],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
        env={**os.environ, "EXP_TELEMETRY_ENABLED": "false"},
    )
    assert json.loads(result.stdout) == []
