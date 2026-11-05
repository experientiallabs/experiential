"""The actual default tagger's native local ledger covers its full planned decision work."""

from pathlib import Path

import pytest

from exp.common.core.artifacts import sha256_json
from exp.optimize.evaluation.variance.pipeline_errors import (
    error_tag_costs,
    traceml_decision_attempts,
    traceml_label_attempts,
)
from exp.optimize.evaluation.variance.pipeline_spec import StageContext
from exp.optimize.evaluation.variance.pipeline_stages import _tagger
from exp.optimize.evaluation.variance.pipeline_test import make_spec, wired
from exp.runtime.models.providers.transport import RetryPolicy
from exp.runtime.sessions.dispatch_ledger import DispatchLedger
from exp.simulation.mining.tagging import TraceTaggingError


def test_real_tagger_ledger_counts_retries_chunks_and_whole_attempt_error_tags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A chat-only test alias stops before dispatch but still exercises real ledger composition."""
    run = wired(tmp_path, monkeypatch)
    spec = make_spec(run.fixture, run.rubric, run.bank)
    before = run.calls()
    context = StageContext(
        project=run.project, spec=spec, identity="native-tagger-bound", code_revision="offline"
    )
    steps = spec.mining.attempts * spec.mining.registered.maximum_model_calls
    steps *= spec.traceml.steps_per_call
    labels = traceml_label_attempts(spec)
    assert labels >= steps * RetryPolicy().maximum_attempts
    errors = sum(row.attempts for row in error_tag_costs(run.project, spec))
    assert errors > 0
    expected = len(spec.mining.actors) * labels + errors
    assert traceml_decision_attempts(run.project, spec) == expected
    # The fixture intentionally has no native Jev wire; no replacement client is invented.
    with pytest.raises(TraceTaggingError, match="native decision client|decision endpoint"):
        _tagger(context)
    path = run.project.artifacts.project_directory / (context.identity + "-traceml.sqlite")
    assert path.is_file()
    DispatchLedger(path, expected, sha256_json(context.identity + "-traceml"))
    assert run.calls() == before
