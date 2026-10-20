"""Shared author calls retain exact requests and finite dispatch authority."""

from datetime import timedelta
from pathlib import Path

import pytest

from exp.optimize.evaluation.authoring import author_response
from exp.optimize.evaluation.task_proposals import proposal_request, propose_tasks
from exp.optimize.evaluation.task_proposals_test import proposal_fixture


def test_saved_author_response_cannot_be_rebound_to_another_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reusing an identity with changed context refuses instead of relabeling a saved draw."""
    project, spec, transport = proposal_fixture(tmp_path, monkeypatch)
    propose_tasks(project, spec, code_revision="test")
    request = proposal_request(project, spec)
    assert request.maximum_output_tokens is not None
    changed = request.model_copy(
        update={"maximum_output_tokens": request.maximum_output_tokens + 1}
    )
    with pytest.raises(ValueError, match="original request"):
        author_response(
            project,
            spec,
            changed,
            identity=spec.identity("test"),
            sources=(spec.base, spec.brief),
            code_revision="test",
        )
    assert len(transport.requests) == 1


def test_expired_author_has_no_intent_or_provider_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deadlines block new dispatch but do not consume a draw or falsely create ambiguity."""
    project, spec, transport = proposal_fixture(tmp_path, monkeypatch)
    spec = spec.model_copy(update={"deadline": spec.deadline - timedelta(days=1)})
    with pytest.raises(ValueError, match="deadline expired"):
        propose_tasks(project, spec, code_revision="test")
    assert not transport.requests
    assert not project.artifacts.exists(spec.identity("test") + "-intent")
