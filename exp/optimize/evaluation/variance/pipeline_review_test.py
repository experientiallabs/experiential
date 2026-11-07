"""Admitted compatibility settings stay bound through build and draw replay."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactInput, JsonObject
from exp.common.models import (
    AssistantAction,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    OperationEconomics,
)
from exp.common.project import ProjectStore
from exp.common.state_machine import TransitionJournal
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.variance.build import plan_variance_build, run_variance_build
from exp.optimize.evaluation.variance.build_test import FakeDecisions, _tagger, variance_project
from exp.optimize.evaluation.variance.family_scopes_test import spec as family_spec
from exp.optimize.evaluation.variance.pipeline_review import TRANSIENT_TYPE, family_reviewer
from exp.optimize.evaluation.variance.pipeline_spec import BuildSettings
from exp.runtime.models.providers.transport import ProviderTransportError
from exp.simulation.mining.environment_families_test import answer, reviewer
from exp.simulation.mining.review_states import TransientReviewFailure


def test_changed_reasoning_settings_rekey_build_family_and_review_draws(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stronger requested review cannot silently adopt the earlier model's verdict."""
    project = variance_project(tmp_path, monkeypatch)

    def compatible(members: frozenset[str], _variant: str) -> JsonObject:
        """Answer one compatible environment from the scripted evidence."""
        return answer("compatible", members)

    script, original = reviewer(compatible)
    settings = AuthorSettings(
        model_alias="reviewer",
        model=original.model,
        catalog_sha256="0" * 64,
        output_tokens=2000,
        reasoning_effort="low",
        deadline=datetime(2099, 1, 1, tzinfo=UTC),
    )
    calls: list[str | None] = []

    def offline_response(
        _project: ProjectStore,
        author: AuthorSettings,
        request: ModelRequest,
        *,
        identity: str,
        sources: tuple[ArtifactInput, ...],
        code_revision: str,
    ) -> tuple[None, ModelResponse]:
        """Serve a deterministic review while preserving the real owner-supplied identities."""
        del sources, code_revision
        calls.append(author.reasoning_effort)
        return None, script(request, identity=identity)

    monkeypatch.setattr(
        "exp.optimize.evaluation.variance.pipeline_review.author_response", offline_response
    )
    spec = family_spec()
    reports = []
    for author in (settings, settings.model_copy(update={"reasoning_effort": "high"})):
        bound = family_reviewer(
            project, BuildSettings(spec=spec, reviewer=author), code_revision="test"
        )
        plan = plan_variance_build(
            project,
            spec,
            code_revision="test",
            reviewer_model=bound.model,
            reviewer_configuration_sha256=bound.configuration_sha256,
        )
        assert not plan.saved
        report = run_variance_build(
            project,
            spec,
            code_revision="test",
            reviewer=bound,
            tagger=_tagger(project, FakeDecisions()),
        )
        assert report.identity == plan.identity
        count = len(calls)
        assert (
            run_variance_build(
                project,
                spec,
                code_revision="test",
                reviewer=bound,
                tagger=_tagger(project, FakeDecisions()),
            )
            == report
        )
        assert len(calls) == count
        reports.append(report)
    assert reports[0].identity != reports[1].identity
    assert reports[0].environment_families != reports[1].environment_families
    assert calls == ["low"] * 7 + ["high"] * 7
    assert len(script.saved) == 14


def test_a_transient_draw_failure_is_a_receipt_and_the_next_run_draws_a_fresh_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 503 saves a failed-attempt receipt and raises; the retry uses a fresh attempt identity."""
    project = ProjectStore(tmp_path, "case")
    _, original = reviewer(lambda members, _variant: answer("compatible", members))
    served = ModelResponse(
        output=AssistantAction(content="reviewed"),
        model=original.model,
        economics=OperationEconomics(),
    )
    settings = AuthorSettings(
        model_alias="reviewer",
        model=original.model,
        catalog_sha256="0" * 64,
        output_tokens=2000,
        deadline=datetime(2099, 1, 1, tzinfo=UTC),
    )
    attempts: list[str] = []

    def offline_response(
        _project: ProjectStore,
        _author: AuthorSettings,
        request: ModelRequest,
        *,
        identity: str,
        sources: tuple[ArtifactInput, ...],
        code_revision: str,
    ) -> tuple[None, ModelResponse]:
        """Answer 503 on the first attempt of every draw, then serve a completed response."""
        del request, sources, code_revision
        attempts.append(identity)
        if "-retry-" not in identity:
            raise ProviderTransportError("gateway busy", status_code=503)
        return None, served

    monkeypatch.setattr(
        "exp.optimize.evaluation.variance.pipeline_review.author_response", offline_response
    )
    bound = family_reviewer(
        project, BuildSettings(spec=family_spec(), reviewer=settings), code_revision="test"
    )
    request = ModelRequest(messages=(ModelMessage(role="user", content="{}"),))
    with pytest.raises(TransientReviewFailure, match="http_503"):
        bound.draw(request, identity="draw-a")
    receipt = project.artifacts.read("draw-a-transient-1")
    assert receipt.manifest.artifact_type == TRANSIENT_TYPE
    assert bound.draw(request, identity="draw-a") == served
    assert bound.draw(request, identity="draw-a") == served
    assert attempts == ["draw-a", "draw-a-retry-1", "draw-a-retry-1"]
    rows = TransitionJournal(
        project.artifacts.project_directory / "environment-review" / "draws.jsonl"
    ).rows()
    assert [row["event"] for row in rows] == ["failed_transiently", "answered", "answered"]


def test_an_exhausted_output_allowance_fails_that_review_draw_without_aborting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A length-limit rejection is a refused draw (ReviewerError), not a transient abort."""
    from exp.runtime.models.providers.errors import ProviderRetryableResponseError
    from exp.simulation.mining.environment_review import ReviewerError

    project = ProjectStore(tmp_path, "case")
    _, original = reviewer(lambda members, _variant: answer("compatible", members))
    settings = AuthorSettings(
        model_alias="reviewer",
        model=original.model,
        catalog_sha256="0" * 64,
        output_tokens=2000,
        deadline=datetime(2099, 1, 1, tzinfo=UTC),
    )

    def offline_response(*_args: object, **_kwargs: object) -> tuple[None, ModelResponse]:
        """Every draw ends at the output allowance, as a saved rejection replays."""
        raise ProviderRetryableResponseError("length", hit_length_limit=True)

    monkeypatch.setattr(
        "exp.optimize.evaluation.variance.pipeline_review.author_response", offline_response
    )
    bound = family_reviewer(
        project, BuildSettings(spec=family_spec(), reviewer=settings), code_revision="test"
    )
    request = ModelRequest(messages=(ModelMessage(role="user", content="{}"),))
    with pytest.raises(ReviewerError, match="2000-token output allowance"):
        bound.draw(request, identity="draw-b")
    assert not project.artifacts.exists("draw-b-transient-1")
    rows = TransitionJournal(
        project.artifacts.project_directory / "environment-review" / "draws.jsonl"
    ).rows()
    assert [row["event"] for row in rows] == ["refused"]
