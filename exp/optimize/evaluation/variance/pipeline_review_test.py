"""Admitted compatibility settings stay bound through build and draw replay."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactInput, JsonObject
from exp.common.models import ModelRequest, ModelResponse
from exp.common.project import ProjectStore
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.variance.build import plan_variance_build, run_variance_build
from exp.optimize.evaluation.variance.build_test import FakeDecisions, _tagger, variance_project
from exp.optimize.evaluation.variance.family_scopes_test import spec as family_spec
from exp.optimize.evaluation.variance.pipeline_review import family_reviewer
from exp.optimize.evaluation.variance.pipeline_spec import BuildSettings
from exp.simulation.mining.environment_families_test import answer, reviewer


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


def test_only_named_family_review_receives_explicit_retry_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Operational retry registration leaves the semantic reviewer configuration unchanged."""
    from exp.common.core.artifacts import sha256_json
    from exp.common.models import ModelMessage
    from exp.optimize.evaluation.variance.evidence_tools_test import _response

    project = variance_project(tmp_path, monkeypatch)
    _, original = reviewer(lambda members, variant: answer("compatible", members))
    settings = AuthorSettings(
        model_alias="reviewer",
        model=original.model,
        catalog_sha256="0" * 64,
        output_tokens=2000,
        deadline=datetime(2099, 1, 1, tzinfo=UTC),
    )
    registration = ArtifactInput(artifact_id="author-retry-test", sha256="a" * 64)
    window = ArtifactInput(artifact_id="author-execution-test", sha256="b" * 64)
    observed: list[tuple[str, ArtifactInput | None, ArtifactInput | None]] = []

    def offline_response(
        _project: ProjectStore,
        author: AuthorSettings,
        request: ModelRequest,
        *,
        identity: str,
        sources: tuple[ArtifactInput, ...],
        code_revision: str,
        retry_authorization: ArtifactInput | None = None,
        execution_window: ArtifactInput | None = None,
    ) -> tuple[None, ModelResponse]:
        """Capture the exact named retry reference without provider access."""
        observed.append((identity, retry_authorization, execution_window))
        return None, _response(author.model, None)

    monkeypatch.setattr(
        "exp.optimize.evaluation.variance.pipeline_review.author_response", offline_response
    )
    bound = family_reviewer(
        project,
        BuildSettings(spec=family_spec(), reviewer=settings),
        code_revision="test",
        author_retries={"named-review": registration},
        execution_window=window,
    )
    request = ModelRequest(messages=(ModelMessage(role="user", content="review"),))
    bound.draw(request, identity="named-review")
    bound.draw(request, identity="another-review")
    assert observed == [("named-review", registration, window), ("another-review", None, window)]
    assert bound.configuration_sha256 == sha256_json(settings)
