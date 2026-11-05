"""Admitted compatibility settings stay bound through build and draw replay."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactInput, JsonObject
from exp.common.models import ModelMessage, ModelRequest, ModelResponse
from exp.common.project import ProjectStore
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.variance.build import plan_variance_build, run_variance_build
from exp.optimize.evaluation.variance.build_test import FakeDecisions, _tagger, variance_project
from exp.optimize.evaluation.variance.family_scopes_test import spec as family_spec
from exp.optimize.evaluation.variance.pipeline_review import family_reviewer
from exp.optimize.evaluation.variance.pipeline_spec import BuildSettings
from exp.simulation.mining.environment_families_test import answer, order_trace, reviewer
from exp.simulation.mining.environment_review import CompatibilitySpec, checked_group_packet


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


def test_bounded_inventory_never_bypasses_full_native_author_byte_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fitting initial shard still refuses an oversized full request before intent or dispatch."""
    import json

    project = variance_project(tmp_path, monkeypatch)
    _, original = reviewer(lambda members, _variant: answer("compatible", members))
    settings = AuthorSettings(
        model_alias="reviewer",
        model=original.model,
        catalog_sha256="0" * 64,
        output_tokens=2000,
        maximum_input_bytes=128_000,
        deadline=datetime(2099, 1, 1, tzinfo=UTC),
    )
    spec = family_spec(
        review=CompatibilitySpec(
            maximum_candidate_members=64,
            maximum_candidate_packet_bytes=48_000,
        )
    )
    assert spec.environment_families is not None
    packet, _ = checked_group_packet((order_trace("o0"),), spec.environment_families.review)
    body = json.dumps(packet, sort_keys=True)
    assert len(body.encode()) < 48_000
    request = ModelRequest(
        messages=(
            ModelMessage(role="user", content=body),
            ModelMessage(role="assistant", content="x" * 128_000),
        ),
        maximum_output_tokens=settings.output_tokens,
    )
    assert len(request.model_dump_json().encode()) > settings.maximum_input_bytes
    bound = family_reviewer(
        project, BuildSettings(spec=spec, reviewer=settings), code_revision="test"
    )
    with pytest.raises(ValueError, match="author context exceeds its frozen byte allowance"):
        bound.draw(request, identity="oversize-bounded-review")
    assert not project.artifacts.exists("oversize-bounded-review-intent")
    assert not project.artifacts.exists("oversize-bounded-review-response")


def test_native_parallel_roots_overlap_with_bounded_workers_context_and_cached_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real author intents and receipts preserve authority across isolated review threads."""
    import json
    from collections.abc import Mapping
    from contextvars import ContextVar
    from threading import Barrier, Lock

    from exp.optimize.evaluation.factory.authoring import author_response
    from exp.optimize.evaluation.factory.proposals_test import proposal_fixture
    from exp.runtime.models.call_evidence import (
        call_consumer,
        canonical_call,
        current_call_intent,
        current_consumer,
    )
    from exp.runtime.models.providers.transport import JsonHttpResponse
    from exp.runtime.models.spend_ceiling import (
        OperationSpendCeiling,
        current_spend_ceiling,
        install_spend_ceiling,
        operation_spend_scope,
    )
    from exp.simulation.mining.environment_families import (
        EnvironmentReviewWindow,
        environment_families,
    )
    from exp.simulation.mining.environment_families_test import clustering

    project, proposal, transport = proposal_fixture(tmp_path, monkeypatch)
    settings = AuthorSettings(
        model_alias=proposal.model_alias,
        model=proposal.model,
        catalog_sha256=proposal.catalog_sha256,
        output_tokens=2000,
        deadline=proposal.deadline,
    )
    review = CompatibilitySpec(
        variants=("contract_first",),
        min_reviews=1,
        max_turns=1,
        repairs=0,
        maximum_parallel_groups=2,
    )
    build_spec = family_spec(review=review)
    bound = family_reviewer(
        project, BuildSettings(spec=build_spec, reviewer=settings), code_revision="test"
    )
    traces = {f"o{i}": order_trace(f"o{i}") for i in range(6)}
    groups = clustering({f"c{i}": (key,) for i, key in enumerate(traces)})
    ceiling = OperationSpendCeiling(tmp_path / "parallel-spend.sqlite", tariffs={})
    ceiling.authorize_policy(ceiling_usd=0, unpriced_attempts=12)
    outer = ArtifactInput(artifact_id="parent-review-intent", sha256="c" * 64)
    local: ContextVar[str] = ContextVar("root-review-test", default="outside")
    barrier, lock = Barrier(2, timeout=10), Lock()
    active = maximum = calls = 0
    roots: set[str] = set()
    observed: list[str] = []

    def native_author(
        source: ProjectStore,
        author: AuthorSettings,
        request: ModelRequest,
        *,
        identity: str,
        sources: tuple[ArtifactInput, ...],
        code_revision: str,
    ) -> tuple[ArtifactInput, ModelResponse]:
        """Check the original context before delegating to the unmodified author owner."""
        packet = json.loads(request.messages[1].content or "{}")
        member = str(packet["members"][0]["trace_id"])
        with lock:
            assert local.get() in ("parent", member)
            roots.add(member)
        local.set(member)
        assert current_spend_ceiling() is ceiling
        assert current_call_intent() == outer
        assert current_consumer() == "investigator"
        return author_response(
            source,
            author,
            request,
            identity=identity,
            sources=sources,
            code_revision=code_revision,
        )

    def post(
        _url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Serve complete native typed answers after two physical attempts overlap."""
        nonlocal active, maximum, calls
        del headers, timeout_seconds
        intent = current_call_intent()
        assert intent is not None and intent != outer
        assert current_spend_ceiling() is ceiling
        assert current_consumer() == "investigator"
        messages = payload["messages"]
        assert isinstance(messages, list)
        message = messages[1]
        assert isinstance(message, dict)
        packet = json.loads(str(message["content"]))
        members = frozenset(str(row["trace_id"]) for row in packet["members"])
        assert local.get() in members
        with lock:
            active += 1
            maximum = max(maximum, active)
            calls += 1
            observed.append(intent.artifact_id)
        try:
            barrier.wait()
            arguments = answer("compatible", members)
            return JsonHttpResponse(
                status_code=200,
                body={
                    "id": intent.artifact_id,
                    "model": settings.model.model_id,
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "submit",
                                        "type": "function",
                                        "function": {
                                            "name": "submit_review",
                                            "arguments": json.dumps(arguments),
                                        },
                                    }
                                ],
                            },
                        }
                    ],
                },
            )
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(transport, "post", post)
    monkeypatch.setattr(
        "exp.optimize.evaluation.variance.pipeline_review.author_response", native_author
    )
    with operation_spend_scope(), canonical_call(outer), call_consumer("investigator"):
        install_spend_ceiling(ceiling)
        token = local.set("parent")
        try:
            record, reference = environment_families(
                project.artifacts,
                groups,
                bound,
                traces=traces,
                eligible=tuple(traces),
                held_out=(),
                spec=review,
                review_window=EnvironmentReviewWindow(start=0, count=4),
                code_revision="test",
            )
            overlapping, _ = environment_families(
                project.artifacts,
                groups,
                bound,
                traces=traces,
                eligible=tuple(traces),
                held_out=(),
                spec=review,
                review_window=EnvironmentReviewWindow(start=2, count=4),
                code_revision="test",
            )
            assert local.get() == "parent"
            assert current_call_intent() == outer
            assert current_spend_ceiling() is ceiling
            replay, replay_reference = environment_families(
                project.artifacts,
                groups,
                bound,
                traces=traces,
                eligible=tuple(traces),
                held_out=(),
                spec=review,
                review_window=EnvironmentReviewWindow(start=0, count=4),
                code_revision="test",
            )
        finally:
            local.reset(token)
    assert replay == record and replay_reference == reference
    assert maximum == 2 and active == 0
    assert calls == 12 and len(set(observed)) == 12
    assert len(roots) == 6
    assert [family.members for family in record.families] == [(key,) for key in list(traces)[:4]]
    assert [family.members for family in overlapping.families] == [
        (key,) for key in list(traces)[2:]
    ]
    assert [root.members for root in record.queued] == [(key,) for key in list(traces)[4:]]
    assert [root.members for root in overlapping.queued] == [(key,) for key in list(traces)[:2]]
    assert len(project.artifacts.list_ids(artifact_type="author-intent")) == 12
    assert len(project.artifacts.list_ids(artifact_type="author-response")) == 12
