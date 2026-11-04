"""Dense pipeline bindings and actual confirmation records with offline provider transports."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from exp.common.core.artifacts import JsonObject
from exp.optimize.evaluation.executable.costs import evaluation_costs
from exp.optimize.evaluation.variance.dense_judge_test import call
from exp.optimize.evaluation.variance.dense_prompt import (
    PLAN_TOOL,
    REPORT_TOOL,
    dense_model_calls,
    dense_profile,
)
from exp.optimize.evaluation.variance.dense_reward import DenseTag
from exp.optimize.evaluation.variance.family import family_executable_task
from exp.optimize.evaluation.variance.pipeline_dense import (
    DenseSettings,
    attach_family_dense,
    bind_dense_task,
    dense_costs,
)
from exp.optimize.evaluation.variance.tests.family_end_to_end_test import _run
from exp.runtime.models.providers.transport import JsonHttpResponse, JsonHttpTransport

TAG = DenseTag(
    tag_id="actor_contribution",
    title="Recorded actor contribution",
    polarity="credit",
    axis="correctness",
    magnitude=0.05,
    grading_instructions="Credit a recorded actor action in the scored suffix.",
)


class DenseWire(JsonHttpTransport):
    """Script each dense judge from the actual saved episode's actor-owned events.

    Attributes:
        requests: Every request supplied to the scripted transport.
    """

    def __init__(self) -> None:
        """Start with an empty request inventory."""
        self.requests: list[JsonObject] = []

    def get(
        self, url: str, *, headers: Mapping[str, str], timeout_seconds: float
    ) -> JsonHttpResponse:
        """Refuse any unexpected network discovery request."""
        raise AssertionError("no GET belongs to the scripted dense run")

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Plan and report a credit for the first recorded actor-owned scored event."""
        del url, headers, timeout_seconds
        self.requests.append(payload)
        messages = payload["messages"]
        assert isinstance(messages, list) and isinstance(messages[1], dict)
        content = messages[1]["content"]
        assert isinstance(content, str)
        data = json.loads(content)
        candidate_ids = [candidate["candidate_id"] for candidate in data["candidates"]]
        choice = payload.get("tool_choice")
        forced = choice.get("function") if isinstance(choice, dict) else None
        if isinstance(forced, dict) and forced.get("name") == PLAN_TOOL:
            return call(
                PLAN_TOOL,
                {
                    "checks": [
                        {"candidate_id": key, "evidence_needed": "actor event"}
                        for key in candidate_ids
                    ]
                },
            )
        episode = data["episode_evidence"]
        actor = next(
            (
                event["event_id"]
                for event in episode["events"]
                if event["owner"] == "actor" and event["scope"] == "scored"
            ),
            None,
        )
        return call(
            REPORT_TOOL,
            {
                "findings": [
                    {
                        "candidate_id": key,
                        "tag_id": TAG.tag_id,
                        "decision": "absent" if actor is None else "established",
                        "observed_refs": [] if actor is None else [actor],
                        "rationale": "The scripted check cites only a recorded actor event.",
                    }
                    for key in candidate_ids
                ]
            },
        )


def test_dense_definitions_and_confirmations_bind_exact_variants_and_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Actual confirmation cells keep the original outcome signal and actor-only dense spans."""
    run = _run(tmp_path, monkeypatch, first="read_ticket")
    first = run.family()
    family = run.family(run.disposed(first))
    assert family.records is not None and family.eligibility.eligible
    author = run.fixture.settings
    settings = DenseSettings(
        tags=(TAG,),
        profile=dense_profile(
            model_alias=author.model_alias,
            model=author.model,
            maximum_output_tokens=author.output_tokens,
            maximum_input_bytes=author.maximum_input_bytes,
            reasoning_effort=author.reasoning_effort,
        ),
        settings=author,
    )
    task = family_executable_task(family, "urgent", intent="Handle the ticket correctly")
    bound = bind_dense_task(task, settings)
    assert bound.judge == task.judge and bound.job == task.job
    assert bound.profile() != task.profile()
    assert bound.dense is not None and bound.judge is not None
    assert bound.dense.definition.task_sha256 == bound.judge.definition.task_sha256
    assert bind_dense_task(bound, settings) == bound
    changed = settings.model_copy(update={"tags": (TAG.model_copy(update={"magnitude": 0.04}),)})
    assert bind_dense_task(task, changed).profile() != bound.profile()
    with pytest.raises(ValueError, match="already binds"):
        bind_dense_task(bound, changed)

    wire = DenseWire()
    monkeypatch.setattr(
        "exp.optimize.evaluation.factory.authoring.HttpxJsonTransport", lambda: wire
    )
    attached, references = attach_family_dense(run.project, family, settings, code_revision="test")
    assert references and wire.requests
    for tier, groups in attached.groups.items():
        for original, group in zip(family.records.groups[tier], groups, strict=True):
            assert group.advantages == original.advantages
            assert group.episodes == original.episodes and not group.imposed_action_trainable
            assert len(group.dense) == len(group.episodes)
            expected = bind_dense_task(
                family_executable_task(family, group.scenario_id, intent="Handle the ticket"),
                settings,
            ).dense
            assert expected is not None
            for dense in group.dense:
                assert dense is not None and dense.definition_sha256 == expected.identity
                assert dense.episode.total == pytest.approx(0.05)
                spans = [span for finding in dense.findings for span in finding.spans]
                assert spans and all(span.event_id != "imposed.call" for span in spans)
                assert all(span.sampled is None and span.granularity == "action" for span in spans)

    def fail_transport() -> JsonHttpTransport:
        """A replay must never construct a provider transport."""
        raise AssertionError("dense replay attempted provider construction")

    monkeypatch.setattr(
        "exp.optimize.evaluation.factory.authoring.HttpxJsonTransport", fail_transport
    )
    assert attach_family_dense(run.project, family, settings, code_revision="test") == (
        attached,
        references,
    )
    count = sum(len(group.episodes) for groups in attached.groups.values() for group in groups)
    costs = dense_costs(run.project, settings, count)
    calls = dense_model_calls(settings.profile) * count
    assert costs == tuple(row.scaled(calls) for row in evaluation_costs(run.project, author))
    assert dense_costs(run.project, settings, 0) == ()
    with pytest.raises(ValueError, match="negative"):
        dense_costs(run.project, settings, -1)

    groups = family.records.groups
    tier = next(iter(groups))
    row = groups[tier][0]
    episode = row.episodes[0].episode
    assert episode is not None
    wrong = row.model_copy(
        update={
            "episodes": (
                row.episodes[0].model_copy(
                    update={
                        "episode": episode.model_copy(update={"episode_id": "not-a-confirmation"})
                    }
                ),
                *row.episodes[1:],
            )
        }
    )
    records = family.records.model_copy(
        update={"groups": {**groups, tier: (wrong, *groups[tier][1:])}}
    )
    with pytest.raises(ValueError, match="lack exact confirmation"):
        attach_family_dense(
            run.project,
            family.model_copy(update={"records": records}),
            settings,
            code_revision="test",
        )
