"""Executed witnesses, forced-option contrasts and family flips on a restored native workbook."""

from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactInput, JsonObject, sha256_bytes
from exp.common.evaluations.uncertainty import wilson_interval
from exp.common.models import ToolCall
from exp.common.project import ProjectStore
from exp.optimize.evaluation.continuations import action_contrast
from exp.optimize.evaluation.continuations.job import ContinuationJob
from exp.optimize.evaluation.continuations.job_test import job_fixture
from exp.optimize.evaluation.continuations.study import ContinuationStudy
from exp.optimize.evaluation.variance.contracts import (
    Criterion,
    DecisionOption,
    DecisionState,
    Scenario,
    StateEdit,
    VariationAxis,
    WitnessStep,
)
from exp.optimize.evaluation.variance.contrast import (
    OptionContrast,
    OptionOutcome,
    WitnessResult,
    family_flip,
    run_observation_witness,
    run_option_contrast,
    shows_value,
    states_action,
    witness_reference,
)
from exp.runtime.environments.interface import Observation
from exp.runtime.environments.tests.world_conformance_test import workbook_case
from exp.runtime.environments.worlds import ExecutableWorld
from exp.runtime.models.providers.transport import JsonHttpResponse, ScriptedJsonTransport

VARIANT_RUNTIME = """from exp.optimize.evaluation.executable.loader import ContinuationRuntime
from exp.optimize.evaluation.continuations.study_test import probe, workbook_runtime
from exp.simulation.continuations import load_continuation_checkpoint
from exp.common.core.artifacts import ArtifactInput
def continuation_runtime(project, configuration):
    reference = ArtifactInput.model_validate(configuration["checkpoint"])
    checkpoint = load_continuation_checkpoint(project.artifacts, reference)
    def variant(image):
        return {**probe(image), "account": configuration["account"]}
    return ContinuationRuntime(workbook_runtime(checkpoint.world), variant, configuration["probe"])
"""


def _decision() -> DecisionState:
    """Return a two-option refund decision deciding on the account tier."""
    return DecisionState(
        step=0,
        decision="Whether to refund the order.",
        options=(
            DecisionOption(option_id="refund", action="Issue a full refund"),
            DecisionOption(option_id="decline", action="Decline the refund request"),
        ),
        relevant_paths=("/account/tier",),
        axes=(VariationAxis(path="/account/tier", values=("gold", "basic"), effect="tier"),),
        criteria=(Criterion(criterion_id="right-call", text="The right call was made."),),
        evidence_refs=("run:a:step:0",),
    )


def _scenario(*, witness_calls: int = 1) -> Scenario:
    """Return the gold-tier variant, whose deciding tier is revealed by one workbook call."""
    step = WitnessStep(
        tool="edit_workbook", arguments=workbook_case().first.arguments, reveals=("/account/tier",)
    )
    return Scenario(
        scenario_id="gold",
        edits=(StateEdit(path="/account/tier", value="gold"),),
        correct_option="refund",
        deciding_paths=("/account/tier",),
        witness=(step,) * witness_calls,
        rationale="Gold accounts are refunded.",
        criteria=("right-call",),
    )


def _variant_job(tmp_path: Path, job: ContinuationJob, account: JsonObject) -> ContinuationJob:
    """Bind the job to a runtime whose evaluator state carries the variant's account."""
    runtime = tmp_path / "variant_runtime.py"
    runtime.write_text(VARIANT_RUNTIME)
    return job.model_copy(
        update={
            "runtime_file": runtime,
            "runtime_sha256": sha256_bytes(runtime.read_bytes()),
            "configuration": {**job.configuration, "account": account},
        }
    )


def _witness_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: str, tier: str = "gold"
) -> tuple[ProjectStore, ContinuationJob, list[ToolCall]]:
    """Restore the real workbook world but script what each witness observation says."""
    project, job, transport = job_fixture(tmp_path, monkeypatch)
    calls: list[ToolCall] = []

    def observe(world: ExecutableWorld, action: ToolCall) -> Observation:
        """Record the executed witness call and return the scripted observation."""
        calls.append(action)
        return Observation(content=content)

    monkeypatch.setattr(ExecutableWorld, "execute", observe)
    assert not transport.requests
    return project, _variant_job(tmp_path, job, {"tier": tier}), calls


def test_witness_shows_deciding_fact_and_replays_without_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tier shown in a structured observation holds, and the saved witness replays exactly."""
    project, job, calls = _witness_setup(
        tmp_path, monkeypatch, '{"account": {"tier": "gold", "orders": 3}}'
    )

    def run() -> WitnessResult:
        """Run or replay the gold witness."""
        return run_observation_witness(
            project,
            checkpoint=job.study.checkpoint,
            scenario=_scenario(),
            decision=_decision(),
            job=job,
            code_revision="test",
        )

    with pytest.raises(ValueError, match="no saved witness"):
        witness_reference(
            project,
            checkpoint=job.study.checkpoint,
            scenario=_scenario(),
            job=job,
            code_revision="test",
        )
    result = run()
    assert result.holds, result.problems
    cited = witness_reference(
        project,
        checkpoint=job.study.checkpoint,
        scenario=_scenario(),
        job=job,
        code_revision="test",
    )
    assert project.artifacts.read(cited.artifact_id).manifest.artifact_type == (
        "observation-witness"
    )
    assert result.state["account"] == {"tier": "gold"}
    assert result.paths[0].shown_by == (0,) and result.paths[0].declared_by == (0,)
    assert result.remaining_calls == 4 and len(calls) == 1
    job.runtime_file.unlink()
    assert run() == result
    assert len(calls) == 1


def test_witness_fails_when_no_observation_shows_the_fact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A declared reveal that the executed observation does not back is not evidence."""
    project, job, _ = _witness_setup(tmp_path, monkeypatch, '{"written": {"output.xlsx": {}}}')
    result = run_observation_witness(
        project,
        checkpoint=job.study.checkpoint,
        scenario=_scenario(),
        decision=_decision(),
        job=job,
        code_revision="test",
    )
    assert not result.holds
    assert result.paths[0].shown_by == ()
    assert result.problems == ("no witness observation shows deciding fact /account/tier",)


def test_witness_rejects_an_observation_that_states_the_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Showing the tier is not enough when the observation also tells the actor what to do."""
    project, job, _ = _witness_setup(
        tmp_path, monkeypatch, "Tier gold. Recommended next step: issue a  FULL refund."
    )
    result = run_observation_witness(
        project,
        checkpoint=job.study.checkpoint,
        scenario=_scenario(),
        decision=_decision(),
        job=job,
        code_revision="test",
    )
    assert result.paths[0].shown_by == (0,)
    assert result.leaks == (0,)
    assert any("verbatim" in problem for problem in result.problems)


def test_witness_rejects_wrong_variant_and_exhausted_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpoint lacking the edit is not this variant; four calls leave no decision call."""
    project, job, calls = _witness_setup(tmp_path, monkeypatch, "tier: gold", tier="basic")
    result = run_observation_witness(
        project,
        checkpoint=job.study.checkpoint,
        scenario=_scenario(witness_calls=4),
        decision=_decision(),
        job=job,
        code_revision="test",
    )
    assert len(calls) == 4
    assert any("not this variant" in problem for problem in result.problems)
    assert any("plus the decision" in problem for problem in result.problems)
    assert any("shows deciding fact" in problem for problem in result.problems)


def test_value_matching_respects_token_boundaries() -> None:
    """Numbers and words match whole tokens; structured content matches exact typed values."""
    assert shows_value(12, "limit is 12 today")
    assert not shows_value(12, "limit is 120") and not shows_value(12, "limit 1.12")
    assert not shows_value(12, "limit 12.5")
    assert shows_value(12.0, "limit 12.")
    assert shows_value("gold", "tier=gold-plus")
    assert not shows_value("gold", "goldfish")
    assert shows_value(True, '{"flag": true}') and not shows_value(1, '{"flag": true}')
    assert shows_value({"a": 1}, '{"x": [{"a": 1}]}')
    assert states_action("Issue a full refund", "please ISSUE a\nfull refund now")
    assert not states_action("Issue a full refund", "a refund is possible")


def _finish_transport(monkeypatch: pytest.MonkeyPatch, count: int) -> ScriptedJsonTransport:
    """Script ``count`` finish-only actor replies for every suffix."""
    response = JsonHttpResponse(
        status_code=200,
        body={
            "id": "finish",
            "model": "world-v1",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "done"},
                }
            ],
        },
    )
    transport = ScriptedJsonTransport((response,) * count)
    monkeypatch.setattr(
        "exp.optimize.evaluation.continuations.job.HttpxJsonTransport", lambda: transport
    )
    return transport


def _deletion(count: int) -> ToolCall:
    """Delete ``count`` data rows; the fixture verifier passes only with one deletion."""
    call = workbook_case().first
    return call.model_copy(
        update={
            "arguments": {
                **call.arguments,
                "deletions": [{"sheet": "Orders", "start": 2, "count": count}],
            }
        }
    )


def _row_decision(*counts: int) -> DecisionState:
    """Return a decision whose options delete different numbers of rows."""
    return DecisionState(
        step=0,
        decision="How many rows to delete.",
        options=tuple(
            DecisionOption(option_id=f"delete-{count}", action=f"Delete {count} rows")
            for count in counts
        ),
        relevant_paths=("/rows",),
        axes=(VariationAxis(path="/rows", values=(6, 7), effect="row count"),),
        criteria=(Criterion(criterion_id="rows", text="The right rows were deleted."),),
        evidence_refs=("run:a:step:0",),
    )


def test_forced_options_separate_under_one_frozen_job_and_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deleting one row passes every draw and two fails every draw, separated at four pairs."""
    project, job, _ = job_fixture(tmp_path, monkeypatch)
    transport = _finish_transport(monkeypatch, 8)
    decision = _row_decision(1, 2)

    def run() -> OptionContrast:
        """Run or replay the one-versus-two row contrast."""
        return run_option_contrast(
            project,
            checkpoint=job.study.checkpoint,
            decision=decision,
            actions={"delete-1": _deletion(1), "delete-2": _deletion(2)},
            job=job,
            repeats=4,
            code_revision="test",
        )

    contrast = run()
    one, two = contrast.options
    assert (one.option_id, one.planned, one.successes, one.unknown) == ("delete-1", 4, 4, 0)
    assert (two.option_id, two.planned, two.successes, two.goal_failures) == ("delete-2", 4, 0, 4)
    assert one.bounds[0] > two.bounds[1]
    assert one.wilson_95 == one.bounds and len(contrast.contrasts) == 1
    assert len(transport.requests) == 8
    job.runtime_file.unlink()
    assert run() == contrast
    assert len(transport.requests) == 8
    scenario = _scenario().model_copy(update={"correct_option": "delete-1"})
    verdict = family_flip({"gold": contrast}, [scenario])
    assert verdict.variants[0].status == "separated_matches" and not verdict.qualifies


def test_three_options_pool_the_reference_and_keep_unknowns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each non-reference option is one contrast; an unavailable suffix widens its bounds."""
    project, job, _ = job_fixture(tmp_path, monkeypatch)
    transport = _finish_transport(monkeypatch, 3)
    original = action_contrast.execute_continuation_job
    calls = []

    def fail_once(
        project: ProjectStore, job: ContinuationJob, *, code_revision: str
    ) -> ContinuationStudy:
        """Lose the first suffix before it returns."""
        calls.append(job)
        if len(calls) == 1:
            raise RuntimeError("unknown provider outcome")
        return original(project, job, code_revision=code_revision)

    monkeypatch.setattr(action_contrast, "execute_continuation_job", fail_once)
    contrast = run_option_contrast(
        project,
        checkpoint=job.study.checkpoint,
        decision=_row_decision(1, 2, 3),
        actions={f"delete-{n}": _deletion(n) for n in (1, 2, 3)},
        job=job,
        repeats=1,
        code_revision="test",
    )
    reference, two, three = contrast.options
    assert contrast.reference_option == "delete-1" and len(contrast.contrasts) == 2
    assert (reference.planned, reference.assessable, reference.unknown) == (2, 1, 1)
    assert reference.failures == ("RuntimeError",)
    assert reference.wilson_95 is not None and reference.bounds[1] == 1.0
    assert reference.bounds[0] < reference.wilson_95[0]
    assert (two.planned, two.successes, three.planned, three.successes) == (1, 0, 1, 0)
    assert len(transport.requests) == 3


def test_option_actions_must_cover_distinct_declared_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing or duplicate concrete actions refuse before any dispatch."""
    project, job, transport = job_fixture(tmp_path, monkeypatch)
    for actions in (
        {"delete-1": _deletion(1)},
        {"delete-1": _deletion(1), "delete-2": _deletion(1)},
    ):
        with pytest.raises(ValueError, match="option"):
            run_option_contrast(
                project,
                checkpoint=job.study.checkpoint,
                decision=_row_decision(1, 2),
                actions=actions,
                job=job,
                repeats=1,
                code_revision="test",
            )
    assert not transport.requests


def _outcome(option_id: str, successes: int, planned: int, unknown: int = 0) -> OptionOutcome:
    """Build a synthetic option outcome with its conservative separation bounds."""
    return OptionOutcome(
        option_id=option_id,
        action=ToolCall(call_id=option_id, name="act", arguments={"option": option_id}),
        planned=planned,
        assessable=planned - unknown,
        successes=successes,
        unknown=unknown,
        bounds=(
            wilson_interval(successes / planned, planned)[0],
            wilson_interval((successes + unknown) / planned, planned)[1],
        ),
    )


def _contrast(refund: int, decline: int, planned: int = 20, unknown: int = 0) -> OptionContrast:
    """Build a synthetic two-option contrast."""
    return OptionContrast(
        step=0,
        decision="Whether to refund the order.",
        checkpoint=_checkpoint(),
        reference_option="refund",
        repeats=planned,
        options=(
            _outcome("refund", refund, planned, unknown),
            _outcome("decline", decline, planned),
        ),
    )


def _checkpoint() -> ArtifactInput:
    """Return a syntactically valid artifact reference."""
    return ArtifactInput(artifact_id="checkpoint-a", sha256="0" * 64)


def _variants(*correct: str) -> list[Scenario]:
    """Build one scenario per declared correct option, in order."""
    base = _scenario()
    return [
        base.model_copy(update={"scenario_id": f"v{index}", "correct_option": option})
        for index, option in enumerate(correct)
    ]


def test_family_qualifies_when_the_separated_winner_flips() -> None:
    """Refund wins one variant and decline another, each as declared."""
    verdict = family_flip(
        {"v0": _contrast(18, 2), "v1": _contrast(3, 17)}, _variants("refund", "decline")
    )
    assert verdict.qualifies and verdict.winners == ("decline", "refund")
    assert [v.status for v in verdict.variants] == ["separated_matches", "separated_matches"]


def test_null_family_without_a_flip_does_not_qualify() -> None:
    """Different success rates with the same winner everywhere are spread, not a flip."""
    verdict = family_flip(
        {"v0": _contrast(19, 1), "v1": _contrast(15, 3)}, _variants("refund", "refund")
    )
    assert not verdict.qualifies and verdict.winners == ("refund",)


def test_mismatch_needs_triage_and_blocks_qualification() -> None:
    """A separated winner contradicting its scenario flags the author or simulator."""
    verdict = family_flip(
        {"v0": _contrast(18, 2), "v1": _contrast(3, 17), "v2": _contrast(19, 1)},
        _variants("refund", "decline", "decline"),
    )
    assert verdict.variants[2].status == "separated_mismatch"
    assert verdict.triage == ("v2",) and not verdict.qualifies


def test_overlapping_or_unknown_heavy_variants_stay_unseparated() -> None:
    """Overlapping intervals and unknowns counted both ways never produce a winner."""
    verdict = family_flip(
        {"v0": _contrast(12, 8), "v1": _contrast(10, 2, planned=20, unknown=10)},
        _variants("refund", "refund", "decline"),
    )
    assert [v.status for v in verdict.variants] == ["unseparated", "unseparated", "unmeasured"]
    assert verdict.undersampled == ("v0", "v1") and not verdict.qualifies


def test_family_refuses_unknown_scenarios_and_mismatched_options() -> None:
    """Contrasts must belong to the family and contrast the same options."""
    with pytest.raises(ValueError, match="unknown scenarios"):
        family_flip({"other": _contrast(1, 1)}, _variants("refund"))
    odd = _contrast(1, 1).model_copy(
        update={"options": (_outcome("refund", 1, 2), _outcome("escalate", 1, 2))}
    )
    with pytest.raises(ValueError, match="same decision options"):
        family_flip({"v0": _contrast(1, 1), "v1": odd}, _variants("refund", "refund"))
