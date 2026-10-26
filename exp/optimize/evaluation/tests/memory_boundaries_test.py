"""Indirect provenance, retirement in the serving view, and reservation at registration."""

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from exp.cli.app import app
from exp.common.core.artifacts import JsonObject
from exp.common.project import ProjectStore
from exp.optimize.evaluation import fidelity
from exp.optimize.evaluation.fidelity import (
    FidelitySpec,
    execute_fidelity_evaluation,
    prepare_fidelity_evaluation,
)
from exp.optimize.evaluation.fidelity_memory import saved_memory_receipt, trajectory_record
from exp.optimize.evaluation.tests.fidelity_memory_test import (
    RoutedTransport,
    judged,
    memory_project,
    minimal,
    notes,
    predicted,
    trajectory,
)
from exp.optimize.evaluation.tests.memory_architecture_test import with_lineage
from exp.runtime.models.providers.transport import JsonHttpResponse
from exp.simulation.world_model.experience_memory import (
    Eligibility,
    MemoryVersion,
    is_reserved,
    load_source,
    memory_catalog,
    memory_head,
    public_trajectory_identity,
    publish_trajectory,
)
from exp.simulation.world_model.memory_tree import overview
from exp.simulation.world_model.session_research_test import payload, structured_reply


def note(label: str, hypothesis: str, supersedes: list[str] | None = None) -> JsonHttpResponse:
    """One scripted investigation."""
    return structured_reply(
        {
            "discrepancy": f"{label} differed",
            "hypothesis": hypothesis,
            "scope": f"{label} reads",
            "uncertainty": "one observation",
            "supersedes": supersedes or [],
            "transferable": True,
            "observable_cue": f"{label} reads",
        }
    )


def learn(
    project: ProjectStore,
    spec: FidelitySpec,
    monkeypatch: pytest.MonkeyPatch,
    responses: list[JsonHttpResponse],
) -> tuple[RoutedTransport, list[str]]:
    """Run one learning trajectory and return its wire and published research IDs."""
    wire = RoutedTransport(responses)
    monkeypatch.setattr(fidelity, "HttpxJsonTransport", lambda: wire)
    prepared = prepare_fidelity_evaluation(project, spec, code_revision="fixture")
    execute_fidelity_evaluation(project, prepared)
    receipt = saved_memory_receipt(project, prepared)
    assert receipt is not None and receipt.get("published") is True, receipt
    ids = receipt["research_ids"]
    assert isinstance(ids, list)
    return wire, [str(i) for i in ids]


def alpha_then_beta(
    project: ProjectStore, base: FidelitySpec, monkeypatch: pytest.MonkeyPatch, *, retire: bool
) -> tuple[FidelitySpec, list[str], list[str], RoutedTransport]:
    """A publishes a note; B (another task lineage) sees it and writes a dependent note."""
    a = trajectory(base, "traj-a", "alpha", "learn")
    _, alpha = learn(
        project,
        a,
        monkeypatch,
        [
            predicted("traj-a-0", "wrong"),
            note("alpha", "entries return alpha-reply-0"),
            predicted("traj-a-1", "alpha-reply-1"),
            judged(a),
        ],
    )
    b = with_lineage(trajectory(base, "traj-b", "beta", "learn"), "beta-lineage")
    wire, beta = learn(
        project,
        b,
        monkeypatch,
        [
            predicted("traj-b-0", "wrong"),
            note(
                "beta",
                "copying the earlier note: entries return alpha-reply-0",
                alpha if retire else None,
            ),
            predicted("traj-b-1", "beta-reply-1"),
            judged(b),
        ],
    )
    return a, alpha, beta, wire


def test_research_derived_from_withheld_memory_is_withheld_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B's note was written after seeing A's note, so excluding A excludes B's note."""
    project, base = memory_project(tmp_path)
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    a, alpha, beta, wire = alpha_then_beta(project, base, monkeypatch, retire=False)
    assert "alpha-reply-0" in json.dumps(payload(wire, 1))
    head = memory_head(project.artifacts, "fixture-memory")
    rule = Eligibility(exclude_public=public_trajectory_identity(trajectory_record(a)))
    records, withheld = memory_catalog(project.artifacts, head, eligibility=rule)
    assert not set(alpha) & set(records)
    assert set(beta) <= set(withheld), "derivative note must not survive A's exclusion"
    assert "alpha-reply-0" not in json.dumps(records)


def test_retired_hypothesis_never_reaches_a_later_predictor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retirement applies to the overview lines exactly as to the exact-record catalog."""
    project, base = memory_project(tmp_path)
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    _, alpha, _, _ = alpha_then_beta(project, base, monkeypatch, retire=True)
    c = with_lineage(trajectory(base, "traj-c", "gamma", "frozen", checkpoint=2), "gamma-lineage")
    wire = RoutedTransport([predicted("traj-c-0", "x"), predicted("traj-c-1", "y"), judged(c)])
    monkeypatch.setattr(fidelity, "HttpxJsonTransport", lambda: wire)
    execute_fidelity_evaluation(
        project, prepare_fidelity_evaluation(project, c, code_revision="fixture")
    )
    first = payload(wire, 0)
    history = first["research_history"]
    assert isinstance(history, dict)
    overview = history["memory_overview"]
    assert isinstance(overview, list)
    assert not any(isinstance(r, dict) and r.get("origin") in alpha for r in overview)
    assert not set(alpha) & set(notes(wire, 0, "prior_trajectories"))


def test_preparing_a_reserved_evaluation_registers_the_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Registration protects the panel before execution: learning it is refused at once."""
    project, base = memory_project(tmp_path)
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    a = trajectory(base, "traj-a", "alpha", "learn")
    learn(
        project,
        a,
        monkeypatch,
        [predicted("traj-a-0", "alpha-reply-0"), predicted("traj-a-1", "alpha-reply-1"), judged(a)],
    )
    held = with_lineage(
        trajectory(base, "held", "gamma", "frozen", checkpoint=1), "held-lineage", reserved=True
    )
    prepare_fidelity_evaluation(project, held, code_revision="fixture")
    assert is_reserved(project.artifacts, "fixture-memory", "held-lineage")
    learner = with_lineage(trajectory(base, "learner", "delta", "learn"), "held-lineage")
    with pytest.raises(ValueError, match="reserved for evaluation"):
        prepare_fidelity_evaluation(project, learner, code_revision="fixture")


def test_retirement_and_dependencies_also_govern_lines_and_the_current_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B's own retirement leaves B's next overview; B's derivative line follows A's exclusion."""
    project, base = memory_project(tmp_path)
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    a, alpha, beta, wire = alpha_then_beta(project, base, monkeypatch, retire=True)
    before = payload(wire, 0)["research_history"]
    after = payload(wire, 2)["research_history"]
    assert isinstance(before, dict) and isinstance(after, dict)
    origins = [r.get("origin") for r in before["memory_overview"] if isinstance(r, dict)]
    assert set(alpha) <= set(origins)
    later = [r.get("origin") for r in after["memory_overview"] if isinstance(r, dict)]
    assert not set(alpha) & set(later)
    head = memory_head(project.artifacts, "fixture-memory")
    assert head is not None and head.log_length is not None
    excluded = public_trajectory_identity(trajectory_record(a))
    rule = Eligibility(exclude_public=excluded)
    rows = overview(
        project.artifacts,
        "fixture-memory",
        head.log_length,
        32,
        admits=lambda ref: rule.admits(load_source(project.artifacts, ref)),
    )
    assert not {r.get("origin") for r in rows} & set(beta)
    assert all(r["kind"] != "hypothesis" for r in rows)


class SelectsNothing(RoutedTransport):
    """Native decisions that score every record 0, so JEV delivers no record."""

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Zero relevance on /systemone; model calls follow the script."""
        if not url.endswith("/systemone"):
            return super().post(
                url, headers=headers, payload=payload, timeout_seconds=timeout_seconds
            )
        questions = payload["questions"]
        assert isinstance(questions, dict)
        answers: JsonObject = {
            key: {"type": "choice", "choice": "unclear"}
            if key == "dependency"
            else {"type": "score", "score": 0}
            for key in questions
        }
        return JsonHttpResponse(status_code=200, body={"answers": answers})


def test_dependencies_follow_successive_notes_within_a_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reaches B's second note only through B's first note; excluding A withholds both."""
    project, base = memory_project(tmp_path)
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    a = trajectory(base, "traj-a", "alpha", "learn")
    _, alpha = learn(
        project,
        a,
        monkeypatch,
        [
            predicted("traj-a-0", "wrong"),
            note("alpha", "entries return alpha-reply-0"),
            predicted("traj-a-1", "alpha-reply-1"),
            judged(a),
        ],
    )
    c = with_lineage(trajectory(base, "traj-c", "gamma", "learn"), "gamma-lineage")
    learn(
        project,
        c,
        monkeypatch,
        [predicted("traj-c-0", "gamma-reply-0"), predicted("traj-c-1", "gamma-reply-1"), judged(c)],
    )
    b = with_lineage(trajectory(base, "traj-b", "beta", "learn"), "beta-lineage", overview_lines=1)
    wire = SelectsNothing(
        [
            predicted("traj-b-0", "wrong"),
            note("beta", "first: entries return alpha-reply-0", alpha),
            predicted("traj-b-1", "wrong"),
            note("beta", "second: still alpha-reply-0 per the first beta note"),
            judged(b),
        ]
    )
    monkeypatch.setattr(fidelity, "HttpxJsonTransport", lambda: wire)
    prepared = prepare_fidelity_evaluation(project, b, code_revision="fixture")
    execute_fidelity_evaluation(project, prepared)
    first_overview = payload(wire, 0)["research_history"]
    assert isinstance(first_overview, dict)
    assert "alpha" not in json.dumps(first_overview["memory_overview"])
    receipt = saved_memory_receipt(project, prepared)
    assert receipt is not None and isinstance(receipt["research_ids"], list)
    beta = [str(i) for i in receipt["research_ids"]]
    assert len(beta) == 2
    head = memory_head(project.artifacts, "fixture-memory")
    rule = Eligibility(exclude_public=public_trajectory_identity(trajectory_record(a)))
    records, withheld = memory_catalog(project.artifacts, head, eligibility=rule)
    assert set(beta) <= set(withheld), "the second note inherits A through the first"
    assert "alpha-reply-0" not in json.dumps(records)


def test_saved_reserved_plans_reserve_and_failed_preparation_reserves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validate first, then reserve under the lock, on new and saved registrations alike."""
    project, base = memory_project(tmp_path)
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    early = with_lineage(
        trajectory(base, "early", "gamma", "frozen", checkpoint=1), "early-lineage", reserved=True
    )
    with pytest.raises(ValueError, match="has no version 1"):
        prepare_fidelity_evaluation(project, early, code_revision="fixture")
    assert not is_reserved(project.artifacts, "fixture-memory", "early-lineage")
    a = trajectory(base, "traj-a", "alpha", "learn")
    learn(
        project,
        a,
        monkeypatch,
        [predicted("traj-a-0", "alpha-reply-0"), predicted("traj-a-1", "alpha-reply-1"), judged(a)],
    )
    held = with_lineage(
        trajectory(base, "held", "gamma", "frozen", checkpoint=1), "held-lineage", reserved=True
    )
    monkeypatch.setattr(fidelity, "reserve_task_lineage", lambda *args, **kwargs: None)
    prepared = prepare_fidelity_evaluation(project, held, code_revision="fixture")
    assert not is_reserved(project.artifacts, "fixture-memory", "held-lineage")
    monkeypatch.undo()
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    assert prepare_fidelity_evaluation(project, held, code_revision="fixture") == prepared
    assert is_reserved(project.artifacts, "fixture-memory", "held-lineage")


def test_publication_refuses_a_parent_from_another_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A version from lineage A can never become the parent of lineage B's next version."""
    project, base = memory_project(tmp_path)
    store = project.artifacts
    now = datetime.now(UTC)

    def publish(lineage: str, parent: MemoryVersion | None, reply: str) -> MemoryVersion:
        """Publish one minimal trajectory into a lineage."""
        version, _ = publish_trajectory(
            store,
            lineage=lineage,
            parent=parent,
            world_model=base.world_model,
            selection_policy_sha256="0" * 64,
            trajectory=minimal(reply),
            research=(),
            created_at=now,
            code_revision="fixture",
            publisher="fixture",
        )
        return version

    a1 = publish("lineage-a", None, "a")
    publish("lineage-b", None, "b")
    with pytest.raises(ValueError, match="parent"):
        publish("lineage-b", a1, "b2")
    head = memory_head(store, "lineage-b")
    assert head is not None and head.version == 1


def test_exhausted_compression_is_reported_not_left_as_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After the lifetime limit, the receipt and the terminal name the refused block."""
    project, base = memory_project(tmp_path)
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    overlong = structured_reply({"summary": "z" * 300})
    a = trajectory(base, "traj-a", "alpha", "learn")
    learn(
        project,
        a,
        monkeypatch,
        [
            predicted("traj-a-0", "wrong"),
            note("alpha", "entries differ"),
            predicted("traj-a-1", "alpha-reply-1"),
            judged(a),
            overlong,
            overlong,
            overlong,
        ],
    )
    b = with_lineage(trajectory(base, "traj-b", "beta", "learn"), "beta-lineage")
    learn(
        project,
        b,
        monkeypatch,
        [
            predicted("traj-b-0", "beta-reply-0"),
            predicted("traj-b-1", "beta-reply-1"),
            judged(b),
            overlong,
            overlong,
            overlong,
        ],
    )
    second = prepare_fidelity_evaluation(project, b, code_revision="fixture")
    receipt = json.loads(
        project.artifacts.read_bytes(second.id + "-memory-maintenance", "maintenance.json")
    )
    assert receipt["refused"] == ["#0-1"]
    path = tmp_path / "b.json"
    path.write_text(b.model_dump_json())
    shown = CliRunner().invoke(
        app,
        ["eval", "fidelity-test", "--root", str(tmp_path), "--fidelity", str(path)]
        + ["--non-interactive"],
    )
    assert shown.exit_code == 0, shown.output
    assert "refused #0-1" in shown.output


def test_sources_revealed_after_retirement_are_inherited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When retirement reveals an older row, later notes inherit that row's source."""
    project, base = memory_project(tmp_path)
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")

    def correct(name: str, lineage: str) -> FidelitySpec:
        """Consume an all-correct trajectory: one observed line, no notes."""
        spec = with_lineage(trajectory(base, name, name, "learn"), lineage)
        learn(
            project,
            spec,
            monkeypatch,
            [
                predicted(f"{name}-0", f"{name}-reply-0"),
                predicted(f"{name}-1", f"{name}-reply-1"),
                judged(spec),
            ],
        )
        return spec

    a = correct("traj-a", "a-lineage")
    correct("traj-d", "d-lineage")
    c = with_lineage(trajectory(base, "traj-c", "gamma", "learn"), "c-lineage", overview_lines=1)
    wire = SelectsNothing(
        [
            predicted("traj-c-0", "wrong"),
            note("gamma", "entries return gamma-reply-0"),
            predicted("traj-c-1", "gamma-reply-1"),
            judged(c),
        ]
    )
    monkeypatch.setattr(fidelity, "HttpxJsonTransport", lambda: wire)
    plan_c = prepare_fidelity_evaluation(project, c, code_revision="fixture")
    execute_fidelity_evaluation(project, plan_c)
    receipt_c = saved_memory_receipt(project, plan_c)
    assert receipt_c is not None and isinstance(receipt_c["research_ids"], list)
    gamma = [str(i) for i in receipt_c["research_ids"]]
    a_public = public_trajectory_identity(trajectory_record(a))
    rule = Eligibility(exclude_public=a_public)
    head = memory_head(project.artifacts, "fixture-memory")
    assert set(gamma) <= set(memory_catalog(project.artifacts, head, eligibility=rule)[0])
    b = with_lineage(trajectory(base, "traj-b", "beta", "learn"), "b-lineage", overview_lines=3)
    wire = SelectsNothing(
        [
            predicted("traj-b-0", "wrong"),
            note("beta", "the gamma note was wrong", gamma),
            predicted("traj-b-1", "wrong"),
            note("beta", "second note, after the older history appeared"),
            judged(b),
        ]
    )
    monkeypatch.setattr(fidelity, "HttpxJsonTransport", lambda: wire)
    plan_b = prepare_fidelity_evaluation(project, b, code_revision="fixture")
    execute_fidelity_evaluation(project, plan_b)
    first = payload(wire, 0)["research_history"]
    later = payload(wire, 2)["research_history"]
    assert isinstance(first, dict) and isinstance(later, dict)
    origins = [[r.get("origin") for r in h["memory_overview"]] for h in (first, later)]
    assert "a-lineage" not in origins[0] and "a-lineage" in origins[1]
    receipt = saved_memory_receipt(project, plan_b)
    assert receipt is not None and isinstance(receipt["research_ids"], list)
    first_note, second_note = (str(i) for i in receipt["research_ids"])
    head = memory_head(project.artifacts, "fixture-memory")
    _, withheld = memory_catalog(project.artifacts, head, eligibility=rule)
    assert first_note not in withheld
    assert second_note in withheld, "a note written after A became visible depends on A"
