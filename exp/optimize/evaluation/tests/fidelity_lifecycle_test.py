"""Lifecycle boundaries: CLI recovery, same-trajectory isolation, supersession, observations."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from typer.testing import CliRunner

from exp.cli.app import app
from exp.common.project import ProjectStore
from exp.optimize.evaluation import fidelity, fidelity_memory
from exp.optimize.evaluation.fidelity import (
    FidelitySpec,
    execute_fidelity_evaluation,
    prepare_fidelity_evaluation,
)
from exp.optimize.evaluation.fidelity_memory import (
    saved_memory_receipt,
)
from exp.optimize.evaluation.tests.fidelity_memory_test import (
    RoutedTransport,
    investigation,
    judged,
    memory_project,
    notes,
    predicted,
    trajectory,
)
from exp.runtime.sessions.store import SessionStore
from exp.simulation.world_model.experience_memory import (
    memory_catalog,
    memory_head,
    memory_records,
    public_trajectory_identity,
    read_memory_version,
)
from exp.simulation.world_model.session_research_test import payload, structured_reply


def learn_alpha(
    project: ProjectStore, base: FidelitySpec, monkeypatch: pytest.MonkeyPatch
) -> tuple[FidelitySpec, str]:
    """Consume trajectory A (step 0 wrong, investigated) and return its research ID."""
    a = trajectory(base, "traj-a", "alpha", "learn")
    wire = RoutedTransport(
        [
            predicted("traj-a-0", "wrong"),
            investigation("alpha"),
            predicted("traj-a-1", "alpha-reply-1"),
            judged(a),
        ]
    )
    monkeypatch.setattr(fidelity, "HttpxJsonTransport", lambda: wire)
    prepared = prepare_fidelity_evaluation(project, a, code_revision="fixture")
    execute_fidelity_evaluation(project, prepared)
    receipt = saved_memory_receipt(project, prepared)
    assert receipt is not None and isinstance(receipt["research_ids"], list)
    return a, str(receipt["research_ids"][0])


def run_cli(root: Path, spec: FidelitySpec) -> str:
    """Invoke the public command for one saved spec."""
    path = root / (spec.run_id + ".json")
    path.write_text(spec.model_dump_json())
    result = CliRunner().invoke(
        app,
        ["eval", "fidelity-test", "--root", str(root), "--fidelity", str(path)]
        + ["--non-interactive"],
    )
    assert result.exit_code == 0, result.output
    return result.output


@pytest.mark.parametrize("stage", ["before-publication", "after-version-write"])
def test_cli_restart_recovers_interrupted_publication_without_purchase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    """A crash after report persistence is completed exactly once by the public CLI."""
    project, base = memory_project(tmp_path)
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    a = trajectory(base, "traj-a", "alpha", "learn")
    wire = RoutedTransport(
        [
            predicted("traj-a-0", "wrong"),
            investigation("alpha"),
            predicted("traj-a-1", "alpha-reply-1"),
            judged(a),
        ]
    )
    monkeypatch.setattr(fidelity, "HttpxJsonTransport", lambda: wire)
    real = fidelity_memory.publish_trajectory

    def crash(*args: object, **kwargs: object) -> object:
        """Interrupt before or after the durable version write."""
        if stage == "after-version-write":
            cast("Callable[..., object]", real)(*args, **kwargs)
        raise KeyboardInterrupt(stage)

    monkeypatch.setattr(fidelity_memory, "publish_trajectory", crash)
    with pytest.raises(KeyboardInterrupt):
        execute_fidelity_evaluation(
            project, prepare_fidelity_evaluation(project, a, code_revision="fixture")
        )
    assert project.artifacts.exists(
        prepare_fidelity_evaluation(project, a, code_revision="fixture").id + "-report"
    )
    monkeypatch.setattr(fidelity_memory, "publish_trajectory", real)
    monkeypatch.delenv("FIDELITY_TEST_KEY")
    purchased = len(wire.requests)
    output = run_cli(tmp_path, a)
    assert "Saved result. No provider calls." in output
    assert "v0 → v1" in output
    assert len(wire.requests) == purchased
    head = memory_head(project.artifacts, "fixture-memory")
    assert head is not None and head.version == 1 and len(head.consumed) == 1
    run_cli(tmp_path, a)
    assert memory_head(project.artifacts, "fixture-memory") == head
    receipt = saved_memory_receipt(
        project, prepare_fidelity_evaluation(project, a, code_revision="fixture")
    )
    assert receipt is not None and receipt["published"] is True


def test_consumed_trajectory_never_informs_its_own_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A frozen rerun of a consumed trajectory receives none of its disclosed replies."""
    project, base = memory_project(tmp_path)
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    a, _ = learn_alpha(project, base, monkeypatch)
    rerun = trajectory(base, "traj-a-frozen", "alpha", "frozen", checkpoint=1)
    rerun = FidelitySpec.model_validate(
        {
            **rerun.model_dump(mode="json"),
            "workload": {**a.workload.model_dump(mode="json"), "id": "traj-a-frozen"},
            "references": a.references.model_dump(mode="json"),
        }
    )
    wire = RoutedTransport([predicted("traj-a-0", "x"), predicted("traj-a-1", "y"), judged(rerun)])
    monkeypatch.setattr(fidelity, "HttpxJsonTransport", lambda: wire)
    execute_fidelity_evaluation(
        project, prepare_fidelity_evaluation(project, rerun, code_revision="fixture")
    )
    for request in wire.requests[:2]:
        assert "alpha-reply" not in json.dumps(request.payload)
    assert notes(wire, 0, "prior_trajectories") == []


def test_later_trajectory_supersedes_earlier_hypothesis_and_keeps_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contradicting feedback retires an earlier trajectory's note without deleting it."""
    project, base = memory_project(tmp_path)
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    _, old = learn_alpha(project, base, monkeypatch)
    b = trajectory(base, "traj-b", "beta", "learn")
    contradiction = structured_reply(
        {
            "discrepancy": "beta reply contradicts the alpha hypothesis",
            "hypothesis": "entries keep their stored value",
            "scope": "entry reads",
            "uncertainty": "two observations",
            "supersedes": [old],
            "transferable": True,
            "observable_cue": "the same tool and arguments",
        }
    )
    wire = RoutedTransport(
        [
            predicted("traj-b-0", "wrong"),
            contradiction,
            predicted("traj-b-1", "beta-reply-1"),
            judged(b),
        ]
    )
    monkeypatch.setattr(fidelity, "HttpxJsonTransport", lambda: wire)
    prepared = prepare_fidelity_evaluation(project, b, code_revision="fixture")
    report = execute_fidelity_evaluation(project, prepared)
    assert report.mechanical["valid_steps"] == 2
    investigator = payload(wire, 1)
    assert old in json.dumps(investigator)
    assert old not in notes(wire, 2, "prior_trajectories")
    head = memory_head(project.artifacts, "fixture-memory")
    assert head is not None and head.version == 2
    active = memory_records(project.artifacts, head)
    assert old not in active
    v1 = read_memory_version(project.artifacts, "fixture-memory", 1)
    assert old in memory_records(project.artifacts, v1)
    library = json.loads(project.artifacts.read_bytes(head.library.artifact_id, "library.json"))
    assert old in {entry["entry_id"] for entry in library["entries"]}


def test_consumed_observations_are_retrievable_but_never_initialize_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Native observations of A reach B as labelled history, never as B's starting state."""
    project, base = memory_project(tmp_path)
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    learn_alpha(project, base, monkeypatch)
    b = trajectory(base, "traj-b", "beta", "learn")
    wire = RoutedTransport(
        [
            predicted("traj-b-0", "beta-reply-0"),
            predicted("traj-b-1", "beta-reply-1"),
            judged(b),
        ]
    )
    monkeypatch.setattr(fidelity, "HttpxJsonTransport", lambda: wire)
    prepared = prepare_fidelity_evaluation(project, b, code_revision="fixture")
    execute_fidelity_evaluation(project, prepared)
    first = payload(wire, 0)
    assert first["environment_state"] == {}
    history = first["research_history"]
    assert isinstance(history, dict) and "prior_observations" not in history
    assert "original-observation" not in json.dumps(history)
    observations = first["memory_observations"]
    assert isinstance(observations, list) and observations
    replies = {json.dumps(row["observed_reply"]) for row in observations if isinstance(row, dict)}
    assert json.dumps("alpha-reply-1") in replies
    assert all(
        isinstance(row, dict) and row["authority"].startswith("Original native observation")
        for row in observations
    )
    assert "beta-reply" not in json.dumps(first)
    store = SessionStore(project.artifacts.project_directory / (prepared.id + "-sessions.sqlite"))
    transitions = store.read_session(prepared.id).state["transitions"]
    assert isinstance(transitions, list) and isinstance(transitions[0], dict)
    assert transitions[0]["before"] == {}
    evidence = transitions[0]["evidence"]
    assert isinstance(evidence, dict) and evidence["session_research_selection"]


def test_withheld_investigation_cannot_retire_hypotheses_for_its_own_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Supersession counts only eligible evidence; withheld feedback has no indirect effect."""
    project, base = memory_project(tmp_path)
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    _, old = learn_alpha(project, base, monkeypatch)
    b = trajectory(base, "traj-b", "beta", "learn")
    contradiction = structured_reply(
        {
            "discrepancy": "beta reply contradicts the alpha hypothesis",
            "hypothesis": "entries keep their stored value",
            "scope": "entry reads",
            "uncertainty": "two observations",
            "supersedes": [old],
            "transferable": True,
            "observable_cue": "the same tool and arguments",
        }
    )
    wire = RoutedTransport(
        [
            predicted("traj-b-0", "wrong"),
            contradiction,
            predicted("traj-b-1", "beta-reply-1"),
            judged(b),
        ]
    )
    monkeypatch.setattr(fidelity, "HttpxJsonTransport", lambda: wire)
    execute_fidelity_evaluation(
        project, prepare_fidelity_evaluation(project, b, code_revision="fixture")
    )
    head = memory_head(project.artifacts, "fixture-memory")
    assert head is not None and head.version == 2
    unrelated, _ = memory_catalog(project.artifacts, head, exclude_public="0" * 64)
    assert old not in unrelated
    own = public_trajectory_identity(fidelity_memory.trajectory_record(b))
    eligible, withheld = memory_catalog(project.artifacts, head, exclude_public=own)
    assert old in eligible
    assert len(withheld) == 3 and not set(withheld) & set(eligible)
    rerun = trajectory(base, "traj-b-frozen", "beta", "frozen", checkpoint=2)
    rerun = FidelitySpec.model_validate(
        {
            **rerun.model_dump(mode="json"),
            "workload": {**b.workload.model_dump(mode="json"), "id": "traj-b-frozen"},
            "references": b.references.model_dump(mode="json"),
        }
    )
    wire = RoutedTransport([predicted("traj-b-0", "x"), predicted("traj-b-1", "y"), judged(rerun)])
    monkeypatch.setattr(fidelity, "HttpxJsonTransport", lambda: wire)
    execute_fidelity_evaluation(
        project, prepare_fidelity_evaluation(project, rerun, code_revision="fixture")
    )
    assert old in notes(wire, 0, "prior_trajectories")
    assert "beta-reply" not in json.dumps(wire.requests[0].payload)
