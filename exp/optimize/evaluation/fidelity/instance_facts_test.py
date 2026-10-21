"""Instance facts through learning, held-out audits and miss retests, offline.

A trace package supplies each trace's facts in its request context. Every simulator request is
recorded by the routed fake, so these tests check what each request actually carried.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from exp.cli.app import app
from exp.common.core.artifacts import JsonObject
from exp.common.tasks.facts import InstanceFacts, InstanceFactsError
from exp.common.tasks.facts_test import fact
from exp.optimize.evaluation.audit.panel_test import (
    PROJECT,
    REVISION,
    USERS,
    SimulatedProvider,
    _at,
    _import_id,
    _obj,
    build_project,
    build_report,
    episode,
    held_out_command,
)
from exp.optimize.evaluation.fidelity.instance_facts import ReplayFacts
from exp.optimize.evaluation.fidelity.memory import trajectory_record
from exp.optimize.evaluation.fidelity.replay import DEADLINE, content_run_id
from exp.optimize.evaluation.fidelity.runner import FidelitySpec, prepare_fidelity_evaluation
from exp.optimize.evaluation.fidelity.runner_test import source_project
from exp.optimize.evaluation.learning.traces import learning_cases, learning_plan, learning_spec
from exp.simulation.retrieval.build_inputs import load_completed_build_rag_lineage_bindings

MARKERS = ("FACT-", "LATE-", "PRIVATE-")


def with_facts(index: int) -> JsonObject:
    """An episode whose package supplies its starting world, a later read, and a private fact.

    ``FACT-<user>`` describes the starting state, ``LATE-<user>`` the state after the second
    recorded action, and ``PRIVATE-<user>`` is verifier-only.
    """
    item = episode(index)
    user = str(item["user"])
    item["facts"] = {
        "facts": [
            fact(f"orders/{user}/note", f"FACT-{user}"),
            fact(f"orders/{user}/after-read", f"LATE-{user}", version=2),
            fact(f"orders/{user}/expected", f"PRIVATE-{user}", visibility="verifier_private"),
        ]
    }
    return item


def test_every_replay_conditions_on_its_own_facts_and_nothing_else(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The build's learning and retest, and the audit, see only their own trace's facts.

    Facts reach predictions in the ``instance_facts`` channel; a fact read after an action never
    conditions that action's prediction; verifier-private facts reach no request at all; and no
    fact reaches retrieval, investigation, judging, recall decisions, or published memory.
    """
    provider = SimulatedProvider([episode(i) for i in range(USERS)], learnable=True)
    root, store, output = build_project(
        tmp_path,
        monkeypatch,
        provider,
        episodes=[with_facts(i) for i in range(USERS)],
        learn=True,
    )
    assert "learned" in output
    assert "Report:" in output
    assert "Miss retest" in build_report(store)
    held_out_command(root, 3)

    predictions = provider.predictions()
    assert predictions
    for prediction in predictions:
        own = str(_at(prediction, "task", "initial_context", "task_lineage"))
        channel = _obj(prediction["instance_facts"])
        # The predicted action's position: the public prefix plus the actions already replayed.
        prefix = _at(prediction, "task", "initial_context", "public_prefix")
        conversation = prediction["visible_conversation"]
        assert isinstance(conversation, list)
        replayed = [message for message in conversation if _obj(message)["role"] == "tool"]
        assert isinstance(prefix, list) and len(prefix) == 1
        version = len(prefix) + len(replayed)
        assert channel["state_version"] == version
        rows = channel["facts"]
        assert isinstance(rows, list)
        values = {_obj(row)["value"] for row in rows}
        assert values == ({f"FACT-{own}", f"LATE-{own}"} if version >= 2 else {f"FACT-{own}"})
        rest = {key: value for key, value in prediction.items() if key != "instance_facts"}
        assert not any(marker in json.dumps(rest) for marker in MARKERS)
        assert "PRIVATE-" not in json.dumps(prediction)
    others = [
        request
        for request in provider.requests
        if "submit_transition" not in json.dumps(request.payload.get("tool_choice"))
    ]
    assert others
    keys = [f"orders/user_{i:02d}/" for i in range(USERS)]
    for request in others:
        payload = json.dumps(request.payload)
        assert not any(marker in payload for marker in MARKERS)
        assert not any(key in payload for key in keys), "fact keys never reach investigations"
    consumed = store.artifacts.list_ids(artifact_type="consumed-trajectory")
    assert consumed
    for identity in consumed:
        published = store.artifacts.read_bytes(identity, "trajectory.json").decode()
        assert not any(marker in published for marker in MARKERS)


def test_unvalidated_package_facts_stop_learning_with_a_clear_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A package fact that is not validated is refused before any prediction is bought."""
    episodes = [episode(i) for i in range(USERS)]
    episodes[0]["facts"] = {"facts": [fact("orders/x/note", "x", status="unvalidated")]}
    provider = SimulatedProvider(episodes)
    root, store, _ = build_project(tmp_path, monkeypatch, provider, episodes=episodes)
    built = len(provider.requests)
    result = CliRunner().invoke(
        app,
        ["build", PROJECT, "--import-id", _import_id(store), "--root", str(root)]
        + ["--yes", "--non-interactive"],
    )
    assert result.exit_code != 0
    message = " ".join(result.output.replace("│", " ").split())
    assert "supplies instance facts that are not validated: orders/x/note (unvalidated)" in message
    assert len(provider.requests) == built


def test_changing_a_fact_changes_the_run_id(tmp_path: Path) -> None:
    """Facts are run inputs: a different value addresses a different run."""
    _, spec = source_project(tmp_path)

    def facts(value: str) -> FidelitySpec:
        """The spec conditioned on one starting fact."""
        supplied = InstanceFacts.model_validate({"facts": [fact("entry/current", value)]})
        return FidelitySpec.model_validate(
            {
                **spec.model_dump(mode="json"),
                "facts": ReplayFacts(facts=supplied).model_dump(mode="json"),
            }
        )

    plain = content_run_id(spec, "learn", "t", code_revision=REVISION)
    first = content_run_id(facts("a"), "learn", "t", code_revision=REVISION)
    second = content_run_id(facts("b"), "learn", "t", code_revision=REVISION)
    assert len({plain, first, second}) == 3
    assert content_run_id(facts("a"), "learn", "t", code_revision=REVISION) == first
    assert "entry/current" not in json.dumps(trajectory_record(facts("a")))
    with pytest.raises(ValueError, match="not validated"):
        ReplayFacts(
            facts=InstanceFacts.model_validate({"facts": [fact("a", status="unvalidated")]})
        )


def test_held_out_derived_facts_condition_only_their_own_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fit trace's run refuses facts derived from a held-out trace, or an unknown one."""
    provider = SimulatedProvider([episode(i) for i in range(USERS)])
    _, store, _ = build_project(tmp_path, monkeypatch, provider)
    build = store.load_project().build
    assert build is not None
    bindings = load_completed_build_rag_lineage_bindings(store.artifacts, build)
    held = next(row.trace_id for row in bindings if row.partition == "held_out")
    fit_traces = [row.trace_id for row in bindings if row.partition == "fit"]
    case = learning_cases(store, "fit")[0]
    plan = learning_plan(store, "jev", "fit")

    def register(derived: tuple[str, ...]) -> None:
        """Register the fit case's learning run with one fact derived from ``derived``."""
        row = fact("orders/derived", "value")
        provenance = row["provenance"]
        assert isinstance(provenance, dict)
        provenance["derived_from_traces"] = list(derived)
        facts = InstanceFacts.model_validate({"facts": [row]})
        spec = learning_spec(
            store,
            plan,
            case.model_copy(update={"facts": facts}),
            deadline=datetime.now(UTC) + DEADLINE,
        )
        prepare_fidelity_evaluation(store, spec, code_revision=REVISION)

    with pytest.raises(InstanceFactsError, match=f"derived from held-out trace {held}"):
        register((held,))
    with pytest.raises(InstanceFactsError, match="not in this project's build"):
        register(("unknown-trace",))
    other = next(trace for trace in fit_traces if trace != case.trace_id)
    register((case.trace_id, other))
