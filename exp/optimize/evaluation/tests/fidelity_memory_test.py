"""Cross-trajectory memory through the TUI executor, native JEV wire and durable stores."""

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from exp.cli.app import app
from exp.common.core.artifacts import ArtifactInput, JsonObject
from exp.common.evaluations.reconstruction import EvaluationReferences, ReferenceStep
from exp.common.models import (
    BillingSource,
    ConnectionConfig,
    ModelCapabilities,
    ModelRecord,
    ToolCall,
    load_model_catalog,
    write_model_catalog,
)
from exp.common.project import ArtifactStore, ProjectStore, artifact_input
from exp.optimize.evaluation import fidelity
from exp.optimize.evaluation.fidelity import (
    FidelityRetrieval,
    FidelitySpec,
    execute_fidelity_evaluation,
    prepare_fidelity_evaluation,
)
from exp.optimize.evaluation.fidelity_memory import (
    saved_memory_receipt,
)
from exp.optimize.evaluation.fidelity_test import source_project, transition_reply
from exp.runtime.models.providers.transport import JsonHttpResponse, ScriptedJsonTransport
from exp.simulation.engines.text.knowledge import SimulationKnowledge
from exp.simulation.evaluation import FixedWorkflow
from exp.simulation.retrieval.jev import JEVCueSelector, JEVPolicy
from exp.simulation.world_model.artifact import (
    GroundedWorldModelArtifact,
    persist_grounded_world_model,
)
from exp.simulation.world_model.experience_memory import (
    MemoryVersion,
    memory_head,
    memory_records,
    publish_trajectory,
)
from exp.simulation.world_model.runtime import load_grounded_world_model_artifact
from exp.simulation.world_model.session_research import OnlineResearchPolicy
from exp.simulation.world_model.session_research_test import payload, structured_reply


class RoutedTransport(ScriptedJsonTransport):
    """Serve model calls from a script and native decisions from rubric-shaped answers."""

    def __init__(self, responses: list[JsonHttpResponse]) -> None:
        """Keep model responses ordered; decisions score every presented record."""
        super().__init__(responses)
        self.decisions: list[JsonObject] = []

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Answer /systemone with exactly the registered question inventory."""
        if not url.endswith("/systemone"):
            return super().post(
                url, headers=headers, payload=payload, timeout_seconds=timeout_seconds
            )
        self.decisions.append(payload)
        questions = payload["questions"]
        assert isinstance(questions, dict)
        answers: JsonObject = {
            key: {"type": "choice", "choice": "entity-state"}
            if key == "dependency"
            else {"type": "score", "score": 2}
            for key in questions
        }
        return JsonHttpResponse(status_code=200, body={"answers": answers})


def memory_project(root: Path) -> tuple[ProjectStore, FidelitySpec]:
    """Add a native decision alias to the shared fixture project."""
    project, spec = source_project(root)
    catalog = load_model_catalog(project.model_catalog_path)
    catalog = catalog.model_copy(
        update={
            "connections": {
                **catalog.connections,
                "decisions": ConnectionConfig(
                    provider="typesafe",
                    base_url="https://fixture.invalid/v1",
                    api_key_env="FIDELITY_TEST_KEY",
                    trusted_custom_origin=True,
                ),
            },
            "models": {
                **catalog.models,
                "jev": ModelRecord(
                    connection="decisions",
                    model="jev-fixture",
                    billing_source=BillingSource.CUSTOMER_MANAGED,
                    capabilities=ModelCapabilities(supports_completions=False),
                ),
            },
        }
    )
    write_model_catalog(project.model_catalog_path, catalog)
    return project, spec


def test_retrieval_without_memory_has_no_research_delivery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retrieval-only runs predict and judge without an unbound memory selector."""
    project, base = memory_project(tmp_path)
    spec = FidelitySpec.model_validate(
        {
            **base.model_dump(mode="json"),
            "output_protocol": "json-tool",
            "retrieval": FidelityRetrieval(
                jev_alias="jev", jev=JEVPolicy(candidate_limit=4), memory_top_k=4
            ).model_dump(mode="json"),
        }
    )
    wire = RoutedTransport(
        [predicted("call-0", "reply-0"), predicted("call-1", "reply-1"), judged(spec)]
    )
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    monkeypatch.setattr(fidelity, "HttpxJsonTransport", lambda: wire)
    result = execute_fidelity_evaluation(
        project, prepare_fidelity_evaluation(project, spec, code_revision="fixture")
    )
    assert result.mechanical["valid_steps"] == 2
    assert result.judge_status == "completed"
    assert wire.decisions == []


def trajectory(
    base: FidelitySpec, run_id: str, key: str, mode: str, checkpoint: int | None = None
) -> FidelitySpec:
    """Build one distinct trajectory spec sharing the simulator and memory lineage."""
    actions = tuple(
        ToolCall(call_id=f"{run_id}-{i}", name="inspect", arguments={"key": key}) for i in range(2)
    )
    references = EvaluationReferences(
        capability="recorded_reference_only",
        description="REFERENCE_ONLY_SENTINEL",
        steps=tuple(
            ReferenceStep(
                step=i,
                action={"name": a.name, "arguments": a.arguments},
                reply_available=True,
                reply=f"{key}-reply-{i}",
            )
            for i, a in enumerate(actions)
        ),
    )
    value = {
        **base.model_dump(mode="json"),
        "run_id": run_id,
        "workload": FixedWorkflow(
            id=run_id, task=base.task.instruction, actions=actions
        ).model_dump(mode="json"),
        "references": references.model_dump(mode="json"),
        "output_protocol": "json-tool",
        "maximum_physical_attempts": 20,
        "retrieval": {
            "jev_alias": "jev",
            "jev": JEVPolicy(candidate_limit=4).model_dump(mode="json"),
            "memory_top_k": 4,
        },
        "memory": {"lineage": "fixture-memory", "mode": mode, "checkpoint": checkpoint},
    }
    if mode == "learn":
        selector = JEVCueSelector(
            policy=JEVPolicy(candidate_limit=4),
            top_k=4,
            decide=lambda request: {},
            record=lambda receipt: None,
        )
        value["online_research"] = OnlineResearchPolicy(
            feedback_source="disclosed-native-reply", selection_policy_sha256=selector.identity
        ).model_dump(mode="json")
    return FidelitySpec.model_validate(value)


def notes(wire: ScriptedJsonTransport, index: int, key: str) -> list[str]:
    """Return delivered research record IDs for one history partition of a request."""
    history = payload(wire, index)["research_history"]
    assert isinstance(history, dict)
    rows = history[key]
    assert isinstance(rows, list)
    return [str(row["id"]) for row in rows if isinstance(row, dict)]


def predicted(call_id: str, reply: str) -> JsonHttpResponse:
    """Return one typed transition proposal."""
    return transition_reply(
        {
            "observations": [{"call_id": call_id, "reply": reply, "encoding": "json"}],
            "effect": {
                "kind": "edits",
                "edits": [{"op": "set", "path": ["last"], "value": call_id}],
            },
        }
    )


def investigation(label: str) -> JsonHttpResponse:
    """Return one provisional investigator output."""
    return structured_reply(
        {
            "discrepancy": f"{label} reply differed",
            "hypothesis": f"{label} entries may use a different value",
            "scope": f"{label} entry reads",
            "uncertainty": "One observation only",
            "supersedes": [],
            "transferable": True,
            "observable_cue": "the same tool and arguments",
        }
    )


def judged(spec: FidelitySpec) -> JsonHttpResponse:
    """Return a complete five-axis judge response."""
    return structured_reply(
        {
            "axes": {
                d.dimension_id: {
                    "status": "assessed",
                    "score": 3,
                    "rationale": "Fixture",
                    "evidence_refs": ["step-0"],
                    "critical_flags": [],
                }
                for d in spec.judge.definition.dimensions
            }
        }
    )


def test_learning_reuse_restart_duplicate_and_frozen_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Investigations persist across trajectories and frozen checkpoints never learn."""
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
    result_a = execute_fidelity_evaluation(
        project, prepare_fidelity_evaluation(project, a, code_revision="fixture")
    )
    rows = result_a.mechanical["steps"]
    assert isinstance(rows, list) and [r["trajectory_reply_exact"] for r in rows] == [False, True]
    receipt_a = saved_memory_receipt(
        project, prepare_fidelity_evaluation(project, a, code_revision="fixture")
    )
    assert receipt_a is not None and receipt_a["published"] is True
    assert receipt_a["version"] == 1 and receipt_a["consumed_trajectories"] == 1
    research_a = receipt_a["research_ids"]
    assert isinstance(research_a, list) and len(research_a) == 1
    assert notes(wire, 2, "session") == research_a
    assert len(wire.decisions) == 1

    b = trajectory(base, "traj-b", "beta", "learn")
    wire = RoutedTransport(
        [
            predicted("traj-b-0", "wrong"),
            investigation("beta"),
            predicted("traj-b-1", "beta-reply-1"),
            judged(b),
        ]
    )
    monkeypatch.setattr(fidelity, "HttpxJsonTransport", lambda: wire)
    prepared_b = prepare_fidelity_evaluation(project, b, code_revision="fixture")
    execute_fidelity_evaluation(project, prepared_b)
    first_b = payload(wire, 0)
    assert first_b["environment_state"] == {}
    assert notes(wire, 0, "session") == []
    assert notes(wire, 0, "prior_trajectories") == research_a
    assert "beta-reply" not in json.dumps(first_b)
    receipt_b = saved_memory_receipt(project, prepared_b)
    assert receipt_b is not None and receipt_b["version"] == 2
    research_b = receipt_b["research_ids"]
    assert isinstance(research_b, list) and len(research_b) == 1
    head = memory_head(project.artifacts, "fixture-memory")
    assert head is not None and len(head.consumed) == 2
    assert set(memory_records(project.artifacts, head)) == {*research_a, *research_b}

    # Restart: completed trajectories replay without provider work or a new version.
    wire = RoutedTransport([])
    monkeypatch.setattr(fidelity, "HttpxJsonTransport", lambda: wire)
    monkeypatch.delenv("FIDELITY_TEST_KEY")
    prepared_a = prepare_fidelity_evaluation(project, a, code_revision="fixture")
    assert execute_fidelity_evaluation(project, prepared_a) == result_a
    assert not wire.requests and not wire.decisions
    assert memory_head(project.artifacts, "fixture-memory") == head
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")

    # A repeated draw of trajectory A is predicted afresh but is not consumed again.
    again = trajectory(base, "traj-a-draw-2", "alpha", "learn")
    again = FidelitySpec.model_validate(
        {
            **again.model_dump(mode="json"),
            "workload": {**a.workload.model_dump(mode="json"), "id": "traj-a-draw-2"},
            "references": a.references.model_dump(mode="json"),
        }
    )
    wire = RoutedTransport(
        [
            predicted("traj-a-0", "alpha-reply-0"),
            predicted("traj-a-1", "alpha-reply-1"),
            judged(again),
        ]
    )
    monkeypatch.setattr(fidelity, "HttpxJsonTransport", lambda: wire)
    prepared_again = prepare_fidelity_evaluation(project, again, code_revision="fixture")
    execute_fidelity_evaluation(project, prepared_again)
    duplicate = saved_memory_receipt(project, prepared_again)
    assert duplicate is not None and duplicate["published"] is False
    assert duplicate["consumed_trajectories"] == 2
    assert memory_head(project.artifacts, "fixture-memory") == head

    # Reserved evaluation reads checkpoint 1 only and never discloses or learns.
    held = trajectory(base, "traj-held", "gamma", "frozen", checkpoint=1)
    assert held.online_research is None
    wire = RoutedTransport(
        [predicted("traj-held-0", "x"), predicted("traj-held-1", "y"), judged(held)]
    )
    monkeypatch.setattr(fidelity, "HttpxJsonTransport", lambda: wire)
    prepared_held = prepare_fidelity_evaluation(project, held, code_revision="fixture")
    held_result = execute_fidelity_evaluation(project, prepared_held)
    assert held_result.mechanical["valid_steps"] == 2
    assert notes(wire, 1, "prior_trajectories") == research_a
    assert notes(wire, 1, "session") == []
    assert "gamma-reply" not in json.dumps([r.payload for r in wire.requests[:2]])
    assert all(str(research_b[0]) not in json.dumps(d) for d in wire.decisions)
    assert saved_memory_receipt(project, prepared_held) is None
    assert memory_head(project.artifacts, "fixture-memory") == head

    spec_path = tmp_path / "held.json"
    spec_path.write_text(held.model_dump_json())
    shown = CliRunner().invoke(
        app,
        ["eval", "fidelity-test", "--root", str(tmp_path), "--fidelity", str(spec_path)]
        + ["--non-interactive", "--verbose"],
    )
    assert shown.exit_code == 0, shown.output
    assert "frozen run read v1" in shown.output and "Whole trajectory exact: no" in shown.output
    assert "read v1, unchanged" in shown.output


def minimal(reply: str) -> JsonObject:
    """A one-step consumed trajectory owned by its own task lineage."""
    return {
        "task_lineage": "task-" + reply,
        "actions": [{"name": "read", "arguments": {}}],
        "observed_replies": [reply],
    }


def test_boundaries_refuse_wrong_scope_stale_parent_and_missing_reranker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Memory cannot cross knowledge, fork, or bypass an artifact-bound reranker."""
    project, base = memory_project(tmp_path)
    with pytest.raises(ValueError, match="checkpoint"):
        trajectory(base, "bad", "k", "frozen")
    missing = trajectory(base, "missing", "k", "frozen", checkpoint=1)
    with pytest.raises(ValueError, match="has no version 1"):
        prepare_fidelity_evaluation(project, missing, code_revision="fixture")
    store = project.artifacts
    now = datetime.now(UTC)
    simulator = load_grounded_world_model_artifact(store, base.world_model)

    def publish(parent: MemoryVersion | None, reply: str, selector: str) -> MemoryVersion:
        """Publish one minimal trajectory under the given memory selector."""
        version, _ = publish_trajectory(
            store,
            lineage="scope",
            parent=parent,
            world_model=base.world_model,
            selection_policy_sha256=selector,
            trajectory=minimal(reply),
            research=(),
            created_at=now,
            code_revision="fixture",
            publisher="fixture",
        )
        return version

    first = publish(None, "one", "0" * 64)
    assert first.version == 1 and first.knowledge_identity is not None
    second = publish(first, "two", "1" * 64)
    assert second.version == 2
    assert second.serving_policy is not None
    assert second.serving_policy["memory_selector_sha256"] == "1" * 64
    with pytest.raises(ValueError, match="advanced"):
        publish(first, "three", "0" * 64)
    changed = persist_grounded_world_model(
        store,
        simulator.serving_rag,
        model_alias=simulator.model_alias,
        model=simulator.model,
        created_at=now,
        code_revision="fixture",
        top_k=simulator.top_k,
        knowledge=SimulationKnowledge(rules="other"),
        selection_policy_sha256=simulator.selection_policy_sha256,
    )
    with pytest.raises(ValueError, match="different knowledge"):
        publish_trajectory(
            store,
            lineage="scope",
            parent=second,
            world_model=artifact_input(changed.manifest),
            selection_policy_sha256="0" * 64,
            trajectory=minimal("four"),
            research=(),
            created_at=now,
            code_revision="fixture",
            publisher="fixture",
        )
    unbound = base.model_copy(update={"run_id": "unbound"})
    original = fidelity.load_grounded_world_model_artifact

    def bound(store: ArtifactStore, reference: ArtifactInput) -> GroundedWorldModelArtifact:
        """Load the fixture artifact as if it bound a JEV reranker policy."""
        artifact = original(store, reference)
        return artifact.model_copy(
            update={"selection_policy_sha256": JEVPolicy(candidate_limit=4).identity}
        )

    monkeypatch.setattr(fidelity, "load_grounded_world_model_artifact", bound)
    with pytest.raises(ValueError, match="binds a JEV reranker"):
        prepare_fidelity_evaluation(project, unbound, code_revision="fixture")


def test_wire_tool_schemas_are_reference_free_and_string_effects_stay_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Served models see inline shapes; a stringified effect is still refused, not coerced.

    The refused shape is the one GLM 5.3 Flash returned under the referenced schema
    (saved tau-efficacy-flash proposals): ``effect`` as a JSON-encoded string.
    """
    project, base = memory_project(tmp_path)
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    a = trajectory(base, "traj-wire", "alpha", "learn")
    stringified = transition_reply(
        {
            "observations": [
                {"call_id": "traj-wire-1", "reply": "alpha-reply-1", "encoding": "json"}
            ],
            "effect": json.dumps(
                {"kind": "edits", "edits": [{"op": "set", "path": ["last"], "value": "x"}]}
            ),
        }
    )
    wire = RoutedTransport(
        [predicted("traj-wire-0", "wrong"), investigation("alpha"), stringified, judged(a)]
    )
    monkeypatch.setattr(fidelity, "HttpxJsonTransport", lambda: wire)
    prepared = prepare_fidelity_evaluation(project, a, code_revision="fixture")
    result = execute_fidelity_evaluation(project, prepared)
    schemas = []
    for request in wire.requests:
        tools = request.payload.get("tools")
        for tool in tools if isinstance(tools, list) else []:
            if isinstance(tool, dict) and isinstance(tool.get("function"), dict):
                schemas.append(tool["function"]["parameters"])
    assert len(schemas) == 4
    assert all("$ref" not in json.dumps(schema) for schema in schemas)
    assert result.mechanical["valid_steps"] == 1 and result.mechanical["failed_steps"] == 1
    receipt = saved_memory_receipt(project, prepared)
    assert receipt is not None and receipt["published"] is False
