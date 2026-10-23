"""Memory belongs to knowledge, reserved partitions are declared, and the tree reaches prompts."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from exp.common.core.artifacts import ArtifactInput
from exp.common.models import (
    BillingSource,
    ModelRecord,
    load_model_catalog,
    write_model_catalog,
)
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation import fidelity
from exp.optimize.evaluation.fidelity import (
    FidelitySpec,
    PreparedFidelity,
    execute_fidelity_evaluation,
    prepare_fidelity_evaluation,
)
from exp.optimize.evaluation.fidelity_memory import (
    saved_memory_receipt,
)
from exp.optimize.evaluation.tests.fidelity_lifecycle_test import learn_alpha
from exp.optimize.evaluation.tests.fidelity_memory_test import (
    RoutedTransport,
    investigation,
    judged,
    memory_project,
    predicted,
    trajectory,
)
from exp.runtime.models import RuntimeModelCatalog
from exp.runtime.models.providers.transport import JsonHttpResponse
from exp.simulation.engines.text.knowledge import SimulationKnowledge
from exp.simulation.reconstruction.library import LibraryEntry, LibrarySnapshot, extend_library
from exp.simulation.world_model.artifact import persist_grounded_world_model
from exp.simulation.world_model.experience_memory import (
    Eligibility,
    MemoryVersion,
    is_reserved,
    load_source,
    memory_catalog,
    memory_head,
    provenance_gaps,
)
from exp.simulation.world_model.memory_tree import overview, read_summary
from exp.simulation.world_model.runtime import load_grounded_world_model_artifact
from exp.simulation.world_model.session_research_test import payload, structured_reply


def with_lineage(spec: FidelitySpec, task_lineage: str, **memory: object) -> FidelitySpec:
    """Same spec under another task lineage and memory options."""
    value = spec.model_dump(mode="json")
    value["task"]["lineage_group_id"] = task_lineage
    value["memory"] = {**value["memory"], **memory}
    return FidelitySpec.model_validate(value)


def rebind(
    project: ProjectStore,
    spec: FidelitySpec,
    *,
    top_k: int | None = None,
    knowledge: SimulationKnowledge | None = None,
) -> FidelitySpec:
    """Serve the same knowledge with another model or serving policy, or change knowledge."""
    catalog = load_model_catalog(project.model_catalog_path)
    record = catalog.models["model"]
    catalog = catalog.model_copy(
        update={
            "models": {
                **catalog.models,
                "model-b": ModelRecord(
                    connection=record.connection,
                    model="fixture-v2",
                    billing_source=BillingSource.CUSTOMER_MANAGED,
                    capabilities=record.capabilities,
                ),
            }
        }
    )
    write_model_catalog(project.model_catalog_path, catalog)
    original = load_grounded_world_model_artifact(project.artifacts, spec.world_model)
    snapshot, _ = RuntimeModelCatalog(catalog).snapshot("model-b")
    rebound = persist_grounded_world_model(
        project.artifacts,
        original.serving_rag,
        model_alias="model-b",
        model=snapshot,
        created_at=datetime.now(UTC),
        code_revision="fixture",
        top_k=top_k or original.top_k,
        knowledge=knowledge or original.knowledge,
        selection_policy_sha256=original.selection_policy_sha256,
    )
    return spec.model_copy(update={"world_model": artifact_input(rebound.manifest)})


def served(response: JsonHttpResponse, model: str) -> JsonHttpResponse:
    """The same provider response, reported by another served model."""
    body = response.body
    assert isinstance(body, dict)
    return JsonHttpResponse(status_code=response.status_code, body={**body, "model": model})


def test_rebinding_the_model_continues_memory_but_new_knowledge_does_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lineage belongs to what the simulator knows, not to the model serving it."""
    project, base = memory_project(tmp_path)
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    learn_alpha(project, base, monkeypatch)
    b = rebind(project, trajectory(base, "traj-b", "beta", "learn"))
    wire = RoutedTransport(
        [
            served(predicted("traj-b-0", "beta-reply-0"), "fixture-v2"),
            served(predicted("traj-b-1", "beta-reply-1"), "fixture-v2"),
            judged(b),
            served(structured_reply({"summary": "beta and alpha compressed"}), "fixture-v2"),
        ]
    )
    monkeypatch.setattr(fidelity, "HttpxJsonTransport", lambda: wire)
    prepared = prepare_fidelity_evaluation(project, b, code_revision="fixture")
    execute_fidelity_evaluation(project, prepared)
    receipt = saved_memory_receipt(project, prepared)
    assert receipt is not None and receipt["version"] == 2 and receipt["published"] is True
    head = memory_head(project.artifacts, "fixture-memory")
    assert head is not None and head.serving_policy is not None
    assert head.serving_policy["model_id"] == "fixture-v2"
    policy = rebind(project, trajectory(base, "traj-c", "gamma", "learn"), top_k=2)
    prepare_fidelity_evaluation(project, policy, code_revision="fixture")
    other = rebind(
        project,
        trajectory(base, "traj-d", "delta", "learn"),
        knowledge=SimulationKnowledge(rules="a different behavioral model"),
    )
    with pytest.raises(ValueError, match="different knowledge"):
        prepare_fidelity_evaluation(project, other, code_revision="fixture")


def test_prediction_guidance_is_serving_policy_but_rules_are_knowledge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tuning only prediction guidance reuses learned memory; obsolete checkpoints are rejected."""
    project, base = memory_project(tmp_path)
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    known = SimulationKnowledge(rules="Retain observed identity and state.", authored=True)
    served_by = load_grounded_world_model_artifact(project.artifacts, base.world_model)
    learner = persist_grounded_world_model(
        project.artifacts,
        served_by.serving_rag,
        model_alias=served_by.model_alias,
        model=served_by.model,
        created_at=datetime.now(UTC),
        code_revision="fixture",
        top_k=served_by.top_k,
        knowledge=known,
        selection_policy_sha256=served_by.selection_policy_sha256,
    )
    base = base.model_copy(update={"world_model": artifact_input(learner.manifest)})
    learn_alpha(project, base, monkeypatch)
    tuned = known.model_copy(update={"prediction_guidance": "Give concise tool replies."})
    reuse = rebind(
        project, trajectory(base, "new-guide", "beta", "frozen", checkpoint=1), knowledge=tuned
    )
    prepare_fidelity_evaluation(project, reuse, code_revision="fixture")
    changed = known.model_copy(update={"rules": "A different behavioral model."})
    other = rebind(
        project, trajectory(base, "new-rules", "gamma", "frozen", checkpoint=1), knowledge=changed
    )
    with pytest.raises(ValueError, match="different knowledge"):
        prepare_fidelity_evaluation(project, other, code_revision="fixture")
    head = memory_head(project.artifacts, "fixture-memory")
    assert head is not None and head.knowledge_identity_scheme == 2
    assert head.serving_policy is not None and "prediction_guidance_sha256" in head.serving_policy
    legacy = head.model_dump(mode="json")
    legacy["knowledge_identity_scheme"] = 1
    with pytest.raises(ValidationError, match="knowledge_identity_scheme"):
        MemoryVersion.model_validate(legacy)
    for field in (
        "knowledge_identity",
        "knowledge_identity_scheme",
        "publisher",
        "serving_policy",
        "log_length",
        "appended",
    ):
        incomplete = head.model_dump(mode="json")
        del incomplete[field]
        with pytest.raises(ValidationError, match=field):
            MemoryVersion.model_validate(incomplete)


def test_reserved_partition_is_declared_and_enforced_both_ways(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reserved lineages are never learned; learned lineages are never reserved."""
    project, base = memory_project(tmp_path)
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    learn_alpha(project, base, monkeypatch)
    learned = with_lineage(
        trajectory(base, "held-learned", "gamma", "frozen", checkpoint=1), "other", reserved=True
    )
    with pytest.raises(ValueError, match="already consumed"):
        prepare_fidelity_evaluation(project, learned, code_revision="fixture")
    late = with_lineage(trajectory(base, "late-learner", "delta", "learn"), "held")
    late_plan = prepare_fidelity_evaluation(project, late, code_revision="fixture")
    held = with_lineage(
        trajectory(base, "held-eval", "gamma", "frozen", checkpoint=1), "held", reserved=True
    )
    wire = RoutedTransport(
        [predicted("held-eval-0", "x"), predicted("held-eval-1", "y"), judged(held)]
    )
    monkeypatch.setattr(fidelity, "HttpxJsonTransport", lambda: wire)
    execute_fidelity_evaluation(
        project, prepare_fidelity_evaluation(project, held, code_revision="fixture")
    )
    assert is_reserved(project.artifacts, "fixture-memory", "held")
    early = with_lineage(trajectory(base, "early-learner", "delta", "learn"), "held")
    with pytest.raises(ValueError, match="reserved for evaluation"):
        prepare_fidelity_evaluation(project, early, code_revision="fixture")
    with pytest.raises(ValueError, match="reserved for evaluation"):
        execute_fidelity_evaluation(project, late_plan)
    head = memory_head(project.artifacts, "fixture-memory")
    assert head is not None and head.version == 1


def test_compressed_memory_reaches_the_next_trajectory_overview(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A learning run compresses after publishing; the next run sees the summary."""
    project, base = memory_project(tmp_path)
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    a = trajectory(base, "traj-a", "alpha", "learn")
    wire = RoutedTransport(
        [
            predicted("traj-a-0", "wrong"),
            investigation("alpha"),
            predicted("traj-a-1", "alpha-reply-1"),
            judged(a),
            structured_reply({"summary": "alpha: one wrong read, [unresolved] value differs"}),
        ]
    )
    monkeypatch.setattr(fidelity, "HttpxJsonTransport", lambda: wire)
    first = prepare_fidelity_evaluation(project, a, code_revision="fixture")
    execute_fidelity_evaluation(project, first)
    maintenance = json.loads(
        project.artifacts.read_bytes(first.id + "-memory-maintenance", "maintenance.json")
    )
    assert maintenance["log_length"] == 2 and len(maintenance["built"]) == 1
    summary = read_summary(project.artifacts, "fixture-memory", 0, 2)
    assert summary is not None and summary.text.startswith("alpha")
    b = with_lineage(trajectory(base, "traj-b", "beta", "learn"), "other", overview_lines=1)
    wire = RoutedTransport(
        [
            predicted("traj-b-0", "beta-reply-0"),
            predicted("traj-b-1", "beta-reply-1"),
            judged(b),
        ]
    )
    monkeypatch.setattr(fidelity, "HttpxJsonTransport", lambda: wire)
    execute_fidelity_evaluation(
        project, prepare_fidelity_evaluation(project, b, code_revision="fixture")
    )
    history = payload(wire, 0)["research_history"]
    assert isinstance(history, dict)
    rows = history["memory_overview"]
    assert isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0], dict)
    assert rows[0]["id"] == "#0-1" and rows[0]["kind"] == "summary"
    assert rows[0]["text"] == summary.text
    assert "beta-reply" not in json.dumps(payload(wire, 0))


def test_reserved_source_and_dependent_summaries_leave_the_eligible_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Eligibility follows typed provenance through records, lines and compressed summaries."""
    project, base = memory_project(tmp_path)
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    a = trajectory(base, "traj-a", "alpha", "learn")
    wire = RoutedTransport(
        [
            predicted("traj-a-0", "wrong"),
            investigation("alpha"),
            predicted("traj-a-1", "alpha-reply-1"),
            judged(a),
            structured_reply({"summary": "alpha compressed"}),
        ]
    )
    monkeypatch.setattr(fidelity, "HttpxJsonTransport", lambda: wire)
    first = prepare_fidelity_evaluation(project, a, code_revision="fixture")
    execute_fidelity_evaluation(project, first)
    b = with_lineage(trajectory(base, "traj-b", "beta", "learn"), "beta-lineage")
    wire = RoutedTransport(
        [
            predicted("traj-b-0", "wrong"),
            investigation("beta"),
            predicted("traj-b-1", "beta-reply-1"),
            judged(b),
            structured_reply({"summary": "beta compressed"}),
            structured_reply({"summary": "alpha and beta compressed"}),
        ]
    )
    monkeypatch.setattr(fidelity, "HttpxJsonTransport", lambda: wire)
    second = prepare_fidelity_evaluation(project, b, code_revision="fixture")
    execute_fidelity_evaluation(project, second)
    head = memory_head(project.artifacts, "fixture-memory")
    assert head is not None and head.log_length == 4
    root = read_summary(project.artifacts, "fixture-memory", 0, 4)
    assert root is not None and root.text == "alpha and beta compressed"
    alpha_ids, beta_ids = research_ids(project, first), research_ids(project, second)
    rule = Eligibility(reserved_task_lineages=("other",), strict=True)
    records, withheld = memory_catalog(project.artifacts, head, eligibility=rule)
    assert not set(alpha_ids) & set(records) and set(alpha_ids) <= set(withheld)
    # B's notes were written after B was shown A's memory, so they depend on A as well.
    assert set(beta_ids) <= set(withheld)
    sources = {
        row["trajectory"] for row in records.values() if row.get("kind") == "original-observation"
    }
    assert len(sources) == 1

    def admits(reference: ArtifactInput) -> bool:
        """The run-time predicate: resolve the typed source, then apply the rule."""
        return rule.admits(load_source(project.artifacts, reference))

    full = overview(project.artifacts, "fixture-memory", 4, 1)
    assert [row["id"] for row in full] == ["#0-3"]
    visible = overview(project.artifacts, "fixture-memory", 4, 2, admits=admits)
    # B's summary depends on A through B's hypothesis, so only B's observed line remains.
    assert [row["id"] for row in visible] == ["#2"]
    assert visible[0]["kind"] == "observed"
    assert "alpha" not in json.dumps(visible)

    library = _library_of(project, head)
    legacy = LibraryEntry(
        entry_id="legacy-note",
        kind="research",
        artifact=library.entries[1].artifact,
        path=library.entries[1].path,
        payload_sha256=library.entries[1].payload_sha256,
        disposition="unresolved",
    )
    snapshot = extend_library(
        project.artifacts,
        [legacy],
        parent=library,
        created_at=datetime.now(UTC),
        code_revision="fixture",
    )
    unknown = head.model_copy(
        update={"library": artifact_input(project.artifacts.read(snapshot.library_id).manifest)}
    )
    assert "legacy-note" in provenance_gaps(project.artifacts, unknown)
    strict, hidden = memory_catalog(
        project.artifacts, unknown, eligibility=Eligibility(strict=True)
    )
    assert "legacy-note" not in strict and "legacy-note" in hidden
    relaxed, _ = memory_catalog(project.artifacts, unknown, eligibility=Eligibility())
    assert "legacy-note" in relaxed


def _library_of(project: ProjectStore, version: MemoryVersion) -> LibrarySnapshot:
    """The exact library snapshot a version pins."""
    raw = project.artifacts.read_bytes(version.library.artifact_id, "library.json")
    return LibrarySnapshot.model_validate_json(raw)


def research_ids(project: ProjectStore, prepared: PreparedFidelity) -> list[str]:
    """Research record identities a learning run published."""
    receipt = saved_memory_receipt(project, prepared)
    assert receipt is not None
    ids = receipt["research_ids"]
    assert isinstance(ids, list)
    return [str(identity) for identity in ids]
