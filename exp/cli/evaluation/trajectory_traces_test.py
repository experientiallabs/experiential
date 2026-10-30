"""``exp eval --continuations`` exports saved trajectories that ``exp build`` learns from.

The whole loop runs offline: native and simulated ticket-world studies, a dry run, the import,
a learning build on the native import through one scripted provider, the build's exposure-ledger
grouping of forks, and the refusal of the simulated import.
"""

import functools
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from exp.cli.app import app
from exp.cli.evaluation.continuations import ContinuationCommand
from exp.common.core.artifacts import JsonObject
from exp.common.project import ProjectStore
from exp.common.traces import Trace
from exp.common.traces.ingest.persistence import read_ingested_traces
from exp.common.traces.sqlite import SQLiteTraceStore
from exp.common.traces.sqlite_schema import trace_database_path
from exp.common.traces.store import load_trace_dataset
from exp.optimize.evaluation.audit.exposure import held_out_exposure, held_out_members
from exp.optimize.evaluation.audit.panel_test import (
    REVISION,
    SimulatedProvider,
    _call_key,
    build_report,
    write_catalog,
)
from exp.optimize.evaluation.continuations.traces import TrajectoryTraceExport
from exp.optimize.evaluation.continuations.traces_test import (
    capture,
    exchanges,
    panel,
    simulated_tickets,
    study,
    ticket_fork,
    ticket_studies,
    ticket_task,
)
from exp.optimize.evaluation.equivalence.policy_test import registered_policy
from exp.optimize.evaluation.fidelity import runner as fidelity
from exp.optimize.evaluation.learning.traces import learning_cases
from exp.runtime.models import RuntimeModelCatalog
from exp.simulation.mining.bindings import load_task_set_lineage_bindings
from exp.simulation.mining.exposure_ledger import (
    ExposureLedger,
    LedgerExposure,
    LedgerReservation,
    SourceRecord,
    node_vertex,
    record_sha256,
)

TASKS = 6
FORKS = 2


def eval_command(root: Path, request: Path, *flags: str) -> tuple[int, str]:
    """Run one continuation action and return its exit code and flattened output."""
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "case",
            "--root",
            str(root),
            "--continuations",
            str(request),
            "--non-interactive",
            *flags,
        ],
    )
    return result.exit_code, " ".join(result.output.split())


def build_command(root: Path, project: str, import_id: str) -> tuple[int, str]:
    """Run ``exp build PROJECT --import-id ID`` and return its exit code and flattened output."""
    result = CliRunner().invoke(
        app,
        [
            "build",
            project,
            "--import-id",
            import_id,
            "--root",
            str(root),
            "--yes",
            "--non-interactive",
        ],
    )
    return result.exit_code, " ".join(result.output.split())


def offline_provider(monkeypatch: pytest.MonkeyPatch, traces: tuple[Trace, ...]) -> None:
    """Route every build, learning and fidelity request to a provider scripted by the traces."""
    provider = SimulatedProvider([])
    provider.truth = {
        _call_key(name, arguments): reply for name, arguments, reply in exchanges(traces)
    }
    monkeypatch.setenv("HELD_OUT_TEST_KEY", "fixture-only")
    monkeypatch.setenv("EXP_RELEASE_REVISION", REVISION)
    monkeypatch.setattr(
        "exp.cli.build.app.RuntimeModelCatalog",
        functools.partial(RuntimeModelCatalog, transport_factory=lambda: provider),
    )
    monkeypatch.setattr("exp.cli.build.app.capture_build_completed", lambda **_kwargs: None)
    monkeypatch.setattr(fidelity, "HttpxJsonTransport", lambda: provider)


def vertex(ledger: ExposureLedger, record: SourceRecord, trace: Trace) -> str:
    """The trace's content vertex under the ledger's newest computable scheme."""
    scheme = ledger.computable[-1].identity
    return node_vertex(scheme, ledger.vertices(trace)[scheme], record)


def test_export_traces_dry_runs_imports_and_exp_build_learns_with_forks_grouped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Native forks learn as one lineage each; the simulated import is refused."""
    root = tmp_path / ".exp"
    root.mkdir()
    project = ProjectStore(root, "case")
    store = project.artifacts
    policy = registered_policy(store)
    native = ticket_studies(store, tasks=TASKS, forks=FORKS)
    reference, _ = capture(store, simulated_tickets(), ticket_task(99))
    simulated = study(store, reference, simulated_tickets(), [ticket_fork(99, 0)])
    generator = panel(store, "actor-panel-fixture", [*native, simulated])
    request = tmp_path / "traces.json"
    request.write_text(
        ContinuationCommand(
            action="export-traces",
            traces=TrajectoryTraceExport(policy=policy, generators=(generator,), project="learner"),
        ).model_dump_json()
    )
    counts = (
        f"Trajectories: {TASKS * FORKS + 1} branches from {TASKS + 1} studies; generated-native "
        f"{TASKS * FORKS} (0 with simulated or unproven replies excluded), generated-simulated "
        "1, 0 identical collapsed, 0 skipped. No provider calls."
    )
    code, dry = eval_command(root, request, "--dry-run")
    assert code == 0, dry
    assert counts in dry and "Dry run: nothing imported." in dry
    imports = SQLiteTraceStore(trace_database_path(root))
    assert not imports.list_imports("learner")

    code, output = eval_command(root, request)
    assert code == 0, output
    assert counts in output and "exp build refuses them as learning source" in output
    learnable = re.search(r"exp build learner --import-id (import-[0-9a-f]+)", output)
    labelled = re.search(r"generated-simulated: (import-[0-9a-f]+)", output)
    assert learnable is not None and labelled is not None
    native_id, simulated_id = learnable.group(1), labelled.group(1)
    assert set(imports.list_imports("learner")) == {native_id, simulated_id}

    traces = read_ingested_traces(root, native_id).traces
    write_catalog(root, recall=True)
    offline_provider(monkeypatch, traces)
    code, built = build_command(root, "learner", native_id)
    assert code == 0, built
    learner = ProjectStore(root, "learner")
    assert len(learning_cases(learner, "all")) == TASKS * FORKS
    assert f"{TASKS * FORKS} traces" in build_report(learner)

    # Every fork of one captured task stays in its source task's lineage, fit or held out.
    build = learner.load_project().build
    assert build is not None
    dataset_input = build.trace_dataset
    bindings = {
        row.trace_id: row
        for row in load_task_set_lineage_bindings(
            learner.artifacts, build.task_set.artifact_id
        ).bindings
    }
    dataset = load_trace_dataset(learner.artifacts, build.trace_dataset.artifact_id).traces
    forks: dict[str, list[Trace]] = {}
    for trace in dataset:
        assert trace.conversation_id is not None
        forks.setdefault(trace.conversation_id, []).append(trace)
    assert sorted(forks) == [f"ticket-lineage-{index}" for index in range(TASKS)]
    groups = {
        source: {(bindings[t.trace_id].lineage_id, bindings[t.trace_id].partition) for t in rows}
        for source, rows in forks.items()
    }
    assert all(len(rows) == FORKS and len(groups[s]) == 1 for s, rows in forks.items())
    held_out = [s for s, group in groups.items() if next(iter(group))[1] == "held_out"]
    assert held_out and len(held_out) < TASKS

    # The ledger joins forks by declared task identity, so one fork's hold covers the other.
    ledger = ExposureLedger(root)
    snapshot = ledger.snapshot()
    declared = {
        edge.task_lineage
        for edge in snapshot.records.edges
        if edge.reason == "declared-task-identity" and edge.project == "learner"
    }
    assert declared == set(forks)

    def fork_vertices(source: str) -> list[str]:
        """Content vertices of one source task's forks, in dataset order."""
        return [
            vertex(
                ledger,
                SourceRecord(
                    project="learner",
                    dataset=dataset_input,
                    trace_id=trace.trace_id,
                    record_sha256=record_sha256(trace),
                ),
                trace,
            )
            for trace in forks[source]
        ]

    reserved_source = held_out[0]
    lineage = next(iter(groups[reserved_source]))[0]
    members = held_out_members(learner)[lineage].vertices
    first, second = fork_vertices(reserved_source)
    assert {first, second} <= set(members)
    with ledger.transaction() as transaction:
        transaction.reserve(
            LedgerReservation(
                receipt="receipt-other-audit",
                panel="panel-other-audit",
                project="learner",
                lineages=(lineage,),
                vertices=(first,),
                snapshot=transaction.snapshot().identity,
                at=datetime.now(UTC),
            )
        )
    snapshot = ledger.snapshot()
    taken = snapshot.taken((second,), owner=None)
    assert taken is not None and taken.by == ("receipt-other-audit",)
    assert any(
        edge.reason == "declared-task-identity"
        and {edge.left, edge.right} == {first, second}
        and edge.task_lineage == reserved_source
        for edge in snapshot.records.edges
    ), "the forks are joined by their declared task identity, not only by similarity"
    decision = {item.lineage: item.kind for item in held_out_exposure(learner).exposed}
    assert decision == {lineage: "reserved"}

    exposed_source = next(source for source in forks if source != reserved_source)
    shown, sibling = fork_vertices(exposed_source)
    ledger.record_exposure(
        LedgerExposure(
            vertices=(shown,),
            how="inspected",
            panel="panel-other-audit",
            report="report-other-audit",
            receipt="receipt-other-audit",
            project="learner",
            at=datetime.now(UTC),
        )
    )
    exposure = ledger.snapshot().exposure((sibling,))
    assert exposure is not None and exposure.exposed_node == shown

    # The simulated import is refused as learning source in a fresh project.
    request.write_text(
        ContinuationCommand(
            action="export-traces",
            traces=TrajectoryTraceExport(
                policy=policy, generators=(generator,), project="simulated-learner"
            ),
        ).model_dump_json()
    )
    code, output = eval_command(root, request)
    assert code == 0 and simulated_id in output, output
    code, refused = build_command(root, "simulated-learner", simulated_id)
    assert code != 0
    assert "carry simulation provenance" in refused
    assert ProjectStore(root, "simulated-learner").artifacts.list_ids() == ()


def test_export_traces_refuses_a_foreign_payload(tmp_path: Path) -> None:
    """The action accepts only its own payload, like every continuation action."""
    payload: JsonObject = {"action": "export-traces", "analysis": None}
    with pytest.raises(ValueError, match="export-traces requires only its traces payload"):
        ContinuationCommand.model_validate(payload)
