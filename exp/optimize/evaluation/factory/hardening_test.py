"""Saturated worlds are hardened level by level on workbook and record worlds, with lineage kept."""

import io
import json
from collections.abc import Mapping
from pathlib import Path

import pytest
from openpyxl import Workbook
from pydantic import BaseModel, JsonValue

from exp.common.core.artifacts import ArtifactEnvelope, JsonObject, sha256_json
from exp.common.models import load_model_catalog
from exp.common.project import ProjectStore, artifact_input
from exp.common.project.manifests import ArtifactManifest
from exp.optimize.evaluation.continuations.actors import PanelActor
from exp.optimize.evaluation.executable.task import load_executable_task
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.factory.difficulty import SaturationRule
from exp.optimize.evaluation.factory.hardening import (
    TaskHardeningSpec,
    run_task_hardening,
)
from exp.optimize.evaluation.factory.leakage import ValidityRule
from exp.optimize.evaluation.factory.validity_test import (
    closed,
    judge,
    tool_reply,
    validity_settings,
)
from exp.optimize.evaluation.factory.variants import load_task_variant
from exp.optimize.evaluation.factory.worlds import WorldProposalSpec, propose_worlds
from exp.optimize.evaluation.factory.worlds_test import world_fixture
from exp.runtime.environments.workbook_edits import workbook_projection
from exp.runtime.models.providers.transport import JsonHttpResponse, ScriptedJsonTransport
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.simulation.continuations import load_continuation_checkpoint

AUTHOR = "exp.optimize.evaluation.factory.authoring.HttpxJsonTransport"
ACTOR = "exp.optimize.evaluation.continuations.job.HttpxJsonTransport"


def _drops(keys: list[int]) -> tuple[list[int], list[int]]:
    """Sheet rows the global-last rule deletes, and the adjacency-only mistake, bottom up."""
    global_drop = [i + 2 for i, key in enumerate(keys) if key in keys[i + 1 :]][::-1]
    adjacent_drop = [i + 2 for i in range(len(keys) - 1) if keys[i] == keys[i + 1]][::-1]
    return global_drop, adjacent_drop


def _edit(call_id: str, drops: list[int], **options: object) -> JsonObject:
    """One workbook edit deleting the given Data rows."""
    deletions: list[JsonObject] = [
        {"sheet": "Missing" if options.get("missing") else "Data", "start": row, "count": 1}
        for row in drops
    ]
    if options.get("extra"):
        deletions.append({"sheet": "Protected", "start": 2, "count": 1})
    return {
        "name": "edit_workbook",
        "call_id": call_id,
        "arguments": {
            "source": options.get("source", "input.xlsx"),
            "destination": "output.xlsx",
            "deletions": deletions,
        },
    }


def workbook_world(keys: list[int]) -> JsonObject:
    """A world proposal with its own verifier and control scripts for these keys."""
    rows = [[i + 1, key] for i, key in enumerate(keys)]
    keep = [row for i, row in enumerate(rows) if row[1] not in keys[i + 1 :]]
    workbook = Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.title = "Data"
    for row in (["id", "key"], *keep):
        sheet.append(row)
    workbook.create_sheet("Protected").append(["do not change"])
    buffer = io.BytesIO()
    workbook.save(buffer)
    sheets = workbook_projection(workbook)["sheets"]
    assert isinstance(sheets, list) and isinstance(sheets[0], dict)
    global_drop, adjacent_drop = _drops(keys)
    programs = (
        ("solution", (_edit("global", global_drop),)),
        (
            "alternative",
            tuple(
                _edit(f"split-{i}", [row], source="input.xlsx" if i == 0 else "output.xlsx")
                for i, row in enumerate(global_drop)
            ),
        ),
        ("recovery", (_edit("bad", [2], missing=True), _edit("fixed", global_drop))),
        ("targeted_error", (_edit("adjacent", adjacent_drop),)),
        ("unwanted_effect", (_edit("extra", global_drop, extra=True),)),
    )
    return {
        "construction": {
            "name": f"harder-{len(keys)}",
            "assumptions": ["Authored harder world; no production frequency claim."],
            "tables": {
                "Data": {
                    "rows": len(keys) + 1,
                    "columns": {
                        "id": {"kind": "values", "values": ["id", *range(1, len(keys) + 1)]},
                        "key": {"kind": "values", "values": ["key", *keys]},
                    },
                },
                "Protected": {
                    "rows": 2,
                    "columns": {"sentinel": {"kind": "values", "values": ["do not change", 42]}},
                },
            },
            "workbooks": {"original.xlsx": ["Data", "Protected"]},
        },
        "rationale": "More rows and scattered repeats around adjacent duplicates.",
        "verifier": {
            "description": "Global last occurrence, retaining input and Protected.",
            "outcome_schema": {},
            "required_paths": ["/final_state/views/main/sha256"],
            "rules": [
                {
                    "path": "/final_state/views/main/sha256",
                    "operator": "equals",
                    "value": sha256_json([sheets[0]]),
                },
                {
                    "path": "/final_state/views/main/sheets/0/rows",
                    "operator": "equals",
                    "value": sheets[0]["rows"],
                },
            ],
            "allowed_tools": ["edit_workbook", "describe_workbook", "inspect_workbook"],
            "preserved_paths": ["/files/input.xlsx", "/views/protected/sha256"],
        },
        "controls": [
            {"role": role, "actions": list(actions), "finish_tick": len(actions)}
            for role, actions in programs
        ],
    }


def actor_turn(model: str, drops: list[int]) -> tuple[JsonHttpResponse, JsonHttpResponse]:
    """A scripted actor attempt: one workbook edit, then a final reply."""
    call = _edit("edit", drops)
    arguments = call["arguments"]
    assert isinstance(arguments, dict)
    edit = tool_reply("edit_workbook", arguments, model=model)
    done = JsonHttpResponse(
        status_code=200,
        body={
            "id": "done",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "done"},
                }
            ],
        },
    )
    return edit, done


def panel_actors(project: ProjectStore) -> tuple[PanelActor, PanelActor]:
    """Two strong actors from the fixture catalog."""
    catalog = RuntimeModelCatalog(load_model_catalog(project.model_catalog_path), environment={})
    return (
        PanelActor(alias="world", model=catalog.snapshot("world")[0], label="strong a"),
        PanelActor(alias="judge", model=catalog.snapshot("judge")[0], label="strong b"),
    )


def author_settings(spec: WorldProposalSpec) -> AuthorSettings:
    """The seed author model, registered again for harder levels."""
    return AuthorSettings(
        model_alias=spec.model_alias,
        model=spec.model,
        catalog_sha256=spec.catalog_sha256,
        deadline=spec.deadline,
    )


def workbook_hardening(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, levels: int = 2
) -> tuple[ProjectStore, TaskHardeningSpec, WorldProposalSpec]:
    """A saved seed experiment whose second world qualified, and a registered hardening."""
    project, world, _ = world_fixture(tmp_path, monkeypatch)
    propose_worlds(project, world, code_revision="test")
    seed = artifact_input(project.artifacts.read(world.identity("test")).manifest)
    spec = TaskHardeningSpec(
        seed=seed,
        candidate=1,
        author=author_settings(world),
        actors=panel_actors(project),
        repeats=1,
        saturation=SaturationRule(),
        validity=validity_settings(project, world.base, rule=ValidityRule()),
        maximum_levels=levels,
    )
    return project, spec, world


def test_saturated_workbook_world_is_hardened_until_the_panel_separates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Level 0 saturates; level 1 keeps a separating world and rejects a leaky one unpanelled."""
    project, spec, world = workbook_hardening(tmp_path, monkeypatch)
    harder = ([1, 1, 2, 1, 3, 2, 3], [2, 2, 1, 3, 1, 2, 3, 3])
    seed_drop, _ = _drops([1, 2, 1, 1])
    first_drop, first_mistake = _drops(harder[0])
    author = ScriptedJsonTransport(
        (
            judge(),
            closed(),
            tool_reply("propose_worlds", {"candidates": [workbook_world(keys) for keys in harder]}),
            judge(),
            closed(),
            judge("derivable", "Keep the global last row for each key"),
        )
    )
    actor = ScriptedJsonTransport(
        (
            *actor_turn("world-v1", seed_drop),
            *actor_turn("judge-v1", seed_drop),
            *actor_turn("world-v1", first_drop),
            *actor_turn("judge-v1", first_mistake),
        )
    )
    monkeypatch.setattr(AUTHOR, lambda: author)
    monkeypatch.setattr(ACTOR, lambda: actor)
    report = run_task_hardening(project, spec, code_revision="test")
    assert report.status == "kept", report.model_dump_json()
    assert [level.level for level in report.levels] == [0, 1]
    (seed_row,) = report.levels[0].candidates
    assert seed_row.disposition == "saturated" and seed_row.difficulty is not None
    assert seed_row.difficulty.pass_rate == 1.0 and seed_row.difficulty.band == "above"
    kept, leaky = report.levels[1].candidates
    assert report.levels[1].source == 1
    assert kept.disposition == "kept" and kept.difficulty is not None
    assert kept.difficulty.pass_rate == 0.5 and kept.difficulty.band == "within"
    assert kept.difficulty.interval is not None and kept.difficulty.interval[0] < 0.5
    assert leaky.disposition == "invalid" and leaky.panel is None
    assert leaky.rejections == (
        "model_leak: judge: derivable from public context: Keep the global last row for each key",
    )
    assert report.counts["families"] == 1 and report.counts["variants"] == 3
    hardening = json.dumps(author.requests[2].payload)
    assert "Hardening:" in hardening and "previous_world" in hardening
    assert "harder-" not in hardening and '\\"successes\\": 1' in hardening
    base = load_executable_task(project, world.base)
    lineage = load_continuation_checkpoint(
        project.artifacts, base.job.study.checkpoint
    ).task.lineage_group_id
    assert report.lineage == lineage
    for row in (seed_row, kept, leaky):
        assert row.variant is not None
        assert load_task_variant(project, row.variant).lineage == lineage
    requests = len(author.requests), len(actor.requests)
    assert requests == (6, 8)
    project.model_catalog_path.unlink()
    world.template.capture.runtime_file.unlink()
    assert run_task_hardening(project, spec, code_revision="test") == report
    assert (len(author.requests), len(actor.requests)) == requests


def test_level_limit_stops_a_world_that_stays_saturated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every level is retained; the registered limit stops without another author draw."""
    project, spec, _ = workbook_hardening(tmp_path, monkeypatch, levels=1)
    with pytest.raises(ValueError, match="qualified candidate"):
        run_task_hardening(project, spec.model_copy(update={"candidate": 0}), code_revision="test")
    harder = ([1, 1, 2, 1, 3, 2, 3], [2, 2, 1, 3, 1, 2, 3, 3])
    seed_drop, _ = _drops([1, 2, 1, 1])
    author = ScriptedJsonTransport(
        (
            judge(),
            closed(),
            tool_reply("propose_worlds", {"candidates": [workbook_world(keys) for keys in harder]}),
            *(judge(), closed()) * 2,
        )
    )
    turns = [actor_turn(model, seed_drop) for model in ("world-v1", "judge-v1")]
    for keys in harder:
        turns.extend(actor_turn(model, _drops(keys)[0]) for model in ("world-v1", "judge-v1"))
    actor = ScriptedJsonTransport(tuple(reply for turn in turns for reply in turn))
    monkeypatch.setattr(AUTHOR, lambda: author)
    monkeypatch.setattr(ACTOR, lambda: actor)
    write = project.artifacts.write_json

    def interrupt(
        *,
        artifact_id: str,
        artifact_type: str,
        envelope: ArtifactEnvelope,
        files: Mapping[str, BaseModel | JsonValue],
    ) -> ArtifactManifest:
        """Crash after every level is saved, before the report is published."""
        if artifact_type == "task-hardening-report":
            raise KeyboardInterrupt("report interrupted")
        return write(
            artifact_id=artifact_id, artifact_type=artifact_type, envelope=envelope, files=files
        )

    with monkeypatch.context() as patch:
        patch.setattr(project.artifacts, "write_json", interrupt)
        with pytest.raises(KeyboardInterrupt):
            run_task_hardening(project, spec, code_revision="test")
    assert len(author.requests) == 7 and len(actor.requests) == 12
    report = run_task_hardening(project, spec, code_revision="test")
    assert report.status == "saturated_at_limit", report.model_dump_json()
    assert [row.disposition for level in report.levels for row in level.candidates] == [
        "saturated"
    ] * 3
    assert len(author.requests) == 7 and len(actor.requests) == 12


RECORD_RUNTIME = '''"""Test-only record world seeded with orders from a bound construction."""
from functools import partial

from exp.optimize.evaluation.executable.construction import ConstructionBinding
from exp.optimize.evaluation.executable.loader import ContinuationRuntime
from exp.runtime.environments.record_world import RecordTool, RecordWorldFactory
from exp.runtime.environments.worlds import ExecutableWorldRuntime
from exp.simulation.evaluation.record_execution import record_state
from exp.simulation.evaluation.record_execution_test import access, change, fixture
from exp.simulation.world_construction import load_construction, materialize_tables


def _schema(orders):
    """Create the orders and outbox tables with the bound initial orders."""
    values = ", ".join("(" + ", ".join(repr(str(value)) for value in row) + ")" for row in orders)
    return (
        "CREATE TABLE orders(id TEXT PRIMARY KEY, address TEXT, status TEXT);"
        f"INSERT INTO orders VALUES {values}; CREATE TABLE outbox(message TEXT);"
    )


def continuation_runtime(project, configuration):
    """Bind the actor tool to an isolated record world seeded from configuration."""
    replay, _ = fixture()
    factory = RecordWorldFactory(
        service={"name": "constructed-orders"},
        schema=_schema(configuration["orders"]),
        tools={"change": RecordTool(access, change)},
    )
    world = ExecutableWorldRuntime(replay.world.manifest, factory, seed=7)
    return ContinuationRuntime(world, partial(record_state, world), replay.probe_sha256)


def bind_construction(project, configuration, assets):
    """Replace only the initial orders with the constructed table."""
    table = materialize_tables(load_construction(project.artifacts, assets))["orders"]
    order = [table.columns.index(name) for name in ("id", "address", "status")]
    replay, _ = fixture()
    return ConstructionBinding(
        configuration={**configuration, "orders": [[row[i] for i in order] for row in table.rows]},
        probe_sha256=replay.probe_sha256,
    )
'''


def record_world(rows: list[list[str]], name: str) -> JsonObject:
    """A four-order world proposal with its verifier and control scripts."""
    final = [["a", "new", rows[0][2]], *rows[1:]]
    programs: tuple[tuple[str, tuple[JsonObject, ...]], ...] = (
        ("solution", ({},)),
        ("alternative", ({"address": "new"}, {"noop": True})),
        ("recovery", ({"id": "absent"}, {})),
        ("targeted_error", ({"noop": True},)),
        ("unwanted_effect", ({"email": True},)),
    )
    return {
        "construction": {
            "name": name,
            "assumptions": ["Authored orders; no production frequency claim."],
            "tables": {
                "orders": {
                    "rows": len(rows),
                    "columns": {
                        field: {"kind": "values", "values": [row[i] for row in rows]}
                        for i, field in enumerate(("id", "address", "status"))
                    },
                }
            },
        },
        "rationale": "Decoy orders share the old address and status.",
        "verifier": {
            "description": "Order a has the new address; other orders and the outbox unchanged.",
            "outcome_schema": {"type": "object"},
            "required_paths": ["/final_state/tables/orders/rows"],
            "rules": [
                {"path": "/final_state/tables/orders/rows", "operator": "equals", "value": final}
            ],
            "allowed_tools": ["change"],
            "preserved_paths": [
                "/tables/orders/rows/1",
                "/tables/orders/rows/2",
                "/tables/orders/rows/3",
                "/tables/outbox",
            ],
        },
        "controls": [
            {
                "role": role,
                "actions": [
                    {"name": "change", "call_id": f"step-{i}", "arguments": arguments}
                    for i, arguments in enumerate(actions)
                ],
                "finish_tick": 1 + len(actions),
            }
            for role, actions in programs
        ],
    }


def record_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ProjectStore, WorldProposalSpec]:
    """A record-world author experiment bound through a trusted construction hook."""
    from datetime import UTC, datetime

    from exp.common.core.artifacts import sha256_bytes
    from exp.optimize.evaluation.analysis.research import (
        FailureHypothesis,
        ResearchEvidence,
        ResearchFailure,
    )
    from exp.optimize.evaluation.audit.panel_test import write_catalog
    from exp.optimize.evaluation.continuations.capture import CaptureDefinition
    from exp.optimize.evaluation.continuations.job_test import job_fixture
    from exp.optimize.evaluation.executable.scenarios import (
        RecordScenarioSpec,
        prepare_record_scenario,
    )
    from exp.optimize.evaluation.executable.task import ExecutableTask, persist_executable_task
    from exp.optimize.evaluation.factory.scenario_search import ScenarioResearchTarget
    from exp.simulation.capability_briefs import persist_capability_brief
    from exp.simulation.capability_briefs_test import brief
    from exp.simulation.evaluation.admission import EvaluationSpecification, persist_specification
    from exp.simulation.evaluation.record_execution_test import acceptance, fixture, source
    from exp.simulation.evaluation.tasks import OutcomeRule

    project = ProjectStore(tmp_path, "records")
    _, template, _ = job_fixture(tmp_path / "template", monkeypatch)
    write_catalog(tmp_path)
    runtime = tmp_path / "record_runtime.py"
    runtime.write_text(RECORD_RUNTIME)
    replay, task = fixture()
    capture = CaptureDefinition(
        task=task,
        sources=(source(project),),
        runtime_file=runtime,
        runtime_sha256=sha256_bytes(runtime.read_bytes()),
        configuration={"orders": [["a", "old", "pending"], ["b", "other", "pending"]]},
        probe_sha256=replay.probe_sha256,
        maximum_model_calls=5,
        system_prompt="Use only the actor's public tools.",
    )
    scenario = RecordScenarioSpec(capture=capture, finish_tick=0)
    reached = prepare_record_scenario(project, scenario, code_revision="test")
    job = template.model_copy(
        update={
            "runtime_file": runtime,
            "runtime_sha256": capture.runtime_sha256,
            "configuration": capture.configuration,
            "maximum_model_calls": capture.maximum_model_calls,
            "system_prompt": capture.system_prompt,
            "study": template.study.model_copy(
                update={
                    "checkpoint": reached.checkpoint,
                    "probe_sha256": capture.probe_sha256,
                    "verifier": acceptance(),
                    "training_specification": None,
                }
            ),
        }
    )
    now = datetime.now(UTC)
    base = persist_executable_task(
        project,
        ExecutableTask(intent=task.instruction, job=job),
        created_at=now,
        code_revision="test",
    )
    capability = persist_capability_brief(
        project.artifacts,
        brief().model_copy(update={"outcome": task.instruction, "required_tools": ("change",)}),
        created_at=now,
        code_revision="test",
    )
    independent = persist_specification(
        project.artifacts,
        EvaluationSpecification(
            description="Order a gets the new address; three other orders and email unchanged.",
            predicate=acceptance().model_copy(
                update={
                    "rules": (
                        OutcomeRule(
                            path="/final_state/tables/orders/rows/0/1",
                            operator="equals",
                            value="new",
                        ),
                    ),
                    "preserved_paths": (
                        "/tables/orders/rows/1",
                        "/tables/orders/rows/2",
                        "/tables/orders/rows/3",
                        "/tables/outbox",
                    ),
                }
            ),
            subject="native_observation",
        ),
        sources=(source(project),),
        created_at=now,
        code_revision="test",
    )
    target = ScenarioResearchTarget(
        failure=ResearchFailure(
            symptom="Decoy orders may hide whether the requested order is changed.",
            observations=(
                ResearchEvidence(source=source(project), file="intent.json", pointer="/scope"),
            ),
            hypotheses=(
                FailureHypothesis(
                    id="decoys",
                    destination="actor",
                    explanation="Orders sharing the old address invite the wrong change.",
                    falsifier="Every actor changes only order a.",
                ),
            ),
        ),
        hypothesis="decoys",
    )
    spec = WorldProposalSpec(
        base=base,
        brief=capability,
        target=target,
        template=scenario.model_copy(
            update={
                "capture": capture.model_copy(
                    update={"sources": (*capture.sources, reached.checkpoint)}
                )
            }
        ),
        acceptance=independent,
        asset_requirements="One orders table with id, address and status; order a is first.",
        model_alias=job.actor_alias,
        model=job.study.actor_model,
        catalog_sha256=job.catalog_sha256,
        deadline=job.deadline,
    )
    return project, spec


def record_turn(model: str, arguments: JsonObject) -> tuple[JsonHttpResponse, JsonHttpResponse]:
    """A scripted record-world attempt: one change call, then a final reply."""
    done = JsonHttpResponse(
        status_code=200,
        body={
            "id": "done",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "done"},
                }
            ],
        },
    )
    return tool_reply("change", arguments, model=model), done


def record_hardening(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ProjectStore, TaskHardeningSpec, ScriptedJsonTransport, ScriptedJsonTransport]:
    """A saved record-world seed whose panel saturates, and one harder level that separates."""
    project, world = record_seed(tmp_path, monkeypatch)
    solved = [
        ["a", "new", "pending"],
        ["b", "old", "pending"],
        ["c", "old", "held"],
        ["d", "x", "y"],
    ]
    seed_rows = [
        ["a", "old", "pending"],
        ["b", "other", "pending"],
        ["c", "far", "held"],
        ["d", "x", "y"],
    ]
    harder = [
        ["a", "old", "pending"],
        ["b", "old", "pending"],
        ["c", "old", "pending"],
        ["e", "old", "held"],
    ]
    author = ScriptedJsonTransport(
        (
            tool_reply(
                "propose_worlds",
                {
                    "candidates": [
                        record_world(solved, "already solved"),
                        record_world(seed_rows, "seed orders"),
                    ]
                },
            ),
            judge(),
            closed(),
            tool_reply(
                "propose_worlds",
                {
                    "candidates": [
                        record_world(harder, "decoy orders"),
                        record_world(solved, "already solved"),
                    ]
                },
            ),
            judge(),
            closed(),
        )
    )
    monkeypatch.setattr(AUTHOR, lambda: author)
    seed_report = propose_worlds(project, world, code_revision="test")
    assert seed_report.search is not None
    assert seed_report.search.cells[0].variant is None
    assert seed_report.search.cells[1].variant is not None, seed_report.search.cells[1].refusals
    actor = ScriptedJsonTransport(
        (
            *record_turn("world-v1", {}),
            *record_turn("judge-v1", {}),
            *record_turn("world-v1", {}),
            *record_turn("judge-v1", {"noop": True}),
        )
    )
    monkeypatch.setattr(ACTOR, lambda: actor)
    spec = TaskHardeningSpec(
        seed=artifact_input(project.artifacts.read(world.identity("test")).manifest),
        candidate=1,
        author=author_settings(world),
        actors=panel_actors(project),
        repeats=1,
        validity=validity_settings(
            project, world.base, rule=ValidityRule(specified_values=("new",))
        ),
    )
    return project, spec, author, actor


def test_record_world_hardens_through_its_own_construction_hook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same loop runs on a record world; an unqualified harder world stays retained."""
    project, spec, author, actor = record_hardening(tmp_path, monkeypatch)
    report = run_task_hardening(project, spec, code_revision="test")
    assert report.status == "kept", report.model_dump_json()
    kept, unqualified = report.levels[1].candidates
    assert kept.disposition == "kept" and kept.difficulty is not None
    assert kept.difficulty.pass_rate == 0.5
    assert unqualified.disposition == "unqualified" and unqualified.validity is None
    assert any("targeted_error" in reason for reason in unqualified.refusals)
    assert (len(author.requests), len(actor.requests)) == (6, 8)
    assert run_task_hardening(project, spec, code_revision="test") == report
    assert (len(author.requests), len(actor.requests)) == (6, 8)
