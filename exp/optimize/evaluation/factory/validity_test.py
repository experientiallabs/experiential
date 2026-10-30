"""Validity checks reject stated answers and memorizable tasks on record and workbook worlds."""

import json
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactInput, JsonObject
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.audit.panel_test import write_catalog
from exp.optimize.evaluation.executable.task import load_executable_task
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.factory.leakage import ValidityRule
from exp.optimize.evaluation.factory.scenario_search import (
    ScenarioSearchSpec,
    run_scenario_search,
)
from exp.optimize.evaluation.factory.scenario_search_test import search_fixture
from exp.optimize.evaluation.factory.validity import (
    TaskValiditySpec,
    ValiditySettings,
    run_task_validity,
)
from exp.optimize.evaluation.factory.worlds import propose_worlds
from exp.optimize.evaluation.factory.worlds_test import world_fixture
from exp.runtime.models.providers.transport import JsonHttpResponse, ScriptedJsonTransport

PATCH = "exp.optimize.evaluation.factory.authoring.HttpxJsonTransport"


def tool_reply(name: str, arguments: JsonObject, model: str = "world-v1") -> JsonHttpResponse:
    """One provider-native forced tool call."""
    return JsonHttpResponse(
        status_code=200,
        body={
            "id": "draw-" + name,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": name,
                                "type": "function",
                                "function": {"name": name, "arguments": json.dumps(arguments)},
                            }
                        ],
                    },
                }
            ],
        },
    )


def judge(verdict: str = "not_derivable", quote: str | None = None) -> JsonHttpResponse:
    """A scripted leak-judge verdict."""
    return tool_reply(
        "assess_leakage", {"verdict": verdict, "quote": quote, "rationale": "Checked context."}
    )


def closed(
    answer: str = "I cannot see the records.", values: tuple[str | int | float, ...] = ()
) -> JsonHttpResponse:
    """A scripted closed-book attempt."""
    return tool_reply("submit_answer", {"final_answer": answer, "values": list(values)})


def validity_settings(
    project: ProjectStore, base: ArtifactInput, *, rule: ValidityRule, attempts: int = 1
) -> ValiditySettings:
    """Registered judge and closed-book models drawn from the project's catalog."""
    job = load_executable_task(project, base).job
    author = AuthorSettings(
        model_alias=job.actor_alias,
        model=job.study.actor_model,
        catalog_sha256=job.catalog_sha256,
        deadline=job.deadline,
    )
    return ValiditySettings(
        leak_judge=author, closed_book=author, closed_book_attempts=attempts, rule=rule
    )


def record_validity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    rule: ValidityRule,
    prompt: str | None = None,
    attempts: int = 1,
) -> tuple[ProjectStore, TaskValiditySpec]:
    """Qualify the record-world variant and register validity checks for it."""
    project, search = search_fixture(tmp_path, monkeypatch)
    write_catalog(tmp_path)
    if prompt is not None:
        search = ScenarioSearchSpec.model_validate(
            {**search.model_dump(mode="json"), "prompt": prompt}
        )
    result = run_scenario_search(project, search, code_revision="test")
    assert result.cells[1].variant is not None, result.cells[1].refusals
    reference = artifact_input(project.artifacts.read(search.identity("test") + "-result").manifest)
    settings = validity_settings(project, search.base, rule=rule, attempts=attempts)
    return project, TaskValiditySpec(**settings.model_dump(), search=reference, cell=1)


def script(
    monkeypatch: pytest.MonkeyPatch, *responses: JsonHttpResponse | Exception
) -> ScriptedJsonTransport:
    """Serve the given validity draws in order."""
    transport = ScriptedJsonTransport(responses)
    monkeypatch.setattr(PATCH, lambda: transport)
    return transport


def test_stated_answer_rejects_before_any_model_call_and_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The record task states its new address; undeclared, that value is a leak."""
    project, spec = record_validity(tmp_path, monkeypatch, rule=ValidityRule())
    transport = script(monkeypatch)
    report = run_task_validity(project, spec, code_revision="test")
    assert report.status == "rejected" and report.produced_values == ("new",)
    checks = {check.name: check for check in report.checks}
    assert checks["controls"].status == "pass"
    assert checks["mechanical_leak"].reasons == ("answer value in public context: new",)
    assert checks["model_leak"].status == checks["closed_book"].status == "not_run"
    assert not transport.requests
    project.model_catalog_path.unlink()
    assert run_task_validity(project, spec, code_revision="test") == report


def test_specified_value_then_model_checks_keep_a_valid_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A declared requested value is not a leak; judge and closed-book draws then decide."""
    project, spec = record_validity(
        tmp_path, monkeypatch, rule=ValidityRule(specified_values=("new",)), attempts=2
    )
    transport = script(monkeypatch, judge(), closed(), closed("Change it.", ("old",)))
    report = run_task_validity(project, spec, code_revision="test")
    assert report.status == "valid", report.model_dump_json()
    assert report.produced_values == ()
    assert [check.status for check in report.checks] == ["pass"] * 4
    assert len(transport.requests) == 3
    closed_book = json.dumps(transport.requests[1].payload["tools"])
    assert '"submit_answer"' in closed_book and '"change"' not in closed_book
    project.model_catalog_path.unlink()
    assert run_task_validity(project, spec, code_revision="test") == report
    assert len(transport.requests) == 3


@pytest.mark.parametrize("quoted,status", [(True, "rejected"), (False, "undetermined")])
def test_judge_rejects_only_with_an_exact_public_quote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, quoted: bool, status: str
) -> None:
    """A derivable verdict must cite actor-visible text; an unsupported claim stays unknown."""
    project, spec = record_validity(
        tmp_path, monkeypatch, rule=ValidityRule(specified_values=("new",))
    )
    quote = "Change order a to new" if quoted else "the answer is in the hidden table"
    transport = script(monkeypatch, judge("derivable", quote), closed())
    report = run_task_validity(project, spec, code_revision="test")
    assert report.status == status
    leak = report.checks[2]
    assert leak.name == "model_leak" and len(leak.responses) == 1
    if quoted:
        assert leak.status == "reject" and report.checks[3].status == "not_run"
        assert len(transport.requests) == 1
    else:
        assert leak.status == "unknown" and report.checks[3].status == "pass"


def test_closed_book_recall_rejects_a_memorizable_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the address in the prompt, a closed-book model that names it has memorized it."""
    project, spec = record_validity(
        tmp_path,
        monkeypatch,
        rule=ValidityRule(),
        prompt="Apply the customer's address update to order a; send no email.",
        attempts=2,
    )
    script(monkeypatch, judge(), closed(), closed("Set the address to new.", ("new",)))
    report = run_task_validity(project, spec, code_revision="test")
    assert report.produced_values == ("new",)
    assert report.checks[1].status == "pass"
    book = report.checks[3]
    assert report.status == "rejected" and book.status == "reject"
    assert book.reasons == ("attempt 1: recalled 1 of 1 produced values without the world",)


def test_unknown_dispatch_stays_undetermined_without_redraw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lost closed-book response is retained; a rerun replays instead of buying another."""
    project, spec = record_validity(
        tmp_path, monkeypatch, rule=ValidityRule(specified_values=("new",))
    )
    transport = script(monkeypatch, judge(), RuntimeError("lost response"))
    report = run_task_validity(project, spec, code_revision="test")
    assert report.status == "undetermined"
    assert report.checks[3].status == "unknown" and not report.checks[3].responses
    count = len(transport.requests)
    assert run_task_validity(project, spec, code_revision="test") == report
    assert len(transport.requests) == count


def test_workbook_world_variant_passes_every_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Row deletion adds no values; the closed-book attempt cannot write the output workbook."""
    project, world, _ = world_fixture(tmp_path, monkeypatch)
    proposals = propose_worlds(project, world, code_revision="test")
    assert proposals.search is not None
    search = artifact_input(
        project.artifacts.read(proposals.search.spec.identity("test") + "-result").manifest
    )
    settings = validity_settings(project, world.base, rule=ValidityRule())
    spec = TaskValiditySpec(**settings.model_dump(), search=search, cell=1)
    transport = script(monkeypatch, judge(), closed("Kept the last rows.", ("id", "key")))
    report = run_task_validity(project, spec, code_revision="test")
    assert report.status == "valid", report.model_dump_json()
    assert report.produced_values == ()
    assert len(transport.requests) == 2
    with pytest.raises(ValueError, match="qualified scenario-search variant"):
        run_task_validity(project, spec.model_copy(update={"cell": 0}), code_revision="test")
