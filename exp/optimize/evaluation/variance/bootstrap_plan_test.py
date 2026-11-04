"""No-start planning, exact consent, family reuse and provider-free replay contracts."""

import json
from collections.abc import Mapping
from pathlib import Path

import pytest
from pydantic import ValidationError

from exp.cli.build.app_test import _EmbeddingClient
from exp.common.core.artifacts import JsonObject, sha256_bytes
from exp.common.project import ProjectStore
from exp.optimize.evaluation.executable import grounded
from exp.optimize.evaluation.executable.costs import grounded_allowance_costs, runtime_costs
from exp.optimize.evaluation.executable.grounded import GroundedRuntimeDefinition
from exp.optimize.evaluation.variance import bootstrap_plan as module
from exp.optimize.evaluation.variance.bootstrap import registered_job
from exp.optimize.evaluation.variance.bootstrap_contracts import MAX_REVEAL_PROBES, StartDraft
from exp.optimize.evaluation.variance.bootstrap_plan import (
    BootstrapApproval,
    BootstrapDependencies,
    BootstrapPreparationReport,
    BootstrapPreparationSpec,
    plan_bootstrap_preparation,
    run_bootstrap_preparation,
)
from exp.optimize.evaluation.variance.bootstrap_selection_test import (
    preparation_fixture,
    saved_build,
)
from exp.optimize.evaluation.variance.bootstrap_test import FINALS, MARKERS, _draft
from exp.optimize.evaluation.variance.build_test import FakeDecisions, _tagger
from exp.optimize.evaluation.variance.evidence_tools_test import _reply
from exp.optimize.evaluation.variance.family_simulators import load_family_simulator
from exp.optimize.evaluation.variance.tests.grounded_world_test import (
    TicketPredictor,
    grounded_fixture,
)
from exp.runtime.models.providers.transport import (
    JsonHttpResponse,
    RecordedRequest,
    ScriptedJsonTransport,
)


def forbidden(*args: object, **kwargs: object) -> None:
    """Fail if an offline plan or replay tries constructing a provider or executing a stage."""
    raise AssertionError("unexpected paid execution")


def test_provider_free_plan_counts_all_role_and_restore_multiplicities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A completed source build needs no starting job, clients or credentials to price work."""
    project, spec, _ = preparation_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr("exp.optimize.evaluation.factory.authoring.HttpxJsonTransport", forbidden)
    monkeypatch.setattr("exp.runtime.models.registry.RuntimeModelCatalog.resolve", forbidden)
    plan = plan_bootstrap_preparation(project, spec, code_revision="test")
    assert not plan.build_saved and plan.task_bound == 2 and plan.family_bound == 2
    assert not plan.completed_tasks and not plan.remaining_tasks and not plan.complete
    components = {row.name: row for row in plan.costs}
    settings = spec.bootstrap
    assert (
        components["bootstrap author: world"].attempts
        == 2 * 2 * settings.author_profile.budget.model_calls
    )
    assert (
        components["bootstrap reviewer: world"].attempts
        == 2 * 2 * len(settings.review_variants) * settings.review_profile.budget.model_calls
    )
    assert (
        components["bootstrap researcher: world"].attempts
        == 2 * 2 * settings.research_profile.budget.model_calls
    )
    assert components["bootstrap validation: world"].attempts == 2 * (MAX_REVEAL_PROBES + 2) * 2
    assert plan.unpriced_attempts > 0
    with pytest.raises(ValueError, match="explicit approval"):
        run_bootstrap_preparation(project, spec, code_revision="test")
    wrong = BootstrapApproval(
        plan_sha256="0" * 64, ceiling_usd=plan.known_usd, unpriced_attempts=plan.unpriced_attempts
    )
    with pytest.raises(ValueError, match="exact current plan"):
        run_bootstrap_preparation(project, spec, code_revision="test", approval=wrong)
    insufficient = wrong.model_copy(update={"plan_sha256": plan.digest, "unpriced_attempts": 0})
    with pytest.raises(ValueError, match="priced and unpriced"):
        run_bootstrap_preparation(project, spec, code_revision="test", approval=insufficient)


def test_saved_build_uses_exact_pending_tasks_and_rejects_stale_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After real family review, remaining costs shrink and approval names the new inventory."""
    project, spec, bound = preparation_fixture(tmp_path, monkeypatch)
    before = plan_bootstrap_preparation(project, spec, code_revision="test")
    saved_build(project, spec, bound)
    after = plan_bootstrap_preparation(project, spec, code_revision="test")
    assert after.build_saved and len(after.remaining_tasks) == after.task_bound == 2
    assert after.family_bound == 1 and after.unpriced_attempts < before.unpriced_attempts
    approval = BootstrapApproval(
        plan_sha256=before.digest,
        ceiling_usd=before.known_usd,
        unpriced_attempts=before.unpriced_attempts,
    )
    monkeypatch.setattr(module, "family_reviewer", forbidden)
    with pytest.raises(ValueError, match="exact current plan"):
        run_bootstrap_preparation(project, spec, code_revision="test", approval=approval)


def test_failed_preparation_preserves_pending_tasks_without_fabricating_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unavailable roles remain retryable failures and never produce fabricated mining jobs."""
    project, spec, bound = preparation_fixture(tmp_path, monkeypatch)

    def no_provider() -> None:
        """Refuse provider construction while exercising the bootstrap failure contract."""
        raise RuntimeError("offline author unavailable")

    monkeypatch.setattr("exp.optimize.evaluation.factory.authoring.HttpxJsonTransport", no_provider)
    plan = plan_bootstrap_preparation(project, spec, code_revision="test")
    approval = BootstrapApproval(
        plan_sha256=plan.digest,
        ceiling_usd=plan.known_usd,
        unpriced_attempts=plan.unpriced_attempts,
    )
    result = run_bootstrap_preparation(
        project,
        spec,
        code_revision="test",
        approval=approval,
        dependencies=BootstrapDependencies(
            reviewer=bound, tagger=_tagger(project, FakeDecisions())
        ),
    )
    assert not result.bindings
    assert len(result.failed_tasks) == 2 and not result.refused_tasks
    assert not project.artifacts.exists(result.identity + "-result")
    remaining = plan_bootstrap_preparation(project, spec, code_revision="test")
    assert set(remaining.remaining_tasks) == set(result.failed_tasks)
    assert not remaining.complete


def test_unbounded_rules_custom_runtime_and_excess_reveals_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every preparatory simulator lifetime is counted before any executable runtime loads."""
    project, spec, _ = preparation_fixture(tmp_path, monkeypatch)
    assert spec.bootstrap.rules is not None
    settings = spec.bootstrap.model_copy(
        update={"rules": spec.bootstrap.rules.model_copy(update={"profile": "agentic_research"})}
    )
    with pytest.raises(ValidationError, match="finite merged-rule count"):
        BootstrapPreparationSpec.model_validate({**spec.model_dump(), "bootstrap": settings})
    source = tmp_path / "custom.py"
    source.write_text("raise AssertionError('custom runtime must not execute')")
    runtime = spec.bootstrap.runtime.model_copy(
        update={"runtime_file": source, "runtime_sha256": sha256_bytes(source.read_bytes())}
    )
    custom = spec.model_copy(
        update={"bootstrap": spec.bootstrap.model_copy(update={"runtime": runtime})}
    )
    with pytest.raises(ValueError, match="maintained grounded.py"):
        plan_bootstrap_preparation(project, custom, code_revision="test")
    draft = _draft("auth", {"findings": []})
    draft["reveal_probes"] = [{"name": "check_sla", "arguments": {}}] * (MAX_REVEAL_PROBES + 1)
    with pytest.raises(ValidationError, match="at most 16"):
        StartDraft.model_validate(draft)


def test_pre_task_grounded_allowance_matches_real_runtime_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pricing with no task uses the same owner and counts as an actual grounded definition."""
    fixture = grounded_fixture(tmp_path, monkeypatch)
    runtime = fixture.definition
    expected = runtime_costs(
        fixture.project,
        sha256_bytes(Path(grounded.__file__).read_bytes()),
        runtime.model_dump(mode="json"),
    )
    assert (
        grounded_allowance_costs(
            fixture.project,
            runtime.world_model,
            attempts=runtime.maximum_physical_attempts,
            output_tokens=runtime.maximum_output_tokens,
        )
        == expected
    )


class PreparationScript(ScriptedJsonTransport):
    """Answer real extraction and bootstrap role requests from their delivered evidence."""

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Supply bounded role phases, keeping every cited trace in the actual inventory."""
        del timeout_seconds
        self.requests.append(RecordedRequest(url, dict(headers), payload))
        messages = payload["messages"]
        assert isinstance(messages, list)
        opening = json.loads(str(messages[1]["content"]))
        system = str(messages[0]["content"])
        if system.startswith("You learn the rules"):
            result = _reply(
                [
                    (
                        "propose_rules",
                        {
                            "trace_id": opening["trajectory"]["trace_id"],
                            "rules": [],
                            "questions": [],
                            "limitations": ["No tool behavior recorded."],
                        },
                    )
                ]
            )
        else:
            role = next(name for marker, name in MARKERS.items() if system.startswith(marker))
            body: JsonObject
            if role == "start_author":
                body = _draft("auth", opening)
                facts = body["facts"]
                assert isinstance(facts, list) and isinstance(facts[0], dict)
                facts[0]["references"] = [opening["inventory"][0]]
            elif role == "reveal_checker":
                body = {"consistent": True, "summary": "Observed tier matches the authored fact."}
            else:
                body = {
                    "verdict": "accept",
                    "summary": "Coherent authored ticket, not an observation.",
                }
            reading: tuple[str, JsonObject] = ("read_findings", {})
            if "family" in opening:
                reading = ("read_trace", {"trace_id": opening["family"]["member_traces"][0]})
            script: list[tuple[str, JsonObject]] = [
                ("record_plan", {"hypotheses": ["authored ticket"], "planned_reads": ["traces"]}),
                reading,
                ("submit_draft", body),
                ("record_self_check", {"issues": [], "revise": False}),
                (FINALS[role], body),
            ]
            turns = sum(
                1 for row in messages if isinstance(row, dict) and row.get("role") == "assistant"
            )
            result = _reply([script[min(turns, 4)]])
        return JsonHttpResponse(result.status_code, {**result.body, "model": payload["model"]})


class PreparationPredictor(TicketPredictor):
    """Serve real semantic retrieval and grounded ticket predictions without network access."""

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Embed through the same deterministic build client or return a ticket transition."""
        if "input" in payload:
            inputs = payload["input"]
            assert isinstance(inputs, list) and all(isinstance(row, str) for row in inputs)
            vectors = _EmbeddingClient().embed(tuple(str(row) for row in inputs))
            return JsonHttpResponse(
                200,
                {
                    "object": "list",
                    "model": payload["model"],
                    "data": [
                        {"object": "embedding", "index": index, "embedding": list(vector.values)}
                        for index, vector in enumerate(vectors)
                    ],
                    "usage": {"prompt_tokens": 10, "total_tokens": 10},
                },
            )
        response = super().post(
            url, headers=headers, payload=payload, timeout_seconds=timeout_seconds
        )
        return JsonHttpResponse(response.status_code, {**response.body, "model": payload["model"]})


def successful_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ProjectStore, BootstrapPreparationSpec, BootstrapPreparationReport, PreparationScript]:
    """Run real build, family review, bootstrap and capture owners with scripted wires."""
    project, spec, bound = preparation_fixture(tmp_path, monkeypatch)
    spec = spec.model_copy(
        update={
            "bootstrap": spec.bootstrap.model_copy(
                update={
                    "runtime": spec.bootstrap.runtime.model_copy(
                        update={"maximum_physical_attempts": 20}
                    )
                }
            )
        }
    )
    script = PreparationScript()
    predictor = PreparationPredictor()
    monkeypatch.setattr(
        "exp.optimize.evaluation.factory.authoring.HttpxJsonTransport", lambda: script
    )
    monkeypatch.setattr(grounded, "HttpxJsonTransport", lambda: predictor)
    monkeypatch.setattr(
        "exp.runtime.models.providers.transport.HttpxJsonTransport", lambda: predictor
    )
    plan = plan_bootstrap_preparation(project, spec, code_revision="test")
    result = run_bootstrap_preparation(
        project,
        spec,
        code_revision="test",
        approval=BootstrapApproval(
            plan_sha256=plan.digest,
            ceiling_usd=plan.known_usd,
            unpriced_attempts=plan.unpriced_attempts,
        ),
        dependencies=BootstrapDependencies(
            reviewer=bound, tagger=_tagger(project, FakeDecisions())
        ),
    )
    return project, spec, result, script


def test_successful_preparation_registers_exact_family_tasks_and_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A completed build creates usable starts and frozen family rules without an initial job."""
    project, spec, result, script = successful_preparation(tmp_path, monkeypatch)
    assert len(result.bindings) == 2, result.model_dump_json()
    assert not result.failed_tasks and not result.refused_tasks
    assert len(result.simulators) == 1 and script.requests
    shared = load_family_simulator(project.artifacts, result.simulators[0])
    assert shared.rules_report is not None and shared.rule_book is not None
    for binding in result.bindings:
        job = registered_job(project, binding)
        runtime = GroundedRuntimeDefinition.model_validate(job.configuration)
        assert runtime.world_model == shared.world_model
        assert runtime.rule_book == shared.rule_book
        assert runtime.tag_retrieval == shared.tag_retrieval
    next_task = result.bindings[0].task_id
    reuse = spec.model_copy(
        update={
            "bootstrap": spec.bootstrap.model_copy(update={"rules": None}),
            "simulators": result.simulators,
            "task_ids": (next_task,),
        }
    )
    reuse_plan = plan_bootstrap_preparation(project, reuse, code_revision="test")
    assert reuse_plan.family_bound == 0 and reuse_plan.remaining_tasks == (next_task,)
    extractions = sum("You learn the rules" in json.dumps(row.payload) for row in script.requests)
    another = run_bootstrap_preparation(
        project,
        reuse,
        code_revision="test",
        approval=BootstrapApproval(
            plan_sha256=reuse_plan.digest,
            ceiling_usd=reuse_plan.known_usd,
            unpriced_attempts=reuse_plan.unpriced_attempts,
        ),
    )
    assert len(another.bindings) == 1 and not another.failed_tasks
    assert (
        sum("You learn the rules" in json.dumps(row.payload) for row in script.requests)
        == extractions
    )
    assert (
        GroundedRuntimeDefinition.model_validate(
            registered_job(project, another.bindings[0]).configuration
        ).rule_book
        == shared.rule_book
    )
    monkeypatch.setattr(module, "family_reviewer", forbidden)
    monkeypatch.setattr(module, "run_variance_build", forbidden)
    monkeypatch.setattr("exp.optimize.evaluation.factory.authoring.HttpxJsonTransport", forbidden)
    monkeypatch.setattr(grounded, "HttpxJsonTransport", forbidden)
    assert run_bootstrap_preparation(project, spec, code_revision="test") == result
    saved = plan_bootstrap_preparation(project, spec, code_revision="test")
    assert saved.complete and len(saved.completed_tasks) == 2 and not saved.costs


def test_resumed_approval_includes_prior_unknown_liability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Consent for remaining work retains previous unpriced requests in the same operation cap."""
    project, spec, _ = preparation_fixture(tmp_path, monkeypatch)
    first = plan_bootstrap_preparation(project, spec, code_revision="test")
    ceiling = module.operation_spend_ceiling(project, first.identity)
    ceiling.authorize("preparation", ceiling_usd=0, unpriced_attempts=first.unpriced_attempts)
    reservation = ceiling.reserve(
        alias="unpriced-fixture", url="https://fixture.invalid/v1", payload={}
    )
    ceiling.mark_dispatched(reservation, "scripted-accounting-only")
    ceiling.retain_unknown(reservation)
    resumed = plan_bootstrap_preparation(project, spec, code_revision="test")
    assert resumed.accounted_unpriced_attempts == 1
    assert resumed.unpriced_attempts == first.unpriced_attempts + 1
    assert resumed.digest != first.digest
