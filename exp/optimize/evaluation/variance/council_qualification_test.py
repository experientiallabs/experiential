"""Native owner qualification and original-ledger registration, using scripted wires only."""

import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, JsonObject, sha256_json
from exp.common.models import ModelCapabilities, load_model_catalog, write_model_catalog
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.executable import grounded
from exp.optimize.evaluation.rules.locator_inventory import LocatorInventory
from exp.optimize.evaluation.spending import operation_spend_ceiling
from exp.optimize.evaluation.variance import bootstrap_selection_test as fixtures
from exp.optimize.evaluation.variance.actor_critic import ActorCriticSpec
from exp.optimize.evaluation.variance.actor_critic_test import _spec
from exp.optimize.evaluation.variance.bootstrap_plan import (
    BootstrapDependencies,
    BootstrapPreparationReport,
    run_bootstrap_preparation,
)
from exp.optimize.evaluation.variance.bootstrap_plan_test import (
    PreparationPredictor,
    PreparationScript,
)
from exp.optimize.evaluation.variance.bootstrap_resume_test import _origin, _register
from exp.optimize.evaluation.variance.bootstrap_stage import bootstrapped_registration
from exp.optimize.evaluation.variance.build_records import VarianceBuildReport
from exp.optimize.evaluation.variance.build_test import FakeDecisions, _tagger
from exp.optimize.evaluation.variance.council_qualification import (
    CONSENT_TYPE,
    CouncilConditionalConsent,
    CouncilStudyContract,
    _native_start,
    configured_council_actors,
    council_aliases,
    council_sources,
    qualify_council,
    register_conditional_council,
)
from exp.optimize.evaluation.variance.evidence import ReferenceSource, TraceSearchSpec
from exp.optimize.evaluation.variance.family_simulators import load_family_simulator
from exp.optimize.evaluation.variance.pipeline_spec import MiningSettings
from exp.runtime.models.providers.transport import JsonHttpResponse
from exp.runtime.models.registry import RuntimeModelCatalog


class NativePreparationScript(PreparationScript):
    """Decode the native lossless locator grammar for the legacy fixture responder."""

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Keep every delivered canonical reference while adapting fixture response construction."""
        messages = payload["messages"]
        assert isinstance(messages, list) and isinstance(messages[1], dict)
        opening: JsonObject = json.loads(str(messages[1]["content"]))
        inventory = opening.get("inventory")
        if isinstance(inventory, dict):
            opening["inventory"] = list(LocatorInventory.model_validate(inventory).decode())
            messages = list(messages)
            messages[1] = {**messages[1], "content": json.dumps(opening)}
            payload = {**payload, "messages": messages}
        return super().post(url, headers=headers, payload=payload, timeout_seconds=timeout_seconds)


def _fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ProjectStore, ArtifactInput, ArtifactInput, CouncilStudyContract, PreparationScript]:
    """Run resumed preparation owners, then freeze five real configured native studies."""
    initial_catalog = fixtures._catalog
    aliases = tuple(f"open-actor-{index}" for index in range(5))

    def catalog(root: Path) -> None:
        """Add five explicitly generative fixture routes before any evidence freezes."""
        initial_catalog(root)
        path = root / "models.toml"
        current = load_model_catalog(path)
        caps = ModelCapabilities(
            supports_tools=True,
            supports_completions=True,
            context_window_tokens=200000,
            maximum_output_tokens=16000,
        )
        actors = {
            alias: current.models["world"].model_copy(
                update={"model": f"open-fixture-{index}", "capabilities": caps}
            )
            for index, alias in enumerate(aliases)
        }
        write_model_catalog(
            path, current.model_copy(update={"models": {**current.models, **actors}})
        )

    monkeypatch.setattr(fixtures, "_catalog", catalog)
    project, preparation, reviewer = fixtures.preparation_fixture(tmp_path, monkeypatch)
    preparation = preparation.model_copy(
        update={
            "bootstrap": preparation.bootstrap.model_copy(
                update={
                    "runtime": preparation.bootstrap.runtime.model_copy(
                        update={"maximum_physical_attempts": 20}
                    ),
                    "job": preparation.bootstrap.job.model_copy(update={"maximum_model_calls": 20}),
                }
            )
        }
    )
    paths = _origin(project, preparation, tmp_path)
    resume = _register(project, paths)
    wire = NativePreparationScript()
    predictor = PreparationPredictor()
    monkeypatch.setattr(
        "exp.optimize.evaluation.factory.authoring.HttpxJsonTransport", lambda: wire
    )
    monkeypatch.setattr(grounded, "HttpxJsonTransport", lambda: predictor)
    monkeypatch.setattr(
        "exp.runtime.models.providers.transport.HttpxJsonTransport", lambda: predictor
    )
    report = run_bootstrap_preparation(
        project,
        preparation,
        code_revision="corrected",
        resume_binding=resume,
        dependencies=BootstrapDependencies(
            reviewer=reviewer, tagger=_tagger(project, FakeDecisions())
        ),
    )
    assert not report.failed_tasks and not report.refused_tasks
    prepared = artifact_input(project.artifacts.read(report.identity + "-result").manifest)
    bootstrap = report.bindings[0]
    start = bootstrapped_registration(project, bootstrap)
    base = _spec(project, start.job)
    role = base.judges[0]
    mining = MiningSettings(
        bootstrap=bootstrap,
        actors=aliases,
        judge=role,
        selector=role,
        critic=role,
    )
    family = load_family_simulator(project.artifacts, report.simulators[0])
    registry = RuntimeModelCatalog(load_model_catalog(project.model_catalog_path), environment={})
    studies: list[ActorCriticSpec] = []
    for alias in aliases:
        initial = start.job.model_copy(
            update={
                "actor_alias": alias,
                "study": start.job.study.model_copy(
                    update={"actor_model": registry.snapshot(alias)[0]}
                ),
            }
        )
        study = ActorCriticSpec.error_mining(
            initial,
            episode_judge=base.episode_judge,
            selector=role,
            critic=role,
            actor_tier=alias,
            references=(
                ReferenceSource(
                    kind="rule_book",
                    name="rules",
                    artifact=start.rulebook,
                    file="rules.json",
                ),
            ),
        ).model_copy(
            update={
                "retrieval": TraceSearchSpec(
                    index=family.index,
                    scope=family.corpus,
                    task_id=start.task_id,
                )
            }
        )
        studies.append(study)
    build = VarianceBuildReport.model_validate_json(
        project.artifacts.read_bytes(report.build_report.artifact_id, "report.json")
    )
    contract = CouncilStudyContract(
        preparation_spec=preparation,
        bootstrap=bootstrap,
        mining=mining,
        studies=tuple(studies),
        exposure=build.exposure,
        family_simulator=report.simulators[0],
        sources=council_sources(),
    )
    return project, prepared, resume, contract, wire


def test_native_task_start_requires_actual_completed_binding_and_qa(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A genuinely authored start qualifies; an unrelated task cannot borrow that QA."""
    project, prepared, _, contract, _ = _fixture(tmp_path, monkeypatch)
    report = BootstrapPreparationReport.model_validate_json(
        project.artifacts.read_bytes(prepared.artifact_id, "report.json")
    )
    binding = report.bindings[0]
    start, reference = _native_start(project, binding, report, contract.preparation_spec.bootstrap)
    assert start.validation.passed and start.rounds[-1].verdict == "accept"
    assert reference.artifact_id.endswith("-registration")
    wrong = binding.model_copy(update={"task_id": "another-task"})
    with pytest.raises(ValueError, match="no binding"):
        _native_start(project, wrong, report, contract.preparation_spec.bootstrap)


def test_complete_native_resume_qualifies_without_calls_or_ledger_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Read actual native completion, settings and QA with provider construction forbidden."""
    project, prepared, resume, spec, wire = _fixture(tmp_path, monkeypatch)
    calls = len(wire.requests)
    report = BootstrapPreparationReport.model_validate_json(
        project.artifacts.read_bytes(prepared.artifact_id, "report.json")
    )
    ceiling = operation_spend_ceiling(project, report.identity)
    before = ceiling.path.read_bytes()

    def forbidden() -> None:
        """Reject any provider-client construction during read-only qualification."""
        raise AssertionError("qualification dispatched a provider")

    monkeypatch.setattr(grounded, "HttpxJsonTransport", forbidden)
    monkeypatch.setattr("exp.optimize.evaluation.factory.authoring.HttpxJsonTransport", forbidden)
    qualified = qualify_council(project, prepared, resume, spec, code_revision="corrected")
    assert tuple(row.alias for row in qualified.actors) == spec.mining.actors
    assert (
        configured_council_actors(
            project,
            aliases=spec.mining.actors,
            model_ids=tuple(f"open-fixture-{index}" for index in range(5)),
        )
        == qualified.actors
    )
    assert len(set(row.model for row in qualified.actors)) == 5
    assert set(council_aliases(project, qualified)) == {
        *spec.mining.actors,
        "judge",
        "world",
        "embed",
    }
    assert len(wire.requests) == calls and ceiling.path.read_bytes() == before


@pytest.mark.parametrize("change", ["producer", "source", "G", "H", "roles", "RAG"])
def test_drift_cannot_borrow_completed_preparation_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    """Native job, prompt, producer, sampling and retrieval drift refuse before dispatch."""
    project, prepared, resume, spec, wire = _fixture(tmp_path, monkeypatch)
    revision = "corrected"
    if change == "producer":
        revision = "unreviewed"
    elif change == "source":
        spec = spec.model_copy(update={"sources": {}})
    else:
        study = spec.studies[0]
        if change in ("G", "H"):
            initial = study.initial
            initial = initial.model_copy(
                update=(
                    {"maximum_model_calls": 19}
                    if change == "H"
                    else {"study": initial.study.model_copy(update={"repeats": 15})}
                )
            )
            study = study.model_copy(update={"initial": initial})
        elif change == "roles":
            study = study.model_copy(update={"critics": study.critics[:-1]})
        else:
            assert study.retrieval is not None
            study = study.model_copy(
                update={"retrieval": study.retrieval.model_copy(update={"task_id": "another-task"})}
            )
        spec = spec.model_copy(update={"studies": (study, *spec.studies[1:])})
    calls = len(wire.requests)
    with pytest.raises(ValueError):
        qualify_council(project, prepared, resume, spec, code_revision=revision)
    assert len(wire.requests) == calls


def test_missing_preparation_cannot_write_conditional_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A build, task start or invented result reference never substitutes for preparation."""
    project, prepared, resume, spec, _ = _fixture(tmp_path, monkeypatch)
    missing = ArtifactInput(artifact_id="no-completed-preparation", sha256="0" * 64)
    before = project.artifacts.list_ids()
    with pytest.raises((ValueError, FileNotFoundError)):
        register_conditional_council(
            project, missing, resume, spec, prepared, code_revision="corrected"
        )
    assert project.artifacts.list_ids() == before
    with pytest.raises(ValueError, match="native bootstrap-preparation"):
        qualify_council(project, spec.family_simulator, resume, spec, code_revision="corrected")


def test_registration_preserves_native_original_ledger_and_legacy_grants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only exact consent registers additive parent authority after real offline preparation."""
    project, prepared, resume, spec, _ = _fixture(tmp_path, monkeypatch)
    qualified = qualify_council(project, prepared, resume, spec, code_revision="corrected")
    from exp.optimize.evaluation.variance.bootstrap_resume import load_bootstrap_resume

    binding = load_bootstrap_resume(
        project, resume, spec=spec.preparation_spec, code_revision="corrected"
    )
    ceiling = operation_spend_ceiling(project, binding.original_operation)
    summary = ceiling.summary()
    approval = CouncilConditionalConsent(
        operation=binding.original_operation,
        ledger_sha256=binding.ledger_sha256,
        contract_sha256=sha256_json(spec),
        preparation=prepared,
        resume_binding=resume,
        open_actors=qualified.actors,
        legacy_attempt_limit=binding.unpriced_attempts,
        maximum_known_usd=binding.ceiling_usd,
        deadline=datetime.now(UTC) + timedelta(hours=1),
        producer_revision="corrected",
    )
    manifest = project.artifacts.write_json(
        artifact_id="explicit-conditional-consent",
        artifact_type=CONSENT_TYPE,
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=datetime.now(UTC),
            code_revision="corrected",
            inputs=(prepared, resume),
        ),
        files={"consent.json": approval},
    )
    result = register_conditional_council(
        project, prepared, resume, spec, artifact_input(manifest), code_revision="corrected"
    )
    assert result.policy.ledger_sha256 == binding.ledger_sha256
    assert ceiling.summary() == summary
    assert (
        register_conditional_council(
            project, prepared, resume, spec, artifact_input(manifest), code_revision="corrected"
        )
        == result
    )
