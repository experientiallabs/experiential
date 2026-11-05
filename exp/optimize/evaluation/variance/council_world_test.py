"""Exact native per-job world authority with scripted preparation and no real services."""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    JsonObject,
    sha256_json,
    stable_id,
)
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.job import (
    continuation_result_reference,
    load_continuation_runtime,
)
from exp.optimize.evaluation.executable.runtime_authority import runtime_authority_scope
from exp.optimize.evaluation.spending import operation_spend_ceiling
from exp.optimize.evaluation.variance import council_descendant as descendant_owner
from exp.optimize.evaluation.variance import council_world as world_owner
from exp.optimize.evaluation.variance.actor_critic import ActorCriticSpec
from exp.optimize.evaluation.variance.actor_critic_test import _done
from exp.optimize.evaluation.variance.bootstrap_history import (
    read_bootstrap_preparation_provenance,
)
from exp.optimize.evaluation.variance.bootstrap_resume import load_bootstrap_resume
from exp.optimize.evaluation.variance.build_test import FakeDecisions, _tagger
from exp.optimize.evaluation.variance.council_checkpoint import CouncilCheckpoint
from exp.optimize.evaluation.variance.council_descendant import (
    load_council_descendant,
    register_full_council_from_prelude,
)
from exp.optimize.evaluation.variance.council_inventory import CouncilPhaseInventory
from exp.optimize.evaluation.variance.council_parallel import council_draws
from exp.optimize.evaluation.variance.council_prelude import (
    freeze_council_prelude,
    qualify_council_prelude,
)
from exp.optimize.evaluation.variance.council_prelude_hook import CouncilPreludeCheckpoint
from exp.optimize.evaluation.variance.council_prelude_inventory import prelude_inventory
from exp.optimize.evaluation.variance.council_prelude_test import (
    _prelude_fixture,
    _register_fixture,
)
from exp.optimize.evaluation.variance.council_prelude_world import (
    register_prelude_runtime,
    save_prelude_inventory,
)
from exp.optimize.evaluation.variance.council_qualification import (
    CONSENT_TYPE,
    ConditionalCouncilRegistration,
    CouncilConditionalConsent,
    CouncilStudyContract,
    qualify_council,
    register_conditional_council,
)
from exp.optimize.evaluation.variance.council_qualification_test import _fixture
from exp.optimize.evaluation.variance.council_world import (
    register_council_runtime,
    resolve_council_qualification,
    save_council_inventory,
)
from exp.optimize.evaluation.variance.pipeline import (
    plan_decision_variance,
    run_decision_variance,
)
from exp.optimize.evaluation.variance.pipeline_spec import (
    PipelineApproval,
    StageContext,
    StageDependencies,
)
from exp.optimize.evaluation.variance.rubric_author import Rubric, RubricBody
from exp.optimize.evaluation.variance.rubric_author_test import rubric_answer
from exp.runtime.models.providers.transport import (
    JsonHttpResponse,
    RecordedRequest,
    ScriptedJsonTransport,
)
from exp.runtime.models.scoped_spending import (
    SpendEnvelopePlan,
    open_spend_envelope,
    spend_envelope_scope,
    verify_spend_envelope,
)
from exp.runtime.models.spend_ceiling import install_spend_ceiling, operation_spend_scope


def register_fixture(
    project: ProjectStore,
    prepared: ArtifactInput,
    resume: ArtifactInput,
    spec: CouncilStudyContract,
) -> ConditionalCouncilRegistration:
    """Register actual native offline preparation, its QA and explicit synthetic test consent."""
    qualified = qualify_council(project, prepared, resume, spec, code_revision="corrected")
    binding = load_bootstrap_resume(
        project, resume, spec=spec.preparation_spec, code_revision="corrected"
    )
    consent = CouncilConditionalConsent(
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
    reference = artifact_input(
        project.artifacts.write_json(
            artifact_id="explicit-native-world-test-consent",
            artifact_type=CONSENT_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision="corrected",
                inputs=(prepared, resume),
            ),
            files={"consent.json": consent},
        )
    )
    return register_conditional_council(
        project, prepared, resume, spec, reference, code_revision="corrected"
    )


def test_actual_loader_requires_exact_registered_job_envelope_and_preserves_resume_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real native actor wave restores the frozen world with no trial-receipt fabrication."""
    project, prepared, resume, contract, wire = _fixture(tmp_path, monkeypatch)
    registration = register_fixture(project, prepared, resume, contract)
    binding = load_bootstrap_resume(
        project, resume, spec=contract.preparation_spec, code_revision="corrected"
    )
    ceiling = operation_spend_ceiling(project, binding.original_operation)
    spec = contract.studies[0]
    checkpoint = CouncilCheckpoint(
        run_identity=spec.identity("corrected"), phase="actors", jobs=(spec.initial,)
    )
    reference = save_council_inventory(
        project, registration.qualification, spec, checkpoint, code_revision="corrected"
    )
    inventory = CouncilPhaseInventory.model_validate_json(
        project.artifacts.read_bytes(reference.artifact_id, "inventory.json")
    )
    # This test exercises one native runtime boundary, not a full-study future-floor planner.
    envelope_plan = SpendEnvelopePlan(
        policy=registration.policy,
        scope="actual-actor-wave",
        inventory=reference,
        maximum_attempts=inventory.attempts,
        maximum_known_usd=inventory.known_usd,
        remaining_minimum_attempts=0,
        requirements_sha256=sha256_json(contract),
    )
    envelope = open_spend_envelope(ceiling, envelope_plan)
    calls = len(wire.requests)
    before = ceiling.summary()
    with spend_envelope_scope(ceiling, envelope):
        grant = register_council_runtime(
            project,
            registration.qualification,
            reference,
            spec.initial,
            code_revision="corrected",
        )
        assert grant == register_council_runtime(
            project,
            registration.qualification,
            reference,
            spec.initial,
            code_revision="corrected",
        )
        grant.require_configuration(
            project, spec.initial.configuration, runtime_sha256=spec.initial.runtime_sha256
        )
        with pytest.raises(ValueError, match="exact job or native wave"):
            grant.require_configuration(project, {})
        with pytest.raises(ValueError, match="exact job or native wave"):
            grant.require_configuration(
                project, spec.initial.configuration, runtime_sha256="0" * 64
            )
        with pytest.raises(ValueError, match="exact admitted native wave"):
            register_council_runtime(
                project,
                registration.qualification,
                reference,
                contract.studies[1].initial,
                code_revision="corrected",
            )

        def restore() -> str:
            """Use the actual loader in copied bounded worker contexts without model calls."""
            with load_continuation_runtime(project, spec.initial) as runtime:
                return runtime.probe_sha256

        with runtime_authority_scope(lambda project, job: grant):
            with load_continuation_runtime(project, spec.initial) as runtime:
                assert runtime.probe_sha256 == spec.initial.study.probe_sha256
            other = contract.studies[1].initial
            assert other.configuration == spec.initial.configuration
            assert other.runtime_sha256 == spec.initial.runtime_sha256
            assert other != spec.initial
            with pytest.raises(ValueError, match="another exact incoming job"):
                load_continuation_runtime(project, other)

        with runtime_authority_scope(
            lambda project, job: register_council_runtime(
                project, registration.qualification, reference, job, code_revision="corrected"
            )
        ):
            assert council_draws((restore, restore), 2) == (
                spec.initial.study.probe_sha256,
                spec.initial.study.probe_sha256,
            )
    assert ceiling.summary() == before and len(wire.requests) == calls
    with pytest.raises(ValueError, match="active native spend envelope"):
        grant.require_configuration(project, spec.initial.configuration)
    second = open_spend_envelope(
        ceiling, envelope_plan.model_copy(update={"scope": "must-not-refill-same-job"})
    )
    with spend_envelope_scope(ceiling, second):
        with pytest.raises(ValueError, match="cannot refill"):
            register_council_runtime(
                project,
                registration.qualification,
                reference,
                spec.initial,
                code_revision="corrected",
            )
        with pytest.raises(ValueError, match="exact job or native wave"):
            grant.require_configuration(project, spec.initial.configuration)
    assert not project.artifacts.list_ids(artifact_type="executable-trial")


def test_suffix_authority_rechecks_seeded_group_and_actual_selector_citations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A suffix cannot borrow a different boundary, even when its job shape matches."""
    from exp.optimize.evaluation.continuations.job_test import job_fixture
    from exp.optimize.evaluation.variance.actor_critic import run_actor_critic
    from exp.optimize.evaluation.variance.actor_critic_test import (
        _done,
        _edit,
        _exposure,
        _point,
        _spec,
        _tool_reply,
        _wire,
    )
    from exp.optimize.evaluation.variance.council_checkpoint import point_job
    from exp.optimize.evaluation.variance.council_inventory import phase_inventory
    from exp.optimize.evaluation.variance.council_world import _selected_jobs

    project, job, _ = job_fixture(tmp_path, monkeypatch)
    spec = _spec(project, job)
    point = {**_point(0), "evidence_refs": ["point:0", "judgment:0"]}
    roles = [
        _tool_reply(
            "choose_points",
            {"points": [point], "audit": [], "limitations": ["Scripted offline fixture."]},
        ),
        _tool_reply(
            "estimate_points",
            {
                "estimates": [
                    {
                        "step": 0,
                        "probability": 0.5,
                        "rationale": "Attempt-specific initial LLM feedback is supplied.",
                        "evidence_refs": ["judgment:0"],
                    }
                ],
                "limitations": ["Scripted offline fixture."],
            },
        ),
    ]
    actor, judge = _wire(
        monkeypatch,
        [reply for row in (2, 3, 4, 2, 3) for reply in (_edit(row), _done())],
        roles,
    )
    report = run_actor_critic(
        project,
        spec,
        exposure=_exposure(project, job, "mining_probes"),
        code_revision="suffix-authority",
    )
    assert report.initial_judged is not None and len(report.points) == 1
    checkpoint = CouncilCheckpoint(
        run_identity=spec.identity("suffix-authority"),
        phase="continuations",
        group=report.group,
        initial_judged=report.initial_judged,
        selected=(0,),
        wave=(0,),
        jobs=(point_job(spec, report.points[0].checkpoint),),
    )
    inventory = phase_inventory(project, spec, checkpoint, code_revision="suffix-authority")
    calls = (len(actor.requests), len(judge.judged))
    _selected_jobs(project, inventory, spec, code_revision="suffix-authority")
    with pytest.raises(ValueError, match="selected native checkpoint"):
        _selected_jobs(
            project,
            inventory.model_copy(
                update={"checkpoint": checkpoint.model_copy(update={"wave": (1,)})}
            ),
            spec,
            code_revision="suffix-authority",
        )
    with pytest.raises(ValueError, match="actual initial LLM feedback"):
        _selected_jobs(
            project,
            inventory.model_copy(
                update={"checkpoint": checkpoint.model_copy(update={"initial_judged": None})}
            ),
            spec,
            code_revision="suffix-authority",
        )
    assert (len(actor.requests), len(judge.judged)) == calls


class RegistrationReached(Exception):
    """Stop the native pipeline after its actual full studies have been registered."""


def test_native_descendant_keeps_original_prelude_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Native produced groups and labels admit a full descendant without replacing its parent."""

    project, prepared, resume, contract, wire = _prelude_fixture(tmp_path, monkeypatch)
    rubric_ids = project.artifacts.list_ids(artifact_type="episode-rubric")
    assert len(rubric_ids) == 1
    declared = Rubric(
        protocol="explicit-fixture-declaration-v1",
        family_id="native-descendant-fixture",
        packet_sha256=sha256_json(contract.preparation),
        body=RubricBody.model_validate(rubric_answer()["rubric"]),
    )
    rubric = artifact_input(
        project.artifacts.write_json(
            artifact_id="explicit-native-descendant-rubric",
            artifact_type="episode-rubric",
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision="explicit-fixture-declaration",
                inputs=(prepared,),
            ),
            files={"rubric.json": declared},
        )
    )
    pipeline = contract.pipeline.model_copy(
        update={
            "mining": contract.pipeline.mining.model_copy(
                update={"rubric": rubric, "behaviors": None}
            )
        }
    )
    contract = freeze_council_prelude(
        contract.preparation,
        pipeline,
        actor_model_ids=contract.actor_model_ids,
        code_revision="council-executor",
    )
    registration = _register_fixture(project, prepared, resume, contract)
    qualified = qualify_council_prelude(
        project, prepared, resume, contract, code_revision="council-executor"
    )
    binding = read_bootstrap_preparation_provenance(
        project, resume, spec=contract.preparation.preparation_spec
    ).binding
    ceiling = operation_spend_ceiling(project, binding.original_operation)
    authorizations = ceiling.summary().authorizations

    class DoneActors(ScriptedJsonTransport):
        """Return a complete actor message using each requested model's actual wire identity."""

        def post(
            self,
            url: str,
            *,
            headers: Mapping[str, str],
            payload: JsonObject,
            timeout_seconds: float,
        ) -> JsonHttpResponse:
            """Record one actual admitted actor request without creating artificial study rows."""
            del timeout_seconds
            self.requests.append(RecordedRequest(url, dict(headers), payload))
            response = _done()
            return JsonHttpResponse(
                response.status_code, {**response.body, "model": payload["model"]}
            )

    actors = DoneActors(())
    monkeypatch.setattr(
        "exp.optimize.evaluation.continuations.job.HttpxJsonTransport", lambda: actors
    )
    descendants: list[ArtifactInput] = []

    @contextmanager
    def admit(context: StageContext, checkpoint: CouncilPreludeCheckpoint) -> Iterator[None]:
        """Use native quotes and real original-ledger envelopes for actual prelude owners."""
        del context
        if checkpoint.phase == "behavior_reference":
            assert pipeline.mining.behaviors is None
            yield
            return
        inventory = prelude_inventory(project, qualified, checkpoint)
        if not inventory.attempts:
            yield
            return
        reference = save_prelude_inventory(
            project, registration.qualification, checkpoint, code_revision="council-executor"
        )
        envelope = open_spend_envelope(
            ceiling,
            SpendEnvelopePlan(
                policy=registration.policy,
                scope=f"{checkpoint.phase}-{checkpoint.actor_alias}",
                inventory=reference,
                maximum_attempts=inventory.attempts,
                maximum_known_usd=inventory.known_usd,
                remaining_minimum_attempts=inventory.remaining.attempts,
                requirements_sha256=sha256_json(contract),
            ),
        )
        with (
            spend_envelope_scope(ceiling, envelope),
            runtime_authority_scope(
                lambda owner, job: register_prelude_runtime(
                    owner,
                    registration.qualification,
                    reference,
                    job,
                    code_revision="council-executor",
                )
            ),
        ):
            yield

    def register(context: StageContext, studies: tuple[ActorCriticSpec, ...]) -> None:
        """Let the descendant owner verify actual pipeline-produced studies and native groups."""
        del context
        assert len(actors.requests) == 5 * 16
        descendants.append(
            register_full_council_from_prelude(
                project, registration.qualification, studies, code_revision="council-executor"
            )
        )
        raise RegistrationReached()

    plan = plan_decision_variance(project, pipeline, code_revision="council-executor")
    with operation_spend_scope():
        install_spend_ceiling(ceiling)
        with pytest.raises(RegistrationReached):
            run_decision_variance(
                project,
                pipeline,
                code_revision="council-executor",
                approval=PipelineApproval(
                    plan_sha256=plan.digest, ceiling_usd=plan.known_usd, unpriced_allowed=True
                ),
                dependencies=StageDependencies(
                    council_prelude_admission=admit,
                    council_registration=register,
                    traceml_tagger=_tagger(project, FakeDecisions()),
                ),
            )
    assert len(actors.requests) == 5 * 16
    assert len(descendants) == 1
    descendant = load_council_descendant(project, descendants[0])
    assert descendant.prelude == registration.qualification
    assert ceiling.summary().authorizations == authorizations
    altered = descendant.qualified.contract.studies[1].model_copy(
        update={"references": descendant.qualified.contract.studies[0].references}
    )
    with pytest.raises(ValueError, match="native model group"):
        register_full_council_from_prelude(
            project,
            registration.qualification,
            (
                descendant.qualified.contract.studies[0],
                altered,
                *descendant.qualified.contract.studies[2:],
            ),
            code_revision="council-executor",
        )
    with pytest.raises(ValueError, match="another original parent"):
        resolve_council_qualification(project, descendants[0], parent_execution=prepared)
    with pytest.raises(ValueError):
        resolve_council_qualification(
            project, descendants[0].model_copy(update={"sha256": "0" * 64})
        )

    first = descendant.qualified.contract.studies[0]
    assert first.retrieval is not None
    changed_search = first.model_copy(
        update={"retrieval": first.retrieval.model_copy(update={"index": prepared})}
    )
    foreign = descendant.model_copy(
        update={
            "qualified": descendant.qualified.model_copy(
                update={
                    "contract": descendant.qualified.contract.model_copy(
                        update={
                            "studies": (changed_search, *descendant.qualified.contract.studies[1:])
                        }
                    )
                }
            )
        }
    )
    foreign_ref = artifact_input(
        project.artifacts.write_json(
            artifact_id=stable_id(
                descendant_owner.DESCENDANT_TYPE, {"descendant": sha256_json(foreign)}
            ),
            artifact_type=descendant_owner.DESCENDANT_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=descendant.qualified.producer_revision,
                inputs=project.artifacts.read(descendants[0].artifact_id).manifest.inputs,
            ),
            files={"descendant.json": foreign},
        )
    )
    with pytest.raises(ValueError, match="family RAG scope"):
        resolve_council_qualification(
            project, foreign_ref, parent_execution=registration.qualification
        )

    def history_must_not_replay(owner: ProjectStore, reference: ArtifactInput) -> None:
        """Prove parent-bound current reads do not replay historical preparation qualification."""
        del owner, reference
        raise AssertionError("historical preparation replayed after immutable parent registration")

    with monkeypatch.context() as bounded:
        bounded.setattr(descendant_owner, "load_qualified_prelude", history_must_not_replay)
        for study in descendant.qualified.contract.studies:
            reference = save_council_inventory(
                project,
                descendants[0],
                study,
                CouncilCheckpoint(
                    run_identity=study.identity("council-executor"),
                    phase="selectors",
                    group=continuation_result_reference(
                        project, study.initial, code_revision="council-executor"
                    ),
                    draws=tuple(range(8)),
                ),
                code_revision="council-executor",
                parent_execution=registration.qualification,
            )
            native_quote = CouncilPhaseInventory.model_validate_json(
                project.artifacts.read_bytes(reference.artifact_id, "inventory.json")
            )
            envelope = open_spend_envelope(
                ceiling,
                SpendEnvelopePlan(
                    policy=registration.policy,
                    scope=f"actual-full-study-{study.initial.actor_alias}",
                    inventory=reference,
                    maximum_attempts=max(1, native_quote.attempts),
                    maximum_known_usd=native_quote.known_usd,
                    remaining_minimum_attempts=0,
                    requirements_sha256=sha256_json(contract),
                ),
            )
            with spend_envelope_scope(ceiling, envelope):
                actual, parent = resolve_council_qualification(
                    project, descendants[0], parent_execution=registration.qualification
                )
                opened = verify_spend_envelope(ceiling, envelope)
                assert actual == descendant.qualified
                assert parent == opened.policy.execution == registration.qualification
    with monkeypatch.context() as changed_source:
        changed_source.setattr(world_owner, "council_sources", lambda: {})
        with pytest.raises(ValueError, match="qualified council producer"):
            resolve_council_qualification(
                project, descendants[0], parent_execution=registration.qualification
            )
    study = descendant.qualified.contract.studies[0]
    cached = CouncilCheckpoint(
        run_identity=study.identity("council-executor"), phase="actors", jobs=(study.initial,)
    )
    reference = save_council_inventory(
        project,
        descendants[0],
        study,
        cached,
        code_revision="council-executor",
        parent_execution=registration.qualification,
    )
    native_quote = CouncilPhaseInventory.model_validate_json(
        project.artifacts.read_bytes(reference.artifact_id, "inventory.json")
    )
    envelope = open_spend_envelope(
        ceiling,
        SpendEnvelopePlan(
            policy=registration.policy,
            scope="no-initial-world-refill",
            inventory=reference,
            maximum_attempts=max(1, native_quote.attempts),
            maximum_known_usd=native_quote.known_usd,
            remaining_minimum_attempts=0,
            requirements_sha256=sha256_json(contract),
        ),
    )
    with spend_envelope_scope(ceiling, envelope):
        with pytest.raises(ValueError, match="cannot refill"):
            register_council_runtime(
                project, descendants[0], reference, study.initial, code_revision="council-executor"
            )
    assert ceiling.summary().authorizations == authorizations
