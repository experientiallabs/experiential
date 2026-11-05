"""Native paid prelude scopes and completion certificates from actual owner returns."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    sha256_json,
    stable_id,
)
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.job import (
    continuation_result_reference,
    saved_continuation_job,
)
from exp.optimize.evaluation.executable.runtime_authority import runtime_authority_scope
from exp.optimize.evaluation.spending import operation_spend_ceiling
from exp.optimize.evaluation.variance.authoring_packet import load_packet
from exp.optimize.evaluation.variance.behavior_pairs import load_behavior_differences
from exp.optimize.evaluation.variance.bootstrap import resolve_bootstrap
from exp.optimize.evaluation.variance.bootstrap_resume import BootstrapResumeBinding
from exp.optimize.evaluation.variance.council_prelude import (
    ConditionalCouncilPreludeRegistration,
    _rubric_proof,
    load_qualified_prelude,
)
from exp.optimize.evaluation.variance.council_prelude_hook import CouncilPreludeCheckpoint
from exp.optimize.evaluation.variance.council_prelude_inventory import (
    CouncilPreludeInventory,
    _completed_errors,
    _completed_labels,
    _judgments_pending,
)
from exp.optimize.evaluation.variance.council_prelude_world import (
    register_prelude_runtime,
    save_prelude_inventory,
)
from exp.optimize.evaluation.variance.pipeline_inputs import rubric_author_sha256
from exp.optimize.evaluation.variance.pipeline_spec import StageContext
from exp.optimize.evaluation.variance.rubric_agent import AGENTIC_TYPE, AgenticRubric
from exp.optimize.evaluation.variance.rubric_author import load_rubric
from exp.runtime.models.scoped_spending import (
    EnvelopeCompletion,
    SpendEnvelopePlan,
    SpendEnvelopeRef,
    open_spend_envelope,
    settle_spend_envelope,
    spend_envelope_scope,
    verify_spend_envelope,
)

COMPLETION_TYPE = "native-spend-envelope-completion"


@dataclass(frozen=True)
class ActivePrelude:
    """One immutable phase context copied into its native bounded workers.

    Attributes:
        qualification: Exact registered native prelude.
        checkpoint: Exact real owner being executed.
        inventory: Original native quote, including every model's mandatory future floor.
        reference: Exact quote artifact backing this paid scope.
        envelope: Paid envelope, or None for proven native replay with no pending requests.
    """

    qualification: ArtifactInput
    checkpoint: CouncilPreludeCheckpoint
    inventory: CouncilPreludeInventory
    reference: ArtifactInput
    envelope: SpendEnvelopeRef | None


_ACTIVE: ContextVar[ActivePrelude | None] = ContextVar("native-active-prelude", default=None)
_COMPLETED: ContextVar[ArtifactInput | None] = ContextVar("native-prelude-completed", default=None)


class PreludeDisposition(ContractModel):
    """Known native owner failures, kept apart from physical settlement or task success.

    Attributes:
        failures: Explicit terminal failures from the returned native owner records.
        authority: Budget completion does not certify task success or training eligibility.
    """

    failures: tuple[str, ...] = ()
    authority: str = (
        "native owner return only; task success and training eligibility remain separate"
    )


class PreludeCompletionProof(ContractModel):
    """Exact owner-return inputs behind a replayable native phase completion.

    Attributes:
        qualification: Original exact prelude registration.
        checkpoint: Exact native owner whose result was returned.
        returned: References returned directly by that native owner.
    """

    qualification: ArtifactInput
    checkpoint: CouncilPreludeCheckpoint
    returned: tuple[ArtifactInput, ...]


class NativePreludeExecution:
    """Admit actual native prelude owners under their original parent and source proof.

    Attributes:
        project: Exact original project holding all immutable native evidence.
        registration: Native prelude qualification and original-ledger policy.
        code_revision: Actual reviewed executor.
        qualified: Immutable native prelude with exact source, catalog and job proofs.
        ceiling: The unchanged original physical spend ledger.
    """

    def __init__(
        self,
        project: ProjectStore,
        registration: ConditionalCouncilPreludeRegistration,
        *,
        code_revision: str,
    ) -> None:
        """Read the genuine prelude and original ledger without adding grants or claims."""
        self.project, self.registration, self.code_revision = project, registration, code_revision
        self.qualified = load_qualified_prelude(project, registration.qualification)
        if self.qualified.producer_revision != code_revision:
            raise ValueError("prelude executor differs from its native qualification")
        binding = BootstrapResumeBinding.model_validate_json(
            project.artifacts.read_bytes(self.qualified.resume_binding.artifact_id, "binding.json")
        )
        self.ceiling = operation_spend_ceiling(project, binding.original_operation)

    def _context(self, context: StageContext) -> None:
        """Refuse another pipeline, project or producer before native quote construction."""
        if (
            context.project.paths != self.project.paths
            or context.code_revision != self.code_revision
            or context.spec != resolve_bootstrap(self.project, self.qualified.contract.pipeline)
        ):
            raise ValueError("prelude execution changed its exact native pipeline or producer")

    def _completion_identity(self, checkpoint: CouncilPreludeCheckpoint) -> str:
        """Name one current-source native phase completion without an artifact scan."""
        return stable_id(
            COMPLETION_TYPE,
            {
                "qualification": self.registration.qualification.model_dump(mode="json"),
                "checkpoint": sha256_json(checkpoint),
            },
        )

    def _replay(
        self, checkpoint: CouncilPreludeCheckpoint
    ) -> tuple[ArtifactInput, EnvelopeCompletion] | None:
        """Verify completed native owners and actual physical settlement before exact replay."""
        identity = self._completion_identity(checkpoint)
        if not self.project.artifacts.exists(identity):
            return None
        stored = self.project.artifacts.read(identity)
        proof = PreludeCompletionProof.model_validate_json(stored.payloads["proof.json"])
        record = EnvelopeCompletion.model_validate_json(stored.payloads["completion.json"])
        if (
            stored.manifest.artifact_type != COMPLETION_TYPE
            or stored.manifest.code_revision != self.code_revision
            or proof.qualification != self.registration.qualification
            or proof.checkpoint != checkpoint
            or record.envelope.policy != self.registration.policy
        ):
            raise ValueError("native prelude completion changed its exact source or phase")
        verified, _ = self._owners(checkpoint, proof.returned)
        if verified != record.owners:
            raise ValueError("native prelude completion changed its actual returned owners")
        reference = artifact_input(stored.manifest)
        settle_spend_envelope(self.project, self.ceiling, record.envelope, reference)
        return reference, record

    @contextmanager
    def __call__(
        self, context: StageContext, checkpoint: CouncilPreludeCheckpoint
    ) -> Iterator[None]:
        """Open the exact native current-wave bound and retain every model's future work."""
        self._context(context)
        if _ACTIVE.get() is not None:
            raise ValueError("cannot nest another native prelude owner")
        replay = self._replay(checkpoint)
        if replay is not None:
            reference, record = replay
            inventory = CouncilPreludeInventory.model_validate_json(
                self.project.artifacts.read_bytes(record.inventory.artifact_id, "inventory.json")
            )
            token = _ACTIVE.set(
                ActivePrelude(
                    self.registration.qualification, checkpoint, inventory, record.inventory, None
                )
            )
            finished = _COMPLETED.set(None)
            try:
                yield
                if _COMPLETED.get() is None:
                    raise ValueError("native replay returned without verified owner references")
            finally:
                _COMPLETED.reset(finished)
                _ACTIVE.reset(token)
            return
        reference = save_prelude_inventory(
            self.project,
            self.registration.qualification,
            checkpoint,
            code_revision=self.code_revision,
        )
        inventory = CouncilPreludeInventory.model_validate_json(
            self.project.artifacts.read_bytes(reference.artifact_id, "inventory.json")
        )
        envelope = None
        if inventory.attempts:
            envelope = open_spend_envelope(
                self.ceiling,
                SpendEnvelopePlan(
                    policy=self.registration.policy,
                    scope=stable_id("native-prelude-wave", {"checkpoint": sha256_json(checkpoint)}),
                    inventory=reference,
                    maximum_attempts=inventory.attempts,
                    maximum_known_usd=inventory.known_usd,
                    remaining_minimum_attempts=inventory.remaining.attempts,
                    requirements_sha256=inventory.contract_sha256,
                ),
            )
        active = ActivePrelude(
            self.registration.qualification, checkpoint, inventory, reference, envelope
        )
        token, finished = _ACTIVE.set(active), _COMPLETED.set(None)
        try:
            if envelope is None:
                yield
            else:
                with (
                    spend_envelope_scope(self.ceiling, envelope),
                    runtime_authority_scope(
                        lambda project, job: register_prelude_runtime(
                            project,
                            self.registration.qualification,
                            reference,
                            job,
                            code_revision=self.code_revision,
                        )
                    ),
                ):
                    yield
            if _COMPLETED.get() is None:
                raise ValueError(
                    "native prelude owner returned without verified completion references"
                )
        finally:
            _COMPLETED.reset(finished)
            _ACTIVE.reset(token)

    def _owners(
        self, checkpoint: CouncilPreludeCheckpoint, owners: tuple[ArtifactInput, ...]
    ) -> tuple[tuple[ArtifactInput, ...], PreludeDisposition]:
        """Independently verify exact native owner identities, controls and source ancestry."""
        if not owners or len(set(owners)) != len(owners):
            raise ValueError("native prelude completion needs distinct actual owner references")
        store = self.project.artifacts
        for reference in owners:
            manifest = store.read(reference.artifact_id).manifest
            if artifact_input(manifest) != reference or (
                checkpoint.phase != "traceml_group" and manifest.code_revision != self.code_revision
            ):
                raise ValueError("native prelude completion changed owner bytes or producer")
        job = checkpoint.job
        failures: tuple[str, ...] = ()
        verified = set(owners)
        if checkpoint.phase == "traceml_group":
            expected = continuation_result_reference(
                self.project, job, code_revision=self.code_revision
            )
            group = saved_continuation_job(self.project, job, code_revision=self.code_revision)
            if owners != (expected,) or group is None or len(group.branches) != job.study.repeats:
                raise ValueError("native group completion changed its exact job or attempts")
            failures = tuple(
                f"attempt {row.index}: {row.actor_stop or row.failure or row.status}"
                for row in group.branches
                if row.status != "completed" or row.actor_stop == "actor_budget_exhausted"
            )
        elif checkpoint.phase == "traceml_labels":
            expected = 4 if self.qualified.contract.pipeline.traceml.error_bank is not None else 1
            if len(owners) != expected:
                raise ValueError("native labels omitted a mandatory returned owner")
            completed = checkpoint.model_copy(update={"completed_labels": (owners[0],)})
            if checkpoint.actor_alias not in _completed_labels(
                self.project, self.qualified, completed
            ):
                raise ValueError("native labels belong to another actor's group")
            if expected == 4:
                completed = completed.model_copy(update={"completed_error_points": (owners[3],)})
                if checkpoint.actor_alias not in _completed_errors(
                    self.project, self.qualified, completed
                ):
                    raise ValueError("native error points belong to another actor's group")
                if store.read(owners[3].artifact_id).manifest.inputs != (owners[2],):
                    raise ValueError("native error points differ from the returned step owner")
                if store.read(owners[1].artifact_id).manifest.inputs != (checkpoint.group,):
                    raise ValueError("native attempt errors lost their actual group")
        elif checkpoint.phase == "mining_rubric":
            if len(owners) != 2:
                raise ValueError("native rubric completion needs the rubric and authoring packet")
            rubric, packet = owners
            _rubric_proof(self.project, self.qualified, rubric)
            material = load_packet(self.project, packet)
            if material.checkpoint != job.study.checkpoint:
                raise ValueError("native rubric packet describes another initial job")
            identity = (
                stable_id(
                    "mining-rubric",
                    {
                        "packet": packet.sha256,
                        "author": rubric_author_sha256(self.qualified.contract.pipeline),
                    },
                )
                + "-result"
            )
            stored = store.read(identity)
            run = AgenticRubric.model_validate_json(stored.payloads["agentic.json"])
            if (
                stored.manifest.artifact_type != AGENTIC_TYPE
                or stored.manifest.code_revision != self.code_revision
                or packet not in stored.manifest.inputs
                or run.profile != self.qualified.contract.pipeline.authoring.rubric_profile
                or run.draft.status != "ready"
                or run.draft.rubric != load_rubric(self.project, rubric)
                or store.read(rubric.artifact_id).manifest.inputs != (run.draft.response,)
            ):
                raise ValueError("native rubric lost its exact ready authoring owner or profile")
            verified.add(artifact_input(stored.manifest))
        else:
            if len(owners) != 1 or checkpoint.judge is None:
                raise ValueError(
                    "native behavior completion needs its exact judged difference owner"
                )
            report = load_behavior_differences(self.project, owners[0])
            expected = continuation_result_reference(
                self.project, job, code_revision=self.code_revision
            )
            settings = self.qualified.contract.pipeline.mining.behaviors
            if (
                report.study != expected
                or report.tier != checkpoint.actor_alias
                or report.spec != settings
                or report.judge != checkpoint.judge.identity
                or sha256_json(report.bank) != self.qualified.contract.banks["behavior"]
            ):
                raise ValueError("native behavior completion changed model, group, judge or bank")
            pending, judged = _judgments_pending(self.project, job, checkpoint, self.code_revision)
            if pending or judged != report.judged:
                raise ValueError("native behavior completion lacks its complete group judgments")
            verified.update((expected, report.judged))
            failures = tuple(pair.failure for pair in report.pairs if pair.failure is not None)
        return tuple(sorted(verified, key=lambda ref: ref.artifact_id)), PreludeDisposition(
            failures=failures
        )

    def completed(
        self,
        context: StageContext,
        checkpoint: CouncilPreludeCheckpoint,
        owners: tuple[ArtifactInput, ...],
        /,
    ) -> None:
        """Settle only actual returned native owners; unknown physical claims retain liability."""
        self._context(context)
        active = _ACTIVE.get()
        if (
            active is None
            or active.qualification != self.registration.qualification
            or active.checkpoint != checkpoint
            or _COMPLETED.get() is not None
        ):
            raise ValueError("native completion belongs to another active prelude wave")
        verified, disposition = self._owners(checkpoint, owners)
        if active.envelope is None:
            _COMPLETED.set(active.reference)
            return
        opened = verify_spend_envelope(self.ceiling, active.envelope)
        record = EnvelopeCompletion(
            envelope=active.envelope,
            inventory=active.reference,
            owners=verified,
            requirements_sha256=active.inventory.contract_sha256,
        )
        identity = self._completion_identity(checkpoint)
        inputs = tuple(
            sorted(
                {
                    active.reference,
                    *verified,
                    opened.policy.authorization,
                    opened.policy.preparation,
                    opened.policy.execution,
                },
                key=lambda ref: ref.artifact_id,
            )
        )
        if not self.project.artifacts.exists(identity):
            self.project.artifacts.write_json(
                artifact_id=identity,
                artifact_type=COMPLETION_TYPE,
                envelope=ArtifactEnvelope(
                    schema_version=1,
                    created_at=datetime.now(UTC),
                    code_revision=self.code_revision,
                    inputs=inputs,
                ),
                files={
                    "completion.json": record,
                    "disposition.json": disposition,
                    "proof.json": PreludeCompletionProof(
                        qualification=self.registration.qualification,
                        checkpoint=checkpoint,
                        returned=owners,
                    ),
                },
            )
        reference = artifact_input(self.project.artifacts.read(identity).manifest)
        settle_spend_envelope(self.project, self.ceiling, active.envelope, reference)
        _COMPLETED.set(reference)
