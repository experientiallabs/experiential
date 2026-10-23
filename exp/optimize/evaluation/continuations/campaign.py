"""Registered source attempts, retrospective selectors and prospective continuation experiments."""

from datetime import UTC, datetime
from typing import Self

from filelock import Timeout
from pydantic import Field, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    Sha256,
    redact_secret_text,
    sha256_json,
    stable_id,
)
from exp.common.core.locks import CoordinationLock
from exp.common.models import ModelSnapshot, ReasoningEffort, load_model_catalog
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.critic import CriticSpec, preflight_critic
from exp.optimize.evaluation.continuations.experiment import (
    CheckpointExperimentSpec,
    run_checkpoint_experiment,
    saved_checkpoint_experiment,
)
from exp.optimize.evaluation.continuations.job import (
    MAXIMUM_PHYSICAL_ATTEMPTS,
    ContinuationJob,
    continuation_result_reference,
    execute_continuation_job,
    preflight_continuation_job,
    saved_continuation_job,
)
from exp.optimize.evaluation.continuations.proposals import (
    CheckpointProposalSpec,
    SourceBranchPolicy,
    propose_checkpoints,
)
from exp.runtime.models.providers.admitted_transport import require_admitted_transport
from exp.runtime.models.registry import RuntimeModelCatalog


class CampaignSelector(ContractModel):
    """One registered retrospective selector draw, without claiming model independence.

    Attributes:
        alias: Configured selector alias.
        model: Exact model snapshot.
        catalog_sha256: Frozen credential-free provider configuration.
        output_tokens: Bound for this selector's one draw.
        reasoning_effort: Requested reasoning effort for the draw, or None for the model's
            catalog default. A lower effort leaves the output allowance for the typed selection
            instead of spending it on reasoning.
    """

    alias: str = Field(min_length=1)
    model: ModelSnapshot
    catalog_sha256: Sha256
    output_tokens: int = Field(default=4096, ge=1, le=8192)
    reasoning_effort: ReasoningEffort | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class ContinuationCampaignSpec(ContractModel):
    """An evaluation schedule, with no weight update, promotion or protected-audit authority.

    Attributes:
        initial: Complete source attempt job; its study repeats are the source attempts
            every selector inspects and the critics' base rate is measured on.
        selectors: Registered retrospective selector models, all retained without winner picking.
        critics: Prospective critic templates; every selector is evaluated with each critic.
        points: Maximum selected points per selector, default four.
        continuations: Equal repeat count per selected and random point, default eight.
        continuation_physical_attempts: Finite per-point actor HTTP ceiling, including retries.
        selector_input_bytes: Complete retrospective packet ceiling, with no truncation.
        selector_source_branches: Registered selector packet policy: ``complete`` refuses above
            the ceiling; ``registered_prefix`` sends the longest registered-order prefix of whole
            source branches that fits and names every omitted branch.
        seed: Registered seed offset for each selector's random control draw.
        horizon_tolerance: Remaining-call tolerance for matched points.
    """

    initial: ContinuationJob
    selectors: tuple[CampaignSelector, ...] = Field(min_length=1, max_length=8)
    critics: tuple[CriticSpec, ...] = Field(min_length=1, max_length=8)
    points: int = Field(default=4, ge=1, le=32, strict=True)
    continuations: int = Field(default=8, ge=1, le=100, strict=True)
    continuation_physical_attempts: int = Field(default=64, ge=1, le=10000, strict=True)
    selector_input_bytes: int = Field(default=128000, ge=512, le=512000, strict=True)
    selector_source_branches: SourceBranchPolicy = "complete"
    seed: int = Field(default=0, ge=0, le=2**32 - 9, strict=True)
    horizon_tolerance: int = Field(default=0, ge=0, le=10, strict=True)

    @model_validator(mode="after")
    def protocol(self) -> Self:
        """Fix policy and pre-outcome critic identities before purchasing the initial attempts."""
        job = self.initial
        if job.study.critic_prediction is not None or job.study.selection_plan is not None:
            raise ValueError(
                "campaign source must be an initial study, not an already selected arm"
            )
        if len({sha256_json(row) for row in self.selectors}) != len(self.selectors):
            raise ValueError("campaign selectors must have distinct registered configurations")
        if len({sha256_json(row) for row in self.critics}) != len(self.critics):
            raise ValueError("campaign critics must have distinct registered configurations")
        for critic in self.critics:
            if (
                critic.checkpoint != job.study.checkpoint
                or critic.actor_model != job.study.actor_model
                or critic.maximum_model_calls != job.maximum_model_calls
                or critic.system_prompt != job.system_prompt
                or critic.temperature != job.study.temperature
                or critic.actor_output_tokens != job.study.maximum_output_tokens
                or critic.actor_reasoning_effort != job.study.reasoning_effort
                or critic.verifier_sha256 != sha256_json(job.study.verifier)
                or critic.probe_sha256 != job.study.probe_sha256
                or critic.deadline != job.deadline
            ):
                raise ValueError("campaign critic must match the frozen actor/verifier/deadline")
        if self.attempt_bound() > MAXIMUM_PHYSICAL_ATTEMPTS:
            raise ValueError(
                f"campaign exceeds the {MAXIMUM_PHYSICAL_ATTEMPTS} actor/selector/critic "
                "attempt ceiling"
            )
        return self

    def attempt_bound(self) -> int:
        """Conservative aggregate HTTP bound; simulator calls retain their separate pinned cap."""
        return self.initial.maximum_physical_attempts + len(self.selectors) * (
            1
            + 2
            * self.points
            * sum(
                self.continuation_physical_attempts + critic.maximum_physical_attempts
                for critic in self.critics
            )
        )

    def identity(self, revision: str) -> str:
        """Bind the whole schedule before any source outcome is observed."""
        return stable_id("continuation-campaign", {"spec": sha256_json(self), "code": revision})


class CampaignCell(ContractModel):
    """One planned selector/critic pair, including failure or explicit selector abstention.

    Attributes:
        selector: Original selector position.
        critic: Original critic position.
        proposals: Saved automatic point proposals, absent if selector failed.
        experiment: Saved matched execution, absent for abstention or unresolved execution.
        failure: Safe stage code, retaining unsuccessful cells in the original inventory.
        reason: Exception type and secret-redacted message of the refusal behind ``failure``,
            at most 2000 characters; absent for success and explicit selector abstention.
    """

    selector: int
    critic: int
    proposals: ArtifactInput | None
    experiment: ArtifactInput | None
    failure: str | None
    reason: str | None = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def explained(self) -> Self:
        """Attach a refusal reason only to an unsuccessful cell."""
        if self.reason is not None and self.failure is None:
            raise ValueError("campaign cell reason needs a failure code")
        return self


class ContinuationCampaign(ContractModel):
    """Complete evaluation schedule with stage accounting and no training interpretation.

    Attributes:
        spec: Original frozen population and allowances.
        source: Complete initial attempt result, including outcome-unknown branches.
        cells: Every selector/critic pair in registered order.
        metrics: Stage counts and per-experiment measurements, never a pooled independent score.
        scope: Explicit limits on hindsight, model correlation, qualification and training.
    """

    spec: ContinuationCampaignSpec
    source: ArtifactInput
    cells: tuple[CampaignCell, ...]
    metrics: JsonObject
    scope: str = (
        "development evaluation schedule; retrospective selectors, prospective critics; "
        "correlated source attempts and model errors; no weight updates, "
        "independent task correctness or promotion"
    )


def _selector(
    spec: ContinuationCampaignSpec, index: int, source: ArtifactInput
) -> CheckpointProposalSpec:
    """Use the same frozen source population for every selector, changing only registered policy."""
    row = spec.selectors[index]
    base = spec.initial.model_copy(
        update={
            "maximum_physical_attempts": spec.continuation_physical_attempts,
            "study": spec.initial.study.model_copy(
                update={
                    "repeats": spec.continuations,
                    "critic_prediction": None,
                    "predicted_success": None,
                    "selection_plan": None,
                }
            ),
        }
    )
    return CheckpointProposalSpec(
        base=base,
        studies=(source,),
        selector_alias=row.alias,
        selector_model=row.model,
        catalog_sha256=row.catalog_sha256,
        points=spec.points,
        output_tokens=row.output_tokens,
        reasoning_effort=row.reasoning_effort,
        maximum_input_bytes=spec.selector_input_bytes,
        source_branches=spec.selector_source_branches,
        deadline=spec.initial.deadline,
        seed=spec.seed + index,
        horizon_tolerance=spec.horizon_tolerance,
    )


def preflight_continuation_campaign(project: ProjectStore, spec: ContinuationCampaignSpec) -> int:
    """Validate known models and budgets before source execution; bound later packets separately."""
    preflight_continuation_job(project, spec.initial)
    catalog = load_model_catalog(project.model_catalog_path)
    digest = sha256_json(catalog.model_dump(mode="json"))
    registry = RuntimeModelCatalog(catalog, environment={})
    for selector in spec.selectors:
        model, capability = registry.snapshot(selector.alias)
        if (
            selector.catalog_sha256 != digest
            or model != selector.model
            or capability.supports_tools is not True
        ):
            raise ValueError("campaign selector differs from the frozen tool-capable model")
        if (
            capability.maximum_output_tokens is not None
            and selector.output_tokens > capability.maximum_output_tokens
        ):
            raise ValueError("campaign selector output allowance exceeds model capacity")
        require_admitted_transport(catalog, selector.alias, consumer="campaign selector")
    for critic in spec.critics:
        preflight_critic(project, critic)
    return spec.attempt_bound()


def _cell_inputs(
    registration: ArtifactInput, source: ArtifactInput, cell: CampaignCell
) -> tuple[ArtifactInput, ...]:
    """Bind each planned role pair to the source population and every completed child."""
    return tuple(
        sorted(
            {
                registration,
                source,
                *(ref for ref in (cell.proposals, cell.experiment) if ref is not None),
            },
            key=lambda ref: ref.artifact_id,
        )
    )


def _metrics(
    project: ProjectStore,
    spec: ContinuationCampaignSpec,
    source: ArtifactInput,
    cells: tuple[CampaignCell, ...],
    revision: str,
) -> JsonObject:
    """Keep all original attempts, failed role cells and matched-study unknown bounds visible."""
    source_result = saved_continuation_job(project, spec.initial, code_revision=revision)
    if (
        source_result is None
        or continuation_result_reference(project, spec.initial, code_revision=revision) != source
    ):
        raise ValueError("campaign source differs from its registered initial job")
    measured = []
    for cell in cells:
        if cell.proposals is None:
            if cell.experiment is not None or cell.failure is None:
                raise ValueError("campaign cell lost its selector failure")
            continue
        selector_spec = _selector(spec, cell.selector, source)
        expected_id = selector_spec.identity(revision)
        if (
            not project.artifacts.exists(expected_id)
            or not project.artifacts.exists(expected_id + "-response")
            or artifact_input(project.artifacts.read(expected_id).manifest) != cell.proposals
        ):
            raise ValueError("campaign selector differs from its frozen draw")
        proposals = propose_checkpoints(project, selector_spec, code_revision=revision)
        if proposals is None:
            raise RuntimeError("checkpoint proposals returned no report outside a dry run")
        if cell.experiment is None:
            if cell.failure is None:
                raise ValueError("campaign cell lost its unfinished or abstained experiment")
            continue
        if proposals.selection is None or cell.failure is not None:
            raise ValueError("campaign experiment has no selected population")
        experiment_spec = CheckpointExperimentSpec(
            selection=proposals.selection, critic=spec.critics[cell.critic]
        )
        expected = experiment_spec.identity(revision) + "-result"
        if artifact_input(project.artifacts.read(expected).manifest) != cell.experiment:
            raise ValueError("campaign experiment differs from its frozen selector/critic pair")
        report = saved_checkpoint_experiment(project, experiment_spec, code_revision=revision)
        if report is None:
            raise ValueError("campaign completed experiment is missing")
        measured.append(
            {
                "selector": cell.selector,
                "critic": cell.critic,
                "unmatched": report.unmatched,
                "metrics": report.metrics,
            }
        )
    return {
        "initial_attempts": source_result.metrics(),
        "planned_role_pairs": len(cells),
        "completed_experiments": len(measured),
        "unavailable_or_abstained": len(cells) - len(measured),
        "experiments": measured,
        "failures": [
            {
                "selector": cell.selector,
                "critic": cell.critic,
                "code": cell.failure,
                "reason": cell.reason,
                "selector_artifact_prefix": _selector(spec, cell.selector, source).identity(
                    revision
                ),
                "guidance": (
                    "Inspect the saved selector request/response and matched-plan receipts. "
                    "Missing responses remain unknown; malformed or out-of-pool choices are "
                    "retained, not redrawn. Source limits and abstentions are not task failures."
                ),
            }
            for cell in cells
            if cell.failure is not None
        ],
        "maximum_actor_selector_critic_http_attempts": spec.attempt_bound(),
        "resource_scope": "simulator/embedding/reader calls retain their frozen runtime ceiling",
    }


def saved_continuation_campaign(
    project: ProjectStore, spec: ContinuationCampaignSpec, *, code_revision: str
) -> ContinuationCampaign | None:
    """Recompute every completed stage without running missing children or calling providers."""
    store, identity = project.artifacts, spec.identity(code_revision)
    if not store.exists(identity + "-result"):
        return None
    report = ContinuationCampaign.model_validate_json(
        store.read_bytes(identity + "-result", "report.json")
    )
    planned = [(i, j) for i in range(len(spec.selectors)) for j in range(len(spec.critics))]
    if (
        report.spec != spec
        or [(c.selector, c.critic) for c in report.cells] != planned
        or ContinuationCampaignSpec.model_validate_json(store.read_bytes(identity, "spec.json"))
        != spec
        or store.read(identity).manifest.inputs != (spec.initial.study.checkpoint,)
    ):
        raise ValueError("campaign changed its original population or configuration")
    inputs = {artifact_input(store.read(identity).manifest), report.source}
    for index, cell in enumerate(report.cells):
        ref = artifact_input(store.read(f"{identity}-cell-{index}").manifest)
        if CampaignCell.model_validate_json(
            store.read_bytes(ref.artifact_id, "cell.json")
        ) != cell or store.read(ref.artifact_id).manifest.inputs != _cell_inputs(
            artifact_input(store.read(identity).manifest), report.source, cell
        ):
            raise ValueError("campaign report differs from its saved cell")
        inputs.add(ref)
    if set(
        store.read(identity + "-result").manifest.inputs
    ) != inputs or report.metrics != _metrics(
        project, spec, report.source, report.cells, code_revision
    ):
        raise ValueError("campaign metrics or dependencies differ from retained evidence")
    return report


def campaign_requires_execution(
    project: ProjectStore, spec: ContinuationCampaignSpec, *, code_revision: str
) -> bool:
    """Distinguish new child work from publication of already committed cells."""
    store, identity = project.artifacts, spec.identity(code_revision)
    if saved_continuation_campaign(project, spec, code_revision=code_revision) is not None:
        return False
    if not store.exists(identity):
        return True
    if ContinuationCampaignSpec.model_validate_json(
        store.read_bytes(identity, "spec.json")
    ) != spec or store.read(identity).manifest.inputs != (spec.initial.study.checkpoint,):
        raise ValueError("campaign registration differs from its frozen population")
    if saved_continuation_job(project, spec.initial, code_revision=code_revision) is None:
        return True
    source = continuation_result_reference(project, spec.initial, code_revision=code_revision)
    registration = artifact_input(store.read(identity).manifest)
    cells = []
    for i in range(len(spec.selectors)):
        for j in range(len(spec.critics)):
            key = f"{identity}-cell-{i * len(spec.critics) + j}"
            if not store.exists(key):
                return True
            cell = CampaignCell.model_validate_json(store.read_bytes(key, "cell.json"))
            if (cell.selector, cell.critic) != (i, j) or store.read(
                key
            ).manifest.inputs != _cell_inputs(registration, source, cell):
                raise ValueError("campaign cell differs from its registered role or dependencies")
            cells.append(cell)
    _metrics(project, spec, source, tuple(cells), code_revision)
    return False


def run_continuation_campaign(
    project: ProjectStore, spec: ContinuationCampaignSpec, *, code_revision: str
) -> ContinuationCampaign:
    """Execute existing owners; reuse saved draws and never silently reroll unknowns."""
    spec = ContinuationCampaignSpec.model_validate_json(spec.model_dump_json())
    store, identity = project.artifacts, spec.identity(code_revision)
    with CoordinationLock(store.project_directory / (identity + ".lock"), timeout=0):
        saved = saved_continuation_campaign(project, spec, code_revision=code_revision)
        if saved is not None:
            return saved
        if not store.exists(identity):
            preflight_continuation_campaign(project, spec)
            store.write_json(
                artifact_id=identity,
                artifact_type="continuation-campaign-registration",
                envelope=ArtifactEnvelope(
                    schema_version=1,
                    created_at=datetime.now(UTC),
                    code_revision=code_revision,
                    inputs=(spec.initial.study.checkpoint,),
                ),
                files={"spec.json": spec},
            )
        elif ContinuationCampaignSpec.model_validate_json(
            store.read_bytes(identity, "spec.json")
        ) != spec or store.read(identity).manifest.inputs != (spec.initial.study.checkpoint,):
            raise ValueError("campaign registration changed; inspect its frozen source population")
        registration = artifact_input(store.read(identity).manifest)
        source_result = saved_continuation_job(project, spec.initial, code_revision=code_revision)
        if source_result is None:
            execute_continuation_job(project, spec.initial, code_revision=code_revision)
        source = continuation_result_reference(project, spec.initial, code_revision=code_revision)
        cells, dependencies = [], {registration, source}
        for i in range(len(spec.selectors)):
            for j in range(len(spec.critics)):
                key = f"{identity}-cell-{i * len(spec.critics) + j}"
                if store.exists(key):
                    cell = CampaignCell.model_validate_json(store.read_bytes(key, "cell.json"))
                    if (cell.selector, cell.critic) != (i, j) or store.read(
                        key
                    ).manifest.inputs != _cell_inputs(registration, source, cell):
                        raise ValueError("campaign saved cell has changed roles or dependencies")
                else:
                    proposals_ref = experiment_ref = reason = None
                    failure = "SelectorUnavailable"
                    try:
                        selector_spec = _selector(spec, i, source)
                        proposals = propose_checkpoints(
                            project, selector_spec, code_revision=code_revision
                        )
                        if proposals is None:
                            raise RuntimeError(
                                "checkpoint proposals returned no report outside a dry run"
                            )
                        proposals_ref = artifact_input(
                            store.read(selector_spec.identity(code_revision)).manifest
                        )
                        if proposals.selection is None:
                            failure = "SelectorAbstained"
                        else:
                            failure = "ExperimentUnavailable"
                            experiment = CheckpointExperimentSpec(
                                selection=proposals.selection, critic=spec.critics[j]
                            )
                            run_checkpoint_experiment(
                                project, experiment, code_revision=code_revision
                            )
                            experiment_ref = artifact_input(
                                store.read(experiment.identity(code_revision) + "-result").manifest
                            )
                            failure = None
                    except Timeout:
                        raise  # Busy work remains pending; the owning process may still finish it.
                    except (ValueError, RuntimeError, OSError) as error:
                        reason = (type(error).__name__ + ": " + redact_secret_text(str(error))[0])[
                            :2000
                        ]
                    cell = CampaignCell(
                        selector=i,
                        critic=j,
                        proposals=proposals_ref,
                        experiment=experiment_ref,
                        failure=failure,
                        reason=reason,
                    )
                    store.write_json(
                        artifact_id=key,
                        artifact_type="continuation-campaign-cell",
                        envelope=ArtifactEnvelope(
                            schema_version=1,
                            created_at=datetime.now(UTC),
                            code_revision=code_revision,
                            inputs=_cell_inputs(registration, source, cell),
                        ),
                        files={"cell.json": cell},
                    )
                cells.append(cell)
                dependencies.add(artifact_input(store.read(key).manifest))
        report = ContinuationCampaign(
            spec=spec,
            source=source,
            cells=tuple(cells),
            metrics=_metrics(project, spec, source, tuple(cells), code_revision),
        )
        store.write_json(
            artifact_id=identity + "-result",
            artifact_type="continuation-campaign",
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=code_revision,
                inputs=tuple(sorted(dependencies, key=lambda r: r.artifact_id)),
            ),
            files={"report.json": report},
        )
        return report
