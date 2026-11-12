"""The front door's registration, plan, consent and report contracts.

:class:`FrontDoorSpec` is one registration (a trace source and every downstream setting);
:class:`FrontDoorPlan` lists every stage credential-free; :class:`FrontDoorApproval` is the one
consent for the whole run; :class:`FrontDoorReport` is what a run did. The sequence that plans
and runs them lives in :mod:`~exp.optimize.evaluation.variance.frontdoor`.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from exp.common.core.artifacts import ArtifactInput, ContractModel, JsonObject, Sha256, sha256_json
from exp.optimize.evaluation.variance.astra_role import refuse_astra_outside_its_roles
from exp.optimize.evaluation.variance.bootstrap_contracts import BootstrapBinding
from exp.optimize.evaluation.variance.bootstrap_plan import (
    BootstrapPreparationReport,
    BootstrapPreparationSpec,
)
from exp.optimize.evaluation.variance.frontdoor_executors import PIPELINE_SETTINGS, SCOPED_SLOTS
from exp.optimize.evaluation.variance.frontdoor_slots import SlotName
from exp.optimize.evaluation.variance.frontdoor_source import FRONT_DOOR_PROTOCOL, TraceSource
from exp.optimize.evaluation.variance.frontdoor_states import StageState
from exp.optimize.evaluation.variance.pipeline_spec import (
    AuthoringSettings,
    CostLine,
    DecisionVarianceSpec,
    MiningSettings,
    RecordsSettings,
    RulesSettings,
    TraceMLSettings,
)


class PipelineTemplate(ContractModel):
    """Every pipeline setting except the build and the start, which the front door supplies.

    Attributes:
        rules: Rules-stage settings, or None for the preparation's own rules registration. A
            bootstrapped pipeline reuses the family's frozen rules with no call either way.
        traceml: TraceML settings.
        mining: Error-mining settings with neither ``job`` nor ``bootstrap``: each registered
            start fills ``bootstrap``.
        authoring: Family settings.
        records: Records settings.
    """

    rules: RulesSettings | None = None
    traceml: TraceMLSettings
    mining: MiningSettings
    authoring: AuthoringSettings
    records: RecordsSettings

    @model_validator(mode="after")
    def _unbound(self) -> Self:
        """Refuse a hand-registered job or start; the preparation registers them."""
        if self.mining.job is not None or self.mining.bootstrap is not None:
            raise ValueError(
                "the front door binds every start its preparation registers; leave "
                "pipeline.mining.job and pipeline.mining.bootstrap empty"
            )
        return self


class FrontDoorSpec(ContractModel):
    """One front-door registration: a trace source and every downstream setting.

    Attributes:
        traces: The trace source, or None to use the project's completed build as it is.
        preparation: Variance build, environment families, bootstrap roles and finite limits.
        pipeline: Pipeline settings applied to every registered start.
        slots: Settings of the slot stages to run, by slot; each needs a registered executor.
        astra_alias: The one approved closed alias, for the build-time tag roles only (the
            tag-bank and state-tag authors and the tag-monitor reviewer at checkpoints 1 to 5);
            refused in every other role (:mod:`~.astra_role`). None keeps every role open-weight.
        save_partial: When some tasks failed but at least one start registered, save the
            partial preparation and run the registered starts' pipelines; the failed tasks
            retry on the next run. False stops before any pipeline instead.
        protocol: Front-door protocol.
    """

    traces: TraceSource | None = None
    preparation: BootstrapPreparationSpec
    pipeline: PipelineTemplate
    slots: dict[SlotName, JsonObject] = Field(default_factory=dict)
    astra_alias: str | None = Field(
        default=None, min_length=1, exclude_if=lambda value: value is None
    )
    save_partial: bool = Field(default=True, exclude_if=lambda value: value)
    protocol: Literal["decision-variance-front-door-v1"] = FRONT_DOOR_PROTOCOL

    @model_validator(mode="after")
    def _rules(self) -> Self:
        """Require rules, each in-pipeline slot's template setting, and a scoped slot's tasks."""
        if self.pipeline.rules is None and self.preparation.bootstrap.rules is None:
            raise ValueError("give pipeline.rules when the preparation registers no rules")
        enabled = {
            "blast-radius": self.pipeline.mining.qualification is not None
            and self.pipeline.mining.qualification.blast is not None,
            "curriculum": self.pipeline.records.schedule,
            "dense-hybrid": self.pipeline.records.dense is None
            or self.pipeline.records.dense.hybrid is not None,
        }
        for name in self.slots:
            if name in enabled and not enabled[name]:
                raise ValueError(
                    f"slot {name} runs inside each pipeline; set {PIPELINE_SETTINGS[name][1]}"
                )
        scoped = sorted(
            name
            for name, settings in self.slots.items()
            if name in SCOPED_SLOTS and settings.get("scope", "all") != "all"
        )
        if scoped and not self.preparation.task_ids:
            raise ValueError(
                f"slots {', '.join(scoped)} are scoped to the run's tasks; give "
                "preparation.task_ids, whose families they cover"
            )
        refuse_astra_outside_its_roles(self.model_dump(mode="json"), self.astra_alias)
        return self

    def pipeline_spec(
        self, report: BootstrapPreparationReport, binding: BootstrapBinding
    ) -> DecisionVarianceSpec:
        """Return one registered start's pipeline registration.

        Args:
            report: The saved preparation.
            binding: One of its registered starts.

        Returns:
            The template with the preparation's saved build pinned and the start bound.
        """
        rules = self.pipeline.rules
        if rules is None:
            assert self.preparation.bootstrap.rules is not None
            rules = RulesSettings(spec=self.preparation.bootstrap.rules)
        return DecisionVarianceSpec(
            build=self.preparation.build.model_copy(update={"saved_report": report.build_report}),
            rules=rules,
            traceml=self.pipeline.traceml,
            mining=self.pipeline.mining.model_copy(update={"bootstrap": binding}),
            authoring=self.pipeline.authoring,
            records=self.pipeline.records,
        )


class FrontDoorStage(ContractModel):
    """One front-door stage before it runs.

    Attributes:
        stage: Its name (``project-build``, ``slot <name>``, ``bootstrap-preparation`` or
            ``pipeline <task>``).
        status: Saved, priced and runnable, or waiting on an earlier stage.
        costs: Upper-bound cost inventory of a pending stage.
        lines: What it does, or did.
        notes: Work without a known bound yet, and what will bound it.
    """

    stage: str
    status: StageState
    costs: tuple[CostLine, ...] = ()
    lines: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def known_usd(self) -> float:
        """Sum of the priced components."""
        return sum(row.usd for row in self.costs if row.usd is not None)

    @property
    def unpriced(self) -> tuple[CostLine, ...]:
        """Components with physical requests but no known price."""
        return tuple(row for row in self.costs if row.usd is None and row.attempts > 0)


class FrontDoorPlan(ContractModel):
    """Every front-door stage, credential-free.

    Attributes:
        spec_sha256: Digest of the registration.
        code_revision: Producing source identity.
        stages: Every stage in order.
    """

    spec_sha256: Sha256
    code_revision: str
    stages: tuple[FrontDoorStage, ...]

    @property
    def digest(self) -> Sha256:
        """Digest a consent binds."""
        return sha256_json(self)

    @property
    def known_usd(self) -> float:
        """Priced upper bound of every pending stage."""
        return sum(row.known_usd for row in self.stages if row.status == StageState.PLANNED)

    @property
    def unpriced(self) -> tuple[CostLine, ...]:
        """Unpriced components of every pending stage, named by stage."""
        return tuple(
            line.model_copy(update={"name": f"{row.stage}: {line.name}"})
            for row in self.stages
            if row.status == StageState.PLANNED
            for line in row.unpriced
        )

    @property
    def complete(self) -> bool:
        """Whether every stage is saved, so a run replays without consent or calls."""
        return all(row.status == StageState.DONE for row in self.stages)


class FrontDoorApproval(ContractModel):
    """One consent for the whole front-door run.

    Attributes:
        plan_sha256: The consented plan.
        ceiling_usd: Dollar ceiling across every phase, including phases priced later.
        unpriced_attempts: Physical attempts allowed to unpriced components across the run.
    """

    plan_sha256: Sha256
    ceiling_usd: float = Field(ge=0)
    unpriced_attempts: int = Field(default=0, ge=0)


class PipelineOutcome(ContractModel):
    """One registered start's pipeline.

    Attributes:
        task_id: The start's task.
        binding: Its registered start.
        run: The saved run registration, once it ran.
        complete: Whether every stage is saved.
        stopped: Why it stopped, when it did.
        stages: Each finished stage's saved record, by stage.
    """

    task_id: str
    binding: BootstrapBinding
    run: ArtifactInput | None = None
    complete: bool = False
    stopped: str | None = None
    stages: dict[str, ArtifactInput] = Field(default_factory=dict)


class FrontDoorReport(ContractModel):
    """What one front-door run did.

    Attributes:
        plan: The plan it ran under.
        source: The saved trace-source record, when a trace source was given.
        preparation: The saved preparation, once it completed.
        bindings: Every registered start.
        failed_tasks: Starts whose preparation failed and that a rerun retries.
        refused_tasks: Starts refused with a saved diagnostic.
        pipelines: One outcome per registered start that was reached.
        slots: Saved result of every slot stage that ran.
        stopped: Why the run stopped early, when it did.
    """

    plan: FrontDoorPlan
    source: ArtifactInput | None = None
    preparation: ArtifactInput | None = None
    bindings: tuple[BootstrapBinding, ...] = ()
    failed_tasks: tuple[str, ...] = ()
    refused_tasks: tuple[str, ...] = ()
    pipelines: tuple[PipelineOutcome, ...] = ()
    slots: dict[SlotName, ArtifactInput] = Field(default_factory=dict)
    stopped: str | None = None

    @property
    def complete(self) -> bool:
        """Whether every start's pipeline is complete and nothing stopped or failed."""
        return (
            self.stopped is None
            and not self.failed_tasks
            and len(self.pipelines) == len(self.bindings)
            and all(row.complete for row in self.pipelines)
        )
