"""Wire one decision-variance front-door run: revision, spec checks, dependencies and spend.

This is what the operator scripts did by hand (``cluster_envs.py`` with ``run_code.sh``), owned
by the CLI so ``exp variance run`` is the whole demo:

- the code revision is the checkout's git ``HEAD``, refused when tracked files are modified (the
  revision would not name the code that runs); it keys the plan digest and the preparation;
- the spec is refused unless every decision (build, rule memory, runtime, TraceML, and the
  tag-banks and state-rag slots) is answered by Clef, no alias is the retired ``jev``, and no
  ``astra_alias`` is set (every pipeline role is open-weight);
- the dependencies carry every configured slot's executor, the rule-memory decisions client
  (the spec's Clef alias over the project decision cache), the ``exp build`` builder, the
  phase announcer and progress hooks; pipeline stage executors are the owner-bound defaults,
  wrapped only to report each stage's start and finish (planning and identities delegate
  unchanged, so the plan digest is the one the scripts produced);
- after consent the operation ``front-door-cluster-<digest16>`` gets the scripts' ceiling: the
  priced bound capped at ``ceiling_usd`` dollars and ``real_attempts`` real physical attempts,
  while the front door's phase ledger admits phases by their worst case up to
  ``admission_attempts``.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from exp.common.core.artifacts import JsonObject
from exp.common.progress import ProgressHook, report
from exp.common.project import ProjectStore
from exp.optimize.evaluation.variance.frontdoor import (
    FrontDoorApproval,
    FrontDoorDependencies,
    FrontDoorPlan,
    FrontDoorSpec,
    configured_slots,
)
from exp.optimize.evaluation.variance.pipeline import DEFAULT_EXECUTORS, StageExecutor
from exp.optimize.evaluation.variance.pipeline_spec import (
    CostLine,
    DecisionVarianceSpec,
    StageContext,
    StageName,
    StageRun,
)

DEFAULT_CEILING_USD = 250.0
"""Hard real-spend cap of the operation (the plan prices each Clef call at its worst case)."""

DEFAULT_REAL_ATTEMPTS = 1_000_000
"""Real physical attempts allowed per run (Raghav, 2026-11-09), enforced by the spend ceiling."""

DEFAULT_ADMISSION_ATTEMPTS = 2_000_000_000
"""Phase ledger allowance: phases are admitted by worst-case bounds (a start is ~2.6M)."""

RULE_MEMORY_PURPOSE = "bootstrap-rule-memory"
RULE_MEMORY_ATTEMPTS = 100_000

OPERATION_PREFIX = "front-door-cluster-"
"""Spend operation name of one consented plan, shared with the operator scripts' runs."""

CLEF_SLOTS = ("tag-banks", "state-rag")
"""Slots whose decisions must be answered by Clef."""


class SpecRefused(ValueError):
    """The front-door spec breaks a rule of this pipeline (Clef decisions, open-weight roles)."""


def operation_name(digest: str) -> str:
    """The spend operation of one consented plan."""
    return OPERATION_PREFIX + digest[:16]


def source_revision(source: Path | None = None) -> str:
    """Return the checkout's git ``HEAD``, refusing modified tracked files.

    Args:
        source: A directory inside the checkout; defaults to the one holding this ``exp``.

    Raises:
        ValueError: The source is not a git checkout, or tracked files are modified.
    """
    import exp

    directory = source or Path(exp.__file__).resolve().parents[1]

    def git(*args: str) -> str:
        """Run one git query in the checkout."""
        return subprocess.run(  # noqa: S603 - fixed argument vector
            ["git", *args], capture_output=True, text=True, check=True, cwd=directory
        ).stdout.strip()

    try:
        dirty = git("status", "--porcelain", "--untracked-files=no")
        revision = git("rev-parse", "HEAD")
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(
            f"{directory} is not a git checkout; pass --code-revision to name the code"
        ) from error
    if dirty:
        raise ValueError(
            f"{directory} has uncommitted changes to tracked files; the code revision would "
            "not name the code that runs. Commit them, or pass --code-revision explicitly"
        )
    return revision


def decision_aliases(spec: FrontDoorSpec) -> dict[str, str | None]:
    """Every decision alias the spec names, by where it is set."""
    bootstrap = spec.preparation.bootstrap
    aliases: dict[str, str | None] = {
        "build.decision_alias": spec.preparation.build.spec.decision_alias,
        "runtime.jev_alias": bootstrap.runtime.jev_alias,
        "pipeline.traceml.decision_alias": spec.pipeline.traceml.decision_alias,
    }
    if bootstrap.rules is not None and bootstrap.rules.memory is not None:
        aliases["rules.memory.decision_alias"] = bootstrap.rules.memory.decision_alias
    slots = configured_slots(spec)
    for name in CLEF_SLOTS:
        if name in slots:
            value = slots[name].get("decision_alias")
            aliases[f"slots.{name}.decision_alias"] = None if value is None else str(value)
    return aliases


def check_spec(spec: FrontDoorSpec) -> dict[str, str | None]:
    """Refuse a spec that breaks the pipeline's rules; return its decision aliases.

    Raises:
        SpecRefused: An ``astra_alias`` is set, an alias is ``jev``, or a decision is not Clef.
    """
    aliases = decision_aliases(spec)
    check_aliases(aliases, astra_alias=spec.astra_alias, text=spec.model_dump_json())
    return aliases


def check_aliases(aliases: dict[str, str | None], *, astra_alias: str | None, text: str) -> None:
    """Refuse an Astra role, the retired ``jev`` alias and any decision not answered by Clef.

    Args:
        aliases: Every decision alias, by where it is set.
        astra_alias: The spec's ``astra_alias``.
        text: The spec's JSON.

    Raises:
        SpecRefused: The first rule broken.
    """
    if astra_alias is not None:
        raise SpecRefused(
            f"the spec declares astra_alias {astra_alias!r}; every pipeline role must be "
            "open-weight, so remove it"
        )
    if '"jev"' in text:
        raise SpecRefused("the spec still names the retired alias jev")
    wrong = {where: alias for where, alias in aliases.items() if alias != "clef"}
    if wrong:
        named = ", ".join(f"{where}={alias}" for where, alias in sorted(wrong.items()))
        raise SpecRefused(f"every decision must be answered by clef; these are not: {named}")


def load_spec(path: Path) -> FrontDoorSpec:
    """Read and validate one front-door spec file."""
    return FrontDoorSpec.model_validate_json(path.read_bytes())


def slot_names(spec: FrontDoorSpec) -> list[str]:
    """The configured slots, in their fixed order."""
    return [str(name) for name in configured_slots(spec)]


def journal_name(project: ProjectStore, spec: FrontDoorSpec) -> str:
    """The registration's run directory name, ``front-door-<spec16>``."""
    from exp.optimize.evaluation.variance.frontdoor import run_directory

    return run_directory(project, spec).name


def priced_bound(plan: FrontDoorPlan) -> float:
    """Every stage's priced cost lines, as the scripts summed them."""
    return sum(line.usd or 0.0 for row in plan.stages for line in row.costs)


@dataclass(frozen=True)
class ReportingExecutor:
    """A pipeline stage executor that reports its start and finish; everything else delegates.

    Attributes:
        inner: The owner-bound executor.
        stage: Its stage name.
        hook: Receives ``pipeline <task>`` events with the stage as detail and counts in stages.
    """

    inner: StageExecutor
    stage: StageName
    hook: ProgressHook

    def keys(self, project: ProjectStore, spec: DecisionVarianceSpec) -> JsonObject:
        """Delegate: the identity is the owner's."""
        return self.inner.keys(project, spec)

    def plan(
        self, project: ProjectStore, spec: DecisionVarianceSpec, code_revision: str
    ) -> tuple[tuple[CostLine, ...], tuple[str, ...]]:
        """Delegate: the plan is the owner's."""
        return self.inner.plan(project, spec, code_revision)

    def run(self, context: StageContext) -> StageRun:
        """Report the start, run the owner, report the finish."""
        bootstrap = context.spec.mining.bootstrap
        task = bootstrap.task_id if bootstrap is not None else context.identity
        stages = tuple(DEFAULT_EXECUTORS)
        index = stages.index(self.stage)
        name = f"pipeline {task}"
        report(self.hook, name, completed=index, total=len(stages), detail=self.stage)
        result = self.inner.run(context)
        report(self.hook, name, completed=index + 1, total=len(stages), detail=self.stage)
        return result


def reporting_executors(hook: ProgressHook) -> dict[StageName, StageExecutor]:
    """The owner-bound pipeline executors, each wrapped to report its stage."""
    return {
        stage: ReportingExecutor(executor, stage, hook)
        for stage, executor in DEFAULT_EXECUTORS.items()
    }


def front_door_dependencies(
    project: ProjectStore,
    spec: FrontDoorSpec,
    revision: str,
    *,
    root: Path,
    announce: Callable[[tuple[str, ...]], None] | None = None,
    progress: ProgressHook | None = None,
) -> FrontDoorDependencies:
    """Every collaborator the run needs, as the operator scripts bound them.

    Args:
        project: The project store.
        spec: The checked spec.
        revision: The code revision.
        root: The EXP root, for the ``exp build`` builder.
        announce: Shown each phase's preview before it is admitted.
        progress: Observer of phases and pipeline stages.
    """
    from exp.cli.evaluation.front_door_action import ExpBuild
    from exp.optimize.evaluation.variance.bootstrap_plan import BootstrapDependencies
    from exp.optimize.evaluation.variance.frontdoor_executors import decision_tagger
    from exp.optimize.evaluation.variance.frontdoor_slots import slot_executor

    rules = spec.preparation.bootstrap.rules
    memory = (
        None
        if rules is None or rules.memory is None
        else decision_tagger(
            project,
            rules.memory.decision_alias,
            RULE_MEMORY_PURPOSE,
            RULE_MEMORY_ATTEMPTS,
            revision,
        )
    )
    return FrontDoorDependencies(
        builder=ExpBuild(root=root, project=project.paths.project_id),
        bootstrap=BootstrapDependencies(decisions=memory),
        executors=None if progress is None else reporting_executors(progress),
        slots={name: slot_executor(name) for name in configured_slots(spec)},
        announce=announce,
        progress=progress,
    )


def matches(consent: str, digest: str) -> bool:
    """Whether a typed consent names this plan: the whole digest or at least its first 12."""
    consent = consent.strip().lower()
    return len(consent) >= 12 and digest.startswith(consent)


def install_ceiling(
    project: ProjectStore,
    plan: FrontDoorPlan,
    *,
    ceiling_usd: float,
    real_attempts: int,
    command: str,
) -> tuple[str, Path]:
    """Authorize and install the operation's spend ceiling; call inside an operation scope.

    Returns:
        The operation name and its saved spend receipt.
    """
    from exp.optimize.evaluation.spending import operation_spend_ceiling, save_spend_receipt
    from exp.runtime.models.spend_ceiling import install_spend_ceiling

    operation = operation_name(plan.digest)
    ceiling = operation_spend_ceiling(project, operation)
    ceiling.authorize(plan.digest, ceiling_usd=min(priced_bound(plan), ceiling_usd))
    ceiling.authorize_policy(ceiling_usd=None, unpriced_attempts=real_attempts)
    install_spend_ceiling(ceiling)
    return operation, save_spend_receipt(project, operation, ceiling, command)


def approval(plan: FrontDoorPlan, admission_attempts: int) -> FrontDoorApproval:
    """The front door's consent to this exact plan, with the phase ledger's allowance."""
    return FrontDoorApproval(
        plan_sha256=plan.digest,
        ceiling_usd=priced_bound(plan),
        unpriced_attempts=admission_attempts,
    )
