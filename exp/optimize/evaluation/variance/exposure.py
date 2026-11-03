"""Which data each stage may see, frozen before any tag is bought or any trace is analyzed.

A trace's leakage lineage decides its role, and every role is fixed before tagging, so a tag
predicate can never move evidence from one role to another:

- **final_evaluation**: every held-out lineage. Nothing in this pipeline reads it: it is not
  tagged, mined, continued or used to author rubrics, and continuation preflight already refuses
  held-out checkpoints.
- **world_learning**: fit lineages the world model and its learned rules are built from. Retrieval
  draws from the same role and may overlap construction deliberately.
- **mining_probes**: fit lineages held out from world learning and used to choose points, author
  rubrics and generate variants. Once their results guide those choices they are development
  data, which is why they are not the final evaluation.

Fit lineages are split between world learning and probes by a seeded hash of the lineage identity,
so the split is reproducible and cannot be tuned after seeing results.

Every success estimate also has an exact meaning: the probability that one actor policy, under one
harness, budget, sampling setting, world version and verifier, succeeds from one checkpoint. An
:class:`EstimateMeaning` records all of it, so estimates are compared and trained on only within
the meaning they were made under.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Literal

from pydantic import Field

from exp.common.core.artifacts import ArtifactInput, ContractModel, sha256_json
from exp.common.models import ModelSnapshot
from exp.optimize.evaluation.continuations.job import ContinuationJob
from exp.simulation.continuations import ContinuationCheckpoint
from exp.simulation.mining.bindings import TaskTraceLineageBinding
from exp.simulation.mining.tag_selection import TagSelection

Role = Literal["final_evaluation", "world_learning", "mining_probes"]
"""The stage a lineage's traces are reserved for."""


class ExposurePlan(ContractModel):
    """The frozen role of every lineage in one task set.

    Attributes:
        bindings: The task set's lineage bindings this plan was derived from.
        seed: Seed of the fit-lineage split.
        probe_fraction: Share of fit lineages reserved as mining probes, in (0, 1).
        roles: Role of each lineage identity.
        traces: Lineage identity of each trace.
    """

    bindings: ArtifactInput
    seed: int = Field(ge=0)
    probe_fraction: float = Field(gt=0.0, lt=1.0)
    roles: dict[str, Role]
    traces: dict[str, str]

    def role_of(self, trace_id: str) -> Role:
        """Return the role of one trace.

        Args:
            trace_id: Canonical source trace identity.

        Returns:
            Its lineage's role.

        Raises:
            ValueError: The trace is not in the plan's task set.
        """
        lineage = self.traces.get(trace_id)
        if lineage is None:
            raise ValueError(f"trace {trace_id} has no lineage in this exposure plan")
        return self.roles[lineage]

    def require(self, trace_ids: Iterable[str], role: Role) -> None:
        """Refuse any trace outside one role.

        Args:
            trace_ids: Traces a stage is about to read.
            role: The role the stage may read.

        Raises:
            ValueError: A trace belongs to another role.
        """
        outside = sorted(t for t in set(trace_ids) if self.role_of(t) != role)
        if outside:
            raise ValueError(
                f"{len(outside)} traces are not reserved for {role}, for example {outside[0]}; "
                "a stage may read only its own role"
            )

    def require_lineage(self, lineage_id: str, role: Role) -> None:
        """Refuse a lineage outside one role.

        Args:
            lineage_id: Leakage lineage of a task about to be used.
            role: The role the stage may read.

        Raises:
            ValueError: The lineage is unknown or reserved for another role.
        """
        actual = self.roles.get(lineage_id)
        if actual != role:
            raise ValueError(
                f"lineage {lineage_id} is reserved for {actual or 'nothing in this plan'}, "
                f"not {role}"
            )

    def restrict(self, selection: TagSelection, role: Role) -> TagSelection:
        """Check that a tag selection stayed inside one role.

        Args:
            selection: A selection made over traces the role may read.
            role: The role the selection is for.

        Returns:
            The same selection.

        Raises:
            ValueError: The selection reaches into another role.
        """
        self.require(selection.trace_ids, role)
        return selection


def plan_exposure(
    bindings: ArtifactInput,
    rows: Iterable[TaskTraceLineageBinding],
    *,
    seed: int = 0,
    probe_fraction: float = 0.3,
) -> ExposurePlan:
    """Assign every lineage a role before any tagging or analysis.

    Args:
        bindings: Saved lineage bindings of the task set.
        rows: Those bindings' rows.
        seed: Seed of the fit-lineage split.
        probe_fraction: Share of fit lineages reserved as mining probes.

    Returns:
        The frozen plan.

    Raises:
        ValueError: A lineage appears in both partitions, or the split leaves a role empty.
    """
    partitions: dict[str, set[str]] = {}
    traces: dict[str, str] = {}
    for row in rows:
        partitions.setdefault(row.lineage_id, set()).add(row.partition)
        traces[row.trace_id] = row.lineage_id
    mixed = sorted(lineage for lineage, seen in partitions.items() if len(seen) > 1)
    if mixed:
        raise ValueError(f"lineage {mixed[0]} spans both fit and held-out partitions")
    roles: dict[str, Role] = {}
    for lineage, seen in partitions.items():
        if "held_out" in seen:
            roles[lineage] = "final_evaluation"
        else:
            roles[lineage] = (
                "mining_probes" if _unit(seed, lineage) < probe_fraction else "world_learning"
            )
    for role in ("world_learning", "mining_probes"):
        if role not in roles.values():
            raise ValueError(f"the fit split leaves no lineage for {role}; adjust probe_fraction")
    return ExposurePlan(
        bindings=bindings, seed=seed, probe_fraction=probe_fraction, roles=roles, traces=traces
    )


class EstimateMeaning(ContractModel):
    """Everything a success probability from one checkpoint is conditional on.

    Attributes:
        checkpoint: The paired actor/world checkpoint.
        actor_model: Actor policy snapshot.
        harness_sha256: Identity of the actor runtime and its trusted environment code.
        system_prompt_sha256: Identity of the actor's system prompt.
        maximum_model_calls: The actor's total call allowance.
        remaining_calls: Calls left at the checkpoint.
        temperature: Continuation sampling temperature.
        output_tokens: Output allowance per actor call.
        reasoning_effort: Requested reasoning effort, or None for the model default.
        world_sha256: Identity of the world runtime configuration, including the world model.
        verifier_sha256: Identity of the terminal verifier.
        probe_sha256: Identity of the evaluator state projection.
    """

    checkpoint: ArtifactInput
    actor_model: ModelSnapshot
    harness_sha256: str
    system_prompt_sha256: str
    maximum_model_calls: int
    remaining_calls: int
    temperature: float
    output_tokens: int
    reasoning_effort: str | None
    world_sha256: str
    verifier_sha256: str
    probe_sha256: str


def estimate_meaning(job: ContinuationJob, checkpoint: ContinuationCheckpoint) -> EstimateMeaning:
    """Record what a success estimate from a job's checkpoint is conditional on.

    Args:
        job: The continuation job that measures success from the checkpoint.
        checkpoint: The loaded checkpoint the job starts from.

    Returns:
        The estimate's frozen meaning.
    """
    study = job.study
    return EstimateMeaning(
        checkpoint=study.checkpoint,
        actor_model=study.actor_model,
        harness_sha256=job.runtime_sha256,
        system_prompt_sha256=sha256_json(job.system_prompt),
        maximum_model_calls=job.maximum_model_calls,
        remaining_calls=max(0, job.maximum_model_calls - checkpoint.actor.calls),
        temperature=study.temperature,
        output_tokens=study.maximum_output_tokens,
        reasoning_effort=study.reasoning_effort,
        world_sha256=sha256_json(job.configuration),
        verifier_sha256=sha256_json(study.verifier),
        probe_sha256=study.probe_sha256,
    )


def _unit(seed: int, lineage: str) -> float:
    """Map a lineage to a reproducible number in [0, 1)."""
    digest = hashlib.sha256(f"variance-exposure-v1\0{seed}\0{lineage}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64
