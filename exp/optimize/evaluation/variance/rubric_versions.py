"""Versions of the two-part rubric: admitted items, each compiled into both rubric parts.

A training task carries a two-part rubric: the frozen LLM episode judge (the outcome reward,
:mod:`~.judge`) and the frozen decision-model dense tags run as hybrid dense (:mod:`~.dense_hybrid`,
vocabulary from :mod:`~.dense_vocabulary`). The evolving rubric keeps both parts ahead of a policy
learning to exploit them by adding :class:`RubricItem` rows between epochs. Each item is one
admitted reward-hacking signature compiled into:

- a dense decision question (family :data:`ITEM_FAMILY`), which the in-flow vocabulary turns into
  a decisive penalty that needs the cited obligation and keeps the legitimate look-alikes
  acceptable, so the decision model tags it on every step and the dense judge adjudicates it;
- an outcome-judge criterion (role ``required_correctness``), appended to the task's frozen
  rubric, so the episode judge must find it met before a pass.

A :class:`RubricVersion` is immutable: a revision makes a new version whose ``parent`` is the old
one's identity. A version rebinds every task's judge and dense definition
(:func:`versioned_judge`, :func:`versioned_dense`); both identities change with the version, so a
label (an assessment or a dense judgment) made under an earlier version is refused by the
unchanged judge-identity checks (:func:`~.judged.judged_episode`). :func:`rubric_provenance` is
the record a Harbor package keeps in its
:class:`~exp.optimize.evaluation.executable.task.MiningProvenance`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime

from pydantic import Field

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    Sha256,
    sha256_json,
    stable_id,
)
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.executable.task import (
    ExecutableTask,
    MiningProvenance,
    RubricProvenance,
)
from exp.optimize.evaluation.variance.dense_judge import DenseRewardJudge
from exp.optimize.evaluation.variance.dense_vocabulary import vocabulary
from exp.optimize.evaluation.variance.episodes import JudgeDefinition
from exp.optimize.evaluation.variance.judged import EpisodeJudge
from exp.simulation.mining.questions import Question, QuestionBank

RUBRIC_VERSION_PROTOCOL = "rubric-version-v1"
"""Version of the item layout and of how a version rebinds both rubric parts."""

ITEM_FAMILY = "rubric.negative"
"""Bank family of an item's dense question; ``.negative`` makes it a decisive cited penalty."""

ITEM_ROLE = "required_correctness"
"""Outcome-judge role of an item's criterion: a pass needs it established as met."""

VERSIONED_RUBRIC_TYPE = "episode-rubric"
"""Artifact type of a task's rubric rebound to a version (the episode judge's rubric type)."""


class JudgeCriterion(ContractModel):
    """An item compiled into one outcome-judge criterion.

    Attributes:
        criterion_id: Identity, unique in the rubric.
        question: What the judge must establish as met for a pass.
        fail_when: What, cited from the episode's events, makes it not met.
        acceptable_when: Legitimate look-alikes the judge must keep met.
    """

    criterion_id: str = Field(min_length=1, max_length=128, pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    question: str = Field(min_length=1, max_length=1500)
    fail_when: str = Field(min_length=1, max_length=1500)
    acceptable_when: tuple[str, ...] = Field(min_length=2, max_length=8)


class RubricItem(ContractModel):
    """One admitted reward-hacking signature, compiled into both rubric parts.

    Attributes:
        item_id: Identity, stable across versions.
        title: Short name.
        signature: The Added/Removed signature it was mined from.
        statement: The admitted claim about rollouts showing the signature.
        dense_question: The decision question tagged on every step.
        criterion: The outcome-judge criterion.
        admitted_epoch: Boundary epoch it was admitted at.
        supporting: Supporting rollouts.
        supporting_lineages: Their distinct lineages (at least two).
        evidence_refs: Cited ``<rollout>/<event>`` locators the compiler read.
        merged_signatures: Later signatures consolidation merged into it.
        quiet_epochs: Consecutive boundaries at which it did not discriminate.
        status: ``active`` or ``retired``.
        retired_epoch: Boundary epoch it was retired at.
    """

    item_id: str = Field(min_length=1, max_length=128, pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    title: str = Field(min_length=1, max_length=200)
    signature: str = Field(min_length=1, max_length=1000)
    statement: str = Field(min_length=1, max_length=2000)
    dense_question: Question
    criterion: JudgeCriterion
    admitted_epoch: int = Field(ge=1)
    supporting: tuple[str, ...] = Field(min_length=2)
    supporting_lineages: tuple[str, ...] = Field(min_length=2)
    evidence_refs: tuple[str, ...] = ()
    merged_signatures: tuple[str, ...] = ()
    quiet_epochs: int = Field(default=0, ge=0)
    status: str = Field(default="active", pattern=r"^(active|retired)$")
    retired_epoch: int | None = Field(default=None, ge=1)

    @property
    def identity(self) -> Sha256:
        """Digest of what the item asks; its bookkeeping (quiet epochs) is excluded."""
        return sha256_json(
            self.model_dump(mode="json", exclude={"quiet_epochs", "status", "retired_epoch"})
        )


class BankPin(ContractModel):
    """A tag bank a version's dense part is frozen from.

    Attributes:
        bank_id: The bank.
        version: Its published version.
        sha256: Digest of its exact questions.
    """

    bank_id: str
    version: int = Field(ge=1)
    sha256: Sha256

    @classmethod
    def of(cls, bank: QuestionBank) -> BankPin:
        """Pin one bank."""
        return cls(bank_id=bank.bank_id, version=bank.version, sha256=sha256_json(bank))


class RecallRecord(ContractModel):
    """Held-out known-hack recall of one version's two parts.

    Attributes:
        held_out_sha256: Identity of the held-out set.
        hacks: Known hacks graded.
        outcome_caught: Hacks the outcome judge failed.
        dense_caught: Hacks the dense part decisively penalized.
        outcome_unscored: Hacks the outcome judge left unscored.
        dense_unscored: Hacks the dense part left unavailable.
        caught_by: Per hack, which parts caught it.
    """

    held_out_sha256: Sha256
    hacks: int = Field(ge=1)
    outcome_caught: int = Field(ge=0)
    dense_caught: int = Field(ge=0)
    outcome_unscored: int = Field(default=0, ge=0)
    dense_unscored: int = Field(default=0, ge=0)
    caught_by: dict[str, tuple[str, ...]] = Field(default_factory=dict)

    @property
    def outcome_recall(self) -> float:
        """Share of known hacks the outcome judge failed; unscored hacks count as missed."""
        return self.outcome_caught / self.hacks

    @property
    def dense_recall(self) -> float:
        """Share of known hacks the dense part penalized; unavailable counts as missed."""
        return self.dense_caught / self.hacks

    def lowered_by(self, candidate: RecallRecord) -> tuple[str, ...]:
        """Return which recall the candidate lowers, empty when neither drops.

        Raises:
            ValueError: The records were measured on different held-out sets.
        """
        if candidate.held_out_sha256 != self.held_out_sha256:
            raise ValueError("recall records measured on different held-out sets")
        lowered = []
        if candidate.outcome_recall < self.outcome_recall:
            lowered.append("outcome")
        if candidate.dense_recall < self.dense_recall:
            lowered.append("dense")
        return tuple(lowered)


class RubricVersion(ContractModel):
    """One immutable version of the evolving rubric.

    Attributes:
        protocol: Version protocol.
        run_id: The training run it belongs to.
        version: Number, from one.
        epoch: First training epoch frozen under it.
        parent: Identity of the version it was revised from.
        banks: Banks its dense part is frozen from, in order.
        items: Every item ever admitted, active and retired, in admission order.
        recall: Held-out recall measured when it was published (not part of its identity).
    """

    protocol: str = RUBRIC_VERSION_PROTOCOL
    run_id: str = Field(min_length=1)
    version: int = Field(ge=1)
    epoch: int = Field(ge=1)
    parent: Sha256 | None = None
    banks: tuple[BankPin, ...] = Field(min_length=1)
    items: tuple[RubricItem, ...] = ()
    recall: RecallRecord | None = None

    @property
    def identity(self) -> Sha256:
        """Digest of the run, number, parent, banks and the questions each item asks."""
        return sha256_json(
            {
                "protocol": self.protocol,
                "run": self.run_id,
                "version": self.version,
                "parent": self.parent,
                "banks": [pin.model_dump(mode="json") for pin in self.banks],
                "items": [[row.identity, row.status] for row in self.items],
            }
        )

    def active(self) -> tuple[RubricItem, ...]:
        """Return the active items, in admission order."""
        return tuple(row for row in self.items if row.status == "active")


def item_bank(version: RubricVersion) -> QuestionBank | None:
    """Return the active items' dense questions as one bank, or None when there are none."""
    active = version.active()
    if not active:
        return None
    return QuestionBank(
        bank_id=f"rubric-items-{version.run_id[:48]}",
        version=version.version,
        questions=tuple(row.dense_question for row in active),
    )


def version_banks(version: RubricVersion, banks: Sequence[QuestionBank]) -> list[QuestionBank]:
    """Return the version's frozen banks plus its item bank, refusing a changed bank.

    Raises:
        ValueError: A bank differs from the pinned one or is missing.
    """
    pinned = {pin.bank_id: pin for pin in version.banks}
    given = {bank.bank_id: bank for bank in banks}
    if set(given) != set(pinned):
        raise ValueError(f"version {version.version} pins banks {sorted(pinned)}")
    for bank_id, pin in pinned.items():
        if BankPin.of(given[bank_id]) != pin:
            raise ValueError(f"bank {bank_id} changed since version {version.version} pinned it")
    ordered = [given[pin.bank_id] for pin in version.banks]
    extra = item_bank(version)
    return [*ordered, *([] if extra is None else [extra])]


def versioned_dense(
    version: RubricVersion, banks: Sequence[QuestionBank], base: DenseRewardJudge
) -> tuple[DenseRewardJudge, QuestionBank]:
    """Rebind a task's frozen dense judge to the version's vocabulary and step bank.

    Args:
        version: The rubric version.
        banks: The pinned banks.
        base: The task's frozen dense judge (bounds, profile, task and world digests).

    Returns:
        The dense judge whose tags are the version's vocabulary, and the step bank the decision
        model tags every step with.
    """
    tags, step = vocabulary(version_banks(version, banks))
    definition = base.definition.model_copy(update={"tags": tags})
    return base.model_copy(update={"definition": definition}), step


def versioned_judge(
    project: ProjectStore, version: RubricVersion, base: EpisodeJudge, *, code_revision: str
) -> EpisodeJudge:
    """Rebind a task's frozen episode judge to a rubric holding the version's item criteria.

    The base rubric's content is kept exactly; each active item's criterion is appended to its
    ``body.criteria`` with role ``required_correctness``, and ``rubric_version`` records the
    version. The new rubric is saved once and the judge definition binds its digest.

    Raises:
        ValueError: The base rubric is not a single JSON object with a criteria list, or an item
            criterion repeats a base criterion id.
    """
    store = project.artifacts
    manifest = store.read(base.definition.rubric.artifact_id).manifest
    if artifact_input(manifest) != base.definition.rubric or len(manifest.files) != 1:
        raise ValueError("the base judge rubric must be its referenced single-file artifact")
    file = manifest.files[0]
    content = json.loads(store.read_bytes(manifest.artifact_id, file.path))
    body = content.get("body") if isinstance(content, dict) else None
    rows = body.get("criteria") if isinstance(body, dict) else None
    if not isinstance(content, dict) or not isinstance(body, dict) or not isinstance(rows, list):
        raise ValueError("the base judge rubric must hold body.criteria")
    if not version.active():
        return base
    known = {row.get("criterion_id") for row in rows if isinstance(row, dict)}
    added: list[JsonObject] = []
    for item in version.active():
        criterion = item.criterion
        if criterion.criterion_id in known:
            raise ValueError(f"item criterion {criterion.criterion_id} repeats a base criterion")
        added.append(
            {
                "criterion_id": criterion.criterion_id,
                "role": ITEM_ROLE,
                "question": criterion.question,
                "fail_when": criterion.fail_when,
                "acceptable_when": list(criterion.acceptable_when),
                "source": f"rubric-evolution item {item.item_id} ({item.signature})",
            }
        )
    rebound: JsonObject = {
        **content,
        "body": {**body, "criteria": [*rows, *added]},
        "rubric_version": {"version": version.version, "sha256": version.identity},
    }
    identity = stable_id(
        "rubric-version-judge",
        {"base": base.definition.rubric.model_dump(mode="json"), "version": version.identity},
    )
    if not store.exists(identity):
        store.write_json(
            artifact_id=identity,
            artifact_type=VERSIONED_RUBRIC_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=code_revision,
                inputs=(base.definition.rubric,),
            ),
            files={file.path: rebound},
        )
    saved = store.read(identity).manifest
    definition: JudgeDefinition = base.definition.model_copy(
        update={"rubric": artifact_input(saved), "rubric_sha256": saved.files[0].sha256}
    )
    return base.model_copy(update={"definition": definition})


def rubric_provenance(version: RubricVersion) -> RubricProvenance:
    """Return the version record a Harbor package keeps in its mining provenance."""
    recall = version.recall
    return RubricProvenance(
        version=version.version,
        version_sha256=version.identity,
        epoch=version.epoch,
        parent_sha256=version.parent,
        items=tuple(row.identity for row in version.active()),
        retired=tuple(row.identity for row in version.items if row.status == "retired"),
        outcome_recall=None if recall is None else recall.outcome_recall,
        dense_recall=None if recall is None else recall.dense_recall,
        held_out_sha256=None if recall is None else recall.held_out_sha256,
    )


def bind_version(
    task: ExecutableTask, version: RubricVersion, judge: EpisodeJudge, dense: DenseRewardJudge
) -> ExecutableTask:
    """Bind a mined task to a version's two rubric parts and record the version in provenance.

    Args:
        task: The exported task; it must carry mining provenance.
        version: The rubric version.
        judge: The task's judge rebound to the version (:func:`versioned_judge`).
        dense: The task's dense judge rebound to the version (:func:`versioned_dense`).

    Returns:
        The task, whose identity now includes the version.

    Raises:
        ValueError: The task has no provenance, or a rebound part belongs to another task.
    """
    provenance: MiningProvenance | None = task.provenance
    if provenance is None:
        raise ValueError("a rubric version is recorded only on a task with mining provenance")
    if task.judge is not None and (
        judge.definition.task_sha256 != task.judge.definition.task_sha256
        or judge.definition.world_sha256 != task.judge.definition.world_sha256
    ):
        raise ValueError("the rebound judge belongs to another task or world")
    record = provenance.model_copy(update={"rubric": rubric_provenance(version)})
    return task.model_copy(update={"judge": judge, "dense": dense, "provenance": record})


def first_version(run_id: str, banks: Sequence[QuestionBank], epoch: int = 1) -> RubricVersion:
    """Return version 1: the pinned banks and no item."""
    return RubricVersion(
        run_id=run_id, version=1, epoch=epoch, banks=tuple(BankPin.of(b) for b in banks)
    )


def load_version(project: ProjectStore, reference: ArtifactInput) -> RubricVersion:
    """Read a saved version, refusing a changed artifact."""
    stored = project.artifacts.read(reference.artifact_id)
    if artifact_input(stored.manifest) != reference:
        raise ValueError(f"rubric version {reference.artifact_id} changed")
    return RubricVersion.model_validate_json(stored.payloads["version.json"])


def save_version(
    project: ProjectStore,
    version: RubricVersion,
    *,
    code_revision: str,
    inputs: tuple[ArtifactInput, ...] = (),
) -> ArtifactInput:
    """Save a version once; its recall and item bookkeeping are part of the saved record."""
    identity = stable_id(
        "rubric-version", {"version": version.identity, "record": sha256_json(version)}
    )
    store = project.artifacts
    if not store.exists(identity):
        store.write_json(
            artifact_id=identity,
            artifact_type="rubric-version",
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=code_revision,
                inputs=tuple(sorted(set(inputs), key=lambda ref: ref.artifact_id)),
            ),
            files={"version.json": version},
        )
    return artifact_input(store.read(identity).manifest)
