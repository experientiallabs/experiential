"""One family's authored bank, re-authored with feedback while a large family shows one task type.

A first authoring pass (:func:`~exp.simulation.mining.tag_bank_selection.corpus_tag_bank`) can
keep questions that are each discriminating and distinct yet, together, show no structure: the
family's members then cluster into a single task type (:mod:`~exp.simulation.mining.task_types`)
even when the family is large. :func:`author_family_bank` checks for that and, for a family with
at least ``min_members_to_retry`` members, re-runs the author up to ``max_rounds`` times. Each
re-authoring round is told (:class:`~exp.simulation.mining.tag_authoring.AuthoringFeedback`)
every earlier question with its fire rate, correlation and verdict from selection, that the
family collapsed to one type, and is asked for traits that split it roughly 20 to 80 percent
along an axis the existing questions do not cover.

A round's candidates go through the unchanged selection, the kept ones are tagged over every
member, and any that correlates above ``max_correlation`` with a question already in the bank
is dropped as a near-duplicate of it. The rest join the bank, which is published as its next
version, and the members are clustered again over the shared matrices and the union bank with
the same fit task types uses. The loop stops early when a round yields two or more stable task
types, or when it adds no kept question.

Every step replays: authoring and selection are content-addressed by protocol, scope, evidence,
feedback and request (never by code revision), and each round is saved as a content-addressed
``tag-authoring-round`` record.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import Field

from exp.common.core.artifacts import (
    ArtifactInput,
    ContractModel,
    JsonObject,
    sha256_json,
    stable_id,
)
from exp.common.project import ArtifactStore, artifact_input
from exp.simulation.mining.questions import Question, QuestionBank, QuestionBankError
from exp.simulation.mining.tag_authoring import (
    AUTHORING_LOOP_PROTOCOL,
    AuthoringFeedback,
    QuestionFeedback,
    TagAuthor,
    TagAuthoringSpec,
    authored_bank_for_view,
    authored_questions,
)
from exp.simulation.mining.tag_bank_selection import (
    AuthoredBank,
    BankSelection,
    BankSelectionSpec,
    answer_correlation,
    publish_question_bank,
    selected_tag_bank,
)
from exp.simulation.mining.tag_clusters import (
    ClusteringSpec,
    cluster_tags,
    write_record,
)
from exp.simulation.mining.tagging import TagMatrix, TagRow, TraceTagger
from exp.simulation.mining.task_types import TaskTypeSpec, family_matrices
from exp.simulation.mining.views import Unit

AUTHORING_ROUND_TYPE = "tag-authoring-round"
"""Artifact type of a saved :class:`AuthoringRound` (``round.json``)."""

Columns = dict[str, dict[str, float]]
"""Each question's answers by member trace."""


class AuthoringLoopSpec(ContractModel):
    """When a family's bank is re-authored with feedback.

    Attributes:
        min_members_to_retry: Fewest members a family needs before a one-type split is retried.
        max_rounds: Most re-authoring rounds per family after the first pass; zero disables.
        target_split: Share of members a new trait is asked to hold for, low and high.
        clustering: The fit that counts task types; keep it equal to the task-types slot's.
    """

    min_members_to_retry: int = Field(default=100, ge=2, le=1_000_000)
    max_rounds: int = Field(default=2, ge=0, le=8)
    target_split: tuple[float, float] = (0.2, 0.8)
    clustering: ClusteringSpec = Field(default_factory=lambda: TaskTypeSpec().clustering)

    def loops(self, members: int) -> bool:
        """Return whether a family this large may be re-authored."""
        return self.max_rounds > 0 and members >= self.min_members_to_retry


class AuthoringRound(ContractModel):
    """The ``tag-authoring-round`` record: one re-authoring round of one family.

    Attributes:
        protocol: Loop protocol.
        scope: The family.
        feedback: What the round's author was told.
        authoring: The round's saved ``tag-authoring`` record.
        selection: Its saved ``tag-bank-selection`` record, when a candidate was authored and
            survived selection.
        added: Questions that joined the bank, as authored (publishing may suffix an identity).
        redundant: Kept questions dropped as near-duplicates of a bank question, with why.
        task_types: Stable task types after the round, when questions were added.
        outcome: What the round led to.
    """

    protocol: str = AUTHORING_LOOP_PROTOCOL
    scope: str
    feedback: AuthoringFeedback
    authoring: ArtifactInput
    selection: ArtifactInput | None = None
    added: tuple[str, ...] = ()
    redundant: dict[str, str] = Field(default_factory=dict)
    task_types: int | None = None
    outcome: str

    def identity(self) -> str:
        """Return the content-addressed identity of this record."""
        return stable_id(
            AUTHORING_ROUND_TYPE, {"protocol": self.protocol, "record": sha256_json(self)}
        )


@dataclass(frozen=True)
class FamilyBank:
    """One family's final bank, its member matrices and how it got there.

    Attributes:
        bank: The authored bank as published, citing every round.
        matrices: The members' answers to it, per view.
        task_types: Stable task types its members cluster into, when the family may loop.
        rounds: Every re-authoring round run, in order.
    """

    bank: AuthoredBank
    matrices: dict[str, TagMatrix]
    task_types: int | None
    rounds: tuple[AuthoringRound, ...] = ()


def author_family_bank(
    store: ArtifactStore,
    author: TagAuthor,
    tagger: TraceTagger,
    *,
    scope: str,
    bank_id: str,
    rendered: Sequence[Mapping[str, Unit]],
    views: Sequence[str],
    shared: Sequence[TagMatrix],
    tagger_identity: JsonObject,
    authoring: TagAuthoringSpec | None = None,
    selection: BankSelectionSpec | None = None,
    loop: AuthoringLoopSpec | None = None,
    root: Path | None = None,
    code_revision: str,
) -> FamilyBank:
    """Author, select and tag one family's bank, re-authoring while it shows one task type.

    Args:
        store: Artifact store holding every saved step.
        author: Injected author model (offline authoring scope only).
        tagger: The unchanged decision tagging seam.
        scope: The family identity.
        bank_id: Identity of the published bank.
        rendered: Each member's rendered units, by view.
        views: Views the authored questions are asked over.
        shared: Matrices every family is clustered on besides its own bank (the fixed banks).
        tagger_identity: Identity of the decision model behind ``tagger``.
        authoring: Authoring registration; defaults apply when omitted.
        selection: Selection registration; defaults apply when omitted.
        loop: Re-authoring registration; defaults apply when omitted.
        root: Bank directory, or None for the local bank root.
        code_revision: Code revision recorded on saved artifacts, never in an identity.
    """
    selection = selection or BankSelectionSpec()
    loop = loop or AuthoringLoopSpec()
    units = [unit for row in rendered for unit in row.values()]
    members = tuple(next(iter(row.values())).trace_id for row in rendered)
    record, first_authoring = authored_questions(
        store, author, scope=scope, units=units, spec=authoring, code_revision=code_revision
    )
    first, first_selection = selected_tag_bank(
        store,
        record.candidate_bank(bank_id),
        tagger,
        units=units,
        tagger_identity=tagger_identity,
        spec=selection,
        code_revision=code_revision,
        inputs=(first_authoring,),
    )
    bank = first.bank
    _, columns = _member_tags(tagger, rendered, views, bank)
    count: int | None = None
    rounds: list[AuthoringRound] = []
    references: list[ArtifactInput] = []
    if loop.loops(len(members)):
        count, reason = _task_types(
            shared, _matrices(bank, rendered, views, columns), members, loop
        )
        history = _measured(first, {})
        for pass_number in range(1, loop.max_rounds + 1):
            if count >= 2:
                break
            feedback = AuthoringFeedback(
                pass_number=pass_number,
                members=len(members),
                task_types=count,
                clustering_reason=reason[:2_000],
                questions=tuple(history),
                target_split=loop.target_split,
            )
            record, authored_ref = authored_questions(
                store,
                author,
                scope=scope,
                units=units,
                spec=authoring,
                feedback=feedback,
                code_revision=code_revision,
                inputs=(first_selection, *references),
            )
            fields: dict[str, object] = {
                "scope": scope,
                "feedback": feedback,
                "authoring": authored_ref,
            }
            try:
                chosen, selected_ref = selected_tag_bank(
                    store,
                    record.candidate_bank(bank_id),
                    tagger,
                    units=units,
                    tagger_identity=tagger_identity,
                    spec=selection,
                    code_revision=code_revision,
                    inputs=(authored_ref,),
                )
            except QuestionBankError as error:
                fields["outcome"] = f"stopped: the round kept no question ({error})"[:2_000]
                references.append(_saved(store, rounds, fields, code_revision))
                break
            _, answers = _member_tags(tagger, rendered, views, chosen.bank)
            added, redundant = _distinct(chosen.bank, answers, columns, bank, selection)
            history.extend(_measured(chosen, redundant))
            fields |= {"selection": selected_ref, "redundant": redundant}
            if not added:
                fields["outcome"] = "stopped: every kept question duplicates a bank question"
                references.append(_saved(store, rounds, fields, code_revision))
                break
            bank = bank.model_copy(update={"questions": bank.questions + tuple(added)})
            columns |= {q.question_id: answers[q.question_id] for q in added}
            matrices = _matrices(bank, rendered, views, columns)
            count, reason = _task_types(shared, matrices, members, loop)
            fields |= {
                "added": tuple(q.question_id for q in added),
                "task_types": count,
                "outcome": (
                    f"split: {count} stable task types"
                    if count >= 2
                    else "one task type remains after the last round"
                    if pass_number == loop.max_rounds
                    else "one task type remains; authoring again"
                ),
            }
            references.append(_saved(store, rounds, fields, code_revision))
    # Only the final bank is published, so a replay never publishes an intermediate version.
    published, path = publish_question_bank(bank, root=root)
    rename = dict(zip(bank.question_ids(), published.question_ids(), strict=True))
    columns = {rename[q]: answers for q, answers in columns.items() if q in rename}
    authored = AuthoredBank(
        authoring=first_authoring,
        selection=first_selection,
        bank=published,
        path=str(path),
        rounds=tuple(references),
    )
    return FamilyBank(
        bank=authored,
        matrices=_matrices(published, rendered, views, columns),
        task_types=count,
        rounds=tuple(rounds),
    )


def _saved(
    store: ArtifactStore,
    rounds: list[AuthoringRound],
    fields: Mapping[str, object],
    code_revision: str,
) -> ArtifactInput:
    """Save one round record, or adopt the identical saved one, and append it."""
    record = AuthoringRound.model_validate(dict(fields))
    identity = record.identity()
    cited = tuple(ref for ref in (record.authoring, record.selection) if ref is not None)
    rounds.append(record)
    if store.exists(identity):
        return artifact_input(store.read(identity).manifest)
    return write_record(
        store, identity, AUTHORING_ROUND_TYPE, "round.json", record, code_revision, cited
    )


def _member_tags(
    tagger: TraceTagger,
    rendered: Sequence[Mapping[str, Unit]],
    views: Sequence[str],
    bank: QuestionBank,
) -> tuple[dict[str, TagMatrix], Columns]:
    """Tag every member with a bank, per view; return the matrices and each question's column."""
    matrices: dict[str, TagMatrix] = {}
    columns: Columns = {}
    for view in views:
        asked = authored_bank_for_view(bank, view)
        if asked is None:
            continue
        matrix = tagger.tag([row[view] for row in rendered], asked)
        matrices[view] = matrix
        for position, question_id in enumerate(matrix.question_ids):
            columns[question_id] = {row.trace_id: row.values[position] for row in matrix.rows}
    return matrices, columns


def _matrices(
    bank: QuestionBank,
    rendered: Sequence[Mapping[str, Unit]],
    views: Sequence[str],
    columns: Columns,
) -> dict[str, TagMatrix]:
    """Assemble the members' matrices of a bank from answers already bought, per view."""
    matrices: dict[str, TagMatrix] = {}
    for view in views:
        asked = authored_bank_for_view(bank, view)
        if asked is None:
            continue
        ids = asked.question_ids()
        rows = tuple(
            TagRow(
                trace_id=unit.trace_id,
                unit_hash=unit.unit_hash,
                values=tuple(columns[q][unit.trace_id] for q in ids),
            )
            for unit in (row[view] for row in rendered)
            if all(unit.trace_id in columns[q] for q in ids)
        )
        matrices[view] = TagMatrix(
            bank_id=asked.bank_id,
            bank_version=asked.version,
            question_ids=ids,
            view=view,
            rows=rows,
        )
    return matrices


def _task_types(
    shared: Sequence[TagMatrix],
    matrices: Mapping[str, TagMatrix],
    members: Sequence[str],
    loop: AuthoringLoopSpec,
) -> tuple[int, str]:
    """Count the stable task types the members cluster into, with the fit task types uses."""
    own = family_matrices(members, [*shared, *matrices.values()])
    clustering = cluster_tags(own, loop.clustering, trace_ids=members)
    return sum(1 for cluster in clustering.clusters if cluster.size > 0), clustering.reason


def _distinct(
    kept: QuestionBank,
    answers: Columns,
    bank_columns: Columns,
    bank: QuestionBank,
    spec: BankSelectionSpec,
) -> tuple[list[Question], dict[str, str]]:
    """Split a round's kept questions into ones new to the bank and near-duplicates of it."""
    added: list[Question] = []
    redundant: dict[str, str] = {}
    present = dict(bank_columns)
    for question in kept.questions:
        column = answers[question.question_id]
        best: tuple[float, str] | None = None
        for other, values in present.items():
            value = answer_correlation(column, values, spec.min_overlap)
            if value is not None and (best is None or value > best[0]):
                best = (value, other)
        if best is not None and best[0] > spec.max_correlation:
            redundant[question.question_id] = (
                f"redundant: correlation {best[0]:.2f} with bank question {best[1]}"
            )
        elif len(bank.questions) + len(added) >= spec.max_questions:
            redundant[question.question_id] = f"over the {spec.max_questions}-question limit"
        else:
            added.append(question)
            present[question.question_id] = column
    return added, redundant


def _measured(selection: BankSelection, redundant: Mapping[str, str]) -> list[QuestionFeedback]:
    """Return every candidate of a selection as feedback, with near-duplicates marked dropped."""
    texts = {q.question_id: q.text for q in selection.candidate.questions}
    return [
        QuestionFeedback(
            question_id=score.question_id,
            text=texts[score.question_id],
            kept=score.kept and score.question_id not in redundant,
            fire_rate=score.fire_rate,
            max_correlation=score.max_correlation,
            correlated_with=score.correlated_with,
            reason=redundant.get(score.question_id, score.reason),
        )
        for score in selection.scores
    ]
