"""Select authored questions that split the corpus, and publish them as a versioned bank.

An author (:mod:`~exp.simulation.mining.tag_authoring`) proposes more questions than a bank
should carry, and its own belief about which split a question makes is not evidence. Selection
asks the decision model instead: the unchanged tagger
(:class:`~exp.simulation.mining.tagging.TraceTagger`, in production a
:class:`~exp.simulation.mining.decisions.DecisionsTraceTagger` with its decision cache) answers
every candidate over a sample of the scope's rendered units, each authored family over its own
view, and the answers decide:

- **discriminating**: the share of sampled traces where the question fires lies inside
  ``[min_fire_rate, max_fire_rate]`` and its probabilities spread by at least ``min_spread``, so
  a near-constant question, which no geometry can use, is dropped;
- **answerable**: at most ``max_ambiguous`` of its answers fall inside the ambiguity band, so a
  question the decision model cannot settle from the text is dropped;
- **not redundant**: candidates are taken greedily, most balanced first, and one whose absolute
  correlation with an already kept question exceeds ``max_correlation`` (over the traces both
  were answered on) is dropped as a near-duplicate of it.

Every candidate's measurements and verdict are recorded in a content-addressed
``tag-bank-selection`` artifact, and :func:`selected_tag_bank` replays it with no call.
:func:`publish_question_bank` writes the kept questions as an ordinary local bank file
(``<root>/<bank_id>.toml``, the file :func:`~exp.simulation.mining.questions.resolve_question_bank`
reads) with a monotonic version: identical content keeps its version, changed content gets the
next one, and every published version is kept under ``<root>/history/<bank_id>/v<N>.toml``. A
question identity never changes meaning across versions: a reworded question published under an
identity an earlier version used gets a revision suffix instead.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import tomli_w
from pydantic import Field, model_validator

from exp.common.core.artifacts import (
    ArtifactInput,
    ContractModel,
    JsonObject,
    Sha256,
    sha256_json,
    stable_id,
)
from exp.common.project import ArtifactStore, artifact_input
from exp.simulation.mining.questions import (
    QuestionBank,
    QuestionBankError,
    load_question_bank,
    local_bank_root,
)
from exp.simulation.mining.tag_authoring import (
    TagAuthor,
    TagAuthoringSpec,
    authored_bank_for_view,
    authored_questions,
    view_of_family,
)
from exp.simulation.mining.tag_clusters import read_record, write_record
from exp.simulation.mining.tag_selection import DEFAULT_AMBIGUITY_BAND, DEFAULT_FIRE_THRESHOLD
from exp.simulation.mining.tagging import TagMatrix, TraceTagger, TraceTaggingError
from exp.simulation.mining.views import Unit

logger = logging.getLogger(__name__)

TAG_BANK_SELECTION_PROTOCOL = "tag-bank-selection-v1"
"""Version of the discrimination, answerability and redundancy rules; bump when any changes."""

SELECTION_TYPE = "tag-bank-selection"
"""Artifact type of a saved :class:`BankSelection` (``selection.json``)."""

HISTORY_DIRECTORY = "history"
"""Directory under the bank root holding every published version of every authored bank."""


class BankSelectionSpec(ContractModel):
    """Which candidate questions a bank keeps.

    Attributes:
        min_fire_rate: Smallest share of sampled traces a kept question fires on.
        max_fire_rate: Largest such share.
        min_spread: Smallest standard deviation of a kept question's probabilities.
        max_ambiguous: Largest share of a kept question's answers inside the ambiguity band.
        max_correlation: Largest absolute correlation with any already kept question.
        max_questions: Most questions the bank keeps.
        min_overlap: Fewest traces two questions must share before their correlation is read.
        threshold: Probability strictly above which a question fires.
        ambiguity_band: Closed band of answers counted as ambiguous.
    """

    min_fire_rate: float = Field(default=0.05, ge=0.0, lt=0.5)
    max_fire_rate: float = Field(default=0.95, gt=0.5, le=1.0)
    min_spread: float = Field(default=0.1, ge=0.0, le=0.5)
    max_ambiguous: float = Field(default=0.25, ge=0.0, le=1.0)
    max_correlation: float = Field(default=0.8, gt=0.0, le=1.0)
    max_questions: int = Field(default=48, ge=1, le=512)
    min_overlap: int = Field(default=3, ge=2, le=1_000)
    threshold: float = Field(default=DEFAULT_FIRE_THRESHOLD, gt=0.0, lt=1.0)
    ambiguity_band: tuple[float, float] = DEFAULT_AMBIGUITY_BAND

    @model_validator(mode="after")
    def _band(self) -> BankSelectionSpec:
        """Require an ordered ambiguity band inside the unit interval."""
        low, high = self.ambiguity_band
        if not 0.0 <= low <= high <= 1.0:
            raise ValueError("ambiguity_band must be an ordered pair inside [0, 1]")
        return self


class QuestionScore(ContractModel):
    """One candidate's measurements and verdict.

    Attributes:
        question_id: The candidate.
        view: View it was answered over.
        answered: Sampled traces it was answered on.
        fire_rate: Share of those where it fired.
        spread: Standard deviation of its probabilities.
        ambiguous: Share of its answers inside the ambiguity band.
        balance: Binary entropy of the fire rate, the greedy order (most balanced first).
        max_correlation: Largest absolute correlation with a kept question, when one was read.
        correlated_with: That kept question.
        kept: Whether the bank keeps it.
        reason: Why it was kept or dropped.
    """

    question_id: str
    view: str
    answered: int = Field(ge=0)
    fire_rate: float = Field(ge=0.0, le=1.0)
    spread: float = Field(ge=0.0)
    ambiguous: float = Field(ge=0.0, le=1.0)
    balance: float = Field(ge=0.0, le=1.0)
    max_correlation: float | None = None
    correlated_with: str | None = None
    kept: bool = False
    reason: str


class BankSelection(ContractModel):
    """The ``tag-bank-selection`` record: every candidate's verdict and the kept bank.

    Attributes:
        protocol: Selection protocol.
        spec: The selection registration.
        candidate: The candidate bank asked.
        tagger: Identity of the decision model that answered.
        sample: Sampled unit hashes, by view.
        matrices: Digest of each view's answers.
        scores: Every candidate's measurements, in candidate order.
        bank: The kept questions, in candidate order, under the candidate's identity.
    """

    protocol: str = TAG_BANK_SELECTION_PROTOCOL
    spec: BankSelectionSpec
    candidate: QuestionBank
    tagger: JsonObject
    sample: dict[str, tuple[str, ...]]
    matrices: dict[str, Sha256]
    scores: tuple[QuestionScore, ...]
    bank: QuestionBank

    def identity(self) -> str:
        """Return the content-addressed identity of this record."""
        return stable_id(SELECTION_TYPE, {"protocol": self.protocol, "record": sha256_json(self)})

    def kept(self) -> tuple[str, ...]:
        """Return the kept question identities, in candidate order."""
        return self.bank.question_ids()


def selection_identity(
    candidate: QuestionBank,
    units: Sequence[Unit],
    spec: BankSelectionSpec,
    tagger_identity: JsonObject,
) -> str:
    """Return the identity a selection is saved under, computed before any call."""
    return stable_id(
        SELECTION_TYPE,
        {
            "protocol": TAG_BANK_SELECTION_PROTOCOL,
            "candidate": sha256_json(candidate),
            "units": sorted({(u.view, u.trace_id, u.unit_hash) for u in units}),
            "spec": sha256_json(spec),
            "tagger": tagger_identity,
        },
    )


def select_tag_bank(
    candidate: QuestionBank,
    tagger: TraceTagger,
    *,
    units: Sequence[Unit],
    tagger_identity: JsonObject,
    spec: BankSelectionSpec | None = None,
) -> BankSelection:
    """Tag a sample with the decision model and keep the questions that split it.

    Args:
        candidate: Authored candidate bank; every family is an author variant.
        tagger: The unchanged decision tagging seam.
        units: The sample's rendered units, over the views the candidate's families read.
        tagger_identity: Identity of the decision model behind ``tagger``.
        spec: Selection registration; defaults apply when omitted.

    Raises:
        TraceTaggingError: The sample renders no view a candidate family is asked over.
        QuestionBankError: No candidate survives selection.
    """
    spec = spec or BankSelectionSpec()
    by_view: dict[str, list[Unit]] = {}
    for unit in units:
        by_view.setdefault(unit.view, []).append(unit)
    needed = sorted({view_of_family(q.family) for q in candidate.questions})
    missing = [view for view in needed if view not in by_view]
    if missing:
        raise TraceTaggingError(
            f"candidate bank {candidate.bank_id} asks over views {missing} that the sample does "
            "not render; render them for every sampled trace"
        )
    columns: dict[str, dict[str, float]] = {}
    views: dict[str, str] = {}
    matrices: dict[str, str] = {}
    sample: dict[str, tuple[str, ...]] = {}
    for view in needed:
        asked = authored_bank_for_view(candidate, view)
        if asked is None:
            continue
        matrix = tagger.tag(by_view[view], asked)
        _check(matrix, asked)
        matrices[view] = sha256_json(matrix)
        sample[view] = tuple(sorted({unit.unit_hash for unit in by_view[view]}))
        for position, question_id in enumerate(matrix.question_ids):
            columns[question_id] = {row.trace_id: row.values[position] for row in matrix.rows}
            views[question_id] = view
    scores = _scores(candidate, columns, views, spec)
    kept = frozenset(score.question_id for score in scores if score.kept)
    if not kept:
        raise QuestionBankError(
            f"no candidate of {candidate.bank_id} discriminates, is answerable and is distinct on "
            "this sample; author again with a larger or more varied sample"
        )
    bank = candidate.model_copy(
        update={"questions": tuple(q for q in candidate.questions if q.question_id in kept)}
    )
    return BankSelection(
        spec=spec,
        candidate=candidate,
        tagger=tagger_identity,
        sample=sample,
        matrices=matrices,
        scores=scores,
        bank=bank,
    )


def selected_tag_bank(
    store: ArtifactStore,
    candidate: QuestionBank,
    tagger: TraceTagger,
    *,
    units: Sequence[Unit],
    tagger_identity: JsonObject,
    spec: BankSelectionSpec | None = None,
    code_revision: str,
    inputs: Sequence[ArtifactInput] = (),
) -> tuple[BankSelection, ArtifactInput]:
    """Replay the saved selection of exactly this candidate, sample and model, or select.

    Returns:
        The record and its saved artifact.
    """
    spec = spec or BankSelectionSpec()
    identity = selection_identity(candidate, units, spec, tagger_identity)
    if store.exists(identity):
        reference = artifact_input(store.read(identity).manifest)
        return load_bank_selection(store, reference), reference
    record = select_tag_bank(
        candidate, tagger, units=units, tagger_identity=tagger_identity, spec=spec
    )
    return record, write_record(
        store, identity, SELECTION_TYPE, "selection.json", record, code_revision, inputs
    )


def load_bank_selection(store: ArtifactStore, reference: ArtifactInput) -> BankSelection:
    """Read a saved selection, refusing a changed or mistyped artifact."""
    payload = read_record(store, reference, SELECTION_TYPE, "selection.json")
    return BankSelection.model_validate_json(payload)


def publish_question_bank(
    bank: QuestionBank, *, root: Path | None = None
) -> tuple[QuestionBank, Path]:
    """Write kept questions as the local bank file the unchanged tagger resolves.

    The written bank keeps ``bank.bank_id``. Its version is the published one when the questions
    are identical, otherwise one more than the latest published version. A question identity an
    earlier version used for different text is published as ``<id>.r<version>``, so an identity
    never changes meaning.

    Args:
        bank: Kept questions; its version is ignored in favour of the published history.
        root: Bank directory, or None for :func:`~exp.simulation.mining.questions.local_bank_root`.

    Returns:
        The published bank and the path of its current file.
    """
    directory = local_bank_root() if root is None else root
    history = directory / HISTORY_DIRECTORY / bank.bank_id
    published = _history(history)
    latest = published[-1] if published else None
    if latest is not None and latest.questions == bank.questions:
        current = directory / f"{bank.bank_id}.toml"
        if not current.is_file() or load_question_bank(current) != latest:
            _write(current, latest)
        return latest, current
    version = 1 if latest is None else latest.version + 1
    used = {q.question_id: q.text for old in published for q in old.questions}
    questions = tuple(
        q
        if used.get(q.question_id, q.text) == q.text
        else q.model_copy(update={"question_id": f"{q.question_id}.r{version}"})
        for q in bank.questions
    )
    result = QuestionBank(bank_id=bank.bank_id, version=version, questions=questions)
    _write(history / f"v{version}.toml", result)
    current = directory / f"{bank.bank_id}.toml"
    _write(current, result)
    logger.info("published question bank %s version %d at %s", bank.bank_id, version, current)
    return result, current


class AuthoredBank(ContractModel):
    """Where one scope's corpus-authored bank came from and where it was published.

    Attributes:
        authoring: The saved ``tag-authoring`` record.
        selection: The saved ``tag-bank-selection`` record.
        bank: The published bank, as the unchanged tagger resolves it.
        path: The published bank file.
    """

    authoring: ArtifactInput
    selection: ArtifactInput
    bank: QuestionBank
    path: str


def corpus_tag_bank(
    store: ArtifactStore,
    author: TagAuthor,
    tagger: TraceTagger,
    *,
    scope: str,
    bank_id: str,
    units: Sequence[Unit],
    selection_units: Sequence[Unit] | None = None,
    tagger_identity: JsonObject,
    authoring: TagAuthoringSpec | None = None,
    selection: BankSelectionSpec | None = None,
    root: Path | None = None,
    code_revision: str,
) -> AuthoredBank:
    """Author, select and publish one scope's bank; every saved step replays with no call.

    Args:
        store: Artifact store holding the authoring and selection records.
        author: Injected author model.
        tagger: The unchanged decision tagging seam used for selection.
        scope: Stable scope identity, such as a family or cluster identity.
        bank_id: Identity of the published bank.
        units: Rendered units the author samples from.
        selection_units: Units tagged for selection; ``units`` when None.
        tagger_identity: Identity of the decision model behind ``tagger``.
        authoring: Authoring registration; defaults apply when omitted.
        selection: Selection registration; defaults apply when omitted.
        root: Bank directory, or None for the local bank root.
        code_revision: Code revision recorded on saved artifacts.
    """
    record, authored = authored_questions(
        store, author, scope=scope, units=units, spec=authoring, code_revision=code_revision
    )
    candidate = record.candidate_bank(bank_id)
    chosen, selected = selected_tag_bank(
        store,
        candidate,
        tagger,
        units=units if selection_units is None else selection_units,
        tagger_identity=tagger_identity,
        spec=selection,
        code_revision=code_revision,
        inputs=(authored,),
    )
    bank, path = publish_question_bank(chosen.bank, root=root)
    return AuthoredBank(authoring=authored, selection=selected, bank=bank, path=str(path))


def score_candidates(
    candidate: QuestionBank,
    columns: Mapping[str, Mapping[str, float]],
    views: Mapping[str, str],
    spec: BankSelectionSpec | None = None,
) -> tuple[QuestionScore, ...]:
    """Measure already bought answers with the selection rules, without asking anything.

    The same discrimination, answerability and redundancy rules :func:`select_tag_bank` applies,
    for a caller that already holds the decision model's answers (for example a rubric that
    re-checks whether its questions still split the latest rollouts).

    Args:
        candidate: The questions to measure, in candidate order.
        columns: Each question's answers by trace identity.
        views: The view each question was answered over.
        spec: Selection rules; defaults apply when omitted.

    Returns:
        Every question's measurements and verdict, in candidate order.

    Raises:
        KeyError: A candidate question has no column or view.
    """
    return _scores(candidate, columns, views, spec or BankSelectionSpec())


def _check(matrix: TagMatrix, asked: QuestionBank) -> None:
    """Refuse a matrix that does not answer exactly the asked questions."""
    if matrix.question_ids != asked.question_ids() or not matrix.rows:
        raise TraceTaggingError(
            f"the tagger answered {len(matrix.question_ids)} columns over {len(matrix.rows)} "
            f"traces for bank {asked.bank_id}; it must answer every asked question"
        )


def _scores(
    candidate: QuestionBank,
    columns: Mapping[str, Mapping[str, float]],
    views: Mapping[str, str],
    spec: BankSelectionSpec,
) -> tuple[QuestionScore, ...]:
    """Measure every candidate and decide its verdict greedily, most balanced first."""
    low, high = spec.ambiguity_band
    measured: dict[str, QuestionScore] = {}
    for question in candidate.questions:
        values = np.array(list(columns[question.question_id].values()), dtype=np.float64)
        rate = float((values > spec.threshold).mean())
        spread = float(values.std())
        ambiguous = float(((values >= low) & (values <= high)).mean())
        problems = []
        if not spec.min_fire_rate <= rate <= spec.max_fire_rate:
            problems.append(
                f"near-constant: fires on {rate:.2f} of the sample, outside "
                f"[{spec.min_fire_rate}, {spec.max_fire_rate}]"
            )
        elif spread < spec.min_spread:
            problems.append(f"near-constant: answers spread only {spread:.3f}")
        if ambiguous > spec.max_ambiguous:
            problems.append(
                f"unanswerable: {ambiguous:.2f} of answers are inside the ambiguity band"
            )
        measured[question.question_id] = QuestionScore(
            question_id=question.question_id,
            view=views[question.question_id],
            answered=int(values.size),
            fire_rate=round(rate, 6),
            spread=round(spread, 6),
            ambiguous=round(ambiguous, 6),
            balance=round(_entropy(rate), 6),
            reason="; ".join(problems) or "pending",
        )
    order = sorted(
        (s for s in measured.values() if s.reason == "pending"),
        key=lambda s: (-s.balance, s.ambiguous, candidate.question_ids().index(s.question_id)),
    )
    kept: list[str] = []
    for score in order:
        best: tuple[float, str] | None = None
        for other in kept:
            value = _correlation(columns[score.question_id], columns[other], spec.min_overlap)
            if value is not None and (best is None or value > best[0]):
                best = (value, other)
        update: JsonObject = {}
        if best is not None:
            update = {"max_correlation": round(best[0], 6), "correlated_with": best[1]}
        if best is not None and best[0] > spec.max_correlation:
            update["reason"] = f"redundant: correlation {best[0]:.2f} with kept {best[1]}"
        elif len(kept) >= spec.max_questions:
            update["reason"] = f"over the {spec.max_questions}-question limit"
        else:
            kept.append(score.question_id)
            update |= {"kept": True, "reason": "discriminating, answerable and distinct"}
        measured[score.question_id] = score.model_copy(update=update)
    return tuple(measured[q.question_id] for q in candidate.questions)


def _correlation(
    left: Mapping[str, float], right: Mapping[str, float], minimum: int
) -> float | None:
    """Return the absolute Pearson correlation over shared traces, or None when unreadable."""
    shared = sorted(set(left) & set(right))
    if len(shared) < minimum:
        return None
    a = np.array([left[t] for t in shared])
    b = np.array([right[t] for t in shared])
    if a.std() == 0.0 or b.std() == 0.0:
        return 1.0 if a.std() == b.std() == 0.0 else 0.0
    return float(abs(np.corrcoef(a, b)[0, 1]))


def _entropy(rate: float) -> float:
    """Return the binary entropy of a fire rate, in bits."""
    if rate <= 0.0 or rate >= 1.0:
        return 0.0
    return -(rate * math.log2(rate) + (1.0 - rate) * math.log2(1.0 - rate))


def _history(directory: Path) -> list[QuestionBank]:
    """Return every published version of one bank, oldest first."""
    if not directory.is_dir():
        return []
    banks = [load_question_bank(path) for path in directory.glob("v*.toml")]
    return sorted(banks, key=lambda bank: bank.version)


def _write(path: Path, bank: QuestionBank) -> None:
    """Write one bank file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "bank_id": bank.bank_id,
        "version": bank.version,
        "questions": [q.model_dump(mode="json") for q in bank.questions],
    }
    staging = path.with_suffix(".toml.tmp")
    staging.write_text(tomli_w.dumps(payload), encoding="utf-8")
    staging.replace(path)
