"""Decision tagging of the variance build: per-unit view tagging and the error tags.

:func:`tag_units` answers one view's units one at a time, so a unit the tagger refuses (for
example at the credential boundary) is recorded as a failure instead of aborting the view.

:func:`asked_error_bank` decides what the error tagging asks. By default it asks the error bank's
observations-view propositions only. A spec naming ``error_consumed_families`` asks every
selected question of the bank over the observations view instead, which is exactly the request a
full-bank tagging pass makes, so its saved decisions replay. :func:`tag_build_errors` then saves
that full matrix and keeps only the declared families' columns in the ``error-tags`` record the
build consumes. Columns are selected by the questions' declared families, never by their names.
"""

from __future__ import annotations

from collections.abc import Sequence

from exp.common.core.artifacts import ArtifactInput, sha256_json
from exp.common.project import ArtifactStore
from exp.optimize.evaluation.variance.build_records import (
    MATRIX_TYPE,
    TagMatrixRecord,
    VarianceBuildPlan,
    VarianceBuildSpec,
    save_artifact,
)
from exp.runtime.models.providers.transport import ProviderTransportError
from exp.simulation.mining.decisions import DecisionsTraceTagger
from exp.simulation.mining.dispatch import dispatch_tagging
from exp.simulation.mining.error_tags import (
    ErrorTags,
    bank_for_view,
    error_bank,
    persist_error_tags,
    summarize_row,
    tag_errors,
)
from exp.simulation.mining.questions import QuestionBank
from exp.simulation.mining.tag_selection import DEFAULT_AMBIGUITY_BAND, DEFAULT_FIRE_THRESHOLD
from exp.simulation.mining.tagging import (
    TagMatrix,
    TagRow,
    TraceTaggingError,
    distinct_units,
    tagged_view,
)
from exp.simulation.mining.views import Unit

SavedErrors = tuple[ErrorTags, ArtifactInput, ArtifactInput | None]
"""The consumed error-tags record, its artifact, and the saved full matrix when one was asked."""


def tag_units(
    tagger: DecisionsTraceTagger, units: Sequence[Unit], bank: QuestionBank
) -> tuple[TagMatrix, tuple[tuple[str, str], ...]]:
    """Tag independent units, keeping input failures apart from fatal refusals.

    Args:
        tagger: Decision client with its concurrency bound.
        units: Rendered traces of one view.
        bank: Questions to ask.

    Returns:
        Tagged rows and recoverable failures; fatal refusals propagate unchanged.

    Raises:
        TraceTaggingError: Every unit failed.
    """
    view = tagged_view(units)
    decided = distinct_units(units)

    def answer(unit: Unit) -> TagRow | str:
        """Return one row or a recoverable failure, preserving fatal refusals."""
        try:
            return tagger.tag([unit], bank).rows[0]
        except (TraceTaggingError, ProviderTransportError) as exc:
            return str(exc)

    outcomes = dispatch_tagging(decided, answer, maximum_concurrency=tagger.maximum_concurrency)
    by_hash = {unit.unit_hash: result for unit, result in zip(decided, outcomes, strict=True)}
    rows: dict[str, TagRow] = {}
    failed: list[tuple[str, str]] = []
    for unit in units:
        result = by_hash[unit.unit_hash]
        if isinstance(result, str):
            failed.append((unit.trace_id, result))
        else:
            rows.setdefault(unit.trace_id, result.model_copy(update={"trace_id": unit.trace_id}))
    if not rows:
        raise TraceTaggingError(
            f"every {units[0].view} unit failed tagging with bank {bank.bank_id}, for example "
            f"trace {failed[0][0]}: {failed[0][1]}"
        )
    return TagMatrix(
        bank_id=bank.bank_id,
        bank_version=bank.version,
        question_ids=bank.question_ids(),
        view=view,
        rows=tuple(rows.values()),
    ), tuple(failed)


def asked_error_bank(spec: VarianceBuildSpec) -> QuestionBank | None:
    """Return the error questions asked over the observations view, or None without a bank.

    Raises:
        QuestionBankError: The bank is not an error bank, or a named family is not in it.
    """
    if spec.error_bank is None:
        return None
    bank = error_bank(spec.error_bank, families=spec.error_families)
    if spec.error_consumed_families is None:
        return bank_for_view(bank, "observations")
    bank.families(*spec.error_consumed_families)
    return bank


def tag_build_errors(
    store: ArtifactStore,
    spec: VarianceBuildSpec,
    plan: VarianceBuildPlan,
    asked: tuple[QuestionBank, tuple[Unit, ...]] | None,
    tagger: DecisionsTraceTagger,
    *,
    inputs: tuple[ArtifactInput, ...],
    code_revision: str,
) -> SavedErrors | None:
    """Tag world-learning and mining-probe observations with the error bank and save them.

    Args:
        store: Artifact store receiving the records.
        spec: The registration.
        plan: The run's summary, naming the error view's roles and the decision model.
        asked: The asked error bank and its units, or None when no error bank is named.
        tagger: The decision tagger.
        inputs: Inputs recorded on the saved artifacts.
        code_revision: Producing source identity.

    Returns:
        The consumed record, its artifact and the full matrix's artifact (None unless the spec
        names consumed families), or None when no error bank is named.
    """
    if asked is None:
        return None
    bank, units = asked
    consumed = spec.error_consumed_families
    if consumed is None:
        errors = tag_errors(
            units, tagger=tagger, bank=bank, unit_concurrency=tagger.maximum_concurrency
        )
        return (
            errors,
            persist_error_tags(store, errors, code_revision=code_revision, inputs=inputs),
            None,
        )
    matrix, failed = tag_units(tagger, units, bank)
    record = TagMatrixRecord(
        bank_id=bank.bank_id,
        bank_version=bank.version,
        bank_sha256=sha256_json(bank),
        view="observations",
        roles=() if plan.errors is None else plan.errors.roles,
        decision_model=plan.decision_model,
        matrix_sha256=sha256_json(matrix),
        traces=len(matrix.rows),
    )
    full = save_artifact(
        store, MATRIX_TYPE, {"record.json": record, "matrix.json": matrix}, inputs, code_revision
    )
    kept = bank.families(*consumed)
    columns = [matrix.question_ids.index(question) for question in kept.question_ids()]
    narrowed = TagMatrix(
        bank_id=bank.bank_id,
        bank_version=bank.version,
        question_ids=kept.question_ids(),
        view=matrix.view,
        rows=tuple(
            row.model_copy(update={"values": tuple(row.values[i] for i in columns)})
            for row in matrix.rows
        ),
    )
    refused: dict[str, str] = {}
    for trace, error in failed:
        refused.setdefault(trace, error[:2000])
    errors = ErrorTags(
        bank_id=bank.bank_id,
        bank_version=bank.version,
        bank_sha256=sha256_json(kept),
        view=matrix.view,
        threshold=DEFAULT_FIRE_THRESHOLD,
        ambiguity_band=DEFAULT_AMBIGUITY_BAND,
        matrix=narrowed,
        traces=tuple(
            summarize_row(row, narrowed.question_ids, threshold=DEFAULT_FIRE_THRESHOLD)
            for row in narrowed.rows
        ),
        planned=len({unit.trace_id for unit in units}),
        failed=refused,
    )
    saved = persist_error_tags(store, errors, code_revision=code_revision, inputs=inputs)
    return errors, saved, full
