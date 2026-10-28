"""The trajectory-count question as select-on-development, confirm-on-untouched-lineages."""

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

import exp.optimize.evaluation.learning.sample_count as sample_count
from exp.common.core.artifacts import JsonObject
from exp.common.evaluations.uncertainty import wilson_interval
from exp.optimize.evaluation.audit.lifecycle import open_held_out_audit
from exp.optimize.evaluation.audit.lifecycle_test import clock, run
from exp.optimize.evaluation.audit.panel import plan_held_out_panel, saved_held_out_results
from exp.optimize.evaluation.audit.panel_test import (
    REVISION,
    USERS,
    SimulatedProvider,
    build_project,
    episode,
    learn_fit,
)
from exp.optimize.evaluation.audit.receipts import AuditReceipt, receipt_reference, record_report
from exp.optimize.evaluation.learning.sample_count import (
    CheckpointMeasurement,
    LineageMeasurement,
    SampleCountRegistration,
    SampleCountSelection,
    audit_measurement,
    confirm_sample_count,
    register_sample_count_study,
    select_sample_count,
)
from exp.optimize.evaluation.learning.traces import fit_cases

REGISTERED = datetime(2026, 10, 20, 9, tzinfo=UTC)
CONFIGURATION = "c" * 64


def registration(
    *, criterion: str = "point_estimate", lineages: tuple[str, ...] = ("dev-1", "dev-2")
) -> SampleCountRegistration:
    """A registered grid of 4, 8 and 16 trajectories at a 75 percent threshold."""
    return SampleCountRegistration.model_validate(
        {
            "project": "retail",
            "memory_lineage": "fit-memory",
            "trace_order_sha256": "a" * 64,
            "grid": (4, 8, 16),
            "threshold": 0.75,
            "criterion": criterion,
            "development_lineages": lineages,
            "registered_at": REGISTERED,
            "receipts_before": 0,
        }
    )


def measured(
    count: int,
    shares: dict[str, tuple[int, int]],
    *,
    sequence: int,
    minutes: int = 10,
    receipt: str | None = None,
    basis: str = "development-re-audit",
    configuration: str = CONFIGURATION,
) -> CheckpointMeasurement:
    """One audit of the checkpoint that learned ``count`` trajectories."""
    return CheckpointMeasurement.model_validate(
        {
            "receipt_id": receipt or f"receipt-{count}",
            "sequence": sequence,
            "opened_at": REGISTERED + timedelta(minutes=minutes),
            "basis": basis,
            "headline": "independent" if basis == "independent" else "development re-audit",
            "memory_lineage": "fit-memory",
            "memory_version": count,
            "trajectories": count,
            "configuration_sha256": configuration,
            "lineages": [
                {"lineage": name, "traces": 1, "measured": 1, "exact": exact, "steps": steps}
                for name, (exact, steps) in sorted(shares.items())
            ],
        }
    )


DEV_4 = measured(4, {"dev-1": (5, 10), "dev-2": (6, 10)}, sequence=1, minutes=1)
DEV_8 = measured(8, {"dev-1": (8, 10), "dev-2": (9, 10)}, sequence=2, minutes=2)
DEV_16 = measured(16, {"dev-1": (10, 10), "dev-2": (10, 10)}, sequence=3, minutes=3)
SELECTED_AFTER = 5  # latest receipt sequence when the fixture selection was saved


def saved_selection(rows: tuple[CheckpointMeasurement, ...]) -> SampleCountSelection:
    """The selection over ``rows`` as recorded after receipt ``SELECTED_AFTER``."""
    selection = select_sample_count(registration(), rows)
    return selection.model_copy(update={"receipts_before": SELECTED_AFTER})


def test_selection_takes_the_first_registered_count_meeting_the_criterion() -> None:
    """Grid order decides; a later, better count does not replace the first that qualified."""
    selection = select_sample_count(registration(), (DEV_16, DEV_4, DEV_8))
    assert selection.status == "selected" and selection.selected_count == 8
    assert selection.memory_version == 8 and selection.configuration_sha256 == CONFIGURATION
    assert [row["met"] for row in selection.evaluated] == [False, True]
    assert set(selection.development_receipts) == {"receipt-4", "receipt-8", "receipt-16"}
    assert selection.last_development_sequence == DEV_16.sequence
    assert selection.receipts_before is None, "only a saved selection records its order"
    strict = select_sample_count(
        registration(criterion="wilson_lower_bound"), (DEV_4, DEV_8, DEV_16)
    )
    assert strict.status == "threshold_not_met", "two lineages cannot bound accuracy above 75%"
    lower = wilson_interval(1.0, 2)[0]
    assert lower < 0.75


def test_selection_stops_at_a_missing_or_incomplete_grid_count() -> None:
    """Skipping an unmeasured or partly failed count would be choosing after the fact."""
    stopped = select_sample_count(registration(), (DEV_4, DEV_16))
    assert stopped.status == "not_measured" and stopped.stopped_at == 8
    partial = DEV_8.model_copy(
        update={
            "lineages": (
                DEV_8.lineages[0],
                LineageMeasurement(lineage="dev-2", traces=1, measured=0, exact=0, steps=0),
            )
        }
    )
    incomplete = select_sample_count(registration(), (DEV_4, partial, DEV_16))
    assert incomplete.status == "incomplete" and incomplete.stopped_at == 8
    assert incomplete.selected_count is None


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ((DEV_4.model_copy(update={"memory_lineage": "all-memory"}),), "another memory"),
        ((measured(4, {"dev-1": (5, 10)}, sequence=1),), "exactly the development lineages"),
        ((DEV_4, DEV_4), "measured once"),
        ((measured(5, {"dev-1": (5, 10), "dev-2": (5, 10)}, sequence=1),), "measured once"),
        ((DEV_4, DEV_8.model_copy(update={"configuration_sha256": "d" * 64})), "frozen config"),
    ],
)
def test_development_evidence_must_follow_the_registration(
    rows: tuple[CheckpointMeasurement, ...], message: str
) -> None:
    """Audits before registration, of other memories, lineages or configurations are refused."""
    with pytest.raises(ValueError, match=message):
        select_sample_count(registration(), rows)


@pytest.mark.parametrize("clock_skew", [timedelta(days=-1), timedelta(0), timedelta(days=1)])
def test_registration_order_is_the_receipt_sequence_whatever_the_clocks_claim(
    clock_skew: timedelta,
) -> None:
    """An audit minted before registration is refused even when its clock claims later, and an
    audit minted after it counts even when its clock claims earlier."""
    registered = registration().model_copy(update={"receipts_before": 3})
    skewed = REGISTERED + clock_skew
    early = DEV_4.model_copy(update={"sequence": 3, "opened_at": skewed + timedelta(hours=2)})
    with pytest.raises(ValueError, match="predates the registration"):
        select_sample_count(registered, (early,))
    late = DEV_4.model_copy(update={"sequence": 4, "opened_at": skewed - timedelta(hours=2)})
    assert select_sample_count(registered, (late,)).status == "not_measured"


def confirmation(**update: object) -> CheckpointMeasurement:
    """An independent audit of the frozen 8-trajectory checkpoint on untouched lineages."""
    row = measured(
        8,
        {"held-1": (9, 10), "held-2": (8, 10), "held-3": (10, 10)},
        sequence=SELECTED_AFTER + 1,
        minutes=30,
        receipt="receipt-confirm",
        basis="independent",
    )
    return row.model_copy(update=update)


def test_confirmation_reports_count_metric_uncertainty_and_failures() -> None:
    """The frozen count is reported from untouched lineages, beside its optimistic development."""
    selection = saved_selection((DEV_4, DEV_8, DEV_16))
    report = confirm_sample_count(registration(), selection, confirmation())
    assert report["trajectories"] == 8 and report["metric"] == "exact_replies"
    measured_confirmation = report["confirmation"]
    assert isinstance(measured_confirmation, dict)
    assert measured_confirmation["independent_units"] == 3
    assert measured_confirmation["estimate"] == pytest.approx(0.9)
    assert measured_confirmation["wilson_95"] == list(wilson_interval(0.9, 3))
    assert measured_confirmation["coverage"] == {"traces": 3, "measured": 3}
    assert measured_confirmation["unmeasured_lineages"] == 0
    assert measured_confirmation["headline"] == "independent"
    assert report["criterion_met_on_confirmation"] is True
    development = report["development_at_selected_count"]
    assert isinstance(development, dict) and development["estimate"] == pytest.approx(0.85)
    assert "optimistic" in str(report["scope"])
    lost = confirmation(
        lineages=(
            *confirmation().lineages[:2],
            LineageMeasurement(lineage="held-3", traces=1, measured=0, exact=0, steps=0),
        )
    )
    partial = confirm_sample_count(registration(), selection, lost)["confirmation"]
    assert isinstance(partial, dict) and partial["unmeasured_lineages"] == 1
    assert partial["lineage_bounds"] == pytest.approx([(0.9 + 0.8) / 3, (0.9 + 0.8 + 1) / 3])


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"receipt_id": "receipt-8"}, "cannot be reported as confirmation"),
        (
            {
                "lineages": (
                    LineageMeasurement(lineage="dev-1", traces=1, measured=1, exact=9, steps=10),
                )
            },
            "reused lineages: dev-1",
        ),
        ({"basis": "development-re-audit"}, "independent audit"),
        ({"notices": 1}, "invalidated"),
        ({"sequence": 2}, "after every development"),
        ({"sequence": SELECTED_AFTER}, "after the saved sample-count selection"),
        ({"memory_version": 16, "trajectories": 16}, "frozen selected checkpoint"),
        ({"configuration_sha256": "d" * 64}, "frozen selected checkpoint"),
    ],
)
def test_a_development_selected_checkpoint_is_never_reported_as_confirmation(
    update: JsonObject, message: str
) -> None:
    """Reused receipts or lineages, re-audits, early or changed confirmations are refused."""
    selection = saved_selection((DEV_4, DEV_8, DEV_16))
    with pytest.raises(ValueError, match=message):
        confirm_sample_count(registration(), selection, confirmation(**update))


def test_nothing_is_confirmed_without_a_selection() -> None:
    """A walk that stopped or never met the threshold has no count to confirm."""
    stopped = select_sample_count(registration(), (DEV_4,))
    with pytest.raises(ValueError, match="no development-selected count"):
        confirm_sample_count(registration(), stopped, confirmation())
    foreign = SampleCountSelection(registration="another", status="selected", selected_count=8)
    with pytest.raises(ValueError, match="another registration"):
        confirm_sample_count(registration(), foreign, confirmation())
    unsaved = select_sample_count(registration(), (DEV_4, DEV_8, DEV_16))
    with pytest.raises(ValueError, match="confirm a saved selection"):
        confirm_sample_count(registration(), unsaved, confirmation())


@pytest.mark.parametrize("clock_skew", [timedelta(days=-1), timedelta(0), timedelta(days=1)])
def test_selection_order_is_the_receipt_sequence_whatever_the_clocks_claim(
    clock_skew: timedelta,
) -> None:
    """A confirmation minted before the saved selection is refused even when its clock claims
    a later time, and one minted after it is accepted even when its clock claims earlier."""
    selection = saved_selection((DEV_4, DEV_8, DEV_16))
    selected_at = REGISTERED + timedelta(minutes=40) + clock_skew
    before = confirmation(sequence=SELECTED_AFTER, opened_at=selected_at + timedelta(days=2))
    with pytest.raises(ValueError, match="minted before the selection"):
        confirm_sample_count(registration(), selection, before)
    after = confirmation(opened_at=selected_at - timedelta(days=2))
    report = confirm_sample_count(registration(), selection, after)
    assert report["trajectories"] == 8


def test_registration_requires_an_increasing_grid_and_sorted_lineages() -> None:
    """The grid and its order are fixed before any result."""
    with pytest.raises(ValidationError, match="strictly increasing"):
        SampleCountRegistration.model_validate({**registration().model_dump(), "grid": (8, 4)})
    with pytest.raises(ValidationError, match="unique and sorted"):
        registration(lineages=("dev-2", "dev-1"))
    assert registration().identity != registration(lineages=("dev-1",)).identity


def skewed(offset: timedelta) -> Callable[[], datetime]:
    """An aware clock running ``offset`` away from this host's clock."""
    return lambda: datetime.now(UTC) + offset


def test_the_procedure_runs_on_held_out_audits_of_the_learned_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Register, select on a development audit, then confirm on a fresh independent audit.

    The registration, selection and early-audit hosts run days fast.
    The order that counts is the receipt sequence: an audit minted before the saved selection is
    refused although its clock claims a later time, and audits minted after the registration and
    the selection count although their clocks claim earlier times. (Audit hosts never run slow
    here: a run's deadline is wall time on the host that executes it.)
    """
    provider = SimulatedProvider([episode(i) for i in range(USERS)], learnable=True)
    root, store, _ = build_project(tmp_path, monkeypatch, provider)
    learn_fit(root, store)
    fit = len(fit_cases(store))
    development = plan_held_out_panel(
        store, 1, profile="memory", code_revision="fixture", clock=clock
    ).lineages
    with pytest.raises(ValueError, match="fit traces"):
        register_sample_count_study(
            store,
            grid=(fit + 1,),
            threshold=0.75,
            criterion="point_estimate",
            development_lineages=development,
            registered_at=clock(),
            code_revision=REVISION,
        )
    registered, artifact = register_sample_count_study(
        store,
        grid=(fit,),
        threshold=0.75,
        criterion="point_estimate",
        development_lineages=development,
        registered_at=clock() + timedelta(days=3),
        code_revision=REVISION,
    )
    assert artifact.artifact_id == registered.identity and registered.receipts_before == 0
    replayed, _ = register_sample_count_study(
        store,
        grid=(fit,),
        threshold=0.75,
        criterion="point_estimate",
        development_lineages=development,
        registered_at=registered.registered_at,
        code_revision=REVISION,
    )
    assert replayed == registered

    # Development audit minted after registration, on a clock days behind the registration's.
    with open_held_out_audit(
        store, 1, profile="memory", code_revision=REVISION, clock=clock
    ) as opened:
        run(store, opened)
        dev = audit_measurement(
            opened.view, opened.panel, saved_held_out_results(store, opened.panel)
        )
        # Development results are inspected per trace, so later audits move to fresh content.
        record_report(
            store, opened.receipt, opened.panel, kind="evaluation", detail_shown=True, clock=clock
        )
    assert dev.opened_at < registered.registered_at and dev.sequence == 1
    assert dev.trajectories == fit and tuple(row.lineage for row in dev.lineages) == development

    # A confirmation audit minted before the selection is saved, on a clock days ahead.
    # It asks for two traces, so the later one-trace request opens a new receipt, not this one.
    with open_held_out_audit(
        store, 2, profile="memory", code_revision=REVISION, clock=skewed(timedelta(days=3))
    ) as early:
        run(store, early)
    selection, selection_ref = sample_count.record_sample_count_selection(
        store,
        sample_count.SampleCountSelect(
            resolution="select", registration=artifact, development=(dev.receipt_id,)
        ),
        created_at=clock() + timedelta(days=2),
        code_revision=REVISION,
    )
    assert selection_ref is not None
    assert selection.status == "selected" and selection.selected_count == fit
    assert selection.receipts_before == early.receipt.sequence == 2
    assert (
        early.receipt.opened_at
        > store.artifacts.read(selection_ref.artifact_id).manifest.created_at
    )
    cited = set(store.artifacts.read(selection_ref.artifact_id).manifest.inputs)
    assert cited == {artifact, receipt_reference(store, dev.receipt_id)}
    with pytest.raises(ValueError, match="minted before the selection"):
        sample_count.record_sample_count_confirmation(
            store,
            sample_count.SampleCountConfirm(
                resolution="confirm", selection=selection_ref, confirmation=early.receipt.receipt_id
            ),
            created_at=clock(),
            code_revision=REVISION,
        )
    with pytest.raises(ValueError, match="cannot be reported as confirmation"):
        confirm_sample_count(registered, selection, dev)

    # Minted after the selection, on a clock that claims it opened before the selection.
    with open_held_out_audit(
        store, 1, profile="memory", code_revision=REVISION, clock=clock
    ) as fresh:
        run(store, fresh)
        held = audit_measurement(
            fresh.view, fresh.panel, saved_held_out_results(store, fresh.panel)
        )
    assert held.basis == "independent" and not set(development) & {
        row.lineage for row in held.lineages
    }
    selected_at = store.artifacts.read(selection_ref.artifact_id).manifest.created_at
    assert held.opened_at < selected_at and held.opened_at < early.receipt.opened_at
    assert held.sequence == 3
    assert held.configuration_sha256 == dev.configuration_sha256
    report, confirmed = sample_count.record_sample_count_confirmation(
        store,
        sample_count.SampleCountConfirm(
            resolution="confirm", selection=selection_ref, confirmation=held.receipt_id
        ),
        created_at=clock(),
        code_revision=REVISION,
    )
    assert report["trajectories"] == fit and report["criterion_met_on_confirmation"] is True
    assert confirmed is not None
    assert set(store.artifacts.read(confirmed.artifact_id).manifest.inputs) == {
        selection_ref,
        receipt_reference(store, held.receipt_id),
    }

    with open_held_out_audit(
        store, 1, profile="memory", code_revision=REVISION, clock=clock, include_inspected=True
    ) as reaudit:
        run(store, reaudit)
        again = audit_measurement(
            reaudit.view, reaudit.panel, saved_held_out_results(store, reaudit.panel)
        )
    with pytest.raises(ValueError, match="cannot be reported|independent audit"):
        confirm_sample_count(registered, selection, again)


def test_only_the_first_audit_after_the_selection_can_confirm_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second confirmation audit can be neither previewed nor saved in place of the first.

    Otherwise an operator could open several independent audits of the frozen checkpoint, read
    each score through a dry run, and save only a passing one: a search on the outcome.
    """
    provider = SimulatedProvider([episode(i) for i in range(USERS)], learnable=True)
    root, store, _ = build_project(tmp_path, monkeypatch, provider)
    learn_fit(root, store)
    development = plan_held_out_panel(
        store, 1, profile="memory", code_revision="fixture", clock=clock
    ).lineages
    _, registration_ref = register_sample_count_study(
        store,
        grid=(len(fit_cases(store)),),
        threshold=0.75,
        criterion="point_estimate",
        development_lineages=development,
        registered_at=clock(),
        code_revision=REVISION,
    )

    def audited() -> AuditReceipt:
        """Open, run and show one independent one-trace audit of the final memory."""
        with open_held_out_audit(
            store, 1, profile="memory", code_revision=REVISION, clock=clock
        ) as opened:
            run(store, opened)
            record_report(
                store,
                opened.receipt,
                opened.panel,
                kind="evaluation",
                detail_shown=True,
                clock=clock,
            )
        return opened.receipt

    _, selection_ref = sample_count.record_sample_count_selection(
        store,
        sample_count.SampleCountSelect(
            resolution="select", registration=registration_ref, development=(audited().receipt_id,)
        ),
        created_at=clock(),
        code_revision=REVISION,
    )
    assert selection_ref is not None
    first, second = audited(), audited()
    assert first.basis == second.basis == "independent"
    for dry_run in (True, False):
        with pytest.raises(ValueError, match=f"first independent audit .*\\({first.receipt_id}\\)"):
            sample_count.record_sample_count_confirmation(
                store,
                sample_count.SampleCountConfirm(
                    resolution="confirm", selection=selection_ref, confirmation=second.receipt_id
                ),
                created_at=clock(),
                code_revision=REVISION,
                dry_run=dry_run,
            )
    _, confirmed = sample_count.record_sample_count_confirmation(
        store,
        sample_count.SampleCountConfirm(
            resolution="confirm", selection=selection_ref, confirmation=first.receipt_id
        ),
        created_at=clock(),
        code_revision=REVISION,
    )
    assert confirmed is not None
    cited = store.artifacts.read(confirmed.artifact_id).manifest.inputs
    assert receipt_reference(store, first.receipt_id) in cited
