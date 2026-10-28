"""A replay panel stops readably when a provider request fails or runs out of time."""

from pathlib import Path

import pytest
import typer
from rich.console import Console

from exp.cli.evaluation import replay_view
from exp.common.models.client import RequestAdmissionRefused
from exp.common.models.unknown_outcome import DispatchOutcomeUnknown
from exp.common.project import ProjectStore
from exp.common.tasks.facts import FactProvenance, FactValidation, InstanceFact, InstanceFacts
from exp.optimize.evaluation.fidelity.instance_facts import ReplayFacts
from exp.optimize.evaluation.fidelity.replay import ReplayAggregate, ReplyKindCount
from exp.optimize.evaluation.fidelity.runner import FidelitySpec, PreparedFidelity
from exp.runtime.models.providers.async_transport import ProviderDeadlineExceeded
from exp.runtime.models.providers.transport import ProviderTransportError
from exp.runtime.sessions.dispatch_ledger import DispatchCapExhaustedError


@pytest.mark.parametrize(
    "failure",
    [
        ProviderTransportError("provider returned HTTP 400", status_code=400),
        ProviderDeadlineExceeded("provider request deadline exceeded"),
    ],
    ids=["transport", "deadline"],
)
def test_a_provider_stop_keeps_completed_traces_and_names_the_resume_flag(
    failure: Exception, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The stopped trace is never bought again silently, whichever way the request failed."""

    def stop(*_args: object, **_kwargs: object) -> None:
        """Fail the panel the way the provider failed."""
        raise failure

    monkeypatch.setattr(replay_view, "require_resumable", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(replay_view, "execute_panel", stop)
    console = Console(record=True, width=200)
    with pytest.raises(typer.Exit) as stopped:
        replay_view.run_panel(console, ProjectStore(tmp_path, "panel"), ())
    assert stopped.value.exit_code == 1
    output = console.export_text()
    assert f"error a provider request failed: {failure}" in output
    assert "--retry-incomplete" in output


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        (DispatchCapExhaustedError("cap reached"), "spent its physical-attempt ceiling"),
        (DispatchOutcomeUnknown("no saved response"), "has an unknown outcome"),
        (RequestAdmissionRefused("lease is not held"), "refused before dispatch"),
    ],
    ids=["cap", "unknown", "refused"],
)
def test_every_infrastructure_stop_is_explained_and_never_scored(
    failure: BaseException, reason: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Stops beyond provider failures end the panel the same way, naming what stopped."""

    def stop(*_args: object, **_kwargs: object) -> None:
        """Fail the panel with the infrastructure stop."""
        raise failure

    monkeypatch.setattr(replay_view, "require_resumable", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(replay_view, "execute_panel", stop)
    console = Console(record=True, width=200)
    with pytest.raises(typer.Exit):
        replay_view.run_panel(console, ProjectStore(tmp_path, "panel"), ())
    output = console.export_text()
    assert reason in output and "--retry-incomplete" in output


def test_the_headline_states_lineage_uncertainty_reply_kinds_judge_coverage_and_time() -> None:
    """Pooled turns come with the per-lineage rate, its interval, and every denominator."""
    total = ReplayAggregate(
        traces=3,
        exact=5,
        steps=9,
        fields_matched=20,
        fields_total=30,
        judge_mean=3.5,
        judged_traces=2,
        judge_failures=1,
        kinds=(
            ReplyKindCount(kind="trivial", exact=4, steps=4),
            ReplyKindCount(kind="text", exact=0, steps=0),
            ReplyKindCount(kind="structured", exact=1, steps=5),
        ),
        lineages=3,
        lineage_exact=0.5,
        lineage_interval=(0.1, 0.9),
        replay_seconds=(30.0, 90.0, 60.0),
    )
    line = " ".join(replay_view.aggregate_line(total, 4).split())
    assert "5/9 replies exact (56%)" in line
    assert "judge 3.5/5 (2/3 traces judged, 1 judge failure)" in line
    assert "3/4 traces" in line
    assert "per-lineage mean exact 50% (95% interval 10% to 90%, 3 lineages)" in line
    assert "exact by recorded reply: trivial 4/4 (100%) · structured 1/5 (20%)" in line
    assert "text 0/0" not in line, "a kind with no turns is left out"
    assert "under 60 characters" in line
    assert "replay time per trace 1m00s median, 1m30s longest" in line


def test_an_unjudged_panel_says_how_many_traces_the_judge_covered() -> None:
    """Without an assessed score the judge reads as a dash, still with its coverage."""
    total = ReplayAggregate(
        traces=2, exact=0, steps=2, fields_matched=0, fields_total=0, judge_mean=None
    )
    assert "judge - (0/2 traces judged, 0 judge failures)" in replay_view.aggregate_line(total, 2)


def test_a_panel_states_which_traces_were_measured_given_supplied_facts() -> None:
    """Exactness given supplied instance facts is labelled, and its absence is said too."""
    fact = InstanceFact(
        key="orders/#W1/status",
        value="pending",
        visibility="conditioning",
        provenance=FactProvenance(
            adapter="fixture",
            source="order service",
            validation=FactValidation(status="validated", method="read"),
            state_version=0,
        ),
    )
    facts = ReplayFacts(facts=InstanceFacts(facts=(fact,)))
    with_facts = PreparedFidelity.model_construct(spec=FidelitySpec.model_construct(facts=facts))
    bare = PreparedFidelity.model_construct(spec=FidelitySpec.model_construct(facts=None))
    line = replay_view.facts_line((with_facts, bare))
    assert "supplied to 1/2 traces (1 validated facts)" in line
    assert "measured given those facts" in line
    assert "none supplied" in replay_view.facts_line((bare,))
