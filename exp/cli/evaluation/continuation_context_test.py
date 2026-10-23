"""Every continuation spend prompt names the spec file the user actually passed."""

from dataclasses import replace
from io import StringIO
from pathlib import Path
from typing import cast

import pytest
from rich.console import Console

from exp.cli.evaluation import continuation_context
from exp.cli.evaluation.continuation_context import ContinuationContext
from exp.common.core.artifacts import ArtifactInput
from exp.common.project import ProjectStore
from exp.optimize.evaluation.executable.costs import CostedSpec
from exp.optimize.evaluation.executable.model_costs import CostComponent
from exp.optimize.evaluation.spending import SpendPolicy, operation_spend_ceiling
from exp.runtime.models.spend_ceiling import UnpricedDispatchRefused, operation_spend_scope

UNPRICED_SPEND = SpendPolicy(unpriced_attempts=1000)
"""Explicit unpriced-attempt ceiling for CLI fixtures whose catalogs carry no prices."""


def test_consent_reviews_the_spec_and_names_the_real_spec_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one consent step reviews the payload's cost and shows the exact command."""
    # review_ceiling is replaced below, so any costed model can stand in for a spec.
    spec = cast("CostedSpec", ArtifactInput(artifact_id="study", sha256="a" * 64))
    reviewed: list[ArtifactInput] = []
    requests: list[dict[str, str | bool | float | None]] = []

    def review(
        console: Console, store: ProjectStore, payload: ArtifactInput
    ) -> tuple[float, tuple[CostComponent, ...]]:
        """Record the reviewed payload and return a fixed fully priced estimate.

        Args:
            console: Command console, unused.
            store: Project store, unused.
            payload: Payload whose cost is reviewed.

        Returns:
            A fixed known subtotal and no unpriced component.
        """
        reviewed.append(payload)
        return 0.25, ()

    def consent(
        console: Console,
        *,
        root: Path,
        yes: bool,
        estimated_cost_usd: float | None,
        cost_is_upper_bound: bool,
        command: str,
        non_interactive: bool,
    ) -> bool:
        """Record the consent request and decline it.

        Args:
            console: Command console, unused.
            root: EXP root that owns the settings.
            yes: Explicit authorization flag.
            estimated_cost_usd: Reviewed estimate.
            cost_is_upper_bound: Whether the estimate is an enforced allowance.
            command: Command identity shown to the operator.
            non_interactive: Whether prompting is forbidden.

        Returns:
            False, declining the spend.
        """
        requests.append(
            {
                "yes": yes,
                "estimate": estimated_cost_usd,
                "capped": cost_is_upper_bound,
                "command": command,
                "non_interactive": non_interactive,
            }
        )
        return False

    monkeypatch.setattr(continuation_context, "review_ceiling", review)
    monkeypatch.setattr(continuation_context, "require_spend_consent", consent)
    spec_path = tmp_path / "chosen-by-user.json"
    context = ContinuationContext(
        store=ProjectStore(tmp_path, "case"),
        console=Console(file=StringIO()),
        revision="test",
        root=tmp_path,
        project="case",
        spec_path=spec_path,
        yes=True,
        non_interactive=True,
        dry_run=False,
        operation="operation-fixture",
        spend=SpendPolicy(),
    )
    assert context.consent(spec) is False
    assert reviewed == [spec]
    assert requests == [
        {
            "yes": True,
            "estimate": 0.25,
            "capped": True,
            "command": f"exp eval case --continuations {spec_path}",
            "non_interactive": True,
        }
    ]


@pytest.mark.parametrize("explicit_cap, expected", [(10.0, 10.0), (None, 5.0)])
def test_stages_share_explicit_operation_limits_and_resume_without_growth(
    tmp_path: Path, explicit_cap: float | None, expected: float
) -> None:
    """Stage consent preserves one explicit cap while estimates sum only when no cap is set."""
    store = ProjectStore(tmp_path, "case")
    context = ContinuationContext(
        store=store,
        console=Console(file=StringIO()),
        revision="test",
        root=tmp_path,
        project="case",
        spec_path=tmp_path / "operation.json",
        yes=True,
        non_interactive=True,
        dry_run=False,
        operation="operation-fixture",
        spend=SpendPolicy(ceiling_usd=explicit_cap, unpriced_attempts=2),
    )
    unpriced = (CostComponent(name="actor", attempts=2, usd=None),)
    with operation_spend_scope():
        assert context.consent_ceiling(stage="first", known_usd=2.0, unpriced=unpriced)
        assert context.consent_ceiling(stage="second", known_usd=3.0, unpriced=unpriced)
        ceiling = operation_spend_ceiling(store, context.operation)
        summary = ceiling.summary()
        assert summary.ceiling_usd == expected
        assert summary.unpriced_attempt_ceiling == 2
        # Re-consenting an earlier stage is a resume, not another grant.
        assert context.consent_ceiling(stage="first", known_usd=2.0, unpriced=unpriced)
        summary = ceiling.summary()
        assert summary.ceiling_usd == expected
        assert summary.unpriced_attempt_ceiling == 2
        # Even different paid roles cannot multiply the unpriced permission.
        for alias in ("actor", "simulator"):
            ceiling.reserve(
                alias=alias, url="https://fixture.invalid/v1/chat/completions", payload={}
            )
        with pytest.raises(UnpricedDispatchRefused):
            ceiling.reserve(
                alias="judge", url="https://fixture.invalid/v1/chat/completions", payload={}
            )


def test_explicit_policy_can_replace_estimates_on_the_same_operation(tmp_path: Path) -> None:
    """Changing an operation's policy overrides prior grants and preserves resumed identity."""
    store = ProjectStore(tmp_path, "case")
    context = ContinuationContext(
        store=store,
        console=Console(file=StringIO()),
        revision="test",
        root=tmp_path,
        project="case",
        spec_path=tmp_path / "operation.json",
        yes=True,
        non_interactive=True,
        dry_run=False,
        operation="operation-fixture",
        spend=SpendPolicy(),
    )
    with operation_spend_scope():
        assert context.consent_ceiling(stage="estimated", known_usd=2.0)
        ceiling = operation_spend_ceiling(store, context.operation)
        assert ceiling.summary().ceiling_usd == 2.0
        context = replace(context, spend=SpendPolicy(ceiling_usd=1.0, unpriced_attempts=2))
        for stage in ("explicit-first", "explicit-second", "explicit-first"):
            assert context.consent_ceiling(stage=stage, known_usd=9.0)
            summary = ceiling.summary()
            assert summary.ceiling_usd == 1.0
            assert summary.unpriced_attempt_ceiling == 2
        context = replace(context, spend=SpendPolicy(ceiling_usd=3.0, unpriced_attempts=1))
        assert context.consent_ceiling(stage="raised", known_usd=9.0)
        assert ceiling.summary().ceiling_usd == 3.0
        assert ceiling.summary().unpriced_attempt_ceiling == 1
