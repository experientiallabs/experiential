"""Every continuation spend prompt names the spec file the user actually passed."""

from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from exp.cli.evaluation import continuation_context
from exp.cli.evaluation.continuation_context import ContinuationContext
from exp.common.core.artifacts import ArtifactInput
from exp.common.project import ProjectStore


def test_consent_reviews_the_spec_and_names_the_real_spec_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one consent step reviews the payload's cost and shows the exact command."""
    spec = ArtifactInput(artifact_id="study", sha256="a" * 64)
    reviewed: list[ArtifactInput] = []
    requests: list[dict[str, str | bool | float | None]] = []

    def review(console: Console, store: ProjectStore, payload: ArtifactInput) -> float:
        """Record the reviewed payload and return a fixed estimate.

        Args:
            console: Command console, unused.
            store: Project store, unused.
            payload: Payload whose cost is reviewed.

        Returns:
            A fixed known estimate.
        """
        reviewed.append(payload)
        return 0.25

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

    monkeypatch.setattr(continuation_context, "review_cost", review)
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
    )
    assert context.consent(spec) is False
    assert reviewed == [spec]
    assert requests == [
        {
            "yes": True,
            "estimate": 0.25,
            "capped": False,
            "command": f"exp eval case --continuations {spec_path}",
            "non_interactive": True,
        }
    ]
