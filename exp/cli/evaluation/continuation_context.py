"""Invocation context and the one spend-consent step shared by ``--continuations`` handlers."""

from dataclasses import dataclass
from pathlib import Path

from rich.console import Console

from exp.cli.evaluation.spend import review_cost
from exp.cli.shared.consent import require_spend_consent
from exp.common.project import ProjectStore
from exp.optimize.evaluation.executable.costs import CostedSpec


@dataclass(frozen=True)
class ContinuationContext:
    """One validated ``exp eval PROJECT --continuations SPEC`` invocation.

    Every action handler receives this context with its own typed payload, so options, the
    project store and the execution revision are resolved once by the dispatcher.

    Attributes:
        store: Project store the action reads from and publishes into.
        console: Command-owned output console.
        revision: Installed execution revision recorded in published artifacts.
        root: EXP root that owns the configured spend warning budget.
        project: Project name exactly as typed on the command line.
        spec_path: The ``--continuations`` spec file, named in spend prompts.
        yes: Explicit authorization of the displayed estimate, including budget overruns.
        non_interactive: Fail rather than prompt when paid work lacks authorization.
        dry_run: Validate inputs without provider calls, runtime execution or publication.
    """

    store: ProjectStore
    console: Console
    revision: str
    root: Path
    project: str
    spec_path: Path
    yes: bool
    non_interactive: bool
    dry_run: bool

    @property
    def command(self) -> str:
        """The command identity shown to the operator in spend prompts.

        Returns:
            ``exp eval PROJECT --continuations SPEC`` with the real spec path.
        """
        return f"exp eval {self.project} --continuations {self.spec_path}"

    def consent(self, spec: CostedSpec) -> bool:
        """Show the spec's credential-free cost inventory and request spend consent once.

        The estimate covers known components only; unpriced dependencies stay visible and the
        execution is never presented as capped, so explicit authorization is always required.

        Args:
            spec: Paid action payload whose complete cost inventory is reviewed.

        Returns:
            True when execution is authorized, or False after an interactive decline.

        Raises:
            typer.BadParameter: Settings or cost arithmetic are invalid.
            typer.Exit: No explicit confirmation is available in a noninteractive session.
        """
        return require_spend_consent(
            self.console,
            root=self.root,
            yes=self.yes,
            estimated_cost_usd=review_cost(self.console, self.store, spec),
            cost_is_upper_bound=False,
            command=self.command,
            non_interactive=self.non_interactive,
        )
