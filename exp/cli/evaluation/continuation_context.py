"""Invocation context, spend consent and prepared-operation output for continuation handlers."""

import json
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.text import Text

from exp.cli.evaluation.spend import review_ceiling
from exp.cli.shared.consent import require_spend_consent
from exp.common.core.artifacts import JsonObject, sha256_json
from exp.common.project import ProjectStore
from exp.optimize.evaluation.executable.costs import CostedSpec
from exp.optimize.evaluation.executable.model_costs import CostComponent
from exp.optimize.evaluation.spending import (
    SpendPolicy,
    operation_spend_ceiling,
    save_spend_receipt,
)
from exp.runtime.models.spend_ceiling import install_spend_ceiling


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
        operation: Stable identity of this operation and its resumes, naming its spend ledger.
        spend: Operator spend-ceiling choices from the operation file.
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
    operation: str
    spend: SpendPolicy

    @property
    def command(self) -> str:
        """The command identity shown to the operator in spend prompts.

        Returns:
            ``exp eval PROJECT --continuations SPEC`` with the real spec path.
        """
        return f"exp eval {self.project} --continuations {self.spec_path}"

    def consent(self, spec: CostedSpec) -> bool:
        """Show the spec's cost inventory, ask once, then enforce the consented ceiling.

        The ceiling is the known-rate subtotal of every priced component, or the operation
        file's explicit ``spend.ceiling_usd``. Unpriced components cannot be bounded, so the
        operation needs an explicit ``spend.unpriced_attempts`` ceiling for them.

        Args:
            spec: Paid action payload whose complete cost inventory is reviewed.

        Returns:
            True when execution is authorized, or False after an interactive decline.

        Raises:
            typer.BadParameter: Settings or cost arithmetic are invalid.
            typer.Exit: No explicit confirmation is available in a noninteractive session.
            ValueError: A paid unpriced component has no explicit unpriced-attempt ceiling.
        """
        known, unpriced = review_ceiling(self.console, self.store, spec)
        return self.consent_ceiling(
            stage=sha256_json(spec.model_dump(mode="json")), known_usd=known, unpriced=unpriced
        )

    def consent_ceiling(
        self,
        *,
        stage: str,
        known_usd: float | None,
        unpriced: tuple[CostComponent, ...] = (),
        remaining: bool = False,
    ) -> bool:
        """Ask once for one stage's ceiling and install it as the operation's ambient ceiling.

        ``--yes`` authorizes the displayed ceiling. The authorization is appended to the
        operation's durable ledger, replacing an earlier consent of the same stage on resume,
        and recorded in the operation's spend receipt. Explicit dollar and unpriced limits
        authorize the whole operation once, independently of how many stages request consent.

        Args:
            stage: Identity of the consented cost inventory.
            known_usd: Known-rate subtotal, or None when the stage has no cost inventory and
                needs an explicit ``spend.ceiling_usd``.
            unpriced: Unpriced components shown beside the ceiling.
            remaining: The estimate prices only work not yet saved, so the spend this
                operation already accounted is added to the authorized stage ceiling.

        Returns:
            True when execution is authorized, or False after an interactive decline.

        Raises:
            ValueError: The ceiling or the unpriced-attempt allowance cannot be determined.
        """
        allowance = self.spend.unpriced_attempts
        ceiling = self.spend.ceiling_usd if self.spend.ceiling_usd is not None else known_usd
        if ceiling is None:
            raise ValueError(
                "this operation has no cost inventory; set spend.ceiling_usd (and, for unpriced "
                "roles, spend.unpriced_attempts) in the operation file to authorize a ceiling"
            )
        paid = sorted({row.name for row in unpriced if row.attempts > 0})
        if paid and not allowance:
            raise ValueError(
                f"unpriced roles ({', '.join(paid)}) cannot be bounded by a spend ceiling; add "
                "their prices to the model catalog, or set spend.unpriced_attempts in the "
                "operation file to allow that many unpriced physical attempts"
            )
        self.console.print(
            f"Spend ceiling: ${ceiling:.6f} across every paid role of this operation"
            + (f", plus at most {allowance} unpriced physical attempts" if allowance else "")
            + "."
        )
        consented = require_spend_consent(
            self.console,
            root=self.root,
            yes=self.yes,
            estimated_cost_usd=ceiling,
            cost_is_upper_bound=not allowance,
            command=self.command,
            non_interactive=self.non_interactive,
        )
        if consented:
            operation = operation_spend_ceiling(self.store, self.operation)
            if remaining and self.spend.ceiling_usd is None:
                ceiling += operation.summary().accounted_usd
            if self.spend.ceiling_usd is None:
                operation.authorize(stage, ceiling_usd=ceiling)
            operation.authorize_policy(
                ceiling_usd=self.spend.ceiling_usd, unpriced_attempts=allowance
            )
            install_spend_ceiling(operation)
            receipt = save_spend_receipt(self.store, self.operation, operation, self.command)
            self.console.print(Text(f"Spend receipt: {receipt}", style="dim"))
        return consented


def require_new_operation_file(path: Path | None) -> None:
    """Refuse an operation file that exists or whose directory is missing, before any work.

    Args:
        path: Requested new JSON file, or None when the operation is only printed.

    Raises:
        ValueError: The file exists or its parent directory is missing.
    """
    if path is not None and (path.exists() or not path.parent.is_dir()):
        raise ValueError("operation_file needs a new file in an existing directory")


def emit_operation(
    console: Console, operation: JsonObject, path: Path | None, *, dry_run: bool
) -> None:
    """Print a prepared ``--continuations`` operation and, unless dry-run, save it as a new file.

    Args:
        console: Command-owned output console.
        operation: Ready continuation command for a later invocation.
        path: New JSON file to create exclusively, or None to print only.
        dry_run: Print without creating the file.
    """
    if path is not None and not dry_run:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(operation, stream, indent=2)
            stream.write("\n")
    console.print_json(data=operation)
