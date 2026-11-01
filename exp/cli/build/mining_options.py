"""Command-line choice of mining geometry, clustering backend and sampling strategy.

A build mines representative tasks by one of two routes. The default route reads the text a
request made visible: it needs no provider, costs nothing, and groups cases by how they were
worded. The second route asks a decision model what each episode turned out to contain and groups
cases by that, which separates two runs of one template that differ in what the environment
revealed. Those two look alike in the first geometry and belong apart in the second, and that is
the whole reason the choice exists.

The options live here rather than in the command module for two reasons. The command module is
near its line limit, and more usefully, every refusal in this file is a sentence a user reads
instead of a stack trace: naming an incompatible combination at parse time costs nothing, while
discovering it after a corpus has been tagged costs real money.
"""

from __future__ import annotations

from pathlib import Path

import typer

from exp.common.models import ModelCatalog
from exp.common.project import ProjectTracePreparationSettings
from exp.common.project.paths import ProjectPaths
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.simulation.build import mining_spec_for
from exp.simulation.mining.decisions import DecisionsTraceTagger
from exp.simulation.mining.dispatch import DEFAULT_TAGGING_CONCURRENCY
from exp.simulation.mining.questions import (
    QuestionBankError,
    local_bank_root,
    resolve_question_bank,
)
from exp.simulation.mining.service import MiningSpec
from exp.simulation.mining.tagging import DecisionCache, TraceTagger

MINE_WITH_CHOICES = ("hashing", "decision-tag")
CLUSTER_WITH_CHOICES = ("descriptor", "kmeans", "bernoulli-mixture", "atoms-hierarchy")
SAMPLE_WITH_CHOICES = ("medoid", "proportional", "diversity", "stratified")

MINE_WITH_OPTION = typer.Option(
    "hashing",
    "--mine-with",
    metavar="GEOMETRY",
    help=(
        "Selection geometry: hashing reads request-visible text and needs no provider; "
        "decision-tag asks --tagger what each episode contained (paid, cached per episode)."
    ),
)
QUESTION_BANK_OPTION = typer.Option(
    None,
    "--question-bank",
    metavar="BANK_ID",
    help=(
        "Question bank the decision-tag geometry reads, by identity. It is resolved from the "
        "local bank directory rather than from the installed package, so the propositions stay "
        "local configuration; set EXP_QUESTION_BANK_ROOT to read them from elsewhere."
    ),
)
TAGGER_OPTION = typer.Option(
    None,
    "--tagger",
    metavar="ALIAS",
    help="Decision-model alias answering the question bank. Required by --mine-with decision-tag.",
)
CLUSTER_WITH_OPTION = typer.Option(
    "descriptor",
    "--cluster-with",
    metavar="BACKEND",
    help=(
        "Clustering backend: descriptor (default, seedless), kmeans, bernoulli-mixture "
        "(needs decision-tag) or atoms-hierarchy."
    ),
)
SAMPLE_WITH_OPTION = typer.Option(
    "medoid",
    "--sample-with",
    metavar="STRATEGY",
    help=(
        "How leftover task budget is spent: medoid (default), proportional to workload, "
        "diversity, or stratified evenly across groups."
    ),
)
MINING_SEED_OPTION = typer.Option(
    42,
    "--mining-seed",
    help="Seed for a clustering backend with random initialization. The default path uses none.",
)
TAGGER_CONCURRENCY_OPTION = typer.Option(
    DEFAULT_TAGGING_CONCURRENCY,
    "--tagger-concurrency",
    min=1,
    metavar="CALLS",
    help=(
        "Decision calls in flight at once while tagging. Raises throughput only; the tasks, "
        "their identities and what the build pays are the same at any value. 1 tags serially."
    ),
)


def mining_geometry(
    *,
    mine_with: str,
    cluster_with: str,
    sample_with: str,
    tagger: str | None,
    question_bank: str | None = None,
    seed: int = 42,
    fit_task_budget: int = 50,
    held_out_task_budget: int = 20,
    descriptor_dimensions: int = 64,
) -> MiningSpec:
    """Turn the command line's names into one validated mining spec.

    Args:
        mine_with: Selection geometry name.
        cluster_with: Clustering backend name.
        sample_with: Sampling strategy name.
        tagger: Decision-model alias, or None.
        question_bank: Question bank identity, or None.
        seed: Seed for a randomly initialized backend.
        fit_task_budget: Representative tasks reserved for fit-only work.
        held_out_task_budget: Representative tasks sealed for final reporting.
        descriptor_dimensions: Width of the request-visible hashing vector.

    Returns:
        The mining spec, identical to the provider-free default when nothing was named.

    Raises:
        typer.BadParameter: A name is unknown, or the combination cannot describe one corpus.
    """
    _require_choice("--mine-with", mine_with, MINE_WITH_CHOICES)
    _require_choice("--cluster-with", cluster_with, CLUSTER_WITH_CHOICES)
    _require_choice("--sample-with", sample_with, SAMPLE_WITH_CHOICES)
    if mine_with == "decision-tag" and tagger is None:
        raise typer.BadParameter(
            "--mine-with decision-tag asks a decision model about every episode, so it needs "
            "--tagger ALIAS naming a configured decision alias"
        )
    if mine_with != "decision-tag" and tagger is not None:
        raise typer.BadParameter(
            "--tagger is only used by --mine-with decision-tag; add that flag to tag the "
            "episodes, or drop --tagger to mine request-visible text alone"
        )
    _require_bank(mine_with, question_bank)
    settings = _settings(
        mine_with=mine_with,
        cluster_with=cluster_with,
        sample_with=sample_with,
        tagger=tagger,
        question_bank=question_bank,
        seed=seed,
        fit_task_budget=fit_task_budget,
        held_out_task_budget=held_out_task_budget,
        descriptor_dimensions=descriptor_dimensions,
    )
    try:
        return mining_spec_for(settings)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def mining_preparation_settings(
    *,
    source_kind: str,
    mine_with: str,
    cluster_with: str,
    sample_with: str,
    tagger: str | None,
    question_bank: str | None = None,
    seed: int = 42,
    fit_task_budget: int = 50,
    held_out_task_budget: int = 20,
    descriptor_dimensions: int = 64,
) -> ProjectTracePreparationSettings:
    """Return the persisted preparation settings for one named geometry.

    Args:
        source_kind: Declared canonical trace source format.
        mine_with: Selection geometry name.
        cluster_with: Clustering backend name.
        sample_with: Sampling strategy name.
        tagger: Decision-model alias, or None.
        question_bank: Question bank identity, or None.
        seed: Seed for a randomly initialized backend.
        fit_task_budget: Representative tasks reserved for fit-only work.
        held_out_task_budget: Representative tasks sealed for final reporting.
        descriptor_dimensions: Width of the request-visible hashing vector.

    Returns:
        Settings a project stores, from which the same spec is rebuilt on every later run.
    """
    _require_bank(mine_with, question_bank)
    return _settings(
        source_kind=source_kind,
        mine_with=mine_with,
        cluster_with=cluster_with,
        sample_with=sample_with,
        tagger=tagger,
        question_bank=question_bank,
        seed=seed,
        fit_task_budget=fit_task_budget,
        held_out_task_budget=held_out_task_budget,
        descriptor_dimensions=descriptor_dimensions,
    )


def build_tagger(
    alias: str,
    *,
    catalog: ModelCatalog,
    root: Path,
    project: str,
    maximum_concurrency: int = DEFAULT_TAGGING_CONCURRENCY,
) -> TraceTagger:
    """Resolve one alias into a decision-model tagger with a project-scoped answer cache.

    The cache is what makes a tagged corpus affordable to iterate on: a rerun pays only for
    episodes it has not tagged and questions whose text changed. Concurrency is passed here
    rather than put in the mining spec on purpose: it cannot change an answer, a task or a
    digest, so it is a property of this invocation and not of the evidence the build produces.

    Args:
        alias: Configured decision-model alias.
        catalog: Loaded model catalog.
        root: Local EXP artifact root.
        project: Project owning the saved answers.
        maximum_concurrency: Decision calls in flight at once.

    Returns:
        A tagger bound to that alias's typed decision endpoint.

    Raises:
        typer.BadParameter: The alias is absent or does not serve typed decisions.
    """
    try:
        resolved = RuntimeModelCatalog(catalog).resolve(alias)
    except (KeyError, ValueError) as exc:
        raise typer.BadParameter(
            f"--tagger {alias!r} is not a configured model alias ({exc}); "
            "add it with exp config providers, then rerun"
        ) from exc
    try:
        return DecisionsTraceTagger.for_alias(
            resolved,
            cache=DecisionCache(ProjectPaths(root=root, project_id=project)),
            maximum_concurrency=maximum_concurrency,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _settings(
    *,
    source_kind: str = "otlp",
    mine_with: str,
    cluster_with: str,
    sample_with: str,
    tagger: str | None,
    question_bank: str | None,
    seed: int,
    fit_task_budget: int,
    held_out_task_budget: int,
    descriptor_dimensions: int,
) -> ProjectTracePreparationSettings:
    """Build preparation settings from validated command-line names."""
    return ProjectTracePreparationSettings(
        source_kind=source_kind,
        fit_task_budget=fit_task_budget,
        held_out_task_budget=held_out_task_budget,
        descriptor_dimensions=descriptor_dimensions,
        mining_geometry="decision_tag" if mine_with == "decision-tag" else "hashing",
        mining_cluster=cluster_with.replace("-", "_"),  # ty: ignore[invalid-argument-type]
        mining_sampling=sample_with,  # ty: ignore[invalid-argument-type]
        mining_tagger_alias=tagger,
        mining_bank=question_bank or "",
        mining_seed=seed,
    )


def _require_bank(mine_with: str, question_bank: str | None) -> None:
    """Refuse a geometry and bank that do not go together, and a bank nobody installed.

    Resolving the bank here rather than at mining time is the point of this module: a missing or
    misspelled bank costs nothing to report now and costs a tagged corpus to discover later.
    """
    if mine_with == "decision-tag" and not question_bank:
        raise typer.BadParameter(
            "--mine-with decision-tag groups cases by a decision model's answers, and which "
            "questions it answers decides what the geometry can tell apart, so it needs "
            f"--question-bank BANK_ID naming a bank in {local_bank_root()}"
        )
    if mine_with != "decision-tag" and question_bank:
        raise typer.BadParameter(
            "--question-bank is only read by --mine-with decision-tag; add that flag to tag the "
            "episodes, or drop --question-bank to mine request-visible text alone"
        )
    if not question_bank:
        return
    try:
        resolve_question_bank(question_bank)
    except QuestionBankError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _require_choice(flag: str, value: str, choices: tuple[str, ...]) -> None:
    """Reject an unknown name by listing the ones that exist."""
    if value not in choices:
        raise typer.BadParameter(f"{flag} must be one of: {', '.join(choices)}; got {value!r}")
