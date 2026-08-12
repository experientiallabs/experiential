"""Representative scenario construction and verification commands."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from wmo.common.config import (
    ARTIFACT_DIR,
    WorldModelStore,
)

if TYPE_CHECKING:
    from wmo.common.core.types import Trace
    from wmo.simulation.scenarios import ScenarioSet

from wmo.cli.catalog_cmd import _load_model, _resolve_name
from wmo.cli.command_common import (
    _resolve_scenario_embedder,
    _role_provider_config,
    _scenario_role_llms,
    _worker_role_provider_config,
)

scenarios_app = typer.Typer(
    help="Construct and verify representative eval scenario sets from traces.",
    no_args_is_help=True,
)
_console = Console()


@scenarios_app.command("build")
def scenarios_build(
    file: str = typer.Option(..., "--file", help="Path to exported traces (OTLP-JSON / JSONL)."),
    out: str = typer.Option("scenarios.json", "--out", help="Where to write the scenario set."),
    budget: int = typer.Option(20, help="Number of scenarios to construct."),
    k: int = typer.Option(None, help="Cluster count (default: sqrt(corpus size))."),
    limit: int = typer.Option(None, help="Only use the first N ingested traces (cost control)."),
    provider: str = typer.Option(
        None,
        "--provider",
        help=(
            "Pin ONE LLM for every role (facets/naming/synthesis/validation). When omitted, "
            "roles resolve from .wmo/settings.toml \\[models.worker|judge|summary]."
        ),
    ),
    model: str = typer.Option(None, help="Model id (pins all roles, like --provider)."),
    region: str = typer.Option(None, help="AWS region (Bedrock)."),
    embed_provider: str = typer.Option(
        "hashing",
        help=(
            "Facet embedder: hashing (offline but lexical-only; clusters by wording, not "
            "meaning; prefer a semantic embedder for real corpora) | local (in-process "
            "Qwen3, semantic and credential-free; pass --embed-dim 1024) | bedrock | "
            "openai | azure."
        ),
    ),
    embed_model: str = typer.Option(None, help="Embeddings model id / Azure deployment."),
    embed_dim: int = typer.Option(512, help="Embedding dimensionality."),
    seed: int = typer.Option(0, help="Clustering seed."),
) -> None:
    """Distill a trace corpus into a representative scenario set (facets -> cluster -> select).

    Writes a `ScenarioSet` JSON: scenarios (task, seed state, checklist, weight, provenance),
    the named clusters they came from, and the corpus-coverage number that justifies them.

    Args:
        options: Inputs accepted by this callable.
    Raises:
        ValueError: If the requested operation cannot be completed.
    """
    from wmo.simulation.scenarios import FacetExtractor, ScenarioBuildConfig, build_scenario_set

    traces = _ingest_scenario_corpus(file)
    if limit is not None:
        traces = traces[:limit]
    if not traces:
        raise typer.BadParameter(f"no traces ingested from {file}")
    summary_llm, worker_llm, judge_llm = _scenario_role_llms(provider, model, region)
    embedder = _resolve_scenario_embedder(embed_provider, embed_model, embed_dim, region)

    _console.print(f"extracting facets for {len(traces)} traces…")
    facets = FacetExtractor(summary_llm).extract_all(traces)
    config = ScenarioBuildConfig(budget=budget, k=k, seed=seed)
    scenario_set = build_scenario_set(
        traces, facets, worker_llm, embedder, config, judge_provider=judge_llm
    )
    scenario_set.save(out)

    table = Table(title="Scenario set")
    table.add_column("Cluster", no_wrap=True)
    table.add_column("Scenario task")
    table.add_column("Weight", justify="right")
    table.add_column("Source", no_wrap=True)
    for scenario in scenario_set.scenarios:
        source = scenario.failure_category or scenario.source_outcome.value
        table.add_row(scenario.cluster_name, scenario.task[:80], f"{scenario.weight:.3f}", source)
    _console.print(table)
    _console.print(
        f"{len(scenario_set.scenarios)} scenarios from {scenario_set.corpus_traces} traces; "
        f"coverage {scenario_set.corpus_coverage:.0%} at tau={scenario_set.coverage_tau} -> {out}"
    )


@scenarios_app.command("verify")
def scenarios_verify(
    scenarios_file: str = typer.Argument(..., help="Scenario set JSON from `wmo scenarios build`."),
    file: str = typer.Option(..., "--file", help="Source trace corpus (for back-agreement)."),
    name: str = typer.Option(None, "--name", help="World model to roll against."),
    root: str = typer.Option(ARTIFACT_DIR, help="Project dir holding world models."),
    provider: str = typer.Option(None, "--provider", help="Override serve provider kind."),
    model: str = typer.Option(None, help="Override canonical serve model type."),
    region: str = typer.Option(None, help="AWS region (Bedrock)."),
    max_steps: int = typer.Option(12, help="Rollout step budget per scenario."),
    drop: bool = typer.Option(False, "--drop", help="Write back only verified scenarios."),
) -> None:
    """Closed-loop verification: back-agreement on source traces + solvability rollouts.

    Loads the world model (optionally overriding its serve provider with a cheaper model), rolls a
    baseline LLM agent on every scenario, and grades episodes against each scenario's checklist.
    With `--drop`, unverified scenarios are removed from the set in place.

    Args:
        options: Inputs accepted by this callable.
    """
    import wmo.common.providers as providers
    from wmo.common.providers.retry import wrap_provider_with_retries
    from wmo.runtime.agents.llm import LLMAgent
    from wmo.simulation.model.world_model import WorldModel
    from wmo.simulation.scenarios import ChecklistJudge, verify_scenarios

    scenario_set = _load_scenario_set(scenarios_file)
    traces = _ingest_scenario_corpus(file)
    if provider is not None or model is not None:
        store = WorldModelStore(root)
        model_dir = store.resolve(_resolve_name(store, name))
        override = _worker_role_provider_config(provider, model, region)
        llm = wrap_provider_with_retries(providers.get_provider(override))
        world_model = WorldModel.load(str(model_dir), llm)
    else:
        world_model, _resolved_name, llm = _load_model(name, root)

    # The rollout agent takes the worker role and the grader the judge role when configured in
    # settings (judge should differ in family from the generator); both fall back to the world
    # model's serve provider, which was the only behavior before roles existed.
    worker_config = _role_provider_config("worker", region)
    judge_config = _role_provider_config("judge", region)
    agent_llm = providers.get_provider(worker_config) if worker_config else llm
    judge_llm = providers.get_provider(judge_config) if judge_config else llm
    report = verify_scenarios(
        scenario_set,
        traces,
        world_model,
        LLMAgent(agent_llm),
        ChecklistJudge(judge_llm),
        max_steps=max_steps,
    )
    table = Table(title="Scenario verification")
    table.add_column("Scenario", no_wrap=True)
    table.add_column("Back-agree")
    table.add_column("Solvable")
    table.add_column("Pass rate", justify="right")
    for verdict in report.verdicts:
        if verdict.back_agreement is None:
            agree = "-"
        else:
            agree = "yes" if verdict.back_agreement else "NO"
        table.add_row(
            verdict.scenario_id,
            agree,
            "yes" if verdict.solvable else "NO",
            f"{verdict.rollout_pass_rate:.2f}",
        )
    _console.print(table)
    _console.print(
        f"back-agreement {report.back_agreement_rate:.0%}, solvable {report.solvable_rate:.0%} "
        f"over {len(report.verdicts)} scenarios"
    )
    if drop:
        verified = {v.scenario_id for v in report.verdicts if v.ok}
        scenario_set.retain(verified)
        scenario_set.save(scenarios_file)
        _console.print(
            f"kept {len(scenario_set.scenarios)} verified scenarios "
            f"(weights renormalized, coverage reset) -> {scenarios_file}"
        )


def _ingest_scenario_corpus(file: str) -> list[Trace]:
    """Ingest a `--file` trace corpus for the scenarios commands as a validated CLI input.

    `--file` is the only required option on `wmo scenarios build`, so a mistyped path is the
    likeliest first-run mistake. Guard it here rather than letting `Path.read_text` raise, which
    reaches the user as a stdlib FileNotFoundError/IsADirectoryError traceback.
    """
    from wmo.simulation.ingest import get_adapter

    path = Path(file)
    if path.is_dir():
        raise typer.BadParameter(
            f"--file {file} is a directory; point it at the trace file itself, "
            f"e.g. `--file {Path(file) / 'traces.otel.jsonl'}`"
        )
    if not path.exists():
        raise typer.BadParameter(
            f"--file {file} does not exist; pass an exported OTel-GenAI corpus, or fetch a "
            "benchmark one with `wmo download <benchmark>`"
        )
    try:
        return get_adapter("otel-genai").from_file(file)
    except (OSError, UnicodeDecodeError) as exc:
        raise _unreadable_input(f"--file {file}", path, exc) from exc


def _load_scenario_set(scenarios_file: str) -> ScenarioSet:
    """Load a `ScenarioSet` argument, reporting a missing or malformed file as a usage error.

    The raw failures are a stdlib FileNotFoundError/IsADirectoryError and a pydantic
    ValidationError that sends the user to pydantic's docs; neither says the file is supposed to
    be the output of `wmo scenarios build`.
    """
    from wmo.simulation.scenarios import ScenarioSet

    path = Path(scenarios_file)
    build_hint = (
        f"build one with `wmo scenarios build --file <traces.jsonl> --out {scenarios_file}`"
    )
    if path.is_dir():
        raise typer.BadParameter(
            f"scenario set {scenarios_file} is a directory; pass the JSON file written by "
            "`wmo scenarios build --out <scenarios.json>`"
        )
    if not path.exists():
        raise typer.BadParameter(f"scenario set {scenarios_file} does not exist; {build_hint}")
    try:
        return ScenarioSet.load(path)
    except ValidationError as exc:
        raise typer.BadParameter(
            f"{scenarios_file} is not a scenario set written by `wmo scenarios build` "
            f"({exc.error_count()} validation error(s), first: {exc.errors()[0]['msg']}); "
            f"{build_hint}"
        ) from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise _unreadable_input(f"scenario set {scenarios_file}", path, exc) from exc


def _unreadable_input(
    label: str, path: Path, exc: OSError | UnicodeDecodeError
) -> typer.BadParameter:
    """Report the two read failures an exists/is-dir check cannot predict as a usage error.

    A path that passes the shape checks can still fail inside the read: no permission on it, or
    bytes that are not UTF-8 (a compressed or binary export). Both otherwise reach the user as a
    stdlib traceback, which is the thing these guards exist to prevent.
    """
    if isinstance(exc, UnicodeDecodeError):
        return typer.BadParameter(
            f"{label} is not UTF-8 text; pass the decompressed JSON/JSONL export "
            f"(`file {path}` says what it actually is)"
        )
    return typer.BadParameter(
        f"{label} could not be read ({exc.strerror or exc}); "
        f"`ls -l {path}` shows its owner and mode"
    )
