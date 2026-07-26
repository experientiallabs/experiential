"""`wmh optimize route`: fit, tune, and report learned inference policies from outcome matrices.

The routing optimizer's CLI face, sitting beside `wmh optimize harness` in the optimizer
family. Consumes a persisted `OutcomeMatrix` (produced by closed-loop pool evaluation or a
research adapter such as RouterBench) and emits the policy artifact serving loads, plus the
improvement report the endpoint cites. `tune` is the one post-fit control: it moves a fitted
policy's cost/quality dial without refitting. Vocabulary note: "route" is developer-facing CLI
only; customer copy never says router.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console

from wmh.optimize.knn import (
    COST_QUALITY_ANCHORS,
    apply_cost_quality,
    cost_quality_knobs,
    cost_quality_named_point,
    fit_knn_policy,
)
from wmh.optimize.outcomes import OutcomeMatrix
from wmh.optimize.policy import (
    KNN_BANK_FILENAME,
    POLICY_FILENAME,
    EmbedderSpec,
    RoutingPolicy,
)
from wmh.optimize.report import build_report
from wmh.optimize.routing import evaluate_policy, fit_rank_policy, rerank_policy

route_app = typer.Typer(
    help="Fit and report learned inference policies from closed-loop outcome matrices.",
    no_args_is_help=True,
)

_console = Console()


@route_app.command("fit")
def fit(
    matrix_file: str = typer.Argument(..., help="OutcomeMatrix JSON (closed-loop eval output)."),
    kind: str = typer.Option(
        "rank",
        "--kind",
        help="knn (guarded nearest-neighbor evidence, the validated champion) | rank "
        "(Avengers cluster ranks).",
    ),
    out: str = typer.Option(
        POLICY_FILENAME, "--out", help="Where to write the fitted policy JSON."
    ),
    fallback: str = typer.Option(
        None,
        "--fallback",
        help="(knn) Baseline model every request uses unless the evidence says otherwise. "
        "Default: the best single model on the fit set.",
    ),
    z: float = typer.Option(
        0.5,
        "--z",
        min=0.0,
        help="(knn) Confidence knob: standard errors of paired evidence a pick must clear to "
        "leave the fallback (doubled when it is also pricier). Higher = stricter = more "
        "requests stay on the fallback; 0 routes on any positive difference.",
    ),
    rag_num: int = typer.Option(50, "--rag-num", min=1, help="(knn) Neighbor budget."),
    rag_thres: float = typer.Option(
        0.95,
        "--rag-thres",
        min=0.0,
        max=1.0,
        help="(knn) Keep neighbors above this fraction of the rag-num-th best similarity.",
    ),
    min_pairs: int = typer.Option(
        8, "--min-pairs", min=0, help="(knn) Neighbors scored on both sides before routing away."
    ),
    floor_q: float = typer.Option(
        0.05,
        "--floor-q",
        min=0.0,
        max=1.0,
        help="Novelty floor quantile: abstain to the fallback when a query's best bank "
        "similarity is below this quantile of the bank's own nearest-neighbor sims "
        "(coverage/robustness knob for task drift; 0 = off, the exact validated champion).",
    ),
    se_floor: bool = typer.Option(
        True,
        "--se-floor/--no-se-floor",
        help="(knn) Floor the guard's standard error on thin neighborhoods (small-bank safety).",
    ),
    clusters: int = typer.Option(64, "--clusters", min=1, help="k-means cluster count."),
    seed: int = typer.Option(42, "--seed", help="Clustering seed."),
    top_k_clusters: int = typer.Option(2, "--top-k-clusters", min=1),
    beta: float = typer.Option(6.0, "--beta", help="Cluster softmax sharpness."),
    cost_weight: float = typer.Option(
        0.0,
        "--cost-weight",
        min=0.0,
        help="Quality/cost knob: reward points paid per average-call-cost unit (0 = pure "
        "accuracy ranking, the Avengers reference behavior).",
    ),
    embedder: str = typer.Option("hashing", "--embedder", help="hashing | azure"),
    dim: int = typer.Option(512, "--dim", help="Embedding dimension."),
    deployment: str = typer.Option(None, "--deployment", help="(azure) embedding deployment."),
    endpoint: str = typer.Option(None, "--endpoint", help="(azure) resource endpoint."),
    api_key_env: str = typer.Option(
        None, "--api-key-env", help="(azure) env var holding the account key."
    ),
) -> None:
    """Fit a routing policy on an outcome matrix (kNN evidence or Avengers cluster ranks)."""
    if kind not in ("rank", "knn"):
        raise typer.BadParameter(f"unknown kind '{kind}'; use knn or rank")
    matrix = OutcomeMatrix.load(Path(matrix_file))
    if embedder not in ("hashing", "azure"):
        raise typer.BadParameter(f"unknown embedder '{embedder}'; use hashing or azure")
    spec = (
        EmbedderSpec(dim=dim)
        if embedder == "hashing"
        else EmbedderSpec(
            kind="azure",
            dim=dim,
            deployment=deployment,
            endpoint=endpoint,
            api_key_env=api_key_env,
        )
    )
    out_path = Path(out)
    if rag_thres <= 0.0:
        # typer's min is inclusive but the artifact field requires > 0; fail before the fit
        # writes a sidecar it will then abandon.
        raise typer.BadParameter("--rag-thres must be greater than 0")
    built = spec.build()  # ONE embedder for fit and evaluation; azure would otherwise embed twice
    if kind == "knn":
        if cost_weight > 0.0:
            raise typer.BadParameter(
                "--cost-weight re-ranks cluster evidence and applies to --kind rank only; a knn "
                "policy trades cost through its dial instead: fit it, then "
                "`wmh optimize route tune <policy.json> --cost-quality <0..1>`"
            )
        # The sidecar goes beside the policy file: that is where serving resolves it from.
        policy = fit_knn_policy(
            matrix,
            bank_path=out_path.parent / KNN_BANK_FILENAME,
            embedder=spec,
            embed_with=built,
            guard_model=fallback,
            rag_num=rag_num,
            rag_thres=rag_thres,
            z=z,
            min_pairs=min_pairs,
            se_floor=se_floor,
            floor_q=floor_q,
            fitted_from=f"{matrix_file} knn z={z} k={rag_num} q={floor_q} {embedder}-{dim}",
        )
    else:
        policy = fit_rank_policy(
            matrix,
            embedder=spec,
            n_clusters=clusters,
            seed=seed,
            top_k_clusters=top_k_clusters,
            beta=beta,
            fitted_from=f"{matrix_file} seed={seed} k={clusters} {embedder}-{dim}",
        )
        if cost_weight > 0.0:
            policy = rerank_policy(policy, cost_weight=cost_weight)
    policy.save(out_path)
    result = evaluate_policy(policy, matrix, matrix.scenario_ids(), embedder=built)
    if kind == "knn":
        routed = 1.0 - result.model_mix.get(policy.default_model, 0.0)
        _console.print(
            f"[green]✓[/green] fitted knn policy over {result.scenarios} scenarios -> {out}\n"
            f"  bank {out_path.parent / KNN_BANK_FILENAME}, fallback {policy.default_model}, "
            f"z={z}\n"
            f"  routed away from the fallback {routed:.1%} of the time; cost/scenario "
            f"${result.cost_per_scenario:.5f}\n"
            f"  fit-set accuracy {result.accuracy:.4f} is IN-SAMPLE (every request retrieves its "
            "own row); measure on held-out scenarios with `wmh optimize route report`"
        )
        return
    _console.print(
        f"[green]✓[/green] fitted {len(policy.clusters)} clusters over "
        f"{result.scenarios} scenarios -> {out}\n"
        f"  fit-set accuracy {result.accuracy:.4f}, cost/scenario ${result.cost_per_scenario:.5f}"
    )


@route_app.command("tune")
def tune(
    policy_file: str = typer.Argument(POLICY_FILENAME, help="Fitted knn policy JSON to re-tune."),
    cost_quality: float = typer.Option(
        ...,
        "--cost-quality",
        min=0.0,
        max=1.0,
        help="The endpoint's one dial: 0.0 = max quality, 1.0 = max savings. 0.25 is the "
        "shipped default. See the anchor table this command prints for what each end measured.",
    ),
) -> None:
    """Set a fitted policy's cost/quality dial in place, without refitting anything.

    The dial maps to the policy's knobs along the measured frontier (see
    `wmh.optimize.knn.apply_cost_quality`). The first run copies the un-tuned artifact to
    `policy.base.json` and every later run re-reads THAT, so the dial is always applied to the
    policy as fitted and sliding twice never compounds:

        wmh optimize route tune models/support/policy.json --cost-quality 0.6

    The evidence bank is untouched, so this is instant. A served endpoint can be dialed without
    touching files at all: `PUT /v1/endpoints/{name}/config`.
    """
    path = Path(policy_file)
    if not path.is_file():
        raise typer.BadParameter(f"no policy file at {path}")
    base_path = path.with_name(f"{path.stem}.base{path.suffix}")
    if not base_path.is_file():
        # Preserve the artifact as fitted the first time, so `tune` is always re-appliable from
        # the fit and never from an already-slid copy of itself.
        base_path.write_bytes(path.read_bytes())
    base = RoutingPolicy.load(base_path)
    try:
        tuned = apply_cost_quality(base, cost_quality)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    tuned.save(path)
    knobs = cost_quality_knobs(cost_quality)
    _console.print(
        f"[green]✓[/green] cost_quality={cost_quality:g} "
        f"({cost_quality_named_point(cost_quality)}) -> {path}\n"
        f"  knobs: floor_q={knobs.floor_q:g}, cost knob lam={knobs.pick_lam:g}, "
        f"guard={knobs.guard_mode}, z={knobs.knn_z:g}\n"
        f"  as fitted: {base_path}\n"
        f"  measured on routerbench-ours9 (5 held-out splits, vs the best single model):"
    )
    for anchor in COST_QUALITY_ANCHORS:
        marker = "->" if anchor.cost_quality == cost_quality else "  "
        _console.print(
            f"  {marker} {anchor.cost_quality:<5g} {anchor.quality_delta_points:+.2f}pt "
            f"@ {anchor.cost_delta_percent:+.1f}% cost"
            + (f"  [dim]{anchor.named_point}[/dim]" if anchor.named_point != "custom" else "")
        )


@route_app.command("report")
def report(
    matrix_file: str = typer.Argument(..., help="OutcomeMatrix JSON with held-out scenarios."),
    policy_file: str = typer.Argument(..., help="Fitted policy JSON."),
    baseline: str = typer.Option(
        ..., "--baseline", help="Frontier pool model the report compares against."
    ),
    endpoint: str = typer.Option("endpoint", "--endpoint", help="Endpoint id for the report."),
    out: str = typer.Option("report.json", "--out", help="Where to write the report JSON."),
) -> None:
    """Build the improvement report for a fitted policy over a matrix."""
    matrix = OutcomeMatrix.load(Path(matrix_file))
    policy = RoutingPolicy.load(Path(policy_file))
    improvement = build_report(
        matrix,
        policy,
        baseline=baseline,
        endpoint=endpoint,
        generated_at=datetime.now(tz=UTC).isoformat(),
    )
    Path(out).write_text(improvement.model_dump_json(indent=2), encoding="utf-8")
    headline = improvement.headline
    _console.print(
        f"[green]✓[/green] report -> {out}\n"
        f"  routed acc {headline.accuracy:.4f} @ ${headline.cost_per_run_usd:.5f}/run vs "
        f"{baseline} {headline.baseline_accuracy:.4f} @ ${headline.baseline_cost_per_run_usd:.5f}"
    )
