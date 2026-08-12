"""Route fitting, tuning, and reporting commands."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from pydantic import ValidationError
from rich.console import Console

if TYPE_CHECKING:
    # Type-only: real imports are local to the commands and helpers that construct or inspect
    # these values, so importing this module never pulls the optimize/engine/env/distill/pool
    # bodies behind it.
    from wmo.optimize.routing.knn import DialResult, KnnFitOutcome
    from wmo.optimize.routing.outcomes import OutcomeMatrix
    from wmo.optimize.routing.policy import RoutingPolicy

from wmo.cli.route_constants import (
    _AZURE_EMBEDDER_DIM,
    _AZURE_EMBEDDER_ENV,
    _DEFAULT_WM_JUDGE,
    _HASHING_EMBEDDER_DIM,
    _LOCAL_EMBEDDER_DIM,
    _MATRIX_DIGEST_MARK,
    _POLICY_FILENAME,
    _REAL_EPISODE,
    _WM_SIMULATED,
    COMPRESSOR_IDS_HELP,
    DEFAULT_MATRIX_FILENAME,
)

_console = Console()


def _reads_as(model: type[OutcomeMatrix] | type[RoutingPolicy], path: Path) -> bool:
    """Return whether a path parses as the alternate route artifact type."""
    try:
        model.model_validate_json(path.read_bytes())
    except (ValidationError, OSError):
        return False
    return True


def _load_matrix(matrix_file: str) -> tuple[OutcomeMatrix, str]:
    """Load a matrix with its digest provenance, or raise a CLI usage error."""
    from wmo.optimize.routing.outcomes import load_matrix_with_digest

    path = Path(matrix_file)
    try:
        return load_matrix_with_digest(path)
    except FileNotFoundError as exc:
        raise typer.BadParameter(
            f"no outcome matrix at {path}; `wmo optimize route sweep <world-model>` measures the "
            f"pool and writes one (its --out, default {DEFAULT_MATRIX_FILENAME})"
        ) from exc
    except OSError as exc:
        raise typer.BadParameter(f"cannot read the outcome matrix at {path}: {exc}") from exc
    except ValidationError as exc:
        from wmo.optimize.routing.policy import RoutingPolicy

        if _reads_as(RoutingPolicy, path):
            raise typer.BadParameter(
                f"{path} holds a fitted policy, not an outcome matrix. The matrix is what "
                "`wmo optimize route sweep` writes, and it comes FIRST: "
                "`wmo optimize route report <matrix.json> <policy.json>`"
            ) from exc
        raise typer.BadParameter(f"{path} is not a readable OutcomeMatrix: {exc}") from exc


def _load_policy(policy_file: str) -> RoutingPolicy:
    """Load a fitted policy or turn artifact failures into CLI usage errors."""
    from wmo.optimize.routing.policy import RoutingPolicy

    path = Path(policy_file)
    try:
        return RoutingPolicy.load(path)
    except FileNotFoundError as exc:
        raise typer.BadParameter(
            f"no policy file at {path}; fit one with "
            "`wmo optimize route fit <matrix.json> --kind knn`"
        ) from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise typer.BadParameter(f"cannot read the policy at {path}: {exc}") from exc
    except ValidationError as exc:
        from wmo.optimize.routing.outcomes import OutcomeMatrix

        if _reads_as(OutcomeMatrix, path):
            raise typer.BadParameter(
                f"{path} holds an outcome matrix, not a fitted policy. The policy is what "
                "`wmo optimize route fit` writes, and it comes SECOND: "
                "`wmo optimize route report <matrix.json> <policy.json>`"
            ) from exc
        raise typer.BadParameter(f"{path} is not a readable routing policy: {exc}") from exc


def fit(
    matrix_file: str = typer.Argument(..., help="OutcomeMatrix JSON (closed-loop eval output)."),
    kind: str = typer.Option(
        "knn",
        "--kind",
        help="knn (guarded nearest-neighbor evidence, the validated champion, and what `tune`, "
        "`sweep`'s handoff and the docs all assume) | rank (Avengers cluster ranks).",
    ),
    out: str = typer.Option(
        _POLICY_FILENAME, "--out", help="Where to write the fitted policy JSON."
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
    embedder: str = typer.Option(
        "auto",
        "--embedder",
        help="auto | local | hashing | azure. auto prefers the in-process local model "
        "(Qwen3-Embedding-0.6B) when its weights are already cached, then the Azure "
        f"text-embedding-3-large deployment when {' and '.join(_AZURE_EMBEDDER_ENV)} are set, "
        "and hashing otherwise; either way it says which one it picked. --embedder local runs "
        "with NO embedding API in the loop (first ever use downloads the weights from Hugging "
        "Face); resolving to azure means this fit CALLS A PAID EMBEDDING API (billed to that "
        "resource); --embedder hashing keeps it offline, free, and lexical.",
    ),
    dim: int = typer.Option(
        None,
        "--dim",
        min=1,
        help="Embedding dimension, sent as the request's `dimensions`. Default: the resolved "
        f"model's native width ({_HASHING_EMBEDDER_DIM} hashing, {_LOCAL_EMBEDDER_DIM} local, "
        f"{_AZURE_EMBEDDER_DIM} text-embedding-3-large). Set it only to reduce a model's "
        "output deliberately.",
    ),
    deployment: str = typer.Option(None, "--deployment", help="(azure) embedding deployment."),
    endpoint: str = typer.Option(None, "--endpoint", help="(azure) resource endpoint."),
    api_key_env: str = typer.Option(
        None, "--api-key-env", help="(azure) env var holding the account key."
    ),
    embed_model: str = typer.Option(
        None,
        "--embed-model",
        help="(local) Hugging Face embedding model id; default Qwen/Qwen3-Embedding-0.6B.",
    ),
    compressor: str = typer.Option(
        None,
        "--compressor",
        help="D-COMPRESS: compressor id the endpoint applies before routing "
        f"({COMPRESSOR_IDS_HELP}). Default: compression off.",
    ),
    aggressiveness: float = typer.Option(
        0.0,
        "--aggressiveness",
        min=0.0,
        max=1.0,
        help="Compressor-defined dial in [0, 1]: 0.0 is a no-op and higher never removes "
        "less, but it is not an exact removal fraction (the achieved ratio is reported per "
        "call). Only meaningful with --compressor.",
    ),
) -> None:
    """Fit a routing policy on an outcome matrix (kNN evidence or Avengers cluster ranks).

    `--kind knn` is the product router and what `wmo optimize model` fits. `--kind rank` is a
    retained research direction (a faithful Avengers replication kept for comparison); the
    staged pipeline never fits it and no served endpoint carries one, so choose it only to
    measure against the champion.

    Args:
        options: Inputs accepted by this callable.
    Raises:
        ValueError: If the requested operation cannot be completed.
    """
    from wmo.optimize.routing import evaluate_policy, fit_rank_policy, rerank_policy
    from wmo.optimize.routing.compression import (
        CompressingEmbedder,
        compression_signature,
        resolve_compression,
        same_compression,
    )
    from wmo.optimize.routing.knn import fit_knn_artifact
    from wmo.optimize.routing.outcomes import ROUTER_SPLIT_VERSION, split_router_scenarios
    from wmo.optimize.routing.policy import embedder_provenance, probe_embedder, resolve_embedder

    if kind not in ("rank", "knn"):
        raise typer.BadParameter(f"unknown kind '{kind}'; use knn or rank")
    matrix, source = _load_matrix(matrix_file)
    if not any(outcome.scored for outcome in matrix.outcomes):
        # `sweep` already exits 1 saying "fitting will fail" on this matrix, so it is a state a
        # user arrives here from: answer it the way sweep does instead of letting the fitter's
        # own ValueError out (the rank one names no remedy at all).
        raise typer.BadParameter(
            f"no cell in {matrix_file} carries a reward, so there is nothing to fit: read the "
            "`error` field of a row to see what broke, fix it, then re-run "
            "`wmo optimize route sweep <world-model>`"
        )
    try:
        spec, resolution = resolve_embedder(
            embedder,
            dim=dim,
            deployment=deployment,
            endpoint=endpoint,
            api_key_env=api_key_env,
            model=embed_model,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    # Printed before the fit, not after: the embedder decides what the policy can route on, and an
    # operator who meant to fit on semantic vectors should see that it fell back BEFORE paying for
    # the fit and reading an accuracy number that quietly came from hashed features.
    _console.print(resolution)
    out_path = Path(out)
    if rag_thres <= 0.0:
        # typer's min is inclusive but the artifact field requires > 0; fail before the fit
        # writes a sidecar it will then abandon.
        raise typer.BadParameter("--rag-thres must be greater than 0")
    # One throwaway embedding BEFORE the bulk work: an unreachable or embedding-less resource is
    # a usage error at the boundary here, instead of a traceback from inside the fit.
    try:
        probe_embedder(spec)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if compressor is None and aggressiveness > 0.0:
        raise typer.BadParameter("--aggressiveness needs --compressor to apply it")
    compression = None
    if compressor is not None:
        try:
            # Fail before the fit spends anything: model_copy below skips validators, and an
            # unservable compressor would otherwise only surface when serving mounts the result.
            compression = resolve_compression(compressor, aggressiveness)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
    # The rewards in this matrix were produced under SOME compression config, and a joint fit is
    # only joint if that config is the one being fitted. `--compressor` moves the fit-side
    # representation (embeddings), but it cannot retroactively change what the episodes ran
    # under: fitting a compressed policy over uncompressed rewards would stamp an arm that was
    # never measured. Checked both directions, since compressed rewards under a raw fit is the
    # same mistake mirrored.
    measured = matrix.measured_compression()
    if not same_compression(measured, compression):
        raise typer.BadParameter(
            f"this matrix's rewards were measured with {compression_signature(measured)}, but "
            f"the fit would stamp {compression_signature(compression)}. Rewards cannot be "
            "recompressed after the fact, so measure the arm you intend to serve: "
            "`wmo optimize route sweep <model> --compressor <id> --aggressiveness <a>` writes a "
            "matrix whose episodes actually ran that way (one matrix per arm)."
        )
    try:
        router_split = split_router_scenarios(matrix.scenario_ids())
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    fit_ids = list(router_split.fit_ids)
    if kind == "knn":
        if cost_weight > 0.0:
            raise typer.BadParameter(
                "--cost-weight re-ranks cluster evidence and applies to --kind rank only; a knn "
                "policy trades cost through its dial instead: fit it, then "
                "`wmo optimize route tune <policy.json> --cost-quality <0..1>`"
            )
        try:
            fitted = fit_knn_artifact(
                matrix,
                out_path=out_path,
                matrix_source=source,
                embedder=spec,
                fit_ids=fit_ids,
                fallback=fallback,
                z=z,
                rag_num=rag_num,
                rag_thres=rag_thres,
                min_pairs=min_pairs,
                se_floor=se_floor,
                floor_q=floor_q,
                compression=compression,
            )
        except ValueError as exc:
            # What the fitter can still refuse once the matrix loads: an unknown --fallback, or a
            # fit split whose own rows are all unscored. Both are about the arguments.
            raise typer.BadParameter(str(exc)) from exc
        print_knn_fit(_console, fitted, out=out, z=z)
        return
    built = spec.build()  # ONE embedder for fit and evaluation; azure would otherwise embed twice
    if compression is not None:
        # Representation consistency: the cluster centroids have to live in the geometry of the
        # text serving will embed, which is the COMPRESSED text (see `fit_knn_artifact`, which
        # applies the same rule to the bank on the knn path).
        built = CompressingEmbedder(built, compression)
    try:
        policy = fit_rank_policy(
            matrix,
            fit_ids=fit_ids,
            embedder=spec,
            n_clusters=clusters,
            seed=seed,
            top_k_clusters=top_k_clusters,
            beta=beta,
            fitted_from=(
                f"{source} split={ROUTER_SPLIT_VERSION} "
                f"rank seed={seed} k={clusters} topk={top_k_clusters} beta={beta:g} "
                f"cost_weight={cost_weight:g} {embedder_provenance(spec)}"
            ),
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if cost_weight > 0.0:
        policy = rerank_policy(policy, cost_weight=cost_weight)
    if compression is not None:
        # Stamped as BOTH halves of the contract: what this endpoint serves, and what its
        # evidence was fitted under. They are the same config here by construction (the fit just
        # embedded through it), which is exactly what the mount gate re-checks. The knn path
        # stamps inside `fit_knn_artifact`, which saves and returns before this line.
        policy = policy.model_copy(
            update={"compression": compression, "fit_compression": compression}
        )
    policy.save(out_path)
    result = evaluate_policy(policy, matrix, fit_ids, embedder=built)
    _console.print(
        f"[green]✓[/green] fitted {len(policy.clusters)} clusters over "
        f"{result.scenarios} scenarios -> {out}\n"
        f"  fit-set accuracy {result.accuracy:.4f}, cost/scenario ${result.cost_per_scenario:.5f}"
    )


def print_knn_fit(console: Console, fitted: KnnFitOutcome, *, out: str, z: float) -> None:
    """Report a written knn policy: where its evidence is, and what it scored in-sample.

    Args:
        options: Inputs accepted by this callable.
    """
    console.print(
        f"[green]✓[/green] fitted knn policy over {fitted.scenarios} scenarios -> {out}\n"
        f"  bank {fitted.bank_path}, fallback {fitted.policy.default_model}, z={z}\n"
        f"  routed away from the fallback {fitted.routed_share:.1%} of the time; cost/scenario "
        f"${fitted.cost_per_scenario:.5f}\n"
        f"  fit-set accuracy {fitted.fit_accuracy:.4f} is IN-SAMPLE (every request retrieves its "
        "own row); measure on held-out scenarios with `wmo optimize route report`"
    )


def tune(
    policy_file: str = typer.Argument(_POLICY_FILENAME, help="Fitted knn policy JSON to re-tune."),
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
    `wmo.optimize.routing.knn.apply_cost_quality`). The first successful run copies the un-tuned
    artifact to `policy.base.json` and every later run re-reads THAT, so the dial is always applied
    to the policy as fitted and sliding twice never compounds:

        wmo optimize route tune models/support/policy.json --cost-quality 0.6

    That snapshot is only a valid baseline for the fit it came from, so this command refuses to
    run when the two disagree (refit the policy and the stale snapshot must be deleted, not
    silently dialed back over the new fit). A tune that is rejected writes nothing at all, and
    every write it does make is atomic.

    The evidence bank is untouched, so this is instant. A served endpoint can be dialed without
    touching files at all: `PUT /v1/endpoints/{name}/config`.

    Args:
        options: Inputs accepted by this callable.
    Raises:
        ValueError: If the requested operation cannot be completed.
    """
    from wmo.optimize.routing.knn import tune_policy_dial

    try:
        dialed = tune_policy_dial(Path(policy_file), cost_quality)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    print_dial(_console, dialed)


def print_dial(console: Console, dialed: DialResult) -> None:
    """Report an applied dial position against the frontier that was actually measured.

    Args:
        options: Inputs accepted by this callable.
    """
    from wmo.optimize.routing.knn import COST_QUALITY_ANCHORS

    knobs = dialed.knobs
    console.print(
        f"[green]✓[/green] cost_quality={dialed.cost_quality:g} "
        f"({dialed.named_point}) -> {dialed.policy_path}\n"
        f"  knobs: floor_q={knobs.floor_q:g}, cost knob lam={knobs.pick_lam:g}, "
        f"guard={knobs.guard_mode}, z={knobs.knn_z:g}\n"
        f"  as fitted: {dialed.base_path}\n"
        f"  measured on routerbench-ours9 (5 held-out splits, vs the best single model):"
    )
    for anchor in COST_QUALITY_ANCHORS:
        marker = "->" if anchor.cost_quality == dialed.cost_quality else "  "
        console.print(
            f"  {marker} {anchor.cost_quality:<5g} {anchor.quality_delta_points:+.2f}pt "
            f"@ {anchor.cost_delta_percent:+.1f}% cost"
            + (f"  [dim]{anchor.named_point}[/dim]" if anchor.named_point != "Custom" else "")
        )


def report(
    matrix_file: str = typer.Argument(..., help="OutcomeMatrix JSON with held-out scenarios."),
    policy_file: str = typer.Argument(..., help="Fitted policy JSON."),
    baseline: str = typer.Option(
        ...,
        "--baseline",
        # The doubled brackets are escaped for the same reason `sweep --pool`'s are: typer
        # renders help through rich markup, which otherwise prints an empty pair.
        help="Pool entry HANDLE to compare against: the `name` of a \\[\\[model]] table in the "
        "matrix's pool, NOT the model id. Normally the frontier candidate.",
    ),
    endpoint: str = typer.Option("endpoint", "--endpoint", help="Endpoint id for the report."),
    out: str = typer.Option("report.json", "--out", help="Where to write the report JSON."),
    provenance: str = typer.Option(
        _WM_SIMULATED,
        "--provenance",
        help=f"How this matrix's rewards were produced: {_WM_SIMULATED} (closed-loop against a "
        f"world model, the default) or {_REAL_EPISODE} (episodes of the real benchmark). It rides "
        "on the pareto curve and must never be wrong: consumers refuse to blend the two.",
    ),
    judge: str = typer.Option(
        _DEFAULT_WM_JUDGE,
        "--judge",
        help="What scored the episodes, printed beside every rendering of the curve. Pass the "
        "real scorer for a real-benchmark matrix (for example \\[tau2 reward]).",
    ),
    scenario_label: str = typer.Option(
        "",
        "--scenario-label",
        help="The report's customer-facing sentence describing WHAT was measured. Defaults to the "
        "world-model phrasing ('reconstructed from your traces'), which is false for a real "
        "benchmark, so pass the truth there (for example 'on the 20 pinned tau2-bench eval "
        "tasks').",
    ),
) -> None:
    """Build the improvement report for a fitted policy over a matrix.

    Args:
        options: Inputs accepted by this callable.
    Raises:
        ValueError: If the requested operation cannot be completed.
    """
    from wmo.optimize.routing.pareto import (
        PARETO_FILENAME,
        REAL_EPISODE,
        WM_SIMULATED,
        held_out_curve,
    )
    from wmo.optimize.routing.policy import write_artifact_atomically
    from wmo.optimize.routing.report import build_report

    if provenance not in {WM_SIMULATED, REAL_EPISODE}:
        # A typo here would silently label real measurements as simulated, which is the one
        # mistake the curve's provenance field exists to prevent.
        raise typer.BadParameter(
            f"--provenance must be {WM_SIMULATED} or {REAL_EPISODE}, not {provenance!r}"
        )
    matrix, matrix_source = _load_matrix(matrix_file)
    policy = _load_policy(policy_file)
    try:
        improvement = build_report(
            matrix,
            policy,
            baseline=baseline,
            endpoint=endpoint,
            generated_at=datetime.now(tz=UTC).isoformat(),
            scenario_label=scenario_label or None,
        )
    except KeyError as exc:
        # `--baseline` is a pool entry handle; the KeyError already lists the ones this matrix
        # has. `str()` on a KeyError quotes its own argument, so unwrap it.
        raise typer.BadParameter(
            f"{exc.args[0]}. --baseline names a pool entry handle (the `name` of a [[model]] "
            "table), not a model id."
        ) from exc
    except FileNotFoundError as exc:
        # A knn policy carries its evidence in a sidecar; `knn_bank` says how to restore it.
        raise typer.BadParameter(str(exc)) from exc
    except ValueError as exc:
        # Nothing scored on both sides, so there is no paired comparison to report.
        raise typer.BadParameter(str(exc)) from exc
    # mkdir + atomic, exactly as `fit --out` and `pin --out` write their policies: a report whose
    # parent directory does not exist must not throw away the work that produced it.
    write_artifact_atomically(Path(out), improvement.model_dump_json(indent=2).encode("utf-8"))
    # The measured cost/quality curve rides beside every report (D-PARETO): GET /config
    # serves it from the model dir so the platform's graph renders this workload's frontier.
    try:
        curve = held_out_curve(matrix, policy, judge=judge, provenance=provenance)
        pareto_out = Path(out).parent / PARETO_FILENAME
        write_artifact_atomically(pareto_out, curve.model_dump_json(indent=2).encode("utf-8"))
        _console.print(f"[green]✓[/green] pareto curve -> {pareto_out}")
        # Same foot-gun as `pin --out`: serving loads policy.json and pareto.json from ONE
        # model dir, so a curve written apart from the policy it describes is a curve
        # `wmo serve` and GET /config never show. Succeeding silently here is what hid it.
        if Path(policy_file).resolve().parent != pareto_out.resolve().parent:
            _console.print(
                f"[yellow]![/yellow] the curve landed apart from {policy_file}; serving reads "
                "pareto.json from the same directory as the policy it mounts, so point --out "
                "there for the endpoint to show this curve"
            )
    except (ValueError, FileNotFoundError) as exc:
        _console.print(f"[yellow]![/yellow] pareto curve skipped: {exc}")
    headline = improvement.headline
    _console.print(
        f"[green]✓[/green] report -> {out}\n"
        f"  routed acc {headline.accuracy:.4f} @ ${headline.cost_per_run_usd:.5f}/run vs "
        f"{baseline} {headline.baseline_accuracy:.4f} @ ${headline.baseline_cost_per_run_usd:.5f}"
    )
    in_sample = _in_sample_warning(policy, matrix_source)
    if in_sample is not None:
        _console.print(in_sample)


def _in_sample_warning(policy: RoutingPolicy, matrix_source: str) -> str | None:
    """The caveat for a report measured on the very matrix the policy was fitted on.

    Since the router split (#308), `build_report` excludes the policy's recorded fit scenarios,
    so a report over the fit matrix IS held out whenever that split is recoverable - the note
    says so, with the count. The in-sample WARNING remains only for a policy that records no
    split and whose evidence cannot name one: those numbers retrieve their own rows. The digest
    in `fitted_from` is an identity rather than a label (`load_matrix_with_digest`), so the
    collision is detectable even when the matrix was renamed or moved after the fit, and a
    matrix with the same path but different bytes does not trip it.
    """
    # `load_matrix_with_digest` appends the marker LAST, so split from the right: a matrix under a
    # content-addressed directory (`artifacts/sha256=.../matrix.json`) carries the marker in its
    # path too, and splitting from the left would read that one and silently drop the warning.
    _, mark, digest = matrix_source.rpartition(_MATRIX_DIGEST_MARK)
    stamped = policy.fitted_from or ""
    if not mark or not digest or f"{_MATRIX_DIGEST_MARK}{digest}" not in stamped:
        return None
    fit_ids = set(policy.fit_scenario_ids)
    if not fit_ids and policy.kind == "knn":
        # The same recovery `build_report` uses for legacy kNN artifacts: their evidence bank
        # names the fit scenarios even when the policy predates recording them.
        try:
            fit_ids = set(policy.knn_bank().scenario_ids)
        except (FileNotFoundError, ValueError):
            fit_ids = set()
    if fit_ids:
        # Same matrix as the fit, but the report excluded the fit scenarios (the split is
        # recorded on the policy), so the numbers above ARE held out. Say what happened instead
        # of contradicting the report's own label.
        return (
            f"note: same matrix as the fit ({_MATRIX_DIGEST_MARK}{digest}); the "
            f"{len(fit_ids)} fit scenario(s) were excluded, so the numbers above are over "
            "held-out scenarios only."
        )
    return (
        f"[yellow]warning[/yellow] this policy was FITTED on this matrix "
        f"({_MATRIX_DIGEST_MARK}{digest}) and records no fit split, so these numbers are "
        "IN-SAMPLE, not held out: every request retrieves its own row. Sweep a second matrix "
        "over scenarios the fit never saw and report against that one."
    )


def register(app: typer.Typer) -> None:
    """Register route fitting commands on their parent Typer app.

    Args:
        options: Inputs accepted by this callable.
    """
    app.command("fit")(fit)
    app.command("tune")(tune)
    app.command("report")(report)
