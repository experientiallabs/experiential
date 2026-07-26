"""`wmo optimize route`: fit, tune, and report learned inference policies from outcome matrices.

The routing optimizer's CLI face, sitting beside `wmo optimize harness` in the optimizer
family. Consumes a persisted `OutcomeMatrix` (produced by closed-loop pool evaluation or a
research adapter such as RouterBench) and emits the policy artifact serving loads, plus the
improvement report the endpoint cites. `tune` is the one post-fit control: it moves a fitted
policy's cost/quality dial without refitting. Vocabulary note: "route" is developer-facing CLI
only; customer copy never says router.

Two commands bracket the fit rather than consuming a matrix. `student` puts a freshly distilled
adapter into the candidate pool, which is what makes a trained model routable at all. `pin`
installs a `kind="static"` policy for one pool model, so a single candidate is serveable before
any measurement exists: the honest zero-evidence starting point a fit is compared against.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import typer
from llm_waterfall import ChatMaxTokensField
from pydantic import ValidationError
from rich.console import Console
from rich.markup import escape
from rich.prompt import Confirm

from wmo.config import ARTIFACT_DIR, WorldModelStore
from wmo.distill.store import MODEL_CARD_FILE, DistillModelCard, student_pool_entry
from wmo.optimize.knn import (
    COST_QUALITY_ANCHORS,
    apply_cost_quality,
    cost_quality_knobs,
    cost_quality_named_point,
    fit_knn_policy,
)
from wmo.optimize.outcomes import OutcomeMatrix
from wmo.optimize.policy import (
    KNN_BANK_FILENAME,
    POLICY_FILENAME,
    EmbedderSpec,
    RoutingPolicy,
)
from wmo.optimize.report import build_report
from wmo.optimize.routing import evaluate_policy, fit_rank_policy, rerank_policy
from wmo.providers.pool import (
    DEFAULT_POOL_PATH,
    PoolEntry,
    PoolLockTimeout,
    load_pool,
    upsert_pool_entry,
)

# The two output-budget parameter names any OpenAI-compatible backend accepts.
_MAX_TOKENS_FIELDS: tuple[ChatMaxTokensField, ...] = ("max_tokens", "max_completion_tokens")

route_app = typer.Typer(
    help="Make models routable, then fit, tune, and report inference policies over them.",
    no_args_is_help=True,
)

_console = Console()


@route_app.command("student")
def student(
    card_dir: str = typer.Argument(
        ...,
        help="The distillation run dir, or an adapter version dir: whichever holds "
        "model_card.json.",
    ),
    input_per_mtok: float = typer.Option(
        ...,
        "--input-per-mtok",
        min=0.0,
        help="Prompt-token price at the serving endpoint, USD per 1M tokens. Required: an "
        "unpriced candidate reports $0 and a cost-aware policy would route everything to it.",
    ),
    output_per_mtok: float = typer.Option(
        ...,
        "--output-per-mtok",
        min=0.0,
        help="Completion-token price at the serving endpoint, USD per 1M tokens.",
    ),
    name: str = typer.Option(
        "student", "--name", help="Pool handle: what policy artifacts and request logs call it."
    ),
    pool: str = typer.Option(
        str(DEFAULT_POOL_PATH), "--pool", help="Candidate pool TOML to add the entry to."
    ),
    endpoint: str = typer.Option(
        None,
        "--endpoint",
        help="OpenAI-compatible base URL. Default: Tinker's serving endpoint.",
    ),
    api_key_env: str = typer.Option(
        None,
        "--api-key-env",
        help="Env var holding the endpoint's API key. Default: TINKER_API_KEY on Tinker's own "
        "endpoint; on any other --endpoint the provider's WMO_ENDPOINT_API_KEY fallback is used, "
        "so a Tinker key is never sent to a host you named.",
    ),
    chat_max_tokens_field: str = typer.Option(
        None,
        "--chat-max-tokens-field",
        help="Output-budget parameter the endpoint accepts: max_tokens | max_completion_tokens. "
        "Default: max_tokens on Tinker's endpoint, max_completion_tokens on any other.",
    ),
    yes: bool = typer.Option(
        False, "--yes", help="Skip the confirmation when an entry of this name already exists."
    ),
) -> None:
    """Add a distilled student to the candidate pool, so the router can select it.

    The keystone step between training and serving: a run produces a `tinker://` adapter, and this
    turns it into a `[[model]]` entry the sweep measures, the fitter routes to, and the endpoint
    calls, with no hand-edited TOML in between:

        wmo optimize route student .wmo/distill/support --input-per-mtok 0.1 --output-per-mtok 0.4

    On Tinker's own endpoint the entry reads its credential from `TINKER_API_KEY`, so export that
    before serving. Point `--endpoint` somewhere else and the Tinker defaults do NOT follow: the
    entry falls back to `WMO_ENDPOINT_API_KEY` and to `max_completion_tokens`, so a Tinker key is
    never sent to a host you named. `--api-key-env` and `--chat-max-tokens-field` set either
    explicitly.

    To serve the student on its own with no measurement at all, follow this with
    `wmo optimize route pin <world-model> --model student`; to have the router CHOOSE between the
    student and the rest of the roster, run `wmo optimize route fit` on a matrix that covers both.
    """
    card_path = Path(card_dir) / MODEL_CARD_FILE
    if not card_path.is_file():
        raise typer.BadParameter(
            f"no {MODEL_CARD_FILE} at {card_path}; pass a distillation run directory (the one "
            "holding config.toml and metrics.jsonl) or an adapter version directory "
            "(.wmo/adapters/<name>/vN)"
        )
    if endpoint is not None and not endpoint.strip():
        # `--endpoint "$UNSET_VAR"` is the way this happens. Falling back to Tinker's endpoint
        # would silently serve a different host than the script meant to name.
        raise typer.BadParameter(
            "--endpoint is empty; give the OpenAI-compatible base URL, or drop the flag to use "
            "Tinker's serving endpoint"
        )
    if api_key_env is not None and not api_key_env.strip():
        # Same accident as an empty --endpoint. An empty string reaches `pool_provider` as a
        # falsy api_key_env, which it reads as "no explicit credential" and skips its own
        # unset-variable check, so the misconfiguration would only surface as a 401 at request
        # time with no hint.
        raise typer.BadParameter(
            "--api-key-env is empty; name the environment variable holding the endpoint's key, "
            "or drop the flag to use the provider's default credentials"
        )
    if chat_max_tokens_field is not None and chat_max_tokens_field not in _MAX_TOKENS_FIELDS:
        raise typer.BadParameter(
            f"unknown --chat-max-tokens-field {chat_max_tokens_field!r}; use "
            f"{' or '.join(_MAX_TOKENS_FIELDS)}"
        )
    try:
        card = DistillModelCard.model_validate_json(card_path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise typer.BadParameter(f"cannot read the model card at {card_path}: {exc}") from exc
    try:
        entry = student_pool_entry(
            card,
            name=name,
            input_per_mtok=input_per_mtok,
            output_per_mtok=output_per_mtok,
            endpoint=endpoint,
            api_key_env=api_key_env,
            chat_max_tokens_field=cast("ChatMaxTokensField | None", chat_max_tokens_field),
        )
    except ValidationError as exc:
        raise typer.BadParameter(f"cannot build a pool entry for '{name}': {exc}") from exc
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    pool_path = Path(pool)
    if _pool_has(pool_path, name) and not yes and not _confirm_replace(pool_path, name):
        _console.print(
            f"left {pool_path} unchanged; pass --yes to replace '{name}' (comments in the file "
            "are not preserved by a replacement), or --name <other> to keep both"
        )
        raise typer.Exit(0)
    try:
        replaced = upsert_pool_entry(entry, pool_path)
    except PoolLockTimeout as exc:
        # Nothing is wrong with the flags, so this is not a BadParameter: another writer is in the
        # way. Exit non-zero (and say to retry) so a script does not read it as a registration.
        _console.print(f"[red]pool busy[/red] {escape(str(exc))}")
        raise typer.Exit(1) from exc
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    verb = "replaced" if replaced else "added"
    _console.print(
        f"[green]✓[/green] {verb} pool candidate [bold]{name}[/bold] -> {pool_path}\n"
        f"  {card.base_model} adapter at {entry.model}\n"
        f"  ${input_per_mtok:g}/${output_per_mtok:g} per 1M in/out tokens, "
        f"{_credential_note(entry)}\n"
        f"  serve it directly: wmo optimize route pin <world-model> --model {name}",
        soft_wrap=True,
    )


def _credential_note(entry: PoolEntry) -> str:
    """How this entry authenticates, so the summary never names a key it will not send."""
    if entry.api_key_env is not None:
        return f"credential from {entry.api_key_env}"
    return "credential from WMO_ENDPOINT_API_KEY (the custom-endpoint fallback)"


def _pool_has(path: Path, name: str) -> bool:
    """Whether `path` already carries an entry called `name` (False when there is no pool yet)."""
    if not path.is_file():
        return False
    try:
        return any(entry.name == name for entry in load_pool(path).models)
    except (ValueError, FileNotFoundError):
        # An unreadable pool is upsert_pool_entry's error to raise, with its own message; do not
        # pre-empt it here with a confirmation prompt about an entry we cannot see.
        return False


def _confirm_replace(path: Path, name: str) -> bool:
    """Confirm repointing an existing pool handle; a non-interactive run declines."""
    try:
        return Confirm.ask(f"Replace the existing '{name}' entry in {path}?", default=False)
    except EOFError:
        return False


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
                "`wmo optimize route tune <policy.json> --cost-quality <0..1>`"
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
            "own row); measure on held-out scenarios with `wmo optimize route report`"
        )
        return
    _console.print(
        f"[green]✓[/green] fitted {len(policy.clusters)} clusters over "
        f"{result.scenarios} scenarios -> {out}\n"
        f"  fit-set accuracy {result.accuracy:.4f}, cost/scenario ${result.cost_per_scenario:.5f}"
    )


@route_app.command("pin")
def pin(
    world_model: str = typer.Argument(
        None, help="Built world model whose endpoint serves this policy. Default: the only one."
    ),
    model: str = typer.Option(
        ...,
        "--model",
        help="Pool entry every request goes to (a `wmo optimize route student` name).",
    ),
    pool: str = typer.Option(
        str(DEFAULT_POOL_PATH), "--pool", help="Candidate pool TOML to snapshot into the policy."
    ),
    root: str = typer.Option(ARTIFACT_DIR, "--root", help="Project dir holding the built models."),
    out: str = typer.Option(
        None, "--out", help="Override where the policy JSON lands (default: the model's own dir)."
    ),
    yes: bool = typer.Option(
        False, "--yes", help="Skip the confirmation when a policy is already installed."
    ),
) -> None:
    """Serve one pool model as an endpoint, with no matrix and no fit.

    A `kind="static"` policy sends every request to `--model`, which is all a single distilled
    student needs to be reachable through the OpenAI-compatible endpoint:

        wmo optimize route student .wmo/distill/support --input-per-mtok 0.1 --output-per-mtok 0.4
        wmo optimize route pin support --model student
        wmo serve --name support

    The policy is written to the world model's artifact dir, because that is where `wmo serve`
    looks for one. This is the honest "before" state the routing story is told against: a static
    endpoint has learned nothing and saves nothing, and `GET /v1/endpoints/<name>/savings` will
    say so. Replace it with `wmo optimize route fit` on a real outcome matrix to let the router
    choose per request.
    """
    try:
        model_dir = WorldModelStore(root).resolve(world_model)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    pool_path = Path(pool)
    try:
        roster = load_pool(pool_path)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except ValueError as exc:
        raise typer.BadParameter(f"cannot read the pool at {pool_path}: {exc}") from exc
    if all(entry.name != model for entry in roster.models):
        available = ", ".join(entry.name for entry in roster.models)
        raise typer.BadParameter(
            f"no pool model named '{model}' in {pool_path}; available: {available}"
        )
    out_path = Path(out) if out else model_dir / POLICY_FILENAME
    if out_path.is_file() and not yes and not _confirm_overwrite(out_path):
        _console.print(f"left {out_path} in place")
        raise typer.Exit(0)
    policy = RoutingPolicy(
        kind="static",
        default_model=model,
        pool=roster.models,
        fitted_from=f"pinned to {model} from {pool_path} (no outcome matrix)",
    )
    policy.save(out_path)
    _console.print(
        f"[green]✓[/green] pinned endpoint [bold]{model_dir.name}[/bold] to "
        f"[bold]{model}[/bold] -> {out_path}\n"
        f"  every request goes to {model}; nothing is measured and nothing is saved yet\n"
        f"  serve it: wmo serve --name {model_dir.name}\n"
        "  to let the router choose per request instead, replace this with "
        "`wmo optimize route fit <matrix.json>`",
        soft_wrap=True,
    )


def _confirm_overwrite(path: Path) -> bool:
    """Confirm replacing an installed policy; a non-interactive run declines.

    Worth asking about: the file being replaced may be a fitted knn policy, whose evidence bank
    sidecar this static policy will not use and does not remove.
    """
    try:
        return Confirm.ask(f"Replace the policy already at {path}?", default=False)
    except EOFError:
        return False


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
    `wmo.optimize.knn.apply_cost_quality`). The first run copies the un-tuned artifact to
    `policy.base.json` and every later run re-reads THAT, so the dial is always applied to the
    policy as fitted and sliding twice never compounds:

        wmo optimize route tune models/support/policy.json --cost-quality 0.6

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
            + (f"  [dim]{anchor.named_point}[/dim]" if anchor.named_point != "Custom" else "")
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
