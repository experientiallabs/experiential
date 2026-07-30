"""The reproduction runner: download, replay, compare, say REPRODUCED or DIVERGED.

Verdict semantics, per published row:

- `bit-exact` manifests compare at a float-noise epsilon on top of the row's tolerance
  (which is normally zero), so "REPRODUCED" there means the artifact's numbers ARE the
  published numbers.
- `protocol-exact` manifests compare within the row's stated relative tolerances, and the
  result says so: the claim is "the protocol produces numbers like these", not "these
  digits".

The runner never edits a manifest and never widens a tolerance: a DIVERGED verdict is a
result, and the caller decides what it means.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from wmo.optimize.knn import fit_knn_artifact
from wmo.optimize.outcomes import OutcomeMatrix, split_router_scenarios
from wmo.optimize.policy import EmbedderSpec, RoutingPolicy
from wmo.optimize.report import build_report
from wmo.reproduce.embedding import CachedTaskEmbedder
from wmo.reproduce.manifest import Manifest, PublishedRow

_EPSILON = 1e-9  # float-noise allowance on bit-exact comparisons


class RowVerdict(BaseModel):
    """One published row against what the reproduction measured."""

    model_config = ConfigDict(frozen=True)

    label: str
    reproduced: bool
    fields: dict[str, tuple[float, float, bool]]  # name -> (published, measured, ok)


class ReproduceResult(BaseModel):
    """The whole run: where the artifacts landed and how every row compared."""

    benchmark: str
    exactness: str
    generated_at: str
    out_dir: str
    rows: list[RowVerdict]
    notes: list[str]

    @property
    def reproduced(self) -> bool:
        return all(row.reproduced for row in self.rows)


def run_reproduction(
    manifest: Manifest,
    *,
    out_dir: Path,
    data_dir: Path | None = None,
    approve_spend: bool = False,
) -> ReproduceResult:
    """Run one manifest end to end.

    Args:
        manifest: the recipe (see `wmo.reproduce.manifest`).
        out_dir: where artifacts (policy, reports, verdict.json) land; created if missing.
        data_dir: an existing snapshot directory to use instead of downloading - the seam
            tests use, and the offline path for someone who already has the data.
        approve_spend: `commands` manifests spend real money; the CLI passes this only
            after explicit consent. `matrix` manifests ignore it (they are free).

    Raises:
        PermissionError: a `commands` manifest without `approve_spend` (the message carries
            the manifest's own spend estimate; nothing has run).
        ValueError: the downloaded data does not match the manifest's expectations.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot = data_dir if data_dir is not None else _download(manifest, out_dir)

    if manifest.kind == "matrix":
        report_paths = _run_matrix(manifest, snapshot, out_dir)
    else:
        if not approve_spend:
            assert manifest.commands is not None
            raise PermissionError(
                f"reproducing '{manifest.name}' replays live commands with an estimated spend "
                f"of ~${manifest.commands.estimated_spend_usd:.0f}; pass --yes to approve"
            )
        report_paths = _run_commands(manifest, snapshot, out_dir)

    rows = [_compare(row, report_paths, manifest) for row in manifest.published]
    result = ReproduceResult(
        benchmark=manifest.name,
        exactness=manifest.exactness,
        generated_at=datetime.now(UTC).isoformat(),
        out_dir=str(out_dir),
        rows=rows,
        notes=list(manifest.notes),
    )
    (out_dir / "verdict.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return result


def _download(manifest: Manifest, out_dir: Path) -> Path:
    """Fetch the manifest's pinned files from Hugging Face into `<out>/data`."""
    from huggingface_hub import hf_hub_download  # deferred: heavy import

    target = out_dir / "data"
    target.mkdir(parents=True, exist_ok=True)
    for filename in manifest.data.files:
        hf_hub_download(
            repo_id=manifest.data.hf_repo,
            repo_type=manifest.data.repo_type,
            revision=manifest.data.revision,
            filename=filename,
            local_dir=target,
        )
    return target


def _run_matrix(manifest: Manifest, snapshot: Path, out_dir: Path) -> dict[str, Path]:
    """The offline path: fit + dial + one report per baseline, recorded vectors throughout."""
    protocol = manifest.matrix
    assert protocol is not None
    matrix_path = snapshot / protocol.matrix_file
    matrix = OutcomeMatrix.load(matrix_path)

    built = None
    if protocol.embedding_cache_file is not None:
        built = CachedTaskEmbedder(matrix, snapshot / protocol.embedding_cache_file)

    policy_path = out_dir / "policy.json"
    if protocol.pin_model is not None:
        # The bench-defaults protocol: a static pin (what `wmo optimize route
        # pin` writes) instead of a kNN fit. A pin routes nothing, so the
        # replay needs no embedder and no cache; it is arithmetic over the
        # matrix, which is why these manifests can honestly claim bit-exact.
        pin = RoutingPolicy(
            kind="static",
            default_model=protocol.pin_model,
            pool=list(matrix.pool),
            fitted_from=f"pinned to {protocol.pin_model} from the published matrix (reproduction)",
        )
        pin.save(policy_path)
        policy = pin
    else:
        spec = EmbedderSpec(
            kind=protocol.embedder_kind,
            dim=protocol.embedder_dim,
            deployment=protocol.embedder_deployment,
            endpoint=protocol.embedder_endpoint,
        )
        # The same deterministic 70/30 scenario split the CLI fit computes, so the manifest
        # reproduces the shipped protocol rather than a private variant of it.
        split = split_router_scenarios(matrix.scenario_ids())
        fit_knn_artifact(
            matrix,
            out_path=policy_path,
            matrix_source=str(matrix_path),
            embedder=spec,
            fit_ids=list(split.fit_ids),
            fallback=protocol.fallback,
            built=built,
        )
        policy = RoutingPolicy.load(policy_path)
        policy = policy.model_copy(update={"cost_quality": protocol.cost_quality})

    reports: dict[str, Path] = {}
    for baseline in protocol.baselines:
        report = build_report(
            matrix,
            policy,
            baseline=baseline,
            endpoint=f"reproduce-{manifest.name}",
            generated_at=datetime.now(UTC).isoformat(),
            built=built,
        )
        path = out_dir / f"report_vs_{baseline}.json"
        path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        reports[baseline] = path
    return reports


def _run_commands(manifest: Manifest, snapshot: Path, out_dir: Path) -> dict[str, Path]:
    """The live path: the cookbook's own commands, verbatim, with {data}/{out} expanded."""
    protocol = manifest.commands
    assert protocol is not None
    if protocol.pool_file is not None:
        from importlib import resources

        from wmo.reproduce.manifest import MANIFEST_PACKAGE

        pool_text = (resources.files(MANIFEST_PACKAGE) / protocol.pool_file).read_text(
            encoding="utf-8"
        )
        (out_dir / "pool.toml").write_text(pool_text, encoding="utf-8")
    for argv in protocol.steps:
        expanded = [
            arg.replace("{data}", str(snapshot)).replace("{out}", str(out_dir)) for arg in argv
        ]
        # Through the installed CLI so the replay is exactly what the cookbook documents.
        completed = subprocess.run(  # noqa: S603 - argv comes from a committed manifest
            [sys.executable, "-m", "wmo", *expanded],
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"reproduction step failed (exit {completed.returncode}): wmo {' '.join(expanded)}"
            )
    return {"__report__": out_dir / protocol.report_file}


def _compare(row: PublishedRow, reports: dict[str, Path], manifest: Manifest) -> RowVerdict:
    """One published row against the produced report's headline."""
    path = reports.get(row.baseline or "__report__")
    if path is None and len(reports) == 1:
        path = next(iter(reports.values()))
    if path is None:
        raise ValueError(
            f"published row '{row.label}' names baseline '{row.baseline}' but the run produced "
            f"reports for: {', '.join(sorted(reports))}"
        )
    headline = json.loads(path.read_text(encoding="utf-8"))["headline"]

    fields: dict[str, tuple[float, float, bool]] = {}

    def check(name: str, published: float, measured: float, tolerance: float) -> None:
        allowed = abs(published) * tolerance + _EPSILON
        fields[name] = (published, measured, abs(measured - published) <= allowed)

    check("accuracy", row.accuracy, headline["accuracy"], row.tolerance_accuracy)
    check(
        "cost_per_run_usd", row.cost_per_run_usd, headline["cost_per_run_usd"], row.tolerance_cost
    )
    if row.latency_p50_ms is not None:
        check(
            "latency_p50_ms", row.latency_p50_ms, headline["latency_p50_ms"], row.tolerance_latency
        )
    return RowVerdict(
        label=row.label,
        reproduced=all(ok for _, _, ok in fields.values()),
        fields=fields,
    )
