"""Search external-only coding router algorithms and evaluate frozen policies on DeepSWE.

This research runner has two explicit phases:

* ``fit`` reads only external coding-task text and execution outcomes. It performs
  repository-grouped cross-validation, freezes candidate models and operating points, and writes
  an append-only trial ledger.
* ``evaluate`` loads those frozen candidates, then opens the published DeepSWE v1.1 matrix exactly
  for evaluation. DeepSWE outcomes never enter a fitted feature transform, estimator, or threshold.

The intended execution location is remote compute. The local workstation may upload inputs, start
the job, and sync artifacts, but must not run the fitting or bootstrap phases.
"""

from __future__ import annotations

import argparse
import collections
import csv
import dataclasses
import hashlib
import importlib.util
import json
import logging
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Literal, Protocol, TypedDict, cast

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("coding-model_router_autoresearch")

QUALITY_FLOORS = (0.95, 0.97, 0.99)
FOLDS = 5
BOOTSTRAP_SAMPLES = 10_000
PRIMARY_TARGET_FAMILY = "mini_swe_agent_claude_opus_5"
TARGET_LADDERS: dict[str, tuple[str, ...]] = {
    "opus-low-high": (
        "mini_swe_agent_claude_opus_5_low",
        "mini_swe_agent_claude_opus_5_high",
    ),
    "luna-xhigh-max": (
        "mini_swe_agent_gpt_5_6_luna_xhigh",
        "mini_swe_agent_gpt_5_6_luna_max",
    ),
    "luna-xhigh-opus-high": (
        "mini_swe_agent_gpt_5_6_luna_xhigh",
        "mini_swe_agent_claude_opus_5_high",
    ),
    "luna-xhigh-max-opus-high": (
        "mini_swe_agent_gpt_5_6_luna_xhigh",
        "mini_swe_agent_gpt_5_6_luna_max",
        "mini_swe_agent_claude_opus_5_high",
    ),
}


class ExternalTaskRow(TypedDict):
    """One cached Nebius task row."""

    instance_id: str
    repo: str
    text: str
    cheap_reward: float
    strong_reward: float
    cheap_attempts: int
    strong_attempts: int


class JsonObject(Protocol):
    """Protocol for JSON mappings used by the one-off runner."""

    def get(self, key: str, default: object = ...) -> object: ...


class FittedRegressor(Protocol):
    """Estimator surface shared by the searched scikit-learn regressors."""

    def fit(
        self,
        features: np.ndarray,
        target: np.ndarray,
        *,
        sample_weight: np.ndarray,
    ) -> object: ...

    def predict(self, features: np.ndarray) -> np.ndarray: ...


class StaticRow(TypedDict):
    """One DeepSWE static-arm aggregate."""

    arm: str
    reward: float
    cost_usd: float


@dataclasses.dataclass(frozen=True)
class SourceData:
    """One external source normalized to weak and strong execution outcomes."""

    name: str
    task_ids: list[str]
    groups: list[str]
    texts: list[str]
    weak: np.ndarray
    strong: np.ndarray
    weak_attempts: np.ndarray
    strong_attempts: np.ndarray


@dataclasses.dataclass(frozen=True)
class CombinedData:
    """Deduplicated external task rows and source-balanced sample weights."""

    source_names: list[str]
    task_ids: list[str]
    groups: list[str]
    texts: list[str]
    weak: np.ndarray
    strong: np.ndarray
    sample_weight: np.ndarray


@dataclasses.dataclass(frozen=True)
class CandidateSpec:
    """One mechanically searchable static-text estimator."""

    name: str
    analyzer: Literal["word", "char"]
    components: int
    estimator: Literal["ridge-uplift", "ridge-heads", "extra-heads", "hist-heads"]
    alpha: float = 1.0
    min_leaf: int = 10


@dataclasses.dataclass(frozen=True)
class TargetData:
    """Published DeepSWE task text, group, arm reward, and arm cost matrices."""

    task_ids: list[str]
    texts: list[str]
    groups: list[str]
    arms: list[str]
    rewards: np.ndarray
    costs: np.ndarray


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise TypeError(f"expected numeric value, found {type(value).__name__}")
    return float(value)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()


def _canonical_group(value: str) -> str:
    normalized = value.strip().lower().replace("__", "/")
    return normalized or "unknown"


def _empirical_bayes_rate(successes: float, attempts: int, global_mean: float) -> float:
    """Shrink a repeated success rate toward its source-wide mean."""
    prior_strength = 4.0
    alpha = 1.0 + prior_strength * global_mean
    beta = 1.0 + prior_strength * (1.0 - global_mean)
    return (successes + alpha) / (attempts + alpha + beta)


def _load_nebius(path: Path) -> SourceData:
    raw = _read_json(path)
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a JSON list")
    rows = [cast(ExternalTaskRow, item) for item in raw if isinstance(item, dict)]
    weak_total = sum(row["cheap_reward"] * row["cheap_attempts"] for row in rows)
    weak_attempts = sum(row["cheap_attempts"] for row in rows)
    strong_total = sum(row["strong_reward"] * row["strong_attempts"] for row in rows)
    strong_attempts = sum(row["strong_attempts"] for row in rows)
    weak_mean = weak_total / weak_attempts
    strong_mean = strong_total / strong_attempts
    weak = np.asarray(
        [
            _empirical_bayes_rate(
                row["cheap_reward"] * row["cheap_attempts"],
                row["cheap_attempts"],
                weak_mean,
            )
            for row in rows
        ],
        dtype=np.float64,
    )
    strong = np.asarray(
        [
            _empirical_bayes_rate(
                row["strong_reward"] * row["strong_attempts"],
                row["strong_attempts"],
                strong_mean,
            )
            for row in rows
        ],
        dtype=np.float64,
    )
    return SourceData(
        name="nebius-swe-agent-8b-70b",
        task_ids=[row["instance_id"] for row in rows],
        groups=[_canonical_group(row["repo"]) for row in rows],
        texts=[row["text"] for row in rows],
        weak=weak,
        strong=strong,
        weak_attempts=np.asarray([row["cheap_attempts"] for row in rows], dtype=np.float64),
        strong_attempts=np.asarray([row["strong_attempts"] for row in rows], dtype=np.float64),
    )


def _load_r2e(loader_path: Path) -> SourceData:
    spec = importlib.util.spec_from_file_location("external_r2e_loader", loader_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load R2E loader from {loader_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    loaded = cast(dict[str, object], module.load())  # type: ignore[attr-defined]
    tasks = cast(list[str], loaded["tasks"])
    arms = cast(list[str], loaded["arms"])
    scores = np.asarray(loaded["score_binary"], dtype=np.float64)
    means = np.nanmean(scores, axis=1)
    strong_index = int(np.nanargmax(means))
    weak_index = int(np.nanargmin(means))
    texts = cast(dict[str, str], loaded["text"])
    groups = cast(dict[str, str], loaded["group"])
    logger.info(
        "R2E normalized with weak=%s mean=%.4f strong=%s mean=%.4f",
        arms[weak_index],
        means[weak_index],
        arms[strong_index],
        means[strong_index],
    )
    return SourceData(
        name="r2e-gym-terminus",
        task_ids=tasks,
        groups=[_canonical_group(groups[task_id]) for task_id in tasks],
        texts=[texts[task_id] for task_id in tasks],
        weak=scores[weak_index],
        strong=scores[strong_index],
        weak_attempts=np.ones(len(tasks), dtype=np.float64),
        strong_attempts=np.ones(len(tasks), dtype=np.float64),
    )


def _coderouter_group(row: dict[str, object]) -> str:
    original = str(row.get("original_task_id", ""))
    if "__" in original:
        return _canonical_group(original.split("__", 1)[0])
    source = str(row.get("source_dataset", row.get("bench", "unknown")))
    return _canonical_group(source)


def _load_coderouter(root: Path) -> SourceData:
    task_path = root / "data" / "coderouterbench" / "ood176_tasks.jsonl"
    result_path = root / "data" / "coderouterbench" / "ood176_results_long.csv"
    tasks: dict[str, dict[str, object]] = {}
    for line in task_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if isinstance(row, dict):
            tasks[str(row["task_id"])] = {str(key): value for key, value in row.items()}
    cells: dict[tuple[str, str], float] = {}
    with result_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["resolved"] != "":
                cells[(row["model"], row["task_id"])] = float(row["resolved"])
    requested = ("Qwen3-Max", "gpt-5.4")
    task_ids = sorted(
        task_id for task_id in tasks if all((arm, task_id) in cells for arm in requested)
    )
    return SourceData(
        name="coderouterbench-ood176",
        task_ids=task_ids,
        groups=[_coderouter_group(tasks[task_id]) for task_id in task_ids],
        texts=[str(tasks[task_id]["prompt"]) for task_id in task_ids],
        weak=np.asarray([cells[(requested[0], task_id)] for task_id in task_ids]),
        strong=np.asarray([cells[(requested[1], task_id)] for task_id in task_ids]),
        weak_attempts=np.ones(len(task_ids), dtype=np.float64),
        strong_attempts=np.ones(len(task_ids), dtype=np.float64),
    )


def _combine(sources: list[SourceData]) -> CombinedData:
    counts = {source.name: len(source.task_ids) for source in sources}
    seen_text: set[str] = set()
    source_names: list[str] = []
    task_ids: list[str] = []
    groups: list[str] = []
    texts: list[str] = []
    weak: list[float] = []
    strong: list[float] = []
    weights: list[float] = []
    for source in sources:
        for index, text in enumerate(source.texts):
            digest = hashlib.sha256(" ".join(text.split()).encode()).hexdigest()
            if digest in seen_text:
                continue
            seen_text.add(digest)
            source_names.append(source.name)
            task_ids.append(source.task_ids[index])
            groups.append(source.groups[index])
            texts.append(text)
            weak.append(float(source.weak[index]))
            strong.append(float(source.strong[index]))
            weights.append(1.0 / counts[source.name])
    weight_array = np.asarray(weights, dtype=np.float64)
    weight_array *= len(weight_array) / weight_array.sum()
    return CombinedData(
        source_names=source_names,
        task_ids=task_ids,
        groups=groups,
        texts=texts,
        weak=np.asarray(weak, dtype=np.float64),
        strong=np.asarray(strong, dtype=np.float64),
        sample_weight=weight_array,
    )


def _candidate_space() -> list[CandidateSpec]:
    return [
        CandidateSpec("word64-ridge-uplift-a1", "word", 64, "ridge-uplift", alpha=1.0),
        CandidateSpec("word128-ridge-uplift-a10", "word", 128, "ridge-uplift", alpha=10.0),
        CandidateSpec("char128-ridge-uplift-a10", "char", 128, "ridge-uplift", alpha=10.0),
        CandidateSpec("word128-ridge-heads-a1", "word", 128, "ridge-heads", alpha=1.0),
        CandidateSpec("char128-ridge-heads-a1", "char", 128, "ridge-heads", alpha=1.0),
        CandidateSpec("word128-extra-heads-l5", "word", 128, "extra-heads", min_leaf=5),
        CandidateSpec("word128-extra-heads-l20", "word", 128, "extra-heads", min_leaf=20),
        CandidateSpec("word128-hist-heads-l10", "word", 128, "hist-heads", min_leaf=10),
        CandidateSpec("word128-hist-heads-l30", "word", 128, "hist-heads", min_leaf=30),
    ]


def _features(spec: CandidateSpec) -> Pipeline:
    if spec.analyzer == "word":
        vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.995,
            max_features=50_000,
            sublinear_tf=True,
            strip_accents="unicode",
        )
    else:
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=3,
            max_features=60_000,
            sublinear_tf=True,
        )
    return Pipeline(
        [
            ("tfidf", vectorizer),
            ("svd", TruncatedSVD(n_components=spec.components, random_state=17)),
            ("scale", StandardScaler()),
        ]
    )


def _estimators(spec: CandidateSpec) -> tuple[FittedRegressor, FittedRegressor | None]:
    if spec.estimator in ("ridge-uplift", "ridge-heads"):
        first = Ridge(alpha=spec.alpha)
        second = Ridge(alpha=spec.alpha) if spec.estimator == "ridge-heads" else None
        return cast(FittedRegressor, first), cast(FittedRegressor | None, second)
    if spec.estimator == "extra-heads":
        first = ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=spec.min_leaf,
            max_features=0.7,
            n_jobs=-1,
            random_state=29,
        )
        second = ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=spec.min_leaf,
            max_features=0.7,
            n_jobs=-1,
            random_state=31,
        )
        return cast(FittedRegressor, first), cast(FittedRegressor, second)
    first = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=spec.min_leaf,
        l2_regularization=1.0,
        random_state=37,
    )
    second = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=spec.min_leaf,
        l2_regularization=1.0,
        random_state=41,
    )
    return cast(FittedRegressor, first), cast(FittedRegressor, second)


def _fit_estimators(
    spec: CandidateSpec,
    features: np.ndarray,
    weak: np.ndarray,
    strong: np.ndarray,
    weights: np.ndarray,
) -> tuple[FittedRegressor, FittedRegressor | None]:
    first, second = _estimators(spec)
    if spec.estimator == "ridge-uplift":
        first.fit(features, strong - weak, sample_weight=weights)
        return first, None
    first.fit(features, weak, sample_weight=weights)
    if second is None:
        raise AssertionError(f"{spec.name} requires two potential-outcome heads")
    second.fit(features, strong, sample_weight=weights)
    return first, second


def _predict_score(
    spec: CandidateSpec,
    estimators: tuple[FittedRegressor, FittedRegressor | None],
    features: np.ndarray,
) -> np.ndarray:
    first, second = estimators
    if spec.estimator == "ridge-uplift":
        return np.asarray(first.predict(features), dtype=np.float64)
    if second is None:
        raise AssertionError(f"{spec.name} has no strong-outcome head")
    weak = np.clip(np.asarray(first.predict(features), dtype=np.float64), 0.0, 1.0)
    strong = np.clip(np.asarray(second.predict(features), dtype=np.float64), 0.0, 1.0)
    return strong - weak


def _operating_point(
    scores: np.ndarray,
    weak: np.ndarray,
    strong: np.ndarray,
    source_names: list[str],
    quality_floor: float,
) -> dict[str, float]:
    """Choose the least strong traffic that meets source-balanced external quality."""
    unique_sources = sorted(set(source_names))
    source_array = np.asarray(source_names, dtype=object)
    thresholds = np.unique(
        np.concatenate(
            [
                np.quantile(scores, np.linspace(0.0, 1.0, 401)),
                np.asarray([np.nextafter(scores.max(), math.inf)]),
            ]
        )
    )
    feasible: list[dict[str, float]] = []
    for threshold in thresholds:
        use_strong = scores >= threshold
        routed = np.where(use_strong, strong, weak)
        retentions: list[float] = []
        traffic: list[float] = []
        for source in unique_sources:
            mask = source_array == source
            baseline = float(strong[mask].mean())
            retentions.append(float(routed[mask].mean() / baseline) if baseline else 1.0)
            traffic.append(float(use_strong[mask].mean()))
        mean_retention = float(np.mean(retentions))
        minimum_retention = float(np.min(retentions))
        if mean_retention >= quality_floor and minimum_retention >= quality_floor - 0.05:
            feasible.append(
                {
                    "threshold": float(threshold),
                    "strong_traffic": float(np.mean(traffic)),
                    "mean_retention": mean_retention,
                    "minimum_source_retention": minimum_retention,
                }
            )
    if not feasible:
        return {
            "threshold": float(np.nextafter(scores.min(), -math.inf)),
            "strong_traffic": 1.0,
            "mean_retention": 1.0,
            "minimum_source_retention": 1.0,
        }
    return min(
        feasible,
        key=lambda row: (
            row["strong_traffic"],
            -row["minimum_source_retention"],
            -row["mean_retention"],
        ),
    )


def _source_metrics(
    scores: np.ndarray,
    weak: np.ndarray,
    strong: np.ndarray,
    source_names: list[str],
    threshold: float,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    source_array = np.asarray(source_names, dtype=object)
    for source in sorted(set(source_names)):
        mask = source_array == source
        use_strong = scores[mask] >= threshold
        routed = np.where(use_strong, strong[mask], weak[mask])
        baseline = float(strong[mask].mean())
        result[source] = {
            "tasks": int(mask.sum()),
            "weak_reward": float(weak[mask].mean()),
            "strong_reward": baseline,
            "router_reward": float(routed.mean()),
            "quality_retention": float(routed.mean() / baseline) if baseline else 1.0,
            "strong_traffic": float(use_strong.mean()),
            "uplift_spearman": _spearman(scores[mask], (strong - weak)[mask]),
        }
    return result


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.all(left == left[0]) or np.all(right == right[0]):
        return 0.0
    left_rank = np.argsort(np.argsort(left, kind="stable"), kind="stable").astype(np.float64)
    right_rank = np.argsort(np.argsort(right, kind="stable"), kind="stable").astype(np.float64)
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _fit(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = [
        _load_nebius(args.nebius_tasks.resolve()),
        _load_r2e(args.r2e_loader.resolve()),
        _load_coderouter(args.coderouter_root.resolve()),
    ]
    combined = _combine(sources)
    _write_json(
        output / "external-sources.json",
        {
            "sources": [
                {
                    "name": source.name,
                    "tasks": len(source.task_ids),
                    "groups": len(set(source.groups)),
                    "weak_mean": float(source.weak.mean()),
                    "strong_mean": float(source.strong.mean()),
                }
                for source in sources
            ],
            "deduplicated_tasks": len(combined.task_ids),
            "deduplicated_groups": len(set(combined.groups)),
            "deep_swe_labels_read": False,
        },
    )
    folds = list(
        GroupKFold(n_splits=FOLDS).split(
            np.arange(len(combined.task_ids)),
            groups=np.asarray(combined.groups, dtype=object),
        )
    )
    leaderboard: list[dict[str, object]] = []
    for spec in _candidate_space():
        oof = np.empty(len(combined.task_ids), dtype=np.float64)
        for fold_index, (train, heldout) in enumerate(folds):
            transformer = _features(spec)
            train_features = np.asarray(
                transformer.fit_transform([combined.texts[index] for index in train]),
                dtype=np.float64,
            )
            heldout_features = np.asarray(
                transformer.transform([combined.texts[index] for index in heldout]),
                dtype=np.float64,
            )
            estimators = _fit_estimators(
                spec,
                train_features,
                combined.weak[train],
                combined.strong[train],
                combined.sample_weight[train],
            )
            oof[heldout] = _predict_score(spec, estimators, heldout_features)
            logger.info("candidate=%s fold=%d/%d complete", spec.name, fold_index + 1, FOLDS)
        operating_points = {
            str(floor): _operating_point(
                oof,
                combined.weak,
                combined.strong,
                combined.source_names,
                floor,
            )
            for floor in QUALITY_FLOORS
        }
        primary = operating_points[str(QUALITY_FLOORS[0])]
        source_metrics = _source_metrics(
            oof,
            combined.weak,
            combined.strong,
            combined.source_names,
            primary["threshold"],
        )
        row: dict[str, object] = {
            "candidate": dataclasses.asdict(spec),
            "external_oof_uplift_spearman": _spearman(
                oof,
                combined.strong - combined.weak,
            ),
            "operating_points": operating_points,
            "primary_source_metrics": source_metrics,
            "deep_swe_labels_read": False,
        }
        leaderboard.append(row)
        _append_jsonl(output / "trials.jsonl", row)

        transformer = _features(spec)
        full_features = np.asarray(transformer.fit_transform(combined.texts), dtype=np.float64)
        estimators = _fit_estimators(
            spec,
            full_features,
            combined.weak,
            combined.strong,
            combined.sample_weight,
        )
        joblib.dump(
            {
                "spec": dataclasses.asdict(spec),
                "transformer": transformer,
                "weak_estimator": estimators[0],
                "strong_estimator": estimators[1],
            },
            output / f"{spec.name}.joblib",
            compress=3,
        )
    leaderboard.sort(
        key=lambda row: (
            cast(dict[str, dict[str, float]], row["operating_points"])["0.95"][
                "strong_traffic"
            ],
            -float(row["external_oof_uplift_spearman"]),
        )
    )
    _write_json(
        output / "frozen-candidates.json",
        {
            "protocol": {
                "fit_sources": [source.name for source in sources],
                "folds": FOLDS,
                "grouping": "canonical_repository",
                "source_weighting": "equal_total_weight_per_source",
                "quality_floors": list(QUALITY_FLOORS),
                "target_outcomes_used": False,
                "target_embeddings_used": False,
                "selection": "minimum_source_balanced_strong_traffic_then_uplift_spearman",
            },
            "leaderboard": leaderboard,
        },
    )
    logger.info(
        "external fit complete: tasks=%d candidates=%d leader=%s",
        len(combined.task_ids),
        len(leaderboard),
        cast(dict[str, object], leaderboard[0]["candidate"])["name"],
    )


def _load_target(matrix_path: Path, task_meta_path: Path) -> TargetData:
    matrix = _read_json(matrix_path)
    if not isinstance(matrix, dict):
        raise ValueError(f"{matrix_path} must contain one JSON object")
    matrix_object = {str(key): value for key, value in matrix.items()}
    outcomes = matrix_object.get("outcomes")
    if not isinstance(outcomes, list):
        raise ValueError(f"{matrix_path} has no outcomes list")
    meta = _read_json(task_meta_path)
    if not isinstance(meta, dict):
        raise ValueError(f"{task_meta_path} must contain one JSON object")
    meta_object = {str(key): value for key, value in meta.items()}
    if not isinstance(meta_object.get("rows"), list):
        raise ValueError(f"{task_meta_path} has no task rows")
    task_rows = cast(list[dict[str, object]], meta_object["rows"])
    groups = {str(row["id"]): str(row["repository"]) for row in task_rows}
    texts: dict[str, str] = {}
    cells: dict[tuple[str, str], tuple[float, float]] = {}
    for untyped in outcomes:
        if not isinstance(untyped, dict):
            continue
        row = {str(key): value for key, value in untyped.items()}
        task_id = str(row["scenario_id"])
        arm = str(row["model"])
        texts.setdefault(task_id, str(row["task"]))
        cells[(arm, task_id)] = (_as_float(row["reward"]), _as_float(row["cost_usd"]))
    arms = sorted({arm for arm, _ in cells})
    task_ids = sorted(texts)
    complete = [
        task_id
        for task_id in task_ids
        if task_id in groups and all((arm, task_id) in cells for arm in arms)
    ]
    rewards = np.asarray(
        [[cells[(arm, task_id)][0] for task_id in complete] for arm in arms],
        dtype=np.float64,
    )
    costs = np.asarray(
        [[cells[(arm, task_id)][1] for task_id in complete] for arm in arms],
        dtype=np.float64,
    )
    return TargetData(
        task_ids=complete,
        texts=[texts[task_id] for task_id in complete],
        groups=[_canonical_group(groups[task_id]) for task_id in complete],
        arms=arms,
        rewards=rewards,
        costs=costs,
    )


def _candidate_score(path: Path, texts: list[str]) -> np.ndarray:
    fitted = cast(dict[str, object], joblib.load(path))
    raw_spec = cast(dict[str, object], fitted["spec"])
    analyzer = str(raw_spec["analyzer"])
    estimator = str(raw_spec["estimator"])
    if analyzer not in ("word", "char"):
        raise ValueError(f"invalid frozen analyzer {analyzer!r}")
    if estimator not in ("ridge-uplift", "ridge-heads", "extra-heads", "hist-heads"):
        raise ValueError(f"invalid frozen estimator {estimator!r}")
    spec = CandidateSpec(
        name=str(raw_spec["name"]),
        analyzer=cast(Literal["word", "char"], analyzer),
        components=int(_as_float(raw_spec["components"])),
        estimator=cast(
            Literal["ridge-uplift", "ridge-heads", "extra-heads", "hist-heads"],
            estimator,
        ),
        alpha=_as_float(raw_spec.get("alpha", 1.0)),
        min_leaf=int(_as_float(raw_spec.get("min_leaf", 10))),
    )
    transformer = cast(Pipeline, fitted["transformer"])
    features = np.asarray(transformer.transform(texts), dtype=np.float64)
    estimators = (
        cast(FittedRegressor, fitted["weak_estimator"]),
        cast(FittedRegressor | None, fitted["strong_estimator"]),
    )
    return _predict_score(spec, estimators, features)


def _ladder_indices(target: TargetData, ladder: tuple[str, ...]) -> np.ndarray:
    positions = {arm: index for index, arm in enumerate(target.arms)}
    missing = [arm for arm in ladder if arm not in positions]
    if missing:
        raise ValueError(f"DeepSWE matrix is missing frozen ladder arms: {missing}")
    return np.asarray([positions[arm] for arm in ladder], dtype=np.int64)


def _thresholds(
    operating_points: dict[str, dict[str, float]],
    ladder_size: int,
) -> list[float]:
    if ladder_size == 2:
        return [operating_points["0.95"]["threshold"]]
    if ladder_size == 3:
        return [
            operating_points["0.99"]["threshold"],
            operating_points["0.95"]["threshold"],
        ]
    raise ValueError(f"unsupported target ladder size {ladder_size}")


def _route_indices(scores: np.ndarray, thresholds: list[float], ladder_size: int) -> np.ndarray:
    if ladder_size == 2:
        return (scores >= thresholds[0]).astype(np.int64)
    lower, upper = sorted(thresholds)
    return np.where(scores >= upper, 2, np.where(scores >= lower, 1, 0)).astype(np.int64)


def _bootstrap(
    router_reward: np.ndarray,
    router_cost: np.ndarray,
    baseline_reward: np.ndarray,
    baseline_cost: np.ndarray,
    groups: list[str],
) -> dict[str, list[float]]:
    rng = np.random.default_rng(73)
    unique = sorted(set(groups))
    group_array = np.asarray(groups, dtype=object)
    by_group = {group: np.flatnonzero(group_array == group) for group in unique}
    deltas = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    ratios = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    for sample in range(BOOTSTRAP_SAMPLES):
        selected_groups = rng.choice(unique, size=len(unique), replace=True)
        selected = np.concatenate([by_group[str(group)] for group in selected_groups])
        deltas[sample] = float(
            router_reward[selected].mean() - baseline_reward[selected].mean()
        )
        router_total = float(router_cost[selected].sum())
        ratios[sample] = (
            float(baseline_cost[selected].sum() / router_total)
            if router_total
            else math.inf
        )
    return {
        "quality_delta_95ci": [
            float(value) for value in np.quantile(deltas, [0.025, 0.975])
        ],
        "cost_ratio_95ci": [
            float(value) for value in np.quantile(ratios, [0.025, 0.975])
        ],
    }


def _static_rows(target: TargetData) -> list[StaticRow]:
    return [
        {
            "arm": arm,
            "reward": float(target.rewards[index].mean()),
            "cost_usd": float(target.costs[index].sum()),
        }
        for index, arm in enumerate(target.arms)
    ]


def _evaluate(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    frozen = _read_json(output / "frozen-candidates.json")
    if not isinstance(frozen, dict):
        raise ValueError("frozen-candidates.json is absent or invalid")
    frozen_object = {str(key): value for key, value in frozen.items()}
    if not isinstance(frozen_object.get("leaderboard"), list):
        raise ValueError("frozen-candidates.json is absent or invalid")
    target = _load_target(args.deep_matrix.resolve(), args.deep_tasks.resolve())
    static = _static_rows(target)
    best_quality = max(static, key=lambda row: float(row["reward"]))
    rows: list[dict[str, object]] = []
    for untyped in cast(list[dict[str, object]], frozen_object["leaderboard"]):
        candidate = cast(dict[str, object], untyped["candidate"])
        name = str(candidate["name"])
        scores = _candidate_score(output / f"{name}.joblib", target.texts)
        operating_points = cast(dict[str, dict[str, float]], untyped["operating_points"])
        for ladder_name, ladder in TARGET_LADDERS.items():
            arm_indices = _ladder_indices(target, ladder)
            thresholds = _thresholds(operating_points, len(ladder))
            decisions = _route_indices(scores, thresholds, len(ladder))
            columns = np.arange(len(target.task_ids), dtype=np.int64)
            selected_arms = arm_indices[decisions]
            routed_reward = target.rewards[selected_arms, columns]
            routed_cost = target.costs[selected_arms, columns]
            router_reward = float(routed_reward.mean())
            router_cost = float(routed_cost.sum())
            dominating = [
                row
                for row in static
                if row["reward"] >= router_reward and row["cost_usd"] <= router_cost
            ]
            comparable = [
                row for row in static if row["reward"] >= router_reward
            ]
            matched_static = (
                min(comparable, key=lambda row: row["cost_usd"])
                if comparable
                else best_quality
            )
            baseline_index = target.arms.index(str(matched_static["arm"]))
            counts = collections.Counter(int(value) for value in decisions)
            row: dict[str, object] = {
                "candidate": name,
                "ladder": ladder_name,
                "arms": list(ladder),
                "thresholds": thresholds,
                "tasks": len(target.task_ids),
                "repositories": len(set(target.groups)),
                "router_reward": router_reward,
                "router_cost_usd": router_cost,
                "quality_retention_vs_best_static": (
                    router_reward / best_quality["reward"]
                    if best_quality["reward"]
                    else 0.0
                ),
                "best_static_quality_arm": best_quality,
                "matched_static": matched_static,
                "cost_ratio_vs_matched_static": (
                    matched_static["cost_usd"] / router_cost
                    if router_cost
                    else math.inf
                ),
                "dominated_by_static": bool(dominating),
                "dominating_static_arms": dominating,
                "traffic": {
                    ladder[index]: counts.get(index, 0) for index in range(len(ladder))
                },
                "target_labels_used_for_fit": False,
                "target_labels_used_for_thresholds": False,
                "target_static_aggregates_used_for_ladder_design": True,
                **_bootstrap(
                    routed_reward,
                    routed_cost,
                    target.rewards[baseline_index],
                    target.costs[baseline_index],
                    target.groups,
                ),
            }
            rows.append(row)
            _append_jsonl(output / "target-trials.jsonl", row)
    rows.sort(
        key=lambda row: (
            bool(row["dominated_by_static"]),
            -float(row["quality_retention_vs_best_static"]),
            -float(row["cost_ratio_vs_matched_static"]),
        )
    )
    _write_json(
        output / "deep-swe-evaluation.json",
        {
            "dataset": "DeepSWE v1.1 published execution-scored outcomes",
            "target_tasks": len(target.task_ids),
            "target_repositories": len(set(target.groups)),
            "static_arms": static,
            "results": rows,
            "target_labels_used_for_fit": False,
            "target_labels_used_for_thresholds": False,
            "research_adaptation_note": (
                "The external candidate grid and thresholds were frozen before this phase. "
                "The target ladders use previously known DeepSWE static aggregate results, so "
                "this is deployment calibration rather than untouched confirmation."
            ),
        },
    )
    logger.info(
        "DeepSWE evaluation complete: rows=%d leader=%s/%s reward=%.4f cost=%.2f",
        len(rows),
        rows[0]["candidate"],
        rows[0]["ladder"],
        _as_float(rows[0]["router_reward"]),
        _as_float(rows[0]["router_cost_usd"]),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser("fit")
    fit.add_argument("--nebius-tasks", type=Path, required=True)
    fit.add_argument("--r2e-loader", type=Path, required=True)
    fit.add_argument("--coderouter-root", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--deep-matrix", type=Path, required=True)
    evaluate.add_argument("--deep-tasks", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "fit":
        _fit(args)
    else:
        _evaluate(args)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    main()
