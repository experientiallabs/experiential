"""Converter tests: synthetic-fixture mechanics plus a data-gated check of the lab goldens."""

from __future__ import annotations

import json
import os
from pathlib import Path
from statistics import fmean

import numpy as np
import pytest

from wmo.optimize.deepswe import (
    EMBEDDINGS_FILENAME,
    GROUPS_FILENAME,
    PROMPT_BOILERPLATE,
    DeepsweConversion,
    convert_deepswe,
    price_table_span,
    top_arm,
)
from wmo.optimize.outcomes import OutcomeMatrix
from wmo.reproduce.embedding import CachedTaskEmbedder

# The machine-local published-artifact cache (trials/tasks/leaderboard + instruction texts) and
# the recorded Qwen3 embedding cache. Unset means the live reproduction tests skip; the numbers
# they pin are the port's hard gate wherever the data exists.
SOURCE_ENV = "WMO_DEEPSWE_SOURCE"
EMBED_CACHE_ENV = "WMO_DEEPSWE_EMBED_CACHE"


def _trial(
    config: str,
    task: str,
    trial: str,
    *,
    model: str,
    effort: str | None = "high",
    f2p: float = 1.0,
    passed: bool = True,
    cost: float | None = 0.5,
    scored: bool = True,
) -> dict[str, object]:
    return {
        "config": config,
        "task_name": task,
        "trial_name": trial,
        "included_in_score": scored,
        "model": model,
        "reasoning_effort": effort,
        "f2p": f2p,
        "passed": passed,
        "cost_usd": cost,
        "outcome": "pass" if passed else "fail",
        "n_agent_steps": 7,
        "n_input_tokens": 1000,
        "n_output_tokens": 100,
        "n_cache_tokens": 600,
    }


def _write_source(tmp_path: Path, trials: list[dict[str, object]]) -> Path:
    """A tiny publisher-shaped source directory: three artifacts plus instruction texts."""
    source = tmp_path / "source"
    tasks = sorted({str(trial["task_name"]) for trial in trials})
    (source / "deep-swe-main" / "tasks").mkdir(parents=True)
    for task in tasks:
        task_dir = source / "deep-swe-main" / "tasks" / task
        task_dir.mkdir()
        (task_dir / "instruction.md").write_text(
            f"Fix the {task} bug.{PROMPT_BOILERPLATE}", encoding="utf-8"
        )
    (source / "trials.json").write_text(
        json.dumps({"n_trials": len(trials), "rows": trials}), encoding="utf-8"
    )
    (source / "tasks.json").write_text(
        json.dumps(
            {
                "n_tasks": len(tasks),
                "rows": [{"id": task, "repository": f"org/{task[0]}"} for task in tasks],
            }
        ),
        encoding="utf-8",
    )
    by_config: dict[str, list[float]] = {}
    for trial in trials:
        if trial["included_in_score"]:
            by_config.setdefault(str(trial["config"]), []).append(float(bool(trial["passed"])))
    (source / "leaderboard-live.json").write_text(
        json.dumps({"rows": [{"config": c, "pass_at_1": fmean(v)} for c, v in by_config.items()]}),
        encoding="utf-8",
    )
    return source


def _write_cache(tmp_path: Path, tasks: list[str], *, dim: int = 8) -> Path:
    cache = tmp_path / "embeddings.json"
    rng = np.random.default_rng(7)
    cache.write_text(
        json.dumps({task: rng.normal(size=dim).tolist() for task in tasks}), encoding="utf-8"
    )
    return cache


def _convert(tmp_path: Path, trials: list[dict[str, object]]) -> DeepsweConversion:
    source = _write_source(tmp_path, trials)
    tasks = sorted({str(trial["task_name"]) for trial in trials})
    return convert_deepswe(
        source, embedding_cache=_write_cache(tmp_path, tasks), out=tmp_path / "bundle"
    )


def _tiny_trials() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task in ("alpha", "beta", "gamma"):
        for index in range(2):  # two episodes per cell
            rows.append(
                _trial(
                    "mini_swe_agent_claude_opus_5_high",
                    task,
                    f"opus-{task}-{index}",
                    model="claude-opus-5",
                    f2p=1.0,
                    passed=True,
                    cost=2.0,
                )
            )
            rows.append(
                _trial(
                    "mini_swe_agent_gpt_5_6_sol_medium",
                    task,
                    f"sol-{task}-{index}",
                    model="gpt-5-6-sol",
                    effort="medium",
                    f2p=0.5,
                    passed=False,
                    cost=1.0,
                )
            )
    return rows


def test_conversion_shapes_pool_outcomes_embeddings_and_groups(tmp_path: Path) -> None:
    result = _convert(tmp_path, _tiny_trials())
    matrix = OutcomeMatrix.load(result.matrix_path)
    assert sorted(matrix.model_names()) == ["claude-opus-5@high", "gpt-5.6-sol@medium"]
    assert matrix.scenario_ids() == ["alpha", "beta", "gamma"]
    assert result.scored_outcomes == 12 and result.unscored_outcomes == 0
    # Episodes within a cell are distinct, deterministic indices.
    episodes = {
        (o.scenario_id, o.model, o.episode) for o in matrix.outcomes if o.model.startswith("claude")
    }
    assert len(episodes) == 6
    # The .npy is row-aligned to first appearance: CachedTaskEmbedder must accept it verbatim.
    embedder = CachedTaskEmbedder(matrix, result.embeddings_path)
    assert embedder.dim == 8
    groups = json.loads(result.groups_path.read_text(encoding="utf-8"))
    assert groups == {"alpha": "org/a", "beta": "org/b", "gamma": "org/g"}
    assert (tmp_path / "bundle" / EMBEDDINGS_FILENAME).is_file()
    assert (tmp_path / "bundle" / GROUPS_FILENAME).is_file()


def test_pool_entries_carry_the_real_prices_and_measured_usage_lands_on_rows(
    tmp_path: Path,
) -> None:
    matrix = OutcomeMatrix.load(_convert(tmp_path, _tiny_trials()).matrix_path)
    opus = next(entry for entry in matrix.pool if entry.name == "claude-opus-5@high")
    assert (opus.input_per_mtok, opus.output_per_mtok) == (5.00, 25.00)
    assert opus.cached_input_per_mtok == 0.50
    row = matrix.outcomes[0]
    assert row.usage.input_tokens == 1000 and row.usage.cached_input_tokens == 600
    assert row.cost_usd > 0.0


def test_a_model_the_price_table_does_not_cover_is_dropped_and_named(tmp_path: Path) -> None:
    trials = _tiny_trials() + [
        _trial(
            "mini_swe_agent_muse_spark_1_1_xhigh",
            "alpha",
            "muse-alpha-0",
            model="muse-spark-1-1",
            effort="xhigh",
        )
    ]
    result = _convert(tmp_path, trials)
    assert result.dropped_configs == ["mini_swe_agent_muse_spark_1_1_xhigh"]
    assert result.models == 2  # nothing unpriced entered the pool


def test_an_unpriced_trial_becomes_unscored_evidence_not_a_free_run(tmp_path: Path) -> None:
    trials = _tiny_trials()
    trials.append(
        _trial(
            "mini_swe_agent_claude_opus_5_high",
            "alpha",
            "opus-alpha-nocost",
            model="claude-opus-5",
            cost=None,
        )
    )
    result = _convert(tmp_path, trials)
    assert result.unscored_outcomes == 1
    matrix = OutcomeMatrix.load(result.matrix_path)
    unpriced = [o for o in matrix.outcomes if o.error]
    assert len(unpriced) == 1
    assert unpriced[0].reward is None and unpriced[0].cost_usd == 0.0
    assert "cost_usd" in (unpriced[0].error or "")


def test_the_integrity_gate_refuses_trials_that_contradict_the_leaderboard(
    tmp_path: Path,
) -> None:
    trials = _tiny_trials()
    source = _write_source(tmp_path, trials)
    leaderboard = json.loads((source / "leaderboard-live.json").read_text(encoding="utf-8"))
    leaderboard["rows"][0]["pass_at_1"] = 0.123  # not what the trials say
    (source / "leaderboard-live.json").write_text(json.dumps(leaderboard), encoding="utf-8")
    with pytest.raises(ValueError, match="published pass@1"):
        convert_deepswe(
            source,
            embedding_cache=_write_cache(tmp_path, ["alpha", "beta", "gamma"]),
            out=tmp_path / "bundle",
        )


def test_a_gap_in_the_embedding_cache_names_the_missing_tasks(tmp_path: Path) -> None:
    trials = _tiny_trials()
    source = _write_source(tmp_path, trials)
    with pytest.raises(ValueError, match="misses"):
        convert_deepswe(
            source,
            embedding_cache=_write_cache(tmp_path, ["alpha", "beta"]),  # gamma missing
            out=tmp_path / "bundle",
        )


def test_the_price_table_ported_intact() -> None:
    # The pre-split lab's `arms` golden: blended $/1M span 0.72 -> 30.00, a 41x spread.
    lo, hi = price_table_span()
    assert (round(lo, 3), round(hi, 2)) == (0.725, 30.00)
    assert round(hi / lo) == 41


_HAVE_DATA = bool(os.environ.get(SOURCE_ENV)) and bool(os.environ.get(EMBED_CACHE_ENV))


@pytest.mark.skipif(
    not _HAVE_DATA, reason=f"needs {SOURCE_ENV} and {EMBED_CACHE_ENV} pointing at local data"
)
def test_live_conversion_reproduces_the_lab_goldens(tmp_path: Path) -> None:
    """The port's hard gate: the converted matrix must reproduce the pre-split lab's numbers."""
    result = convert_deepswe(
        Path(os.environ[SOURCE_ENV]),
        embedding_cache=Path(os.environ[EMBED_CACHE_ENV]),
        out=tmp_path / "bundle",
    )
    assert result.crosscheck == "50/50 configs reproduce published pass@1, 0 off"
    assert (result.models, result.scenarios) == (41, 113)
    assert (result.scored_outcomes, result.unscored_outcomes) == (18354, 21)
    matrix = OutcomeMatrix.load(result.matrix_path)
    top = top_arm(matrix)
    assert top.name == "claude-opus-5@high"
    assert (round(top.graded, 3), round(top.pass_at_1, 3)) == (0.955, 0.729)
    assert round(top.cost_per_task, 2) == 6.09
    assert top.tasks == 113
    # And the recorded vectors serve verbatim (row-aligned, one width, no duplicate texts).
    assert CachedTaskEmbedder(matrix, result.embeddings_path).dim == 1024
