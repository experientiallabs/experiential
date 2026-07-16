"""Named eval suites for repeatable reconstruction-fidelity runs.

Suites live next to examples (`examples/<task>/evals/*.toml`) and point at trace files relative to
the suite file. Generated run results are local artifacts, normally written under `.wmh/evals/`.
"""

from __future__ import annotations

import json
import logging
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

logger = logging.getLogger(__name__)

SampleTurns = Literal["all", "sampled"]


class EvalSuiteConfig(BaseModel):
    """A TOML-backed eval suite definition."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    description: str | None = None
    files: list[str] = Field(default_factory=lambda: ["../traces.otel.jsonl"])
    prompt: str | None = None
    train_split: float = Field(default=0.7, gt=0.0, lt=1.0)
    top_k: int = Field(default=5, ge=0)
    sample_turns: SampleTurns = "all"
    seed: int = 0
    no_rag: bool = False
    embed_dim: int = Field(default=512, gt=0)
    # Agentic mode (default off): seed a train-split knowledge base into every prediction, and/or
    # request the deliberate-then-answer output contract. See `wmh.engine.knowledge`.
    knowledge: bool = False
    reasoning: bool = False


@dataclass(frozen=True)
class EvalSuite:
    """A discovered suite plus its parsed config."""

    id: str
    example: str
    name: str
    path: Path
    config: EvalSuiteConfig

    @property
    def aliases(self) -> tuple[str, ...]:
        if self.name == "default":
            return (self.id, self.example)
        return (self.id,)

    def resolve_files(self) -> list[Path]:
        return [_resolve_relative(self.path.parent, value) for value in self.config.files]

    def resolve_prompt(self) -> Path | None:
        if self.config.prompt is None:
            return None
        return _resolve_relative(self.path.parent, self.config.prompt)


@dataclass(frozen=True)
class EvalResultSummary:
    """One persisted eval result, for `wmh eval results`."""

    path: Path
    suite: str
    run_id: str
    started_at: str
    provider: str
    model: str
    overall_fidelity: float
    overall_std: float
    total_steps: int  # all steps attempted, including judge-invalid ones
    total_invalid: int  # judge failures, excluded from overall_fidelity (0 for old results)


def discover_eval_suites(examples_root: str | Path | Iterable[str | Path]) -> list[EvalSuite]:
    """Find every example-local suite under `<root>/*/evals/*.toml` across one or more roots.

    A malformed suite file is skipped with a warning naming the file and the parse error —
    one broken benchmark dir must not take down listing/resolution for every other suite.
    Loading the broken suite directly (by path) still raises the full error.
    """
    suites: list[EvalSuite] = []
    for root in _as_roots(examples_root):
        if not root.exists():
            continue
        for path in sorted(root.glob("*/evals/*.toml")):
            try:
                suites.append(load_eval_suite(path))
            except ValueError as exc:
                logger.warning("skipping eval suite %s: %s", path, exc)
    return suites


def _as_roots(examples_root: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(examples_root, (str, Path)):
        return [Path(examples_root)]
    return [Path(root) for root in examples_root]


def load_eval_suite(path: str | Path) -> EvalSuite:
    """Read and validate one eval suite TOML file."""
    suite_path = Path(path)
    try:
        with suite_path.open("rb") as fh:
            raw = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"{suite_path} is not valid TOML ({exc})") from exc
    try:
        config = EvalSuiteConfig.model_validate(raw)
    except ValidationError as exc:
        has_judge = isinstance(raw, dict) and "judge" in raw
        if has_judge and exc.error_count() == 1:
            # Pre-overhaul suites (and the old shipped defaults) carried this knob; the generic
            # schema error would never say it was removed or what to do. Only replace the error
            # when `judge` is the sole problem — otherwise the full listing must surface so the
            # user fixes everything in one pass.
            raise ValueError(
                f"{suite_path} sets `judge`, an option that no longer exists — the harness has a "
                "single judge (the 5-dimension rubric); delete the `judge` line from the suite file"
            ) from exc
        hint = (
            " (note: the `judge` option no longer exists; delete that line too)"
            if has_judge
            else ""
        )
        raise ValueError(
            f"{suite_path} does not match the eval suite schema ({exc}){hint}"
        ) from exc
    example = (
        suite_path.parent.parent.name
        if suite_path.parent.name == "evals"
        else suite_path.parent.name
    )
    name = suite_path.stem
    return EvalSuite(
        id=f"{example}/{name}",
        example=example,
        name=name,
        path=suite_path,
        config=config,
    )


def resolve_eval_suite(
    selector: str, examples_root: str | Path | Iterable[str | Path]
) -> EvalSuite:
    """Resolve `selector` as `example/suite`, `example` for default, or a direct TOML path."""
    direct = Path(selector)
    if direct.suffix == ".toml" and direct.exists():
        return load_eval_suite(direct)

    suites = discover_eval_suites(examples_root)
    exact = [suite for suite in suites if suite.id == selector]
    if len(exact) > 1:
        paths = ", ".join(str(suite.path) for suite in exact)
        raise ValueError(f"eval suite {selector!r} exists in multiple roots: {paths}")
    if exact:
        return exact[0]
    aliased = [suite for suite in suites if selector in suite.aliases]
    if len(aliased) == 1:
        return aliased[0]
    if len(aliased) > 1:
        choices = ", ".join(suite.id for suite in aliased)
        raise ValueError(f"ambiguous eval suite {selector!r}; choose one of: {choices}")
    available = ", ".join(suite.id for suite in suites)
    hint = f" (available: {available})" if available else ""
    raise ValueError(f"unknown eval suite {selector!r}{hint}")


def result_path(results_root: str | Path, suite: EvalSuite, run_id: str) -> Path:
    """Default JSON output path for one suite run."""
    return Path(results_root) / suite.example / suite.name / f"{run_id}.json"


def list_eval_results(
    results_root: str | Path, suite: str | None = None, *, limit: int = 20
) -> list[EvalResultSummary]:
    """Read persisted eval result summaries, newest first."""
    root = Path(results_root)
    if not root.exists():
        return []
    summaries: list[EvalResultSummary] = []
    for path in sorted(root.glob("**/*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        summary = _read_result_summary(path)
        if summary is None:
            continue
        if suite is not None and summary.suite != suite:
            continue
        summaries.append(summary)
        if len(summaries) >= limit:
            break
    return summaries


def _resolve_relative(base: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve(strict=False) if path.is_absolute() else (base / path).resolve(strict=False)


def _read_result_summary(path: Path) -> EvalResultSummary | None:
    try:
        raw: JsonValue = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    report = raw.get("report")
    config = raw.get("config")
    if not isinstance(report, dict) or not isinstance(config, dict):
        return None
    return EvalResultSummary(
        path=path,
        suite=_as_str(raw.get("suite"), default="unknown"),
        run_id=_as_str(raw.get("run_id"), default=path.stem),
        started_at=_as_str(raw.get("started_at"), default="unknown"),
        provider=_as_str(config.get("provider"), default="unknown"),
        model=_as_str(config.get("model"), default="unknown"),
        overall_fidelity=_as_float(report.get("overall_fidelity")),
        overall_std=_as_float(report.get("overall_std")),
        total_steps=_as_int(report.get("total_steps")),
        total_invalid=_as_int(report.get("total_invalid")),
    )


def _as_str(value: JsonValue | None, *, default: str) -> str:
    return value if isinstance(value, str) else default


def _as_float(value: JsonValue | None) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def _as_int(value: JsonValue | None) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0
