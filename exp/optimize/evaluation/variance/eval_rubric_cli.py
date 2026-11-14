"""Run the held-out eval rubric on finished continuation groups of one task (CLI).

Reads the project store read-only, authors and freezes the eval rubric once (reused on reruns),
judges every attempt of every finished group, and writes a JSON and Markdown report under --out.
Groups whose study has no "-result" yet are skipped and listed.

Reading the studies needs the same EXP_REDRAW_STUDIES as the run that drew them.

Example:
    python -m exp.optimize.evaluation.variance.eval_rubric_cli --task task-a5728ea9a683a6e1aab2 \
        --group glm-5.3=continuation-study-39e4497037f923e8589a \
        --group kimi-k3=continuation-study-323ccbdbd20557d7b4e6
"""

from __future__ import annotations

import argparse
import json
import secrets
import subprocess
import sys
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

from exp.common.core.artifacts import sha256_json
from exp.common.core.locks import CoordinationLock
from exp.common.models import ModelRequest, ModelResponse, load_model_catalog
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.records import load_study
from exp.optimize.evaluation.factory.authoring import AuthorSettings, author_response
from exp.optimize.evaluation.spending import operation_spend_ceiling
from exp.optimize.evaluation.variance import eval_rubric as ev
from exp.optimize.evaluation.variance.actor_critic import admitted_concurrency
from exp.optimize.evaluation.variance.author_harness import scratch_project
from exp.optimize.evaluation.variance.bounded import run_at_once
from exp.optimize.evaluation.variance.episode_evidence import evidence_from_branch
from exp.optimize.evaluation.variance.judge import load_assessment
from exp.optimize.evaluation.variance.judged import JUDGED_STUDY_TYPE, JudgedStudy
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.runtime.models.spend_ceiling import (
    SpendCeilingReached,
    install_spend_ceiling,
    operation_spend_scope,
)
from exp.simulation.continuations import load_continuation_checkpoint

ALLOWED_ALIASES = frozenset({"glm", "deepseek-judge"})
"""Open-weight aliases this eval may call."""


def say(text: str) -> None:
    """Print one progress line."""
    print(text, flush=True)  # noqa: T201 - CLI progress


class LiveCaller:
    """Sends draws through the repo's author-response owner on a private scratch project."""

    def __init__(
        self, scratch: ProjectStore, settings: AuthorSettings, run: str, revision: str
    ) -> None:
        """Bind the scratch project, frozen settings and this invocation's run token."""
        self.scratch, self.settings, self.run, self.revision = scratch, settings, run, revision
        self.calls = 0
        self._lock = threading.Lock()

    def __call__(self, request: ModelRequest, identity: str) -> ModelResponse:
        """Buy one draw."""
        draw = f"{self.run}-{identity}"
        with self._lock:
            self.calls += 1
        lock = self.scratch.artifacts.project_directory / (draw + ".lock")
        with CoordinationLock(lock, timeout=0):
            _, response = author_response(
                self.scratch,
                self.settings,
                request,
                identity=draw,
                sources=(),
                code_revision=self.revision,
            )
        return response


def settings_for(scratch: ProjectStore, alias: str, output_tokens: int) -> AuthorSettings:
    """Frozen request settings of one allowed alias."""
    if alias not in ALLOWED_ALIASES:
        raise SystemExit(f"alias {alias} is not an allowed open-weight eval alias")
    catalog = load_model_catalog(scratch.model_catalog_path)
    return AuthorSettings(
        model_alias=alias,
        model=RuntimeModelCatalog(catalog, environment={}).snapshot(alias)[0],
        catalog_sha256=sha256_json(catalog.model_dump(mode="json")),
        output_tokens=output_tokens,
        maximum_input_bytes=512_000,
        maximum_physical_attempts=3,
        deadline=datetime.now(UTC) + timedelta(hours=6),
    )


def judged_studies(project: ProjectStore) -> dict[str, list[str]]:
    """Saved judged-continuation-study ids by the study result they judge."""
    store = project.artifacts
    found: dict[str, list[str]] = {}
    for identity in store.list_ids(artifact_type=JUDGED_STUDY_TYPE):
        for source in store.read(identity).manifest.inputs:
            found.setdefault(source.artifact_id, []).append(identity)
    return found


def main(argv: list[str] | None = None) -> int:
    """Run the eval; see the module docstring."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--root", default="/workspace/hf-traces/build-root")
    parser.add_argument("--project", default="hf-agents")
    parser.add_argument("--task", required=True)
    parser.add_argument("--group", action="append", required=True, help="tier=study-id")
    parser.add_argument("--training-rubric", default="episode-rubric-94a857462834411b9243")
    parser.add_argument("--out", type=Path, default=Path("/workspace/t/dv-eval/out"))
    parser.add_argument("--author", default="glm")
    parser.add_argument("--judge", default="deepseek-judge")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--max-usd", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true", help="build inputs, call no model")
    parser.add_argument("--protocol", choices=("v1", "v2"), default="v1")
    parser.add_argument("--rules-document", type=Path, help="v2: supporting specifications (MD)")
    parser.add_argument("--coverage", type=Path, help="v2: coverage dimensions (JSON)")
    args = parser.parse_args(argv)

    project = ProjectStore(args.root, args.project)
    store = project.artifacts
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    revision = subprocess.run(
        ["git", "-C", str(Path(ev.__file__).resolve().parent), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    groups: list[tuple[str, str]] = []
    for spec in args.group:
        tier, _, study = spec.partition("=")
        study = study.removesuffix("-result")
        if not store.exists(study + "-result"):
            say(f"skip {tier}: {study} has no -result yet")
            continue
        groups.append((tier, study))
    if not groups:
        say("no finished group")
        return 1

    # Load every finished group and its evidence before any model call.
    judged_index = judged_studies(project)
    checkpoint = None
    checkpoint_ref = None
    attempts: list[tuple[str, str, int, object]] = []
    training: dict[tuple[str, int], ev.TrainingVerdict | None] = {}
    findings: list[dict[str, object]] = []
    sources: list[ev.ContextSource] = []
    for tier, study_id in groups:
        reference = artifact_input(store.read(study_id + "-result").manifest)
        study = load_study(store, reference)
        group_checkpoint = load_continuation_checkpoint(store, study.spec.checkpoint)
        if group_checkpoint.task.task_id != args.task:
            raise SystemExit(
                f"{study_id} attempts {group_checkpoint.task.task_id}, not {args.task}"
            )
        if checkpoint is None:
            checkpoint, checkpoint_ref = group_checkpoint, study.spec.checkpoint
        for branch in study.branches:
            source = f"{study_id}-branch-{branch.index}"
            try:
                evidence = evidence_from_branch(
                    branch,
                    group_checkpoint,
                    episode_id=source,
                    # The eval reads no world digest; the configuration only feeds that digest.
                    configuration={},
                    checkpoint_source=study.spec.checkpoint.artifact_id,
                    branch_source=source,
                )
            except ValueError as error:
                evidence = str(error)
            attempts.append((tier, study_id, branch.index, evidence))
        verdicts: dict[int, ev.TrainingVerdict] = {}
        judged_ids = sorted(
            judged_index.get(reference.artifact_id, []),
            key=lambda identity: store.read(identity).manifest.created_at,
        )
        for judged_id in judged_ids:  # the latest judging wins the side-by-side
            sources.append(
                ev.ContextSource(
                    artifact_id=judged_id,
                    sha256=artifact_input(store.read(judged_id).manifest).sha256,
                    role="training judge findings (not-met / insufficient) on " + study_id,
                )
            )
            judged = JudgedStudy.model_validate_json(store.read_bytes(judged_id, "judged.json"))
            for slot in judged.branches:
                if slot.episode is None:
                    verdicts[slot.position] = ev.TrainingVerdict(
                        verdict="no evidence", judged_study=judged_id, judge=judged.judge
                    )
                    continue
                assessment = load_assessment(project, slot.episode.assessment)
                verdicts[slot.position] = ev.TrainingVerdict(
                    verdict=assessment.verdict, judged_study=judged_id, judge=judged.judge
                )
                for finding in assessment.findings:
                    if finding.status in ("not_met", "insufficient_evidence"):
                        findings.append(finding.model_dump(mode="json"))
        for branch in study.branches:
            training[(study_id, branch.index)] = verdicts.get(branch.index)
        say(f"{tier}: {study_id} {len(study.branches)} attempts, judged by {judged_ids or 'none'}")

    # Error context also comes from training judgings of other groups of the same task.
    used = {source.artifact_id for source in sources}
    task_of: dict[str, str | None] = {}
    for judged_id in sorted({i for ids in judged_index.values() for i in ids} - used):
        judged = JudgedStudy.model_validate_json(store.read_bytes(judged_id, "judged.json"))
        result = judged.study.artifact_id
        try:
            spec = json.loads(store.read_bytes(result, "record.json"))["spec"]
            point = spec["checkpoint"]["artifact_id"]
            if point not in task_of:
                task_of[point] = json.loads(store.read_bytes(point, "checkpoint.json"))["task"][
                    "task_id"
                ]
        except (KeyError, ValueError, OSError) as error:
            say(f"  cannot read the task of {judged_id}: {type(error).__name__}")
            continue
        if task_of[point] != args.task:
            continue
        before = len(findings)
        for slot in judged.branches:
            if slot.episode is not None:
                for finding in load_assessment(project, slot.episode.assessment).findings:
                    if finding.status in ("not_met", "insufficient_evidence"):
                        findings.append(finding.model_dump(mode="json"))
        if len(findings) == before:
            continue
        sources.append(
            ev.ContextSource(
                artifact_id=judged_id,
                sha256=artifact_input(store.read(judged_id).manifest).sha256,
                role=f"training judge findings (not-met / insufficient) on other group {result}",
            )
        )
        say(f"  error context from {judged_id} ({result}): {len(findings) - before} findings")

    assert checkpoint is not None and checkpoint_ref is not None
    initial_state = None
    predictor = checkpoint.world.image.parts.get("predictor.json")
    if predictor is not None:
        initial_state = json.loads(predictor).get("state")
    task = ev.task_context(checkpoint.task.model_dump(mode="json"), initial_state)
    rubric_manifest = store.read(args.training_rubric).manifest
    training_rubric = json.loads(store.read_bytes(args.training_rubric, "rubric.json"))
    error_context = ev.training_error_context(training_rubric, findings)
    sources = [
        ev.ContextSource(
            artifact_id=checkpoint_ref.artifact_id,
            sha256=checkpoint_ref.sha256,
            role="task instruction and starting environment",
        ),
        ev.ContextSource(
            artifact_id=args.training_rubric,
            sha256=artifact_input(rubric_manifest).sha256,
            role="training rubric failure patterns and not-met conditions",
        ),
        *sources,
    ]
    say(
        f"error context: {len(error_context['training_rubric_failure_patterns'])} training "
        f"criteria, {len(error_context['observed_training_findings'])} observed findings"
    )

    scratch = scratch_project(out / "scratch-project", project)
    run = "e" + secrets.token_hex(3)
    judge_settings = settings_for(scratch, args.judge, 16384)
    author_settings = settings_for(scratch, args.author, 16384)
    concurrency = max(1, min(args.concurrency, 3, admitted_concurrency(scratch, args.judge)))
    if args.dry_run:
        for tier, study_id, index, evidence in attempts[:3]:
            if isinstance(evidence, str):
                continue
            size = len(json.dumps(ev.render_trajectory(evidence)))
            say(f"dry run {tier} {study_id} b{index}: trajectory {size} bytes")
        say(f"dry run: {len(attempts)} attempts, concurrency {concurrency}, no model called")
        return 0

    author = LiveCaller(scratch, author_settings, run, revision)
    judge = LiveCaller(scratch, judge_settings, run, revision)
    if args.protocol == "v2":
        return _run_v2(args, out, task, error_context, sources, attempts, scratch, author, judge,
                       concurrency, groups)
    with operation_spend_scope():
        ceiling = operation_spend_ceiling(scratch, f"dv-eval-{run}")
        ceiling.authorize(
            "dv-eval", ceiling_usd=args.max_usd, unpriced_attempts=12 * (len(attempts) + 1)
        )
        install_spend_ceiling(ceiling)
        rubric, rubric_file, authored = ev.freeze_rubric(
            out,
            task=task,
            error_context=error_context,
            context_sources=sources,
            caller=author,
            author_alias=args.author,
        )
        say(f"eval rubric {'authored' if authored else 'reused'}: {rubric_file}")
        folder = out / "judgments" / rubric.sha256[:16]

        def one(item: tuple[str, str, int, object]) -> ev.EvalJudgment:
            tier, study_id, index, evidence = item
            if isinstance(evidence, str):
                return ev.EvalJudgment(
                    rubric_sha256=rubric.sha256,
                    tier=tier,
                    study=study_id,
                    branch=index,
                    status="no_evidence",
                    failure=evidence[:2000],
                )
            path = folder / tier / f"{study_id}-b{index}-{evidence.identity[:16]}.json"
            if path.exists():
                return ev.EvalJudgment.model_validate_json(path.read_bytes())
            try:
                judgment = ev.judge_attempt(
                    rubric, task, evidence, tier=tier, study=study_id, branch=index, caller=judge
                )
            except SpendCeilingReached:
                raise
            except Exception as error:  # noqa: BLE001 - a provider failure is an unscored row
                judgment = ev.EvalJudgment(
                    rubric_sha256=rubric.sha256,
                    tier=tier,
                    study=study_id,
                    branch=index,
                    evidence_sha256=evidence.identity,
                    status="unavailable",
                    failure=f"{type(error).__name__}: {error}"[:2000],
                )
                say(f"  {tier} b{index}: unavailable ({judgment.failure[:200]})")
                return judgment  # not saved, so a rerun retries it
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(judgment.model_dump_json(indent=2), encoding="utf-8")
            say(f"  {tier} b{index}: {judgment.status} {judgment.overall_score}")
            return judgment

        judgments = run_at_once([lambda item=item: one(item) for item in attempts], concurrency)
        spend = ceiling.summary()
    report = ev.aggregate(rubric, judgments, training)
    report["rubric_file"] = str(rubric_file)
    report["rubric_authored_this_run"] = authored
    report["calls_this_run"] = {"author": author.calls, "judge": judge.calls}
    report["spend_this_run"] = spend.model_dump(mode="json")
    report["groups"] = dict((tier, study) for tier, study in groups)
    report["skipped_groups"] = [
        spec for spec in args.group if spec.partition("=")[2] not in {s for _, s in groups}
    ]
    report["judgments"] = [row.model_dump(mode="json") for row in judgments]
    stem = f"report-{args.task}-{rubric.sha256[:16]}"
    (out / f"{stem}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / f"{stem}.md").write_text(ev.markdown_report(report, rubric), encoding="utf-8")
    say(f"report: {out / stem}.json and .md; calls {report['calls_this_run']}")
    return 0


def _run_v2(args, out, task, error_context, sources, attempts, scratch, author, judge,  # noqa: ANN001, PLR0913
            concurrency, groups) -> int:
    """Author, review and freeze the v2 rubric, judge every attempt, write the report."""
    from exp.optimize.evaluation.variance import eval_rubric_v2 as v2
    from exp.simulation.mining.decisions import DecisionsTraceTagger

    model = RuntimeModelCatalog(load_model_catalog(scratch.model_catalog_path)).resolve("clef")
    decide = DecisionsTraceTagger.for_alias(model, maximum_concurrency=32)._decide
    specifications = args.rules_document.read_text() if args.rules_document else None
    coverage = json.loads(args.coverage.read_text()) if args.coverage else None
    author.settings = author.settings.model_copy(update={"output_tokens": 32768, "maximum_input_bytes": 4_000_000})
    judge.settings = judge.settings.model_copy(update={"maximum_input_bytes": 4_000_000, "output_tokens": 32768})
    rubric, rubric_file, authored = v2.freeze_rubric_v2(
        out, task=task, trace_analysis=error_context, specifications=specifications,
        coverage=coverage, context_sources=sources, caller=author, decide=decide,
        author_alias=args.author,
    )  # fmt: skip
    say(f"v2 rubric {'authored' if authored else 'reused'}: {rubric_file} "
        f"({len(rubric.body.criteria)} criteria, review flags on "
        f"{sum(bool(r.flags) for r in rubric.review)})")  # fmt: skip
    folder = out / "judgments-v2" / rubric.sha256[:16]

    def one(item):  # noqa: ANN001, ANN202
        tier, study_id, index, evidence = item
        if isinstance(evidence, str):
            return v2.AttemptVerdict(rubric_sha256=rubric.sha256, tier=tier, study=study_id,
                                     branch=index, status="no_evidence", failure=evidence[:2000])  # fmt: skip
        path = folder / tier / f"{study_id}-b{index}-{evidence.identity[:16]}.json"
        if path.exists():
            return v2.AttemptVerdict.model_validate_json(path.read_bytes())
        try:
            row = v2.judge_attempt_v2(rubric, task, evidence, tier=tier, study=study_id,
                                      branch=index, decide=decide, caller=judge)  # fmt: skip
        except Exception as error:  # noqa: BLE001 - an unscored row, retried on rerun
            say(f"  {tier} b{index}: unavailable ({type(error).__name__}: {str(error)[:160]})")
            return v2.AttemptVerdict(rubric_sha256=rubric.sha256, tier=tier, study=study_id,
                                     branch=index, evidence_sha256=evidence.identity,
                                     status="unavailable", failure=str(error)[:2000])  # fmt: skip
        if row.status == "judged":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(row.model_dump_json(indent=2), encoding="utf-8")
        say(f"  {tier} b{index}: {row.verdict or row.status}")
        return row

    rows = run_at_once([lambda item=item: one(item) for item in attempts], max(concurrency, 8))
    report = v2.aggregate_v2(rubric, rows)
    report["rubric_file"] = str(rubric_file)
    report["groups"] = dict(groups)
    report["attempts"] = [row.model_dump(mode="json") for row in rows]
    stem = f"report-v2-{args.task}-{rubric.sha256[:16]}"
    (out / f"{stem}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    say(f"report: {out / stem}.json")
    say(json.dumps({k: report[k] for k in ("tiers", "pass_rate", "clef_verdicts",
                                            "escalated_to_llm_judge", "review_flags",
                                            "uncovered_requirements")}, indent=1))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
