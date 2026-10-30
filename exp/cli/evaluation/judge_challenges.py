"""The ``challenge-judges`` continuation action: a fixed-output judge challenge study.

The registered study, its arm models and every judgment belong to
:mod:`exp.optimize.evaluation.tasks.challenge_runs`. The handler checks the study and models
without credentials, prints the plan, stops there on ``--dry-run``, asks for spend consent for
the judge calls, and prints the recorded report. A recorded study replays without calls.
"""

import json

from exp.cli.evaluation.continuation_context import ContinuationContext
from exp.common.core.artifacts import JsonObject, JsonValue
from exp.optimize.evaluation.tasks.challenge_runs import (
    ChallengeRunSpec,
    preflight_challenge_run,
    run_challenge,
    saved_challenge_run,
)


def handle_challenge_judges(context: ContinuationContext, spec: ChallengeRunSpec) -> None:
    """Validate, authorize and run one judge challenge study, then print its report.

    Args:
        context: Validated ``--continuations`` invocation.
        spec: Registered study, one pinned model per arm, allowances and deadline.

    Returns:
        None. The plan, any consent question, a per-arm summary and the report are printed.
    """
    store, console = context.store, context.console
    saved = saved_challenge_run(store, spec)
    if saved is None:
        preflight_challenge_run(store, spec)
    console.print(
        f"Judge challenge {spec.study.identity}: {len(spec.study.judges)} arms judge the same "
        f"saved outputs; {sum(spec.planned(row.arm_id) for row in spec.models)} judgments, "
        f"physical attempt ceiling {spec.maximum_physical_attempts}."
    )
    if context.dry_run:
        console.print("Registered study validated; no calls or publication.")
        return
    if saved is None and not context.consent(spec):
        return
    reference = saved or run_challenge(store, spec, code_revision=context.revision)
    report = json.loads(store.artifacts.read_bytes(reference.artifact_id, "report.json"))
    arms = report["arms"]
    for arm_id, row in arms.items():
        console.print(f"  {arm_id}: {_summary(row)}", markup=False)
    console.print(f"Saved: {reference.artifact_id}. Labels are never rewritten.")
    console.print_json(data={"reference": reference.model_dump(mode="json"), "report": report})


def _summary(row: JsonObject) -> str:
    """One arm's base agreement and its judge sensitivity per declared defect kind."""
    agreement = row["base_label_agreement"]
    parts = [f"base agreement {_rate(agreement)}"]
    sensitivity = row["sensitivity"]
    if isinstance(sensitivity, dict):
        for kind, measured in sensitivity.items():
            if isinstance(measured, dict):
                parts.append(f"{kind} detected {_rate(measured['judge_sensitivity'])}")
    return ", ".join(parts)


def _rate(summary: JsonValue) -> str:
    """``successes/trials`` of a proportion summary, or ``none`` when nothing was decided."""
    if isinstance(summary, dict) and summary.get("trials"):
        return f"{summary.get('successes')}/{summary.get('trials')}"
    return "none"
