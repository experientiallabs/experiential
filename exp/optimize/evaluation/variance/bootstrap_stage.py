"""The pipeline's rules stage for a bootstrapped mining job: reuse the family's frozen rules.

When ``mining.bootstrap`` names a task's registered start, that start already binds its family's
frozen rule book and rule memory. The rules stage then learns nothing: it runs no global
extraction over the build's world-learning traces. It verifies the registration's family rules
and hands them, with the registered job and the general rubric authored over them, to the later
stages, so every stage reads the same family-scoped rules the start was captured with.
"""

from __future__ import annotations

from exp.common.core.artifacts import JsonObject
from exp.common.project import ProjectStore
from exp.optimize.evaluation.variance.bootstrap import (
    frozen_report,
    load_outcome,
    registered_job,
    registration_entry,
)
from exp.optimize.evaluation.variance.bootstrap_contracts import (
    BootstrapBinding,
    BootstrapRegistration,
)
from exp.optimize.evaluation.variance.pipeline_inputs import general_rubric
from exp.optimize.evaluation.variance.pipeline_spec import RulesOutputs, StageContext, StageRun

PLAN_LINE = (
    "Rules stage: reuses the bootstrapped family's frozen rule book and rule memory; no "
    "extraction and no challenges."
)
"""The rules stage's plan line for a bootstrapped mining job."""


def bootstrapped_registration(
    project: ProjectStore, binding: BootstrapBinding
) -> BootstrapRegistration:
    """Return the registration a binding names, verified against its frozen batch.

    Raises:
        ValueError: The binding does not resolve to a saved registration.
    """
    registered_job(project, binding)
    entry = registration_entry(project, binding)
    registration = load_outcome(project, entry.identity)
    assert isinstance(registration, BootstrapRegistration)
    return registration


def run_bootstrapped_rules(context: StageContext) -> StageRun:
    """Hand the registered job and its family's frozen rules to the later stages.

    Args:
        context: The running rules stage of a bootstrapped registration.

    Returns:
        The stage's outputs.

    Raises:
        ValueError: The registration names no rules-stage report (the families stage reads
            one), or its report names another rule book.
    """
    project, spec = context.project, context.spec
    binding = spec.mining.bootstrap
    assert binding is not None
    registration = bootstrapped_registration(project, binding)
    report = registration.rules_report
    if report is None:
        raise ValueError(
            f"the bootstrapped family {registration.family_id} carries no rules-stage report; "
            "the family simulator's rules_report must name the report its rule book came from"
        )
    if frozen_report(project, report).rulebook != registration.rulebook:
        raise ValueError("the bootstrapped family's rules report names another rule book")
    rubric = general_rubric(context, spec.mining.registered, registration.rulebook)
    outputs = RulesOutputs(
        report=report,
        rulebook=registration.rulebook,
        challenges=0,
        job=spec.mining.registered,
        rule_memory=registration.rule_memory,
        rubric=rubric.rubric,
        rubric_packet=rubric.packet,
        rubric_library=rubric.library,
    )
    summary: JsonObject = {
        "bootstrapped_family": registration.family_id,
        "rules_learned": False,
        "challenges_executed": 0,
        **rubric.summary(),
    }
    references = (report, registration.rulebook, registration.checkpoint, *rubric.references())
    return StageRun(outputs, references, summary)
