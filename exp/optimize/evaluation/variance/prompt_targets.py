"""Every open-weight, non-actor role prompt and prompt variant the Astra refinement may rewrite.

The registry is declared, never inferred from names: each :class:`TargetSource` names one role,
the module and attribute holding its prompt text, and the shape of that attribute. A ``text``
attribute is one target; a ``dict`` of texts, a tuple of ``variants`` (objects with
``variant_id`` and ``stance``) or a dict of ``world_sets`` (objects with ``approach``) expands to
one target per entry (a ``pair`` is one named stance), so every prompt variant of a multi-step
role is its own target.
:func:`prompt_targets` reads the live module text each time, so the registry cannot drift from
the code the roles run.

Not targeted, by design:

- the five actor tiers (the policies being measured): their system prompt is the task itself and
  the harness's, and nudges and briefings reach them as authored environment text, not prompts;
- the rubric-evolution council audit (:mod:`.rubric_audit`): its draws are made by the five
  council actor aliases, so one open-weight runtime prompt cannot stand for them.

``runner_role`` names the label a role passes to the shared role runner
(:func:`~.family_stages.run_role`). Those targets take an approved frozen prompt at runtime
through :func:`~.refined_prompts.refined_role_prompt`; the others are refinable offline but read
their code default at runtime until their call site is wired (listed by :func:`unwired_targets`).
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from exp.common.core.artifacts import ContractModel, Sha256, sha256_json

Shape = Literal["text", "pair", "dict", "variants", "world_sets"]
"""How a source attribute holds its prompt text."""

Part = Literal["instruction", "stance", "phase", "system", "guidance"]
"""What part of a role's prompt a target is."""

V = "exp.optimize.evaluation.variance"
R = "exp.optimize.evaluation.rules"
S = "exp.simulation"
W = f"{S}.world_model"
P = f"{S}.engines.text.prompt"


@dataclass(frozen=True)
class TargetSource:
    """One declared prompt attribute of one role.

    Attributes:
        role: The pipeline role.
        module: Module holding the attribute.
        attribute: Attribute name.
        shape: How the attribute holds its text.
        part: What part of the role's prompt it is.
        runner_role: The shared role runner's label for the role, when it runs there.
        judge: Whether the role grades episodes, so a rewrite must keep the labeled acceptance
            cases (weakened evaluation fails, genuine and alternative repairs pass).
    """

    role: str
    module: str
    attribute: str
    shape: Shape = "text"
    part: Part = "instruction"
    runner_role: str | None = None
    judge: bool = False


def _src(
    role: str,
    module: str,
    attribute: str,
    shape: Shape = "text",
    part: Part = "instruction",
    runner: str | None = None,
    judge: bool = False,
) -> TargetSource:
    """Shorthand for one source row."""
    return TargetSource(role, module, attribute, shape, part, runner, judge)


SOURCES: tuple[TargetSource, ...] = (
    # Environment families.
    _src("families-compatibility-reviewer", f"{S}.mining.environment_review", "REVIEW_PROMPT"),
    _src("families-compatibility-reviewer", f"{S}.mining.environment_review", "RECONCILE_PROMPT"),
    # Corpus tags, cluster names and task types.
    _src("tag-bank-author", f"{S}.mining.tag_authoring", "AUTHOR_PROMPT"),
    _src("decision-tagger", f"{S}.mining.tagging", "_INSTRUCTIONS"),
    _src("cluster-naming", f"{S}.mining.tag_cluster_names", "_NAMING_SYSTEM", part="system"),
    _src(
        "cluster-naming",
        f"{S}.mining.tag_cluster_names",
        "DEFAULT_NAMING_PROMPTS",
        "variants",
        "stance",
    ),
    _src(
        "task-type-naming",
        f"{S}.mining.task_types",
        "DEFAULT_TASK_TYPE_PROMPTS",
        "variants",
        "stance",
    ),
    _src("traceml-moves", f"{S}.mining.traceml_moves", "MOVE_PROMPT"),
    # World model (the simulator) and its learning.
    _src("world-model", P, "WORLD_MODEL_TEXT_SYSTEM_PROMPT", part="system"),
    _src("world-model", P, "RECONSTRUCTED_TEXT_SYSTEM_PROMPT", part="system"),
    _src(
        "world-model",
        P,
        "RECONSTRUCTED_JSON_TOOL_SYSTEM_PROMPT",
        part="system",
    ),
    _src("world-model", P, "INSTANCE_FACTS_GUIDANCE", part="guidance"),
    _src("world-model", P, "MEMORY_CONTEXT_GUIDANCE", part="guidance"),
    _src("world-model", P, "NATIVE_ARTIFACT_PLAN_GUIDANCE", part="guidance"),
    _src(
        "world-model",
        f"{W}.artifact",
        "GROUNDED_WORLD_MODEL_SYSTEM_PROMPT",
        part="system",
    ),
    _src(
        "world-model",
        f"{W}.grounded_session",
        "LEARNED_RULES_GUIDANCE",
        part="guidance",
    ),
    _src(
        "world-model",
        f"{W}.grounded_session",
        "STATE_PRECEDENTS_GUIDANCE",
        part="guidance",
    ),
    _src("world-model", f"{W}.state_memory", "STATE_MEMORY_GUIDANCE", part="guidance"),
    _src("world-model-inspection", f"{S}.engines.inspection", "_INSPECTION_PROMPT"),
    _src("world-model-dreaming", f"{W}.dreaming", "PROPOSE_PROMPT"),
    _src("world-model-dreaming", f"{W}.dreaming", "CHECK_PROMPT"),
    _src("world-model-dreaming", f"{W}.dreaming", "JOINT_CHECK_PROMPT"),
    _src("world-model-memory", f"{W}.memory_tree", "COMPRESSION_PROMPT"),
    _src("world-model-guidance", f"{W}.session_guidance", "GUIDANCE_PROMPT"),
    _src("world-model-research", f"{W}.session_research", "INVESTIGATION_PROMPT"),
    _src("memory-analyst", f"{S}.evaluation.analysis", "ANALYSIS_PROMPT"),
    _src("memory-analyst", f"{S}.evaluation.analysis", "MEMORY_ANALYSIS_GUIDANCE", part="guidance"),
    _src("memory-analyst", f"{S}.evaluation.analysis", "PAIRED_ANALYSIS_GUIDANCE", part="guidance"),
    # World authoring.
    _src("world-author", f"{V}.world_author", "WORLD_AUTHOR_PROMPT"),
    _src("world-author", f"{V}.world_agent", "PROMPT_SETS", "world_sets", "stance"),
    # Rule drafting, compilation and consolidation.
    _src("rule-extraction", f"{R}.extraction", "EXTRACTION_PROMPT"),
    _src("rule-extraction", f"{R}.agent_loop", "LOOP_INSTRUCTION"),
    _src("rule-extraction", f"{R}.agent_loop", "PHASE_PROMPTS", "dict", "phase"),
    _src("rule-research", f"{R}.research", "RESEARCH_INSTRUCTION"),
    _src("rule-weighing", f"{R}.weighing", "WEIGHING_INSTRUCTION"),
    _src("rule-weighing", f"{R}.weighing", "RATIONALE_INSTRUCTION"),
    _src("rule-condense", f"{R}.condense", "CONDENSE_INSTRUCTION"),
    _src("rule-planner", f"{R}.system_two", "PLANNER_INSTRUCTION"),
    _src("rule-tag-compiler", f"{R}.rule_tags", "COMPILER_INSTRUCTION"),
    _src("rule-consolidation", f"{R}.rule_consolidation", "CONSOLIDATION_INSTRUCTION"),
    # State-tag RAG and per-task state memory.
    _src("state-tag-author", f"{S}.retrieval.state_tags", "_AUTHOR_SYSTEM", part="system"),
    _src(
        "state-tag-author",
        f"{S}.retrieval.state_tags",
        "DEFAULT_STATE_TAG_PROMPTS",
        "variants",
        "stance",
    ),  # fmt: skip
    _src("state-memory-author", f"{W}.state_memory", "_SYSTEM", part="system"),
    _src(
        "state-memory-author",
        f"{W}.state_memory",
        "DEFAULT_STATE_MEMORY_PROMPTS",
        "variants",
        "stance",
    ),
    _src("evidence-reader", f"{S}.retrieval.evidence_reading", "PLAN_PROMPT"),
    _src("evidence-reader", f"{S}.retrieval.evidence_reading", "READER_PROMPT"),
    _src("evidence-reader", f"{S}.retrieval.evidence_reading", "SYNTHESIS_PROMPT"),
    # Start authoring and review (bootstrap).
    _src("start-author", f"{V}.bootstrap_prompts", "AUTHOR_INSTRUCTION"),
    _src("start-author", f"{V}.bootstrap_prompts", "RESEARCH_INSTRUCTION"),
    _src("start-author", f"{V}.bootstrap_prompts", "REVEAL_INSTRUCTION"),
    _src("start-review", f"{V}.bootstrap_prompts", "REVIEW_INSTRUCTION"),
    # Re-authoring continue-from-here tasks, variants and nudges.
    _src("scenario-author", f"{V}.authors", "SCENARIO_PROMPT"),
    _src("decision-state-author", f"{V}.authors", "DECISION_STATE_PROMPT"),
    _src("briefing-review", f"{V}.briefing_review", "REVIEW_PROMPT"),
    _src("briefing-review", f"{V}.briefing_review", "NUDGE_REVIEW_PROMPT"),
    _src("variant-search", f"{V}.variant_search", "SEARCH_ADDENDUM", runner="search"),
    _src("variant-search", f"{V}.variant_search", "SEARCH_STANCES", "dict", "stance", "search"),
    _src("nudge-author", f"{V}.nudges", "NUDGE_AUTHOR_INSTRUCTION", runner="nudge-author"),
    _src("nudge-author", f"{V}.nudges", "NUDGE_AUTHOR_STANCES", "dict", "stance", "nudge-author"),
    _src("nudge-realism", f"{V}.nudges", "REALISM_INSTRUCTION", runner="nudge-realism"),
    _src("nudge-realism", f"{V}.nudges", "REALISM_STANCES", "dict", "stance", "nudge-realism"),
    _src("choice-reader", f"{V}.decision_measure", "CHOICE_INSTRUCTION", runner="choice"),
    _src("choice-reader", f"{V}.decision_measure", "CHOICE_STANCES", "dict", "stance", "choice"),
    _src("clarity-reviewer", f"{V}.family_stages", "CLARITY_INSTRUCTION", runner="clarity"),
    _src("clarity-reviewer", f"{V}.family_stages", "CLARITY_STANCES", "dict", "stance", "clarity"),
    _src("point-options", f"{V}.point_options", "OPTIONS_INSTRUCTION", runner="point-options"),
    _src(
        "point-options", f"{V}.point_options", "OPTION_STANCES", "dict", "stance", "point-options"
    ),
    # Selectors and critics.
    _src("point-selector", f"{V}.roles", "JUDGE_PROMPT"),
    _src("point-selector", f"{V}.role_prompts", "SELECTOR_STANCES", "dict", "stance"),
    _src("point-selector", f"{V}.point_selection", "BAR_SELECTOR_PROMPT"),
    _src("success-critic", f"{V}.roles", "CRITIC_PROMPT"),
    _src("success-critic", f"{V}.role_prompts", "CRITIC_STANCES", "dict", "stance"),
    _src("selector-critic-loop", f"{V}.role_prompts", "AGENTIC_INSTRUCTION"),
    _src("selector-critic-loop", f"{V}.role_prompts", "READ_ONLY_INSTRUCTION"),
    _src("selector-critic-loop", f"{V}.role_prompts", "PHASE_PROMPTS", "dict", "phase"),
    # Behavior differences.
    _src("behavior-labeler", f"{V}.behavior_labeling", "LABELER_PROMPT"),
    _src("behavior-labeler", f"{V}.behavior_labeling", "REPAIR_INSTRUCTION"),
    _src("behavior-pairs", f"{V}.behavior_pairs", "ALIGN_PROMPT"),
    _src("behavior-pairs", f"{V}.behavior_pairs", "AUDIT_PROMPT"),
    _src("behavior-curator", f"{V}.behavior_curation", "CURATOR_PROMPT"),
    # The two-part rubric: outcome judge and dense violation judge (hard-gated).
    _src("outcome-judge", f"{V}.judge_prompt", "JUDGE_PROMPT", judge=True),
    _src("outcome-judge", f"{V}.judge_prompt", "JUDGE_STANCES", "dict", "stance", judge=True),
    _src("outcome-judge", f"{V}.judge_prompt", "JUDGE_PHASE_PROMPTS", "dict", "phase", judge=True),
    _src("dense-judge", f"{V}.dense_prompt", "DENSE_JUDGE_PROMPT", judge=True),
    _src("dense-judge", f"{V}.dense_prompt", "DENSE_STANCES", "dict", "stance", judge=True),
    _src("dense-judge", f"{V}.dense_prompt", "DENSE_PHASE_PROMPTS", "dict", "phase", judge=True),
    # Rubric authoring, critique and revision.
    _src("rubric-author", f"{V}.rubric_author", "RUBRIC_AUTHOR_PROMPT"),
    _src("rubric-author", f"{V}.rubric_agent", "PROMPT_VARIANTS", "dict", "stance"),
    _src("rubric-author", f"{V}.rubric_agent", "PHASE_INSTRUCTIONS", "dict", "phase"),
    _src("rubric-author", f"{V}.rubric_agent", "RECONCILE_PROMPT"),
    _src("rubric-critic", f"{V}.rubric_critic", "CRITIC_PROMPT"),
    _src("rubric-critic", f"{V}.rubric_critic", "REPAIR_INSTRUCTION"),
    _src("rubric-reviser", f"{V}.rubric_revision", "REVISION_PROMPT"),
    # Adversarial mining and blast radius.
    _src(
        "adversarial-linter",
        f"{V}.adversarial_rubric",
        "LINTER_INSTRUCTION",
        runner="adversarial-linter",
    ),  # fmt: skip
    _src(
        "adversarial-linter",
        f"{V}.adversarial_rubric",
        "LINTER_STANCES",
        "dict",
        "stance",
        "adversarial-linter",
    ),  # fmt: skip
    _src(
        "adversarial-linter-review",
        f"{V}.adversarial_rubric",
        "REVIEW_INSTRUCTION",
        runner="adversarial-linter-review",
    ),  # fmt: skip
    _src(
        "adversarial-linter-review",
        f"{V}.adversarial_rubric",
        "REVIEW_STANCE",
        "pair",
        "stance",
        "adversarial-linter-review",
    ),  # fmt: skip
    _src("blast-severity", f"{V}.blast_radius", "SEVERITY_INSTRUCTION", runner="severity"),
    _src("blast-severity", f"{V}.blast_radius", "SEVERITY_STANCES", "dict", "stance", "severity"),
    # Rubric evolution author.
    _src(
        "rubric-evolution-author",
        f"{V}.rubric_signatures",
        "PROBE_INSTRUCTION",
        runner="rubric-probe-author",
    ),  # fmt: skip
    _src(
        "rubric-evolution-author",
        f"{V}.rubric_signatures",
        "PROBE_STANCES",
        "dict",
        "stance",
        "rubric-probe-author",
    ),  # fmt: skip
    _src(
        "rubric-evolution-author",
        f"{V}.rubric_signatures",
        "COMPILE_INSTRUCTION",
        runner="rubric-signature-compiler",
    ),  # fmt: skip
    _src(
        "rubric-evolution-author",
        f"{V}.rubric_signatures",
        "COMPILE_STANCES",
        "dict",
        "stance",
        "rubric-signature-compiler",
    ),  # fmt: skip
    _src(
        "rubric-evolution-author",
        f"{V}.rubric_items",
        "ITEM_INSTRUCTION",
        runner="rubric-item-compiler",
    ),  # fmt: skip
    _src(
        "rubric-evolution-author",
        f"{V}.rubric_items",
        "ITEM_STANCES",
        "dict",
        "stance",
        "rubric-item-compiler",
    ),  # fmt: skip
    _src(
        "rubric-evolution-author",
        f"{V}.rubric_items",
        "CONSOLIDATE_INSTRUCTION",
        runner="rubric-consolidation",
    ),  # fmt: skip
    _src(
        "rubric-evolution-author",
        f"{V}.rubric_items",
        "CONSOLIDATE_STANCE",
        "pair",
        "stance",
        "rubric-consolidation",
    ),  # fmt: skip
    # Tag and epoch monitor.
    _src(
        "tag-epoch-monitor", f"{V}.tag_monitor_review", "INSTRUCTION", runner="tag-monitor-review"
    ),
    _src(
        "tag-epoch-monitor",
        f"{V}.tag_monitor_review",
        "REVIEW_STANCES",
        "dict",
        "stance",
        "tag-monitor-review",
    ),  # fmt: skip
)
"""Every declared prompt attribute of every targeted role."""

EXCLUDED_ROLES: Mapping[str, str] = {
    "actor": "the five actor tiers are the policies being measured; their prompts are not tuned",
    "rubric-audit": "the council audit is drawn by the five council actor aliases, not one "
    "open-weight runtime alias",
}
"""Roles deliberately left out, with the reason."""


class PromptTarget(ContractModel):
    """One refinable prompt or prompt variant.

    Attributes:
        target_id: ``<role>/<attribute>`` or ``<role>/<attribute>/<variant>``.
        role: The pipeline role.
        module: Module holding the text.
        attribute: Attribute name.
        variant: The entry within the attribute, for an expanded attribute.
        part: What part of the role's prompt it is.
        runner_role: The shared role runner's label, when the role runs there.
        judge: Whether the labeled acceptance cases gate a rewrite.
        text: The live code text.
        sha256: Digest of ``text``.
    """

    target_id: str
    role: str
    module: str
    attribute: str
    variant: str | None = None
    part: Part
    runner_role: str | None = None
    judge: bool = False
    text: str
    sha256: Sha256


def _entries(source: TargetSource) -> list[tuple[str | None, str]]:
    """The ``(variant, text)`` entries of one source attribute, read from the live module."""
    value = getattr(importlib.import_module(source.module), source.attribute)
    if source.shape == "text":
        if not isinstance(value, str):
            raise TypeError(f"{source.module}.{source.attribute} is not a text")
        return [(None, value)]
    if source.shape == "pair":
        name, text = value
        return [(str(name), str(text))]
    if source.shape == "dict":
        if not isinstance(value, dict):
            raise TypeError(f"{source.module}.{source.attribute} is not a dict of texts")
        return [(str(key), str(text)) for key, text in value.items()]
    if source.shape == "variants":
        return [(str(row.variant_id), str(row.stance)) for row in value]
    return [(str(name), str(row.approach)) for name, row in value.items()]


def prompt_targets() -> tuple[PromptTarget, ...]:
    """Return every target, reading each prompt's live text.

    Raises:
        ValueError: Two targets share an identity.
    """
    rows: list[PromptTarget] = []
    for source in SOURCES:
        for variant, text in _entries(source):
            suffix = "" if variant is None else f"/{variant}"
            rows.append(
                PromptTarget(
                    target_id=f"{source.role}/{source.attribute}{suffix}",
                    role=source.role,
                    module=source.module,
                    attribute=source.attribute,
                    variant=variant,
                    part=source.part,
                    runner_role=source.runner_role,
                    judge=source.judge,
                    text=text,
                    sha256=sha256_json(text),
                )
            )
    if len({row.target_id for row in rows}) != len(rows):
        raise ValueError("two prompt targets share an identity")
    return tuple(rows)


def target(target_id: str) -> PromptTarget:
    """Return one target by identity.

    Raises:
        KeyError: No such target.
    """
    for row in prompt_targets():
        if row.target_id == target_id:
            return row
    raise KeyError(f"no prompt target {target_id!r}")


def unwired_targets() -> tuple[str, ...]:
    """Targets refinable offline whose runtime call site does not yet read a frozen prompt."""
    return tuple(row.target_id for row in prompt_targets() if row.runner_role is None)


def roles() -> dict[str, tuple[str, ...]]:
    """Every targeted role with its target identities, in declaration order."""
    found: dict[str, list[str]] = {}
    for row in prompt_targets():
        found.setdefault(row.role, []).append(row.target_id)
    return {role: tuple(ids) for role, ids in found.items()}
