"""Versioned prompt variants and loop-phase prompts of the point selectors and success critics.

A role's system instruction is its fixed contract (:mod:`~exp.optimize.evaluation.variance.roles`)
plus one **stance**: a distinct way of investigating the same evidence. Several stances exist per
role so that a set of draws explores the group differently instead of repeating one reading:

- selector: ``divergence_seeking``, ``failure_forensics``, ``counterfactual_branching`` and
  ``pivot_and_return``;
- critic: ``base_rate_anchored``, ``mechanism_tracing`` and ``adversarial_skeptical``.

Every draw is assigned exactly one stance by :func:`assign_prompts`: the configured
:class:`PromptDistribution` fixes integer weights, the draws are apportioned to stances by largest
remainder, and the apportioned list is shuffled with a seed derived from the registration seed,
the role and the weights. The assignment is deterministic, balanced, and recorded per draw as a
:class:`PromptAssignment` carrying the digest of the exact stance, phase prompts and phase tools.

The agentic loop (:mod:`~exp.optimize.evaluation.variance.evidence_tools`) runs the phases
``plan``, ``investigate``, ``draft``, ``self_check`` and ``final``. Each phase's instruction is
in :data:`PHASE_PROMPTS` and travels inside the tool result of the previous phase, so the role
always knows which phase it is in. :func:`role_prompts_sha256` pins every text and schema here.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from typing import Literal, Self

from pydantic import Field, model_validator

from exp.common.core.artifacts import ContractModel, JsonObject, Sha256, sha256_json
from exp.common.tasks import ToolSchema

ROLE_PROMPTS_VERSION = "variance-role-prompts-v1"
"""Version of the stances, phase prompts and phase tools; change it whenever any changes."""

RoleName = Literal["selector", "critic"]
"""The two estimation roles that read the privileged evidence packet."""

PhaseName = Literal["plan", "investigate", "draft", "self_check", "final"]
"""The phases of one agentic draw, in order."""

PHASES: tuple[PhaseName, ...] = ("plan", "investigate", "draft", "self_check", "final")
"""Every phase, in the order the loop runs them."""

SELECTOR_STANCES: dict[str, str] = {
    "divergence_seeking": """Start from where attempts in the group stop agreeing. Use
list_divergence_points and compare_attempts to find the earliest boundaries at which attempts
that shared a history took different actions, and prefer points of the picked attempt that sit at
or just before such a split when the branches end differently.""",
    "failure_forensics": """Work backwards from failure. Find the first observable mistake in the
picked attempt (and in other failing attempts), read the tool results and world state just before
it, and locate the last boundary where the actor still had the information and the options to
avoid it. That pre-error decision, not the visible failure itself, is the point to choose.""",
    "counterfactual_branching": """Think in executable counterfactuals. For each candidate
boundary, write down the concrete alternative actions the actor could have issued there, check in
the world state and tool contracts that they were legal and would have changed what followed, and
keep only points where at least one such alternative plausibly flips the outcome.""",
    "pivot_and_return": """Track the actor's strategy over time. Find where it abandoned an
approach, switched to a new one, or returned to something it had dropped, and read the evidence
around each pivot. Choose the boundaries where committing to, or abandoning, an approach decided
what the remaining budget could still achieve.""",
}
"""Selector stances: distinct ways of finding consequential decisions."""

CRITIC_STANCES: dict[str, str] = {
    "base_rate_anchored": """Anchor on base rates. Start every estimate at the rate observed for
attempts that passed through a comparable state, adjust only for evidence specific to this point,
and state each adjustment and the evidence behind it. Without specific evidence, stay at the base
rate.""",
    "mechanism_tracing": """Trace mechanisms. For each point, identify what must still happen for
success (remaining obligations, preconditions in the world state, tool contracts, remaining
budget), read the evidence for each, and combine how likely the actor is to complete every step
from exactly that point.""",
    "adversarial_skeptical": """Be adversarial toward your own first estimate. After forming it,
look for the strongest evidence that it is wrong in either direction (an attempt that recovered
from a similar state, or one that failed after looking safe), read it, and move the estimate only
as far as that evidence warrants.""",
}
"""Critic stances: distinct ways of estimating continuation success."""

STANCES: dict[RoleName, dict[str, str]] = {
    "selector": SELECTOR_STANCES,
    "critic": CRITIC_STANCES,
}
"""Every stance of every role."""

READ_ONLY_INSTRUCTION = """

You may read evidence before answering. The opening message is an overview: the group, the
verifier, attempt summaries, the checkpoint inventory, the evidence inventory (including material
that was not supplied) and every locator. The read-only tools fetch exact evidence by locator;
they cannot run actions, change any state or grade, or reach the actor. Your budget is frozen:
{budget}. When it runs low you must submit your answer with {answer}. Cite only listed locators."""
"""Addendum of the read-only tool profile: read, then answer."""

AGENTIC_INSTRUCTION = """

You investigate as an agent, in phases. The opening message is an overview: the group, the
verifier, attempt summaries, the checkpoint inventory, the evidence inventory (including material
that was not supplied), whether world-learning retrieval is available, and every locator. The
read-only tools fetch exact evidence, world state at a checkpoint, tool contracts and the harness
definition, registered references, side-by-side attempt comparisons, divergence points across the
group, step labels and world-learning traces. They cannot run actions, change any state or grade,
or reach the actor. Phases:
1. plan: record what you will investigate and why, with record_plan.
2. investigate: read the evidence your plan needs, then submit a draft with submit_draft.
3. self_check: check the draft against the evidence (and the host's validation of it), read more
   if a claim is unsupported, then record the outcome with record_self_check.
4. final: give the final answer with {answer}. It may differ from the draft.
Each tool result names the next phase. Your budget is frozen: {budget}. When it runs low the
host moves you to the final phase. Cite only listed locators."""
"""Addendum of the agentic profile: the phase protocol."""

PHASE_PROMPTS: dict[PhaseName, str] = {
    "plan": """PLAN. Before reading anything, call record_plan: the hypotheses you will test, the
evidence you will read for each (by locator or tool), and what would change your mind.""",
    "investigate": """INVESTIGATE. Read the evidence your plan names with the read-only tools.
Follow surprises, not only the plan. When you can support an answer, call submit_draft with the
complete answer you would give now.""",
    "draft": """DRAFT RECORDED. The host validated the draft against the packet; the result is in
host_validation. Continue with the self check.""",
    "self_check": """SELF CHECK. Check every claim in your draft against evidence you actually
read: each cited locator, each chosen point or estimate, and the host validation. Read more when a
claim is unsupported or a contrary reading is plausible. Then call record_self_check with the
issues found and whether the final answer must differ from the draft.""",
    "final": """FINAL. Give your final answer now with {answer}, corrected for every issue the
self check found. Cite only supplied locators.""",
}
"""The instruction delivered at the start of each phase."""

PLAN_TOOL = ToolSchema(
    name="record_plan",
    description="Record the investigation plan before reading evidence.",
    input_schema={
        "type": "object",
        "properties": {
            "hypotheses": {"type": "array", "items": {"type": "string"}, "maxItems": 16},
            "planned_reads": {"type": "array", "items": {"type": "string"}, "maxItems": 32},
            "disconfirming_evidence": {"type": "string", "maxLength": 4000},
        },
        "required": ["hypotheses", "planned_reads"],
        "additionalProperties": False,
    },
)
"""The plan phase's tool."""

SELF_CHECK_TOOL = ToolSchema(
    name="record_self_check",
    description="Record the self check of the draft against the evidence.",
    input_schema={
        "type": "object",
        "properties": {
            "issues": {"type": "array", "items": {"type": "string"}, "maxItems": 32},
            "revise": {"type": "boolean"},
            "notes": {"type": "string", "maxLength": 4000},
        },
        "required": ["issues", "revise"],
        "additionalProperties": False,
    },
)
"""The self-check phase's tool."""


def _schema(properties: JsonObject, required: Sequence[str]) -> JsonObject:
    """Return a closed object schema."""
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


_INT: JsonObject = {"type": "integer", "minimum": 0}
_TEXT: JsonObject = {"type": "string", "maxLength": 256}
_POINTER: JsonObject = {"type": "string", "maxLength": 512, "description": "JSON pointer"}

EVIDENCE_TOOLS: tuple[ToolSchema, ...] = (
    ToolSchema(
        name="list_evidence",
        description="List supplied evidence locators that start with a prefix.",
        input_schema=_schema({"prefix": {"type": "string", "maxLength": 128}}, ("prefix",)),
    ),
    ToolSchema(
        name="read_trace_step",
        description="Read one exact tool turn (1-based) of one attempt's trajectory.",
        input_schema=_schema({"attempt": _INT, "step": _INT}, ("attempt", "step")),
    ),
    ToolSchema(
        name="read_checkpoint_prefix",
        description="Read actor-visible prefix messages at one offered point, by slice.",
        input_schema=_schema(
            {"point": _INT, "start": _INT, "count": {"type": "integer", "minimum": 1}},
            ("point", "start", "count"),
        ),
    ),
    ToolSchema(
        name="read_rule",
        description="Read one supplied rule-book entry by rule id.",
        input_schema=_schema({"rule_id": {"type": "string", "maxLength": 128}}, ("rule_id",)),
    ),
    ToolSchema(
        name="read_grader_evidence",
        description="Read one attempt's grade and the evaluator projection it was graded on.",
        input_schema=_schema({"attempt": _INT}, ("attempt",)),
    ),
    ToolSchema(
        name="read_evidence",
        description="Read any supplied evidence by its exact locator.",
        input_schema=_schema({"locator": {"type": "string", "maxLength": 256}}, ("locator",)),
    ),
    ToolSchema(
        name="read_world_state",
        description="Read the saved world state at one offered point, at a JSON pointer path "
        "('' is the whole checkpoint); an oversized result lists the children to narrow to.",
        input_schema=_schema({"point": _INT, "path": _POINTER}, ("point", "path")),
    ),
    ToolSchema(
        name="read_environment",
        description="Read the public task, harness settings and world kind of the group.",
        input_schema=_schema({}, ()),
    ),
    ToolSchema(
        name="read_tool_contract",
        description="Read one actor tool's declared schema and its rule-book contract.",
        input_schema=_schema({"tool": _TEXT}, ("tool",)),
    ),
    ToolSchema(
        name="read_reference",
        description="Read a registered reference by name, at a JSON pointer path into it.",
        input_schema=_schema({"name": _TEXT, "path": _POINTER}, ("name", "path")),
    ),
    ToolSchema(
        name="compare_attempts",
        description="Line two to four attempts up step by step over a window of tool turns.",
        input_schema=_schema(
            {
                "attempts": {"type": "array", "items": _INT, "minItems": 2, "maxItems": 4},
                "start": {"type": "integer", "minimum": 1},
                "count": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            ("attempts", "start", "count"),
        ),
    ),
    ToolSchema(
        name="list_divergence_points",
        description="List where attempts with identical earlier actions split, with grades.",
        input_schema=_schema({}, ()),
    ),
    ToolSchema(
        name="read_step_labels",
        description="Read registered trace step labels for one attempt's tool turn.",
        input_schema=_schema({"attempt": _INT, "step": _INT}, ("attempt", "step")),
    ),
    ToolSchema(
        name="search_world_traces",
        description="Search world-learning traces (never held out) for observed outcomes of an "
        "action in a situation; give tool_name and arguments, or a message.",
        input_schema=_schema(
            {
                "situation": {"type": "string", "minLength": 1, "maxLength": 4000},
                "tool_name": _TEXT,
                "arguments": {"type": "object"},
                "message": {"type": "string", "maxLength": 4000},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            ("situation", "top_k"),
        ),
    ),
)
"""The complete read-only tool set; nothing else is ever executed for a role."""

DRAFT_TOOL_NAME = "submit_draft"
"""Name of the draft tool; its schema is the role's own answer schema."""


def draft_tool(answer: ToolSchema) -> ToolSchema:
    """Return the draft tool, which takes exactly the role's answer schema."""
    return ToolSchema(
        name=DRAFT_TOOL_NAME,
        description="Submit a complete draft answer for the self check.",
        input_schema=answer.input_schema,
    )


def phase_prompt(phase: PhaseName, answer: str) -> str:
    """Return one phase's instruction for a role answering with ``answer``."""
    return PHASE_PROMPTS[phase].format(answer=answer)


def loop_instruction(mode: Literal["read_only_tools", "agentic"], budget: JsonObject) -> str:
    """Return the addendum for a tool-reading mode, with ``{answer}`` still open."""
    text = READ_ONLY_INSTRUCTION if mode == "read_only_tools" else AGENTIC_INSTRUCTION
    return text.replace("{budget}", json.dumps(budget, sort_keys=True))


def _phases_sha256() -> str:
    """Return the digest of the loop instructions, phase prompts and phase tools."""
    return sha256_json(
        {
            "read_only": READ_ONLY_INSTRUCTION,
            "agentic": AGENTIC_INSTRUCTION,
            "phases": PHASE_PROMPTS,
            "tools": [PLAN_TOOL.model_dump(mode="json"), SELF_CHECK_TOOL.model_dump(mode="json")],
            "draft": DRAFT_TOOL_NAME,
            "evidence_tools": [tool.model_dump(mode="json") for tool in EVIDENCE_TOOLS],
        }
    )


class PromptAssignment(ContractModel):
    """The prompt variant one draw ran under, recorded with the draw.

    Attributes:
        version: Prompt-set version.
        role: Role of the draw.
        variant: Stance name.
        draw: Draw index within the role.
        stance_sha256: Digest of the exact stance text.
        phases_sha256: Digest of the loop instructions, phase prompts and phase tools.
    """

    version: str = ROLE_PROMPTS_VERSION
    role: RoleName
    variant: str
    draw: int = Field(ge=0)
    stance_sha256: Sha256
    phases_sha256: Sha256

    @model_validator(mode="after")
    def _known(self) -> Self:
        """Refuse a stance or digest that is not this module's exact text."""
        stance = STANCES[self.role].get(self.variant)
        if stance is None:
            raise ValueError(f"unknown {self.role} prompt variant {self.variant!r}")
        if self.version != ROLE_PROMPTS_VERSION or self.stance_sha256 != sha256_json(stance):
            raise ValueError(f"{self.role} variant {self.variant} is not this prompt set's text")
        if self.phases_sha256 != _phases_sha256():
            raise ValueError("phase prompts differ from this prompt set")
        return self

    @property
    def stance(self) -> str:
        """Return the exact stance text."""
        return STANCES[self.role][self.variant]


def assignment(role: RoleName, variant: str, draw: int = 0) -> PromptAssignment:
    """Return the assignment of one named variant to one draw."""
    return PromptAssignment(
        role=role,
        variant=variant,
        draw=draw,
        stance_sha256=sha256_json(STANCES[role].get(variant, "")),
        phases_sha256=_phases_sha256(),
    )


class PromptDistribution(ContractModel):
    """Integer weights over one role's stances; draws are apportioned in proportion.

    Attributes:
        role: Role whose stances are weighted.
        weights: Positive weight per stance; an absent stance is never assigned.
    """

    role: RoleName
    weights: dict[str, int] = Field(min_length=1)

    @model_validator(mode="after")
    def _known(self) -> Self:
        """Refuse unknown stances and non-positive weights."""
        unknown = sorted(set(self.weights) - set(STANCES[self.role]))
        if unknown:
            raise ValueError(f"unknown {self.role} prompt variants {unknown}")
        if any(weight < 1 or weight > 1000 for weight in self.weights.values()):
            raise ValueError("prompt variant weights must be between 1 and 1000")
        return self

    @classmethod
    def uniform(cls, role: RoleName) -> PromptDistribution:
        """Return equal weights over every stance of a role."""
        return cls(role=role, weights=dict.fromkeys(STANCES[role], 1))


def assign_prompts(
    distribution: PromptDistribution, *, seed: int, draws: int
) -> tuple[PromptAssignment, ...]:
    """Deterministically assign one stance to each of ``draws`` draws.

    Draws are apportioned to stances by largest remainder over the weights (ties broken by stance
    name), then the apportioned list is shuffled with a seed derived from ``seed``, the role and
    the weights. The same inputs always give the same assignments.

    Args:
        distribution: Weights over the role's stances.
        seed: Registration seed.
        draws: Number of draws of the role.

    Returns:
        One assignment per draw, in draw order.
    """
    names = sorted(distribution.weights)
    total = sum(distribution.weights.values())
    exact = {name: draws * distribution.weights[name] / total for name in names}
    quota = {name: int(exact[name]) for name in names}
    left = draws - sum(quota.values())
    for name in sorted(names, key=lambda n: (-(exact[n] - quota[n]), n))[:left]:
        quota[name] += 1
    order = [name for name in names for _ in range(quota[name])]
    material = sha256_json(
        {"seed": seed, "role": distribution.role, "weights": distribution.weights}
    )
    random.Random(material).shuffle(order)
    return tuple(assignment(distribution.role, name, draw) for draw, name in enumerate(order))


def role_prompts_sha256() -> str:
    """Return the identity of every stance, loop instruction, phase prompt and phase tool."""
    return sha256_json(
        {"version": ROLE_PROMPTS_VERSION, "stances": STANCES, "phases": _phases_sha256()}
    )
