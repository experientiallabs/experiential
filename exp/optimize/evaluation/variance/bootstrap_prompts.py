"""Prompt variants, read-only tools and answer tools of the start bootstrap roles.

Four roles take part in bootstrapping a task's starting state from traces
(:mod:`~exp.optimize.evaluation.variance.bootstrap`):

- ``start_author`` drafts the task's public instruction, tool schemas, world manifest, initial
  state, evaluator projection, verifier and fact ledger, separating observed, inferred and unknown
  facts with their supporting references.
- ``start_reviewer`` checks whether that start is coherent, sufficiently specified and supports
  the task, and names missing deciding facts as research items or refuses with a diagnosis.
- ``start_researcher`` answers one research item from the family's evidence.
- ``reveal_checker`` compares what the captured start shows on its first reads with the authored
  facts.

Every role runs the shared bounded agentic loop
(:func:`~exp.optimize.evaluation.rules.agent_loop.run_agent`) under one of several stances.
Every instruction, stance and tool here is pinned by :func:`bootstrap_prompts_sha256`, so a
changed prompt re-keys every draw that used it; no role writes a prompt inline.
"""

from __future__ import annotations

from exp.common.core.artifacts import JsonObject, sha256_json
from exp.common.tasks import ToolSchema
from exp.optimize.evaluation.rules.agent_loop import tool_schema

BOOTSTRAP_PROTOCOL = "start-bootstrap-v3"
"""Version of the roles, their prompts, tools and checks; change it whenever any changes."""

_REFERENCES = """References are strings from the material inventory: trace:<trace_id> or
trace:<trace_id>:span:<span_id> for recorded evidence, rule:<rule_id> for the family's learned
rules, contract:<tool> for parsed tool contracts, finding:<finding_id> for research findings and
recorded-state for a recorded environment snapshot. Cite only references in the inventory.
The opening context counts them by kind; read_trace, read_rules and your findings return the
exact references, and every one they return is valid."""

AUTHOR_INSTRUCTION = (
    """You author the starting state of one executable task from recorded agent traces.

Produce the task's public instruction, the tool schemas the actor sees, the world manifest (every
state domain simulated), the exact initial state, the evaluator projection (probe_paths, JSON
pointers into the initial state), a verifier over the final state, and a ledger of facts.
Every fact is observed (recorded evidence shows it), inferred (it follows from cited evidence
or rules) or unknown (nothing recorded decides it). Mark a fact deciding when the right course
of action depends on it. Never present an inferred or unknown fact as observed.

Provenance: observed only when the start restores a recorded state snapshot unchanged;
reconstructed when it is rebuilt from recorded evidence and the recorded prefix of a source trace
must replay to the same observations (name replay_trace and prefix_steps); authored when it is
generated. An authored start is never the original environment; give reveal_probes (read-only
calls whose first answers show the facts the task depends on).
"""
    + _REFERENCES
)

REVIEW_INSTRUCTION = (
    """You review the starting state of one executable task before it is captured.

Decide whether the start is coherent (state, manifest, tools, instruction and verifier agree),
sufficiently specified (every deciding fact is settled, not unknown or guessed) and supports the
task (an actor can succeed or fail on its own decisions). Answer accept, research or refuse.
Choose research when a deciding fact is missing and the family's evidence could settle it: name
each missing fact as a research item with the question and why it decides. Choose refuse, with a
diagnosis, when the start cannot be made sound from the evidence. Cite what supports each finding.
"""
    + _REFERENCES
)

RESEARCH_INSTRUCTION = (
    """You research one missing fact for the starting state of an executable task.

Read the family's recorded traces, contracts and learned rules to answer the research item.
Answer resolved only with evidence you actually read; otherwise answer unresolved and say what is
missing. State whether the answer is observed (recorded evidence shows it) or inferred.
"""
    + _REFERENCES
)

REVEAL_INSTRUCTION = """You check the first reveal of an authored executable start.

The host captured the start and executed each reveal probe once, each from a fresh copy of the
start. Compare what each probe revealed with the start's fact ledger and public instruction.
Answer consistent only when no reveal contradicts a fact or the instruction; otherwise list each
conflict with the probe index and the facts it contradicts."""

AUTHOR_VARIANTS: dict[str, str] = {
    "evidence_first": """Stance: start from the recorded evidence. Settle each state field from
what the traces show, then fill the rest from rules, marking each fill inferred or unknown.""",
    "task_first": """Stance: start from the task. List the decisions the actor must make, the
facts each decision depends on, and settle those facts first from the evidence.""",
}
"""Author stances; each round's draw uses one, assigned by task and round."""

REVIEW_VARIANTS: dict[str, str] = {
    "coherence_first": """Stance: check coherence first. Look for state fields, tools, manifest
effects, instruction and verifier that disagree with each other or with the cited evidence.""",
    "specification_first": """Stance: check specification first. Find every fact the right course
of action depends on and ask whether the start settles it from evidence.""",
    "task_support_first": """Stance: check task support first. Ask whether an actor can reach a
verifiable success or failure from this start through its own decisions.""",
}
"""Reviewer stances; every stance reviews every draft."""

RESEARCH_VARIANTS: dict[str, str] = {
    "trace_evidence_first": """Stance: read the recorded traces first and answer from what they
show; use rules only to connect observations.""",
    "rule_book_first": """Stance: read the learned rules and contracts first, then confirm or
refute them against the recorded traces they cite.""",
}
"""Researcher stances, assigned per research item."""

REVEAL_VARIANTS: dict[str, str] = {
    "fact_by_fact": """Stance: go fact by fact through the ledger and find the reveal that bears
on each.""",
    "actor_view": """Stance: read the reveals as the actor would, then ask whether the
instruction and facts still describe the world the actor sees.""",
}
"""First-reveal checker stances; every stance checks every authored start."""

_STRINGS: JsonObject = {"type": "array", "items": {"type": "string"}}

READ_TOOLS: tuple[ToolSchema, ...] = (
    tool_schema(
        "read_trace",
        "Read one recorded trajectory's ordered tool steps and their locators.",
        {"trace_id": {"type": "string"}},
        ("trace_id",),
    ),
    tool_schema(
        "read_contract",
        "Read the parsed contract of one tool: argument types, result shapes, failures.",
        {"tool": {"type": "string"}},
        ("tool",),
    ),
    tool_schema(
        "read_rules",
        "Read the family's learned rules, optionally only those about one tool.",
        {"tool": {"type": ["string", "null"]}},
        (),
    ),
    tool_schema("read_findings", "Read every research finding made so far.", {}, ()),
    tool_schema(
        "read_recorded_state", "Read the recorded environment state snapshot, if any.", {}, ()
    ),
)
"""Read-only tools every bootstrap role may call."""

_FACT: JsonObject = {
    "type": "object",
    "properties": {
        "fact_id": {"type": "string"},
        "statement": {"type": "string"},
        "status": {"type": "string", "enum": ["observed", "inferred", "unknown"]},
        "deciding": {"type": "boolean"},
        "state_paths": _STRINGS,
        "references": _STRINGS,
    },
    "required": ["fact_id", "statement", "status", "deciding", "references"],
}

AUTHOR_ANSWER = tool_schema(
    "submit_start",
    "Submit the complete starting state.",
    {
        "instruction": {"type": "string"},
        "tools": {"type": "array", "items": {"type": "object"}},
        "manifest": {"type": "object"},
        "initial_state": {"type": "object"},
        "probe_paths": {"type": "object"},
        "verifier": {"type": "object"},
        "provenance": {"type": "string", "enum": ["observed", "reconstructed", "authored"]},
        "replay_trace": {"type": ["string", "null"]},
        "prefix_steps": {"type": "integer"},
        "facts": {"type": "array", "items": _FACT},
        "reveal_probes": {"type": "array", "items": {"type": "object"}},
        "limitations": _STRINGS,
    },
    (
        "instruction",
        "tools",
        "manifest",
        "initial_state",
        "probe_paths",
        "verifier",
        "provenance",
        "facts",
    ),
)
"""The author's answer tool."""

REVIEW_ANSWER = tool_schema(
    "submit_review",
    "Submit the review verdict.",
    {
        "verdict": {"type": "string", "enum": ["accept", "research", "refuse"]},
        "summary": {"type": "string"},
        "findings": {"type": "array", "items": {"type": "object"}},
        "research_items": {"type": "array", "items": {"type": "object"}},
        "refusal": {"type": ["string", "null"]},
    },
    ("verdict", "summary"),
)
"""The reviewer's answer tool."""

RESEARCH_ANSWER = tool_schema(
    "submit_finding",
    "Submit the answer to the research item.",
    {
        "status": {"type": "string", "enum": ["resolved", "unresolved"]},
        "answer": {"type": "string"},
        "fact_status": {"type": "string", "enum": ["observed", "inferred", "unknown"]},
        "evidence": _STRINGS,
    },
    ("status", "answer", "fact_status", "evidence"),
)
"""The researcher's answer tool."""

REVEAL_ANSWER = tool_schema(
    "submit_reveal_check",
    "Submit whether the first reveals agree with the start.",
    {
        "consistent": {"type": "boolean"},
        "summary": {"type": "string"},
        "conflicts": {"type": "array", "items": {"type": "object"}},
    },
    ("consistent", "summary"),
)
"""The first-reveal checker's answer tool."""


def instruction(base: str, stances: dict[str, str], variant: str) -> str:
    """Return a role's system instruction under one stance.

    Args:
        base: The role instruction.
        stances: The role's stances.
        variant: The stance name.

    Returns:
        The instruction followed by the stance.

    Raises:
        ValueError: The stance is unknown.
    """
    if variant not in stances:
        raise ValueError(f"unknown prompt variant {variant!r}; choose one of {sorted(stances)}")
    return base + "\n\n" + stances[variant]


def bootstrap_prompts_sha256() -> str:
    """Return the digest of every bootstrap instruction, stance and tool."""
    return sha256_json(
        {
            "protocol": BOOTSTRAP_PROTOCOL,
            "instructions": [
                AUTHOR_INSTRUCTION,
                REVIEW_INSTRUCTION,
                RESEARCH_INSTRUCTION,
                REVEAL_INSTRUCTION,
            ],
            "variants": [AUTHOR_VARIANTS, REVIEW_VARIANTS, RESEARCH_VARIANTS, REVEAL_VARIANTS],
            "tools": [
                tool.model_dump(mode="json")
                for tool in (
                    *READ_TOOLS,
                    AUTHOR_ANSWER,
                    REVIEW_ANSWER,
                    RESEARCH_ANSWER,
                    REVEAL_ANSWER,
                )
            ],
        }
    )
