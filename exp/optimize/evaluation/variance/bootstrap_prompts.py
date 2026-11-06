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
from exp.optimize.evaluation.rules.agent_loop import AGENT_LOOP_PROTOCOL, tool_schema
from exp.optimize.evaluation.rules.evidence_pages import (
    CONTINUE_EVIDENCE,
    EVIDENCE_DELIVERY_PROTOCOL,
    PAGE_INSTRUCTION,
)
from exp.optimize.evaluation.rules.locator_inventory import LOCATOR_GRAMMAR
from exp.optimize.evaluation.variance.bootstrap_reads import STEP_READ_TOOL

BOOTSTRAP_PROTOCOL = "start-bootstrap-v11"
"""Version of the roles, their prompts, tools and checks; change it whenever any changes."""

_REFERENCES = (
    """References are strings from the material inventory: trace:<trace_id> or
trace:<trace_id>:span:<span_id> for recorded evidence, rule:<rule_id> for the family's learned
rules, contract:<tool> for parsed tool contracts, finding:<finding_id> for research findings and
recorded-state for a recorded environment snapshot. Cite only references in the inventory."""
    + "\n"
    + LOCATOR_GRAMMAR
)

AUTHOR_INSTRUCTION = (
    """You author the starting state of one executable task from recorded agent traces.

Produce the task's public instruction, exact initial state, tool_effects, evaluator projection
(probe_paths, JSON pointers into the initial state), executable verifier and fact ledger.
For submit_draft and submit_start, pass the proposal fields directly as tool arguments,
not a JSON string nested inside draft or answer.
Declare each tool once in tools, including name, description, input_schema, reads and writes.
Use the supplied task tool schemas when they support the task. The host derives the manifest
from those declarations and initial_state; do not submit manifest. Each read/write effect names
domains in initial_state. When family.state_domains is supplied, initial_state
must contain exactly those keys, including their exact spelling. family.domain_state_roots gives
their valid pointer prefixes. Use these for probe_paths and concrete fact state_paths; leave a
fact's state_paths empty when it names no concrete state field. Every domain is simulated.
A verifier's description is explanatory; only outcome_schema, required_paths and rules execute.
Encode required behavior and verification in those checks, rather than describing checks that
will never run. final_state in the episode outcome is the evaluator projection: each
probe_paths key names the value selected by its pointer into the state. Verifier state paths
start /final_state/<projection-key>/... . Do not require invented outcome fields such as passed;
the verifier itself computes pass/fail.
Use the smallest sufficient subset of supplied tools to keep the draft compact.
Project the full mutable state domains needed to verify results, rather than an initial empty
marker or note that never captures the actor's work. For example, probe_paths={"workspace":
"/Workspace filesystem (~1workspace)"} exposes that domain under /final_state/workspace.
A rule can then test /final_state/workspace/pytest_exit_code equals 0; the start must define
where tool execution stores that result. A required_paths check alone does not assert success.
An authored start must include at least one read-only reveal_probes call.
Task requirements already appear in the public instruction: they are objectives, not
environment facts. Do not duplicate them in the fact ledger or cite task.instruction.
Every fact is observed (recorded evidence shows it), inferred (it follows from cited evidence
or rules) or unknown (nothing recorded decides it). Mark a fact deciding when the right course
of action depends on it. Never present an inferred or unknown fact as observed.

The opening source_traces field lists identities and step counts only; it does not deliver
trace contents. Read the exact evidence needed through read_trace or read_trace_steps. An
inventory locator is not evidence you have read. Fact citations must come from successful,
complete evidence reads, or research findings supplied in the opening.

Provenance: observed only when the start restores a recorded state snapshot unchanged;
reconstructed when it is rebuilt from recorded evidence and the recorded prefix of a source trace
must replay to the same observations (name replay_trace and prefix_steps). Declare every tool
used by that recorded prefix; its source schema is in task.tools. A start is authored when it is
generated. An authored start is never the original environment; give reveal_probes (read-only
calls whose first answers show the facts the task depends on).
"""
    + _REFERENCES
)

REVIEW_INSTRUCTION = (
    """You review the starting state of one executable task before it is captured.

Decide whether the start is coherent (state, manifest, tools, instruction and verifier agree),
sufficiently specified (every deciding fact is settled, not unknown or guessed) and supports the
task (an actor can succeed or fail on its own decisions). Inspect the executable verifier's
rules, required_paths and outcome_schema, not only its description. When the task requires
behavior or tests, file presence alone is insufficient. Answer accept, research or refuse.
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
    STEP_READ_TOOL,
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


def author_answer_tool(domain_names: tuple[str, ...] = ()) -> ToolSchema:
    """Expose state and a single schema-and-effects declaration for each authored tool."""
    from exp.optimize.evaluation.variance.bootstrap_contracts import (
        StartDraft,
        StartTool,
        domain_state_roots,
    )

    schema = StartDraft.model_json_schema()
    schema["title"] = "StartProposal"
    schema["description"] = (
        "State, facts and verification with each tool's schema and effects declared together. "
        "The host derives the manifest from these tools and initial-state domain keys."
    )
    properties = schema["properties"]
    del properties["manifest"]
    schema["required"].remove("manifest")
    schema["allOf"] = [
        {
            "if": {"properties": {"provenance": {"const": "authored"}}},
            "then": {
                "required": ["reveal_probes"],
                "properties": {"reveal_probes": {"minItems": 1}},
            },
        }
    ]
    tool = StartTool.model_json_schema()
    schema["$defs"].update(tool.pop("$defs", {}))
    if domain_names:
        for field in ("reads", "writes"):
            tool["properties"][field]["items"] = {"type": "string", "enum": list(domain_names)}
        roots = domain_state_roots(domain_names)
        escapes = {ord(char): "\\" + char for char in r"\.^$|?*+()[]{}"}
        pattern = "^(?:" + "|".join(root.translate(escapes) for root in roots.values()) + ")(?:/|$)"
        pointer = {"type": "string", "pattern": pattern, "examples": list(roots.values())}
        properties["probe_paths"]["additionalProperties"] = pointer
        schema["$defs"]["FactClaim"]["properties"]["state_paths"]["items"] = pointer
        state = properties["initial_state"]
        value = state["additionalProperties"]
        state.update(
            properties={name: value for name in domain_names},
            required=list(domain_names),
            additionalProperties=False,
        )
    schema["$defs"]["StartTool"] = tool
    properties["tools"]["items"] = {"$ref": "#/$defs/StartTool"}
    properties["probe_paths"]["description"] = (
        "JSON pointers into initial_state. Escape each domain-key segment: "
        "escape '~' as '~0', then '/' as '~1'."
    )
    return ToolSchema(
        name="submit_start",
        description="Submit state, tool declarations, facts and executable verification.",
        input_schema=schema,
    )


def review_answer_tool() -> ToolSchema:
    """Expose the accepted review verdict, findings and research-item structures."""
    from exp.optimize.evaluation.variance.bootstrap_contracts import ReviewAnswer

    return ToolSchema(
        name="submit_review",
        description="Submit the review verdict.",
        input_schema=ReviewAnswer.model_json_schema(),
    )


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


def reveal_answer_tool() -> ToolSchema:
    """Expose the accepted first-reveal verdict and conflict records."""
    from exp.optimize.evaluation.variance.bootstrap_contracts import RevealAnswer

    return ToolSchema(
        name="submit_reveal_check",
        description="Submit whether the first reveals agree with the start.",
        input_schema=RevealAnswer.model_json_schema(),
    )


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
            "loop_protocol": AGENT_LOOP_PROTOCOL,
            "delivery_protocol": EVIDENCE_DELIVERY_PROTOCOL,
            "delivery_instruction": PAGE_INSTRUCTION,
            "delivery_tool": CONTINUE_EVIDENCE.model_dump(mode="json"),
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
                    author_answer_tool(),
                    review_answer_tool(),
                    RESEARCH_ANSWER,
                    reveal_answer_tool(),
                )
            ],
        }
    )
