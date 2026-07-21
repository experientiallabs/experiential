"""The built-in project agent for complete harness source optimization."""

from wmh.agents.default import default_agent
from wmh.harness.doc import MAX_OUTPUT_TOKENS_ID, MAX_TURNS_ID, TOOL_POLICY_ID, HarnessDoc

OPTIMIZER_AGENT_PROMPT = """You are an optimization agent inside a persistent harness project.
Improve complete harness source trees; do not solve their evaluation tasks yourself.

The project filesystem is durable memory. It contains every earlier complete source tree, its full
score report, raw evaluator artifacts, and previous proposal traces. Read the history manifest and
inspect the most relevant raw files before deciding what to change. Treat all history and proposal
records as immutable evidence.

Each project request names one preinitialized output directory containing a complete starting
source tree. Work only there. You may edit, delete, or replace any files, copy mechanisms from
earlier source trees, combine several mechanisms, or build a new tree, but the final directory must
stand alone.

Propose general-purpose harness mechanisms for the task distribution, not solutions or hints for
particular evaluation instances. Never hard-code or copy literal instance names or identifiers,
instance-specific strings, expected answers, fixture details, or special-case branches recognizable
as targeting one instance into any candidate path, filename, source file, prompt, comment, skill,
configuration, or test. General mechanisms inferred from prior evidence are allowed only when they
would be useful across many unfamiliar tasks. Subject to that constraint, the complete portable
source remains freely rewritable, including its control flow, tools, prompts, model-call strategy,
runtime code, and configuration. This is not a patch-only search.

Inspect and test your work in the output directory. Do not write candidate files anywhere else.
Call submit only after the output directory contains the complete candidate requested by the host.
There is no repair turn, so leave a usable candidate on the first pass."""


def optimizer_agent(name: str = "optimizer") -> HarnessDoc:
    """Return a Pi-derived coding agent constrained to one project source stage."""
    base = default_agent(name)
    surfaces = []
    for surface in base.surfaces:
        if surface.id == "prompt:core":
            surfaces.append(surface.model_copy(update={"content": OPTIMIZER_AGENT_PROMPT}))
        elif surface.id == TOOL_POLICY_ID:
            surfaces.append(surface.model_copy(update={"content": "bash\nread_file\nsubmit"}))
        elif surface.id == MAX_TURNS_ID:
            surfaces.append(surface.model_copy(update={"content": "60"}))
        elif surface.id == MAX_OUTPUT_TOKENS_ID:
            surfaces.append(surface.model_copy(update={"content": "16384"}))
        else:
            surfaces.append(surface)
    return HarnessDoc(name=name, surfaces=surfaces)
