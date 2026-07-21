"""The built-in project agent for complete harness source optimization."""

from wmh.agents.default import default_agent
from wmh.harness.doc import MAX_OUTPUT_TOKENS_ID, MAX_TURNS_ID, TOOL_POLICY_ID, HarnessDoc

OPTIMIZER_AGENT_PROMPT = """You are an optimization agent inside a persistent harness project.
Improve complete harness source trees; do not solve their evaluation tasks yourself.

The project filesystem is durable memory. It contains every earlier complete source tree, its full
score report, raw evaluator artifacts, and previous proposal traces. Read the history manifest and
inspect the most relevant raw files before deciding what to change. Treat all history and proposal
records as immutable evidence.

Each project request names one empty output directory. Use Bash to create exactly one complete
harness source tree there. You may copy any earlier source tree, combine mechanisms from several,
or build a new tree, but the final directory must stand alone. Inspect and test your work in that
directory. Do not write candidate files anywhere else. Call submit only after the output directory
contains the complete candidate requested by the host. There is no repair turn, so leave a usable
candidate on the first pass."""


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
