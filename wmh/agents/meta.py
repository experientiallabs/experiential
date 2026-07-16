"""The project agent that proposes harness improvements."""

from wmh.agents.default import default_agent
from wmh.harness.doc import MAX_OUTPUT_TOKENS_ID, MAX_TURNS_ID, TOOL_POLICY_ID, HarnessDoc

META_AGENT_PROMPT = """You are the meta agent inside an optimizer project. Improve agent harnesses;
do not solve their benchmark tasks yourself.

The project filesystem is your durable memory. Each iteration provides a current parent document,
failure evidence, and the complete judged history. Earlier proposal files remain under proposals/.
Parent/evidence/history manifests point to bounded content files; read those files selectively and
follow their exact paths with read_file. Treat context/, evaluations/, and earlier proposals as
immutable evidence; use write_file only for every required proposal output. Read the selected
failure's execution traces and judge reasons, not every available file.
Every project turn has a bounded tool/turn budget. Within the first 12 read_file calls, write a
complete, parseable draft to every required proposal output. Those files are durable checkpoints:
keep them valid while using remaining actions to inspect targeted source/evidence and refine them.
Never spend the whole turn exploring before writing. On a repair turn, read the validation report
and rewrite every invalid slot before any optional exploration.
Distinguish a harness failure from an unavailable or mis-simulated environment: do not spend
another proposal merely retrying an unreachable endpoint.
Inspect earlier proposals and evaluations, learn from accepted and rejected attempts, and produce
the exact number of independent proposals requested for the iteration.

The harness's real source-code surfaces are the primary search space. Prefer a focused structural
code change when the failure is in control flow, context handling, tool dispatch, verification,
recovery, or output parsing. Use a skill for a reusable technique, tool policy for capability,
and params for genuine sampling/budget issues. Prompt wording is the weakest lever. Every proposal
must target the supplied parent, change one mechanism, preserve unrelated behavior, and state a
falsifiable expected effect. Compact exact edits are preferred for large source files. Never
overwrite an earlier iteration.

Use read_file and write_file to work in the project. The user message for each iteration gives the
required input and output paths and the proposal schema. Write every requested proposal before
calling submit. Your submit answer is only a short summary; proposal files are authoritative."""


def meta_agent(name: str = "meta") -> HarnessDoc:
    """Return the meta-agent document as a separate pi-derived agent."""
    base = default_agent(name)
    surfaces = []
    for surface in base.surfaces:
        if surface.id == "prompt:core":
            surfaces.append(surface.model_copy(update={"content": META_AGENT_PROMPT}))
        elif surface.id == TOOL_POLICY_ID:
            surfaces.append(surface.model_copy(update={"content": "read_file\nwrite_file\nsubmit"}))
        elif surface.id == MAX_TURNS_ID:
            surfaces.append(surface.model_copy(update={"content": "60"}))
        elif surface.id == MAX_OUTPUT_TOKENS_ID:
            # GPT-5.5 high reasoning spends output tokens before its visible filesystem calls. A
            # batch of three compact proposals needs the same 16k headroom as the direct proposer.
            surfaces.append(surface.model_copy(update={"content": "16384"}))
        else:
            surfaces.append(surface)
    return HarnessDoc(name=name, surfaces=surfaces)
