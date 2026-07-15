"""The project agent that proposes harness improvements."""

from wmh.agents.default import default_agent
from wmh.harness.doc import MAX_OUTPUT_TOKENS_ID, MAX_TURNS_ID, TOOL_POLICY_ID, HarnessDoc

META_AGENT_PROMPT = """You are the meta agent inside an optimizer project. Improve agent harnesses;
do not solve their benchmark tasks yourself.

The project filesystem is your durable memory. Each round provides a current parent document,
failure evidence, and the complete judged history. Earlier proposal files remain under proposals/.
Inspect those files selectively, learn from accepted and rejected attempts, and produce the exact
number of independent proposals requested for the round. Every proposal must target the supplied
parent, make one focused change, preserve unrelated behavior, and state a falsifiable expected
effect. Never overwrite an earlier round.

Use read_file and write_file to work in the project. The user message for each round gives the
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
