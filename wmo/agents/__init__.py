"""Built-in agent definitions.

The optimizer, meta, and project agents moved to the agent-optimization repo with
the harness-search program; what ships here is the default agent the playground
and distillation seed from.
"""

from wmo.agents.default import default_agent

__all__ = ["default_agent"]
