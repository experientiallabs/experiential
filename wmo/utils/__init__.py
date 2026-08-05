"""Self-contained utilities WMO depends on but that carry no WMO domain concepts.

A module lands here when it is a general-purpose building block with its own tests and no
import back into the rest of `wmo` — today that is `wmo.utils.waterfall`, the stateless LLM
failover chain. Anything that knows about worlds, traces, evals, or optimizers belongs in the
domain package that owns it, not here.
"""
