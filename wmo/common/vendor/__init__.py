"""Vendored utilities WMO depends on that carry no WMO domain concepts.

A module lands here when it is a general-purpose building block with its own tests and no import
back into the rest of `wmo`. Today that is `wmo.common.vendor.waterfall`, the stateless LLM
failover chain. Code that knows about worlds, traces, evaluation, or optimizers belongs in its
domain package.
"""
