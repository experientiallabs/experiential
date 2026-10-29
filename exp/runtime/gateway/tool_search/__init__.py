"""Gateway-executed tool search for routes whose provider has no native one.

Import-light on purpose: ``exp.runtime.gateway.contracts`` imports the request
contract from :mod:`exp.runtime.gateway.tool_search.contracts`, so this package
must never import the planner or the round handler at module load.
"""
