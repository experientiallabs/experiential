"""Gateway-executed web search for routes whose provider has no native search.

Kept import-light: ``exp.runtime.gateway.contracts`` imports the request
contract from :mod:`exp.runtime.gateway.web_search.contracts`, so this package
must never import the planner (which imports the gateway contracts back).
"""
