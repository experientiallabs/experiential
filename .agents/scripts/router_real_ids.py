"""Canonical identifiers shared by the router reproduction analysis scripts."""


def canonical_tau2_scenario_id(scenario_id: str) -> str:
    """Convert runner-local ``domain:task`` IDs without corrupting canonical task text."""
    domain, separator, task_id = scenario_id.partition(":")
    if separator and "/" not in domain:
        return f"{domain}/{task_id}"
    return scenario_id
