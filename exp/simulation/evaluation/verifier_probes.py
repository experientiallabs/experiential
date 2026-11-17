"""Bounded synthetic counterexamples to a candidate verifier's specified guarantees.

Perturbations are specification tests, not native witnesses or training examples. A false
pass falsifies predicate coverage; surviving this finite search does not certify correctness.
"""

from collections import deque
from copy import deepcopy
from itertools import islice
from typing import Literal

from pydantic import Field

from exp.common.core.artifacts import ContractModel, JsonObject, JsonValue, sha256_json
from exp.simulation.evaluation.tasks import TaskVerifier, grade


class VerifierCounterexample(ContractModel):
    """A reproducible synthetic change rejected by the independent specification.

    Attributes:
        path: Exact keys/indices inside final_state, relative to that state.
        replacement: Value substituted at that path, never claimed to be natively observed.
        outcome_sha256: Digest of the entire mutated observation.
    """

    path: tuple[str | int, ...]
    replacement: JsonValue
    outcome_sha256: str


class VerifierProbeReport(ContractModel):
    """Finite differential tests with explicit coverage and authority.

    Attributes:
        tested: Number of actual perturbations evaluated.
        truncated: More perturbations existed than the finite budget allowed.
        false_passes: Candidate passes that the independent specification rejects.
        authority: These cases are synthetic predicate checks, never native labels.
    """

    tested: int = Field(ge=0)
    truncated: bool
    false_passes: tuple[VerifierCounterexample, ...]
    authority: Literal["synthetic_specification_probe"] = "synthetic_specification_probe"


def _replacements(value: JsonValue) -> tuple[JsonValue, ...]:
    """Use small, deterministic value and cardinality changes without hidden expected answers."""
    if isinstance(value, bool):
        return (not value, int(value), None)
    if isinstance(value, str):
        return (value + "__probe__", None, 0)
    if isinstance(value, int):
        # Preserve numeric equality while changing type, without overflowing large integers.
        typed = (float(value),) if abs(value) <= 2**53 else ()
        return (value + 1, *typed, str(value), None, bool(value))
    if isinstance(value, float):
        typed = (int(value),) if value.is_integer() else ()
        return (value + 1, *typed, str(value), None)
    if isinstance(value, list):
        return ([], [*value, deepcopy(value[-1])], None, {}) if value else ([None], None, {})
    if isinstance(value, dict):
        return ({**value, "__unexpected_effect__": True}, None, [])
    return ("__unknown_changed__",)


def probe_verifier(
    candidate: TaskVerifier,
    specification: TaskVerifier,
    observation: JsonObject,
    *,
    maximum: int = 256,
) -> VerifierProbeReport:
    """Attempt to falsify a passing candidate using an independently supplied goal predicate."""
    if maximum < 1 or maximum > 4096:
        raise ValueError("verifier probes require a budget between 1 and 4096")
    if grade(specification, observation).status != "pass":
        raise ValueError("verifier probes require an independently passing baseline")
    state = observation.get("final_state")
    if not isinstance(state, dict):
        raise ValueError("verifier probes require observed final state")
    queue: deque[tuple[tuple[str | int, ...], JsonValue]] = deque([((), state)])
    tested, failures = 0, []
    truncated = False
    while queue and tested < maximum:
        path, original = queue.popleft()
        if isinstance(original, dict):
            queue.extend(
                ((*path, key), value) for key, value in islice(original.items(), maximum + 1)
            )
        elif isinstance(original, list):
            queue.extend(
                ((*path, i), value) for i, value in enumerate(islice(original, maximum + 1))
            )
        for replacement in _replacements(original):
            if tested == maximum:
                truncated = True
                break
            mutated = deepcopy(observation)
            target = mutated["final_state"]
            for part in path[:-1]:
                if isinstance(target, dict) and isinstance(part, str):
                    target = target[part]
                elif isinstance(target, list) and isinstance(part, int):
                    target = target[part]
                else:
                    raise ValueError("probe path no longer resolves")
            if path:
                key = path[-1]
                if isinstance(target, dict) and isinstance(key, str):
                    target[key] = replacement
                elif isinstance(target, list) and isinstance(key, int):
                    target[key] = replacement
                else:
                    raise ValueError("probe target no longer resolves")
            else:
                mutated["final_state"] = replacement
            tested += 1
            if (
                grade(specification, mutated).status == "fail"
                and grade(candidate, mutated).status == "pass"
            ):
                failures.append(
                    VerifierCounterexample(
                        path=path, replacement=replacement, outcome_sha256=sha256_json(mutated)
                    )
                )
    return VerifierProbeReport(
        tested=tested, truncated=truncated or bool(queue), false_passes=tuple(failures)
    )
