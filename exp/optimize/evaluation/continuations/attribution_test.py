"""Every branch ending names its owner; only the actor's count as actor evidence."""

from exp.common.core.artifacts import FailureAttribution
from exp.optimize.evaluation.continuations.attribution import attributed, ending_owner
from exp.simulation.evaluation.tasks import VerifierGrade


def _grade(
    status: str, policy: str = "observed workflow satisfies the declared policy"
) -> VerifierGrade:
    """A grade with the given combined status and policy reason."""
    return VerifierGrade.model_validate(
        {
            "status": status,
            "reason": "r",
            "goal": {"status": status, "reason": "r"},
            "policy": {
                "status": "unassessable" if status == "unassessable" else "pass",
                "reason": policy,
            },
            "complete": status != "unassessable",
        }
    )


def test_each_execution_fault_is_owned_by_its_boundary() -> None:
    """Run 31's endings: simulator refusals, store errors and provider deadlines."""
    assert (
        ending_owner("InvalidPredictionError", "tool_dispatch_failed_known_effects") == "simulator"
    )
    assert ending_owner("ArtifactStoreError", "actor_model_dispatch") == "store"
    assert ending_owner("ProviderDeadlineExceeded", "actor_model_dispatch") == "provider"
    assert ending_owner("ProviderRetryableResponseError", "tool_dispatch_failed_known_effects") == (
        "provider"
    )
    assert ending_owner("InvisibleToolCallError", "actor_tool_authorization") == "actor"
    assert ending_owner("UnfinishedDispatchIntent", None) == "store"
    assert ending_owner("RuntimeError", "actor_model_dispatch") == "provider"
    assert ending_owner(None, None) is None


def test_a_branch_record_names_the_owner_of_its_result() -> None:
    """A known fail is the actor's even when interrupted; a pass carries no failure."""
    unassessable = _grade("unassessable")
    detail = attributed(
        None, unassessable, "InvalidPredictionError", "tool_dispatch_failed_known_effects"
    )
    assert detail is not None and detail.attribution == FailureAttribution.SIMULATOR
    failed = attributed(None, _grade("fail"), None, "verifier_grade")
    assert failed is not None and failed.attribution == FailureAttribution.ACTOR
    assert attributed(None, _grade("pass"), None, "verifier_grade") is None
    drift = _grade(
        "unassessable", "protected state changed by the simulator, not by an actor action: x"
    )
    masked = attributed(None, drift, None, "verifier_grade")
    assert masked is not None and masked.attribution == FailureAttribution.SIMULATOR
