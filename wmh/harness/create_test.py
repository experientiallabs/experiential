"""End-to-end create-loop tests: one provider plays all four roles, no network, no entropy.

Extends the closed-loop test pattern (`closed_loop_test.RoleProvider`) with a fourth role: the
meta-agent, keyed on `MUTATE_SYSTEM`'s distinctive phrase. The agent role is the FALLBACK after
the judge/meta/world-model markers, because a variant's system prompt is exactly what the search
rewrites — no marker on it is stable across generations. The fake judge echoes the gold assertions
verbatim from its prompt (the real judge is scored by text-matching those echoes) and passes or
fails a run based on the submitted answer, so seed and child scores can genuinely differ.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

import pytest
from llm_waterfall import ChatRequest, ChatResponse

from wmh.core.types import JsonObject
from wmh.engine.world_model import WorldModel
from wmh.evals.closed_loop import ClosedLoopReport, RolloutEvidence, TaskOutcome
from wmh.evals.gold import GoldJudge, GoldVerdict
from wmh.evals.tasks import TaskSpec
from wmh.harness import create as create_module
from wmh.harness.cost import (
    ProviderCostBinding,
    SearchComponentCostBinding,
    SearchComponentRole,
    SearchCostBinding,
)
from wmh.harness.create import (
    CreateResult,
    HarnessSearchCancelled,
    ProposalRecord,
    SearchCheckpoint,
    SearchProposalBatchWitness,
    _create_harness_nonpaid,
    load_search_checkpoint,
    load_search_proposal_batch_witness,
    search_harness,
    select_failure_cluster,
    write_search_checkpoint,
    write_search_proposal_batch_witness,
)
from wmh.harness.create import (
    create_harness as _production_create_harness,
)
from wmh.harness.delta import FailureSignature, GateRecord, HarnessDelta
from wmh.harness.doc import HarnessDoc
from wmh.harness.e2b_sandbox import SandboxCleanupError, SandboxUsage
from wmh.harness.mutate import parse_delta
from wmh.harness.proposer import (
    ProposalFailure,
)
from wmh.harness.proposer import (
    ProviderDeltaProposer as _ProductionProviderDeltaProposer,
)
from wmh.harness.runtime import Runtime, StopReason
from wmh.harness.scoring import (
    HarnessScoreArchive,
    HarnessScoreReport,
    ScoreArchiveTier,
    ScoreArchiveVisibility,
    ScoreCapabilities,
    ScoreRequest,
    ScoreRunHealth,
    ScoreRunHealthError,
    TaskScore,
)
from wmh.providers.base import Completion, Message, Provider, ProviderConfig, ProviderKind
from wmh.providers.process_worker import ProviderWorkerCleanupError
from wmh.providers.receipt import ProviderResponseIdentity
from wmh.retrieval import EmbeddingRetriever, HashingEmbedder
from wmh.tracking._testing import (
    synthetic_provider_cost_meter,
    synthetic_tariff_provenance,
)
from wmh.tracking.budget import (
    BudgetBreachError,
    BudgetExceededError,
    BudgetIntegrityError,
    BudgetPolicy,
    BudgetScope,
    UnpricedProviderUsageError,
    bind_budget_account,
    bootstrap_budget_ledger,
)
from wmh.tracking.rate_limit import ExternalDispatchRateIntegrityError

_CAREFUL_PROMPT = "You are a careful agent. Verify the state of the system before submitting."


# Every provider in this module is a deterministic in-memory fake. Keep the production surfaces
# fail closed while making that nonpaid declaration explicit at this test boundary.
class ProviderDeltaProposer(_ProductionProviderDeltaProposer):
    """Direct proposer over the deterministic in-memory provider used in this module."""

    requires_search_cost_binding = False


create_harness = _create_harness_nonpaid


def _meta_reply(parent: HarnessDoc, new_prompt: str) -> str:
    """A well-formed delta against `parent`, preconditioned on its actual prompt hash."""
    core = parent.surface("prompt:core")
    assert core is not None
    return json.dumps(
        {
            "expected_effect": "the failing tasks flip to pass",
            "preconditions": {"prompt:core": core.content_hash},
            "ops": [
                {
                    "op": "replace",
                    "surface_id": "prompt:core",
                    "content": new_prompt,
                    "rationale": "make the agent verify before submitting",
                }
            ],
        }
    )


class RoleProvider:
    """Plays agent, world model, gold judge, and meta-agent, keyed off the system prompt."""

    def __init__(
        self,
        *,
        meta_reply: str = "not json at all",
        judge_fn: Callable[[str], bool] | None = None,
    ) -> None:
        self.config = ProviderConfig(kind=ProviderKind.BEDROCK, model="m")
        self._meta_reply = meta_reply
        self.meta_users: list[str] = []  # every proposer prompt, for history assertions
        # Default: a run passes iff the agent submitted the verified answer.
        self._judge_fn = judge_fn if judge_fn is not None else lambda u: "done-verified" in u

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Completion:
        user = messages[-1].content
        if "grade whether an agent completed a task" in system:
            passed = self._judge_fn(user)
            results = [
                {"assertion": a, "passed": passed, "why": "x"} for a in _gold_assertions(user)
            ]
            return Completion(text=json.dumps({"assertions": results, "passed": passed}))
        if "meta-agent improving an agent harness" in system:
            self.meta_users.append(user)
            return Completion(text=self._meta_reply)
        if "You ARE the environment" in system:
            return Completion(text='{"output": "ok", "is_error": false}')
        # Fallback: the agent role. What it submits depends on the prompt the variant carries.
        if "careful agent" in system:
            answer = "done-verified"
        elif "broken agent" in system:
            answer = "done-broken"
        else:
            answer = "done"
        return Completion(text=json.dumps({"tool": "submit", "arguments": {"answer": answer}}))

    def complete_chat(self, request: ChatRequest) -> ChatResponse:
        del request
        return ChatResponse.model_validate(
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ]
            }
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self):  # noqa: ANN201 - test fake never calls it
        raise NotImplementedError


def _gold_assertions(user: str) -> list[str]:
    """The gold list the judge prompt carries, echoed back verbatim."""
    _, _, tail = user.partition("GOLD ASSERTIONS")
    return [line[2:] for line in tail.splitlines() if line.startswith("- ")]


def _wm(provider: RoleProvider) -> WorldModel:
    return WorldModel(provider, EmbeddingRetriever(HashingEmbedder(dim=16)))


def _tasks() -> list[TaskSpec]:
    return [TaskSpec(task_id="t1", instruction="answer it", gold=["the work was verified"])]


def _run(
    provider: RoleProvider,
    *,
    iterations: int = 1,
    k: int = 3,
    proposal_batch_size: int = 1,
    holdout: list[TaskSpec] | None = None,
    on_progress: Callable[[int, str, float, bool], None] | None = None,
    on_note: Callable[[str], None] | None = None,
    on_proposal: Callable[[ProposalRecord], None] | None = None,
    on_accept: Callable[[HarnessDoc, HarnessDelta, float], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> CreateResult:
    return create_harness(
        "winner",
        HarnessDoc.baseline("seed"),
        _tasks(),
        _wm(provider),
        provider,
        ProviderDeltaProposer(provider),
        GoldJudge(provider),
        iterations=iterations,
        proposal_batch_size=proposal_batch_size,
        k=k,
        holdout=holdout,
        on_progress=on_progress,
        on_note=on_note,
        on_proposal=on_proposal,
        on_accept=on_accept,
        should_cancel=should_cancel,
    )


class _NeutralScorer:
    """A ground-truth-like scorer with no world-model dependencies."""

    capabilities = ScoreCapabilities()
    configuration_id = "neutral-scorer-v1"
    default_attempts = 2
    task_ids = ("ground-truth-task",)

    def __init__(self) -> None:
        self.requests: list[tuple[str, ScoreRequest]] = []
        self.before_proposal_calls = 0

    def validate_candidate(self, candidate: HarnessDoc) -> str | None:
        return None

    def before_proposal_batch(self) -> None:
        self.before_proposal_calls += 1

    def score(self, candidate: HarnessDoc, *, request: ScoreRequest) -> HarnessScoreReport:
        self.requests.append((candidate.doc_hash, request))
        prompt = candidate.surface("prompt:core")
        passed = prompt is not None and "careful agent" in prompt.content
        score = 1.0 if passed else 0.0
        return HarnessScoreReport(
            evaluation_id=f"fake:{candidate.doc_hash}:{request.purpose}",
            label=candidate.name,
            score=score,
            secondary_score=score,
            attempts=self.default_attempts,
            run_health=ScoreRunHealth.VALID,
            per_task={
                "ground-truth-task": TaskScore(
                    task_id="ground-truth-task",
                    score=score,
                    secondary_score=score,
                    passed=passed,
                    description="repair the repository",
                    mechanisms=() if passed else ("missing verification",),
                    evidence="verifier reward and execution trace",
                )
            },
        )


class _EvidenceRecordingProposer:
    """Return one deterministic delta and retain the scorer evidence."""

    configuration_id = "evidence-proposer-v1"

    def __init__(self) -> None:
        self.evidence: list[str] = []
        self.archives: list[tuple[str, HarnessScoreArchive]] = []
        self.feedback: list[str] = []

    def propose_batch(
        self,
        parent: HarnessDoc,
        trigger: FailureSignature,
        evidence: str,
        *,
        history: list[HarnessDelta],
        count: int,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[HarnessDelta | ProposalFailure | None]:
        del history, should_cancel
        assert count == 1
        self.evidence.append(evidence)
        proposal = parse_delta(parent, trigger, _meta_reply(parent, _CAREFUL_PROMPT))
        assert proposal is not None
        return [proposal]

    def record_harness_evaluation(
        self,
        harness: HarnessDoc,
        *,
        archive: HarnessScoreArchive,
        should_cancel: Callable[[], bool] | None = None,
    ) -> None:
        del should_cancel
        self.archives.append((harness.doc_hash, archive))

    def record_evaluation(self, delta: HarnessDelta, *, stage: str, content: str) -> None:
        del delta, stage
        self.feedback.append(content)


class _HistoryProposer(_EvidenceRecordingProposer):
    """Derive unique deterministic proposals only from the complete judged history."""

    configuration_id = "history-proposer-v1"

    def __init__(self) -> None:
        super().__init__()
        self.history_ids: list[list[str]] = []
        self.resumed: tuple[int, list[str]] | None = None

    def resume_from_history(
        self,
        *,
        completed_iteration: int,
        proposal_records: list[ProposalRecord],
    ) -> None:
        self.resumed = (
            completed_iteration,
            [record.delta_id for record in proposal_records if record.delta_id is not None],
        )

    def propose_batch(
        self,
        parent: HarnessDoc,
        trigger: FailureSignature,
        evidence: str,
        *,
        history: list[HarnessDelta],
        count: int,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[HarnessDelta | ProposalFailure | None]:
        del evidence, should_cancel
        assert count == 1
        self.history_ids.append([delta.delta_id for delta in history])
        prompt = f"{_CAREFUL_PROMPT} Revision {len(history) + 1}."
        proposal = parse_delta(parent, trigger, _meta_reply(parent, prompt))
        assert proposal is not None
        return [proposal]


def _interrupt_after_iteration(
    checkpoints: list[SearchCheckpoint],
    *,
    iteration: int,
) -> Callable[[SearchCheckpoint], None]:
    """Capture committed checkpoints and simulate a process failure at one boundary."""

    def _callback(checkpoint: SearchCheckpoint) -> None:
        checkpoints.append(checkpoint)
        if checkpoint.completed_iteration == iteration:
            raise RuntimeError("simulated checkpoint interruption")

    return _callback


def _rehash_checkpoint_payload(payload: dict[str, object]) -> None:
    unsigned = {key: value for key, value in payload.items() if key != "payload_sha256"}
    canonical = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    payload["payload_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()


def _search_cost_binding(tmp_path: Path, *, label: str) -> SearchCostBinding:
    provider_config = ProviderConfig(
        kind=ProviderKind.BEDROCK,
        model="budgeted-model",
        region="test-region",
    )
    tariff_provenance = synthetic_tariff_provenance(provider_config)
    policy = BudgetPolicy(
        study_id=f"search-{label}",
        manifest_digest="sha256:" + hashlib.sha256(label.encode()).hexdigest(),
        hard_limit_nano_usd=1_000_000,
        phase_limits_nano_usd={"search": 1_000_000},
        meters={
            role: synthetic_provider_cost_meter(
                provider_config=provider_config,
                provenance=tariff_provenance,
                input_nano_usd_per_token=1,
                output_nano_usd_per_token=1,
            )
            for role in ("proposer", "scorer")
        },
    )
    authority = bootstrap_budget_ledger(tmp_path / f"{label}.sqlite3", policy)

    def component(
        role: SearchComponentRole,
        configuration_id: str,
    ) -> SearchComponentCostBinding:
        category = role.value
        account = authority.provider_account(
            scope=BudgetScope(
                phase="search",
                category=category,
                run_id="search-run",
            ),
            meter_id=category,
        )
        return SearchComponentCostBinding(
            role=role,
            configuration_id=configuration_id,
            scope_category=category,
            providers=(
                ProviderCostBinding(
                    component_configuration_id=configuration_id,
                    provider_config=provider_config,
                    response_identity=ProviderResponseIdentity(provider=ProviderKind.BEDROCK),
                    account=bind_budget_account(account),
                ),
            ),
        )

    return SearchCostBinding(
        declared_hard_limit_nano_usd=policy.hard_limit_nano_usd,
        policy=policy,
        ledger_identity=authority.ledger_identity,
        phase="search",
        run_id="search-run",
        proposer=component(SearchComponentRole.PROPOSER, _HistoryProposer.configuration_id),
        scorer=component(SearchComponentRole.SCORER, _NeutralScorer.configuration_id),
    )


class _CostBoundScorer(_NeutralScorer):
    """Neutral scorer exposing the exact path-free paid accounts it will use."""

    def __init__(self, binding: SearchComponentCostBinding) -> None:
        super().__init__()
        self.search_cost_binding = binding


class _CostBoundProposer(_HistoryProposer):
    """History proposer exposing the exact path-free paid accounts it will use."""

    def __init__(self, binding: SearchComponentCostBinding) -> None:
        super().__init__()
        self.search_cost_binding = binding


class _AuthorizingCostBoundProposer(_CostBoundProposer):
    """Record the complete search contract admitted before any external score call."""

    def __init__(self, binding: SearchComponentCostBinding) -> None:
        super().__init__(binding)
        self.authorizations: list[SearchCostBinding] = []

    def authorize_search_dispatch(self, binding: SearchCostBinding) -> None:
        self.authorizations.append(SearchCostBinding.model_validate(binding.model_dump()))


class _NeverCalledCostBoundScorer(_CostBoundScorer):
    """Expose cost state while making every executable scorer hook fail the test."""

    def validate_candidate(self, candidate: HarnessDoc) -> str | None:
        del candidate
        raise AssertionError("cost-bound scorer was called before omission rejection")

    def score(self, candidate: HarnessDoc, *, request: ScoreRequest) -> HarnessScoreReport:
        del candidate, request
        raise AssertionError("cost-bound scorer was called before omission rejection")


class _NeverCalledCostBoundProposer(_CostBoundProposer):
    """Expose cost state while making every executable proposer hook fail the test."""

    def propose_batch(
        self,
        parent: HarnessDoc,
        trigger: FailureSignature,
        evidence: str,
        *,
        history: list[HarnessDelta],
        count: int,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[HarnessDelta | ProposalFailure | None]:
        del parent, trigger, evidence, history, count, should_cancel
        raise AssertionError("cost-bound proposer was called before omission rejection")


class _NeverCalledBudgetMarkerScorer(_NeutralScorer):
    """Expose a lower-level budget marker without a component cost binding."""

    def __init__(self) -> None:
        super().__init__()
        self._budget_account = object()

    def validate_candidate(self, candidate: HarnessDoc) -> str | None:
        del candidate
        raise AssertionError("budget-marked scorer was called before omission rejection")

    def score(self, candidate: HarnessDoc, *, request: ScoreRequest) -> HarnessScoreReport:
        del candidate, request
        raise AssertionError("budget-marked scorer was called before omission rejection")


class _PaidDispatchScorer(_NeverCalledBudgetMarkerScorer):
    """Declare a paid dispatch even before its exact account is attached."""

    requires_search_cost_binding = True

    def __init__(self) -> None:
        _NeutralScorer.__init__(self)


@pytest.mark.parametrize("bound_role", ["proposer", "scorer", "holdout_scorer"])
def test_cost_bound_component_requires_top_level_binding_before_component_calls(
    tmp_path: Path,
    bound_role: str,
) -> None:
    binding = _search_cost_binding(tmp_path, label=f"omitted-{bound_role}")
    proposer: _HistoryProposer = _HistoryProposer()
    scorer: _NeutralScorer = _NeutralScorer()
    holdout: _NeutralScorer | None = None
    if bound_role == "proposer":
        proposer = _NeverCalledCostBoundProposer(binding.proposer)
    elif bound_role == "scorer":
        scorer = _NeverCalledCostBoundScorer(binding.scorer)
    else:
        holdout_binding = binding.scorer.model_copy(
            update={"role": SearchComponentRole.HOLDOUT_SCORER}
        )
        holdout = _NeverCalledCostBoundScorer(holdout_binding)

    with pytest.raises(ValueError, match=f"{bound_role} exposes budgeted cost state"):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            scorer,
            proposer,
            iterations=0,
            screen_proposals=False,
            holdout_scorer=holdout,
            confirm_narrow_vetoes=False,
        )

    assert scorer.requests == []


def test_budget_account_marker_requires_top_level_binding_before_component_calls() -> None:
    scorer = _NeverCalledBudgetMarkerScorer()

    with pytest.raises(ValueError, match="scorer exposes budgeted cost state"):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            scorer,
            _HistoryProposer(),
            iterations=0,
            screen_proposals=False,
            confirm_narrow_vetoes=False,
        )

    assert scorer.requests == []


def test_paid_dispatch_marker_requires_top_level_binding_before_component_calls() -> None:
    scorer = _PaidDispatchScorer()

    with pytest.raises(ValueError, match="scorer exposes budgeted cost state"):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            scorer,
            _HistoryProposer(),
            iterations=0,
            screen_proposals=False,
            confirm_narrow_vetoes=False,
        )

    assert scorer.requests == []


@pytest.mark.parametrize(
    "error",
    [
        BudgetIntegrityError("ledger drift"),
        BudgetBreachError("accounting breach"),
        BudgetExceededError("hard budget exhausted"),
        UnpricedProviderUsageError("tariff missing a billable dimension"),
        ExternalDispatchRateIntegrityError("rate authority drift"),
        SandboxCleanupError("sandbox cleanup unproved"),
        ProviderWorkerCleanupError("provider cleanup unproved"),
    ],
)
def test_search_harness_propagates_safety_terminal_proposal_errors(error: Exception) -> None:
    class _SafetyTerminalProposer(_HistoryProposer):
        def propose_batch(
            self,
            parent: HarnessDoc,
            trigger: FailureSignature,
            evidence: str,
            *,
            history: list[HarnessDelta],
            count: int,
            should_cancel: Callable[[], bool] | None = None,
        ) -> list[HarnessDelta | ProposalFailure | None]:
            del parent, trigger, evidence, history, count, should_cancel
            raise error

    scorer = _NeutralScorer()

    with pytest.raises(type(error), match=str(error)):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            scorer,
            _SafetyTerminalProposer(),
            iterations=1,
            screen_proposals=False,
            confirm_narrow_vetoes=False,
        )

    assert len(scorer.requests) == 1


def test_search_harness_restores_nested_safety_type_from_score_archive_write() -> None:
    terminal = BudgetIntegrityError("wrapped archive ledger drift")
    wrapped = RuntimeError("fresh project sandbox recovery failed")
    wrapped.__cause__ = terminal

    class _TerminalArchiveProposer(_HistoryProposer):
        def record_harness_evaluation(
            self,
            harness: HarnessDoc,
            *,
            archive: HarnessScoreArchive,
            should_cancel: Callable[[], bool] | None = None,
        ) -> None:
            del harness, archive, should_cancel
            raise wrapped

    with pytest.raises(BudgetIntegrityError, match="wrapped archive ledger drift") as raised:
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            _NeutralScorer(),
            _TerminalArchiveProposer(),
            iterations=0,
            screen_proposals=False,
            confirm_narrow_vetoes=False,
        )

    assert raised.value is terminal


def test_search_harness_restores_nested_budget_exhaustion_from_feedback_write() -> None:
    terminal = BudgetExceededError("wrapped feedback budget exhausted")
    wrapped = RuntimeError("fresh project sandbox recovery failed")
    wrapped.__cause__ = terminal

    class _TerminalFeedbackProposer(_EvidenceRecordingProposer):
        def record_evaluation(self, delta: HarnessDelta, *, stage: str, content: str) -> None:
            del delta, stage, content
            raise wrapped

    with pytest.raises(BudgetExceededError, match="wrapped feedback budget exhausted") as raised:
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            _NeutralScorer(),
            _TerminalFeedbackProposer(),
            iterations=1,
            screen_proposals=False,
            confirm_narrow_vetoes=False,
        )

    assert raised.value is terminal


def test_create_harness_rejects_raw_unbudgeted_paid_path_before_dispatch() -> None:
    provider = RoleProvider()
    proposer = _ProductionProviderDeltaProposer(provider)

    with pytest.raises(ValueError, match="cannot dispatch raw paid providers"):
        _production_create_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            _tasks(),
            _wm(provider),
            provider,
            proposer,
            GoldJudge(provider),
        )

    assert provider.meta_users == []


def test_budgeted_search_rejects_unbound_components_before_scoring(tmp_path: Path) -> None:
    scorer = _NeutralScorer()
    with pytest.raises(ValueError, match="proposer.search_cost_binding"):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            scorer,
            _HistoryProposer(),
            iterations=0,
            screen_proposals=False,
            confirm_narrow_vetoes=False,
            cost_binding=_search_cost_binding(tmp_path, label="unbound"),
        )

    assert scorer.requests == []


def test_search_authorizes_deferred_resources_only_after_all_pure_validation(
    tmp_path: Path,
) -> None:
    binding = _search_cost_binding(tmp_path, label="deferred-authorization-invalid")
    proposer = _AuthorizingCostBoundProposer(binding.proposer)
    scorer = _CostBoundScorer(binding.scorer)

    with pytest.raises(ValueError, match="cannot evaluate task subsets"):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            scorer,
            proposer,
            iterations=0,
            cost_binding=binding,
        )

    assert proposer.authorizations == []
    assert scorer.requests == []


def test_search_authorizes_deferred_resources_before_first_score(tmp_path: Path) -> None:
    binding = _search_cost_binding(tmp_path, label="deferred-authorization-valid")
    proposer = _AuthorizingCostBoundProposer(binding.proposer)

    class _AuthorizationCheckingScorer(_CostBoundScorer):
        def score(self, candidate: HarnessDoc, *, request: ScoreRequest) -> HarnessScoreReport:
            assert proposer.authorizations == [binding]
            return super().score(candidate, request=request)

    scorer = _AuthorizationCheckingScorer(binding.scorer)
    search_harness(
        "winner",
        HarnessDoc.baseline("seed"),
        scorer,
        proposer,
        iterations=0,
        screen_proposals=False,
        confirm_narrow_vetoes=False,
        cost_binding=binding,
    )

    assert proposer.authorizations == [binding]
    assert len(scorer.requests) == 1


def test_search_result_retains_full_path_free_cost_contract_without_checkpointing(
    tmp_path: Path,
) -> None:
    binding = _search_cost_binding(tmp_path, label="result-provenance")

    result = search_harness(
        "winner",
        HarnessDoc.baseline("seed"),
        _CostBoundScorer(binding.scorer),
        _CostBoundProposer(binding.proposer),
        iterations=0,
        screen_proposals=False,
        confirm_narrow_vetoes=False,
        cost_binding=binding,
    )

    retained = result.search_cost_binding
    assert retained is not None
    assert retained == binding
    assert retained.digest == binding.digest
    assert str(tmp_path) not in retained.model_dump_json()


def test_search_checkpoint_binds_cost_provenance_and_rejects_drift(
    tmp_path: Path,
) -> None:
    first = _search_cost_binding(tmp_path, label="first")
    checkpoints: list[SearchCheckpoint] = []
    with pytest.raises(RuntimeError, match="simulated checkpoint interruption"):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            _CostBoundScorer(first.scorer),
            _CostBoundProposer(first.proposer),
            iterations=0,
            screen_proposals=False,
            confirm_narrow_vetoes=False,
            cost_binding=first,
            on_checkpoint=_interrupt_after_iteration(checkpoints, iteration=0),
        )

    assert checkpoints[-1].configuration.search_cost_binding_digest == first.digest

    second = _search_cost_binding(tmp_path, label="second")
    scorer = _CostBoundScorer(second.scorer)
    with pytest.raises(
        ValueError,
        match="search checkpoint configuration drift.*search_cost_binding_digest",
    ):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            scorer,
            _CostBoundProposer(second.proposer),
            iterations=0,
            screen_proposals=False,
            confirm_narrow_vetoes=False,
            cost_binding=second,
            resume_from=checkpoints[-1],
        )

    assert scorer.requests == []


def _rehash_witness_payload(payload: dict[str, object]) -> None:
    """Recompute a proposal transaction checksum after an adversarial test edit."""
    unsigned = {key: value for key, value in payload.items() if key != "payload_sha256"}
    canonical = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    payload["payload_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()


def test_search_harness_resumes_exactly_from_a_committed_iteration() -> None:
    seed = HarnessDoc.baseline("seed")
    checkpoints: list[SearchCheckpoint] = []
    interrupted_scorer = _NeutralScorer()

    with pytest.raises(RuntimeError, match="simulated checkpoint interruption"):
        search_harness(
            "winner",
            seed,
            interrupted_scorer,
            _HistoryProposer(),
            iterations=3,
            screen_proposals=False,
            confirm_narrow_vetoes=False,
            on_checkpoint=_interrupt_after_iteration(checkpoints, iteration=1),
        )

    committed = checkpoints[-1]
    assert committed.completed_iteration == 1
    assert committed.champion_doc_hash != seed.doc_hash
    assert set(committed.docs) == set(committed.reports)
    assert len(committed.docs) == 2
    assert len(committed.archive.deltas) == 1
    assert len(committed.proposal_records) == 1
    assert committed.failure_cluster_expansions[0].count == 1
    assert [request.purpose for _, request in interrupted_scorer.requests] == ["seed", "full"]

    resumed_scorer = _NeutralScorer()
    resumed_proposer = _HistoryProposer()
    resumed = search_harness(
        "winner",
        seed,
        resumed_scorer,
        resumed_proposer,
        iterations=3,
        screen_proposals=False,
        confirm_narrow_vetoes=False,
        resume_from=committed,
    )
    uninterrupted_proposer = _HistoryProposer()
    uninterrupted = search_harness(
        "winner",
        seed,
        _NeutralScorer(),
        uninterrupted_proposer,
        iterations=3,
        screen_proposals=False,
        confirm_narrow_vetoes=False,
    )

    assert resumed == uninterrupted
    assert [request.purpose for _, request in resumed_scorer.requests] == ["full", "full"]
    assert resumed_proposer.resumed == (1, [committed.archive.deltas[0].delta_id])
    assert [len(history) for history in resumed_proposer.history_ids] == [1, 2]
    assert [len(history) for history in uninterrupted_proposer.history_ids] == [0, 1, 2]


def test_search_harness_replays_a_witnessed_batch_without_recalling_proposer() -> None:
    """A crash after proposal return cannot resample a different candidate on resume."""

    class StatefulProposer(_HistoryProposer):
        configuration_id = "stateful-witness-proposer-v1"
        durable_state_required = True

        def __init__(self) -> None:
            super().__init__()
            self.calls = 0
            self.state = 0
            self.restored: list[int] = []

        def export_search_state(self) -> JsonObject:
            return {"state": self.state}

        def restore_search_state(self, raw_state: JsonObject) -> None:
            state = raw_state["state"]
            assert isinstance(state, int)
            self.state = state
            self.restored.append(state)

        def propose_batch(
            self,
            parent: HarnessDoc,
            trigger: FailureSignature,
            evidence: str,
            *,
            history: list[HarnessDelta],
            count: int,
            should_cancel: Callable[[], bool] | None = None,
        ) -> list[HarnessDelta | ProposalFailure | None]:
            self.calls += 1
            self.state += 1
            return super().propose_batch(
                parent,
                trigger,
                evidence,
                history=history,
                count=count,
                should_cancel=should_cancel,
            )

    seed = HarnessDoc.baseline("seed")
    checkpoints: list[SearchCheckpoint] = []
    preparations: list[SearchProposalBatchWitness] = []
    witnesses: list[SearchProposalBatchWitness] = []
    first_proposer = StatefulProposer()

    def _interrupt_after_witness(witness: SearchProposalBatchWitness) -> None:
        witnesses.append(witness)
        raise RuntimeError("simulated post-proposal process failure")

    with pytest.raises(RuntimeError, match="post-proposal process failure"):
        search_harness(
            "winner",
            seed,
            _NeutralScorer(),
            first_proposer,
            iterations=1,
            screen_proposals=False,
            confirm_narrow_vetoes=False,
            on_checkpoint=checkpoints.append,
            on_proposal_batch_prepare=preparations.append,
            on_proposal_batch_witness=_interrupt_after_witness,
        )

    assert first_proposer.calls == 1
    assert [checkpoint.completed_iteration for checkpoint in checkpoints] == [0]
    [witness] = witnesses
    [preparation] = preparations
    assert preparation.phase == "prepared"
    assert witness.phase == "completed"
    assert witness.prepared_payload_sha256 == preparation.payload_sha256
    assert witness.iteration == 1
    assert witness.prior_checkpoint_payload_sha256 == checkpoints[0].payload_sha256
    assert witness.proposer_state_before == {"state": 0}
    assert witness.proposer_state_after == {"state": 1}

    resumed_proposer = StatefulProposer()
    resumed_scorer = _NeutralScorer()
    resumed_checkpoints: list[SearchCheckpoint] = []
    resumed = search_harness(
        "winner",
        seed,
        resumed_scorer,
        resumed_proposer,
        iterations=1,
        screen_proposals=False,
        confirm_narrow_vetoes=False,
        resume_from=checkpoints[0],
        resume_proposal_batch_witness=witness,
        on_checkpoint=resumed_checkpoints.append,
        on_proposal_batch_prepare=lambda prepared: (_ for _ in ()).throw(
            AssertionError("a replayed witness must not be prepared again")
        ),
        on_proposal_batch_witness=lambda replayed: (_ for _ in ()).throw(
            AssertionError("a replayed witness must not be republished")
        ),
    )

    uninterrupted_proposer = StatefulProposer()
    uninterrupted = search_harness(
        "winner",
        seed,
        _NeutralScorer(),
        uninterrupted_proposer,
        iterations=1,
        screen_proposals=False,
        confirm_narrow_vetoes=False,
        on_checkpoint=lambda checkpoint: None,
        on_proposal_batch_prepare=lambda prepared: None,
        on_proposal_batch_witness=lambda witnessed: None,
    )
    assert resumed == uninterrupted
    assert resumed_proposer.calls == 0
    assert resumed_proposer.state == 1
    assert resumed_proposer.restored == [0, 1]
    assert [request.purpose for _, request in resumed_scorer.requests] == ["full"]
    assert resumed_checkpoints[-1].proposal_batch_witness_digests == (witness.payload_sha256,)


def test_replay_preserves_pre_batch_callback_order_in_witnessed_proposer_state() -> None:
    """The pre-batch hook must run before both snapshots, not twice around a replay."""

    class StatefulProposer(_HistoryProposer):
        configuration_id = "callback-ordered-stateful-proposer-v1"
        durable_state_required = True

        def __init__(self) -> None:
            super().__init__()
            self.state = 0
            self.calls = 0

        def export_search_state(self) -> JsonObject:
            return {"state": self.state}

        def restore_search_state(self, raw_state: JsonObject) -> None:
            state = raw_state["state"]
            assert isinstance(state, int)
            self.state = state

        def propose_batch(
            self,
            parent: HarnessDoc,
            trigger: FailureSignature,
            evidence: str,
            *,
            history: list[HarnessDelta],
            count: int,
            should_cancel: Callable[[], bool] | None = None,
        ) -> list[HarnessDelta | ProposalFailure | None]:
            self.calls += 1
            self.state += 1
            return super().propose_batch(
                parent,
                trigger,
                evidence,
                history=history,
                count=count,
                should_cancel=should_cancel,
            )

    class StatefulScorer(_NeutralScorer):
        configuration_id = "callback-ordered-stateful-scorer-v1"

        def __init__(self, proposer: StatefulProposer) -> None:
            super().__init__()
            self._proposer = proposer

        def before_proposal_batch(self) -> None:
            super().before_proposal_batch()
            self._proposer.state += 10

    checkpoints: list[SearchCheckpoint] = []
    witnesses: list[SearchProposalBatchWitness] = []
    first_proposer = StatefulProposer()
    with pytest.raises(RuntimeError, match="stop after witness"):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            StatefulScorer(first_proposer),
            first_proposer,
            iterations=1,
            screen_proposals=False,
            confirm_narrow_vetoes=False,
            on_checkpoint=checkpoints.append,
            on_proposal_batch_prepare=lambda prepared: None,
            on_proposal_batch_witness=lambda witness: (
                witnesses.append(witness),
                (_ for _ in ()).throw(RuntimeError("stop after witness")),
            ),
        )

    [witness] = witnesses
    assert witness.proposer_state_before == {"state": 10}
    assert witness.proposer_state_after == {"state": 11}

    resumed_proposer = StatefulProposer()
    resumed = search_harness(
        "winner",
        HarnessDoc.baseline("seed"),
        StatefulScorer(resumed_proposer),
        resumed_proposer,
        iterations=1,
        screen_proposals=False,
        confirm_narrow_vetoes=False,
        resume_from=checkpoints[0],
        resume_proposal_batch_witness=witness,
    )

    assert resumed.iterations == 1
    assert resumed_proposer.calls == 0
    assert resumed_proposer.state == 11


def test_search_harness_witnesses_proposer_failures_before_any_candidate_score() -> None:
    class FailingStatefulProposer(_HistoryProposer):
        configuration_id = "failing-stateful-witness-proposer-v1"
        durable_state_required = True

        def __init__(self) -> None:
            super().__init__()
            self.state = 0

        def export_search_state(self) -> JsonObject:
            return {"state": self.state}

        def restore_search_state(self, raw_state: JsonObject) -> None:
            state = raw_state["state"]
            assert isinstance(state, int)
            self.state = state

        def propose_batch(
            self,
            parent: HarnessDoc,
            trigger: FailureSignature,
            evidence: str,
            *,
            history: list[HarnessDelta],
            count: int,
            should_cancel: Callable[[], bool] | None = None,
        ) -> list[HarnessDelta | ProposalFailure | None]:
            del parent, trigger, evidence, history, count, should_cancel
            self.state += 1
            raise RuntimeError("provider returned an unusable response")

    checkpoints: list[SearchCheckpoint] = []
    preparations: list[SearchProposalBatchWitness] = []
    witnesses: list[SearchProposalBatchWitness] = []
    scorer = _NeutralScorer()

    def _stop(witness: SearchProposalBatchWitness) -> None:
        witnesses.append(witness)
        raise RuntimeError("stop after durable witness")

    with pytest.raises(RuntimeError, match="stop after durable witness"):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            scorer,
            FailingStatefulProposer(),
            iterations=1,
            screen_proposals=False,
            confirm_narrow_vetoes=False,
            on_checkpoint=checkpoints.append,
            on_proposal_batch_prepare=preparations.append,
            on_proposal_batch_witness=_stop,
        )

    assert [request.purpose for _, request in scorer.requests] == ["seed"]
    [witness] = witnesses
    [preparation] = preparations
    assert witness.prepared_payload_sha256 == preparation.payload_sha256
    assert witness.proposer_state_after == {"state": 1}
    assert witness.slots[0].kind == "failure"
    assert witness.slots[0].reason == "provider returned an unusable response"


def test_search_harness_witnesses_and_replays_every_proposal_slot_kind() -> None:
    class MixedSlotProposer(_HistoryProposer):
        configuration_id = "mixed-slot-witness-proposer-v1"

        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def propose_batch(
            self,
            parent: HarnessDoc,
            trigger: FailureSignature,
            evidence: str,
            *,
            history: list[HarnessDelta],
            count: int,
            should_cancel: Callable[[], bool] | None = None,
        ) -> list[HarnessDelta | ProposalFailure | None]:
            self.calls += 1
            assert count == 3
            [delta] = super().propose_batch(
                parent,
                trigger,
                evidence,
                history=history,
                count=1,
                should_cancel=should_cancel,
            )
            return [delta, ProposalFailure(reason="provider failure"), None]

    checkpoints: list[SearchCheckpoint] = []
    witnesses: list[SearchProposalBatchWitness] = []
    with pytest.raises(RuntimeError, match="stop after witness"):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            _NeutralScorer(),
            MixedSlotProposer(),
            iterations=1,
            proposal_batch_size=3,
            screen_proposals=False,
            confirm_narrow_vetoes=False,
            on_checkpoint=checkpoints.append,
            on_proposal_batch_prepare=lambda prepared: None,
            on_proposal_batch_witness=lambda witness: (
                witnesses.append(witness),
                (_ for _ in ()).throw(RuntimeError("stop after witness")),
            ),
        )
    [witness] = witnesses
    assert [slot.kind for slot in witness.slots] == ["delta", "failure", "none"]

    resumed_proposer = MixedSlotProposer()
    resumed = search_harness(
        "winner",
        HarnessDoc.baseline("seed"),
        _NeutralScorer(),
        resumed_proposer,
        iterations=1,
        proposal_batch_size=3,
        screen_proposals=False,
        confirm_narrow_vetoes=False,
        resume_from=checkpoints[0],
        resume_proposal_batch_witness=witness,
    )
    assert resumed_proposer.calls == 0
    assert [record.outcome for record in resumed.proposal_records] == [
        "scored",
        "proposer_error",
        "unusable",
    ]


def test_search_harness_rejects_a_drifted_or_unbound_proposal_witness_before_scoring() -> None:
    checkpoints: list[SearchCheckpoint] = []
    preparations: list[SearchProposalBatchWitness] = []
    witnesses: list[SearchProposalBatchWitness] = []

    with pytest.raises(RuntimeError, match="stop after witness"):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            _NeutralScorer(),
            _HistoryProposer(),
            iterations=2,
            screen_proposals=False,
            confirm_narrow_vetoes=False,
            on_checkpoint=checkpoints.append,
            on_proposal_batch_prepare=preparations.append,
            on_proposal_batch_witness=lambda witness: (
                witnesses.append(witness),
                (_ for _ in ()).throw(RuntimeError("stop after witness")),
            ),
        )

    payload = witnesses[0].model_dump(mode="json")
    payload["prior_checkpoint_payload_sha256"] = "0" * 64
    unsigned = {key: value for key, value in payload.items() if key != "payload_sha256"}
    payload["payload_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    drifted = SearchProposalBatchWitness.model_validate(payload)
    scorer = _NeutralScorer()
    with pytest.raises(ValueError, match="prior checkpoint"):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            scorer,
            _HistoryProposer(),
            iterations=2,
            screen_proposals=False,
            confirm_narrow_vetoes=False,
            resume_from=checkpoints[0],
            resume_proposal_batch_witness=drifted,
        )
    assert scorer.requests == []

    with pytest.raises(ValueError, match="requires resume_from"):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            _NeutralScorer(),
            _HistoryProposer(),
            iterations=1,
            screen_proposals=False,
            confirm_narrow_vetoes=False,
            resume_proposal_batch_witness=witnesses[0],
        )


def test_proposal_batch_witness_file_is_atomic_and_rejects_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoints: list[SearchCheckpoint] = []
    preparations: list[SearchProposalBatchWitness] = []
    witnesses: list[SearchProposalBatchWitness] = []
    with pytest.raises(RuntimeError, match="stop after witness"):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            _NeutralScorer(),
            _HistoryProposer(),
            iterations=1,
            screen_proposals=False,
            confirm_narrow_vetoes=False,
            on_checkpoint=checkpoints.append,
            on_proposal_batch_prepare=preparations.append,
            on_proposal_batch_witness=lambda witness: (
                witnesses.append(witness),
                (_ for _ in ()).throw(RuntimeError("stop after witness")),
            ),
        )
    path = tmp_path / "proposal-witness.json"
    write_search_proposal_batch_witness(path, witnesses[0])
    assert load_search_proposal_batch_witness(path) == witnesses[0]

    corrupt = json.loads(path.read_text())
    corrupt["iteration"] = 2
    path.write_text(json.dumps(corrupt))
    with pytest.raises(ValueError, match="witness payload digest"):
        load_search_proposal_batch_witness(path)

    write_search_proposal_batch_witness(path, witnesses[0])

    def _fail_replace(source: Path, target: Path) -> None:
        del source, target
        raise OSError("simulated atomic replacement failure")

    monkeypatch.setattr(create_module.os, "replace", _fail_replace)
    with pytest.raises(OSError, match="simulated atomic replacement failure"):
        write_search_proposal_batch_witness(path, witnesses[0])
    assert load_search_proposal_batch_witness(path) == witnesses[0]


def test_search_harness_fails_closed_on_an_unwitnessed_prepared_proposer_call() -> None:
    """A prepared call is ambiguous after restart until its completed witness exists."""
    checkpoints: list[SearchCheckpoint] = []
    preparations: list[SearchProposalBatchWitness] = []

    def _stop_after_prepare(prepared: SearchProposalBatchWitness) -> None:
        preparations.append(prepared)
        raise RuntimeError("simulated crash after preparing proposer call")

    proposer = _HistoryProposer()
    with pytest.raises(RuntimeError, match="crash after preparing"):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            _NeutralScorer(),
            proposer,
            iterations=1,
            screen_proposals=False,
            confirm_narrow_vetoes=False,
            on_checkpoint=checkpoints.append,
            on_proposal_batch_prepare=_stop_after_prepare,
            on_proposal_batch_witness=lambda witnessed: None,
        )

    assert proposer.history_ids == []
    [prepared] = preparations
    assert prepared.phase == "prepared"
    scorer = _NeutralScorer()
    resumed_proposer = _HistoryProposer()
    with pytest.raises(RuntimeError, match="ambiguous prepared proposal batch"):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            scorer,
            resumed_proposer,
            iterations=1,
            screen_proposals=False,
            confirm_narrow_vetoes=False,
            resume_from=checkpoints[0],
            resume_proposal_batch_witness=prepared,
        )
    assert scorer.requests == []
    assert scorer.before_proposal_calls == 0
    assert resumed_proposer.history_ids == []


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("iteration", 2, "iteration"),
        ("parent_doc_hash", "drifted-parent", "parent document"),
        ("parent_execution_hash", "drifted-execution", "parent execution"),
        ("evidence", "drifted evidence", "evidence"),
        ("count", 2, "slot count"),
        ("proposer_state_before", {"drifted": True}, "proposer state before"),
    ],
)
def test_search_harness_rejects_rehashed_witness_binding_drift_before_scoring(
    field: str,
    replacement: object,
    message: str,
) -> None:
    checkpoints: list[SearchCheckpoint] = []
    witnesses: list[SearchProposalBatchWitness] = []
    with pytest.raises(RuntimeError, match="stop after witness"):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            _NeutralScorer(),
            _HistoryProposer(),
            iterations=2,
            screen_proposals=False,
            confirm_narrow_vetoes=False,
            on_checkpoint=checkpoints.append,
            on_proposal_batch_prepare=lambda prepared: None,
            on_proposal_batch_witness=lambda witness: (
                witnesses.append(witness),
                (_ for _ in ()).throw(RuntimeError("stop after witness")),
            ),
        )
    payload = witnesses[0].model_dump(mode="json")
    payload[field] = replacement
    if field == "count":
        payload["slots"] = [*payload["slots"], payload["slots"][0]]
    _rehash_witness_payload(payload)
    drifted = SearchProposalBatchWitness.model_validate(payload)
    scorer = _NeutralScorer()
    with pytest.raises(ValueError, match=message):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            scorer,
            _HistoryProposer(),
            iterations=2,
            screen_proposals=False,
            confirm_narrow_vetoes=False,
            resume_from=checkpoints[0],
            resume_proposal_batch_witness=drifted,
        )
    assert scorer.requests == []
    assert scorer.before_proposal_calls == 0


def test_search_harness_rejects_history_drift_and_replays_witnessed_slots_exactly() -> None:
    checkpoints: list[SearchCheckpoint] = []
    witnesses: list[SearchProposalBatchWitness] = []
    with pytest.raises(RuntimeError, match="stop after witness"):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            _NeutralScorer(),
            _HistoryProposer(),
            iterations=1,
            screen_proposals=False,
            confirm_narrow_vetoes=False,
            on_checkpoint=checkpoints.append,
            on_proposal_batch_prepare=lambda prepared: None,
            on_proposal_batch_witness=lambda witness: (
                witnesses.append(witness),
                (_ for _ in ()).throw(RuntimeError("stop after witness")),
            ),
        )

    history_payload = witnesses[0].model_dump(mode="json")
    history_payload["history"] = [history_payload["slots"][0]["delta"]]
    _rehash_witness_payload(history_payload)
    history_drifted = SearchProposalBatchWitness.model_validate(history_payload)
    with pytest.raises(ValueError, match="full proposal history"):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            _NeutralScorer(),
            _HistoryProposer(),
            iterations=1,
            screen_proposals=False,
            confirm_narrow_vetoes=False,
            resume_from=checkpoints[0],
            resume_proposal_batch_witness=history_drifted,
        )

    slot_payload = witnesses[0].model_dump(mode="json")
    slot_payload["slots"][0]["delta"]["expected_effect"] = "mutated witnessed output"
    _rehash_witness_payload(slot_payload)
    slot_drifted = SearchProposalBatchWitness.model_validate(slot_payload)
    resumed = search_harness(
        "winner",
        HarnessDoc.baseline("seed"),
        _NeutralScorer(),
        _HistoryProposer(),
        iterations=1,
        screen_proposals=False,
        confirm_narrow_vetoes=False,
        resume_from=checkpoints[0],
        resume_proposal_batch_witness=slot_drifted,
    )
    assert resumed.proposal_records[0].expected_effect == "mutated witnessed output"


def test_completed_proposal_witness_precedes_cancellation_and_is_observer_detached() -> None:
    checkpoints: list[SearchCheckpoint] = []
    witnesses: list[SearchProposalBatchWitness] = []
    cancel_now = False

    def _should_cancel() -> bool:
        return cancel_now

    def _capture_and_mutate(witness: SearchProposalBatchWitness) -> None:
        nonlocal cancel_now
        witnesses.append(witness.model_copy(deep=True))
        delta = witness.slots[0].delta
        assert delta is not None
        delta.expected_effect = "observer mutation"
        cancel_now = True

    with pytest.raises(HarnessSearchCancelled):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            _NeutralScorer(),
            _HistoryProposer(),
            iterations=1,
            screen_proposals=False,
            confirm_narrow_vetoes=False,
            on_checkpoint=checkpoints.append,
            on_proposal_batch_prepare=lambda prepared: None,
            on_proposal_batch_witness=_capture_and_mutate,
            should_cancel=_should_cancel,
        )
    [witness] = witnesses
    assert witness.phase == "completed"
    assert witness.slots[0].delta is not None
    assert witness.slots[0].delta.expected_effect != "observer mutation"


def test_search_checkpoint_commits_exactly_one_witness_digest_per_iteration() -> None:
    checkpoints: list[SearchCheckpoint] = []
    witnesses: list[SearchProposalBatchWitness] = []
    search_harness(
        "winner",
        HarnessDoc.baseline("seed"),
        _NeutralScorer(),
        _HistoryProposer(),
        iterations=2,
        screen_proposals=False,
        confirm_narrow_vetoes=False,
        on_checkpoint=checkpoints.append,
        on_proposal_batch_prepare=lambda prepared: None,
        on_proposal_batch_witness=witnesses.append,
    )
    assert [checkpoint.schema_version for checkpoint in checkpoints] == [
        "wmh.search-checkpoint.v2",
        "wmh.search-checkpoint.v2",
        "wmh.search-checkpoint.v2",
    ]
    assert checkpoints[0].proposal_batch_witness_digests == ()
    assert checkpoints[1].proposal_batch_witness_digests == (witnesses[0].payload_sha256,)
    assert checkpoints[2].proposal_batch_witness_digests == tuple(
        witness.payload_sha256 for witness in witnesses
    )


def test_search_harness_recognizes_only_the_exact_already_consumed_witness() -> None:
    checkpoints: list[SearchCheckpoint] = []
    witnesses: list[SearchProposalBatchWitness] = []

    def _stop_at_iteration_one(checkpoint: SearchCheckpoint) -> None:
        checkpoints.append(checkpoint)
        if checkpoint.completed_iteration == 1:
            raise RuntimeError("stop after first completed iteration")

    with pytest.raises(RuntimeError, match="first completed iteration"):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            _NeutralScorer(),
            _HistoryProposer(),
            iterations=2,
            screen_proposals=False,
            confirm_narrow_vetoes=False,
            on_checkpoint=_stop_at_iteration_one,
            on_proposal_batch_prepare=lambda prepared: None,
            on_proposal_batch_witness=witnesses.append,
        )
    committed = checkpoints[-1]
    assert committed.proposal_batch_witness_digests == (witnesses[0].payload_sha256,)

    proposer = _HistoryProposer()
    resumed = search_harness(
        "winner",
        HarnessDoc.baseline("seed"),
        _NeutralScorer(),
        proposer,
        iterations=2,
        screen_proposals=False,
        confirm_narrow_vetoes=False,
        resume_from=committed,
        resume_proposal_batch_witness=witnesses[0],
    )
    assert resumed.iterations == 2
    assert len(proposer.history_ids) == 1

    tampered_payload = witnesses[0].model_dump(mode="json")
    tampered_payload["slots"][0]["delta"]["expected_effect"] = "different batch"
    _rehash_witness_payload(tampered_payload)
    tampered = SearchProposalBatchWitness.model_validate(tampered_payload)
    scorer = _NeutralScorer()
    with pytest.raises(ValueError, match="committed witness digest"):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            scorer,
            _HistoryProposer(),
            iterations=2,
            screen_proposals=False,
            confirm_narrow_vetoes=False,
            resume_from=committed,
            resume_proposal_batch_witness=tampered,
        )
    assert scorer.requests == []


def test_consumed_witness_cannot_substitute_output_behind_a_rehashed_checkpoint() -> None:
    checkpoints: list[SearchCheckpoint] = []
    witnesses: list[SearchProposalBatchWitness] = []

    def _stop_after_first_iteration(checkpoint: SearchCheckpoint) -> None:
        checkpoints.append(checkpoint)
        if checkpoint.completed_iteration == 1:
            raise RuntimeError("stop after first iteration")

    with pytest.raises(RuntimeError, match="stop after first iteration"):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            _NeutralScorer(),
            _HistoryProposer(),
            iterations=2,
            screen_proposals=False,
            confirm_narrow_vetoes=False,
            on_checkpoint=_stop_after_first_iteration,
            on_proposal_batch_prepare=lambda prepared: None,
            on_proposal_batch_witness=witnesses.append,
        )

    witness_payload = witnesses[0].model_dump(mode="json")
    witness_payload["slots"][0]["delta"]["expected_effect"] = "stale substituted output"
    _rehash_witness_payload(witness_payload)
    substituted_witness = SearchProposalBatchWitness.model_validate(witness_payload)

    checkpoint_payload = checkpoints[-1].model_dump(mode="json")
    checkpoint_payload["proposal_batch_witness_digests"][0] = substituted_witness.payload_sha256
    _rehash_checkpoint_payload(checkpoint_payload)
    substituted_checkpoint = SearchCheckpoint.model_validate(checkpoint_payload)

    scorer = _NeutralScorer()
    with pytest.raises(ValueError, match="committed proposal output"):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            scorer,
            _HistoryProposer(),
            iterations=2,
            screen_proposals=False,
            confirm_narrow_vetoes=False,
            resume_from=substituted_checkpoint,
            resume_proposal_batch_witness=substituted_witness,
        )
    assert scorer.requests == []


def test_search_harness_rejects_resume_configuration_drift_before_scoring() -> None:
    checkpoints: list[SearchCheckpoint] = []
    with pytest.raises(RuntimeError, match="simulated checkpoint interruption"):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            _NeutralScorer(),
            _HistoryProposer(),
            iterations=2,
            screen_proposals=False,
            confirm_narrow_vetoes=False,
            on_checkpoint=_interrupt_after_iteration(checkpoints, iteration=0),
        )

    scorer = _NeutralScorer()
    with pytest.raises(ValueError, match="search checkpoint configuration drift"):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            scorer,
            _HistoryProposer(),
            iterations=2,
            proposal_batch_size=2,
            screen_proposals=False,
            confirm_narrow_vetoes=False,
            resume_from=checkpoints[-1],
        )

    assert scorer.requests == []
    assert scorer.before_proposal_calls == 0


def test_search_harness_binds_search_run_id_into_checkpoint_and_witness() -> None:
    checkpoints: list[SearchCheckpoint] = []
    witnesses: list[SearchProposalBatchWitness] = []
    with pytest.raises(RuntimeError, match="stop after witness"):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            _NeutralScorer(),
            _HistoryProposer(),
            search_run_id="discovery-run-001",
            iterations=1,
            screen_proposals=False,
            confirm_narrow_vetoes=False,
            on_checkpoint=checkpoints.append,
            on_proposal_batch_prepare=lambda prepared: None,
            on_proposal_batch_witness=lambda witness: (
                witnesses.append(witness),
                (_ for _ in ()).throw(RuntimeError("stop after witness")),
            ),
        )
    assert checkpoints[0].configuration.search_run_id == "discovery-run-001"
    assert witnesses[0].configuration.search_run_id == "discovery-run-001"

    scorer = _NeutralScorer()
    with pytest.raises(ValueError, match="search checkpoint configuration drift"):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            scorer,
            _HistoryProposer(),
            search_run_id="discovery-run-002",
            iterations=1,
            screen_proposals=False,
            confirm_narrow_vetoes=False,
            resume_from=checkpoints[0],
            resume_proposal_batch_witness=witnesses[0],
        )
    assert scorer.requests == []


def test_generated_search_run_id_rejects_stale_cross_run_witness_substitution() -> None:
    first_checkpoints: list[SearchCheckpoint] = []
    first_witnesses: list[SearchProposalBatchWitness] = []
    with pytest.raises(RuntimeError, match="stop after witness"):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            _NeutralScorer(),
            _HistoryProposer(),
            iterations=1,
            screen_proposals=False,
            confirm_narrow_vetoes=False,
            on_checkpoint=first_checkpoints.append,
            on_proposal_batch_prepare=lambda prepared: None,
            on_proposal_batch_witness=lambda witness: (
                first_witnesses.append(witness),
                (_ for _ in ()).throw(RuntimeError("stop after witness")),
            ),
        )

    second_checkpoints: list[SearchCheckpoint] = []
    with pytest.raises(RuntimeError, match="stop at second seed checkpoint"):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            _NeutralScorer(),
            _HistoryProposer(),
            iterations=1,
            screen_proposals=False,
            confirm_narrow_vetoes=False,
            on_checkpoint=lambda checkpoint: (
                second_checkpoints.append(checkpoint),
                (_ for _ in ()).throw(RuntimeError("stop at second seed checkpoint")),
            ),
        )

    first_run_id = first_checkpoints[0].configuration.search_run_id
    second_run_id = second_checkpoints[0].configuration.search_run_id
    assert first_run_id is not None
    assert second_run_id is not None
    assert first_run_id != second_run_id
    assert first_checkpoints[0].payload_sha256 != second_checkpoints[0].payload_sha256

    scorer = _NeutralScorer()
    proposer = _HistoryProposer()
    with pytest.raises(ValueError, match="proposal batch witness (configuration|prior checkpoint)"):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            scorer,
            proposer,
            iterations=1,
            screen_proposals=False,
            confirm_narrow_vetoes=False,
            resume_from=second_checkpoints[0],
            resume_proposal_batch_witness=first_witnesses[0],
        )
    assert scorer.requests == []
    assert proposer.history_ids == []


def test_search_harness_rejects_resume_scorer_matrix_drift_before_scoring() -> None:
    checkpoints: list[SearchCheckpoint] = []
    with pytest.raises(RuntimeError, match="simulated checkpoint interruption"):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            _NeutralScorer(),
            _HistoryProposer(),
            iterations=1,
            screen_proposals=False,
            confirm_narrow_vetoes=False,
            on_checkpoint=_interrupt_after_iteration(checkpoints, iteration=0),
        )

    scorer = _NeutralScorer()
    scorer.task_ids = ("drifted-task",)
    with pytest.raises(ValueError, match="search checkpoint configuration drift"):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            scorer,
            _HistoryProposer(),
            iterations=1,
            screen_proposals=False,
            confirm_narrow_vetoes=False,
            resume_from=checkpoints[-1],
        )
    assert scorer.requests == []


def test_checkpointed_search_requires_an_independently_configured_holdout() -> None:
    discovery = _NeutralScorer()
    holdout = _NeutralScorer()

    with pytest.raises(ValueError, match="independent scorer configuration"):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            discovery,
            _HistoryProposer(),
            iterations=0,
            screen_proposals=False,
            holdout_scorer=holdout,
            confirm_narrow_vetoes=False,
            on_checkpoint=lambda checkpoint: None,
        )

    assert discovery.requests == []
    assert holdout.requests == []


def test_checkpointed_search_requires_disjoint_discovery_and_holdout_tasks() -> None:
    class DistinctRouteSameTasks(_NeutralScorer):
        configuration_id = "distinct-route-same-tasks-v1"

    discovery = _NeutralScorer()
    holdout = DistinctRouteSameTasks()

    with pytest.raises(ValueError, match="task identities must be disjoint"):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            discovery,
            _HistoryProposer(),
            iterations=0,
            screen_proposals=False,
            holdout_scorer=holdout,
            confirm_narrow_vetoes=False,
            on_checkpoint=lambda checkpoint: None,
        )

    assert discovery.requests == []
    assert holdout.requests == []


def test_search_harness_fails_before_proposals_when_required_state_cannot_export() -> None:
    class MissingDurableExport(_HistoryProposer):
        configuration_id = "missing-durable-export-v1"
        durable_state_required = True

    scorer = _NeutralScorer()
    with pytest.raises(RuntimeError, match="exposes no export_search_state"):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            scorer,
            MissingDurableExport(),
            iterations=2,
            screen_proposals=False,
            confirm_narrow_vetoes=False,
            on_checkpoint=lambda checkpoint: None,
        )

    assert scorer.requests == []
    assert scorer.before_proposal_calls == 0


def test_search_harness_rejects_export_only_checkpoint_state_before_scoring() -> None:
    class ExportOnlyProposer(_HistoryProposer):
        configuration_id = "export-only-proposer-v1"

        def export_search_state(self) -> JsonObject:
            return {"state": "cannot be restored"}

    scorer = _NeutralScorer()
    with pytest.raises(RuntimeError, match="exposes no restore_search_state"):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            scorer,
            ExportOnlyProposer(),
            iterations=1,
            screen_proposals=False,
            confirm_narrow_vetoes=False,
            on_checkpoint=lambda checkpoint: None,
        )

    assert scorer.requests == []
    assert scorer.before_proposal_calls == 0


def test_search_harness_preserves_evaluation_id_collisions_across_resume() -> None:
    class CollidingCheckpointScorer(_NeutralScorer):
        configuration_id = "colliding-scorer-v1"

        def score(self, candidate: HarnessDoc, *, request: ScoreRequest) -> HarnessScoreReport:
            report = super().score(candidate, request=request)
            if request.purpose == "full":
                return report.model_copy(update={"evaluation_id": "shared-candidate-evaluation"})
            return report

    checkpoints: list[SearchCheckpoint] = []
    with pytest.raises(RuntimeError, match="simulated checkpoint interruption"):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            CollidingCheckpointScorer(),
            _HistoryProposer(),
            iterations=2,
            screen_proposals=False,
            confirm_narrow_vetoes=False,
            on_checkpoint=_interrupt_after_iteration(checkpoints, iteration=1),
        )

    assert any(
        record.evaluation_id == "shared-candidate-evaluation"
        for record in checkpoints[-1].evaluation_records
    )
    with pytest.raises(ValueError, match="shared-candidate-evaluation.*different report"):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            CollidingCheckpointScorer(),
            _HistoryProposer(),
            iterations=2,
            screen_proposals=False,
            confirm_narrow_vetoes=False,
            resume_from=checkpoints[-1],
        )


def test_search_checkpoint_file_rejects_corruption(tmp_path: Path) -> None:
    checkpoints: list[SearchCheckpoint] = []
    with pytest.raises(RuntimeError, match="simulated checkpoint interruption"):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            _NeutralScorer(),
            _HistoryProposer(),
            iterations=1,
            screen_proposals=False,
            confirm_narrow_vetoes=False,
            on_checkpoint=_interrupt_after_iteration(checkpoints, iteration=0),
        )
    path = tmp_path / "search-checkpoint.json"
    write_search_checkpoint(path, checkpoints[-1])
    assert load_search_checkpoint(path) == checkpoints[-1]

    content = json.loads(path.read_text())
    content["configuration"]["name"] = "corrupted-name"
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="checkpoint payload digest"):
        load_search_checkpoint(path)

    adversarial = checkpoints[-1].model_dump(mode="json")
    adversarial["evaluation_records"] = []
    payload = {key: value for key, value in adversarial.items() if key != "payload_sha256"}
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    adversarial["payload_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    path.write_text(json.dumps(adversarial))
    with pytest.raises(ValueError, match="missing a discovery evaluation identity"):
        load_search_checkpoint(path)

    completed: list[SearchCheckpoint] = []
    with pytest.raises(RuntimeError, match="simulated checkpoint interruption"):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            _NeutralScorer(),
            _HistoryProposer(),
            iterations=1,
            screen_proposals=False,
            confirm_narrow_vetoes=False,
            on_checkpoint=_interrupt_after_iteration(completed, iteration=1),
        )
    drifted_counts = completed[-1].model_dump(mode="json")
    drifted_counts["failure_cluster_expansions"][0]["count"] = 2
    count_payload = {key: value for key, value in drifted_counts.items() if key != "payload_sha256"}
    count_canonical = json.dumps(
        count_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    drifted_counts["payload_sha256"] = hashlib.sha256(count_canonical.encode()).hexdigest()
    path.write_text(json.dumps(drifted_counts))
    with pytest.raises(ValueError, match="expansion counts do not match history"):
        load_search_checkpoint(path)


def test_search_checkpoint_rejects_report_request_identity_drift() -> None:
    checkpoints: list[SearchCheckpoint] = []
    with pytest.raises(RuntimeError, match="simulated checkpoint interruption"):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            _NeutralScorer(),
            _HistoryProposer(),
            iterations=0,
            screen_proposals=False,
            confirm_narrow_vetoes=False,
            on_checkpoint=_interrupt_after_iteration(checkpoints, iteration=0),
        )
    payload = checkpoints[-1].model_dump(mode="json")
    [seed_record] = payload["evaluation_records"]
    seed_record["request"]["purpose"] = "full"
    _rehash_checkpoint_payload(payload)

    with pytest.raises(ValueError, match="discovery report request identity drifted"):
        SearchCheckpoint.model_validate(payload)


def test_search_checkpoint_binds_seed_execution_to_the_archive() -> None:
    checkpoints: list[SearchCheckpoint] = []
    with pytest.raises(RuntimeError, match="simulated checkpoint interruption"):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            _NeutralScorer(),
            _HistoryProposer(),
            iterations=0,
            screen_proposals=False,
            confirm_narrow_vetoes=False,
            on_checkpoint=_interrupt_after_iteration(checkpoints, iteration=0),
        )
    payload = checkpoints[-1].model_dump(mode="json")
    seed_hash = payload["configuration"]["seed_doc_hash"]
    seed_payload = payload["docs"][seed_hash]
    core_payload = next(
        surface for surface in seed_payload["surfaces"] if surface["id"] == "prompt:core"
    )
    core_payload["budget"] = len(core_payload["content"]) + 1
    drifted_seed = HarnessDoc.model_validate(seed_payload)
    assert drifted_seed.doc_hash == seed_hash
    assert drifted_seed.execution_hash != payload["configuration"]["seed_execution_hash"]
    [seed_evaluation] = payload["evaluation_records"]
    seed_evaluation["harness_execution_hash"] = drifted_seed.execution_hash
    _rehash_checkpoint_payload(payload)

    with pytest.raises(ValueError, match="seed document differs from archive seed"):
        SearchCheckpoint.model_validate(payload)


def test_search_checkpoint_rejects_an_unselected_accepted_delta() -> None:
    class TwoProposalProposer(_HistoryProposer):
        configuration_id = "two-proposal-proposer-v1"

        def propose_batch(
            self,
            parent: HarnessDoc,
            trigger: FailureSignature,
            evidence: str,
            *,
            history: list[HarnessDelta],
            count: int,
            should_cancel: Callable[[], bool] | None = None,
        ) -> list[HarnessDelta | ProposalFailure | None]:
            del evidence, history, should_cancel
            assert count == 2
            proposals = [
                parse_delta(parent, trigger, _meta_reply(parent, f"{_CAREFUL_PROMPT} {suffix}"))
                for suffix in ("first", "second")
            ]
            assert all(proposal is not None for proposal in proposals)
            return [proposal for proposal in proposals if proposal is not None]

    checkpoints: list[SearchCheckpoint] = []
    search_harness(
        "winner",
        HarnessDoc.baseline("seed"),
        _NeutralScorer(),
        TwoProposalProposer(),
        iterations=1,
        proposal_batch_size=2,
        screen_proposals=False,
        confirm_narrow_vetoes=False,
        on_checkpoint=checkpoints.append,
    )
    payload = checkpoints[-1].model_dump(mode="json")
    assert payload["proposal_records"][1]["selected"] is False
    payload["archive"]["deltas"][1]["verdict"]["accepted"] = True
    _rehash_checkpoint_payload(payload)

    with pytest.raises(ValueError, match="selection does not match delta verdict"):
        SearchCheckpoint.model_validate(payload)


def test_search_checkpoint_atomic_replace_preserves_last_commit_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoints: list[SearchCheckpoint] = []
    with pytest.raises(RuntimeError, match="simulated checkpoint interruption"):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            _NeutralScorer(),
            _HistoryProposer(),
            iterations=1,
            screen_proposals=False,
            confirm_narrow_vetoes=False,
            on_checkpoint=_interrupt_after_iteration(checkpoints, iteration=0),
        )
    path = tmp_path / "search-checkpoint.json"
    write_search_checkpoint(path, checkpoints[-1])

    def _fail_replace(source: Path, target: Path) -> None:
        del source, target
        raise OSError("simulated atomic replacement failure")

    monkeypatch.setattr(create_module.os, "replace", _fail_replace)
    with pytest.raises(OSError, match="simulated atomic replacement failure"):
        write_search_checkpoint(path, checkpoints[-1])

    assert load_search_checkpoint(path) == checkpoints[-1]
    assert list(tmp_path.iterdir()) == [path]


def test_iteration_checkpoint_precedes_fallible_observer_callbacks() -> None:
    checkpoints: list[SearchCheckpoint] = []
    events: list[str] = []

    def _checkpoint(checkpoint: SearchCheckpoint) -> None:
        checkpoints.append(checkpoint)
        events.append(f"checkpoint:{checkpoint.completed_iteration}")

    def _fail_accept(candidate: HarnessDoc, delta: HarnessDelta, score: float) -> None:
        del candidate, delta, score
        events.append("accept")
        raise RuntimeError("simulated observer failure")

    with pytest.raises(RuntimeError, match="simulated observer failure"):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            _NeutralScorer(),
            _HistoryProposer(),
            iterations=1,
            screen_proposals=False,
            confirm_narrow_vetoes=False,
            on_accept=_fail_accept,
            on_checkpoint=_checkpoint,
        )

    assert [checkpoint.completed_iteration for checkpoint in checkpoints] == [0, 1]
    assert events == ["checkpoint:0", "checkpoint:1", "accept"]


def test_seed_checkpoint_precedes_fallible_progress_callback() -> None:
    checkpoints: list[SearchCheckpoint] = []

    def _fail_progress(iteration: int, name: str, score: float, changed: bool) -> None:
        del name, score, changed
        assert iteration == 0
        raise RuntimeError("simulated progress failure")

    with pytest.raises(RuntimeError, match="simulated progress failure"):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            _NeutralScorer(),
            _HistoryProposer(),
            iterations=0,
            screen_proposals=False,
            confirm_narrow_vetoes=False,
            on_progress=_fail_progress,
            on_checkpoint=checkpoints.append,
        )

    assert [checkpoint.completed_iteration for checkpoint in checkpoints] == [0]


def test_observer_callbacks_cannot_mutate_live_search_state() -> None:
    checkpoints: list[SearchCheckpoint] = []

    def _mutate_accept(candidate: HarnessDoc, delta: HarnessDelta, score: float) -> None:
        del score
        candidate.surfaces[0].content = "observer-corrupted candidate"
        assert delta.verdict is not None
        delta.verdict.accepted = False

    def _mutate_proposal(record: ProposalRecord) -> None:
        record.selected = False
        record.reason = "observer-corrupted record"

    result = search_harness(
        "winner",
        HarnessDoc.baseline("seed"),
        _NeutralScorer(),
        _HistoryProposer(),
        iterations=2,
        screen_proposals=False,
        confirm_narrow_vetoes=False,
        on_accept=_mutate_accept,
        on_proposal=_mutate_proposal,
        on_checkpoint=checkpoints.append,
    )

    assert [checkpoint.completed_iteration for checkpoint in checkpoints] == [0, 1, 2]
    assert len(result.archive.accepted()) == 2
    assert [record.selected for record in result.proposal_records] == [True, True]
    assert all("observer-corrupted" not in surface.content for surface in result.best.surfaces)


def test_scorer_and_proposer_hooks_cannot_mutate_live_search_state() -> None:
    class MutatingScorer(_NeutralScorer):
        def validate_candidate(self, candidate: HarnessDoc) -> str | None:
            candidate.name = "validation-corrupted"
            return None

        def score(self, candidate: HarnessDoc, *, request: ScoreRequest) -> HarnessScoreReport:
            report = super().score(candidate, request=request)
            candidate.surfaces[0].content = "scorer-corrupted"
            return report

    class MutatingProposer(_HistoryProposer):
        configuration_id = "mutating-proposer-v1"

        def propose_batch(
            self,
            parent: HarnessDoc,
            trigger: FailureSignature,
            evidence: str,
            *,
            history: list[HarnessDelta],
            count: int,
            should_cancel: Callable[[], bool] | None = None,
        ) -> list[HarnessDelta | ProposalFailure | None]:
            proposals = super().propose_batch(
                parent,
                trigger,
                evidence,
                history=history,
                count=count,
                should_cancel=should_cancel,
            )
            parent.surfaces[0].content = "proposer-corrupted"
            for delta in history:
                if delta.verdict is not None:
                    delta.verdict.accepted = False
            return proposals

        def record_harness_evaluation(
            self,
            harness: HarnessDoc,
            *,
            archive: HarnessScoreArchive,
            should_cancel: Callable[[], bool] | None = None,
        ) -> None:
            super().record_harness_evaluation(
                harness,
                archive=archive,
                should_cancel=should_cancel,
            )
            harness.surfaces[0].content = "archive-hook-corrupted"
            archive.report.label = "archive-hook-corrupted"

        def record_evaluation(self, delta: HarnessDelta, *, stage: str, content: str) -> None:
            super().record_evaluation(delta, stage=stage, content=content)
            assert delta.verdict is not None or stage == "screen"
            if delta.verdict is not None:
                delta.verdict.accepted = False

    checkpoints: list[SearchCheckpoint] = []
    result = search_harness(
        "winner",
        HarnessDoc.baseline("seed"),
        MutatingScorer(),
        MutatingProposer(),
        iterations=2,
        screen_proposals=False,
        confirm_narrow_vetoes=False,
        on_checkpoint=checkpoints.append,
    )

    assert [checkpoint.completed_iteration for checkpoint in checkpoints] == [0, 1, 2]
    assert len(result.archive.accepted()) == 2
    assert all("corrupted" not in surface.content for surface in result.best.surfaces)
    assert all("corrupted" not in report.label for report in result.reports.values())


def test_search_rejects_a_distinct_execution_with_the_same_document_hash() -> None:
    class MetadataOnlyProposer(_HistoryProposer):
        configuration_id = "metadata-only-proposer-v1"

        def propose_batch(
            self,
            parent: HarnessDoc,
            trigger: FailureSignature,
            evidence: str,
            *,
            history: list[HarnessDelta],
            count: int,
            should_cancel: Callable[[], bool] | None = None,
        ) -> list[HarnessDelta | ProposalFailure | None]:
            del evidence, history, should_cancel
            assert count == 1
            core = parent.surface("prompt:core")
            assert core is not None
            proposal = parse_delta(
                parent,
                trigger,
                json.dumps(
                    {
                        "expected_effect": "metadata-only changes must not shadow the seed",
                        "preconditions": {"prompt:core": core.content_hash},
                        "ops": [
                            {
                                "op": "replace",
                                "surface_id": "prompt:core",
                                "content": core.content,
                                "budget": len(core.content) + 1,
                                "rationale": "change execution metadata only",
                            }
                        ],
                    }
                ),
            )
            assert proposal is not None
            return [proposal]

    scorer = _NeutralScorer()
    result = search_harness(
        "winner",
        HarnessDoc.baseline("seed"),
        scorer,
        MetadataOnlyProposer(),
        iterations=1,
        screen_proposals=False,
        confirm_narrow_vetoes=False,
        on_checkpoint=lambda checkpoint: None,
    )

    assert [request.purpose for _, request in scorer.requests] == ["seed"]
    assert result.proposal_records[0].outcome == "invalid"
    assert "document hash" in (result.proposal_records[0].reason or "")


def test_search_harness_scores_with_no_world_model_or_gold_judge() -> None:
    seed = HarnessDoc.baseline("seed")
    scorer = _NeutralScorer()
    proposer = _EvidenceRecordingProposer()

    result = search_harness(
        "winner",
        seed,
        scorer,
        proposer,
        iterations=1,
        screen_proposals=False,
        confirm_narrow_vetoes=False,
    )

    assert result.best_score == 1.0
    assert result.best.name == "winner"
    assert [request.purpose for _, request in scorer.requests] == ["seed", "full"]
    assert scorer.before_proposal_calls == 1
    assert "verifier reward and execution trace" in proposer.evidence[0]
    assert [archive.request.purpose for _, archive in proposer.archives] == ["seed", "full"]
    assert all(
        archive.scorer_tier is ScoreArchiveTier.DISCOVERY for _, archive in proposer.archives
    )
    assert all(
        archive.visibility is ScoreArchiveVisibility.PROPOSER for _, archive in proposer.archives
    )
    assert all(
        "verifier reward and execution trace"
        in archive.report.per_task["ground-truth-task"].evidence
        for _, archive in proposer.archives
    )
    assert "Hidden scorer tiers and confirmation measurements" in proposer.feedback[0]
    assert result.suite == ["ground-truth-task"]


def test_search_harness_fails_closed_when_complete_project_evidence_cannot_be_archived() -> None:
    class FailingArchiveProposer(_EvidenceRecordingProposer):
        def record_harness_evaluation(
            self,
            harness: HarnessDoc,
            *,
            archive: HarnessScoreArchive,
            should_cancel: Callable[[], bool] | None = None,
        ) -> None:
            del harness, archive, should_cancel
            raise OSError("durable project unavailable")

    with pytest.raises(RuntimeError, match="complete seed score evidence.*durable project"):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            _NeutralScorer(),
            FailingArchiveProposer(),
            iterations=1,
            screen_proposals=False,
            confirm_narrow_vetoes=False,
        )


def test_search_harness_fails_when_required_archive_capability_has_no_recorder() -> None:
    class RequiredArchiveProposer:
        score_archive_required = True

        def propose_batch(
            self,
            parent: HarnessDoc,
            trigger: FailureSignature,
            evidence: str,
            *,
            history: list[HarnessDelta],
            count: int,
            should_cancel: Callable[[], bool] | None = None,
        ) -> list[HarnessDelta | ProposalFailure | None]:
            del parent, trigger, evidence, history, count, should_cancel
            raise AssertionError("scoring must fail before proposal generation")

    scorer = _NeutralScorer()
    with pytest.raises(RuntimeError, match="requires durable score archives"):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            scorer,
            RequiredArchiveProposer(),
            iterations=1,
            screen_proposals=False,
            confirm_narrow_vetoes=False,
        )
    assert scorer.requests == []


def test_search_harness_marks_holdout_reports_audit_only() -> None:
    seed = HarnessDoc.baseline("seed")
    discovery = _NeutralScorer()
    holdout = _NeutralScorer()
    proposer = _EvidenceRecordingProposer()

    search_harness(
        "winner",
        seed,
        discovery,
        proposer,
        holdout_scorer=holdout,
        iterations=1,
        screen_proposals=False,
        confirm_narrow_vetoes=False,
    )

    assert [archive.request.purpose for _, archive in proposer.archives] == [
        "seed",
        "holdout",
        "full",
        "holdout",
    ]
    discovery_archive = proposer.archives[0][1]
    holdout_archive = proposer.archives[1][1]
    assert discovery_archive.scorer_tier is ScoreArchiveTier.DISCOVERY
    assert discovery_archive.visibility is ScoreArchiveVisibility.PROPOSER
    assert holdout_archive.scorer_tier is ScoreArchiveTier.HOLDOUT
    assert holdout_archive.visibility is ScoreArchiveVisibility.AUDIT_ONLY
    assert all(
        archive.visibility is ScoreArchiveVisibility.AUDIT_ONLY
        for _, archive in proposer.archives
        if archive.request.purpose == "holdout"
    )
    assert "held-out" not in proposer.feedback[0]
    assert "Hidden scorer tiers and confirmation measurements" in proposer.feedback[0]


def test_search_harness_never_gates_on_retryable_run_health() -> None:
    class RetryableScorer(_NeutralScorer):
        def score(self, candidate: HarnessDoc, *, request: ScoreRequest) -> HarnessScoreReport:
            report = super().score(candidate, request=request)
            return report.model_copy(update={"run_health": ScoreRunHealth.RETRY_REQUIRED})

    with pytest.raises(ScoreRunHealthError, match="retry or invalidate") as caught:
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            RetryableScorer(),
            _EvidenceRecordingProposer(),
            iterations=0,
            screen_proposals=False,
            confirm_narrow_vetoes=False,
        )

    assert caught.value.run_health is ScoreRunHealth.RETRY_REQUIRED


def test_search_harness_rejects_unsupported_paid_stages_before_scoring() -> None:
    scorer = _NeutralScorer()

    with pytest.raises(ValueError, match="screen_proposals=False"):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            scorer,
            _EvidenceRecordingProposer(),
            iterations=0,
        )

    assert scorer.requests == []


def test_search_harness_rejects_negative_iterations() -> None:
    scorer = _NeutralScorer()
    with pytest.raises(ValueError, match="iterations must be non-negative"):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            scorer,
            _EvidenceRecordingProposer(),
            iterations=-1,
            screen_proposals=False,
            confirm_narrow_vetoes=False,
        )
    assert scorer.requests == []


def test_search_harness_rejects_default_attempt_count_drift() -> None:
    class WrongAttemptsScorer(_NeutralScorer):
        def score(self, candidate: HarnessDoc, *, request: ScoreRequest) -> HarnessScoreReport:
            report = super().score(candidate, request=request)
            return report.model_copy(update={"attempts": self.default_attempts + 1})

    with pytest.raises(ValueError, match="attempts=3.*expected attempts=2"):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            WrongAttemptsScorer(),
            _EvidenceRecordingProposer(),
            iterations=0,
            screen_proposals=False,
            confirm_narrow_vetoes=False,
        )


def test_search_harness_rejects_candidate_that_omits_a_discovery_task() -> None:
    class OmittingScorer(_NeutralScorer):
        def score(self, candidate: HarnessDoc, *, request: ScoreRequest) -> HarnessScoreReport:
            report = super().score(candidate, request=request)
            if request.purpose != "seed":
                return report
            required = TaskScore(
                task_id="required-task",
                score=0.0,
                secondary_score=0.0,
                passed=False,
            )
            return report.model_copy(
                update={"per_task": {**report.per_task, required.task_id: required}}
            )

    with pytest.raises(ValueError, match="wrong task set.*required-task"):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            OmittingScorer(),
            _EvidenceRecordingProposer(),
            iterations=1,
            screen_proposals=False,
            confirm_narrow_vetoes=False,
        )


def test_search_harness_rejects_candidate_that_omits_a_holdout_task() -> None:
    class OmittingHoldoutScorer(_NeutralScorer):
        def score(self, candidate: HarnessDoc, *, request: ScoreRequest) -> HarnessScoreReport:
            report = super().score(candidate, request=request)
            if len(self.requests) != 1:
                return report
            required = TaskScore(
                task_id="held-out-required",
                score=0.0,
                secondary_score=0.0,
                passed=False,
            )
            return report.model_copy(
                update={"per_task": {**report.per_task, required.task_id: required}}
            )

    with pytest.raises(ValueError, match="wrong task set.*held-out-required"):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            _NeutralScorer(),
            _EvidenceRecordingProposer(),
            iterations=1,
            screen_proposals=False,
            holdout_scorer=OmittingHoldoutScorer(),
            confirm_narrow_vetoes=False,
        )


def test_search_harness_rejects_evaluation_identity_collision() -> None:
    class CollidingScorer(_NeutralScorer):
        def score(self, candidate: HarnessDoc, *, request: ScoreRequest) -> HarnessScoreReport:
            report = super().score(candidate, request=request)
            return report.model_copy(update={"evaluation_id": "reused-id"})

    with pytest.raises(ValueError, match="evaluation_id 'reused-id'.*different report"):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            CollidingScorer(),
            _EvidenceRecordingProposer(),
            iterations=1,
            screen_proposals=False,
            confirm_narrow_vetoes=False,
        )


def test_search_harness_snapshots_and_canonicalizes_scorer_owned_reports() -> None:
    class RetainingScorer(_NeutralScorer):
        source_report: HarnessScoreReport | None = None

        def score(self, candidate: HarnessDoc, *, request: ScoreRequest) -> HarnessScoreReport:
            report = super().score(candidate, request=request)
            extra = TaskScore(
                task_id="another-task",
                score=0.0,
                secondary_score=0.0,
                passed=False,
                evidence="another immutable trace",
            )
            self.source_report = report.model_copy(
                update={
                    "per_task": {
                        "ground-truth-task": report.per_task["ground-truth-task"],
                        "another-task": extra,
                    }
                }
            )
            return self.source_report

    seed = HarnessDoc.baseline("seed")
    scorer = RetainingScorer()
    result = search_harness(
        "winner",
        seed,
        scorer,
        _EvidenceRecordingProposer(),
        iterations=0,
        screen_proposals=False,
        confirm_narrow_vetoes=False,
    )
    source = scorer.source_report
    assert source is not None
    audited = result.reports[seed.doc_hash]
    assert list(audited.per_task) == ["another-task", "ground-truth-task"]

    source.score = 1.0
    source.per_task["ground-truth-task"].score = 1.0
    source.per_task["ground-truth-task"].evidence = "mutated after return"
    source.per_task["late-task"] = TaskScore(
        task_id="late-task",
        score=1.0,
        secondary_score=1.0,
        passed=True,
    )

    assert audited.score == 0.0
    assert audited.per_task["ground-truth-task"].score == 0.0
    assert audited.per_task["ground-truth-task"].evidence == "verifier reward and execution trace"
    assert "late-task" not in audited.per_task


def test_search_harness_rejects_empty_seed_and_holdout_matrices() -> None:
    class EmptyScorer(_NeutralScorer):
        def score(self, candidate: HarnessDoc, *, request: ScoreRequest) -> HarnessScoreReport:
            report = super().score(candidate, request=request)
            return report.model_copy(update={"score": 0.0, "secondary_score": 0.0, "per_task": {}})

    with pytest.raises(ValueError, match="seed score report contains no tasks"):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            EmptyScorer(),
            _EvidenceRecordingProposer(),
            iterations=0,
            screen_proposals=False,
            confirm_narrow_vetoes=False,
        )
    with pytest.raises(ValueError, match="holdout seed score report contains no tasks"):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            _NeutralScorer(),
            _EvidenceRecordingProposer(),
            iterations=0,
            screen_proposals=False,
            holdout_scorer=EmptyScorer(),
            confirm_narrow_vetoes=False,
        )


def test_search_harness_validates_and_retires_both_scorers() -> None:
    class RejectingHoldoutScorer(_NeutralScorer):
        def validate_candidate(self, candidate: HarnessDoc) -> str | None:
            prompt = candidate.surface("prompt:core")
            return (
                "holdout runtime rejected candidate"
                if prompt and "careful" in prompt.content
                else None
            )

    discovery = _NeutralScorer()
    holdout = RejectingHoldoutScorer()
    result = search_harness(
        "winner",
        HarnessDoc.baseline("seed"),
        discovery,
        _EvidenceRecordingProposer(),
        iterations=1,
        screen_proposals=False,
        holdout_scorer=holdout,
        confirm_narrow_vetoes=False,
    )

    assert [request.purpose for _, request in discovery.requests] == ["seed"]
    assert [request.purpose for _, request in holdout.requests] == ["holdout"]
    assert discovery.before_proposal_calls == holdout.before_proposal_calls == 1
    assert result.proposal_records[0].outcome == "invalid"
    assert "holdout runtime rejected candidate" in (result.proposal_records[0].reason or "")


def test_search_harness_validates_seed_against_holdout_before_scoring() -> None:
    class RejectingHoldoutScorer(_NeutralScorer):
        def validate_candidate(self, candidate: HarnessDoc) -> str | None:
            return "holdout runtime rejected seed"

    discovery = _NeutralScorer()
    holdout = RejectingHoldoutScorer()
    with pytest.raises(ValueError, match="holdout seed is not eligible"):
        search_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            discovery,
            _EvidenceRecordingProposer(),
            iterations=0,
            screen_proposals=False,
            holdout_scorer=holdout,
            confirm_narrow_vetoes=False,
        )
    assert discovery.requests == holdout.requests == []


def test_create_accepts_improving_delta_and_promotes_suite() -> None:
    seed = HarnessDoc.baseline("seed")
    provider = RoleProvider(meta_reply=_meta_reply(seed, _CAREFUL_PROMPT))
    progress: list[tuple[int, str, float, bool]] = []
    result = _run(provider, on_progress=lambda i, n, r, a: progress.append((i, n, r, a)))

    assert result.skipped == 0
    assert result.best_score == 1.0
    assert result.best.name == "winner"
    assert result.best.system_prompt() == _CAREFUL_PROMPT
    assert progress == [(0, "seed", 0.0, True), (1, "winner-i1-p1", 1.0, True)]

    [delta] = result.archive.deltas
    assert delta.verdict is not None and delta.verdict.accepted
    assert delta.verdict.full_delta == 1.0
    assert delta.verdict.holdout_delta is None
    assert "1/1 tasks now pass" in delta.verdict.reason
    # The trigger came from deterministic clustering of the seed's failures.
    assert delta.trigger.mechanism == "the work was verified"
    assert delta.trigger.task_ids == ["t1"]
    # The newly-passing task promoted into the regression suite.
    assert result.suite == ["t1"]
    # Reports are keyed by content: seed and child doc hashes, k=3 passes each.
    assert set(result.reports) == {seed.doc_hash, delta.child_doc_hash}
    assert all(r.k == 3 for r in result.reports.values())


def test_archive_reconstructs_children_by_folding_deltas() -> None:
    seed = HarnessDoc.baseline("seed")
    provider = RoleProvider(meta_reply=_meta_reply(seed, _CAREFUL_PROMPT))
    result = _run(provider)
    [delta] = result.archive.deltas
    assert delta.child_doc_hash is not None
    rebuilt = result.archive.reconstruct(delta.child_doc_hash)
    assert rebuilt.surfaces == result.best.surfaces
    with pytest.raises(ValueError, match="not in this archive"):
        result.archive.reconstruct("0" * 32)


def test_create_rejects_regressing_delta_and_keeps_champion() -> None:
    # The seed already passes; the proposed prompt makes the agent submit a broken answer.
    seed = HarnessDoc.baseline("seed")
    provider = RoleProvider(
        meta_reply=_meta_reply(seed, "You are a broken agent."),
        judge_fn=lambda user: "done-broken" not in user,
    )
    result = _run(provider)

    assert result.skipped == 0
    [delta] = result.archive.deltas
    assert delta.verdict is not None and not delta.verdict.accepted
    assert "full split" in delta.verdict.reason
    assert result.archive.accepted() == []
    # The champion never moved: the winner is the (renamed) seed at its original score.
    assert result.best_score == 1.0
    assert result.best.system_prompt() == seed.system_prompt()
    assert result.suite == ["t1"]  # and the suite kept the seed's win
    # An all-pass parent gets the generalization trigger, not a fabricated failure.
    assert delta.trigger.mechanism == "none: all tasks pass"


def test_create_skips_unusable_proposals() -> None:
    provider = RoleProvider(meta_reply="not json at all")
    result = _run(provider, iterations=2)
    assert result.skipped == 2
    assert result.archive.deltas == []  # nothing to audit: no delta object ever existed
    assert result.best.name == "winner"  # even a search with no children yields the renamed seed
    # Every iteration is recorded even when its proposal dies before producing anything.
    assert [(r.iteration, r.outcome) for r in result.proposal_records] == [
        (1, "unusable"),
        (2, "unusable"),
    ]


def test_create_stops_before_the_next_expensive_phase_when_cancelled() -> None:
    provider = RoleProvider(meta_reply="not json at all")
    checks = 0

    def should_cancel() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 2

    with pytest.raises(HarnessSearchCancelled):
        _run(provider, should_cancel=should_cancel)

    assert provider.meta_users == []


def test_create_passes_cancellation_into_a_batched_provider_proposer() -> None:
    seed = HarnessDoc.baseline("seed")
    provider = RoleProvider(meta_reply=_meta_reply(seed, _CAREFUL_PROMPT))

    with pytest.raises(HarnessSearchCancelled, match="cancelled"):
        _run(
            provider,
            proposal_batch_size=3,
            should_cancel=lambda: len(provider.meta_users) >= 1,
        )

    assert len(provider.meta_users) == 1


def test_create_never_converts_explicit_proposer_cancellation_to_failures() -> None:
    class _CancellingMetaProvider(RoleProvider):
        def complete(
            self,
            system: str,
            messages: list[Message],
            *,
            temperature: float = 0.7,
            max_tokens: int = 2048,
        ) -> Completion:
            if "meta-agent improving an agent harness" in system:
                raise HarnessSearchCancelled("harness search cancelled")
            return super().complete(
                system,
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    with pytest.raises(HarnessSearchCancelled, match="cancelled"):
        _run(_CancellingMetaProvider())


def test_cancellation_wins_before_accepted_lineage_and_callback_mutate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = HarnessDoc.baseline("seed")
    provider = RoleProvider(meta_reply=_meta_reply(seed, _CAREFUL_PROMPT))
    cancelled = False
    crowned: list[str] = []
    gate_delta = create_module.gate_score_delta

    def cancelling_gate(
        delta: HarnessDelta,
        *,
        child: HarnessScoreReport,
        champion: HarnessScoreReport,
        best_full: float,
        suite: list[str],
        child_holdout: HarnessScoreReport | None = None,
        champion_holdout: HarnessScoreReport | None = None,
    ) -> GateRecord:
        nonlocal cancelled
        verdict = gate_delta(
            delta,
            child=child,
            champion=champion,
            best_full=best_full,
            suite=suite,
            child_holdout=child_holdout,
            champion_holdout=champion_holdout,
        )
        if verdict.accepted:
            cancelled = True
        return verdict

    monkeypatch.setattr(create_module, "gate_score_delta", cancelling_gate)

    with pytest.raises(HarnessSearchCancelled, match="cancelled"):
        _run(
            provider,
            should_cancel=lambda: cancelled,
            on_accept=lambda doc, delta, score: crowned.append(doc.doc_hash),
        )

    assert crowned == []


def test_create_audits_invalid_delta_without_spending_eval() -> None:
    stale = json.dumps(
        {
            "expected_effect": "x",
            "preconditions": {"prompt:core": "0" * 32},
            "ops": [
                {
                    "op": "replace",
                    "surface_id": "prompt:core",
                    "content": _CAREFUL_PROMPT,
                    "rationale": "r",
                }
            ],
        }
    )
    provider = RoleProvider(meta_reply=stale)
    result = _run(provider)
    assert result.skipped == 1
    [delta] = result.archive.deltas
    assert delta.verdict is not None and not delta.verdict.accepted
    assert delta.verdict.reason.startswith("invalid before eval")
    assert delta.child_doc_hash is None  # it never applied, so it never produced a doc
    assert len(result.reports) == 1  # only the seed was ever scored


def test_holdout_regression_rejects_a_full_split_win() -> None:
    # The delta fixes the main task but breaks the held-out one: tiers 1-2 pass, tier 3 rejects.
    seed = HarnessDoc.baseline("seed")

    def judge(user: str) -> bool:
        if "the holdout task" in user:
            return "done-verified" not in user  # holdout passes only for the seed's plain answer
        return "done-verified" in user

    provider = RoleProvider(meta_reply=_meta_reply(seed, _CAREFUL_PROMPT), judge_fn=judge)
    holdout = [TaskSpec(task_id="h1", instruction="the holdout task", gold=["the base flow works"])]
    result = _run(provider, holdout=holdout)

    [delta] = result.archive.deltas
    assert delta.verdict is not None and not delta.verdict.accepted
    assert delta.verdict.full_delta == 1.0  # it really did win the full split...
    assert delta.verdict.holdout_delta == -1.0  # ...and really did regress held-out
    assert "held-out regressed" in delta.verdict.reason
    assert result.best_score == 0.0  # champion stays the seed
    assert set(result.holdout_reports) == {seed.doc_hash, delta.child_doc_hash}


# -- deterministic failure clustering ---------------------------------------------------------


def test_select_failure_cluster_rotates_equally_sized_failures() -> None:
    clusters = [
        FailureSignature(mechanism=mechanism, task_ids=[f"t-{mechanism}"])
        for mechanism in ("a", "b", "c")
    ]
    counts: dict[tuple[str, str, tuple[str, ...]], int] = {}
    selected: list[str] = []
    for _ in range(4):
        cluster = select_failure_cluster(clusters, counts, parent_doc_hash="parent")
        selected.append(cluster.mechanism)
        key = ("parent", cluster.mechanism, tuple(cluster.task_ids))
        counts[key] = counts.get(key, 0) + 1

    assert selected == ["a", "b", "c", "a"]


def test_create_rotates_failure_evidence_after_a_screened_batch() -> None:
    seed = HarnessDoc.baseline("seed")
    provider = RoleProvider(meta_reply=_useless_meta_reply(seed), judge_fn=lambda _user: False)
    tasks = [
        TaskSpec(task_id="t1", instruction="first failure", gold=["alpha assertion"]),
        TaskSpec(task_id="t2", instruction="second failure", gold=["beta assertion"]),
    ]

    create_harness(
        "winner",
        seed,
        tasks,
        _wm(provider),
        provider,
        ProviderDeltaProposer(provider),
        GoldJudge(provider),
        iterations=2,
        k=1,
    )

    assert '[TARGET] task_id="t1"' in provider.meta_users[0]
    assert '[other] task_id="t2"' in provider.meta_users[0]
    assert '[TARGET] task_id="t2"' in provider.meta_users[1]
    assert '[other] task_id="t1"' in provider.meta_users[1]


def test_create_does_not_discount_a_cluster_when_the_proposer_failed() -> None:
    seed = HarnessDoc.baseline("seed")
    provider = RoleProvider(judge_fn=lambda _user: False)
    tasks = [
        TaskSpec(task_id="t1", instruction="first failure", gold=["alpha assertion"]),
        TaskSpec(task_id="t2", instruction="second failure", gold=["beta assertion"]),
    ]

    class FailingProposer:
        def __init__(self) -> None:
            self.triggers: list[FailureSignature] = []

        def propose_batch(  # noqa: PLR0913 - mirrors the proposer protocol
            self,
            parent: HarnessDoc,
            trigger: FailureSignature,
            evidence: str,
            *,
            history: list[HarnessDelta],
            count: int,
            should_cancel: Callable[[], bool] | None = None,
        ) -> list[HarnessDelta | ProposalFailure | None]:
            del parent, evidence, history, should_cancel
            self.triggers.append(trigger)
            return [ProposalFailure(reason="temporary transport failure")] * count

    proposer = FailingProposer()
    create_harness(
        "winner",
        seed,
        tasks,
        _wm(provider),
        provider,
        proposer,
        GoldJudge(provider),
        iterations=2,
        k=1,
    )

    assert [trigger.task_ids for trigger in proposer.triggers] == [["t1"], ["t1"]]


def test_create_does_not_discount_a_cluster_when_every_delta_is_invalid() -> None:
    """Parsed deltas spend no cluster allocation until one can enter evaluation."""
    seed = HarnessDoc.baseline("seed")
    stale = json.dumps(
        {
            "expected_effect": "fix the selected failure",
            "preconditions": {"prompt:core": "0" * 32},
            "ops": [
                {
                    "op": "replace",
                    "surface_id": "prompt:core",
                    "content": _CAREFUL_PROMPT,
                    "rationale": "exercise the invalid-before-eval path",
                }
            ],
        }
    )
    provider = RoleProvider(meta_reply=stale, judge_fn=lambda _user: False)
    tasks = [
        TaskSpec(task_id="t1", instruction="first failure", gold=["alpha assertion"]),
        TaskSpec(task_id="t2", instruction="second failure", gold=["beta assertion"]),
    ]

    result = create_harness(
        "winner",
        seed,
        tasks,
        _wm(provider),
        provider,
        ProviderDeltaProposer(provider),
        GoldJudge(provider),
        iterations=2,
        k=1,
    )

    assert result.skipped == 2
    assert '[TARGET] task_id="t1"' in provider.meta_users[0]
    assert '[TARGET] task_id="t1"' in provider.meta_users[1]
    assert '[other] task_id="t2"' in provider.meta_users[0]
    assert '[other] task_id="t2"' in provider.meta_users[1]


# -- staged verification: screening + history ---------------------------------------------------


def _useless_meta_reply(parent: HarnessDoc) -> str:
    """A well-formed delta that changes wording but cannot fix the failing task."""
    return _meta_reply(parent, "You are an agent. Do the task.")


def test_screen_rejects_delta_that_does_not_improve_its_trigger() -> None:
    seed = HarnessDoc.baseline("seed")
    provider = RoleProvider(meta_reply=_useless_meta_reply(seed))
    progress: list[tuple[int, str, float, bool]] = []
    result = _run(provider, on_progress=lambda i, n, r, a: progress.append((i, n, r, a)))

    assert result.screened == 1 and result.skipped == 0
    [delta] = result.archive.deltas
    assert delta.verdict is not None and not delta.verdict.accepted
    assert delta.verdict.reason.startswith("screened out")
    # The dead iteration is still a first-class record, with its screen means attached.
    [record] = result.proposal_records
    assert record.iteration == 1 and record.outcome == "screened"
    assert record.screen_child is not None and record.screen_parent is not None
    # The cheap screen replaced the full eval: only the seed has a full-split report. The
    # iteration still emits one unchanged champion checkpoint.
    assert len(result.reports) == 1
    assert progress == [(0, "seed", 0.0, True), (1, "seed", 0.0, False)]


def test_screen_uses_assertion_fraction_to_admit_partial_improvement() -> None:
    seed = HarnessDoc.baseline("seed")

    class PartialJudgeProvider(RoleProvider):
        def complete(
            self,
            system: str,
            messages: list[Message],
            *,
            temperature: float = 0.7,
            max_tokens: int = 2048,
        ) -> Completion:
            if "grade whether an agent completed a task" not in system:
                return super().complete(
                    system,
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            user = messages[-1].content
            improved = "done-verified" in user
            assertions = _gold_assertions(user)
            results = [
                {
                    "assertion": assertion,
                    "passed": improved and index == 0,
                    "why": "one subgoal improved" if improved and index == 0 else "still missing",
                }
                for index, assertion in enumerate(assertions)
            ]
            return Completion(text=json.dumps({"assertions": results, "passed": False}))

    provider = PartialJudgeProvider(meta_reply=_meta_reply(seed, _CAREFUL_PROMPT))
    tasks = [
        TaskSpec(
            task_id="t1",
            instruction="complete both parts",
            gold=["part one complete", "part two complete"],
        )
    ]

    result = create_harness(
        "winner",
        seed,
        tasks,
        _wm(provider),
        provider,
        ProviderDeltaProposer(provider),
        GoldJudge(provider),
        iterations=1,
        k=1,
    )

    assert result.screened == 0
    [record] = result.proposal_records
    assert record.outcome == "scored"
    assert record.screen_child == record.screen_parent == 0.0
    assert record.screen_parent_secondary == 0.0
    assert record.screen_child_secondary == 0.5


def test_gate_rejects_target_partial_lift_when_full_split_partial_credit_regresses() -> None:
    """Dense screening is a prefilter; the authoritative full gate protects other tasks."""
    delta = HarnessDelta.model_construct(
        trigger=FailureSignature(mechanism="target", task_ids=["target"])
    )
    champion = HarnessScoreReport(
        evaluation_id="champion",
        score=0.0,
        secondary_score=0.45,
        attempts=1,
        run_health=ScoreRunHealth.VALID,
        per_task={
            "target": TaskScore(task_id="target", score=0.0, secondary_score=0.0, passed=False),
            "other": TaskScore(task_id="other", score=0.0, secondary_score=0.9, passed=False),
        },
    )
    child = HarnessScoreReport(
        evaluation_id="child",
        score=0.0,
        secondary_score=0.25,
        attempts=1,
        run_health=ScoreRunHealth.VALID,
        per_task={
            "target": TaskScore(task_id="target", score=0.0, secondary_score=0.5, passed=False),
            "other": TaskScore(task_id="other", score=0.0, secondary_score=0.0, passed=False),
        },
    )

    verdict = create_module.gate_score_delta(
        delta,
        child=child,
        champion=champion,
        best_full=0.0,
        suite=[],
    )

    assert verdict.accepted is False
    assert verdict.full_delta == 0.0
    assert verdict.full_secondary_delta == pytest.approx(-0.2)
    assert "full-split secondary score regressed" in verdict.reason


def test_gate_accepts_binary_tie_with_nonregressing_global_partial_progress() -> None:
    delta = HarnessDelta.model_construct(
        trigger=FailureSignature(mechanism="target", task_ids=["target"])
    )
    champion = HarnessScoreReport(
        evaluation_id="champion",
        score=0.0,
        secondary_score=0.1,
        attempts=1,
        run_health=ScoreRunHealth.VALID,
        per_task={
            "target": TaskScore(task_id="target", score=0.0, secondary_score=0.0, passed=False),
            "other": TaskScore(task_id="other", score=0.0, secondary_score=0.2, passed=False),
        },
    )
    child = HarnessScoreReport(
        evaluation_id="child",
        score=0.0,
        secondary_score=0.35,
        attempts=1,
        run_health=ScoreRunHealth.VALID,
        per_task={
            "target": TaskScore(task_id="target", score=0.0, secondary_score=0.5, passed=False),
            "other": TaskScore(task_id="other", score=0.0, secondary_score=0.2, passed=False),
        },
    )

    verdict = create_module.gate_score_delta(
        delta,
        child=child,
        champion=champion,
        best_full=0.0,
        suite=[],
    )

    assert verdict.accepted is True
    assert verdict.full_secondary_delta == pytest.approx(0.25)


def test_search_records_screen_and_full_trace_feedback_for_project_proposers() -> None:
    seed = HarnessDoc.baseline("seed")
    provider = RoleProvider(meta_reply=_meta_reply(seed, _CAREFUL_PROMPT))

    class RecordingProposer(ProviderDeltaProposer):
        def __init__(self, wrapped: Provider) -> None:
            super().__init__(wrapped)
            self.evaluations: list[tuple[str, str]] = []

        def record_evaluation(self, delta: HarnessDelta, *, stage: str, content: str) -> None:
            del delta
            self.evaluations.append((stage, content))

    proposer = RecordingProposer(provider)
    create_harness(
        "winner",
        seed,
        _tasks(),
        _wm(provider),
        provider,
        proposer,
        GoldJudge(provider),
        iterations=1,
        k=1,
    )

    assert [stage for stage, _content in proposer.evaluations] == ["screen", "full"]
    assert all("Execution transcript" in content for _stage, content in proposer.evaluations)
    assert all("Judge feedback" in content for _stage, content in proposer.evaluations)


def test_feedback_persistence_failure_does_not_abort_scored_search() -> None:
    seed = HarnessDoc.baseline("seed")
    provider = RoleProvider(meta_reply=_meta_reply(seed, _CAREFUL_PROMPT))
    notes: list[str] = []

    class BrokenFeedbackProposer(ProviderDeltaProposer):
        def record_evaluation(self, delta: HarnessDelta, *, stage: str, content: str) -> None:
            del delta, stage, content
            raise RuntimeError("project filesystem disconnected")

    result = create_harness(
        "winner",
        seed,
        _tasks(),
        _wm(provider),
        provider,
        BrokenFeedbackProposer(provider),
        GoldJudge(provider),
        iterations=1,
        k=1,
        on_note=notes.append,
    )

    assert result.best_score == 1.0
    assert len(result.proposal_records) == 1
    assert any("screen feedback could not be persisted" in note for note in notes)
    assert any("full feedback could not be persisted" in note for note in notes)


def test_feedback_persistence_preserves_explicit_cancellation() -> None:
    seed = HarnessDoc.baseline("seed")
    provider = RoleProvider(meta_reply=_meta_reply(seed, _CAREFUL_PROMPT))

    class CancellingFeedbackProposer(ProviderDeltaProposer):
        def record_evaluation(self, delta: HarnessDelta, *, stage: str, content: str) -> None:
            del delta, stage, content
            raise HarnessSearchCancelled("harness search cancelled")

    with pytest.raises(HarnessSearchCancelled, match="cancelled"):
        create_harness(
            "winner",
            seed,
            _tasks(),
            _wm(provider),
            provider,
            CancellingFeedbackProposer(provider),
            GoldJudge(provider),
            iterations=1,
            k=1,
        )


def test_rejected_history_reaches_the_next_proposal() -> None:
    seed = HarnessDoc.baseline("seed")
    provider = RoleProvider(meta_reply=_useless_meta_reply(seed))
    _run(provider, iterations=2)
    assert len(provider.meta_users) == 2
    assert "Previous attempts" not in provider.meta_users[0]
    assert "Previous attempts" in provider.meta_users[1]
    assert "screened out" in provider.meta_users[1]  # the verdict itself is the lesson


# -- code deltas end to end ----------------------------------------------------------------------


def _code_meta_reply(parent: HarnessDoc) -> str:
    from wmh.harness.doc import CODE_RUNTIME_ID

    code_surface = parent.surface(CODE_RUNTIME_ID)
    assert code_surface is not None
    new_code = (
        "def run(kit):\n"
        '    kit.execute("bash", {"command": "verify the work"})\n'
        '    return "done-verified"\n'
    )
    return json.dumps(
        {
            "expected_effect": "the failing task flips to pass",
            "preconditions": {CODE_RUNTIME_ID: code_surface.content_hash},
            "ops": [
                {
                    "op": "replace",
                    "surface_id": CODE_RUNTIME_ID,
                    "content": new_code,
                    "rationale": "verify via the environment before submitting",
                }
            ],
        }
    )


def test_code_delta_passes_screen_and_gate_end_to_end() -> None:
    from wmh.harness.doc import CODE_RUNTIME_ID, code_baseline

    seed = code_baseline("seed")
    provider = RoleProvider(meta_reply=_code_meta_reply(seed))
    result = create_harness(
        "winner",
        seed,
        _tasks(),
        _wm(provider),
        provider,
        ProviderDeltaProposer(provider),
        GoldJudge(provider),
        iterations=1,
        k=3,
    )
    assert result.screened == 0 and result.skipped == 0
    [delta] = result.archive.deltas
    assert delta.verdict is not None and delta.verdict.accepted
    assert [op.surface_id for op in delta.ops] == [CODE_RUNTIME_ID]
    assert result.best_score == 1.0
    winner_code = result.best.surface(CODE_RUNTIME_ID)
    assert winner_code is not None and "done-verified" in winner_code.content


def test_broken_code_delta_is_rejected_before_any_eval() -> None:
    from wmh.harness.doc import CODE_RUNTIME_ID, code_baseline

    seed = code_baseline("seed")
    code_surface = seed.surface(CODE_RUNTIME_ID)
    assert code_surface is not None
    broken = json.dumps(
        {
            "expected_effect": "x",
            "preconditions": {CODE_RUNTIME_ID: code_surface.content_hash},
            "ops": [
                {
                    "op": "replace",
                    "surface_id": CODE_RUNTIME_ID,
                    "content": "def run(kit:\n",
                    "rationale": "r",
                }
            ],
        }
    )
    provider = RoleProvider(meta_reply=broken)
    result = create_harness(
        "winner",
        seed,
        _tasks(),
        _wm(provider),
        provider,
        ProviderDeltaProposer(provider),
        GoldJudge(provider),
        iterations=1,
        k=2,
    )
    assert result.skipped == 1
    [delta] = result.archive.deltas
    assert delta.verdict is not None and "does not compile" in delta.verdict.reason
    assert len(result.reports) == 1  # only the seed was ever scored


# -- confirmation re-runs -----------------------------------------------------------------------


def test_narrow_failing_tiers_eligibility() -> None:
    from wmh.harness.create import narrow_failing_tiers
    from wmh.harness.delta import GateRecord

    def record(**kw) -> GateRecord:  # noqa: ANN003
        return GateRecord(accepted=False, reason="r", **kw)

    # Narrow holdout veto on a full-split win -> retry that tier.
    narrow = record(full_delta=0.05, holdout_delta=-0.1)
    assert narrow_failing_tiers(narrow, k=5, n_suite=4, n_holdout=4) == ["holdout"]
    # A wide veto is a real regression, not noise: ineligible.
    wide = record(full_delta=0.05, holdout_delta=-1.0)
    assert narrow_failing_tiers(wide, k=5, n_suite=4, n_holdout=4) is None
    # No full-split win: nothing to confirm.
    no_win = record(full_delta=0.0, holdout_delta=-0.1)
    assert narrow_failing_tiers(no_win, k=5, n_suite=4, n_holdout=4) is None
    # Both tiers narrowly failing -> both retried.
    both = record(full_delta=0.05, suite_delta=-0.05, holdout_delta=-0.1)
    assert narrow_failing_tiers(both, k=5, n_suite=8, n_holdout=4) == ["suite", "holdout"]
    # Confirmation of one binary veto cannot erase a separate dense-signal veto.
    dense_veto = record(
        full_delta=0.05,
        suite_delta=-0.05,
        holdout_delta=0.0,
        holdout_secondary_delta=-0.2,
    )
    assert narrow_failing_tiers(dense_veto, k=5, n_suite=8, n_holdout=4) is None
    # Accepted verdicts are never retried.
    ok = GateRecord(accepted=True, reason="r", full_delta=0.05)
    assert narrow_failing_tiers(ok, k=5, n_suite=4, n_holdout=4) is None


def test_flaky_holdout_veto_is_overturned_by_confirmation() -> None:
    # The child genuinely fixes the train task; the holdout task fails exactly ONE child
    # attempt (judge flakiness). The initial k-pass gate vetoes; the 2k re-measurement of
    # child AND champion overturns it.
    seed = HarnessDoc.baseline("seed")
    child_h1_calls = {"n": 0}

    def judge(user: str) -> bool:
        if "the holdout task" in user:
            if "done-verified" in user:  # the child's answer style
                child_h1_calls["n"] += 1
                return child_h1_calls["n"] != 1  # fail only the first child attempt
            return True  # the seed always passes holdout
        return "done-verified" in user  # train task needs the careful child

    provider = RoleProvider(meta_reply=_meta_reply(seed, _CAREFUL_PROMPT), judge_fn=judge)
    holdout = [TaskSpec(task_id="h1", instruction="the holdout task", gold=["the base flow works"])]
    result = _run(provider, holdout=holdout, k=5)

    assert result.confirmations == 1
    [delta] = result.archive.deltas
    assert delta.verdict is not None and delta.verdict.accepted
    assert "veto overturned" in delta.verdict.reason
    assert "initially: rejected" in delta.verdict.reason
    assert result.best_score == 1.0  # the win was kept


def test_wide_holdout_regression_skips_confirmation() -> None:
    # Same setup as the second-iteration holdout test: the child ALWAYS fails held-out. -1.0 is far
    # beyond the narrow margin, so no re-measurement is spent and the plain rejection stands.
    seed = HarnessDoc.baseline("seed")

    def judge(user: str) -> bool:
        if "the holdout task" in user:
            return "done-verified" not in user
        return "done-verified" in user

    provider = RoleProvider(meta_reply=_meta_reply(seed, _CAREFUL_PROMPT), judge_fn=judge)
    holdout = [TaskSpec(task_id="h1", instruction="the holdout task", gold=["the base flow works"])]
    result = _run(provider, holdout=holdout)
    assert result.confirmations == 0
    [delta] = result.archive.deltas
    assert delta.verdict is not None and not delta.verdict.accepted
    assert "held-out regressed" in delta.verdict.reason


def test_confirmed_suite_overturn_still_faces_the_holdout_tier() -> None:
    # A suite veto narrow enough to overturn must NOT smuggle the child past held-out
    # verification: here the suite flake clears on re-measurement but the child genuinely
    # regresses held-out, so the final verdict is a holdout rejection.
    seed = HarnessDoc.baseline("seed")
    suite_flake = {"n": 0}

    def judge(user: str) -> bool:
        if "the holdout task" in user:
            return "done-verified" not in user  # child ALWAYS fails held-out (wide, real)
        if "suite task" in user:
            if "done-verified" in user:
                suite_flake["n"] += 1
                return suite_flake["n"] != 1  # one flaky failure for the child
            return True  # seed masters the suite task
        return "done-verified" in user  # the trigger task needs the careful child

    tasks = [
        TaskSpec(task_id="t1", instruction="answer it", gold=["the work was verified"]),
        TaskSpec(task_id="s1", instruction="the suite task", gold=["steady state holds"]),
    ]
    holdout = [TaskSpec(task_id="h1", instruction="the holdout task", gold=["the base flow works"])]
    provider = RoleProvider(meta_reply=_meta_reply(seed, _CAREFUL_PROMPT), judge_fn=judge)
    result = create_harness(
        "winner",
        seed,
        tasks,
        _wm(provider),
        provider,
        ProviderDeltaProposer(provider),
        GoldJudge(provider),
        iterations=1,
        k=5,
        holdout=holdout,
    )
    [delta] = result.archive.deltas
    assert delta.verdict is not None and not delta.verdict.accepted
    # The holdout tier was measured (not bypassed) and its regression is the rejection.
    assert delta.verdict.holdout_delta is not None and delta.verdict.holdout_delta < 0
    assert result.best_score == pytest.approx(0.5)  # champion stayed the seed


# -- harness backends: local (in-process) vs e2b (the pi process in pooled sandboxes) ------------


def _pi_seed() -> HarnessDoc:
    from wmh.harness.doc import RUNTIME_KIND_ID, TOOL_POLICY_ID, Surface, SurfaceKind

    return HarnessDoc(
        name="seed",
        surfaces=[
            Surface(id="prompt:core", kind=SurfaceKind.PROMPT, content="p"),
            Surface(id=TOOL_POLICY_ID, kind=SurfaceKind.TOOL_POLICY, content="bash\nsubmit"),
            Surface(id=RUNTIME_KIND_ID, kind=SurfaceKind.PARAM, content="pi-node"),
            Surface(id="code:a", kind=SurfaceKind.CODE, path="src/agent.ts", content="// a"),
        ],
    )


def _canned_report(rate: float, *, k: int = 3) -> ClosedLoopReport:
    successes = round(rate * k)
    if abs(successes / k - rate) > 1e-9:
        raise ValueError(f"rate {rate} cannot be represented by k={k} binary verdicts")
    verdicts = [
        GoldVerdict(passed=index < successes, fraction=1.0 if index < successes else 0.0)
        for index in range(k)
    ]
    attempts = [RolloutEvidence(stop_reason=StopReason.SUBMITTED) for _ in range(k)]
    outcome = TaskOutcome(
        task_id="t1",
        success_rate=rate,
        mean_fraction=rate,
        passes=k,
        verdicts=verdicts,
        attempts=attempts,
    )
    return ClosedLoopReport(
        label="x", success_rate=rate, mean_fraction=rate, k=k, per_task={"t1": outcome}
    )


def _corrupt_closed_loop_report(kind: str) -> ClosedLoopReport:
    """Return one malformed raw evaluator result for adapter fail-closed tests."""
    report = _canned_report(1.0)
    outcome = report.per_task["t1"].model_copy(deep=True)
    if kind == "missing task":
        return report.model_copy(update={"per_task": {}})
    if kind == "extra task":
        extra = outcome.model_copy(update={"task_id": "t2"})
        return report.model_copy(update={"per_task": {"t1": outcome, "t2": extra}})
    if kind == "mismatched task id":
        outcome = outcome.model_copy(update={"task_id": "not-t1"})
    elif kind == "wrong pass count":
        outcome = outcome.model_copy(update={"passes": 2})
    elif kind == "missing verdict":
        outcome = outcome.model_copy(update={"verdicts": outcome.verdicts[:-1]})
    elif kind == "missing evidence":
        outcome = outcome.model_copy(update={"attempts": outcome.attempts[:-1]})
    elif kind == "task score drift":
        outcome = outcome.model_copy(update={"success_rate": 0.5})
    elif kind == "task secondary drift":
        outcome = outcome.model_copy(update={"mean_fraction": 0.5})
    elif kind == "aggregate score drift":
        return report.model_copy(update={"success_rate": 0.5})
    elif kind == "aggregate secondary drift":
        return report.model_copy(update={"mean_fraction": 0.5})
    else:
        raise AssertionError(f"unknown corruption {kind!r}")
    return report.model_copy(update={"per_task": {"t1": outcome}})


@pytest.mark.parametrize(
    ("kind", "error"),
    [
        ("missing task", "wrong task set.*missing=\\['t1'\\]"),
        ("extra task", "wrong task set.*extra=\\['t2'\\]"),
        ("mismatched task id", "key 't1'.*task_id 'not-t1'"),
        ("wrong pass count", "task 't1'.*passes=2.*expected 3"),
        ("missing verdict", "task 't1'.*2 verdicts.*expected 3"),
        ("missing evidence", "task 't1'.*2 evidence.*expected 3"),
        ("task score drift", "task 't1'.*success_rate=0.5.*verdicts=1.0"),
        ("task secondary drift", "task 't1'.*mean_fraction=0.5.*verdicts=1.0"),
        ("aggregate score drift", "report success_rate=0.5.*per-task mean=1.0"),
        ("aggregate secondary drift", "report mean_fraction=0.5.*per-task mean=1.0"),
    ],
)
def test_closed_loop_adapter_rejects_incomplete_or_inconsistent_raw_reports(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    error: str,
) -> None:
    provider = RoleProvider()
    malformed = _corrupt_closed_loop_report(kind)
    monkeypatch.setattr(create_module, "evaluate_closed_loop", lambda *args, **kwargs: malformed)

    with pytest.raises(ValueError, match=error):
        create_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            _tasks(),
            _wm(provider),
            provider,
            ProviderDeltaProposer(provider),
            GoldJudge(provider),
            iterations=0,
        )


class _ScriptedPoolChannel:
    """Plays the runner peer for one pooled episode: a tool_request, then done.

    The same frame script `runner_link_test._FakeChannel` speaks; recv() hands frames to the
    real `RunnerLink`, send() records what the host answered — the tool_response content is how
    a test observes WHO answered the tool call.
    """

    def __init__(self) -> None:
        self.sent: list[JsonObject] = []
        self._script: list[JsonObject] = [
            {
                "type": "tool_request",
                "req_id": 1,
                "name": "bash",
                "arguments": {"command": "verify the work"},
            },
            {"type": "done", "answer": "done-verified"},
        ]

    def send(self, frame: JsonObject) -> None:
        self.sent.append(frame)

    def recv(self, timeout: float | None = None) -> JsonObject | None:
        del timeout
        return self._script.pop(0) if self._script else None


class _FakePool:
    """Stands in for `E2BSandboxPool`: no sandboxes, one scripted runner channel per acquire."""

    instances: ClassVar[list[_FakePool]] = []

    def __init__(
        self,
        *,
        template: str | None = None,
        api_key: str | None = None,
        metadata: dict[str, str] | None = None,
        sandbox_factory: object = None,
        hello_timeout: float = 0.0,
    ) -> None:
        self.template = template
        self.metadata = metadata
        self.channels: list[_ScriptedPoolChannel] = []
        self.releases: list[bool] = []
        self.retire_idle_calls = 0
        self.closes = 0
        self.close_failures = 0
        _FakePool.instances.append(self)

    def usage(self) -> SandboxUsage:
        return SandboxUsage(count=len(self.channels), seconds=1.5 * len(self.channels))

    def acquire(self) -> tuple[object, _ScriptedPoolChannel]:
        channel = _ScriptedPoolChannel()
        self.channels.append(channel)
        return object(), channel

    def release(self, sandbox: object, channel: object, *, healthy: bool) -> None:
        self.releases.append(healthy)

    def retire_idle(self) -> int:
        self.retire_idle_calls += 1
        return 0

    def close(self) -> None:
        self.closes += 1
        if self.closes <= self.close_failures:
            from wmh.harness.e2b_sandbox import SandboxCleanupError

            raise SandboxCleanupError("evaluator cleanup unproven")


@pytest.fixture
def fake_pool_cls(monkeypatch: pytest.MonkeyPatch) -> type[_FakePool]:
    """Patch the pool at its source module (create_harness imports it lazily from there)."""
    _FakePool.instances = []
    monkeypatch.setattr("wmh.harness.pi_e2b.E2BSandboxPool", _FakePool)
    return _FakePool


def test_unknown_harness_backend_is_rejected() -> None:
    from typing import Literal, cast

    provider = RoleProvider()
    # Dynamic callers (the platform's optimizer passes a plain str) can hand in anything;
    # the runtime guard, not the type annotation, is what this test pins.
    bogus = cast("Literal['local', 'e2b']", "banana")
    with pytest.raises(ValueError, match="choose local or e2b"):
        create_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            _tasks(),
            _wm(provider),
            provider,
            ProviderDeltaProposer(provider),
            GoldJudge(provider),
            harness_backend=bogus,
        )


def test_e2b_backend_rejects_non_pi_node_seeds() -> None:
    """e2b moves the pi-node harness PROCESS into sandboxes; in-process seeds must fail early."""
    provider = RoleProvider()
    with pytest.raises(ValueError, match="use harness_backend='local'"):
        create_harness(
            "winner",
            HarnessDoc.baseline("seed"),
            _tasks(),
            _wm(provider),
            provider,
            ProviderDeltaProposer(provider),
            GoldJudge(provider),
            harness_backend="e2b",
        )


def test_local_backend_rejects_parallel_pi_node_scoring() -> None:
    """Local pi runtimes are single-episode (one port/workdir/channel): local stays sequential.

    The guard fires per-doc at scoring time, before any rollout, so a parallel request fails
    loudly instead of colliding episodes.
    """
    provider = RoleProvider()
    with pytest.raises(ValueError, match="one episode at a time"):
        create_harness(
            "winner",
            _pi_seed(),
            _tasks(),
            _wm(provider),
            provider,
            ProviderDeltaProposer(provider),
            GoldJudge(provider),
            eval_concurrency=2,
        )


def test_e2b_backend_scores_against_the_world_model_through_the_shared_pool(
    monkeypatch: pytest.MonkeyPatch, fake_pool_cls: type[_FakePool]
) -> None:
    """harness_backend='e2b': the pi process lives in pooled sandboxes, the env stays the WM.

    The pool is faked (its channels play the runner peer), `evaluate_closed_loop` is the real
    one wrapped only to record the concurrency each eval was asked for — so every scripted
    tool_request is really brokered by `RunnerLink` into `WorldModelEnvironment`, and the
    tool_response carries the world model's marker reply ("ok" from the RoleProvider env role).
    """
    provider = RoleProvider()  # default judge passes on the runner's "done-verified" answer
    concurrencies: list[int] = []
    real_evaluate = create_module.evaluate_closed_loop

    def spying_evaluate(
        tasks: list[TaskSpec],
        world_model: WorldModel,
        agent_provider: Provider,
        judge: GoldJudge,
        *,
        label: str,
        k: int,
        concurrency: int,
        runtime: Runtime | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> ClosedLoopReport:
        concurrencies.append(concurrency)
        return real_evaluate(
            tasks,
            world_model,
            agent_provider,
            judge,
            label=label,
            k=k,
            concurrency=concurrency,
            runtime=runtime,
            should_cancel=should_cancel,
        )

    monkeypatch.setattr(create_module, "evaluate_closed_loop", spying_evaluate)

    result = create_harness(
        "winner",
        _pi_seed(),
        _tasks(),
        _wm(provider),
        provider,
        ProviderDeltaProposer(provider),
        GoldJudge(provider),
        iterations=0,  # the seed eval alone exercises the whole scoring path
        k=3,
        harness_backend="e2b",
        e2b_template="tmpl-1",
        e2b_metadata={"optimizer_run_id": "run-1", "purpose": "evaluation"},
    )

    # The judge passed the runner's submitted answer: the eval genuinely ran end to end.
    assert result.best_score == 1.0
    assert concurrencies == [0]  # e2b default: every (task, attempt) cell at once
    [pool] = fake_pool_cls.instances  # ONE shared pool for the whole search
    assert pool.template == "tmpl-1"
    assert pool.metadata == {"optimizer_run_id": "run-1", "purpose": "evaluation"}
    # One finally owns teardown and mutates the returned model with the finalized meter.
    assert pool.closes == 1
    assert result.sandbox_usage is not None
    assert result.sandbox_usage.count == len(pool.channels)  # the fake meters per acquire
    assert len(pool.channels) == 3  # one pooled runner episode per (task, attempt) cell
    assert pool.releases == [True, True, True]  # healthy episodes return their sandboxes
    for channel in pool.channels:
        kinds = [f.get("type") for f in channel.sent]
        assert kinds == ["episode_start", "tool_response"]
        response = channel.sent[1]
        # The WORLD MODEL answered the tool: "ok" is the RoleProvider env-role marker reply.
        assert response.get("content") == "ok" and response.get("is_error") is False


def test_e2b_pool_is_closed_exactly_once_when_the_search_raises(
    monkeypatch: pytest.MonkeyPatch, fake_pool_cls: type[_FakePool]
) -> None:
    provider = RoleProvider()
    observed_usage: list[SandboxUsage] = []

    def exploding_evaluate(*args: object, **kwargs: object) -> ClosedLoopReport:
        raise RuntimeError("boom mid-eval")

    monkeypatch.setattr(create_module, "evaluate_closed_loop", exploding_evaluate)
    with pytest.raises(RuntimeError, match="boom mid-eval"):
        create_harness(
            "winner",
            _pi_seed(),
            _tasks(),
            _wm(provider),
            provider,
            ProviderDeltaProposer(provider),
            GoldJudge(provider),
            harness_backend="e2b",
            on_sandbox_usage=observed_usage.append,
        )
    [pool] = fake_pool_cls.instances
    assert pool.closes == 1  # the try/finally tears the pool down even on failure
    assert observed_usage == [SandboxUsage(count=0, seconds=0.0)]


@pytest.mark.parametrize("duplicate_split", ["discovery", "holdout"])
def test_e2b_pool_is_closed_when_scorer_construction_rejects_duplicate_task_ids(
    fake_pool_cls: type[_FakePool], duplicate_split: str
) -> None:
    """Task validation after pool creation remains inside the pool's lifecycle boundary."""
    provider = RoleProvider()
    duplicate_tasks = [*_tasks(), *_tasks()]
    discovery = duplicate_tasks if duplicate_split == "discovery" else _tasks()
    holdout = duplicate_tasks if duplicate_split == "holdout" else None
    observed_usage: list[SandboxUsage] = []

    with pytest.raises(ValueError, match="closed-loop search task ids must be unique"):
        create_harness(
            "winner",
            _pi_seed(),
            discovery,
            _wm(provider),
            provider,
            ProviderDeltaProposer(provider),
            GoldJudge(provider),
            iterations=0,
            holdout=holdout,
            harness_backend="e2b",
            on_sandbox_usage=observed_usage.append,
        )

    [pool] = fake_pool_cls.instances
    assert pool.closes == 1
    assert observed_usage == [SandboxUsage(count=0, seconds=0.0)]


def test_e2b_cleanup_failure_replaces_cancellation_and_withholds_final_usage(
    monkeypatch: pytest.MonkeyPatch, fake_pool_cls: type[_FakePool]
) -> None:
    """Cancellation cannot look clean when evaluator release remains unproven."""
    from wmh.harness.e2b_sandbox import SandboxCleanupError
    from wmh.harness.runtime import RuntimeCancelled

    provider = RoleProvider()
    observed_usage: list[SandboxUsage] = []

    def cancelled_evaluate(*args: object, **kwargs: object) -> ClosedLoopReport:
        del args, kwargs
        [pool] = fake_pool_cls.instances
        pool.close_failures = 1
        raise RuntimeCancelled("runtime episode cancelled")

    monkeypatch.setattr(create_module, "evaluate_closed_loop", cancelled_evaluate)

    with pytest.raises(SandboxCleanupError, match="cleanup unproven") as raised:
        create_harness(
            "winner",
            _pi_seed(),
            _tasks(),
            _wm(provider),
            provider,
            ProviderDeltaProposer(provider),
            GoldJudge(provider),
            harness_backend="e2b",
            on_sandbox_usage=observed_usage.append,
        )

    assert isinstance(raised.value.__context__, HarnessSearchCancelled)
    assert observed_usage == []


def test_runtime_cancellation_aborts_the_wave_without_judging_and_closes_pool(
    monkeypatch: pytest.MonkeyPatch, fake_pool_cls: type[_FakePool]
) -> None:
    from wmh.harness.pi_e2b import E2BPiRuntime
    from wmh.harness.runtime import RuntimeCancelled

    provider = RoleProvider()
    callback_seen = False

    def should_cancel() -> bool:
        return False

    def cancelled_evaluate(*args: object, **kwargs: object) -> ClosedLoopReport:
        nonlocal callback_seen
        runtime = kwargs.get("runtime")
        assert isinstance(runtime, E2BPiRuntime)
        callback_seen = runtime._should_cancel is should_cancel  # noqa: SLF001
        raise RuntimeCancelled("runtime episode cancelled")

    monkeypatch.setattr(create_module, "evaluate_closed_loop", cancelled_evaluate)

    with pytest.raises(HarnessSearchCancelled, match="cancelled") as raised:
        create_harness(
            "winner",
            _pi_seed(),
            _tasks(),
            _wm(provider),
            provider,
            ProviderDeltaProposer(provider),
            GoldJudge(provider),
            harness_backend="e2b",
            should_cancel=should_cancel,
        )

    assert callback_seen
    [pool] = fake_pool_cls.instances
    assert pool.closes == 1
    assert raised.value.sandbox_usage == SandboxUsage(count=0, seconds=0.0)


def test_cancellation_carries_completed_and_partial_wave_worker_usage(
    monkeypatch: pytest.MonkeyPatch, fake_pool_cls: type[_FakePool]
) -> None:
    """The public cancellation result owns all worker spend without a partial CreateResult."""
    from wmh.harness.runtime import RuntimeCancelled, TokenUsage

    seed = _pi_seed()
    provider = RoleProvider(meta_reply=_meta_reply(seed, _CAREFUL_PROMPT))
    evaluate_calls = 0

    def cancel_second_wave(*args: object, **kwargs: object) -> ClosedLoopReport:
        nonlocal evaluate_calls
        del args
        evaluate_calls += 1
        if evaluate_calls == 1:
            k = kwargs.get("k", 3)
            assert isinstance(k, int)
            return _canned_report(0.5, k=k).model_copy(
                update={"worker_usage": TokenUsage(input_tokens=100, output_tokens=10, calls=2)}
            )
        raise RuntimeCancelled(
            "runtime episode cancelled",
            worker_usage=TokenUsage(input_tokens=7, output_tokens=2, calls=1),
        )

    monkeypatch.setattr(create_module, "evaluate_closed_loop", cancel_second_wave)

    with pytest.raises(HarnessSearchCancelled, match="cancelled") as raised:
        create_harness(
            "winner",
            seed,
            _tasks(),
            _wm(provider),
            provider,
            ProviderDeltaProposer(provider),
            GoldJudge(provider),
            iterations=1,
            k=2,
            harness_backend="e2b",
        )

    assert evaluate_calls == 2
    assert raised.value.worker_usage == TokenUsage(input_tokens=107, output_tokens=12, calls=3)
    assert raised.value.sandbox_usage == SandboxUsage(count=0, seconds=0.0)
    [pool] = fake_pool_cls.instances
    assert pool.closes == 1


def test_e2b_pool_retires_idle_runners_once_per_proposal_batch(
    monkeypatch: pytest.MonkeyPatch, fake_pool_cls: type[_FakePool]
) -> None:
    """Round boundaries rotate eval streams without rotating between sibling proposals."""
    provider = RoleProvider()
    monkeypatch.setattr(
        create_module,
        "evaluate_closed_loop",
        lambda *a, **k: _canned_report(0.5, k=k.get("k", 3)),
    )

    result = create_harness(
        "winner",
        _pi_seed(),
        _tasks(),
        _wm(provider),
        provider,
        ProviderDeltaProposer(provider),
        GoldJudge(provider),
        iterations=2,
        proposal_batch_size=3,
        k=2,
        harness_backend="e2b",
    )

    assert result.iterations == 2 and len(result.proposal_records) == 6
    [pool] = fake_pool_cls.instances
    assert pool.retire_idle_calls == 2  # once per batch, never between its three siblings


def test_eval_concurrency_overrides_both_backend_defaults(
    monkeypatch: pytest.MonkeyPatch, fake_pool_cls: type[_FakePool]
) -> None:
    """An explicit eval_concurrency reaches the scorer; unset local keeps the sequential default."""
    provider = RoleProvider()
    concurrencies: list[int] = []

    def fake_evaluate(
        tasks: list[TaskSpec],
        world_model: WorldModel,
        agent_provider: Provider,
        judge: GoldJudge,
        *,
        label: str,
        k: int,
        concurrency: int,
        runtime: Runtime | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> ClosedLoopReport:
        del should_cancel
        concurrencies.append(concurrency)
        return _canned_report(1.0, k=k)

    monkeypatch.setattr(create_module, "evaluate_closed_loop", fake_evaluate)

    def run(seed: HarnessDoc, *, harness_backend: str, eval_concurrency: int | None) -> None:
        create_harness(
            "winner",
            seed,
            _tasks(),
            _wm(provider),
            provider,
            ProviderDeltaProposer(provider),
            GoldJudge(provider),
            iterations=0,  # score the seed only: one eval call per run
            harness_backend="local" if harness_backend == "local" else "e2b",
            eval_concurrency=eval_concurrency,
        )

    run(HarnessDoc.baseline("seed"), harness_backend="local", eval_concurrency=None)
    run(HarnessDoc.baseline("seed"), harness_backend="local", eval_concurrency=4)
    run(_pi_seed(), harness_backend="e2b", eval_concurrency=2)
    assert concurrencies == [1, 4, 2]  # local defaults sequential; explicit caps pass through


def test_create_sums_worker_usage_across_score_waves(
    monkeypatch: pytest.MonkeyPatch, fake_pool_cls: type[_FakePool]
) -> None:
    """CreateResult.worker_usage is the sum of every score wave's report.worker_usage.

    Regression: the pi worker path self-meters tokens on each ClosedLoopReport, but the search
    dropped them on the floor (the accumulator list was declared and summed, never appended to),
    so CreateResult.worker_usage came back None and the platform's worker cost booked $0.00
    despite real agent LLM spend. Seed + one screened child = two waves here.
    """
    from wmh.harness.runtime import TokenUsage

    provider = RoleProvider(meta_reply=_meta_reply(HarnessDoc.baseline("seed"), _CAREFUL_PROMPT))

    def fake_evaluate(
        tasks: list[TaskSpec],
        world_model: WorldModel,
        agent_provider: Provider,
        judge: GoldJudge,
        *,
        label: str,
        k: int,
        concurrency: int,
        runtime: Runtime | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> ClosedLoopReport:
        del should_cancel
        report = _canned_report(0.5, k=k)
        return report.model_copy(
            update={"worker_usage": TokenUsage(input_tokens=100, output_tokens=10, calls=2)}
        )

    monkeypatch.setattr(create_module, "evaluate_closed_loop", fake_evaluate)

    result = create_harness(
        "winner",
        _pi_seed(),
        _tasks(),
        _wm(provider),
        provider,
        ProviderDeltaProposer(provider),
        GoldJudge(provider),
        iterations=1,
        k=2,
        harness_backend="e2b",
    )

    # Before the fix this was None (each wave's usage was never accumulated); now it sums
    # every wave. At least the seed wave ran (2 calls / 100in / 10out per wave), and the totals
    # hold that exact per-call ratio however many waves the search took.
    assert result.worker_usage is not None
    assert result.worker_usage.calls >= 2
    assert result.worker_usage.calls % 2 == 0
    assert result.worker_usage.input_tokens == 50 * result.worker_usage.calls
    assert result.worker_usage.output_tokens == 5 * result.worker_usage.calls


def test_create_worker_usage_is_none_when_no_wave_reports_it(
    monkeypatch: pytest.MonkeyPatch, fake_pool_cls: type[_FakePool]
) -> None:
    """Local runtimes don't self-meter: worker_usage stays None (never a zero-token TokenUsage)."""
    provider = RoleProvider()

    monkeypatch.setattr(
        create_module,
        "evaluate_closed_loop",
        lambda *a, **k: _canned_report(1.0, k=k.get("k", 3)),
    )

    result = create_harness(
        "winner",
        HarnessDoc.baseline("seed"),
        _tasks(),
        _wm(provider),
        provider,
        ProviderDeltaProposer(provider),
        GoldJudge(provider),
        iterations=0,
        harness_backend="local",
    )

    assert result.worker_usage is None


def test_e2b_rejects_a_delta_that_abandons_the_pi_runtime(
    monkeypatch: pytest.MonkeyPatch, fake_pool_cls: type[_FakePool]
) -> None:
    """A candidate that flips param:runtime-kind is archived invalid, not a run-aborting raise.

    Regression (Greptile P1): `doc.runtime(backend="e2b")` raises for non-pi-node docs; a meta
    proposal that rewrote the runtime-kind surface escaped the invalid-delta handling and
    aborted the whole search.
    """
    seed = _pi_seed()
    kind = seed.surface("param:runtime-kind")
    assert kind is not None
    escape = json.dumps(
        {
            "expected_effect": "run in-process instead",
            "preconditions": {"param:runtime-kind": kind.content_hash},
            "ops": [
                {
                    "op": "replace",
                    "surface_id": "param:runtime-kind",
                    "content": "kit-python",
                    "rationale": "abandon the pi runtime",
                }
            ],
        }
    )
    provider = RoleProvider(meta_reply=escape)
    monkeypatch.setattr(
        create_module,
        "evaluate_closed_loop",
        lambda *a, **k: _canned_report(0.5, k=k.get("k", 3)),
    )

    result = create_harness(
        "winner",
        seed,
        _tasks(),
        _wm(provider),
        provider,
        ProviderDeltaProposer(provider),
        GoldJudge(provider),
        iterations=1,
        k=2,
        harness_backend="e2b",
    )

    assert result.skipped == 1  # the escape delta was rejected, not fatal
    [delta] = result.archive.deltas
    assert delta.verdict is not None and not delta.verdict.accepted
    assert "pi-node only" in delta.verdict.reason
    assert result.best_score == 0.5  # the seed stayed champion and the search finished


class _MetaExplodingProvider(RoleProvider):
    """RoleProvider whose meta-agent calls raise (an API rejecting the request outright)."""

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Completion:
        if "meta-agent improving an agent harness" in system:
            msg = "max_tokens above model output limit"
            raise RuntimeError(msg)
        return super().complete(system, messages, temperature=temperature, max_tokens=max_tokens)


def test_proposer_call_failure_skips_the_iteration_not_the_run() -> None:
    """A meta-provider exception (output-cap rejection, rate limit) costs one iteration.

    Same contract as an unusable reply, but narrated with the error; the search must not
    abort on the first provider fault.
    """
    provider = _MetaExplodingProvider()
    notes: list[str] = []
    result = _run(provider, iterations=2, on_note=notes.append)

    assert result.skipped == 2
    assert result.best.name == "winner"  # the seed still wins; the run completed
    assert len(notes) == 2
    assert all("proposer call failed" in note for note in notes)
    assert all("max_tokens above model output limit" in note for note in notes)
    assert [(r.iteration, r.outcome) for r in result.proposal_records] == [
        (1, "proposer_error"),
        (2, "proposer_error"),
    ]


class _SequencedMetaProvider(RoleProvider):
    """RoleProvider whose meta-agent replies follow a script, one per proposal call."""

    def __init__(self, replies: list[str]) -> None:
        super().__init__()
        self._replies = replies

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Completion:
        if "meta-agent improving an agent harness" in system:
            self.meta_users.append(messages[-1].content)
            reply = self._replies[min(len(self.meta_users) - 1, len(self._replies) - 1)]
            return Completion(text=reply)
        return super().complete(system, messages, temperature=temperature, max_tokens=max_tokens)


class _RankedMetaProvider(_SequencedMetaProvider):
    """Give weak and strong proposal prompts distinct, deterministic task scores."""

    def __init__(self, replies: list[str]) -> None:
        super().__init__(replies)
        self._judge_fn = lambda user: (
            "done-strong" in user or ("done-weak" in user and "task one" in user)
        )

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Completion:
        special = (
            "grade whether an agent completed a task",
            "meta-agent improving an agent harness",
            "You ARE the environment",
        )
        if not any(marker in system for marker in special):
            if "strong agent" in system:
                answer = "done-strong"
            elif "weak agent" in system:
                answer = "done-weak"
            else:
                answer = "done"
            return Completion(text=json.dumps({"tool": "submit", "arguments": {"answer": answer}}))
        return super().complete(
            system,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )


class _ParentRecordingProposer:
    """Propose against the live parent and remember each iteration's parent hash."""

    def __init__(self) -> None:
        self.parent_hashes: list[str] = []

    def propose_batch(
        self,
        parent: HarnessDoc,
        trigger: FailureSignature,
        evidence: str,
        *,
        history: list[HarnessDelta],
        count: int,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[HarnessDelta | ProposalFailure | None]:
        del evidence, history, should_cancel
        self.parent_hashes.append(parent.doc_hash)
        prompt = f"{_CAREFUL_PROMPT} Iteration {len(self.parent_hashes)}."
        proposal = parse_delta(parent, trigger, _meta_reply(parent, prompt))
        assert proposal is not None and count == 1
        return [proposal]


class _FeedbackRecordingProposer:
    """Return scripted siblings and retain the final evaluation feedback for each."""

    def __init__(self, prompts: list[str]) -> None:
        self.prompts = prompts
        self.feedback: list[tuple[str, str, str]] = []

    def propose_batch(
        self,
        parent: HarnessDoc,
        trigger: FailureSignature,
        evidence: str,
        *,
        history: list[HarnessDelta],
        count: int,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[HarnessDelta | ProposalFailure | None]:
        del evidence, history, should_cancel
        assert count == len(self.prompts)
        proposals: list[HarnessDelta | ProposalFailure | None] = [
            parse_delta(parent, trigger, _meta_reply(parent, prompt)) for prompt in self.prompts
        ]
        assert all(proposal is not None for proposal in proposals)
        return proposals

    def record_evaluation(self, delta: HarnessDelta, *, stage: str, content: str) -> None:
        self.feedback.append((delta.delta_id, stage, content))


def test_proposal_batch_is_generated_before_siblings_are_evaluated() -> None:
    """One iteration expands one parent into independently tracked sibling candidates."""
    seed = HarnessDoc.baseline("seed")
    provider = _SequencedMetaProvider(
        [
            _meta_reply(seed, _CAREFUL_PROMPT),
            _meta_reply(seed, f"{_CAREFUL_PROMPT} Double-check the result."),
        ]
    )

    result = _run(provider, iterations=1, proposal_batch_size=2)

    assert len(provider.meta_users) == 2
    assert [(record.iteration, record.proposal_index) for record in result.proposal_records] == [
        (1, 1),
        (1, 2),
    ]
    assert [record.candidate for record in result.proposal_records] == [
        "winner-i1-p1",
        "winner-i1-p2",
    ]
    assert len(result.archive.deltas) == 2


def test_iteration_batch_commits_one_winner_and_one_progress_point() -> None:
    """Three eligible siblings yield one deterministic winner and one champion update."""
    seed = HarnessDoc.baseline("seed")
    provider = _SequencedMetaProvider(
        [_meta_reply(seed, f"{_CAREFUL_PROMPT} Candidate {index}.") for index in range(1, 4)]
    )
    progress: list[tuple[int, str, float, bool]] = []
    proposals: list[ProposalRecord] = []
    crowned: list[str] = []

    result = create_harness(
        "winner",
        seed,
        _tasks(),
        _wm(provider),
        provider,
        ProviderDeltaProposer(provider),
        GoldJudge(provider),
        iterations=1,
        proposal_batch_size=3,
        on_progress=lambda i, n, r, a: progress.append((i, n, r, a)),
        on_proposal=proposals.append,
        on_accept=lambda doc, delta, score: crowned.append(doc.name),
    )

    assert result.iterations == 1
    assert len(result.proposal_records) == 3
    assert [(record.iteration, record.proposal_index) for record in result.proposal_records] == [
        (1, 1),
        (1, 2),
        (1, 3),
    ]
    assert [record.gate_eligible for record in result.proposal_records] == [True, True, True]
    assert [record.selected for record in result.proposal_records] == [True, False, False]
    assert len(proposals) == 3
    assert crowned == ["winner-i1-p1"]
    assert progress == [
        (0, "seed", 0.0, True),
        (1, "winner-i1-p1", 1.0, True),
    ]
    assert [delta.verdict.accepted for delta in result.archive.deltas if delta.verdict] == [
        True,
        False,
        False,
    ]


def test_duplicate_sibling_is_archived_without_duplicate_evaluation() -> None:
    """The search boundary rejects duplicate sibling deltas before spending another screen."""
    seed = HarnessDoc.baseline("seed")
    provider = RoleProvider(meta_reply=_meta_reply(seed, _CAREFUL_PROMPT))

    result = _run(provider, iterations=1, proposal_batch_size=2)

    assert result.skipped == 1
    assert len(result.archive.deltas) == 2
    assert [record.outcome for record in result.proposal_records] == ["scored", "invalid"]
    assert [record.selected for record in result.proposal_records] == [True, False]
    assert "already-proposed" in (result.proposal_records[1].reason or "")
    assert len(result.reports) == 2  # seed plus the first sibling, never the duplicate


def test_cancellation_during_later_sibling_commits_no_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later-sibling cancellation cannot publish an earlier sibling as the winner."""
    seed = HarnessDoc.baseline("seed")
    provider = _SequencedMetaProvider(
        [
            _meta_reply(seed, _CAREFUL_PROMPT),
            _meta_reply(seed, f"{_CAREFUL_PROMPT} Another candidate."),
        ]
    )
    cancelled = False
    gate_delta = create_module.gate_score_delta

    def cancel_after_first_gate(
        delta: HarnessDelta,
        *,
        child: HarnessScoreReport,
        champion: HarnessScoreReport,
        best_full: float,
        suite: list[str],
        child_holdout: HarnessScoreReport | None = None,
        champion_holdout: HarnessScoreReport | None = None,
    ) -> GateRecord:
        nonlocal cancelled
        verdict = gate_delta(
            delta,
            child=child,
            champion=champion,
            best_full=best_full,
            suite=suite,
            child_holdout=child_holdout,
            champion_holdout=champion_holdout,
        )
        cancelled = True
        return verdict

    monkeypatch.setattr(create_module, "gate_score_delta", cancel_after_first_gate)
    progress: list[tuple[int, str, float, bool]] = []
    crowned: list[str] = []
    proposals: list[ProposalRecord] = []

    with pytest.raises(HarnessSearchCancelled, match="cancelled"):
        _run(
            provider,
            proposal_batch_size=2,
            should_cancel=lambda: cancelled,
            on_progress=lambda i, n, r, a: progress.append((i, n, r, a)),
            on_accept=lambda doc, delta, score: crowned.append(doc.doc_hash),
            on_proposal=proposals.append,
        )

    assert progress == [(0, "seed", 0.0, True)]
    assert crowned == []
    assert proposals == []


@pytest.mark.parametrize(
    ("prompts", "selected"),
    [
        (["You are a weak agent.", "You are a strong agent."], [False, True]),
        (["You are a strong agent.", "You are a weak agent."], [True, False]),
    ],
)
def test_iteration_selects_best_eligible_score_independent_of_proposal_order(
    prompts: list[str], selected: list[bool]
) -> None:
    """Full success outranks assertion fraction and proposal order within a frozen batch."""
    seed = HarnessDoc.baseline("seed")
    provider = _RankedMetaProvider([_meta_reply(seed, prompt) for prompt in prompts])
    tasks = [
        TaskSpec(task_id="t1", instruction="task one", gold=["the work completed"]),
        TaskSpec(task_id="t2", instruction="task two", gold=["the work completed"]),
    ]
    crowned: list[str] = []

    result = create_harness(
        "winner",
        seed,
        tasks,
        _wm(provider),
        provider,
        ProviderDeltaProposer(provider),
        GoldJudge(provider),
        iterations=1,
        proposal_batch_size=2,
        k=1,
        on_accept=lambda doc, delta, score: crowned.append(doc.system_prompt()),
    )

    assert result.best_score == 1.0
    assert result.best.system_prompt() == "You are a strong agent."
    assert [record.gate_eligible for record in result.proposal_records] == [True, True]
    assert [record.selected for record in result.proposal_records] == selected
    assert crowned == ["You are a strong agent."]
    accepted = result.archive.accepted()
    assert len(accepted) == 1
    winner_hash = next(
        record.candidate_doc_hash for record in result.proposal_records if record.selected
    )
    assert accepted[0].child_doc_hash == winner_hash
    assert winner_hash is not None
    assert result.archive.reconstruct(winner_hash).system_prompt() == "You are a strong agent."
    loser_hash = next(
        record.candidate_doc_hash for record in result.proposal_records if not record.selected
    )
    assert loser_hash is not None
    with pytest.raises(ValueError, match="not in this archive"):
        result.archive.reconstruct(loser_hash)
    loser_delta = next(
        delta
        for delta in result.archive.deltas
        if delta.verdict is not None and not delta.verdict.accepted
    )
    assert loser_delta.verdict is not None
    assert "gate eligible but not selected" in loser_delta.verdict.reason


def test_full_feedback_records_final_batch_selection() -> None:
    """Eligible losers teach the proposer that ranking, not the gate, rejected them."""
    provider = _RankedMetaProvider([])
    proposer = _FeedbackRecordingProposer(["You are a weak agent.", "You are a strong agent."])
    tasks = [
        TaskSpec(task_id="t1", instruction="task one", gold=["the work completed"]),
        TaskSpec(task_id="t2", instruction="task two", gold=["the work completed"]),
    ]

    result = create_harness(
        "winner",
        HarnessDoc.baseline("seed"),
        tasks,
        _wm(provider),
        provider,
        proposer,
        GoldJudge(provider),
        iterations=1,
        proposal_batch_size=2,
        k=1,
    )

    loser = next(
        delta
        for delta in result.archive.deltas
        if delta.verdict is not None and not delta.verdict.accepted
    )
    winner = result.archive.accepted()[0]
    loser_feedback = next(
        content
        for delta_id, stage, content in proposer.feedback
        if delta_id == loser.delta_id and stage == "full"
    )
    winner_feedback = next(
        content
        for delta_id, stage, content in proposer.feedback
        if delta_id == winner.delta_id and stage == "full"
    )
    assert "gate eligible but not selected" in loser_feedback
    assert "gate eligible but not selected" not in winner_feedback


def test_sibling_holdout_gates_use_frozen_iteration_champion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A selected-looking first sibling cannot become the second sibling's gate baseline."""
    seed = HarnessDoc.baseline("seed")
    provider = _SequencedMetaProvider(
        [
            _meta_reply(seed, _CAREFUL_PROMPT),
            _meta_reply(seed, f"{_CAREFUL_PROMPT} Another candidate."),
        ]
    )
    provider._judge_fn = lambda user: (
        True if "the holdout task" in user else "done-verified" in user
    )
    holdout = [TaskSpec(task_id="h1", instruction="the holdout task", gold=["still works"])]
    gate_delta = create_module.gate_score_delta
    holdout_gate_champions: list[tuple[float, float]] = []

    def capture_gate_baselines(
        delta: HarnessDelta,
        *,
        child: HarnessScoreReport,
        champion: HarnessScoreReport,
        best_full: float,
        suite: list[str],
        child_holdout: HarnessScoreReport | None = None,
        champion_holdout: HarnessScoreReport | None = None,
    ) -> GateRecord:
        if child_holdout is not None and champion_holdout is not None:
            holdout_gate_champions.append((champion.score, champion_holdout.score))
        return gate_delta(
            delta,
            child=child,
            champion=champion,
            best_full=best_full,
            suite=suite,
            child_holdout=child_holdout,
            champion_holdout=champion_holdout,
        )

    monkeypatch.setattr(create_module, "gate_score_delta", capture_gate_baselines)

    result = _run(provider, proposal_batch_size=2, holdout=holdout)

    assert holdout_gate_champions == [(0.0, 1.0), (0.0, 1.0)]
    assert [record.gate_eligible for record in result.proposal_records] == [True, True]
    assert [record.selected for record in result.proposal_records] == [True, False]


def test_next_iteration_proposes_from_previous_iteration_winner() -> None:
    """The selected winner is the next parent, with no stepping-stone parent pool."""
    seed = HarnessDoc.baseline("seed")
    provider = RoleProvider()
    proposer = _ParentRecordingProposer()

    result = create_harness(
        "winner",
        seed,
        _tasks(),
        _wm(provider),
        provider,
        proposer,
        GoldJudge(provider),
        iterations=2,
        proposal_batch_size=1,
    )

    assert len(result.archive.accepted()) == 2
    first_winner_hash = result.proposal_records[0].candidate_doc_hash
    assert proposer.parent_hashes == [seed.doc_hash, first_winner_hash]
    assert [record.selected for record in result.proposal_records] == [True, True]


def test_on_accept_delivers_the_new_champion_the_moment_it_is_crowned() -> None:
    """Accepted champions stream out live, so callers can persist them in real time."""
    seed = HarnessDoc.baseline("seed")
    provider = RoleProvider(meta_reply=_meta_reply(seed, _CAREFUL_PROMPT))
    crowned: list[tuple[str, bool, float]] = []
    result = _run(
        provider,
        on_accept=lambda doc, delta, score: crowned.append(
            (doc.system_prompt(), delta.verdict is not None and delta.verdict.accepted, score)
        ),
    )

    assert result.best_score == 1.0
    [(prompt, verdict_accepted, score)] = crowned
    assert prompt == _CAREFUL_PROMPT  # the actual champion doc, not a name or hash
    assert verdict_accepted is True  # the delta arrives with its verdict already attached
    assert score == 1.0


def test_dead_iteration_ends_early_and_the_search_moves_on() -> None:
    """A dead proposal costs its iteration cheaply; the next iteration proceeds normally.

    Iteration 1's proposal is unusable; iteration 2 proposes the genuine fix. Both appear
    in the records, and the scored one keeps its own iteration number.
    """
    seed = HarnessDoc.baseline("seed")
    provider = _SequencedMetaProvider(["garbage, not json", _meta_reply(seed, _CAREFUL_PROMPT)])
    progress: list[tuple[int, str, float, bool]] = []
    result = _run(
        provider, iterations=2, on_progress=lambda i, n, r, a: progress.append((i, n, r, a))
    )

    assert result.skipped == 1
    assert result.best_score == 1.0
    assert result.best.system_prompt() == _CAREFUL_PROMPT
    assert [e[0] for e in progress] == [0, 1, 2]
    assert progress[1] == (1, "seed", 0.0, False)
    assert [(r.iteration, r.outcome) for r in result.proposal_records] == [
        (1, "unusable"),
        (2, "scored"),
    ]
    scored = result.proposal_records[-1]
    assert scored.selected is True and scored.score == 1.0
    assert scored.candidate == "winner-i2-p1"


def test_skipped_proposals_narrate_through_on_note() -> None:
    """Every proposal that dies before scoring narrates itself.

    Regression: a run whose proposals were all unusable (e.g. truncated meta replies on huge
    pi code surfaces) emitted NO progress events at all; five iterations looked like one.
    """
    provider = RoleProvider(meta_reply="truncated garbage that is not json")
    notes: list[str] = []
    result = _run(provider, iterations=3, on_note=notes.append)

    assert result.skipped == 3
    assert [note.split(":")[0] for note in notes] == [
        "iteration 1/3",
        "iteration 2/3",
        "iteration 3/3",
    ]
    assert all("proposal unusable" in note for note in notes)


def test_dead_notes_precede_final_proposal_records_and_iteration_checkpoint() -> None:
    """Eager diagnostics precede the batch's ordered records and champion checkpoint."""
    events: list[str] = []

    result = _run(
        RoleProvider(meta_reply="truncated garbage that is not json"),
        proposal_batch_size=2,
        on_progress=lambda iteration, name, score, changed: events.append(f"progress:{iteration}"),
        on_note=lambda message: events.append(f"note:{message.split(':', 1)[0]}"),
        on_proposal=lambda record: events.append(f"proposal:{record.proposal_index}"),
    )

    assert result.skipped == 2
    assert events == [
        "progress:0",
        "note:iteration 1/1 proposal 1/2",
        "note:iteration 1/1 proposal 2/2",
        "proposal:1",
        "proposal:2",
        "progress:1",
    ]
