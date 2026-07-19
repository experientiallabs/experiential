"""Tests for provider and persistent-project delta proposers."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable, Collection
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
from llm_waterfall import ChatRequest, ChatResponse
from llm_waterfall.types import ChatMessage, ChatUsage

from wmh.agents.default import default_agent
from wmh.agents.meta import meta_agent
from wmh.agents.project import AgentProjectRun
from wmh.core.types import JsonObject
from wmh.harness.cost import (
    ProviderCostBinding,
    SearchComponentCostBinding,
    SearchComponentCostRuntime,
    SearchComponentRole,
    SearchCostBinding,
    SearchCostRuntime,
    TimedResourceCostBinding,
)
from wmh.harness.delta import FailureSignature, GateRecord, HarnessDelta, apply_delta
from wmh.harness.doc import HarnessDoc, Surface, SurfaceKind
from wmh.harness.e2b_sandbox import SandboxCleanupError
from wmh.harness.mutate import parse_delta
from wmh.harness.proposer import (
    ProjectDeltaProposer as _ProductionProjectDeltaProposer,
)
from wmh.harness.proposer import ProposalFailure, ProviderDeltaProposer
from wmh.harness.runtime import HarnessSearchCancelled, TokenUsage
from wmh.harness.scoring import (
    HarnessScoreArchive,
    HarnessScoreReport,
    ScoreArchiveTier,
    ScoreArchiveVisibility,
    ScoreRequest,
    ScoreRunHealth,
    TaskScore,
    canonical_score_json,
)
from wmh.providers.base import (
    Completion,
    Message,
    ProviderConfig,
    ProviderKind,
    ToolCallingProvider,
    VerifyResult,
)
from wmh.providers.base import TokenUsage as ProviderTokenUsage
from wmh.providers.process_worker import ProviderWorkerCleanupError
from wmh.providers.receipt import (
    ProviderResponseIdentity,
    ProviderResponseIdentityError,
    build_chat_provider_receipt,
    freeze_provider_response_identity,
)
from wmh.tracking._testing import (
    synthetic_provider_cost_meter,
    synthetic_tariff_provenance,
)
from wmh.tracking.budget import (
    BudgetBreachError,
    BudgetedProvider,
    BudgetExceededError,
    BudgetIntegrityError,
    BudgetPolicy,
    BudgetScope,
    ReservationStatus,
    SpendLedger,
    TimedResourceCostMeter,
    UnpricedProviderUsageError,
    bind_budget_account,
    bind_timed_resource_account,
    bootstrap_budget_ledger,
)
from wmh.tracking.rate_limit import (
    E2B_SANDBOX_CREATE_RATE_POLICY,
    ExternalDispatchRateAuthority,
    ExternalDispatchRateIntegrityError,
)


def _trigger() -> FailureSignature:
    return FailureSignature(mechanism="verification", task_ids=["t1"])


def _payload(parent: HarnessDoc, content: str) -> str:
    core = parent.surface("prompt:core")
    assert core is not None
    return json.dumps(
        {
            "expected_effect": "t1 passes",
            "preconditions": {"prompt:core": core.content_hash},
            "ops": [
                {
                    "op": "replace",
                    "surface_id": "prompt:core",
                    "content": content,
                    "rationale": "verify before submit",
                }
            ],
        }
    )


def _skill_payload(content: str, *, slug: str = "parse-json") -> str:
    return json.dumps(
        {
            "expected_effect": "the agent parses JSON responses reliably",
            "preconditions": {},
            "ops": [
                {
                    "op": "add",
                    "surface_id": f"skill:{slug}",
                    "kind": "skill",
                    "content": content,
                    "rationale": "teach response parsing without changing unrelated behavior",
                }
            ],
        }
    )


def _proposal_failure_reason(proposal: object) -> str:
    assert isinstance(proposal, ProposalFailure)
    return proposal.reason


class _Provider:
    config = ProviderConfig(kind=ProviderKind.BEDROCK, model="m", region="test-region")
    paid_request_attempts = 1

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0

    def complete(self, system: str, messages: list[Message], **kwargs: object) -> Completion:
        del system, messages, kwargs
        self.calls += 1
        return Completion(text=self.reply)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self) -> VerifyResult:
        return VerifyResult(ok=True, kind=ProviderKind.BEDROCK, model="m")

    def complete_chat(self, request: object) -> ChatResponse:
        del request
        raise AssertionError("fake project never calls the provider")


class ProjectDeltaProposer(_ProductionProjectDeltaProposer):
    """Exercise proposal projection with fake projects and providers that cannot spend."""

    requires_search_cost_binding = False


class _NonpaidProviderDeltaProposer(ProviderDeltaProposer):
    """Exercise direct proposal parsing with deterministic in-memory providers."""

    requires_search_cost_binding = False


class _FlakyProvider(_Provider):
    def __init__(self, replies: list[str | Exception]) -> None:
        super().__init__("")
        self.replies = replies

    def complete(self, system: str, messages: list[Message], **kwargs: object) -> Completion:
        del system, messages, kwargs
        reply = self.replies[self.calls]
        self.calls += 1
        if isinstance(reply, Exception):
            raise reply
        return Completion(text=reply)


class _IdentityProvider(_Provider):
    """Return controlled direct and chat response identities for proposer regressions."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        completions: list[Completion] | None = None,
        chat_responses: list[ChatResponse] | None = None,
    ) -> None:
        super().__init__("")
        self.config = config
        self.completions = completions or []
        self.chat_responses = chat_responses or []
        self.chat_calls = 0

    def complete(self, system: str, messages: list[Message], **kwargs: object) -> Completion:
        del system, messages, kwargs
        completion = self.completions[self.calls]
        self.calls += 1
        return completion

    def complete_chat(self, request: object) -> ChatResponse:
        del request
        response = self.chat_responses[self.chat_calls]
        self.chat_calls += 1
        return response

    def verify(self) -> VerifyResult:
        return VerifyResult(ok=True, kind=self.config.kind, model=self.config.model)


def _chat_response(
    config: ProviderConfig,
    *,
    index: int,
    response_model: str,
    system_fingerprint: str | None,
) -> ChatResponse:
    requested_model = (
        config.deployment if config.kind is ProviderKind.AZURE_OPENAI else config.model
    )
    assert requested_model is not None
    receipt = build_chat_provider_receipt(
        provider=config.kind.value,
        provider_request_id=f"request-{index}",
        response_id=f"response-{index}",
        requested_model=requested_model,
        response_model=response_model,
        system_fingerprint=system_fingerprint,
        request_payload={"messages": [], "max_completion_tokens": 16},
        temperature=None,
        max_tokens=16,
        max_tokens_field=config.resolved_chat_max_tokens_field(),
        started_at_unix_s=1.0,
        finished_at_unix_s=2.0,
    )
    return ChatResponse.model_validate(
        {
            "id": f"response-{index}",
            "model": response_model,
            "system_fingerprint": system_fingerprint,
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            "provider_receipt": receipt,
        }
    )


_PROJECT_EXECUTION_CONFIGURATION_ID = "sha256:" + "e" * 64


class _Project:
    workspace = "/home/user/project"
    execution_configuration_id = _PROJECT_EXECUTION_CONFIGURATION_ID

    def __init__(self, outputs: list[str]) -> None:
        self.files: dict[str, str] = {}
        self.private_files: dict[str, str] = {}
        self.outputs = outputs
        self.runs = 0

    def write_text(self, path: str, content: str) -> None:
        self.files[path] = content

    def read_text(self, path: str) -> str:
        return self.files[path]

    def write_private_text(self, path: str, content: str) -> None:
        self.private_files[path] = content

    def read_private_text(self, path: str) -> str:
        return self.private_files[path]

    def export_search_state(self) -> JsonObject:
        return {
            "visible_files": dict(self.files),
            "private_files": dict(self.private_files),
        }

    def restore_search_state(self, state: JsonObject) -> None:
        visible = state.get("visible_files")
        private = state.get("private_files")
        assert isinstance(visible, dict)
        assert isinstance(private, dict)
        self.files = {str(path): str(content) for path, content in visible.items()}
        self.private_files = {str(path): str(content) for path, content in private.items()}

    def run(
        self,
        agent: HarnessDoc,
        provider: ToolCallingProvider,
        instruction: str,
        *,
        should_cancel: Callable[[], bool] | None = None,
        writable_files: Collection[str] | None = None,
    ) -> AgentProjectRun:
        del agent, provider, should_cancel, writable_files
        self.runs += 1
        assert f"exactly {len(self.outputs)}" in instruction
        iteration_dir = f"iteration-{self.runs:04d}"
        for index, output in enumerate(self.outputs, start=1):
            self.files[f"proposals/{iteration_dir}/proposal-{index:02d}.json"] = output
        return AgentProjectRun(answer="done", events=(), worker_usage=TokenUsage())


class _CostBoundProject(_Project):
    def __init__(
        self,
        outputs: list[str],
        runtime: SearchComponentCostRuntime,
    ) -> None:
        super().__init__(outputs)
        self.search_cost_binding = runtime.binding
        [self.timed_resource_binding] = runtime.binding.timed_resources
        self.budget_policy_digest = runtime.authority.policy.policy_digest
        self.budget_ledger_path = runtime.authority.ledger_path
        self.observed_provider: ToolCallingProvider | None = None
        self.authorizations: list[SearchCostBinding] = []

    def authorize_search_dispatch(self, binding: SearchCostBinding) -> None:
        self.authorizations.append(SearchCostBinding.model_validate(binding.model_dump()))

    def run(
        self,
        agent: HarnessDoc,
        provider: ToolCallingProvider,
        instruction: str,
        *,
        should_cancel: Callable[[], bool] | None = None,
        writable_files: Collection[str] | None = None,
    ) -> AgentProjectRun:
        self.observed_provider = provider
        return super().run(
            agent,
            provider,
            instruction,
            should_cancel=should_cancel,
            writable_files=writable_files,
        )


def _cost_digest(character: str) -> str:
    return "sha256:" + character * 64


def _proposer_cost_runtime(
    tmp_path: Path,
    *,
    configuration_id: str,
    provider_config: ProviderConfig | None = None,
    response_identity: ProviderResponseIdentity | None = None,
    include_project_resource: bool = True,
) -> SearchComponentCostRuntime:
    proposer_provider = provider_config or _Provider.config
    proposer_identity = freeze_provider_response_identity(proposer_provider, response_identity)
    policy = BudgetPolicy(
        study_id="project-proposer-cost-wiring",
        manifest_digest="sha256:" + hashlib.sha256(str(tmp_path).encode()).hexdigest(),
        hard_limit_nano_usd=1_000_000,
        phase_limits_nano_usd={"search": 1_000_000},
        meters={
            "proposer-provider": synthetic_provider_cost_meter(
                provider_config=proposer_provider,
                provenance=synthetic_tariff_provenance(proposer_provider),
                input_nano_usd_per_token=1,
                output_nano_usd_per_token=1,
            ),
            "scorer-provider": synthetic_provider_cost_meter(
                provider_config=_Provider.config,
                provenance=synthetic_tariff_provenance(_Provider.config),
                input_nano_usd_per_token=1,
                output_nano_usd_per_token=1,
            ),
            "project-sandbox": TimedResourceCostMeter(
                resource_type="proposer_project",
                resource_class_digest=_cost_digest("b"),
                nano_usd_per_second=1,
                max_billing_seconds=30,
            ),
        },
    )
    authority = bootstrap_budget_ledger(tmp_path / "proposer-cost.sqlite3", policy)
    proposer_scope = BudgetScope(phase="search", category="proposer", run_id="optimizer-run")
    scorer_scope = BudgetScope(phase="search", category="scorer", run_id="optimizer-run")
    proposer_account = authority.provider_account(
        scope=proposer_scope,
        meter_id="proposer-provider",
    )
    project_account = authority.timed_resource_account(
        scope=proposer_scope,
        meter_id="project-sandbox",
    )
    scorer_account = authority.provider_account(
        scope=scorer_scope,
        meter_id="scorer-provider",
    )
    binding = SearchCostBinding(
        declared_hard_limit_nano_usd=policy.hard_limit_nano_usd,
        policy=policy,
        ledger_identity=authority.ledger_identity,
        phase="search",
        run_id="optimizer-run",
        proposer=SearchComponentCostBinding(
            role=SearchComponentRole.PROPOSER,
            configuration_id=configuration_id,
            scope_category="proposer",
            providers=(
                ProviderCostBinding(
                    component_configuration_id=configuration_id,
                    provider_config=proposer_provider,
                    response_identity=proposer_identity,
                    account=bind_budget_account(proposer_account),
                ),
            ),
            timed_resources=(
                (
                    TimedResourceCostBinding(
                        component_configuration_id=configuration_id,
                        resource_type="proposer_project",
                        resource_class_digest=_cost_digest("b"),
                        account=bind_timed_resource_account(project_account),
                    ),
                )
                if include_project_resource
                else ()
            ),
        ),
        scorer=SearchComponentCostBinding(
            role=SearchComponentRole.SCORER,
            configuration_id="unused-scorer",
            scope_category="scorer",
            providers=(
                ProviderCostBinding(
                    component_configuration_id="unused-scorer",
                    provider_config=_Provider.config,
                    response_identity=ProviderResponseIdentity(provider=ProviderKind.BEDROCK),
                    account=bind_budget_account(scorer_account),
                ),
            ),
        ),
    )
    return SearchCostRuntime(authority=authority, binding=binding).for_component(
        SearchComponentRole.PROPOSER
    )


def test_project_proposer_binds_exact_runtime_and_reaudits_before_project_dispatch(
    tmp_path: Path,
) -> None:
    parent = HarnessDoc.baseline("parent")
    raw_provider = _Provider(_payload(parent, "cost-bound"))
    agent = meta_agent()
    configuration_id = ProjectDeltaProposer.configuration_id_for(
        project_type=_CostBoundProject,
        project_workspace=_Project.workspace,
        project_execution_configuration_id=_PROJECT_EXECUTION_CONFIGURATION_ID,
        agent=agent,
        provider=raw_provider,
    )
    runtime = _proposer_cost_runtime(tmp_path, configuration_id=configuration_id)
    project = _CostBoundProject([_payload(parent, "cost-bound")], runtime)
    proposer = ProjectDeltaProposer(project, agent, raw_provider, cost_runtime=runtime)

    with sqlite3.connect(runtime.authority.ledger_path) as connection:
        connection.execute("DROP TRIGGER budget_metadata_no_update")
        connection.execute("UPDATE budget_metadata SET schema_version = 1 WHERE id = 1")

    with pytest.raises(BudgetIntegrityError, match="unsupported budget schema version"):
        proposer.propose_batch(parent, _trigger(), "evidence", history=[], count=1)

    assert proposer.search_cost_binding == runtime.binding
    assert isinstance(proposer._provider, BudgetedProvider)  # noqa: SLF001
    assert project.runs == 0
    assert project.files == {}
    assert raw_provider.calls == 0


def test_project_proposer_forwards_only_its_exact_complete_search_binding(tmp_path: Path) -> None:
    provider = _Provider("unused")
    agent = meta_agent()
    configuration_id = ProjectDeltaProposer.configuration_id_for(
        project_type=_CostBoundProject,
        project_workspace=_Project.workspace,
        project_execution_configuration_id=_PROJECT_EXECUTION_CONFIGURATION_ID,
        agent=agent,
        provider=provider,
    )
    runtime = _proposer_cost_runtime(tmp_path, configuration_id=configuration_id)
    project = _CostBoundProject([], runtime)
    proposer = ProjectDeltaProposer(project, agent, provider, cost_runtime=runtime)

    proposer.authorize_search_dispatch(runtime.search_binding)

    assert project.authorizations == [runtime.search_binding]

    other_runtime = _proposer_cost_runtime(
        tmp_path / "other-runtime",
        configuration_id=configuration_id,
    )
    with pytest.raises(ValueError, match="differs from the proposer cost runtime"):
        proposer.authorize_search_dispatch(other_runtime.search_binding)

    assert project.authorizations == [runtime.search_binding]


def test_project_proposer_configuration_binds_create_rate_authority(tmp_path: Path) -> None:
    provider = _Provider("unused")
    agent = meta_agent()
    first = ExternalDispatchRateAuthority.bootstrap(
        (tmp_path / "first-rate.json").resolve(),
        E2B_SANDBOX_CREATE_RATE_POLICY,
    )
    second = ExternalDispatchRateAuthority.bootstrap(
        (tmp_path / "second-rate.json").resolve(),
        E2B_SANDBOX_CREATE_RATE_POLICY,
    )

    first_id = ProjectDeltaProposer.configuration_id_for(
        project_type=_CostBoundProject,
        project_workspace=_Project.workspace,
        project_execution_configuration_id=_PROJECT_EXECUTION_CONFIGURATION_ID,
        agent=agent,
        provider=provider,
        project_create_rate_binding=first.binding,
    )
    second_id = ProjectDeltaProposer.configuration_id_for(
        project_type=_CostBoundProject,
        project_workspace=_Project.workspace,
        project_execution_configuration_id=_PROJECT_EXECUTION_CONFIGURATION_ID,
        agent=agent,
        provider=provider,
        project_create_rate_binding=second.binding,
    )

    assert first_id != second_id


def test_project_proposer_configuration_binds_project_execution_commitment() -> None:
    provider = _Provider("unused")
    agent = meta_agent()

    first_id = ProjectDeltaProposer.configuration_id_for(
        project_type=_CostBoundProject,
        project_workspace=_Project.workspace,
        project_execution_configuration_id="sha256:" + "1" * 64,
        agent=agent,
        provider=provider,
    )
    second_id = ProjectDeltaProposer.configuration_id_for(
        project_type=_CostBoundProject,
        project_workspace=_Project.workspace,
        project_execution_configuration_id="sha256:" + "2" * 64,
        agent=agent,
        provider=provider,
    )

    assert first_id != second_id


def test_project_execution_drift_stops_before_replayed_project_dispatch(tmp_path: Path) -> None:
    parent = HarnessDoc.baseline("parent")
    provider = _Provider(_payload(parent, "must-not-dispatch"))
    agent = meta_agent()
    configuration_id = ProjectDeltaProposer.configuration_id_for(
        project_type=_CostBoundProject,
        project_workspace=_Project.workspace,
        project_execution_configuration_id=_PROJECT_EXECUTION_CONFIGURATION_ID,
        agent=agent,
        provider=provider,
    )
    runtime = _proposer_cost_runtime(tmp_path, configuration_id=configuration_id)
    project = _CostBoundProject([_payload(parent, "must-not-dispatch")], runtime)
    proposer = ProjectDeltaProposer(project, agent, provider, cost_runtime=runtime)
    project.execution_configuration_id = "sha256:" + "f" * 64

    with pytest.raises(ValueError, match="configuration_id differs"):
        proposer.propose_batch(parent, _trigger(), "evidence", history=[], count=1)

    assert project.runs == 0
    assert project.files == {}
    assert provider.calls == 0


@dataclass(frozen=True)
class _ResumeRecord:
    iteration: int
    proposal_index: int
    delta_id: str | None


class _BudgetProject(_Project):
    def __init__(
        self,
        outputs: list[str],
        *,
        policy_digest: str | None,
        ledger_path: Path | None,
    ) -> None:
        super().__init__(outputs)
        self.budget_policy_digest = policy_digest
        self.budget_ledger_path = ledger_path


class _BudgetProvider(_Provider):
    def __init__(
        self,
        reply: str,
        *,
        policy_digest: str | None,
        ledger_path: Path | None,
    ) -> None:
        super().__init__(reply)
        self.budget_policy_digest = policy_digest
        self.budget_ledger_path = ledger_path


def _manifest_content(project: _Project, path: str) -> tuple[dict[str, object], list[str]]:
    manifest = json.loads(project.files[path])
    chunks = [
        project.files[str(absolute).removeprefix(f"{project.workspace}/")]
        for absolute in manifest["content_files"]
    ]
    return manifest, chunks


def _context_chunks(project: _Project, context: object) -> list[str]:
    assert isinstance(context, dict)
    manifest = cast("dict[str, object]", context)
    files = manifest["content_files"]
    assert isinstance(files, list)
    return [
        project.files[str(absolute).removeprefix(f"{project.workspace}/")] for absolute in files
    ]


def _score_archive(
    evaluation_id: str,
    *,
    purpose: str = "seed",
    tier: ScoreArchiveTier = ScoreArchiveTier.DISCOVERY,
    visibility: ScoreArchiveVisibility | None = None,
    evidence: str = "successful parent trace\nfailed parent trace",
) -> HarnessScoreArchive:
    if visibility is None:
        visibility = (
            ScoreArchiveVisibility.PROPOSER
            if tier is ScoreArchiveTier.DISCOVERY and purpose in {"seed", "screen", "full"}
            else ScoreArchiveVisibility.AUDIT_ONLY
        )
    tasks = {
        "pass-task": TaskScore(
            task_id="pass-task",
            score=1.0,
            secondary_score=0.9999999994,
            passed=True,
            description="keep the successful behavior",
            evidence="successful trajectory",
        ),
        "fail-task": TaskScore(
            task_id="fail-task",
            score=0.1234567894,
            secondary_score=0.2345678912,
            passed=False,
            description="find the requested artifact",
            mechanisms=("verification",),
            evidence=evidence,
        ),
    }
    return HarnessScoreArchive(
        scorer_tier=tier,
        visibility=visibility,
        request=ScoreRequest.model_validate({"purpose": purpose}),
        report=HarnessScoreReport(
            evaluation_id=evaluation_id,
            label="candidate",
            score=0.5617283947,
            secondary_score=0.6172839453,
            attempts=2,
            run_health=ScoreRunHealth.VALID,
            per_task=tasks,
        ),
    )


def _parent_surface_manifests(project: _Project, root_path: str) -> list[dict[str, object]]:
    root = json.loads(project.files[root_path])
    index_path = str(root["surface_index_manifest"]).removeprefix(f"{project.workspace}/")
    _index_manifest, index_chunks = _manifest_content(project, index_path)
    index = json.loads("".join(index_chunks))
    return [
        json.loads(project.files[str(item["manifest_file"]).removeprefix(f"{project.workspace}/")])
        for item in index
    ]


def test_production_project_proposer_rejects_missing_cost_runtime_before_effects() -> None:
    project = _Project([])
    provider = _Provider("unused")

    with pytest.raises(ValueError, match="complete search cost runtime"):
        _ProductionProjectDeltaProposer(project, meta_agent(), provider)

    assert project.runs == 0
    assert project.files == {}
    assert provider.calls == 0


@pytest.mark.parametrize("kind", [ProviderKind.AZURE_OPENAI, ProviderKind.BEDROCK])
def test_production_provider_proposer_rejects_missing_cost_runtime_before_effects(
    kind: ProviderKind,
) -> None:
    parent = HarnessDoc.baseline("parent")
    provider = _Provider("unused")
    provider.config = ProviderConfig(kind=kind, model="paid-model")
    response_identity = (
        ProviderResponseIdentity(
            provider=kind,
            response_model="served-model",
            system_fingerprint=None,
        )
        if kind is ProviderKind.AZURE_OPENAI
        else None
    )
    proposer = ProviderDeltaProposer(provider, response_identity=response_identity)

    with pytest.raises(ValueError, match="complete search cost runtime"):
        proposer.propose_batch(parent, _trigger(), "evidence", history=[], count=1)

    assert provider.calls == 0


def test_provider_proposer_binds_exact_runtime_and_reaudits_before_dispatch(
    tmp_path: Path,
) -> None:
    parent = HarnessDoc.baseline("parent")
    raw_provider = _Provider(_payload(parent, "cost-bound"))
    configuration_id = ProviderDeltaProposer.configuration_id_for(raw_provider)
    runtime = _proposer_cost_runtime(
        tmp_path,
        configuration_id=configuration_id,
        include_project_resource=False,
    )
    proposer = ProviderDeltaProposer(raw_provider, cost_runtime=runtime)

    with sqlite3.connect(runtime.authority.ledger_path) as connection:
        connection.execute("DROP TRIGGER budget_metadata_no_update")
        connection.execute("UPDATE budget_metadata SET schema_version = 1 WHERE id = 1")

    with pytest.raises(BudgetIntegrityError, match="unsupported budget schema version"):
        proposer.propose_batch(parent, _trigger(), "evidence", history=[], count=1)

    assert proposer.search_cost_binding == runtime.binding
    assert isinstance(proposer.provider, BudgetedProvider)
    assert raw_provider.calls == 0


def test_direct_proposer_response_identity_drift_stops_before_next_sibling(
    tmp_path: Path,
) -> None:
    parent = HarnessDoc.baseline("parent")
    config = ProviderConfig(
        kind=ProviderKind.AZURE_OPENAI,
        model_type="gpt-5.5",
        model="gpt-5.5",
        deployment="proposer-deployment",
        endpoint="https://example.openai.azure.com",
        api_version="2026-01-01",
    )
    response_identity = ProviderResponseIdentity(
        provider=ProviderKind.AZURE_OPENAI,
        response_model="served-stable",
        system_fingerprint=None,
    )
    raw_provider = _IdentityProvider(
        config,
        completions=[
            Completion(
                text=_payload(parent, "drifted"),
                usage=ProviderTokenUsage(input_tokens=1, output_tokens=1),
                model="served-drifted",
                system_fingerprint=None,
            ),
            Completion(
                text=_payload(parent, "must-not-dispatch"),
                usage=ProviderTokenUsage(input_tokens=1, output_tokens=1),
                model="served-stable",
                system_fingerprint=None,
            ),
        ],
    )
    configuration_id = ProviderDeltaProposer.configuration_id_for(
        raw_provider,
        response_identity=response_identity,
    )
    runtime = _proposer_cost_runtime(
        tmp_path,
        configuration_id=configuration_id,
        provider_config=config,
        response_identity=response_identity,
        include_project_resource=False,
    )
    proposer = ProviderDeltaProposer(
        raw_provider,
        cost_runtime=runtime,
        response_identity=response_identity,
    )

    with pytest.raises(ProviderResponseIdentityError, match="frozen response identity"):
        proposer.propose_batch(parent, _trigger(), "evidence", history=[], count=2)

    assert raw_provider.calls == 1
    [reservation] = SpendLedger(
        runtime.authority.ledger_path,
        runtime.authority.policy,
    ).reservations()
    assert reservation.status is ReservationStatus.FORFEITED
    assert reservation.failure_type == "ProviderIdentityInvalid"
    assert reservation.charged_nano_usd == reservation.max_nano_usd


def test_direct_proposer_invalid_usage_forfeits_and_stops_before_next_sibling(
    tmp_path: Path,
) -> None:
    parent = HarnessDoc.baseline("parent")
    config = ProviderConfig(
        kind=ProviderKind.AZURE_OPENAI,
        model_type="gpt-5.5",
        model="gpt-5.5",
        deployment="proposer-deployment",
        endpoint="https://example.openai.azure.com",
        api_version="2026-01-01",
    )
    response_identity = ProviderResponseIdentity(
        provider=ProviderKind.AZURE_OPENAI,
        response_model="served-stable",
        system_fingerprint=None,
    )
    raw_provider = _IdentityProvider(
        config,
        completions=[
            Completion(
                text=_payload(parent, "invalid-usage"),
                usage=ProviderTokenUsage(input_tokens=-1, output_tokens=1),
                model="served-stable",
                system_fingerprint=None,
            ),
            Completion(
                text=_payload(parent, "must-not-dispatch"),
                usage=ProviderTokenUsage(input_tokens=1, output_tokens=1),
                model="served-stable",
                system_fingerprint=None,
            ),
        ],
    )
    configuration_id = ProviderDeltaProposer.configuration_id_for(
        raw_provider,
        response_identity=response_identity,
    )
    runtime = _proposer_cost_runtime(
        tmp_path,
        configuration_id=configuration_id,
        provider_config=config,
        response_identity=response_identity,
        include_project_resource=False,
    )
    proposer = ProviderDeltaProposer(
        raw_provider,
        cost_runtime=runtime,
        response_identity=response_identity,
    )

    with pytest.raises(BudgetIntegrityError, match="nonnegative integer counts"):
        proposer.propose_batch(parent, _trigger(), "evidence", history=[], count=2)

    assert raw_provider.calls == 1
    [reservation] = SpendLedger(
        runtime.authority.ledger_path,
        runtime.authority.policy,
    ).reservations()
    assert reservation.status is ReservationStatus.FORFEITED
    assert reservation.failure_type == "UsageInvalid"
    assert reservation.charged_nano_usd == reservation.max_nano_usd


def test_direct_proposer_accepts_exact_response_identity_for_every_sibling(
    tmp_path: Path,
) -> None:
    parent = HarnessDoc.baseline("parent")
    config = ProviderConfig(
        kind=ProviderKind.AZURE_OPENAI,
        model_type="gpt-5.5",
        model="gpt-5.5",
        deployment="proposer-deployment",
        endpoint="https://example.openai.azure.com",
        api_version="2026-01-01",
    )
    response_identity = ProviderResponseIdentity(
        provider=ProviderKind.AZURE_OPENAI,
        response_model="served-stable",
        system_fingerprint="fp-stable",
    )
    raw_provider = _IdentityProvider(
        config,
        completions=[
            Completion(
                text=_payload(parent, "first"),
                usage=ProviderTokenUsage(input_tokens=1, output_tokens=1),
                model="served-stable",
                system_fingerprint="fp-stable",
            ),
            Completion(
                text=_payload(parent, "second"),
                usage=ProviderTokenUsage(input_tokens=1, output_tokens=1),
                model="served-stable",
                system_fingerprint="fp-stable",
            ),
        ],
    )
    configuration_id = ProviderDeltaProposer.configuration_id_for(
        raw_provider,
        response_identity=response_identity,
    )
    runtime = _proposer_cost_runtime(
        tmp_path,
        configuration_id=configuration_id,
        provider_config=config,
        response_identity=response_identity,
        include_project_resource=False,
    )
    proposer = ProviderDeltaProposer(
        raw_provider,
        cost_runtime=runtime,
        response_identity=response_identity,
    )

    proposals = proposer.propose_batch(parent, _trigger(), "evidence", history=[], count=2)

    assert raw_provider.calls == 2
    assert all(isinstance(proposal, HarnessDelta) for proposal in proposals)


def test_proposer_configuration_ids_bind_served_response_identity() -> None:
    config = ProviderConfig(
        kind=ProviderKind.AZURE_OPENAI,
        model="gpt-5.5",
        deployment="proposer-deployment",
    )
    provider = _IdentityProvider(config)
    first = ProviderResponseIdentity(
        provider=ProviderKind.AZURE_OPENAI,
        response_model="served-first",
        system_fingerprint=None,
    )
    second = first.model_copy(update={"response_model": "served-second"})

    assert ProviderDeltaProposer.configuration_id_for(
        provider,
        response_identity=first,
    ) != ProviderDeltaProposer.configuration_id_for(
        provider,
        response_identity=second,
    )
    assert ProjectDeltaProposer.configuration_id_for(
        project_type=_CostBoundProject,
        project_workspace=_Project.workspace,
        project_execution_configuration_id=_PROJECT_EXECUTION_CONFIGURATION_ID,
        agent=meta_agent(),
        provider=provider,
        response_identity=first,
    ) != ProjectDeltaProposer.configuration_id_for(
        project_type=_CostBoundProject,
        project_workspace=_Project.workspace,
        project_execution_configuration_id=_PROJECT_EXECUTION_CONFIGURATION_ID,
        agent=meta_agent(),
        provider=provider,
        response_identity=second,
    )


def test_project_proposer_response_identity_drift_stops_live_session_dispatch(
    tmp_path: Path,
) -> None:
    parent = HarnessDoc.baseline("parent")
    config = ProviderConfig(
        kind=ProviderKind.AZURE_OPENAI,
        model_type="gpt-5.5",
        model="gpt-5.5",
        deployment="project-proposer-deployment",
        endpoint="https://example.openai.azure.com",
        api_version="2026-01-01",
    )
    response_identity = ProviderResponseIdentity(
        provider=ProviderKind.AZURE_OPENAI,
        response_model="served-stable",
        system_fingerprint=None,
    )
    raw_provider = _IdentityProvider(
        config,
        chat_responses=[
            _chat_response(
                config,
                index=1,
                response_model="served-drifted",
                system_fingerprint=None,
            ),
            _chat_response(
                config,
                index=2,
                response_model="served-stable",
                system_fingerprint=None,
            ),
        ],
    )
    agent = meta_agent()

    class _DispatchingProject(_CostBoundProject):
        def run(
            self,
            agent: HarnessDoc,
            provider: ToolCallingProvider,
            instruction: str,
            *,
            should_cancel: Callable[[], bool] | None = None,
            writable_files: Collection[str] | None = None,
        ) -> AgentProjectRun:
            del agent, instruction, should_cancel, writable_files
            self.runs += 1
            request = ChatRequest(
                messages=[ChatMessage(role="user", content="propose")],
                max_completion_tokens=16,
            )
            provider.complete_chat(request)
            provider.complete_chat(request)
            raise AssertionError("identity drift must stop before the second provider dispatch")

    configuration_id = ProjectDeltaProposer.configuration_id_for(
        project_type=_DispatchingProject,
        project_workspace=_Project.workspace,
        project_execution_configuration_id=_PROJECT_EXECUTION_CONFIGURATION_ID,
        agent=agent,
        provider=raw_provider,
        response_identity=response_identity,
    )
    runtime = _proposer_cost_runtime(
        tmp_path,
        configuration_id=configuration_id,
        provider_config=config,
        response_identity=response_identity,
    )
    project = _DispatchingProject([], runtime)
    proposer = ProjectDeltaProposer(
        project,
        agent,
        raw_provider,
        cost_runtime=runtime,
        response_identity=response_identity,
    )

    with pytest.raises(ProviderResponseIdentityError, match="frozen response identity"):
        proposer.propose_batch(parent, _trigger(), "evidence", history=[], count=1)

    assert project.runs == 1
    assert raw_provider.chat_calls == 1
    [reservation] = SpendLedger(
        runtime.authority.ledger_path,
        runtime.authority.policy,
    ).reservations()
    assert reservation.status is ReservationStatus.FORFEITED
    assert reservation.failure_type == "ProviderIdentityInvalid"
    assert reservation.charged_nano_usd == reservation.max_nano_usd


def test_project_proposer_invalid_chat_usage_forfeits_and_stops_next_dispatch(
    tmp_path: Path,
) -> None:
    parent = HarnessDoc.baseline("parent")
    config = ProviderConfig(
        kind=ProviderKind.AZURE_OPENAI,
        model_type="gpt-5.5",
        model="gpt-5.5",
        deployment="project-proposer-deployment",
        endpoint="https://example.openai.azure.com",
        api_version="2026-01-01",
    )
    response_identity = ProviderResponseIdentity(
        provider=ProviderKind.AZURE_OPENAI,
        response_model="served-stable",
        system_fingerprint=None,
    )
    first_response = _chat_response(
        config,
        index=1,
        response_model="served-stable",
        system_fingerprint=None,
    ).model_copy(update={"usage": ChatUsage(prompt_tokens=1, completion_tokens=-1)})
    raw_provider = _IdentityProvider(
        config,
        chat_responses=[
            first_response,
            _chat_response(
                config,
                index=2,
                response_model="served-stable",
                system_fingerprint=None,
            ),
        ],
    )
    agent = meta_agent()

    class _DispatchingProject(_CostBoundProject):
        def run(
            self,
            agent: HarnessDoc,
            provider: ToolCallingProvider,
            instruction: str,
            *,
            should_cancel: Callable[[], bool] | None = None,
            writable_files: Collection[str] | None = None,
        ) -> AgentProjectRun:
            del agent, instruction, should_cancel, writable_files
            self.runs += 1
            request = ChatRequest(
                messages=[ChatMessage(role="user", content="propose")],
                max_completion_tokens=16,
            )
            provider.complete_chat(request)
            provider.complete_chat(request)
            raise AssertionError("invalid usage must stop before the second provider dispatch")

    configuration_id = ProjectDeltaProposer.configuration_id_for(
        project_type=_DispatchingProject,
        project_workspace=_Project.workspace,
        project_execution_configuration_id=_PROJECT_EXECUTION_CONFIGURATION_ID,
        agent=agent,
        provider=raw_provider,
        response_identity=response_identity,
    )
    runtime = _proposer_cost_runtime(
        tmp_path,
        configuration_id=configuration_id,
        provider_config=config,
        response_identity=response_identity,
    )
    project = _DispatchingProject([], runtime)
    proposer = ProjectDeltaProposer(
        project,
        agent,
        raw_provider,
        cost_runtime=runtime,
        response_identity=response_identity,
    )

    with pytest.raises(BudgetIntegrityError, match="nonnegative integer counts"):
        proposer.propose_batch(parent, _trigger(), "evidence", history=[], count=1)

    assert project.runs == 1
    assert raw_provider.chat_calls == 1
    [reservation] = SpendLedger(
        runtime.authority.ledger_path,
        runtime.authority.policy,
    ).reservations()
    assert reservation.status is ReservationStatus.FORFEITED
    assert reservation.failure_type == "UsageInvalid"
    assert reservation.charged_nano_usd == reservation.max_nano_usd


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
def test_provider_proposer_propagates_search_safety_terminal_errors(error: Exception) -> None:
    parent = HarnessDoc.baseline("parent")
    provider = _FlakyProvider([error])

    with pytest.raises(type(error), match=str(error)):
        _NonpaidProviderDeltaProposer(provider).propose_batch(
            parent,
            _trigger(),
            "evidence",
            history=[],
            count=1,
        )


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
def test_project_proposer_cannot_salvage_durable_output_after_safety_error(
    error: Exception,
) -> None:
    parent = HarnessDoc.baseline("parent")

    class _SafetyTerminalProject(_Project):
        def run(
            self,
            agent: HarnessDoc,
            provider: ToolCallingProvider,
            instruction: str,
            *,
            should_cancel: Callable[[], bool] | None = None,
            writable_files: Collection[str] | None = None,
        ) -> AgentProjectRun:
            super().run(
                agent,
                provider,
                instruction,
                should_cancel=should_cancel,
                writable_files=writable_files,
            )
            raise error

    project = _SafetyTerminalProject([_payload(parent, "durable-but-terminal")])

    with pytest.raises(type(error), match=str(error)):
        ProjectDeltaProposer(project, meta_agent(), _Provider("unused")).propose_batch(
            parent,
            _trigger(),
            "evidence",
            history=[],
            count=1,
        )

    assert project.runs == 1


@pytest.mark.parametrize(
    ("project_policy", "provider_policy"),
    [("sha256:" + "a" * 64, None), (None, "sha256:" + "a" * 64)],
)
def test_project_proposer_requires_budget_presence_parity(
    tmp_path: Path,
    project_policy: str | None,
    provider_policy: str | None,
) -> None:
    ledger = (tmp_path / "budget.sqlite3").resolve()
    project = _BudgetProject(
        [],
        policy_digest=project_policy,
        ledger_path=ledger if project_policy is not None else None,
    )
    provider = _BudgetProvider(
        "",
        policy_digest=provider_policy,
        ledger_path=ledger if provider_policy is not None else None,
    )

    with pytest.raises(ValueError, match="both use one hard-budget policy"):
        ProjectDeltaProposer(project, meta_agent(), provider)


def test_project_proposer_requires_the_same_canonical_budget_ledger(tmp_path: Path) -> None:
    policy = "sha256:" + "b" * 64
    project = _BudgetProject(
        [],
        policy_digest=policy,
        ledger_path=(tmp_path / "project.sqlite3").resolve(),
    )
    provider = _BudgetProvider(
        "",
        policy_digest=policy,
        ledger_path=(tmp_path / "provider.sqlite3").resolve(),
    )

    with pytest.raises(ValueError, match="share one hard-budget ledger"):
        ProjectDeltaProposer(project, meta_agent(), provider)


class _InterruptedProject(_Project):
    """Write a prefix of the batch, then lose the runner's terminal control frame."""

    def run(
        self,
        agent: HarnessDoc,
        provider: ToolCallingProvider,
        instruction: str,
        *,
        should_cancel: Callable[[], bool] | None = None,
        writable_files: Collection[str] | None = None,
    ) -> AgentProjectRun:
        del agent, provider, instruction, should_cancel, writable_files
        self.runs += 1
        iteration_dir = f"iteration-{self.runs:04d}"
        for index, output in enumerate(self.outputs, start=1):
            self.files[f"proposals/{iteration_dir}/proposal-{index:02d}.json"] = output
        raise RuntimeError("Server disconnected after durable writes")


class _FailedProject(_Project):
    """Fail the project turn before any proposal output reaches durable storage."""

    def run(
        self,
        agent: HarnessDoc,
        provider: ToolCallingProvider,
        instruction: str,
        *,
        should_cancel: Callable[[], bool] | None = None,
        writable_files: Collection[str] | None = None,
    ) -> AgentProjectRun:
        del agent, provider, instruction, should_cancel, writable_files
        self.runs += 1
        raise RuntimeError("provider down")


class _RepairingProject(_Project):
    """Drive one proposal iteration with explicit per-turn rewrites of the same slot files."""

    def __init__(
        self,
        outputs: list[str],
        repairs: list[dict[int, str]],
        *,
        extra_rewrites: dict[int, str] | None = None,
        initial_error: str | None = None,
    ) -> None:
        super().__init__(outputs)
        self.repairs = repairs
        self.extra_rewrites = extra_rewrites or {}
        self.initial_error = initial_error
        self.instructions: list[str] = []
        self.write_grants: list[tuple[str, ...] | None] = []

    def run(
        self,
        agent: HarnessDoc,
        provider: ToolCallingProvider,
        instruction: str,
        *,
        should_cancel: Callable[[], bool] | None = None,
        writable_files: Collection[str] | None = None,
    ) -> AgentProjectRun:
        del agent, provider, should_cancel
        self.runs += 1
        self.instructions.append(instruction)
        self.write_grants.append(None if writable_files is None else tuple(sorted(writable_files)))
        if self.runs == 1:
            writes = {index: output for index, output in enumerate(self.outputs, start=1)}
        else:
            repair_index = self.runs - 2
            writes = self.repairs[repair_index] if repair_index < len(self.repairs) else {}
            writes = {**self.extra_rewrites, **writes}
        for index, output in writes.items():
            self.files[f"proposals/iteration-0001/proposal-{index:02d}.json"] = output
        if self.runs == 1 and self.initial_error is not None:
            raise RuntimeError(self.initial_error)
        return AgentProjectRun(answer="done", events=(), worker_usage=TokenUsage())


class _RestoreFailingProject(_RepairingProject):
    """Lose the durable-write channel while the host restores a valid sibling."""

    def write_text(self, path: str, content: str) -> None:
        if self.runs >= 2 and path == "proposals/iteration-0001/proposal-01.json":
            raise RuntimeError("valid sibling restoration failed")
        super().write_text(path, content)


def test_provider_proposer_produces_requested_sibling_count() -> None:
    parent = HarnessDoc.baseline("parent")
    provider = _Provider(_payload(parent, "careful"))

    proposals = _NonpaidProviderDeltaProposer(provider).propose_batch(
        parent, _trigger(), "evidence", history=[], count=3
    )

    assert provider.calls == 3
    assert len(proposals) == 3
    assert all(proposal is not None for proposal in proposals)


def test_provider_proposer_isolates_one_failed_sibling_call() -> None:
    parent = HarnessDoc.baseline("parent")
    provider = _FlakyProvider(
        [_payload(parent, "first"), RuntimeError("rate limited"), _payload(parent, "third")]
    )

    proposals = _NonpaidProviderDeltaProposer(provider).propose_batch(
        parent, _trigger(), "evidence", history=[], count=3
    )

    assert provider.calls == 3
    assert proposals[0] is not None and not isinstance(proposals[0], ProposalFailure)
    assert proposals[1] == ProposalFailure(reason="rate limited")
    assert proposals[2] is not None and not isinstance(proposals[2], ProposalFailure)


def test_provider_proposer_checks_cancellation_between_sibling_calls() -> None:
    parent = HarnessDoc.baseline("parent")
    provider = _Provider(_payload(parent, "careful"))

    with pytest.raises(HarnessSearchCancelled, match="cancelled"):
        _NonpaidProviderDeltaProposer(provider).propose_batch(
            parent,
            _trigger(),
            "evidence",
            history=[],
            count=3,
            should_cancel=lambda: provider.calls >= 1,
        )

    assert provider.calls == 1


def test_project_proposer_propagates_project_cancellation() -> None:
    parent = HarnessDoc.baseline("parent")
    callback = lambda: False  # noqa: E731 - identity is the behavior under test

    class _CancellingProject(_Project):
        def run(
            self,
            agent: HarnessDoc,
            provider: ToolCallingProvider,
            instruction: str,
            *,
            should_cancel: Callable[[], bool] | None = None,
            writable_files: Collection[str] | None = None,
        ) -> AgentProjectRun:
            del agent, provider, instruction, writable_files
            assert should_cancel is callback
            raise HarnessSearchCancelled("harness search cancelled")

    proposer = ProjectDeltaProposer(_CancellingProject([]), meta_agent(), _Provider("unused"))

    with pytest.raises(HarnessSearchCancelled, match="cancelled"):
        proposer.propose_batch(
            parent,
            _trigger(),
            "inspect failures",
            history=[],
            count=2,
            should_cancel=callback,
        )


def test_project_proposer_checks_cancellation_between_context_writes() -> None:
    parent = HarnessDoc.baseline("parent")

    class _CountingProject(_Project):
        def __init__(self) -> None:
            super().__init__([_payload(parent, "careful")])
            self.writes = 0

        def write_text(self, path: str, content: str) -> None:
            self.writes += 1
            super().write_text(path, content)

    project = _CountingProject()
    proposer = ProjectDeltaProposer(project, meta_agent(), _Provider("unused"))

    with pytest.raises(HarnessSearchCancelled, match="cancelled"):
        proposer.propose_batch(
            parent,
            _trigger(),
            "inspect failures",
            history=[],
            count=1,
            should_cancel=lambda: project.writes >= 2,
        )

    assert project.writes == 2
    assert project.runs == 0


def test_project_proposer_uses_one_agent_turn_and_keeps_iteration_files() -> None:
    parent = HarnessDoc.baseline("parent")
    project = _Project([_payload(parent, "careful"), _payload(parent, "verify")])
    provider = _Provider("unused")

    proposer = ProjectDeltaProposer(project, meta_agent(), provider)
    proposals = proposer.propose_batch(parent, _trigger(), "inspect failures", history=[], count=2)
    first_files = dict(project.files)
    project.outputs = [_payload(parent, "careful next"), _payload(parent, "verify next")]
    second = proposer.propose_batch(
        parent,
        _trigger(),
        "inspect the next failures",
        history=[proposal for proposal in proposals if isinstance(proposal, HarnessDelta)],
        count=2,
    )

    assert project.runs == 2
    assert len(proposals) == 2
    assert len(second) == 2
    assert all(proposal is not None for proposal in proposals)
    assert "context/iteration-0001/parent.json" in project.files
    assert "context/iteration-0001/parent-evaluations.json" in project.files
    assert "context/iteration-0001/evidence.json" in project.files
    assert "context/iteration-0001/history.json" in project.files
    assert "context/iteration-0002/history.json" in project.files
    assert "proposal-01.json" in project.files["context/iteration-0002/REQUEST.md"]
    assert "complete parent evaluation index" in project.files["context/iteration-0002/REQUEST.md"]
    assert "failure evidence manifest" in project.files["context/iteration-0002/REQUEST.md"]
    assert "content_files in listed order" in project.files["context/iteration-0002/REQUEST.md"]
    assert all(project.files[path] == content for path, content in first_files.items())
    assert {path for path in project.files if path.startswith("parents/")} == {
        path for path in first_files if path.startswith("parents/")
    }
    parent_context = json.loads(project.files["context/iteration-0001/parent.json"])
    surface_manifests = _parent_surface_manifests(project, "context/iteration-0001/parent.json")
    assert parent_context["doc_hash"] == parent.doc_hash
    assert {
        surface["id"]: surface["content_hash"] for surface in surface_manifests
    } == parent.surface_hashes()
    assert all("content" not in surface for surface in surface_manifests)
    for surface, manifest_surface in zip(parent.surfaces, surface_manifests, strict=True):
        files = [
            path.removeprefix(f"{project.workspace}/")
            for path in cast("list[str]", manifest_surface["content_files"])
        ]
        assert "".join(project.files[path] for path in files) == surface.content
        assert "source_file" not in manifest_surface


def test_project_parent_manifest_splits_large_surfaces_below_read_cap() -> None:
    content = "0123456789" * 4_001
    parent = HarnessDoc(
        name="large",
        surfaces=[
            Surface(id="prompt:core", kind=SurfaceKind.PROMPT, content="p"),
            Surface(
                id="tool_policy:main",
                kind=SurfaceKind.TOOL_POLICY,
                content="submit",
            ),
            Surface(
                id="code:large",
                kind=SurfaceKind.CODE,
                content=content,
                path="src/large.ts",
            ),
        ],
    )
    project = _Project([_payload(parent, "careful")])

    ProjectDeltaProposer(project, meta_agent(), _Provider("unused")).propose_batch(
        parent, _trigger(), "inspect failures", history=[], count=1
    )

    manifest_text = project.files["context/iteration-0001/parent.json"]
    surfaces = _parent_surface_manifests(project, "context/iteration-0001/parent.json")
    code_surface = next(surface for surface in surfaces if surface["id"] == "code:large")
    relative_files = [
        path.removeprefix(f"{project.workspace}/")
        for path in cast("list[str]", code_surface["content_files"])
    ]
    chunks = [project.files[path] for path in relative_files]
    source_file = cast("str", code_surface["source_file"]).removeprefix(f"{project.workspace}/")
    assert len(manifest_text) < 16_000
    assert len(chunks) > 1
    assert all(len(chunk) <= 12_000 for chunk in chunks)
    assert "".join(chunks) == content
    assert source_file == f"parents/{parent.doc_hash}/parent-source/src/large.ts"
    assert project.files[source_file] == content


def test_real_pi_parent_manifest_itself_fits_one_project_read() -> None:
    parent = default_agent("parent")
    project = _Project([_payload(parent, "careful")])

    ProjectDeltaProposer(project, meta_agent(), _Provider("unused")).propose_batch(
        parent, _trigger(), "inspect failures", history=[], count=1
    )

    manifest_text = project.files["context/iteration-0001/parent.json"]
    manifest = json.loads(manifest_text)
    surfaces = _parent_surface_manifests(project, "context/iteration-0001/parent.json")
    assert len(manifest_text) < 16_000
    assert manifest["surface_count"] == len(parent.surfaces)
    assert len(surfaces) == len(parent.surfaces)
    for surface in surfaces:
        chunks = [
            project.files[path.removeprefix(f"{project.workspace}/")]
            for path in cast("list[str]", surface["content_files"])
        ]
        assert all(len(chunk) <= 12_000 for chunk in chunks)
    assert all(surface.get("source_file") for surface in surfaces if surface["kind"] == "code")


def test_project_context_preserves_evidence_and_compacts_judged_history() -> None:
    parent = HarnessDoc.baseline("parent")
    large_change = "changed source\n" * 2_001
    project = _Project([_payload(parent, large_change)])
    proposer = ProjectDeltaProposer(project, meta_agent(), _Provider("unused"))

    first = proposer.propose_batch(parent, _trigger(), "first evidence", history=[], count=1)[0]
    assert isinstance(first, HarnessDelta)
    first.verdict = GateRecord(
        accepted=False,
        holdout_delta=-0.5,
        holdout_secondary_delta=-0.25,
        reason="secret holdout task regressed during confirmation",
    )
    second = first.model_copy(
        deep=True,
        update={"delta_id": "second-history-entry", "expected_effect": "different prediction"},
    )
    evidence = "failure trace line\n" * 1_501
    history = [second, first]

    proposer.propose_batch(parent, _trigger(), evidence, history=history, count=1)

    evidence_manifest, evidence_chunks = _manifest_content(
        project, "context/iteration-0002/evidence.json"
    )
    history_manifest, history_chunks = _manifest_content(
        project, "context/iteration-0002/history.json"
    )
    reconstructed_history = "".join(history_chunks)
    judged_history = json.loads(reconstructed_history)

    assert evidence_manifest["format"] == "markdown"
    assert evidence_manifest["content_length"] == len(evidence)
    assert len(evidence_chunks) > 1
    assert all(len(chunk) <= 12_000 for chunk in evidence_chunks)
    assert "".join(evidence_chunks) == evidence
    assert history_manifest["format"] == "json-array"
    assert history_manifest["entry_count"] == 2
    assert history_manifest["content_length"] == len(reconstructed_history)
    assert all(len(chunk) <= 12_000 for chunk in history_chunks)
    assert len(reconstructed_history) < len(large_change)
    assert [entry["delta_id"] for entry in judged_history] == [second.delta_id, first.delta_id]
    assert all("content" not in entry["ops"][0] for entry in judged_history)
    assert all(entry["ops"][0]["content_length"] == len(large_change) for entry in judged_history)
    assert "secret holdout task" not in reconstructed_history
    assert all("holdout_delta" not in entry["verdict"] for entry in judged_history)
    assert all(
        "confirmation measurements are not proposer-visible" in entry["verdict"]["reason"]
        for entry in judged_history
    )
    assert judged_history[0]["proposal_file"] is None
    assert judged_history[1]["proposal_file"].endswith("/proposals/iteration-0001/proposal-01.json")


def test_project_proposer_persists_candidate_evaluation_beside_its_proposal() -> None:
    parent = HarnessDoc.baseline("parent")
    project = _Project([_payload(parent, "careful")])
    proposer = ProjectDeltaProposer(project, meta_agent(), _Provider("unused"))
    proposal = proposer.propose_batch(parent, _trigger(), "inspect failures", history=[], count=1)[
        0
    ]
    assert isinstance(proposal, HarnessDelta)
    evidence = "candidate trace\n" * 2_001

    proposer.record_evaluation(proposal, stage="screen", content=evidence)

    manifest_path = "evaluations/iteration-0001/proposal-01/screen.json"
    manifest, chunks = _manifest_content(project, manifest_path)
    assert manifest["delta_id"] == proposal.delta_id
    assert manifest["stage"] == "screen"
    assert len(chunks) > 1
    assert all(len(chunk) <= 12_000 for chunk in chunks)
    assert "".join(chunks) == evidence


def test_project_proposer_indexes_complete_harness_evidence_for_parent_and_history() -> None:
    parent = HarnessDoc.baseline("parent")
    project = _Project([_payload(parent, "careful")])
    proposer = ProjectDeltaProposer(project, meta_agent(), _Provider("unused"))
    parent_archive = _score_archive(
        "eval-parent",
        evidence="successful parent trace\nfailed parent trace\n" * 1_001,
    )

    proposer.record_harness_evaluation(parent, archive=parent_archive)
    proposal = proposer.propose_batch(
        parent,
        _trigger(),
        "inspect failures",
        history=[],
        count=1,
    )[0]
    assert isinstance(proposal, HarnessDelta)
    child = apply_delta(parent, proposal, "child")
    child_archive = _score_archive(
        "eval-child",
        purpose="full",
        evidence="complete child evidence",
    )
    proposer.record_harness_evaluation(child, archive=child_archive)
    proposer.propose_batch(
        child,
        _trigger(),
        "inspect child failures",
        history=[proposal],
        count=1,
    )

    parent_index = json.loads(project.files["context/iteration-0001/parent-evaluations.json"])
    assert parent_index["harness_doc_hash"] == parent.doc_hash
    [parent_manifest_absolute] = parent_index["report_manifests"]
    parent_manifest_relative = str(parent_manifest_absolute).removeprefix(f"{project.workspace}/")
    parent_manifest = json.loads(project.files[parent_manifest_relative])
    assert parent_manifest["kind"] == "harness-score-report-index"
    assert parent_manifest["purpose"] == "seed"
    assert parent_manifest["scorer_tier"] == "discovery"
    assert parent_manifest["visibility"] == "proposer"
    assert parent_manifest["evaluation_id"] == "eval-parent"
    assert json.loads(parent_manifest["canonical_request_json"]) == {  # exact typed request
        "attempts": None,
        "purpose": "seed",
        "task_ids": None,
    }
    task_index = json.loads("".join(_context_chunks(project, parent_manifest["task_index"])))
    assert [item["task_id"] for item in task_index] == ["fail-task", "pass-task"]
    fail_manifest_path = str(task_index[0]["manifest_file"]).removeprefix(f"{project.workspace}/")
    fail_manifest = json.loads(project.files[fail_manifest_path])
    exact_task = "".join(_context_chunks(project, fail_manifest["canonical_record"]))
    assert exact_task == canonical_score_json(parent_archive.report.per_task["fail-task"])
    derived = "".join(_context_chunks(project, fail_manifest["derived_evidence"]))
    assert "untrusted benchmark data" in derived
    assert "failed parent trace" in derived

    child_index = json.loads(project.files["context/iteration-0002/parent-evaluations.json"])
    [child_manifest_absolute] = child_index["report_manifests"]
    child_manifest_relative = str(child_manifest_absolute).removeprefix(f"{project.workspace}/")
    child_manifest = json.loads(project.files[child_manifest_relative])
    assert child_manifest["harness_doc_hash"] == child.doc_hash
    assert child_manifest["purpose"] == "full"

    _history_manifest, history_chunks = _manifest_content(
        project, "context/iteration-0002/history.json"
    )
    [history_entry] = json.loads("".join(history_chunks))
    assert history_entry["score_report_manifests"] == [child_manifest_absolute]


def test_project_proposer_complete_evaluation_identity_is_idempotent_and_immutable() -> None:
    harness = HarnessDoc.baseline("parent")
    project = _Project([])
    proposer = ProjectDeltaProposer(project, meta_agent(), _Provider("unused"))

    archive = _score_archive("eval-1", evidence="same evidence")
    proposer.record_harness_evaluation(harness, archive=archive)
    first_files = dict(project.files)
    first_private_files = dict(project.private_files)
    reconstructed = ProjectDeltaProposer(project, meta_agent(), _Provider("unused"))
    reconstructed.record_harness_evaluation(harness, archive=archive)

    assert project.files == first_files
    assert project.private_files == first_private_files
    with pytest.raises(ValueError, match="cannot name different archived content"):
        reconstructed.record_harness_evaluation(
            harness,
            archive=_score_archive("eval-1", evidence="different evidence"),
        )

    reconstructed.propose_batch(harness, _trigger(), "inspect failures", history=[], count=1)
    parent_index = json.loads(project.files["context/iteration-0001/parent-evaluations.json"])
    assert len(parent_index["report_manifests"]) == 1


def test_project_proposer_checkpoint_restores_private_archives_and_iteration_links() -> None:
    parent = HarnessDoc.baseline("parent")
    first_project = _Project([_payload(parent, "first revision")])
    first = ProjectDeltaProposer(first_project, meta_agent(), _Provider("unused"))
    [proposal] = first.propose_batch(
        parent,
        _trigger(),
        "inspect failures",
        history=[],
        count=1,
    )
    assert isinstance(proposal, HarnessDelta)
    child = apply_delta(parent, proposal, "child")
    first.record_harness_evaluation(
        child,
        archive=_score_archive(
            "holdout-private",
            tier=ScoreArchiveTier.HOLDOUT,
            purpose="holdout",
        ),
    )
    state = first.export_search_state()

    restored_project = _Project([_payload(child, "second revision")])
    restored = ProjectDeltaProposer(restored_project, meta_agent(), _Provider("unused"))
    restored.restore_search_state(state)
    restored.resume_from_history(
        completed_iteration=1,
        proposal_records=[_ResumeRecord(iteration=1, proposal_index=1, delta_id=proposal.delta_id)],
    )
    restored.propose_batch(
        child,
        _trigger(),
        "inspect next failure",
        history=[proposal],
        count=1,
    )

    assert restored_project.private_files == first_project.private_files
    assert "context/iteration-0002/history.json" in restored_project.files
    _manifest, chunks = _manifest_content(
        restored_project,
        "context/iteration-0002/history.json",
    )
    [history] = json.loads("".join(chunks))
    assert history["proposal_file"].endswith("/proposals/iteration-0001/proposal-01.json")


def test_project_proposer_restores_one_exact_witnessed_batch_transition() -> None:
    parent = HarnessDoc.baseline("parent")
    first_project = _Project([_payload(parent, "first revision")])
    first = ProjectDeltaProposer(first_project, meta_agent(), _Provider("unused"))
    state_before = first.export_search_state()
    first.propose_batch(parent, _trigger(), "inspect failures", history=[], count=1)
    state_after = first.export_search_state()

    restored_project = _Project([])
    restored = ProjectDeltaProposer(restored_project, meta_agent(), _Provider("unused"))
    restored.restore_search_state(state_before)
    restored.restore_proposal_batch_state(
        state_before=state_before,
        state_after=state_after,
    )

    assert restored.export_search_state() == state_after
    assert restored_project.files == first_project.files

    with pytest.raises(ValueError, match="current state does not match witness pre-call"):
        restored.restore_proposal_batch_state(
            state_before=state_before,
            state_after=state_after,
        )


def test_project_proposer_resume_rejects_uncommitted_link_state() -> None:
    parent = HarnessDoc.baseline("parent")
    first_project = _Project([_payload(parent, "first revision")])
    first = ProjectDeltaProposer(first_project, meta_agent(), _Provider("unused"))
    [proposal] = first.propose_batch(
        parent,
        _trigger(),
        "inspect failures",
        history=[],
        count=1,
    )
    assert isinstance(proposal, HarnessDelta)
    state = first.export_search_state()
    proposal_files = state["proposal_files"]
    evaluation_dirs = state["evaluation_dirs"]
    assert isinstance(proposal_files, dict)
    assert isinstance(evaluation_dirs, dict)
    proposal_files["uncommitted-delta"] = "/home/user/project/private-evidence.json"
    evaluation_dirs["uncommitted-delta"] = "evaluations/uncommitted"

    restored = ProjectDeltaProposer(
        _Project([]),
        meta_agent(),
        _Provider("unused"),
    )
    restored.restore_search_state(state)
    with pytest.raises(ValueError, match="committed proposal history"):
        restored.resume_from_history(
            completed_iteration=1,
            proposal_records=[
                _ResumeRecord(iteration=1, proposal_index=1, delta_id=proposal.delta_id)
            ],
        )


def test_project_proposer_recovers_post_manifest_cancellation_without_identity_overwrite() -> None:
    harness = HarnessDoc.baseline("parent")
    project = _Project([])
    proposer = ProjectDeltaProposer(project, meta_agent(), _Provider("unused"))

    with pytest.raises(HarnessSearchCancelled, match="cancelled"):
        proposer.record_harness_evaluation(
            harness,
            archive=_score_archive("eval-1", evidence="large evidence\n" * 2_000),
            should_cancel=lambda: len(project.private_files) >= 3,
        )

    assert len(project.private_files) == 3
    assert any(path.endswith("/manifest.json") for path in project.private_files)
    assert not any(path.startswith("score-archives/by-harness/") for path in project.private_files)
    assert not project.files

    reconstructed = ProjectDeltaProposer(project, meta_agent(), _Provider("unused"))
    with pytest.raises(ValueError, match="cannot name different archived content"):
        reconstructed.record_harness_evaluation(
            harness,
            archive=_score_archive("eval-1", evidence="different evidence"),
        )

    reconstructed.record_harness_evaluation(
        harness,
        archive=_score_archive("eval-1", evidence="large evidence\n" * 2_000),
    )
    assert any(path.startswith("score-archives/by-harness/") for path in project.private_files)
    assert any(path.endswith(".json") for path in project.files)


def test_project_proposer_never_exposes_holdout_or_confirmation_archives() -> None:
    harness = HarnessDoc.baseline("parent")
    project = _Project([_payload(harness, "careful")])
    proposer = ProjectDeltaProposer(project, meta_agent(), _Provider("unused"))

    proposer.record_harness_evaluation(harness, archive=_score_archive("discovery-seed"))
    proposer.record_harness_evaluation(
        harness,
        archive=_score_archive(
            "holdout-seed",
            purpose="holdout",
            tier=ScoreArchiveTier.HOLDOUT,
            evidence="secret holdout instruction",
        ),
    )
    proposer.record_harness_evaluation(
        harness,
        archive=_score_archive(
            "discovery-confirmation",
            purpose="confirmation",
            evidence="secret confirmation instruction",
        ),
    )

    assert "secret holdout instruction" not in "\n".join(project.files.values())
    assert "secret confirmation instruction" not in "\n".join(project.files.values())
    assert "secret holdout instruction" in "\n".join(project.private_files.values())
    assert "secret confirmation instruction" in "\n".join(project.private_files.values())
    proposer.propose_batch(harness, _trigger(), "inspect failures", history=[], count=1)
    parent_index = json.loads(project.files["context/iteration-0001/parent-evaluations.json"])
    assert len(parent_index["report_manifests"]) == 1
    private_index_path = f"score-archives/by-harness/{harness.doc_hash}.json"
    private_index = json.loads(project.private_files[private_index_path])
    assert len(private_index["records"]) == 3
    assert len(private_index["proposer_report_manifests"]) == 1


def test_project_proposer_fails_closed_on_corrupt_committed_score_content() -> None:
    harness = HarnessDoc.baseline("parent")
    project = _Project([])
    archive = _score_archive("eval-corrupt")
    proposer = ProjectDeltaProposer(project, meta_agent(), _Provider("unused"))
    proposer.record_harness_evaluation(harness, archive=archive)
    report_path = next(path for path in project.private_files if path.endswith("/report.json"))
    project.private_files[report_path] = "{}"

    with pytest.raises(ValueError, match="report is corrupt"):
        ProjectDeltaProposer(project, meta_agent(), _Provider("unused")).record_harness_evaluation(
            harness,
            archive=archive,
        )


def test_project_proposer_rejects_a_private_index_that_exposes_hidden_metadata() -> None:
    harness = HarnessDoc.baseline("parent")
    project = _Project([_payload(harness, "careful")])
    archive = _score_archive("eval-visible")
    ProjectDeltaProposer(project, meta_agent(), _Provider("unused")).record_harness_evaluation(
        harness,
        archive=archive,
    )
    report_path = next(
        path
        for path, content in project.files.items()
        if path.endswith(".json") and '"kind": "harness-score-report-index"' in content
    )
    manifest = json.loads(project.files[report_path])
    manifest["visibility"] = "audit_only"
    project.files[report_path] = json.dumps(manifest)

    with pytest.raises(ValueError, match="attempted to expose a hidden score report"):
        ProjectDeltaProposer(project, meta_agent(), _Provider("unused")).propose_batch(
            harness,
            _trigger(),
            "inspect failures",
            history=[],
            count=1,
        )


def test_project_proposer_large_report_uses_compact_selective_task_index() -> None:
    harness = HarnessDoc.baseline("parent")
    project = _Project([])
    tasks = {
        f"task-{index:03d}": TaskScore(
            task_id=f"task-{index:03d}",
            score=float(index % 2),
            secondary_score=index / 100,
            passed=bool(index % 2),
            description=f"instruction {index}",
            evidence="x" * 64_000,
        )
        for index in range(89)
    }
    archive = HarnessScoreArchive(
        scorer_tier=ScoreArchiveTier.DISCOVERY,
        visibility=ScoreArchiveVisibility.PROPOSER,
        request=ScoreRequest(purpose="seed"),
        report=HarnessScoreReport(
            evaluation_id="large-eval",
            score=44 / 89,
            secondary_score=0.44,
            attempts=1,
            run_health=ScoreRunHealth.VALID,
            per_task=tasks,
        ),
    )

    ProjectDeltaProposer(project, meta_agent(), _Provider("unused")).record_harness_evaluation(
        harness,
        archive=archive,
    )

    report_path = next(
        path
        for path, content in project.files.items()
        if path.endswith(".json") and '"kind": "harness-score-report-index"' in content
    )
    report_manifest = json.loads(project.files[report_path])
    assert len(project.files[report_path]) < 16_000
    task_index_chunks = _context_chunks(project, report_manifest["task_index"])
    assert len(task_index_chunks) < 5
    assert all(len(chunk) <= 12_000 for chunk in task_index_chunks)
    task_index = json.loads("".join(task_index_chunks))
    assert len(task_index) == 89
    selected_path = str(task_index[42]["manifest_file"]).removeprefix(f"{project.workspace}/")
    selected_manifest = json.loads(project.files[selected_path])
    selected_record = "".join(_context_chunks(project, selected_manifest["canonical_record"]))
    assert selected_record == canonical_score_json(tasks["task-042"])


def test_project_proposer_checks_cancellation_before_evaluation_writes() -> None:
    parent = HarnessDoc.baseline("parent")
    project = _Project([_payload(parent, "careful")])
    cancelled = False
    proposer = ProjectDeltaProposer(project, meta_agent(), _Provider("unused"))
    proposal = proposer.propose_batch(
        parent,
        _trigger(),
        "inspect failures",
        history=[],
        count=1,
        should_cancel=lambda: cancelled,
    )[0]
    assert isinstance(proposal, HarnessDelta)
    cancelled = True

    with pytest.raises(HarnessSearchCancelled, match="cancelled"):
        proposer.record_evaluation(proposal, stage="screen", content="candidate trace")

    assert not any(path.startswith("evaluations/") for path in project.files)


def test_project_proposer_stamps_missing_parent_preconditions() -> None:
    parent = HarnessDoc.baseline("parent")
    raw = json.loads(_payload(parent, "careful"))
    raw["preconditions"] = {}
    project = _Project([json.dumps(raw)])

    proposals = ProjectDeltaProposer(project, meta_agent(), _Provider("unused")).propose_batch(
        parent, _trigger(), "inspect failures", history=[], count=1
    )

    proposal = proposals[0]
    assert isinstance(proposal, HarnessDelta)
    assert proposal.preconditions == {"prompt:core": parent.surface_hashes()["prompt:core"]}


def test_project_proposer_repairs_skill_frontmatter_and_protects_valid_sibling() -> None:
    parent = HarnessDoc.baseline("parent")
    valid_raw = _payload(parent, "careful")
    invalid_skill = _skill_payload("Parse the JSON response before formatting the answer.")
    repaired_skill = _skill_payload(
        "---\n"
        "name: parse-json\n"
        "description: Parse a JSON API response before formatting the requested fields\n"
        "---\n"
        "Parse the JSON response, select the requested fields, and format only after parsing."
    )
    # The fake repair agent also tries to overwrite slot 1. The host must restore that valid file
    # byte-for-byte and re-read only slot 2.
    project = _RepairingProject(
        [valid_raw, invalid_skill],
        [{2: repaired_skill}],
        extra_rewrites={1: "{"},
    )

    proposals = ProjectDeltaProposer(project, meta_agent(), _Provider("unused")).propose_batch(
        parent, _trigger(), "inspect failures", history=[], count=2
    )

    assert project.runs == 2
    assert all(isinstance(proposal, HarnessDelta) for proposal in proposals)
    assert project.write_grants == [
        (
            "proposals/iteration-0001/proposal-01.json",
            "proposals/iteration-0001/proposal-02.json",
        ),
        ("proposals/iteration-0001/proposal-02.json",),
    ]
    assert project.files["proposals/iteration-0001/proposal-01.json"] == valid_raw
    assert "name: <slug>" in project.instructions[0]
    assert "rewrite ONLY these" in project.instructions[1]
    assert "invalid files:" in project.instructions[1]
    assert "proposals/iteration-0001/proposal-02.json" in project.instructions[1]
    assert "Do not rewrite them" in project.instructions[1]
    first_report = json.loads(
        project.files["context/iteration-0001/proposal-validation-attempt-01.json"]
    )
    final_report = json.loads(
        project.files["context/iteration-0001/proposal-validation-attempt-02.json"]
    )
    assert first_report["valid_slots"] == [1]
    assert "skill file has no frontmatter" in first_report["errors"][0]["reason"]
    assert final_report["valid_slots"] == [1, 2]
    assert final_report["errors"] == []
    for proposal in proposals:
        assert isinstance(proposal, HarnessDelta)
        assert proposal.child_doc_hash is None
        apply_delta(parent, proposal.model_copy(deep=True), "preflight-proven")


def test_project_proposer_fails_if_valid_sibling_provenance_cannot_be_restored() -> None:
    """Never return an in-memory delta whose durable proposal file may have changed."""
    parent = HarnessDoc.baseline("parent")
    project = _RestoreFailingProject(
        [_payload(parent, "careful"), _skill_payload("missing frontmatter")],
        [
            {
                2: _skill_payload(
                    "---\nname: parse-json\ndescription: Parse JSON responses\n---\nParse first."
                )
            }
        ],
        extra_rewrites={1: "{"},
    )

    with pytest.raises(RuntimeError, match="valid sibling restoration failed"):
        ProjectDeltaProposer(project, meta_agent(), _Provider("unused")).propose_batch(
            parent, _trigger(), "inspect failures", history=[], count=2
        )


def test_project_proposer_repairs_history_and_sibling_duplicates() -> None:
    parent = HarnessDoc.baseline("parent")
    sibling = _payload(parent, "sibling")
    historic_raw = _payload(parent, "historic")
    historic = parse_delta(parent, _trigger(), historic_raw)
    assert isinstance(historic, HarnessDelta)
    project = _RepairingProject(
        [sibling, sibling, historic_raw],
        [{2: _payload(parent, "repaired sibling"), 3: _payload(parent, "repaired history")}],
    )

    proposals = ProjectDeltaProposer(project, meta_agent(), _Provider("unused")).propose_batch(
        parent, _trigger(), "inspect failures", history=[historic], count=3
    )

    assert project.runs == 2
    assert all(isinstance(proposal, HarnessDelta) for proposal in proposals)
    delta_ids = [proposal.delta_id for proposal in proposals if isinstance(proposal, HarnessDelta)]
    assert len(delta_ids) == len(set(delta_ids)) == 3
    report = json.loads(project.files["context/iteration-0001/proposal-validation-attempt-01.json"])
    reasons = [error["reason"] for error in report["errors"]]
    assert any("duplicates valid sibling proposal-01" in reason for reason in reasons)
    assert any("already present in judged history" in reason for reason in reasons)


def test_project_proposer_repairs_semantically_identical_children_and_no_ops() -> None:
    base = HarnessDoc.baseline("parent")
    parent = HarnessDoc(
        name="parent",
        surfaces=[
            *base.surfaces,
            Surface(id="prompt:extra", kind=SurfaceKind.PROMPT, content="extra"),
        ],
    )

    def payload(*, reverse: bool, core: str = "core changed", extra: str = "extra changed") -> str:
        ops = [
            {
                "op": "replace",
                "surface_id": "prompt:core",
                "content": core,
                "rationale": "change the main instruction",
            },
            {
                "op": "replace",
                "surface_id": "prompt:extra",
                "content": extra,
                "rationale": "change the supporting instruction",
            },
        ]
        if reverse:
            ops.reverse()
        return json.dumps(
            {
                "expected_effect": "the two prompt sections work together",
                "preconditions": {},
                "ops": ops,
            }
        )

    core_surface = parent.surface("prompt:core")
    assert core_surface is not None
    no_op = _payload(parent, core_surface.content)
    project = _RepairingProject(
        [payload(reverse=False), payload(reverse=True), no_op],
        [
            {
                2: payload(reverse=True, extra="independent extra"),
                3: _payload(parent, "nonempty semantic change"),
            }
        ],
    )

    proposals = ProjectDeltaProposer(project, meta_agent(), _Provider("unused")).propose_batch(
        parent, _trigger(), "inspect failures", history=[], count=3
    )

    assert all(isinstance(proposal, HarnessDelta) for proposal in proposals)
    report = json.loads(project.files["context/iteration-0001/proposal-validation-attempt-01.json"])
    reasons = [error["reason"] for error in report["errors"]]
    assert any("duplicates valid sibling proposal-01" in reason for reason in reasons)
    assert any("semantic no-op" in reason for reason in reasons)


def test_project_proposer_never_returns_a_delta_that_remains_invalid_after_two_repairs() -> None:
    parent = HarnessDoc.baseline("parent")
    invalid = _skill_payload("Still missing required frontmatter.")
    project = _RepairingProject([invalid], [{1: invalid}, {1: invalid}])

    proposals = ProjectDeltaProposer(project, meta_agent(), _Provider("unused")).propose_batch(
        parent, _trigger(), "inspect failures", history=[], count=1
    )

    assert project.runs == 3
    assert "skill file has no frontmatter" in _proposal_failure_reason(proposals[0])
    final_report = json.loads(
        project.files["context/iteration-0001/proposal-validation-attempt-03.json"]
    )
    assert final_report["valid_slots"] == []
    assert "skill file has no frontmatter" in final_report["errors"][0]["reason"]


def test_project_proposer_repairs_partial_durable_outputs_after_runner_disconnect() -> None:
    parent = HarnessDoc.baseline("parent")
    project = _RepairingProject(
        [_payload(parent, "careful"), _skill_payload("missing frontmatter")],
        [
            {
                2: _skill_payload(
                    "---\nname: parse-json\ndescription: Parse JSON responses\n---\nParse first."
                )
            }
        ],
        initial_error="Server disconnected after durable writes",
    )

    proposals = ProjectDeltaProposer(project, meta_agent(), _Provider("unused")).propose_batch(
        parent, _trigger(), "inspect failures", history=[], count=2
    )

    assert project.runs == 2
    assert all(isinstance(proposal, HarnessDelta) for proposal in proposals)
    final_report = json.loads(
        project.files["context/iteration-0001/proposal-validation-attempt-02.json"]
    )
    assert final_report["valid_slots"] == [1, 2]
    assert final_report["errors"] == []


def test_project_proposer_rejects_a_runtime_kind_switch_when_fixed_before_search() -> None:
    parent = HarnessDoc.baseline("parent")
    switch = json.dumps(
        {
            "expected_effect": "run the vendored node harness",
            "preconditions": {},
            "ops": [
                {
                    "op": "add",
                    "surface_id": "param:runtime-kind",
                    "kind": "param",
                    "content": "pi-node",
                    "rationale": "switch execution engines",
                }
            ],
        }
    )
    project = _RepairingProject([switch], [{1: switch}, {1: switch}])

    proposals = ProjectDeltaProposer(
        project,
        meta_agent(),
        _Provider("unused"),
        preserve_runtime_kind=True,
    ).propose_batch(parent, _trigger(), "inspect failures", history=[], count=1)

    assert "must preserve the parent's runtime kind 'kit-python'" in _proposal_failure_reason(
        proposals[0]
    )
    report = json.loads(project.files["context/iteration-0001/proposal-validation-attempt-03.json"])
    assert "must preserve the parent's runtime kind 'kit-python'" in report["errors"][0]["reason"]


def test_project_proposer_allows_runtime_kind_switch_for_generic_local_search() -> None:
    parent = HarnessDoc.baseline("parent")
    switch = json.dumps(
        {
            "expected_effect": "run the vendored node harness",
            "preconditions": {},
            "ops": [
                {
                    "op": "add",
                    "surface_id": "param:runtime-kind",
                    "kind": "param",
                    "content": "pi-node",
                    "rationale": "switch execution engines",
                }
            ],
        }
    )
    project = _Project([switch])

    proposals = ProjectDeltaProposer(project, meta_agent(), _Provider("unused")).propose_batch(
        parent, _trigger(), "inspect failures", history=[], count=1
    )

    proposal = proposals[0]
    assert isinstance(proposal, HarnessDelta)
    assert apply_delta(parent, proposal, "child").runtime_kind() == "pi-node"


def test_project_proposer_repairs_an_unknown_runtime_kind_before_search() -> None:
    parent = HarnessDoc.baseline("parent")
    invalid = json.dumps(
        {
            "expected_effect": "run a misspelled execution engine",
            "preconditions": {},
            "ops": [
                {
                    "op": "add",
                    "surface_id": "param:runtime-kind",
                    "kind": "param",
                    "content": "pi-nod",
                    "rationale": "switch execution engines",
                }
            ],
        }
    )
    project = _RepairingProject([invalid], [{1: invalid}, {1: invalid}])

    proposals = ProjectDeltaProposer(project, meta_agent(), _Provider("unused")).propose_batch(
        parent, _trigger(), "inspect failures", history=[], count=1
    )

    assert "unsupported runtime kind 'pi-nod'" in _proposal_failure_reason(proposals[0])
    report = json.loads(project.files["context/iteration-0001/proposal-validation-attempt-03.json"])
    assert "unsupported runtime kind 'pi-nod'" in report["errors"][0]["reason"]


def test_project_proposer_salvages_outputs_written_before_runner_disconnect() -> None:
    parent = HarnessDoc.baseline("parent")
    project = _InterruptedProject([_payload(parent, "careful")])

    proposals = ProjectDeltaProposer(project, meta_agent(), _Provider("unused")).propose_batch(
        parent, _trigger(), "inspect failures", history=[], count=2
    )

    assert isinstance(proposals[0], HarnessDelta)
    assert proposals[1] == ProposalFailure(reason="Server disconnected after durable writes")


def test_project_proposer_only_salvages_fully_parsed_outputs_after_failure() -> None:
    parent = HarnessDoc.baseline("parent")
    project = _InterruptedProject([_payload(parent, "careful"), "{"])

    proposals = ProjectDeltaProposer(project, meta_agent(), _Provider("unused")).propose_batch(
        parent, _trigger(), "inspect failures", history=[], count=3
    )

    assert isinstance(proposals[0], HarnessDelta)
    assert proposals[1:] == [
        ProposalFailure(reason="Server disconnected after durable writes"),
        ProposalFailure(reason="Server disconnected after durable writes"),
    ]


def test_project_proposer_reports_the_exact_clean_malformed_output_failure() -> None:
    parent = HarnessDoc.baseline("parent")
    project = _Project(["{"])

    proposals = ProjectDeltaProposer(project, meta_agent(), _Provider("unused")).propose_batch(
        parent, _trigger(), "inspect failures", history=[], count=1
    )

    assert _proposal_failure_reason(proposals[0]) == (
        "proposal is not a parseable typed delta JSON object"
    )


def test_project_proposer_marks_every_missing_output_as_a_proposal_failure() -> None:
    parent = HarnessDoc.baseline("parent")
    project = _FailedProject([])

    proposals = ProjectDeltaProposer(project, meta_agent(), _Provider("unused")).propose_batch(
        parent, _trigger(), "inspect failures", history=[], count=3
    )

    assert proposals == [ProposalFailure(reason="provider down")] * 3
