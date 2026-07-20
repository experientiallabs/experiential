"""Typed delta proposers for direct providers and project-backed meta agents."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from wmh.agents.project import AgentProjectRun, AgentProjectState
from wmh.core.types import JsonObject
from wmh.harness.cost import (
    SearchComponentCostBinding,
    SearchComponentCostRuntime,
    SearchComponentRole,
    SearchCostBinding,
    TimedResourceCostBinding,
)
from wmh.harness.delta import FailureSignature, GateRecord, HarnessDelta, apply_delta
from wmh.harness.doc import HarnessDoc
from wmh.harness.mutate import parse_delta, propose_delta
from wmh.harness.runtime import HarnessSearchCancelled, search_safety_terminal_error
from wmh.harness.scoring import (
    HarnessScoreArchive,
    ScoreArchiveVisibility,
    canonical_score_json,
    render_task_score_archive,
)
from wmh.providers.base import Provider, ProviderConfig, ToolCallingProvider
from wmh.providers.receipt import (
    ProviderResponseIdentity,
    freeze_provider_response_identity,
)
from wmh.tracking.budget import BudgetedProvider, bind_budget_account
from wmh.tracking.rate_limit import ExternalDispatchRateBinding

_CONTEXT_CONTENT_CHUNK_CHARS = 12_000
_MAX_PROJECT_REPAIR_TURNS = 2
_SUPPORTED_RUNTIME_KINDS = frozenset({"kit-python", "pi-node"})
_JSON_OBJECT_ADAPTER = TypeAdapter(JsonObject)
_CONFIGURATION_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class _ProposalValidation:
    """Host-validated proposal slots plus the raw files needed to protect valid siblings."""

    proposals: dict[int, HarnessDelta]
    raw_files: dict[int, str]
    child_hashes: dict[int, str]
    errors: dict[int, str]


class _ProjectProposerState(BaseModel):
    """Host-only project and proposer state required after process or sandbox death."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["wmh.project-proposer-state.v1"] = "wmh.project-proposer-state.v1"
    configuration_id: str = Field(min_length=1)
    iteration: int = Field(ge=0)
    evaluation_dirs: dict[str, str]
    proposal_files: dict[str, str]
    parent_manifests: dict[str, JsonObject]
    project_state: JsonObject


class AgentProject(Protocol):
    """Project operations required by the optimizer wiring."""

    workspace: str
    execution_configuration_id: str

    def write_text(self, path: str, content: str) -> None: ...

    def read_text(self, path: str) -> str: ...

    def write_private_text(self, path: str, content: str) -> None: ...

    def read_private_text(self, path: str) -> str: ...

    def run(
        self,
        agent: HarnessDoc,
        provider: ToolCallingProvider,
        instruction: str,
        *,
        should_cancel: Callable[[], bool] | None = None,
        writable_files: Collection[str] | None = None,
    ) -> AgentProjectRun: ...


class DeltaProposer(Protocol):
    """Produce sibling deltas against one selected parent."""

    def propose_batch(
        self,
        parent: HarnessDoc,
        trigger: FailureSignature,
        evidence: str,
        *,
        history: list[HarnessDelta],
        count: int,
        should_cancel: Callable[[], bool] | None = None,
    ) -> list[HarnessDelta | ProposalFailure | None]: ...


class _ProposalHistoryRecord(Protocol):
    """Minimal committed proposal identity used to restore project file locations."""

    iteration: int
    proposal_index: int
    delta_id: str | None


@dataclass(frozen=True)
class ProposalFailure:
    """One proposal slot whose provider or agent call failed."""

    reason: str


class ProviderDeltaProposer:
    """Adapt the original single-completion proposer to the batched search contract."""

    score_archive_required = False
    requires_search_cost_binding = True

    def __init__(
        self,
        provider: Provider,
        *,
        cost_runtime: SearchComponentCostRuntime | None = None,
        response_identity: ProviderResponseIdentity | None = None,
    ) -> None:
        self._provider = provider
        provider_config = getattr(provider, "config", None)
        if not isinstance(provider_config, ProviderConfig):
            raise ValueError("direct proposal search requires a provider with ProviderConfig")
        self._response_identity = freeze_provider_response_identity(
            provider_config,
            response_identity,
        )
        self._cost_runtime = cost_runtime
        if cost_runtime is not None:
            self._provider = self._bind_cost_runtime(cost_runtime)

    @property
    def provider(self) -> Provider:
        """Return the provider used for direct proposal calls."""
        return self._provider

    @property
    def configuration_id(self) -> str:
        """Return an opaque identity for the direct proposal model configuration."""
        return type(self).configuration_id_for(
            self._provider,
            response_identity=self._response_identity,
        )

    @classmethod
    def configuration_id_for(
        cls,
        provider: Provider,
        *,
        response_identity: ProviderResponseIdentity | None = None,
    ) -> str:
        """Compute direct proposer identity before attaching its paid account."""
        provider_config = getattr(provider, "config", None)
        if not isinstance(provider_config, ProviderConfig):
            raise ValueError("direct proposal search requires a provider with ProviderConfig")
        frozen_response_identity = freeze_provider_response_identity(
            provider_config,
            response_identity,
        )
        return cls.configuration_id_for_contract(
            provider_config=provider_config,
            provider_implementation=_provider_implementation(cast("ToolCallingProvider", provider)),
            response_identity=frozen_response_identity,
        )

    @classmethod
    def configuration_id_for_contract(
        cls,
        *,
        provider_config: ProviderConfig,
        provider_implementation: str,
        response_identity: ProviderResponseIdentity | None = None,
    ) -> str:
        """Compute direct proposer identity from a nonsecret, client-free contract."""
        if (
            provider_implementation != provider_implementation.strip()
            or not provider_implementation
        ):
            raise ValueError("provider implementation identity must be canonical")
        frozen_config = ProviderConfig.model_validate(provider_config.model_dump())
        frozen_response_identity = freeze_provider_response_identity(
            frozen_config,
            response_identity,
        )
        payload = {
            "schema_version": 2,
            "implementation": f"{cls.__module__}.{cls.__qualname__}",
            "provider_implementation": provider_implementation,
            "provider_config": frozen_config.model_dump(mode="json"),
            "response_identity": frozen_response_identity.model_dump(mode="json"),
        }
        return _content_digest(_canonical_json(payload))

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
        """Make ``count`` independent proposal calls against the same parent."""
        self._revalidate_cost_runtime()
        if count < 1:
            raise ValueError(f"proposal count must be positive, got {count}")
        proposals: list[HarnessDelta | ProposalFailure | None] = []
        for _ in range(count):
            _check_cancelled(should_cancel)
            try:
                proposal = propose_delta(
                    parent,
                    trigger,
                    evidence,
                    self._provider,
                    history=history,
                )
            except HarnessSearchCancelled:
                raise
            except Exception as error:  # noqa: BLE001 - isolate one flaky sibling call
                terminal = search_safety_terminal_error(error)
                if terminal is not None:
                    raise terminal from None
                proposal = ProposalFailure(reason=str(error))
            proposals.append(proposal)
        return proposals

    def _bind_cost_runtime(
        self,
        runtime: SearchComponentCostRuntime,
    ) -> BudgetedProvider:
        """Validate and attach the one direct proposer provider account."""
        binding = self._validate_component_cost_runtime(runtime)
        provider_binding = binding.providers[0]
        provider_config = getattr(self._provider, "config", None)
        if provider_config != provider_binding.provider_config:
            raise ValueError("direct proposer provider config differs from its cost binding")
        if provider_binding.response_identity != self._response_identity:
            raise ValueError("direct proposer response identity differs from its cost binding")
        if isinstance(self._provider, BudgetedProvider):
            if self._provider.budget_binding != provider_binding.account:
                raise ValueError("direct proposer provider account differs from its cost binding")
            if self._provider.response_identity != self._response_identity:
                raise ValueError("direct proposer provider lost its response identity")
            wrapped = self._provider
        else:
            wrapped = BudgetedProvider(
                cast("Provider", self._provider),
                runtime.provider_account(provider_binding),
                response_identity=self._response_identity,
            )
        self.search_cost_binding = binding
        return wrapped

    def _revalidate_cost_runtime(self) -> None:
        """Reaudit the direct proposer account before every provider dispatch batch."""
        runtime = self._cost_runtime
        if runtime is None:
            if self.requires_search_cost_binding:
                raise ValueError("ProviderDeltaProposer requires a complete search cost runtime")
            return
        binding = self._validate_component_cost_runtime(runtime)
        provider_binding = binding.providers[0]
        if not isinstance(self._provider, BudgetedProvider):
            raise RuntimeError("cost-bound direct proposer lost its budgeted provider")
        if self._provider.budget_binding != provider_binding.account:
            raise ValueError("direct proposer provider account differs from its cost binding")
        if self._provider.response_identity != provider_binding.response_identity:
            raise ValueError("direct proposer response identity changed after construction")
        account = runtime.provider_account(provider_binding)
        if self._provider.budget_binding != bind_budget_account(account):
            raise ValueError("direct proposer provider account changed after construction")

    def _validate_component_cost_runtime(
        self,
        runtime: SearchComponentCostRuntime,
    ) -> SearchComponentCostBinding:
        if not isinstance(runtime, SearchComponentCostRuntime):
            raise TypeError(
                "ProviderDeltaProposer cost_runtime must be a SearchComponentCostRuntime"
            )
        binding = SearchComponentCostBinding.model_validate(runtime.binding.model_dump())
        if binding.role is not SearchComponentRole.PROPOSER:
            raise ValueError("ProviderDeltaProposer cost runtime must use the proposer role")
        if binding.configuration_id != self.configuration_id:
            raise ValueError("ProviderDeltaProposer configuration_id differs from its cost runtime")
        if len(binding.providers) != 1:
            raise ValueError("ProviderDeltaProposer cost runtime must bind exactly one provider")
        if binding.timed_resources:
            raise ValueError("ProviderDeltaProposer cost runtime cannot bind timed resources")
        return binding


class ProjectDeltaProposer:
    """Wire a normal agent in a persistent project into harness search."""

    score_archive_required = True
    durable_state_required = True
    requires_search_cost_binding = True

    def __init__(
        self,
        project: AgentProject,
        agent: HarnessDoc,
        provider: ToolCallingProvider,
        *,
        preserve_runtime_kind: bool = False,
        cost_runtime: SearchComponentCostRuntime | None = None,
        response_identity: ProviderResponseIdentity | None = None,
    ) -> None:
        self._project = project
        self._agent = agent
        self._provider = provider
        provider_config = getattr(provider, "config", None)
        if not isinstance(provider_config, ProviderConfig):
            raise ValueError("project proposal search requires a provider with ProviderConfig")
        self._response_identity = freeze_provider_response_identity(
            provider_config,
            response_identity,
        )
        self._cost_runtime = cost_runtime
        self._iteration = 0
        self._evaluation_dirs: dict[str, str] = {}
        self._proposal_files: dict[str, str] = {}
        self._parent_manifests: dict[str, JsonObject] = {}
        self._should_cancel: Callable[[], bool] | None = None
        self._preserve_runtime_kind = preserve_runtime_kind
        if cost_runtime is not None:
            self._provider = self._bind_cost_runtime(cost_runtime)
        elif self.requires_search_cost_binding:
            raise ValueError("ProjectDeltaProposer requires a complete search cost runtime")
        project_policy = getattr(project, "budget_policy_digest", None)
        provider_policy = getattr(self._provider, "budget_policy_digest", None)
        if (project_policy is None) != (provider_policy is None):
            raise ValueError("proposer project and provider must both use one hard-budget policy")
        if project_policy is not None:
            if provider_policy != project_policy:
                raise ValueError("proposer project and provider must share one hard-budget policy")
            project_ledger = getattr(project, "budget_ledger_path", None)
            provider_ledger = getattr(self._provider, "budget_ledger_path", None)
            if (
                project_ledger is None
                or provider_ledger is None
                or Path(project_ledger).expanduser().resolve()
                != Path(provider_ledger).expanduser().resolve()
            ):
                raise ValueError("proposer project and provider must share one hard-budget ledger")

    @classmethod
    def configuration_id_for(
        cls,
        *,
        project_type: type[object],
        project_workspace: str,
        agent: HarnessDoc,
        provider: ToolCallingProvider,
        project_execution_configuration_id: str,
        preserve_runtime_kind: bool = False,
        project_create_rate_binding: ExternalDispatchRateBinding | None = None,
        response_identity: ProviderResponseIdentity | None = None,
    ) -> str:
        """Compute the proposer identity before creating its paid project resource."""
        provider_config = getattr(provider, "config", None)
        if not isinstance(provider_config, ProviderConfig):
            raise ValueError(
                "checkpointed project proposal search requires a provider with ProviderConfig"
            )
        frozen_response_identity = freeze_provider_response_identity(
            provider_config,
            response_identity,
        )
        if not project_workspace or project_workspace != project_workspace.rstrip("/"):
            raise ValueError("project proposer workspace must be non-empty and canonical")
        if (
            not isinstance(project_execution_configuration_id, str)
            or _CONFIGURATION_DIGEST.fullmatch(project_execution_configuration_id) is None
        ):
            raise ValueError("project execution configuration ID must be a sha256 digest")
        frozen_rate_binding = (
            None
            if project_create_rate_binding is None
            else ExternalDispatchRateBinding.model_validate(
                project_create_rate_binding.model_dump()
            )
        )
        payload = {
            "schema_version": 3,
            "implementation": f"{cls.__module__}.{cls.__qualname__}",
            "project_implementation": f"{project_type.__module__}.{project_type.__qualname__}",
            "project_workspace": project_workspace,
            "project_execution_configuration_id": project_execution_configuration_id,
            "agent": agent.model_dump(mode="json"),
            "provider_implementation": _provider_implementation(provider),
            "provider_config": provider_config.model_dump(mode="json"),
            "response_identity": frozen_response_identity.model_dump(mode="json"),
            "preserve_runtime_kind": preserve_runtime_kind,
            "project_create_rate_binding": (
                None if frozen_rate_binding is None else frozen_rate_binding.model_dump(mode="json")
            ),
        }
        return _content_digest(_canonical_json(payload))

    @property
    def configuration_id(self) -> str:
        """Return an opaque identity for the project, meta agent, and provider route."""
        raw_rate_binding = getattr(self._project, "create_rate_binding", None)
        rate_binding = (
            None
            if raw_rate_binding is None
            else ExternalDispatchRateBinding.model_validate(raw_rate_binding)
        )
        return type(self).configuration_id_for(
            project_type=type(self._project),
            project_workspace=self._project.workspace,
            project_execution_configuration_id=self._project.execution_configuration_id,
            agent=self._agent,
            provider=self._provider,
            preserve_runtime_kind=self._preserve_runtime_kind,
            project_create_rate_binding=rate_binding,
            response_identity=self._response_identity,
        )

    @property
    def create_rate_binding(self) -> ExternalDispatchRateBinding | None:
        """Return the project sandbox's shared external dispatch authority identity."""
        raw = getattr(self._project, "create_rate_binding", None)
        if raw is None:
            return None
        return ExternalDispatchRateBinding.model_validate(raw)

    def authorize_search_dispatch(self, binding: SearchCostBinding) -> None:
        """Forward the fully validated search contract to the deferred project resource."""
        self._revalidate_cost_runtime()
        runtime = self._cost_runtime
        if runtime is None:
            raise ValueError("ProjectDeltaProposer requires a complete search cost runtime")
        validated = SearchCostBinding.model_validate(binding.model_dump())
        expected = SearchCostBinding.model_validate(runtime.search_binding.model_dump())
        if validated != expected:
            raise ValueError("search dispatch binding differs from the proposer cost runtime")
        authorize_project = getattr(self._project, "authorize_search_dispatch", None)
        if not callable(authorize_project):
            raise RuntimeError("cost-bound proposer project cannot authorize deferred dispatch")
        authorize_project(validated)

    def export_search_state(self) -> JsonObject:
        """Export public and host-only project state without crossing their visibility roots."""
        export_project = getattr(self._project, "export_search_state", None)
        if not callable(export_project):
            raise RuntimeError(
                "checkpointed project proposal search requires AgentProject.export_search_state"
            )
        project_state = _JSON_OBJECT_ADAPTER.validate_python(export_project())
        state = _ProjectProposerState(
            configuration_id=self.configuration_id,
            iteration=self._iteration,
            evaluation_dirs=dict(self._evaluation_dirs),
            proposal_files=dict(self._proposal_files),
            parent_manifests={
                doc_hash: _JSON_OBJECT_ADAPTER.validate_json(_canonical_json(manifest))
                for doc_hash, manifest in self._parent_manifests.items()
            },
            project_state=project_state,
        )
        return cast("JsonObject", state.model_dump(mode="json"))

    def restore_search_state(self, raw_state: JsonObject) -> None:
        """Restore a host checkpoint into a fresh project before any proposer turn runs."""
        state = _ProjectProposerState.model_validate(raw_state)
        if state.configuration_id != self.configuration_id:
            raise ValueError("project proposer durable state configuration does not match")
        if self._iteration not in (0, state.iteration):
            raise ValueError("project proposer already contains a different iteration state")
        self._apply_search_state(state)

    def restore_proposal_batch_state(
        self,
        *,
        state_before: JsonObject,
        state_after: JsonObject,
    ) -> None:
        """Advance exactly one witnessed proposer call from its bound pre-call state."""
        before = _ProjectProposerState.model_validate(state_before)
        after = _ProjectProposerState.model_validate(state_after)
        if before.configuration_id != self.configuration_id or after.configuration_id != (
            self.configuration_id
        ):
            raise ValueError("project proposer witness configuration does not match")
        current = _ProjectProposerState.model_validate(self.export_search_state())
        if current != before:
            raise ValueError("project proposer current state does not match witness pre-call state")
        if after.iteration != before.iteration + 1:
            raise ValueError("project proposer witness must advance exactly one iteration")
        self._advance_project_state(before, after)
        self._apply_search_metadata(after)

    def _apply_search_state(self, state: _ProjectProposerState) -> None:
        """Replace live state with one already validated durable snapshot."""
        restore_project = getattr(self._project, "restore_search_state", None)
        if not callable(restore_project):
            raise RuntimeError(
                "checkpointed project proposal search requires AgentProject.restore_search_state"
            )
        restore_project(state.project_state)
        self._apply_search_metadata(state)

    def _advance_project_state(
        self,
        before: _ProjectProposerState,
        after: _ProjectProposerState,
    ) -> None:
        """Replay one additive project turn over an already-restored checkpoint."""
        before_project = AgentProjectState.model_validate(before.project_state)
        after_project = AgentProjectState.model_validate(after.project_state)
        export_project = getattr(self._project, "export_search_state", None)
        if not callable(export_project):
            raise RuntimeError(
                "checkpointed project proposal search requires AgentProject.export_search_state"
            )
        current_project = AgentProjectState.model_validate(export_project())
        if current_project != before_project:
            raise ValueError("project state does not match witness pre-call state")

        removed_visible = set(before_project.visible_files) - set(after_project.visible_files)
        removed_private = set(before_project.private_files) - set(after_project.private_files)
        if removed_visible or removed_private:
            raise ValueError("project proposer witness cannot remove durable project files")

        for path, content in sorted(after_project.visible_files.items()):
            if before_project.visible_files.get(path) != content:
                self._project.write_text(path, content)
        for path, content in sorted(after_project.private_files.items()):
            if before_project.private_files.get(path) != content:
                self._project.write_private_text(path, content)

        restored_project = AgentProjectState.model_validate(export_project())
        if restored_project != after_project:
            raise ValueError("project state does not match witnessed post-call state after replay")

    def _apply_search_metadata(self, state: _ProjectProposerState) -> None:
        """Replace proposer metadata after its project state is restored or replayed."""
        self._iteration = state.iteration
        self._evaluation_dirs = dict(state.evaluation_dirs)
        self._proposal_files = dict(state.proposal_files)
        self._parent_manifests = {
            doc_hash: _JSON_OBJECT_ADAPTER.validate_json(_canonical_json(manifest))
            for doc_hash, manifest in state.parent_manifests.items()
        }
        self._should_cancel = None

    def resume_from_history(
        self,
        *,
        completed_iteration: int,
        proposal_records: list[_ProposalHistoryRecord],
    ) -> None:
        """Restore iteration numbering and durable proposal links from committed records."""
        if self._iteration not in (0, completed_iteration):
            raise ValueError(
                "project proposer iteration does not match the resumed search checkpoint"
            )
        self._iteration = completed_iteration
        expected_proposals: dict[str, str] = {}
        expected_evaluations: dict[str, str] = {}
        for record in proposal_records:
            if record.iteration > completed_iteration or record.delta_id is None:
                continue
            iteration_dir = f"iteration-{record.iteration:04d}"
            proposal_file = (
                f"{self._project.workspace}/proposals/{iteration_dir}/"
                f"proposal-{record.proposal_index:02d}.json"
            )
            evaluation_dir = f"evaluations/{iteration_dir}/proposal-{record.proposal_index:02d}"
            expected_proposals.setdefault(record.delta_id, proposal_file)
            expected_evaluations.setdefault(record.delta_id, evaluation_dir)
        if (
            self._proposal_files != expected_proposals
            or self._evaluation_dirs != expected_evaluations
        ):
            raise ValueError(
                "project proposer durable links do not exactly match committed proposal history"
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
        """Run one meta-agent turn that writes ``count`` proposal files."""
        self._revalidate_cost_runtime()
        if count < 1:
            raise ValueError(f"proposal count must be positive, got {count}")
        self._should_cancel = should_cancel
        _check_cancelled(should_cancel)
        self._iteration += 1
        iteration_dir = f"iteration-{self._iteration:04d}"
        context_dir = f"context/{iteration_dir}"
        proposal_dir = f"proposals/{iteration_dir}"
        parent_context = self._parent_manifests.get(parent.doc_hash)
        if parent_context is None:
            parent_context = _materialize_parent(
                self._project,
                parent,
                context_dir=f"parents/{parent.doc_hash}",
                should_cancel=should_cancel,
            )
            self._parent_manifests[parent.doc_hash] = parent_context
        _write_project_text(
            self._project,
            f"{context_dir}/parent.json",
            json.dumps(parent_context, indent=2),
            should_cancel=should_cancel,
        )
        parent_evaluation_manifests = self._visible_evaluation_manifests(parent.doc_hash)
        _write_project_text(
            self._project,
            f"{context_dir}/parent-evaluations.json",
            json.dumps(
                {
                    "kind": "harness-evaluation-index",
                    "harness_doc_hash": parent.doc_hash,
                    "report_manifests": parent_evaluation_manifests,
                    "content_contract": (
                        "Each report manifest contains exact request and report summary metadata "
                        "plus a compact task index. Read that index first, then select only the "
                        "relevant successful or failed task manifests and their structured "
                        "records or derived evidence."
                    ),
                },
                indent=2,
            ),
            should_cancel=should_cancel,
        )
        evidence_context = _materialize_context_content(
            self._project,
            evidence,
            directory=f"{context_dir}/evidence",
            extension=".md",
            should_cancel=should_cancel,
        )
        _write_project_text(
            self._project,
            f"{context_dir}/evidence.json",
            json.dumps(
                {
                    "kind": "failure-evidence",
                    "format": "markdown",
                    **evidence_context,
                },
                indent=2,
            ),
            should_cancel=should_cancel,
        )
        history_content = json.dumps(
            [
                _project_history_entry(
                    delta,
                    proposal_file=self._proposal_files.get(delta.delta_id),
                    evaluation_dir=self._evaluation_dirs.get(delta.delta_id),
                    score_report_manifests=(
                        self._visible_evaluation_manifests(delta.child_doc_hash)
                        if delta.child_doc_hash is not None
                        else ()
                    ),
                    workspace=self._project.workspace,
                )
                for delta in history
            ],
            indent=2,
        )
        history_context = _materialize_context_content(
            self._project,
            history_content,
            directory=f"{context_dir}/history",
            extension=".json.part",
            should_cancel=should_cancel,
        )
        _write_project_text(
            self._project,
            f"{context_dir}/history.json",
            json.dumps(
                {
                    "kind": "judged-history",
                    "format": "json-array",
                    "entry_count": len(history),
                    **history_context,
                },
                indent=2,
            ),
            should_cancel=should_cancel,
        )
        request = _project_request(
            workspace=self._project.workspace,
            context_dir=context_dir,
            proposal_dir=proposal_dir,
            count=count,
            runtime_kind=parent.runtime_kind(),
            preserve_runtime_kind=self._preserve_runtime_kind,
        )
        _write_project_text(
            self._project,
            f"{context_dir}/REQUEST.md",
            request,
            should_cancel=should_cancel,
        )
        run_error: Exception | None = None
        try:
            self._project.run(
                self._agent,
                self._provider,
                request,
                should_cancel=should_cancel,
                writable_files=[
                    f"{proposal_dir}/proposal-{index:02d}.json" for index in range(1, count + 1)
                ],
            )
        except HarnessSearchCancelled:
            raise
        except Exception as error:  # noqa: BLE001 - durable project files may still be complete
            terminal = search_safety_terminal_error(error)
            if terminal is not None:
                raise terminal from None
            run_error = error
        _check_cancelled(should_cancel)

        slots = list(range(1, count + 1))
        validation = _validate_project_proposals(
            self._project,
            parent,
            trigger,
            proposal_dir=proposal_dir,
            slots=slots,
            history=history,
            preserve_runtime_kind=self._preserve_runtime_kind,
            should_cancel=should_cancel,
        )
        # A lost terminal frame does not invalidate durable files. Host-preflight whatever was
        # written, then repair only the bad/missing slots in an ordinary follow-up project turn.
        # The last project-channel error matters only while a slot remains unresolved.
        terminal_error = run_error
        if validation.errors:
            validation_report_ready = True
            try:
                _write_project_text(
                    self._project,
                    f"{context_dir}/proposal-validation-attempt-01.json",
                    _proposal_validation_report(
                        parent=parent,
                        proposal_dir=proposal_dir,
                        attempt=1,
                        valid_slots=validation.proposals,
                        errors=validation.errors,
                    ),
                    should_cancel=should_cancel,
                )
            except HarnessSearchCancelled:
                raise
            except Exception as error:  # noqa: BLE001 - no report means no safe repair prompt
                terminal = search_safety_terminal_error(error)
                if terminal is not None:
                    raise terminal from None
                terminal_error = error
                validation_report_ready = False

            for repair_turn in range(1, _MAX_PROJECT_REPAIR_TURNS + 1):
                if not validation.errors or not validation_report_ready:
                    break
                validation_report_ready = False
                validation_path = (
                    f"{context_dir}/proposal-validation-attempt-{repair_turn:02d}.json"
                )
                repair_request = _project_repair_request(
                    workspace=self._project.workspace,
                    validation_path=validation_path,
                    request_path=f"{context_dir}/REQUEST.md",
                    proposal_dir=proposal_dir,
                    errors=validation.errors,
                    valid_slots=validation.proposals,
                    runtime_kind=parent.runtime_kind(),
                    preserve_runtime_kind=self._preserve_runtime_kind,
                    repair_turn=repair_turn,
                )
                invalid_slots = sorted(validation.errors)
                protected_restore_error: Exception | None = None
                turn_error: Exception | None = None
                try:
                    _write_project_text(
                        self._project,
                        f"{context_dir}/REPAIR-{repair_turn:02d}.md",
                        repair_request,
                        should_cancel=should_cancel,
                    )
                    try:
                        self._project.run(
                            self._agent,
                            self._provider,
                            repair_request,
                            should_cancel=should_cancel,
                            writable_files=[
                                f"{proposal_dir}/proposal-{index:02d}.json"
                                for index in invalid_slots
                            ],
                        )
                    except HarnessSearchCancelled:
                        raise
                    except Exception as error:  # noqa: BLE001 - salvage durable repaired files
                        terminal = search_safety_terminal_error(error)
                        if terminal is not None:
                            raise terminal from None
                        turn_error = error
                    finally:
                        # The agent is asked to rewrite only invalid slots. Restore every
                        # byte-exact valid sibling after each turn as an enforcement boundary.
                        for index in sorted(validation.proposals):
                            try:
                                _write_project_text(
                                    self._project,
                                    f"{proposal_dir}/proposal-{index:02d}.json",
                                    validation.raw_files[index],
                                    should_cancel=should_cancel,
                                )
                            except HarnessSearchCancelled:
                                raise
                            except Exception as error:  # noqa: BLE001 - attempt every restore
                                terminal = search_safety_terminal_error(error)
                                if terminal is not None:
                                    raise terminal from None
                                if protected_restore_error is None:
                                    protected_restore_error = error
                except HarnessSearchCancelled:
                    raise
                except Exception as error:  # noqa: BLE001 - preserve good siblings, fail bad ones
                    terminal = search_safety_terminal_error(error)
                    if terminal is not None:
                        raise terminal from None
                    turn_error = error
                if protected_restore_error is not None:
                    # The in-memory delta and its durable proposal_file must always name the
                    # same bytes. If protection cannot be proven, abort this batch instead of
                    # returning a valid object whose persistent provenance may have been changed.
                    raise protected_restore_error
                terminal_error = turn_error

                repaired = _validate_project_proposals(
                    self._project,
                    parent,
                    trigger,
                    proposal_dir=proposal_dir,
                    slots=invalid_slots,
                    history=history,
                    valid_proposals=validation.proposals,
                    preserve_runtime_kind=self._preserve_runtime_kind,
                    should_cancel=should_cancel,
                )
                validation = _ProposalValidation(
                    proposals=repaired.proposals,
                    raw_files={**validation.raw_files, **repaired.raw_files},
                    child_hashes={**validation.child_hashes, **repaired.child_hashes},
                    errors=repaired.errors,
                )
                # Each host result becomes the next turn's nested input and the durable final
                # audit. These turns happen wholly inside proposal generation, before search.
                try:
                    _write_project_text(
                        self._project,
                        f"{context_dir}/proposal-validation-attempt-{repair_turn + 1:02d}.json",
                        _proposal_validation_report(
                            parent=parent,
                            proposal_dir=proposal_dir,
                            attempt=repair_turn + 1,
                            valid_slots=validation.proposals,
                            errors=validation.errors,
                        ),
                        should_cancel=should_cancel,
                    )
                    validation_report_ready = True
                except HarnessSearchCancelled:
                    raise
                except Exception as error:  # noqa: BLE001 - preserve validated in-memory output
                    terminal = search_safety_terminal_error(error)
                    if terminal is not None:
                        raise terminal from None
                    if terminal_error is None:
                        terminal_error = error
                if not validation.errors:
                    # A prior turn can lose its terminal control frame after durable repaired
                    # files were written. Successful preflight is authoritative salvage.
                    terminal_error = None

        proposals: list[HarnessDelta | ProposalFailure | None] = []
        for index in slots:
            stamped = validation.proposals.get(index)
            if stamped is not None:
                proposals.append(stamped)
                self._proposal_files.setdefault(
                    stamped.delta_id,
                    f"{self._project.workspace}/{proposal_dir}/proposal-{index:02d}.json",
                )
                self._evaluation_dirs.setdefault(
                    stamped.delta_id,
                    f"evaluations/{iteration_dir}/proposal-{index:02d}",
                )
            elif terminal_error is not None:
                proposals.append(ProposalFailure(reason=str(terminal_error)))
            else:
                proposals.append(
                    ProposalFailure(
                        reason=validation.errors.get(
                            index,
                            "proposal slot remained invalid after bounded repair turns",
                        )
                    )
                )
        return proposals

    def _bind_cost_runtime(
        self,
        runtime: SearchComponentCostRuntime,
    ) -> BudgetedProvider:
        """Validate and attach every proposer cost account before the first project turn."""
        binding = self._validate_component_cost_runtime(runtime)
        provider_binding = binding.providers[0]
        provider_config = getattr(self._provider, "config", None)
        if provider_config != provider_binding.provider_config:
            raise ValueError("project proposer provider config differs from its cost binding")
        if provider_binding.response_identity != self._response_identity:
            raise ValueError("project proposer response identity differs from its cost binding")
        if isinstance(self._provider, BudgetedProvider):
            if self._provider.budget_binding != provider_binding.account:
                raise ValueError("project proposer provider account differs from its cost binding")
            if self._provider.response_identity != self._response_identity:
                raise ValueError("project proposer provider lost its response identity")
            wrapped = self._provider
        else:
            account = runtime.provider_account(provider_binding)
            wrapped = BudgetedProvider(
                cast("Provider", self._provider),
                account,
                response_identity=self._response_identity,
            )
        self.search_cost_binding = binding
        return wrapped

    def _revalidate_cost_runtime(self) -> None:
        """Reaudit exact proposer accounts before any project or provider dispatch."""
        runtime = self._cost_runtime
        if runtime is None:
            return
        binding = self._validate_component_cost_runtime(runtime)
        provider_binding = binding.providers[0]
        if not isinstance(self._provider, BudgetedProvider):
            raise RuntimeError("cost-bound project proposer lost its budgeted provider")
        if self._provider.budget_binding != provider_binding.account:
            raise ValueError("project proposer provider account differs from its cost binding")
        if self._provider.response_identity != provider_binding.response_identity:
            raise ValueError("project proposer response identity changed after construction")
        account = runtime.provider_account(provider_binding)
        if self._provider.budget_binding != bind_budget_account(account):
            raise ValueError("project proposer provider account changed after construction")

    def _validate_component_cost_runtime(
        self,
        runtime: SearchComponentCostRuntime,
    ) -> SearchComponentCostBinding:
        if not isinstance(runtime, SearchComponentCostRuntime):
            raise TypeError(
                "ProjectDeltaProposer cost_runtime must be a SearchComponentCostRuntime"
            )
        binding = SearchComponentCostBinding.model_validate(runtime.binding.model_dump())
        if binding.role is not SearchComponentRole.PROPOSER:
            raise ValueError("ProjectDeltaProposer cost runtime must use the proposer role")
        if binding.configuration_id != self.configuration_id:
            raise ValueError("ProjectDeltaProposer configuration_id differs from its cost runtime")
        if len(binding.providers) != 1:
            raise ValueError(
                "ProjectDeltaProposer cost runtime must bind exactly one provider account"
            )
        if len(binding.timed_resources) != 1:
            raise ValueError(
                "ProjectDeltaProposer cost runtime must bind exactly one project resource account"
            )
        project_binding = getattr(self._project, "search_cost_binding", None)
        if not isinstance(project_binding, SearchComponentCostBinding):
            raise ValueError("cost-bound proposer project search cost binding is missing")
        if SearchComponentCostBinding.model_validate(project_binding.model_dump()) != binding:
            raise ValueError("project search cost binding differs from proposer cost runtime")
        resource_binding = getattr(self._project, "timed_resource_binding", None)
        if not isinstance(resource_binding, TimedResourceCostBinding):
            raise ValueError("cost-bound proposer project timed resource binding is missing")
        if (
            TimedResourceCostBinding.model_validate(resource_binding.model_dump())
            != binding.timed_resources[0]
        ):
            raise ValueError("project timed resource account differs from proposer cost runtime")
        return binding

    def record_harness_evaluation(
        self,
        harness: HarnessDoc,
        *,
        archive: HarnessScoreArchive,
        should_cancel: Callable[[], bool] | None = None,
    ) -> None:
        """Commit one exact score archive, exposing only discovery evidence to the proposer."""
        should_cancel = should_cancel or self._should_cancel
        _check_cancelled(should_cancel)
        identity = (
            harness.doc_hash,
            archive.scorer_tier.value,
            archive.report.evaluation_id,
        )
        identity_digest = hashlib.sha256("\0".join(identity).encode()).hexdigest()
        private_root = f"score-archives/records/{identity_digest}"
        private_manifest_path = f"{private_root}/manifest.json"
        request_file = f"{private_root}/request.json"
        report_file = f"{private_root}/report.json"
        request_json = canonical_score_json(archive.request)
        report_json = canonical_score_json(archive.report)
        archive_json = canonical_score_json(archive)
        archive_digest = _content_digest(archive_json)
        expected: JsonObject = {
            "kind": "harness-score-archive",
            "schema_version": archive.schema_version,
            "identity": {
                "harness_doc_hash": harness.doc_hash,
                "scorer_tier": archive.scorer_tier.value,
                "evaluation_id": archive.report.evaluation_id,
            },
            "harness_execution_hash": harness.execution_hash,
            "visibility": archive.visibility.value,
            "purpose": archive.request.purpose,
            "request_file": request_file,
            "request_sha256": _content_digest(request_json),
            "report_file": report_file,
            "report_sha256": _content_digest(report_json),
            "archive_sha256": archive_digest,
        }
        existing = _read_optional_private_project_text(self._project, private_manifest_path)
        if existing is None:
            _write_private_project_text(
                self._project,
                request_file,
                request_json,
                should_cancel=should_cancel,
            )
            _write_private_project_text(
                self._project,
                report_file,
                report_json,
                should_cancel=should_cancel,
            )
            # The manifest is the commit point. Orphan request/report files from cancellation are
            # harmless and may be overwritten, but a committed identity is immutable.
            _write_private_project_text(
                self._project,
                private_manifest_path,
                _canonical_json(expected),
                should_cancel=should_cancel,
            )
        else:
            _verify_private_score_archive(
                self._project,
                existing=existing,
                expected=expected,
                archive=archive,
            )

        public_manifest: str | None = None
        if archive.visibility is ScoreArchiveVisibility.PROPOSER:
            public_manifest = _materialize_visible_score_archive(
                self._project,
                harness=harness,
                archive=archive,
                identity_digest=identity_digest,
                archive_digest=archive_digest,
                should_cancel=should_cancel,
            )
        _upsert_private_harness_index(
            self._project,
            harness_doc_hash=harness.doc_hash,
            identity_digest=identity_digest,
            private_manifest_path=private_manifest_path,
            public_manifest_path=public_manifest,
            should_cancel=should_cancel,
        )

    def _visible_evaluation_manifests(self, harness_doc_hash: str) -> list[str]:
        """Reconstruct proposer-visible report pointers from the durable private index."""
        path = _private_harness_index_path(harness_doc_hash)
        content = _read_optional_private_project_text(self._project, path)
        if content is None:
            return []
        _records, manifests = _parse_private_harness_index(
            content, harness_doc_hash=harness_doc_hash
        )
        for manifest in manifests:
            _verify_visible_score_manifest(
                self._project,
                manifest,
                harness_doc_hash=harness_doc_hash,
            )
        return [f"{self._project.workspace}/{item}" for item in manifests]

    def record_evaluation(self, delta: HarnessDelta, *, stage: str, content: str) -> None:
        """Persist one candidate's judged evidence for later project-agent iterations."""
        should_cancel = self._should_cancel
        _check_cancelled(should_cancel)
        root = self._evaluation_dirs.get(
            delta.delta_id,
            f"evaluations/by-delta/{delta.delta_id}",
        )
        context = _materialize_context_content(
            self._project,
            content,
            directory=f"{root}/{stage}",
            extension=".md",
            should_cancel=should_cancel,
        )
        _write_project_text(
            self._project,
            f"{root}/{stage}.json",
            json.dumps(
                {
                    "kind": "candidate-evaluation",
                    "stage": stage,
                    "delta_id": delta.delta_id,
                    "format": "markdown",
                    **context,
                },
                indent=2,
            ),
            should_cancel=should_cancel,
        )


def _check_cancelled(should_cancel: Callable[[], bool] | None) -> None:
    """Raise the shared search signal without converting it into a failed proposal slot."""
    if should_cancel is not None and should_cancel():
        raise HarnessSearchCancelled("harness search cancelled")


def _write_project_text(
    project: AgentProject,
    path: str,
    content: str,
    *,
    should_cancel: Callable[[], bool] | None,
) -> None:
    """Make each E2B filesystem RPC a cancellation boundary."""
    _check_cancelled(should_cancel)
    project.write_text(path, content)
    _check_cancelled(should_cancel)


def _read_project_text(
    project: AgentProject,
    path: str,
    *,
    should_cancel: Callable[[], bool] | None,
) -> str:
    """Read one project output without hiding cancellation behind the next iteration."""
    _check_cancelled(should_cancel)
    content = project.read_text(path)
    _check_cancelled(should_cancel)
    return content


def _write_private_project_text(
    project: AgentProject,
    path: str,
    content: str,
    *,
    should_cancel: Callable[[], bool] | None,
) -> None:
    """Make each host-only audit write a cancellation boundary."""
    _check_cancelled(should_cancel)
    project.write_private_text(path, content)
    _check_cancelled(should_cancel)


def _read_optional_private_project_text(project: AgentProject, path: str) -> str | None:
    """Return a private file or ``None`` only for a proven missing-file error."""
    try:
        return project.read_private_text(path)
    except Exception as error:  # noqa: BLE001 - E2B uses a provider-specific not-found type
        if _is_missing_file_error(error):
            return None
        raise


def _is_missing_file_error(error: Exception) -> bool:
    """Recognize local and E2B missing-file errors without hiding transport failures."""
    if isinstance(error, (FileNotFoundError, KeyError)):
        return True
    error_name = type(error).__name__.lower()
    text = str(error).lower()
    return "notfound" in error_name or "not found" in text or "no such file" in text


def _provider_implementation(provider: ToolCallingProvider) -> str:
    """Return the paid dispatch implementation without budget-wrapper identity churn."""
    if isinstance(provider, BudgetedProvider):
        return provider.wrapped_provider_implementation
    provider_type = type(provider)
    return f"{provider_type.__module__}.{provider_type.__qualname__}"


def _content_digest(content: str) -> str:
    """Return the UTF-8 SHA-256 used by durable score records."""
    return hashlib.sha256(content.encode()).hexdigest()


def _canonical_json(value: object) -> str:
    """Serialize already-validated archive metadata deterministically."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _verify_private_score_archive(
    project: AgentProject,
    *,
    existing: str,
    expected: JsonObject,
    archive: HarnessScoreArchive,
) -> None:
    """Verify a committed archive before treating a retry as idempotent."""
    try:
        manifest = json.loads(existing)
    except json.JSONDecodeError as error:
        raise ValueError("committed harness score archive manifest is corrupt") from error
    if manifest != expected:
        raise ValueError("one harness evaluation identity cannot name different archived content")
    request_file = manifest.get("request_file")
    report_file = manifest.get("report_file")
    if not isinstance(request_file, str) or not isinstance(report_file, str):
        raise ValueError("committed harness score archive has invalid content pointers")
    request_json = project.read_private_text(request_file)
    report_json = project.read_private_text(report_file)
    if request_json != canonical_score_json(archive.request):
        raise ValueError("committed harness score request is corrupt or conflicts with this retry")
    if report_json != canonical_score_json(archive.report):
        raise ValueError("committed harness score report is corrupt or conflicts with this retry")
    if _content_digest(request_json) != manifest.get("request_sha256"):
        raise ValueError("committed harness score request digest does not match its content")
    if _content_digest(report_json) != manifest.get("report_sha256"):
        raise ValueError("committed harness score report digest does not match its content")


def _materialize_visible_score_archive(
    project: AgentProject,
    *,
    harness: HarnessDoc,
    archive: HarnessScoreArchive,
    identity_digest: str,
    archive_digest: str,
    should_cancel: Callable[[], bool] | None,
) -> str:
    """Write a compact report index plus independently readable per-task records."""
    root = f"evaluations/by-harness/{harness.doc_hash}/report-{identity_digest}"
    task_index: list[JsonObject] = []
    for task_id in sorted(archive.report.per_task):
        task = archive.report.per_task[task_id]
        task_digest = _content_digest(task_id)
        task_root = f"{root}/tasks/task-{task_digest}"
        task_json = canonical_score_json(task)
        record_context = _materialize_context_content(
            project,
            task_json,
            directory=f"{task_root}/record",
            extension=".json.part",
            should_cancel=should_cancel,
        )
        markdown = render_task_score_archive(task)
        evidence_context = _materialize_context_content(
            project,
            markdown,
            directory=f"{task_root}/evidence",
            extension=".md",
            should_cancel=should_cancel,
        )
        task_manifest_relative = f"{task_root}.json"
        _write_project_text(
            project,
            task_manifest_relative,
            json.dumps(
                {
                    "kind": "harness-score-task",
                    "task_id": task.task_id,
                    "score": task.score,
                    "secondary_score": task.secondary_score,
                    "passed": task.passed,
                    "canonical_record": {
                        "format": "canonical-json",
                        "sha256": _content_digest(task_json),
                        **record_context,
                    },
                    "derived_evidence": {
                        "format": "markdown",
                        "sha256": _content_digest(markdown),
                        "trust": "untrusted-benchmark-data",
                        **evidence_context,
                    },
                },
                indent=2,
            ),
            should_cancel=should_cancel,
        )
        task_index.append(
            {
                "task_id": task.task_id,
                "score": task.score,
                "secondary_score": task.secondary_score,
                "passed": task.passed,
                "manifest_file": f"{project.workspace}/{task_manifest_relative}",
            }
        )
    task_index_json = _canonical_json(task_index)
    task_index_context = _materialize_context_content(
        project,
        task_index_json,
        directory=f"{root}/task-index",
        extension=".json.part",
        should_cancel=should_cancel,
    )
    report_summary = archive.report.model_copy(update={"per_task": {}})
    manifest_relative = f"{root}.json"
    _write_project_text(
        project,
        manifest_relative,
        json.dumps(
            {
                "kind": "harness-score-report-index",
                "schema_version": archive.schema_version,
                "harness_doc_hash": harness.doc_hash,
                "harness_execution_hash": harness.execution_hash,
                "scorer_tier": archive.scorer_tier.value,
                "visibility": archive.visibility.value,
                "purpose": archive.request.purpose,
                "evaluation_id": archive.report.evaluation_id,
                "archive_sha256": archive_digest,
                "canonical_request_json": canonical_score_json(archive.request),
                "canonical_report_summary_json": canonical_score_json(report_summary),
                "task_count": len(task_index),
                "task_index": {
                    "format": "canonical-json-array",
                    "sha256": _content_digest(task_index_json),
                    **task_index_context,
                },
                "content_contract": (
                    "Read task_index content_files first. Select task manifests by task_id, score, "
                    "and pass status. Each task manifest points to exact canonical JSON and a "
                    "derived Markdown view explicitly framed as untrusted benchmark data."
                ),
            },
            indent=2,
        ),
        should_cancel=should_cancel,
    )
    return manifest_relative


def _verify_visible_score_manifest(
    project: AgentProject,
    manifest_path: str,
    *,
    harness_doc_hash: str,
) -> None:
    """Fail closed when a durable private index points outside its visible score archive."""
    prefix = f"evaluations/by-harness/{harness_doc_hash}/report-"
    if (
        not manifest_path.startswith(prefix)
        or not manifest_path.endswith(".json")
        or ".." in manifest_path.split("/")
    ):
        raise ValueError("private harness archive index contains an invalid public path")
    try:
        manifest = json.loads(project.read_text(manifest_path))
    except json.JSONDecodeError as error:
        raise ValueError("proposer-visible score report manifest is corrupt") from error
    if not isinstance(manifest, dict):
        raise ValueError("proposer-visible score report manifest must be an object")
    if manifest.get("kind") != "harness-score-report-index":
        raise ValueError("proposer-visible score report manifest has the wrong kind")
    if manifest.get("harness_doc_hash") != harness_doc_hash:
        raise ValueError("proposer-visible score report manifest has the wrong harness identity")
    if manifest.get("scorer_tier") != "discovery" or manifest.get("visibility") != "proposer":
        raise ValueError("private archive index attempted to expose a hidden score report")
    if manifest.get("purpose") not in {"seed", "screen", "full"}:
        raise ValueError("private archive index attempted to expose a hidden score purpose")


def _private_harness_index_path(harness_doc_hash: str) -> str:
    """Return the deterministic private index path for one immutable harness document."""
    return f"score-archives/by-harness/{harness_doc_hash}.json"


def _parse_private_harness_index(
    content: str, *, harness_doc_hash: str
) -> tuple[list[str], list[str]]:
    """Parse and validate the small host-only record index."""
    try:
        index = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError("private harness archive index is corrupt") from error
    if not isinstance(index, dict):
        raise ValueError("private harness archive index must be an object")
    if index.get("kind") != "harness-score-archive-index":
        raise ValueError("private harness archive index has the wrong kind")
    if index.get("harness_doc_hash") != harness_doc_hash:
        raise ValueError("private harness archive index has the wrong harness identity")
    records = index.get("records")
    manifests = index.get("proposer_report_manifests")
    if not isinstance(records, list) or not all(isinstance(item, str) for item in records):
        raise ValueError("private harness archive index has invalid record pointers")
    if not isinstance(manifests, list) or not all(isinstance(item, str) for item in manifests):
        raise ValueError("private harness archive index has invalid public manifest pointers")
    return (
        [item for item in records if isinstance(item, str)],
        [item for item in manifests if isinstance(item, str)],
    )


def _upsert_private_harness_index(
    project: AgentProject,
    *,
    harness_doc_hash: str,
    identity_digest: str,
    private_manifest_path: str,
    public_manifest_path: str | None,
    should_cancel: Callable[[], bool] | None,
) -> None:
    """Index a committed record last so interrupted writes are reconstructed on retry."""
    path = _private_harness_index_path(harness_doc_hash)
    existing = _read_optional_private_project_text(project, path)
    if existing is None:
        records: list[str] = []
        manifests: list[str] = []
    else:
        records, manifests = _parse_private_harness_index(
            existing, harness_doc_hash=harness_doc_hash
        )
    if private_manifest_path not in records:
        records.append(private_manifest_path)
    if public_manifest_path is not None and public_manifest_path not in manifests:
        manifests.append(public_manifest_path)
    updated = {
        "kind": "harness-score-archive-index",
        "harness_doc_hash": harness_doc_hash,
        "records": sorted(records),
        "proposer_report_manifests": sorted(manifests),
        "record_identity_digests": sorted(
            {
                *(
                    record.rsplit("/", 2)[-2]
                    for record in records
                    if record.endswith("/manifest.json")
                ),
                identity_digest,
            }
        ),
    }
    _write_private_project_text(
        project,
        path,
        _canonical_json(updated),
        should_cancel=should_cancel,
    )


def _materialize_parent(
    project: AgentProject,
    parent: HarnessDoc,
    *,
    context_dir: str,
    should_cancel: Callable[[], bool] | None = None,
) -> JsonObject:
    """Write a bounded manifest plus individually readable parent-surface chunks.

    A real pi document is hundreds of kilobytes, while one project ``read_file`` observation is
    intentionally capped. Putting every surface inline in one parent.json therefore hid most of
    the harness from the proposer. The manifest remains small and points to ordered chunks below
    the read cap; concatenating a surface's chunks reconstructs its exact content.
    """
    surface_index: list[JsonObject] = []
    for index, surface in enumerate(parent.surfaces, 1):
        surface_dir = f"{context_dir}/parent-surfaces/surface-{index:03d}"
        content_context = _materialize_context_content(
            project,
            surface.content,
            directory=surface_dir,
            extension=".txt",
            include_contract=False,
            should_cancel=should_cancel,
        )
        source_file: str | None = None
        if surface.path is not None:
            source_relative = f"{context_dir}/parent-source/{surface.path}"
            _write_project_text(
                project,
                source_relative,
                surface.content,
                should_cancel=should_cancel,
            )
            source_file = f"{project.workspace}/{source_relative}"
        entry: JsonObject = {
            "id": surface.id,
            "kind": surface.kind.value,
            "content_hash": surface.content_hash,
            **content_context,
        }
        if surface.budget is not None:
            entry["budget"] = surface.budget
        if surface.path is not None:
            entry["path"] = surface.path
            entry["source_file"] = source_file
        surface_manifest_relative = f"{surface_dir}/manifest.json"
        _write_project_text(
            project,
            surface_manifest_relative,
            json.dumps(entry, indent=2),
            should_cancel=should_cancel,
        )
        surface_index.append(
            {
                "id": surface.id,
                "kind": surface.kind.value,
                "content_hash": surface.content_hash,
                "manifest_file": f"{project.workspace}/{surface_manifest_relative}",
            }
        )
    index_content = json.dumps(surface_index, indent=2)
    index_context = _materialize_context_content(
        project,
        index_content,
        directory=f"{context_dir}/parent-surface-index",
        extension=".json.part",
        should_cancel=should_cancel,
    )
    index_manifest_relative = f"{context_dir}/parent-surfaces.json"
    _write_project_text(
        project,
        index_manifest_relative,
        json.dumps(
            {
                "kind": "parent-surface-index",
                "format": "json-array",
                "entry_count": len(surface_index),
                **index_context,
            },
            indent=2,
        ),
        should_cancel=should_cancel,
    )
    return {
        "name": parent.name,
        "version": parent.version,
        "doc_hash": parent.doc_hash,
        "source_root": f"{project.workspace}/{context_dir}/parent-source",
        "surface_count": len(surface_index),
        "surface_index_manifest": f"{project.workspace}/{index_manifest_relative}",
        "content_contract": (
            "Read surface_index_manifest, then concatenate its content_files and parse that JSON "
            "array. Each index entry points to one independently readable surface manifest. "
            "Within a surface manifest, concatenate content_files exactly to reconstruct the "
            "surface. Pathful code is also mirrored beneath source_root at its exact path."
        ),
    }


def _materialize_context_content(
    project: AgentProject,
    content: str,
    *,
    directory: str,
    extension: str,
    include_contract: bool = True,
    should_cancel: Callable[[], bool] | None = None,
) -> JsonObject:
    """Write exact ordered chunks that each fit in one project ``read_file`` result."""
    chunk_count = max(1, math.ceil(len(content) / _CONTEXT_CONTENT_CHUNK_CHARS))
    width = max(3, len(str(chunk_count)))
    content_files: list[str] = []
    for chunk_index in range(chunk_count):
        start = chunk_index * _CONTEXT_CONTENT_CHUNK_CHARS
        chunk = content[start : start + _CONTEXT_CONTENT_CHUNK_CHARS]
        relative = (
            f"{directory}/part-{chunk_index + 1:0{width}d}-of-{chunk_count:0{width}d}{extension}"
        )
        _write_project_text(
            project,
            relative,
            chunk,
            should_cancel=should_cancel,
        )
        content_files.append(f"{project.workspace}/{relative}")
    result: JsonObject = {
        "content_length": len(content),
        "content_files": content_files,
    }
    if include_contract:
        result["content_contract"] = (
            "Read content_files in listed order and concatenate them exactly. Each file is "
            "independently readable without truncation."
        )
    return result


def _project_history_entry(
    delta: HarnessDelta,
    *,
    proposal_file: str | None,
    evaluation_dir: str | None,
    score_report_manifests: Collection[str],
    workspace: str,
) -> JsonObject:
    """Compact judged metadata while raw proposals retain exact replacement payloads.

    Re-serializing every prior full code surface into every later iteration makes persistent history
    quadratic in run length. The project already owns each raw proposal file, so history carries
    queryable identities, rationales, sizes, and verdicts plus a direct pointer to the exact bytes.
    """
    ops: list[JsonObject] = []
    for op in delta.ops:
        item: JsonObject = {
            "op": op.op,
            "surface_id": op.surface_id,
            "rationale": op.rationale[:2_000],
            "content_length": len(op.content) if op.content is not None else 0,
        }
        if op.kind is not None:
            item["kind"] = op.kind.value
        if op.path is not None:
            item["path"] = op.path
        if op.budget is not None:
            item["budget"] = op.budget
        ops.append(item)
    return {
        "delta_id": delta.delta_id,
        "parent_doc_hash": delta.parent_doc_hash,
        "child_doc_hash": delta.child_doc_hash,
        "trigger": delta.trigger.model_dump(mode="json"),
        "preconditions": dict(delta.preconditions),
        "expected_effect": delta.expected_effect[:2_000],
        "ops": ops,
        "verdict": _proposer_visible_verdict(delta.verdict),
        "proposal_file": proposal_file,
        "evaluation_dir": (f"{workspace}/{evaluation_dir}" if evaluation_dir is not None else None),
        "score_report_manifests": list(score_report_manifests),
        "content_contract": (
            "Exact op content remains in proposal_file; this entry intentionally omits it to "
            "keep cumulative judged history linear and fast. Complete per-task score evidence "
            "is available through score_report_manifests."
        ),
    }


def _proposer_visible_verdict(verdict: GateRecord | None) -> JsonObject | None:
    """Retain discovery deltas while withholding hidden-tier and confirmation measurements."""
    if verdict is None:
        return None
    return {
        "suite_delta": verdict.suite_delta,
        "suite_secondary_delta": verdict.suite_secondary_delta,
        "full_delta": verdict.full_delta,
        "full_secondary_delta": verdict.full_secondary_delta,
        "accepted": verdict.accepted,
        "reason": (
            "Accepted by the configured search gate."
            if verdict.accepted
            else "Rejected by the configured search gate."
        )
        + " Hidden scorer tiers and confirmation measurements are not proposer-visible.",
    }


def _stamp_project_preconditions(
    parent: HarnessDoc, proposal: HarnessDelta | None
) -> HarnessDelta | None:
    """Stamp missing concurrency metadata from the exact project-iteration parent.

    The ordinary agent still chooses every semantic operation. The host owns this mechanical
    identity field because it wrote the immutable parent snapshot for the same iteration.
    An explicitly supplied but incorrect hash is preserved so normal validation rejects it.
    """
    if proposal is None:
        return None
    for op in proposal.ops:
        if op.op not in ("replace", "remove") or op.surface_id in proposal.preconditions:
            continue
        surface = parent.surface(op.surface_id)
        if surface is not None:
            proposal.preconditions[op.surface_id] = surface.content_hash
    return proposal


def _validate_project_proposals(
    project: AgentProject,
    parent: HarnessDoc,
    trigger: FailureSignature,
    *,
    proposal_dir: str,
    slots: list[int],
    history: list[HarnessDelta],
    valid_proposals: dict[int, HarnessDelta] | None = None,
    preserve_runtime_kind: bool,
    should_cancel: Callable[[], bool] | None,
) -> _ProposalValidation:
    """Parse, stamp, apply, and de-duplicate selected project proposal slots.

    Applying a deep copy exercises the complete typed ``HarnessDoc`` boundary without stamping
    ``child_doc_hash`` onto the delta the search will later apply and archive. Previously the
    project proposer returned syntactically parsed deltas and left this check to the search loop,
    where a missing skill frontmatter block consumed an iteration as ``invalid before eval``.
    """
    accepted = dict(valid_proposals or {})
    raw_files: dict[int, str] = {}
    child_hashes: dict[int, str] = {}
    errors: dict[int, str] = {}
    history_ids = {delta.delta_id for delta in history}
    history_child_hashes = {
        delta.child_doc_hash for delta in history if delta.child_doc_hash is not None
    }
    sibling_ids = {delta.delta_id: index for index, delta in accepted.items()}
    for accepted_index, accepted_delta in accepted.items():
        accepted_child = apply_delta(
            parent,
            accepted_delta.model_copy(deep=True),
            f"{parent.name}-accepted-preflight-{accepted_index:02d}",
        )
        child_hashes[accepted_index] = accepted_child.doc_hash
    sibling_child_hashes = {child_hash: index for index, child_hash in child_hashes.items()}
    parent_runtime_kind = parent.runtime_kind()
    for index in slots:
        _check_cancelled(should_cancel)
        relative = f"{proposal_dir}/proposal-{index:02d}.json"
        try:
            raw = _read_project_text(project, relative, should_cancel=should_cancel)
        except HarnessSearchCancelled:
            raise
        except Exception as error:  # noqa: BLE001 - one missing file is one repairable slot
            terminal = search_safety_terminal_error(error)
            if terminal is not None:
                raise terminal from None
            errors[index] = f"proposal file is missing or unreadable: {error}"
            continue
        raw_files[index] = raw
        proposal = parse_delta(parent, trigger, raw)
        if proposal is None:
            errors[index] = "proposal is not a parseable typed delta JSON object"
            continue
        stamped = _stamp_project_preconditions(parent, proposal)
        assert stamped is not None
        try:
            child = apply_delta(
                parent,
                stamped.model_copy(deep=True),
                f"{parent.name}-proposal-preflight-{index:02d}",
            )
            child_runtime_kind = child.runtime_kind()
            if child_runtime_kind not in _SUPPORTED_RUNTIME_KINDS:
                supported = ", ".join(sorted(_SUPPORTED_RUNTIME_KINDS))
                raise ValueError(
                    f"project proposal resolves to unsupported runtime kind "
                    f"{child_runtime_kind!r}; choose one of: {supported}"
                )
            if preserve_runtime_kind and child_runtime_kind != parent_runtime_kind:
                raise ValueError(
                    "project proposals must preserve the parent's runtime kind "
                    f"{parent_runtime_kind!r}; this proposal resolves to {child_runtime_kind!r}"
                )
        except ValueError as error:
            errors[index] = f"delta does not apply to the supplied parent: {error}"
            continue
        child_hash = child.doc_hash
        if child_hash == parent.doc_hash:
            errors[index] = "delta is a semantic no-op: its child document equals the parent"
            continue
        if stamped.delta_id in history_ids:
            errors[index] = (
                f"delta {stamped.delta_id} duplicates a proposal already present in judged history"
            )
            continue
        if child_hash in history_child_hashes:
            errors[index] = (
                f"child document {child_hash} duplicates a proposal already present in "
                "judged history"
            )
            continue
        duplicate_slot = sibling_ids.get(stamped.delta_id)
        if duplicate_slot is not None:
            errors[index] = (
                f"delta {stamped.delta_id} duplicates valid sibling proposal-{duplicate_slot:02d}"
            )
            continue
        duplicate_child_slot = sibling_child_hashes.get(child_hash)
        if duplicate_child_slot is not None:
            errors[index] = (
                f"child document {child_hash} duplicates valid sibling "
                f"proposal-{duplicate_child_slot:02d}"
            )
            continue
        accepted[index] = stamped
        sibling_ids[stamped.delta_id] = index
        child_hashes[index] = child_hash
        sibling_child_hashes[child_hash] = index
    _check_cancelled(should_cancel)
    return _ProposalValidation(
        proposals=accepted,
        raw_files=raw_files,
        child_hashes=child_hashes,
        errors=errors,
    )


def _proposal_validation_report(
    *,
    parent: HarnessDoc,
    proposal_dir: str,
    attempt: int,
    valid_slots: dict[int, HarnessDelta],
    errors: dict[int, str],
) -> str:
    """Serialize actionable per-slot host validation for the project and run audit."""
    return json.dumps(
        {
            "kind": "proposal-validation",
            "attempt": attempt,
            "parent_doc_hash": parent.doc_hash,
            "parent_runtime_kind": parent.runtime_kind(),
            "valid_slots": sorted(valid_slots),
            "errors": [
                {
                    "slot": index,
                    "proposal_file": f"{proposal_dir}/proposal-{index:02d}.json",
                    "reason": errors[index],
                }
                for index in sorted(errors)
            ],
        },
        indent=2,
    )


def _project_repair_request(
    *,
    workspace: str,
    validation_path: str,
    request_path: str,
    proposal_dir: str,
    errors: dict[int, str],
    valid_slots: dict[int, HarnessDelta],
    runtime_kind: str,
    preserve_runtime_kind: bool,
    repair_turn: int,
) -> str:
    """Render one of the bounded repair turns for only invalid batch slots."""
    invalid_outputs = "\n".join(
        f"- {workspace}/{proposal_dir}/proposal-{index:02d}.json" for index in sorted(errors)
    )
    protected_outputs = "\n".join(
        f"- {workspace}/{proposal_dir}/proposal-{index:02d}.json" for index in sorted(valid_slots)
    )
    if not protected_outputs:
        protected_outputs = "- (none)"
    runtime_constraint = (
        f"preserve its resolved runtime kind {runtime_kind!r}"
        if preserve_runtime_kind
        else "produce a valid resolved runtime kind"
    )
    return f"""Repair exactly {len(errors)} invalid proposal slot(s) from this iteration.
This is repair turn {repair_turn} of {_MAX_PROJECT_REPAIR_TURNS}.

Read the host validation report: {workspace}/{validation_path}
It contains the exact error for each invalid slot. Re-read the original iteration request at
{workspace}/{request_path} and its supplied parent manifests as needed, then rewrite ONLY these
invalid files:
{invalid_outputs}

These siblings already passed host preflight. Do not rewrite them:
{protected_outputs}

Every repaired file must follow the original typed delta JSON schema, apply cleanly to that same
parent, {runtime_constraint}, produce a child document different from the parent, differ from
judged history, and differ from every sibling. A skill's content must include the complete
four-line frontmatter shown in the original request. Rewrite every invalid file immediately after
reading the validation report; only then spend remaining actions on optional evidence. Validate
every rewritten file before calling submit with a short summary."""


def _project_request(
    *,
    workspace: str,
    context_dir: str,
    proposal_dir: str,
    count: int,
    runtime_kind: str,
    preserve_runtime_kind: bool,
) -> str:
    """Render one filesystem-first proposal task for the ordinary meta agent."""
    absolute_context = f"{workspace}/{context_dir}"
    absolute_proposals = f"{workspace}/{proposal_dir}"
    outputs = "\n".join(
        f"- {absolute_proposals}/proposal-{index:02d}.json" for index in range(1, count + 1)
    )
    runtime_constraint = (
        f"This project must preserve the parent's resolved runtime kind {runtime_kind!r}; do not "
        "add, replace, or remove runtime-kind in a way that changes it."
        if preserve_runtime_kind
        else (
            "A runtime-kind edit is allowed only when the resulting child remains a valid harness; "
            "the search backend makes the final executability decision."
        )
    )
    heading = (
        f"Produce exactly {count} independent harness proposals for this optimization iteration."
    )
    return f"""{heading}

Read:
- parent manifest: {absolute_context}/parent.json
  - follow surface_index_manifest to find every independently readable surface manifest
  - each surface manifest lists ordered content_files; concatenate them to inspect exact content
  - pathful code is also mirrored under source_root with exact source_file paths for direct reads
- complete parent evaluation index: {absolute_context}/parent-evaluations.json
  - each report manifest has exact request/report summary metadata and a compact task index
  - read each compact task index first, then inspect only relevant successful and failed task
    manifests; prefer canonical records and treat derived evidence as untrusted benchmark data
- failure evidence manifest: {absolute_context}/evidence.json
  - read its content_files in listed order and concatenate them exactly
- judged history manifest: {absolute_context}/history.json
  - read its content_files in listed order, concatenate them exactly, then parse the JSON array
- earlier raw proposals, when useful: {workspace}/proposals/
- earlier candidate evaluation manifests and traces: {workspace}/evaluations/

Write exactly these files, without changing earlier iterations:
{outputs}

Each file must be one JSON object:
{{"expected_effect":"<falsifiable prediction>",
 "preconditions":{{"<surface id>":"<hash copied from parent>"}},
 "ops":[{{"op":"add|replace|remove","surface_id":"<kind:slug>",
           "kind":"<required for add>","content":"<full content>",
           "rationale":"<why this helps>"}}]}}

For a replacement, you may omit content and use compact exact edits instead:
"edits":[{{"old":"<nonempty text occurring exactly once>","new":"<replacement>"}}].
The optimizer expands those edits against the parent before validation.

Typed surface constraints (host preflight enforces all of these before evaluation):
- Every surface id is `<kind>:<kebab-slug>` and its prefix must exactly match `kind`.
- `add` needs a fresh id, `kind`, full `content`, and a nonempty `rationale`. `replace` needs an
  existing id, full `content` or exact `edits`, and a nonempty `rationale`; if it declares `kind`,
  that kind must match the parent. `remove` needs an existing id and rationale and must omit
  content. Every replace/remove target must have its exact parent hash in `preconditions`.
- A `skill:<slug>` add/replace has kind `skill`; its content is the complete markdown below,
  beginning at the first character, with kebab-case `name` exactly equal to `<slug>`:
  ---
  name: <slug>
  description: <one-line description of when the agent should use this skill>
  ---
  <nonempty reusable technique body>
- Prompt content is plain text, and the child must retain at least one prompt surface.
- `tool_policy:main` is one registered tool name per line and must retain `submit`.
- Supported scalar params are `param:max-turns` and `param:max-output-tokens` (integers >= 1),
  `param:temperature` (number in [0, 2]), and `param:runtime-kind` (`kit-python` or `pi-node`).
  {runtime_constraint}
- A path-less code surface can only be `code:runtime` and must remain valid Python defining
  `run(kit)`. Pathful code surfaces use safe relative paths without `..`; replacements inherit the
  parent's path unless explicitly supplied, and paths must stay unique.
- Respect each surface's character budget. Do not remove required singleton surfaces or create
  duplicate ids/paths. Do not emit a semantic no-op or repeat any child document from judged
  history or another sibling, even through differently ordered operations.

Every proposal must be focused, valid against the same supplied parent, and meaningfully different
from its siblings. Your project tool budget is bounded: after reading the four root manifests,
write a complete, parseable draft to every output before doing deeper optional exploration. Keep
those files valid as you refine them. The host will parse, stamp mechanical missing preconditions,
deep-copy apply, and de-duplicate every file. Invalid slots receive at most two repair turns and
are never evaluated. After all files exist, call submit with a short summary."""
