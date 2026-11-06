"""The front door's owner-bound slot executors, run against the preparation's saved families.

Each executor prices its work credential-free and runs, or replays, it through the owner's own
entry points; every result is a saved record the next phase reads.

- ``tag-banks`` (:class:`TagBankExecutor`): per admitted environment family, the corpus tag
  author drafts yes/no questions from a sample of the members' task and observation views and
  the selection keeps the discriminating, answerable, non-redundant ones (``corpus_tag_bank``);
  every member is then tagged with ``authored_bank_for_view`` per view by the decision model.
- ``task-types`` (:class:`TaskTypeExecutor`): each family is split into named task types over
  its task-view tags plus the authored matrices (``task_types``); ``task_type_family`` narrows a
  family to one type.
- ``state-rag`` (:class:`StateRagExecutor`): per family, a state-tag bank is authored from
  sampled pre-turn states of its fit steps and every fit step is tagged (``authored_state_tags``,
  ``build_state_tag_index``); the saved index binds into the family's served runtimes as
  ``state_recall`` with the optional ``state_memory`` policy.
- ``rule-admission`` (:class:`RuleAdmissionExecutor`): the family rules already ran under tag
  admission inside the preparation; the executor loads every family's guard table, checks it
  converts to the runtime rule book (``guard_rulebook``) and records the bindings.

The decision model is the admitted native decision alias (Clef by default); authors and namers
are admitted open-weight completion aliases. Collaborators may be injected for offline tests.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import Field

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    sha256_json,
    stable_id,
)
from exp.common.models import ModelRequest, ModelResponse, load_model_catalog
from exp.common.project import ProjectStore, artifact_input
from exp.common.traces import Trace, load_trace_dataset
from exp.optimize.evaluation.executable.costs import evaluation_costs
from exp.optimize.evaluation.executable.model_costs import CostComponent, decision_cost, model_cost
from exp.optimize.evaluation.executable.state_recall import StateRecallBinding
from exp.optimize.evaluation.factory.authoring import AuthorSettings, author_response
from exp.optimize.evaluation.fidelity.decision_access import decision_caller
from exp.optimize.evaluation.rules.memory_admission import load_guard_table
from exp.optimize.evaluation.variance.bootstrap_plan import (
    BootstrapDependencies,
    BootstrapPreparationReport,
    BootstrapPreparationSpec,
)
from exp.optimize.evaluation.variance.build import plan_variance_build, run_variance_build
from exp.optimize.evaluation.variance.build_records import VarianceBuildReport, load_tag_matrix
from exp.optimize.evaluation.variance.dense_prompt import dense_model_calls
from exp.optimize.evaluation.variance.dense_vocabulary import inflow_dense, vocabulary
from exp.optimize.evaluation.variance.family_simulators import load_family_simulator
from exp.optimize.evaluation.variance.pipeline_review import family_reviewer, review_costs
from exp.optimize.evaluation.variance.pipeline_spec import CostLine, cost_lines
from exp.runtime.gateway.decisions_contracts import MAX_DECISION_QUESTIONS
from exp.runtime.models.providers.admitted_transport import AdmittedJsonTransport
from exp.runtime.models.providers.transport import HttpxJsonTransport
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.runtime.sessions.dispatch_ledger import DispatchLedger
from exp.simulation.mining.decisions import DEFAULT_DECISION_ALIAS, DecisionsTraceTagger
from exp.simulation.mining.environment_families import (
    EnvironmentFamily,
    admitted_families,
    load_environment_families,
)
from exp.simulation.mining.questions import resolve_question_bank
from exp.simulation.mining.tag_authoring import (
    TagAuthor,
    TagAuthoringSpec,
    authored_bank_for_view,
)
from exp.simulation.mining.tag_bank_selection import (
    AuthoredBank,
    BankSelectionSpec,
    corpus_tag_bank,
)
from exp.simulation.mining.tag_cluster_names import ClusterNamer
from exp.simulation.mining.tagging import DecisionCache, TagMatrix, TraceTagger
from exp.simulation.mining.task_types import TaskTypeSpec, task_types
from exp.simulation.mining.views import build_units
from exp.simulation.retrieval.contracts import RAGLineageBinding
from exp.simulation.retrieval.state_rag import (
    StateRetrievalPolicy,
    build_state_tag_index,
    plan_state_rag_calls,
    save_state_tag_index,
)
from exp.simulation.retrieval.state_tags import (
    StateTagAuthor,
    StateTagAuthoringSpec,
    authored_state_tags,
    cached_state_decider,
    pre_turn_steps,
    trace_order,
)
from exp.simulation.retrieval.store import load_rag_index
from exp.simulation.rules.admission import guard_rulebook
from exp.simulation.world_model.state_memory import StateMemoryPolicy

SLOT_RECORD_TYPE = "front-door-slot"
"""Artifact type of one slot stage's saved result (``slot.json``)."""

AUTHORED_VIEWS = ("task", "observations")
"""Views the corpus tag author samples and the authored banks are asked over."""


@dataclass(frozen=True)
class SlotContext:
    """What a slot executor reads besides its own settings.

    Attributes:
        project: The project.
        preparation: The front door's preparation registration (build, families, bootstrap).
        code_revision: Producing source identity.
        results: Saved results of slots that already ran, by slot name.
    """

    project: ProjectStore
    preparation: BootstrapPreparationSpec
    code_revision: str
    results: Mapping[str, ArtifactInput] = field(default_factory=dict)


@dataclass(frozen=True)
class Corpus:
    """The preparation's saved variance build, its admitted families and their traces.

    Attributes:
        report: The saved variance build.
        families: Admitted environment families, in review order.
        traces: Every trace of the build's dataset, by identity.
    """

    report: VarianceBuildReport
    families: tuple[EnvironmentFamily, ...]
    traces: Mapping[str, Trace]


_CORPORA: dict[str, Corpus] = {}
"""Saved corpora already read in this process, by build plan identity and project root."""


def saved_corpus(context: SlotContext) -> Corpus | None:
    """Return the saved build and its admitted families, or None before family review.

    A saved build is immutable, so it is read once per process and plan identity.
    """
    build = context.preparation.build
    early = sha256_json(
        {
            "build": build.model_dump(mode="json"),
            "code": context.code_revision,
            "project": str(context.project.artifacts.project_directory),
        }
    )
    if early in _CORPORA:
        return _CORPORA[early]
    reviewer = build.reviewer
    plan = plan_variance_build(
        context.project,
        build.spec,
        code_revision=context.code_revision,
        reviewer_model=None if reviewer is None else reviewer.model,
        reviewer_configuration_sha256=None if reviewer is None else sha256_json(reviewer),
    )
    key = plan.identity + "-result"
    store = context.project.artifacts
    memo = f"{store.project_directory}:{key}"
    if memo in _CORPORA:
        _CORPORA[early] = _CORPORA[memo]
        return _CORPORA[memo]
    if not store.exists(key):
        return None
    report = VarianceBuildReport.model_validate_json(store.read_bytes(key, "report.json"))
    scoping = report.environment_families
    settings = report.spec.environment_families
    if scoping is None or settings is None:
        raise ValueError("front-door slots after the families need an environment-family build")
    traces = {
        row.trace_id: row
        for row in load_trace_dataset(store, report.build.trace_dataset.artifact_id).traces
    }
    record = load_environment_families(store, scoping.families)
    families = admitted_families(record, settings.admit_uncertain, traces=traces)
    _CORPORA[memo] = _CORPORA[early] = Corpus(report=report, families=families, traces=traces)
    return _CORPORA[memo]


def slot_identity(
    name: str,
    settings: JsonObject,
    context: SlotContext,
    corpus: Corpus,
    depends: tuple[str, ...] = (),
) -> str:
    """Return the identity a slot's result is saved under, computed before any call.

    Args:
        name: The slot.
        settings: Its settings.
        context: The front-door context.
        corpus: The saved build and families.
        depends: Earlier results it reads; only these join the identity.
    """
    return stable_id(
        SLOT_RECORD_TYPE,
        {
            "slot": name,
            "settings": sha256_json(settings),
            "build": corpus.report.identity,
            "inputs": {
                key: context.results[key].model_dump(mode="json")
                for key in depends
                if key in context.results
            },
            "code": context.code_revision,
        },
    )


def save_slot(
    context: SlotContext, identity: str, record: ContractModel, inputs: tuple[ArtifactInput, ...]
) -> ArtifactInput:
    """Save one slot result, adopting an identical saved one."""
    store = context.project.artifacts
    if not store.exists(identity):
        store.write_json(
            artifact_id=identity,
            artifact_type=SLOT_RECORD_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=context.code_revision,
                inputs=tuple(sorted(set(inputs), key=lambda ref: ref.artifact_id)),
            ),
            files={"slot.json": record},
        )
    return artifact_input(store.read(identity).manifest)


def read_slot[Record: ContractModel](
    context: SlotContext, reference: ArtifactInput, model: type[Record]
) -> Record:
    """Read one saved slot result, refusing a changed artifact."""
    stored = context.project.artifacts.read(reference.artifact_id)
    if artifact_input(stored.manifest) != reference:
        raise ValueError(f"slot result {reference.artifact_id} changed")
    return model.model_validate_json(stored.payloads["slot.json"])


def review_families(
    project: ProjectStore,
    preparation: BootstrapPreparationSpec,
    code_revision: str,
    collaborators: BootstrapDependencies,
) -> None:
    """Run the preparation's own variance build first, so its saved families exist.

    The call is the one the preparation makes, with the same collaborators, so the preparation
    replays it with no call. It runs only when an ``after-families`` slot needs the families.
    """
    build = preparation.build
    reviewer = collaborators.reviewer or family_reviewer(
        project, build, code_revision=code_revision
    )
    run_variance_build(
        project,
        build.spec,
        code_revision=code_revision,
        tagger=collaborators.tagger,
        reviewer=reviewer,
        namer=collaborators.namer,
    )


def families_review_plan(
    context: SlotContext,
) -> tuple[bool, tuple[CostLine, ...], tuple[str, ...]]:
    """Price the preparation's variance build that ``after-families`` slots need saved first.

    Returns:
        Whether it is saved, its decision and review costs, and its plan line.
    """
    build = context.preparation.build
    reviewer = build.reviewer
    plan = plan_variance_build(
        context.project,
        build.spec,
        code_revision=context.code_revision,
        reviewer_model=None if reviewer is None else reviewer.model,
        reviewer_configuration_sha256=None if reviewer is None else sha256_json(reviewer),
    )
    if plan.saved:
        return True, (), (f"Variance build {plan.identity}: saved, replays with no call.",)
    alias = build.spec.decision_alias
    costs = (
        decision_cost(context.project, alias, plan.decision_attempts),
        *review_costs(context.project, build, plan),
    )
    line = (
        f"Variance build {plan.identity}: {plan.decision_calls} decision calls with {alias} "
        f"and {plan.review_calls} compatibility-review calls, before the families slots."
    )
    return False, cost_lines(costs), (line,)


def admitted(
    project: ProjectStore, purpose: str, attempts: int, code_revision: str, hours: float = 24.0
) -> AdmittedJsonTransport:
    """Return an admitted physical transport bounded by its own attempt ledger."""
    identity = stable_id("front-door-transport", {"purpose": purpose, "code": code_revision})
    return AdmittedJsonTransport(
        HttpxJsonTransport(),
        ledger=DispatchLedger(
            project.artifacts.project_directory / (identity + "-physical.sqlite"),
            max(1, attempts),
            sha256_json(identity),
        ),
        artifacts=project.artifacts,
        code_revision=code_revision,
        deadline=datetime.now(UTC) + timedelta(hours=hours),
    )


def decision_tagger(
    project: ProjectStore, alias: str, purpose: str, attempts: int, code_revision: str
) -> DecisionsTraceTagger:
    """Return the native decision tagger of an alias over the project decision cache."""
    resolved = RuntimeModelCatalog(load_model_catalog(project.model_catalog_path)).resolve(alias)
    transport = admitted(project, purpose, attempts, code_revision).priced_as(alias)
    return DecisionsTraceTagger.for_alias(
        resolved, transport=transport, cache=DecisionCache(project.paths)
    )


def author_draw(
    project: ProjectStore, settings: AuthorSettings, code_revision: str
) -> Callable[..., ModelResponse]:
    """Return a buy-or-replay draw through the admitted author-response owner."""

    def draw(request: ModelRequest, *, identity: str) -> ModelResponse:
        """Buy or replay one exact draw."""
        _, response = author_response(
            project, settings, request, identity=identity, sources=(), code_revision=code_revision
        )
        return response

    return draw


class FamilyBanks(ContractModel):
    """The ``tag-banks`` slot's result.

    Attributes:
        banks: Each family's authored, selected and published bank.
        matrices: Each family's member matrices per view, as saved tag matrices.
    """

    banks: dict[str, AuthoredBank]
    matrices: dict[str, dict[str, TagMatrix]] = Field(default_factory=dict)


class TagBankSlot(ContractModel):
    """Settings of the ``tag-banks`` slot.

    Attributes:
        author: Admitted open-weight author settings.
        authoring: Variants, rounds and sample size per family.
        selection: Discrimination, answerability and redundancy bounds.
        bank_root: Directory the banks are published under; None for the local bank root.
        decision_alias: Native decision alias tagging the selection sample and the members.
    """

    author: AuthorSettings
    authoring: TagAuthoringSpec = TagAuthoringSpec()
    selection: BankSelectionSpec = BankSelectionSpec()
    bank_root: Path | None = None
    decision_alias: str = Field(default=DEFAULT_DECISION_ALIAS, min_length=1)


@dataclass
class TagBankExecutor:
    """Authors, selects, publishes and applies one bank per family.

    Attributes:
        tagger: Decision tagger replacing the admitted default, for offline tests.
    """

    tagger: TraceTagger | None = None

    def plan(
        self, context: SlotContext, settings: JsonObject
    ) -> tuple[bool, tuple[CostLine, ...], tuple[str, ...]]:
        """Price authoring and tagging per family, or report the saved result."""
        slot = TagBankSlot.model_validate(settings)
        corpus = saved_corpus(context)
        if corpus is not None and context.project.artifacts.exists(
            slot_identity("tag-banks", settings, context, corpus)
        ):
            return True, (), ("Saved: authored banks per family; replays with no call.",)
        families = (
            context.preparation.limits.maximum_families if corpus is None else len(corpus.families)
        )
        members = sum(len(row.members) for row in corpus.families) if corpus else 0
        members = members or families * context.preparation.limits.maximum_family_traces
        author = slot.authoring.calls() * families
        chunks = math.ceil(slot.selection.max_questions / MAX_DECISION_QUESTIONS)
        decisions = len(AUTHORED_VIEWS) * members * chunks * 2
        costs: list[CostComponent] = [
            row.scaled(author) for row in evaluation_costs(context.project, slot.author)
        ]
        costs.append(decision_cost(context.project, slot.decision_alias, decisions))
        line = (
            f"{families} families: up to {author} author calls with "
            f"{slot.author.model_alias}, then at most {decisions} decisions with "
            f"{slot.decision_alias} (selection sample and every member, per view)."
        )
        return False, cost_lines(tuple(costs)), (line,)

    def run(self, context: SlotContext, settings: JsonObject) -> ArtifactInput:
        """Run, or replay, every family's bank and member tagging."""
        slot = TagBankSlot.model_validate(settings)
        corpus = saved_corpus(context)
        if corpus is None:
            raise ValueError("the tag-banks slot runs after the families review is saved")
        identity = slot_identity("tag-banks", settings, context, corpus)
        if context.project.artifacts.exists(identity):
            return artifact_input(context.project.artifacts.read(identity).manifest)
        project = context.project
        members_total = sum(len(row.members) for row in corpus.families)
        tagger = self.tagger or decision_tagger(
            project, slot.decision_alias, "tag-banks", members_total * 64, context.code_revision
        )
        author = TagAuthor(
            draw=author_draw(project, slot.author, context.code_revision),
            model=slot.author.model,
            configuration_sha256=sha256_json(slot.author),
        )
        banks: dict[str, AuthoredBank] = {}
        matrices: dict[str, dict[str, TagMatrix]] = {}
        scoping = corpus.report.environment_families
        assert scoping is not None
        inputs: list[ArtifactInput] = [scoping.families]
        for family in corpus.families:
            rendered = [build_units(corpus.traces[t], AUTHORED_VIEWS) for t in family.members]
            units = [unit for row in rendered for unit in row.values()]
            authored = corpus_tag_bank(
                project.artifacts,
                author,
                tagger,
                scope=family.family_id,
                bank_id=f"authored-{family.family_id[:24]}",
                units=units,
                tagger_identity=getattr(tagger, "model_identity", {}),
                authoring=slot.authoring,
                selection=slot.selection,
                root=slot.bank_root,
                code_revision=context.code_revision,
            )
            banks[family.family_id] = authored
            inputs.extend((authored.authoring, authored.selection))
            per_view: dict[str, TagMatrix] = {}
            for view in AUTHORED_VIEWS:
                asked = authored_bank_for_view(authored.bank, view)
                if asked is not None:
                    per_view[view] = tagger.tag([row[view] for row in rendered], asked)
            matrices[family.family_id] = per_view
        record = FamilyBanks(banks=banks, matrices=matrices)
        return save_slot(context, identity, record, tuple(inputs))


class TaskTypeSlot(ContractModel):
    """Settings of the ``task-types`` slot.

    Attributes:
        namer_alias: Admitted open-weight naming alias.
        spec: Second-level clustering and naming.
    """

    namer_alias: str = Field(min_length=1)
    spec: TaskTypeSpec = TaskTypeSpec()


@dataclass
class TaskTypeExecutor:
    """Splits every family into named task types over task and authored tags.

    Attributes:
        namer: Naming model replacing the admitted default, for offline tests.
    """

    namer: ClusterNamer | None = None

    def plan(
        self, context: SlotContext, settings: JsonObject
    ) -> tuple[bool, tuple[CostLine, ...], tuple[str, ...]]:
        """Price naming every possible type of every family."""
        slot = TaskTypeSlot.model_validate(settings)
        corpus = saved_corpus(context)
        if (
            corpus is not None
            and "tag-banks" in context.results
            and context.project.artifacts.exists(
                slot_identity("task-types", settings, context, corpus, ("tag-banks",))
            )
        ):
            return True, (), ("Saved: task types per family; replays with no call.",)
        families = (
            context.preparation.limits.maximum_families if corpus is None else len(corpus.families)
        )
        calls = families * slot.spec.naming.calls(slot.spec.clustering.max_clusters)
        costs = (model_cost(context.project, slot.namer_alias, calls, 1200),)
        return False, cost_lines(costs), (f"Up to {calls} naming calls with {slot.namer_alias}.",)

    def run(self, context: SlotContext, settings: JsonObject) -> ArtifactInput:
        """Run, or replay, every family's task types over the authored banks' matrices."""
        slot = TaskTypeSlot.model_validate(settings)
        corpus = saved_corpus(context)
        banks_ref = context.results.get("tag-banks")
        if corpus is None or banks_ref is None:
            raise ValueError("the task-types slot needs the families and the tag-banks slot")
        identity = slot_identity("task-types", settings, context, corpus, ("tag-banks",))
        project = context.project
        if project.artifacts.exists(identity):
            return artifact_input(project.artifacts.read(identity).manifest)
        banks = read_slot(context, banks_ref, FamilyBanks)
        task_view = next(row for row in corpus.report.tagging if row.plan.view == "task")
        matrices: list[TagMatrix] = [load_tag_matrix(project, task_view.matrix)]
        questions: dict[str, str] = {}
        for family_id, per_view in banks.matrices.items():
            matrices.extend(per_view.values())
            questions.update({q.question_id: q.text for q in banks.banks[family_id].bank.questions})
        namer = self.namer
        if namer is None:
            transport = admitted(project, "task-types", 1000, context.code_revision)
            catalog = load_model_catalog(project.model_catalog_path)
            resolved = RuntimeModelCatalog(catalog, transport_factory=lambda: transport).resolve(
                slot.namer_alias
            )
            namer = ClusterNamer(client=resolved.client, model=resolved.snapshot)
        exemplars = {
            trace_id: build_units(trace, ("task",))["task"].text
            for trace_id, trace in corpus.traces.items()
            if any(trace_id in row.members for row in corpus.families)
        }
        scoping = corpus.report.environment_families
        assert scoping is not None
        record, saved = task_types(
            project.artifacts,
            corpus.families,
            matrices,
            namer,
            families_record=scoping.families.artifact_id,
            questions=questions,
            exemplar_texts=exemplars,
            spec=slot.spec,
            code_revision=context.code_revision,
            inputs=(banks_ref,),
        )
        del record
        return save_slot(context, identity, SlotReference(reference=saved), (saved, banks_ref))


class SlotReference(ContractModel):
    """A slot result that is one saved owner record.

    Attributes:
        reference: The owner's saved record.
    """

    reference: ArtifactInput


class FamilyStateIndexes(ContractModel):
    """The ``state-rag`` slot's result.

    Attributes:
        bindings: Each family's served state recall binding.
        banks: Each family's saved state-tag bank.
    """

    bindings: dict[str, StateRecallBinding]
    banks: dict[str, ArtifactInput]


class StateRagSlot(ContractModel):
    """Settings of the ``state-rag`` slot.

    Attributes:
        author_alias: Admitted open-weight completion alias authoring the state questions.
        authoring: Draft prompts and the bank bound.
        policy: Frozen agreement scoring bound into the served runtimes.
        decision_alias: Native decision alias tagging every fit step and every live state.
        samples: Pre-turn states the author samples per family.
        live_steps: Live steps a served family runtime may tag, for pricing.
        memory: Per-task state description policy bound into every start's simulator.
    """

    author_alias: str = Field(min_length=1)
    authoring: StateTagAuthoringSpec = StateTagAuthoringSpec()
    policy: StateRetrievalPolicy = StateRetrievalPolicy()
    decision_alias: str = Field(default=DEFAULT_DECISION_ALIAS, min_length=1)
    samples: int = Field(default=24, ge=2, le=512)
    live_steps: int = Field(default=0, ge=0)
    memory: StateMemoryPolicy | None = None


def state_rag_preparation(
    context: SlotContext, settings: JsonObject
) -> BootstrapPreparationSpec | None:
    """Return the preparation with each family's saved state recall bound, or None unsaved.

    The bindings and the state memory policy enter the runtime template every start is
    captured in, so every served world of a family retrieves by its own state-tag index.
    """
    corpus = saved_corpus(context)
    if corpus is None:
        return None
    identity = slot_identity("state-rag", settings, context, corpus)
    store = context.project.artifacts
    if not store.exists(identity):
        return None
    record = read_slot(context, artifact_input(store.read(identity).manifest), FamilyStateIndexes)
    slot = StateRagSlot.model_validate(settings)
    preparation = context.preparation
    runtime = preparation.bootstrap.runtime.model_copy(
        update={"state_recall": record.bindings, "state_memory": slot.memory}
    )
    bootstrap = preparation.bootstrap.model_copy(update={"runtime": runtime})
    return preparation.model_copy(update={"bootstrap": bootstrap})


@dataclass
class StateRagExecutor:
    """Authors a state-tag bank per family and tags every fit step's pre-turn state.

    Attributes:
        author: State-tag author replacing the admitted default, for offline tests.
        decide: Native decision call replacing the admitted default, for offline tests.
    """

    author: StateTagAuthor | None = None
    decide: Callable[..., JsonObject] | None = None
    decision_model: JsonObject | None = None

    def plan(
        self, context: SlotContext, settings: JsonObject
    ) -> tuple[bool, tuple[CostLine, ...], tuple[str, ...]]:
        """Price authoring per family and one tagging pass over every fit step."""
        slot = StateRagSlot.model_validate(settings)
        corpus = saved_corpus(context)
        if corpus is not None and context.project.artifacts.exists(
            slot_identity("state-rag", settings, context, corpus)
        ):
            return True, (), ("Saved: state-tag index per family; replays with no call.",)
        families = (
            context.preparation.limits.maximum_families if corpus is None else len(corpus.families)
        )
        steps = 0
        if corpus is not None:
            scoping = corpus.report.environment_families
            assert scoping is not None
            for scope in scoping.scopes:
                steps += len(
                    load_rag_index(context.project.artifacts, scope.index.artifact_id).transitions
                )
        steps = steps or families * context.preparation.limits.maximum_family_traces * 50
        calls = plan_state_rag_calls(
            authoring_calls=slot.authoring.calls * families,
            questions=slot.authoring.maximum_questions,
            corpus_steps=steps,
            live_steps=slot.live_steps,
        )
        costs = (
            model_cost(
                context.project,
                slot.author_alias,
                calls.authoring_completions,
                slot.authoring.maximum_output_tokens,
            ),
            decision_cost(context.project, slot.decision_alias, calls.decisions),
        )
        line = (
            f"{families} families: {calls.authoring_completions} authoring completions with "
            f"{slot.author_alias}, {calls.corpus_decisions} corpus and {calls.live_decisions} "
            f"live decisions with {slot.decision_alias}."
        )
        return False, cost_lines(costs), (line,)

    def run(self, context: SlotContext, settings: JsonObject) -> ArtifactInput:
        """Run, or replay, every family's state-tag bank and index."""
        slot = StateRagSlot.model_validate(settings)
        corpus = saved_corpus(context)
        if corpus is None:
            raise ValueError("the state-rag slot runs after the families review is saved")
        identity = slot_identity("state-rag", settings, context, corpus)
        project = context.project
        if project.artifacts.exists(identity):
            return artifact_input(project.artifacts.read(identity).manifest)
        author, decide, model = self._collaborators(context, slot)
        scoping = corpus.report.environment_families
        assert scoping is not None
        bindings: dict[str, StateRecallBinding] = {}
        banks: dict[str, ArtifactInput] = {}
        for scope in scoping.scopes:
            loaded = load_rag_index(project.artifacts, scope.index.artifact_id)
            traces = sorted({row.trace_id for row in loaded.transitions})
            ordered = trace_order([corpus.traces[t] for t in traces], loaded.transitions)
            steps = [pair for rows in ordered.values() for pair in pre_turn_steps(rows)]
            samples = [state for _, state in steps[:: max(1, len(steps) // slot.samples)]]
            bank, bank_ref = authored_state_tags(
                project.artifacts,
                scope.family_id,
                samples[: slot.samples],
                author,
                spec=slot.authoring,
                code_revision=context.code_revision,
                inputs=(scope.index,),
            )
            lineages = {row.trace_id: row.lineage_id for row in loaded.transitions}
            index = build_state_tag_index(
                bank,
                steps,
                [
                    RAGLineageBinding(trace_id=t, lineage_id=lineages[t], partition="fit")
                    for t in traces
                ],
                decide=decide,
                decision_model=model,
            )
            saved = save_state_tag_index(
                project.artifacts,
                index,
                code_revision=context.code_revision,
                inputs=(bank_ref, scope.index),
            )
            banks[scope.family_id] = bank_ref
            bindings[scope.family_id] = StateRecallBinding(
                index=saved, policy=slot.policy, decision_alias=slot.decision_alias
            )
        record = FamilyStateIndexes(bindings=bindings, banks=banks)
        inputs = (*banks.values(), *(row.index for row in bindings.values()))
        return save_slot(context, identity, record, inputs)

    def _collaborators(
        self, context: SlotContext, slot: StateRagSlot
    ) -> tuple[StateTagAuthor, Callable[..., JsonObject], JsonObject]:
        """Return the author, the cached decision call and the decision model identity."""
        project = context.project
        catalog = load_model_catalog(project.model_catalog_path)
        author = self.author
        if author is None:
            transport = admitted(project, "state-rag-author", 100, context.code_revision)
            resolved = RuntimeModelCatalog(catalog, transport_factory=lambda: transport).resolve(
                slot.author_alias
            )
            author = StateTagAuthor(client=resolved.client, model=resolved.snapshot)
        if self.decide is not None and self.decision_model is not None:
            decide, model = self.decide, self.decision_model
        else:
            native = RuntimeModelCatalog(catalog).resolve(slot.decision_alias)
            transport = admitted(project, "state-rag-decisions", 100_000, context.code_revision)
            model = native.snapshot.model_dump(mode="json")
            deadline = datetime.now(UTC) + timedelta(hours=24)
            decide = decision_caller(native, transport, deadline, lambda: datetime.now(UTC))
        return author, cached_state_decider(DecisionCache(project.paths), decide, model), model


class FamilyGuards(ContractModel):
    """The ``rule-admission`` slot's result.

    Attributes:
        guards: Each family's admitted guard table, by family.
        checkable: Entries of each table the runtime checker enforces.
    """

    guards: dict[str, ArtifactInput]
    checkable: dict[str, int]


@dataclass
class RuleAdmissionExecutor:
    """Records and checks every family's admitted guard table; makes no call."""

    def plan(
        self, context: SlotContext, settings: JsonObject
    ) -> tuple[bool, tuple[CostLine, ...], tuple[str, ...]]:
        """The admission ran, and was priced, inside the preparation's family rules."""
        del settings
        return (
            False,
            (),
            (
                "Family rules are admitted by corpus-wide decision tags inside the preparation "
                "(priced there); this stage records and checks each family's guard table.",
            ),
        )

    def run(self, context: SlotContext, settings: JsonObject) -> ArtifactInput:
        """Load every frozen family simulator's guard table and check it converts.

        Raises:
            ValueError: The preparation has not saved its report.
        """
        del settings
        corpus = saved_corpus(context)
        prepared = context.results.get("preparation")
        if corpus is None or prepared is None:
            raise ValueError("the rule-admission slot runs after the preparation")
        project = context.project
        report = BootstrapPreparationReport.model_validate_json(
            project.artifacts.read_bytes(prepared.artifact_id, "report.json")
        )
        guards: dict[str, ArtifactInput] = {}
        checkable: dict[str, int] = {}
        for reference in report.simulators:
            simulator = load_family_simulator(project.artifacts, reference)
            memory = simulator.rule_memory
            if memory is None or memory.guards is None:
                continue
            table = load_guard_table(project, memory.guards)
            guards[simulator.family_id] = memory.guards
            checkable[simulator.family_id] = len(guard_rulebook(table).rules)
        identity = slot_identity("rule-admission", {}, context, corpus, ("preparation",))
        record = FamilyGuards(guards=guards, checkable=checkable)
        return save_slot(context, identity, record, (prepared, *guards.values()))


PIPELINE_SETTINGS: dict[str, tuple[str, str]] = {
    "blast-radius": ("error-mining", "pipeline.mining.qualification (with its blast reviewers)"),
    "curriculum": ("records", "pipeline.records.schedule"),
    "dense-hybrid": ("records", "in-flow dense vocabulary, or pipeline.records.dense with hybrid"),
}
"""The pipeline stage that runs each in-pipeline slot and the template setting that enables it."""


class PipelineCollection(ContractModel):
    """An in-pipeline slot's cross-start result: what every start's pipeline saved for it.

    Attributes:
        slot: The slot.
        starts: Each start's saved artifacts for the slot, by task.
    """

    slot: str
    starts: dict[str, tuple[ArtifactInput, ...]]


def stage_outputs(context: SlotContext, task_id: str, stage: str) -> JsonObject | None:
    """Return one start's saved stage outputs, or None when that stage is not saved."""
    reference = context.results.get(f"pipeline:{task_id}:{stage}")
    if reference is None:
        return None
    stored = context.project.artifacts.read(reference.artifact_id)
    if artifact_input(stored.manifest) != reference:
        raise ValueError(f"stage record {reference.artifact_id} changed")
    outputs = json.loads(stored.payloads["record.json"])["outputs"]
    assert isinstance(outputs, dict)
    return outputs


@dataclass
class PipelineSlotExecutor:
    """Collects one in-pipeline slot's saved results across every start; makes no call.

    The work itself runs inside each start's pipeline stage (blast radius in error mining, the
    difficulty curriculum and hybrid dense in records), priced in that pipeline's plan.

    Attributes:
        slot: The slot.
    """

    slot: str

    def plan(
        self, context: SlotContext, settings: JsonObject
    ) -> tuple[bool, tuple[CostLine, ...], tuple[str, ...]]:
        """Name the pipeline stage that runs it; its calls are priced in each pipeline."""
        stage, setting = PIPELINE_SETTINGS[self.slot]
        lines = [f"Runs inside each start's {stage} stage ({setting}); collected here."]
        if self.slot == "dense-hybrid" and settings:
            lines.append(
                f"In-flow vocabulary: {settings['tags']} tags frozen from {settings['banks']} "
                "plus each family's mined adversarial linters; per confirmation episode at most "
                f"{settings['decisions_per_episode']} step decisions with "
                f"{settings['decision_alias']} and {settings['judge_calls_per_episode']} judge "
                f"calls with {settings['judge']} on routed tags only."
            )
        done = any(key.startswith("pipeline:") for key in context.results)
        if done:
            identity, _, _ = self.collect(context)
            done = context.project.artifacts.exists(identity)
        return done, (), tuple(lines)

    def run(self, context: SlotContext, settings: JsonObject) -> ArtifactInput:
        """Save every start's results for the slot as one record."""
        del settings
        identity, record, inputs = self.collect(context)
        return save_slot(context, identity, record, inputs)

    def collect(self, context: SlotContext) -> tuple[str, ContractModel, tuple[ArtifactInput, ...]]:
        """Return the record of every start's results and the identity it is saved under."""
        starts: dict[str, tuple[ArtifactInput, ...]] = {}
        tasks = sorted(
            {key.split(":")[1] for key in context.results if key.startswith("pipeline:")}
        )
        for task in tasks:
            found: list[ArtifactInput] = []
            if self.slot == "blast-radius":
                mining = stage_outputs(context, task, "error-mining") or {}
                if mining.get("blast") is not None:
                    found.append(ArtifactInput.model_validate(mining["blast"]))
            else:
                records = stage_outputs(context, task, "records") or {}
                if self.slot == "curriculum":
                    schedules = records.get("schedules")
                    rows = schedules.values() if isinstance(schedules, dict) else ()
                    for row in rows:
                        if isinstance(row, dict) and row.get("difficulty") is not None:
                            found.append(ArtifactInput.model_validate(row["difficulty"]))
                else:
                    training = records.get("training")
                    rows = training.values() if isinstance(training, dict) else ()
                    found.extend(ArtifactInput.model_validate(row) for row in rows)
            starts[task] = tuple(found)
        identity = stable_id(
            SLOT_RECORD_TYPE,
            {"slot": self.slot, "starts": sha256_json(starts), "code": context.code_revision},
        )
        record = PipelineCollection(slot=self.slot, starts=starts)
        inputs = tuple(ref for refs in starts.values() for ref in refs)
        return identity, record, inputs


def dense_slot_settings(
    behavior_bank: str,
    error_bank: str | None,
    bank_root: Path | None,
    author: AuthorSettings,
    decision_alias: str,
    maximum_steps: int,
) -> JsonObject:
    """Describe the in-flow dense vocabulary every exported task binds, for the dense slot.

    Raises:
        QuestionBankError: A vocabulary bank is not published.
    """
    banks = [resolve_question_bank(behavior_bank, root=bank_root)]
    if error_bank is not None:
        banks.append(resolve_question_bank(error_bank, root=bank_root))
    tags, step = vocabulary(banks)
    dense = inflow_dense(tags, step, author, decision_alias)
    chunks = math.ceil(len(step.questions) / MAX_DECISION_QUESTIONS)
    return {
        "tags": len(tags),
        "banks": [bank.bank_id for bank in banks],
        "step_bank": step.bank_id,
        "decision_alias": decision_alias,
        "judge": author.model_alias,
        "decisions_per_episode": maximum_steps * chunks,
        "judge_calls_per_episode": dense_model_calls(dense.profile),
    }
