"""The front door's owner-bound slot executors, run against the preparation's saved families.

Each executor prices its work credential-free and runs, or replays, it through the owner's own
entry points; every result is a saved record the next phase reads.

- ``tag-banks`` (:class:`TagBankExecutor`): per admitted environment family, the corpus tag
  author drafts yes/no questions from a sample of the members' task and observation views and
  the selection keeps the discriminating, answerable, non-redundant ones (``corpus_tag_bank``);
  every member is then tagged with ``authored_bank_for_view`` per view by the decision model.
  A family with at least ``loop.min_members_to_retry`` members whose tags cluster into a single
  task type is re-authored with feedback, up to ``loop.max_rounds`` rounds
  (``author_family_bank``); the plan prices every such round as its worst case.
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
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, NamedTuple

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
from exp.optimize.evaluation.variance.build_records import (
    FamilyScope,
    VarianceBuildReport,
    load_tag_matrix,
)
from exp.optimize.evaluation.variance.dense_prompt import dense_model_calls
from exp.optimize.evaluation.variance.dense_vocabulary import inflow_dense, vocabulary
from exp.optimize.evaluation.variance.family_simulators import load_family_simulator
from exp.optimize.evaluation.variance.pipeline_review import family_reviewer, review_costs
from exp.optimize.evaluation.variance.pipeline_spec import CostLine, cost_lines
from exp.optimize.evaluation.variance.refined_prompts import SLOT_ROLES, refined_identity
from exp.runtime.gateway.decisions_contracts import MAX_DECISION_QUESTIONS
from exp.runtime.gateway.reservation_tokenizer import count_reservation_tokens
from exp.runtime.models.providers.admitted_transport import AdmittedJsonTransport
from exp.runtime.models.providers.codex_cli import offline_closed_model
from exp.runtime.models.providers.errors import ProviderRetryableResponseError
from exp.runtime.models.providers.transport import (
    HttpxJsonTransport,
    JsonHttpResponse,
    JsonHttpTransport,
    ProviderTransportError,
)
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.runtime.sessions.dispatch_ledger import DispatchLedger
from exp.simulation.mining.authoring_loop import AuthoringLoopSpec, author_family_bank
from exp.simulation.mining.decisions import DEFAULT_DECISION_ALIAS, DecisionsTraceTagger
from exp.simulation.mining.dispatch import DEFAULT_TAGGING_CONCURRENCY
from exp.simulation.mining.environment_families import (
    EnvironmentFamily,
    admitted_families,
    load_environment_families,
)
from exp.simulation.mining.environment_review import ReviewerError
from exp.simulation.mining.questions import Question, resolve_question_bank
from exp.simulation.mining.scrub import scrub_units
from exp.simulation.mining.tag_authoring import (
    TagAuthor,
    TagAuthoringSpec,
)
from exp.simulation.mining.tag_bank_selection import (
    AuthoredBank,
    BankSelectionSpec,
)
from exp.simulation.mining.tag_cluster_names import ClusterNamer
from exp.simulation.mining.tagging import DecisionCache, TagMatrix, TraceTagger
from exp.simulation.mining.task_types import TaskTypeSpec, task_types
from exp.simulation.mining.views import TraceViewName, Unit, build_units
from exp.simulation.retrieval.contracts import RAGLineageBinding
from exp.simulation.retrieval.state_rag import (
    StateRAGCalls,
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


def _views(corpus: Corpus, trace_id: str, views: tuple[TraceViewName, ...]) -> dict[str, Unit]:
    """Render a corpus trace's views, scrubbed exactly as its build scrubbed them."""
    rendered: dict[str, Unit] = dict(build_units(corpus.traces[trace_id], views))
    return scrub_units(rendered) if corpus.report.spec.credential_scrub else rendered


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


FamilyCoverage = Literal["all", "run_tasks"]
"""Which admitted families a per-family slot works on: every one, or only the run's own.

``run_tasks`` keeps the families that hold the preparation's selected ``task_ids``, read from the
variance build's probe placements, so a cluster run authors banks and indexes for its own family
only and a spec never names a family by hand.
"""

SCOPED_SLOTS = frozenset({"tag-banks", "task-types", "state-rag"})
"""Per-family slots that take a ``scope`` setting (:data:`FamilyCoverage`)."""


def run_families(context: SlotContext, corpus: Corpus) -> tuple[str, ...]:
    """Return the families holding the preparation's selected tasks, sorted.

    Raises:
        ValueError: The preparation names no ``task_ids``, or one of them is not placed in a
            family of the saved build.
    """
    task_ids = context.preparation.task_ids
    if not task_ids:
        raise ValueError(
            "scope run_tasks needs preparation.task_ids: the run's families are those holding "
            "its selected tasks"
        )
    scoping = corpus.report.environment_families
    assert scoping is not None
    placed = {row.task_id: row.family_id for row in scoping.placements if row.family_id}
    unplaced = sorted(task for task in task_ids if task not in placed)
    if unplaced:
        raise ValueError(f"scope run_tasks: tasks not placed in any family: {', '.join(unplaced)}")
    return tuple(sorted({placed[task] for task in task_ids}))


def slot_families(
    context: SlotContext, corpus: Corpus, scope: FamilyCoverage
) -> frozenset[str] | None:
    """Return the family identities a slot is scoped to, or None for every family."""
    return None if scope == "all" else frozenset(run_families(context, corpus))


def scoped_families(
    corpus: Corpus, families: frozenset[str] | None
) -> tuple[EnvironmentFamily, ...]:
    """Return the admitted families within a scope, in review order."""
    if families is None:
        return corpus.families
    return tuple(row for row in corpus.families if row.family_id in families)


def scope_note(corpus: Corpus, families: frozenset[str] | None) -> str:
    """Return the plan text naming a narrowed scope, empty for every family."""
    if families is None:
        return ""
    scoping = corpus.report.environment_families
    assert scoping is not None
    paths = {row.family_id: row.path for row in scoping.scopes}
    named = ", ".join(sorted(paths.get(f, f[:24]) for f in families))
    return f" (scope run_tasks: only {named}, of {len(corpus.families)} admitted)"


def _identity_settings(
    name: str, settings: JsonObject, context: SlotContext, corpus: Corpus
) -> tuple[JsonObject, JsonObject | None]:
    """Return the settings a slot's identity hashes, and its resolved scope.

    An explicit ``scope: "all"`` hashes like an absent one, so it keeps every saved result. A
    narrowed scope joins the identity with the families it resolved to, so an all-family result
    and a scoped one, or two runs scoped to different families, never share a result.
    """
    if name not in SCOPED_SLOTS:
        return settings, None
    scope = settings.get("scope", "all")
    keyed = {key: value for key, value in settings.items() if key != "scope"}
    if scope == "all":
        return keyed, None
    return keyed, {"mode": scope, "families": list(run_families(context, corpus))}


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
    refined = refined_identity(context.project, SLOT_ROLES.get(name, frozenset()))
    keyed, scope = _identity_settings(name, settings, context, corpus)
    return stable_id(
        SLOT_RECORD_TYPE,
        {
            **refined,
            "slot": name,
            "settings": sha256_json(keyed),
            **({} if scope is None else {"scope": scope}),
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


MAXIMUM_DECISION_STATE_TOKENS = 100_000
"""Largest rendered state, in o200k reservation tokens, one front-door decision call carries.

Hosted Clef refuses an over-long input with a bare HTTP 400 ("rejected the request without giving
a reason"): a 130,740-token observations view (about 151,000 Clef input tokens) is refused while a
100,628-input-token state is answered. The observations view's own budget cannot bound a view
whose fixed head is the problem, such as a request that inlines a whole repository, so the state
is fitted here, at the call, instead of in the view (which would re-key the saved variance build).
"""

_ELISION = "\n\n[... {count} characters elided to fit the decision model's input ...]\n\n"


def fit_decision_unit(unit: Unit, maximum_tokens: int = MAXIMUM_DECISION_STATE_TOKENS) -> Unit:
    """Return the unit unchanged when it fits, otherwise its head and tail around an elision.

    The start (what was asked) and the end (the final answer) carry most of what a question
    reads, so the middle gives way. The cut is a pure function of the text and the ceiling, and
    the decision cache keys on the fitted text, so a rerun replays the same answer, and a view
    that already fits keeps every answer it was saved with.
    """
    text = unit.text
    if count_reservation_tokens(text) <= maximum_tokens:
        return unit

    def cut(keep: int) -> str:
        head = (keep * 3) // 4
        marker = _ELISION.format(count=len(text) - keep)
        return text[:head] + marker + text[len(text) - (keep - head) :]

    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if count_reservation_tokens(cut(middle)) <= maximum_tokens:
            low = middle
        else:
            high = middle - 1
    return Unit(unit.trace_id, unit.view, cut(low), dict(unit.meta))


class FittedDecisionsTagger(DecisionsTraceTagger):
    """The native decision tagger, sending each unit fitted to the decision model's input.

    Rows still carry the rendered unit's own identity; only the request and its cache key use the
    fitted text.
    """

    maximum_state_tokens = MAXIMUM_DECISION_STATE_TOKENS

    def _unit_answers(self, unit: Unit, chunks: Sequence[Sequence[Question]]) -> Any:  # noqa: ANN401
        """Answer the bank about the unit as fitted to the input ceiling."""
        return super()._unit_answers(fit_decision_unit(unit, self.maximum_state_tokens), chunks)


def _refusal_reason(body: object) -> str:
    """Return the provider's stated error type, code and message, bounded, never the request."""
    error = body.get("error") if isinstance(body, dict) else None
    if not isinstance(error, dict):
        return "no error object in the response"
    parts = [
        f"{key}={str(error[key])[:300]}"
        for key in ("type", "code", "message")
        if isinstance(error.get(key), str)
    ]
    return "; ".join(parts) or "an error object with no type, code or message"


@dataclass(frozen=True)
class ReasonedDecisionTransport(JsonHttpTransport):
    """Raise a refused decision call with the provider's stated reason, not just its status.

    The decision tagger raises the same error, with the same status, for any response that is not
    an HTTP 200 object, so retry classification is unchanged; only the message gains the reason.
    """

    inner: JsonHttpTransport

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Post through the inner transport and name a refusal's reason."""
        response = self.inner.post(
            url, headers=headers, payload=payload, timeout_seconds=timeout_seconds
        )
        if response.status_code != 200 or not isinstance(response.body, dict):
            raise ProviderTransportError(
                f"tagging decision returned HTTP {response.status_code}: "
                f"{_refusal_reason(response.body)}",
                status_code=response.status_code,
            )
        return response


def decision_tagger(
    project: ProjectStore, alias: str, purpose: str, attempts: int, code_revision: str
) -> DecisionsTraceTagger:
    """Return the native decision tagger of an alias over the project decision cache."""
    resolved = RuntimeModelCatalog(load_model_catalog(project.model_catalog_path)).resolve(alias)
    transport = admitted(project, purpose, attempts, code_revision).priced_as(alias)
    return FittedDecisionsTagger.for_alias(
        resolved,
        transport=ReasonedDecisionTransport(transport),
        cache=DecisionCache(project.paths),
        maximum_concurrency=tagging_concurrency(),
    )


TAGGING_CONCURRENCY_ENV = "EXP_TAGGING_CONCURRENCY"
"""Environment override for decision calls in flight; it changes only how long a pass takes."""


def tagging_concurrency() -> int:
    """Return the decision calls in flight, from the environment or the shared default."""
    raw = os.environ.get(TAGGING_CONCURRENCY_ENV, "").strip()
    if not raw:
        return DEFAULT_TAGGING_CONCURRENCY
    value = int(raw)
    if value < 1:
        raise ValueError(f"{TAGGING_CONCURRENCY_ENV} must be at least 1, not {value}")
    return value


UNKNOWN_AUTHOR_OUTCOME = "author dispatch has an unknown outcome"
"""Prefix of the refusal for a draw whose earlier dispatch has no recorded outcome."""


def author_draw(
    project: ProjectStore, settings: AuthorSettings, code_revision: str
) -> Callable[..., ModelResponse]:
    """Return a buy-or-replay draw through the admitted author-response owner.

    A draw that spends its whole output allowance without a usable answer (a reasoning model
    that never reaches its tool call) fails only that draw, as :class:`ReviewerError`, the
    way the compatibility reviewer's draws do. Its rejection stays saved and replays as the
    same failure; the authoring loop records it and goes on with its other draws instead of
    aborting the whole slot.
    """

    def draw(request: ModelRequest, *, identity: str) -> ModelResponse:
        """Buy or replay one exact draw."""
        try:
            _, response = author_response(
                project,
                settings,
                request,
                identity=identity,
                sources=(),
                code_revision=code_revision,
            )
        except ProviderRetryableResponseError as error:
            if not error.hit_length_limit:
                raise
            raise ReviewerError(
                f"the author exhausted its {settings.output_tokens}-token output allowance"
            ) from error
        except ValueError as error:
            # A draw interrupted mid-dispatch (its run was stopped) has an unknown outcome and is
            # never drawn again; fail only that draw so the slot goes on with its other draws.
            if not str(error).startswith(UNKNOWN_AUTHOR_OUTCOME):
                raise
            raise ReviewerError(
                "the author draw has an unknown outcome from an earlier run"
            ) from error
        return response

    return draw


MIN_AUTHORED_MEMBERS = 2
"""Fewest members a family needs before questions are authored for it."""


class FamilyBanks(ContractModel):
    """The ``tag-banks`` slot's result.

    Attributes:
        banks: Each family's authored, selected and published bank; a family with fewer than
            :data:`MIN_AUTHORED_MEMBERS` members has none.
        matrices: Each family's member matrices per view, as saved tag matrices.
        scope: The families the slot was scoped to, sorted; None when it covered every one.
    """

    banks: dict[str, AuthoredBank]
    matrices: dict[str, dict[str, TagMatrix]] = Field(default_factory=dict)
    scope: tuple[str, ...] | None = None


class TagBankSlot(ContractModel):
    """Settings of the ``tag-banks`` slot.

    Attributes:
        author: Admitted open-weight author settings.
        authoring: Variants, rounds and sample size per family.
        selection: Discrimination, answerability and redundancy bounds.
        bank_root: Directory the banks are published under; None for the local bank root.
        decision_alias: Native decision alias tagging the selection sample and the members.
        loop: When a large family that collapses to one task type is re-authored with feedback.
        scope: Every admitted family, or only the families of the run's selected tasks.
    """

    author: AuthorSettings
    authoring: TagAuthoringSpec = TagAuthoringSpec()
    selection: BankSelectionSpec = BankSelectionSpec()
    loop: AuthoringLoopSpec = AuthoringLoopSpec()
    bank_root: Path | None = None
    decision_alias: str = Field(default=DEFAULT_DECISION_ALIAS, min_length=1)
    scope: FamilyCoverage = "all"


def _unsaved_families(context: SlotContext, scope: FamilyCoverage) -> int:
    """Families a slot prices before the families are saved: the limit, or the run's tasks."""
    bound = context.preparation.limits.maximum_families
    if scope == "run_tasks" and context.preparation.task_ids:
        bound = min(bound, len(context.preparation.task_ids))
    return bound


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
        bounds = tag_bank_bounds(context, slot, corpus)
        author, decisions = bounds.author, bounds.decisions
        loop_author, loop_decisions = bounds.loop_author, bounds.loop_decisions
        costs: list[CostComponent] = [
            row.scaled(author + loop_author)
            for row in evaluation_costs(context.project, slot.author)
        ]
        costs.append(
            decision_cost(context.project, slot.decision_alias, decisions + loop_decisions)
        )
        lines = (
            f"{bounds.families} families{bounds.note}: up to {author} author calls with "
            f"{slot.author.model_alias}, then at most {decisions} decisions with "
            f"{slot.decision_alias} (selection sample and every member, per view).",
            f"Re-authoring loop: {bounds.looping} families of at least "
            f"{slot.loop.min_members_to_retry} members may collapse to one task type; at worst "
            f"{slot.loop.max_rounds} rounds each add {loop_author} author calls and "
            f"{loop_decisions} decisions.",
        )
        return False, cost_lines(tuple(costs)), lines

    def run(self, context: SlotContext, settings: JsonObject) -> ArtifactInput:
        """Run, or replay, every family's bank and member tagging.

        Build-time tag authoring is the one offline role the closed Astra alias may fill, so
        the slot runs inside :func:`offline_closed_model`; the front door's registration check
        already refuses that alias on every other role.
        """
        with offline_closed_model("build-time tag authoring: tag-banks"):
            return self._run(context, settings)

    def _run(self, context: SlotContext, settings: JsonObject) -> ArtifactInput:
        """Run, or replay, every family's bank and member tagging."""
        slot = TagBankSlot.model_validate(settings)
        corpus = saved_corpus(context)
        if corpus is None:
            raise ValueError("the tag-banks slot runs after the families review is saved")
        identity = slot_identity("tag-banks", settings, context, corpus)
        if context.project.artifacts.exists(identity):
            return artifact_input(context.project.artifacts.read(identity).manifest)
        project = context.project
        scope = slot_families(context, corpus, slot.scope)
        families = scoped_families(corpus, scope)
        members_total = sum(len(row.members) for row in families)
        tagger = self.tagger or decision_tagger(
            project,
            slot.decision_alias,
            "tag-banks",
            members_total * 64 * (1 + slot.loop.max_rounds),
            context.code_revision,
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
        task_view = next(row for row in corpus.report.tagging if row.plan.view == "task")
        shared = (load_tag_matrix(project, task_view.matrix),)
        for family in families:
            rendered = [_views(corpus, t, AUTHORED_VIEWS) for t in family.members]
            if len(rendered) < MIN_AUTHORED_MEMBERS:
                # A question must hold in some members and fail in others, so a family this
                # small cannot be authored; it keeps its fixed-bank tags and gets one task type.
                continue
            result = author_family_bank(
                project.artifacts,
                author,
                tagger,
                scope=family.family_id,
                bank_id=f"authored-{family.family_id[:24]}",
                rendered=rendered,
                views=AUTHORED_VIEWS,
                shared=shared,
                tagger_identity=getattr(tagger, "model_identity", {}),
                authoring=slot.authoring,
                selection=slot.selection,
                loop=slot.loop,
                root=slot.bank_root,
                code_revision=context.code_revision,
            )
            banks[family.family_id] = result.bank
            inputs.extend((result.bank.authoring, result.bank.selection, *result.bank.rounds))
            matrices[family.family_id] = result.matrices
        record = FamilyBanks(
            banks=banks, matrices=matrices, scope=None if scope is None else tuple(sorted(scope))
        )
        return save_slot(context, identity, record, tuple(inputs))


class TagBankBounds(NamedTuple):
    """The tag-banks slot's call bound over its scoped families.

    Attributes:
        families: Families priced.
        author: Author calls.
        decisions: Decisions over the selection sample and every member, per view.
        looping: Families that may run the re-authoring loop.
        loop_author: The loop's extra author calls over all rounds.
        loop_decisions: The loop's extra decisions over all rounds.
        note: Plan text naming a narrowed scope, empty for every family.
    """

    families: int
    author: int
    decisions: int
    looping: int
    loop_author: int
    loop_decisions: int
    note: str


def tag_bank_bounds(
    context: SlotContext, slot: TagBankSlot, corpus: Corpus | None
) -> TagBankBounds:
    """Bound the tag-banks slot's calls over its scoped families, with no call."""
    if corpus is None:
        families = _unsaved_families(context, slot.scope)
        members = families * context.preparation.limits.maximum_family_traces
        note = ""
    else:
        scope = slot_families(context, corpus, slot.scope)
        rows = scoped_families(corpus, scope)
        families = len(rows)
        members = sum(len(row.members) for row in rows)
        note = scope_note(corpus, scope)
    author = slot.authoring.calls() * families
    chunks = math.ceil(slot.selection.max_questions / MAX_DECISION_QUESTIONS)
    decisions = len(AUTHORED_VIEWS) * members * chunks * 2
    looping, loop_author, loop_decisions = loop_worst_case(context, slot, corpus)
    return TagBankBounds(families, author, decisions, looping, loop_author, loop_decisions, note)


def loop_worst_case(
    context: SlotContext, slot: TagBankSlot, corpus: Corpus | None
) -> tuple[int, int, int]:
    """Price the re-authoring loop at its worst: every large family runs every round.

    A round asks the author for a full authoring pass and the decision model for every
    candidate over every member and view, then for the kept ones again over the members.

    Returns:
        Families that may loop, and the extra author calls and decisions over all rounds.
    """
    loop = slot.loop
    if corpus is None:
        size = context.preparation.limits.maximum_family_traces
        sizes = [size] * _unsaved_families(context, slot.scope)
    else:
        scope = slot_families(context, corpus, slot.scope)
        sizes = [len(row.members) for row in scoped_families(corpus, scope)]
    looping = [size for size in sizes if size >= MIN_AUTHORED_MEMBERS and loop.loops(size)]
    spec = slot.authoring
    candidates = len(spec.variants) * spec.rounds * spec.questions_per_draw
    chunks = math.ceil(candidates / MAX_DECISION_QUESTIONS)
    author = loop.max_rounds * spec.calls() * len(looping)
    decisions = loop.max_rounds * len(AUTHORED_VIEWS) * sum(looping) * chunks * 2
    return len(looping), author, decisions


class TaskTypeSlot(ContractModel):
    """Settings of the ``task-types`` slot.

    Attributes:
        namer_alias: Admitted open-weight naming alias.
        spec: Second-level clustering and naming.
        scope: Every admitted family, or only the families of the run's selected tasks; no
            wider than the ``tag-banks`` slot's scope.
    """

    namer_alias: str = Field(min_length=1)
    spec: TaskTypeSpec = TaskTypeSpec()
    scope: FamilyCoverage = "all"


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
        families, note = _unsaved_families(context, slot.scope), ""
        if corpus is not None:
            scope = slot_families(context, corpus, slot.scope)
            families, note = len(scoped_families(corpus, scope)), scope_note(corpus, scope)
        calls = families * slot.spec.naming.calls(slot.spec.clustering.max_clusters)
        costs = (model_cost(context.project, slot.namer_alias, calls, 1200),)
        line = f"Up to {calls} naming calls with {slot.namer_alias}{note}."
        return False, cost_lines(costs), (line,)

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
        scope = slot_families(context, corpus, slot.scope)
        if banks.scope is not None and (scope is None or not scope <= set(banks.scope)):
            raise ValueError(
                "the task-types scope is wider than the tag-banks scope: families outside "
                "the banks' scope have no authored bank; give both slots the same scope"
            )
        families = scoped_families(corpus, scope)
        task_view = next(row for row in corpus.report.tagging if row.plan.view == "task")
        matrices: list[TagMatrix] = [load_tag_matrix(project, task_view.matrix)]
        questions: dict[str, str] = {}
        for family_id, per_view in banks.matrices.items():
            if scope is not None and family_id not in scope:
                continue
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
            trace_id: _views(corpus, trace_id, ("task",))["task"].text
            for trace_id in corpus.traces
            if any(trace_id in row.members for row in families)
        }
        scoping = corpus.report.environment_families
        assert scoping is not None
        record, saved = task_types(
            project.artifacts,
            families,
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
        concurrency: Fit steps tagged at once; the saved index does not depend on it.
        scope: Every family scope of the build, or only the families of the run's selected
            tasks; a start of a family with no index runs without state recall.
    """

    author_alias: str = Field(min_length=1)
    authoring: StateTagAuthoringSpec = StateTagAuthoringSpec()
    policy: StateRetrievalPolicy = StateRetrievalPolicy()
    decision_alias: str = Field(default=DEFAULT_DECISION_ALIAS, min_length=1)
    samples: int = Field(default=24, ge=2, le=512)
    live_steps: int = Field(default=0, ge=0)
    memory: StateMemoryPolicy | None = None
    concurrency: int = Field(default=1, ge=1, le=64)
    scope: FamilyCoverage = "all"


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


def state_rag_scopes(
    context: SlotContext, slot: StateRagSlot, corpus: Corpus
) -> tuple[FamilyScope, ...]:
    """Return the build's family scopes the slot indexes, in build order."""
    scoping = corpus.report.environment_families
    assert scoping is not None
    families = slot_families(context, corpus, slot.scope)
    return tuple(row for row in scoping.scopes if families is None or row.family_id in families)


def state_rag_bounds(
    context: SlotContext, slot: StateRagSlot, corpus: Corpus | None
) -> tuple[int, StateRAGCalls, str]:
    """Bound the state-rag slot's calls over its scoped families, with no call.

    Returns:
        The families priced, the call bound, and the plan text naming a narrowed scope.
    """
    families = _unsaved_families(context, slot.scope)
    steps, note = 0, ""
    if corpus is not None:
        scopes = state_rag_scopes(context, slot, corpus)
        families = len(scopes)
        store = context.project.artifacts
        for scope in scopes:
            steps += len(load_rag_index(store, scope.index.artifact_id).transitions)
        note = scope_note(corpus, slot_families(context, corpus, slot.scope))
        if slot.scope == "all":
            families = len(corpus.families)
    steps = steps or families * context.preparation.limits.maximum_family_traces * 50
    calls = plan_state_rag_calls(
        authoring_calls=slot.authoring.calls * families,
        questions=slot.authoring.maximum_questions,
        corpus_steps=steps,
        live_steps=slot.live_steps,
    )
    return families, calls, note


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
        families, calls, note = state_rag_bounds(context, slot, corpus)
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
            f"{families} families{note}: {calls.authoring_completions} authoring completions with "
            f"{slot.author_alias}, {calls.corpus_decisions} corpus and {calls.live_decisions} "
            f"live decisions with {slot.decision_alias}."
        )
        return False, cost_lines(costs), (line,)

    def run(self, context: SlotContext, settings: JsonObject) -> ArtifactInput:
        """Run, or replay, every family's state-tag bank and index.

        Build-time tag authoring is the one offline role the closed Astra alias may fill, so
        the slot runs inside :func:`offline_closed_model`; the front door's registration check
        already refuses that alias on every other role.
        """
        with offline_closed_model("build-time tag authoring: state-rag"):
            return self._run(context, settings)

    def _run(self, context: SlotContext, settings: JsonObject) -> ArtifactInput:
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
        for scope in state_rag_scopes(context, slot, corpus):
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
                concurrency=slot.concurrency,
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


def pipeline_tasks(results: Mapping[str, object]) -> list[str]:
    """Return the start tasks of ``pipeline:<task>:<stage>`` keys, sorted.

    A task id may itself hold a colon (``arvo:3938``), so the stage is split off the right.
    """
    prefix = "pipeline:"
    return sorted(
        {key[len(prefix) :].rsplit(":", 1)[0] for key in results if key.startswith(prefix)}
    )


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
        tasks = pipeline_tasks(context.results)
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
