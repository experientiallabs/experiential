"""Read each front-door checkpoint's monitor inputs from records the pipeline already saved.

Each gatherer takes the front-door slot context and returns the checkpoint's inputs, the saved
records they were read from and the checkpoint's scope, or None while a record it needs is not
saved. Nothing here calls a model.

- corpus (checkpoint 1): the variance build's tag matrices per view, its tagging failures at the
  credential boundary per view, and every family's authored-bank matrices from ``tag-banks``.
- families (checkpoint 2): the admitted families with each member's declared tools, the build's
  matrices, the authored-bank matrices and, when saved, each member's task type.
- rules (checkpoint 3): every family's admitted guard table (support per rule) and every family's
  state-tag index, each step paired with the tool its recorded transition called.
- mining (checkpoint 4): every start's saved point-bar records and blast-radius ranking.
- dense (checkpoint 5): every saved hybrid dense assessment in the project.
"""

from __future__ import annotations

from collections.abc import Callable

from exp.common.core.artifacts import ArtifactInput
from exp.common.project import artifact_input
from exp.optimize.evaluation.rules.memory_admission import load_guard_table
from exp.optimize.evaluation.variance.bootstrap_plan import BootstrapPreparationReport
from exp.optimize.evaluation.variance.build_records import load_tag_matrix
from exp.optimize.evaluation.variance.dense_hybrid import DENSE_HYBRID_TYPE, HybridDense
from exp.optimize.evaluation.variance.family_simulators import load_family_simulator
from exp.optimize.evaluation.variance.frontdoor_executors import (
    FamilyBanks,
    FamilyStateIndexes,
    SlotContext,
    SlotReference,
    pipeline_tasks,
    read_slot,
    saved_corpus,
    stage_outputs,
)
from exp.optimize.evaluation.variance.pipeline_points import load_blast_ranking
from exp.optimize.evaluation.variance.point_selection import load_point_bar
from exp.optimize.evaluation.variance.tag_monitor import CheckpointInputs
from exp.optimize.evaluation.variance.tag_monitor_contracts import CheckpointName
from exp.optimize.evaluation.variance.tag_monitor_reward_stats import (
    BarRule,
    DenseEpisode,
    DenseInputs,
    MinedPoint,
    MiningInputs,
)
from exp.optimize.evaluation.variance.tag_monitor_stats import (
    CorpusInputs,
    FamilyInputs,
    FamilyMembers,
    RuleEvidence,
    RuleInputs,
    StateStep,
)
from exp.simulation.mining.tag_selection import DEFAULT_FIRE_THRESHOLD
from exp.simulation.mining.tagging import TagMatrix
from exp.simulation.mining.task_types import load_task_types
from exp.simulation.retrieval.state_rag import load_state_tag_index
from exp.simulation.retrieval.store import load_rag_index
from exp.simulation.rules.admission import AdmissionVerdict

Gathered = tuple[CheckpointInputs, tuple[ArtifactInput, ...], str]
"""A checkpoint's inputs, the saved records they came from, and its scope."""

Gatherer = Callable[[SlotContext], Gathered | None]
"""Reads one checkpoint's inputs from the front-door context, or None until they are saved."""

CREDENTIAL_BOUNDARY = "credential boundary"
"""The tagger's fixed refusal wording for an input that crosses the credential boundary."""


def _build_matrices(
    context: SlotContext,
) -> tuple[list[TagMatrix], dict[str, int], list[ArtifactInput]] | None:
    """The variance build's matrices, credential-boundary exclusions per view, and sources."""
    corpus = saved_corpus(context)
    if corpus is None:
        return None
    matrices, excluded, sources = [], {}, []
    for view in corpus.report.tagging:
        matrices.append(load_tag_matrix(context.project, view.matrix))
        sources.append(view.matrix)
        boundary = [f for f in view.failures if CREDENTIAL_BOUNDARY in f.error]
        excluded[str(view.plan.view)] = len(boundary)
    return matrices, excluded, sources


def _authored(context: SlotContext) -> tuple[FamilyBanks | None, list[ArtifactInput]]:
    """The ``tag-banks`` result, when that slot ran."""
    reference = context.results.get("tag-banks")
    if reference is None:
        return None, []
    return read_slot(context, reference, FamilyBanks), [reference]


def gather_corpus(context: SlotContext) -> Gathered | None:
    """Checkpoint 1 inputs: every build and authored matrix, exclusions and question texts."""
    built = _build_matrices(context)
    if built is None:
        return None
    matrices, excluded, sources = built
    banks, more = _authored(context)
    texts: dict[str, str] = {}
    if banks is not None:
        for family in sorted(banks.matrices):
            matrices.extend(banks.matrices[family][view] for view in sorted(banks.matrices[family]))
        for authored in banks.banks.values():
            texts |= {q.question_id: q.text for q in authored.bank.questions}
    inputs = CorpusInputs(matrices=tuple(matrices), excluded=excluded, texts=texts)
    corpus = saved_corpus(context)
    assert corpus is not None
    return inputs, (*sources, *more), f"corpus-{corpus.report.identity[-16:]}"


def gather_families(context: SlotContext) -> Gathered | None:
    """Checkpoint 2 inputs: families, member tools, matrices, authored banks, task types."""
    corpus = saved_corpus(context)
    built = _build_matrices(context)
    if corpus is None or built is None:
        return None
    matrices, _, sources = built
    banks, more = _authored(context)
    types: dict[str, dict[str, str]] = {}
    typed = context.results.get("task-types")
    if typed is not None:
        saved = read_slot(context, typed, SlotReference).reference
        more.append(saved)
        for family in load_task_types(context.project.artifacts, saved).by_family:
            types[family.family_id] = {
                member: row.task_type_id for row in family.task_types for member in row.members
            }
    families = tuple(
        FamilyMembers(
            family_id=family.family_id,
            members=family.members,
            tools={
                m: tuple(sorted(t.name for t in corpus.traces[m].tools))
                for m in family.members
                if m in corpus.traces
            },
            task_types=types.get(family.family_id, {}),
        )
        for family in corpus.families
    )
    authored = {}
    if banks is not None:
        authored = {f: tuple(by_view.values()) for f, by_view in banks.matrices.items()}
    inputs = FamilyInputs(families=families, matrices=tuple(matrices), authored=authored)
    return inputs, (*sources, *more), f"families-{corpus.report.identity[-16:]}"


def gather_rules(context: SlotContext) -> Gathered | None:
    """Checkpoint 3 inputs: admitted rules from the guard tables, state steps from the indexes."""
    prepared = context.results.get("preparation")
    corpus = saved_corpus(context)
    if prepared is None or corpus is None:
        return None
    project = context.project
    report = BootstrapPreparationReport.model_validate_json(
        project.artifacts.read_bytes(prepared.artifact_id, "report.json")
    )
    rules: list[RuleEvidence] = []
    sources: list[ArtifactInput] = [prepared]
    for reference in report.simulators:
        memory = load_family_simulator(project.artifacts, reference).rule_memory
        if memory is None or memory.guards is None:
            continue
        sources.append(memory.guards)
        for entry in load_guard_table(project, memory.guards).entries():
            rules.append(
                RuleEvidence(
                    verdict=AdmissionVerdict(
                        rule_id=entry.rule_id,
                        admitted=True,
                        status="supported",
                        supporting=entry.supporting,
                        supporting_lineages=entry.supporting_lineages,
                        reason="admitted into the family guard table",
                    )
                )
            )
    steps: list[StateStep] = []
    questions: tuple[str, ...] = ()
    bank = "state"
    recall = context.results.get("state-rag")
    if recall is not None:
        indexes = read_slot(context, recall, FamilyStateIndexes)
        scoping = corpus.report.environment_families
        for family_id, binding in sorted(indexes.bindings.items()):
            index = load_state_tag_index(project.artifacts, binding.index)
            sources.append(binding.index)
            if not questions:
                questions = tuple(q.question_id for q in index.bank.questions)
                bank = f"state.{family_id[:24]}"
            elif questions != tuple(q.question_id for q in index.bank.questions):
                continue
            tools: dict[str, str] = {}
            if scoping is not None:
                rag = load_rag_index(project.artifacts, scoping.scope(family_id).index.artifact_id)
                tools = {
                    t.transition_id: t.action.tool_name or t.action.kind for t in rag.transitions
                }
            steps.extend(
                StateStep(
                    step_id=row.transition_id,
                    lineage_id=row.lineage_id,
                    probabilities=row.probabilities,
                    next_action=tools.get(row.transition_id, "unknown"),
                )
                for row in index.rows
            )
    inputs = RuleInputs(
        rules=tuple(rules), state_bank=bank, state_questions=questions, state_steps=tuple(steps)
    )
    return inputs, tuple(sources), f"rules-{prepared.artifact_id[-16:]}"


def _pipeline_tasks(context: SlotContext) -> list[str]:
    """Every start whose pipeline stages are in the context."""
    return pipeline_tasks(context.results)


def gather_mining(context: SlotContext) -> Gathered | None:
    """Checkpoint 4 inputs: every tier's bar records and the blast-radius ranking."""
    store = context.project.artifacts
    rules: dict[str, bool] = {}
    steps: list[dict[str, float]] = []
    points: list[MinedPoint] = []
    sources: list[ArtifactInput] = []
    for task in _pipeline_tasks(context):
        mining = stage_outputs(context, task, "error-mining") or {}
        ranks: dict[tuple[int | None, int], tuple[int, float | None]] = {}
        blast = mining.get("blast")
        if isinstance(blast, dict):
            reference = ArtifactInput.model_validate(blast)
            sources.append(reference)
            for row in load_blast_ranking(store, reference):
                ranks[(row.attempt, row.point)] = (row.rank, row.score)
        bars = mining.get("point_bars")
        for tier, value in sorted(bars.items()) if isinstance(bars, dict) else ():
            reference = ArtifactInput.model_validate(value)
            sources.append(reference)
            for attempt, record in sorted(load_point_bar(store, reference).items()):
                for point in record.points:
                    held: set[str] = set()
                    for step in point.steps:
                        steps.append(dict(step.values))
                        for rule, probability in step.values.items():
                            rules.setdefault(rule, True)
                            if probability > DEFAULT_FIRE_THRESHOLD:
                                held.add(rule)
                    rank, score = ranks.get((attempt, point.point), (0, None))
                    points.append(
                        MinedPoint(
                            point_id=f"{task}/{tier}/attempt:{attempt}/point:{point.point}",
                            qualified=point.state in ("qualified", "continued", "deferred"),
                            tags=tuple(sorted(held)),
                            rank=rank or None,
                            score=score,
                        )
                    )
    if not rules:
        return None
    inputs = MiningInputs(
        bar=tuple(BarRule(rule_id=r, required=req) for r, req in sorted(rules.items())),
        steps=tuple(steps),
        points=tuple(points),
    )
    return inputs, tuple(sources), f"mining-{len(points)}-{len(sources)}"


def gather_dense(context: SlotContext) -> Gathered | None:
    """Checkpoint 5 inputs: every saved hybrid dense assessment, one episode each."""
    store = context.project.artifacts
    episodes: list[DenseEpisode] = []
    sources: list[ArtifactInput] = []
    for identity in sorted(store.list_ids(artifact_type=DENSE_HYBRID_TYPE)):
        stored = store.read(identity)
        hybrid = HybridDense.model_validate_json(stored.payloads["hybrid.json"])
        sources.append(artifact_input(stored.manifest))
        strongest: dict[str, str] = {}
        order = {"clear": 0, "uncertain": 1, "fired": 2}
        for row in hybrid.routing.step_tags:
            if order[row.status] >= order.get(strongest.get(row.tag_id, "clear"), 0):
                strongest[row.tag_id] = row.status
        decisions = {
            f.tag_id: f.decision
            for f in hybrid.export.findings
            if f.decision in ("established", "acceptable", "absent", "uncertain")
        }
        routed = tuple(r.tag_id for r in hybrid.routing.routes if r.route == "judge")
        episodes.append(
            DenseEpisode.model_validate(
                {
                    "episode_id": hybrid.export.episode_id,
                    "tier": "all",
                    "clef": strongest,
                    "routed": routed,
                    "decisions": {t: d for t, d in decisions.items() if t in routed},
                }
            )
        )
    if not episodes:
        return None
    behavior = tuple(sorted({t for e in episodes for t in e.clef}))
    inputs = DenseInputs(episodes=tuple(episodes), behavior_tags=behavior)
    return inputs, tuple(sources), f"dense-{len(episodes)}"


GATHERERS: dict[CheckpointName, Gatherer] = {
    "corpus": gather_corpus,
    "families": gather_families,
    "rules": gather_rules,
    "mining": gather_mining,
    "dense": gather_dense,
}
"""The production gatherer of every front-door checkpoint."""
