"""What the decision-variance stages read beyond their own owners.

- World-learning traces from the variance build's pinned dataset, named as authoring sources
  and replayed as observed episodes for the world author's program validation.
- Tag-aware retrieval over the build's scoped, fit-only world-learning index for error-mining
  selectors and critics, with the build's catalog embedder role bound for a semantic index.
- The mining-probe task's rubric, authored by the agentic rubric author over a whole-task
  authoring packet when the spec does not register one.

Held-out traces never enter: every trace read here is checked against the frozen exposure plan,
and the retrieval index is refused unless it is fit-only and scoped to world learning.
"""

from __future__ import annotations

import re
from datetime import datetime

from exp.common.core.artifacts import ArtifactInput, sha256_json, stable_id
from exp.common.models import load_model_catalog
from exp.common.project import ProjectStore, artifact_input
from exp.common.traces import Trace
from exp.common.traces.store import load_trace_dataset
from exp.optimize.evaluation.executable.costs import evaluation_costs
from exp.optimize.evaluation.executable.grounded import GroundedRuntimeDefinition
from exp.optimize.evaluation.executable.model_costs import CostComponent, decision_cost
from exp.optimize.evaluation.fidelity.retrieval_binding import rag_index_embedder_alias
from exp.optimize.evaluation.induction.episodes import trace_episode
from exp.optimize.evaluation.rules.memory_stage import memory_stage_costs
from exp.optimize.evaluation.rules.rule_tags import rule_tag_call_bounds
from exp.optimize.evaluation.rules.stage import RulesStageSpec, load_rulebook
from exp.optimize.evaluation.variance.authoring_packet import (
    SourceRef,
    build_authoring_packet,
    save_packet,
)
from exp.optimize.evaluation.variance.build_records import (
    load_exposure_plan,
    load_tag_retrieval_binding,
)
from exp.optimize.evaluation.variance.evidence import TraceSearchSpec
from exp.optimize.evaluation.variance.exposure import ExposurePlan
from exp.optimize.evaluation.variance.pipeline_spec import (
    AuthoringSettings,
    BuildOutputs,
    DecisionVarianceSpec,
    RulesOutputs,
    StageContext,
)
from exp.optimize.evaluation.variance.rubric_agent import (
    RubricAuthoringProfile,
    author_rubric_agentic,
)
from exp.optimize.evaluation.variance.rubric_author import save_rubric
from exp.runtime.agents.chat import ChatAgentRuntime
from exp.runtime.models.providers.admitted_transport import AdmittedJsonTransport
from exp.runtime.models.providers.transport import HttpxJsonTransport, RetryPolicy
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.runtime.sessions.dispatch_ledger import DispatchLedger
from exp.simulation.continuations import load_continuation_checkpoint
from exp.simulation.induction.contracts import InductionEpisode
from exp.simulation.retrieval import load_rag_index
from exp.simulation.retrieval.embedding import RAGEmbedderBinding, default_rag_embedder


def rubric_draws(profile: RubricAuthoringProfile) -> int:
    """Return the most draws one agentic rubric authoring makes, design critic reviews included."""
    return profile.draws()


def library_draws(authoring: AuthoringSettings) -> int:
    """Return the most labeling and curation draws one family's library update makes.

    Each labeler variant labels and may repair once; the curator plans, proposes operations and
    may repair once.
    """
    if authoring.library_id is None:
        return 0
    return 2 * len(authoring.labeler_variants) + 3


def revision_draws(authoring: AuthoringSettings) -> int:
    """Return the most reviser and design-critic draws of one family's rubric revision."""
    critic = authoring.rubric_profile.critic
    per_round = 1 + (0 if critic is None else critic.draws())
    return authoring.revision_rounds * per_round


def local_id(value: str) -> str:
    """Turn a label into a locator-safe identity."""
    cleaned = re.sub(r"[^A-Za-z0-9_.:-]", "-", value)
    return (cleaned if cleaned[:1].isalnum() else "x" + cleaned)[:128]


def world_sources(build: BuildOutputs) -> tuple[SourceRef, ...]:
    """Name every world-learning trace as an authoring source."""
    return tuple(
        SourceRef(
            source_id=local_id(f"trace-{trace_id}"),
            kind="trace",
            trace_id=trace_id,
            summary="World-learning trajectory the rule book was learned from.",
        )
        for trace_id in build.rules_traces
    )


def observed_episodes(
    project: ProjectStore,
    build: BuildOutputs,
    exposure: ExposurePlan,
    tools: tuple[str, ...],
    maximum: int,
) -> tuple[InductionEpisode, ...]:
    """Return world-learning traces calling any of ``tools`` as replay episodes, in order.

    A trace's starting state is not recorded, so each episode starts unknown: steps that read
    an unknown domain stay unverifiable, and only recorded states are compared exactly.

    Args:
        project: Project holding the trace dataset.
        build: The variance build naming the world-learning traces.
        exposure: The frozen data roles; only world-learning traces are read.
        tools: The task's tools.
        maximum: Most episodes returned.

    Returns:
        The episodes.
    """
    traces = world_learning_traces(project, build)
    exposure.require((trace.trace_id for trace in traces), "world_learning")
    episodes: list[InductionEpisode] = []
    for trace in traces:
        if len(episodes) >= maximum:
            break
        lineage = exposure.traces[trace.trace_id]
        episode = trace_episode(trace, lineage, tools=tools, initial_state=None)
        if episode is not None:
            episodes.append(episode)
    return tuple(episodes)


def rules_costs(
    project: ProjectStore, spec: RulesStageSpec, traces: int, tools: int = 0
) -> tuple[tuple[CostComponent, ...], str]:
    """Price a rules stage's draws and describe them.

    The ``jev_memory`` profile is priced by its owner over one topic per tool plus the
    environment, capped by its topic limit. Agentic research buys one bounded draw per trace,
    variant and round, then one weighing per merged rule and one condensation. The rule count is
    known only after research, so the weighings are listed with their per-rule bound and left
    unpriced in the total.

    Args:
        project: Project whose catalog prices the author.
        spec: The registration.
        traces: World-learning traces read, at most.
        tools: Tools of the task, for the memory profile's topics.

    Returns:
        The cost components and one line describing them.
    """
    author = spec.author
    if spec.profile == "trajectory_tags":
        assert spec.tagging is not None
        logical_author, physical_decisions = rule_tag_call_bounds(
            traces=traces,
            maximum_rules=spec.tagging.maximum_rules,
            draws_per_trace=spec.draws_per_trace,
            rounds=spec.rounds,
            research_calls=spec.research.budget.model_calls,
            condensation_calls=spec.condensation.budget.model_calls,
            author_attempts=1,
        )
        rows = tuple(row.scaled(logical_author) for row in evaluation_costs(project, author))
        rows += (decision_cost(project, spec.tagging.decision_alias, physical_decisions),)
        return rows, (
            f"{traces} parallel trajectory research inputs, at most "
            f"{spec.tagging.maximum_rules} separate rule-question conversions, "
            f"{physical_decisions} native evidence-tag attempts and one condensation"
        )
    if spec.profile == "jev_memory" and spec.memory is not None:
        topics = min(tools + 1, spec.memory.maximum_topics)
        return memory_stage_costs(project, author, spec.weighing, spec.memory, topics=topics)
    per_call = evaluation_costs(project, author)
    if spec.profile == "single_draw":
        line = f"{traces} extraction draws with {author.model_alias}"
        return tuple(row.scaled(traces) for row in per_call), line
    research = traces * spec.draws_per_trace * spec.rounds * spec.research.budget.model_calls
    condensation = spec.condensation.budget.model_calls
    weighing = spec.weighing.budget.model_calls
    known = [row.scaled(research + condensation) for row in per_call]
    each = sum(row.usd or 0.0 for row in per_call) * weighing
    priced = all(row.usd is not None for row in per_call)
    known.append(CostComponent(f"{author.model_alias} weighing (per merged rule)", weighing, None))
    line = (
        f"{traces * spec.draws_per_trace * spec.rounds} research draws (up to {research} calls), "
        f"one weighing draw per merged rule (up to {weighing} calls, "
        + (f"${each:.6f}" if priced else "unpriced")
        + f" each; the rule count is known after research) and one condensation draw (up to "
        f"{condensation} calls) with {author.model_alias}"
    )
    return tuple(known), line


def trace_search(
    project: ProjectStore, build: BuildOutputs, task_id: str, top_k: int
) -> TraceSearchSpec | None:
    """Register tag-aware search over the build's scoped world-learning index.

    Args:
        project: Project holding the tag binding.
        build: The variance build.
        task_id: The mining-probe task whose query tags rank matches.
        top_k: Most matches one search returns.

    Returns:
        The registration, or None when the build saved no scoped index.
    """
    if build.index is None or build.retrieval_scope is None:
        return None
    tags = (
        None
        if build.tag_retrieval is None
        else load_tag_retrieval_binding(project, build.tag_retrieval)
    )
    return TraceSearchSpec(
        index=build.index,
        scope=build.retrieval_scope,
        tags=tags,
        task_id=task_id,
        maximum_top_k=top_k,
    )


def mining_rubric(
    context: StageContext, build: BuildOutputs, rules: RulesOutputs
) -> tuple[ArtifactInput, ArtifactInput]:
    """Author, or replay, the mining-probe task's rubric over a whole-task packet.

    Args:
        context: The running stage.
        build: The variance build.
        rules: The rules stage, whose bound job starts at the task's initial context.

    Returns:
        The saved rubric and the packet it was authored from.

    Raises:
        ValueError: The bound job does not start at the task's initial context, or the rubric
            author returned no ready rubric.
    """
    project, spec, revision = context.project, context.spec, context.code_revision
    job = rules.job
    start = load_continuation_checkpoint(project.artifacts, job.study.checkpoint)
    actor = ChatAgentRuntime(
        maximum_model_calls=job.maximum_model_calls, system_prompt=job.system_prompt
    )
    if start.actor != actor.initial_checkpoint(start.task):
        raise ValueError(
            "authoring the mining rubric needs a mining job starting at the task's initial "
            "context; register mining.rubric instead"
        )
    exposure = load_exposure_plan(project, build.exposure)
    tools = tuple(tool.name for tool in start.task.tools)
    packet = build_authoring_packet(
        family_id=local_id(f"mining-{start.task.task_id}"),
        task=start.task,
        checkpoint=job.study.checkpoint,
        before_state=dict(
            GroundedRuntimeDefinition.model_validate(job.configuration).initial_state
        ),
        public_history=start.actor.messages,
        remaining_calls=actor.remaining_calls(start.actor),
        sources=world_sources(build),
        exposure=exposure,
        rule_book=rules.rulebook,
        book=load_rulebook(project, rules.rulebook),
        decision=None,
        scenarios=(),
        observed_episodes=observed_episodes(
            project, build, exposure, tools, spec.authoring.maximum_episodes
        ),
    )
    packet_ref = save_packet(project, packet, code_revision=revision)
    identity = stable_id(
        "mining-rubric",
        {"packet": packet_ref.sha256, "author": rubric_author_sha256(spec)},
    )
    run = author_rubric_agentic(
        project,
        spec.authoring.rubric_author,
        packet_ref,
        identity=identity,
        code_revision=revision,
        profile=spec.authoring.rubric_profile,
    )
    if run.draft.status != "ready":
        raise ValueError(
            f"the mining rubric author is {run.draft.status}: {run.draft.failure}; "
            "register mining.rubric or fix the author settings"
        )
    return save_rubric(project, run.draft, code_revision=revision), packet_ref


def rubric_author_sha256(spec: DecisionVarianceSpec) -> str:
    """Digest of the rubric author settings and profile an authored mining rubric reads."""
    return sha256_json(
        {
            "author": spec.authoring.rubric_author.model_dump(mode="json"),
            "profile": spec.authoring.rubric_profile.model_dump(mode="json"),
        }
    )


def world_learning_traces(project: ProjectStore, build: BuildOutputs) -> tuple[Trace, ...]:
    """Read the build's world-learning traces from its pinned dataset, in order."""
    dataset = build.trace_dataset
    stored = project.artifacts.read(dataset.artifact_id)
    if artifact_input(stored.manifest) != dataset:
        raise ValueError(f"trace dataset {dataset.artifact_id} differs from its pinned reference")
    by_id = {
        trace.trace_id: trace
        for trace in load_trace_dataset(project.artifacts, dataset.artifact_id).traces
    }
    missing = [trace_id for trace_id in build.rules_traces if trace_id not in by_id]
    if missing:
        raise ValueError(f"world-learning traces missing from the dataset: {', '.join(missing)}")
    return tuple(by_id[trace_id] for trace_id in build.rules_traces)


def retrieval_embedder(context: StageContext, index: ArtifactInput) -> RAGEmbedderBinding | None:
    """Bind the build's catalog embedder role for a semantic index, behind the admitted transport.

    Args:
        context: The running stage.
        index: The scoped world-learning index.

    Returns:
        The query embedder, or None for a local hashing index, which needs no provider.

    Raises:
        ValueError: The configured embedder differs from the index's or cannot embed.
    """
    project = context.project
    catalog = load_model_catalog(project.model_catalog_path)
    loaded = load_rag_index(project.artifacts, index.artifact_id).index
    alias = rag_index_embedder_alias(project, catalog, loaded)
    if alias is None:
        return None
    transport = admitted_transport(
        context, "retrieval", _searches(context.spec), context.spec.mining.registered.deadline
    )
    resolved = RuntimeModelCatalog(catalog, transport_factory=lambda: transport).resolve(alias)
    if resolved.embedding_client is None or resolved.snapshot != loaded.embedder:
        raise ValueError(f"embedder alias {alias!r} cannot embed this index's queries")
    return RAGEmbedderBinding(
        client=resolved.embedding_client,
        snapshot=resolved.snapshot,
        maximum_attempts=RetryPolicy().maximum_attempts,
        maximum_input_tokens=resolved.capabilities.context_window_tokens,
    )


def retrieval_costs(project: ProjectStore, spec: DecisionVarianceSpec) -> tuple[CostComponent, ...]:
    """Name query embeddings as unpriced when the build's retrieval is semantic.

    The scoped index reuses the completed build's embedder, so the build's serving index decides
    before the variance build runs whether queries need a provider.
    """
    build = project.load_project().build
    if not spec.mining.retrieval or build is None:
        return ()
    serving = load_rag_index(project.artifacts, build.serving_rag.artifact_id).index
    if default_rag_embedder(serving.embedding_dimension).snapshot == serving.embedder:
        return ()
    return (CostComponent("embedder", _searches(spec) * RetryPolicy().maximum_attempts, None),)


def _searches(spec: DecisionVarianceSpec) -> int:
    """Most searches every tier's selectors and critics can make, one per evidence-tool call."""
    mining = spec.mining
    per_tier = mining.selector_draws * mining.selector_profile.budget.tool_calls
    per_tier += mining.critic_draws * mining.critic_profile.budget.tool_calls
    total = per_tier * len(mining.actors)
    schedule = mining.schedule
    if schedule is not None and schedule.base.initial.actor_alias in mining.actors:
        base = schedule.base
        selectors = max(len(base.judges), schedule.steady_draws)
        critics = max(len(base.critics), schedule.steady_draws)
        scheduled = selectors * base.selector_profile.budget.tool_calls
        scheduled += critics * base.critic_profile.budget.tool_calls
        total += max(0, scheduled - per_tier)
    return max(1, total)


def admitted_transport(
    context: StageContext, purpose: str, attempts: int, deadline: datetime
) -> AdmittedJsonTransport:
    """Return the admitted transport of one stage purpose, with its own dispatch ledger."""
    project = context.project
    return AdmittedJsonTransport(
        HttpxJsonTransport(),
        ledger=DispatchLedger(
            project.artifacts.project_directory / f"{context.identity}-{purpose}.sqlite",
            max(1, attempts),
            sha256_json(f"{context.identity}-{purpose}"),
        ),
        artifacts=project.artifacts,
        code_revision=context.code_revision,
        deadline=deadline,
    )
