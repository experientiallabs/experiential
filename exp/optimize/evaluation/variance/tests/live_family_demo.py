"""Run the judged decision-family path at small scale against real open-weight models.

This is an operator script, not a pytest module (its name does not end in ``_test.py``). It runs
the same path as :mod:`~exp.optimize.evaluation.variance.tests.judged_proof_test` on the support
ticket task, with every scripted wire replaced by admitted model calls through the project's
``models.toml``:

1. One authoring packet, independent world and rubric author draws, a world freeze and a
   compatibility review; one judge profile fixed before the rubric and one judge definition per
   frozen variant.
2. Per actor: a small error-mining run (G/J/C/K/M) from the frozen urgent start and one decision
   family judged by the same frozen judge. Every actor is its own tier: its runs, families and
   records are never pooled with another actor's.
3. Per actor: the urgent variant's executable task, ready for Harbor packaging.

The default is ``--dry-run``: it reads the catalog, checks every alias, prints the planned calls
per role and a conservative upper-bound spend over every actor, and makes no provider call and
builds no credential or client. A live run needs ``--no-dry-run --yes``; ``--yes`` authorizes
the displayed estimate, and an estimate the catalog cannot price is shown as undefined and still
needs ``--yes``. Live execution also needs a project built by ``exp build`` whose grounded world
model artifact is passed with ``--world-model``.

Example::

    python -m exp.optimize.evaluation.variance.tests.live_family_demo --root .exp --project demo
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from rich.console import Console

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    JsonObject,
    sha256_bytes,
    sha256_json,
)
from exp.common.models import ModelCatalog, ModelCatalogError, ToolCall, load_model_catalog
from exp.common.project import ProjectStore, artifact_input
from exp.common.tasks import TaskCase, ToolSchema
from exp.optimize.evaluation.continuations.job import ContinuationJob
from exp.optimize.evaluation.continuations.study import ContinuationStudySpec
from exp.optimize.evaluation.executable.grounded import (
    GroundedRuntimeDefinition,
    continuation_runtime,
)
from exp.optimize.evaluation.executable.task import persist_executable_task
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.rules.stage import (
    REPORT_TYPE,
    RULE_BOOK_TYPE,
    RulesStageReport,
    RulesStageSpec,
    rulebook_files,
)
from exp.optimize.evaluation.variance.actor_critic import ActorCriticSpec, run_actor_critic
from exp.optimize.evaluation.variance.admission import VariantJudging, frozen_run_digests
from exp.optimize.evaluation.variance.authoring_packet import (
    SourceRef,
    build_authoring_packet,
    save_packet,
)
from exp.optimize.evaluation.variance.contracts import (
    Criterion,
    DecisionOption,
    DecisionState,
    Scenario,
    StateEdit,
    VariationAxis,
    WitnessStep,
)
from exp.optimize.evaluation.variance.exposure import ExposurePlan
from exp.optimize.evaluation.variance.family import (
    DecisionFamilySpec,
    family_executable_task,
    run_decision_family,
)
from exp.optimize.evaluation.variance.judge_prompt import judge_profile
from exp.optimize.evaluation.variance.judged import EpisodeJudge
from exp.optimize.evaluation.variance.rubric_author import (
    author_rubric,
    judge_definition,
    load_rubric,
    save_rubric,
)
from exp.optimize.evaluation.variance.world_author import (
    author_world,
    load_world_draft,
    world_draft_reference,
    world_runtime_definition,
)
from exp.optimize.evaluation.variance.world_freeze import freeze_world, review_compatibility
from exp.runtime.agents.chat import ChatAgentRuntime
from exp.runtime.environments.world_contracts import (
    StateDomain,
    ToolEffectPolicy,
    WorldManifest,
)
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.simulation.continuations import (
    ContinuationCheckpoint,
    persist_continuation_checkpoint,
)
from exp.simulation.evaluation.tasks import OutcomeRule, TaskVerifier
from exp.simulation.induction.sandbox import ProgramSandbox
from exp.simulation.rules.model import Operand, Predicate, Rule, RuleBook, ToolContract

logger = logging.getLogger(__name__)

COUNCIL = ("gemma-4-26b-a4b-it", "glm-5.3-flash", "glm-5.3", "kimi-k3", "mimo-v2.6-pro")
"""Default actor aliases, all served through the gateway; each is its own tier."""

JUDGE = "deepseek-judge"
"""Default open-weight episode judge alias."""

TOOLS = (
    ToolSchema(
        name="read_ticket",
        description="Read the ticket's handling status.",
        input_schema={"type": "object"},
    ),
    ToolSchema(
        name="check_sla",
        description="Read the ticket's service tier and lock.",
        input_schema={"type": "object"},
    ),
    ToolSchema(
        name="escalate",
        description="Hand the ticket to the second line.",
        input_schema={"type": "object"},
    ),
    ToolSchema(
        name="close", description="Close the ticket as handled.", input_schema={"type": "object"}
    ),
)
"""The ticket task's tools."""

INITIAL: JsonObject = {
    "tickets": {"severity": "high", "lock": "none", "status": "open", "outcome": "pending"}
}
"""The ticket world's start state."""

ACTIONS = {
    "escalate": ToolCall(name="escalate", call_id="force-escalate", arguments={}),
    "close": ToolCall(name="close", call_id="force-close", arguments={}),
}
"""Concrete forced first action per decision option."""


@dataclass(frozen=True)
class Scale:
    """Small-scale counts for one live run.

    Attributes:
        attempts: Error-mining group size ``G`` per actor.
        selectors: Point-selector draws ``J`` per actor.
        critics: Critic draws ``C`` per actor.
        points: Most points per selector ``K``.
        suffixes: Continuations per distinct point ``M``.
        repeats: Pairs per selection contrast.
        confirmation: Pairs per fresh confirmation contrast.
        actor_calls: Actor call allowance per episode.
        output_tokens: Output allowance per request.
        input_tokens: Conservative input tokens per request.
    """

    attempts: int = 4
    selectors: int = 2
    critics: int = 2
    points: int = 2
    suffixes: int = 3
    repeats: int = 4
    confirmation: int = 4
    actor_calls: int = 6
    output_tokens: int = 2048
    input_tokens: int = 24_000

    def calls(self, variants: int = 2, options: int = 2) -> dict[str, int]:
        """Upper-bound request counts per role for one actor.

        Args:
            variants: Family variants.
            options: Decision options per variant.

        Returns:
            Requests per role: actor, judge and role draws.
        """
        distinct = self.selectors * self.points
        episodes = self.attempts + distinct * self.suffixes
        cells = variants * options * (self.repeats + self.confirmation)
        return {
            "actor": (episodes + cells) * self.actor_calls,
            "judge": distinct * self.suffixes + cells,
            "roles": self.selectors + self.critics + 2,
        }


def price(catalog: ModelCatalog, alias: str, inputs: int, outputs: int) -> float | None:
    """Return a conservative dollar cost of one alias's tokens, or None when unpriced.

    Declared capability prices win; discovered prices are the fallback.
    """
    record = catalog.models[alias]
    for source in (record.capabilities, record.discovery):
        if source is None:
            continue
        rates = (
            source.input_cost_per_million_tokens_usd,
            source.output_cost_per_million_tokens_usd,
        )
        if rates[0] is not None and rates[1] is not None:
            return (inputs * rates[0] + outputs * rates[1]) / 1_000_000
    return None


def estimate(
    catalog: ModelCatalog, actors: Sequence[str], judge: str, author: str, scale: Scale
) -> tuple[float | None, list[tuple[str, str, int, float | None]]]:
    """Estimate the whole run's upper-bound spend over every actor tier.

    Args:
        catalog: The project catalog.
        actors: Actor aliases, one tier each.
        judge: Judge alias.
        author: Author and role-draw alias.
        scale: Counts.

    Returns:
        The total (None when any alias is unpriced) and one row per (tier, role).
    """
    rows: list[tuple[str, str, int, float | None]] = []
    authoring = 2
    rows.append(("shared", author, authoring, _cost(catalog, author, authoring, scale)))
    for actor in actors:
        calls = scale.calls()
        rows.append((actor, actor, calls["actor"], _cost(catalog, actor, calls["actor"], scale)))
        rows.append((actor, judge, calls["judge"], _cost(catalog, judge, calls["judge"], scale)))
        rows.append((actor, author, calls["roles"], _cost(catalog, author, calls["roles"], scale)))
    costs = [row[3] for row in rows]
    total = None if any(cost is None for cost in costs) else sum(c for c in costs if c)
    return total, rows


def _cost(catalog: ModelCatalog, alias: str, requests: int, scale: Scale) -> float | None:
    """Price ``requests`` full-allowance requests of one alias."""
    return price(catalog, alias, requests * scale.input_tokens, requests * scale.output_tokens)


def parse(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        description="Run the judged decision-family demo against real models."
    )
    parser.add_argument("--root", type=Path, required=True, help="EXP root holding models.toml")
    parser.add_argument("--project", default="demo", help="project identity under the root")
    parser.add_argument("--actors", nargs="+", default=list(COUNCIL), help="actor aliases")
    parser.add_argument("--judge", default=JUDGE, help="episode judge alias")
    parser.add_argument("--author", default=None, help="author/role alias; default the judge")
    parser.add_argument("--world-model", default=None, help="grounded world model artifact id")
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--yes", action="store_true", help="authorize the displayed estimate")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Plan, and with ``--no-dry-run --yes`` run, the live family demo.

    Args:
        argv: Arguments, or None for ``sys.argv``.

    Returns:
        Process exit status.
    """
    args = parse(argv)
    console = Console()
    author = args.author or args.judge
    try:
        catalog = load_model_catalog(args.root / "models.toml")
    except ModelCatalogError as error:
        console.print(f"[red]{error}[/red]")
        return 2
    actors = tuple(dict.fromkeys(args.actors))
    missing = sorted({*actors, args.judge, author} - set(catalog.models))
    if missing:
        console.print(f"[red]aliases missing from {args.root / 'models.toml'}: {missing}[/red]")
        return 2
    scale = Scale()
    total, rows = estimate(catalog, actors, args.judge, author, scale)
    console.print(
        f"decision-family demo: {len(actors)} actor tiers (never pooled), judge {args.judge}, "
        f"authors {author}; G={scale.attempts} J={scale.selectors} C={scale.critics} "
        f"K={scale.points} M={scale.suffixes}, repeats {scale.repeats}+{scale.confirmation}"
    )
    for tier, alias, requests, cost in rows:
        shown = "undefined" if cost is None else f"${cost:.2f}"
        console.print(f"  {tier:<20} {alias:<22} {requests:>5} requests  <= {shown}")
    shown = "undefined (unpriced alias)" if total is None else f"${total:.2f}"
    console.print(f"conservative upper bound: {shown}")
    if args.dry_run:
        console.print("dry run: no provider call was made")
        return 0
    if not args.yes:
        console.print("[red]a live run needs --yes to authorize the estimate above[/red]")
        return 2
    if args.world_model is None:
        console.print("[red]a live run needs --world-model from an exp build project[/red]")
        return 2
    run_live(args, catalog, actors, author, scale, console)
    return 0


def run_live(
    args: argparse.Namespace,
    catalog: ModelCatalog,
    actors: tuple[str, ...],
    author: str,
    scale: Scale,
    console: Console,
) -> None:
    """Run the authored, judged family path once per actor tier against real models.

    Args:
        args: Parsed arguments.
        catalog: The project catalog.
        actors: Actor aliases, one tier each.
        author: Author and role-draw alias.
        scale: Counts.
        console: Output console.
    """
    revision = "live-family-demo"
    project = ProjectStore(args.root, args.project)
    store = project.artifacts
    registry = RuntimeModelCatalog(catalog)
    catalog_sha256 = sha256_json(catalog.model_dump(mode="json"))
    deadline = datetime.now(UTC) + timedelta(hours=6)

    def settings(alias: str) -> AuthorSettings:
        """Frozen request settings for one alias."""
        return AuthorSettings(
            model_alias=alias,
            model=registry.snapshot(alias)[0],
            catalog_sha256=catalog_sha256,
            output_tokens=scale.output_tokens,
            deadline=deadline,
        )

    judge_settings, author_settings = settings(args.judge), settings(author)
    task = TaskCase(
        task_id="ticket",
        lineage_group_id="tickets",
        origin="authored",
        partition="fit",
        instruction="Resolve the support ticket with the tools you have.",
        tools=TOOLS,
        workload_weight=1,
    )
    world_model = artifact_input(store.read(args.world_model).manifest)
    base = GroundedRuntimeDefinition(
        task=task,
        world_model=world_model,
        catalog_sha256=catalog_sha256,
        manifest=WorldManifest(
            domains=(StateDomain(name="tickets", owner="simulated"),),
            tools=tuple(
                ToolEffectPolicy(
                    name=tool.name,
                    reads=("tickets",),
                    writes=("tickets",) if tool.name in ACTIONS else (),
                )
                for tool in TOOLS
            ),
        ),
        initial_state=INITIAL,
        probe_paths={"tickets": "/tickets"},
        maximum_physical_attempts=20_000,
        deadline=deadline,
    )
    tier_rule = Rule(
        rule_id="tickets.tier",
        tool="escalate",
        kind="effect",
        statement="Escalating resolves a high tier ticket; closing resolves a low tier one.",
        status="supported",
        support=2,
        contradiction=0,
    )
    lock_rule = Rule(
        rule_id="tickets.lock",
        tool="close",
        kind="precondition",
        statement="A ticket another agent holds cannot be closed.",
        predicate=Predicate(
            op="eq", left=Operand(scope="before", path="/tickets/lock"), right=Operand(value="none")
        ),
        status="supported",
        support=3,
        contradiction=0,
    )
    book = RuleBook(
        contracts={tool.name: ToolContract(name=tool.name) for tool in TOOLS},
        rules=(tier_rule, lock_rule),
        sources=("trace-1",),
    )
    envelope = ArtifactEnvelope(
        schema_version=1, created_at=datetime.now(UTC), code_revision=revision
    )
    rulebook_id = "live-demo-rule-book"
    if not store.exists(rulebook_id):
        store.write(
            artifact_id=rulebook_id,
            artifact_type=RULE_BOOK_TYPE,
            envelope=envelope,
            files=rulebook_files(book),
        )
    rulebook = artifact_input(store.read(rulebook_id).manifest)
    policy = ChatAgentRuntime(maximum_model_calls=scale.actor_calls)
    with continuation_runtime(project, base.model_dump(mode="json")) as runtime:
        with runtime.world.open(task) as session:
            start = ContinuationCheckpoint(
                task=task,
                actor=policy.initial_checkpoint(task),
                world=session.checkpoint(),
                sources=(world_model,),
            )
    checkpoint = persist_continuation_checkpoint(
        store, start, created_at=datetime.now(UTC), code_revision=revision
    )
    decision = DecisionState(
        step=0,
        decision="Escalate or close the ticket.",
        options=(
            DecisionOption(option_id="escalate", action="Hand the ticket to the second line"),
            DecisionOption(option_id="close", action="Close the ticket as handled"),
        ),
        relevant_paths=("/tickets",),
        axes=(
            VariationAxis(
                path="/tickets/severity", values=("high", "low"), effect="The tier decides."
            ),
        ),
        criteria=(Criterion(criterion_id="handled", text="The ticket was handled correctly."),),
        evidence_refs=("point", "rule:tickets.tier"),
    )
    scenarios = tuple(
        Scenario(
            scenario_id=name,
            edits=(StateEdit(path="/tickets/severity", value=severity),),
            correct_option="escalate" if severity == "high" else "close",
            deciding_paths=("/tickets/severity",),
            witness=(WitnessStep(tool="check_sla", arguments={}, reveals=("/tickets/severity",)),),
            rationale=f"A {severity} tier ticket is resolved by the matching handling.",
            criteria=("handled",),
        )
        for name, severity in (("urgent", "high"), ("routine", "low"))
    )
    exposure = ExposurePlan(
        bindings=ArtifactInput(artifact_id="bindings", sha256="0" * 64),
        seed=0,
        probe_fraction=0.5,
        roles={"tickets": "mining_probes", "sources": "world_learning"},
        traces={"trace-1": "sources"},
    )
    packet = build_authoring_packet(
        family_id="ticket-tier",
        task=task,
        checkpoint=checkpoint,
        before_state=INITIAL,
        public_history=start.actor.messages,
        remaining_calls=scale.actor_calls,
        sources=(
            SourceRef(
                source_id="trace-1",
                kind="trace",
                trace_id="trace-1",
                summary="An agent closed a low tier ticket.",
            ),
        ),
        exposure=exposure,
        rule_book=rulebook,
        book=book,
        decision=decision,
        scenarios=scenarios,
    )
    packet_ref = save_packet(project, packet, code_revision=revision)
    profile = judge_profile(
        model_alias=args.judge,
        model=judge_settings.model,
        maximum_output_tokens=judge_settings.output_tokens,
    )
    rubric_draft = author_rubric(
        project, author_settings, packet_ref, identity="live-rubric", code_revision=revision
    )
    if rubric_draft.rubric is None:
        raise SystemExit(f"rubric author: {rubric_draft.status} {rubric_draft.failure}")
    rubric = save_rubric(project, rubric_draft, code_revision=revision)
    with ProgramSandbox() as sandbox:
        draft = author_world(
            project,
            author_settings,
            packet_ref,
            sandbox=sandbox,
            identity="live-world",
            code_revision=revision,
        )
        if draft.world is None:
            raise SystemExit(f"world author: {draft.status} {draft.failure}")
        world_ref = world_draft_reference(project, draft)
        freeze = freeze_world(
            project,
            packet_ref,
            world_ref,
            sandbox=sandbox,
            code_revision=revision,
            base=base,
            actor=policy,
        )
    world = load_world_draft(project, world_ref).world
    assert world is not None
    review = review_compatibility(packet, world, load_rubric(project, rubric))
    console.print(f"freeze trainable={freeze.trainable}; review compatible={review.compatible}")
    freeze_ref = artifact_input(store.read(freeze.identity + "-result").manifest)
    judging: dict[str, VariantJudging] = {}
    for variant in ("urgent", "routine"):
        task_sha256, world_sha256 = frozen_run_digests(project, freeze_ref, variant)
        runtime_definition = world_runtime_definition(packet, world, variant, base=base)
        judging[variant] = VariantJudging(
            judge=EpisodeJudge(
                definition=judge_definition(
                    project, profile, rubric, task_sha256=task_sha256, world_sha256=world_sha256
                ),
                settings=judge_settings,
            ),
            world_freeze=freeze_ref,
            freeze_variant=variant,
            runtime=runtime_definition.model_dump(mode="json"),
            compatibility=review,
        )
    urgent = judging["urgent"]
    frozen = next(row for row in freeze.variants if row.variant_id == "urgent")
    if frozen.checkpoint is None or urgent.runtime is None:
        raise SystemExit("the urgent variant did not freeze; see the freeze report")
    runtime_file = args.root / "live_family_runtime.py"
    runtime_file.write_text(
        "from exp.optimize.evaluation.executable.grounded import continuation_runtime\n"
    )
    probe = GroundedRuntimeDefinition.model_validate(urgent.runtime).probe_sha256
    for actor in actors:
        tier = "".join(c if c.isalnum() or c in "._-" else "-" for c in actor.lower())[:64]
        job = ContinuationJob(
            study=ContinuationStudySpec(
                checkpoint=frozen.checkpoint,
                actor_model=registry.snapshot(actor)[0],
                verifier=TaskVerifier(
                    description="Simulator diagnostics only; the episode judge labels outcomes.",
                    outcome_schema={},
                    required_paths=("/final_state/tickets/outcome",),
                    rules=(
                        OutcomeRule(
                            path="/final_state/tickets/outcome", operator="equals", value="right"
                        ),
                    ),
                    allowed_tools=tuple(tool.name for tool in TOOLS),
                ),
                probe_sha256=probe,
                repeats=scale.attempts,
                selector=f"live family demo; frozen urgent start; tier {tier}",
            ),
            runtime_file=runtime_file,
            runtime_sha256=sha256_bytes(runtime_file.read_bytes()),
            configuration=urgent.runtime,
            actor_alias=actor,
            catalog_sha256=catalog_sha256,
            maximum_model_calls=scale.actor_calls,
            maximum_physical_attempts=400,
            deadline=deadline,
        )
        mining = ActorCriticSpec.error_mining(
            job,
            episode_judge=urgent.judge,
            selector=author_settings,
            critic=author_settings,
            attempts=scale.attempts,
            selector_draws=scale.selectors,
            critic_draws=scale.critics,
            points=scale.points,
            continuations=scale.suffixes,
            actor_tier=tier,
        )
        report = run_actor_critic(project, mining, exposure=exposure, code_revision=revision)
        family = run_decision_family(
            project,
            DecisionFamilySpec(
                decision_author=author_settings,
                scenario_author=author_settings,
                rules_stage=_rules_stage(project, author_settings, rulebook, book, revision),
                actions=ACTIONS,
                judging=judging,
                repeats=scale.repeats,
                confirmation_repeats=scale.confirmation,
                tiers={job.study.actor_model.model_id: tier},
            ),
            exposure=exposure,
            report=report,
            point=0,
            code_revision=revision,
        )
        console.print(
            f"[bold]{tier}[/bold]: eligible={family.eligibility.eligible} "
            f"failure={family.failure} reasons={list(family.eligibility.reasons)}"
        )
        if any(row.scenario_id == "urgent" and row.decision_checkpoint for row in family.variants):
            executable = family_executable_task(family, "urgent", intent="Ticket tier handling")
            reference = persist_executable_task(
                project, executable, created_at=datetime.now(UTC), code_revision=revision
            )
            console.print(f"  urgent executable task: {reference.artifact_id}")


def _rules_stage(
    project: ProjectStore,
    settings: AuthorSettings,
    rulebook: ArtifactInput,
    book: RuleBook,
    revision: str,
) -> ArtifactInput:
    """Save, or adopt, the rules-stage report naming the demo's rule book."""
    store, artifact_id = project.artifacts, "live-demo-rules-stage-result"
    if not store.exists(artifact_id):
        report = RulesStageReport(
            spec=RulesStageSpec(author=settings),
            identity="live-demo-rules-stage",
            extraction_identity="live-demo-rules-extraction",
            contracts=dict(book.contracts),
            draws=(),
            failures=(),
            rulebook=rulebook,
            rules=len(book.rules),
            rule_counts={"supported": len(book.rules), "contradicted": 0, "unknown": 0},
            open_questions=0,
            diagnostic=(),
            requires_witness=(),
            sessions=(),
            session_ids=(),
            properties=None,
            workloads=(),
            triage=(),
            families_must_stay_diagnostic=(),
        )
        store.write_json(
            artifact_id=artifact_id,
            artifact_type=REPORT_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=1,
                created_at=datetime.now(UTC),
                code_revision=revision,
                inputs=(rulebook,),
            ),
            files={"report.json": report},
        )
    return artifact_input(store.read(artifact_id).manifest)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
