"""Offline acceptance test of the whole Experiential pipeline, driven through the public CLI.

One test per world kind walks every stage through ``exp`` commands, with scripted providers
only, so a break between two stages fails here even when every stage's own tests pass:

1. ``exp build`` imports chat-json traces recorded in the native world, grounds a world model
   and learns from every trace.
2. ``exp eval --audit`` measures the simulator on held-out traces with frozen fit-only memory,
   then replays the saved audit without a provider call.
3. ``register-admission-policy`` fixes the per-dimension admission policy before any result;
   ``development-tasks`` chooses a fit task of the build, ``ground-runtime`` grounds the built
   simulator with its learned memory and the trace's instance facts,
   ``derive-equivalence-rubric`` and ``freeze-equivalence-rubric`` fix the judge's rubric from an
   admitted development trajectory of another family, and ``compare-environments`` runs the
   native world and that simulator on the captured task under one actor policy, that policy's
   margins and the frozen rubric.
4. ``register-specification`` saves an independent acceptance specification, and ``capture``
   and ``run`` fork native branches from the captured checkpoint.
5. ``export-traces`` turns those trajectories into imports under the policy: native ones are
   admitted and learned by ``exp build --import-id``; simulated ones stay unassessable without
   an applicable calibration and are refused as learning source.
6. ``repair-learning`` screens learning misses into lessons, and the audit compares the three
   memory profiles.
7. ``harbor`` registers the native task, witnesses its controls, qualifies it, exports and
   verifies the package, and serves an offline trial to separate actor and verifier processes.

Finally the trajectories are exported to their default destination, the source project
itself, and ``exp build`` runs the command the export prints.

The two world kinds are the package-owned native workbook adapter and an operator-selected
SQLite record world. Scripted actor, simulator, judge, embedder and reviewer transports answer
every request in process; nothing leaves the machine.
"""

import functools
import json
import re
import sqlite3
import tempfile
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from exp.cli.app import app
from exp.cli.evaluation.executable import ExecutableCommand
from exp.common.core.artifacts import (
    ArtifactInput,
    JsonObject,
    JsonValue,
    sha256_bytes,
    sha256_json,
)
from exp.common.evaluations.actor_equivalence import RUBRIC_DIMENSIONS
from exp.common.evaluations.reconstruction import WorkflowEvidence, WorkflowStep
from exp.common.models import ToolCall, load_model_catalog
from exp.common.project import ProjectStore, artifact_input
from exp.common.tasks import TaskCase, ToolSchema
from exp.common.tasks.facts import FactProvenance, FactValidation, InstanceFact, InstanceFacts
from exp.common.tasks.store import load_task_set
from exp.common.traces.ingest.persistence import read_ingested_traces
from exp.optimize.evaluation.analysis.source_access import declare_source_access
from exp.optimize.evaluation.audit.panel import MemoryProfile, plan_held_out_panel
from exp.optimize.evaluation.audit.panel_test import (
    SimulatedProvider,
    _at,
    _call_key,
    _tool_reply,
    _user_content,
    write_catalog,
)
from exp.optimize.evaluation.continuations.capture import CaptureDefinition
from exp.optimize.evaluation.continuations.job import ContinuationJob
from exp.optimize.evaluation.continuations.study import ContinuationStudySpec
from exp.optimize.evaluation.equivalence.policy import policy_identity
from exp.optimize.evaluation.equivalence.policy_test import fixture_policy
from exp.optimize.evaluation.equivalence.review import ReviewModel
from exp.optimize.evaluation.equivalence.rubric import (
    RubricDerivationSpec,
    RubricTrajectory,
    register_rubric_trajectory,
)
from exp.optimize.evaluation.equivalence.runner import EquivalenceCase, EquivalenceSpec
from exp.optimize.evaluation.executable import workbook as workbook_runtime
from exp.optimize.evaluation.executable.loader import ContinuationRuntime
from exp.optimize.evaluation.executable.task import ExecutableTask
from exp.optimize.evaluation.executable.workbook import (
    WorkbookRuntimeDefinition,
    WorkbookStateView,
)
from exp.optimize.evaluation.fidelity import runner as fidelity
from exp.optimize.evaluation.harbor.tests.package_trial_test import _run_trial
from exp.optimize.evaluation.learning import lesson_evolution, screening
from exp.optimize.evaluation.learning.lessons import (
    latest_lesson_set,
    load_lesson_set,
    screening_plan,
)
from exp.optimize.evaluation.learning.traces import learning_plan
from exp.runtime.environments.artifact_world import ArtifactWorldFactory
from exp.runtime.environments.record_world import RecordTool, RecordToolError, RecordWorldFactory
from exp.runtime.environments.workbook_edits import (
    WorkbookEditAdapter,
    read_supported_workbook,
    workbook_projection,
)
from exp.runtime.environments.world_contracts import (
    ExecutableCheckpoint,
    StateDomain,
    ToolEffectPolicy,
    WorldAccess,
    WorldManifest,
)
from exp.runtime.environments.worlds import ExecutableWorldRuntime, WorldContext
from exp.runtime.models import RuntimeModelCatalog
from exp.runtime.models.providers import transport
from exp.runtime.models.providers.transport import (
    JsonHttpResponse,
    RecordedRequest,
    ScriptedJsonTransport,
)
from exp.simulation.continuations import WorldCheckpoint
from exp.simulation.evaluation.admission import (
    EvaluationSpecification,
)
from exp.simulation.evaluation.record_execution import record_state
from exp.simulation.evaluation.tasks import OutcomeRule, TaskVerifier
from exp.simulation.mining.bindings import load_task_set_lineage_bindings
from exp.simulation.world_construction import (
    ConstructionSpec,
    TablePlan,
    ValuesColumn,
    compile_construction,
)
from exp.simulation.world_model.experience_memory import memory_head

REVISION = "c" * 40
EPISODES = 10
AUDITED = 2
DEADLINE = datetime.now(UTC) + timedelta(hours=2)
ACTOR_CALLS = 8


def schema(name: str, description: str, **properties: str) -> ToolSchema:
    """Declare one public tool with typed object arguments."""
    return ToolSchema(
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "properties": {key: {"type": kind} for key, kind in properties.items()},
            "required": sorted(properties),
        },
    )


@dataclass(frozen=True)
class Episode:
    """One recorded customer episode: its request, lineage and the actions its agent took.

    Attributes:
        lineage: Declared task identity of the episode.
        instruction: The customer's request.
        actions: Tool calls in order; the first is the disclosed public prefix.
    """

    lineage: str
    instruction: str
    actions: tuple[ToolCall, ...]


@dataclass(frozen=True)
class World:
    """One world kind: its native runtime, recorded traffic and the roles of its tools.

    Attributes:
        kind: Short name used for projects and parameter IDs.
        tools: Public tools every episode and task declares.
        manifest: Native domain ownership and per-tool effects.
        episodes: Recorded episodes, one lineage each.
        native: Opens the native world one recorded episode ran in.
        learnable: Tool whose reply format memory can teach.
        instance: Tool whose replies read unseen instance data and stay mispredicted.
    """

    kind: str
    tools: tuple[ToolSchema, ...]
    manifest: WorldManifest
    episodes: tuple[Episode, ...]
    native: Callable[[int], ExecutableWorldRuntime]
    learnable: str
    instance: str


# Record world: a retail order service on SQLite, one user and order per episode.

RECORD_TOOLS = (
    schema("find_user_id_by_email", "Find a user ID by email.", email="string"),
    schema("get_order_details", "Read one order.", order_id="string"),
    schema("get_user_details", "Read one user's profile and orders.", user_id="string"),
    schema("cancel_pending_order", "Cancel a pending order.", order_id="string", reason="string"),
)


def record_schema() -> str:
    """Seed users and orders; every third order has already been delivered."""
    rows = []
    for index in range(EPISODES):
        status = "delivered" if index % 3 == 0 else "pending"
        rows.append(
            f"INSERT INTO users VALUES ('user_{index:02d}', 'user_{index:02d}@example.test', "
            f"'{'gift_card' if index % 2 else 'credit_card'}');\n"
            f"INSERT INTO orders VALUES ('#W{1000 + index}', 'user_{index:02d}', '{status}', "
            f"{10.5 + index});\n"
        )
    return (
        "CREATE TABLE users(user_id TEXT PRIMARY KEY, email TEXT, payment TEXT);\n"
        "CREATE TABLE orders(order_id TEXT PRIMARY KEY, user_id TEXT, status TEXT, "
        "total REAL);\n" + "".join(rows)
    )


def _reads_users(arguments: JsonObject) -> WorldAccess:
    """User lookups read the users table."""
    del arguments
    return WorldAccess(reads=frozenset({"users"}))


def _reads_orders(arguments: JsonObject) -> WorldAccess:
    """Order lookups read the orders table."""
    del arguments
    return WorldAccess(reads=frozenset({"orders"}))


def _reads_both(arguments: JsonObject) -> WorldAccess:
    """A profile joins users and orders."""
    del arguments
    return WorldAccess(reads=frozenset({"users", "orders"}))


def _writes_orders(arguments: JsonObject) -> WorldAccess:
    """A cancellation reads and writes orders."""
    del arguments
    return WorldAccess(reads=frozenset({"orders"}), writes=frozenset({"orders"}))


def _find_user(db: sqlite3.Connection, arguments: JsonObject, context: WorldContext) -> JsonValue:
    """Resolve an email to its user ID."""
    del context
    row = db.execute(
        "SELECT user_id FROM users WHERE email=?", (str(arguments.get("email")),)
    ).fetchone()
    if row is None:
        raise RecordToolError("no user has that email")
    return {"user_id": row[0]}


def _order(db: sqlite3.Connection, arguments: JsonObject, context: WorldContext) -> JsonValue:
    """Read one order's status and total."""
    del context
    row = db.execute(
        "SELECT order_id, status, total FROM orders WHERE order_id=?",
        (str(arguments.get("order_id")),),
    ).fetchone()
    if row is None:
        raise RecordToolError("order not found")
    return {"order_id": row[0], "status": row[1], "total": row[2]}


def _user(db: sqlite3.Connection, arguments: JsonObject, context: WorldContext) -> JsonValue:
    """Read a user's payment method and order IDs."""
    del context
    user = str(arguments.get("user_id"))
    row = db.execute("SELECT payment FROM users WHERE user_id=?", (user,)).fetchone()
    if row is None:
        raise RecordToolError("user not found")
    orders = [
        order
        for (order,) in db.execute(
            "SELECT order_id FROM orders WHERE user_id=? ORDER BY order_id", (user,)
        )
    ]
    return {"user_id": user, "orders": orders, "payment": row[0]}


def _cancel(db: sqlite3.Connection, arguments: JsonObject, context: WorldContext) -> JsonValue:
    """Cancel a pending order; a delivered order keeps its status."""
    del context
    order = str(arguments.get("order_id"))
    row = db.execute("SELECT status FROM orders WHERE order_id=?", (order,)).fetchone()
    if row is None:
        raise RecordToolError("order not found")
    if row[0] == "pending":
        db.execute("UPDATE orders SET status='cancelled' WHERE order_id=?", (order,))
        return {"order_id": order, "status": "cancelled"}
    return {"order_id": order, "status": row[0]}


RECORD_MANIFEST = WorldManifest(
    domains=(StateDomain(name="users", owner="native"), StateDomain(name="orders", owner="native")),
    tools=(
        ToolEffectPolicy(name="find_user_id_by_email", reads=("users",)),
        ToolEffectPolicy(name="get_order_details", reads=("orders",)),
        ToolEffectPolicy(name="get_user_details", reads=("users", "orders")),
        ToolEffectPolicy(name="cancel_pending_order", reads=("orders",), writes=("orders",)),
    ),
)


def record_world(index: int = 0) -> ExecutableWorldRuntime:
    """The native retail record service; every episode starts from the same database."""
    del index
    return ExecutableWorldRuntime(
        RECORD_MANIFEST,
        RecordWorldFactory(
            service={"name": "acceptance-retail", "version": 1},
            schema=record_schema(),
            tools={
                "find_user_id_by_email": RecordTool(_reads_users, _find_user),
                "get_order_details": RecordTool(_reads_orders, _order),
                "get_user_details": RecordTool(_reads_both, _user),
                "cancel_pending_order": RecordTool(_writes_orders, _cancel),
            },
        ),
        seed=7,
    )


def record_episode(index: int) -> Episode:
    """A customer asks to cancel their order; the agent looks it up first."""
    user, order = f"user_{index:02d}", f"#W{1000 + index}"

    def call(position: int, name: str, **arguments: JsonValue) -> ToolCall:
        """One numbered tool call of this episode."""
        return ToolCall(call_id=f"r{index}-{position}", name=name, arguments=arguments)

    return Episode(
        lineage=f"retail-{user}",
        instruction=f"I am {user}. Please cancel order {order} if it has not shipped.",
        actions=(
            call(0, "find_user_id_by_email", email=f"{user}@example.test"),
            call(1, "get_order_details", order_id=order),
            call(2, "get_user_details", user_id=user),
            call(3, "cancel_pending_order", order_id=order, reason="no longer needed"),
        ),
    )


# Workbook world: the package-owned native XLSX adapter, one order sheet per episode.

WORKBOOK_TOOLS = (
    schema("describe_workbook", "List worksheet names and sizes.", source="string"),
    schema(
        "inspect_workbook",
        "Read a rectangle of cells.",
        source="string",
        sheet="string",
        start_row="integer",
        row_count="integer",
        column_count="integer",
    ),
    schema(
        "edit_workbook",
        "Delete rows from an input into the output workbook.",
        source="string",
        destination="string",
        deletions="array",
    ),
)
WORKBOOK_MANIFEST = WorldManifest(
    domains=(
        StateDomain(name="input.xlsx", owner="native"),
        StateDomain(name="output.xlsx", owner="native"),
    ),
    tools=(
        ToolEffectPolicy(name="describe_workbook", reads=("input.xlsx",)),
        ToolEffectPolicy(name="inspect_workbook", reads=("input.xlsx",)),
        ToolEffectPolicy(name="edit_workbook", reads=("input.xlsx",), writes=("output.xlsx",)),
    ),
)


def orders_construction(index: int) -> ConstructionSpec:
    """One clerk's order sheet, with a duplicated second row, beside a protected audit sheet."""
    rows = 3 + index % 3
    return ConstructionSpec(
        name=f"orders-sheet-{index:02d}",
        assumptions=("Authored order sheet; not production data.",),
        tables={
            "Orders": TablePlan(
                rows=rows,
                columns={
                    "order": ValuesColumn(
                        kind="values",
                        values=tuple(f"#W{1000 + 10 * index + row}" for row in range(rows)),
                    ),
                    "status": ValuesColumn(
                        kind="values",
                        values=tuple("duplicate" if row == 1 else "open" for row in range(rows)),
                    ),
                },
            ),
            "Audit": TablePlan(
                rows=1, columns={"reviewer": ValuesColumn(kind="values", values=(f"a{index}",))}
            ),
        },
        workbooks={"orders.xlsx": ("Orders", "Audit")},
    )


def orders_workbook(index: int) -> bytes:
    """The XLSX bytes ``construct-assets`` publishes for one episode's construction."""
    return compile_construction(orders_construction(index))["assets/orders.xlsx"]


def workbook_world(index: int) -> ExecutableWorldRuntime:
    """The native workbook adapter over one episode's input workbook."""
    return ExecutableWorldRuntime(
        WORKBOOK_MANIFEST,
        ArtifactWorldFactory(
            WorkbookEditAdapter(),
            files={"input.xlsx": orders_workbook(index)},
            outputs=("output.xlsx",),
        ),
    )


def workbook_episode(index: int) -> Episode:
    """A clerk asks to drop the duplicated order row; the agent inspects before editing."""

    def call(position: int, name: str, **arguments: JsonValue) -> ToolCall:
        """One numbered tool call of this episode."""
        return ToolCall(call_id=f"w{index}-{position}", name=name, arguments=arguments)

    return Episode(
        lineage=f"orders-sheet-{index:02d}",
        instruction=f"Sheet {index:02d}: delete the duplicated order row into output.xlsx.",
        actions=(
            call(0, "inspect_workbook", source="input.xlsx", sheet="Orders", column_count=2),
            call(1, "describe_workbook", source="input.xlsx"),
            call(
                2,
                "inspect_workbook",
                source="input.xlsx",
                sheet="Orders",
                start_row=2,
                column_count=2,
            ),
            call(
                3,
                "edit_workbook",
                source="input.xlsx",
                destination="output.xlsx",
                deletions=[{"sheet": "Orders", "start": 2, "count": 1}],
            ),
        ),
    )


WORLDS = {
    "records": World(
        kind="records",
        tools=RECORD_TOOLS,
        manifest=RECORD_MANIFEST,
        episodes=tuple(record_episode(index) for index in range(EPISODES)),
        native=record_world,
        learnable="get_user_details",
        instance="get_order_details",
    ),
    "workbook": World(
        kind="workbook",
        tools=WORKBOOK_TOOLS,
        manifest=WORKBOOK_MANIFEST,
        episodes=tuple(workbook_episode(index) for index in range(EPISODES)),
        native=workbook_world,
        learnable="describe_workbook",
        instance="inspect_workbook",
    ),
}


def task_of(world: World, episode: Episode) -> TaskCase:
    """The public task an episode's agent was given."""
    return TaskCase(
        task_id=episode.lineage,
        lineage_group_id=episode.lineage,
        origin="authored",
        partition="fit",
        instruction=episode.instruction,
        workload_weight=1,
        tools=world.tools,
    )


def starting_fact(world: World, index: int) -> InstanceFact:
    """One validated fact about an episode's starting world, read from the native world.

    Records state the task order's status; the workbook states the duplicated row's order.
    """
    if world.kind == "records":
        key, value = f"orders/#W{1000 + index}/status", "pending" if index % 3 else "delivered"
    else:
        key, value = "input.xlsx/Orders/A2", f"#W{1000 + 10 * index + 1}"
    return InstanceFact(
        key=key,
        value=value,
        visibility="conditioning",
        provenance=FactProvenance(
            adapter=f"acceptance-{world.kind}",
            source="native world before the recorded episode",
            validation=FactValidation(status="validated", method="read from the native world"),
            state_version=0,
        ),
    )


def recorded_conversation(world: World, index: int) -> JsonObject:
    """Execute one episode in its native world and record it as a chat-json conversation.

    Every conversation declares the same complete tool catalogue, as real agent traces do.
    Shared schemas must not collapse independent tasks into one leakage group. Request
    context supplies the starting world's instance facts, as a trace package does.
    """
    episode = world.episodes[index]
    messages: list[JsonValue] = [{"role": "user", "content": episode.instruction}]
    with world.native(index).open(task_of(world, episode)) as session:
        for action in episode.actions:
            reply = session.execute(action)
            assert not reply.is_error, reply.content
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": action.call_id,
                            "type": "function",
                            "function": {
                                "name": action.name,
                                "arguments": json.dumps(action.arguments, sort_keys=True),
                            },
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": action.call_id,
                    "name": action.name,
                    "content": reply.content,
                }
            )
    messages.append({"role": "assistant", "content": "done"})
    return {
        "id": f"{world.kind}-{episode.lineage}",
        "provider": "recording",
        "model": "recorded-agent",
        "tools": [tool.model_dump(mode="json") for tool in world.tools],
        "messages": messages,
        # A declared customer keeps lineage identity, and so the split, independent of the
        # path the export is read from.
        "exp.customer.id": "acceptance-customer",
        "exp.conversation.id": episode.lineage,
        "exp.request.context": {
            "task_lineage": episode.lineage,
            "replay_from": 1,
            "instance_facts": InstanceFacts(facts=(starting_fact(world, index),)).model_dump(
                mode="json"
            ),
        },
    }


def decoded(content: str) -> JsonValue:
    """A recorded reply as replay decodes it: JSON when it parses, the text otherwise."""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return content


def recorded_key(lineage: str, name: str, arguments: JsonValue) -> str:
    """Identity of one recorded call within one episode lineage."""
    return json.dumps([lineage, _call_key(name, arguments)])


class PipelineProvider(SimulatedProvider):
    """Every model wire of one world, answered from its recorded native replies.

    The routed retail fake answers embeddings, native decisions, judges and compressions. This
    subclass adds the world's own tool roles: replies of ``instance`` are always mispredicted,
    replies of ``learnable`` until delivered memory names that tool, and investigations,
    screening decisions and library edits name the learnable tool. Replies are recorded per
    task lineage, because one workbook call reads a different sheet in every episode; a
    derived task's lineage is mapped to the episode it came from. Calls no trace recorded are
    answered with ``unrecorded``.

    Attributes:
        world: The world whose tools are simulated.
        aliases: Derived task lineages mapped to the recorded episode lineage.
        unrecorded: Reply for a call no recorded trace contains.
        effects: State edits an executable simulated world commits for one predicted call.
    """

    def __init__(self, world: World, truth: Mapping[str, JsonValue]) -> None:
        """Index the recorded replies and adopt the world's tool roles."""
        super().__init__([], frozenset({world.instance}), learnable=True)
        self.truth = dict(truth)
        self.world = world
        self.aliases: dict[str, str] = {}
        self.unrecorded: JsonValue = {"error": "no recorded reply"}
        self.effects: Callable[[str, JsonValue, JsonValue], list[JsonValue]] = (
            lambda name, arguments, reply: []
        )

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Answer screening, library edits and investigations; route the rest as before."""
        questions = payload.get("questions")
        if url.endswith("/systemone") and isinstance(questions, dict) and "covered" in questions:
            self.requests.append(RecordedRequest(url, dict(headers), payload))
            behavioural = 0.9 if _miss_tool(payload["state"]) == self.world.learnable else 0.2
            answers = {"behavioural": behavioural, "covered": 0.1, "supported": 0.9}
            return JsonHttpResponse(
                status_code=200,
                body={"answers": {key: {"type": "noul", "noul": p} for key, p in answers.items()}},
            )
        tools = payload.get("tools")
        if not url.endswith("/chat/completions") or not isinstance(tools, list):
            return super().post(
                url, headers=headers, payload=payload, timeout_seconds=timeout_seconds
            )
        name = str(_at(payload, "tool_choice", "function", "name"))
        properties = json.dumps(_at(payload, "tools", 0, "function", "parameters"))
        if '"rationale"' in properties and '"rules"' in properties:
            self.requests.append(RecordedRequest(url, dict(headers), payload))
            training = _user_content(payload)["training"]
            assert isinstance(training, list)
            sources = sorted(
                {
                    str(trace)
                    for row in training
                    for trace in _arr(_at(row, "case", "source_trace_ids"))
                }
            )
            rules = f"{self.world.learnable} returns its documented fields"
            return _tool_reply(
                "world-v1",
                name,
                {"rules": rules, "rationale": "Fixture edit", "source_trace_ids": sources},
            )
        if '"hypothesis"' in properties and name != "submit_transition":
            self.requests.append(RecordedRequest(url, dict(headers), payload))
            learnable = self.world.learnable
            return _tool_reply(
                "world-v1",
                name,
                {
                    "discrepancy": f"{learnable} reply differed from the prediction",
                    "hypothesis": f"{learnable} returns its documented fields",
                    "scope": f"{learnable} reads",
                    "uncertainty": "One observation",
                    "supersedes": [],
                    "transferable": True,
                    "observable_cue": "the same tool and arguments",
                },
            )
        return super().post(url, headers=headers, payload=payload, timeout_seconds=timeout_seconds)

    def _transition(self, payload: JsonObject) -> JsonObject:
        """Propose recorded replies, mispredicting the instance tool and the untaught one."""
        request = _user_content(payload)
        memory = json.dumps(
            [request.get("research_history") or {}, request.get("behavioral_knowledge") or {}]
        )
        taught = self.world.learnable in memory
        # Executable worlds frame simulated domains; fidelity replay frames its own state.
        executable = _is_executable(request)
        observations: list[JsonValue] = []
        edits: list[JsonValue] = []
        for item in _arr(_at(request, "candidate_response", "tool_calls")):
            call = _obj(item)
            tool, arguments = str(call["name"]), call["arguments"]
            lineage = str(_at(request, "task", "initial_context", "task_lineage"))
            key = recorded_key(self.aliases.get(lineage, lineage), tool, arguments)
            reply = self.truth.get(key, self.unrecorded)
            if not executable and (
                tool == self.world.instance or (tool == self.world.learnable and not taught)
            ):
                reply = {"error": "simulated mismatch"}
            if executable:
                edits.extend(self.effects(tool, arguments, reply))
            observations.append({"call_id": call["call_id"], "reply": reply, "encoding": "json"})
        if not executable:
            edits = [{"op": "set", "path": ["calls"], "value": len(observations)}]
        return {"observations": observations, "effect": {"kind": "edits", "edits": edits}}


def _is_executable(request: JsonObject) -> bool:
    """Whether a prediction comes from an executable world over simulated domains."""
    state = request.get("environment_state")
    return isinstance(state, dict) and "domains" in state


def _miss_tool(state: JsonValue) -> str:
    """Tool of the learning miss one screening request frames."""
    action = _at(state, "miss", "action")
    first = _arr(action)[0] if isinstance(action, list) else action
    return str(_at(first, "name"))


def _obj(value: JsonValue) -> JsonObject:
    """Narrow a JSON value to an object."""
    assert isinstance(value, dict)
    return value


def _arr(value: JsonValue) -> list[JsonValue]:
    """Narrow a JSON value to an array."""
    assert isinstance(value, list)
    return value


def flat(text: str) -> str:
    """Collapse terminal wrapping and error-panel borders so assertions read one line."""
    return " ".join(text.replace("│", " ").split())


def exp(*arguments: str) -> str:
    """Run one ``exp`` command, require success and return its flattened output."""
    result = CliRunner().invoke(app, list(arguments))
    assert result.exit_code == 0, result.output
    return flat(result.output)


UNPRICED_SPEND: JsonObject = {"unpriced_attempts": 10_000}
"""Explicit unpriced-attempt ceiling for the fixture's unpriced simulator and decision wire."""


def refused(*arguments: str) -> str:
    """Run one ``exp`` command that must fail and return its flattened output."""
    result = CliRunner().invoke(app, list(arguments))
    assert result.exit_code != 0, result.output
    return flat(result.output)


@dataclass
class Pipeline:
    """One world's project, root and scripted providers, shared by every stage.

    Attributes:
        world: The world kind under test.
        root: The shared ``.exp`` root.
        tmp: Scratch directory for operation files.
        project: The project every stage operates on.
        provider: Simulator, judge, embedder and decision wires.
    """

    world: World
    root: Path
    tmp: Path
    project: ProjectStore
    provider: PipelineProvider

    def eval(self, *flags: str) -> str:
        """Run ``exp eval PROJECT`` with the given flags."""
        return exp("eval", self.project.paths.project_id, "--root", str(self.root), *flags)

    def operation(self, name: str, operation: JsonObject, *flags: str) -> str:
        """Run one ``exp eval --continuations`` operation saved as ``name``.json.

        The fixture catalog prices the embedder but not the simulator or decision wire, so
        every operation explicitly authorizes unpriced attempts beside its dollar ceiling.
        """
        path = self.tmp / f"{name}.json"
        path.write_text(json.dumps({"spend": UNPRICED_SPEND, **operation}))
        return self.eval("--continuations", str(path), "--non-interactive", *flags)

    def emitted(self, path: Path, *flags: str) -> str:
        """Run an operation file another stage emitted, after authorizing its unpriced roles."""
        path.write_text(json.dumps({"spend": UNPRICED_SPEND, **json.loads(path.read_text())}))
        return self.eval("--continuations", str(path), "--non-interactive", *flags)

    def refused(self, name: str, operation: JsonObject) -> str:
        """Run one ``exp eval --continuations`` operation that must fail; return its output."""
        path = self.tmp / f"{name}.json"
        path.write_text(json.dumps({"spend": UNPRICED_SPEND, **operation}))
        project, root = self.project.paths.project_id, str(self.root)
        return refused(
            "eval", project, "--root", root, "--continuations", str(path), "--non-interactive"
        )


def offline_pipeline(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Pipeline, Path]:
    """Record the world's traces as chat-json and route every model wire offline."""
    conversations = [recorded_conversation(world, index) for index in range(EPISODES)]
    truth: dict[str, JsonValue] = {}
    for conversation in conversations:
        calls: dict[str, tuple[str, JsonValue]] = {}
        for message in _arr(conversation["messages"]):
            row = _obj(message)
            for call in _arr(row.get("tool_calls", [])):
                function = _obj(_at(call, "function"))
                calls[str(_at(call, "id"))] = (
                    str(function["name"]),
                    json.loads(str(function["arguments"])),
                )
            if row["role"] == "tool":
                name, arguments = calls[str(row["tool_call_id"])]
                lineage = str(_at(conversation, "exp.request.context", "task_lineage"))
                truth[recorded_key(lineage, name, arguments)] = decoded(str(row["content"]))
    traces = tmp_path / f"{world.kind}.chat.json"
    traces.write_text(json.dumps({"conversations": conversations}))
    root = tmp_path / ".exp"
    root.mkdir()
    write_catalog(root, recall=True)
    provider = PipelineProvider(world, truth)
    monkeypatch.setenv("HELD_OUT_TEST_KEY", "fixture-only")
    monkeypatch.setenv("EXP_RELEASE_REVISION", REVISION)
    monkeypatch.setattr(
        "exp.cli.build.app.RuntimeModelCatalog",
        functools.partial(RuntimeModelCatalog, transport_factory=lambda: provider),
    )
    monkeypatch.setattr("exp.cli.build.app.capture_build_completed", lambda **_kwargs: None)
    for owner in (fidelity, screening, lesson_evolution):
        monkeypatch.setattr(owner, "HttpxJsonTransport", lambda: provider)
    pipeline = Pipeline(
        world=world,
        root=root,
        tmp=tmp_path,
        project=ProjectStore(root, world.kind),
        provider=provider,
    )
    return pipeline, traces


def build_and_audit(pipeline: Pipeline, traces: Path) -> None:
    """Stages 1 and 2: build and learn from chat-json traces, then audit held-out traces."""
    root, project = str(pipeline.root), pipeline.project.paths.project_id
    built = exp(
        "build", project, "-t", str(traces), "--source", "chat-json", "--root", root,
        "--yes", "--non-interactive",
    )  # fmt: skip
    learned = re.search(r"learned (\d+)/(\d+) traces · memory v(\d+)", built)
    assert learned is not None, built
    assert learned.group(1) == learned.group(2) == learned.group(3)
    assert re.search(r"split \d+ fit / [1-9]\d* held out", built), built
    assert "Report:" in built
    selected = pipeline.project.load_project().build
    assert selected is not None
    tasks = load_task_set(pipeline.project.artifacts, selected.task_set.artifact_id).tasks
    expected_tools = tuple(sorted(pipeline.world.tools, key=lambda tool: tool.name))
    assert tasks and all(task.tools == expected_tools for task in tasks)

    audit = ("--audit", str(AUDITED), "--yes", "--non-interactive")
    audited = pipeline.eval(*audit)
    assert "Held-out fidelity" in audited, audited
    assert "exact by recorded reply:" in audited and "trivial" in audited
    assert re.search(
        rf"per-lineage mean exact \d+% \(95% interval \d+% to \d+%, {AUDITED} lineages\)", audited
    )
    assert f"({AUDITED}/{AUDITED} traces judged, 0 judge failures)" in audited
    fit = re.search(r"learned (\d+)/\1 fit traces · memory v(\d+)", audited)
    assert fit is not None and fit.group(1) == fit.group(2), audited
    calls = len(pipeline.provider.requests)
    replay = pipeline.eval(*audit)
    assert "no new model calls (saved results)" in replay, replay
    assert len(pipeline.provider.requests) == calls, "the saved audit replays without calls"


# Operator-selected runtime for the record world: the native service and its private probe.

RECORD_PROBE = "acceptance-order-status-v1"
RECORD_RUNTIME = '''"""Operator-selected native retail record runtime for acceptance tests."""

from exp.cli.tests.acceptance_pipeline_test import record_runtime as continuation_runtime

__all__ = ["continuation_runtime"]
'''


def record_probe_sha256(order: str) -> str:
    """Identity of the record probe that reads one order's status beside all others."""
    return sha256_json({"probe": RECORD_PROBE, "order": order})


def record_runtime(project: ProjectStore, configuration: JsonObject) -> ContinuationRuntime:
    """Restore the native record service with an evaluator-only order-status projection."""
    del project
    order = str(configuration["order_id"])
    world = record_world()

    def probe(checkpoint: WorldCheckpoint) -> JsonObject:
        """The task order's status, and every other order's, read from native SQLite."""
        assert isinstance(checkpoint, ExecutableCheckpoint)
        orders = _obj(_obj(record_state(world, checkpoint)["tables"])["orders"])
        statuses: JsonObject = {str(_arr(row)[0]): _arr(row)[2] for row in _arr(orders["rows"])}
        return {"status": statuses.pop(order), "others": statuses}

    return ContinuationRuntime(world, probe, record_probe_sha256(order))


@dataclass(frozen=True)
class Native:
    """The native side of one world: its runtime file, configuration and private probe.

    Attributes:
        runtime_file: Trusted runtime source the operator selected.
        configuration: Frozen factory configuration.
        probe_sha256: Identity of the runtime's private projection.
        sources: Evidence the capture binds.
        outputs: Native files returned after a Harbor submission.
    """

    runtime_file: Path
    configuration: JsonObject
    probe_sha256: str
    sources: tuple[ArtifactInput, ...]
    outputs: tuple[str, ...] = ()


@dataclass(frozen=True)
class Simulated:
    """The simulated side of one world: owned domains, their state and their projection.

    Attributes:
        manifest: Simulated ownership of the same tools.
        initial_state: Starting value of every simulated domain.
        probe_paths: Projection names mapped to pointers into simulated state.
    """

    manifest: WorldManifest
    initial_state: JsonObject
    probe_paths: dict[str, str]


@dataclass(frozen=True)
class Contract:
    """What the task asks for, as each consumer checks it.

    Attributes:
        plans: Actor plans; the first passes, later ones vary the route to the same goal.
        missing: A call sequence that misses the goal, for the Harbor trial.
        paired: Material paths compared across native and simulated worlds.
        verifier: Development verifier over the native projection.
        acceptance: Independent specification the qualification witnesses grade.
        controls: Edits of a passing outcome into a goal-miss or side-effect control, per
            control kind and partition.
    """

    plans: tuple[tuple[ToolCall, ...], ...]
    missing: tuple[ToolCall, ...]
    paired: tuple[str, ...]
    verifier: TaskVerifier
    acceptance: TaskVerifier
    controls: Callable[[str, str], list[JsonValue]]


def episode_index(world: World, task: TaskCase) -> int:
    """The recorded episode a build task was mined from, by its declared task lineage."""
    lineage = task.initial_context["task_lineage"]
    return next(index for index, row in enumerate(world.episodes) if row.lineage == lineage)


def development_task(pipeline: Pipeline) -> tuple[TaskCase, int, ArtifactInput]:
    """Choose a fit task of the build through ``development-tasks``, keeping its lineage.

    The action lists only fit tasks and refuses a held-out one by ID. The recorded
    conversations declared no tools, so the chosen task publishes the world's tool schemas.
    Records need an order that is still pending, so its cancellation changes native state.

    Returns:
        The development task, the recorded episode it was mined from, and the build's trace
        dataset, which the native capture and the specification cite as their evidence.
    """
    world = pipeline.world
    listed = json_tail(
        pipeline.operation("tasks", {"action": "development-tasks", "development_tasks": {}})
    )
    tasks = [TaskCase.model_validate(row) for row in _arr(listed["tasks"])]
    assert tasks and {task.partition for task in tasks} == {"fit"}, listed
    # The store is read only to name a held-out task the action must refuse.
    build = pipeline.project.load_project().build
    assert build is not None
    held_out = next(
        task.task_id
        for task in load_task_set(pipeline.project.artifacts, build.task_set.artifact_id).tasks
        if task.partition == "held_out"
    )
    query: JsonObject = {"task_id": held_out}
    reserved = pipeline.refused(
        "held-out-task", {"action": "development-tasks", "development_tasks": query}
    )
    assert f"build task {held_out} is held out" in reserved, reserved
    for task in tasks:
        index = episode_index(world, task)
        if world.kind != "records" or index % 3:
            query = {
                "task_id": task.task_id,
                "tools": [tool.model_dump(mode="json") for tool in world.tools],
            }
            chosen = json_tail(
                pipeline.operation(
                    "task", {"action": "development-tasks", "development_tasks": query}
                )
            )
            (published,) = _arr(chosen["tasks"])
            evidence = ArtifactInput.model_validate(_at(chosen, "build", "trace_dataset"))
            return TaskCase.model_validate(published), index, evidence
    raise AssertionError("the build kept no usable fit task")


def native_side(pipeline: Pipeline, index: int, evidence: ArtifactInput) -> Native:
    """Publish the native world's inputs through the CLI and bind its runtime."""
    if pipeline.world.kind == "records":
        runtime = pipeline.tmp / "record_runtime.py"
        runtime.write_text(RECORD_RUNTIME)
        order = f"#W{1000 + index}"
        return Native(runtime, {"order_id": order}, record_probe_sha256(order), (evidence,))
    command = ExecutableCommand(action="construct-assets", construction=orders_construction(index))
    published = pipeline.operation(
        "construct", {"action": "harbor", "harbor": command.model_dump(mode="json")}
    )
    assets = ArtifactInput.model_validate(
        json.loads(published[published.index("{") :])["reference"]
    )
    definition = WorkbookRuntimeDefinition(
        input_artifact=assets,
        files={"input.xlsx": "assets/orders.xlsx"},
        output_paths=("output.xlsx",),
        manifest=WORKBOOK_MANIFEST,
        views={
            "audit": WorkbookStateView(
                input_file="input.xlsx",
                output_file="output.xlsx",
                sheets=("Audit",),
                baseline_until_written=True,
            ),
            "orders": WorkbookStateView(
                input_file="input.xlsx",
                output_file="output.xlsx",
                sheets=("Orders",),
                include_rows=True,
            ),
        },
    )
    return Native(
        Path(workbook_runtime.__file__),
        definition.model_dump(mode="json"),
        definition.probe_sha256,
        (assets,),
        ("output.xlsx",),
    )


def simulated_side(world: World, index: int) -> Simulated:
    """Declare the simulated domains and their projection; facts come from the trace."""
    simulated = world.manifest.model_copy(
        update={
            "domains": tuple(
                row.model_copy(update={"owner": "simulated"}) for row in world.manifest.domains
            )
        }
    )
    if world.kind == "records":
        order = f"#W{1000 + index}"
        return Simulated(
            manifest=simulated,
            initial_state={"users": {}, "orders": {order: {"status": "pending"}}},
            probe_paths={"status": f"/orders/{order}/status"},
        )
    sheets = workbook_projection(read_supported_workbook(orders_workbook(index)))["sheets"]
    audit = next(row for row in _arr(sheets) if _obj(row)["name"] == "Audit")
    return Simulated(
        manifest=simulated,
        initial_state={
            "input.xlsx": {"views": {"audit": {"sha256": sha256_json([audit])}}},
            "output.xlsx": {},
        },
        probe_paths={"views": "/input.xlsx/views"},
    )


def contract(world: World, index: int) -> Contract:
    """The task contract: actor plans, paired paths, verifier, specification and controls."""
    episode = world.episodes[index]
    if world.kind == "records":
        order = f"#W{1000 + index}"
        lookup, _, profile, cancel = episode.actions[0:4]
        rules = (OutcomeRule(path="/final_state/status", operator="equals", value="cancelled"),)

        others = sorted(f"#W{1000 + row}" for row in range(EPISODES) if row != index)

        def record_controls(kind: str, partition: str) -> list[JsonValue]:
            """Leave the order pending, or cancel another customer's order too."""
            if kind == "goal-miss":
                status = "pending " + partition
                return [{"op": "set", "path": ["final_state", "status"], "value": status}]
            other = others[0 if partition == "development" else 1]
            path: list[JsonValue] = ["final_state", "others", other]
            return [{"op": "set", "path": path, "value": "cancelled"}]

        return Contract(
            plans=(
                (episode.actions[1], cancel),
                (lookup, profile, cancel),
            ),
            missing=(episode.actions[1],),
            paired=("/status",),
            verifier=TaskVerifier(
                description=f"Order {order} is cancelled; no other order changes.",
                outcome_schema={},
                rules=rules,
                allowed_tools=tuple(tool.name for tool in world.tools),
                preserved_paths=("/others",),
            ),
            acceptance=TaskVerifier(
                description=f"Independent specification: {order} cancelled, others unchanged.",
                outcome_schema={"type": "object"},
                required_paths=("/final_state/status",),
                rules=rules,
                preserved_paths=("/others",),
                allowed_tools=tuple(tool.name for tool in world.tools),
            ),
            controls=record_controls,
        )
    kept = f"#W{1000 + 10 * index + 2}"
    goal = "/final_state/views/orders/sheets/0/rows/1/0/value"
    rules = (OutcomeRule(path=goal, operator="equals", value=kept),)
    _, describe, row, edit = episode.actions

    def workbook_controls(kind: str, partition: str) -> list[JsonValue]:
        """Delete the wrong row, or change the protected audit sheet."""
        if kind == "goal-miss":
            cell: list[JsonValue] = ["final_state", "views", "orders", "sheets", 0, "rows", 1, 0]
            return [{"op": "set", "path": [*cell, "value"], "value": "#W-wrong-" + partition}]
        digest = "0" * 63 + ("1" if partition == "development" else "2")
        path: list[JsonValue] = ["final_state", "views", "audit", "sha256"]
        return [{"op": "set", "path": path, "value": digest}]

    return Contract(
        plans=((row, edit), (describe, edit)),
        missing=(
            edit.model_copy(
                update={
                    "call_id": "wrong-row",
                    "arguments": {
                        **edit.arguments,
                        "deletions": [{"sheet": "Orders", "start": 3, "count": 1}],
                    },
                }
            ),
        ),
        paired=("/views/audit/sha256",),
        verifier=TaskVerifier(
            description="The duplicated row is deleted into output.xlsx; Audit is preserved.",
            outcome_schema={},
            rules=rules,
            allowed_tools=tuple(tool.name for tool in world.tools),
            preserved_paths=("/views/audit/sha256",),
        ),
        acceptance=TaskVerifier(
            description="Independent specification: the second order row is gone, Audit kept.",
            outcome_schema={"type": "object"},
            required_paths=(goal,),
            rules=rules,
            preserved_paths=("/views/audit/sha256",),
            allowed_tools=tuple(tool.name for tool in world.tools),
        ),
        controls=workbook_controls,
    )


class ScriptedActor(ScriptedJsonTransport):
    """An actor that follows one plan per episode, then answers ``done``.

    Each request that opens an episode (no assistant turn yet) takes the next plan in turn,
    so repeated branches of one study follow different routes to the same goal.

    Attributes:
        plans: Tool-call plans, used in rotation.
        episodes: Episodes opened so far.
        plan: The plan of the current episode; a resumed episode continues it.
    """

    def __init__(self, plans: Sequence[Sequence[ToolCall]]) -> None:
        """Rotate through the given plans."""
        super().__init__()
        self.plans = [tuple(plan) for plan in plans]
        self.episodes = 0
        self.plan: tuple[ToolCall, ...] = self.plans[0]

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Answer the next planned call, or finish once the plan is spent."""
        del timeout_seconds
        self.requests.append(RecordedRequest(url, dict(headers), payload))
        turns = sum(_obj(row)["role"] == "assistant" for row in _arr(payload["messages"]))
        if turns == 0:
            self.plan = self.plans[self.episodes % len(self.plans)]
            self.episodes += 1
        message: JsonObject = {"role": "assistant", "content": "done"}
        if turns < len(self.plan):
            call = self.plan[turns]
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "id": f"{call.call_id}-{turns}",
                        "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                    }
                ],
            }
        return JsonHttpResponse(
            status_code=200,
            body={
                "id": f"actor-{len(self.requests)}",
                "model": "world-v1",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls" if "tool_calls" in message else "stop",
                        "message": message,
                    }
                ],
            },
        )


class ScriptedReviewer(ScriptedJsonTransport):
    """Pair judges rate every dimension; council analysts return one bounded analysis."""

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Answer one review by the shape of its requested proposal."""
        del timeout_seconds
        self.requests.append(RecordedRequest(url, dict(headers), payload))
        properties = _obj(_at(payload, "tools", 0, "function", "parameters", "properties"))
        proposal: JsonObject = {
            "summary": "Compare committed consequences; labels stay provisional.",
            "findings": [],
            "limitations": ["Scripted reviewer; no live-model equivalence measured."],
        }
        if "grounding" in properties:
            proposal = _rubric_proposal(_user_content(payload))
        elif "summary" not in properties:
            proposal = {}
            for name in properties:
                proposal[name] = {
                    "dimension_id": name,
                    "status": "assessed",
                    "score": 5,
                    "rationale": "Fixture rating, not calibrated evidence.",
                    "evidence_refs": ["A:task_checks", "B:task_checks"],
                    "critical_flags": [],
                }
        return _tool_reply(
            "judge-v1", str(_at(payload, "tool_choice", "function", "name")), proposal
        )


def _rubric_proposal(request: JsonObject) -> JsonObject:
    """Propose a rubric that cites an exact excerpt of the first supplied public field."""
    allowed = _obj(request["allowed_evidence_refs"])
    reference, text = next((key, str(value)) for key, value in allowed.items() if value)
    dimensions = RUBRIC_DIMENSIONS
    return {
        "rubric": {
            "definition": {
                "name": "Order handling",
                "syllabus": "Compare how both executions read state and commit the request.",
                "dimensions": [
                    {
                        "dimension_id": name,
                        "name": name,
                        "description": f"Paired {name} relationships observed in development.",
                        "min_score": 1,
                        "max_score": 5,
                        "anchors": [
                            {"score": score, "description": f"Paired {name} level {score}"}
                            for score in range(1, 6)
                        ],
                    }
                    for name in dimensions
                ],
            },
            "state_coverage": request["state_coverage"],
            "criteria": [
                {
                    "dimension_id": name,
                    "valid_variations": ["Instance identifiers may differ."],
                    "unassessable_when": "The relevant effect is not observed.",
                }
                for name in dimensions
            ],
        },
        "grounding": [
            {"dimension_id": name, "citations": [{"reference": reference, "quote": text[:40]}]}
            for name in dimensions
        ],
    }


def json_tail(output: str) -> JsonObject:
    """The JSON object a continuation operation printed after its progress lines."""
    return _obj(json.loads(output[output.index("{") : output.rindex("}") + 1]))


@dataclass(frozen=True)
class Development:
    """The development task, both captured worlds and the jobs every later stage reuses.

    Attributes:
        task: Development task derived from a fit task of the build.
        index: Recorded episode the task came from.
        native: Native runtime binding.
        contract: The task's plans, verifier and controls.
        checkpoint: Native capture at the actor's initial turn.
        catalog_sha256: Frozen model catalog.
        evidence: The build's trace dataset, which the task's contract derives from.
    """

    task: TaskCase
    index: int
    native: Native
    contract: Contract
    checkpoint: ArtifactInput
    catalog_sha256: str
    evidence: ArtifactInput


def capture(pipeline: Pipeline, name: str, definition: CaptureDefinition) -> ArtifactInput:
    """``capture`` an initial world and actor context, without any model call."""
    calls = len(pipeline.provider.requests)
    operation = {"action": "capture", "capture": definition.model_dump(mode="json")}
    output = pipeline.operation(name, operation, "--yes")
    assert len(pipeline.provider.requests) == calls
    return ArtifactInput.model_validate(json_tail(output))


def job(
    pipeline: Pipeline,
    development: Development,
    *,
    checkpoint: ArtifactInput,
    runtime_file: Path,
    configuration: JsonObject,
    probe_sha256: str,
    verifier: TaskVerifier,
    repeats: int,
    specification: ArtifactInput | None = None,
) -> ContinuationJob:
    """One frozen continuation job under the shared actor policy.

    The study pins the admission policy registered in stage 3a, so its later export can prove
    the policy was fixed before the study ran.
    """
    catalog = load_model_catalog(pipeline.project.model_catalog_path)
    model = RuntimeModelCatalog(catalog, environment={}).snapshot("world")[0]
    policy = artifact_input(
        pipeline.project.artifacts.read(policy_identity(fixture_policy())).manifest
    )
    return ContinuationJob(
        study=ContinuationStudySpec(
            checkpoint=checkpoint,
            actor_model=model,
            verifier=verifier,
            probe_sha256=probe_sha256,
            repeats=repeats,
            selector="acceptance development task",
            training_specification=specification,
            admission_policy=policy,
        ),
        runtime_file=runtime_file,
        runtime_sha256=sha256_bytes(runtime_file.read_bytes()),
        configuration=configuration,
        actor_alias="world",
        catalog_sha256=development.catalog_sha256,
        maximum_model_calls=ACTOR_CALLS,
        maximum_physical_attempts=8,
        deadline=DEADLINE,
    )


def register_policy(pipeline: Pipeline) -> ArtifactInput:
    """Stage 3a: register the admission policy before any comparison or trajectory exists."""
    policy = fixture_policy()
    operation = {
        "action": "register-admission-policy",
        "admission_policy": policy.model_dump(mode="json"),
    }
    assert "Dry run: not registered" in pipeline.operation("policy", operation, "--dry-run")
    output = pipeline.operation("policy", operation)
    reference = artifact_input(pipeline.project.artifacts.read(policy_identity(policy)).manifest)
    assert reference.artifact_id in output, output
    return reference


def freeze_rubric(pipeline: Pipeline, reviewer: ReviewModel, catalog_sha256: str) -> ArtifactInput:
    """Stage 3b: derive the judge's rubric from a development trajectory, review and freeze it.

    Registering the trajectory and classifying it are operator Python calls; derivation and the
    explicit review run through the CLI, and the frozen draft replays without a model call.
    """
    project = pipeline.project
    task = TaskCase(
        task_id="rubric-development",
        lineage_group_id="rubric-development-family",
        partition="fit",
        workload_weight=1.0,
        origin="authored",
        instruction="Look up the order and report its status",
    )
    source = register_rubric_trajectory(
        project,
        RubricTrajectory(
            task=task,
            trajectory=WorkflowEvidence(
                workflow_id="rubric-development",
                task=task.instruction,
                initial_state={},
                steps=(
                    WorkflowStep(
                        step=0,
                        action={"name": "get_order_details", "arguments": {"order_id": "#W1"}},
                        status="valid",
                        reply="order #W1 is pending",
                    ),
                ),
                final_answer="Order #W1 is pending",
                stop_reason="completed",
            ),
        ),
        inputs=(),
        code_revision=REVISION,
    )
    declare_source_access(
        project.artifacts,
        source,
        scope="development",
        rationale="Authored development trajectory outside the evaluated family",
        code_revision=REVISION,
    )
    derivation = RubricDerivationSpec(
        sources=(source,),
        proposer=reviewer,
        catalog_sha256=catalog_sha256,
        state_coverage="partial",
        deadline=DEADLINE,
    )
    operation: JsonObject = {
        "action": "derive-equivalence-rubric",
        "rubric_derivation": derivation.model_dump(mode="json"),
    }
    assert "no calls or publication" in pipeline.operation("rubric", operation, "--dry-run")
    draft = ArtifactInput.model_validate(
        json_tail(pipeline.operation("rubric", operation, "--yes"))["draft"]
    )
    assert json_tail(pipeline.operation("rubric", operation))["draft"] == draft.model_dump(
        mode="json"
    ), "a saved draft replays without consent or a model call"
    review: JsonObject = {
        "action": "freeze-equivalence-rubric",
        "rubric_review": {
            "draft": draft.model_dump(mode="json"),
            "reviewer": "acceptance author",
            "rationale": "Citations and paired anchors reviewed; calibration remains pending.",
        },
    }
    assert "no publication" in pipeline.operation("rubric-review", review, "--dry-run")
    return ArtifactInput.model_validate(
        json_tail(pipeline.operation("rubric-review", review))["rubric"]
    )


def compare_environments(
    pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch, policy: ArtifactInput
) -> tuple[Development, ArtifactInput]:
    """Stage 3: pair the native world with the built simulator on one captured fit task."""
    world, project = pipeline.world, pipeline.project
    task, index, evidence = development_task(pipeline)
    pipeline.provider.aliases[task.lineage_group_id] = world.episodes[index].lineage
    native = native_side(pipeline, index, evidence)
    terms = contract(world, index)
    catalog = load_model_catalog(project.model_catalog_path)
    development = Development(
        task=task,
        index=index,
        native=native,
        contract=terms,
        checkpoint=capture(
            pipeline,
            "capture-native",
            CaptureDefinition(
                task=task,
                sources=native.sources,
                runtime_file=native.runtime_file,
                runtime_sha256=sha256_bytes(native.runtime_file.read_bytes()),
                configuration=native.configuration,
                probe_sha256=native.probe_sha256,
                maximum_model_calls=ACTOR_CALLS,
            ),
        ),
        catalog_sha256=sha256_json(catalog.model_dump(mode="json")),
        evidence=evidence,
    )
    # The build's simulator, memory and catalog are grounded through the CLI; the starting
    # world's facts come from the task's own source trace.
    simulated = simulated_side(world, index)
    grounding: JsonObject = {
        "task_id": task.task_id,
        "tools": [tool.model_dump(mode="json") for tool in world.tools],
        "manifest": simulated.manifest.model_dump(mode="json"),
        "initial_state": simulated.initial_state,
        "probe_paths": simulated.probe_paths,
        "trace_facts": True,
        "maximum_model_calls": ACTOR_CALLS,
        "maximum_physical_attempts": 64,
        "deadline": DEADLINE.isoformat(),
        "operation_file": str(pipeline.tmp / "capture-simulated.json"),
    }
    calls = len(pipeline.provider.requests)
    grounded = json_tail(
        pipeline.operation("ground", {"action": "ground-runtime", "grounded": grounding})
    )
    assert len(pipeline.provider.requests) == calls, "grounding makes no model call"
    simulated_capture = CaptureDefinition.model_validate(grounded["capture"])
    assert simulated_capture.task == task
    assert json.loads((pipeline.tmp / "capture-simulated.json").read_text()) == grounded
    configuration = simulated_capture.configuration
    assert _at(configuration, "memory", "version"), "the build's learned memory is pinned"
    expected_fact = starting_fact(world, index)
    assert _at(configuration, "facts", "facts", "facts", 0, "key") == expected_fact.key
    candidate_checkpoint = ArtifactInput.model_validate(
        json_tail(pipeline.emitted(pipeline.tmp / "capture-simulated.json", "--yes"))
    )
    assert len(pipeline.provider.requests) == calls, "capture makes no model call"
    if world.kind == "records":
        order = f"#W{1000 + index}"

        def cancel(name: str, arguments: JsonValue, reply: JsonValue) -> list[JsonValue]:
            """A predicted cancellation commits the predicted status to simulated state."""
            if name != "cancel_pending_order" or not isinstance(reply, dict):
                return []
            return [
                {
                    "op": "set",
                    "path": ["domains", "orders", order, "status"],
                    "value": reply["status"],
                }
            ]

        pipeline.provider.effects = cancel
    paired = TaskVerifier(
        description="Material consequence both environments must commit.",
        outcome_schema={},
        rules=(
            OutcomeRule(path="/final_state/status", operator="equals", value="cancelled")
            if world.kind == "records"
            else OutcomeRule(
                path="/final_state/views/audit/sha256",
                operator="equals",
                value=_at(simulated.initial_state, "input.xlsx", "views", "audit", "sha256"),
            ),
        ),
    )
    reference = job(
        pipeline,
        development,
        checkpoint=development.checkpoint,
        runtime_file=native.runtime_file,
        configuration=native.configuration,
        probe_sha256=native.probe_sha256,
        verifier=paired,
        repeats=1,
    )
    candidate = job(
        pipeline,
        development,
        checkpoint=candidate_checkpoint,
        runtime_file=simulated_capture.runtime_file,
        configuration=configuration,
        probe_sha256=simulated_capture.probe_sha256,
        verifier=paired,
        repeats=1,
    )
    model = RuntimeModelCatalog(catalog, environment={}).snapshot("judge")[0]
    reviewer = ReviewModel(alias="judge", model=model, output_tokens=1024)
    reviews = ScriptedReviewer()
    monkeypatch.setattr(
        "exp.optimize.evaluation.equivalence.review.HttpxJsonTransport", lambda: reviews
    )
    rubric = freeze_rubric(pipeline, reviewer, development.catalog_sha256)
    assert len(reviews.requests) == 1, "one proposal; the replay and the freeze make no call"
    spec = EquivalenceSpec(
        rubric=rubric,
        cases=(
            EquivalenceCase(
                reference=reference,
                candidate=candidate,
                initial_state_paths=terms.paired,
                final_state_paths=terms.paired,
            ),
        ),
        judge=reviewer,
        analysts=(reviewer,),
        maximum_review_attempts=3,
        deadline=DEADLINE,
        policy=policy,
    )
    actor = ScriptedActor(terms.plans[:1])
    monkeypatch.setattr(
        "exp.optimize.evaluation.continuations.job.HttpxJsonTransport", lambda: actor
    )
    # The operator's runtime file executes as a fresh module, so it imports this transport anew.
    monkeypatch.setattr(transport, "HttpxJsonTransport", lambda: pipeline.provider)
    operation = {"action": "compare-environments", "equivalence": spec.model_dump(mode="json")}
    output = pipeline.operation("equivalence", operation, "--yes")
    report = json_tail(output)
    # Every paid role reserved against the one operation ceiling: priced embeddings settled
    # in dollars, the unpriced simulator and decision wire against the explicit attempt cap.
    receipts = [
        json.loads(path.read_text())
        for path in (project.paths.runtime_directory / "spend").glob("*/receipt.json")
    ]
    spend = next(row for row in receipts if str(row["command"]).endswith("equivalence.json"))
    assert spend["accounted_usd"] <= spend["ceiling_usd"] and spend["in_flight_usd"] == 0
    assert spend["unpriced_attempts"] > 0 and spend["unknown_usd"] == 0, spend
    roles = {row["role"]: row for row in spend["roles"]}
    assert set(roles) == {"embed", "jev", "judge", "world"}, spend
    assert 0 < roles["embed"]["settled_usd"] <= spend["ceiling_usd"], spend
    assert all(roles[role]["unpriced_attempts"] for role in ("jev", "judge", "world")), spend
    metrics = _obj(report["metrics"])
    # Paired evidence: the simulator, conditioned on the instance facts and frozen memory,
    # commits the same material consequence as the native world.
    candidate, control = _obj(metrics["candidate"]), _obj(metrics["reference_control"])
    assert (candidate["planned_pairs"], candidate["material_matches"]) == (1, 1), metrics
    assert candidate["task_result_matrix_reference_then_other"] == {"pass/pass": 1}
    # The noise floor: the reference repeated against itself, and the candidate's shortfall.
    assert (control["planned_pairs"], control["material_matches"]) == (1, 1)
    assert _obj(control["agreement"])["wilson_95"]
    assert _obj(metrics["candidate_minus_control"])["estimate"] == 0.0
    assert set(_obj(report["analyses"])) == {"case-0:judge"}
    # The judge rated under the frozen rubric; its ratings stay separate from admission.
    assert metrics["rubric"] == rubric.model_dump(mode="json")
    assert metrics["rubric_calibration"] == "provisional"
    goal_ratings = _obj(_obj(candidate["rubric_dimensions"])["goal"])
    assert (goal_ratings["assessed"], goal_ratings["not_judged"]) == (1, 0), goal_ratings
    # One task family is too few independent units for any equivalence verdict.
    calibration = _obj(report["calibration"])
    assert calibration["policy"] == policy.artifact_id
    goal = _obj(_obj(calibration["dimensions"])["goal_success"])
    assert goal["status"] == "unassessable" and "underpowered: 1" in str(goal["reason"]), goal
    predictions = [
        _user_content(row.payload)
        for row in pipeline.provider.requests
        if row.url.endswith("/chat/completions")
        and _at(row.payload, "tool_choice", "function", "name") == "submit_transition"
        and _is_executable(_user_content(row.payload))
    ]
    assert predictions and all(
        _at(row, "instance_facts", "facts", 0, "key") == expected_fact.key for row in predictions
    ), "every simulated turn is conditioned on the trace's instance facts"
    assert all(row["research_history"] for row in predictions), "the frozen memory is read"
    calls = (len(actor.requests), len(reviews.requests), len(pipeline.provider.requests))
    assert json_tail(pipeline.operation("equivalence", operation, "--yes")) == report
    assert calls == (len(actor.requests), len(reviews.requests), len(pipeline.provider.requests))
    return development, ArtifactInput.model_validate(report["reference"])


def fork_native_branches(
    pipeline: Pipeline, development: Development, monkeypatch: pytest.MonkeyPatch
) -> tuple[ContinuationJob, ArtifactInput, tuple[ArtifactInput, ...]]:
    """Stage 4: fork native branches from the captured checkpoint, then from a branch's own.

    Returns:
        The frozen job, its saved result, and every study result the stage produced.
    """
    native, terms = development.native, development.contract
    registration: JsonObject = {
        "specification": EvaluationSpecification(
            description=terms.acceptance.description,
            predicate=terms.acceptance,
            subject="outcome",
        ).model_dump(mode="json"),
        # The acceptance contract derives from the recorded traces the task came from, not
        # from the constructed world it is checked against.
        "sources": [development.evidence.model_dump(mode="json")],
    }
    operation: JsonObject = {"action": "register-specification", "specification": registration}
    assert "Nothing saved" in pipeline.operation("specification", operation, "--dry-run")
    registered = json_tail(pipeline.operation("specification", operation))
    specification = ArtifactInput.model_validate(registered["reference"])
    assert json_tail(pipeline.operation("specification", operation)) == registered
    forked = job(
        pipeline,
        development,
        checkpoint=development.checkpoint,
        runtime_file=native.runtime_file,
        configuration=native.configuration,
        probe_sha256=native.probe_sha256,
        verifier=terms.verifier,
        repeats=2,
        specification=specification,
    )
    actor = ScriptedActor(terms.plans)
    monkeypatch.setattr(
        "exp.optimize.evaluation.continuations.job.HttpxJsonTransport", lambda: actor
    )
    operation = {"action": "run", "job": forked.model_dump(mode="json")}
    pipeline.operation("run", operation, "--dry-run")
    assert not actor.requests, "a dry run makes no actor call"
    ran = json_tail(pipeline.operation("run", operation, "--yes"))
    assert (ran["planned"], ran["successes"]) == (2, 2), ran
    first = next(
        ArtifactInput.model_validate(_obj(row)["reference"])
        for row in _arr(ran["checkpoints"])
        if _obj(row)["branch"] == 0
    )
    # A fork of a fork: resume branch 0 after its first tool turn, along its own plan.
    actor.plan = terms.plans[0]
    child = forked.model_copy(
        update={
            "study": forked.study.model_copy(
                update={"checkpoint": first, "repeats": 1, "training_specification": None}
            )
        }
    )
    resumed = json_tail(
        pipeline.operation(
            "run-child", {"action": "run", "job": child.model_dump(mode="json")}, "--yes"
        )
    )
    assert (resumed["planned"], resumed["successes"]) == (1, 1), resumed
    result = ArtifactInput.model_validate(ran["reference"])
    return forked, result, (result, ArtifactInput.model_validate(resumed["reference"]))


def export_and_learn(
    pipeline: Pipeline,
    development: Development,
    generators: Sequence[ArtifactInput],
    policy: ArtifactInput,
) -> None:
    """Stage 5: export the generated trajectories; only native ones become learning evidence."""
    learner = f"{pipeline.world.kind}-learner"
    operation: JsonObject = {
        "action": "export-traces",
        "traces": {
            "generators": [row.model_dump(mode="json") for row in generators],
            "policy": policy.model_dump(mode="json"),
            "calibrations": [generators[0].model_dump(mode="json")],
            "project": learner,
        },
    }
    calls = len(pipeline.provider.requests)
    dry = pipeline.operation("export-traces", operation, "--dry-run")
    counts = re.search(
        r"generated-native (\d+) \(0 with simulated or unproven replies excluded\), "
        r"generated-simulated 1, \d+ identical collapsed, 0 skipped\. No provider calls\.",
        dry,
    )
    assert counts is not None and "Dry run: nothing imported." in dry, dry
    exported = pipeline.operation("export-traces", operation)
    assert counts.group(0) in exported, exported
    admission = re.search(
        rf"Admission policy {policy.artifact_id}: (\d+) admitted, 0 refused, 1 unassessable",
        exported,
    )
    assert admission is not None and int(admission.group(1)) >= int(counts.group(1)), exported
    assert len(pipeline.provider.requests) == calls, "the export opens no runtime or provider"
    native = re.search(rf"exp build {learner} --import-id (import-[0-9a-f]+)", exported)
    simulated = re.search(r"generated-simulated: (import-[0-9a-f]+)", exported)
    assert native is not None and simulated is not None, exported
    traces = read_ingested_traces(pipeline.root, native.group(1)).traces
    assert len(traces) == int(counts.group(1)) >= 2
    lineage = development.task.lineage_group_id
    for trace in traces:
        assert trace.source.identity.kind == "native-world"
        assert trace.conversation_id == lineage, "every fork keeps its source task's lineage"
        provenance = _obj(trace.spans[0].attributes["exp.trace.metadata"])
        assert {_obj(row)["owner"] for row in _arr(provenance["steps"])} == {"native"}
        for occurrence in _arr(provenance["occurrences"]):
            verdict = _obj(_obj(occurrence)["admission"])
            assert (verdict["status"], verdict["simulated"]) == ("admitted", False)
            assert _obj(verdict["policy"])["artifact_id"] == policy.artifact_id
    (labelled,) = read_ingested_traces(pipeline.root, simulated.group(1)).traces
    assert labelled.source.identity.kind == "simulation"
    (occurrence,) = _arr(_obj(labelled.spans[0].attributes["exp.trace.metadata"])["occurrences"])
    withheld = _obj(_obj(occurrence)["admission"])
    assert (withheld["status"], withheld["simulated"]) == ("unassessable", True)
    assert any("underpowered" in str(reason) for reason in _arr(withheld["reasons"])), withheld

    root = str(pipeline.root)
    built = exp(
        "build", learner, "--import-id", native.group(1), "--root", root, "--yes",
        "--non-interactive",
    )  # fmt: skip
    learned = re.search(r"learned (\d+)/\1 traces · memory v(\d+)", built)
    assert learned is not None and learned.group(1) == learned.group(2) == str(len(traces)), built
    store = ProjectStore(pipeline.root, learner)
    head = memory_head(store.artifacts, learning_plan(store, scope="all").lineage)
    assert head is not None and head.version == len(traces), "learning advanced the memory"
    build = store.load_project().build
    assert build is not None
    bindings = load_task_set_lineage_bindings(store.artifacts, build.task_set.artifact_id).bindings
    assert len({row.lineage_id for row in bindings}) == 1, "the forks share one leakage lineage"
    selected = store.load_project()
    rejected = refused(
        "build", learner, "--import-id", simulated.group(1), "--root", root, "--yes",
        "--non-interactive",
    )  # fmt: skip
    assert "carry simulation provenance" in rejected, rejected
    assert store.load_project() == selected, "the refused import selects nothing"


def screen_lessons(pipeline: Pipeline) -> None:
    """Stage 6: screen learning misses into lessons and audit the three memory profiles."""
    operation: JsonObject = {
        "action": "repair-learning",
        "learning_repair": {"resolution": "screen-lessons"},
    }
    calls = len(pipeline.provider.requests)
    preview = pipeline.operation("screen", operation, "--dry-run")
    misses = re.search(r"(\d+) misses, \1 new decisions", preview)
    assert misses is not None and len(pipeline.provider.requests) == calls, preview
    screened = json_tail(pipeline.operation("screen", operation, "--yes"))
    assert len(pipeline.provider.requests) == calls + int(misses.group(1)), "one call per miss"
    assert (screened["screened_in"], screened["unknown"]) == (1, 0), screened
    store, world = pipeline.project, pipeline.world
    plan, head = screening_plan(store, "fit")
    reference = latest_lesson_set(store, plan.lineage, head.version)
    assert reference == ArtifactInput.model_validate(screened["lesson_set"])
    lessons = load_lesson_set(store, reference)
    assert [row.miss.tool for row in lessons.screened_in] == [world.learnable]
    assert {row.miss.tool for row in lessons.screened_out} == {world.instance}

    def specs(profile: MemoryProfile) -> list[JsonObject]:
        """Planned audit specs of one memory profile, without memory and its call ceiling."""
        planned = plan_held_out_panel(
            store,
            AUDITED,
            profile="memory",
            code_revision=REVISION,
            clock=lambda: store.artifacts.read(reference.artifact_id).manifest.created_at,
            memory_profile=profile,
        )
        excluded = {"run_id", "deadline", "memory", "maximum_physical_attempts"}
        return [row.model_dump(mode="json", exclude=excluded) for row in planned.specs]

    # Only memory differs; without a recall call per turn the attempt ceiling is lower.
    assert specs("learned") == specs("none") == specs("lessons")
    audit = ("--audit", str(AUDITED), "--yes", "--non-interactive")
    lessoned = pipeline.eval(*audit, "--audit-memory", "lessons")
    assert f"replays read only lesson set {reference.artifact_id}" in lessoned, lessoned
    assert "Held-out fidelity" in lessoned, "untouched held-out lineages stay independent"
    none = pipeline.eval(*audit, "--audit-memory", "none", "--include-inspected")
    assert "replays read no memory" in none and "development re-audit" in none, none
    learned = pipeline.eval(*audit, "--include-inspected")
    assert "replays read" not in learned.split("Memory profile")[-1]

    def exact(output: str) -> int:
        """Exact held-out replies an audit reported."""
        found = re.search(r"(\d+)/\d+ replies exact", output.split("Replaying held-out")[-1])
        assert found is not None, output
        return int(found.group(1))

    assert exact(none) < exact(learned), "learned memory improves held-out replies"


def harbor(pipeline: Pipeline, name: str, command: ExecutableCommand) -> str:
    """Run one ``harbor`` continuation action and return its output."""
    operation = {"action": "harbor", "harbor": command.model_dump(mode="json")}
    return pipeline.operation(name, operation, "--yes")


def package_and_trial(
    pipeline: Pipeline,
    development: Development,
    forked: ContinuationJob,
    feasibility: ArtifactInput,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage 7: register the forked native task, qualify, export, verify and serve a trial."""
    project, terms = pipeline.project, development.contract
    scratch = pipeline.tmp / "system-temp"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    definition = ExecutableTask(
        intent=development.task.instruction,
        job=forked,
        output_paths=development.native.outputs,
    )
    task = ArtifactInput.model_validate(
        json_tail(
            harbor(
                pipeline, "register", ExecutableCommand(action="register", definition=definition)
            )
        )
    )
    # Witnessed controls over the feasibility study's own native outcomes: both routes pass,
    # edited copies miss the goal or change protected state, and erased evidence is unknown.
    # Each is graded by the job's independent specification, the same check qualification runs.
    controls: list[JsonValue] = []
    for partition, offset in (("development", 0), ("confirmation", 10)):
        for index, kind in enumerate(
            ("valid-one", "valid-two", "goal-miss", "side-effect", "unknown")
        ):
            marker: list[JsonValue] = ["steps", 0, "action", "arguments", "control"]
            edits: list[JsonValue] = [
                {"op": "set", "path": marker, "value": f"{partition}-{index}"}
            ]
            if kind in ("goal-miss", "side-effect"):
                edits.extend(terms.controls(kind, partition))
            elif kind == "unknown":
                erased = ["steps"] if partition == "development" else ["stop_reason"]
                value = None if partition == "development" else "failure"
                edits.append({"op": "set", "path": erased, "value": value})
            controls.append(
                {
                    "id": f"{partition}-{kind}",
                    "lineage": f"acceptance-control-{offset + index}",
                    "partition": partition,
                    "expected": "pass"
                    if kind.startswith("valid")
                    else "unassessable"
                    if kind == "unknown"
                    else "fail",
                    "branch": 1 if kind == "valid-two" else 0,
                    "edits": edits,
                    "rationale": "Authored control over the feasibility study's native outcome.",
                }
            )
    witnessing: JsonObject = {
        "action": "witness-controls",
        "controls": {
            "task": task.model_dump(mode="json"),
            "feasibility": feasibility.model_dump(mode="json"),
            "controls": controls,
        },
        "operation_file": str(pipeline.tmp / "qualify.json"),
    }
    witness: JsonObject = {"action": "harbor", "harbor": witnessing}
    planned = pipeline.operation("controls", witness, "--dry-run")
    assert "10 control outcomes resolved" in planned, planned
    assert not (pipeline.tmp / "qualify.json").exists(), "a dry run saves no operation"
    # A control whose label the specification contradicts is refused before qualification.
    mislabeled = deepcopy(witnessing)
    _obj(_arr(_obj(mislabeled["controls"])["controls"])[2])["expected"] = "pass"
    del mislabeled["operation_file"]
    wrong = pipeline.refused("mislabeled", {"action": "harbor", "harbor": mislabeled})
    assert "the witness establishes fail, but the label declares pass" in wrong, wrong
    witnessed = json_tail(pipeline.operation("controls", witness))
    assert json.loads((pipeline.tmp / "qualify.json").read_text()) == witnessed
    qualified = json_tail(pipeline.emitted(pipeline.tmp / "qualify.json", "--yes"))
    assert qualified["qualified"] is True, qualified["refusals"]
    package = pipeline.tmp / "packages" / pipeline.world.kind
    exported = harbor(
        pipeline,
        "export",
        ExecutableCommand(
            action="export",
            task=task,
            qualification=ArtifactInput.model_validate(qualified["reference"]),
            directory=package,
            name="experiential/acceptance-" + pipeline.world.kind,
        ),
    )
    assert "Harbor package:" in exported and str(package) in exported.replace(" ", "")
    verified = json_tail(
        harbor(pipeline, "verify", ExecutableCommand(action="verify", directory=package))
    )
    assert ArtifactInput.model_validate(verified["task"]) == task
    assert verified["qualification"] == "qualified_observable_contract"
    assert 'environment_mode = "separate"' in (package / "task.toml").read_text()
    passing = terms.plans[0][-1:]
    rewards, verdict, result, _ = _run_trial(
        project,
        pipeline.tmp,
        monkeypatch,
        package,
        trial_id="pass",
        actions=passing,
        submit=True,
        output_paths=development.native.outputs,
    )
    assert rewards == {"reward": 1, "scored": 1} and verdict["reward"] == 1 and result == verdict
    rewards, verdict, result, _ = _run_trial(
        project,
        pipeline.tmp,
        monkeypatch,
        package,
        trial_id="miss",
        actions=terms.missing,
        submit=True,
        output_paths=development.native.outputs,
    )
    assert rewards == {"reward": 0, "scored": 1} and verdict["reward"] == 0 and result == verdict
    assert not list(scratch.iterdir()), "every trial removed its temporary files"


def relearn_in_place(
    pipeline: Pipeline, generators: Sequence[ArtifactInput], policy: ArtifactInput
) -> None:
    """Export to the default destination, the source project, and rebuild it from the import.

    The command the export prints must run as printed, although the project was first built
    from chat-json traces and the import declares another format.
    """
    project = pipeline.project.paths.project_id
    operation: JsonObject = {
        "action": "export-traces",
        "traces": {
            "generators": [row.model_dump(mode="json") for row in generators],
            "policy": policy.model_dump(mode="json"),
        },
    }
    exported = pipeline.operation("export-in-place", operation)
    printed = re.search(
        rf"Learn from it: (exp build {project} --import-id import-[0-9a-f]+)", exported
    )
    assert printed is not None, exported
    rebuilt = exp(
        *printed.group(1).split()[1:], "--root", str(pipeline.root), "--yes", "--non-interactive"
    )
    assert re.search(r"learned (\d+)/\1 traces · memory v\1", rebuilt), rebuilt
    selected = pipeline.project.load_project()
    assert selected.trace_source == "generated-native"
    assert selected.trace_import_id == printed.group(1).split()[-1]


@pytest.mark.parametrize("kind", sorted(WORLDS))
def test_pipeline_connects_every_stage_through_the_cli(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Build, audit, compare, fork, export, learn, screen and package one world end to end."""
    pipeline, traces = offline_pipeline(WORLDS[kind], tmp_path, monkeypatch)
    build_and_audit(pipeline, traces)
    policy = register_policy(pipeline)
    development, comparison = compare_environments(pipeline, monkeypatch, policy)
    forked, feasibility, studies = fork_native_branches(pipeline, development, monkeypatch)
    export_and_learn(pipeline, development, (comparison, *studies), policy)
    screen_lessons(pipeline)
    package_and_trial(pipeline, development, forked, feasibility, monkeypatch)
    relearn_in_place(pipeline, studies, policy)
