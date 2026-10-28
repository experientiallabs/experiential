"""Source and recovery boundaries of the integrated pipeline, proven through the public CLI.

Every recorded trace carries a marker of its own task lineage in its customer request and in
every recorded tool reply. The pipeline then runs offline through ``exp`` commands with
scripted providers: build, fit learning, the held-out audit, the paired environment
comparison with its rubric, judge and analyst, native forks, trajectory export and learning,
lesson screening, a success critic and transition induction. Every request body each scripted
transport received, and every artifact each stage saved (saved request intents included), is
searched for the held-out lineages' markers:

- ``exp build`` learns the serving memory from every trace, so held-out markers reach it by
  design; the scan proves the markers are detectable.
- Fit learning sends and saves no held-out marker.
- The held-out audit sends each request at most one held-out lineage, the one it replays.
- No development stage (comparison, forks, export, screening, critic, induction) sends or saves
  any held-out marker, while their requests do carry recorded development evidence.

The same run checks provenance: every artifact's manifest inputs resolve to the exact saved
manifests they cite, and the published learning artifacts cite their evidence. Separate tests
drive an unknown critic call and an unknown quality-proxy call through the CLI.
"""

import functools
import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from types import TracebackType

import pytest
from typer.testing import CliRunner

from exp.cli.app import app
from exp.cli.evaluation import held_out_fidelity
from exp.cli.evaluation.continuations import ContinuationCommand
from exp.cli.tests import acceptance_pipeline_test as acceptance
from exp.common.core.artifacts import ArtifactInput, JsonObject, sha256_json
from exp.common.models import ToolCall, load_model_catalog
from exp.common.project import ProjectStore, artifact_input
from exp.common.project.causality import cites
from exp.common.tasks.store import load_task_set
from exp.common.traces.ingest.persistence import read_ingested_traces
from exp.optimize.evaluation.continuations import critic as actor_critic
from exp.optimize.evaluation.continuations.critic import CriticSpec
from exp.optimize.evaluation.continuations.job import ContinuationJob
from exp.optimize.evaluation.induction import service as induction_service
from exp.optimize.evaluation.induction.service import InductionSpec
from exp.optimize.evaluation.learning.lessons import load_lesson_set
from exp.optimize.evaluation.learning.traces import learning_plan
from exp.optimize.evaluation.spending import SpendPolicy
from exp.runtime.models import RuntimeModelCatalog
from exp.runtime.models.providers.transport import (
    JsonHttpResponse,
    ProviderTransportError,
    RecordedRequest,
    ScriptedJsonTransport,
)
from exp.simulation.induction.proposals import InductionSettings
from exp.simulation.world_model.experience_memory import (
    memory_head,
    memory_version_id,
    provenance_gaps,
)

OUT_OF_DOMAIN = "def transition(state, args, facts):\n    raise OutOfDomain('not induced')\n"
DEVELOPMENT_STAGES = ("compare", "fork", "export", "screen", "critic", "induction")


def marker(lineage: str) -> str:
    """The marker planted in the request and every reply of one recorded lineage."""
    return "BOUNDARYMARK" + "".join(ch for ch in lineage if ch.isalnum()) + "END"


def planted(
    original: Callable[[acceptance.World, int], JsonObject], world: acceptance.World, index: int
) -> JsonObject:
    """One recorded conversation with its lineage marker in the request and every reply."""
    conversation = original(world, index)
    tag = marker(world.episodes[index].lineage)
    messages = conversation["messages"]
    assert isinstance(messages, list)
    for message in messages:
        assert isinstance(message, dict)
        if message["role"] == "tool":
            reply = acceptance.decoded(str(message["content"]))
            message["content"] = (
                json.dumps({**reply, "audit_note": tag})
                if isinstance(reply, dict)
                else f"{reply} {tag}"
            )
        elif message["role"] == "user":
            message["content"] = f"{message['content']} (ticket {tag})"
    return conversation


class Recording:
    """Every scripted transport of the run, and what each stage sent and saved.

    Attributes:
        transports: Scripted transports in registration order, each named by its role.
        project: Project whose saved artifacts are scanned.
        sent: Per stage, the (role, request body) pairs its transports received.
        saved: Per stage, the (artifact ID, payload text) pairs it published.
    """

    def __init__(self, project: ProjectStore) -> None:
        """Start with no transport and no stage."""
        self.transports: list[tuple[str, ScriptedJsonTransport]] = []
        self.project = project
        self.sent: dict[str, list[tuple[str, JsonObject]]] = {}
        self.saved: dict[str, list[tuple[str, str]]] = {}

    def add[TransportT: ScriptedJsonTransport](
        self, role: str, transport: TransportT
    ) -> TransportT:
        """Record one more transport under its role."""
        self.transports.append((role, transport))
        return transport

    def stage(self, name: str) -> "Stage":
        """A context recording what one stage sends and saves."""
        return Stage(self, name)

    def requests(self, *names: str) -> Iterator[tuple[str, str, JsonObject]]:
        """Every (stage, role, body) of the named stages."""
        for name in names:
            for role, body in self.sent[name]:
                yield name, role, body

    def artifacts(self, *names: str) -> Iterator[tuple[str, str, str]]:
        """Every (stage, artifact ID, payload text) the named stages saved."""
        for name in names:
            for identity, text in self.saved[name]:
                yield name, identity, text


class Stage:
    """Snapshot request counts and artifact IDs around one stage.

    Attributes:
        recording: Owner of the stage records.
        name: Stage name.
        counts: Requests each transport had received when the stage began.
        ids: Artifacts saved when the stage began.
    """

    def __init__(self, recording: Recording, name: str) -> None:
        """Bind the stage to its recording."""
        self.recording = recording
        self.name = name
        self.counts: dict[int, int] = {}
        self.ids: set[str] = set()

    def __enter__(self) -> None:
        """Remember what had been sent and saved before the stage."""
        self.counts = {id(row): len(row.requests) for _, row in self.recording.transports}
        self.ids = self._ids()

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        """Keep the requests and artifacts the stage added."""
        if kind is not None:
            return
        self.recording.sent[self.name] = [
            (role, row.payload)
            for role, transport in self.recording.transports
            for row in transport.requests[self.counts.get(id(transport), 0) :]
        ]
        store = self.recording.project.artifacts
        self.recording.saved[self.name] = [
            (
                identity,
                " ".join(
                    payload.decode("utf-8", "replace")
                    for payload in store.read(identity).payloads.values()
                ),
            )
            for identity in sorted(self._ids() - self.ids)
        ]

    def _ids(self) -> set[str]:
        """Artifacts the project has saved, none before ``exp build`` creates it."""
        project = self.recording.project
        return set(project.artifacts.list_ids()) if project.exists() else set()


def found(text: str, lineages: set[str]) -> set[str]:
    """The lineages whose marker appears in one serialized body."""
    return {lineage for lineage in lineages if marker(lineage) in text}


def instrument(recording: Recording, monkeypatch: pytest.MonkeyPatch) -> None:
    """Plant markers, and record every reviewer and actor the acceptance stages construct."""
    monkeypatch.setattr(
        acceptance,
        "recorded_conversation",
        functools.partial(planted, acceptance.recorded_conversation),
    )

    class Reviewer(acceptance.ScriptedReviewer):
        """The acceptance pair judge, rubric proposer and analyst, recorded."""

        def __init__(self) -> None:
            """Register this reviewer."""
            super().__init__()
            recording.add("equivalence reviewer", self)

    class Actor(acceptance.ScriptedActor):
        """The acceptance actor, recorded."""

        def __init__(self, plans: Sequence[Sequence[ToolCall]]) -> None:
            """Register this actor."""
            super().__init__(plans)
            recording.add("actor", self)

    monkeypatch.setattr(acceptance, "ScriptedReviewer", Reviewer)
    monkeypatch.setattr(acceptance, "ScriptedActor", Actor)


def held_out_lineages(project: ProjectStore) -> set[str]:
    """Declared task lineages the build reserved as held out."""
    build = project.load_project().build
    assert build is not None
    tasks = load_task_set(project.artifacts, build.task_set.artifact_id).tasks
    return {
        str(task.initial_context["task_lineage"]) for task in tasks if task.partition == "held_out"
    }


class Answering(ScriptedJsonTransport):
    """Answer every request with one fixed forced tool call, recording the body.

    Attributes:
        name: Tool the answer calls.
        arguments: Its arguments.
    """

    def __init__(self, name: str, arguments: JsonObject) -> None:
        """Fix the tool call every request receives."""
        super().__init__()
        self.name = name
        self.arguments = arguments

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Record the request and return the fixed tool call."""
        del timeout_seconds
        self.requests.append(RecordedRequest(url, dict(headers), payload))
        call = {
            "id": "call",
            "type": "function",
            "function": {"name": self.name, "arguments": json.dumps(self.arguments)},
        }
        return JsonHttpResponse(
            status_code=200,
            body={
                "id": f"scripted-{len(self.requests)}",
                "model": "world-v1",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {"role": "assistant", "content": None, "tool_calls": [call]},
                    }
                ],
            },
        )


def critic_spec(job: ContinuationJob) -> CriticSpec:
    """A success estimate for a job's checkpoint under the job's own actor policy."""
    return CriticSpec(
        checkpoint=job.study.checkpoint,
        actor_model=job.study.actor_model,
        verifier_sha256=sha256_json(job.study.verifier),
        probe_sha256=job.study.probe_sha256,
        maximum_model_calls=job.maximum_model_calls,
        system_prompt=job.system_prompt,
        temperature=job.study.temperature,
        actor_output_tokens=job.study.maximum_output_tokens,
        critic_alias=job.actor_alias,
        critic_model=job.study.actor_model,
        catalog_sha256=job.catalog_sha256,
        deadline=job.deadline,
    )


def screen(pipeline: acceptance.Pipeline) -> ArtifactInput:
    """Screen the fit memory's learning misses into a lesson set through ``repair-learning``."""
    operation: JsonObject = {
        "action": "repair-learning",
        "learning_repair": {"resolution": "screen-lessons"},
    }
    screened = acceptance.json_tail(pipeline.operation("screen", operation, "--yes"))
    assert screened["unknown"] == 0, screened
    return ArtifactInput.model_validate(screened["lesson_set"])


def induce(pipeline: acceptance.Pipeline) -> ArtifactInput:
    """Propose one transition program per inducible tool through ``induce-transitions``."""
    catalog = load_model_catalog(pipeline.project.model_catalog_path)
    snapshot, _ = RuntimeModelCatalog(catalog, environment={}).snapshot("world")
    spec = InductionSpec(
        manifest=acceptance.simulated_side(pipeline.world, 0).manifest,
        inducer_alias="world",
        inducer_model=snapshot,
        catalog_sha256=sha256_json(catalog.model_dump(mode="json")),
        settings=InductionSettings(maximum_attempts=1, minimum_support=2),
        maximum_physical_attempts=16,
        deadline=acceptance.DEADLINE,
    )
    operation: JsonObject = {
        "action": "induce-transitions",
        "induction": spec.model_dump(mode="json"),
    }
    induced = acceptance.json_tail(pipeline.operation("induce", operation, "--yes"))
    return ArtifactInput.model_validate(induced["reference"])


def require_citations_verify(project: ProjectStore) -> int:
    """Every saved artifact's inputs resolve to the exact saved manifests they cite.

    Returns:
        How many citations were checked.
    """
    store, checked = project.artifacts, 0
    for identity in store.list_ids():
        for cited in store.read(identity).manifest.inputs:
            assert store.exists(cited.artifact_id), f"{identity} cites missing {cited.artifact_id}"
            assert artifact_input(store.read(cited.artifact_id).manifest) == cited, (
                f"{identity} cites a changed {cited.artifact_id}"
            )
            checked += 1
    return checked


def reference(store: ProjectStore, identity: str) -> ArtifactInput:
    """The exact saved reference of one artifact."""
    return artifact_input(store.artifacts.read(identity).manifest)


@pytest.mark.parametrize("kind", sorted(acceptance.WORLDS))
def test_held_out_evidence_reaches_no_development_request_and_provenance_verifies(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Planted held-out markers stay inside the serving build and their own audit replays."""
    world = acceptance.WORLDS[kind]
    root = tmp_path / ".exp"
    recording = Recording(ProjectStore(root, world.kind))
    instrument(recording, monkeypatch)
    pipeline, traces = acceptance.offline_pipeline(world, tmp_path, monkeypatch)
    recording.add("provider", pipeline.provider)
    project = pipeline.project.paths.project_id

    with recording.stage("build"):
        acceptance.exp(
            "build", project, "-t", str(traces), "--source", "chat-json", "--root", str(root),
            "--yes", "--non-interactive",
        )  # fmt: skip
    held = held_out_lineages(pipeline.project)
    lineages = {episode.lineage for episode in world.episodes}
    assert held and held < lineages

    learning = held_out_fidelity.run_learning

    def staged_learning(*args: object, **kwargs: object) -> object:
        """Record the audit's fit learning as its own stage."""
        with recording.stage("fit learning"):
            return learning(*args, **kwargs)  # ty: ignore[invalid-argument-type]

    monkeypatch.setattr(held_out_fidelity, "run_learning", staged_learning)
    with recording.stage("audit"):
        audited = pipeline.eval("--audit", str(acceptance.AUDITED), "--yes", "--non-interactive")
    assert "Held-out fidelity" in audited, audited
    supplied = f"instance facts: supplied to {acceptance.AUDITED}/{acceptance.AUDITED} traces"
    assert supplied in audited, "exactness measured given supplied facts is labelled"

    policy = acceptance.register_policy(pipeline)
    with recording.stage("compare"):
        development, comparison = acceptance.compare_environments(pipeline, monkeypatch, policy)
    with recording.stage("fork"):
        forked, _, studies = acceptance.fork_native_branches(pipeline, development, monkeypatch)
    with recording.stage("export"):
        acceptance.export_and_learn(pipeline, development, (comparison, *studies), policy)
    with recording.stage("screen"):
        lesson_set = screen(pipeline)
    estimate: JsonObject = {"probability": 0.6, "rationale": "Public evidence only."}
    critic = recording.add("critic", Answering("estimate_success", estimate))
    monkeypatch.setattr(actor_critic, "HttpxJsonTransport", lambda: critic)
    with recording.stage("critic"):
        operation: JsonObject = {
            "action": "predict-success",
            "critic": critic_spec(forked).model_dump(mode="json"),
        }
        predicted = acceptance.json_tail(pipeline.operation("critic", operation, "--yes"))
    assert predicted["predicted_success"] == 0.6
    inducer = recording.add("inducer", Answering("submit_program", {"source": OUT_OF_DOMAIN}))
    monkeypatch.setattr(induction_service, "HttpxJsonTransport", lambda: inducer)
    with recording.stage("induction"):
        induced = induce(pipeline)
    assert inducer.requests, "induction drew at least one program"

    # 1. Protected-source admission, read from the actual request bodies and saved artifacts.
    served = set().union(
        *(found(json.dumps(body), held) for *_, body in recording.requests("build"))
    )
    assert served == held, "the serving build learns every trace by design"
    fit_bodies = {id(body) for *_, body in recording.requests("fit learning")}
    assert fit_bodies, "the audit learned the fit memory"
    for _, role, body in recording.requests("audit"):
        reached = found(json.dumps(body), held)
        if id(body) in fit_bodies:
            assert not reached, f"fit learning ({role}) received held-out {reached}"
        else:
            assert len(reached) <= 1, f"an audit request mixed held-out lineages {reached}"
    for stage, identity, text in recording.artifacts("fit learning", *DEVELOPMENT_STAGES):
        assert not found(text, held), f"{stage} saved held-out evidence in {identity}"
    evidence: dict[str, set[str]] = {stage: set() for stage in DEVELOPMENT_STAGES}
    roles: set[str] = set()
    for stage, role, body in recording.requests(*DEVELOPMENT_STAGES):
        text = json.dumps(body)
        assert not found(text, held), f"{stage} {role} request received held-out evidence"
        evidence[stage] |= found(text, lineages)
        roles.add(role)
    assert {"provider", "actor", "equivalence reviewer", "critic", "inducer"} <= roles
    for stage in ("compare", "screen", "induction"):
        assert evidence[stage], f"{stage} requests carried no recorded evidence to check"
    own = world.episodes[development.index].lineage
    assert own in evidence["critic"], "the critic read the development task's own request"

    # 2. Provenance: every citation verifies, and published artifacts cite their evidence.
    store = pipeline.project
    assert require_citations_verify(store) > 0
    learner = ProjectStore(pipeline.root, f"{kind}-learner")
    assert require_citations_verify(learner) > 0
    lineage = learning_plan(store, scope="fit").lineage
    head = memory_head(store.artifacts, lineage)
    assert head is not None and head.version > 1
    first = reference(store, memory_version_id(lineage, 1))
    assert cites(store.artifacts, memory_version_id(lineage, head.version), first)
    assert not provenance_gaps(store.artifacts, head), "every memory entry resolves its source"
    lessons = load_lesson_set(store, lesson_set)
    assert lessons.screened_in
    for row in lessons.screened_in:
        assert cites(store.artifacts, lesson_set.artifact_id, row.miss.source)
    build = store.load_project().build
    assert build is not None
    assert cites(store.artifacts, induced.artifact_id, build.trace_dataset), (
        "induced programs cite the trace evidence they were fit to"
    )
    imported = learner.load_project().trace_import_id
    assert imported is not None
    for trace in read_ingested_traces(pipeline.root, imported).traces:
        assert trace.source.identity.kind == "native-world"


def test_an_unknown_critic_call_is_never_redrawn_through_the_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lost estimate stays unknown: the rerun refuses instead of buying another draw."""
    from exp.optimize.evaluation.continuations.job_test import job_fixture

    project, job, _ = job_fixture(tmp_path, monkeypatch)
    lost = ScriptedJsonTransport([ProviderTransportError("connection reset after send")] * 8)
    monkeypatch.setattr(actor_critic, "HttpxJsonTransport", lambda: lost)
    request = tmp_path / "critic.json"
    command = ContinuationCommand(
        action="predict-success", critic=critic_spec(job), spend=SpendPolicy(unpriced_attempts=8)
    )
    request.write_text(command.model_dump_json())
    arguments = [
        "eval", project.paths.project_id, "--root", str(project.paths.root),
        "--continuations", str(request), "--yes", "--non-interactive",
    ]  # fmt: skip
    first = CliRunner().invoke(app, arguments)
    assert first.exit_code != 0, first.output
    sent = len(lost.requests)
    keys = {row.headers["Idempotency-Key"] for row in lost.requests}
    assert sent >= 1 and len(keys) == 1, "every attempt of the one estimate carries one key"
    again = CliRunner().invoke(app, arguments)
    assert again.exit_code != 0
    assert "unknown outcome; no implicit redraw" in acceptance.flat(again.output), again.output
    assert len(lost.requests) == sent, "the rerun sends nothing"


def test_an_unknown_quality_proxy_call_blocks_packaging_and_stays_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unknown proxy judgment never approves a task; every candidate stays in the counts."""
    from exp.optimize.evaluation.factory.candidates_test import batch_fixture

    project, original, _ = batch_fixture(tmp_path, monkeypatch)
    del project
    spec = original.model_copy(update={"quality_reviews": ()})
    monkeypatch.setattr(
        "exp.cli.evaluation.continuations.installed_execution_revision", lambda: "test"
    )
    lost = ScriptedJsonTransport([ProviderTransportError("connection reset after send")] * 32)
    monkeypatch.setattr(
        "exp.optimize.evaluation.factory.authoring.HttpxJsonTransport", lambda: lost
    )

    def run(namespace: str) -> JsonObject:
        """Evaluate the batch under one registration namespace through the CLI."""
        request = tmp_path / f"{namespace}.json"
        batch = spec.model_copy(update={"namespace": namespace})
        command = ContinuationCommand(
            action="evaluate-candidates", batch=batch, spend=SpendPolicy(unpriced_attempts=32)
        )
        request.write_text(command.model_dump_json())
        result = CliRunner().invoke(
            app,
            [
                "eval",
                "case",
                "--root",
                str(tmp_path),
                "--continuations",
                str(request),
                "--yes",
                "--non-interactive",
            ],  # fmt: skip
        )
        assert result.exit_code == 0, result.output
        return acceptance.json_tail(result.output)

    counts = run("unknown-proxy")["counts"]
    assert isinstance(counts, dict)
    assert counts["unavailable"] == counts["proposed"] == 2, counts
    assert not counts.get("packaged"), counts
    sent = len(lost.requests)
    assert sent >= 2
    assert run("unknown-proxy-again")["counts"] == counts
    assert len(lost.requests) == sent, "an unknown proxy judgment is never redrawn"
