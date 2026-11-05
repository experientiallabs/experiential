"""Explicit cross-producer preparation resumes preserve original authority and evidence."""

import json
import socket
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta, tzinfo
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactInput, JsonObject, sha256_json
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.factory.author_retries import (
    load_author_retry,
    register_author_execution_window,
    register_author_retry,
)
from exp.optimize.evaluation.spending import operation_directory, operation_spend_ceiling
from exp.optimize.evaluation.variance import bootstrap_plan as preparation
from exp.optimize.evaluation.variance.bootstrap import task_identity
from exp.optimize.evaluation.variance.bootstrap_plan import (
    BootstrapApproval,
    BootstrapDependencies,
    BootstrapPlan,
    BootstrapPreparationReport,
    BootstrapPreparationSpec,
    plan_bootstrap_preparation,
    run_bootstrap_preparation,
)
from exp.optimize.evaluation.variance.bootstrap_resume import (
    BootstrapResumeBinding,
    bootstrap_author_retries,
    bootstrap_execution_spec,
    freeze_bootstrap_resume_inputs,
    load_bootstrap_resume,
    register_bootstrap_resume,
)
from exp.optimize.evaluation.variance.bootstrap_selection import select_bootstrap_families
from exp.optimize.evaluation.variance.bootstrap_selection_test import preparation_fixture
from exp.optimize.evaluation.variance.build import run_variance_build
from exp.optimize.evaluation.variance.build_test import FakeDecisions, _tagger
from exp.optimize.evaluation.variance.evidence_tools_test import _reply
from exp.optimize.evaluation.variance.pipeline_review import family_reviewer
from exp.runtime.models.call_evidence import current_call_intent
from exp.runtime.models.providers.transport import (
    JsonHttpResponse,
    ProviderTransportError,
    RecordedRequest,
    ScriptedJsonTransport,
)
from exp.runtime.models.spend_ceiling import (
    current_spend_ceiling,
    install_spend_ceiling,
    operation_spend_scope,
)
from exp.simulation.mining.environment_families import load_environment_families
from exp.simulation.mining.environment_families_test import answer
from exp.simulation.mining.environment_review import REVIEW_TOOL, CompatibilityReviewer


def _origin(
    project: ProjectStore, spec: BootstrapPreparationSpec, directory: Path
) -> tuple[Path, Path, Path]:
    """Persist exact external consent inputs and authorize their existing operation ledger."""
    plan = plan_bootstrap_preparation(project, spec, code_revision="original")
    approval = BootstrapApproval(
        plan_sha256=plan.digest,
        ceiling_usd=plan.known_usd,
        unpriced_attempts=plan.unpriced_attempts + 10,
    )
    ceiling = operation_spend_ceiling(project, plan.identity)
    ceiling.authorize(
        "preparation",
        ceiling_usd=approval.ceiling_usd,
        unpriced_attempts=approval.unpriced_attempts,
    )
    ceiling.authorize_policy(ceiling_usd=50, unpriced_attempts=100000)
    paths = (directory / "spec.json", directory / "plan.json", directory / "approval.json")
    for path, value in zip(paths, (spec, plan, approval), strict=True):
        path.write_text(value.model_dump_json())
    return paths


def _register(project: ProjectStore, paths: tuple[Path, Path, Path]) -> ArtifactInput:
    """Register a fixed producer repair against exact original consent files."""
    original = freeze_bootstrap_resume_inputs(
        project,
        original_spec_file=paths[0],
        original_plan_file=paths[1],
        original_approval_file=paths[2],
        code_revision="corrected",
    )
    return register_bootstrap_resume(
        project,
        original_inputs=original,
        code_revision="corrected",
        permitted_changes=("cached_tagging_inventory",),
    )


def test_resume_preserves_original_identity_ledger_and_exact_original_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only actual production provenance changes while identity and accounted claims stay put."""
    project, spec, _ = preparation_fixture(tmp_path, monkeypatch)
    paths = _origin(project, spec, tmp_path)
    original = json.loads(paths[1].read_text())
    reference = _register(project, paths)
    binding = load_bootstrap_resume(project, reference, spec=spec, code_revision="corrected")
    resumed = plan_bootstrap_preparation(
        project, spec, code_revision="corrected", resume_binding=reference
    )
    assert resumed.identity == original["identity"]
    assert resumed.build_identity == original["build_identity"]
    assert resumed.code_revision == "corrected" and resumed.resume_binding == reference
    assert binding.original_revision == "original"
    assert binding.ledger_sha256 == sha256_json(
        {
            "ledger_path": str(
                (operation_directory(project, resumed.identity) / "ledger.sqlite").resolve()
            )
        }
    )
    assert (
        json.loads(
            project.artifacts.read_bytes(binding.original_inputs.artifact_id, "inputs.json")
        )["plan"]
        == original
    )
    assert plan_bootstrap_preparation(
        project, spec, code_revision="original"
    ).digest == sha256_json(original)


def test_resume_rejects_settings_producer_and_source_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The repair cannot become authority for a different model, source or producer."""
    project, spec, _ = preparation_fixture(tmp_path, monkeypatch)
    paths = _origin(project, spec, tmp_path)
    reference = _register(project, paths)
    changed = spec.model_copy(update={"task_ids": ("different",)})
    with pytest.raises(ValueError, match="settings"):
        load_bootstrap_resume(project, reference, spec=changed, code_revision="corrected")
    with pytest.raises(ValueError, match="producer"):
        load_bootstrap_resume(project, reference, spec=spec, code_revision="another")
    config = project.load_project()
    assert config.build is not None
    monkeypatch.setattr(project, "load_project", lambda: config.model_copy(update={"build": None}))
    with pytest.raises(ValueError, match="source build"):
        load_bootstrap_resume(project, reference, spec=spec, code_revision="corrected")


def test_resume_rejects_changed_grants_missing_ledger_and_expired_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new cap or lost spend history cannot silently become a fresh resume allowance."""
    project, spec, _ = preparation_fixture(tmp_path, monkeypatch)
    paths = _origin(project, spec, tmp_path)
    reference = _register(project, paths)

    class ExpiredClock(datetime):
        """Observe the original cutoff after it has passed without changing frozen inputs."""

        @classmethod
        def now(cls, tz: tzinfo | None = None) -> datetime:
            """Return a time after every fixture owner deadline."""
            return datetime(2100, 1, 1, tzinfo=UTC)

    with monkeypatch.context() as clock:
        clock.setattr(preparation, "datetime", ExpiredClock)
        with pytest.raises(ValueError, match="deadline"):
            run_bootstrap_preparation(
                project, spec, code_revision="corrected", resume_binding=reference
            )
    binding = load_bootstrap_resume(project, reference, spec=spec, code_revision="corrected")
    ceiling = operation_spend_ceiling(project, binding.original_operation)
    ceiling.authorize_policy(ceiling_usd=51, unpriced_attempts=100000)
    with pytest.raises(ValueError, match="authority"):
        load_bootstrap_resume(project, reference, spec=spec, code_revision="corrected")
    ceiling.path.rename(ceiling.path.with_suffix(".kept"))
    with pytest.raises(ValueError, match="ledger"):
        load_bootstrap_resume(project, reference, spec=spec, code_revision="corrected")


def test_registration_refuses_unapproved_old_inputs_and_arbitrary_operation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Original approval pins exact old JSON; operation identity must recompute from it."""
    project, spec, _ = preparation_fixture(tmp_path, monkeypatch)
    paths = _origin(project, spec, tmp_path)
    original = json.loads(paths[1].read_text())
    paths[1].write_text(json.dumps({**original, "identity": "bootstrap-preparation-other"}))
    with pytest.raises(ValueError, match="approval"):
        _register(project, paths)
    paths[2].write_text(
        json.dumps(
            {
                **json.loads(paths[2].read_text()),
                "plan_sha256": sha256_json(json.loads(paths[1].read_text())),
            }
        )
    )
    with pytest.raises(ValueError, match="identity"):
        _register(project, paths)


def test_resumed_execution_keeps_prior_claims_and_never_adds_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The resumed orchestration installs the original ledger and preserves every prior charge."""
    project, spec, reviewer = preparation_fixture(tmp_path, monkeypatch)
    paths = _origin(project, spec, tmp_path)
    original = json.loads(paths[1].read_text())
    ceiling = operation_spend_ceiling(project, original["identity"])
    reservation = ceiling.reserve(
        alias="unpriced", url="https://fixture.invalid/v1/chat/completions", payload={}
    )
    ceiling.mark_dispatched(reservation, "original-physical-attempt")
    ceiling.retain_unknown(reservation)
    reference = _register(project, paths)
    before = ceiling.summary()

    def run(
        project: ProjectStore,
        settings: BootstrapPreparationSpec,
        plan: BootstrapPlan,
        dependencies: BootstrapDependencies,
        actual_reviewer: CompatibilityReviewer,
        revision: str,
        binding: BootstrapResumeBinding | None,
    ) -> BootstrapPreparationReport:
        """Observe the real operation scope without constructing any provider or child stage."""
        del settings, dependencies, actual_reviewer
        assert binding is not None
        assert revision == "corrected" and binding.original_revision == "original"
        active = current_spend_ceiling()
        assert active is not None and active.path == ceiling.path
        assert active.summary().unpriced_attempts == before.unpriced_attempts == 1
        source = project.load_project().build
        assert source is not None
        return BootstrapPreparationReport(
            identity=plan.identity,
            plan=plan,
            build_report=source.task_set,
            batches=(),
            bindings=(),
            failed_tasks=("pending",),
        )

    monkeypatch.setattr(preparation, "_run", run)
    result = run_bootstrap_preparation(
        project,
        spec,
        code_revision="corrected",
        resume_binding=reference,
        dependencies=preparation.BootstrapDependencies(
            reviewer=reviewer, tagger=_tagger(project, FakeDecisions())
        ),
    )
    assert result.identity == original["identity"]
    assert ceiling.summary() == before


def test_native_watchdog_renewal_changes_only_fresh_operational_owner_deadlines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Original review identities remain frozen while all fresh bootstrap owners get one window."""
    project, spec, reviewer = preparation_fixture(tmp_path, monkeypatch)
    old = datetime(2000, 1, 1, tzinfo=UTC)
    settings = spec.bootstrap
    assert settings.rules is not None and spec.build.reviewer is not None
    spec = spec.model_copy(
        update={
            "build": spec.build.model_copy(
                update={"reviewer": spec.build.reviewer.model_copy(update={"deadline": old})}
            ),
            "bootstrap": settings.model_copy(
                update={
                    "author": settings.author.model_copy(update={"deadline": old}),
                    "runtime": settings.runtime.model_copy(update={"deadline": old}),
                    "job": settings.job.model_copy(update={"deadline": old}),
                    "rules": settings.rules.model_copy(
                        update={
                            "author": settings.rules.author.model_copy(update={"deadline": old}),
                            "base_state": {"deadline": "business state stays exact"},
                        }
                    ),
                }
            ),
        }
    )
    assert spec.build.reviewer is not None
    paths = _origin(project, spec, tmp_path)
    original = freeze_bootstrap_resume_inputs(
        project,
        original_spec_file=paths[0],
        original_plan_file=paths[1],
        original_approval_file=paths[2],
        code_revision="corrected",
    )
    original_plan = json.loads(paths[1].read_text())
    ceiling = operation_spend_ceiling(project, original_plan["identity"])
    renewed = datetime.now(UTC) + timedelta(hours=1)
    with operation_spend_scope():
        install_spend_ceiling(ceiling)
        window = register_author_execution_window(
            project,
            spec.build.reviewer,
            operation=ceiling,
            window_id="preparation-retry",
            execution_deadline=renewed,
            code_revision="corrected",
            sources=(original,),
        )
    reference = register_bootstrap_resume(
        project,
        original_inputs=original,
        code_revision="corrected",
        permitted_changes=("cached_tagging_inventory", "execution_watchdog_renewal"),
        execution_deadline=renewed,
        execution_window=window,
    )
    binding = load_bootstrap_resume(project, reference, spec=spec, code_revision="corrected")
    execution = bootstrap_execution_spec(project, binding, spec)
    assert binding.deadline == old and binding.stop == renewed
    assert execution.build == spec.build
    assert execution.bootstrap.author.deadline == renewed
    assert execution.bootstrap.runtime.deadline == renewed
    assert execution.bootstrap.job.deadline == renewed
    assert execution.bootstrap.rules is not None
    assert execution.bootstrap.rules.author.deadline == renewed
    assert execution.bootstrap.rules.base_state == {"deadline": "business state stays exact"}
    assert execution.bootstrap.author.model_copy(update={"deadline": old}) == spec.bootstrap.author

    def run(
        project: ProjectStore,
        settings: BootstrapPreparationSpec,
        plan: BootstrapPlan,
        dependencies: BootstrapDependencies,
        actual_reviewer: CompatibilityReviewer,
        revision: str,
        binding: BootstrapResumeBinding | None,
    ) -> BootstrapPreparationReport:
        """Observe renewed fresh owners and unchanged review semantics under original authority."""
        del dependencies, revision
        assert settings == execution and settings.build == spec.build
        assert actual_reviewer.configuration_sha256 == sha256_json(spec.build.reviewer)
        active = current_spend_ceiling()
        assert active is not None and active.path == ceiling.path
        source = project.load_project().build
        assert source is not None and binding is not None
        return BootstrapPreparationReport(
            identity=plan.identity,
            plan=plan,
            build_report=source.task_set,
            batches=(),
            bindings=(),
            failed_tasks=("pending",),
        )

    monkeypatch.setattr(preparation, "_run", run)
    result = run_bootstrap_preparation(
        project,
        spec,
        code_revision="corrected",
        resume_binding=reference,
        dependencies=BootstrapDependencies(
            reviewer=replace(reviewer, configuration_sha256=sha256_json(spec.build.reviewer)),
            tagger=_tagger(project, FakeDecisions()),
        ),
    )
    assert result.identity == original_plan["identity"]
    completed = run_variance_build(
        project,
        spec.build.spec,
        code_revision="corrected",
        identity_revision="original",
        reviewer=replace(reviewer, configuration_sha256=sha256_json(spec.build.reviewer)),
        tagger=_tagger(project, FakeDecisions()),
    )
    selected = select_bootstrap_families(
        project,
        completed,
        execution.bootstrap,
        execution.limits,
        task_ids=execution.task_ids,
        simulators=execution.simulators,
    )
    expected = {
        task_identity(row.spec, row.spec.families[0], task, "corrected")
        for row in selected
        for task in row.spec.families[0].tasks
    }
    observed: list[str] = []

    def pending(project: ProjectStore, identity: str) -> None:
        """Observe native pending-task IDs without fabricating saved outcomes."""
        del project
        observed.append(identity)

    monkeypatch.setattr(preparation, "load_outcome", pending)
    plan_bootstrap_preparation(project, spec, code_revision="corrected", resume_binding=reference)
    assert set(observed) == expected
    with operation_spend_scope():
        install_spend_ceiling(ceiling)
        second_window = register_author_execution_window(
            project,
            spec.build.reviewer,
            operation=ceiling,
            window_id="another-preparation-retry",
            execution_deadline=renewed + timedelta(minutes=1),
            code_revision="corrected",
            sources=(original,),
        )
    with pytest.raises(ValueError, match="different resume binding"):
        register_bootstrap_resume(
            project,
            original_inputs=original,
            code_revision="corrected",
            permitted_changes=("cached_tagging_inventory", "execution_watchdog_renewal"),
            execution_deadline=renewed + timedelta(minutes=1),
            execution_window=second_window,
        )
    with pytest.raises(ValueError, match="six hours"):
        register_bootstrap_resume(
            project,
            original_inputs=original,
            code_revision="corrected",
            permitted_changes=("execution_watchdog_renewal",),
            execution_deadline=datetime.now(UTC) + timedelta(hours=7),
            execution_window=window,
        )
    with pytest.raises(ValueError, match="native author execution window"):
        register_bootstrap_resume(
            project,
            original_inputs=original,
            code_revision="corrected",
            permitted_changes=("execution_watchdog_renewal",),
            execution_deadline=renewed,
        )


def test_registration_refuses_a_checkpoint_with_original_bootstrap_task_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The deadline-only fresh-task repair cannot reinterpret an already prepared task schema."""
    project, spec, _ = preparation_fixture(tmp_path, monkeypatch)
    paths = _origin(project, spec, tmp_path)
    plan = json.loads(paths[1].read_text())
    plan["build_saved"] = True
    plan["remaining_tasks"] = ["original-task"]
    paths[1].write_text(json.dumps(plan))
    approval = json.loads(paths[2].read_text())
    approval["plan_sha256"] = sha256_json(plan)
    paths[2].write_text(json.dumps(approval))
    with pytest.raises(ValueError, match="unfinished variance build without task state"):
        _register(project, paths)


def test_resume_anchor_is_idempotent_and_scoped_to_the_original_operation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replays keep one binding while independently approved operations remain separate."""
    project, spec, _ = preparation_fixture(tmp_path, monkeypatch)
    paths = _origin(project, spec, tmp_path)
    first = _register(project, paths)
    assert _register(project, paths) == first
    original = load_bootstrap_resume(project, first, spec=spec, code_revision="corrected")
    before = operation_spend_ceiling(project, original.original_operation).summary()
    other = spec.model_copy(update={"limits": spec.limits.model_copy(update={"maximum_tasks": 1})})
    directory = tmp_path / "other-operation"
    directory.mkdir()
    other_paths = _origin(project, other, directory)
    second = _register(project, other_paths)
    independent = load_bootstrap_resume(project, second, spec=other, code_revision="corrected")
    assert independent.original_operation != original.original_operation
    assert second != first
    assert operation_spend_ceiling(project, original.original_operation).summary() == before


class PartialReviewWire(ScriptedJsonTransport):
    """Serve a real completed review then charged terminal 503s for the next logical draw."""

    def __init__(self) -> None:
        """Start with one original success allowed and no recorded canonical identities."""
        super().__init__()
        self.initial = True
        self.completed: str | None = None
        self.failed: str | None = None
        self.logical_ids: list[str] = []

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject | None,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Answer actual native review requests using source-cited fixture members."""
        del timeout_seconds
        assert payload is not None
        intent = current_call_intent()
        assert intent is not None
        logical = intent.artifact_id.removesuffix("-intent")
        self.logical_ids.append(logical)
        self.requests.append(RecordedRequest(url, dict(headers), payload))
        if self.initial:
            if self.completed is None:
                self.completed = logical
            elif logical != self.completed:
                self.failed = logical
                return JsonHttpResponse(503, {"error": {"message": "fixture overloaded"}})
        messages = payload["messages"]
        assert isinstance(messages, list) and isinstance(messages[1], dict)
        packet = json.loads(str(messages[1]["content"]))
        members = frozenset(row["trace_id"] for row in packet["members"])
        response = _reply([(REVIEW_TOOL, answer("compatible", members))])
        return JsonHttpResponse(response.status_code, {**response.body, "model": payload["model"]})


def test_real_partial_build_reaches_old_review_ids_and_only_its_authorized_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Producer changes retain native scope/draw identities and completed paid reviews."""
    project, spec, _ = preparation_fixture(tmp_path, monkeypatch)
    assert spec.build.reviewer is not None
    reviewer_settings = spec.build.reviewer.model_copy(
        update={
            "model": spec.bootstrap.author.model,
            "maximum_physical_attempts": 3,
            "deadline": datetime.now(UTC) + timedelta(hours=1),
        }
    )
    spec = spec.model_copy(
        update={"build": spec.build.model_copy(update={"reviewer": reviewer_settings})}
    )
    paths = _origin(project, spec, tmp_path)
    original_plan = json.loads(paths[1].read_text())
    operation = operation_spend_ceiling(project, original_plan["identity"])
    wire = PartialReviewWire()
    monkeypatch.setattr(
        "exp.optimize.evaluation.factory.authoring.HttpxJsonTransport", lambda: wire
    )

    def no_socket(_socket: socket.socket, _address: str | tuple[str, int]) -> None:
        """Keep every completion in this proof on the scripted offline wire."""
        raise AssertionError("native partial-build proof must not open a socket")

    monkeypatch.setattr("socket.socket.connect", no_socket)
    old_decisions = FakeDecisions()
    with operation_spend_scope():
        install_spend_ceiling(operation)
        with pytest.raises(ProviderTransportError, match="503"):
            run_variance_build(
                project,
                spec.build.spec,
                code_revision="original",
                tagger=_tagger(project, old_decisions),
                reviewer=family_reviewer(project, spec.build, code_revision="original"),
            )
    assert old_decisions.states and wire.completed is not None and wire.failed is not None
    parent = artifact_input(project.artifacts.read(wire.failed + "-intent").manifest)
    completed = artifact_input(project.artifacts.read(wire.completed + "-response").manifest)
    original = freeze_bootstrap_resume_inputs(
        project,
        original_spec_file=paths[0],
        original_plan_file=paths[1],
        original_approval_file=paths[2],
        code_revision="corrected",
    )
    renewed = datetime.now(UTC) + timedelta(hours=2)
    with operation_spend_scope():
        install_spend_ceiling(operation)
        window = register_author_execution_window(
            project,
            reviewer_settings,
            operation=operation,
            window_id="native-build-retry",
            execution_deadline=renewed,
            code_revision="corrected",
            sources=(original,),
        )
        retry = register_author_retry(
            project,
            parent,
            operation=operation,
            authorization_id="one-native-build-child",
            execution_window=window,
            code_revision="corrected",
        )
    reference = register_bootstrap_resume(
        project,
        original_inputs=original,
        code_revision="corrected",
        permitted_changes=(
            "cached_tagging_inventory",
            "explicit_terminal_503_retry",
            "execution_watchdog_renewal",
        ),
        execution_deadline=renewed,
        execution_window=window,
        author_retries=(retry,),
    )
    binding = load_bootstrap_resume(project, reference, spec=spec, code_revision="corrected")
    grant = load_author_retry(project, retry)
    resumed_decisions = FakeDecisions()
    before = len(wire.logical_ids)
    wire.initial = False
    with operation_spend_scope():
        install_spend_ceiling(operation)
        result = run_variance_build(
            project,
            spec.build.spec,
            code_revision="corrected",
            identity_revision="original",
            tagger=_tagger(project, resumed_decisions),
            reviewer=family_reviewer(
                project,
                spec.build,
                code_revision="corrected",
                execution_window=window,
                author_retries=bootstrap_author_retries(project, binding),
            ),
        )
    fresh = wire.logical_ids[before:]
    assert result.identity == original_plan["build_identity"]
    assert not resumed_decisions.states
    assert fresh and fresh[0] == grant.child_identity
    assert wire.completed not in fresh and wire.failed not in fresh
    assert (
        artifact_input(project.artifacts.read(wire.completed + "-response").manifest) == completed
    )
    assert artifact_input(project.artifacts.read(parent.artifact_id).manifest) == parent
    assert result.environment_families is not None
    families = load_environment_families(project.artifacts, result.environment_families.families)
    reached = {
        identity
        for family in families.families
        for review in family.evidence.reviews
        for identity in review.identities
    }
    assert wire.completed in reached and wire.failed in reached
