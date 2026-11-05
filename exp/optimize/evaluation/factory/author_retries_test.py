"""Explicit author retries retain terminal receipts and bounded operation authority."""

import socket
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from exp.common.core.artifacts import ArtifactInput, sha256_json
from exp.common.models import ModelRequest
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.factory.author_retries import (
    load_author_retry,
    register_author_retry,
    retry_resolution_id,
)
from exp.optimize.evaluation.factory.authoring import author_response
from exp.optimize.evaluation.factory.proposals import TaskProposalSpec, proposal_request
from exp.optimize.evaluation.factory.proposals_test import proposal_fixture
from exp.runtime.models.providers.transport import (
    JsonHttpResponse,
    ProviderTransportError,
    ScriptedJsonTransport,
)
from exp.runtime.models.spend_ceiling import (
    OperationSpendCeiling,
    install_spend_ceiling,
    operation_spend_scope,
)
from exp.runtime.sessions.dispatch_ledger import DispatchLedger


def _no_network(_socket: socket.socket, _address: str | tuple[str, int]) -> None:
    """Fail if an offline fixture tries to open a provider socket."""
    raise AssertionError("author retry fixtures must not open sockets")


def test_explicit_retry_keeps_original_claims_and_replays_without_another_draw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three charged terminal 503s can authorize one child under the same operation."""
    monkeypatch.setattr("socket.socket.connect", _no_network)
    project, settings, success = proposal_fixture(tmp_path, monkeypatch)
    request = proposal_request(project, settings)
    failed = ScriptedJsonTransport(
        tuple(
            JsonHttpResponse(status_code=503, body={"error": {"message": "overloaded"}})
            for _ in range(3)
        )
    )
    monkeypatch.setattr(
        "exp.optimize.evaluation.factory.authoring.HttpxJsonTransport", lambda: failed
    )
    operation = OperationSpendCeiling(tmp_path / "operation.sqlite", tariffs={})
    operation.authorize_policy(ceiling_usd=1.0, unpriced_attempts=20)
    sources = (settings.base, settings.brief)
    with operation_spend_scope():
        install_spend_ceiling(operation)
        with pytest.raises(ProviderTransportError, match="503"):
            author_response(
                project,
                settings,
                request,
                identity="retry-parent",
                sources=sources,
                code_revision="test",
            )
        parent = artifact_input(project.artifacts.read("retry-parent-intent").manifest)
        original = DispatchLedger(
            project.artifacts.project_directory / "retry-parent.sqlite", 3, sha256_json(settings)
        )
        assert original.used() == 3 and operation.summary().unpriced_attempts == 3
        authorization = register_author_retry(
            project,
            parent,
            operation=operation,
            authorization_id="user-retry-once",
            code_revision="test",
        )
        assert (
            register_author_retry(
                project,
                parent,
                operation=operation,
                authorization_id="user-retry-once",
                code_revision="test",
            )
            == authorization
        )
        monkeypatch.setattr(
            "exp.optimize.evaluation.factory.authoring.HttpxJsonTransport", lambda: success
        )
        reference, response = author_response(
            project,
            settings,
            request,
            retry_authorization=authorization,
            identity="retry-parent",
            sources=sources,
            code_revision="test",
        )
    assert reference.artifact_id != "retry-parent-response"
    assert author_response(
        project, settings, request, identity="retry-parent", sources=sources, code_revision="test"
    ) == (reference, response)
    assert len(failed.requests) == 3 and len(success.requests) == 1
    assert failed.requests[0].payload == success.requests[0].payload
    assert original.used() == 3 and operation.summary().unpriced_attempts == 4
    assert artifact_input(project.artifacts.read(parent.artifact_id).manifest) == parent


def _failed_author(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    ambiguous: bool = False,
    cap: int = 20,
    parent_revision: str = "test",
) -> tuple[
    ProjectStore, TaskProposalSpec, ModelRequest, ScriptedJsonTransport, OperationSpendCeiling
]:
    """Retain an exhausted three-attempt author call under an installed finite operation."""
    monkeypatch.setattr("socket.socket.connect", _no_network)
    project, settings, success = proposal_fixture(tmp_path, monkeypatch)
    request = proposal_request(project, settings)
    responses = tuple(
        ProviderTransportError("outcome unknown")
        if ambiguous
        else JsonHttpResponse(status_code=503, body={"error": {"message": "overloaded"}})
        for _ in range(3)
    )
    failed = ScriptedJsonTransport(responses)
    monkeypatch.setattr(
        "exp.optimize.evaluation.factory.authoring.HttpxJsonTransport", lambda: failed
    )
    operation = OperationSpendCeiling(tmp_path / "operation.sqlite", tariffs={})
    operation.authorize_policy(ceiling_usd=1.0, unpriced_attempts=cap)
    install_spend_ceiling(operation)
    with pytest.raises(ProviderTransportError):
        author_response(
            project,
            settings,
            request,
            identity="parent",
            sources=(settings.base, settings.brief),
            code_revision=parent_revision,
        )
    monkeypatch.setattr(
        "exp.optimize.evaluation.factory.authoring.HttpxJsonTransport", lambda: success
    )
    return project, settings, request, success, operation


def _register(project: ProjectStore, operation: OperationSpendCeiling) -> ArtifactInput:
    """Explicitly authorize the fixture parent's one child lifetime."""
    return register_author_retry(
        project,
        artifact_input(project.artifacts.read("parent-intent").manifest),
        operation=operation,
        authorization_id="user-retry",
        code_revision="test",
    )


@pytest.mark.parametrize(
    "change", ["settings", "request", "sources", "operation", "ceilings", "deadline", "reference"]
)
def test_retry_refuses_changed_pins_or_authority_before_child_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    """A retry grant cannot alter original inputs, authority, ceilings, or deadline."""
    with operation_spend_scope():
        project, settings, request, transport, operation = _failed_author(tmp_path, monkeypatch)
        registration = _register(project, operation)
        sources = (settings.base, settings.brief)
        if change == "settings":
            settings = settings.model_copy(update={"output_tokens": settings.output_tokens + 1})
        elif change == "request":
            request = request.model_copy(
                update={"maximum_output_tokens": settings.output_tokens + 1}
            )
        elif change == "sources":
            sources = ()
        elif change == "ceilings":
            operation.authorize_policy(ceiling_usd=1.0, unpriced_attempts=21)
        elif change == "reference":
            registration = registration.model_copy(update={"sha256": "0" * 64})
        with (
            operation_spend_scope()
            if change == "operation"
            else patch(
                "exp.optimize.evaluation.factory.author_retries.datetime", wraps=datetime
            ) as clock
        ):
            if change == "operation":
                other = OperationSpendCeiling(tmp_path / "other.sqlite", tariffs={})
                other.authorize_policy(ceiling_usd=1.0, unpriced_attempts=20)
                install_spend_ceiling(other)
            elif change == "deadline":
                assert clock is not None
                clock.now.return_value = settings.deadline + timedelta(seconds=1)
            else:
                assert clock is not None
                clock.now.return_value = settings.deadline - timedelta(seconds=1)
            reasons = {
                "settings": "original request",
                "request": "original request",
                "sources": "original request",
                "operation": "original active operation",
                "ceilings": "ceilings changed",
                "deadline": "original deadline expired",
                "reference": "immutable evidence",
            }
            with pytest.raises(ValueError, match=reasons[change]):
                author_response(
                    project,
                    settings,
                    request,
                    identity="parent",
                    sources=sources,
                    code_revision="test",
                    retry_authorization=registration,
                )
        assert transport.requests == []
        assert operation.summary().unpriced_attempts == 3


def test_unknown_outcomes_and_unapproved_resumes_do_not_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exhausted ambiguous request stays unknown despite an explicit retry request."""
    with operation_spend_scope():
        project, settings, request, transport, operation = _failed_author(
            tmp_path, monkeypatch, ambiguous=True
        )
        with pytest.raises(ValueError, match="ambiguous"):
            _register(project, operation)
        with pytest.raises(ValueError, match="unknown outcome"):
            author_response(
                project,
                settings,
                request,
                identity="parent",
                sources=(settings.base, settings.brief),
                code_revision="test",
            )
        assert transport.requests == [] and operation.summary().unpriced_attempts == 3


def test_retry_publication_interruption_replays_the_same_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A completed child survives interruption before its parent resolution is published."""
    with operation_spend_scope():
        project, settings, request, transport, operation = _failed_author(tmp_path, monkeypatch)
        registration = _register(project, operation)
        with (
            patch(
                "exp.optimize.evaluation.factory.authoring.AuthorRetryResolution",
                side_effect=RuntimeError("interrupted resolution publication"),
            ),
            pytest.raises(RuntimeError, match="interrupted"),
        ):
            author_response(
                project,
                settings,
                request,
                retry_authorization=registration,
                identity="parent",
                sources=(settings.base, settings.brief),
                code_revision="test",
            )
        assert not project.artifacts.exists(retry_resolution_id("parent"))
    reference, response = author_response(
        project,
        settings,
        request,
        retry_authorization=registration,
        identity="parent",
        sources=(settings.base, settings.brief),
        code_revision="test",
    )
    assert author_response(
        project,
        settings,
        request,
        identity="parent",
        sources=(settings.base, settings.brief),
        code_revision="test",
    ) == (reference, response)
    assert len(transport.requests) == 1 and operation.summary().unpriced_attempts == 4


def test_retry_cannot_expand_its_child_or_replenish_an_existing_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every parent's child allowance is immutable and no greater than the original three."""
    with operation_spend_scope():
        project, _, _, transport, operation = _failed_author(tmp_path, monkeypatch)
        parent = artifact_input(project.artifacts.read("parent-intent").manifest)
        with pytest.raises(ValueError, match="exceeds"):
            register_author_retry(
                project,
                parent,
                operation=operation,
                authorization_id="user-retry",
                code_revision="test",
                maximum_physical_attempts=4,
            )
        registration = _register(project, operation)
        assert load_author_retry(project, registration).maximum_physical_attempts == 3
        with pytest.raises(ValueError, match="different immutable"):
            register_author_retry(
                project,
                parent,
                operation=operation,
                authorization_id="another-retry",
                code_revision="test",
            )
        assert transport.requests == []


def test_exhausted_operation_still_refuses_explicit_child_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Child registration does not replenish the parent's original operation ceiling."""
    with operation_spend_scope():
        project, settings, request, transport, operation = _failed_author(
            tmp_path, monkeypatch, cap=3
        )
        registration = _register(project, operation)
        from exp.runtime.models.spend_ceiling import UnpricedDispatchRefused

        with pytest.raises(UnpricedDispatchRefused):
            author_response(
                project,
                settings,
                request,
                identity="parent",
                sources=(settings.base, settings.brief),
                code_revision="test",
                retry_authorization=registration,
            )
        assert transport.requests == [] and operation.summary().unpriced_attempts == 3


def test_original_failure_receipts_cannot_authorize_a_different_operation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Copied terminal receipts cannot transfer the original operation's charged liability."""
    with operation_spend_scope():
        project, _, _, transport, original = _failed_author(tmp_path, monkeypatch)
        other = OperationSpendCeiling(tmp_path / "other.sqlite", tariffs={})
        other.authorize_policy(ceiling_usd=1.0, unpriced_attempts=20)
        with operation_spend_scope():
            install_spend_ceiling(other)
            with pytest.raises(ValueError, match="accounted"):
                _register(project, other)
        assert original.summary().unpriced_attempts == 3
        assert other.summary().unpriced_attempts == 0 and transport.requests == []


@pytest.mark.parametrize("outcome", ["response", "rejection"])
def test_successful_or_length_rejected_parent_cannot_authorize_a_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    """An explicit retry cannot replace a completed answer or its known output-budget stop."""
    from exp.runtime.models.providers.errors import ProviderRetryableResponseError

    monkeypatch.setattr("socket.socket.connect", _no_network)
    project, settings, _ = proposal_fixture(tmp_path, monkeypatch)
    request = proposal_request(project, settings)
    if outcome == "rejection":
        transport = ScriptedJsonTransport(
            (
                JsonHttpResponse(
                    status_code=200,
                    body={
                        "usage": {
                            "prompt_tokens": 32,
                            "completion_tokens": 4096,
                            "completion_tokens_details": {"reasoning_tokens": 4096},
                        },
                        "id": "length",
                        "model": "world-v1",
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": "length",
                                "message": {"role": "assistant", "content": None},
                            }
                        ],
                    },
                ),
            )
        )
        monkeypatch.setattr(
            "exp.optimize.evaluation.factory.authoring.HttpxJsonTransport", lambda: transport
        )
    operation = OperationSpendCeiling(tmp_path / "operation.sqlite", tariffs={})
    operation.authorize_policy(ceiling_usd=1.0, unpriced_attempts=20)
    with operation_spend_scope():
        install_spend_ceiling(operation)
        if outcome == "rejection":
            with pytest.raises(ProviderRetryableResponseError):
                author_response(
                    project,
                    settings,
                    request,
                    identity="parent",
                    sources=(settings.base, settings.brief),
                    code_revision="test",
                )
        else:
            author_response(
                project,
                settings,
                request,
                identity="parent",
                sources=(settings.base, settings.brief),
                code_revision="test",
            )
        with pytest.raises(ValueError, match="cannot replace"):
            _register(project, operation)
        assert operation.summary().unpriced_attempts == 1


def test_new_retry_registration_cannot_extend_an_expired_original_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """User retry consent supplies no additional time after the original author window."""
    with operation_spend_scope():
        project, settings, _, transport, operation = _failed_author(tmp_path, monkeypatch)
        with patch(
            "exp.optimize.evaluation.factory.author_retries.datetime", wraps=datetime
        ) as clock:
            clock.now.return_value = settings.deadline + timedelta(seconds=1)
            with pytest.raises(ValueError, match="original deadline expired"):
                _register(project, operation)
        assert not project.artifacts.list_ids(artifact_type="author-retry-authorization")
        assert transport.requests == [] and operation.summary().unpriced_attempts == 3


def test_registered_execution_window_renews_only_admission_for_child_and_fresh_authors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit finite window preserves original settings, identities and charged history."""
    from exp.optimize.evaluation.factory.author_retries import (
        load_author_execution_window,
        register_author_execution_window,
    )
    from exp.optimize.evaluation.variance.evidence_tools_test import _reply

    with operation_spend_scope():
        project, settings, request, _, operation = _failed_author(tmp_path, monkeypatch)
        after_original = settings.deadline + timedelta(seconds=1)
        renewed = after_original + timedelta(hours=6)
        reply = _reply([("propose_tasks", {"candidates": []})])
        transport = ScriptedJsonTransport((reply, reply))
        monkeypatch.setattr(
            "exp.optimize.evaluation.factory.authoring.HttpxJsonTransport", lambda: transport
        )
        with (
            patch(
                "exp.optimize.evaluation.factory.author_retries.datetime", wraps=datetime
            ) as retry_clock,
            patch(
                "exp.optimize.evaluation.factory.authoring.datetime", wraps=datetime
            ) as author_clock,
        ):
            retry_clock.now.return_value = author_clock.now.return_value = after_original
            window = register_author_execution_window(
                project,
                settings,
                operation=operation,
                window_id="user-six-hour-retry",
                execution_deadline=renewed,
                code_revision="test",
                sources=(settings.base,),
            )
            held = load_author_execution_window(project, window)
            assert held.settings_sha256 == sha256_json(settings)
            assert (
                held.original_deadline == settings.deadline and held.execution_deadline == renewed
            )
            assert held.sources == (settings.base,)
            registration = register_author_retry(
                project,
                artifact_input(project.artifacts.read("parent-intent").manifest),
                operation=operation,
                authorization_id="user-retry",
                code_revision="test",
                execution_window=window,
            )
            grant = load_author_retry(project, registration)
            assert grant.deadline == settings.deadline and grant.effective_retry_deadline == renewed
            sources = (settings.base, settings.brief)
            reference, response = author_response(
                project,
                settings,
                request,
                identity="parent",
                sources=sources,
                code_revision="test",
                retry_authorization=registration,
                execution_window=window,
            )
            author_response(
                project,
                settings,
                request,
                identity="remaining-review",
                sources=sources,
                code_revision="test",
                execution_window=window,
            )
            with pytest.raises(ValueError, match="deadline expired"):
                author_response(
                    project,
                    settings,
                    request,
                    identity="no-window-review",
                    sources=sources,
                    code_revision="test",
                )
        import json

        original = json.loads(project.artifacts.read_bytes("remaining-review-intent", "spec.json"))
        assert original == settings.model_dump(mode="json")
        assert not project.artifacts.exists("no-window-review-intent")
        assert operation.summary().unpriced_attempts == 5
    assert author_response(
        project, settings, request, identity="parent", sources=sources, code_revision="test"
    ) == (reference, response)
    assert len(transport.requests) == 2


@pytest.mark.parametrize("change", ["settings", "operation", "ceilings", "expired", "future_bound"])
def test_execution_window_refuses_different_authority_and_excess_or_expired_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    """An execution window grants no model, budget, operation or indefinite-time override."""
    from exp.optimize.evaluation.factory.author_retries import register_author_execution_window

    with operation_spend_scope():
        project, settings, request, transport, operation = _failed_author(tmp_path, monkeypatch)
        renewed = datetime.now(settings.deadline.tzinfo) + timedelta(hours=6)
        if change == "future_bound":
            with pytest.raises(ValueError, match="at most six hours"):
                register_author_execution_window(
                    project,
                    settings,
                    operation=operation,
                    window_id="renewal",
                    execution_deadline=renewed + timedelta(hours=1),
                    code_revision="test",
                )
            return
        window = register_author_execution_window(
            project,
            settings,
            operation=operation,
            window_id="renewal",
            execution_deadline=renewed,
            code_revision="test",
        )
        assert (
            register_author_execution_window(
                project,
                settings,
                operation=operation,
                window_id="renewal",
                execution_deadline=renewed,
                code_revision="test",
            )
            == window
        )
        with pytest.raises(ValueError, match="different immutable"):
            register_author_execution_window(
                project,
                settings,
                operation=operation,
                window_id="renewal",
                execution_deadline=renewed - timedelta(seconds=1),
                code_revision="test",
            )
        if change == "settings":
            settings = settings.model_copy(update={"output_tokens": settings.output_tokens + 1})
        elif change == "ceilings":
            operation.authorize_policy(ceiling_usd=1.0, unpriced_attempts=21)
        with (
            operation_spend_scope()
            if change == "operation"
            else patch(
                "exp.optimize.evaluation.factory.author_retries.datetime", wraps=datetime
            ) as clock
        ):
            if change == "operation":
                other = OperationSpendCeiling(tmp_path / "other.sqlite", tariffs={})
                other.authorize_policy(ceiling_usd=1.0, unpriced_attempts=20)
                install_spend_ceiling(other)
            else:
                assert clock is not None
                clock.now.return_value = (
                    renewed + timedelta(seconds=1)
                    if change == "expired"
                    else renewed - timedelta(seconds=1)
                )
            with pytest.raises(ValueError):
                author_response(
                    project,
                    settings,
                    request,
                    identity="remaining-review",
                    sources=(settings.base, settings.brief),
                    code_revision="test",
                    execution_window=window,
                )
        assert transport.requests == [] and operation.summary().unpriced_attempts == 3
        assert not project.artifacts.exists("remaining-review-intent")


def test_new_retry_producer_truthfully_differs_from_the_immutable_parent_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """New authority and child artifacts name their real producer without rewriting history."""
    old, new = "a" * 40, "b" * 40
    with operation_spend_scope():
        project, settings, request, transport, operation = _failed_author(
            tmp_path, monkeypatch, parent_revision=old
        )
        parent = artifact_input(project.artifacts.read("parent-intent").manifest)
        with pytest.raises(ValueError, match="exact original author intent"):
            register_author_retry(
                project,
                parent.model_copy(update={"sha256": "0" * 64}),
                operation=operation,
                authorization_id="user-retry",
                code_revision=new,
            )
        registration = register_author_retry(
            project, parent, operation=operation, authorization_id="user-retry", code_revision=new
        )
        grant = load_author_retry(project, registration)
        assert grant.parent_code_revision == old
        assert project.artifacts.read(registration.artifact_id).manifest.code_revision == new
        sources = (settings.base, settings.brief)
        reference, response = author_response(
            project,
            settings,
            request,
            identity="parent",
            sources=sources,
            code_revision=new,
            retry_authorization=registration,
        )
        assert project.artifacts.read(reference.artifact_id).manifest.code_revision == new
        assert (
            project.artifacts.read(grant.child_identity + "-intent").manifest.code_revision == new
        )
    assert author_response(
        project, settings, request, identity="parent", sources=sources, code_revision="c" * 40
    ) == (reference, response)
    assert project.artifacts.read(retry_resolution_id("parent")).manifest.code_revision == new
    assert artifact_input(project.artifacts.read(parent.artifact_id).manifest) == parent
    assert project.artifacts.read(parent.artifact_id).manifest.code_revision == old
    assert len(transport.requests) == 1 and operation.summary().unpriced_attempts == 4
