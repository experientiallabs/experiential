"""Offline tests for version-bound E2B template control-plane inspection."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from http import HTTPStatus
from types import SimpleNamespace
from uuid import UUID

import pytest
from e2b.api.client.models.template_alias_response import TemplateAliasResponse
from e2b.api.client.models.template_build import TemplateBuild
from e2b.api.client.models.template_build_status import TemplateBuildStatus
from e2b.api.client.models.template_tag import TemplateTag
from e2b.api.client.models.template_with_builds import TemplateWithBuilds

import wmh.evals.harbor.e2b_template_control as control

_TEMPLATE_ID = "template-id"
_BUILD_ID = UUID("00000000-0000-4000-8000-000000000001")
_LATEST_BUILD_ID = UUID("00000000-0000-4000-8000-000000000002")


async def _inspect() -> control.E2BTemplateControlIdentity:
    return await control.inspect_e2b_template(
        "wmh-hb-v1-example",
        expected_cpu_count=4,
        expected_memory_mb=8192,
    )


def _response(parsed: object, status: HTTPStatus = HTTPStatus.OK) -> SimpleNamespace:
    return SimpleNamespace(status_code=status, parsed=parsed)


class _Client:
    def __init__(self) -> None:
        self.exited = False

    async def __aenter__(self) -> _Client:
        return self

    async def __aexit__(self, *_args: object) -> None:
        self.exited = True


def _details(*, status: TemplateBuildStatus = TemplateBuildStatus.READY) -> TemplateWithBuilds:
    now = datetime.now(UTC)
    return TemplateWithBuilds(
        aliases=[],
        builds=[
            TemplateBuild(
                build_id=_BUILD_ID,
                cpu_count=4,
                memory_mb=8192,
                status=status,
                created_at=now,
                updated_at=now,
            ),
            TemplateBuild(
                build_id=_LATEST_BUILD_ID,
                cpu_count=8,
                memory_mb=16384,
                status=TemplateBuildStatus.READY,
                created_at=now,
                updated_at=now,
            ),
        ],
        created_at=now,
        last_spawned_at=None,
        names=["wmh-hb-v1-example"],
        public=False,
        spawn_count=0,
        template_id=_TEMPLATE_ID,
        updated_at=now,
    )


def _install_control_responses(
    monkeypatch: pytest.MonkeyPatch,
    *,
    details: TemplateWithBuilds | None = None,
    tags: list[TemplateTag] | None = None,
) -> _Client:
    async def resolve_alias(**kwargs: object) -> SimpleNamespace:
        assert kwargs["alias"] == "wmh-hb-v1-example"
        return _response(TemplateAliasResponse(public=False, template_id=_TEMPLATE_ID))

    async def get_template(**kwargs: object) -> SimpleNamespace:
        assert kwargs["template_id"] == _TEMPLATE_ID
        return _response(details or _details())

    async def get_tags(**kwargs: object) -> SimpleNamespace:
        assert kwargs["template_id"] == _TEMPLATE_ID
        return _response(
            tags
            or [
                TemplateTag(
                    tag="default",
                    build_id=_BUILD_ID,
                    created_at=datetime.now(UTC),
                ),
                TemplateTag(
                    tag="latest",
                    build_id=_LATEST_BUILD_ID,
                    created_at=datetime.now(UTC),
                ),
            ]
        )

    client = _Client()
    monkeypatch.setattr(control, "get_api_client", lambda _config: client)
    monkeypatch.setattr(control.get_templates_aliases_alias, "asyncio_detailed", resolve_alias)
    monkeypatch.setattr(control.get_templates_template_id, "asyncio_detailed", get_template)
    monkeypatch.setattr(control.get_templates_template_id_tags, "asyncio_detailed", get_tags)
    return client


def test_inspection_returns_unique_ready_default_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(control, "e2b_sdk_version", lambda: "2.31.0")
    client = _install_control_responses(monkeypatch)

    observed = asyncio.run(_inspect())

    assert observed == control.E2BTemplateControlIdentity(
        template_id=_TEMPLATE_ID,
        build_id=str(_BUILD_ID),
        cpu_count=4,
        memory_mb=8192,
    )
    assert client.exited


def test_inspection_rejects_unpinned_e2b_sdk_before_api_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(control, "e2b_sdk_version", lambda: "2.32.0")
    api_calls = 0

    def get_client(_config: object) -> object:
        nonlocal api_calls
        api_calls += 1
        return object()

    monkeypatch.setattr(control, "get_api_client", get_client)

    with pytest.raises(RuntimeError, match="requires e2b==2.31.0"):
        asyncio.run(_inspect())

    assert api_calls == 0


def test_inspection_rejects_ambiguous_default_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(control, "e2b_sdk_version", lambda: "2.31.0")
    now = datetime.now(UTC)
    _install_control_responses(
        monkeypatch,
        tags=[
            TemplateTag(tag="default", build_id=_BUILD_ID, created_at=now),
            TemplateTag(tag="default", build_id=_BUILD_ID, created_at=now),
        ],
    )

    with pytest.raises(RuntimeError, match="unique default tag"):
        asyncio.run(_inspect())


def test_inspection_rejects_nonready_build(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(control, "e2b_sdk_version", lambda: "2.31.0")
    _install_control_responses(
        monkeypatch,
        details=_details(status=TemplateBuildStatus.BUILDING),
    )

    with pytest.raises(RuntimeError, match="not ready"):
        asyncio.run(_inspect())


def test_inspection_rejects_resource_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(control, "e2b_sdk_version", lambda: "2.31.0")
    details = _details()
    details.builds[0].memory_mb = 4096
    _install_control_responses(monkeypatch, details=details)

    with pytest.raises(RuntimeError, match="resource mismatch"):
        asyncio.run(_inspect())


def test_inspection_rejects_template_identity_disagreement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(control, "e2b_sdk_version", lambda: "2.31.0")
    details = _details()
    details.template_id = "other-template"
    _install_control_responses(monkeypatch, details=details)

    with pytest.raises(RuntimeError, match="identity disagreement"):
        asyncio.run(_inspect())


def test_inspection_rejects_alias_change_during_control_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(control, "e2b_sdk_version", lambda: "2.31.0")
    _install_control_responses(monkeypatch)
    alias_calls = 0

    async def resolve_alias(**_kwargs: object) -> SimpleNamespace:
        nonlocal alias_calls
        alias_calls += 1
        template_id = _TEMPLATE_ID if alias_calls == 1 else "other-template"
        return _response(TemplateAliasResponse(public=False, template_id=template_id))

    monkeypatch.setattr(control.get_templates_aliases_alias, "asyncio_detailed", resolve_alias)

    with pytest.raises(RuntimeError, match="changed during inspection"):
        asyncio.run(_inspect())

    assert alias_calls == 2


def test_inspection_distinguishes_exact_alias_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(control, "e2b_sdk_version", lambda: "2.31.0")
    client = _Client()
    monkeypatch.setattr(control, "get_api_client", lambda _config: client)

    async def missing(**_kwargs: object) -> SimpleNamespace:
        return _response(object(), status=HTTPStatus.NOT_FOUND)

    monkeypatch.setattr(control.get_templates_aliases_alias, "asyncio_detailed", missing)

    with pytest.raises(control.E2BTemplateNotFound, match="not found"):
        asyncio.run(_inspect())

    assert client.exited
