"""Version-bound E2B control-plane inspection for prepared Harbor templates."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import cast

from e2b.api.client.api.tags import get_templates_template_id_tags
from e2b.api.client.api.templates import (
    get_templates_aliases_alias,
    get_templates_template_id,
)
from e2b.api.client.models.template_alias_response import TemplateAliasResponse
from e2b.api.client.models.template_build_status import TemplateBuildStatus
from e2b.api.client.models.template_tag import TemplateTag
from e2b.api.client.models.template_with_builds import TemplateWithBuilds
from e2b.api.client_async import get_api_client
from e2b.connection_config import ConnectionConfig

from wmh.evals.harbor.e2b_template_policy import e2b_sdk_version

E2B_CONTROL_SDK_VERSION = "2.31.0"


class E2BTemplateNotFound(RuntimeError):
    """A qualified E2B template name has no current control-plane target."""


@dataclass(frozen=True)
class E2BTemplateControlIdentity:
    """Current provider identity and resources behind one qualified name."""

    template_id: str
    build_id: str
    cpu_count: int
    memory_mb: int


async def inspect_e2b_template(
    template_name: str,
    *,
    expected_cpu_count: int,
    expected_memory_mb: int,
) -> E2BTemplateControlIdentity:
    """Resolve one name to its unique ready default build and exact resources."""
    if e2b_sdk_version() != E2B_CONTROL_SDK_VERSION:
        raise RuntimeError(
            f"Harbor E2B template inspection requires e2b=={E2B_CONTROL_SDK_VERSION}"
        )
    if not template_name:
        raise ValueError("E2B template name must be nonempty")
    if (
        isinstance(expected_cpu_count, bool)
        or not isinstance(expected_cpu_count, int)
        or isinstance(expected_memory_mb, bool)
        or not isinstance(expected_memory_mb, int)
        or expected_cpu_count < 1
        or expected_memory_mb < 128
    ):
        raise ValueError("expected E2B template resources are invalid")

    async with get_api_client(ConnectionConfig()) as api_client:
        alias_response = await get_templates_aliases_alias.asyncio_detailed(
            alias=template_name,
            client=api_client,
        )
        if alias_response.status_code == HTTPStatus.NOT_FOUND:
            raise E2BTemplateNotFound("qualified E2B template name was not found")
        if (
            alias_response.status_code != HTTPStatus.OK
            or not isinstance(alias_response.parsed, TemplateAliasResponse)
            or not alias_response.parsed.template_id
        ):
            raise RuntimeError("qualified E2B template name is unavailable")
        template_id = alias_response.parsed.template_id

        template_response = await get_templates_template_id.asyncio_detailed(
            template_id=template_id,
            client=api_client,
            limit=100,
        )
        if template_response.status_code != HTTPStatus.OK or not isinstance(
            template_response.parsed, TemplateWithBuilds
        ):
            raise RuntimeError("qualified E2B template details are unavailable")
        details = template_response.parsed
        if details.template_id != template_id:
            raise RuntimeError("E2B template control-plane identity disagreement")

        tag_response = await get_templates_template_id_tags.asyncio_detailed(
            template_id=template_id,
            client=api_client,
        )
        if (
            tag_response.status_code != HTTPStatus.OK
            or not isinstance(tag_response.parsed, list)
            or any(not isinstance(tag, TemplateTag) for tag in tag_response.parsed)
        ):
            raise RuntimeError("qualified E2B template tags are unavailable")
        tags = cast(list[TemplateTag], tag_response.parsed)
        confirmation = await get_templates_aliases_alias.asyncio_detailed(
            alias=template_name,
            client=api_client,
        )
        if (
            confirmation.status_code != HTTPStatus.OK
            or not isinstance(confirmation.parsed, TemplateAliasResponse)
            or confirmation.parsed.template_id != template_id
        ):
            raise RuntimeError("qualified E2B template name changed during inspection")
    default_tags = [tag for tag in tags if tag.tag == "default"]
    if len(default_tags) != 1 or not default_tags[0].build_id:
        raise RuntimeError("qualified E2B template has no unique default tag")
    build_id = str(default_tags[0].build_id)
    builds = [build for build in details.builds if str(build.build_id) == build_id]
    if len(builds) != 1:
        raise RuntimeError("qualified E2B default build is absent or ambiguous")
    build = builds[0]
    if build.status is not TemplateBuildStatus.READY:
        raise RuntimeError("qualified E2B default build is not ready")
    if build.cpu_count != expected_cpu_count or build.memory_mb != expected_memory_mb:
        raise RuntimeError("qualified E2B default build resource mismatch")
    return E2BTemplateControlIdentity(
        template_id=template_id,
        build_id=build_id,
        cpu_count=build.cpu_count,
        memory_mb=build.memory_mb,
    )
