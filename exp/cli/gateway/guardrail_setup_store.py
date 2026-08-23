"""Setup-owned guardrail document identity, mutation, and locked publish."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal, cast

from exp.common.core.artifacts import JsonObject, JsonValue, stable_id
from exp.common.core.files import resolve_write_target, write_bytes_atomic
from exp.common.core.locks import file_write_lock
from exp.runtime.gateway.guardrails.config import engine_from_document
from exp.runtime.gateway.guardrails.contracts import (
    DEFAULT_MAX_REQUEST_BYTES,
    DEFAULT_MAX_RESPONSE_BYTES,
)
from exp.runtime.gateway.guardrails.http_json import DEFAULT_HTTP_JSON_MAX_RESPONSE_BYTES
from exp.runtime.gateway.guardrails.preset import (
    STANDARD_DEFAULT_TIMEOUT_MS,
    STANDARD_PRESET_NAME,
    STANDARD_PRESET_STEPS,
    STANDARD_REQUIRED_CAPABILITIES,
)
from exp.runtime.gateway.sqlite.alias_activation import AliasActivationOutcomeUnknownError

_CONFIG_NAME = "guardrails.json"
_ADAPTER_ID_PREFIX = "setup-http-json"
_POLICY_ID_PREFIX = "setup-standard"
_SETUP_ADAPTER_KEYS = frozenset({"adapter_id", "kind", "url", "bearer_env", "max_response_bytes"})
_SETUP_POLICY_KEYS = frozenset(
    {
        "policy_id",
        "organization_id",
        "identity_id",
        "protected",
        "preset",
        "timeout_ms",
        "timeouts",
        "capability_adapters",
        "max_request_bytes",
        "max_response_bytes",
    }
)
GUARDRAILS_OFF = "Off"
GUARDRAILS_STANDARD = "Standard"
GUARDRAILS_CUSTOM = "Custom/preserved"


class GuardrailSetupMode(StrEnum):
    """How setup can represent the selected identity's current policy."""

    OFF = "off"
    STANDARD = "standard"
    CUSTOM = "custom"


@dataclass(frozen=True)
class GuardrailInspection:
    """Content-free view of the selected identity's current guardrail file."""

    mode: GuardrailSetupMode
    display: str
    classifier_url: str | None = None
    bearer_env: str | None = None


@dataclass(frozen=True)
class GuardrailSetupPlan:
    """One explicit mutation of the selected identity's setup-owned policy."""

    action: Literal["off", "standard"]
    organization_id: str
    identity_id: str
    classifier_url: str = ""
    bearer_env: str | None = None
    replace_custom: bool = False


@dataclass(frozen=True)
class GuardrailSetupSelection:
    """Operator choice collected by the defaults screen."""

    plan: GuardrailSetupPlan | None
    display: str


def guardrail_config_path(root: Path) -> Path:
    """Return the configured guardrail document path under ``root``."""
    return root / "gateway" / _CONFIG_NAME


def setup_adapter_id(organization_id: str, identity_id: str) -> str:
    """Return the deterministic setup-owned adapter ID for one identity.

    The identifier is a ``stable_id`` over a structured subject so hyphenated
    organization and identity pairs cannot collide, and long valid identifiers
    still produce a bounded ArtifactId.

    Args:
        organization_id: Local organization that owns the policy.
        identity_id: Identity the operator selected.

    Returns:
        A valid ArtifactId that does not embed raw identity fragments.

    Raises:
        ValueError: The prefix cannot begin an artifact ID.
    """
    return _setup_owned_id(_ADAPTER_ID_PREFIX, organization_id, identity_id)


def setup_policy_id(organization_id: str, identity_id: str) -> str:
    """Return the deterministic setup-owned policy ID for one identity.

    Args:
        organization_id: Local organization that owns the policy.
        identity_id: Identity the operator selected.

    Returns:
        A valid ArtifactId that does not embed raw identity fragments.

    Raises:
        ValueError: The prefix cannot begin an artifact ID.
    """
    return _setup_owned_id(_POLICY_ID_PREFIX, organization_id, identity_id)


def inspect_setup_guardrails(
    root: Path,
    organization_id: str,
    identity_id: str,
) -> GuardrailInspection:
    """Classify the selected identity's current policy without reading content.

    Args:
        root: EXP root that may contain ``gateway/guardrails.json``.
        organization_id: Local organization used for lookup.
        identity_id: Identity shown on the setup defaults screen.

    Returns:
        Off when the pair has no policy, Standard when the file is exactly the
        setup-owned pack, or Custom/preserved otherwise.

    Raises:
        ValueError: The file exists but is not a valid policy document.
    """
    path = guardrail_config_path(root)
    _reject_non_file(path)
    if not path.is_file():
        return GuardrailInspection(mode=GuardrailSetupMode.OFF, display=GUARDRAILS_OFF)
    document = load_authored_document(path)
    engine_from_document(document)
    policy = policy_for(document, organization_id, identity_id)
    if policy is None:
        return GuardrailInspection(mode=GuardrailSetupMode.OFF, display=GUARDRAILS_OFF)
    adapter_id = setup_adapter_id(organization_id, identity_id)
    adapter = adapter_for(document, adapter_id)
    if is_setup_owned_policy(policy, organization_id, identity_id) and is_setup_owned_adapter(
        adapter, adapter_id
    ):
        url = adapter.get("url") if adapter is not None else None
        bearer = adapter.get("bearer_env") if adapter is not None else None
        return GuardrailInspection(
            mode=GuardrailSetupMode.STANDARD,
            display=GUARDRAILS_STANDARD,
            classifier_url=url if isinstance(url, str) else None,
            bearer_env=bearer if isinstance(bearer, str) else None,
        )
    return GuardrailInspection(mode=GuardrailSetupMode.CUSTOM, display=GUARDRAILS_CUSTOM)


def apply_setup_guardrails(root: Path, plan: GuardrailSetupPlan) -> None:
    """Apply one explicit setup-owned mutation under a locked read-modify-write.

    Args:
        root: EXP root that owns ``gateway/guardrails.json``.
        plan: Disable or standard-pack mutation for one identity.

    Raises:
        ValueError: The existing file is malformed, the plan is unsafe, or the
            published document does not match the selected mode.
        OSError: The atomic publish failed, so the previous file is unchanged.
    """
    path = guardrail_config_path(root)
    with file_write_lock(path, what="gateway guardrail configuration"):
        apply_setup_guardrails_unlocked(root, plan)


@contextmanager
def guardrail_setup_compensation(root: Path, plan: GuardrailSetupPlan | None) -> Iterator[None]:
    """Persist the selected guardrail plan and restore only proven setup failures.

    One lock covers the snapshot, the selected mutation, the remainder of the
    setup mutation window, and proven-failure restoration. An indeterminate
    alias-activation outcome keeps the selected file so it stays aligned with
    authority that may already have committed.

    Args:
        root: EXP root owning the guardrail file.
        plan: Mutation to publish, or ``None`` to preserve the current file.

    Yields:
        Control to the rest of gateway setup.

    Raises:
        RuntimeError: The previous guardrail file cannot be restored.
    """
    configured_path = guardrail_config_path(root)
    with file_write_lock(configured_path, what="gateway guardrail configuration"):
        original = _snapshot_guardrail_bytes(configured_path)
        try:
            if plan is not None:
                apply_setup_guardrails_unlocked(root, plan)
            yield
        except AliasActivationOutcomeUnknownError:
            raise
        except BaseException:
            try:
                restore_guardrail_file_unlocked(configured_path, original)
            except BaseException as compensation_error:
                raise RuntimeError(
                    "gateway setup guardrail compensation outcome is unknown; inspect "
                    "gateway/guardrails.json before retrying"
                ) from compensation_error
            raise


def apply_setup_guardrails_unlocked(root: Path, plan: GuardrailSetupPlan) -> None:
    """Apply one setup-owned mutation while the caller already holds the lock.

    Args:
        root: EXP root that owns ``gateway/guardrails.json``.
        plan: Disable or standard-pack mutation for one identity.

    Raises:
        ValueError: The existing file is malformed, the plan is unsafe, or the
            published document does not match the selected mode.
        OSError: The atomic publish failed, so the previous file is unchanged.
    """
    path = guardrail_config_path(root)
    _reject_non_file(path)
    document = load_authored_document(path) if path.is_file() else empty_document()
    if path.is_file():
        engine_from_document(document)
    if plan.action == "off":
        updated = disable_owned(document, plan.organization_id, plan.identity_id)
    else:
        updated = enable_standard(document, plan)
    publish_guardrail_document(path, updated)
    landed = inspect_setup_guardrails(root, plan.organization_id, plan.identity_id)
    expected = GuardrailSetupMode.STANDARD if plan.action == "standard" else GuardrailSetupMode.OFF
    if landed.mode is not expected:
        raise ValueError("gateway guardrail configuration did not land as selected")


def restore_guardrail_file_unlocked(configured_path: Path, original: bytes | None) -> None:
    """Restore the previous guardrail bytes without taking the write lock.

    A missing preimage deletes a regular file setup created. When the
    configured path is a symlink, only the target is removed so the link
    remains.

    Args:
        configured_path: Configured path, possibly a symlink.
        original: Previous file bytes, or ``None`` when the file was absent.

    Raises:
        OSError: The compensation write or unlink failed.
    """
    if original is None:
        target = resolve_write_target(configured_path)
        target.unlink(missing_ok=True)
        if not configured_path.is_symlink():
            configured_path.unlink(missing_ok=True)
        return
    write_bytes_atomic(configured_path, original)


def publish_guardrail_document(configured_path: Path, document: JsonObject) -> None:
    """Validate and publish one authored document, or restore the missing-file path.

    A setup-created regular file with no leftover fields is deleted so the
    unguarded hot path stays a missing file. A symlink is never removed: an
    emptied document is written through it so the link and target stay valid.

    Args:
        configured_path: Configured path, possibly a symlink.
        document: Authored configuration after a setup mutation.

    Raises:
        ValueError: The document is not a valid policy document.
        OSError: The atomic publish or unlink failed.
    """
    engine_from_document(document)
    if is_setup_created_empty(document) and not configured_path.is_symlink():
        resolve_write_target(configured_path).unlink(missing_ok=True)
        return
    write_bytes_atomic(configured_path, encode_document(document))


def enable_standard(document: JsonObject, plan: GuardrailSetupPlan) -> JsonObject:
    """Insert or update the setup-owned adapter and standard policy.

    Args:
        document: Current authored configuration.
        plan: Standard-pack mutation for one identity.

    Returns:
        A new document that preserves unrelated adapters, policies, and
        top-level fields.

    Raises:
        ValueError: A custom policy or colliding adapter is not setup-owned.
    """
    adapter_id = setup_adapter_id(plan.organization_id, plan.identity_id)
    policy_id = setup_policy_id(plan.organization_id, plan.identity_id)
    adapters = list(object_list(document, "adapters"))
    policies = list(object_list(document, "policies"))
    existing = policy_for({"policies": policies}, plan.organization_id, plan.identity_id)
    if existing is not None and not is_setup_owned_policy(
        existing, plan.organization_id, plan.identity_id
    ):
        if not plan.replace_custom:
            raise ValueError(
                "existing guardrails are custom/preserved; type replace to author the standard pack"
            )
        policies = [
            item
            for item in policies
            if not same_identity(item, plan.organization_id, plan.identity_id)
        ]
    existing_adapter = adapter_for({"adapters": adapters}, adapter_id)
    if existing_adapter is not None and not is_setup_owned_adapter(existing_adapter, adapter_id):
        raise ValueError(
            f"adapter {adapter_id!r} exists and is not owned by interactive setup; "
            "leave it unchanged or choose a different identity"
        )
    adapters = [item for item in adapters if item.get("adapter_id") != adapter_id]
    adapters.append(_owned_adapter_document(adapter_id, plan))
    policies = [
        item for item in policies if not same_identity(item, plan.organization_id, plan.identity_id)
    ]
    policies.append(_owned_policy_document(policy_id, adapter_id, plan))
    return _with_lists(document, adapters, policies)


def disable_owned(document: JsonObject, organization_id: str, identity_id: str) -> JsonObject:
    """Remove setup-owned artifacts for one identity and leave everything else.

    Args:
        document: Current authored configuration.
        organization_id: Local organization used for ownership checks.
        identity_id: Identity whose setup-owned pack should be removed.

    Returns:
        A new document that preserves unrelated adapters, policies, and
        top-level fields.
    """
    adapter_id = setup_adapter_id(organization_id, identity_id)
    policies = [
        item
        for item in object_list(document, "policies")
        if not is_setup_owned_policy(item, organization_id, identity_id)
    ]
    referenced = referenced_adapter_ids(policies)
    adapters = []
    for item in object_list(document, "adapters"):
        if item.get("adapter_id") == adapter_id and is_setup_owned_adapter(item, adapter_id):
            if adapter_id in referenced:
                adapters.append(item)
            continue
        adapters.append(item)
    return _with_lists(document, adapters, policies)


def is_setup_owned_policy(
    item: Mapping[str, JsonValue],
    organization_id: str,
    identity_id: str,
) -> bool:
    """Return whether one authored policy is exactly the setup-owned standard pack.

    Args:
        item: One policy object from the document.
        organization_id: Expected organization.
        identity_id: Expected identity.

    Returns:
        True only when every authored field matches the setup-owned contract.
    """
    if set(item) - _SETUP_POLICY_KEYS:
        return False
    if item.get("policy_id") != setup_policy_id(organization_id, identity_id):
        return False
    if item.get("organization_id") != organization_id or item.get("identity_id") != identity_id:
        return False
    if item.get("protected") is not True or item.get("preset") != STANDARD_PRESET_NAME:
        return False
    if "checks" in item:
        return False
    timeout_ms = item.get("timeout_ms", STANDARD_DEFAULT_TIMEOUT_MS)
    if timeout_ms != STANDARD_DEFAULT_TIMEOUT_MS:
        return False
    timeouts = item.get("timeouts", {})
    if timeouts not in ({}, None):
        return False
    if item.get("max_request_bytes", DEFAULT_MAX_REQUEST_BYTES) != DEFAULT_MAX_REQUEST_BYTES:
        return False
    if item.get("max_response_bytes", DEFAULT_MAX_RESPONSE_BYTES) != DEFAULT_MAX_RESPONSE_BYTES:
        return False
    bindings = item.get("capability_adapters")
    if not isinstance(bindings, dict):
        return False
    adapter_id = setup_adapter_id(organization_id, identity_id)
    expected = {capability: adapter_id for capability in capability_order()}
    return bindings == expected


def is_setup_owned_adapter(item: Mapping[str, JsonValue] | None, adapter_id: str) -> bool:
    """Return whether one authored adapter is the setup-owned http_json document.

    Args:
        item: Adapter object, or ``None`` when it is missing.
        adapter_id: Expected setup-owned adapter identity.

    Returns:
        True only when the adapter is the setup-owned http_json form.
    """
    if item is None:
        return False
    if set(item) - _SETUP_ADAPTER_KEYS:
        return False
    if item.get("adapter_id") != adapter_id or item.get("kind") != "http_json":
        return False
    if not isinstance(item.get("url"), str):
        return False
    bearer_env = item.get("bearer_env")
    if bearer_env is not None and not isinstance(bearer_env, str):
        return False
    max_response = item.get("max_response_bytes", DEFAULT_HTTP_JSON_MAX_RESPONSE_BYTES)
    return max_response == DEFAULT_HTTP_JSON_MAX_RESPONSE_BYTES


def capability_order() -> tuple[str, ...]:
    """Return required standard capabilities in documented expansion order."""
    ordered: list[str] = []
    for step in STANDARD_PRESET_STEPS:
        value = step.capability.value
        if value not in ordered:
            ordered.append(value)
    missing = STANDARD_REQUIRED_CAPABILITIES - {step.capability for step in STANDARD_PRESET_STEPS}
    if missing:
        raise ValueError("standard preset steps must include every required capability")
    return tuple(ordered)


def load_authored_document(path: Path) -> JsonObject:
    """Read and type-check the authored JSON object.

    Args:
        path: Existing guardrail file.

    Returns:
        The parsed configuration object.

    Raises:
        ValueError: The file is not a JSON object.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("gateway guardrail configuration is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("gateway guardrail configuration must be a JSON object")
    return cast(JsonObject, payload)


def empty_document() -> JsonObject:
    """Return an empty authored document used when the file is missing."""
    return {"adapters": [], "policies": []}


def is_setup_created_empty(document: JsonObject) -> bool:
    """Return whether the document has only empty setup-owned lists.

    Args:
        document: Candidate document after a setup mutation.

    Returns:
        True when a regular file created solely by setup may be deleted.
    """
    extras = set(document) - {"adapters", "policies"}
    return (
        not extras
        and not object_list(document, "adapters")
        and not object_list(document, "policies")
    )


def encode_document(document: JsonObject) -> bytes:
    """Serialize one operator-facing document without content or secrets.

    Args:
        document: Validated authored configuration.

    Returns:
        Pretty-printed UTF-8 JSON with a trailing newline.
    """
    return (json.dumps(document, indent=2) + "\n").encode("utf-8")


def object_list(document: JsonObject, key: str) -> list[JsonObject]:
    """Return one authored object list, or an empty list when absent.

    Args:
        document: Authored configuration.
        key: ``adapters`` or ``policies``.

    Returns:
        The list of objects under ``key``.

    Raises:
        ValueError: The field exists but is not a list of objects.
    """
    raw = document.get(key, [])
    if raw == [] or raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"guardrail {key} must be a list")
    items: list[JsonObject] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError(f"each guardrail {key[:-1]} must be an object")
        items.append(cast(JsonObject, item))
    return items


def policy_for(
    document: Mapping[str, JsonValue],
    organization_id: str,
    identity_id: str,
) -> JsonObject | None:
    """Return the authored policy for one organization and identity.

    Args:
        document: Authored configuration or a policies-only slice.
        organization_id: Organization to match.
        identity_id: Identity to match.

    Returns:
        The matching policy object, or ``None`` when the pair is unassigned.
    """
    matches = [
        item
        for item in object_list(cast(JsonObject, document), "policies")
        if same_identity(item, organization_id, identity_id)
    ]
    if len(matches) > 1:
        raise ValueError("guardrail policies must be unique per organization and identity")
    return matches[0] if matches else None


def adapter_for(document: Mapping[str, JsonValue], adapter_id: str) -> JsonObject | None:
    """Return the authored adapter with ``adapter_id``, if present.

    Args:
        document: Authored configuration or an adapters-only slice.
        adapter_id: Adapter identity to match.

    Returns:
        The matching adapter object, or ``None`` when it is absent.
    """
    for item in object_list(cast(JsonObject, document), "adapters"):
        if item.get("adapter_id") == adapter_id:
            return item
    return None


def same_identity(item: Mapping[str, JsonValue], organization_id: str, identity_id: str) -> bool:
    """Return whether one policy is assigned to the selected pair.

    Args:
        item: Authored policy object.
        organization_id: Organization to match.
        identity_id: Identity to match.

    Returns:
        True when both identifiers match.
    """
    return item.get("organization_id") == organization_id and item.get("identity_id") == identity_id


def referenced_adapter_ids(policies: list[JsonObject]) -> set[str]:
    """Collect adapter identities still bound by remaining policies.

    Args:
        policies: Authored policies that will remain after a mutation.

    Returns:
        Adapter IDs referenced by capability bindings or manual checks.
    """
    referenced: set[str] = set()
    for policy in policies:
        bindings = policy.get("capability_adapters")
        if isinstance(bindings, dict):
            for value in bindings.values():
                if isinstance(value, str):
                    referenced.add(value)
        checks = policy.get("checks")
        if isinstance(checks, list):
            for check in checks:
                if isinstance(check, dict):
                    adapter_id = check.get("adapter_id")
                    if isinstance(adapter_id, str):
                        referenced.add(adapter_id)
    return referenced


def _setup_owned_id(prefix: str, organization_id: str, identity_id: str) -> str:
    """Build one collision-safe setup-owned ArtifactId.

    Args:
        prefix: Readable setup-owned kind prefix.
        organization_id: Local organization in the structured subject.
        identity_id: Selected identity in the structured subject.

    Returns:
        A bounded ArtifactId derived from canonical JSON of both identifiers.
    """
    return stable_id(
        prefix,
        {
            "organization_id": organization_id,
            "identity_id": identity_id,
        },
    )


def _owned_adapter_document(adapter_id: str, plan: GuardrailSetupPlan) -> JsonObject:
    """Author one capability-aware http_json adapter for the standard pack.

    Args:
        adapter_id: Deterministic setup-owned adapter identity.
        plan: Standard-pack mutation that supplies the dedicated endpoint.

    Returns:
        An adapter object that never includes a credential literal.
    """
    authored: JsonObject = {
        "adapter_id": adapter_id,
        "kind": "http_json",
        "url": plan.classifier_url,
    }
    if plan.bearer_env is not None:
        authored["bearer_env"] = plan.bearer_env
    return authored


def _owned_policy_document(policy_id: str, adapter_id: str, plan: GuardrailSetupPlan) -> JsonObject:
    """Author the documented standard preset for the selected identity.

    Args:
        policy_id: Deterministic setup-owned policy identity.
        adapter_id: Shared adapter bound to every required capability.
        plan: Standard-pack mutation that names the organization and identity.

    Returns:
        A presence-strict standard preset with ``protected`` true.
    """
    bindings: JsonObject = {capability: adapter_id for capability in capability_order()}
    return {
        "policy_id": policy_id,
        "organization_id": plan.organization_id,
        "identity_id": plan.identity_id,
        "protected": True,
        "preset": STANDARD_PRESET_NAME,
        "timeout_ms": STANDARD_DEFAULT_TIMEOUT_MS,
        "capability_adapters": bindings,
    }


def _with_lists(
    document: JsonObject,
    adapters: list[JsonObject],
    policies: list[JsonObject],
) -> JsonObject:
    """Copy one authored document and replace only the adapter and policy lists.

    Args:
        document: Current authored configuration, including unrelated fields.
        adapters: Replacement adapter list.
        policies: Replacement policy list.

    Returns:
        A shallow copy that keeps every unrelated top-level field.
    """
    updated = dict(document)
    updated["adapters"] = adapters
    updated["policies"] = policies
    return updated


def _snapshot_guardrail_bytes(configured_path: Path) -> bytes | None:
    """Read the current target bytes, or ``None`` when no file is present.

    Args:
        configured_path: Configured path, possibly a symlink.

    Returns:
        The existing file bytes after following a symlink.

    Raises:
        ValueError: The configured path exists and is not a file.
    """
    _reject_non_file(configured_path)
    if not configured_path.is_file():
        return None
    return resolve_write_target(configured_path).read_bytes()


def _reject_non_file(path: Path) -> None:
    """Fail closed when the configured path exists but is not a regular file.

    Args:
        path: Configured guardrail path, possibly a symlink.

    Raises:
        ValueError: The path exists and is not a file.
    """
    if path.exists() and not path.is_file():
        raise ValueError("gateway guardrail configuration must be a file")
