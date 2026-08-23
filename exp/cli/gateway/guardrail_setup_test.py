"""Tests for interactive setup authoring of the standard guardrail pack."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pytest

from exp.cli.gateway.guardrail_setup import (
    GUARDRAILS_CUSTOM,
    GUARDRAILS_OFF,
    GUARDRAILS_STANDARD,
    GuardrailSetupMode,
    GuardrailSetupPlan,
    apply_setup_guardrails,
    collect_guardrail_setup,
    guardrail_config_path,
    guardrail_setup_compensation,
    inspect_setup_guardrails,
    setup_adapter_id,
    setup_policy_id,
)
from exp.cli.shared.picker_test import ScriptedConsole
from exp.common.core.artifacts import validate_artifact_id
from exp.common.core.locks import file_write_lock
from exp.runtime.gateway.guardrails.config import engine_from_document, load_guardrail_engine
from exp.runtime.gateway.guardrails.preset import STANDARD_DEFAULT_TIMEOUT_MS, STANDARD_PRESET_STEPS
from exp.runtime.gateway.sqlite.alias_activation import AliasActivationOutcomeUnknownError

_ORG = "local"
_IDENTITY = "default"
_CLASSIFIER_URL = "https://classifier.example.invalid/v1/inspect"
_BEARER_ENV = "CLASSIFIER_BEARER"
_SECRET = "sk-super-secret-literal"


def _plan(
    *,
    action: Literal["off", "standard"] = "standard",
    identity_id: str = _IDENTITY,
    classifier_url: str = _CLASSIFIER_URL,
    bearer_env: str | None = _BEARER_ENV,
    replace_custom: bool = False,
) -> GuardrailSetupPlan:
    """Build one setup-owned mutation for tests."""
    return GuardrailSetupPlan(
        action=action,
        organization_id=_ORG,
        identity_id=identity_id,
        classifier_url=classifier_url,
        bearer_env=bearer_env,
        replace_custom=replace_custom,
    )


def _write_document(root: Path, document: object) -> Path:
    """Write one authored guardrail document under ``root``."""
    path = guardrail_config_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path


def _custom_document(*, identity_id: str = _IDENTITY) -> dict[str, object]:
    """Return a hand-authored policy that setup must not treat as owned."""
    return {
        "adapters": [
            {
                "adapter_id": "keyword-safety",
                "kind": "keyword",
                "needles": ["example-disallowed-phrase"],
            }
        ],
        "policies": [
            {
                "policy_id": "strict-member",
                "organization_id": _ORG,
                "identity_id": identity_id,
                "protected": True,
                "checks": [
                    {
                        "check_id": "input-safety",
                        "capability": "content_safety",
                        "stage": "input",
                        "action": "block",
                        "timeout_ms": 250,
                        "adapter_id": "keyword-safety",
                    }
                ],
            }
        ],
    }


def test_setup_owned_ids_are_deterministic_valid_artifact_ids() -> None:
    """Setup IDs stay scoped to organization and identity and validate as ArtifactIds."""
    adapter_id = setup_adapter_id(_ORG, _IDENTITY)
    policy_id = setup_policy_id(_ORG, "operator")
    assert adapter_id == "setup-http-json-local-default"
    assert policy_id == "setup-standard-local-operator"
    assert validate_artifact_id(adapter_id) == adapter_id
    assert validate_artifact_id(policy_id) == policy_id
    assert setup_adapter_id(_ORG, "operator") != adapter_id


def test_missing_file_is_off_and_disable_does_not_create_it(tmp_path: Path) -> None:
    """Default-off: inspect reports Off and an Off plan leaves the unguarded missing file."""
    inspection = inspect_setup_guardrails(tmp_path, _ORG, _IDENTITY)
    assert inspection.mode is GuardrailSetupMode.OFF
    assert inspection.display == GUARDRAILS_OFF
    apply_setup_guardrails(tmp_path, _plan(action="off", bearer_env=None))
    assert not guardrail_config_path(tmp_path).exists()
    assert load_guardrail_engine(tmp_path) is None


def test_explicit_standard_opt_in_authors_one_shared_adapter(tmp_path: Path) -> None:
    """Enabled setup writes the standard pack with one adapter bound to four capabilities."""
    apply_setup_guardrails(tmp_path, _plan())
    path = guardrail_config_path(tmp_path)
    document = json.loads(path.read_text(encoding="utf-8"))
    adapter_id = setup_adapter_id(_ORG, _IDENTITY)
    assert [item["adapter_id"] for item in document["adapters"]] == [adapter_id]
    policy = document["policies"][0]
    assert policy["policy_id"] == setup_policy_id(_ORG, _IDENTITY)
    assert policy["organization_id"] == _ORG
    assert policy["identity_id"] == _IDENTITY
    assert policy["protected"] is True
    assert policy["preset"] == "standard"
    assert policy["timeout_ms"] == STANDARD_DEFAULT_TIMEOUT_MS
    assert "checks" not in policy
    assert set(policy["capability_adapters"]) == {
        "pii",
        "secret_leakage",
        "prompt_injection",
        "content_safety",
    }
    assert set(policy["capability_adapters"].values()) == {adapter_id}
    engine = load_guardrail_engine(tmp_path)
    assert engine is not None
    loaded = engine.policy_for(_ORG, _IDENTITY)
    assert loaded is not None
    assert loaded.protected is True
    assert len(loaded.checks) == len(STANDARD_PRESET_STEPS)
    assert {check.adapter_id for check in loaded.checks} == {adapter_id}
    inspection = inspect_setup_guardrails(tmp_path, _ORG, _IDENTITY)
    assert inspection.mode is GuardrailSetupMode.STANDARD
    assert inspection.display == GUARDRAILS_STANDARD
    assert inspection.classifier_url == _CLASSIFIER_URL
    assert inspection.bearer_env == _BEARER_ENV


def test_standard_opt_in_stores_environment_name_never_a_secret(tmp_path: Path) -> None:
    """The authored file records bearer_env and never a credential literal."""
    apply_setup_guardrails(tmp_path, _plan())
    payload = guardrail_config_path(tmp_path).read_text(encoding="utf-8")
    assert _BEARER_ENV in payload
    assert _SECRET not in payload
    assert "bearer" not in json.loads(payload)["adapters"][0]
    assert json.loads(payload)["adapters"][0]["bearer_env"] == _BEARER_ENV


def test_standard_opt_in_can_omit_bearer_env(tmp_path: Path) -> None:
    """An enabled setup without a bearer name omits the field instead of storing empty text."""
    apply_setup_guardrails(tmp_path, _plan(bearer_env=None))
    adapter = json.loads(guardrail_config_path(tmp_path).read_text(encoding="utf-8"))["adapters"][0]
    assert "bearer_env" not in adapter


def test_collect_enter_preserves_missing_and_existing_files(tmp_path: Path) -> None:
    """Pressing Enter does not author a plan, so the current file stays untouched."""
    console = ScriptedConsole("")
    missing = collect_guardrail_setup(
        console=console,
        root=tmp_path,
        organization_id=_ORG,
        identity_id=_IDENTITY,
        edit=False,
    )
    assert missing.plan is None
    assert missing.display == GUARDRAILS_OFF
    apply_setup_guardrails(tmp_path, _plan())
    before = guardrail_config_path(tmp_path).read_bytes()
    preserved = collect_guardrail_setup(
        console=ScriptedConsole(""),
        root=tmp_path,
        organization_id=_ORG,
        identity_id=_IDENTITY,
        edit=False,
    )
    assert preserved.plan is None
    assert preserved.display == GUARDRAILS_STANDARD
    assert guardrail_config_path(tmp_path).read_bytes() == before


def test_collect_edit_can_opt_in_and_disable(tmp_path: Path) -> None:
    """Edit collects a short on/off choice plus classifier URL and optional env name."""
    enabled = collect_guardrail_setup(
        console=ScriptedConsole(f"on\n{_CLASSIFIER_URL}\n{_BEARER_ENV}\n"),
        root=tmp_path,
        organization_id=_ORG,
        identity_id=_IDENTITY,
        edit=True,
    )
    assert enabled.display == GUARDRAILS_STANDARD
    assert enabled.plan == _plan()
    disabled = collect_guardrail_setup(
        console=ScriptedConsole("off\n"),
        root=tmp_path,
        organization_id=_ORG,
        identity_id=_IDENTITY,
        edit=True,
    )
    assert disabled.display == GUARDRAILS_OFF
    assert disabled.plan is not None
    assert disabled.plan.action == "off"


def test_custom_policy_is_preserved_unless_replace_is_typed(tmp_path: Path) -> None:
    """A hand-authored policy is Custom/preserved and is not overwritten by a normal opt-in."""
    original = _custom_document()
    _write_document(tmp_path, original)
    inspection = inspect_setup_guardrails(tmp_path, _ORG, _IDENTITY)
    assert inspection.mode is GuardrailSetupMode.CUSTOM
    assert inspection.display == GUARDRAILS_CUSTOM
    keep = collect_guardrail_setup(
        console=ScriptedConsole("keep\n"),
        root=tmp_path,
        organization_id=_ORG,
        identity_id=_IDENTITY,
        edit=True,
    )
    assert keep.plan is None
    assert keep.display == GUARDRAILS_CUSTOM
    with pytest.raises(ValueError, match="custom/preserved"):
        apply_setup_guardrails(tmp_path, _plan())
    assert json.loads(guardrail_config_path(tmp_path).read_text(encoding="utf-8")) == original
    apply_setup_guardrails(tmp_path, _plan(replace_custom=True))
    inspection = inspect_setup_guardrails(tmp_path, _ORG, _IDENTITY)
    assert inspection.mode is GuardrailSetupMode.STANDARD
    document = json.loads(guardrail_config_path(tmp_path).read_text(encoding="utf-8"))
    assert any(item["adapter_id"] == "keyword-safety" for item in document["adapters"])
    assert all(item["policy_id"] != "strict-member" for item in document["policies"])


def test_unrelated_policies_and_adapters_are_preserved(tmp_path: Path) -> None:
    """Setup mutates only the selected identity and leaves other artifacts in place."""
    other = _custom_document(identity_id="operator")
    _write_document(tmp_path, other)
    apply_setup_guardrails(tmp_path, _plan())
    document = json.loads(guardrail_config_path(tmp_path).read_text(encoding="utf-8"))
    adapter_ids = {item["adapter_id"] for item in document["adapters"]}
    identities = {item["identity_id"] for item in document["policies"]}
    assert adapter_ids == {"keyword-safety", setup_adapter_id(_ORG, _IDENTITY)}
    assert identities == {"operator", _IDENTITY}
    apply_setup_guardrails(tmp_path, _plan(action="off", bearer_env=None))
    remaining = json.loads(guardrail_config_path(tmp_path).read_text(encoding="utf-8"))
    assert remaining == other
    assert inspect_setup_guardrails(tmp_path, _ORG, _IDENTITY).mode is GuardrailSetupMode.OFF
    assert inspect_setup_guardrails(tmp_path, _ORG, "operator").mode is GuardrailSetupMode.CUSTOM


def test_disable_owned_standard_deletes_an_emptied_file(tmp_path: Path) -> None:
    """Turning off the last setup-owned pack restores the missing-file hot path."""
    apply_setup_guardrails(tmp_path, _plan())
    apply_setup_guardrails(tmp_path, _plan(action="off", bearer_env=None))
    assert not guardrail_config_path(tmp_path).exists()
    assert load_guardrail_engine(tmp_path) is None


def test_malformed_existing_config_fails_closed(tmp_path: Path) -> None:
    """A present but invalid file is never treated as Off and is never overwritten blindly."""
    path = _write_document(tmp_path, ["not-an-object"])
    before = path.read_bytes()
    with pytest.raises(ValueError, match="JSON object"):
        inspect_setup_guardrails(tmp_path, _ORG, _IDENTITY)
    with pytest.raises(ValueError, match="JSON object"):
        apply_setup_guardrails(tmp_path, _plan())
    assert path.read_bytes() == before


def test_malformed_json_fails_closed(tmp_path: Path) -> None:
    """Corrupt JSON cannot be classified as default-off."""
    path = guardrail_config_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        inspect_setup_guardrails(tmp_path, _ORG, _IDENTITY)


def test_compensation_restores_the_previous_file_on_proven_failure(tmp_path: Path) -> None:
    """A proven setup failure puts the previous guardrail bytes back."""
    apply_setup_guardrails(tmp_path, _plan())
    before = guardrail_config_path(tmp_path).read_bytes()
    with pytest.raises(RuntimeError, match="catalog failed"):
        with guardrail_setup_compensation(tmp_path, _plan(action="off", bearer_env=None)):
            assert not guardrail_config_path(tmp_path).exists()
            raise RuntimeError("catalog failed")
    assert guardrail_config_path(tmp_path).read_bytes() == before
    assert inspect_setup_guardrails(tmp_path, _ORG, _IDENTITY).mode is GuardrailSetupMode.STANDARD


def test_compensation_keeps_selected_file_when_activation_outcome_is_unknown(
    tmp_path: Path,
) -> None:
    """An indeterminate commit keeps the selected guardrail file next to authority."""
    with pytest.raises(AliasActivationOutcomeUnknownError, match="operation_outcome_unknown"):
        with guardrail_setup_compensation(tmp_path, _plan()):
            raise AliasActivationOutcomeUnknownError(
                alias_id="gpt-5-6-luna",
                revision_id="revision-unknown",
            )
    assert inspect_setup_guardrails(tmp_path, _ORG, _IDENTITY).mode is GuardrailSetupMode.STANDARD


def test_compensation_does_not_create_a_file_when_preserving(tmp_path: Path) -> None:
    """A preserve plan writes nothing, including when later setup steps fail."""
    with pytest.raises(RuntimeError, match="later step failed"):
        with guardrail_setup_compensation(tmp_path, None):
            raise RuntimeError("later step failed")
    assert not guardrail_config_path(tmp_path).exists()


def test_apply_writes_through_a_symlink(tmp_path: Path) -> None:
    """A configured symlink keeps pointing at the operator's target file."""
    target = tmp_path / "shared" / "guardrails.json"
    configured = guardrail_config_path(tmp_path)
    configured.parent.mkdir(parents=True, exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    configured.symlink_to(target)
    apply_setup_guardrails(tmp_path, _plan())
    assert configured.is_symlink()
    assert target.is_file()
    assert not target.is_symlink()
    assert inspect_setup_guardrails(tmp_path, _ORG, _IDENTITY).mode is GuardrailSetupMode.STANDARD


def test_apply_takes_the_write_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Read-modify-write holds the same file lock used by other gateway config writes."""
    held: list[str] = []
    real_lock = file_write_lock

    def _capture(path: Path, *, what: str, timeout_s: float = 10.0) -> object:
        """Record the lock noun and delegate to the real lock."""
        held.append(what)
        return real_lock(path, what=what, timeout_s=timeout_s)

    monkeypatch.setattr("exp.cli.gateway.guardrail_setup.file_write_lock", _capture)
    apply_setup_guardrails(tmp_path, _plan())
    assert "gateway guardrail configuration" in held


def test_non_owned_standard_looking_policy_is_custom(tmp_path: Path) -> None:
    """A standard preset that is not the setup-owned form is preserved, not rewritten."""
    adapter_id = "hosted-classifier"
    _write_document(
        tmp_path,
        {
            "adapters": [
                {
                    "adapter_id": adapter_id,
                    "kind": "http_json",
                    "url": _CLASSIFIER_URL,
                }
            ],
            "policies": [
                {
                    "policy_id": "hand-authored-standard",
                    "organization_id": _ORG,
                    "identity_id": _IDENTITY,
                    "protected": True,
                    "preset": "standard",
                    "timeout_ms": STANDARD_DEFAULT_TIMEOUT_MS,
                    "capability_adapters": {
                        "pii": adapter_id,
                        "secret_leakage": adapter_id,
                        "prompt_injection": adapter_id,
                        "content_safety": adapter_id,
                    },
                }
            ],
        },
    )
    inspection = inspect_setup_guardrails(tmp_path, _ORG, _IDENTITY)
    assert inspection.mode is GuardrailSetupMode.CUSTOM
    with pytest.raises(ValueError, match="custom/preserved"):
        apply_setup_guardrails(tmp_path, _plan())
    engine = engine_from_document(
        json.loads(guardrail_config_path(tmp_path).read_text(encoding="utf-8"))
    )
    policy = engine.policy_for(_ORG, _IDENTITY)
    assert policy is not None
    assert policy.policy_id == "hand-authored-standard"


def test_collect_rejects_ambiguous_custom_answers(tmp_path: Path) -> None:
    """Custom policies require an explicit keep or replace; other words fail closed."""
    _write_document(tmp_path, _custom_document())
    with pytest.raises(ValueError, match="keep"):
        collect_guardrail_setup(
            console=ScriptedConsole("wipe\n"),
            root=tmp_path,
            organization_id=_ORG,
            identity_id=_IDENTITY,
            edit=True,
        )
