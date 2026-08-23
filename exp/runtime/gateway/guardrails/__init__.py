"""Identity-scoped, pluggable gateway guardrails.

Policies are looked up by authenticated organization and identity. A pair with
no assigned policy leaves the existing gateway hot path unchanged: no
classifier, no buffering, and no extra native callback. Python owns policy
lookup and replaceable classifier adapters. Adapters are reached only through
an injected internal client that cannot recurse through the public gateway
route.
"""

from exp.runtime.gateway.guardrails.classifiers import (
    BoundedSyncClassifier,
    ClassifierRegistry,
    KeywordClassifier,
    ScriptedClassifier,
)
from exp.runtime.gateway.guardrails.client import (
    DirectClassifierClient,
    GuardrailRecursionError,
    InternalClassifierClient,
    assert_not_internal_classification,
    classification_scope,
)
from exp.runtime.gateway.guardrails.config import load_guardrail_engine
from exp.runtime.gateway.guardrails.contracts import (
    ClassifierVerdict,
    GuardrailAction,
    GuardrailCapabilityKind,
    GuardrailCheck,
    GuardrailCheckStage,
    GuardrailCompletion,
    GuardrailPolicy,
    GuardrailRejected,
    GuardrailToolCall,
)
from exp.runtime.gateway.guardrails.enforcement import GuardrailEngine
from exp.runtime.gateway.guardrails.http_json import (
    ClassifierProtocolError,
    HttpJsonClassifier,
)
from exp.runtime.gateway.guardrails.preset import STANDARD_PRESET_NAME
from exp.runtime.gateway.guardrails.store import MappingGuardrailStore

__all__ = [
    "BoundedSyncClassifier",
    "ClassifierProtocolError",
    "ClassifierRegistry",
    "ClassifierVerdict",
    "DirectClassifierClient",
    "GuardrailAction",
    "GuardrailCapabilityKind",
    "GuardrailCheck",
    "GuardrailCheckStage",
    "GuardrailCompletion",
    "GuardrailEngine",
    "GuardrailPolicy",
    "GuardrailRecursionError",
    "GuardrailRejected",
    "GuardrailToolCall",
    "HttpJsonClassifier",
    "InternalClassifierClient",
    "KeywordClassifier",
    "MappingGuardrailStore",
    "STANDARD_PRESET_NAME",
    "ScriptedClassifier",
    "assert_not_internal_classification",
    "classification_scope",
    "load_guardrail_engine",
]
