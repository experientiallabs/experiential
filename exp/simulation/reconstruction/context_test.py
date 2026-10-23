"""Generated prior knowledge cannot masquerade as observed or published native evidence."""

import pytest

from exp.common.core.artifacts import JsonObject, sha256_json
from exp.simulation.reconstruction.models import (
    Claim,
    Interface,
    RawEpisode,
    RawInputs,
    ReconstructionBundle,
    SourceRef,
    ToolContract,
)


@pytest.mark.parametrize("kind", ["rule", "research"])
@pytest.mark.parametrize("status", ["observed", "documented", "contradicted"])
def test_prior_hypotheses_do_not_gain_original_or_documentary_authority(
    kind: str, status: str
) -> None:
    """Valid JSON pointers into generated text still cannot support native-truth labels."""
    context: JsonObject = {
        "selected": [
            {
                "entry": {
                    "entry_id": "prior",
                    "kind": kind,
                    "disposition": "accepted" if kind == "rule" else "rejected",
                },
                "payload": {"rule": "Unverified prior idea", "content": "Rejected investigation"},
            }
        ]
    }
    raw = RawInputs(
        episodes=(RawEpisode(id="target", records=({"action": "read", "reply": None},)),),
        prior_context=context,
    )
    source_id = "prior:" + kind + ":prior"
    assert raw.description is None and "description" not in raw.source_map()
    assert source_id in raw.source_map()
    with pytest.raises(ValueError):
        ReconstructionBundle(
            artifact_id="candidate",
            sources=raw,
            inputs_sha256=sha256_json(raw),
            interface=Interface(
                tools=(
                    ToolContract(name="read", description="Read", parameters={"type": "object"}),
                ),
                state_schema={"type": "object"},
            ),
            claims=(
                Claim.model_validate(
                    {
                        "statement": "Prior proposal establishes native behavior",
                        "status": status,
                        "source_refs": [
                            SourceRef(source_id=source_id, pointer="/payload").model_dump()
                        ],
                        "scope": "unseen target",
                    }
                ),
            ),
            text_rules="Remain uncertain",
        )
