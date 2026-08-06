"""Per-endpoint serving settings: `endpoint.toml`, next to the model's `policy.json`.

Why a file of its own rather than a key in the model's `config.toml`: that file is the world
model's BUILD configuration, rewritten by every build, and this is a live serving control an
operator (or the platform's slider) turns between builds. Keeping them apart means turning the
dial can never race a rebuild, and a rebuild can never quietly reset the dial.

Why not on `policy.json`: the policy is the optimizer's OUTPUT, and `wmo optimize route tune`
does write the dial into it. This file is the serving-side override for a policy the operator
does not want to rewrite (the common case for the platform, which serves artifacts it did not
fit). At mount time the file wins; with no file the policy is served exactly as fitted.

    # .wmo/models/support-endpoint/endpoint.toml
    cost_quality = 0.6
    log_query_embeddings = false
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import tomli_w
from pydantic import BaseModel, ConfigDict, Field

from wmo.core.files import write_text_atomic

ENDPOINT_CONFIG_FILENAME = "endpoint.toml"


class EndpointConfig(BaseModel):
    """What an operator can set per endpoint without refitting anything.

    `cost_quality` is the one dial (0.0 = max quality, 1.0 = max savings; see
    `wmo.optimize.routing.knn.apply_cost_quality`). None means "serve the policy as fitted",
    which is also what an absent file means.

    `extra="forbid"` for the same reason `PoolEntry` forbids it: a typo like `cost_qualty` must
    fail at load with the key named, not be silently ignored and leave an operator staring at an
    endpoint that ignored the dial they set.
    """

    model_config = ConfigDict(extra="forbid")

    cost_quality: float | None = Field(default=None, ge=0.0, le=1.0)
    # Whether this endpoint records the vector each request was routed on
    # (`wmo.simulation.serving.query_embeddings`). On by default because the store enables offline
    # counterfactual analysis and is bounded by rotation, so leaving it on is safe;
    # exposed here because "default on and undisableable" is not a choice an operator should be
    # denied, and because the one legitimate reason to refuse it (query text is derivable from an
    # embedding, so it is request content at rest) is a tenancy decision, not ours.
    log_query_embeddings: bool = True

    @classmethod
    def load(cls, path: Path) -> EndpointConfig:
        """Read `endpoint.toml`; a missing file is the empty config, not an error."""
        if not path.is_file():
            return cls()
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as error:
            raise ValueError(
                f"invalid endpoint config at {path}: {error}. Expected TOML with at most a "
                "`cost_quality` key between 0.0 and 1.0 and a `log_query_embeddings` "
                "boolean, and no other keys"
            ) from error
        return cls.model_validate(data)

    def save(self, path: Path) -> None:
        """Write the config atomically (a half-written dial must not be loadable)."""
        write_text_atomic(path, tomli_w.dumps(self.model_dump(exclude_none=True)))
