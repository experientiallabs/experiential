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

    [representation]
    compressor_id = "llmlingua2"
    aggressiveness = 0.4
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import tomli_w
from pydantic import BaseModel, ConfigDict, Field

from wmo.core.types import Representation

ENDPOINT_CONFIG_FILENAME = "endpoint.toml"


def _describe(representation: Representation | None) -> str:
    """One side of a representation comparison, in words an operator can act on."""
    if representation is None:
        return "uncompressed requests"
    return (
        f"compressor '{representation.compressor_id}' at aggressiveness "
        f"{representation.aggressiveness:g}"
    )


class EndpointConfig(BaseModel):
    """What an operator can set per endpoint without refitting anything.

    `cost_quality` is the one dial (0.0 = max quality, 1.0 = max savings; see
    `wmo.optimize.knn.apply_cost_quality`). None means "serve the policy as fitted", which is
    also what an absent file means.

    `extra="forbid"` for the same reason `PoolEntry` forbids it: a typo like `cost_qualty` must
    fail at load with the key named, not be silently ignored and leave an operator staring at an
    endpoint that ignored the dial they set.
    """

    model_config = ConfigDict(extra="forbid")

    cost_quality: float | None = Field(default=None, ge=0.0, le=1.0)
    # The compression this endpoint puts requests through before they reach a model, when it
    # compresses at all. Declared here rather than derived, because it is a property of the
    # serving stack in front of the policy, not of the artifact; `check_representation` is what
    # makes the declaration load-bearing. Absent means uncompressed.
    representation: Representation | None = None

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
                "`cost_quality` key between 0.0 and 1.0 and a `[representation]` table "
                "(compressor_id, aggressiveness), and no other keys"
            ) from error
        return cls.model_validate(data)

    def check_representation(self, fitted: Representation | None, *, endpoint: str) -> None:
        """Refuse to serve a policy fitted in a different representation than this endpoint uses.

        The failure this prevents is silent: a policy fitted on compressed prompts and served
        uncompressed (or the reverse) still returns a model for every request, and the evidence it
        returns it from was measured over text the request no longer resembles. Nothing downstream
        can detect that, so it is caught here, at the one point where both sides are known.

        Both absent is the uncompressed case and always valid. Anything else has to match on both
        the compressor and its setting, since the same compressor at two aggressiveness levels
        produces two different representations.

        Raises:
            ValueError: The two sides disagree, naming what each one says.
        """
        if fitted == self.representation:
            return
        raise ValueError(
            f"endpoint {endpoint!r} is configured for {_describe(self.representation)} but its "
            f"policy was fitted on {_describe(fitted)}. The policy's evidence was measured over "
            "text this endpoint does not produce, so its routing would be read off neighbors the "
            "request cannot be compared with. Refit the policy on a matrix captured in this "
            "endpoint's representation, or change the endpoint's `[representation]` to match the "
            "policy."
        )

    def save(self, path: Path) -> None:
        """Write the config atomically (a half-written dial must not be loadable)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        staging = path.with_name(f"{path.name}.partial")
        with staging.open("wb") as handle:
            tomli_w.dump(self.model_dump(exclude_none=True), handle)
        staging.replace(path)
