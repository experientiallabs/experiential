"""Judge-quality meta-eval: labeled cases that pin how the judge must score.

The judge is itself an automated component, so it gets its own eval (AGENTS rule: improve
automated components against real data). Each `JudgeCase` is a hand-labeled (action, actual,
predicted) triple with the score band a trustworthy judge must land in — controls pin correct
behavior we must not regress, and defect cases reproduce known failure modes so a prompt or
parsing change can be *proven* to fix its target without moving the controls.

Run `run_judge_quality` with a real provider-backed judge to grade the judge; unit tests cover
the harness mechanics with a fake. Case content is modeled on real steps from the bundled
example corpora (tau-bench tool JSON, terminal-task stdout) so the meta-eval grades the judge
on the distribution it actually scores.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, Field

from wmh.core.types import Action, ActionKind, Observation, Step
from wmh.optimize.judge import Judge


class ScoreBand(BaseModel):
    """Inclusive [lo, hi] band a score must land in."""

    lo: float = 0.0
    hi: float = 1.0

    def holds(self, value: float) -> bool:
        return self.lo <= value <= self.hi


class JudgeCase(BaseModel):
    """One labeled judging scenario: inputs plus the verdict band a sound judge must produce."""

    id: str
    defect: str  # failure mode this case guards ("control" = correct behavior to preserve)
    rationale: str  # why the expected band is what it is
    action: Action
    actual: Observation
    predicted: Observation
    expected: ScoreBand
    # Optional tighter bands on individual rubric dimensions (e.g. factuality on outcome flips).
    expected_dimensions: dict[str, ScoreBand] = Field(default_factory=dict)

    def step(self) -> Step:
        """The judging context for this case (the actual observation is the recorded truth)."""
        return Step(action=self.action, observation=self.actual)


class CaseVerdict(BaseModel):
    """The judge's output on one case, checked against the case's expected bands."""

    case_id: str
    defect: str
    passed: bool
    score: float
    dimensions: dict[str, float] = Field(default_factory=dict)
    critique: str = ""
    failures: list[str] = Field(default_factory=list)  # which bands were violated, human-readable


class JudgeQualityReport(BaseModel):
    """Aggregate meta-eval result over a case suite (satisfies `wmh.evals.base.EvalResult`)."""

    verdicts: list[CaseVerdict] = Field(default_factory=list)

    @property
    def n_total(self) -> int:
        return len(self.verdicts)

    @property
    def n_passed(self) -> int:
        return sum(1 for v in self.verdicts if v.passed)

    @property
    def headline(self) -> float:
        """Pass fraction in [0, 1], so shared eval tooling can consume the meta-eval too."""
        return self.n_passed / self.n_total if self.n_total else 0.0

    def failed(self) -> list[CaseVerdict]:
        return [v for v in self.verdicts if not v.passed]

    def summary(self) -> str:
        return f"judge quality: {self.n_passed}/{self.n_total} cases passed"


def run_judge_quality(
    judge: Judge,
    cases: Sequence[JudgeCase] | None = None,
    *,
    concurrency: int = 1,
) -> JudgeQualityReport:
    """Score every case with `judge` and check each verdict against its labeled bands.

    Cases are independent single judge calls, so `concurrency > 1` runs them on a thread pool
    with order preserved; the supplied `judge` must be thread-safe under it (`RubricJudge` is —
    it keeps no per-call state). Defaults to the built-in `JUDGE_QUALITY_CASES` suite.
    """
    suite = list(JUDGE_QUALITY_CASES if cases is None else cases)

    def _grade(case: JudgeCase) -> CaseVerdict:
        result = judge.score(case.predicted, case.actual, case.step())
        failures: list[str] = []
        if not result.valid:
            # A judge failure can land in a low band by accident; it never passes a case.
            failures.append(f"judge reply invalid: {result.critique[:160]}")
        if not case.expected.holds(result.score):
            failures.append(
                f"score {result.score:.3f} outside [{case.expected.lo}, {case.expected.hi}]"
            )
        for dim, band in case.expected_dimensions.items():
            value = result.dimensions.get(dim)
            if value is None or not band.holds(value):
                shown = "missing" if value is None else f"{value:.3f}"
                failures.append(f"{dim} {shown} outside [{band.lo}, {band.hi}]")
        return CaseVerdict(
            case_id=case.id,
            defect=case.defect,
            passed=not failures,
            score=result.score,
            dimensions=result.dimensions,
            critique=result.critique,
            failures=failures,
        )

    # max_workers=1 is the sequential case; pool.map preserves order and propagates exceptions.
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        verdicts = list(pool.map(_grade, suite))
    return JudgeQualityReport(verdicts=verdicts)


# --- The labeled suite ----------------------------------------------------------------------
# Content is modeled on real steps from examples/ corpora (tau-bench flight tools, terminal-task
# bash) so bands reflect the judge's actual input distribution, not toy strings.

_RESERVATION_JSON = (
    '{"reservation_id": "EHGLP3", "user_id": "emma_kim_9957", "origin": "PHX", '
    '"destination": "JFK", "flight_type": "one_way", "cabin": "basic_economy", '
    '"flights": [{"flight_number": "HAT156", "origin": "PHX", "destination": "SEA", '
    '"date": "2024-05-17", "price": 50}, {"flight_number": "HAT021", "origin": "SEA", '
    '"destination": "JFK", "date": "2024-05-17", "price": 54}]}'
)

# Same reservation, keys reordered and re-spaced: cosmetically different, functionally identical.
_RESERVATION_JSON_REORDERED = (
    '{"user_id": "emma_kim_9957", "reservation_id": "EHGLP3", '
    '"flight_type": "one_way", "cabin": "basic_economy", "origin": "PHX", '
    '"destination": "JFK", "flights": [{"date": "2024-05-17", "flight_number": "HAT156", '
    '"origin": "PHX", "destination": "SEA", "price": 50}, {"date": "2024-05-17", '
    '"flight_number": "HAT021", "origin": "SEA", "destination": "JFK", "price": 54}]}'
)

# Same shape, fabricated data: right format, wrong facts.
_RESERVATION_JSON_FABRICATED = (
    '{"reservation_id": "EHGLP3", "user_id": "liam_chen_1122", "origin": "LAX", '
    '"destination": "BOS", "flight_type": "round_trip", "cabin": "business", '
    '"flights": [{"flight_number": "HAT902", "origin": "LAX", "destination": "DEN", '
    '"date": "2024-05-18", "price": 210}]}'
)

_VERSIONS_STDOUT = (
    "0.1.0\n0.2.0\n1.0.0\n1.1.0\n2.0.0\n2.4.2\n3.0.0\n4.0.0\n4.1.2\n5.0.0\n5.3.0\n5.6.2\n"
)
_VERSIONS_STDOUT_WRONG = (
    "0.1.0\n0.3.0\n1.0.0\n1.2.0\n2.1.0\n2.5.0\n3.1.0\n4.2.0\n4.9.9\n6.0.0\n6.3.0\n7.1.4\n"
)

_LOOKUP_ACTION = Action(
    kind=ActionKind.TOOL_CALL,
    name="get_reservation_details",
    arguments={"reservation_id": "EHGLP3"},
)
_FIND_USER_ACTION = Action(
    kind=ActionKind.TOOL_CALL,
    name="find_user_id_by_email",
    arguments={"email": "aarav.santos8321@example.com"},
)
_CAT_VERSIONS_ACTION = Action(
    kind=ActionKind.TOOL_CALL,
    name="bash",
    arguments={"command": "cat /tmp/chalk_versions.txt"},
)
_REDIRECT_ACTION = Action(
    kind=ActionKind.TOOL_CALL,
    name="bash",
    arguments={
        "command": "curl -s https://registry.npmjs.org/express"
        " | jq -r '.\"dist-tags\".latest' > /tmp/express_version.txt"
    },
)
_BACKGROUND_ACTION = Action(
    kind=ActionKind.TOOL_CALL,
    name="bash",
    arguments={"command": "nohup python server.py >/dev/null 2>&1 & echo $!"},
)

# Long-output pair: identical head, so a judge that only sees a head-truncated view cannot tell
# them apart — the divergent tail is exactly what truncation must preserve.
_LONG_LINES = [f"{i:04d} OK item-{i}" for i in range(2000)]
_LONG_STDOUT = "\n".join(_LONG_LINES)
_LONG_STDOUT_BAD_TAIL = "\n".join(
    _LONG_LINES[:1900] + [f"{i:04d} FAIL item-{i} corrupted" for i in range(1900, 2000)]
)
# Same length as _LONG_STDOUT, same head and tail, character-swapped middle: invisible to the
# truncated view, exposed only by the content_sha256 mismatch.
_LONG_STDOUT_BAD_MIDDLE = "\n".join(
    line if i < 700 or i >= 1300 else line.replace("OK", "KO") for i, line in enumerate(_LONG_LINES)
)
del _LONG_LINES  # only the joined fixtures are needed at runtime
_LIST_ACTION = Action(
    kind=ActionKind.TOOL_CALL,
    name="bash",
    arguments={"command": "./check_inventory.sh --all"},
)

JUDGE_QUALITY_CASES: tuple[JudgeCase, ...] = (
    # --- controls: correct behavior any fix must preserve ------------------------------------
    JudgeCase(
        id="exact-json-lookup",
        defect="control",
        rationale="A byte-identical prediction of deterministic tool JSON is a perfect stand-in.",
        action=_LOOKUP_ACTION,
        actual=Observation(content=_RESERVATION_JSON),
        predicted=Observation(content=_RESERVATION_JSON),
        expected=ScoreBand(lo=0.9, hi=1.0),
    ),
    JudgeCase(
        id="cosmetic-reordering",
        defect="control",
        rationale="Key order/whitespace changes carry no information; the agent acts identically.",
        action=_LOOKUP_ACTION,
        actual=Observation(content=_RESERVATION_JSON),
        predicted=Observation(content=_RESERVATION_JSON_REORDERED),
        expected=ScoreBand(lo=0.7, hi=1.0),
    ),
    JudgeCase(
        id="volatile-pid-differs",
        defect="control",
        rationale="A PID is volatile; a different-but-plausible value is a faithful simulation.",
        action=_BACKGROUND_ACTION,
        actual=Observation(content="12345\n"),
        predicted=Observation(content="48211\n"),
        expected=ScoreBand(lo=0.6, hi=1.0),
    ),
    JudgeCase(
        id="wrong-computed-values",
        defect="control",
        rationale="`cat` of an existing file is deterministic; wrong values are wrong facts.",
        action=_CAT_VERSIONS_ACTION,
        actual=Observation(content=_VERSIONS_STDOUT),
        predicted=Observation(content=_VERSIONS_STDOUT_WRONG),
        expected=ScoreBand(lo=0.0, hi=0.6),
        expected_dimensions={"factuality": ScoreBand(lo=0.0, hi=0.5)},
    ),
    JudgeCase(
        id="fabricated-lookup-data",
        defect="control",
        rationale="Right shape, wrong reservation: every salient fact the agent acts on differs.",
        action=_LOOKUP_ACTION,
        actual=Observation(content=_RESERVATION_JSON),
        predicted=Observation(content=_RESERVATION_JSON_FABRICATED),
        expected=ScoreBand(lo=0.0, hi=0.4),
        expected_dimensions={"factuality": ScoreBand(lo=0.0, hi=0.3)},
    ),
    JudgeCase(
        id="right-facts-wrong-shape",
        defect="control",
        rationale="A prose explanation carrying the right facts is not an environment response; "
        "the headline must not collapse to factuality alone.",
        action=_LOOKUP_ACTION,
        actual=Observation(content=_RESERVATION_JSON),
        predicted=Observation(
            content="The reservation EHGLP3 belongs to emma_kim_9957: a one-way basic-economy "
            "trip PHX to JFK via SEA on 2024-05-17 (HAT156 at $50, then HAT021 at $54)."
        ),
        expected=ScoreBand(lo=0.15, hi=0.75),
        expected_dimensions={"realism": ScoreBand(lo=0.0, hi=0.4)},
    ),
    # --- defect: empty-prediction guidance missing from the rubric prompt --------------------
    JudgeCase(
        id="empty-pred-nonempty-actual",
        defect="empty-prediction",
        rationale="The environment answered with data the prediction omitted entirely.",
        action=_CAT_VERSIONS_ACTION,
        actual=Observation(content=_VERSIONS_STDOUT),
        predicted=Observation(content=""),
        expected=ScoreBand(lo=0.0, hi=0.25),
    ),
    # --- defect: both-empty must be a match, not penalized -----------------------------------
    JudgeCase(
        id="empty-pred-empty-actual",
        defect="both-empty",
        rationale="Output-redirected commands legitimately print nothing; empty == empty is exact.",
        action=_REDIRECT_ACTION,
        actual=Observation(content=""),
        predicted=Observation(content=""),
        expected=ScoreBand(lo=0.85, hi=1.0),
    ),
    # --- defect: outcome flips scoring deceptively high through the unweighted mean ----------
    JudgeCase(
        id="error-flipped-to-pretty-success",
        defect="outcome-flip",
        rationale="The real environment errored; a well-formatted success is the worst failure "
        "mode because the agent proceeds on a state that does not exist.",
        action=_FIND_USER_ACTION,
        actual=Observation(content="Error: User not found", is_error=True),
        predicted=Observation(content='{"user_id": "aarav_santos_3021"}', is_error=False),
        expected=ScoreBand(lo=0.0, hi=0.35),
        expected_dimensions={"factuality": ScoreBand(lo=0.0, hi=0.2)},
    ),
    JudgeCase(
        id="success-flipped-to-error",
        defect="outcome-flip",
        rationale="The real call succeeded; predicting an error flips the outcome the agent sees.",
        action=_LOOKUP_ACTION,
        actual=Observation(content=_RESERVATION_JSON),
        predicted=Observation(content="Error: reservation not found", is_error=True),
        expected=ScoreBand(lo=0.0, hi=0.35),
        expected_dimensions={"factuality": ScoreBand(lo=0.0, hi=0.2)},
    ),
    # --- defect: very long observations (must stay gradeable, and truncation must not hide
    # --- a divergent tail) --------------------------------------------------------------------
    JudgeCase(
        id="long-output-identical",
        defect="long-observation",
        rationale="A perfect prediction of a huge output is perfect regardless of its size.",
        action=_LIST_ACTION,
        actual=Observation(content=_LONG_STDOUT),
        predicted=Observation(content=_LONG_STDOUT),
        expected=ScoreBand(lo=0.85, hi=1.0),
    ),
    JudgeCase(
        id="long-output-divergent-tail",
        defect="long-observation",
        rationale="Identical for 1900 lines then 100 corrupted lines: salient divergence lives "
        "in the tail, so the judge must still see the tail.",
        action=_LIST_ACTION,
        actual=Observation(content=_LONG_STDOUT),
        predicted=Observation(content=_LONG_STDOUT_BAD_TAIL),
        expected=ScoreBand(lo=0.0, hi=0.65),
    ),
    JudgeCase(
        id="long-output-divergent-middle",
        defect="long-observation",
        rationale="Equal length, identical head and tail, fabricated middle: truncation hides "
        "the divergence, so only the content_sha256 mismatch can expose it. The judge cannot "
        "see HOW divergent the hidden region is, so unlike the visible-tail case the label only "
        "demands it never reads as a verified match (pre-hash this scored ~1.0).",
        action=_LIST_ACTION,
        actual=Observation(content=_LONG_STDOUT),
        predicted=Observation(content=_LONG_STDOUT_BAD_MIDDLE),
        expected=ScoreBand(lo=0.0, hi=0.65),
        expected_dimensions={"factuality": ScoreBand(lo=0.0, hi=0.5)},
    ),
)
