"""The containment gate: global win-vs-baseline evidence that vetoes a neighborhood's pick.

The kNN router (`wmo.optimize.policy.knn_decision`) is entirely LOCAL: it reads the reward
profile over a query's nearest fit scenarios and leaves the baseline when a paired guard says
those neighbors support it. That is exactly what makes it strong on an iid split and exactly what
makes it fragile on a thin bank: a 24-scenario test set can hand the same 12 confident neighbors
to a model that loses to the baseline GLOBALLY, and the local guard has no way to know.

This module is the second opinion. For every non-baseline pool model it fits one RBF-SVM over the
whole fit split (the bank's own embeddings) predicting "does this model beat the baseline on a
query like this one", and serve time requires the neighborhood's pick to also clear
`DRIFT_GATE_P_WIN` under that global model. Local evidence proposes, global evidence may veto: the
gate can only ever move a request back to the baseline, never away from it, so the worst case it
can produce is the baseline's own behavior.

Measured (R3's `r3b-hybrid-svm`, ported here and re-gated by
`.agents/scripts/validate_drift_gate.py`): on `financebench-s80`, an 80-scenario bank where the
ungated champion loses 1.85 accuracy points and gives up 11.15 points on its worst seed, the gated
router is EXACTLY the baseline on every seed. The autopsy is the whole point of the mechanism: the
champion was certifying half its test queries to a model whose fit-wide delta against the baseline
is negative, off neighborhoods that happened to look good. On `routerbench-ours9` (1199 scenarios,
the regime the champion was validated in) the gate keeps the win, trading some of the champion's
cost saving for it, because vetoed picks revert to a pricier baseline.

That trade is why the gate is not unconditional. `wmo optimize route fit --drift-gate auto`, the
default, turns it ON below `DRIFT_GATE_AUTO_MAX_SCENARIOS` fit scenarios and OFF above: a large
bank is the regime the champion was measured in and does not need containing, a small one is where
neighborhoods stop being evidence.

Persistence is a sibling `.npz` beside the policy's bank, never a pickle. A fitted `SVC` is not a
portable artifact (pickles execute on load and break across sklearn versions), so the fit extracts
the decision function into plain arrays (support vectors, dual coefficients, intercept, kernel
width) plus the calibrator's two sigmoid parameters, and `DriftGate.p_win` evaluates them in
numpy. `fit_drift_gate` proves the extraction against sklearn's own `predict_proba` before it
returns, so an artifact that loads is an artifact that scores identically to the estimator that
produced it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

import numpy as np
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

if TYPE_CHECKING:
    # Runtime-free: `wmo.optimize.policy` imports `DriftGate` from here to load and serve it, so
    # importing the bank back at module scope would be a real cycle. Nothing here constructs a
    # bank, it only reads one the caller already has.
    from wmo.optimize.policy import KnnBank

logger = logging.getLogger(__name__)

# The sidecar suffix a fit DERIVES from its policy path, appended for the same injectivity reason
# `KNN_BANK_SUFFIX` is (`support.json` -> `support.json.gate.npz`).
DRIFT_GATE_SUFFIX = ".gate.npz"

# A pick must clear this win probability to survive the gate. R3's threshold, unparameterized: the
# calibrated classifier already answers "is this model more likely than not to beat the baseline
# here", and a tunable bar would just be a second confidence knob competing with `knn_z`.
DRIFT_GATE_P_WIN = 0.5

# Decisive fit scenarios (both models scored, and not a tie) a model needs before it gets a
# classifier at all. Below it the model is left UNFITTED, which reads as p_win 0 and vetoes every
# pick of it: on that little evidence "this model beats the baseline on queries like yours" is not
# a claim the fit split can support, and the baseline is the safe answer. R3's floor.
DRIFT_GATE_MIN_DECISIVE = 12

# `--drift-gate auto` resolves ON below this many fit scenarios. Not a measured cliff: it is the
# boundary between the two regimes that WERE measured (financebench-s80 at 80 scenarios, where the
# ungated champion needs containing, and routerbench-ours9 at 1199, where it does not), placed at
# a round number between them and stated as such wherever it is printed.
DRIFT_GATE_AUTO_MAX_SCENARIOS = 200

# Reward differences below this are a tie, not a win: ties carry no signal about which model is
# better, so they are dropped rather than labeled arbitrarily. R3's `abs(d) > 1e-9`.
_DECISIVE_EPSILON = 1e-9

# Scenarios the RARER class needs before a boundary is fitted at all; below it the model gets its
# empirical win rate as a constant instead. Two is the floor a stratified 2-fold calibration can
# work with, and it is also the honest one: an RBF boundary drawn around a single counter-example
# is a memorized point, which is the overfitting this whole module exists to contain. The constant
# is not a weaker answer on such lopsided evidence, it is a more faithful one ("this model won 11
# of its 12 decisive scenarios" is a global claim, and the gate asks a global question).
_MIN_PER_CLASS = 2

# How closely the extracted arrays must reproduce sklearn's own `predict_proba` on the training
# rows before `fit_drift_gate` will return the artifact. Generous next to the float64 agreement
# actually observed (~1e-16) because the check exists to catch a STRUCTURAL break (an sklearn
# release that changes the calibrator's parameterization), not to police rounding.
_EXTRACTION_TOLERANCE = 1e-6

# Per-model classifier state, as `DriftGate.kinds` records it.
_KIND_UNFITTED = 0  # too little decisive evidence: p_win is 0, so every pick is vetoed
_KIND_CONSTANT = 1  # one class only, so no boundary exists: p_win is that class, everywhere
_KIND_SVM = 2  # a fitted, calibrated RBF decision function


@dataclass(frozen=True)
class DriftGate:
    """Per-model win-vs-baseline classifiers, as the `.npz` sidecar persists them.

    One row per pool model, aligned to `models` (which mirrors the bank's model order, so a gate
    and the bank it was fitted from index alike). `kinds` says how each model answers: an SVM
    evaluates the arrays below, a constant returns `constants`, and an unfitted model returns 0.

    Support vectors for every model live in ONE concatenated `support` matrix sliced by
    `offsets`, rather than an array per model. A model name is operator-chosen text and would
    have to be mangled into an `.npz` key to work as one; an offset table keeps the artifact's
    key set fixed and its arrays dense, which is also what lets a load validate shapes rather
    than trust names.
    """

    models: list[str]
    kinds: np.ndarray  # (models,) int8, one of the _KIND_* constants
    constants: np.ndarray  # (models,) float64, the p_win of a single-class model
    offsets: np.ndarray  # (models + 1,) int64, slice bounds into `support` and `dual`
    support: np.ndarray  # (total support vectors, dim) float32
    dual: np.ndarray  # (total support vectors,) float64, the signed dual coefficients
    intercept: np.ndarray  # (models,) float64
    gamma: np.ndarray  # (models,) float64, the RBF width the fit resolved
    sigmoid_a: np.ndarray  # (models,) float64, calibrator slope
    sigmoid_b: np.ndarray  # (models,) float64, calibrator intercept

    def __post_init__(self) -> None:
        count = len(self.models)
        if not count:
            raise ValueError("a drift gate needs at least one model")
        for name, array in (
            ("kinds", self.kinds),
            ("constants", self.constants),
            ("intercept", self.intercept),
            ("gamma", self.gamma),
            ("sigmoid_a", self.sigmoid_a),
            ("sigmoid_b", self.sigmoid_b),
        ):
            if array.shape != (count,):
                raise ValueError(
                    f"{name} has shape {array.shape}, expected ({count},) from the model list"
                )
        if self.offsets.shape != (count + 1,):
            raise ValueError(
                f"offsets has shape {self.offsets.shape}, expected ({count + 1},) from the "
                "model list"
            )
        if self.support.ndim != 2 or self.support.shape[0] != int(self.offsets[-1]):
            raise ValueError(
                f"support has shape {self.support.shape}, expected "
                f"({int(self.offsets[-1])}, dim) from the offset table"
            )
        if self.dual.shape != (self.support.shape[0],):
            raise ValueError(
                f"dual has shape {self.dual.shape}, expected ({self.support.shape[0]},) from the "
                "support matrix"
            )

    @property
    def dim(self) -> int:
        return int(self.support.shape[1])

    def p_win(self, model: str, query: np.ndarray) -> float:
        """This model's calibrated probability of beating the baseline on `query`.

        `query` is the L2-normalized request embedding the bank is searched with, so the gate
        reads the SAME representation the neighbors were retrieved over. A model the fit could
        not support scores 0, which is a veto at any threshold.
        """
        try:
            index = self.models.index(model)
        except ValueError:
            # Not a gated model at all. 0 rather than a raise: an unknown name reaching here means
            # the pool grew past the gate, and containing that pick is the safe reading.
            return 0.0
        kind = int(self.kinds[index])
        if kind == _KIND_UNFITTED:
            return 0.0
        if kind == _KIND_CONSTANT:
            return float(self.constants[index])
        start, stop = int(self.offsets[index]), int(self.offsets[index + 1])
        vectors = self.support[start:stop].astype(np.float64)
        point = np.asarray(query, dtype=np.float64)
        # ||x - sv||^2 expanded, clipped at 0: the expansion can go slightly negative on
        # near-identical vectors, and a negative squared distance would make the kernel explode.
        squared = (point**2).sum() + (vectors**2).sum(axis=1) - 2.0 * (vectors @ point)
        kernel = np.exp(-float(self.gamma[index]) * np.maximum(squared, 0.0))
        decision = float(kernel @ self.dual[start:stop]) + float(self.intercept[index])
        return _sigmoid(decision, float(self.sigmoid_a[index]), float(self.sigmoid_b[index]))

    def save(self, path: Path) -> None:
        """Write the sidecar atomically, staged under a name unique to this call.

        Same discipline (and the same reason) as `KnnBank.save`: two fits racing on one `--out`
        derive the same gate path, and a shared staging file would let one publish the other's
        classifiers under its own name.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        staging = path.with_name(f".{path.name}.{uuid4().hex}.partial")
        try:
            with staging.open("wb") as handle:
                np.savez(
                    handle,
                    models=np.asarray(self.models),
                    kinds=self.kinds.astype(np.int8),
                    constants=self.constants.astype(np.float64),
                    offsets=self.offsets.astype(np.int64),
                    support=self.support.astype(np.float32),
                    dual=self.dual.astype(np.float64),
                    intercept=self.intercept.astype(np.float64),
                    gamma=self.gamma.astype(np.float64),
                    sigmoid_a=self.sigmoid_a.astype(np.float64),
                    sigmoid_b=self.sigmoid_b.astype(np.float64),
                )
            staging.replace(path)
        except BaseException:
            staging.unlink(missing_ok=True)
            raise

    @classmethod
    def load(cls, path: Path) -> DriftGate:
        with np.load(path, allow_pickle=False) as data:
            return cls(
                models=[str(name) for name in data["models"]],
                kinds=np.asarray(data["kinds"], dtype=np.int8),
                constants=np.asarray(data["constants"], dtype=np.float64),
                offsets=np.asarray(data["offsets"], dtype=np.int64),
                support=np.asarray(data["support"], dtype=np.float32),
                dual=np.asarray(data["dual"], dtype=np.float64),
                intercept=np.asarray(data["intercept"], dtype=np.float64),
                gamma=np.asarray(data["gamma"], dtype=np.float64),
                sigmoid_a=np.asarray(data["sigmoid_a"], dtype=np.float64),
                sigmoid_b=np.asarray(data["sigmoid_b"], dtype=np.float64),
            )


DriftGateSetting = Literal["auto", "on", "off"]


def drift_gate_enabled(setting: DriftGateSetting, fit_scenarios: int) -> bool:
    """Resolve `--drift-gate auto|on|off` against the size of the bank being fitted.

    `auto` is the default and turns the gate ON below `DRIFT_GATE_AUTO_MAX_SCENARIOS` fit
    scenarios, because that is the regime where a neighborhood stops being independent evidence
    and the ungated champion was measured LOSING (see the module docstring). Above it the bank is
    the regime the champion was validated in, and the gate would only trade away its cost saving.
    """
    if setting == "on":
        return True
    if setting == "off":
        return False
    return fit_scenarios < DRIFT_GATE_AUTO_MAX_SCENARIOS


def drift_gate_path_for(policy_path: Path) -> Path:
    """The containment sidecar that belongs to one policy file.

    `models/support.json` -> `models/support.json.gate.npz`, beside that policy's bank. One owner
    for the derivation, for the reason `knn_bank_path_for` documents.
    """
    return policy_path.with_name(f"{policy_path.name}{DRIFT_GATE_SUFFIX}")


def _sigmoid(decision: float, slope: float, intercept: float) -> float:
    """The calibrator's `1 / (1 + exp(a * decision + b))`, evaluated without overflowing.

    `exp` of a large positive exponent overflows to inf and warns; the algebraically identical
    `exp(x) / (1 + exp(x))` form is stable there, so each branch uses whichever exponent is
    negative.
    """
    exponent = slope * decision + intercept
    if exponent >= 0.0:
        decayed = float(np.exp(-exponent))
        return decayed / (1.0 + decayed)
    return 1.0 / (1.0 + float(np.exp(exponent)))


def _resolved_gamma(vectors: np.ndarray) -> float:
    """What `gamma="scale"` resolves to for this training matrix.

    Recomputed rather than read off the estimator's private `_gamma`, so the artifact does not
    depend on an sklearn internal; `fit_drift_gate`'s extraction check is what proves the two
    agree. sklearn falls back to 1.0 on a zero-variance matrix, and so does this.
    """
    variance = float(vectors.var())
    if variance <= 0.0:
        return 1.0
    return 1.0 / (vectors.shape[1] * variance)


def _labels_for(
    bank: KnnBank, model_index: int, baseline_index: int
) -> tuple[np.ndarray, np.ndarray]:
    """Win-vs-baseline training rows for one model: its embeddings and 0/1 labels.

    A fit scenario contributes only when BOTH the model and the baseline were scored on it (a
    reward cell is NaN otherwise), and only when their mean rewards actually differ: a tie says
    nothing about which model to prefer, and labeling it either way would teach the classifier
    noise. The label is 1 where the model beat the baseline on that scenario.
    """
    rewards = bank.rewards.astype(np.float64)
    paired = ~np.isnan(rewards[:, model_index]) & ~np.isnan(rewards[:, baseline_index])
    differences = rewards[:, model_index] - rewards[:, baseline_index]
    decisive = paired & (np.abs(differences) > _DECISIVE_EPSILON)
    rows = np.flatnonzero(decisive)
    return bank.embeddings[rows], (differences[rows] > 0.0).astype(np.int64)


def _rarest_class(labels: np.ndarray) -> int:
    """How many scenarios the less-represented label has (0 when only one label appears)."""
    if len(np.unique(labels)) < 2:
        return 0
    return int(np.bincount(labels).min())


def _calibrated_svm(
    vectors: np.ndarray, labels: np.ndarray
) -> tuple[CalibratedClassifierCV, float]:
    """Fit R3's RBF-SVM with a calibrated win probability, and the gamma it resolved.

    R3 used `SVC(..., probability=True)`, whose internal Platt scaling sklearn deprecated in 1.9
    (removed in 1.11) in favor of exactly this wrapper. `ensemble=False` is the equivalent shape:
    one base estimator fit on all the data, one sigmoid fit on cross-validated decision values,
    which is what `probability=True` did internally. It is also the only one of the two whose
    parameters can be extracted into a portable artifact.

    The fold count follows the rarer class, because a stratified 5-fold needs five of each and a
    thin bank routinely has fewer; below `_MIN_PER_CLASS` it never gets here (see
    `fit_drift_gate`), so the count is always at least 2.
    """
    folds = min(5, _rarest_class(labels))
    estimator = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
    model = CalibratedClassifierCV(estimator, method="sigmoid", cv=folds, ensemble=False)
    model.fit(vectors, labels)
    return model, _resolved_gamma(np.asarray(vectors, dtype=np.float64))


def fit_drift_gate(bank: KnnBank, baseline: str) -> DriftGate:
    """Fit one win-vs-baseline classifier per non-baseline model over the bank's own embeddings.

    The bank IS the training set: its embeddings are the fit split's L2-normalized task vectors
    and its reward cells are the paired evidence, so the gate sees exactly the split the router
    retrieves against and no separate feature pipeline can drift away from it.

    Three outcomes per model, all recorded in the artifact rather than resolved at serve time:
    a model with fewer than `DRIFT_GATE_MIN_DECISIVE` decisive scenarios stays unfitted (p_win 0,
    so it is always vetoed), a model whose decisive scenarios all fall on one side gets that side
    as a constant (no boundary exists to fit), and everything else gets a calibrated RBF-SVM. The
    baseline itself is never fitted: the gate is only ever asked about a pick that is leaving it.

    Raises:
        ValueError: `baseline` is not one of the bank's models, or the extracted arrays do not
            reproduce sklearn's own probabilities (which would mean the artifact scores
            differently from the estimator that produced it).
    """
    if baseline not in bank.models:
        raise ValueError(
            f"baseline '{baseline}' is not one of the bank's models ({sorted(bank.models)}); "
            "the gate scores every other model against it"
        )
    baseline_index = bank.models.index(baseline)
    count = len(bank.models)
    kinds = np.zeros(count, dtype=np.int8)
    constants = np.zeros(count, dtype=np.float64)
    intercept = np.zeros(count, dtype=np.float64)
    gamma = np.zeros(count, dtype=np.float64)
    sigmoid_a = np.zeros(count, dtype=np.float64)
    sigmoid_b = np.zeros(count, dtype=np.float64)
    offsets = np.zeros(count + 1, dtype=np.int64)
    support_blocks: list[np.ndarray] = []
    dual_blocks: list[np.ndarray] = []
    fitted = 0

    for index, model in enumerate(bank.models):
        offsets[index + 1] = offsets[index]
        if index == baseline_index:
            continue
        vectors, labels = _labels_for(bank, index, baseline_index)
        if labels.size < DRIFT_GATE_MIN_DECISIVE:
            logger.debug(
                "drift gate: %s has %d decisive fit scenarios (< %d), left unfitted",
                model,
                labels.size,
                DRIFT_GATE_MIN_DECISIVE,
            )
            continue
        if _rarest_class(labels) < _MIN_PER_CLASS:
            kinds[index] = _KIND_CONSTANT
            constants[index] = float(labels.mean())
            continue
        model_fit, resolved = _calibrated_svm(vectors, labels)
        calibrated = model_fit.calibrated_classifiers_[0]
        base = calibrated.estimator
        sigmoid = calibrated.calibrators[0]
        vectors_out = np.asarray(base.support_vectors_, dtype=np.float32)
        duals = np.asarray(base.dual_coef_[0], dtype=np.float64)
        kinds[index] = _KIND_SVM
        gamma[index] = resolved
        intercept[index] = float(base.intercept_[0])
        sigmoid_a[index] = float(sigmoid.a_)
        sigmoid_b[index] = float(sigmoid.b_)
        support_blocks.append(vectors_out)
        dual_blocks.append(duals)
        offsets[index + 1] = offsets[index] + vectors_out.shape[0]
        _check_extraction(
            model_fit,
            vectors,
            gamma=resolved,
            support=vectors_out,
            dual=duals,
            intercept=intercept[index],
            slope=sigmoid_a[index],
            offset=sigmoid_b[index],
            model=model,
        )
        fitted += 1

    dim = bank.dim
    support = (
        np.concatenate(support_blocks) if support_blocks else np.zeros((0, dim), dtype=np.float32)
    )
    dual = np.concatenate(dual_blocks) if dual_blocks else np.zeros(0, dtype=np.float64)
    logger.info(
        "drift gate: %d of %d models fitted over %d fit scenarios, %d support vectors (%.1f MB)",
        fitted,
        count - 1,
        len(bank.scenario_ids),
        support.shape[0],
        support.nbytes / 1e6,
    )
    return DriftGate(
        models=list(bank.models),
        kinds=kinds,
        constants=constants,
        offsets=offsets,
        support=support,
        dual=dual,
        intercept=intercept,
        gamma=gamma,
        sigmoid_a=sigmoid_a,
        sigmoid_b=sigmoid_b,
    )


def _check_extraction(
    model_fit: CalibratedClassifierCV,
    vectors: np.ndarray,
    *,
    gamma: float,
    support: np.ndarray,
    dual: np.ndarray,
    intercept: float,
    slope: float,
    offset: float,
    model: str,
) -> None:
    """Prove the extracted arrays score identically to the estimator they came from.

    The artifact replaces sklearn at serve time, so the fit is the only place the two can be
    compared at all. Cheap (one kernel evaluation over the training rows) and it fails the FIT
    rather than every later request, which is the difference between an sklearn release breaking
    a build and it silently changing what a served endpoint routes.

    Raises:
        ValueError: The reconstruction disagrees with `predict_proba`.
    """
    points = np.asarray(vectors, dtype=np.float64)
    basis = np.asarray(support, dtype=np.float64)
    squared = (
        (points**2).sum(axis=1)[:, None] + (basis**2).sum(axis=1)[None, :] - 2.0 * points @ basis.T
    )
    kernel = np.exp(-gamma * np.maximum(squared, 0.0))
    decisions = kernel @ dual + intercept
    mine = np.asarray([_sigmoid(float(value), slope, offset) for value in decisions])
    positive = list(model_fit.classes_).index(1)
    theirs = model_fit.predict_proba(points)[:, positive]
    worst = float(np.abs(mine - theirs).max())
    if worst > _EXTRACTION_TOLERANCE:
        raise ValueError(
            f"the drift gate's arrays for '{model}' reproduce sklearn's win probabilities only "
            f"to {worst:.2e} (tolerance {_EXTRACTION_TOLERANCE:.0e}), so the persisted gate would "
            f"route differently from the classifier just fitted. This means scikit-learn "
            f"{sklearn.__version__} parameterizes CalibratedClassifierCV differently than this "
            "extraction expects; pin an earlier scikit-learn or update `wmo.optimize.drift_gate`"
        )
