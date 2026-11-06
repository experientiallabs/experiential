"""Checkpoint statistics 1 to 3: corpus tagging, clusters and families, rules and state recall.

Every function here is arithmetic over answers the decision model already gave. Flags are
threshold crossings named in :class:`~.tag_monitor_contracts.MonitorSpec`; what a flag means for
the bank, family or rule is Astra's judgment, never this module's.

- :func:`corpus_stats` (checkpoint 1): per question fire rate, always-yes / always-no,
  near-constant and low-confidence flags, near-duplicate pairs by answer agreement and fire-set
  overlap, per bank coverage and the views excluded at the credential boundary.
- :func:`family_stats` (checkpoint 2): per family tag profile, the tags separating it from the
  rest of the corpus, families held together by weak tags only, a two-way split candidate inside
  each family (two-means over its members' fired tags) with the tags separating the halves and
  the overlap of the halves' declared tools, indistinct family pairs, how well the authored family
  bank discriminates members, and how far the family's task types separate.
- :func:`rule_stats` (checkpoint 3): per rule support against contradiction and whether its
  compiled tags ever fire; per state tag its fire rate and recall quality, the share of each
  tagged step's nearest neighbours (Jaccard over fired state tags, other lineages only) whose
  next action differs.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from itertools import combinations

import numpy as np
from pydantic import Field

from exp.common.core.artifacts import ContractModel, JsonValue, sha256_json
from exp.optimize.evaluation.variance.tag_monitor_contracts import (
    Citation,
    Halt,
    MonitorSpec,
    SubjectStats,
)
from exp.simulation.mining.tag_health import corpus_health, matrix_key
from exp.simulation.mining.tagging import TagMatrix
from exp.simulation.rules.admission import AdmissionVerdict, RuleTagMatrix


def halt(checkpoint: str, subject: str, condition: str, reason: str, *stats: str) -> Halt:
    """Return a statistics halt with its content identity."""
    cites = tuple(Citation(subject=subject, stat=stat) for stat in stats)
    identity = sha256_json([checkpoint, subject, condition, reason])[:20]
    return Halt.model_validate(
        {
            "halt_id": f"halt-{identity}",
            "checkpoint": checkpoint,
            "subject": subject,
            "condition": condition,
            "reason": reason,
            "cites": cites,
        }
    )


def sample(items: Sequence[str], count: int, salt: str) -> list[str]:
    """Return a deterministic, salted sample of ``count`` items."""

    def key(item: str) -> str:
        """Salted digest order."""
        return hashlib.sha256(f"{salt}\0{item}".encode()).hexdigest()

    return sorted(items, key=key)[:count]


def _round(value: float) -> float:
    """Round a statistic for a stable record."""
    return round(float(value), 6)


class CorpusInputs(ContractModel):
    """What checkpoint 1 reads.

    Attributes:
        matrices: Every saved matrix, one per bank and view.
        excluded: Traces excluded at the credential boundary, by ``<bank>.<view>`` or view.
        texts: Question texts by question identity, when known.
        previews: Clipped rendered views by trace identity, for evidence.
    """

    matrices: tuple[TagMatrix, ...] = Field(min_length=1)
    excluded: dict[str, int] | None = None
    texts: dict[str, str] = Field(default_factory=dict)
    previews: dict[str, str] = Field(default_factory=dict)


def question_subject(bank: str, question: str) -> str:
    """Name a question of a bank (``<bank>.<view>``) as a subject."""
    return f"question:{bank}/{question}"


def corpus_stats(
    inputs: CorpusInputs, spec: MonitorSpec
) -> tuple[list[SubjectStats], list[Halt], dict[str, JsonValue]]:
    """Checkpoint 1: question, pair and bank statistics, halts and sampled evidence."""
    health = corpus_health(inputs.matrices, spec.health, excluded=inputs.excluded)
    duplicates: dict[str, list[str]] = {}
    pairs: list[SubjectStats] = []
    for pair in health.pairs:
        bank = f"{pair.bank_id}.{pair.view}"
        left, right = question_subject(bank, pair.left), question_subject(bank, pair.right)
        duplicates.setdefault(left, []).append(right)
        duplicates.setdefault(right, []).append(left)
        pairs.append(
            SubjectStats(
                subject=f"pair:{bank}/{pair.left}|{pair.right}",
                kind="pair",
                values={
                    "agreement": pair.agreement,
                    "jaccard": pair.jaccard,
                    "correlation": pair.correlation,
                    "shared": float(pair.shared),
                },
                flags=("near_duplicate",),
                refs=(left, right),
            )
        )
    subjects: list[SubjectStats] = []
    for row in health.questions:
        name = question_subject(f"{row.bank_id}.{row.view}", row.question_id)
        flags = set(row.flags) | ({"near_duplicate"} if name in duplicates else set())
        subjects.append(
            SubjectStats(
                subject=name,
                kind="question",
                values={
                    "fire_rate": row.fire_rate,
                    "spread": row.spread,
                    "ambiguous": row.ambiguous,
                    "answered": float(row.answered),
                },
                flags=tuple(sorted(flags)),
                refs=tuple(sorted(duplicates.get(name, ()))),
            )
        )
    subjects.extend(pairs)
    halts: list[Halt] = []
    by_bank = {matrix_key(m): m for m in inputs.matrices}
    for bank in health.banks:
        key = f"{bank.bank_id}.{bank.view}"
        rows = [q for q in health.questions if (q.bank_id, q.view) == (bank.bank_id, bank.view)]
        uninformative = sum(1 for q in rows if {"always_yes", "always_no"} & set(q.flags))
        share = uninformative / max(1, bank.questions)
        values = {
            "coverage": bank.coverage,
            "mean_fired": bank.mean_fired,
            "flagged_share": bank.flagged_share,
            "uninformative_share": _round(share),
            "traces": float(bank.traces),
            "questions": float(bank.questions),
            "near_duplicate_pairs": float(bank.duplicates),
        }
        if bank.excluded is not None:
            values["credential_excluded"] = float(bank.excluded)
        flags = []
        name = f"bank:{key}"
        if share > spec.max_uninformative_share:
            flags.append("uninformative")
            halts.append(
                halt(
                    "corpus",
                    name,
                    "bank_uninformative",
                    f"{uninformative} of {bank.questions} questions of {key} are always-yes or "
                    f"always-no ({share:.2f} > {spec.max_uninformative_share})",
                    "uninformative_share",
                )
            )
        if bank.coverage < spec.min_bank_coverage:
            flags.append("low_coverage")
            halts.append(
                halt(
                    "corpus",
                    name,
                    "bank_low_coverage",
                    f"{key} fires on {bank.coverage:.2f} of traces (< {spec.min_bank_coverage})",
                    "coverage",
                )
            )
        subjects.append(SubjectStats(subject=name, kind="bank", values=values, flags=tuple(flags)))
    evidence: dict[str, JsonValue] = {}
    columns: dict[tuple[str, str], dict[str, float]] = {}

    def answers(bank_key: str, question: str) -> dict[str, float]:
        """One question's answers by trace, read once."""
        if (bank_key, question) not in columns:
            matrix = by_bank[bank_key]
            traces = (r.trace_id for r in matrix.rows)
            columns[bank_key, question] = dict(zip(traces, matrix.column(question), strict=True))
        return columns[bank_key, question]

    for row in subjects:
        if row.kind != "question" or not row.flags or len(evidence) >= spec.max_documents:
            continue
        bank_key, _, question = row.subject.removeprefix("question:").partition("/")
        column = answers(bank_key, question)
        others = [ref.rsplit("/", 1)[1] for ref in row.refs]
        rows: list[JsonValue] = []
        for trace in sample(sorted(column), spec.samples_per_flag, row.subject):
            entry: dict[str, JsonValue] = {"trace_id": trace, "p": column[trace]}
            entry |= {f"p_{other}": answers(bank_key, other)[trace] for other in others}
            if trace in inputs.previews:
                entry["preview"] = inputs.previews[trace]
            rows.append(entry)
        evidence[f"evidence:{row.subject}"] = {
            "question": inputs.texts.get(question, question),
            "rows": rows,
        }
    return subjects, halts, evidence


class FamilyMembers(ContractModel):
    """One environment family as checkpoint 2 reads it.

    Attributes:
        family_id: The family.
        members: Member traces.
        tools: Declared tool names per member, when known.
        task_types: Task type per member, when task types were discovered.
    """

    family_id: str
    members: tuple[str, ...] = Field(min_length=1)
    tools: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    task_types: dict[str, str] = Field(default_factory=dict)


class FamilyInputs(ContractModel):
    """What checkpoint 2 reads.

    Attributes:
        families: Every admitted family.
        matrices: Corpus matrices over every trace (task and observation views).
        authored: Each family's authored-bank matrices over its members, when authored.
    """

    families: tuple[FamilyMembers, ...] = Field(min_length=1)
    matrices: tuple[TagMatrix, ...] = Field(min_length=1)
    authored: dict[str, tuple[TagMatrix, ...]] = Field(default_factory=dict)


def _fired_table(
    matrices: Sequence[TagMatrix], threshold: float
) -> tuple[list[str], dict[str, np.ndarray]]:
    """Return tag names (``<view>/<question>``) and each trace's fired vector over them.

    A trace without a row in some matrix is left out, so every vector covers every tag.
    """
    tags = [f"{m.view}/{q}" for m in matrices for q in m.question_ids]
    rows: dict[str, list[float]] = {}
    counts: dict[str, int] = {}
    for matrix in matrices:
        for row in matrix.rows:
            rows.setdefault(row.trace_id, []).extend(
                1.0 if v > threshold else 0.0 for v in row.values
            )
            counts[row.trace_id] = counts.get(row.trace_id, 0) + 1
    vectors = {t: np.array(v) for t, v in rows.items() if counts[t] == len(matrices)}
    return tags, vectors


def _two_means(table: np.ndarray) -> np.ndarray:
    """Split rows into two groups by two-means on L1 distance, deterministically seeded."""
    centre = table.mean(axis=0)
    first = int(np.argmax(np.abs(table - centre).sum(axis=1)))
    second = int(np.argmax(np.abs(table - table[first]).sum(axis=1)))
    seeds = np.stack([table[first], table[second]])
    labels = np.zeros(len(table), dtype=int)
    for _ in range(20):
        distance = np.abs(table[:, None, :] - seeds[None, :, :]).sum(axis=2)
        updated = np.argmin(distance, axis=1)
        if (updated == labels).all() and _ > 0:
            break
        labels = updated
        for group in (0, 1):
            if (labels == group).any():
                seeds[group] = table[labels == group].mean(axis=0)
    return labels


def family_stats(
    inputs: FamilyInputs, spec: MonitorSpec
) -> tuple[list[SubjectStats], list[Halt], dict[str, JsonValue]]:
    """Checkpoint 2: family profiles, separation, split candidates, authored and type checks."""
    tags, vectors = _fired_table(inputs.matrices, spec.health.threshold)
    profiles: dict[str, np.ndarray] = {}
    subjects: list[SubjectStats] = []
    halts: list[Halt] = []
    evidence: dict[str, JsonValue] = {}
    for family in inputs.families:
        name = f"family:{family.family_id}"
        tagged = [m for m in family.members if m in vectors]
        rest = [t for t in vectors if t not in set(family.members)]
        values: dict[str, float] = {"members": float(len(family.members)), "tagged": len(tagged)}
        flags: list[str] = []
        refs: list[str] = []
        if not tagged:
            flags.append("untagged")
            subjects.append(
                SubjectStats(subject=name, kind="family", values=values, flags=("untagged",))
            )
            continue
        table = np.stack([vectors[m] for m in tagged])
        profile = table.mean(axis=0)
        profiles[family.family_id] = profile
        if rest:
            outside = np.stack([vectors[t] for t in rest]).mean(axis=0)
            delta = profile - outside
            order = np.argsort(-np.abs(delta))
            separating = [tags[i] for i in order if abs(delta[i]) >= spec.separating_delta]
            strongest = float(np.abs(delta).max())
            values |= {"separating": float(len(separating)), "max_separation": _round(strongest)}
            for tag in separating[:5]:
                values[f"separation[{tag}]"] = _round(float(delta[tags.index(tag)]))
            if not separating:
                flags.append("no_separating_tags")
            if strongest < spec.weak_delta:
                flags.append("weak_tags_only")
        for index in np.argsort(-profile)[:5]:
            values[f"profile[{tags[index]}]"] = _round(float(profile[index]))
        if len(tagged) >= 2 * spec.min_split_members:
            labels = _two_means(table)
            left = [m for m, g in zip(tagged, labels, strict=True) if g == 0]
            right = [m for m, g in zip(tagged, labels, strict=True) if g == 1]
            if min(len(left), len(right)) >= spec.min_split_members:
                split = table[labels == 0].mean(axis=0) - table[labels == 1].mean(axis=0)
                order = np.argsort(-np.abs(split))
                apart = [tags[i] for i in order if abs(split[i]) >= spec.separating_delta]
                values |= {
                    "split_left": float(len(left)),
                    "split_right": float(len(right)),
                    "split_separating": float(len(apart)),
                }
                for tag in apart[:8]:
                    values[f"split[{tag}]"] = _round(float(split[tags.index(tag)]))
                refs.extend(f"tag:{tag}" for tag in apart[:8])
                tools_left = {t for m in left for t in family.tools.get(m, ())}
                tools_right = {t for m in right for t in family.tools.get(m, ())}
                if tools_left or tools_right:
                    overlap = len(tools_left & tools_right) / len(tools_left | tools_right)
                    values["split_tool_jaccard"] = _round(overlap)
                if len(apart) >= spec.min_split_tags:
                    flags.append("mixed")
                    evidence[f"evidence:{name}/split"] = {
                        "left": [_member(m, family, tags, vectors, apart) for m in left[:4]],
                        "right": [_member(m, family, tags, vectors, apart) for m in right[:4]],
                    }
        authored = inputs.authored.get(family.family_id, ())
        if authored:
            columns = [
                np.array([r.values[i] for r in m.rows if r.trace_id in set(family.members)])
                for m in authored
                for i in range(len(m.question_ids))
            ]
            usable = [
                c
                for c in columns
                if c.size
                and spec.health.min_fire_rate
                <= float((c > spec.health.threshold).mean())
                <= spec.health.max_fire_rate
                and float(c.std()) >= spec.health.min_spread
            ]
            values |= {
                "authored_questions": float(len(columns)),
                "authored_discriminating": len(usable),
            }
            if not usable:
                flags.append("authored_not_discriminating")
        types = sorted({family.task_types[m] for m in tagged if m in family.task_types})
        if len(types) >= 2:
            means = {
                t: np.stack([vectors[m] for m in tagged if family.task_types.get(m) == t]).mean(
                    axis=0
                )
                for t in types
            }
            gaps = [
                int((np.abs(means[a] - means[b]) >= spec.separating_delta).sum())
                for a, b in combinations(types, 2)
            ]
            values |= {
                "task_types": float(len(types)),
                "task_type_min_separating": float(min(gaps)),
            }
            if min(gaps) == 0:
                flags.append("task_types_inseparable")
        if "no_separating_tags" in flags and (
            not authored or "authored_not_discriminating" in flags
        ):
            halts.append(
                halt(
                    "families",
                    name,
                    "family_no_discriminating_tags",
                    f"family {family.family_id} has no tag separating it from the rest of the "
                    "corpus" + (" and no discriminating authored question" if authored else ""),
                    "flag:no_separating_tags",
                )
            )
        subjects.append(
            SubjectStats(
                subject=name,
                kind="family",
                values=values,
                flags=tuple(sorted(flags)),
                refs=tuple(refs),
            )
        )
    for (a, pa), (b, pb) in combinations(sorted(profiles.items()), 2):
        distance = float(np.abs(pa - pb).mean())
        if distance <= spec.indistinct_distance:
            subjects.append(
                SubjectStats(
                    subject=f"family_pair:{a}|{b}",
                    kind="family_pair",
                    values={"profile_distance": _round(distance)},
                    flags=("indistinct",),
                    refs=(f"family:{a}", f"family:{b}"),
                )
            )
    return subjects, halts, evidence


def _member(
    trace: str,
    family: FamilyMembers,
    tags: list[str],
    vectors: Mapping[str, np.ndarray],
    apart: Sequence[str],
) -> dict[str, JsonValue]:
    """One member's evidence row: its tools and its answers to the separating tags."""
    return {
        "trace_id": trace,
        "tools": list(family.tools.get(trace, ())),
        "fired": [tag for tag in apart[:8] if vectors[trace][tags.index(tag)] > 0.5],
    }


class RuleEvidence(ContractModel):
    """One rule as checkpoint 3 reads it.

    Attributes:
        verdict: Its admission verdict.
        matrix: Its corpus-wide compiled-question answers, when saved.
    """

    verdict: AdmissionVerdict
    matrix: RuleTagMatrix | None = None


class StateStep(ContractModel):
    """One tagged pre-turn state and the action that followed it.

    Attributes:
        step_id: The recorded transition.
        lineage_id: Its leakage lineage.
        probabilities: One answer per state question, in bank order.
        next_action: The action that followed (the called tool).
    """

    step_id: str
    lineage_id: str
    probabilities: tuple[float, ...]
    next_action: str


class RuleInputs(ContractModel):
    """What checkpoint 3 reads.

    Attributes:
        rules: Every rule's admission evidence.
        state_bank: State-tag bank identity (for subject names), when state recall exists.
        state_questions: State question identities, in bank order.
        state_steps: Every tagged fit step.
    """

    rules: tuple[RuleEvidence, ...] = ()
    state_bank: str = "state"
    state_questions: tuple[str, ...] = ()
    state_steps: tuple[StateStep, ...] = ()


def rule_stats(
    inputs: RuleInputs, spec: MonitorSpec
) -> tuple[list[SubjectStats], list[Halt], dict[str, JsonValue]]:
    """Checkpoint 3: rule support and dead tags; state-tag fire rates and recall quality."""
    threshold = spec.health.threshold
    subjects: list[SubjectStats] = []
    halts: list[Halt] = []
    evidence: dict[str, JsonValue] = {}
    dead = 0
    for rule in inputs.rules:
        verdict = rule.verdict
        support, against = len(verdict.supporting_lineages), len(verdict.contradicting)
        values = {
            "supporting_lineages": float(support),
            "contradicting": float(against),
            "uncertain": float(len(verdict.uncertain)),
            "not_applicable": float(len(verdict.not_applicable)),
            "admitted": 1.0 if verdict.admitted else 0.0,
        }
        flags: list[str] = []
        if against:
            flags.append("contradicted")
        matrix = rule.matrix
        if matrix is not None and matrix.rows:
            table = np.array([row.values for row in matrix.rows])
            rates = (table > threshold).mean(axis=0)
            values["tags_fire_rate"] = _round(float((table > threshold).any(axis=1).mean()))
            if not (table > threshold).any():
                flags.append("tags_never_fire")
                dead += 1
            for index, question in enumerate(matrix.question_ids):
                qflags = ()
                if rates[index] == 0.0:
                    qflags = ("never_fires",)
                elif rates[index] >= spec.health.always_rate:
                    qflags = ("always_fires",)
                subjects.append(
                    SubjectStats(
                        subject=f"rule_question:{verdict.rule_id}/{question}",
                        kind="rule_question",
                        values={"fire_rate": _round(float(rates[index]))},
                        flags=qflags,
                        refs=(f"rule:{verdict.rule_id}",),
                    )
                )
        subjects.append(
            SubjectStats(
                subject=f"rule:{verdict.rule_id}", kind="rule", values=values, flags=tuple(flags)
            )
        )
    if inputs.rules and dead / len(inputs.rules) > spec.dead_rule_share:
        halts.append(
            halt(
                "rules",
                "rule:" + inputs.rules[0].verdict.rule_id,
                "rules_never_fire",
                f"{dead} of {len(inputs.rules)} rules have compiled tags that never fire "
                f"(> {spec.dead_rule_share})",
                "flag:tags_never_fire",
            )
        )
    steps = inputs.state_steps
    if steps and inputs.state_questions:
        table = np.array([s.probabilities for s in steps]) > threshold
        weights = table.astype(np.float64)
        shared = weights @ weights.T
        sizes = weights.sum(axis=1)
        union = sizes[:, None] + sizes[None, :] - shared
        with np.errstate(invalid="ignore", divide="ignore"):
            agreement = np.where(union > 0, shared / union, 0.0)
        lineages = np.array([s.lineage_id for s in steps])
        agreement[lineages[:, None] == lineages[None, :]] = -1.0
        actions = [s.next_action for s in steps]
        disagree = np.full(len(steps), np.nan)
        for index in range(len(steps)):
            order = [j for j in np.argsort(-agreement[index]) if agreement[index, j] > 0.0]
            near = order[: spec.recall_neighbours]
            if near:
                disagree[index] = sum(actions[j] != actions[index] for j in near) / len(near)
        uninformative = 0
        for column, question in enumerate(inputs.state_questions):
            rate = float(table[:, column].mean())
            fired = table[:, column] & ~np.isnan(disagree)
            values = {"fire_rate": _round(rate)}
            flags = []
            if rate == 0.0 or rate >= spec.health.always_rate:
                flags.append("never_fires" if rate == 0.0 else "always_fires")
                uninformative += 1
            if fired.any():
                mean = float(np.nanmean(disagree[fired]))
                values["neighbour_disagreement"] = _round(mean)
                if mean >= spec.disagreement_rate:
                    flags.append("recall_disagrees")
            name = f"state_tag:{inputs.state_bank}/{question}"
            subjects.append(
                SubjectStats(subject=name, kind="state_tag", values=values, flags=tuple(flags))
            )
            if "recall_disagrees" in flags and len(evidence) < spec.max_documents:
                rows = [steps[i].step_id for i in np.flatnonzero(fired)]
                evidence[f"evidence:{name}"] = [
                    {"step_id": s, "next_action": actions[[x.step_id for x in steps].index(s)]}
                    for s in sample(rows, spec.samples_per_flag, name)
                ]
        share = uninformative / len(inputs.state_questions)
        bank = f"bank:{inputs.state_bank}"
        mean_disagreement = float(np.nanmean(disagree)) if (~np.isnan(disagree)).any() else 0.0
        subjects.append(
            SubjectStats(
                subject=bank,
                kind="bank",
                values={
                    "uninformative_share": _round(share),
                    "neighbour_disagreement": _round(mean_disagreement),
                    "steps": float(len(steps)),
                },
                flags=("uninformative",) if share > spec.max_uninformative_share else (),
            )
        )
        if share > spec.max_uninformative_share:
            halts.append(
                halt(
                    "rules",
                    bank,
                    "state_bank_uninformative",
                    f"{uninformative} of {len(inputs.state_questions)} state tags never or always "
                    f"fire ({share:.2f} > {spec.max_uninformative_share})",
                    "uninformative_share",
                )
            )
    return subjects, halts, evidence
