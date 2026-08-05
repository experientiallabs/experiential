"""Human-readable cluster labels for the routing request log.

A fitted rank policy's `cluster_label` is a product surface: the request log shows WHY a request
went where it did ("rank router: nearest cluster 12 (airline-cancellation refund)"). Two sources,
best first:

1. The majority scenario-id prefix ("mmlu-anatomy", "tau-bench") when ids carry `prefix:` task
   names. That is exact provenance, so nothing beats it.
2. c-TF-IDF terms otherwise (BERTopic's class-based TF-IDF, arXiv 2203.05794, simplified): score
   each term by its in-cluster frequency times log(1 + K / cluster-df) and take the top
   `max_terms`. Distinctive words beat merely frequent ones, so two retail clusters get different
   labels ("exchange camera zoom" vs "cancel order refund") instead of both saying "order".

The second source is why this module exists. Corpora built from real traces key their scenarios
by trace hash, not by `prefix:` name, so on exactly the corpora WMO builds from, the first source
finds nothing and every cluster used to be labeled with the empty string.

Pure text processing, no model calls, no new dependency. Labeling never affects selection: a
label is read by people, and a wrong one costs a confusing log line, never a misrouted request.
"""

from __future__ import annotations

import math
import re
from collections import Counter

# Three characters or more, starting with a letter. `\w` under Python's default unicode
# semantics, so a CJK or accented corpus yields real tokens instead of "": an ASCII-only class
# silently matched nothing on the first and truncated "echange" out of "exchange" on the second,
# which is a wrong label rather than a missing one. Deliberately still one regex and no word
# segmentation, so an unspaced script yields one token per run rather than per word; that is a
# coarse label, which beats the empty one it used to produce.
_TOKEN = re.compile(r"[^\W\d_][\w-]{2,}", re.UNICODE)

# Function words plus the JSON-scaffolding vocabulary of world-model task payloads. Anything this
# generic would otherwise label every cluster identically, which is worse than no label: it looks
# like information.
_STOP = frozenset(
    """the and for you your are with that this not have has can will its all any out from
    into over under been being what when where which while about after before because but
    they them their there then than each such only also more most some very just like want
    need make sure ask say tell get use used using known info reason call task domain
    instructions user known_info reason_for_call should would could must may might his her
    him she who whom does did doing done was were is it in on at to of as by an or if do be
    no so we he up""".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase content words of one task text, stop words dropped."""
    # Literal escape sequences inside JSON-encoded task payloads ("...\\nYou are...") would
    # otherwise fuse into pseudo-words like "nyou".
    text = text.replace("\\n", " ").replace("\\t", " ").lower()
    return [token for token in _TOKEN.findall(text) if token not in _STOP]


def label_clusters(cluster_texts: list[list[str]], *, max_terms: int = 3) -> list[str]:
    """One c-TF-IDF label per cluster, aligned to `cluster_texts` (see the module docstring).

    A cluster whose texts carry no content word at all gets "", the same empty label it would
    have had without this module; there is nothing honest to call it.
    """
    cluster_count = len(cluster_texts)
    term_counts = [
        Counter(token for text in texts for token in tokenize(text)) for texts in cluster_texts
    ]
    document_frequency = Counter(term for counts in term_counts for term in counts)
    labels: list[str] = []
    for counts in term_counts:
        total = sum(counts.values())
        if not total:
            labels.append("")
            continue
        scored = {
            term: (frequency / total) * math.log(1 + cluster_count / document_frequency[term])
            for term, frequency in counts.items()
        }
        # Ties break alphabetically, so a label is a property of the corpus rather than of dict
        # ordering: two fits of the same matrix must produce the same artifact.
        top = sorted(scored, key=lambda term: (-scored[term], term))[:max_terms]
        labels.append(" ".join(top))
    return labels


def majority_prefix(scenario_ids: list[str]) -> str:
    """The most common `prefix:` among the ids, or "" when none of them carry one."""
    prefixes = Counter(sid.split(":", 1)[0] for sid in scenario_ids if ":" in sid)
    return prefixes.most_common(1)[0][0] if prefixes else ""
