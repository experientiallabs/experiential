"""The query-embedding sidecar: the vector each logged request was routed on.

`requests.jsonl` records what the router DECIDED. This records what it decided FROM, one
L2-normalized query vector per request, so an analysis run offline can ask questions the reasons
string cannot answer: which requests cluster together, which ones sit outside the fit bank's
coverage, and what a different policy would have done with the same traffic (replaying
`knn_decision` needs the vector, and re-embedding a month of logs costs real money and does not
reproduce a retired embedder anyway).

Why JSONL of base64 float16 rather than a chunked `.npz`:

- Append-friendly. A row is one `write` to an open file in append mode, the same discipline the
  request log already uses, and a crash costs the row being written rather than the archive. An
  `.npz` is a zip container: appending means rewriting it, or inventing a chunk-rollover scheme,
  and a torn one does not open at all.
- Id-keyed. The row carries the completion id that `RequestLogRecord.id` holds, so a log row and
  its vector join on a value that already exists. Nothing has to stay positionally aligned with a
  log that skips unreadable lines.
- Compact enough. float16 is 2 bytes per dimension and base64 costs a further third, so a row is
  about `2.67 * dim` bytes plus ~60 bytes of JSON. Measured by `query_embeddings_test.py`:

      dim 512 (hashing):                 1.4 KB per request, 1.4 MB per 1k requests
      dim 3072 (text-embedding-3-large): 8.1 KB per request, 8.3 MB per 1k requests

RETENTION IS BOUNDED, because a default-on store on a busy endpoint is otherwise a slow disk
leak: at 8.3 MB per 1k requests, a million served requests is 8 GB. The current file rolls to a
single `.1` sibling once it passes `DEFAULT_MAX_BYTES` (256 MB), and the previous `.1` is
discarded, so the store occupies at most twice the cap on disk (about 512 MB, roughly the last
60k requests at 3072 dimensions) no matter how long the endpoint runs. `get` reads both files,
so a ref stays resolvable across one rotation and then stops resolving; the log row keeps its
ref either way, and a caller that gets None is looking at traffic older than the window.
`EndpointConfig.log_query_embeddings = false` turns the store off entirely per endpoint.

float16 is lossy and deliberately so. These are unit vectors whose components are ~1e-2, and
half precision holds about three decimal digits, which is far finer than any clustering or
neighbor question asked of them; it is NOT enough to re-derive a guard's standard error to the
last digit, so this store is for analysis, not for auditing a past decision's arithmetic. The
request log's own evidence fields are the audit trail.
"""

from __future__ import annotations

import base64
import json
import logging
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

QUERY_EMBEDDING_FILENAME = "query_embeddings.jsonl"

# Separates the store's filename from the row id inside a `query_embedding_ref`. The ref is
# self-describing rather than a bare id so a reader knows WHICH store to open (and so a future
# rollover can point older rows at an archived file) without the log growing a second field.
REF_SEPARATOR = "#"

# Little-endian half floats, pinned rather than native: the store is a file that outlives the
# process that wrote it and may be read on another machine.
_DTYPE = np.dtype("<f2")

# Roll the current file once it passes this, keeping exactly one previous generation, so the
# store's disk footprint is bounded at twice this no matter how long an endpoint serves. 256 MB
# is about 60k requests at 3072 dimensions (see the module docstring's retention note): long
# enough that a counterfactual replay has real traffic to work with, small enough to be an
# unremarkable amount of disk on any host that is already storing a request log.
DEFAULT_MAX_BYTES = 256 * 1024 * 1024


class QueryEmbeddingStore:
    """Append-only JSONL of the vectors requests were routed on, keyed by completion id.

    `path` of None disables the store entirely (`append` records nothing and returns None), which
    is what an in-memory serving setup and the runtime's off switch both use. Writes are
    serialized on one lock, like `RequestLog`, so concurrent requests cannot interleave a line.

    `max_bytes` caps the CURRENT file; see the module docstring for what rotation retains.
    """

    def __init__(self, path: Path | None, *, max_bytes: int = DEFAULT_MAX_BYTES) -> None:
        self._path = path
        self._max_bytes = max_bytes
        self._lock = threading.Lock()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path | None:
        return self._path

    def _previous_path(self) -> Path | None:
        """The one retained previous generation: `query_embeddings.jsonl` -> `...1.jsonl`."""
        if self._path is None:
            return None
        return self._path.with_name(f"{self._path.stem}.1{self._path.suffix}")

    def _rotate_if_full(self) -> None:
        """Roll the current file aside once it passes the cap, discarding the older generation.

        Called with the write lock held. `replace` is atomic, so a reader either sees the old
        current file or the new empty one, never a half-moved file. A stat failure is ignored
        rather than raised: failing to rotate costs disk, and failing a served request costs
        more.
        """
        if self._path is None:
            return
        try:
            if self._path.stat().st_size < self._max_bytes:
                return
        except OSError:
            return
        previous = self._previous_path()
        if previous is not None:
            self._path.replace(previous)

    def append(self, record_id: str, vector: np.ndarray) -> str | None:
        """Persist one query vector and return the ref that resolves it, or None when disabled.

        A write failure is logged and swallowed: this is an analysis sidecar, and a full disk
        must not turn a served request into a 502. The log row then simply carries no ref, which
        is the same state as the store being switched off.
        """
        if self._path is None:
            return None
        if REF_SEPARATOR in record_id:
            raise ValueError(
                f"query embedding id {record_id!r} contains {REF_SEPARATOR!r}, which separates "
                "the store name from the id in a ref; ids come from the completion id and never "
                "contain it"
            )
        payload = np.asarray(vector, dtype=_DTYPE)
        if payload.ndim != 1:
            raise ValueError(f"expected one query vector, got shape {payload.shape}")
        line = json.dumps(
            {
                "id": record_id,
                "dim": int(payload.shape[0]),
                "f16": base64.b64encode(payload.tobytes()).decode("ascii"),
            }
        )
        try:
            with self._lock:
                self._rotate_if_full()
                with self._path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
        except OSError as error:
            logger.warning("could not append a query embedding to %s: %s", self._path, error)
            return None
        return f"{self._path.name}{REF_SEPARATOR}{record_id}"

    def get(self, ref: str) -> np.ndarray | None:
        """Resolve a `query_embedding_ref` back to its vector, or None when it is not there.

        Reads the current file and then the one retained previous generation, so a ref survives
        a rotation; older than that and it resolves to None, which is what the bounded retention
        in the module docstring buys. Scans rather than indexes, because the store is written by
        serving and read by analysis: an index would be a second artifact to keep consistent with
        an append-only log, for a read path that is neither hot nor latency-bound.
        """
        if self._path is None:
            return None
        _, _, record_id = ref.rpartition(REF_SEPARATOR)
        wanted = record_id or ref
        for path in (self._path, self._previous_path()):
            if path is None or not path.is_file():
                continue
            found = self._scan(path, wanted)
            if found is not None:
                return found
        return None

    def _scan(self, path: Path, wanted: str) -> np.ndarray | None:
        """Find one id in one generation. A row this build cannot parse is skipped, matching
        `RequestLog.replay`, so one truncated line does not hide every vector after it.
        """
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if row["id"] != wanted:
                        continue
                    vector = np.frombuffer(base64.b64decode(row["f16"]), dtype=_DTYPE)
                except (ValueError, KeyError, TypeError):
                    logger.warning("skipping an unreadable query embedding row in %s", path)
                    continue
                if vector.shape[0] != row["dim"]:
                    logger.warning(
                        "query embedding %s claims dim %s but holds %d",
                        wanted,
                        row["dim"],
                        vector.shape[0],
                    )
                    continue
                return np.asarray(vector, dtype=np.float32)
        return None
