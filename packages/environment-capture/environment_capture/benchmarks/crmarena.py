"""CRMArena adapter: professional CRM work over a real Salesforce-org SQLite database.

Upstream: SalesforceAIResearch/CRMArena (NAACL 2025; data CC BY-NC 4.0). Each task is a CRM analyst
question — case routing, handle-time and transfer analytics, top-issue identification, entity
disambiguation, policy-violation checks, knowledge QA — answered by querying a realistic Salesforce
org (accounts, cases, orders, knowledge articles, case history, ...). Upstream the org is a live
Salesforce sandbox the agent hits with SOQL; the project also ships that org as a local SQLite dump
(``local_data/crmarena_data.db``), which is what this adapter materializes so captures need no
Salesforce credentials.

``open_env`` stages a fresh read-only COPY of the org database as ``crm.db`` plus a generated
``schema.md`` and a small ``query.py`` runner into the workspace; the agent explores by running
``python3 query.py "SELECT ..."`` (real rows as structured JSON observations) and submits the value
the question asks for. Because the agent gets a copy opened read-only, nothing it does can corrupt
the org the grader trusts.

Grading is deterministic and LLM-free, mirroring CRMArena's own answer semantics (thresholds
documented here, not inherited from anywhere):

- **exact_match** (the eight analytical task types — answers are a Salesforce Id, a US state code, a
  month name, or ``None``): full credit when the submission, stripped of surrounding whitespace and
  quotes, equals the gold answer, or contains it as a whole token (case-sensitive, so the
  distinctive gold Id/code/month still scores inside a short prose reply). ``None`` gold scores when
  the submission is (or contains) ``None`` / ``N/A`` / ``not applicable``. Upstream resolves these
  cases with an LLM answer-extractor; this token match is the deterministic stand-in.
- **fuzzy_match** (``knowledge_qa`` — free-text answers): the upstream token-level F1 (its
  ``normalize_answer`` — lowercase, strip punctuation, drop a/an/the, collapse whitespace — then
  bag-of-tokens precision/recall) returned as a graded ``0.0..1.0`` reward.
"""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
from collections import Counter
from pathlib import Path

from environment_capture.localexec import LocalBashEnv
from environment_capture.trajectory import Task

# Staged into each workspace: a read-only SQL runner returning JSON rows. Kept tiny and stdlib-only
# so the agent's observations are structured (list-of-record JSON) rather than raw CLI table text.
_QUERY_TOOL = '''"""Run one read-only SQL query against the CRM database and print rows as JSON."""

import json
import sqlite3
import sys

_MAX_ROWS = 50
_MAX_CELL = 2000


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python3 query.py \\"SELECT ... FROM ...\\"")
        raise SystemExit(2)
    sql = sys.argv[1]
    con = sqlite3.connect("file:crm.db?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        cursor = con.execute(sql)
        rows = cursor.fetchmany(_MAX_ROWS + 1)
    except sqlite3.Error as error:
        print(f"SQL error: {error}")
        raise SystemExit(1)
    finally:
        con.close()
    truncated = len(rows) > _MAX_ROWS
    records = []
    for row in rows[:_MAX_ROWS]:
        record = {}
        for key in row.keys():
            value = row[key]
            if isinstance(value, str) and len(value) > _MAX_CELL:
                value = value[:_MAX_CELL] + "...[truncated]"
            record[key] = value
        records.append(record)
    print(json.dumps(records, ensure_ascii=False, indent=2))
    if truncated:
        print(f"...[showing first {_MAX_ROWS} rows; add LIMIT/WHERE to narrow]")


if __name__ == "__main__":
    main()
'''

_SCHEMA_HEADER = """# CRM database (crm.db)

A realistic Salesforce org as a read-only SQLite database. Query it with:

    python3 query.py "SELECT Id, Subject, Status FROM \\"Case\\" WHERE Status = 'Closed' LIMIT 5"

Notes:
- `Case`, `Order`, `User` are SQL reserved-ish words — wrap table names in double quotes.
- Ids are Salesforce ids (e.g. `005Ws000001xSR9IAM`). Foreign keys follow Salesforce naming:
  a column `FooId` or `FooId__c` references the `Foo` object's `Id` (e.g. `Case.ContactId` ->
  `Contact.Id`, `Case.OwnerId` -> `User.Id`, `CaseHistory__c.CaseId__c` -> `Case.Id`).
- Dates are ISO strings (`YYYY-MM-DDTHH:MM:SSZ`); compare/sort them as text.

## Tables
"""


def render_schema_md(db_path: Path) -> str:
    """Render a schema guide for the org db: each table, its row count, and its columns."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        tables = [
            row[0]
            for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        ]
        sections: list[str] = []
        for table in tables:
            count = con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            info = con.execute(f'PRAGMA table_info("{table}")')
            columns = [f"{row[1]} ({row[2]})" for row in info]
            sections.append(f"### {table} — {count} rows\n{', '.join(columns)}")
    finally:
        con.close()
    return _SCHEMA_HEADER + "\n\n".join(sections) + "\n"


# --- grading ---------------------------------------------------------------------------------

_NONE_ANSWERS = {"none", "n/a", "not applicable", "no", ""}


def _clean(submission: str) -> str:
    """Strip surrounding whitespace and quotes, matching CRMArena's answer cleaning."""
    return submission.strip().strip('"').strip("'").strip()


def _contains_token(text: str, token: str) -> bool:
    """Whether `token` appears in `text` as a whole token (case-sensitive)."""
    pattern = r"(?<![A-Za-z0-9])" + re.escape(token) + r"(?![A-Za-z0-9])"
    return re.search(pattern, text) is not None


def exact_reward(gold: str, submission: str) -> float:
    """Deterministic exact/contains match for the analytical (non-knowledge-QA) tasks."""
    cleaned = _clean(submission)
    if gold == "None":
        if cleaned.lower() in _NONE_ANSWERS:
            return 1.0
        return 1.0 if _contains_token(submission, "None") else 0.0
    if cleaned == gold:
        return 1.0
    return 1.0 if _contains_token(submission, gold) else 0.0


_PUNCT = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~" + "‘’´`")


def _normalize_answer(text: str) -> str:
    """CRMArena's normalize_answer: lowercase, strip punctuation, drop articles, fix whitespace."""
    text = text.replace("_", " ").lower()
    text = "".join(ch if ch not in _PUNCT else " " for ch in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split()).strip()


def fuzzy_reward(gold: str, submission: str) -> float:
    """CRMArena's token-level F1 between the submission and the gold free-text answer."""
    pred_tokens = _normalize_answer(submission).split()
    gold_tokens = _normalize_answer(gold).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0 or not pred_tokens or not gold_tokens:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


class CrmArenaAdapter:
    """BenchmarkAdapter over a materialized CRMArena org database."""

    name = "crmarena"

    def __init__(self, data_root: Path, *, timeout_s: int = 120) -> None:
        """`data_root` holds crm.db, data/{train,test}.jsonl, and gold/<task_id>.json."""
        self.data_root = data_root
        self.timeout_s = timeout_s

    @property
    def db_path(self) -> Path:
        return self.data_root / "crm.db"

    def tasks(self, split: str) -> list[Task]:
        path = self.data_root / "data" / f"{split}.jsonl"
        tasks: list[Task] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            tasks.append(
                Task(task_id=raw["task_id"], prompt=raw["prompt"], data=raw.get("data", {}))
            )
        return tasks

    def open_env(self, task: Task) -> LocalBashEnv:
        """Stage a fresh copy of the org db + the query tool + schema guide (never the gold)."""
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"{self.db_path} missing; run "
                "`uv run python packages/environment-capture/crmarena/fetch_data.py` to download it"
            )
        env = LocalBashEnv(timeout_s=self.timeout_s)
        shutil.copy(self.db_path, env.workspace / "crm.db")
        (env.workspace / "query.py").write_text(_QUERY_TOOL, encoding="utf-8")
        (env.workspace / "schema.md").write_text(
            render_schema_md(env.workspace / "crm.db"), encoding="utf-8"
        )
        return env

    def grade(self, task: Task, submission: str) -> float:
        """Score against the gold answer with the task's own metric (exact or fuzzy match)."""
        gold = json.loads(
            (self.data_root / "gold" / f"{task.task_id}.json").read_text(encoding="utf-8")
        )
        answer = str(gold["answer"])
        if gold["reward_metric"] == "fuzzy_match":
            return fuzzy_reward(answer, submission)
        return exact_reward(answer, submission)
