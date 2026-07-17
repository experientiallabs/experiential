## Environment / Tooling
- `bash` tool executes shell commands mechanically; no auth or gating observed.
- SQLite DB accessed via `sqlite3 database.db "<SQL>"` in the working directory (also works with `./database.db`).
- Schema file `./schema.sql` sits alongside `database.db` and contains only `CREATE TABLE` statements (no data). May include commented-out columns (e.g. `-- PctGE1500 double null`) that are NOT in the live table.
- Multiple SQL statements can be issued in one `sqlite3` invocation separated by `;`; each result set is printed sequentially with no separator (may appear ambiguous, e.g. duplicate `-` when values overlap between statements).
- Multiple `sqlite3` invocations and `echo` calls can be chained via newline in a single `bash` command; outputs concatenate in order.
- Default `sqlite3` CLI output: pipe-separated columns, newline-separated rows, no header, no row count.
- Backticks embedded in a double-quoted shell argument must be escaped as `\``; identifiers can alternatively be quoted with `\"...\"` inside the double-quoted argument.
- SQL errors surface as `is_error=True` with format `Error: in prepare, no such column: <col>\n  <query snippet>\n         ^--- error here`.
- No observed timeouts, rate limits, or output truncation.

## Data Integrity
- Foreign-key columns may be NULL (all FK columns declared `default NULL`).
- Numeric values in the DB can carry floating-point imprecision (e.g. `-0.199999999999999` stored as REAL; averages returned to full float precision, e.g. `465.432098765432`, `16.7005649717514`, `137.888888888889`).
- `schools.Virtual` codes are single letters (e.g. `F`, `P`, `N`) — `F` observed to mean non-virtual in query contexts (but virtual-named schools like `California Virtual Academy` also carry `Virtual='F'`, so the code is not a reliable virtual-vs-brick indicator).
- `satscores.sname` may be NULL or empty string `''` (both occur for the same county in results); filter with `sname IS NOT NULL AND sname != ''` to exclude blanks.
- `schools.FundingType` may be NULL (blank in output) for schools not in the CS funding model. Distinct non-null values: `Directly funded`, `Locally funded`, `Not in CS funding model` (plus empty).
- `schools.AdmEmail1/2/3` may be empty strings for some schools.
- `expense.approved` observed only as `true` in the loaded DB (no `false` rows present).
- `atom_id` index component is variable width: `SUBSTR(atom_id,7,2)` returns e.g. `1`,`2`,...,`21`,`22` (trailing chars beyond string end are omitted; no zero-padding). Use range comparisons on the substring as strings only when width is uniform.
