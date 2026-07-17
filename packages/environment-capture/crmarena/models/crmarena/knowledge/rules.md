## Environment access
- CRM database is read-only SQLite (`crm.db`).
- Queried via `python3 query.py "<SQL>"`; returns JSON array of row objects.
- `Case`, `Order`, `User` are reserved-ish — must wrap in double quotes in SQL.
- Ids follow Salesforce convention; FKs named `FooId` or `FooId__c` reference `Foo.Id`.
- Dates are ISO strings `YYYY-MM-DDTHH:MM:SS.sss+0000`; safe to compare/sort as text. `Order.EffectiveDate` is a bare `YYYY-MM-DD` date (no time component).
- `strftime('%m', <date>)` on the `+0000`-suffixed ISO strings returns NULL (SQLite doesn't parse the offset); use `substr(col,6,2)` for month or `substr(col,1,7)` for year-month.
- `julianday()` also fails silently (returns NULL) on `+0000`-suffixed ISO strings; strip the offset with `substr(col,1,19)` before calling `julianday()` for date arithmetic. This silently makes duration calculations return NULL and can cause aggregates/HAVING filters to drop all rows.
- Long text fields (e.g., `Knowledge__kav.FAQ_Answer__c`) are truncated in output with `...[truncated]` marker; use `substr()` to read tail. Truncation happens in query.py output regardless of downstream piping (e.g., `| tail`, `python3 -c` json parse) — the tail marker remains in the emitted field value. To retrieve later portions of a long field, use `SUBSTR(col, <offset>)` in SQL to shift the window (server-side substring is not truncated at the front).
- `query.py` output caps results: shows first 50 rows then appends `...[showing first 50 rows; add LIMIT/WHERE to narrow]`.
- Multiple `python3 query.py` invocations can be chained with `;` or newlines in one bash command; each prints its own JSON array.
- HTML entities in text fields (`&amp;`, `&#39;`) are NOT decoded — stored/returned as-is (persists through JSON extraction).

## SQL gotchas
- `OR` binds looser than `AND`: `WHERE a AND b OR c` = `(a AND b) OR c`. Always parenthesize mixed conditions when scoping by AccountId + name filters.
- Comparing datetime columns with `+0000` suffix against `'YYYY-MM-DDTHH:MM:SSZ'` string bounds works lexicographically because `.` (0x2E) < `Z` (0x5A) and `+` (0x2B) < `Z`, so a bound like `'2023-11-29T23:59:59Z'` correctly bounds the stored form.
- Bare date-string bounds like `<= '2021-11-18'` on `+0000`-suffixed datetime columns exclude that day's rows (since `'2021-11-18T...'` > `'2021-11-18'`); use `< '2021-11-19'` or a `T23:59:59Z` upper bound.
- `GROUP BY substr(col,6,2)` returns `mon: null` when the underlying `WHERE` predicate uses a comparison that eliminates all rows but there's still one NULL-yielding row — verify by first inspecting raw rows.

## Business/domain rules (documented in Knowledge__kav; NOT enforced by DB)
- Incorrect Item Received: Full Refund available within 30 days of purchase; Replacement within 60 days.
- Warranty (Defective Equipment): Replacement within 60 days of purchase (defective product must be returned); Extended Warranty covers defects up to 365 days from purchase.
- Order Cancellation: online cancellation window typically 24 hours after order placement; custom/personalized items and already-shipped orders are ineligible. If within 30 days of purchase, customer service can process a full refund (incl. shipping fees); store credit available within 90 days.
- Return, exchange, warranty, cancellation, loyalty-points policies documented in Knowledge articles — informational only.

## CaseHistory__c invariants
- `Field__c` distinct values are exactly: `Case Creation`, `Owner Assignment`, `Case Closed` (no `Owner` value exists; querying `Field__c='Owner'` returns `[]`).
- Every Case (977) has exactly one `Case Creation` row.
- 1088 `Owner Assignment` rows cover all 977 cases (some cases have multiple = transfers). First `Owner Assignment` per case has `OldValue__c` NULL; subsequent ones represent transfers (max observed 2 per case in samples).
- On transfer rows, `OldValue__c` = previous OwnerId (User Id `005...`), `NewValue__c` = new OwnerId.
- 918 `Case Closed` rows (not all 977 cases are closed); rows have both `OldValue__c` and `NewValue__c` NULL; timestamp lives in `CreatedDate` and matches `Case.ClosedDate`.
- `Case Creation` rows have both `OldValue__c` and `NewValue__c` NULL.
- The initial `Owner Assignment` row's `CreatedDate` typically equals the Case's `CreatedDate` (same timestamp).
- `Owner Assignment` volume by month is uneven; some months in 2023 (e.g., 2023-04=2, 2023-05=3, 2023-07=0, 2023-09=2, 2023-10=0) have very few or zero rows. Data ends around 2024-05.

## Case invariants
- All 977 Cases have a non-null `OrderItemId__c`; it exact-matches `OrderItem.Id`.
- `Case.ClosedDate` range: 2020-01-02 → 2024-05-26 (mirrors CreatedDate range).

## EmailMessage conventions
- Customer emails use `<name>@example.com`; agent replies use `<name>@domain.com`.
- Agent replies typically sign off with role "Customer Service Specialist".
- Reply threads share the same `ParentId` (Case Id) and use `Re: <original subject>` subjects.
