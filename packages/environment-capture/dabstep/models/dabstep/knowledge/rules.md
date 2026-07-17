# Environment Rules

## Bash / Heredoc Execution
- Multi-line python heredocs invoked as `python3 - <<'PY' ... PY'` fail with `SyntaxError: unterminated string literal` when the closing `PY` marker is immediately followed by a trailing quote character (i.e. `PY'` instead of `PY` on its own line). The environment passes the trailing `'` into stdin as part of the script. Workaround: end heredoc with `PY` alone, or use `python3 -c "..."`.
- `python3 << 'EOF' ... EOF` heredocs work correctly.
- `python3 -c "..."` with embedded newlines in the double-quoted string works.
- Empty command string is accepted and returns no output (no error).

## Fee Rule Applicability (fees.json)
- A `null` field in a fee rule means the rule applies to ALL possible values of that field (per manual.md).
- An empty list (`[]`) for `account_type` / `aci` / `merchant_category_code` is also treated as "applies to all" in practice.
- A non-null / non-empty field constrains applicability: the transaction's value must be in the list (or match the string rule) for the fee rule to apply.
- Fields that gate applicability: `card_scheme`, `account_type`, `capture_delay`, `monthly_fraud_level`, `monthly_volume`, `merchant_category_code`, `is_credit`, `aci`, `intracountry`.
- Fee formula: `fee = fixed_amount + rate * transaction_value / 10000`.
- Higher fraud rate → typically more expensive processor fees.
- Faster capture-to-settlement → more expensive.
- Credit transactions typically incur higher fees than non-credit.
- Higher monthly volume merchants typically get cheaper fees.
- Volumes in `monthly_volume` are specified in euros (e.g. `100k-1m` = 100,000–1,000,000).
- Monthly volumes and fraud rates are computed over natural calendar months (day 1 to last day of month; Feb=28/29, others 30/31).
- `monthly_fraud_level` is measured as the ratio of monthly fraud volume (sum of eur_amount where has_fraudulent_dispute=True) to monthly total volume (sum of eur_amount), expressed as percent. Not the count-based mean of the boolean column.
- `intracountry` = True when `issuing_country == acquirer_country`; False otherwise (international, typically more expensive).
- Multiple fee rules can match a single transaction — traces show up to 6+ matching rules per (scheme, aci, is_credit, intracountry) combination; the environment itself does not disambiguate (no single "correct" rule enforced).
- Some (card_scheme, aci) combinations have zero matching rules for a given merchant profile (e.g. ACIs D/E/F/G may have NO matching rule under a given merchant's mcc/account_type/capture_delay/volume/fraud constraints), meaning those ACIs are effectively not offered/priced for that merchant.
- Filtering fees.json for rules where `account_type==['R']` AND `aci==['B']` (exact-list match) is stricter than "applies-to-R and applies-to-B" (which also includes `[]` catch-all rules). Both interpretations appear in traces; the environment does not privilege one.
- Filtering fees.json for `('R' ∈ account_type OR account_type==[])` AND `('B' ∈ aci OR aci==[])` (inclusive catch-all interpretation) yields 416 rules. Strict `'R' ∈ account_type AND 'B' ∈ aci` (non-empty membership) yields 50 rules: [34, 39, 49, 62, 68, 82, 154, 220, 231, 236, 265, 276, 278, 286, 329, 345, 352, 355, 360, 368, 369, 390, 393, 404, 419, 512, 539, 556, 564, 583, 587, 590, 638, 645, 661, 711, 717, 731, 757, 779, 793, 828, 837, 871, 915, 938, 939, 964, 986, 998].
- For Belles_cookbook_store (mcc=5942, account=R, capture_delay=1, Jan 2023 volume=100k-1m, fraud >8.3%): 62 fee rules pass mcc+account+capture_delay filter; 47 rules remain after additionally filtering on monthly_volume and monthly_fraud_level.
- fees.json `monthly_fraud_level` distribution across all 1000 rules: {None: 900, `7.7%-8.3%`: 29, `<7.2%`: 26, `7.2%-7.7%`: 24, `>8.3%`: 21}.
- fees.json `aci` distribution (top combos): `[]` (112), `['A','C']` (110), `['B']` (102), `['A','B','C']` (101), `['B','C']` (94), `['E']` (91), `['C']` (84), `['D']` (82), `['A']` (77), `['A','B']` (74), `['F']` (73).

## Capture Delay Bucketing
- Merchant `capture_delay` stored as a raw integer-string (e.g. `'1'`, `'2'`, `'7'`) or a non-numeric bucket (`'immediate'`, `'manual'`). Observed values in merchant_data.json: {`immediate`, `1`, `2`, `7`, `manual`}.
- Fee-rule `capture_delay` values observed in fees.json: {`<3`, `3-5`, `>5`, `immediate`, `manual`, None}.
- Mapping numeric merchant delays into fee-rule buckets: `1`,`2` → `<3`; `3`,`4`,`5` → `3-5`; `6`+ → `>5`; `immediate` matches `immediate`; `manual` matches `manual`.
- `immediate` corresponds conceptually to capture_delay `0`; `manual` is a distinct non-numeric value.

## PIN Entry (business policy per manual)
- Max 3 consecutive incorrect PIN attempts before temporary card block (documented policy; no tool observed to enforce mechanically). Unblock requires contacting issuing bank.

## Data Access
- All reference data lives under `./data/`. There are no tools beyond `bash`; the agent must read files directly.
- Working directory contains only `data/` — no `docs/` or `data/context/` directory (accessing them errors with non-zero exit).
- Directory listing of `data/`: acquirer_countries.csv, fees.json, manual.md, merchant_category_codes.csv, merchant_data.json, payments-readme.md, payments.csv.
- No write/mutation tools observed — this is a read-only analytical environment.
- payments.csv contains only `year==2023` data; `day_of_year` spans 1..365 (full year).
