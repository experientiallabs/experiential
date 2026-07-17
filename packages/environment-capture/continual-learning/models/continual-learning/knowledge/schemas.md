# Schemas

## items_g1
- ref_id TEXT PRIMARY KEY
- ttl TEXT
- avg_rtg REAL              -- stored/denormalized; may drift from AVG(fdbk.rtg)
- avg_rtg_30d REAL
- rtg_ct INTEGER
- prc REAL                  -- often NULL (7,867/20,000 non-null; 0.18..13914.85)
- str_nm TEXT               -- store/brand name; often NULL
- main_cat TEXT             -- may be NULL; may disagree with taxn_g1 lvl-0
- desc_txt TEXT
- feat_lst TEXT
Indexes: prc, avg_rtg, str_nm

## fdbk_g1
- id INTEGER PK AUTOINCREMENT
- item_id TEXT              -- may differ from ref_id on same row
- ref_id TEXT (FK items_g1.ref_id)
- uid TEXT                  -- reviewer id; can be NULL
- rtg REAL                  -- observed values 1.0..5.0
- ttl TEXT
- body TEXT                 -- never NULL; 375 empty-after-trim
- ts INTEGER                -- epoch MILLISECONDS; some corrupt rows with ts ≥ 2e12
- hlp_ct INTEGER
- vrf INTEGER               -- 1=verified (351,745), 0=unverified (25,727)
Indexes: ref_id, rtg, ts, vrf

## taxn_g1
- ref_id TEXT (FK)
- cat_lvl INTEGER           -- 0..5 observed; level 0 = department (e.g. 'Office Products')
- cat_nm TEXT
Indexes: ref_id, cat_nm, cat_lvl

## attrs_g1
- ref_id TEXT (FK)
- attr_key TEXT             -- free-form; see entities for top keys
- attr_val TEXT             -- free-form; e.g. 'Date First Available' is English date like 'October 31, 2015'
Indexes: ref_id, attr_key

## fdbk_stats_g1
- ref_id TEXT PK (FK items_g1.ref_id)
- avg_rtg REAL
- review_ct INTEGER
- pct_5star REAL
- avg_hlp REAL

## items_g2
- ref_id TEXT PRIMARY KEY
- ttl TEXT
- avg_rtg REAL
- prc INTEGER               -- often NULL (6,570/20,000 non-null; 9..2,688,800). Appears to be prc_usd * 100 (cents) per-row, but aggregate averages diverge from prc_usd — prefer prc_usd, fall back via coalesce(prc_usd, prc/100.0).
- prc_usd REAL              -- often NULL (6,346/20,000 non-null; 0.09..499.99). NOT always populated when prc is.
- str_nm TEXT
- main_cat TEXT             -- department-ish label; distinct value space from taxn_g2 lvl-0. Never literally 'Electronics' or 'Computers & Accessories'.
- feat_lst TEXT
- img_ct INTEGER
Indexes: prc, avg_rtg, str_nm

## fdbk_g2
- id INTEGER PK AUTOINCREMENT
- item_id TEXT
- ref_id TEXT (FK items_g2.ref_id)
- uid TEXT
- rtg REAL
- ttl TEXT
- body TEXT                 -- 23 empty-after-trim out of 77,022
- ts INTEGER                -- epoch SECONDS (differs from g1!)
- hlp_ct INTEGER
- vrf TEXT                  -- note: TEXT here ('true'=71,517 / 'false'=5,505), INTEGER in g1/g3
Indexes: ref_id, rtg, ts, vrf

## taxn_g2
- ref_id TEXT (FK)
- cat_lvl INTEGER           -- 0..N; lvl 0 uniformly 'Electronics'; lvl 1 includes 'Computers & Accessories'
- cat_nm TEXT
Indexes: ref_id, cat_nm, cat_lvl

## items_g3
- ref_id TEXT PRIMARY KEY
- ttl TEXT
- avg_rtg REAL
- rtg_ct INTEGER
- prc REAL                  -- often NULL (7,949/20,000 non-null; 0.05..5369.0)
- str_nm TEXT               -- NULL in all sampled rows; use attrs_g3 Brand instead
- main_cat TEXT
- desc_txt TEXT
Indexes: prc, avg_rtg

## fdbk_g3
- id INTEGER PK AUTOINCREMENT
- ref_id TEXT (FK items_g3.ref_id)   -- no item_id column in g3
- uid TEXT
- rtg REAL
- ttl TEXT
- body TEXT                 -- 297 empty-after-trim out of 286,626
- ts TEXT                   -- 'YYYY-MM-DD' or NULL (differs from g1/g2)
- hlp_ct INTEGER
- vrf INTEGER               -- 1=264,096 / 0=22,530
Indexes: ref_id, rtg, ts, vrf

## attrs_g3
- ref_id TEXT (FK)
- attr_key TEXT             -- 202,543 rows total
- attr_val TEXT             -- Brand values are free-text (case may vary, e.g. 'GIBRALTAR' vs 'Gibraltar')
Indexes: ref_id, attr_key

## Full table list (from .tables)
- attrs_g1, attrs_g3, fdbk_g1, fdbk_g2, fdbk_g3, fdbk_stats_g1, items_g1, items_g2, items_g3, taxn_g1, taxn_g2, sqlite_sequence

## Missing tables (do not exist)
- `taxn_g3` — NOT present
- `attrs_g2` — NOT present
- `fdbk_stats_g2`, `fdbk_stats_g3` — NOT present

## Also
- `sqlite_sequence(name, seq)` — SQLite internal for AUTOINCREMENT.
