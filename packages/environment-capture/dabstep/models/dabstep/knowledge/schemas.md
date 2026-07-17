# Schemas

## fees.json (list of records, length 1000)
- `ID`: int — rule identifier
- `card_scheme`: string — e.g. TransactPlus, NexPay, SwiftCharge, GlobalCard
- `account_type`: list[string] — [] means all
- `capture_delay`: string|null — one of `<3`,`3-5`,`>5`,`immediate`,`manual`,null
- `monthly_fraud_level`: string|null — bucket like `7.7%-8.3%`, `<7.2%`, `>8.3%`, or null
- `monthly_volume`: string|null — bucket like `100k-1m` or null
- `merchant_category_code`: list[int] — [] means all
- `is_credit`: bool|null — null means both
- `aci`: list[string] — [] means all
- `fixed_amount`: float — euros per transaction
- `rate`: int — multiplied by tx value, divided by 10000
- `intracountry`: bool|float|null — 1.0/True=domestic, 0.0/null=other; null means both
- Example record: `{ID:1, card_scheme:'TransactPlus', account_type:[], capture_delay:null, monthly_fraud_level:null, monthly_volume:null, merchant_category_code:[8000,8011,8021,8031,8041,7299,9399,8742], is_credit:false, aci:['C','B'], fixed_amount:0.1, rate:19, intracountry:null}`
- Distribution note: in first 50 records, `account_type=[]` in 42/50; most rules use catch-all account_type.

## merchant_data.json (list of records)
- `merchant`: string
- `capture_delay`: string (raw integer-string like `'1'`, or bucket `'immediate'`/`'manual'`)
- `acquirer`: list[string]
- `merchant_category_code`: int
- `account_type`: string (single letter)
- Example: `{merchant:'Belles_cookbook_store', capture_delay:'1', acquirer:['lehman_brothers'], merchant_category_code:5942, account_type:'R'}`

## payments.csv columns
`psp_reference, merchant, card_scheme, year, hour_of_day, minute_of_hour, day_of_year, is_credit, eur_amount, ip_country, issuing_country, device_type, ip_address, email_address, card_number, shopper_interaction, card_bin, has_fraudulent_dispute, is_refused_by_adyen, aci, acquirer_country`

- `day_of_year`: 1..365 (January = 1..31; 2023 non-leap: Mar 1=60, Mar 31=90)
- `has_fraudulent_dispute`, `is_refused_by_adyen`, `is_credit`: booleans (serialized as `True`/`False` strings in CSV; pandas reads as Python bool)
- `shopper_interaction`: e.g. `Ecommerce`, `POS`
- `aci`: single-letter code A–G
- `psp_reference`: unique payment ID (int)
- `eur_amount`: float
- `year`: int (all observed rows are 2023)
- Hashed ID fields: `ip_address`, `email_address`, `card_number`, `card_bin`
