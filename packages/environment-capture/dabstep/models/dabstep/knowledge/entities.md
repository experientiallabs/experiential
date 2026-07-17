# Entities

## Data Files (./data/)
- `manual.md` — Merchant Guide v2.1 (2024-11-01). Sections: 1.Introduction, 2.Account Type, 3.MCC, 4.ACI, 5.Understanding Payment Processing Fees (5.1.1 Local Acquiring, 5.1.2 Choosing ACI, 5.1.3 Higher Volumes, 5.1.4 Fraud Costs, 5.1.5 Avoiding Downgrades), 6.PIN Entry Attempt Limits, 7.Reducing Fraud-Related Fees (7.1 Proactive Prevention, 7.2 Chargebacks, 7.3 Team Education, 7.4 PCI DSS Compliance), 8.Leveraging Data and Reporting (8.1 Transaction Data Analysis, 8.2 Reporting Tools/KPIs), 9.Appendix/Glossary, 10.Contact Information.
- `fees.json` — 1000 fee rule records.
- `payments.csv` — transaction records (year 2023 only observed; day_of_year 1..365).
- `payments-readme.md` — payments dataset docs.
- `merchant_data.json` — merchant configurations.
- `merchant_category_codes.csv` — MCC list.
- `acquirer_countries.csv` — acquirer country mapping.

## Account Types (single-letter codes)
- R: Enterprise - Retail
- D: Enterprise - Digital
- H: Enterprise - Hospitality
- F: Platform - Franchise
- S: Platform - SaaS
- O: Other

## Authorization Characteristics Indicator (ACI)
- A: Card present - Non-authenticated
- B: Card Present - Authenticated
- C: Tokenized card with mobile device
- D: Card Not Present - Card On File
- E: Card Not Present - Recurring Bill Payment
- F: Card Not Present - 3-D Secure
- G: Card Not Present - Non-3-D Secure

## Card Schemes (observed in fees.json / payments.csv)
- TransactPlus, NexPay, SwiftCharge, GlobalCard

## Example Merchant
- `Belles_cookbook_store`: capture_delay=`'1'`, acquirer=[`lehman_brothers`], MCC=5942, account_type=`R`.
  - Jan 2023: 1201 transactions total; monthly total volume ≈ €113,260.42 (falls in `100k-1m` bucket).
  - Jan 2023 fraud volume ≈ €11,680.62; fraud level (fraud-eur / total-eur) ≈ 10.31% (falls in `>8.3%` bucket).
  - Jan 2023 fraudulent disputes: 94 rows, ALL with aci=`G`, all `is_credit=True`, all `acquirer_country=US`, all `shopper_interaction=Ecommerce` (issuing_country spread across BE, ES, FR, GR, IT ⇒ all international/intracountry=False).
  - Jan 2023 fraud rows by card_scheme: GlobalCard 31, TransactPlus 30, NexPay 19, SwiftCharge 14.
  - Day 10 (2023-01-10): 37 transactions; all with `acquirer_country=US`; card_schemes span {NexPay, GlobalCard, TransactPlus, SwiftCharge}; ACIs seen {A,C,D,F,G}; is_credit both True/False.
  - Fee rules matching `account_type==['R']` AND `aci==['B']` exactly: IDs [236, 368, 404, 539, 564, 757] (6 rules).
  - Fee rules with `'R' ∈ account_type` AND `'B' ∈ aci` (50 rules): [34, 39, 49, 62, 68, 82, 154, 220, 231, 236, 265, 276, 278, 286, 329, 345, 352, 355, 360, 368, 369, 390, 393, 404, 419, 512, 539, 556, 564, 583, 587, 590, 638, 645, 661, 711, 717, 731, 757, 779, 793, 828, 837, 871, 915, 938, 939, 964, 986, 998].
  - 62 rules pass mcc+account+capture_delay filter: [36, 51, 53, 64, 65, 80, 107, 123, 150, 154, 163, 183, 229, 230, 231, 249, 276, 286, 304, 319, 347, 381, 384, 394, 398, 428, 454, 470, 471, 473, 477, 498, 536, 556, 572, 595, 602, 606, 608, 626, 631, 642, 678, 680, 700, 709, 722, 725, 741, 813, 839, 849, 861, 868, 871, 892, 895, 924, 939, 942, 960, 965]; 47 rules remain after adding volume+fraud filters.
  - Illustrative cheapest-per-tx fee totals over the 94 Jan fraud rows if re-priced under each ACI (empty-list treated as catch-all; None indicates rows with no matching rule): A=€81.81, B=€54.21, C=€81.04, D=€36.62 (50 rows unmatched), E=€16.63 (50 unmatched), F=€33.56 (64 unmatched), G=€61.05 (64 unmatched).

## Example Fee Rules
- ID 1: `{card_scheme:'TransactPlus', account_type:[], capture_delay:None, monthly_fraud_level:None, monthly_volume:None, merchant_category_code:[8000,8011,8021,8031,8041,7299,9399,8742], is_credit:False, aci:['C','B'], fixed_amount:0.1, rate:19, intracountry:None}`.
- ID 384: `{card_scheme:'NexPay', account_type:[], capture_delay:None, monthly_fraud_level:None, monthly_volume:None, merchant_category_code:[], is_credit:True, aci:['C','B'], fixed_amount:0.05, rate:14, intracountry:None}`.

## Fee-rule Enumerated Values
- `capture_delay`: {`manual`, `<3`, `3-5`, `>5`, `immediate`, None}
- `monthly_fraud_level`: {`<7.2%`, `7.2%-7.7%`, `7.7%-8.3%`, `>8.3%`, None}
- `monthly_volume`: {`<100k`, `100k-1m`, `1m-5m`, `>5m`, None}

## Merchant-data Enumerated Values
- `capture_delay` values observed: {`immediate`, `1`, `2`, `7`, `manual`}.

## Payments Dataset Enumerations (per payments-readme.md)
- `ip_country`, `issuing_country`, `acquirer_country` (readme): {SE, NL, LU, IT, BE, FR, GR, ES}. In practice `acquirer_country` can also take values outside this list (e.g. `US` observed on Belles_cookbook_store transactions).
- `device_type`: {Windows, Linux, MacOS, iOS, Android, Other}
- `shopper_interaction`: {Ecommerce, POS}
- Readme lists card schemes as [MasterCard, Visa, Amex, Other] but actual data uses {TransactPlus, NexPay, SwiftCharge, GlobalCard}.

## Contact Info (per manual)
- Merchant Support: 1-800-555-1234, support@paymentprocessor.com, www.paymentprocessor.com/support
- Fraud Prevention: 1-800-555-5678, fraud@paymentprocessor.com
- Technical Support: 1-800-555-9876, tech@paymentprocessor.com

## Compliance
- PCI DSS non-compliance penalty: EUR 5,000–100,000 per month.

## Transaction Downgrade Causes (per manual §5.1.5)
- Missing/incomplete data elements, late settlement, non-qualified transaction types, failure to use AVS or 3DS for CNP, excessive transaction size/volume, excessive retrying.

## Glossary (per manual §9)
- AVS: Address Verification Service
- CVV: Card Verification Value
- PCI DSS: Payment Card Industry Data Security Standard
- ACI: Authorization Characteristics Indicator
