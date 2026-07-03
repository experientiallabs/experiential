This is documentation for the payments.csv dataset


- **Description**: Synthetic dataset of payment transactions processed by the Payments Processor.
- **Columns**:
  - `psp_reference`: Unique payment identifier (ID).
  - `merchant`: Merchant name (Categorical), eg Starbucks or Netflix*.
  - `card_scheme`: Card Scheme used (Categorical) - *[MasterCard, Visa, Amex, Other]*.
  - `year`: Payment initiation year (Numeric).
  - `hour_of_day`: Payment initiation hour (Numeric).
  - `minute_of_hour`: Payment initiation minute (Numeric).
  - `day_of_year`: Day of the year of payment initiation (Numeric).
  - `is_credit`: Credit or Debit card indicator (Categorical).
  - `eur_amount`: Payment amount in euro (Numeric).
  - `ip_country`: The country the shopper was in at time of transaction (determined by IP address) (Categorical) - *[SE, NL, LU, IT, BE, FR, GR, ES]*.
  - `issuing_country`: Card-issuing country (Categorical) - *[SE, NL, LU, IT, BE, FR, GR, ES]*.
  - `device_type`: Device type used (Categorical) - *[Windows, Linux, MacOS, iOS, Android, Other]*.
  - `ip_address`: Hashed shopper's IP (ID).
  - `email_address`: Hashed shopper's email (ID).
  - `card_number`: Hashed card number (ID).
  - `shopper_interaction`: Payment method (Categorical) - *[Ecommerce, POS]*. POS means an in-person or in-store transaction.
  - `card_bin`: Bank Identification Number (ID).
  - `has_fraudulent_dispute`: Indicator of fraudulent dispute from issuing bank (Boolean).
  - `is_refused_by_adyen`: Adyen refusal indicator (Boolean).
  - `aci`: Authorization Characteristics Indicator (Categorical).
  - `acquirer_country`: The location (country) of the acquiring bank (Categorical) - *[SE, NL, LU, IT, BE, FR, GR, ES]*.