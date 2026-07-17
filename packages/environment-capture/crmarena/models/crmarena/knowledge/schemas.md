## query.py
- Invocation: `python3 query.py "<SQL>"`
- Output: JSON array of objects keyed by selected column names. Empty result → `[]`.
- Row cap: first 50 rows shown, followed by `...[showing first 50 rows; add LIMIT/WHERE to narrow]`.
- Supports multi-line SQL and CTEs (WITH clauses).
- Long text truncation is applied inside query.py before printing; downstream shell filters (e.g., `| tail`, `python3 -c` json extraction) cannot recover truncated content — the `...[truncated]` marker persists inside the field value. Use `SUBSTR(col, <offset>)` in SQL to page past the truncation point.

## schema.md
- A `schema.md` file at CWD documents tables/columns and describes date format as `YYYY-MM-DDTHH:MM:SSZ` (but actual stored format is `YYYY-MM-DDTHH:MM:SS.sss+0000` for datetime columns; `Order.EffectiveDate` is `YYYY-MM-DD`).

## Tables (columns)
- Account: Id, FirstName, LastName, PersonEmail, Phone (REAL), RecordTypeId, ShippingCity, ShippingState.
- Case: Id, Priority, Subject, Description, Status, ContactId, CreatedDate, ClosedDate, OrderItemId__c, IssueId__c, AccountId, OwnerId.
- CaseHistory__c: Id, CaseId__c, OldValue__c, NewValue__c, CreatedDate, Field__c. `Field__c` ∈ {`Case Creation`, `Owner Assignment`, `Case Closed`}.
- Contact: Id, FirstName, LastName, Email, AccountId, OwnerId.
- EmailMessage: Id, Subject, TextBody, ParentId, FromAddress, ToIds, MessageDate. `ToIds` is a JSON array string of Salesforce Ids.
- Issue__c: Id, Name.
- Knowledge__kav: Id, Title, FAQ_Answer__c, Summary, UrlName.
- LiveChatTranscript: Id, CaseId, AccountId, OwnerId, Body, EndTime, LiveChatVisitorId, ContactId.
- Order: Id, AccountId, Status (e.g., 'Activated'), EffectiveDate (YYYY-MM-DD), Pricebook2Id.
- OrderItem: Id, OrderId, Product2Id, Quantity (TEXT), UnitPrice (TEXT), PriceBookEntryId.
- Pricebook2: Id, Name, Description, IsActive (INTEGER), ValidFrom, ValidTo.
- PricebookEntry: Id, Pricebook2Id, Product2Id, UnitPrice (TEXT).
- Product2: Id, Name, Description, IsActive (INTEGER), External_ID__c.
- ProductCategory: Id, Name, CatalogId.
- ProductCategoryProduct: Id, ProductCategoryId, ProductId.
- User: Id, FirstName, LastName, Email, Phone (REAL), Username, Alias, LanguageLocaleKey, EmailEncodingKey, TimeZoneSidKey, LocaleSidKey.

## Id prefix conventions (Salesforce-style)
- `001` Account, `003` Contact, `005` User, `500` Case, `801` Order, `802` OrderItem, `01t` Product2, `570` LiveChatTranscript, `ka0` Knowledge__kav, `a1M` Issue__c, `a1N` CaseHistory__c, `02s` EmailMessage.

## Date formats
- Datetime columns (Case.CreatedDate, Case.ClosedDate, CaseHistory__c.CreatedDate, EmailMessage.MessageDate, LiveChatTranscript.EndTime, etc.): `YYYY-MM-DDTHH:MM:SS.sss+0000` (text).
- Date-only columns (Order.EffectiveDate): `YYYY-MM-DD`.
- SQLite `strftime()` and `julianday()` cannot parse the `+0000` suffix — return NULL. Use `substr(col,1,19)` to strip suffix, or `substr()` for date-part grouping, or `LIKE 'YYYY-MM-DD%'`.
