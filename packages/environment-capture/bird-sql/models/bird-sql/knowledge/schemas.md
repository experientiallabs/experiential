## Superhero DB (SQLite)
```
alignment(id PK, alignment)
attribute(id PK, attribute_name)
colour(id PK, colour)          -- includes 7 Blue
gender(id PK, gender)          -- rows: 1 Male, 2 Female, 3 N/A
publisher(id PK, publisher_name)
race(id PK, race)
superpower(id PK, power_name)
superhero(id PK, superhero_name, full_name,
          gender_id→gender, eye_colour_id→colour, hair_colour_id→colour,
          skin_colour_id→colour, race_id→race, publisher_id→publisher,
          alignment_id→alignment, height_cm INTEGER, weight_kg INTEGER)
hero_attribute(hero_id→superhero, attribute_id→attribute, attribute_value INTEGER)
hero_power(hero_id→superhero, power_id→superpower)
```

## Molecule / Toxicology DB (SQLite)
```
molecule(molecule_id TEXT PK, label TEXT)          -- label ∈ {+,-}
atom(atom_id TEXT PK, molecule_id→molecule, element TEXT)  -- element lowercase
bond(bond_id TEXT PK, molecule_id→molecule, bond_type TEXT)  -- bond_type ∈ {-,=,#}
connected(atom_id→atom, atom_id2→atom, bond_id→bond,
          PK(atom_id,atom_id2), ON DELETE/UPDATE CASCADE)
          -- each bond stored as two symmetric rows (a,b) and (b,a)
```

## Student Club DB (SQLite)
```
event(event_id TEXT PK, event_name TEXT, event_date TEXT (YYYY-MM-DDTHH:MM:SS),
      type TEXT, notes TEXT, location TEXT, status TEXT)
major(major_id TEXT PK, major_name, department, college)
zip_code(zip_code INTEGER PK, type, city, county, state, short_state)
member(member_id TEXT PK, first_name, last_name, email, position,
       t_shirt_size, phone, zip→zip_code, link_to_major→major)
attendance(link_to_event→event, link_to_member→member, PK both)
budget(budget_id TEXT PK, category, spent REAL, remaining REAL,
       amount INTEGER, event_status, link_to_event→event)
expense(expense_id TEXT PK, expense_description, expense_date TEXT (YYYY-MM-DD), cost REAL,
        approved TEXT (`true`/`false`; only `true` observed), link_to_member→member, link_to_budget→budget)
income(income_id TEXT PK, date_received, amount INTEGER, source, notes,
       link_to_member→member)
```

## California Schools DB (SQLite)
```
frpm(CDSCode TEXT PK →schools,
     `Academic Year`, `County Code`, `District Code` INTEGER, `School Code`,
     `County Name`, `District Name`, `School Name`,
     `District Type`, `School Type`, `Educational Option Type`,
     `NSLP Provision Status`, `Charter School (Y/N)` INTEGER,
     `Charter School Number`, `Charter Funding Type`, IRC INTEGER,
     `Low Grade`, `High Grade`,
     `Enrollment (K-12)` REAL, `Free Meal Count (K-12)` REAL,
     `Percent (%) Eligible Free (K-12)` REAL,
     `FRPM Count (K-12)` REAL, `Percent (%) Eligible FRPM (K-12)` REAL,
     `Enrollment (Ages 5-17)` REAL, `Free Meal Count (Ages 5-17)` REAL,
     `Percent (%) Eligible Free (Ages 5-17)` REAL,
     `FRPM Count (Ages 5-17)` REAL, `Percent (%) Eligible FRPM (Ages 5-17)` REAL,
     `2013-14 CALPADS Fall 1 Certification Status` INTEGER)
satscores(cds TEXT PK →schools, rtype TEXT NOT NULL, sname, dname, cname,
          enroll12 INTEGER NOT NULL, NumTstTakr INTEGER NOT NULL,
          AvgScrRead INTEGER, AvgScrMath INTEGER, AvgScrWrite INTEGER,
          NumGE1500 INTEGER)
          -- schema.sql also contains commented-out `PctGE1500 double` (NOT in table)
          -- sname may be NULL or ''
schools(CDSCode TEXT PK, NCESDist, NCESSchool, StatusType NOT NULL, County NOT NULL,
        District NOT NULL, School, Street, StreetAbr, City, Zip, State,
        MailStreet, MailStrAbr, MailCity, MailZip, MailState,
        Phone, Ext, Website, OpenDate DATE, ClosedDate DATE,
        Charter INTEGER, CharterNum, FundingType,
        DOC NOT NULL, DOCType NOT NULL, SOC, SOCType, EdOpsCode, EdOpsName,
        EILCode, EILName, GSoffered, GSserved, Virtual, Magnet INTEGER,
        Latitude REAL, Longitude REAL,
        AdmFName1, AdmLName1, AdmEmail1,
        AdmFName2, AdmLName2, AdmEmail2,
        AdmFName3, AdmLName3, AdmEmail3,
        LastUpdate DATE NOT NULL)
        -- StatusType ∈ {Active, Closed, Merged, Pending}
        -- Virtual ∈ {F, P, N}
        -- FundingType ∈ {Directly funded, Locally funded, Not in CS funding model, NULL}
        -- DOCType includes: County Office of Education (COE), Unified School District,
        --                   Elementary School District, High School District
        -- DOC observed values: 31, 52, 54
        -- SOC observed values: 60, 62, 65, 66
        -- OpenDate format: YYYY-MM-DD; usable via strftime('%Y', OpenDate)
        -- AdmEmail1/2/3 may be empty strings
```

## Tool response shapes
- `bash` observations: raw stdout text; `is_error=False` on success. No JSON envelope.
- `sqlite3` errors: `is_error=True`, stderr text formatted as `Error: in prepare, <message>\n  <query line>\n         ^--- error here`.
