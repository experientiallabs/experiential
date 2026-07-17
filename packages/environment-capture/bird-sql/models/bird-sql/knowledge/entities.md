## Databases observed (each session uses one `database.db`)

### 1. Superhero DB
- Tables: `alignment`, `attribute`, `colour`, `gender`, `publisher`, `race`, `superhero`, `hero_attribute`, `superpower`, `hero_power`.
- `superhero` links to lookup tables via `gender_id`, `eye_colour_id`, `hair_colour_id`, `skin_colour_id`, `race_id`, `publisher_id`, `alignment_id`; stores `height_cm`, `weight_kg`.
- `hero_attribute(hero_id, attribute_id, attribute_value)` — junction w/ numeric value.
- `hero_power(hero_id, power_id)` — junction, powers named in `superpower.power_name` (includes e.g. `Phoenix Force`, `Agility`, `Super Strength`, `Stamina`, `Super Speed`, `Accelerated Healing`).
- Only 1 hero has `Phoenix Force` power (gender: Female).
- Known attribute names include `Durability`.
- `gender` table full contents: `1|Male`, `2|Female`, `3|N/A`.
- Male:Female hero ratio ≈ 2.5567 (SUM(gender_id=1)/SUM(gender_id=2)).
- `colour` table includes `7|Blue`; other colours include Gold, Black.
- ~31.2% of superheroes have `eye_colour_id=7` (Blue).
- 2 Marvel Comics heroes have Gold eye colour.
- Publishers include: Marvel Comics, DC Comics, Image Comics, Dark Horse Comics, Wildstorm, NBC - Heroes, Icon Comics, SyFy, Star Trek, HarperCollins, Hanna-Barbera, ABC Studios, Sony Pictures, Universal Studios, George Lucas, J. R. R. Tolkien, Rebellion, Shueisha; some heroes have empty publisher.
- Hero names include `3-D Man`, `A-Bomb`, `Abe Sapien`, `Adam Monroe`, `Adam Strange`, `Agent 13`, `Agent Bob`, `Agent Zero`, `Alex Mercer`, `Alex Woolsly`.
- ~150 heroes have hair_colour_id = skin_colour_id = eye_colour_id (across many publishers).

### 2. Molecule / Toxicology DB
- Tables: `molecule(molecule_id PK, label)`, `atom(atom_id PK, molecule_id FK, element)`, `bond(bond_id PK, molecule_id FK, bond_type)`, `connected(atom_id, atom_id2, bond_id)` composite PK w/ ON DELETE/UPDATE CASCADE.
- `molecule_id` values look like `TR000`, `TR001`, ... `atom_id` like `TR000_1` (molecule_id + `_` + index; index NOT zero-padded, so has variable width — 1..9, 10..99, etc.).
- `bond_id` observed as `<molecule_id>_<i>_<j>` (e.g. `TR001_6_9`).
- `connected` stores each bond as TWO rows (both directions): `(atom_id, atom_id2)` and `(atom_id2, atom_id)` with same `bond_id`.
- `molecule.label` domain observed: `+`, `-` (activity classification). Approx 44.315% of molecules are labelled `+`.
- `bond.bond_type` domain observed: `-`, `=`, `#` (single/double/triple).
- Only 3 molecules contain triple bonds (`#`): `TR041`, `TR377`, `TR499`.
- `atom.element` stored lowercase (observed: `ca`, `k`, `pb`, `i`, `s`, `br`, `c`, `cl`, `f`, `h`, `n`, `na`, `o`, `p`, `sn`, `y`). Elements observed connected via `=` bonds: `c`, `o`, `n`, `s`, `ca`.
- Rarest elements among label='-' molecules (ascending): `ca` (1), `k` (1), `pb` (1), `sn` (2), `i` (3).
- Molecule `TR006` label `+`; ~36.17% of its atoms are hydrogen.
- Among `+` molecules, atoms with atom-index starting `4` (i.e. `SUBSTR(atom_id,7,1)='4'`) element counts: c=145, h=59, o=33, cl=9, n=8, br=6, s=5, na=3, f=1.

### 3. Student Club DB
- Tables: `event`, `major`, `zip_code`, `attendance`, `budget`, `expense`, `income`, `member`.
- `event(event_id PK, event_name, event_date, type, notes, location, status)` — `event_date` is TEXT ISO-like timestamp `YYYY-MM-DDTHH:MM:SS`; status values include `Closed`; `type` values include `Meeting`, `Guest Speaker`; event names include recurring `September Speaker`, `October Speaker`, `November Speaker`, `Officers meeting - September`, `Officers meeting - October`, `Officers meeting - November`, `October Meeting`.
- Sample `event_id`s: `recEVTik3MlqbvLFi` (October Speaker, 2019-10-22T12:00:00, MU 215), `recc8dizaKrSz3GmH` (Officers meeting - October, 2019-10-08T09:30:00), `recggMW2eyCYceNcy` (October Meeting, 2019-10-08T12:00:00, MU 215).
- `major(major_id PK, major_name, department, college)`; includes `Environmental Engineering` (department `Civil and Environmental Engineering Department`, college `College of Engineering`).
- `zip_code(zip_code PK INTEGER, type, city, county, state, short_state)`.
- `member(member_id PK, first_name, last_name, email, position, t_shirt_size, phone, zip→zip_code, link_to_major→major)`; `t_shirt_size` values include `Medium`; phone format `NNN-NNN-NNNN` (e.g. `928-555-2577` for Carlo Jacobs).
- `budget(budget_id PK, category, spent, remaining REAL, amount, event_status, link_to_event)`; max observed `spent` = 327.07; `remaining` can be negative (over-budget events include September Speaker −23.06, Officers meetings ≈ −0.2).
- `expense(expense_id PK, expense_description, expense_date TEXT `YYYY-MM-DD`, cost REAL, approved TEXT (`true`/`false` per schema; only `true` observed in data), link_to_member, link_to_budget)`; observed dates in 2019 (e.g. `2019-08-20`, `2019-09-10`, `2019-10-08`, `2019-10-10`, `2019-11-19`); minimum observed cost 6.0 (Speaker events); expense descriptions include `Posters`, `Water, chips, cookies`, `Pizza`.
- `income(income_id PK, date_received, amount, source, notes, link_to_member)`.
- `attendance(link_to_event, link_to_member)` composite PK.

### 4. California Schools DB
- Tables: `frpm`, `satscores`, `schools`.
- `schools.CDSCode` is PK; `frpm.CDSCode` and `satscores.cds` are FKs to it.
- `frpm` uses spaced/backticked column names, e.g. `Academic Year`, `County Code`, `District Code`, `School Code`, `County Name`, `District Name`, `School Name`, `Charter School (Y/N)`, `Enrollment (K-12)`, `Free Meal Count (K-12)`, `Percent (%) Eligible Free (K-12)`, `FRPM Count (K-12)`, plus `(Ages 5-17)` variants, `2013-14 CALPADS Fall 1 Certification Status`.
- `schools` includes `DOC` (district ownership code; `31` = state special schools like Schools for the Deaf/Blind — includes `California School for the Deaf-Fremont` (enrollment 410), `California School for the Deaf-Riverside` (355), `California School for the Blind` (60); other observed DOC: `52`, `54`), `DOCType`, `SOC` (school ownership code; observed: `60`, `62`, `65`, `66`), `SOCType`, `EdOpsCode`, `EILCode`, `Charter`, `FundingType`, `Latitude`, `Longitude`, admin contact fields `AdmFName1..3`, `AdmLName1..3`, `AdmEmail1..3`, `OpenDate` (DATE `YYYY-MM-DD`), `ClosedDate`, `LastUpdate`.
- `schools.StatusType` domain: `Active`, `Closed`, `Merged`, `Pending`.
- `schools.Virtual` domain observed: `F`, `P`, `N` (single-letter codes).
- `schools.FundingType` domain observed: `Directly funded`, `Locally funded`, `Not in CS funding model`, NULL/empty.
- `schools.DOCType` values include: `County Office of Education (COE)`, `Unified School District`, `Elementary School District`, `High School District`.
- `schools.GSoffered` values include e.g. `K-8`, `9-12`; may be NULL/empty.
- Westernmost observed `Longitude` ≈ `-124.28481` (tied K-8 and NULL-GSoffered schools); next `-124.27296` (9-12).
- Orange County `StatusType='Merged'` schools by DOC: DOC 52 → 7 schools, DOC 54 → 4 schools (ratio 54/52 ≈ 0.5714).
- Average (`Enrollment (K-12)` − `Enrollment (Ages 5-17)`) for `FundingType='Locally funded'` ≈ 16.7006.
- Average `satscores.NumTstTakr` for Fresno county schools with `strftime('%Y',OpenDate)='1980'` ≈ 137.89 (9 schools).
- San Bernardino City Unified schools opened 2009–2010 with SOC='62' AND DOC='54': `New Vision Middle` (admin emails a.lucero@realjourney.org, j.hernandez@realjourney.org).
- School names observed include `Buchanan High`, `FAME Public Charter`, `Envision Academy for Arts & Technology`, `Aspire California College Preparatory Academy`, `Alameda Science and Technology Institute`, `Nea Community Learning Center`, `Dunlap Leadership Academy`, `Academy of Arts and Sciences: Fresno`, `Insight School of California`, `California Virtual Academy @ Kings`, `National University Academy, Armona`, `Desert Sands Charter`, `iQ Academy California-Los Angeles`, `River Springs Charter` (Riverside, Directly funded), plus many Riverside County high schools (La Sierra High, Norco High, Palm Desert High, Temecula Valley High, Eleanor Roosevelt High, John F. Kennedy High, Centennial High, Santiago High, Norte Vista High, Banning High, Beaumont Senior High).
- `satscores(cds PK, rtype, sname, dname, cname, enroll12, NumTstTakr, AvgScrRead, AvgScrMath, AvgScrWrite, NumGE1500)`; `rtype`, `enroll12`, `NumTstTakr` are NOT NULL; `sname` may be NULL or `''`.
- `satscores.cname` uses county name without " County" suffix (e.g. `Riverside`); matches `schools.County` and `frpm."County Name"`.
- Avg `AvgScrMath` for Riverside county (non-null) ≈ 465.43.
- 57 `Locally funded` schools have `Enrollment (K-12) - Enrollment (Ages 5-17)` above the average for locally-funded schools.
