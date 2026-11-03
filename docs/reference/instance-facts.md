# Instance facts from task input files

Instance facts describe one task instance's starting world (its workbook cells, orders or
tickets) and condition simulated predictions in their own `instance_facts` channel, apart from
learned memory. When a task's starting world is a file the environment backend can read, the
backend derives those facts itself, validates them against its own reader, and refuses anything
it cannot represent exactly. The workbook backend is the first such deriver.

## Derive and attach: `derive-facts`

`exp eval PROJECT --continuations SPEC.json` with `action: "derive-facts"` runs one deriver over
one declared starting file per trace. It makes no provider call.

```json
{
  "action": "derive-facts",
  "facts": {
    "deriver": {"kind": "xlsx-workbook"},
    "inputs": [
      {"trace": "spreadsheetbench-105-24", "file": "/data/105-24/1_105-24_init.xlsx"},
      {"trace": "spreadsheetbench-13-1", "file": "/data/13-1/1_13-1_init.xlsx", "path": "input.xlsx"}
    ],
    "traces": "traces.chat.json",
    "output": "traces.facts.chat.json"
  }
}
```

- `deriver`: the backend deriver and its bounds. `xlsx-workbook` accepts `max_bytes` (default
  1 MiB, at most 8 MiB) and `allow_unrepresented` (default false, see below).
- `inputs`: `trace` is the trace package's key for that task's trace (for chat JSON, the
  conversation's `trace_id`, `conversation_id`, `session_id`, `thread_id` or `id`, the first
  present). `file` is the host file, read once and never recorded. `path` is the file's path
  inside the task's world (default `input.xlsx`); it names the fact (`input.xlsx/workbook`)
  and its provenance source, together with the file's SHA-256. A trace may declare several files
  at different paths.
- `traces` and `output` (together, optional): a chat JSON or JSONL trace package and the new
  file to write. Each matched conversation's `exp.request.context` gains `instance_facts`; every
  trace with facts must match a conversation, and a conversation that already declares facts is
  refused. Import the copy with `exp build PROJECT --traces OUTPUT --source chat-json`: import
  moves the facts out of the request context, and learning, its retest and `exp eval --audit`
  deliver them to that trace's own predictions (see `docs/usage.md`).
- `allow_refusals` (default false): any refused input stops the action before anything is
  written. Set it to report each refusal and give those traces no facts.

The action prints the counts, every refusal, the import command and a `Report:` line. It saves
`<project>/runtime/facts/<identity>/facts.json`, mapping each trace key to a `ReplayFacts` value
(`{"facts": ..., "first_version": 0}`). That value is what a `GroundedRuntimeDefinition` takes as
`facts`, so the same facts condition a continuation job, a mixed definition's `simulated` part or
an executable task built on it. `--dry-run` derives and checks the package match, then writes
nothing.

In Python, any backend deriver is called the same way:

```python
from exp.runtime.environments.workbook_facts import WorkbookFactsDeriver, decode_workbook_facts

facts = WorkbookFactsDeriver().derive("input.xlsx", payload)  # InstanceFacts, validated
cells = decode_workbook_facts(facts.facts[0].value).sheet("Sheet1").cells
cells["B2"].value, cells["B2"].data_type, cells["B2"].number_format, cells["B2"].cached
```

A new backend implements `exp.common.tasks.facts.InstanceFactsDeriver` (a `kind` and
`derive(path, payload)`), and joins `FactsDeriverDefinition` in
`exp/runtime/environments/facts_derivers.py`, discriminated on `kind`.

## What the workbook facts state

The authority is openpyxl at the installed version, loaded the way an agent's program loads the
file: `load_workbook(path)` for stored values and formulas and `load_workbook(path,
data_only=True)` for cached formula results. The facts carry, for every stored cell, the typed
value, the formula and its cached value, openpyxl's stored cell type and the number format, plus
sheet order and state (hidden sheets, chartsheets), `Worksheet.dimensions`, merged ranges,
comments, hyperlinks, defined names, tables, and the ranges of data validations, conditional
formats and the auto filter. Cells that a program cannot tell from absent ones (empty, General,
no comment or link) are left out; the recorded dimensions still give the used range. Fonts,
fills, borders, widths, heights, hidden rows, views and print setup are presentation, outside
the facts.

The value is row oriented and self-describing; its `legend` field explains it to the simulator:

```json
{"codec": "xlsx-instance-facts-v1", "reader": "openpyxl 3.1.5", "coverage": "complete",
 "inventory": [{"name": "Sheet", "dimensions": "A1:D3", "cells": 12, "formulas": 2,
                "uncached_formulas": 2}],
 "number_formats": ["General", "yyyy-mm-dd h:mm:ss"],
 "sheets": [{"name": "Sheet", "dimensions": "A1:D3",
             "rows": [[1, 1, "date", "name", "amount", "double"],
                      [2, 1, {"d": "2023-02-12"}, "Alice", 4667.98, {"f": "=C2*2"}],
                      [3, 1, {"d": "2023-02-13"}, "Bob", 3959, {"f": "=C3*2"}]],
             "formats": [[1, 2, 3, 1]]}]}
```

(`legend` omitted here.) This workbook was saved by openpyxl, so its formulas have no cached
values; an Excel-saved formula reads `{"f": "=C2*2", "c": 9335.96}`.

A row segment is `[row, first column, values...]`; `null` is no stored cell and `{}` a stored
empty cell. Strings, numbers and booleans are plain JSON; other values are tagged: `{"d": ISO
datetime}`, `{"date": ...}`, `{"t": ISO time}`, `{"td": [days, seconds, microseconds]}`, `{"e":
error}`, `{"f": formula, "c": cached}` and `{"af": array formula, "ref": range, "c": cached}`.
Format runs `[column, first row, last row, index]` give non-General number formats.

## Validation, bounds and refusals

Before a fact is returned, its value is serialized and parsed back, decoded into typed values,
and compared with both openpyxl loads cell by cell: every stored cell by Python type and exact
`repr` of the formula-mode value and the cached value, the stored type and number format; the
sheet order, state, dimensions and features exactly. Any difference refuses the workbook. This
pins the cases a float-day decoder gets wrong: a serial below one day with a time format is a
`time`, not a 1899-12-30 datetime; a fractional serial is openpyxl's millisecond-rounded datetime;
`[h]:mm` values are durations; inline strings exclude phonetic runs; and a value Excel keeps under
a merged range's hidden cells is not a cell, since openpyxl reads it as a merged cell.

The deriver refuses (`WorkbookFactsRefused`, an `InstanceFactsError`) with the reason and the way
forward:

- archive parts whose content lies outside the facts, by declared content type: drawings,
  images, charts and chartsheets, pivot tables, external links, macros and ActiveX, form
  controls, threaded comments, slicers, data connections, embedded objects, and unrecognized
  parts. With `allow_unrepresented: true` the cell facts are derived and each such part is
  listed under `unrepresented` by name and kind, never dropped silently;
- data-table formulas and cell value types the encoding does not define;
- inputs that are not readable XLSX archives (legacy `.xls`, encrypted, corrupt), XML entity
  declarations, archives over 32 MiB or with more than 10,000 parts or 256 MiB expanded;
- more than 500,000 stored cells, or an encoded value over `max_bytes`.

openpyxl's own load warnings are kept under `reader_warnings`.

## Views for prompts

`slice_workbook_facts(value, {"Sheet1": ["A1:D20"], "Lookup": None})` returns a bounded view in
the same encoding: the full inventory, only the selected cells (and their comments and links),
and a `coverage` map naming what was included. Cells outside coverage are unknown, not empty. A
caller that knows an action's read set delivers that slice instead of the whole workbook.

## Coverage on SpreadsheetBench starting workbooks

Offline, with openpyxl 3.1.5: of the 106 initial workbooks of the SpreadsheetBench wide pilot,
97 derive under the default policy and all 106 with `allow_unrepresented`. The nine default
refusals hold drawings (7, two with images and two with form controls), an external link (1) and
pivot tables (1). The nine workbooks an earlier float-day decoder failed on (time-only values,
sub-second datetimes, a merged shadow value) all validate. Encoded sizes have a median of 2.4 KB
and a maximum of 427 KB (1.7 MB for all 97). Across all 395 initial workbooks of the verified 400
release, 360 derive under the default policy and all 395 with `allow_unrepresented`.
