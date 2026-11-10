---
name: xlsx
description: Create or edit Excel workbooks (.xlsx) with openpyxl, including live formulas. Use when the user asks for a spreadsheet, an Excel file, a financial model, a table they will open in Excel, or totals and calculations that must stay live.
compatibility: Requires python3 with openpyxl (pip install openpyxl). pandas is optional for loading data.
---

# Excel workbooks

1. Check the library: `python3 -c "import openpyxl"`. If it fails, run `pip install openpyxl`.
2. Write a Python script in the working directory and run it. Keep the script.
3. Layout:
   - One sheet per topic, with a clear sheet title (`ws.title = "Revenue"`).
   - Bold header row (`Font(bold=True)`), frozen below the header (`ws.freeze_panes = "A2"`), and an autofilter over the data (`ws.auto_filter.ref = ws.dimensions`).
   - Sensible column widths (`ws.column_dimensions["A"].width = 24`).
   - Number formats: `"#,##0"` for counts, `"#,##0.00"` for money, `"0.0%"` for percentages, `"yyyy-mm-dd"` for dates.
4. Formulas, not pasted values, wherever a number depends on other cells:
   - Write the formula as a string: `ws["B21"] = "=SUM(B2:B20)"`, `ws["D2"] = "=B2/C2"`.
   - Keep inputs and assumptions in clearly labeled cells and reference them, so the user can change one input and see the model update.
   - Never compute a total in Python and write the result where a formula belongs.
5. openpyxl does not calculate formulas. To check results, recompute the same numbers in Python (or with pandas) and compare them, and say in your answer that values appear when the file is opened in Excel.
6. To edit an existing workbook, load it with `load_workbook(path)`; pass `data_only=True` only to read cached values, never when you will save, because that drops the formulas.
7. In your answer, give the file name, the sheets, and the key figures.
