---
name: data-analysis
description: Analyze tabular data (CSV, Excel, JSON) with pandas and report findings. Use when the user gives you a dataset to explore, clean, summarize, join, or answer questions about.
compatibility: Requires python3 with pandas (pip install pandas); reading .xlsx also needs openpyxl.
---

# Data analysis

1. Check the library: `python3 -c "import pandas"`. If it fails, run `pip install pandas` (and `openpyxl` for Excel files).
2. Look before you analyze. Print the shape, column names and types, the first rows, missing values per column, and duplicate rows. State anything surprising.
3. Clean deliberately and say what you changed: parse dates, fix numeric columns stored as text, drop or flag duplicates, and decide how to treat missing values.
4. Answer the question with the simplest correct method: group-bys, pivots, and descriptive statistics before models. Keep units and time periods explicit.
5. Check results: totals match the source, row counts survive joins as expected, and percentages add up.
6. Save outputs as files the user can open: cleaned data as CSV or Excel (use the xlsx skill for formatted workbooks), figures with the charts skill, and a short Markdown summary.
7. Keep the analysis script in the working directory so it can be rerun on new data.
8. In your answer, lead with the findings in plain language, then list the files.
