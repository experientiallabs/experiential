---
name: docx
description: Create or edit Word documents (.docx) with python-docx. Use when the user asks for a Word document, a .docx report, memo, letter, or any formatted document they will open in Word.
compatibility: Requires python3 with python-docx (pip install python-docx).
---

# Word documents

1. Check the library: `python3 -c "import docx"`. If it fails, run `pip install python-docx`.
2. Write a Python script in the working directory (for example `make_report.py`) and run it. Keep the script so the document can be regenerated.
3. Structure:
   - `doc.add_heading(text, level=0)` for the title, levels 1 and 2 for sections. Use real headings, never bold paragraphs, so the navigation pane and table of contents work.
   - Short paragraphs. Bulleted lists with `style="List Bullet"`, numbered lists with `style="List Number"`.
   - Figures in tables: `doc.add_table(rows, cols)`, `table.style = "Light Grid Accent 1"`, header row first, numbers right-aligned.
   - Images with `doc.add_picture(path, width=Inches(6))`. Make charts with the charts skill first.
4. To edit an existing document, open it with `Document(path)`, change paragraphs and runs in place, and save under a new name unless the user asked to overwrite.
5. Verify: reopen the file and print the headings and paragraph count, for example `[p.text for p in Document(path).paragraphs if p.style.name.startswith("Heading")]`.
6. In your answer, give the file name and a two-line summary of the contents.
