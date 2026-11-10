---
name: pdf
description: Create PDF documents with reportlab, or read text from existing PDFs. Use when the user asks for a PDF report, invoice, certificate, or printable document, or needs the contents of a PDF.
compatibility: Requires python3 with reportlab (pip install reportlab); reading uses pypdf (pip install pypdf).
---

# PDF documents

## Create

1. Check the library: `python3 -c "import reportlab"`. If it fails, run `pip install reportlab`.
2. For flowing documents use Platypus: `SimpleDocTemplate`, `Paragraph` with `getSampleStyleSheet()` styles (`Title`, `Heading1`, `BodyText`), `Spacer`, `Table` with a `TableStyle`, and `Image` for charts.
3. Anything longer than one page gets a title and page numbers (draw the number in an `onPage` callback with `canvas.drawRightString`).
4. Use A4 or Letter as the user's locale suggests; keep margins of at least 2 cm.
5. Write a script in the working directory, run it, and keep it.

## Read

1. `pip install pypdf` if needed, then `PdfReader(path)` and `page.extract_text()` for each page.
2. Scanned PDFs have no text layer; say so rather than guessing the contents.

## Verify

Reopen the PDF with pypdf and print the page count and the first lines of text, then give the file name in your answer.
