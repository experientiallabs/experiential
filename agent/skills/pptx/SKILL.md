---
name: pptx
description: Create PowerPoint presentations (.pptx) with python-pptx. Use when the user asks for slides, a deck, or a presentation.
compatibility: Requires python3 with python-pptx (pip install python-pptx).
---

# PowerPoint presentations

1. Check the library: `python3 -c "import pptx"`. If it fails, run `pip install python-pptx`.
2. Plan the deck first: a title slide, one idea per slide, a closing slide with the takeaway or next steps. Put the plan in your task list.
3. Write a Python script in the working directory and run it. Keep the script.
4. Slides:
   - Use the built-in layouts: `prs.slide_layouts[0]` for the title, `[1]` for title and content, `[5]` for title only (charts and images), `[6]` blank.
   - At most five bullets per slide, each under about ten words. Put detail in the speaker notes (`slide.notes_slide.notes_text_frame.text`).
   - Charts as PNG images made with the charts skill, added with `slide.shapes.add_picture(path, left, top, width=...)`, or native charts with `slide.shapes.add_chart` when the user wants to edit the data.
   - Use 16:9 (`prs.slide_width = Inches(13.333)`, `prs.slide_height = Inches(7.5)`) unless asked otherwise. Keep fonts and positions consistent across slides.
5. Verify: reopen the file and print each slide's title and bullet count.
6. In your answer, give the file name and the slide titles.
