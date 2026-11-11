---
name: charts
description: Make clear, publication-quality charts as PNG files with matplotlib. Use when the user asks for a chart, graph, plot, or visualization, or when a document or deck needs a figure.
compatibility: Requires python3 with matplotlib (pip install matplotlib).
---

# Charts

1. Check the library: `python3 -c "import matplotlib"`. If it fails, run `pip install matplotlib`. Use the non-interactive backend: `import matplotlib; matplotlib.use("Agg")`.
2. Pick the form from the question: a line for change over time, bars for comparing categories, a scatter for a relationship between two measures, a histogram for a distribution. Avoid pie charts beyond three slices and avoid 3D.
3. Every chart has a title, labeled axes with units, and a legend only when there is more than one series.
4. Keep it clean: remove the top and right spines, use a light grid on the value axis only, start bar axes at zero, and format large numbers with thousands separators.
5. Save with `fig.savefig(path, dpi=150, bbox_inches="tight")` and close the figure.
6. Look at what you made: check the file exists and its size is reasonable, and if you can view images, open it with the read tool and fix anything unreadable.
7. In your answer, give the file name and one sentence on what the chart shows.
