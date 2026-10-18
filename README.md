# Matrix Visualiser

**Open it: <https://bnsergiu.github.io/mn-visualiser/>**

Paste any Python / NumPy exercise and replay it line by line. For every step you see:

- the line being executed, with the current values filled in
  (`c[1, 2] = 0.31 + 0.62 * 0.41  →  0.56`)
- every matrix, vector, list and loop variable, drawn as a grid
- the cells the line **read** (blue) and **wrote or changed** (orange), with the row and column labels lit up
- how many times each line has run so far, and what was printed

Python runs in the browser through [Pyodide](https://pyodide.org) (CPython 3.13 + NumPy compiled to WebAssembly), so there is nothing to install.

## Running it locally

The hosted link above needs no setup. To run your own copy (e.g. while changing the code), note that the page loads `tracer.py` next to it, which browsers refuse to do from `file://`, so serve the folder:

```sh
cd mn-visualiser
python3 -m http.server 8000
```

Then open <http://localhost:8000>. The first load downloads Python and NumPy from the jsDelivr CDN (needs internet; the browser caches them afterwards).

## Using it

1. Paste the exercise code (or pick one from **Load an example…**).
2. If the code calls `input()`, put the answers in the box under the editor, one per line. If the box runs out, the browser asks.
3. Press **Visualise** (Ctrl/⌘ + Enter).
4. Step with the buttons, the slider, or the keyboard: ← → to step, Space to play/pause, Home / End. Clicking a line in the code jumps to the next time that line runs.

Notebook-only lines such as `!pip install ...` or `%timeit` are skipped.

Options: **Shade by value** colours cells by magnitude (green positive, pink negative); **Decimals** sets how numbers are rounded in the grids (hover a cell for its exact value); **Step limit** stops long programs (default 3000 steps; for loops this means keeping `n` small, e.g. 3 or 4).

## Limits

- 1-D and 2-D arrays and rectangular lists of numbers are drawn as grids. Bigger arrays (over 2500 cells) or 3-D ones are shown as text.
- Packages: NumPy is always loaded; other packages Pyodide ships (SciPy, etc.) load automatically when imported. Plots are not displayed.
- The "filled-in" line only re-evaluates simple expressions (names, indexing, arithmetic, NumPy functions). Anything with side effects is shown as written.

## Files

| File | What it does |
| --- | --- |
| `index.html`, `styles.css` | Page layout and styling (light and dark) |
| `app.js` | Loads Pyodide, runs the tracer, renders steps and handles playback |
| `tracer.py` | Runs the code under `sys.settrace` and records each step as JSON (`trace_json(code, stdin, max_steps)`) |
| `examples.js` | The snippets in the examples menu |
