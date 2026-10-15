# Plot Evidence Explorer

Interactive D3 view of what each ML diagnostic plot can reveal. Open `index.html` in a browser (no server needed).

**Views**
1. **Evidential map**: failure × plot matrix, detection or diagnosis, by severity and data source; hover for the
   benchmark AUC with its 95% CI and the cell label (identifiable / detect-only / blind).
2. **Cell view**: click a cell: the cell's statistics at every severity, example plots of held-out healthy and failing
   models (re-trained from their seeds, identical to the study run), and what each AI reader answers on that plot.
3. **My plots**: tick the plots you already look at; see which failures they can identify, only flag, or miss, and the
   single plot that would add the most evidence.
4. **AI readers**: each reader's confusion on the selected plot next to the benchmark, and balanced accuracy by plot.

**Data**: `data_<run>.js` from `code/export_explorer.py`; example plots from `code/make_examples.py`
(`examples/<run>/`, `examples_<run>.js`). Default run: the confirmatory synthetic study; the real-data run and the
pilots are selectable.
