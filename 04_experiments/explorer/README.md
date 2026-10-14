# Plot Evidence Explorer (D3)

Interactive view of the study results: open `index.html` in a browser (no server needed).

- **Evidential map**: failure × plot; colour = detection AUC or diagnosis recall; ● identifiable · ◐ detect-only · ○ neither;
  severity slider; click a column to select a plot.
- **Benchmark confusion** for the selected plot (B1* = best of mark-table / image model).
- **AI readers**: each reader's confusion on the same plot (a single bright column = same answer regardless of the plot).
- **Balanced accuracy by plot**: benchmark vs each reader, with the chance line.

Data files are generated from a run folder:
`python 04_experiments/code/export_explorer.py 04_experiments/code/runs/multi_v3 v3` → `data_v3.js`
(v4: `... runs/v4_synthetic v4syn`, `... runs/v4_magic v4magic`; the page picks them up automatically).
