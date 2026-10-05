# MagTouch dynamic simulation and virtual-data-guided reconstruction

This repository hosts the paper's interactive empirical magnetic-response
simulator and a public, privacy-screened research snapshot. The [live
simulator](https://googo2025.github.io/MagTouch-Dynamic-Simulation/) runs in the
browser. It shows commanded three-dimensional motion, history-dependent
15-channel magnetic response, saved reconstruction examples, and a virtual
laboratory. It does not control hardware.
The R461 simulator interface opens in English and can be switched to Chinese
with the `English / 中文` controls. The older Chinese-only R430 entry remains
available as `web/magnetic-forward-r361/index_r430_legacy.html`.

## Main finding and limits

Simulation suggested an additional causal multiscale-history representation.
After source-side real-data training, the selected research model reduced
original XY error by 6.36% and 6.29% on two separate source-device batches of
new complex trajectories, while reducing the defined residual jump rate by
86.52% and 86.51% and stationary XY variation by 36.80% and 40.63%.
The target-device result primarily supports stability transfer, not a
universal accuracy improvement. See both full manuscripts for every model,
device, accuracy metric, coverage, failure, and negative control. No new XYZ
output smoother was added; the causal input EMA is disclosed.

These are retrospective frozen-model evaluations of previously recorded real
streams. Position references come from command/estimated arrival, not optical
contact truth. The tested sample is two devices, six non-random trajectory
families, and a single optimization seed.

## Contents

- `web/magnetic-forward-r361/`: complete static website for GitHub Pages.
- `paper/`: final Chinese and English manuscripts, PDFs, figures, provenance
  index, and validation record.
- `code/`, `training/`, `tools/`, `config/`: research implementation and
  frozen evaluation helpers.
- `analysis/r372_direct_source_plateau_20261001/`: selected baseline weights.
- `analysis/r460_source_real_history_refinement_20261004/`: source-side
  prepared arrays, frozen research weights, training traces, protocol, and
  aggregate evidence.
- `DATA_AVAILABILITY.md`: exact public/private boundary and interpretation.

To serve the website locally:

```bash
python -m http.server 8767 --directory web/magnetic-forward-r361
```

Open <http://127.0.0.1:8767/>. GitHub Pages serves this same directory.
Browser simulation and saved results work without a Python backend.
Training, original-stream replay, and FEM are separate offline workflows.
The public snapshot is **not** a full raw-Trial replication kit.

Software is MIT-licensed. The paper and figures are CC BY 4.0; see
`LICENSE-CONTENT.md`. No license is granted here for unreleased raw records.

The existing physical-platform photographs are in the
[related hardware repository](https://github.com/googo2025/MagTouch-RA-3d-trajectory-recon/tree/main/media).
