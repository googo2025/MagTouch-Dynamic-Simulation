# MagTouch simulation-assisted magnetic tactile reconstruction

An interactive empirical magnetic-response simulator and the current research materials are provided here. The [live simulator](https://googo2025.github.io/MagTouch-Dynamic-Simulation/) runs in the browser.

## Current manuscript and formal supplement

Use the **[formal revision v2](supplements/micromachines_revision_20261024_formal_v2/README.md)** and its **[versioned data/model/code release](https://github.com/googo2025/MagTouch-Dynamic-Simulation/releases/tag/micromachines-revision-20261024-formal-v2)**.

- [English manuscript](supplements/micromachines_revision_20261024_formal_v2/paper/Manuscript_Revised_EN.md) · [完整中文修订稿](supplements/micromachines_revision_20261024_formal_v2/paper/Manuscript_Revised_ZH.md)
- [Formal GRU results](supplements/micromachines_revision_20261024_formal_v2/results/GRU_table.md) · [Eight-sequence validation](supplements/micromachines_revision_20261024_formal_v2/results/Same_condition_Table9.md)
- [Reproduction instructions](supplements/micromachines_revision_20261024_formal_v2/README.md#download-and-reproduce) · [Data availability](DATA_AVAILABILITY.md)

An empirical dynamic surrogate and virtual exploration informed a multiscale causal-history design. The final model was refined on real source data. In the two original source sessions, command-referenced planar error fell from 0.436 to 0.408 mm and 0.452 to 0.424 mm, with about 45% lower residual-step RMS. Matched continued H0 training and a representative residual GRU did not reproduce the large multiscale stability gain. Eight further same-condition frozen comparisons retained all eight candidates and reduced raw XY from 0.456 to 0.431 mm, step RMS from 0.181 to 0.097 mm and jump fraction from 25.76% to 3.38%.

These are bounded command-referenced results. Original retrospective design exposure, common initialization of training-order repetitions, separate assemblies and unmeasured physical metrology remain explicit. No arbitrary-contact generalization or physical-delay claim is made.

## Contents and use

- `supplements/micromachines_revision_20261024_formal_v2/`: current bilingual manuscripts, figures, formal outputs, model metadata, protocols and portable code.
- The [v2 release](https://github.com/googo2025/MagTouch-Dynamic-Simulation/releases/tag/micromachines-revision-20261024-formal-v2): the self-contained new saved-output reproduction capsule; the [earlier raw-data release](https://github.com/googo2025/MagTouch-Dynamic-Simulation/releases/tag/micromachines-revision-20261024-reproducibility) retains original formal raw streams and prior table code.
- `web/magnetic-forward-r361/`: existing bilingual browser simulator.
- `paper/`, `supplements/micromachines_revision_20261024/` and inherited research snapshots: historical versions. Their table numbers and interpretations belong to those versions; use v2 above for the current manuscript.

The statistical entry points execute offline. GitHub Pages does not acquire data, train models or control hardware. Existing licenses remain applicable.
