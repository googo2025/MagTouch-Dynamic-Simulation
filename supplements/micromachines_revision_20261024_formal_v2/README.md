# Formal Micromachines revision v2 — 24 October 2026

This is the current manuscript and formal supplement. An empirical simulator informed a causal-history design that was refined using real data and evaluated against command-referenced trajectories. Source-condition joint error and stability gains are supported by matched continued-training controls, one representative GRU control and eight additional same-condition frozen comparisons.

## Manuscripts and result scope

- [English full manuscript](paper/Manuscript_Revised_EN.md) and [完整中文论文](paper/Manuscript_Revised_ZH.md), with 18 figures, 17 tables and 50 references each.
- [Formal GRU comparison](results/GRU_table.md): all three order repetitions, with source, source-repeat and target assembly separated.
- [Eight-sequence Table 9](results/Same_condition_Table9.md): all eight fixed candidates retained; raw XY 0.456→0.431 mm, step RMS 0.181→0.097 mm and jump fraction 25.76%→3.38%, at approximately 99.76% mean coverage.
- [Full-precision per-record and common-output statistics](results/scientific_results.json), [record/condition protocol](manifests/study.json), selected models and full histories under `models/`.

The source-session-1 three-order GRU means were raw XY 0.444 mm and jump fraction 22.12%, versus 0.407 mm and 4.08% for the multiscale joint configuration. Same-data continued H0 training yielded 0.458 mm and 22.44%. These results support the specified design under the tested protocol, not universal superiority over recurrent architectures. One DEV-selected GRU checkpoint remains at the baseline-equivalent initial function; all three selected models are retained.

## Download and reproduce

Download the [versioned v2 data/model/code addendum](https://github.com/googo2025/MagTouch-Dynamic-Simulation/releases/tag/micromachines-revision-20261024-formal-v2) and compare its hash with [public_release_manifest.json](public_release_manifest.json). The archive contains the formal frozen comparison outputs, the eight original magnetic/command records and complete native operation prefix, model arrays, selected training histories, fixed roles, exact statistical functions and code. The original formal raw sensor streams and refinement data remain in the [earlier reproduction release](https://github.com/googo2025/MagTouch-Dynamic-Simulation/releases/tag/micromachines-revision-20261024-reproducibility).

From the extracted v2 capsule root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-statistics.txt
.venv/bin/python tools/reproduce_followup_tables.py --root . --output ../followup-check-new
```

The v2 saved-output route is self-contained and passed all per-record, own-coverage, common-valid-output and order-summary fields at absolute tolerance 1e-12. See [the completed reproduction certificate](verification/saved_output_reproduction.json). It uses Python and NumPy and does not train models, run inference or operate hardware. Exact original host-bound training snapshots are under `execution_source/`; they are provenance rather than the portable entry point. Current Table 9 and Tables S3–S4 are distinguished from numbering in earlier manuscript versions.

## Interpretation and provenance

The original evaluation is retrospective. The GRU architecture was added after earlier summaries were viewed; its three repetitions vary training order from one initialization. The eight additional records are outside documented fitting/development/original-evaluation roles and passed original-online baseline parity, but are not certified wholly unseen across earlier research. They were evaluated only with the original H0/H2, not with GRU. Commands/EST timing are not independent physical position or synchronization truth; centering is a diagnostic. Dedicated physical metrology, directional holds and physical-delay measurements were not performed.

Publication metadata screens machine/account/network identifiers with original/public hash provenance. Scientific arrays, observations and timestamps are retained. Reviewer correspondence and private author handoff notes are excluded. Existing repository licenses remain applicable. Historical versions remain available; the simulator and workflows are unchanged.
