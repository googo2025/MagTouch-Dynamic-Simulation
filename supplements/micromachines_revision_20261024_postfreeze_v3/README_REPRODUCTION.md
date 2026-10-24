# Post-freeze acquisition addendum v3

This supplement contains all12 H0 candidate outputs and six H2 comparisons for a separately documented hemispherical indenter on the source magnetic specimen. Acquisition on5October followed the original final-model freeze on4October2026. No original H0/H2 parameters were fitted or selected with these measurements, and no indenter identity is input to either model. Evaluation definitions were fixed during revision; this is not a claim of study-wide prospective blind registration.

The fixed H0 operating-condition rule is raw command-reference XY mean<=0.5301053910924818mm and coverage>=95%. All12 H0 results are retained, including candidates not entering H2. The six conditional comparisons cover nominal depths0.8/1.2mm, speeds1/2mm/s and eight-direction same-command-point holds. All76 arrivals and the unresolved reversal event are retained. Within-hold fluctuation decreased in all four combinations, while between-direction mean-position spread decreased in only one. The paired response-proxy median is0ms; separate group medians must not be subtracted and described as a19ms paired gain.

## Reproduce the principal supplementary statistics

Use Python3.11+ and the pinned NumPy dependency. In a newly extracted copy:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-statistics.txt
.venv/bin/python tools/reproduce_postfreeze.py --root . --output ../postfreeze-check-new
```

This verifies every listed file hash, rebuilds command references and scoring masks from the raw command/metadata files, reproduces all12 H0 and six H2 own-output statistics, their common-valid-output comparisons, the equal-sequence mean/SD, all directional-hold rows and the command-response event statistics. Complete scientific fields must match the frozen expected result at absolute tolerance1e-12. It makes zero model-inference or training calls and requires no private directory, network access or hardware. A successful fresh-copy run is recorded in verification/reproduction_certificate.json.

Raw magnetic streams, actual command timestamps, events, metadata, causal input operations, frozen model heads and versioned native source snapshots are supplied as provenance. The portable entry is tools/reproduce_postfreeze.py; execution_source snapshots retain original runtime interfaces and are not the portable statistics entry. Their unchanged full runtime, original training data and earlier principal-table route remain in the original release: https://github.com/googo2025/MagTouch-Dynamic-Simulation/releases/tag/micromachines-revision-20261024-reproducibility . The formal-v2 release supplies the preceding eight-sequence online-parity study and representative GRU comparison: https://github.com/googo2025/MagTouch-Dynamic-Simulation/releases/tag/micromachines-revision-20261024-formal-v2 . No older tag or archive is replaced.

The raw trajectories and response measures use recorded commands and EST timing. They are not independent physical-position or synchronization truth. Frozen causal replay evaluates the same native algorithm under the recorded input/control protocol; direct historical H0 prediction parity is claimed only for the separate eight-sequence v2 study. The current post-freeze model comparison uses its recorded unloaded-baseline events and lifecycle controls. Centering is diagnostic and does not change outputs. One new indenter and a baseline-conditioned cohort do not prove arbitrary-indenter generalization or removal of all directional mean bias.

Metadata is screened for machine/account/network identifiers with explicit original/public hashes; numeric observations, command times and model/output arrays are retained. Existing project licenses remain applicable; no new data-wide license is invented. Original reviewer correspondence, emails, credentials and private author/handoff documents are excluded.
