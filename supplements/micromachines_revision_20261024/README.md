# Micromachines revision supplement — 24 October 2026

The full revised manuscript and the versioned raw-data/code release are linked here. Historical simulator files and evidence are retained.

## Main result

An empirical simulator and real-record checks informed a multiscale causal-history expansion trained on real data. The original source conditions support joint command-referenced error and stability gains. Matched continued training did not reproduce the large stability benefit. In nine archived acquisitions, stability improved but raw XY error increased. This regression, the retrospective original evaluation, common-initialization order seeds and command-derived references remain explicit.

## Raw inputs and executable reproduction

Download **[the versioned raw-data and code capsule](https://github.com/googo2025/MagTouch-Dynamic-Simulation/releases/tag/micromachines-revision-20261024-reproducibility)** and verify its SHA-256 against [public_release_manifest.json](public_release_manifest.json). It contains 35 TRAIN/DEV records, 57 evaluation records, raw sensor journals and CSVs, command timestamps, 92 record assignments, frozen outputs, model artifacts and versioned statistical sources. Machine identifiers are redacted with hash provenance; numerical observations, timestamps and arrays are preserved.

From the extracted capsule root (Linux/Python 3.11+):

```sh
python3 -m venv .venv-statistics
.venv-statistics/bin/python -m pip install -r requirements-statistics.txt
.venv-statistics/bin/python tools/reproduce_saved_outputs.py --root . --output ../verification-new --phase all
.venv-statistics/bin/python tools/reproduce_principal_tables.py --root . --statistics ../verification-new --output ../principal-tables-new
```

The public copy passed in a fresh virtual environment: the original 504 per-sequence rows and 5,976 numerical cells, scientific fields of both new reports, source/target Table 6 rows and Table 7 bootstrap/sign-flip statistics, and the printed values of Tables 8, 9, S1 and S2. See [the completed verification](verification/public_capsule_verification.json). These are principal trajectory-statistics reproductions rather than new training or hardware experiments.

## Files in this repository directory

- [English manuscript](paper/Manuscript_Revised_EN.md) and [Chinese manuscript](paper/Manuscript_Revised_ZH.md), with figure assets.
- `results/`: original, matched and archived summaries/per-sequence/paired results and training traces, including negative outcomes.
- `models/`: all 18 added selected models and hashes.
- `protocols/` and [record assignments](manifests/record_assignments.csv): protocol summaries, source versions and whole-record roles.
- `tools/` and `expected_tables/`: portable reduction/table entry points and printed-table references. Run them against the downloaded capsule, which contains the required numerical inputs.
- `FILE_MANIFEST.json`: checksums for this repository supplement.

Original native streams retain the input actually received by the online decoder and its target queue-loss limitation. Archived reconstruction lacks the old complete inference journal and uses its disclosed separate policy. Command-derived coordinates are not independent physical ground truth. Source/software and manuscript/figure licenses remain those of the existing repository; no new dataset-wide reuse license is asserted.
