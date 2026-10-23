# Data availability and reproducibility

## Public raw data and versioned code

The **[reproduction release](https://github.com/googo2025/MagTouch-Dynamic-Simulation/releases/tag/micromachines-revision-20261024-reproducibility)** supplies the raw magnetic/command inputs, native sensor journals, timestamped CSV exports, record-level assignments, frozen outputs, models and pinned statistical sources required for the principal trajectory tables. The capsule contains 35 refinement records (32 TRAIN/3 DEV) and 57 evaluation records (48 original, including controls, plus nine archived extension records). The [release manifest](supplements/micromachines_revision_20261024/public_release_manifest.json) specifies the archive, size and SHA-256; [92 record assignments](supplements/micromachines_revision_20261024/manifests/record_assignments.csv) are also browsable in the repository.

The [supplement](supplements/micromachines_revision_20261024/README.md) gives executable commands. In a fresh Python virtual environment the public copy reproduced the original 504-row metric table (5,976 numerical cells at published precision), both added reports' scientific fields, source/target Table 6 rows, Table 7 cluster-bootstrap intervals and exact sign-flip tests, and all printed values of Tables 8, 9, S1 and S2. The capsule and repository contain a completed verification receipt. No undisclosed private input folder is required for this verified route.

## Provenance and scope

Machine usernames, private network addresses and local account identifiers were redacted only in the publication copy. Numerical observations, timestamps and model/scientific arrays are preserved, and affected hash contracts have explicit original/public mappings. The original unredacted author-held archive remains immutable and is separate from the publicly downloadable scientific capsule. Reviewer correspondence, private conversations and connection credentials are not released.

The original native journals record the raw input groups actually received by the online decoder, with its original queue-loss behavior on the target session; flattened sensor CSVs expose ADC values, microtesla channels and trigger/read times without inventing missed groups. Archived data retain their saved raw magnetic and command CSVs and a separately prescribed reconstruction protocol because the old full inference journal is absent. Historical baseline record assignments are included and the baseline is supplied frozen; this release is not described as a new from-scratch reproduction of every prior baseline-training stage.

The original evaluation remains retrospective, the archived extension is not certified wholly blind, and three added order seeds share one initialization. Command-derived references do not establish physical position/synchronization truth. Raw-error regressions and output coverage remain reported. The release is sufficient for the documented principal trajectory tables, not a claim of a new hardware experiment or of completed directional-hold, independent-metrology or recurrent-model studies.

## Manuscripts and simulator

Current revised manuscripts, figures and result tables are under `supplements/micromachines_revision_20261024/`. Older files under `paper/` and the browser's saved paper/result pages are historical; the simulator and website are unchanged. GitHub Pages does not acquire data, train models or run the full offline statistical pipeline. Existing software/content licenses remain applicable; no new dataset-wide reuse license is asserted.
