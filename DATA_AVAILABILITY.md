# Data availability and provenance

## Current revision

The [24 October 2026 revision supplement](supplements/micromachines_revision_20261024/README.md) contains full English and Chinese revised manuscripts and figure assets; the original submitted per-sequence metric table; full-precision matched and archived summaries, per-sequence and paired results; all 18 new selected matched model artifacts; training/development traces; public protocol summaries; source-version hashes; a statistical entry point; and its completed numerical verification record.

The older `paper/`, algorithm sources, prepared arrays, checkpoints and interactive simulator remain historical provenance. The current supplement updates the scientific interpretation without replacing those records or changing the simulator's saved examples. Its file manifest records the current supplement bytes; the historical repository baseline is commit `0048460dd73e104101eea1696bd5c24737c9bd44`.

## Raw records and complete reproduction archive

Original hardware magnetic streams, command timestamps and unredacted acquisition metadata are **not included** in this public repository. A separate complete author-held reproduction archive contains the relevant inputs and frozen outputs for 57 evaluation sequences (48 original records including controls and nine archived-extension records), record-level split/role manifests, refinement inputs, exact statistical sources and model artifacts. It is retained on the original research server for review and distribution by the corresponding author. This public subset must not be described as a complete raw-data reproduction kit or as an already deposited raw-data release.

The public `tools/reproduce_saved_outputs.py` inside the revision supplement is an entry point for that **complete archive**, not a standalone script that can infer missing raw inputs. Use the complete archive's README and file manifest. Verification in a separate package directory reproduced all 504 original submitted per-sequence rows and 5,976 numerical cells at reported precision, and all scientific fields of the matched and effective archived reports at the declared numerical tolerance. This verifies saved-output statistical reduction; it does not represent a new full inference, training or hardware run. Historical absolute paths and version-correction evidence remain in the private provenance material.

## Interpretation

The original complex evaluation records were excluded from final fitting and checkpoint selection but informed earlier representation research, so their evaluation remains retrospective. The nine archived inputs and all model artifacts were fixed before the new scoring, and were not used for this study's fitting or checkpoint selection; their complete researcher-exposure history is not certified blind. Their missing old inference-operation journals require the separately documented causal reconstruction protocol, without a claim of exact historical online-state parity. The three added seeds vary training order from one common trained initialization.

Command-derived coordinates and estimated timing are not independent physical contact-position ground truth. Constant-translation removal is a reference-fitted diagnostic rather than deployed calibration. The archived raw-error regression is preserved together with stability improvement. No independent position/synchronization metrology, new consistently mounted hardware acquisition, dedicated directional-hold test, measured response delay or recurrent-model comparison is implied by this release.

The simulator is an empirical response model and browser visualization, not a validated arbitrary-tip physical digital twin. GitHub Pages does not train models, run full inverse replay, collect measurements or solve FEM. Existing software/content licenses continue to apply; no license is newly assigned to unreleased raw records.
