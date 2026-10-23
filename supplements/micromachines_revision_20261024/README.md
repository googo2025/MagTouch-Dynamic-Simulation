# Micromachines revision supplement — 24 October 2026

This published revision supplements the existing MagTouch-Dynamic-Simulation repository and preserves its simulator website and historical evidence. Use the full manuscripts in this folder for the current scientific interpretation.

## Main result and boundary

Simulation and earlier real-record checks motivated multiscale causal features, which were added to an existing history-aware chain and trained on real data. In the original source conditions the refinement improved command-referenced trajectory error and stability. Matched continued training with the same enlarged sample pool did not reproduce the large stability gain. In nine archived evaluation sequences, stability benefits persisted but raw XY error increased; the raw-error regression is retained alongside the smaller centered diagnostic. The work supports a conditional simulation-assisted design route, not universal accuracy gains, independently measured contact-position precision or the necessity of simulation.

## Contents

- `paper/Manuscript_Revised_EN.md`: complete revised manuscript, retaining the submitted paper's methods, original controls and references and adding the matched and archived studies.
- `paper/assets/`: unchanged-source original figures and new plots from frozen results.
- `results/`: full-precision summary, per-record, pair-common and training-trace CSVs, including negative outcomes. The three order seeds are not independent initializations.
- `models/`: the 18 added selected model artifacts and hashes; no model was selected using archived-extension performance.
- `protocols/`: public, path-screened descriptions of the training and extension protocols and source versions.
- `tools/reproduce_saved_outputs.py`: portable statistical entry point for the **complete reproduction archive**, not a claim that the raw inputs are included in this overlay.
- `verification/`: numerical verification receipt for the separately packaged complete payload.

## Reproduction level

The complete author-held archive contains the raw command/magnetic inputs and frozen outputs required by the verified route. Run its `tools/reproduce_saved_outputs.py --root <complete-archive> --output <fresh-directory> --phase all` using Linux/Python 3.11+ and `requirements-statistics.txt`. Verification covers the original 504 per-record rows (5,976 numerical cells at reported precision) and all scientific sections of the two new reports, without model inference or training. A fresh full training/inference execution and independent physical metrology are distinct and are not claimed here.

This overlay intentionally excludes raw hardware streams and unredacted acquisition metadata. Publication of this supplement does not represent the separate original records as public. The existing project software and paper/figure license notices continue to apply; no raw-data license is newly granted by this supplement.
