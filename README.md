# MagTouch simulation-assisted magnetic tactile reconstruction

This repository provides an interactive empirical magnetic-response simulator and a privacy-screened research snapshot. The [live simulator](https://googo2025.github.io/MagTouch-Dynamic-Simulation/) runs in the browser and does not control hardware.

## Current revised manuscript and supplement

The **24 October 2026 revision** is the current manuscript. It retains the submitted study and adds matched continued-training controls and a fixed-protocol archived extension:

- [Complete revised English manuscript](supplements/micromachines_revision_20261024/paper/Manuscript_Revised_EN.md) · [完整中文修订稿](supplements/micromachines_revision_20261024/paper/Manuscript_Revised_ZH.md)
- [Revision supplement and reproduction scope](supplements/micromachines_revision_20261024/README.md)
- [Full-precision summaries](supplements/micromachines_revision_20261024/results/summary_tables.csv), [paired comparisons](supplements/micromachines_revision_20261024/results/paired_comparisons.csv), [archived per-sequence results](supplements/micromachines_revision_20261024/results/archived_per_record.csv)
- [Selected matched models](supplements/micromachines_revision_20261024/models/model_manifest.json), [training traces](supplements/micromachines_revision_20261024/results/matched_training_traces.csv), and [numerical verification](supplements/micromachines_revision_20261024/verification/saved_output_verification.json)

The earlier files under `paper/` and the browser's saved paper/result pages are retained as historical material. Their results and PDFs predate the added controls; use the revision linked above for the current scientific interpretation. New Markdown manuscripts include their figure assets. No revised PDF is implied.

## Main finding and limits

An empirical dynamic surrogate, controlled virtual exploration and earlier real-record checks motivated a multiscale causal-history expansion of an existing history-aware reconstruction chain. The final refinements were trained on real source measurements; simulation supplied design guidance rather than the final virtual supervision or pretrained weights.

In the original two source sessions, the frozen joint model reduced mean **command-referenced** planar error from 0.436 to 0.408 mm and from 0.452 to 0.424 mm, with approximately 45% lower residual-step RMS. Eighteen added matched runs show that the large stability benefit was not reproduced merely by adding the same dwell data and continuing baseline optimization. Three training-order seeds share one initialization; they are not independent random initializations.

In nine archived acquisitions, stability improved but **raw XY error increased**: the original frozen joint model changed 1.255 to 1.293 mm while residual-step RMS decreased 0.223 to 0.176 mm. The revision retains this regression. Joint error and stability gains are supported in the original source conditions; stability is the more consistent benefit across the tested extensions.

References are reconstructed from stage commands and estimated timing, not independent physical contact-position metrology. The original evaluation is retrospective. The archived extension was fixed before its new scoring and excluded from this study's fitting and checkpoint selection, but it is not certified blind and lacks the old native inference-operation journal. Per-sequence translation removal is a reference-fitted diagnostic, not deployed calibration. No new hardware experiment, independent positioning/synchronization test, dedicated directional-hold or physical-latency experiment, or matched recurrent-network baseline is claimed.

## Contents and use

- `supplements/micromachines_revision_20261024/`: current revision, figures, public results, models, protocols, and saved-output verification record.
- `web/magnetic-forward-r361/`: existing bilingual static simulator for GitHub Pages. The legacy Chinese entry remains available.
- `paper/`: historical manuscripts, PDFs, figures and provenance, with a link to the current revision.
- `code/`, `training/`, `tools/`, `config/`, and historical `analysis/` snapshots: inherited research implementation and evidence.
- [Data availability](DATA_AVAILABILITY.md): exact public/private boundary and verification scope.

```bash
python -m http.server 8767 --directory web/magnetic-forward-r361
```

Open <http://127.0.0.1:8767/>. Browser simulation and saved examples require no Python backend. Training, original-stream replay and FEM are separate offline workflows. This public repository omits original hardware streams and is not a complete raw-data replication kit. The separate complete archive has verified saved-output statistics; full training/inference is a distinct execution level.

Software is MIT-licensed. Manuscripts and figures are CC BY 4.0; see [LICENSE-CONTENT.md](LICENSE-CONTENT.md). No raw-data license is newly granted. Physical-platform photographs remain in the [related hardware repository](https://github.com/googo2025/MagTouch-RA-3d-trajectory-recon/tree/main/media).
