# Data availability and provenance

This public snapshot includes the interactive simulator, frozen paper PDFs and
manuscript sources, figure assets, algorithm source, model checkpoints,
source-side prepared training arrays, protocol, training traces, and the
aggregate 144-condition result table. The virtual laboratory includes saved
synthetic demonstrations. See `paper/source_index.json` for provenance.

Original hardware Trial streams and their unredacted acquisition metadata are
**not included** in this public snapshot. The local standalone archive remains
preserved with its original hashes. The omission prevents accidental release
of machine paths and session metadata; it also means a third party cannot
independently repeat every original hardware replay from this repository
alone. Do not describe this public subset as a full raw-data replication kit.

The paper compares a fully trained baseline, a multiscale causal-history
readout trained on source-side real data, and an objective-augmented readout.
The three new complex-trajectory batches were retrospectively evaluated after
model freezing. They are not a prospective blind clinical or physical-contact
measurement. The command/estimated-arrival reference is not optical ground
truth. All reported models, devices, and negative findings remain in the
manuscripts and aggregate table; no favorable-trial-only result was substituted
for the full evaluation.

The displayed simulator is an empirical forward model and browser visualization,
not a validated arbitrary-tip FEM digital twin. GitHub Pages does not train
networks, run the full Python inverse replay, acquire hardware, or solve FEM.
