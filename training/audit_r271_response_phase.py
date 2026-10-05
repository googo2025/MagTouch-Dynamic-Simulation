"""Whole-action timing and figures for the frozen R271 shadow observer.

Commands group diagnostic episodes only; they never enter the observer. This
audit does not fit a model, modify XYZ, or identify physical shallow contact.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from training.probe_r269_hierarchical_response_state import ROOT, load_rows, prepare_hierarchy
from training.probe_r228_shallow_inverse import reference

INPUT = ROOT / "analysis/r271_rearm_phase_continuity_20260929"
OUT = ROOT / "analysis/r271_response_phase_audit_20260929"


def blocks(mask):
    ids = np.flatnonzero(mask)
    return [b for b in np.split(ids, np.flatnonzero(np.diff(ids) > 1) + 1) if len(b)]


def timing(row, states):
    if row["mode"] == "none":
        return []
    q = reference(row)
    finite = np.isfinite(q).all(1)
    dz = np.r_[0., np.diff(q[:, 2])]
    continuous = np.r_[False, (np.diff(row["time"]) > 0) & (np.diff(row["time"]) <= 250_000_000)]
    result = []
    for sign, desired, name in ((1, 2, "loading"), (-1, 4, "withdrawal")):
        mask = row["eligible"] & finite & continuous & (sign * dz > 1e-4) & (q[:, 2] >= .02)
        for block in blocks(mask):
            responsive = block[row["response"][block]]
            exact = block[states[block] == desired]
            weak = block[(q[block, 2] < .4) & row["response"][block]]
            result.append({"phase": name, "start_ns": int(row["time"][block[0]]),
                           "end_ns": int(row["time"][block[-1]]), "frames": len(block),
                           "response_frames": len(responsive), "weak_reference_response_frames": len(weak),
                           "matched_phase_frames": len(exact),
                           "matched_in_weak_reference_frames": int(np.sum(states[weak] == desired)),
                           "first_phase_ns": None if not len(exact) else int(row["time"][exact[0]]),
                           "first_response_ns": None if not len(responsive) else int(row["time"][responsive[0]]),
                           "delay_from_first_response_s": None if not len(responsive) or not len(exact)
                           else float((row["time"][exact[0]] - row["time"][responsive[0]]) / 1e9),
                           "z_range_diagnostic_only": [float(np.min(q[block, 2])), float(np.max(q[block, 2]))]})
    return result


def plot_trace(row, arrays, target):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    time = (row["time"] - row["time"][0]) / 1e9
    q = reference(row)
    colors = ["#B7BEC8", "#EDF1F5", "#DC8D29", "#63AB91", "#417FBB", "#8B71AA", "#0C6678"]
    fig, axes = plt.subplots(4, 1, figsize=(11.5, 8.2), sharex=True, layout="constrained",
                            gridspec_kw={"height_ratios": [1., 1., 1.2, .48]})
    axes[0].plot(time, q[:, 2], color="#8794A2", lw=1.2)
    axes[0].axhspan(.02, .2, color="#DC8D29", alpha=.15)
    axes[0].set_ylabel("Command Z (mm)\ndiagnostic reference")
    axes[1].plot(time, row["amplitude_uT"], color="#0C6678", lw=1.)
    no_response = row["eligible"] & ~row["response"]
    axes[1].scatter(time[no_response][::10], row["amplitude_uT"][no_response][::10], s=3., color="#AAB1BC")
    axes[1].set_ylabel("Baseline-subtracted\nfield norm (uT)")
    axes[2].plot(time, arrays["context_scores"][:, 1], color="#DC8D29", lw=1., label="weak-response context")
    axes[2].plot(time, arrays["context_scores"][:, 3], color="#8B71AA", lw=1., label="residue context")
    axes[2].plot(time, arrays["direction_scores"][:, 0], color="#B46513", ls="--", lw=.85, label="loading head")
    axes[2].plot(time, arrays["direction_scores"][:, 1], color="#417FBB", ls="--", lw=.85, label="withdrawal head")
    axes[2].set_ylabel("Head score\n(not contact probability)")
    axes[2].set_ylim(-.05, 1.05)
    axes[2].legend(ncol=2, fontsize=8, loc="upper right")
    axes[3].imshow(arrays["state"][None, :], aspect="auto", interpolation="nearest",
                   extent=[time[0], time[-1], 0., 1.], vmin=-.5, vmax=6.5, cmap=ListedColormap(colors))
    axes[3].set_yticks([])
    axes[3].set_ylabel("Causal state")
    axes[3].set_xlabel("Elapsed time (s)")
    for ax in axes[:3]:
        ax.grid(alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    labels = ("unknown", "quiet", "loading-like", "weak hold", "withdrawal-like", "residue-like", "supported")
    from matplotlib.patches import Patch
    fig.legend([Patch(facecolor=c) for c in colors], labels, ncol=7, fontsize=8, loc="outside lower center")
    fig.suptitle(f"{row['device'].capitalize()} device | {row['path'].name.split('_')[0]} {row['path'].name.split('_')[1]}\n"
                 "Frozen causal shadow state: no XYZ modification, no command input", fontsize=12)
    fig.savefig(target.with_suffix(".png"), dpi=160)
    fig.savefig(target.with_suffix(".svg"))
    plt.close(fig)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    rows = load_rows(); prepare_hierarchy(rows)
    OUT.mkdir(parents=True)
    trials, pooled = [], {}
    plotted = set()
    for row in rows:
        if row["partition"] != "evaluation":
            continue
        trial = int(row["path"].name.split("_")[1])
        with np.load(INPUT / f"trial_{trial:06d}_state.npz", allow_pickle=False) as a:
            arrays = {k: a[k].copy() for k in a.files}
        np.testing.assert_array_equal(row["time"], arrays["time_ns"])
        events = timing(row, arrays["state"])
        trials.append({"trial": row["path"].name, "device": row["device"], "mode": row["mode"], "events": events})
        for event in events:
            key = row["device"] + "_" + event["phase"]
            pooled.setdefault(key, []).append(event)
        if row["mode"] == "z_path" and row["device"] not in plotted:
            plot_trace(row, arrays, OUT / f"{row['device']}_z_path_state")
            plotted.add(row["device"])
    summary = {}
    for key, events in pooled.items():
        responsive = [e for e in events if e["response_frames"]]
        found = [e for e in responsive if e["first_phase_ns"] is not None]
        delays = [e["delay_from_first_response_s"] for e in found]
        summary[key] = {"episodes": len(events), "with_response": len(responsive), "phase_found": len(found),
                        "median_delay_from_first_response_s": None if not delays else float(np.median(delays)),
                        "range_delay_s": None if not delays else [float(np.min(delays)), float(np.max(delays))],
                        "note": "Command-grouped episodes, not contact recall. Supported-state frames carry no direction in the seven-state encoding."}
    report = {"schema": "R271-WholeActionTiming-1", "frozen_input": str(INPUT.relative_to(ROOT)),
              "summary": summary, "trials": trials,
              "limitation": "R208 retrospective audit, not blind. Neither nominal shallow XYZ nor physical contact is the target. No XYZ is altered; no translation adjustment is performed or required for these magnetic-state scores."}
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
