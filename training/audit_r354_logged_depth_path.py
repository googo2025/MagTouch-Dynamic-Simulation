"""Compare the logged pre-depth Z and final Z on identical R354 online frames.

The pre-depth value is `original_z_mm` from the shared spatial inverse before
the final causal depth path. It is not a single-frame atlas and is not the
R343-only-increment ablation. Its action/XY permission and timestamps are
identical to the saved final output, so this measures the complete depth-path
contribution without re-running or altering the online experiment.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from training.analyze_r354_final_online import OUT, ROOT


def summary(z, q, use):
    if not use.any():
        return None
    err = np.abs(z[use] - q[use, 2])
    return {"frames": int(use.sum()), "z_mae_mm": float(err.mean()),
            "z_p95_mm": float(np.quantile(err, .95))}


def steps(z, time, q, action, valid):
    constant = action & valid & np.isclose(q[:, 2], 1.4, atol=.01)
    dt = np.diff(time) / 1e9
    pair = constant[:-1] & constant[1:] & (dt > 0) & (dt <= .25)
    dz = np.abs(np.diff(z)[pair])
    return {"pairs": int(pair.sum()),
            "p95_mm": float(np.quantile(dz, .95)) if len(dz) else None,
            "over_0p2_mm": int((dz > .2).sum())}


def main():
    report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    rows = []
    for device in ("source", "target"):
        for trial in report["plans"][device]["trials"]:
            if trial["mode"] == "none":
                continue
            number = trial["trial_id"]
            with np.load(OUT / "streams" / f"{device}_{number}.npz", allow_pickle=False) as f:
                a = {k: f[k].copy() for k in f.files}
            frame = pd.read_csv(ROOT / trial["path"] / "live_predictions.csv",
                                usecols=["extra_outputs_json", "t_prediction_ns"], low_memory=False)
            np.testing.assert_array_equal(frame.t_prediction_ns.to_numpy(np.int64), a["time"])
            extra = [json.loads(v) for v in frame.extra_outputs_json]
            original = np.array([float(e.get("original_z_mm", np.nan)) for e in extra])
            valid = a["valid"]
            if not np.isfinite(original[valid]).all():
                raise ValueError(f"Missing pre-depth Z on a valid frame: {device} {number}")
            primary = a["primary"] & valid
            old = summary(original, a["q"], primary)
            final = summary(a["xyz"][:, 2], a["q"], primary)
            constant_old = constant_final = None
            if trial["mode"] in ("angle_scan", "circle_scan"):
                constant_old = steps(original, a["time"], a["q"], a["action"], valid)
                constant_final = steps(a["xyz"][:, 2], a["time"], a["q"], a["action"], valid)
                assert constant_old["pairs"] == constant_final["pairs"]
            rows.append({"device": device, "trial_id": number, "mode": trial["mode"],
                         "pre_depth": old, "final_online": final,
                         "constant_pre_depth": constant_old,
                         "constant_final_online": constant_final,
                         "primary_mae_change_pct": 100*(final["z_mae_mm"]/old["z_mae_mm"]-1),
                         "same_valid_mask_and_xy": True})
    output = {"schema": "R354-Logged-PreDepth-vs-FinalOnline-1",
              "baseline_definition": "Logged original_z_mm from shared spatial inverse before complete causal depth path; not a single-frame atlas and not increment-only ablation.",
              "reference": report["reference"],
              "translation_or_time_alignment": False,
              "rows": rows}
    path = OUT / "logged_depth_path_ablation.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for r in rows:
        print(r["device"], r["trial_id"], r["mode"],
              f"{r['pre_depth']['z_mae_mm']:.3f}->{r['final_online']['z_mae_mm']:.3f}",
              f"{r['primary_mae_change_pct']:+.1f}%",
              r["constant_pre_depth"], r["constant_final_online"])
    print(path)


if __name__ == "__main__":
    main()
