"""Verify frozen weights -> causal features -> response states on real streams.

This is not a new gate or hardware driver. It verifies the derived offline
outputs and temporal prefix invariance, never using XYZ/reference as input.
"""
from __future__ import annotations

from dataclasses import fields
import hashlib
import json
import numpy as np

from training.probe_r269_hierarchical_response_state import ROOT, load_rows, prepare_hierarchy, head_features, StateMLP, SCALAR_IDS
from training.probe_r245_joint_contact_state import causal_state_features
from training.response_phase_state import PhasePolicy
from training.response_phase_state_v3 import replay_rearm_phase
from training.factorized_response_phase import DirectionPolicy, replay_factorized

HEADS = ROOT / "analysis/r269_hierarchical_response_state_20260929"
STRENGTH = ROOT / "analysis/r271_rearm_phase_continuity_20260929"
CONDITIONAL = ROOT / "analysis/r272_factorized_action_phase_20260929"
PHASE = ROOT / "analysis/r273_confirmed_action_phase_20260929"
OUT = ROOT / "analysis/r273_response_phase_summary_20260929/runtime_verification.json"


def load_model(path):
    with np.load(path, allow_pickle=False) as a:
        model = StateMLP(int(a["width"]), n_classes=int(a["n_classes"]))
        model.mean, model.scale = a["mean"].copy(), a["scale"].copy()
        keys = sorted((k for k in a.files if k.startswith("p") and k[1:].isdigit()), key=lambda k: int(k[1:]))
        model.params = [a[k].copy() for k in keys]
    return model


def policy(cls, values):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in values.items() if k in names})


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    h = json.loads((HEADS / "report.json").read_text(encoding="utf-8"))
    s = json.loads((STRENGTH / "report.json").read_text(encoding="utf-8"))
    b = json.loads((CONDITIONAL / "report.json").read_text(encoding="utf-8"))
    p = json.loads((PHASE / "report.json").read_text(encoding="utf-8"))
    model_files = [HEADS / f"{name}_{h['head_training'][name]['selected']}.npz" for name in ("context", "direction")]
    model_files.append(CONDITIONAL / f"conditional_direction_width{b['selected_width']}.npz")
    models = [load_model(path) for path in model_files]
    strength_policy, phase_policy = policy(PhasePolicy, s["policy"]), policy(DirectionPolicy, p["policy"])
    rows = load_rows(); prepare_hierarchy(rows)
    checked, total = [], 0
    for row in rows:
        n = len(row["time"])
        trial = int(row["path"].name.split("_")[1])
        with np.load(PHASE / f"trial_{trial:06d}_phase.npz", allow_pickle=False) as a:
            saved_strength, saved_phase = a["strength_state"].copy(), a["phase"].copy()
            assert np.all(a["physical_contact_truth"] == -1)
        feature = head_features(row, "scalar")
        context, direction, binary = [model.predict(feature) for model in models]
        fresh = 100.*row["state_x"][:, 1] > s["fresh_rise_limits_uT"][row["device"]]
        states, _, _, _ = replay_rearm_phase(row["time"], context, direction, row["response"], fresh,
                                             row["amplitude_uT"], row["eligible"], strength_policy)
        phase, _, _ = replay_factorized(row["time"], states, direction, binary, row["response"],
                    row["state_x"][:, 1]*100., p["activity_change_limits_uT"][row["device"]], row["eligible"], phase_policy)
        np.testing.assert_array_equal(states, saved_strength)
        np.testing.assert_array_equal(phase, saved_phase)
        cuts = sorted({min(n, 512), n//2, n-1} - {0})
        for cut in cuts:
            x = causal_state_features(row["time"][:cut], row["field_uT"][:cut], row["eligible"][:cut])
            np.testing.assert_allclose(x, row["state_x"][:cut], rtol=0., atol=1e-12)
            c, d, conditional = [model.predict(x[:, SCALAR_IDS]) for model in models]
            for prefix, full in zip((c, d, conditional), (context, direction, binary)):
                np.testing.assert_allclose(prefix, full[:cut], rtol=0., atol=1e-12)
            prefix_states, _, _, _ = replay_rearm_phase(row["time"][:cut], c, d, row["response"][:cut],
                fresh[:cut], row["amplitude_uT"][:cut], row["eligible"][:cut], strength_policy)
            prefix_phase, _, _ = replay_factorized(row["time"][:cut], prefix_states, d, conditional,
                row["response"][:cut], x[:, 1]*100., p["activity_change_limits_uT"][row["device"]], row["eligible"][:cut], phase_policy)
            np.testing.assert_array_equal(prefix_states, states[:cut])
            np.testing.assert_array_equal(prefix_phase, phase[:cut])
        total += n
        checked.append({"trial": row["path"].name, "frames": n, "prefix_cuts": cuts})
    files = model_files + [HEADS / "report.json", STRENGTH / "report.json", CONDITIONAL / "report.json", PHASE / "report.json"]
    report = {"schema": "R273-RealStreamRuntimeVerification-1", "result": "passed",
              "trials": len(checked), "frames": total, "prefixes_checked": sum(len(v["prefix_cuts"]) for v in checked),
              "input_features": "16 scalars of causal baseline-subtracted 15D magnetic field history and elapsed time; no XYZ, command, UI coordinates or future field",
              "coordinate_output": "none; shadow response context and phase only",
              "selected_frozen_files_sha256": [{"path": path.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in files],
              "checks": checked}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("result", "trials", "frames", "prefixes_checked")}), flush=True)


if __name__ == "__main__":
    main()
