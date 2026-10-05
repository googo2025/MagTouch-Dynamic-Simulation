"""Use existing Source grid coverage for fallback-position permission.

All three training grids are chosen by the preexisting R276 whole-Trial
partition and metadata, not by results. No new collection/Target labels.
R330/R331 frozen files are left intact. Decision-focused training only.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import training.probe_localization_evidence_r331 as previous
from training.audit_r274_frozen_phase_expansion import load_record, post_group, save_json
from training.unified_response_runtime_r330 import digest
from training.unified_response_runtime_r330 import FrozenMLP
from training.probe_r276_condition_adaptation import source_partition
from training.probe_r245_joint_contact_state import StateMLP
from training.audit_unified_tactile_r330 import UnifiedTactileEngine, load_profile, R92_PROFILE_PATH, PROFILE_KEY
from training.had239_r133_coverage_xy_runtime import CoverageGuardedNeuralDecoder

ROOT = previous.ROOT
OUT = ROOT/"analysis/r332_source_grid_permission_20260929"
PRIOR = previous.OUT


def protocol():
    return {"schema": "R332-ExistingSourceGridCoverage-1", "source_extra_train_trials": [12, 14, 109],
        "source_grid_rule": "all earlier current-assembly train grid_points under unchanged R276 partitions",
        "selection_data": "Source validation 9/11/12 from R331; no retained Source or Target selection",
        "features_and_labels": "exact R331 61 features and calibrated-range criteria; no nominal extreme-shallow XYZ/contact labels",
        "training_focus": "only finite responsive candidates currently rejected by R330, plus 1/32 original reliable anchors",
        "widths": [0, 24], "limits": [.5, .7, .85, .9, .95, .975, .99, .995, .999],
        "confirm_s": [0., .03], "epochs": 100, "seed": 332,
        "acceptance": "zero added Source validation lift/control XYZ; >=95% reliable among added calibrated-reference outputs; all existing R330 valid XYZ unchanged",
        "selection": "most added reliable Source validation outputs among qualified; smaller model then higher limit",
        "target_fit_rows": 0, "target_selection_rows": 0, "source_test_selection_rows": 0,
        "new_online_test": False, "paper_or_UI_update": False}


def declare():
    OUT.mkdir(exist_ok=True); (OUT/"cache").mkdir(exist_ok=True); (OUT/"seed_streams").mkdir(exist_ok=True)
    file = OUT/"declared_protocol.json"
    if file.exists():
        assert json.loads(file.read_text(encoding="utf-8")) == protocol()
    else: save_json(file, protocol())
    for item in json.loads((PRIOR/"freeze.json").read_text(encoding="utf-8"))["tracked"]:
        assert digest(ROOT/item["path"]) == item["sha256"]


def extra_records():
    items = json.loads((ROOT/"analysis/r274_frozen_phase_expansion_20260929/inventory.json").read_text(encoding="utf-8"))["trials"]
    rows = [r for r in items if r["device"] == "source" and r["date"] >= "S20260926_A" and r["initially_readable"]
            and r["mode"] == "grid_points" and source_partition(r) == "train"]
    assert sorted(int(Path(r["path"]).name.split("_")[1]) for r in rows) == protocol()["source_extra_train_trials"]
    return rows


def seed_and_cache(rec):
    path = ROOT/rec["path"]; stem = digest(path/"meta.json")[:16]
    file = OUT/"seed_streams"/(stem+".npz")
    if not file.exists():
        row = load_record(rec); n = len(row["time"])
        profile = load_profile(R92_PROFILE_PATH, PROFILE_KEY)
        engine = UnifiedTactileEngine(profile, profile["defaults"]["threshold"], profile["defaults"]["min_cosine"],
            profile["defaults"]["smoothing_ms"], 70., CoverageGuardedNeuralDecoder.from_report())
        engine.set_baseline(np.zeros(15)); engine.baseline_effective_ns = row["B0_effective_ns"]
        xyz = np.full((n, 3), np.nan); valid, response = np.zeros(n, bool), np.zeros(n, bool)
        strength, phase = np.zeros(n, int), np.zeros(n, int)
        for i, (t, b, ok) in enumerate(zip(row["time"], row["field_uT"], row["eligible"])):
            out = engine.push(t, b, eligible=ok); obs = out["extra"]["response_observation"]
            response[i], strength[i], phase[i] = obs["response"], obs["strength"], obs["phase"]
            if out["valid"]: valid[i] = True; xyz[i] = out["xyz"]
            if (i+1)%5000 == 0: print("Source grid primary", Path(rec["path"]).name[:20], i+1, "/", n, flush=True)
        # Labels/reference are introduced only after inference.
        groups, q, error = post_group(row, rec)
        if error: raise ValueError(error)
        np.savez_compressed(file, time_ns=row["time"], eligible=row["eligible"], q=q, diagnostic_group=groups,
            xyz_mm=xyz, xyz_valid=valid, response=response, strength=strength, phase=phase)
    previous.OUT = OUT; previous.PRIOR = OUT
    return previous.cache(rec, {"stream": file.relative_to(OUT).as_posix()})


def dataset(caches):
    xs, ys, ws = [], [], []
    for c in caches:
        with np.load(ROOT/c["cache"], allow_pickle=False) as a:
            fallback = (~a["base_valid"] & a["eligible"] & a["response"] &
                ~np.isin(a["strength"], (1, 5)) & np.isfinite(a["candidate_xyz_mm"]).all(1) & (a["y"] >= 0))
            anchor = a["eligible"] & a["base_valid"] & (a["y"] == 1) & (np.arange(len(a["y"]))%32 == 0)
            ids = np.flatnonzero(fallback | anchor)
            if not len(ids): continue
            x, y = a["x"][ids], a["y"][ids]
        w = np.zeros(len(y))
        for label in (0, 1):
            mask = y == label
            if mask.any(): w[mask] = 1./mask.sum()
        xs.append(x); ys.append(y); ws.append(w)
    w = np.concatenate(ws); w *= len(w)/w.sum()
    return np.concatenate(xs), np.concatenate(ys), w


def train():
    if (OUT/"freeze.json").exists(): raise FileExistsError("Source selection already frozen")
    base = [json.loads(j.read_text(encoding="utf-8")) for j in (PRIOR/"cache").glob("*.json")]
    base = [c for c in base if c["split"] in ("train", "validation")]
    extra = [json.loads(j.read_text(encoding="utf-8")) for j in (OUT/"cache").glob("*.json")]
    assert len(extra) == 3 and all(c["split"] == "train" for c in extra)
    rows = base+extra; tr = [c for c in rows if c["split"] == "train"]; va = [c for c in rows if c["split"] == "validation"]
    tx, ty, tw = dataset(tr); vx, vy, vw = dataset(va)
    candidates, specs = [], []
    for width in protocol()["widths"]:
        model = StateMLP(width, seed=332, n_classes=2).fit(tx, ty, tw, (vx, vy, vw), epochs=100)
        file = OUT/f"permission_width{width}.npz"; model.save(file)
        specs.append({"width": width, "epoch": model.best_epoch, "loss": model.best_loss,
                      "file": file.relative_to(ROOT).as_posix()})
        for limit in protocol()["limits"]:
            for confirm in protocol()["confirm_s"]:
                val = previous.evaluate(va, model, limit, confirm)
                candidates.append({"width": width, "score_limit": limit, "confirmation_s": confirm, "validation": val})
        print("Source grid-focused permission", specs[-1], flush=True)
    qualified = [c for c in candidates if c["validation"]["summary"]["qualified"]]
    selected = min(qualified, key=lambda c: (-c["validation"]["summary"]["added_reliable"], c["width"], -c["score_limit"], c["confirmation_s"])) if qualified else None
    save_json(OUT/"training_report.json", {"selected": selected, "candidates": candidates, "models": specs,
        "train_samples": len(tx), "validation_samples": len(vx), "class_counts": np.bincount(ty, minlength=2).tolist(),
        "source_trials": rows, "target_fit_rows": 0, "target_selection_rows": 0, "source_test_selection_rows": 0})
    file = next(s["file"] for s in specs if selected and s["width"] == selected["width"]) if selected else None
    tracked = [Path(__file__), Path(previous.__file__), OUT/"declared_protocol.json", OUT/"training_report.json",
        PRIOR/"freeze.json", ROOT/"training/probe_r245_joint_contact_state.py"]+[ROOT/s["file"] for s in specs]
    save_json(OUT/"freeze.json", {"schema": "R332-FrozenSourceGridPermission-1", "selected": selected, "model": file,
        "target_fit_rows": 0, "target_selection_rows": 0, "source_test_selection_rows": 0,
        "frozen_before_target_replay": True, "tracked": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in tracked],
        "source_feature_caches": [{"path": c["cache"], "sha256": digest(ROOT/c["cache"])} for c in rows]})
    print("FROZEN Source grid-focused permission", None if selected is None else selected["validation"]["summary"], flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--stage", choices=("cache", "train"), default="cache")
    parser.add_argument("--trial", type=int); args = parser.parse_args(); declare()
    if args.stage == "train": return train()
    rows = extra_records()
    if args.trial is not None: rows = [r for r in rows if int(Path(r["path"]).name.split("_")[1]) == args.trial]
    for r in rows: print(json.dumps(seed_and_cache(r), ensure_ascii=False), flush=True)


if __name__ == "__main__": main()
