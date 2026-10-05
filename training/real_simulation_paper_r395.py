"""Frozen-paper models on two complete, newly acquired R385 real batches.

No hardware imports/startup, training, threshold selection, resampling, or
coordinate/time registration. Ordered native journal prefixes are immutable.
Run as a module; derived outputs live outside the raw data tree.
"""
from __future__ import annotations

import argparse
import collections
import copy
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from training.build_trajectory_dataset_v1 import command_segments, command_reference, read_rows, reference_origin
from training.analyze_r354_final_online import mask_for
from training.virtual_discovery_r387 import representation, predict, ARMS
import native_online_replay_r387 as native
import offline_comparison_r390 as comparison

OUT = ROOT / "analysis/r395_new_real_simulation_paper_20261003"
PLANS = {
    "Yellow": ROOT / "data/S20261002_A/Plan_20261002_220737_R385_VALUE_094a3d39c0bc7c_R389_eb5fd3c6e4__mag_5x5-had-Yelow",
    "SOURCE": ROOT / "data/S20261002_A/Plan_20261002_233300_R385_VALUE_01e298cf91cae1_R389_e83c104f9121__mag_5x5-had-239",
}
MODEL_ROOT = ROOT / "analysis/r387_virtual_discovery_20261002"
FILES = ("meta.json", "identity.json", "batch_plan_item.json", "magnetic_raw.csv", "stage_commands.csv", "events.csv", "live_predictions.csv", "live_predictions_virtual1000.csv")


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def clean(v):
    if isinstance(v, dict): return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [clean(x) for x in v]
    if isinstance(v, np.ndarray): return clean(v.tolist())
    if isinstance(v, np.generic): return clean(v.item())
    if isinstance(v, float) and not np.isfinite(v): return None
    return v


def dump(p, v):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(clean(v), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def inventory():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for device, plan in PLANS.items():
        prog = read(plan / "plan_progress.json")
        assert prog["status"] == "COMPLETED" and prog["completed_trials"] == prog["planned_trials"] == 16
        trials = sorted(plan.glob("Trial_*"))
        assert len(trials) == 16
        for k, trial in enumerate(trials):
            comparison._assert_not_excluded(trial)
            m, item = read(trial / "meta.json"), read(trial / "batch_plan_item.json")
            assert m["status"] == "completed"
            tp = m["trial_parameters"]
            assert tp["indenter_id"] == "original_D5_topD3_R1"
            assert item["group"] == prog["queue_status"][k]["group"]
            row = dict(id=f'{device}_{tp["trial_id"]}', device=device, path=trial.relative_to(ROOT).as_posix(),
                       group=item["group"], mode=tp["command_mode"], speed_mm_s=tp["speed_mm_s"],
                       repeats=tp["repeat_count"], origin_mm=reference_origin(m).tolist(),
                       closed_local=m["closed_local"], journal=str(comparison._journal_path(trial, m)),
                       raw_hashes={f: sha(trial / f) for f in FILES if (trial / f).is_file()},
                       model_family_online=m["live_prediction_recording"]["model_family"],
                       recorded_role=m.get("data_role"), analysis_role="frozen_real_test_only",
                       raw_rows=sum(1 for _ in (trial / "magnetic_raw.csv").open(encoding="utf-8-sig")) - 1,
                       prediction_rows=m["live_prediction_recording"].get("rows_written"),
                       queue_drops_recorded=m["live_prediction_recording"].get("input_groups_dropped_by_live_queue"))
            rows.append(row)
    models = {}
    for arm in ARMS:
        f = read(MODEL_ROOT / arm / "freeze.json")
        assert f["status"] == "COMPLETED_PLATEAU" and sha(MODEL_ROOT / arm / "best_model.npz") == f["model_sha256"]
        models[arm] = f
    protocol = dict(status="FROZEN_BEFORE_REPLAY", schema="MagTouch-NewRealPaper-R395-1",
                    real_batches=2, complete_trials=32, complex_trials=24,
                    production_pair=["R373", "VirtualXY1000_R373Z"], research_pair=["short_history", "multiscale_history"],
                    research_reference="current", research_freezes=models,
                    paper=str((ROOT / "paper_export/R386_仿真平台与机制发现_20261002/manuscript_zh.md").relative_to(ROOT)),
                    reporting_name="真实采集数据上的实测效果（离线回测）",
                    reference="same logged serial-dispatch/EST commanded XYZ, not optical/encoder displacement truth",
                    primary="original-paper phase moving, finite command XYZ, nominal Z>=0.4mm; entire action including dwell separately",
                    research_primary="entire action including dwell, nominal Z>=0.45mm, finite actual B0-subtracted field",
                    comparison="same recorded native input subset, ordered controls/B0, no sleep/resampling or added transforms",
                    model_input_excludes=["XYZ commands", "path identity", "future frames", "generator hidden state"],
                    checkpoint_or_threshold_selection_on_these_data=False, training=False,
                    uncertainty="commanded trajectory reference; shallow commanded depth is not contact ground truth",
                    python_script_sha256=sha(__file__), raw_inputs=rows)
    previous = OUT / "protocol.json"
    if previous.exists():
        old = read(previous)
        # Implementation repairs do not change the frozen data/model/metric
        # protocol; retain the original protocol and audit the executed scripts.
        current = clean(protocol)
        assert {k:v for k,v in old.items() if k!="python_script_sha256"} == {k:v for k,v in current.items() if k!="python_script_sha256"}, "Frozen experiment inputs changed"
        dump(OUT / "executed_implementation.json", dict(original_sha256=old["python_script_sha256"], current_sha256=sha(__file__)))
    else: dump(previous, protocol)
    dump(OUT / "work_state.json", dict(status="INVENTORY_COMPLETE", completed=["Two complete batches and all frozen paper models located"],
         pending=["native parity", "augmented full-chain replay", "research readouts", "C-F evidence tables and figures"],
         hardware_operated=False, production_or_raw_changed=False))
    print("Inventory:", len(rows), "complete Trials; frozen paper models verified", flush=True)
    return rows


def snapshot(device, rows):
    dest = OUT / "native" / device / "native_protocol_prefix.jsonl"
    last = ROOT / rows[-1]["path"]
    payload, header, csvs, info = comparison._capture_protocol_prefix(last, Path(rows[-1]["journal"]), read(last / "meta.json"))
    if dest.exists(): assert sha(dest) == info["prefix_sha256"]
    else: dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(payload)
    dump(dest.parent / "prefix_provenance.json", info)
    return dest, header


def selected(rows, trial):
    return [r for r in rows if r.get("online_recorded") and comparison._matches_trial(r.get("trial"), trial)]


def compact(row):
    e = row.get("extra_outputs", {})
    keep = {k: e.get(k) for k in ("baseline_epoch", "external_no_contact_hint", "controlled_pre_action_b0",
             "localization_permission_score", "localization_path", "xyz_valid", "trace_break")}
    de = e.get("final_depth_evidence", {})
    keep["depth_source_support_distance"] = de.get("source_support_distance")
    row["extra_outputs"] = keep
    row.pop("raw_fields", None)


def replay_selection(header, label):
    """Identify the recorded arm by class, never assume A is the baseline."""
    virtual_type="live_paired_virtual_validation_r385:VirtualPredictor"
    baseline_type="live_retrained_reconstruction_r373:LiveRetrainedPredictorR373"
    wanted=virtual_type if label=="VirtualXY1000" else baseline_type
    assert label in ("R373", "VirtualXY1000")
    for target, row in header["models"].items():
        if row["selection"]["predictor_type"] == wanted:
            return copy.deepcopy(row["selection"]), target, True
    spec=copy.deepcopy(header["models"]["A"]["selection"])
    assert spec["predictor_type"] in (virtual_type, baseline_type)
    spec["predictor_type"]=wanted
    if label=="VirtualXY1000":
        VIRTUAL_PRESET_NAME="虚拟扩集1000条XY＋原R373 Z（研究副本）"
        spec.update(label="FIT｜"+VIRTUAL_PRESET_NAME,preset=VIRTUAL_PRESET_NAME,research_only=True)
    else:
        from live_retrained_reconstruction_r373 import PRESET_NAME
        spec.update(label="FIT｜"+PRESET_NAME,preset=PRESET_NAME,research_only=False)
    return spec, "A", False


def replay(device, entries, label, journal, header):
    """One sequential full-prefix pass per native model; no repeated Trial prefixes."""
    dest = OUT / "native" / device / label
    if (dest / "completed.json").exists():
        print("Reuse completed", device, label, flush=True); return read(dest / "completed.json")
    dest.mkdir(parents=True, exist_ok=True)
    from general_model_selection_r387 import load, describe
    spec, target, same = replay_selection(header, label)
    predictor = load(spec)
    description = describe(predictor, spec)
    if same:
        before, now = native._hash_fields(header["models"][target]), native._hash_fields(description)
        assert not {k: (v, now.get(k)) for k, v in before.items() if v != now.get(k)}, "Recorded numerical model/profile hash changed"
        hashes = native.implementation_sources(predictor)
        changed = {k: (v, hashes.get(k)) for k, v in header.get("implementation_sources", {}).get(target, {}).items() if v != hashes.get(k)}
        assert not changed, "Recorded native implementation changed: " + str(changed)
    session = native.NativeReplaySession(predictor, target, annotate_display=(target == "A"),
                                        honor_output_record_markers=same, allow_unsupported_controls=False)
    output, fields, b0_epochs, b0_log = [], {}, {}, []
    completed_b0, groups = 0, 0
    started = time.perf_counter()
    last_ready = False
    for record in native._read_journal(journal):
        old_ready = bool(predictor.ready)
        rr = session.consume(record)
        for r in rr:
            compact(r); output.append(r)
        if record["op"] == "group":
            groups += 1
            if predictor.ready:
                if not old_ready:
                    completed_b0 += 1
                    b0_log.append(dict(t_ns=record["t_ns"], epoch=completed_b0, b0_uT=predictor.b0.reshape(-1).tolist()))
                data = {int(k): v for k, v in record["data"].items()}
                from live_normalized_fit import vector
                v = vector(data)
                if v is not None:
                    fields[int(record["t_ns"])] = v - predictor.b0.reshape(-1)
                    b0_epochs[int(record["t_ns"])] = completed_b0
            if groups % 4000 == 0:
                print(device, label, "groups", groups, "elapsed_s", round(time.perf_counter()-started, 1), flush=True)
        last_ready = bool(predictor.ready)
    parity = []
    for item in entries:
        trial = ROOT / item["path"]
        if same:
            rows = selected(output, trial)
        else:
            # Both arms are the SAME native group-time adapter, with different
            # XY weights only. Preserve the acquisition's buffered Trial
            # assignment for every emitted timestamp (including rejected A).
            # This is not permission filtering or deleting B-only valid XYZ.
            with (trial / "live_predictions.csv").open(encoding="utf-8-sig",newline="") as f:
                stamp={int(r["t_prediction_ns"]) for r in csv.DictReader(f)}
            rows=[r for r in output if int(r["t_ns"]) in stamp]
            assert len(rows)==len(stamp), "Same native adapter did not emit all recorded Trial input timestamps"
        assert rows, "No Trial outputs " + str(trial)
        tn = np.array([r["t_ns"] for r in rows], np.int64)
        assert (np.diff(tn) > 0).all(), "Duplicate/reordered native emission"
        xyz = np.array([r["visual_xyz_mm"] for r in rows], float)
        ok = np.array([r["interaction_valid"] for r in rows], bool) & np.isfinite(xyz).all(1)
        extra = [r["extra_outputs"] for r in rows]
        b = np.array([fields.get(int(t), np.full(15, np.nan)) for t in tn])
        epoch = np.array([b0_epochs.get(int(t), -1) for t in tn])
        np.savez_compressed(dest / (item["id"] + ".npz"), t_ns=tn, xyz=xyz, ok=ok, b=b, b0_epoch=epoch,
                            hint=np.array([e.get("external_no_contact_hint") is True for e in extra]),
                            permission=np.array([e.get("localization_permission_score") if e.get("localization_permission_score") is not None else np.nan for e in extra]))
        if same:
            csvpath = trial / ("live_predictions.csv" if target == "A" else "live_predictions_virtual1000.csv")
            result = native.compare_online_csv(rows, csvpath)
            parity.append(dict(id=item["id"], **result))
            assert result["xyz_and_validity_parity"], "Online replay parity failed: " + str(result)
        if label == "R373":
            saved = np.asarray(read(trial / "meta.json")["dynamic_b0_calibration"]["b0_5x3_uT"]).reshape(-1)
            start_ns = int(read(trial / "meta.json")["live_prediction_recording"]["started_perf_ns"])
            available = [x for x in b0_log if x["t_ns"] <= start_ns]
            assert available and np.max(np.abs(np.asarray(available[-1]["b0_uT"]) - saved)) < 1e-10, "Initial B0 differs from acquisition metadata"
    result = dict(status="COMPLETED", label=label, groups=groups, seconds=time.perf_counter()-started,
                  description=description, implementation_sha256=native.implementation_sources(predictor),
                  same_model_as_recorded=same, parity=parity, b0_windows=b0_log,
                  gaps=session.gaps, snapshot_sha256=sha(journal), controls=session.controls,
                  actual_timestamps_and_native_controls_reproduced=True, external_resampling=False,
                  added_coordinate_transform=False, raw_files_unchanged=True)
    dump(dest / "completed.json", result)
    print("Finished", device, label, "parity", len(parity), "seconds", round(result["seconds"], 1), flush=True)
    del predictor, session, output, fields
    gc.collect()
    return result


def metric(xyz, q, ok, mask, t):
    eligible = np.asarray(mask, bool) & np.isfinite(q).all(1)
    good = eligible & ok & np.isfinite(xyz).all(1)
    n, count = int(eligible.sum()), int(good.sum())
    r = dict(reference_frames=n, output_frames=count, rejected_frames=n-count, coverage=count/n if n else None)
    if count:
        e = xyz[good]-q[good]; bias=e.mean(0); centered=e-bias
        r.update(xyz_mean_mm=np.linalg.norm(e, axis=1).mean(), xy_mean_mm=np.linalg.norm(e[:, :2], axis=1).mean(),
                 z_mae_mm=np.abs(e[:, 2]).mean(), xyz_p95_mm=np.quantile(np.linalg.norm(e, axis=1), .95),
                 bias_x_mm=bias[0], bias_y_mm=bias[1], bias_z_mm=bias[2],
                 translation_removed_xyz_diagnostic_mm=np.linalg.norm(centered, axis=1).mean())
    pair = good[:-1] & good[1:] & (np.diff(t) > 0) & (np.diff(t) <= 250_000_000)
    dz = np.abs(np.diff(xyz[:, 2]))[pair]
    dr = np.abs(np.diff(xyz[:, 2]-q[:, 2]))[pair]
    r.update(adjacent_pairs=int(pair.sum()), z_jump_over_0p2_count=int((dz > .2).sum()),
             z_step_p95_mm=np.quantile(dz, .95) if len(dz) else None,
             z_residual_step_p95_mm=np.quantile(dr, .95) if len(dr) else None)
    return clean(r)


def native_condition_masks(q, t, valid):
    """Original paper thresholds; 1-second tail uses time, not fixed 11 samples."""
    dt = np.r_[1., np.diff(t)/1e9]; v = np.vstack((np.zeros(3), np.diff(q, axis=0))) / dt[:, None]
    v[~np.isfinite(v)] = 0.
    speed = np.linalg.norm(v, axis=1); prev = np.vstack((np.zeros(3), v[:-1]))
    cos = np.sum(v*prev, axis=1)/np.maximum(speed*np.linalg.norm(prev, axis=1), 1e-10)
    events = (speed>.05) & (np.linalg.norm(prev, axis=1)>.05) & (cos<.5) & valid
    recent = np.zeros(len(t), bool)
    for i in np.flatnonzero(events): recent[i:np.searchsorted(t, t[i]+1_000_000_000, side="right")] = True
    return dict(moving=speed>.05, dwell=speed<.02, turn_or_reverse_recent_1s=recent,
                coupled_xyz=(np.linalg.norm(v[:, :2], axis=1)>.05) & (np.abs(v[:, 2])>.03))


def analyze(entries):
    (OUT / "streams").mkdir(exist_ok=True)
    results, conditions, input_audit = [], [], []
    for item in entries:
        trial = ROOT / item["path"]; meta=read(trial / "meta.json")
        with np.load(OUT / "native" / item["device"] / "R373" / (item["id"]+".npz")) as f:
            a = {k: f[k].copy() for k in f.files}
        with np.load(OUT / "native" / item["device"] / "VirtualXY1000" / (item["id"]+".npz")) as f:
            aug = {k: f[k].copy() for k in f.files}
        assert np.array_equal(a["t_ns"], aug["t_ns"]), "Different native emission timestamps"
        np.testing.assert_allclose(a["b"], aug["b"], atol=0, rtol=0, equal_nan=True)
        assert np.array_equal(a["b0_epoch"], aug["b0_epoch"])
        t=a["t_ns"]; seg=command_segments(read_rows(trial / "stage_commands.csv"))
        ref=command_reference(t, seg, reference_origin(meta)); q=ref["xyz_mm"]
        phase=np.array([seg[j]["phase"] if j>=0 else "before_stage" for j in ref["command_segment_id"]])
        primary, action = mask_for(item["mode"], phase, ref)
        valid = np.isfinite(a["b"]).all(1)
        whole = action & (q[:, 2] >= .4) & valid
        primary &= valid
        safe = np.isfinite(q[:, 2]) & (q[:, 2] <= -.2)
        if item["mode"] == "none": safe[:] = True
        rm = action & (q[:, 2] >= .45) & valid
        predictions = {"R373":a["xyz"], "VirtualXY1000":aug["xyz"]}
        permissions = {"R373":a["ok"], "VirtualXY1000":aug["ok"]}
        # Real observed magnetic history, not commands/teacher states.
        # Preserve full native Trial prefix, reset only actual B0/invalid gaps.
        b=np.where(valid[:, None], a["b"], 0.)
        reset=np.r_[True, np.diff(a["b0_epoch"]) != 0]
        reset |= ~valid | np.r_[False, ~valid[:-1]]
        for arm in ARMS:
            with np.load(MODEL_ROOT / arm / "best_model.npz", allow_pickle=False) as f:
                params=[f[f"p{i}"].copy() for i in range(4)]; mean=f["mean"]; scale=f["scale"]; ys=f["yscale"]
            x=representation(b, (t-t[0])/1e9, reset, arm)
            y=predict(params, (x-mean)/scale)*ys
            y[~valid]=np.nan
            predictions[arm]=y; permissions[arm]=valid & np.isfinite(y).all(1)
        masks = {"primary_moving":primary, "whole_action":whole, "shared_full_chain":primary & a["ok"] & aug["ok"],
                 "research_whole_action":rm}
        for label in predictions:
            for maskname in (("primary_moving", "whole_action", "shared_full_chain") if label in ("R373", "VirtualXY1000") else ("research_whole_action",)):
                r=dict(id=item["id"], device=item["device"], group=item["group"], mode=item["mode"], speed_mm_s=item["speed_mm_s"],
                       model=label, scope=maskname, **metric(predictions[label], q, permissions[label], masks[maskname], t))
                r.update(safe_reference_frames=int(safe.sum()), safe_output_frames=int((safe & permissions[label]).sum()),
                         safe_without_external_hint_frames=int((safe & ~a["hint"]).sum()),
                         safe_without_external_hint_outputs=int((safe & ~a["hint"] & permissions[label]).sum()),
                         safety_gate_implemented=label in ("R373", "VirtualXY1000"))
                results.append(r)
        kin = native_condition_masks(q, t, rm)
        for label in predictions:
            base = rm if label in ARMS else whole
            for condition, mask in kin.items():
                conditions.append(dict(id=item["id"], device=item["device"], group=item["group"], model=label, condition=condition,
                                       **metric(predictions[label], q, permissions[label], base & mask, t)))
        # Stable command hold records are described; no equilibrium magnetic truth
        # or fitted no-contact threshold is invented from this test set.
        actual_dt=np.diff(t)/1e9
        input_audit.append(dict(id=item["id"], device=item["device"], group=item["group"], raw_rows=item["raw_rows"],
              actual_native_outputs=len(t), native_dt_median_s=float(np.median(actual_dt)), native_dt_p95_s=float(np.quantile(actual_dt,.95)),
              native_dt_max_s=float(actual_dt.max()), gaps_over_0p5s=int((actual_dt>.5).sum()),
              supported_dwell_frames=int((rm & kin["dwell"]).sum()), nominal_primary_frames=int(primary.sum()),
              whole_action_frames=int(whole.sum()), research_reference_frames=int(rm.sum()),
              fewer_R373_outputs_than_aug=int((primary & ~a["ok"] & aug["ok"]).sum()),
              fewer_aug_outputs_than_R373=int((primary & a["ok"] & ~aug["ok"]).sum()),
              actual_b0_epochs=np.unique(a["b0_epoch"]).tolist()))
        np.savez_compressed(OUT / "streams" / (item["id"]+".npz"), t_ns=t, q=q, b=a["b"], b0_epoch=a["b0_epoch"],
              phase=phase, primary=primary, whole_action=whole, research_primary=rm, safe=safe, hint=a["hint"],
              **{label+"_xyz":p for label,p in predictions.items()}, **{label+"_ok":v for label,v in permissions.items()})
        print("Analyzed", item["id"], item["group"], "reference", int(primary.sum()), flush=True)
    pd.DataFrame(results).to_csv(OUT / "trial_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(conditions).to_csv(OUT / "condition_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(input_audit).to_csv(OUT / "input_audit.csv", index=False, encoding="utf-8-sig")
    dump(OUT / "trial_metrics.json", results); dump(OUT / "input_audit.json", input_audit)
    return results


def main():
    p=argparse.ArgumentParser(); p.add_argument("--phase", choices=("inventory", "native", "analyze", "all"), default="all")
    args=p.parse_args(); rows=inventory()
    if args.phase == "inventory": return
    if args.phase in ("all", "native"):
        for device in PLANS:
            entries=[r for r in rows if r["device"]==device]
            journal, header=snapshot(device, entries)
            for label in ("R373", "VirtualXY1000"):
                dump(OUT / "work_state.json", dict(status="NATIVE_REPLAY_RUNNING", device=device, model=label,
                     pid=os.getpid(), protocol_sha256=sha(OUT / "protocol.json"), hardware_operated=False))
                replay(device, entries, label, journal, header)
    if args.phase in ("all", "analyze"):
        analyze(rows)
    changed=[r["path"]+"/"+f for r in rows for f,h in r["raw_hashes"].items() if sha(ROOT / r["path"] / f)!=h]
    assert not changed, "Raw source changed during replay"
    dump(OUT / "work_state.json", dict(status="REPLAY_AND_TABLES_COMPLETE" if args.phase != "native" else "NATIVE_REPLAY_COMPLETE",
         complete_trials=32, raw_sources_unchanged=True, frozen_models_unchanged=True, hardware_operated=False,
         next=["independent numerical audit", "aggregate C-F evidence and figures"], protocol_sha256=sha(OUT / "protocol.json")))


if __name__ == "__main__": main()
