"""Build a standalone causal CPA trajectory dataset, without importing hardware/UI.

Original recordings are read-only. Labels are commanded, not measured, positions.
Run with --output a NEW directory. NumPy and pandas are the only dependencies.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import time

import numpy as np
import pandas as pd

SCHEMA = "MagTouch-CPA-Trajectory-1.0"
HZ = 40
STEP_NS = 25_000_000
B_COLS = [f"S{s}_B{a}_uT" for s in range(1, 6) for a in "xyz"]
ADC_COLS = [f"S{s}_{a}_raw" for s in range(1, 6) for a in "xyz"]
PRIMARY = {"S20260917_A", "S20260918_A", "S20260919_A", "S20260920_A"}
FILES = ("meta.json", "magnetic_raw.csv", "events.csv", "stage_commands.csv")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def number(value, default=np.nan):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def event_value(row):
    try:
        return json.loads(row["value"])
    except (ValueError, KeyError):
        return {}


def reference_origin(meta):
    # Deliberately no geometry fallback, nominal hardcoded origin, or item center.
    ref = meta.get("acquisition", {}).get("printer_reference", {})
    origin = np.array([number(ref.get(k)) for k in
                       ("sensor_center_x_mm", "sensor_center_y_mm", "contact_z_mm")])
    if not np.isfinite(origin).all():
        raise ValueError("missing_acquisition_printer_reference")
    return origin


def command_segments(rows):
    """Known axis values propagate only forwards; unknown initial axes stay NaN.

    EST duration is the logged host motion estimate, not actual encoder feedback.
    New recordings use the explicitly logged serial-dispatch timestamp; older
    recordings retain their historical post-ACK timestamp without silent repair.
    Unknown initial moves become known after their HOST_WAIT, not at time zero.
    """
    rows = sorted(rows, key=lambda r: int(r["t_perf_ns"]))
    absolute, pos, feed = True, np.full(3, np.nan), np.nan
    result = []
    for i, row in enumerate(rows):
        cmd = row["command"].strip().upper()
        if cmd == "G90":
            absolute = True
        elif cmd == "G91":
            absolute = False
        if not re.match(r"^G0?1(?:\s|$)", cmd):
            continue
        if str(row.get("dry_run", "")).lower() == "true":
            raise ValueError("dry_run_stage_commands")
        if not absolute:
            raise ValueError("relative_stage_mode_not_supported")
        t0 = int(row.get("t_send_perf_ns") or row["t_perf_ns"])
        target = pos.copy()
        specified = np.zeros(3, bool)
        for j, axis in enumerate("xyz"):
            val = number(row.get(f"{axis}_mm"))
            if not np.isfinite(val):
                match = re.search(rf"(?:^|\s){axis.upper()}([-+\d.eE]+)", cmd)
                val = number(match.group(1)) if match else np.nan
            if np.isfinite(val):
                target[j], specified[j] = val, True
        new_feed = number(row.get("feed_mm_min"))
        if np.isfinite(new_feed):
            feed = new_feed
        wait_t, duration, duration_source = None, None, "distance_feed"
        next_move_t = None
        for following in rows[i + 1:]:
            if re.match(r"^G0?1(?:\s|$)", following["command"].strip().upper()):
                next_move_t = int(following.get("t_send_perf_ns") or following["t_perf_ns"])
                break
            if following["command"] == "HOST_WAIT" and following["phase"] == row["phase"] + "_host_wait":
                match = re.search(r"EST=([\d.]+)s", following.get("response", ""))
                wait_t = int(following["t_perf_ns"])
                if match:
                    duration, duration_source = float(match.group(1)), "logged_EST"
        changed_known = specified & np.isfinite(pos)
        if duration is None and np.all(np.isfinite(pos[specified])) and feed > 0:
            duration = float(np.linalg.norm(target[changed_known] - pos[changed_known]) / (feed / 60))
        unknown_start = bool(np.any(specified & ~np.isfinite(pos)))
        if duration is None or unknown_start:
            if wait_t is None:
                raise ValueError("unknown_move_start_without_completion_log")
            duration = max(0., (wait_t - t0) / 1e9)
            duration_source = "unknown_initial_position_wait_bound"
        end = t0 + round(duration * 1e9)
        if next_move_t is not None and end > next_move_t + 10_000_000:
            raise ValueError("overlapping_unresolved_command_estimates")
        result.append({"start_ns": t0, "end_ns": end, "from_mm": pos.copy(),
                       "to_mm": target.copy(), "phase": row["phase"],
                       "duration_source": duration_source, "feed_mm_min": feed})
        pos = target
    if not result:
        raise ValueError("no_G1_reference")
    return result


def command_reference(t, segments, origin):
    n = len(t)
    position = np.full((n, 3), np.nan)
    moving = np.zeros(n, bool)
    boundary = np.zeros(n, bool)
    segment_id = np.full(n, -1, np.int32)
    hold_age = np.zeros(n, np.float32)
    for i, seg in enumerate(segments):
        start, end = seg["start_ns"], seg["end_ns"]
        stop = segments[i + 1]["start_ns"] if i + 1 < len(segments) else int(t[-1]) + 1
        ids = np.flatnonzero((t >= start) & (t < stop))
        if not len(ids):
            continue
        dt = max(end - start, 1)
        fraction = np.clip((t[ids] - start) / dt, 0., 1.)
        a, b = seg["from_mm"], seg["to_mm"]
        values = a[None, :] + fraction[:, None] * (b - a)[None, :]
        values[t[ids] >= end] = b
        position[ids] = values
        known_move = np.isfinite(a).all() and np.linalg.norm(b - a) > 1e-8
        moving[ids] = (t[ids] < end) & known_move
        segment_id[ids] = i
        hold_age[ids] = np.maximum(0., (t[ids] - end) / 1e9)
        # Boundary uncertainty is a weight, never a wholesale deletion of short arcs.
        boundary[ids] = (np.abs(t[ids] - start) < 50_000_000) | (np.abs(t[ids] - end) < 50_000_000)
    xyz = (position - origin) * np.array([1., 1., -1.])
    valid_axes = np.isfinite(xyz)
    contact = np.full(n, -1, np.int8)
    contact[valid_axes[:, 2] & (xyz[:, 2] >= .02)] = 1
    contact[valid_axes[:, 2] & (xyz[:, 2] <= -.02)] = 0
    return {"xyz_mm": xyz.astype(np.float32), "xyz_axis_known": valid_axes,
            "contact_command": contact, "command_moving": moving,
            "command_boundary": boundary, "command_segment_id": segment_id,
            "hold_age_s": hold_age}


def load_magnetic(path):
    fields = B_COLS + ADC_COLS + ["t_group_ns", "t_return_ns", "valid_sensor_count", "error_json"]
    fields += [f"S{s}_read_ns" for s in range(1, 6)]
    df = pd.read_csv(path, usecols=lambda c: c in fields, low_memory=False)
    if len(df) < 5:
        raise ValueError("fewer_than_five_raw_frames")
    group = df.t_group_ns.to_numpy(dtype=np.int64)
    returned = pd.to_numeric(df.t_return_ns, errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
    read_t = np.column_stack([pd.to_numeric(df[f"S{s}_read_ns"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64) for s in range(1, 6)])
    availability = np.maximum(returned, read_t.max(axis=1))
    timestamp_valid = (availability >= group) & (group > 0)
    # Keep bad-timestamp source rows in native archive; do not fabricate availability.
    order = np.flatnonzero(timestamp_valid)
    order = order[np.argsort(availability[order], kind="stable")]
    if len(order) < 5:
        raise ValueError("insufficient_valid_availability_timestamps")
    b = df[B_COLS].to_numpy(dtype=np.float32)[order]
    adc = df[ADC_COLS].to_numpy(dtype=np.float32)[order]
    finite = np.isfinite(b).reshape(-1, 5, 3).all(axis=2) & np.isfinite(adc).reshape(-1, 5, 3).all(axis=2)
    count = df.valid_sensor_count.to_numpy(dtype=np.int16)[order]
    # Without explicit per-sensor success, incomplete groups conservatively mask all.
    valid = finite & (count[:, None] == 5)
    peak = np.nanmax(np.abs(adc).reshape(-1, 5, 3), axis=2)
    jump = np.zeros_like(valid)
    jump[1:] = (np.abs(np.diff(adc.astype(np.float64), axis=0)).reshape(-1, 5, 3) > 60000).any(axis=2)
    wrap = jump.copy()
    for k in range(1, 4):
        wrap[k:] |= jump[:-k]
    valid &= (peak < 32760) & ~wrap
    soft = np.clip((32760 - peak) / (32760 - .85 * 32768), 0., 1.).astype(np.float32)
    return {"t_group_ns": group[order], "t_available_ns": availability[order],
            "source_row": order.astype(np.int32), "b_uT": b, "adc_raw": adc,
            "sensor_valid": valid, "sensor_soft_quality": soft,
            "raw_wrap_mask": wrap, "raw_rows_total": len(df),
            "invalid_time_rows": int((~timestamp_valid).sum())}


def robust_baseline(b):
    median = np.median(b.astype(np.float64), axis=0)
    sigma = 1.4826 * np.median(np.abs(b - median), axis=0)
    return median.astype(np.float32), float(np.linalg.norm(sigma))


def baseline_schedule(events, raw, segments, origin):
    t = raw["t_available_ns"]
    b = raw["b_uT"]
    valid = raw["sensor_valid"].all(axis=1)
    scheduled = []
    for event in events:
        v, te = event_value(event), int(event["t_perf_ns"])
        if event["event"] == "DYNAMIC_B0_APPLIED" and "b0_5x3_uT" in v:
            baseline = np.asarray(v["b0_5x3_uT"], dtype=np.float32).reshape(15)
            effective = max(te, int(v.get("completed_perf_ns", te)))
            if np.isfinite(baseline).all():
                scheduled.append({"effective_ns": effective, "b0_uT": baseline,
                                  "noise_uT": np.nan, "source": "logged_past_unloaded_b0",
                                  "window_start_ns": int(v.get("started_perf_ns", effective)),
                                  "window_end_ns": int(v.get("completed_perf_ns", effective))})
        if event["event"] == "PRE_ACTION_B0_WINDOW":
            end = te + round(float(v.get("capture_seconds", .35)) * 1e9)
            ids = (t >= te) & (t <= end) & valid
            if ids.sum() >= 5:
                base, noise = robust_baseline(b[ids])
                scheduled.append({"effective_ns": end, "b0_uT": base, "noise_uT": noise,
                                  "source": "logged_pre_action_window",
                                  "window_start_ns": te, "window_end_ns": end})
    # Recover first initial baseline only from an already reached, lifted hold.
    # This is a replay of an explicit unloaded calibration step, not an input label.
    rr = command_reference(t, segments, origin)
    safe = valid & (rr["contact_command"] == 0) & (rr["xyz_mm"][:, 2] <= -.2)
    safe &= (rr["hold_age_s"] >= .15) & ~rr["command_moving"]
    ids = np.flatnonzero(safe)
    for block in np.split(ids, np.flatnonzero(np.diff(ids) > 1) + 1):
        if len(block) < 5:
            continue
        start = int(t[block[0]])
        if t[block[-1]] - start < 350_000_000:
            continue
        end = start + 350_000_000
        if scheduled and min(x["effective_ns"] for x in scheduled) <= end:
            break
        take = block[t[block] <= end]
        base, noise = robust_baseline(b[take])
        scheduled.append({"effective_ns": end, "b0_uT": base, "noise_uT": noise,
                          "source": "initial_confirmed_lifted_hold",
                          "window_start_ns": start, "window_end_ns": end})
        break
    return sorted(scheduled, key=lambda x: x["effective_ns"])


def causal_resample(raw):
    avail = raw["t_available_ns"]
    start = ((int(avail[0]) + STEP_NS - 1) // STEP_NS) * STEP_NS
    t = np.arange(start, int(avail[-1]) + 1, STEP_NS, dtype=np.int64)
    idx = np.searchsorted(avail, t, side="right") - 1
    assert np.all(idx >= 0) and np.all(avail[idx] <= t)
    out = {"t_ns": t, "native_index": idx.astype(np.int32)}
    for key in ("t_group_ns", "t_available_ns", "source_row", "b_uT", "adc_raw", "sensor_valid", "sensor_soft_quality"):
        out[key] = raw[key][idx]
    out["observation_age_s"] = ((t - raw["t_group_ns"][idx]) / 1e9).astype(np.float32)
    out["availability_age_s"] = ((t - avail[idx]) / 1e9).astype(np.float32)
    return out


def causal_events(amplitude, valid, baseline_id, noise, reset):
    """Conservative magnetic pseudo-onset. Confirm on 3rd frame, never backdate.

    This is not an independently measured contact event. It never gates XYZ loss.
    A stored unknown noise gets a fixed engineering fallback (10 uT), not fitted
    on validation labels. Thresholds are configurable in a later training version.
    """
    n = len(amplitude)
    event_id = np.full(n, -1, np.int32)
    ref_index = np.full(n, -1, np.int32)
    on, above, below, counter, anchor = False, 0, 0, -1, -1
    for i in range(n):
        changed = i == 0 or baseline_id[i] != baseline_id[i - 1]
        if reset[i] or changed or not valid[i]:
            on, above, below, anchor = False, 0, 0, -1
            continue
        sigma = noise[i] if np.isfinite(noise[i]) else 10.
        high, low = max(30., 5 * sigma), max(15., 3 * sigma)
        if not on:
            above = above + 1 if amplitude[i] >= high else 0
            if above >= 3:
                on, counter, anchor, below = True, counter + 1, i, 0
        else:
            below = below + 1 if amplitude[i] < low else 0
            if below >= 4:
                on, above, anchor = False, 0, -1
        if on:
            event_id[i], ref_index[i] = counter, anchor
    return event_id, ref_index


def assemble(raw, segments, events, origin):
    out = causal_resample(raw)
    t, n = out["t_ns"], len(out["t_ns"])
    out.update(command_reference(t, segments, origin))
    baselines = baseline_schedule(events, raw, segments, origin)
    b0 = np.full((n, 15), np.nan, np.float32)
    noise = np.full(n, np.nan, np.float32)
    baseline_id = np.full(n, -1, np.int16)
    baseline_age = np.full(n, np.nan, np.float32)
    for i, base in enumerate(baselines):
        ids = t >= base["effective_ns"]
        b0[ids], noise[ids], baseline_id[ids] = base["b0_uT"], base["noise_uT"], i
        baseline_age[ids] = (t[ids] - base["effective_ns"]) / 1e9
    delta = out["b_uT"] - b0
    input_valid = out["sensor_valid"].all(axis=1) & np.isfinite(delta).all(axis=1)
    input_valid &= out["observation_age_s"] <= .25
    reset = np.zeros(n, bool)
    reset[0] = True
    reset[1:] = (~input_valid[:-1] & input_valid[1:]) | (baseline_id[1:] != baseline_id[:-1])
    reset |= ~input_valid
    stream = np.cumsum(reset, dtype=np.int32)
    amp = np.linalg.norm(delta.astype(np.float64), axis=1).astype(np.float32)
    eid, anchor = causal_events(amp, input_valid, baseline_id, noise, reset)
    xyz = out["xyz_mm"]
    label_valid = out["xyz_axis_known"].all(axis=1) & (out["contact_command"] == 1)
    relative = np.full((n, 3), np.nan, np.float32)
    rvalid = (anchor >= 0) & label_valid
    # A magnetic onset can precede nominal command contact (e.g. z=-0.18 mm).
    # Its POSITION reference is still known. Requiring positive depth there would
    # silently discard almost all historical straight/compound relative labels.
    rvalid &= out["xyz_axis_known"].all(axis=1)[np.maximum(anchor, 0)]
    relative[rvalid] = xyz[rvalid] - xyz[anchor[rvalid]]
    relative_valid = rvalid & input_valid
    out.update({"delta_b_uT": delta, "b0_id": baseline_id, "b0_age_s": baseline_age,
                "b0_noise_uT": noise, "input_valid": input_valid, "stream_reset": reset,
                "stream_id": stream, "xyz_label_valid": label_valid,
                "xyz_train_mask": label_valid & input_valid,
                "event_id": eid, "event_reference_index": anchor,
                "relative_xyz_mm": relative, "relative_train_mask": relative_valid,
                "label_weight": np.where(out["command_boundary"], .5, 1.).astype(np.float32),
                "core_domain": label_valid & (np.abs(xyz[:, :2]) <= 10).all(axis=1) & (xyz[:, 2] >= .2 - 1e-5) & (xyz[:, 2] <= .8 + 1e-5)})
    out["event_consistency_mask"] = relative_valid & label_valid[np.maximum(anchor, 0)]
    # Nominal command-contact bouts are a SECOND target definition, never inputs
    # and never substitutes for a sensor-observable inference-time event anchor.
    command_eid = np.full(n, -1, np.int32)
    command_anchor = np.full(n, -1, np.int32)
    cevent, cref = -1, -1
    for i in range(n):
        if out["contact_command"][i] == 0 or not out["xyz_axis_known"][i].all():
            cref = -1
        elif out["contact_command"][i] == 1 and cref < 0:
            cevent, cref = cevent + 1, i
        if cref >= 0:
            command_eid[i], command_anchor[i] = cevent, cref
    command_relative = np.full((n, 3), np.nan, np.float32)
    cmdmask = (command_anchor >= 0) & label_valid
    command_relative[cmdmask] = xyz[cmdmask] - xyz[command_anchor[cmdmask]]
    out.update({"command_event_id": command_eid, "command_event_reference_index": command_anchor,
                "command_relative_xyz_mm": command_relative,
                "command_relative_train_mask": cmdmask & input_valid})
    lag_valid = []
    for frames in (1, 4, 10, 20):
        lv = np.zeros(n, bool)
        lv[frames:] = input_valid[frames:] & input_valid[:-frames] & (stream[frames:] == stream[:-frames])
        inc = np.full((n, 3), np.nan, np.float32)
        mask = lv.copy()
        mask[frames:] &= label_valid[frames:] & label_valid[:-frames] & (eid[frames:] == eid[:-frames]) & (eid[frames:] >= 0)
        inc[frames:] = xyz[frames:] - xyz[:-frames]
        out[f"increment_{frames}_mm"] = inc
        out[f"increment_{frames}_mask"] = mask
        if frames in (4, 20):
            lag_valid.append(lv)
    out["lag_valid"] = np.column_stack(lag_valid)
    # Separate static view: one robust observation per reached position hold.
    contacts = []
    for sid in np.unique(out["command_segment_id"]):
        if sid < 0:
            continue
        mask = out["xyz_train_mask"] & (out["command_segment_id"] == sid)
        mask &= ~out["command_moving"] & (out["hold_age_s"] >= .15)
        ids = np.flatnonzero(mask)
        if len(ids) >= 4:
            contacts.append((int(ids[0]), int(ids[-1]), int(sid), len(ids),
                             np.median(delta[ids], axis=0), np.median(xyz[ids], axis=0)))
    out["static_hold_bounds"] = np.asarray([x[:4] for x in contacts], np.int32).reshape(-1, 4)
    out["static_hold_delta_b_uT"] = np.asarray([x[4] for x in contacts], np.float32).reshape(-1, 15)
    out["static_hold_xyz_mm"] = np.asarray([x[5] for x in contacts], np.float32).reshape(-1, 3)
    direct_phases = {"preload_indent", "press_indent", "grid_indent", "grid_reference_indent", "load"}
    out["static_hold_direct_press"] = np.asarray([segments[x[2]]["phase"] in direct_phases for x in contacts], bool)
    for k in ("b0_uT",):
        out["baseline_" + k] = np.asarray([b[k] for b in baselines], np.float32).reshape(-1, 15)
    out["baseline_effective_ns"] = np.asarray([b["effective_ns"] for b in baselines], np.int64)
    out["baseline_window_end_ns"] = np.asarray([b["window_end_ns"] for b in baselines], np.int64)
    return out, baselines


def trajectory_signature(rows, events, origin):
    # The signature uses semantic targets/feed/holds, not actual noisy timestamps.
    # Repeated executions reduce to the first complete repeat. The same primitive
    # with different repeats therefore cannot cross the development partition.
    begins = [int(e["t_perf_ns"]) for e in events if e["event"] == "ACTION_REPEAT_BEGIN"]
    ends = [int(e["t_perf_ns"]) for e in events if e["event"] == "ACTION_REPEAT_END"]
    if begins and ends:
        start = min(begins)
        stop = min((x for x in ends if x > start), default=2**63 - 1)
        rows = [r for r in rows if start <= int(r["t_perf_ns"]) <= stop]
    canonical = []
    for row in rows:
        cmd = row["command"].upper().strip()
        if re.match(r"^G0?1(?:\s|$)", cmd):
            coords = [number(row.get(k)) for k in ("x_mm", "y_mm", "z_mm")]
            relative = [round((v - origin[i]) * (-1 if i == 2 else 1), 3) if np.isfinite(v) else None for i, v in enumerate(coords)]
            canonical.append(["move", *relative, round(number(row.get("feed_mm_min"), 0), 2)])
        elif cmd == "HOST_DWELL" and "repeat_interval" not in row["phase"]:
            match = re.search(r"([\d.]+)s", row.get("response", ""))
            canonical.append(["hold", round(float(match.group(1)), 2) if match else row["response"]])
    return digest(canonical)


def assign_splits(records):
    by_family = defaultdict(lambda: defaultdict(list))
    for row in records:
        by_family[row["family"]][row["trajectory_group"]].append(row)
    # A trajectory signature can span family names. Decide globally and reconcile.
    assignment = {}
    for fam in sorted(by_family):
        groups = sorted(by_family[fam], key=lambda k: digest([20260921, k]))
        nval = max(1, round(.2 * len(groups))) if len(groups) > 1 else 0
        for i, group in enumerate(groups):
            assignment.setdefault(group, "validation" if i < nval else "train")
    for row in records:
        if row["session"] == "S20260918_A" and int(row["trial_id"]) in (149, 150, 153):
            assignment[row["trajectory_group"]] = "train"
    # Previously published static test is a historical regression check, not a
    # new/unseen final test. Reserve its ENTIRE signature to avoid identical repeats.
    for row in records:
        if row["session"] == "S20260918_A" and int(row["trial_id"]) == 154:
            assignment[row["trajectory_group"]] = "paper_static_check"
    for family, groups in by_family.items():
        if not any(assignment[g] == "train" for g in groups):
            available = [g for g in groups if assignment[g] != "paper_static_check"]
            if available:
                assignment[sorted(available)[-1]] = "train"
    for row in records:
        row["split"] = assignment[row["trajectory_group"]]


def primitive_stats(arrays):
    mask = arrays["input_valid"]
    x = arrays["delta_b_uT"][mask].astype(np.float64)
    return {"count": len(x), "sum": x.sum(axis=0).tolist(), "sum_sq": (x * x).sum(axis=0).tolist()}


def summarize(records, excluded):
    summary = {"included_trials": len(records), "excluded_trials": len(excluded),
               "splits": {}, "families": {}, "domains": {}, "final_test_trials": 0}
    for field, key in (("split", "splits"), ("family", "families"), ("data_tier", "domains")):
        for value in sorted({r[field] for r in records}):
            group = [r for r in records if r[field] == value]
            summary[key][value] = {"trials": len(group), **{name: sum(r["counts"][name] for r in group) for name in
                 ("native_frames", "frames_40hz", "xyz_labeled_frames", "xyz_train_frames", "relative_train_frames", "static_holds")}}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--inventory", type=Path, default=Path("docs/trajectory_source_inventory_20260921.json"))
    parser.add_argument("--output", type=Path, default=Path("datasets/MagTouch_CPA_Trajectory_v1_20260921"))
    parser.add_argument("--limit", type=int, default=0, help="smoke build only; selects diverse families/sessions")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rebuild-generated", action="store_true", help="regenerate ONLY this builder's marked unpublished output; never removes source data")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and not (args.resume or args.rebuild_generated):
        raise SystemExit("Output exists. Use a NEW directory or --resume for this builder's own output.")
    out.mkdir(parents=True, exist_ok=True)
    for name in ("sequences", "native", "provenance", "splits", "code"):
        (out / name).mkdir(exist_ok=True)
    builder_hash = sha(__file__)
    marker = out / "build_state.json"
    if marker.exists():
        previous = json.loads(marker.read_text("utf-8"))
        if previous.get("schema") != SCHEMA:
            raise SystemExit("Not a dataset directory owned by this builder")
        if previous["builder_sha256"] != builder_hash and not args.rebuild_generated:
            raise SystemExit("Builder changed: use a NEW output directory, do not silently mix versions.")
    elif args.rebuild_generated:
        raise SystemExit("Rebuild requires the builder's existing ownership marker")
    if args.rebuild_generated and args.resume:
        raise SystemExit("Choose either rebuild or resume")
    save_json(marker, {"schema": SCHEMA, "status": "building", "builder_sha256": builder_hash})
    if (out / "manifest.json").exists():
        previous_manifest = json.loads((out / "manifest.json").read_text("utf-8"))
        previous_manifest["status"] = "building"
        save_json(out / "manifest.json", previous_manifest)
    inv = json.loads(args.inventory.read_text("utf-8"))
    all_cpa = [r for r in inv["records"] if r["layer_hint"] == "CPA"]
    candidates = [r for r in all_cpa if r["candidate_complete"]]
    excluded = [{"source_id": r["source_id"], "reason": "not_completed_or_missing_raw_triplet", "status": r["status"]} for r in all_cpa if not r["candidate_complete"]]
    if args.limit:
        diverse = sorted(candidates, key=lambda r: (r["session"], r["family"], r["source_id"]), reverse=True)
        selected, seen = [], set()
        for row in diverse:
            key = (row["session"], row["family"])
            if key not in seen:
                selected.append(row)
                seen.add(key)
        candidates = selected[:args.limit]
    records, magnetic_hashes = [], {}
    t0 = time.monotonic()
    for index, source in enumerate(candidates):
        folder = args.data_root / source["source_id"]
        sid = digest(source["source_id"])[:20]
        detail_path = out / "provenance" / f"{sid}.json"
        try:
            hashes = {name: sha(folder / name) for name in FILES}
            if hashes["magnetic_raw.csv"] in magnetic_hashes:
                excluded.append({"source_id": source["source_id"], "reason": "duplicate_raw_magnetic_file", "duplicate_of": magnetic_hashes[hashes["magnetic_raw.csv"]]})
                continue
            magnetic_hashes[hashes["magnetic_raw.csv"]] = source["source_id"]
            if args.resume and detail_path.exists() and (out / "sequences" / f"{sid}.npz").exists():
                saved = json.loads(detail_path.read_text("utf-8"))
                if saved["source_sha256"] != hashes:
                    raise ValueError("source_changed_during_resume")
                row = saved["index_record"]
            else:
                meta = json.loads((folder / "meta.json").read_text("utf-8-sig"))
                origin = reference_origin(meta)
                events, stage = read_rows(folder / "events.csv"), read_rows(folder / "stage_commands.csv")
                segments = command_segments(stage)
                raw = load_magnetic(folder / "magnetic_raw.csv")
                arrays, baselines = assemble(raw, segments, events, origin)
                mask = arrays["xyz_train_mask"]
                xyz = arrays["xyz_mm"][mask]
                row = {"id": sid, "source_id": source["source_id"], "source_sha256": hashes,
                       "session": source["session"], "trial_id": source["trial_id"],
                       "family": source["family"], "layer_hint_source": source["layer_hint_source"],
                       "data_tier": "primary_recent" if source["session"] in PRIMARY else "auxiliary_historical",
                       "assembly_id": None, "physical_assembly_verified": False,
                       "domain_group": digest([source["session"], str(folder.parent.name), meta.get("acquisition", {}).get("magnetic_layer_profile"), meta.get("acquisition", {}).get("mlx90393")])[:16],
                       "trajectory_group": trajectory_signature(stage, events, origin),
                       "sequence": f"sequences/{sid}.npz", "native": f"native/{sid}.npz",
                       "provenance": f"provenance/{sid}.json", "split": "unassigned",
                       "reference_origin_printer_mm": origin.tolist(),
                       "contact_z_geometry_conflict": bool(source["contact_z_conflict"]),
                       "counts": {"native_frames": len(raw["b_uT"]), "frames_40hz": len(arrays["t_ns"]),
                           "invalid_timestamp_rows": raw["invalid_time_rows"],
                           "xyz_labeled_frames": int(arrays["xyz_label_valid"].sum()),
                           "xyz_train_frames": int(mask.sum()),
                           "relative_train_frames": int(arrays["relative_train_mask"].sum()),
                           "command_relative_train_frames": int(arrays["command_relative_train_mask"].sum()),
                           "static_holds": len(arrays["static_hold_xyz_mm"]),
                           "direct_press_holds": int(arrays["static_hold_direct_press"].sum()),
                           "baselines": len(baselines),
                           "deep_frames_over_0p8": int((xyz[:, 2] > .80001).sum()),
                           "xy_motion_frames": int((mask & arrays["command_moving"]).sum())},
                       "xyz_min_mm": xyz.min(axis=0).tolist() if len(xyz) else None,
                       "xyz_max_mm": xyz.max(axis=0).tolist() if len(xyz) else None,
                       "normalization_sufficient_stats": primitive_stats(arrays),
                       "warnings": ["command_reference_not_independent_metrology", "physical_assembly_identity_unverified"]}
                if not baselines:
                    row["warnings"].append("no_usable_causal_b0")
                if not mask.any() and source["family"] != "unloaded":
                    row["warnings"].append("no_joint_xyz_training_frames")
                np.savez_compressed(out / row["sequence"], **arrays)
                np.savez_compressed(out / row["native"], **{k: v for k, v in raw.items() if isinstance(v, np.ndarray)})
                # Preserve logs/meta inside the bundle; source raw CSV remains untouched.
                for name in ("meta.json", "events.csv", "stage_commands.csv"):
                    shutil.copy2(folder / name, out / "provenance" / f"{sid}_{name}")
                baseline_meta = [{k: (v.tolist() if isinstance(v, np.ndarray) else (None if isinstance(v, float) and not np.isfinite(v) else v)) for k, v in b.items()} for b in baselines]
                save_json(detail_path, {"schema": SCHEMA, "source_sha256": hashes,
                          "index_record": row, "baseline_provenance": baseline_meta})
            records.append(row)
        except (ValueError, KeyError, OSError, pd.errors.ParserError) as exc:
            excluded.append({"source_id": source["source_id"], "reason": str(exc), "exception": type(exc).__name__})
            print(f"EXCLUDED {source['source_id']}: {exc}", flush=True)
        if (index + 1) % 20 == 0 or index + 1 == len(candidates):
            print(f"{index+1}/{len(candidates)} included={len(records)} excluded={len(excluded)} elapsed={time.monotonic()-t0:.1f}s", flush=True)
    assign_splits(records)
    with (out / "index.jsonl").open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    save_json(out / "exclusions.json", excluded)
    for split in ("train", "validation", "paper_static_check"):
        save_json(out / "splits" / f"{split}.json", [r["id"] for r in records if r["split"] == split])
    save_json(out / "splits" / "final_generalization_test.json", {"status": "not_collected", "ids": [], "must_not_be_replaced_by_random_windows": True})
    stats_rows = [r["normalization_sufficient_stats"] for r in records if r["split"] == "train" and r["normalization_sufficient_stats"]["count"]]
    # Equal weight per Trial avoids long grid recordings dominating normalization.
    if stats_rows:
        means = np.array([np.array(s["sum"]) / s["count"] for s in stats_rows])
        seconds = np.array([np.array(s["sum_sq"]) / s["count"] for s in stats_rows])
        mean = means.mean(axis=0)
        std = np.sqrt(np.maximum(seconds.mean(axis=0) - mean * mean, 1.))
        save_json(out / "normalization.json", {"fit_split": "train", "weighting": "equal_per_trial_valid_input_frames", "fit_trials": len(stats_rows), "delta_b_mean_uT": mean.tolist(), "delta_b_std_uT": std.tolist()})
    summary = summarize(records, excluded)
    save_json(out / "summary.json", summary)
    for filename in ("build_trajectory_dataset_v1.py", "trajectory_dataset_v1.py", "test_trajectory_dataset_v1.py", "verify_trajectory_dataset_v1.py"):
        p = Path(__file__).with_name(filename)
        if p.exists():
            shutil.copy2(p, out / "code" / filename)
    manifest = {"schema": SCHEMA, "status": "built_pending_verification", "smoke_build": bool(args.limit),
                "source_selection": "CPA hint; completed raw triplets; separate from old R15/R19 exports", "source_candidate_trials": len(candidates),
                "grid_hz": HZ, "coordinate_frame": "printer_centered_indentation_v1",
                "labels": "commanded XYZ=[Xp-Xc,Yp-Yc,Zc-Zp] in mm; not actual encoder measurement",
                "causality": "complete magnetic group available by t_return/max sensor read; ZOH only",
                "event_reference": "magnetic threshold, confirmed now, never backdated; can precede nominal command contact; pseudo-label auxiliary only",
                "model_input_forbidden": ["xyz_mm", "relative_xyz_mm", "command_relative_xyz_mm", "command_event_id", "command_event_reference_index", "contact_command", "command_moving", "command_segment_id", "hold_age_s", "family", "session", "trial_id", "trajectory_group"],
                "physical_assembly_verified": False, "new_independent_final_test_collected": False,
                "source_inventory_sha256": sha(args.inventory), "builder_sha256": builder_hash,
                "summary": summary}
    save_json(out / "manifest.json", manifest)
    save_json(marker, {"schema": SCHEMA, "status": "built_pending_verification", "builder_sha256": builder_hash})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
