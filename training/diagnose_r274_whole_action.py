"""Supplementary whole-action and grid re-entry diagnosis, AFTER primary audit.

Preserves R274's narrow-window scores. The whole-action definition matches the
previous R273 note and is NOT a replacement metric selected for higher scores.
No refit, XYZ or physical-contact inference.
"""
from __future__ import annotations

import json
import numpy as np

from training.audit_r274_frozen_phase_expansion import OUT, blocks, save_json
from training.plot_r211_r208_trajectories import SVG


def whole_actions(a):
    t, q, response, phase, eligible = [a[k] for k in ("time_ns", "diagnostic_command_xyz_mm", "response", "phase", "eligible")]
    finite = np.isfinite(q).all(1)
    dz = np.r_[0., np.diff(q[:, 2])]
    continuous = np.r_[False, (np.diff(t) > 0) & (np.diff(t) <= 250_000_000)]
    result = []
    for sign, desired, name in ((1, 2, "loading"), (-1, 4, "withdrawal")):
        mask = eligible & finite & continuous & (sign*dz > 1e-4) & (q[:, 2] >= .02)
        for ids in blocks(mask, t):
            r = ids[response[ids]]
            weak = ids[response[ids] & (q[ids, 2] < .4)]
            hits = r[phase[r] == desired]
            result.append({"phase": name, "frames": len(ids), "response_frames": len(r), "weak_response_frames": len(weak),
                "has_weak_response": bool(len(weak)), "whole_action_hit": bool(len(hits)),
                "weak_lt0p4_hit": bool(np.any(phase[weak] == desired)),
                "first_match_command_z_mm": None if not len(hits) else float(q[hits[0], 2]),
                "delay_from_first_response_s": None if not len(hits) else (int(t[hits[0]])-int(t[r[0]]))/1e9,
                "start_time_ns": int(t[ids[0]]), "end_time_ns": int(t[ids[-1]]), "index_first": int(ids[0]), "index_last": int(ids[-1])})
    return result


def summary(items):
    result = {}
    for name in ("loading", "withdrawal"):
        events = [e for v in items for e in v["events"] if e["phase"] == name]
        responsive = [e for e in events if e["response_frames"]]
        weak = [e for e in events if e["has_weak_response"]]
        delays = [e["delay_from_first_response_s"] for e in weak if e["whole_action_hit"]]
        result[name] = {"all_commanded_runs": len(events), "responsive_runs": len(responsive),
            "responsive_whole_action_hits": sum(e["whole_action_hit"] for e in responsive),
            "runs_with_weak_lt0p4_response": len(weak), "whole_action_hits_in_this_subset": sum(e["whole_action_hit"] for e in weak),
            "weak_lt0p4_hits_in_this_subset": sum(e["weak_lt0p4_hit"] for e in weak),
            "delay_median_s": None if not delays else float(np.median(delays)),
            "delay_p95_s": None if not delays else float(np.percentile(delays, 95))}
    return result


def trace(item, arrays, target):
    """First missing main-grid loading action, deliberately not a best example."""
    events = [e for e in item["events"] if e["phase"] == "loading" and e["has_weak_response"] and not e["whole_action_hit"]]
    if not events:
        return False
    event = events[0]; t = arrays["time_ns"]
    start, end = event["start_time_ns"]-1_000_000_000, event["end_time_ns"]+2_000_000_000
    use = (t >= start) & (t <= end)
    ids = np.flatnonzero(use); tt = (t[ids]-start)/1e9; dur = (end-start)/1e9
    svg = SVG(1080, 615)
    svg.text(40, 35, f"Frozen model | {item['domain']} | first missed grid loading", "title")
    svg.text(40, 61, "Diagnostic command depth is not shallow contact truth. Original states, no alignment or refitting.", "sub")
    x, w = 98, 905
    xp = lambda v: x+w*v/dur
    for pane, key, title in ((0, "diagnostic_command_xyz_mm", "Command Z (mm)"), (1, "amplitude_uT", "Field norm (uT)")):
        y, h = 105+pane*155, 113
        value = arrays[key][ids, 2] if pane == 0 else arrays[key][ids]
        finite = np.isfinite(value)
        lo, hi = float(np.nanmin(value)), float(np.nanmax(value))
        hi = max(hi, lo+1e-3); pad = .06*(hi-lo); lo -= pad; hi += pad
        yp = lambda v: y+h*(hi-v)/(hi-lo)
        svg.rect(x, y, w, h, "#f8fafc", "#d6dfe8")
        for v in np.linspace(lo, hi, 4):
            svg.line(x, yp(v), x+w, yp(v), "#e0e6ec", .7); svg.text(x-7, yp(v)+4, f"{v:.2f}" if pane == 0 else f"{v:.0f}", "axis", "end")
        svg.text(x, y-10, title, "panel")
        for block in blocks(finite, t[ids]):
            svg.path(np.column_stack((xp(tt[block]), yp(value[block]))), "#626d7a" if pane == 0 else "#166d84")
    colors = ("#b5bdc8", "#eff3f7", "#d98b25", "#56a684", "#3f7eb6", "#8f73a9", "#136a76")
    for pane, key, title in ((0, "strength", "Response strength/context"), (1, "phase", "Action phase")):
        y = 425+pane*58
        svg.text(x, y-10, title, "panel")
        for i in ids:
            a = (int(t[i])-start)/1e9
            b = min(dur, (int(t[i+1])-start)/1e9) if i+1 < len(t) else dur
            if b > a:
                svg.rect(xp(a), y, max(.2, xp(b)-xp(a)), 24, colors[int(arrays[key][i])])
    for sec in np.linspace(0, dur, 7):
        svg.text(xp(sec), 548, f"{sec:.1f}", "axis", "middle")
    svg.text(x+w/2, 568, "Time (s)", "axis", "middle")
    svg.text(40, 597, "Gray unknown | white quiet | orange loading | green hold | blue withdrawal | purple residue | teal supported", "note")
    svg.save(target); return True


def main():
    file = OUT / "whole_action_diagnostics.json"
    if file.exists():
        raise FileExistsError(file)
    report = json.loads((OUT / "report.json").read_text(encoding="utf-8"))
    new = [v for v in report["trials"] if not v["original_36_record_scope"]]
    results = []; plotted = set()
    folder = OUT / "figures"; folder.mkdir(exist_ok=True)
    for item in new:
        with np.load(OUT / "streams" / f"{item['key']}.npz", allow_pickle=False) as f:
            a = {k: f[k].copy() for k in f.files}
        events = whole_actions(a)
        result = {"path": item["path"], "domain": item["domain"], "mode": item["mode"], "events": events}
        results.append(result)
        if item["domain"] in ("source_5mm", "target_5mm") and item["mode"] == "grid_points" and item["domain"] not in plotted:
            if trace(result, a, folder / f"{item['domain']}_first_grid_failure.svg"):
                plotted.add(item["domain"])
    by_domain = {d: summary([v for v in results if v["domain"] == d]) for d in sorted({v["domain"] for v in results})}
    by_mode = {d+":"+m: summary([v for v in results if v["domain"] == d and v["mode"] == m])
        for d, m in sorted({(v["domain"], v["mode"]) for v in results})}
    save_json(file, {"schema": "R274-SupplementaryWholeAction-1", "primary_metrics_unchanged": True,
        "reason": "Check consistency with prior R273 whole-action semantics AFTER expanded narrow-window evaluation; does not replace primary results.",
        "definition": "chronological commanded Z motion with z>=0.02; report whole action and subset with responsive nominal z<0.4. Neither is physical shallow-contact recall.",
        "by_domain": by_domain, "by_domain_mode": by_mode, "trials": results})
    print(json.dumps(by_domain), flush=True)


if __name__ == "__main__":
    main()
