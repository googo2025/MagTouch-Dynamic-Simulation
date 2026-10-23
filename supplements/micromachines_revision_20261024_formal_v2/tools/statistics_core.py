"""Exact extracted statistical/reference functions; see function_versions.json."""
from pathlib import Path
import csv,json,re
import numpy as np

def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def number(value, default=np.nan):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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


def mask_for(mode, phase, ref):
    known = np.isfinite(ref["xyz_mm"]).all(1)
    depth = ref["xyz_mm"][:, 2]
    moving = ref["command_moving"]
    if mode == "grid_points":
        primary = (phase == "grid_indent") & ~moving & (
            ref["hold_age_s"] >= .15) & (ref["hold_age_s"] <= .55)
        action = phase == "grid_indent"
    elif mode == "angle_scan":
        action = np.array([bool(re.fullmatch(r"angle_(?:return_)?\d+", p)) for p in phase])
        primary = action & moving
    elif mode == "circle_scan":
        action = np.array([bool(re.fullmatch(r"circle_\d+", p)) for p in phase])
        primary = action & moving
    elif mode == "z_path":
        action = np.array([bool(re.fullmatch(r"path_\d+", p)) for p in phase])
        primary = action & moving
    elif mode == "compound":
        action = phase == "xyz_compound"
        primary = action & moving
    elif mode == "waypoint_xyz":
        action = np.array([bool(re.fullmatch(r"waypoint_\d{3}", p)) for p in phase])
        primary = action & moving
    elif mode == "none":
        primary = action = np.zeros(len(phase), bool)
    else:
        raise ValueError(mode)
    return primary & known & (depth >= .4), action & known


def score(x,q,ok,mask,t):
    eligible = np.asarray(mask,bool)&np.isfinite(q).all(1)
    good = eligible&ok&np.isfinite(x).all(1)
    pairs=good[:-1]&good[1:]&(np.diff(t)>0)&(np.diff(t)<=250_000_000)
    result=dict(reference_frames=int(eligible.sum()),output_frames=int(good.sum()),
        coverage=float(good.sum()/eligible.sum()) if eligible.any() else None,
        adjacent_pairs=int(pairs.sum()))
    if good.any():
        e=x[good]-q[good];bias=e.mean(0)
        result.update(raw_xy_mean_mm=float(np.linalg.norm(e[:,:2],axis=1).mean()),
            centered_xy_diagnostic_mm=float(np.linalg.norm((e-bias)[:,:2],axis=1).mean()),
            raw_z_mae_mm=float(np.abs(e[:,2]).mean()),bias_xyz_mm=bias.tolist())
    if pairs.any():
        dx=np.diff(x,axis=0)[pairs];de=np.diff(x-q,axis=0)[pairs]
        xy=np.linalg.norm(de[:,:2],axis=1);z=np.abs(de[:,2])
        result.update(xy_residual_step_p95_mm=float(np.quantile(xy,.95)),
            xy_residual_step_rms_mm=float(np.sqrt(np.mean(xy**2))),
            xy_residual_jumps_over_0p2=int((xy>.2).sum()),
            xy_residual_jump_rate_percent=float(100*(xy>.2).mean()),
            xy_output_step_p95_mm=float(np.quantile(np.linalg.norm(dx[:,:2],axis=1),.95)),
            xy_output_jumps_over_0p2=int((np.linalg.norm(dx[:,:2],axis=1)>.2).sum()),
            z_residual_step_p95_mm=float(np.quantile(z,.95)),z_residual_jumps_over_0p2=int((z>.2).sum()))
    return result


def dwell_score(x,ok,mask,t,phase):
    """Each contiguous >=1 s command-dwell block independently centered, no filtering."""
    mask=mask&ok&np.isfinite(x).all(1)
    ids=np.flatnonzero(mask);blocks=[]
    if len(ids):
        cuts=np.flatnonzero((np.diff(ids)>1)|(np.diff(t[ids])>250_000_000)|(phase[ids[1:]]!=phase[ids[:-1]]))+1
        for ix in np.split(ids,cuts):
            if len(ix)<3 or (t[ix[-1]]-t[ix[0]])<1e9:continue
            v=x[ix];blocks.append(dict(frames=len(ix),duration_s=float((t[ix[-1]]-t[ix[0]])/1e9),
                xy_wobble_rms_mm=float(np.sqrt(np.mean(np.sum((v[:,:2]-v[:,:2].mean(0))**2,axis=1)))),
                z_std_mm=float(np.std(v[:,2]))))
    return dict(blocks=blocks,blocks_count=len(blocks),
        mean_xy_wobble_rms_mm=float(np.mean([b['xy_wobble_rms_mm'] for b in blocks])) if blocks else None)


def summarize(rows):
    keys=['coverage','raw_xy_mean_mm','centered_xy_diagnostic_mm','raw_z_mae_mm',
          'xy_residual_step_p95_mm','xy_residual_step_rms_mm','xy_residual_jump_rate_percent',
          'xy_output_step_p95_mm','z_residual_step_p95_mm']
    out=dict(trials=len(rows))
    for k in keys:
        values=[r[k] for r in rows if r.get(k) is not None]
        out[k]=float(np.mean(values)) if values else None
    n=sum(r.get('adjacent_pairs',0) for r in rows)
    j=sum(r.get('xy_residual_jumps_over_0p2',0) for r in rows)
    out.update(adjacent_pairs=n,xy_residual_jump_count=j,pooled_pair_jump_rate_percent=100*j/n if n else None,
               z_residual_jump_count=sum(r.get('z_residual_jumps_over_0p2',0) for r in rows))
    return out

