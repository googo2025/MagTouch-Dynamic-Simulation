#!/usr/bin/env python3
"""Draw all R208 saved-online trajectories and per-Trial XY translation audits.

Pure-SVG renderer: no trajectory fitting, no model modification, no hardware I/O.
Centered views are retrospective shape diagnostics, never online calibration.
"""
from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

import numpy as np

from training.audit_r133_online_20260928 import make_atlas
from training.audit_r190_waypoint_repeat_strata import repeat_windows
from training.build_trajectory_dataset_v1 import read_rows
from training.retrain_r133_coverage_xy import load_new


COL = {"source": "#2166ac", "target": "#dc5c28"}
NAME = {"source": "Source device S", "target": "Target device T"}
GEOM = {"space": "Expanded space + depth", "history": "History return"}


class SVG:
    def __init__(self, width, height):
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                      '<rect width="100%" height="100%" fill="white"/>',
                      '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#263445}'
                      '.title{font-size:22px;font-weight:bold}.panel{font-size:15px;font-weight:bold}'
                      '.sub{font-size:14px}.axis{font-size:12px}.note{font-size:12px;fill:#586777}</style>']

    def line(self, a, b, c, d, color="#c8d2dc", width=1, dash=""):
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<line x1="{a:.2f}" y1="{b:.2f}" x2="{c:.2f}" y2="{d:.2f}" stroke="{color}" stroke-width="{width}"{extra}/>')

    def rect(self, x, y, w, h, fill="none", stroke="none", radius=0):
        self.parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" rx="{radius}" fill="{fill}" stroke="{stroke}"/>')

    def text(self, x, y, value, cls="axis", anchor="start"):
        self.parts.append(f'<text x="{x:.2f}" y="{y:.2f}" class="{cls}" text-anchor="{anchor}">{html.escape(str(value))}</text>')

    def path(self, points, color, opacity=1, dash=""):
        points = np.asarray(points)
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) < 2:
            return
        # Display-only decimation. Every eligible frame contributes to audit metrics.
        points = points[::max(1, math.ceil(len(points)/3500))]
        xy = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<polyline points="{xy}" fill="none" stroke="{color}" stroke-width="1.45" opacity="{opacity}" stroke-linejoin="round"{extra}/>')

    def marker(self, x, y, color, radius=4):
        self.line(x-radius, y-radius, x+radius, y+radius, color, 1.4)
        self.line(x-radius, y+radius, x+radius, y-radius, color, 1.4)

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join([*self.parts, "</svg>"]), encoding="utf-8")


def load_rows(audit, detail):
    atlas, basis, strength = make_atlas()
    out = {}
    for device in ("source", "target"):
        for geometry in ("space", "history"):
            key = f"{device}_{geometry}"
            run = audit["runs"][key]
            diagnostics = {r["trial"]: r for r in detail["runs"][key]}
            for trial in run["trials"]:
                path = Path(run["plan_run"]) / trial["trial"]
                data = load_new(path, atlas, basis, strength, min_core_frames=0)
                windows = repeat_windows(read_rows(path / "events.csv"), expected=2)
                masks = [(data["time"] >= start) & (data["time"] <= end)
                         for _, start, end in windows]
                observed = np.linalg.norm(data["live"][data["core"], :2] - data["truth"][data["core"], :2], axis=1).mean()
                if not np.isclose(observed, trial["saved_online_xy_raw_mm"], atol=1e-6):
                    raise ValueError(f"saved online score mismatch: {path}")
                out[(device, geometry, trial["condition"].lower())] = dict(data=data, windows=windows, masks=masks,
                                                                           audit=trial, diag=diagnostics[trial["trial"]])
    if len(out) != 8:
        raise ValueError(f"Expected eight contact Trials, got {len(out)}")
    return out


def backdrop(svg, x, y, w, h, xlow, xhigh, ylow, yhigh, xname, yname):
    svg.rect(x-12, y-48, w+30, h+88, "#f6f8fb", "#d6dfe8", 9)
    svg.rect(x, y, w, h, "white", "#aebbc9")
    xp = lambda v: x + w*(v-xlow)/(xhigh-xlow)
    yp = lambda v: y + h*(yhigh-v)/(yhigh-ylow)
    for f in np.linspace(0, 1, 5):
        xv, yv = xlow+(xhigh-xlow)*f, ylow+(yhigh-ylow)*f
        svg.line(xp(xv), y, xp(xv), y+h, "#e5eaef", .8)
        svg.line(x, yp(yv), x+w, yp(yv), "#e5eaef", .8)
        svg.text(xp(xv), y+h+18, f"{xv:.1f}", "axis", "middle")
        svg.text(x-8, yp(yv)+4, f"{yv:.1f}", "axis", "end")
    if xlow <= 0 <= xhigh:
        svg.line(xp(0), y, xp(0), y+h, "#9daab8", 1)
    if ylow <= 0 <= yhigh:
        svg.line(x, yp(0), x+w, yp(0), "#9daab8", 1)
    svg.text(x+w/2, y+h+36, xname, "axis", "middle")
    svg.text(x-8, y-10, yname, "axis")
    return xp, yp


def plot_xy(rows, geometry, centered, folder):
    svg = SVG(1120, 1050)
    mode = "post-hoc translation removed (shape diagnosis only)" if centered else "raw saved online output (no alignment)"
    svg.text(47, 42, f"R208 | {GEOM[geometry]} | XY trajectories", "title")
    svg.text(47, 68, mode, "sub")
    svg.line(47, 91, 94, 91, "#2e3947", 1.5, "5 4")
    svg.text(104, 95, "stage command", "axis")
    svg.line(234, 91, 281, 91, COL["source"], 1.6)
    svg.text(291, 95, "source online", "axis")
    svg.line(401, 91, 448, 91, COL["target"], 1.6)
    svg.text(458, 95, "target online; two repeats in each panel", "axis")
    lim = 6.0
    for device in ("source", "target"):
        for speed in ("slow", "fast"):
            row = rows[(device, geometry, speed)]
            data = row["data"]
            shift = np.asarray(row["diag"]["xy_mean_error_vector_mm"]) if centered else 0
            lim = max(lim, np.nanmax(np.abs(data["live"][data["core"], :2]-shift)),
                      np.nanmax(np.abs(data["truth"][data["core_ref"], :2])))
    lim = float(math.ceil(lim+.6))
    for di, device in enumerate(("source", "target")):
        for si, speed in enumerate(("slow", "fast")):
            row = rows[(device, geometry, speed)]
            data = row["data"]
            shift = np.asarray(row["diag"]["xy_mean_error_vector_mm"]) if centered else 0
            x, y = 92+si*525, 165+di*435
            xp, yp = backdrop(svg, x, y, 352, 352, -lim, lim, -lim, lim, "x (mm)", "y (mm)")
            value = row["diag"]["xy_translation_removed_mean_mm"] if centered else row["diag"]["xy_error_mean_mm"]
            svg.text(x, y-24, f"{NAME[device]} | {speed.title()} | XY {value:.3f} mm", "panel")
            for k, mask in enumerate(row["masks"]):
                ref = data["truth"][mask & data["core_ref"], :2]
                pred = data["live"][mask & data["core"], :2] - shift
                svg.path(np.column_stack((xp(ref[:, 0]), yp(ref[:, 1]))), "#2e3947", .85 if k == 0 else .45, "5 4")
                svg.path(np.column_stack((xp(pred[:, 0]), yp(pred[:, 1]))), COL[device], .96 if k == 0 else .55)
            if geometry == "history":
                for hold in row["diag"]["return_center"]["holds"]:
                    if "prediction_mm" in hold:
                        p = np.asarray(hold["prediction_mm"][:2])-shift
                        svg.marker(xp(p[0]), yp(p[1]), COL[device], 3)
            if geometry == "space" and device == "target" and speed == "fast" and not centered:
                worst = row["diag"]["worst_xy_frames"][0]["predicted_xyz_mm"]
                px, py = xp(worst[0]), yp(worst[1])
                svg.rect(px-6, py-6, 12, 12, "none", "#a50f15")
                svg.text(px-8, py-11, "9.42 mm peak", "axis", "end")
    svg.text(47, 1017, "All four panels use identical x/y limits. Audit metrics use every eligible frame; display path is size-limited.", "note")
    if centered:
        svg.text(47, 1038, "Translation fitted from that Trial's command labels: not deployable calibration and not the saved online score.", "note")
    svg.save(folder / f"r208_{geometry}_xy_{'centered' if centered else 'raw'}.svg")


def plot_z(rows, geometry, folder):
    svg = SVG(1200, 840)
    svg.text(47, 42, f"R208 | {GEOM[geometry]} | Z trajectories", "title")
    svg.text(47, 68, "Dashed stage command vs saved online Z; two full repeats overlaid", "sub")
    for di, device in enumerate(("source", "target")):
        for si, speed in enumerate(("slow", "fast")):
            row = rows[(device, geometry, speed)]
            data = row["data"]
            x, y = 72+si*560, 146+di*338
            last = max((end-start)/1e9 for _, start, end in row["windows"])
            xp, yp = backdrop(svg, x, y, 475, 222, 0, last, -.1, 3.05,
                              "time from repeat start (s)", "indentation z (mm)")
            svg.text(x, y-24, f"{NAME[device]} | {speed.title()} | Z MAE {row['diag']['z_mae_mm']:.3f} mm", "panel")
            for k, ((_, start, _), mask) in enumerate(zip(row["windows"], row["masks"])):
                ref, valid = mask & data["core_ref"], mask & data["core"]
                tref = (data["time"][ref]-start)/1e9
                tout = (data["time"][valid]-start)/1e9
                svg.path(np.column_stack((xp(tref), yp(data["truth"][ref, 2]))),
                         "#2e3947", .85 if k == 0 else .45, "5 4")
                svg.path(np.column_stack((xp(tout), yp(data["live"][valid, 2]))),
                         COL[device], .96 if k == 0 else .55)
    svg.text(47, 810, "Z is the static-map output. The causal neural network corrects XY only. No Z translation has been removed.", "note")
    svg.save(folder / f"r208_{geometry}_z.svg")


def plot_offset(rows, folder):
    order = [(d, g, s) for d in ("source", "target") for g in ("space", "history") for s in ("slow", "fast")]
    svg = SVG(1210, 670)
    svg.text(43, 42, "R208 | Is the trajectory error just a global translation?", "title")
    svg.text(43, 68, "Raw mean XY error vs per-Trial constant-bias removal; bias vector reported separately", "sub")
    svg.text(43, 113, "Mean XY error (mm)", "panel")
    svg.text(781, 113, "Mean bias vector (x, y), mm", "panel")
    maxerr = max(rows[key]["diag"]["xy_error_mean_mm"] for key in order)
    base, scale = 215, 468/maxerr
    for i, key in enumerate(order):
        device, geometry, speed = key
        diag = rows[key]["diag"]
        raw, center = diag["xy_error_mean_mm"], diag["xy_translation_removed_mean_mm"]
        bx, by = diag["xy_mean_error_vector_mm"]
        y = 150+i*57
        label = f"{'S' if device == 'source' else 'T'} | {'Space' if geometry == 'space' else 'Return'} | {speed.title()}"
        svg.text(43, y+6, label)
        svg.rect(base, y-11, raw*scale, 13, COL[device])
        svg.rect(base, y+5, center*scale, 13, "#9aa7b4")
        svg.text(base+raw*scale+7, y, f"{raw:.3f}")
        svg.text(base+center*scale+7, y+17, f"{center:.3f}")
        svg.text(783, y+6, f"({bx:+.3f}, {by:+.3f})")
        svg.line(43, y+31, 1152, y+31, "#e6ebf0", .8)
    svg.rect(44, 628, 20, 9, COL["source"])
    svg.text(70, 637, "raw (blue source / orange target)", "note")
    svg.rect(373, 628, 20, 9, "#9aa7b4")
    svg.text(400, 637, "after constant XY bias removal", "note")
    svg.text(44, 656, "Post-hoc correction uses command labels from the same Trial; it is a shape diagnosis, not an online score.", "note")
    svg.save(folder / "r208_xy_translation_audit.svg")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--detail", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    audit = json.loads(a.audit.read_text(encoding="utf-8"))
    detail = json.loads(a.detail.read_text(encoding="utf-8"))
    if audit.get("schema") != "MagTouch-R208-Frozen3DStressAudit-1" or detail.get("schema") != "MagTouch-R211-R208PostRunDetail-1":
        raise ValueError("wrong R208 audit inputs")
    rows = load_rows(audit, detail)
    for geometry in ("space", "history"):
        plot_xy(rows, geometry, False, a.output_dir)
        plot_xy(rows, geometry, True, a.output_dir)
        plot_z(rows, geometry, a.output_dir)
    plot_offset(rows, a.output_dir)
    print(f"Wrote seven SVG figures to {a.output_dir}")


if __name__ == "__main__":
    main()
