"""Read-only real-data bias diagnosis; never calibrate or alter model outputs.

Two degrees of freedom are shown separately: one translation for a whole batch,
and an independently fitted translation for each Trial. Both use evaluation
labels and are diagnostics, not deployable corrections or main performance.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from training.real_simulation_paper_r395 import (
    ROOT, OUT, read, dump, sha, command_segments, command_reference,
    read_rows, reference_origin, mask_for, comparison,
)

NEW_SOURCE = ROOT / "data/S20261003_A/Plan_20261003_000334_R391_EXEC_cedd52183703_1b7bb8701c02_fdadea2__mag_5x5-had-239"


def diagnostic(e):
    bias = e.mean(0)
    mse = float(np.mean(np.sum(e[:, :2] ** 2, axis=1)))
    bias_mse = float(np.sum(bias[:2] ** 2))
    centered = e - bias
    # Squared-error decomposition is exact; do not apply it to mean norm.
    np.testing.assert_allclose(mse, bias_mse + np.mean(np.sum(centered[:, :2] ** 2, axis=1)), atol=1e-12)
    return dict(frames=len(e), raw_xy_mm=float(np.linalg.norm(e[:, :2], axis=1).mean()),
                raw_xyz_mm=float(np.linalg.norm(e, axis=1).mean()),
                bias_x_mm=float(bias[0]), bias_y_mm=float(bias[1]), bias_z_mm=float(bias[2]),
                xy_bias_norm_mm=float(np.linalg.norm(bias[:2])),
                per_trial_translation_xy_mm=float(np.linalg.norm(centered[:, :2], axis=1).mean()),
                xy_mse_mm2=mse, xy_bias_mse_mm2=bias_mse,
                xy_bias_squared_error_fraction=bias_mse / mse if mse else None)


def main():
    protocol = read(OUT / "protocol.json")
    errors, detail = {}, []
    for item in protocol["raw_inputs"]:
        if item["mode"] != "waypoint_xyz":
            continue
        with np.load(OUT / "streams" / (item["id"] + ".npz")) as f:
            a = {k: f[k].copy() for k in f.files}
        mask = a["primary"] & a["R373_ok"] & a["VirtualXY1000_ok"]
        for model in ("R373", "VirtualXY1000"):
            e = a[model + "_xyz"][mask] - a["q"][mask]
            assert len(e) and np.isfinite(e).all()
            batch = item["device"] + "_paired_original"
            errors.setdefault((batch, model), []).append(e)
            detail.append(dict(batch=batch, model=model, id=item["id"], group=item["group"], **diagnostic(e)))

    progress = read(NEW_SOURCE / "plan_progress.json")
    assert progress["status"] == "COMPLETED" and progress["completed_trials"] == progress["planned_trials"] == 16
    source_files, source_trials = {}, []
    for trial in sorted(NEW_SOURCE.glob("Trial_*")):
        comparison._assert_not_excluded(trial)
        m, item = read(trial / "meta.json"), read(trial / "batch_plan_item.json")
        assert m["status"] == "completed"
        live = m["live_prediction_recording"]
        assert live["model_family"] == "VirtualExpandedXY1000_R385_R373Z"
        assert live["algorithm_metadata"]["xy_mlp_artifact_sha256"] == sha(ROOT / "analysis/r379_full_virtual_expansion_20261002/arm_1000/best_model.npz")
        files = {f: sha(trial / f) for f in ("meta.json", "magnetic_raw.csv", "stage_commands.csv", "events.csv", "live_predictions.csv")}
        source_files[trial.relative_to(ROOT).as_posix()] = files
        df = pd.read_csv(trial / "live_predictions.csv")
        t = df.t_prediction_ns.to_numpy(dtype=np.int64)
        assert (np.diff(t) > 0).all()
        p = df[["visual_x_mm", "visual_y_mm", "visual_z_mm"]].to_numpy(float)
        ok = df.interaction_valid.eq(True).to_numpy() & np.isfinite(p).all(1)
        seg = command_segments(read_rows(trial / "stage_commands.csv"))
        ref = command_reference(t, seg, reference_origin(m))
        q = ref["xyz_mm"]
        phase = np.array([seg[j]["phase"] if j >= 0 else "before_stage" for j in ref["command_segment_id"]])
        mode = m["trial_parameters"]["command_mode"]
        primary, _ = mask_for(mode, phase, ref)
        valid = primary & ok & np.isfinite(q).all(1)
        source_trials.append(dict(trial=trial.name, group=item["group"], mode=mode, status=m["status"],
                                 online_rows=len(df), reference_frames=int(primary.sum()),
                                 valid_frames=int(valid.sum()), model=live["model_family"]))
        if mode != "waypoint_xyz":
            continue
        e = p[valid] - q[valid]
        assert len(e)
        key = ("SOURCE_latest_augmented_online", "VirtualXY1000")
        errors.setdefault(key, []).append(e)
        detail.append(dict(batch=key[0], model=key[1], id=trial.name, group=item["group"], **diagnostic(e)))

    summary = []
    for (batch, model), es in errors.items():
        selected = [r for r in detail if r["batch"] == batch and r["model"] == model]
        bias = np.mean([e.mean(0) for e in es], axis=0)  # Trial-equal, not frame pooled.
        r = dict(batch=batch, model=model, trials=len(es),
                 raw_xy_mm=float(np.mean([x["raw_xy_mm"] for x in selected])),
                 one_batch_translation_xy_mm=float(np.mean([np.linalg.norm((e-bias)[:, :2], axis=1).mean() for e in es])),
                 per_trial_translation_xy_mm=float(np.mean([x["per_trial_translation_xy_mm"] for x in selected])),
                 bias_x_mm=float(bias[0]), bias_y_mm=float(bias[1]), bias_z_mm=float(bias[2]),
                 xy_mse_mm2=float(np.mean([x["xy_mse_mm2"] for x in selected])),
                 xy_bias_mse_mm2=float(np.mean([x["xy_bias_mse_mm2"] for x in selected])),
                 frames=sum(len(e) for e in es))
        summary.append(r)
    paired = []
    for device in ("Yellow", "SOURCE"):
        rows = {r["model"]: r for r in summary if r["batch"] == device + "_paired_original"}
        old, new = rows["R373"], rows["VirtualXY1000"]
        delta_mse = new["xy_mse_mm2"] - old["xy_mse_mm2"]
        bias_delta = new["xy_bias_mse_mm2"] - old["xy_bias_mse_mm2"]
        d = pd.DataFrame([r for r in detail if r["batch"] == device + "_paired_original"])
        a = d[d.model.eq("R373")].set_index("id")
        b = d[d.model.eq("VirtualXY1000")].set_index("id")
        paired.append(dict(device=device, extra_xy_mse_mm2=delta_mse, extra_per_trial_bias_mse_mm2=bias_delta,
                           bias_explained_fraction_of_extra_mse=bias_delta/delta_mse,
                           residual_shape_mse_change_mm2=delta_mse-bias_delta,
                           raw_improved_trials=int((b.raw_xy_mm<a.raw_xy_mm).sum()),
                           per_trial_translation_improved_trials=int((b.per_trial_translation_xy_mm<a.per_trial_translation_xy_mm).sum())))
    for trial, files in source_files.items():
        assert all(sha(ROOT / trial / f) == h for f, h in files.items()), "Raw acquisition changed"
    report = dict(status="COMPLETE_DIAGNOSTIC", aggregation="equal means of 12 per-Trial scores",
                  summary=summary, paired=paired, details=detail,
                  same_frames_for_paired_models=True, coordinate_or_time_rotation_or_scaling=False,
                  correction_uses_evaluation_labels=True, deployable_calibration=False,
                  production_model_or_raw_data_changed=False, protocol_sha256=sha(OUT / "protocol.json"),
                  new_source=dict(path=NEW_SOURCE.relative_to(ROOT).as_posix(), progress=progress,
                                  trials=source_trials, raw_sha256=source_files,
                                  same_stream_R373_comparison_not_yet_run=True),
                  warning="一整批固定平移与逐Trial平移均为使用测试标签的事后诊断；不能替换原始误差或作为上线参数。最新SOURCE在线批次单列，不与上一批R373跨批次拼接成同流改进。")
    dump(OUT / "translation_diagnostic.json", report)
    pd.DataFrame(summary).to_csv(OUT / "translation_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(detail).to_csv(OUT / "translation_trials.csv", index=False, encoding="utf-8-sig")
    tables = "".join(pd.DataFrame(v).to_html(index=False, float_format=lambda f:f"{f:.4f}") for v in (summary, paired, detail))
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>真实轨迹的固定偏移核对</title>
<style>body{{font:16px/1.7 "Microsoft YaHei",sans-serif;margin:30px;color:#203346}}table{{border-collapse:collapse;font-size:13px;margin:24px 0}}th,td{{padding:8px;border:1px solid #dce5ec;white-space:nowrap}}th{{background:#eef4f8}}</style>
<h1>真实轨迹的固定偏移核对</h1><p>{html.escape(report['warning'])}</p>
<p>raw_xy_mm：原始XY平均误差。one_batch_translation_xy_mm：整批12条轨迹只减一个共同偏移。
per_trial_translation_xy_mm：每个Trial独立减均值偏移。误差和偏移均以mm计，平方误差以mm²计。
同流比较只用两个模型共同有效的相同移动帧；每Trial先计算，再等权汇总。</p>
<p>最新SOURCE在 {progress['updated_local']} 完成16/16个Trial，实际在线模型为VirtualExpandedXY1000_R385_R373Z。
共有6类新三维路径、两档速度、旧圆/纯Z及前后无载。此批单独报告原生在线结果。</p>
<div style="overflow:auto">{tables}</div><p><a href="translation_summary.csv">汇总</a> · <a href="translation_trials.csv">逐Trial</a> · <a href="translation_diagnostic.json">输入索引和核对结果</a> · <a href="index.html">完整模型回测报告</a></p></html>'''
    (OUT / "translation.html").write_text(page, encoding="utf-8")
    print(json.dumps(dict(summary=summary, paired=paired, latest_source_complete_trials=len(source_trials)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
