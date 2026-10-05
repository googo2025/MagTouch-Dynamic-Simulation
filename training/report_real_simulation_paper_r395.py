"""Audit full-prefix causal research inputs and publish the real-data report."""
from __future__ import annotations
import csv
import html
import json
import os
from pathlib import Path
import sys
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
# Reuse the project's already provisioned scientific plotting runtime.
sys.path.append(str(Path(__file__).resolve().parents[1] / "tmp/r360_math_packages"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from training.real_simulation_paper_r395 import (ROOT, OUT, MODEL_ROOT, PLANS, ARMS, read, dump, sha,
                                              representation, predict, metric, native_condition_masks)
sys.path.insert(0, str(ROOT / "code"))
from live_normalized_fit import vector
import native_online_replay_r387 as native


def full_prefix_features(device):
    """Include B0-to-Trial prefix, all previous Trials, and actual B0 resets.

    EMA is the paper's fixed continuous-time equation, with no fit to real data.
    Reconstructed fields MUST equal the independently executed original adapter.
    """
    base_report=read(OUT / "native" / device / "R373/completed.json")
    windows={int(r["t_ns"]): r for r in base_report["b0_windows"]}
    base=None; previous=None; times=[]; fields=[]; epochs=[]
    journal=OUT / "native" / device / "native_protocol_prefix.jsonl"
    for rec in native._read_journal(journal):
        if not native._applies(rec, "A"): continue
        if rec["op"] in ("begin_baseline", "request_confirmed_baseline", "confirmed_baseline"):
            base=None; previous=None
        elif rec["op"] == "group":
            t=int(rec["t_ns"])
            if previous is not None and (t<=previous or t-previous>1_000_000_000): base=None
            previous=t
            if t in windows: base=np.array(windows[t]["b0_uT"]); epoch=windows[t]["epoch"]
            v=vector({int(k):val for k,val in rec["data"].items()})
            valid=base is not None and v is not None
            times.append(t); fields.append(v-base if valid else np.full(15,np.nan)); epochs.append(epoch if valid else -1)
    t=np.array(times,np.int64); b=np.array(fields); epoch=np.array(epochs)
    assert (np.diff(t)>0).all()
    valid=np.isfinite(b).all(1)
    reset=np.r_[True, np.diff(epoch)!=0] | ~valid | np.r_[False,~valid[:-1]]
    result={}
    for arm in ARMS:
        with np.load(MODEL_ROOT / arm / "best_model.npz", allow_pickle=False) as f:
            params=[f[f"p{i}"].copy() for i in range(4)]; mean=f["mean"]; scale=f["scale"]; ys=f["yscale"]
        x=representation(np.where(valid[:,None],b,0), (t-t[0])/1e9, reset, arm)
        result[arm]=predict(params,(x-mean)/scale)*ys
        result[arm][~valid]=np.nan
    return t,b,result


def refine_research_inputs(protocol):
    df=pd.read_csv(OUT / "trial_metrics.csv"); cond=pd.read_csv(OUT / "condition_metrics.csv")
    df=df[~df.model.isin(ARMS)].copy(); cond=cond[~cond.model.isin(ARMS)].copy()
    rr=[]; cc=[]; checks=[]; forward=[]; pauses=[]
    from training.forward_complex_r363 import AxisForward
    gen=AxisForward.load(ROOT / "analysis/r363_complex_forward_20261001/models/generator.json")
    for device in PLANS:
        t,b,pred=full_prefix_features(device)
        for item in [x for x in protocol["raw_inputs"] if x["device"]==device]:
            file=OUT / "streams" / (item["id"]+".npz")
            with np.load(file) as f: a={k:f[k].copy() for k in f.files}
            ix=np.searchsorted(t,a["t_ns"])
            assert np.array_equal(t[ix],a["t_ns"])
            np.testing.assert_allclose(b[ix],a["b"],atol=0,rtol=0,equal_nan=True)
            checks.append(dict(id=item["id"], all_fields_equal_native_baseline=True, full_pre_trial_history_preserved=True,
                               times_exact=True, native_frames=len(ix)))
            q=a["q"]; rm=a["research_primary"]; kin=native_condition_masks(q,a["t_ns"],rm)
            for arm in ARMS:
                p=pred[arm][ix]; ok=np.isfinite(p).all(1)
                a[arm+"_xyz"]=p; a[arm+"_ok"]=ok
                rr.append(dict(id=item["id"],device=device,group=item["group"],mode=item["mode"],speed_mm_s=item["speed_mm_s"],
                    model=arm,scope="research_whole_action",**metric(p,q,ok,rm,a["t_ns"]),
                    safe_reference_frames=int(a["safe"].sum()),safe_output_frames=int((a["safe"] & ok).sum()),
                    safe_without_external_hint_frames=int((a["safe"] & ~a["hint"]).sum()),
                    safe_without_external_hint_outputs=int((a["safe"] & ~a["hint"] & ok).sum()),safety_gate_implemented=False))
                for name, mask in kin.items():
                    cc.append(dict(id=item["id"],device=device,group=item["group"],model=arm,condition=name,
                                   **metric(p,q,ok,rm & mask,a["t_ns"])))
            np.savez_compressed(file,**a)
            # Additional descriptive check only, never adapt the generator.
            reset=np.r_[True,np.diff(a["b0_epoch"])!=0]
            vb=gen.rollout(q,(a["t_ns"]-a["t_ns"][0])/1e9,reset)
            mask=rm & np.isfinite(vb).all(1)
            if mask.any():
                err=vb[mask]-a["b"][mask]
                forward.append(dict(id=item["id"],device=device,group=item["group"],frames=int(mask.sum()),
                    rmse_uT=float(np.sqrt(np.mean(err*err))), nrmse=float(np.sqrt(np.mean(err*err))/max(np.sqrt(np.mean(a["b"][mask]**2)),1e-6)),
                    channel_bias_uT=err.mean(0).tolist(), generator_changed=False))
            hold=rm & kin["dwell"]
            blocks=np.split(np.flatnonzero(hold),np.flatnonzero(np.diff(np.flatnonzero(hold))>1)+1)
            for block in blocks:
                if len(block)<3: continue
                duration=(a["t_ns"][block[-1]]-a["t_ns"][block[0]])/1e9
                if duration<.5: continue
                delta=a["b"][block]-a["b"][block[0]]
                pauses.append(dict(id=item["id"],device=device,group=item["group"],start_ns=int(a["t_ns"][block[0]]),
                    duration_s=float(duration),frames=len(block), field_change_from_hold_start_max_uT=float(np.linalg.norm(delta,axis=1).max()),
                    field_change_from_hold_start_last_uT=float(np.linalg.norm(delta[-1])),
                    stationary_command_span_mm=float(np.linalg.norm(q[block]-q[block[0]],axis=1).max()),
                    is_equilibrium_residual=False, confirms_16s_virtual_nonmonotonic_residual=False))
    df=pd.concat([df,pd.DataFrame(rr)],ignore_index=True)
    cond=pd.concat([cond,pd.DataFrame(cc)],ignore_index=True)
    df.to_csv(OUT / "trial_metrics.csv",index=False,encoding="utf-8-sig")
    cond.to_csv(OUT / "condition_metrics.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(forward).to_csv(OUT / "forward_field_diagnostics.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(pauses).to_csv(OUT / "real_contact_holds.csv",index=False,encoding="utf-8-sig")
    dump(OUT / "research_full_prefix_audit.json",dict(status="PASS",checks=checks,frames=sum(x["native_frames"] for x in checks),
         models_only_use_actual_field_and_past_EMA=True, pre_trial_history_preserved=True,real_rows_used_to_fit_or_select=0,
         normalization_from_virtual_training_only=True, protocol_sha256=sha(OUT / "protocol.json")))
    return df,cond,pauses,forward


def summarize(df,cond,pauses,forward,protocol):
    rows=[]
    for device in PLANS:
        for scope,subset in (("new_complex",df["mode"].eq("waypoint_xyz")),("old_circle",df["mode"].eq("circle_scan")),("old_z",df["mode"].eq("z_path"))):
            for model in ["R373","VirtualXY1000",*ARMS]:
                maskname="shared_full_chain" if model in ("R373","VirtualXY1000") else "research_whole_action"
                group=df[subset & df.device.eq(device) & df.model.eq(model) & df.scope.eq(maskname)]
                numeric=["xyz_mean_mm","xy_mean_mm","z_mae_mm","xyz_p95_mm","bias_x_mm","bias_y_mm","bias_z_mm",
                         "translation_removed_xyz_diagnostic_mm"]
                row=dict(device=device, subset=scope, model=model, scope=maskname,trials=len(group),
                         metric_aggregation="arithmetic mean of per-Trial scores, not pooled frames",
                         **{k:float(group[k].mean()) for k in numeric})
                own=df[subset & df.device.eq(device) & df.model.eq(model) & df.scope.eq("primary_moving" if model in ("R373","VirtualXY1000") else maskname)]
                row.update(own_mask_coverage=float(own.coverage.mean()), own_mask_xyz_mean_mm=float(own.xyz_mean_mm.mean()),
                           own_mask_xy_mean_mm=float(own.xy_mean_mm.mean()), own_mask_z_mae_mm=float(own.z_mae_mm.mean()),
                           total_output_frames=int(own.output_frames.sum()), total_reference_frames=int(own.reference_frames.sum()),
                           safe_frames=int(own.safe_reference_frames.sum()), safe_outputs=int(own.safe_output_frames.sum()),
                           z_jumps_over_0p2_count=int(own.z_jump_over_0p2_count.sum()))
                rows.append(row)
    summary=pd.DataFrame(rows)
    summary.to_csv(OUT / "summary.csv",index=False,encoding="utf-8-sig")
    pairs=[]
    for device in PLANS:
        for baseline,candidate,scope in (("R373","VirtualXY1000","shared_full_chain"),("current","short_history","research_whole_action"),("current","multiscale_history","research_whole_action")):
            sub=df[df.device.eq(device) & df["mode"].eq("waypoint_xyz") & df.scope.eq(scope)]
            left=sub[sub.model.eq(baseline)].set_index("id");right=sub[sub.model.eq(candidate)].set_index("id")
            assert list(left.index)==list(right.index) and len(left)==12
            for metricname in ("xy_mean_mm","xyz_mean_mm","z_mae_mm"):
                old=float(left[metricname].mean());new=float(right[metricname].mean())
                delta=right[metricname]-left[metricname]
                pairs.append(dict(device=device,baseline=baseline,candidate=candidate,scope=scope,metric=metricname,
                    old_mm=old,new_mm=new,relative_change_percent=100*(new/old-1), trials_improved=int((delta<0).sum()),trials=12))
    pd.DataFrame(pairs).to_csv(OUT / "paired_improvements.csv",index=False,encoding="utf-8-sig")
    cs=cond.merge(df[["id","model","mode"]].drop_duplicates(),on=["id","model"])
    cs=cs[cs["mode"].eq("waypoint_xyz") & cs.reference_frames.gt(0)]
    cm=cs.groupby(["device","model","condition"])[["xyz_mean_mm","xy_mean_mm","z_mae_mm","coverage"]].mean().reset_index()
    cm.to_csv(OUT / "condition_summary.csv",index=False,encoding="utf-8-sig")
    parity={d:{m:read(OUT / "native" / d / m / "completed.json")["parity"] for m in ("R373","VirtualXY1000")} for d in PLANS}
    holds_max=max((r["duration_s"] for r in pauses),default=0)
    mapping={
      "C":"用两器件24条真实新三维Trial的冻结基线/虚拟扩集整链同流比较检验新增运动经验的真实迁移；不替换既有64条虚拟结果。",
      "D":"新增实测含同步XYZ与短停驻，可检验覆盖与停驻时信号仍变化。576个虚拟16秒端点的非单调平衡残余假设不能由短停驻直接证明；本批最长名义接触停驻%.3fs。"%holds_max,
      "E":"瞬时输入作为对照，短历史与三尺度两新读出全部冻结后直接在同一真实磁流回测；标准化与权重不使用真实新记录更新。读出没有许可门，接触段误差不等于安全系统性能。",
      "F":"真实采集数据上的实测效果（离线回测）补齐原待填整链结果，标明实际时间/B0/控制与同模型在线输出逐点核验；不要求重新进行双模型在线采集。"}
    report=dict(status="COMPLETE", reporting_name="真实采集数据上的实测效果（离线回测）", complete_trials=32,
                new_complex_trials=24, geometry_clusters=6, speeds_mm_s=[.7,1.3], repeats_nested=2,
                two_batches=PLANS.keys(), summary=rows,paired_improvements=pairs,C_F_mapping=mapping,
                real_contact_hold_count=len(pauses),longest_real_contact_hold_seconds=holds_max,
                native_online_parity=parity,
                protocol_sha256=sha(OUT / "protocol.json"), reference="指令时间/EST路径，不冒充外部测得触头真实位移",
                posthoc_registration=False, normalization_or_weight_changes=False, target_selection=False,
                hardware_operated=False, production_weights_replaced=False,
                online_input_subset="原生日志实际推理输入；原始采样完整数据另存，不偷偷补入在线丢弃帧")
    report["two_batches"]=list(PLANS)
    dump(OUT / "report.json",report)
    return summary,pd.DataFrame(pairs),cm,report


def plots(protocol,summary,cm):
    figdir=OUT / "figures";figdir.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False})
    blue,orange="#2377b9","#df7134"
    files=[]
    for device in PLANS:
        # First prespecified geometry/speed, never selected by favorable error.
        item=next(r for r in protocol["raw_inputs"] if r["device"]==device and r["mode"]=="waypoint_xyz")
        with np.load(OUT / "streams" / (item["id"]+".npz")) as f:a={k:f[k].copy() for k in f.files}
        mask=a["whole_action"];q=a["q"].copy();q[~mask]=np.nan
        vals=np.vstack([q]+[np.where((mask & a[m+"_ok"])[:,None],a[m+"_xyz"],np.nan) for m in ("R373","VirtualXY1000")])
        bounds=np.array([np.nanmin(vals,axis=0),np.nanmax(vals,axis=0)])
        fig=plt.figure(figsize=(11,8),layout="constrained")
        for view,az in enumerate((-55,35)):
            for col,model in enumerate(("R373","VirtualXY1000")):
                ax=fig.add_subplot(2,2,view*2+col+1,projection="3d")
                p=a[model+"_xyz"].copy();p[~(mask & a[model+"_ok"])]=np.nan
                ax.plot(*q.T,"--",color="#616d7b",lw=1,label="Command reference")
                ax.plot(*p.T,color=blue if col==0 else orange,lw=1,label=model)
                ax.set(xlabel="X (mm)",ylabel="Y (mm)",zlabel="Depth (mm)",title=f"{device} | {model} | view {view+1}")
                ax.set_xlim(bounds[:,0]+[-.2,.2]);ax.set_ylim(bounds[:,1]+[-.2,.2]);ax.set_zlim(bounds[:,2]+[-.1,.1])
                ax.view_init(elev=27,azim=az);ax.set_box_aspect([4,4,2]);ax.legend(fontsize=7)
        fig.suptitle(f"{device}: first predeclared Bezier 3D path, 0.7 mm/s | original coordinates",fontsize=12)
        for ext in ("png","svg"): fig.savefig(figdir/f"{device}_native_four_views.{ext}",dpi=190)
        plt.close(fig);files.append(f"figures/{device}_native_four_views.png")
    fig,ax=plt.subplots(2,2,figsize=(11,7),layout="constrained")
    for row,device in enumerate(PLANS):
        s=summary[summary.device.eq(device)&summary.subset.eq("new_complex")]
        labels=["R373","VirtualXY1000"];v=s.set_index("model").loc[labels]
        ax[row,0].bar(labels,v.xy_mean_mm,color=[blue,orange]);ax[row,0].set(title=f"{device}: full-chain paired XY",ylabel="Mean error (mm)")
        for i,value in enumerate(v.xy_mean_mm):ax[row,0].text(i,value,f"{value:.3f}",ha="center",va="bottom")
        v=s.set_index("model").loc[list(ARMS)]
        ax[row,1].bar(["Current","0.3 s","0.3 / 1 / 4 s"],v.xyz_mean_mm,color=["#8b95a1",blue,"#1f927b"])
        ax[row,1].set(title=f"{device}: virtual-only readouts, real XYZ",ylabel="Mean error (mm)")
        for i,value in enumerate(v.xyz_mean_mm):ax[row,1].text(i,value,f"{value:.3f}",ha="center",va="bottom")
    fig.suptitle("Real newly collected trajectories | 12 Trials per device, equal-Trial means",fontsize=12)
    for ext in ("png","svg"):fig.savefig(figdir/f"real_learning_and_transfer.{ext}",dpi=190)
    plt.close(fig);files.append("figures/real_learning_and_transfer.png")
    fig,ax=plt.subplots(2,2,figsize=(12,7),layout="constrained")
    for row,device in enumerate(PLANS):
        item=next(r for r in protocol["raw_inputs"] if r["device"]==device and r["group"].endswith("03_free_stop_return_V0.7"))
        with np.load(OUT / "streams" / (item["id"]+".npz")) as f:a={k:f[k].copy() for k in f.files}
        mask=a["research_primary"]
        q=a["q"].copy();q[~mask]=np.nan;tt=(a["t_ns"]-a["t_ns"][0])/1e9
        ax[row,0].plot(q[:,0],q[:,1],"--",color="#555",label="Command reference")
        ax[row,1].plot(tt,q[:,2],"--",color="#555",label="Command reference")
        for arm,name,color in zip(ARMS,("Current","0.3 s","Multiscale"),("#89939e",blue,"#1f927b")):
            p=a[arm+"_xyz"].copy();p[~mask]=np.nan
            ax[row,0].plot(p[:,0],p[:,1],lw=.8,color=color,label=name)
            ax[row,1].plot(tt,p[:,2],lw=.8,color=color,label=name)
        ax[row,0].set(title=f"{device}: fixed free-stop-return path",xlabel="X (mm)",ylabel="Y (mm)")
        ax[row,1].set(title=f"{device}: same actual-time depth",xlabel="Actual time (s)",ylabel="Depth (mm)")
        for col in range(2):ax[row,col].legend(fontsize=7)
    for ext in ("png","svg"):fig.savefig(figdir/f"real_causal_readout_comparison.{ext}",dpi=190)
    plt.close(fig);files.append("figures/real_causal_readout_comparison.png")
    return files


def page(summary,pairs,cm,report,files):
    def table(df):return df.to_html(index=False,border=0,classes="results",float_format=lambda x:f"{x:.4f}",escape=True)
    s=summary[summary.subset.eq("new_complex")][["device","model","scope","trials","xy_mean_mm","xyz_mean_mm","z_mae_mm","own_mask_coverage","bias_x_mm","bias_y_mm","bias_z_mm"]]
    retention=summary[~summary.subset.eq("new_complex")][["device","subset","model","xy_mean_mm","xyz_mean_mm","z_mae_mm","own_mask_coverage"]]
    data=pd.read_csv(OUT / "trial_metrics.csv")
    detail=data[data["scope"].isin(["shared_full_chain","research_whole_action"]) & data["mode"].eq("waypoint_xyz")][["id","group","model","xy_mean_mm","xyz_mean_mm","z_mae_mm","coverage","bias_x_mm","bias_y_mm","bias_z_mm"]]
    parity=[]
    for device,v in report["native_online_parity"].items():
        for model,rows in v.items():
            if not rows:continue
            parity.append(dict(device=device,model=model,trials=len(rows),online_rows=sum(x["online_rows"] for x in rows),
                matches=sum(x["exact_timestamp_matches"] for x in rows),missing=sum(x["online_rows_not_reproduced"] for x in rows),
                validity_disagreements=sum(x["validity_disagreements"] for x in rows),
                maximum_axis_difference_mm=max(x["max_absolute_axis_difference_mm"] for x in rows),all_parity=all(x["xyz_and_validity_parity"] for x in rows)))
    dump(OUT / "native_parity_summary.json",parity)
    paragraphs=""
    for device in PLANS:
        p=pairs[pairs.device.eq(device)&pairs.candidate.eq("VirtualXY1000")&pairs.metric.eq("xy_mean_mm")].iloc[0]
        sign="下降" if p.relative_change_percent<0 else "上升"
        paragraphs+=f'<p><b>{device}</b>：共享有效帧的原始 XY 平均误差 {p.old_mm:.3f} → {p.new_mm:.3f} mm，{sign} {abs(p.relative_change_percent):.1f}%；12 条新轨迹中 {p.trials_improved} 条改善。</p>'
    mapping="".join(f'<h3>{k}</h3><p>{html.escape(v)}</p>' for k,v in report["C_F_mapping"].items())
    figs="".join(f'<figure><img src="{p}"><figcaption>{html.escape(Path(p).stem)}；按预设首条路径绘制，拒绝输出断线，未做坐标/时间配准。</figcaption></figure>' for p in files)
    bias_section=""
    if (OUT / "translation_diagnostic.json").exists():
        bias=read(OUT / "translation_diagnostic.json")
        cols=["batch","model","raw_xy_mm","one_batch_translation_xy_mm","per_trial_translation_xy_mm","bias_x_mm","bias_y_mm","bias_z_mm"]
        bias_section=f'''<h3>固定位置偏移核对</h3>
<p>Yellow 的误差增大主要体现为位置偏移：原始XY误差 R373 0.521、增强模型0.728 mm；整批只扣一个共同平移后分别0.490、0.505 mm；每Trial独立扣均值偏移后分别0.478、0.471 mm。后者增强模型8/12条改善，原始误差则2/12条改善。不能只按原始平均误差把问题概括为轨迹形状全面退化，也不能把去偏移诊断宣称为明显实际提升。</p>
<p>两种去偏移方式都用到了评价标签，仅供分解误差，不是可部署校准，也没有写回模型、UI或原始输出。SOURCE_latest_augmented_online 是刚完成的第二批SOURCE增强模型原生在线结果，单列展示，不能和上一批R373跨批次拼接成同流比较。</p>
<div class="scroll">{table(pd.DataFrame(bias['summary'])[cols])}</div><p><a href="translation.html">详细偏移诊断及逐Trial结果</a></p>'''
    text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>新增真实三维轨迹：仿真论文模型回测</title>
<style>body{{font:16px/1.75 "Microsoft YaHei",sans-serif;max-width:1280px;margin:32px auto;padding:0 24px;color:#213044}}h1,h2{{color:#173957}}.note{{background:#edf5fb;border-left:4px solid #2377b9;padding:12px 18px}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:8px;border-bottom:1px solid #dce5ed;text-align:right;white-space:nowrap}}th{{background:#e9f0f6}}tr:nth-child(even){{background:#f8fafc}}img{{width:100%}}figure{{margin:24px 0}}figcaption{{font-size:13px;color:#536477}}a{{color:#1872ad}}</style>
<h1>仿真论文 C–F：两组新增真实三维轨迹回测</h1>
<p>真实采集数据上的实测效果（离线回测）｜SOURCE / Yellow｜各 16 个完整 Trial，新增复杂路径各 12 条</p>
<div class="note">不重新训练，不用 Yellow 选参。回测完整重放原生日志的实际输入、时间间隔、B0 和控制顺序。原始误差为主；有符号整体偏移单列，不做事后平移。标签是平台指令/EST 路径参考，不冒充光学位移真值。</div>
<h2>1. 新轨迹的模型对比</h2>{paragraphs}
<p>整链基线/增强的主比较遵循采集前方案：每 Trial 的共同有效移动帧 XY 误差，再按 12 Trial 等权汇总；各模型自己的覆盖和误差另列。新因果读出与其瞬时输入对照使用全部名义接触动作（含停驻），不与带拒绝门的整链直接作公平排名。</p>
<div class="scroll">{table(s)}</div><h3>配对变化与改善条数</h3><div class="scroll">{table(pairs)}</div>
{bias_section}
<h2>2. 回测和在线的一致性核验</h2><div class="scroll">{table(pd.DataFrame(parity))}</div>
<p>Yellow 保存了 A/B 两模型在线输出，因此两边均作逐点核验。SOURCE 为单模型在线，A 核验后，新候选用同一原生接口和同一记录输入回测；不能说 SOURCE 当时在线运行过 B，但它仍是这一真实磁流上的实测效果。三个研究读出没有原在线版本，不虚构在线逐点核验；其输入由同一 B0 磁流和完整因果前缀独立核对。</p>
<h2>3. 原能力保持、安全与数据时间</h2><div class="scroll">{table(retention)}</div>
<p>瞬时、短历史和三尺度研究读出无接触许可门；无载时输出是其功能边界，不能把接触段 100% 有限输出称为完整系统安全覆盖。旧圆、纯 Z 是保持性检查，不替代新轨迹检验。</p>
<div class="scroll">{table(pd.read_csv(OUT / 'input_audit.csv'))}</div>
<h2>4. 对应论文 C–F 的证据</h2>{mapping}
<p>本批有 {report['real_contact_hold_count']} 个至少 0.5 s 的名义接触停驻片段，最长 {report['longest_real_contact_hold_seconds']:.3f} s。磁信号相对停驻起点的变化不等于减去真实静态平衡后的残余；不据此宣布已证实 16 s 非单调松弛假设。</p>
<h2>5. 轨迹与时序图</h2>{figs}
<h2>6. 逐 Trial 与条件明细</h2><details><summary>展开全部复杂动作的原始误差、覆盖与偏移</summary><div class="scroll">{table(detail)}</div></details>
<details><summary>移动、停驻、转向后1秒和同步XYZ</summary><div class="scroll">{table(cm)}</div></details>
<h2>7. 可复算资料</h2><p>六个几何簇，两档速度和两次动作重复嵌套在簇内；不是 24 个独立几何实验，更不是将磁帧数当重复数。本批采集后未修改权重、标准化、阈值或检查点。</p>
<p><a href="summary.csv">汇总 CSV</a> · <a href="trial_metrics.csv">逐 Trial CSV</a> · <a href="condition_metrics.csv">条件明细</a> · <a href="paired_improvements.csv">配对改善</a> · <a href="native_parity_summary.json">在线一致性核验</a> · <a href="protocol.json">数据源/冻结模型/哈希</a> · <a href="research_full_prefix_audit.json">研究读出因果前缀校验</a> · <a href="report.json">完整结果 JSON</a> · <a href="forward_field_diagnostics.csv">正向磁场诊断</a> · <a href="real_contact_holds.csv">真实短停驻</a></p>
<p>旧论文及原始采集文件保留，新增结果已经整理为 C–F 可用证据；本报告不声称 Word/PDF 已替换渲染完成。</p></html>'''
    (OUT / "index.html").write_text(text,encoding="utf-8")


def main():
    protocol=read(OUT / "protocol.json")
    for item in protocol["raw_inputs"]:
        assert all(sha(ROOT / item["path"] / file)==digest for file,digest in item["raw_hashes"].items())
    df,cond,pauses,forward=refine_research_inputs(protocol)
    summary,pairs,cm,report=summarize(df,cond,pauses,forward,protocol)
    files=plots(protocol,summary,cm);page(summary,pairs,cm,report,files)
    dump(OUT / "work_state.json",dict(status="COMPLETE_ANALYSIS_REPORT_READY",complete_trials=32,models_evaluated=5,
         report="index.html",results="report.json",native_parity="native_parity_summary.json", figures=files,
         raw_files_unchanged=True, frozen_weights_unchanged=True, hardware_operated=False, manuscript_pdf_not_replaced=True,
         actual_script_sha256={"replay":sha(ROOT/'training/real_simulation_paper_r395.py'),"report":sha(__file__)},
         remaining=["Visual QA of report figures", "User-authorized manuscript integration if requested"],blockers=[]))
    print(summary[summary.subset.eq("new_complex")].to_string(index=False),flush=True)
    print("Report:",OUT / "index.html",flush=True)


if __name__=="__main__":main()
