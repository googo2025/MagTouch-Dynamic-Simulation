"""Completed native D trials only; separate acquisitions, not same-stream replay."""
from pathlib import Path
import json, re, sys
import numpy as np
import pandas as pd

from training.analyze_d_real_r400 import magnetic, rows, bins
from training.build_trajectory_dataset_v1 import command_segments, command_reference, reference_origin
from training.real_simulation_paper_r395 import sha, dump, read

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis/r401_d_native_model_comparison_20261003'
EXPECTED = {
    'DirectPlateauReconstructionR373': ('R373', '65f3015e8c053b1c806fe6c56cb53e862a8b97c4f1f882cb8f8fbf7b694305ff'),
    'VirtualExpandedXY1000_R385_R373Z': ('VirtualXY1000', '2eadbba142db96654e7a4c215c536f414b05cee14b505f13d16705cfda97533e'),
}


def metrics(x, q, ok, mask, t):
    eligible = mask & np.isfinite(q).all(1)
    good = eligible & ok & np.isfinite(x).all(1)
    n, m = int(eligible.sum()), int(good.sum())
    out = dict(reference_frames=n, output_frames=m, coverage_percent=100*m/n if n else None)
    if not m:
        return out
    e = x[good] - q[good]
    bias = e.mean(0)
    out.update(xy_raw_mean_mm=float(np.linalg.norm(e[:, :2], axis=1).mean()),
               xyz_raw_mean_mm=float(np.linalg.norm(e, axis=1).mean()),
               z_raw_mae_mm=float(np.abs(e[:, 2]).mean()), bias_xyz_mm=bias.tolist(),
               xy_centered_diagnostic_mm=float(np.linalg.norm((e-bias)[:, :2], axis=1).mean()),
               z_centered_diagnostic_mae_mm=float(np.abs((e-bias)[:, 2]).mean()))
    adjacent = good[:-1] & good[1:] & (np.diff(t)>0) & (np.diff(t)<=250_000_000)
    de = np.diff(x-q, axis=0)[adjacent]
    if len(de):
        xy = np.linalg.norm(de[:, :2], axis=1)
        dz = np.abs(de[:, 2])
        out.update(adjacent_pairs=int(len(de)), xy_error_step_p95_mm=float(np.quantile(xy,.95)),
                   z_error_step_p95_mm=float(np.quantile(dz,.95)),
                   xy_error_jumps_over_0p2=int((xy>.2).sum()), z_error_jumps_over_0p2=int((dz>.2).sum()))
    return out


def average(rows_, keys):
    return {k:float(np.mean([r[k] for r in rows_ if r.get(k) is not None]))
            if any(r.get(k) is not None for r in rows_) else None for k in keys}


def trajectory_svg(inventory, groups):
    """Native movement samples; panels share bounds; centering is diagnostic."""
    selected = [g for g in groups if g.startswith('D_HISTORY_')]
    if not selected: return
    colors={'R373':'#1968ad','VirtualXY1000':'#dc6922'}
    out=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 '+str(len(selected)*410)+'"><rect width="100%" height="100%" fill="white"/>']
    for row,group in enumerate(selected):
        curves=[]
        for model in ('R373','VirtualXY1000'):
            rr=next(r for r in inventory if r['group']==group and r['model']==model)
            a=np.load(OUT/'streams'/(Path(rr['trial']).name+'.npz'))
            good=a['moving']&a['ok']; xy=a['xyz'][good,:2]; q=a['q'][good,:2]
            bias=(xy-q).mean(0)
            ii=np.flatnonzero(good)
            cuts=np.flatnonzero((np.diff(ii)>1)|(np.diff(a['t_ns'][ii])>250_000_000))+1
            pieces=np.split(np.arange(len(ii)),cuts)
            curves.append((model,xy,xy-bias,q,pieces))
        lim=max(5.5,max(float(np.max(np.abs(a))) for _,x,c,q,_ in curves for a in (x,c,q)))+.3
        for j,label in enumerate(('Raw native XY','Translation-removed diagnostic XY')):
            left,top,w,h=55+j*550,55+row*410,450,300
            out.append(f'<text x="{left}" y="{top-30}" font-family="Arial" font-size="14">{group}</text><text x="{left}" y="{top-10}" font-family="Arial" font-size="14">{label}</text>')
            def points(a):
                return ' '.join(f'{left+w*(p[0]+lim)/(2*lim):.2f},{top+h*(lim-p[1])/(2*lim):.2f}' for p in a)
            for value in (-5,0,5):
                xx=left+w*(value+lim)/(2*lim); yy=top+h*(lim-value)/(2*lim)
                out.append(f'<path d="M{xx},{top}v{h} M{left},{yy}h{w}" stroke="#ddd" fill="none"/><text x="{xx}" y="{top+h+16}" font-size="12">{value}</text><text x="{left-24}" y="{yy+4}" font-size="12">{value}</text>')
            for k,(model,xy,centered,q,pieces) in enumerate(curves):
                # Only the recorded complete moving samples; do not fill rejected frames.
                for block in pieces:
                    out.append(f'<polyline points="{points((xy if j==0 else centered)[block])}" fill="none" stroke="{colors[model]}" stroke-width="1.1" opacity=".85"/>')
                    if k==0:
                        out.append(f'<polyline points="{points(q[block])}" fill="none" stroke="#444" stroke-width="1.1" stroke-dasharray="5 4"/>')
                out.append(f'<text x="{left+k*170}" y="{top+h+45}" font-size="13" fill="{colors[model]}">{model}</text>')
            out.append(f'<text x="{left+w}" y="{top+h+45}" text-anchor="end" font-size="12">X,Y (mm); dashed: command</text>')
    out.append('</svg>')
    (OUT/'trajectory_comparison.svg').write_text(''.join(out),encoding='utf8')


def analyze_trial(path):
    meta = read(path/'meta.json')
    assert meta['status']=='completed'
    item = read(path/'batch_plan_item.json')
    pr = meta['live_prediction_recording']
    label, weight = EXPECTED[pr['model_family']]
    assert pr['algorithm_metadata']['xy_mlp_artifact_sha256']==weight
    names = ('meta.json','batch_plan_item.json','magnetic_raw.csv','stage_commands.csv','events.csv','live_predictions.csv')
    hashes = {f:sha(path/f) for f in names}
    commands = rows(path/'stage_commands.csv')
    segments = command_segments(commands)
    origin = reference_origin(meta)
    frame = pd.read_csv(path/'live_predictions.csv', usecols=['t_prediction_ns','interaction_valid',
                          'visual_x_mm','visual_y_mm','visual_z_mm','model_family'])
    assert set(frame.model_family)=={pr['model_family']}
    t = frame.t_prediction_ns.to_numpy(np.int64)
    assert np.all(np.diff(t)>0)
    xyz = frame[['visual_x_mm','visual_y_mm','visual_z_mm']].to_numpy(float)
    ok = frame.interaction_valid.astype(str).str.lower().isin(['true','1']).to_numpy() & np.isfinite(xyz).all(1)
    ref = command_reference(t,segments,origin)
    q = ref['xyz_mm'].astype(float)
    phase = np.array([segments[j]['phase'] if j>=0 else '' for j in ref['command_segment_id']])
    moving = np.array([bool(re.fullmatch(r'waypoint_\d{3}',p)) for p in phase]) & ref['command_moving'] & (q[:,2]>=.4)
    (OUT/'streams').mkdir(parents=True,exist_ok=True)
    np.savez_compressed(OUT/'streams'/(path.name+'.npz'),t_ns=t,xyz=xyz,q=q,ok=ok,moving=moving)
    result = dict(trial=path.relative_to(ROOT).as_posix(), group=item['group'], model=label,
                  family=pr['model_family'], xy_sha256=weight, origin_mm=origin.tolist(),
                  moving=metrics(xyz,q,ok,moving,t), raw_hashes=hashes,
                  inference_queue_drops=pr.get('input_groups_dropped_by_live_queue'),
                  has_m400=any(c['command'].strip().upper()=='M400' for c in commands), holds=[])
    rt, rb, rg = magnetic(path/'magnetic_raw.csv')
    result['raw_frames']=len(rt); result['raw_valid_frames']=int(rg.sum())
    result['safe_unloaded_reference_frames']=int((np.isfinite(q).all(1)&(q[:,2]<=-.2)).sum())
    result['safe_unloaded_output_frames']=int((np.isfinite(q).all(1)&(q[:,2]<=-.2)&ok).sum())
    ev = rows(path/'events.csv')
    b0events = [e for e in ev if e['event']=='DYNAMIC_B0_APPLIED']
    hold_indices = [(i,c) for i,c in enumerate(commands) if c['command']=='HOST_DWELL' and
                    (c['phase'].startswith('waypoint_hold_') or c['phase']=='steady_hold')]
    if not hold_indices:
        result['unloaded_mean_uT']=rb[rg].mean(0).tolist()
    for rep,(i,h) in enumerate(hold_indices,1):
        wait = commands[i-1]
        assert wait['command']=='HOST_WAIT'
        start,end=int(wait['t_perf_ns']),int(h['t_perf_ns'])
        age=(t-start)/1e9
        take=(age>=0)&(age<20)
        late_mask=ok&(age>=16)&(age<20)
        early_mask=ok&(age>=0)&(age<1)
        last_segment=next(s for s in reversed(segments) if s['start_ns']<=start)
        endpoint=(last_segment['to_mm']-origin)*[1,1,-1]
        assert np.isfinite(endpoint).all()
        np.testing.assert_allclose(q[take], np.broadcast_to(endpoint,(int(take.sum()),3)),atol=1e-4)
        hold = dict(repeat=rep,start_ns=start,end_ns=end,commanded_endpoint_mm=endpoint.tolist(),
                    native=metrics(xyz,q,ok,take,t))
        if late_mask.any():
            late=xyz[late_mask]; lm=late.mean(0)
            hold['late_bias_xyz_mm']=(lm-endpoint).tolist()
            hold['late_xy_wobble_rms_mm']=float(np.sqrt(np.mean(np.sum((late[:,:2]-lm[:2])**2,1))))
            hold['late_z_std_mm']=float(np.std(late[:,2],ddof=1)) if len(late)>1 else None
            if early_mask.any():
                hold['early_to_late_drift_xyz_mm']=(lm-xyz[early_mask].mean(0)).tolist()
                hold['early_to_late_drift_norm_mm']=float(np.linalg.norm(lm-xyz[early_mask].mean(0)))
        rage=(rt-start)/1e9
        rlate=rb[rg&(rage>=16)&(rage<20)].mean(0)
        initial=max((e for e in b0events if int(e['t_perf_ns'])<=start),key=lambda e:int(e['t_perf_ns']))
        b0=np.asarray(json.loads(initial['value'])['b0_5x3_uT']).reshape(15)
        hold['late_absolute_B_uT']=rlate.tolist()
        hold['late_minus_initial_trial_B0_uT']=(rlate-b0).tolist()
        hold['initial_trial_B0_uT']=b0.tolist()
        bt,bb,_=bins(rage[rg&(rage>=0)&(rage<20)],rb[rg&(rage>=0)&(rage<20)])
        rn=np.linalg.norm(bb-rlate,axis=1)
        hold['first_bin_residual_norm_uT']=float(rn[0])
        hold['peak_rise_uT']=float(rn[bt<16].max()-rn[0])
        # Descriptive timing sensitivity, not a revised predeclared test.
        candidates=[]
        for shift in (-1.,-.5,-.2,0.,.2,.5,1.):
            aa=(rt-start)/1e9-shift
            if not (rg&(aa>=0)&(aa<.2)).any():
                continue
            st,sb,_=bins(aa[rg&(aa>=0)&(aa<20)],rb[rg&(aa>=0)&(aa<20)])
            sl=rb[rg&(aa>=16)&(aa<20)].mean(0)
            sn=np.linalg.norm(sb-sl,axis=1)
            candidates.append(dict(hold_origin_shift_s=shift,peak_rise_uT=float(sn[st<16].max()-sn[0])))
        hold['timing_sensitivity']=candidates
        result['holds'].append(hold)
    for f in names:
        assert sha(path/f)==hashes[f], 'Completed source changed during analysis'
    return result


def main():
    plans=sorted((ROOT/'data/S20261003_A').glob('Plan_*75a70606cd72*'))
    inventory=[]; skipped=[]; progress=[]
    for plan in plans:
        p=read(plan/'plan_progress.json')
        progress.append(dict(plan=plan.name,status=p['status'],completed=p['completed_trials'],total=p['planned_trials']))
        for path in sorted(plan.glob('Trial_*')):
            meta=read(path/'meta.json')
            if meta.get('status')!='completed':
                skipped.append(path.relative_to(ROOT).as_posix()); continue
            inventory.append(analyze_trial(path))
    groups=sorted(set(r['group'] for r in inventory if r['model']=='R373') &
                  set(r['group'] for r in inventory if r['model']=='VirtualXY1000'))
    paired=[]
    for group in groups:
        for model in ('R373','VirtualXY1000'):
            r=next(r for r in inventory if r['group']==group and r['model']==model)
            hold=[h['native'] for h in r['holds']]
            paired.append(dict(group=group,model=model,trial=r['trial'],moving=r['moving'],
                holds=len(hold),hold_mean=average(hold,['coverage_percent','xy_raw_mean_mm','xyz_raw_mean_mm',
                'z_raw_mae_mm','xy_centered_diagnostic_mm','xy_error_step_p95_mm','z_error_step_p95_mm',
                'xy_error_jumps_over_0p2','z_error_jumps_over_0p2']),
                tail=average(r['holds'],['late_xy_wobble_rms_mm','late_z_std_mm','early_to_late_drift_norm_mm'])))
    contrasts=[]
    for model in ('R373','VirtualXY1000'):
        for tag in ('ACTIVE381_001_END456','ACTIVE381_008_END438'):
            a=next((r for r in inventory if r['model']==model and r['group']=='D_HISTORY_'+tag),None)
            b=next((r for r in inventory if r['model']==model and r['group']=='D_DIRECT_Z_'+tag),None)
            if not a or not b:
                continue
            means={}
            for key in ('late_absolute_B_uT','late_minus_initial_trial_B0_uT','initial_trial_B0_uT'):
                am=np.mean([h[key] for h in a['holds']],axis=0); bm=np.mean([h[key] for h in b['holds']],axis=0)
                means[key]=float(np.linalg.norm(am-bm))
            ae=np.array([h['commanded_endpoint_mm'] for h in a['holds']]); be=np.array([h['commanded_endpoint_mm'] for h in b['holds']])
            contrasts.append(dict(model=model,tag=tag,endpoint_difference_mm=float(np.max(np.linalg.norm(ae-be,axis=1))),
                                  between_history_field_norms_uT=means))
    complete=all(p['status']=='COMPLETED' and p['completed']==p['total'] for p in progress) and not skipped
    report=dict(status='COMPLETED_NATIVE_TWO_BATCH_COMPARISON' if complete else 'PARTIAL_COMPLETED_TRIALS_NATIVE_COMPARISON',plans=progress,
                completed_trials=len(inventory),skipped_unfinished_trials=skipped,
                comparison='Two separately acquired native online runs; NOT same-stream replay; actual raw predictions unchanged',
                paired_complete_conditions=paired,field_contrasts=contrasts,trials=inventory,
                centered_values_are_diagnostic_not_calibration=True,
                notes=['Command/EST reference, not encoder/optical truth',
                       'Hold start is HOST_WAIT end; no independently observed motion completion',
                       'Magnetic contrast subtracts initial per-Trial B0; does not claim this equals every pre-action inference B0',
                       'Nested repeats are descriptive, not independent statistical replicates',
                       'Model differences are confounded with separately acquired fields/B0; same-stream replay remains necessary for attribution'],
                hardware_operated=False,production_changed=False,raw_files_unchanged=True)
    dump(OUT/'report.json',report)
    trajectory_svg(inventory,groups)
    def val(v): return '—' if v is None else f'{v:.3f}'
    html=['<meta charset="utf-8"><title>D新旧模型在线对比</title><style>body{font:16px/1.7 system-ui;max-width:1150px;margin:30px auto}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:7px}td{font-variant-numeric:tabular-nums}</style>',
          '<h1>D新轨迹：已完成条件的新旧模型原生在线对比</h1>',
          '<p>原R373与1000条虚拟扩集XY＋原R373 Z。两轮分别采集，当前结果不是同磁场输入的模型净收益；未完成Trial不纳入。原始误差和逐重复去偏移诊断分别报告，未修改输出。</p>',
          '<h2>复杂轨迹移动段</h2><table><tr><th>条件</th><th>模型</th><th>原始XY mm</th><th>去固定偏移XY mm</th><th>Z MAE mm</th><th>覆盖%</th></tr>']
    for r in paired:
        m=r['moving']
        if m['reference_frames']:
            html.append(f'<tr><td>{r["group"]}</td><td>{r["model"]}</td><td>{val(m.get("xy_raw_mean_mm"))}</td><td>{val(m.get("xy_centered_diagnostic_mm"))}</td><td>{val(m.get("z_raw_mae_mm"))}</td><td>{val(m["coverage_percent"])}</td></tr>')
    html.append('</table><h2>同路径原生在线输出</h2><img src="trajectory_comparison.svg" style="width:100%"><p>左：原始输出。右：整条移动轨迹扣除一个固定平均偏移，仅用于形状诊断；没有逐点校正或时间对齐。参考为指令路径。</p><h2>20秒停驻：三个重复等权平均</h2><table><tr><th>条件</th><th>模型</th><th>原始XY mm</th><th>Z MAE mm</th><th>XY误差步长P95 mm</th><th>Z误差步长P95 mm</th><th>末4秒XY波动RMS mm</th><th>末4秒Z标准差 mm</th><th>首秒到末4秒漂移 mm</th><th>覆盖%</th></tr>')
    for r in paired:
        if not r['holds']: continue
        m,tail=r['hold_mean'],r['tail']
        html.append(f'<tr><td>{r["group"]}</td><td>{r["model"]}</td>'+''.join(f'<td>{val(v)}</td>' for v in [m['xy_raw_mean_mm'],m['z_raw_mae_mm'],m['xy_error_step_p95_mm'],m['z_error_step_p95_mm'],tail['late_xy_wobble_rms_mm'],tail['late_z_std_mm'],tail['early_to_late_drift_norm_mm'],m['coverage_percent']])+'</tr>')
    html.append('</table><h2>相同指令终点：晚期磁场差异</h2><table><tr><th>条件</th><th>采集轮</th><th>实际指令终点差 mm</th><th>绝对B差 μT</th><th>各自初始B0扣除后差 μT</th><th>初始B0差 μT</th></tr>')
    for r in contrasts:
        a=r['between_history_field_norms_uT']
        html.append(f'<tr><td>{r["tag"]}</td><td>{r["model"]}</td><td>{val(r["endpoint_difference_mm"])}</td><td>{val(a["late_absolute_B_uT"])}</td><td>{val(a["late_minus_initial_trial_B0_uT"])}</td><td>{val(a["initial_trial_B0_uT"])}</td></tr>')
    html.append('</table><p>这里只能证明相同指令終点下的响应差异，不能唯一排除机械回差、实际触点差异和未建模慢历史。D原先的非单调峰值检验与定位模型收益是两个不同问题。</p><p><a href="report.json">逐重复结果、输入哈希与限制</a></p>')
    (OUT/'index.html').write_text(''.join(html),encoding='utf8')
    print(json.dumps(dict(plans=progress,paired=paired,field_contrasts=contrasts,skipped=skipped),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__': main()
