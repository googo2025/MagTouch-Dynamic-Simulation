"""Same-native-stream stability audit; frozen outputs only, no model fitting."""
from __future__ import annotations
import argparse, hashlib, html, json, socket
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'analysis/r402_previous_real_jump_audit_20261003'
PRIOR = ROOT/'analysis/r395_new_real_simulation_paper_20261003'
MODELS = ('R373', 'VirtualXY1000')


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def dump(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')


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


def load_stream(item):
    if item['device']!='SOURCE_REPEAT':
        path=PRIOR/'streams'/(item['id']+'.npz')
        with np.load(path,allow_pickle=False) as f:a={k:f[k].copy() for k in f.files}
        return a,{str(path.relative_to(ROOT)):sha(path)}
    # Native replay already finished; only reconstruct the unchanged command reference.
    from training.build_trajectory_dataset_v1 import command_segments,command_reference,reference_origin,read_rows
    from training.analyze_r354_final_online import mask_for
    files=[PRIOR/'native/SOURCE_REPEAT'/m/(item['id']+'.npz') for m in MODELS]
    arrays=[]
    for p in files:
        with np.load(p,allow_pickle=False) as f:arrays.append({k:f[k].copy() for k in f.files})
    a,b=arrays
    np.testing.assert_array_equal(a['t_ns'],b['t_ns'])
    np.testing.assert_array_equal(a['b0_epoch'],b['b0_epoch'])
    np.testing.assert_allclose(a['b'],b['b'],atol=0,rtol=0,equal_nan=True)
    trial=ROOT/item['path'];meta=read(trial/'meta.json')
    seg=command_segments(read_rows(trial/'stage_commands.csv'))
    ref=command_reference(a['t_ns'],seg,reference_origin(meta));q=ref['xyz_mm']
    phase=np.array([seg[j]['phase'] if j>=0 else 'before_stage' for j in ref['command_segment_id']])
    primary,action=mask_for(item['mode'],phase,ref)
    result=dict(t_ns=a['t_ns'],q=q,phase=phase,primary=primary&np.isfinite(a['b']).all(1),
                whole_action=action&(q[:,2]>=.4)&np.isfinite(a['b']).all(1))
    for m,p in zip(MODELS,arrays):result.update({m+'_xyz':p['xyz'],m+'_ok':p['ok']})
    files.extend([trial/'meta.json',trial/'stage_commands.csv'])
    return result,{str(p.relative_to(ROOT)):sha(p) for p in files}


def analyze():
    assert socket.gethostname()=='googo-Super-Server','Research reductions execute on authorized server'
    protocol=read(OUT/'input_manifest.json')
    rows=[];sources={}
    for item in protocol['entries']:
        if item['mode']!='waypoint_xyz':continue
        a,hashes=load_stream(item);sources.update(hashes)
        t=a['t_ns'];q=a['q'];shared=a['R373_ok']&a['VirtualXY1000_ok']
        for m in MODELS:
            x=a[m+'_xyz'];ok=a[m+'_ok']
            common=dict(id=item['id'],batch=item['device'],group=item['group'],model=m)
            moving=score(x,q,ok,a['primary']&shared,t)
            own=score(x,q,ok,a['primary'],t)
            # Keep slow/held phase separate from movement. Command-reference speed uses actual dt.
            dt=np.r_[1,np.diff(t)/1e9];v=np.vstack((np.zeros(3),np.diff(q,axis=0)))/dt[:,None]
            dwell=a['whole_action']&(np.linalg.norm(v,axis=1)<.02)
            rows.append(dict(**common,moving_shared=moving,moving_own=own,
                dwell=dwell_score(x,ok,dwell,t,a['phase'])))
    summary=[];comparisons=[]
    for batch in sorted({r['batch'] for r in rows}):
        by={}
        for m in MODELS:
            rr=[r for r in rows if r['batch']==batch and r['model']==m]
            s=summarize([r['moving_shared'] for r in rr]);s['own_coverage']=summarize([r['moving_own'] for r in rr])['coverage']
            vv=[r['dwell']['mean_xy_wobble_rms_mm'] for r in rr if r['dwell']['mean_xy_wobble_rms_mm'] is not None]
            s['dwell_xy_wobble_rms_mm']=float(np.mean(vv)) if vv else None
            s.update(batch=batch,model=m);summary.append(s);by[m]=s
        old,new=by['R373'],by['VirtualXY1000']
        rr={m:{r['id']:r for r in rows if r['batch']==batch and r['model']==m} for m in MODELS}
        paired=[dict(id=id,group=rr['R373'][id]['group'],baseline=rr['R373'][id]['moving_shared'],
                     enhanced=rr['VirtualXY1000'][id]['moving_shared']) for id in rr['R373']]
        k='xy_residual_jump_rate_percent'
        comparisons.append(dict(batch=batch,trials=len(paired),
            jump_rate_relative_reduction_percent=100*(1-new[k]/old[k]) if old[k] else None,
            improved_jump_trials=sum(p['enhanced'][k]<p['baseline'][k] for p in paired),
            unchanged_jump_trials=sum(p['enhanced'][k]==p['baseline'][k] for p in paired),
            worsened_jump_trials=sum(p['enhanced'][k]>p['baseline'][k] for p in paired),paired=paired))
    parity=[]
    for rel in protocol['parity_files']:
        obj=read(ROOT/rel);sources[rel]=sha(ROOT/rel)
        if isinstance(obj,list):parity.extend(obj)
        else:
            pp=obj['parity']
            parity.append(dict(device='SOURCE_REPEAT',model='VirtualXY1000',trials=len(pp),
                online_rows=sum(p['online_rows'] for p in pp),
                missing=sum(p['online_rows_not_reproduced'] for p in pp),
                validity_disagreements=sum(p['validity_disagreements'] for p in pp),
                maximum_axis_difference_mm=max(p['max_absolute_axis_difference_mm'] for p in pp),
                all_parity=all(p['xyz_and_validity_parity'] for p in pp)))
    for rel,h in sources.items():assert sha(ROOT/rel)==h,'Frozen input changed'
    result=dict(status='COMPLETE_SAME_STREAM_STABILITY_AUDIT',implementation_sha256=sha(Path(__file__)),summary=summary,comparisons=comparisons,
        details=rows,native_online_replay_parity=parity,input_hashes=sources,
        aggregation='Trial-equal means primary; pooled jump fraction reported separately with exact denominator',
        threshold_mm=.2,maximum_adjacent_gap_seconds=.25,
        jump_definition='norm of consecutive XY error-vector difference, subtracting same commanded motion; no output filtering',
        reference='Recorded commanded path, EST timing, not independent encoder or optical truth',
        centered_error_is_test_label_diagnostic_not_deployable_calibration=True,
        new_model_training_or_threshold_selection=False,hardware_operated=False,
        running_SOURCE_baseline_plan_not_included=True)
    dump(OUT/'report.json',result)
    render(result)
    print(json.dumps(dict(summary=summary,parity=parity),ensure_ascii=False),flush=True)


def render(r):
    trs=[]
    for s in r['summary']:
        def fmt(k,n=3):return f'{s[k]:.{n}f}' if s[k] is not None else '—'
        trs.append('<tr>'+''.join(f'<td>{v}</td>' for v in [s['batch'],s['model'],s['trials'],
            fmt('raw_xy_mean_mm'),fmt('centered_xy_diagnostic_mm'),fmt('xy_residual_jump_rate_percent',2),
            s['xy_residual_jump_count'],s['adjacent_pairs'],fmt('xy_residual_step_p95_mm'),fmt('dwell_xy_wobble_rms_mm'),
            f"{100*s['own_coverage']:.2f}%"] )+'</tr>')
    details=[]
    for c in r['comparisons']:
        details.append(f"<h2>{c['batch']} 逐轨迹对比</h2><p>跳变比例下降／相同／上升：{c['improved_jump_trials']}／{c['unchanged_jump_trials']}／{c['worsened_jump_trials']}。</p>")
        for p in c['paired']:
            a,b=p['baseline'],p['enhanced']
            details.append(f"<p>{html.escape(p['group'])}：跳变 {a['xy_residual_jumps_over_0p2']}/{a['adjacent_pairs']} → {b['xy_residual_jumps_over_0p2']}/{b['adjacent_pairs']}，残差步长 P95 {a['xy_residual_step_p95_mm']:.3f} → {b['xy_residual_step_p95_mm']:.3f} mm。</p>")
    page='''<!doctype html><meta charset="utf-8"><title>前面真实轨迹的同流跳变核验</title><style>body{font:16px/1.6 system-ui;margin:32px auto;max-width:1450px;color:#172536;padding:0 24px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:right;padding:9px;border-bottom:1px solid #ddd}th:first-child,td:first-child{text-align:left}th{background:#edf2f7}h1{font-size:26px}.scroll{overflow:auto}</style><h1>前面真实轨迹：同一磁场流的跳变与波动</h1><p>R373 基线与 1000 条虚拟扩集 XY＋原 R373 Z，六类路径 × 两档速度，共 12 个 Trial／批。不是两次采集拼接，也不是虚拟测试。</p><p>跳变：相邻有效输出的 XY 误差向量增量 &gt;0.2 mm，dt ≤0.25 s；减去同一指令运动，无输出平滑。主比例为逐 Trial 等权平均；次数／对数供独立复算。固定偏移不改变增量统计。</p><div class="scroll"><table><tr><th>数据批</th><th>模型</th><th>Trial</th><th>原始 XY mm</th><th>去平移诊断 mm</th><th>跳变比例 %</th><th>跳变次数</th><th>相邻对数</th><th>残差步长 P95 mm</th><th>停驻波动 RMS mm</th><th>自身覆盖</th></tr>'''+''.join(trs)+'</table></div><h2>在线与回放是否一致</h2>'
    for p in r['native_online_replay_parity']:
        page+=f"<p>{p['device']} / {p['model']}：{p['online_rows']:,} 在线行，缺失 {p['missing']}，有效性差异 {p['validity_disagreements']}，最大轴差 {p['maximum_axis_difference_mm']} mm。</p>"
    page+='<p>这是此前冻结原生回放的逐点核验。保持实际 B0、输入时间及控制事件；不意味着任何简化回测都自动等同在线。新的 SOURCE 基线采集正在进行，未关闭 Trial 不纳入。</p>'+''.join(details)+'<p><a href="report.json">全部结果、定义与输入哈希</a></p>'
    (OUT/'index.html').write_text(page,encoding='utf8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--render-only',action='store_true');args=parser.parse_args()
    if args.render_only:render(read(OUT/'report.json'))
    else:analyze()
