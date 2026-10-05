"""Systematic causal confuser search on latest measured and rich virtual streams.

Exploration only: no training, target-based tuning, time warping or registration.
Nominal stage XYZ defines separated candidates, not verified physical contact.
The measured empty-load window control preserves temporal noise correlations.
"""
from pathlib import Path
import argparse, json, html
import numpy as np
from training.run_forward_r361 import ROOT, dump, sha
from training.research_ambiguity_r366 import causal_windows

OUT=ROOT/'analysis/r380_latest_sequence_ambiguity_20261002'
MATCHED=ROOT/'analysis/r374_matched_real_virtual_20261002'
VIRTUAL=ROOT/'analysis/r370_broad_xyz_training_20261001'
NOISE=ROOT/'analysis/r363_complex_forward_20261001/synthetic_manifest.json'
WINDOWS=(0.,.1,.4,1.,2.)

def permitted(q,trial,times):
    separated=(np.linalg.norm(q[:,None,:2]-q[None,:,:2],axis=2)>=1.)|(np.abs(q[:,None,2]-q[None,:,2])>=.3)
    same=trial[:,None]==trial[None,:]
    separated&=~same|(np.abs(times[:,None]-times[None,:])>=3.)
    return separated,same

def load_groups(virtual_count):
    manifest=json.loads((MATCHED/'manifest.json').read_text(encoding='utf8'))
    groups={'source_latest':[],'target_latest':[],'virtual_rich':[]};unloaded=[];provenance=[]
    for r in manifest:
        p=MATCHED/'streams'/f"{r['id']}_matched.npz"
        with np.load(p,allow_pickle=False) as f:a={k:f[k].copy() for k in f.files}
        a.update(id=r['id'],mode=r['mode'],path=p.relative_to(ROOT).as_posix())
        a['primary']&=a['valid']&np.isfinite(a['q']).all(1)&(a['q'][:,2]>=.4)
        provenance.append({'id':r['id'],'source_raw_hashes':r['hashes'],'cache_sha256':sha(p)})
        if r['mode']!='none':groups[r['device']+'_latest'].append(a)
        elif r['device']=='source':unloaded.append(a)
    # Evenly spaced trajectory IDs, selected before searching magnetic outcomes.
    for i in np.linspace(0,999,virtual_count,dtype=int):
        p=VIRTUAL/'virtual'/f'V370_train_{i:04d}.npz'
        with np.load(p,allow_pickle=False) as f:a={k:f[k].copy() for k in f.files}
        a.update(id=p.stem,mode='free_xyz',path=p.relative_to(ROOT).as_posix())
        a['primary']&=a['valid']&np.isfinite(a['q']).all(1)&(a['q'][:,2]>=.4)
        groups['virtual_rich'].append(a);provenance.append({'id':p.stem,'cache_sha256':sha(p)})
    return groups,unloaded,provenance

def make_bank(rows,sigma,stride=4,cap=3000):
    fields=[];q=[];pred=[];ok=[];records=[]
    for trial,a in enumerate(rows):
        ids=np.flatnonzero(a['primary']&np.isfinite(a['b']).all(1))[::stride]
        v,good=causal_windows(a['b'],a['t'],a['reset'],ids,2.)
        ids=ids[good];v=v[good].reshape(-1,21,15)/sigma[None,None,:]
        if not len(ids):continue
        fields.append(v);q.append(a['q'][ids])
        pred.append(a['real_new'][ids] if 'real_new' in a else np.full((len(ids),3),np.nan))
        ok.append(a['real_new_ok'][ids] if 'real_new_ok' in a else np.zeros(len(ids),bool))
        records.extend({'id':a['id'],'trial':trial,'index':int(i),'time_s':float(a['t'][i]),'mode':a['mode']} for i in ids)
    if not records:raise ValueError('No complete causal windows')
    n=len(records);choose=np.arange(n) if n<=cap else np.linspace(0,n-1,cap,dtype=int)
    return {'fields':np.concatenate(fields)[choose],'q':np.vstack(q)[choose],'pred':np.vstack(pred)[choose],
        'ok':np.concatenate(ok)[choose],'records':[records[i] for i in choose],'before_cap':n}

def pair_peak(fields,pairs,lastlag):
    if len(pairs)==0:return np.empty(0)
    pairs=np.array(pairs,int);a,b=pairs.T
    return np.sqrt(np.mean((fields[a,:lastlag+1]-fields[b,:lastlag+1])**2,axis=2)).max(axis=1)

def null_control(rows,sigma):
    fields=[];trials=[];times=[]
    for k,a in enumerate(rows):
        ids=np.flatnonzero(a['valid']&np.isfinite(a['b']).all(1))[::2]
        values,good=causal_windows(a['b'],a['t'],a['reset'],ids,2.)
        fields.append(values[good].reshape(-1,21,15)/sigma[None,None,:]);trials.extend([k]*int(good.sum()));times.extend(a['t'][ids[good]])
    fields=np.concatenate(fields);trials=np.array(trials);times=np.array(times)
    pairs=np.argwhere((trials[:,None]==trials[None,:])&(times[:,None]-times[None,:]>=3.))
    current=pair_peak(fields,pairs,0);chosen=pairs[current<=2.]
    if len(chosen)<30:raise ValueError('Insufficient measured empty-load controls')
    thresholds={str(w):float(np.quantile(pair_peak(fields,chosen,round(w*10)),.95)) for w in WINDOWS}
    return {'unloaded_records':[a['id'] for a in rows],'complete_endpoints':len(fields),'all_nonoverlapping_same_trial_pairs':len(pairs),
        'current_distance_at_most_2_pairs':len(chosen),'conditional_max_distance_p95':thresholds,
        'scope':'Descriptive measured empty-load window control; overlapping windows and serial pairs are not independent replicates or calibrated confidence.'}

def search(bank,thresholds,name):
    f=bank['fields'];q=bank['q'];records=bank['records'];n=len(q)
    trials=np.array([r['trial'] for r in records]);times=np.array([r['time_s'] for r in records])
    allowed,same=permitted(q,trials,times);cumulative=np.zeros((n,n));fixed=[];within=[];windows={}
    current=np.full(n,np.inf);current_j=np.full(n,-1,int)
    for lag in range(21):
        x=f[:,lag];norm=np.sum(x*x,axis=1)
        d=np.maximum(norm[:,None]+norm[None,:]-2*x@x.T,0)/15
        cumulative=np.maximum(cumulative,d)
        if lag==0:
            masked=np.where(allowed,d,np.inf);current_j=masked.argmin(1);current=np.sqrt(masked.min(1))
            fixed=sorted({tuple(sorted((i,int(j)))) for i,j in enumerate(current_j) if current[i]<=2.})
            masked=np.where(allowed&same,d,np.inf);j=masked.argmin(1);dist=np.sqrt(masked.min(1))
            within=sorted({tuple(sorted((i,int(k)))) for i,k in enumerate(j) if dist[i]<=2.})
        w=lag/10
        if w not in WINDOWS:continue
        best=np.sqrt(np.where(allowed,cumulative,np.inf).min(1));samebest=np.sqrt(np.where(allowed&same,cumulative,np.inf).min(1))
        finite=np.isfinite(best);fs=np.isfinite(samebest);dp=pair_peak(f,fixed,lag);ds=pair_peak(f,within,lag)
        threshold=thresholds[str(w)]
        windows[str(w)]={'eligible_endpoints':int(finite.sum()),'endpoints_with_confuser_at_2':int((best[finite]<=2.).sum()),
            'confuser_fraction_at_2':float((best[finite]<=2.).mean()),'within_trial_eligible_endpoints':int(fs.sum()),
            'within_trial_confuser_fraction_at_2':float((samebest[fs]<=2.).mean()) if fs.any() else None,
            'fixed_current_pairs':len(fixed),'fixed_pairs_still_at_2':int((dp<=2.).sum()),
            'fixed_pairs_separated_over_2':float((dp>2.).mean()) if len(dp) else None,
            'measured_null_p95':threshold,'fixed_pairs_exceeding_measured_null_p95':int((dp>threshold).sum()),
            'fixed_within_trial_pairs':len(within),'fixed_within_trial_still_at_2':int((ds<=2.).sum()),
            'fixed_within_trial_exceeding_measured_null_p95':int((ds>threshold).sum())}
    # No fitting or outcome-based thresholds: associate current confusers with
    # previously frozen real R373 outputs, with refusal kept separate.
    error=np.linalg.norm(bank['pred']-q,axis=1);accepted=bank['ok']&np.isfinite(error)
    risk={}
    for label,mask in [('current_confuser',current<=2.),('no_current_confuser',current>2.)] if accepted.any() else []:
        valid=mask&accepted
        risk[label]={'endpoints':int(mask.sum()),'accepted':int(valid.sum()),'rejected':int((mask&~accepted).sum()),
            'raw_xyz_mean_mm':float(error[valid].mean()) if valid.any() else None,
            'raw_xyz_p95_mm':float(np.quantile(error[valid],.95)) if valid.any() else None}
    def example(p):
        i,j=p;return {'a':records[i],'b':records[j],'q_a_mm':q[i].tolist(),'q_b_mm':q[j].tolist(),
            'same_trial':bool(trials[i]==trials[j]),'raw_field_distance_by_window':{str(w):float(pair_peak(f,[p],round(w*10))[0]) for w in WINDOWS},
            'raw_xyz_error_mm':[float(error[k]) if accepted[k] else None for k in (i,j)]}
    ordered=sorted(fixed,key=lambda p:pair_peak(f,[p],0)[0])
    # Same nominal XYZ <=0.15 mm, separated history >=3 s; strongest disparities.
    candidates=[]
    for i in range(n):
        mask=(np.linalg.norm(q-q[i],axis=1)<=.15)&((trials!=trials[i])|(np.abs(times-times[i])>=3.))
        for label,m in [('any_trial',mask),('within_trial',mask&(trials==trials[i]))]:
            ids=np.flatnonzero(m)
            if not len(ids):continue
            distances=np.sqrt(np.mean((f[ids,0]-f[i,0])**2,axis=1));j=int(ids[distances.argmax()])
            if i<j:candidates.append({'scope':label,'current_distance':float(distances.max()),**example((i,j))})
    candidates.sort(key=lambda r:r['current_distance'],reverse=True)
    saved={k:bank[k] for k in ('fields','q','pred','ok')};np.savez_compressed(OUT/f'{name}_bank.npz',**saved)
    dump(OUT/f'{name}_bank_metadata.json',{'records':records,'candidate_geometry':'nominal command XYZ','noise_scale_source_only':True})
    dump(OUT/f'{name}_all_current_pairs.json',[example(p) for p in fixed])
    return {'endpoints':n,'before_cap':bank['before_cap'],'windows':windows,'current_confuser_outcome_association':risk,
        'closest_current_confuser_examples':[example(p) for p in ordered[:12]],
        'within_trial_confuser_examples':[example(p) for p in within[:12]],
        'same_nominal_position_history_examples':candidates[:12],
        'within_trial_same_position_history_examples':[r for r in candidates if r['scope']=='within_trial'][:12],
        'inverse_outcomes_available':bool(accepted.any()),'missing_virtual_inverse_outputs_are_not_refusals':True}

def render(report):
    parts=['<!doctype html><meta charset="utf-8"><title>R380 因果磁历史混淆搜索</title><style>body{font:16px/1.65 system-ui;max-width:1120px;margin:32px auto;padding:20px;color:#17324d}table{border-collapse:collapse;width:100%;margin:18px 0}th,td{border:1px solid #cbd6df;padding:9px;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#e9f0f7}h2{margin-top:35px}.note{background:#fff4de;padding:16px}pre{white-space:pre-wrap;background:#f4f7fa;padding:12px}</style><h1>R380：位置—历史混淆的因果序列搜索</h1><p>真实磁场与虚拟磁场分开统计；不修改训练权重、门控、原始CSV或生产UI。沿用冻结源侧通道噪声尺度，比较当前帧和0.1／0.4／1／2秒因果窗口。</p><p class="note">“候选可分开”不是保证坐标恢复；不同窗口包含更多比较，必须同时看实测空载历史窗口的对照。平台XYZ是名义动作参考，跨Trial还混有B0、坐标和时间误差。Yellow只作事后诊断，不调参。</p>']
    labels={'source_latest':'最新源器件实测','target_latest':'最新Yellow实测','virtual_rich':'丰富自由三维虚拟动作'}
    for name,r in report['datasets'].items():
        parts+=['<h2>'+labels[name]+'</h2><p>共同端点 '+str(r['endpoints'])+'；完整候选池 '+str(r['before_cap'])+'。不同XYZ候选：XY差≥1 mm或Z差≥0.3 mm，同Trial时距≥3秒。</p><table><tr><th>历史窗口(s)</th><th>存在近似候选的端点</th><th>同Trial近似候选比例</th><th>固定近似对剩余</th><th>空载窗口P95</th></tr>']
        for w,v in r['windows'].items():
            frac=v['within_trial_confuser_fraction_at_2'];same='—' if frac is None else f'{100*frac:.2f}%'
            parts.append(f'<tr><td>{w}</td><td>{v["endpoints_with_confuser_at_2"]}/{v["eligible_endpoints"]} ({100*v["confuser_fraction_at_2"]:.2f}%)</td><td>{same}</td><td>{v["fixed_pairs_still_at_2"]}/{v["fixed_current_pairs"]}</td><td>{v["measured_null_p95"]:.3f}</td></tr>')
        parts.append('</table><p>窗口距离为各因果滞后处15通道归一化RMS的最大值；阈值2沿用先前协议，不由本次结果选取。</p><h3>同一Trial、近似同位置但历史不同的候选</h3>')
        if name!='virtual_rich' and r['within_trial_same_position_history_examples']:
            parts.append(f'<img style="width:100%" src="{name}_same_position.svg" alt="15个实测磁通道及名义XYZ的两个因果历史窗口">')
        parts.append('<pre>'+html.escape(json.dumps(r['within_trial_same_position_history_examples'][:2],ensure_ascii=False,indent=2))+'</pre>')
    parts+=['<h2>解释范围</h2><p>这是一项候选搜索，而非严格全局不可观测性证明。固定近似对随窗口被分开部分来自max距离的定义，不能仅以此宣称模型收益；需要实际错误关联和同流受控消融。实测空载对照只保留已有时序噪声，不是独立样本置信区间。仿真候选需要真实记录或力学—磁场求解核实。</p><p><a href="report.json">全部统计</a> · <a href="protocol.json">固定协议</a> · <a href="source_latest_all_current_pairs.json">源侧全部当前混淆对</a> · <a href="target_latest_all_current_pairs.json">Yellow全部当前混淆对</a></p>']
    (OUT/'index.html').write_text(''.join(parts),encoding='utf8')

def measured_pair_figure(name,example):
    """18 data panels; signed 15-channel fields plus nominal XYZ histories."""
    a=example['a'];b=example['b'];assert a['id']==b['id']
    with np.load(MATCHED/'streams'/f"{a['id']}_matched.npz",allow_pickle=False) as f:
        stream={k:f[k].copy() for k in f.files}
    endpoints=np.array([a['index'],b['index']]);raw,good=causal_windows(stream['b'],stream['t'],stream['reset'],endpoints,2.)
    assert good.all();fields=raw.reshape(2,21,15)[:,::-1,:]
    indices=np.searchsorted(stream['t'],stream['t'][endpoints,None]-np.arange(21)[None,:]*.1+1e-8,side='right')-1
    nominal=stream['q'][indices][:,::-1,:];curves=np.concatenate([fields,nominal],axis=2)
    width=1080;height=1210;parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="1080" height="1210" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#233b52} .axis{fill:none;stroke:#b6c6d6;stroke-width:1} .grid{fill:none;stroke:#e6edf4;stroke-width:1}</style>',
        '<text x="30" y="32" font-size="23" font-weight="700">Same nominal endpoint, different causal histories</text>',
        f'<text x="30" y="59" font-size="15">{html.escape(a["id"])} | {html.escape(a["mode"])} | A: {a["time_s"]:.1f} s; B: {b["time_s"]:.1f} s</text>',
        f'<text x="30" y="84" font-size="14">Command XYZ A: {tuple(round(x,3) for x in example["q_a_mm"])}; B: {tuple(round(x,3) for x in example["q_b_mm"])} mm</text>',
        '<path d="M30 104H61" stroke="#156bbb" stroke-width="2"/><text x="70" y="110" font-size="14">History A</text><path d="M205 104H236" stroke="#ed742e" stroke-width="2"/><text x="245" y="110" font-size="14">History B</text>',
        '<text x="420" y="110" font-size="13">Measured signed ΔB; no field/coordinate registration or time warping</text>']
    for k in range(18):
        col=k%3;row=k//3;x=30+col*350;y=137+row*171;px=x+47;py=y+26;pw=287;ph=112
        values=curves[:,:,k];lo=float(np.nanmin(values));hi=float(np.nanmax(values));pad=max((hi-lo)*.08,.02 if k>=15 else 1.)
        lo-=pad;hi+=pad
        label=f'Sensor {k//3+1} / B{"xyz"[k%3]} (µT)' if k<15 else f'Command {"XYZ"[k-15]} (mm)'
        parts.append(f'<text x="{x}" y="{y+14}" font-size="14" font-weight="700">{label}</text>')
        for h in range(3):
            yy=py+h*ph/2;v=hi-(hi-lo)*h/2
            parts.append(f'<path class="grid" d="M{px} {yy:.2f}H{px+pw}"/><text x="{px-7}" y="{yy+4:.2f}" text-anchor="end" font-size="10">{v:.2f}</text>')
        parts.append(f'<path class="axis" d="M{px} {py}V{py+ph}H{px+pw}"/>')
        for j,c in enumerate(('#156bbb','#ed742e')):
            points=' '.join(f'{px+i*pw/20:.2f},{py+(hi-v)*ph/(hi-lo):.2f}' for i,v in enumerate(values[j]))
            parts.append(f'<polyline points="{points}" fill="none" stroke="{c}" stroke-width="1.7"/>')
        for sec in (-2,-1,0):
            xx=px+(sec+2)*pw/2
            parts.append(f'<text x="{xx}" y="{py+ph+15}" text-anchor="middle" font-size="11">{sec}</text>')
    parts+=['<text x="30" y="1180" font-size="13">Time relative to each endpoint (s). Endpoints differ by ≤0.15 mm in nominal XYZ, not measured material position.</text>',
        '<text x="30" y="1200" font-size="13">Example selected by maximum same-Trial current-field disparity; illustrative extreme, not a population-average effect.</text></svg>']
    (OUT/f'{name}_same_position.svg').write_text(''.join(parts),encoding='utf8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--virtual-count',type=int,default=40);p.add_argument('--cap',type=int,default=3000);a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);groups,unloaded,provenance=load_groups(a.virtual_count)
    sigma=np.array(json.loads(NOISE.read_text(encoding='utf8'))['noise_sigma_uT'])
    control=null_control(unloaded,sigma)
    protocol={'windows_s':WINDOWS,'candidate_endpoint_stride':4,'deterministic_endpoint_cap_per_group':a.cap,
        'noise_sigma_source_uT':sigma.tolist(),'noise_metadata_sha256':sha(NOISE),'distance':'max causal-lag whitened 15D RMS',
        'no_future':True,'no_dtw':True,'no_pair_time_shift':True,'no_field_or_coordinate_registration':True,
        'position_separation':'XY>=1 mm OR |Z| difference>=0.3 mm; same-Trial time gap>=3 s',
        'current_threshold':2.,'same_position_tolerance_mm':.15,'null_control':control,'provenance':provenance,
        'target_parameter_selection':False,'models_retrained':False,'production_changed':False,
        'interpretation':'nominal command-reference candidate search; cross-Trial origins, B0/timing and simulator mismatch are not physical nonidentifiability proofs'}
    dump(OUT/'protocol.json',protocol);result={}
    for name,rows in groups.items():
        result[name]=search(make_bank(rows,sigma,cap=a.cap),control['conditional_max_distance_p95'],name)
        if name!='virtual_rich' and result[name]['within_trial_same_position_history_examples']:
            measured_pair_figure(name,result[name]['within_trial_same_position_history_examples'][0])
        print(name,result[name]['endpoints'],result[name]['windows']['0.0']['fixed_current_pairs'],flush=True)
    report={'status':'COMPLETE','datasets':result,'measured_unloaded_control':control,'target_fit_or_selection':False,
        'new_physical_law_proven':False,'virtual_hypotheses_require_independent_confirmation':True}
    dump(OUT/'report.json',report);render(report)
if __name__=='__main__':main()
