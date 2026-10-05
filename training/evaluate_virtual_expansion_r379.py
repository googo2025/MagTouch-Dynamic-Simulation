"""Post-freeze real/virtual full-chain checks of the R379 XY research head.

Production R373, action heads, slow state, maps and depth parameters stay fixed.
An XY change may nevertheless alter permission and depth evidence numerically.
Cached R374 real baseline is compared at the same 10-Hz input convention.
"""
import argparse,json,time,sys,os,html
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from training.virtual_expansion_r379 import OUT,OLD,ROOT,ResearchXY,PlateauCausalXY
from training.run_forward_r361 import dump,sha
from training.research_formal_r365 import metrics,macro
from training.matched_real_virtual_r374 import OUT as REAL
sys.path.insert(0,str(ROOT/'code'))
from live_retrained_reconstruction_r373 import DirectTrainedEngineR373,COMPONENTS
from live_algorithm_ui_r92 import R92_PROFILE_PATH
from live_algorithm_ui_r126 import PROFILE_KEY
from replay_backtest import load_profile

def frozen(count):
    folder=OUT/f'arm_{count}';f=json.loads((folder/'freeze.json').read_text(encoding='utf8'))
    assert f['status']=='COMPLETED_PLATEAU','Do not test a budget-truncated training run'
    assert not f['test_opened'] and not f['target_fit_or_selection'] and not f['production_replaced']
    assert sha(folder/'best_model.npz')==f['model_sha256']
    return f

def replay(a,modelpath=None):
    p=load_profile(R92_PROFILE_PATH,PROFILE_KEY);e=DirectTrainedEngineR373(p,p['defaults']['threshold'],p['defaults']['min_cosine'],p['defaults']['smoothing_ms'],70.)
    if modelpath is not None:
        e.memory_decoder.xy_model=ResearchXY.load(modelpath);e.candidate_decoder.xy_model=ResearchXY.load(modelpath)
    e.set_baseline(np.zeros(15));n=len(a['t']);pred=np.full((n,3),np.nan);ok=np.zeros(n,bool)
    permission=np.full(n,np.nan);support=np.full(n,np.nan);route=np.empty(n,dtype='U32')
    for i,t in enumerate(a['t']):
        if a['reset'][i]:e.set_baseline(np.zeros(15))
        eligible=bool(a['input_eligible'][i]);r=e.push(int(round(t*1e9))+10_000_000_000,a['b'][i] if eligible else np.zeros(15),eligible=eligible)
        ok[i]=r['valid'];extra=r['extra'];route[i]=extra.get('localization_path','unknown')
        if ok[i]:pred[i]=r['xyz']
        v=extra.get('localization_permission_score');permission[i]=np.nan if v is None else v
        v=extra.get('final_depth_evidence',{}).get('source_support_distance');support[i]=np.nan if v is None else v
    return {'pred':pred,'ok':ok,'permission':permission,'support':support,'route':route}

def quantile(x):
    x=np.asarray(x);x=x[np.isfinite(x)]
    return {'n':len(x),'median':float(np.median(x)),'p95':float(np.quantile(x,.95))} if len(x) else {'n':0}

def trial(task):
    count,kind,entry=task;f=frozen(count);start=time.perf_counter();folder=OUT/f'arm_{count}'/'evaluation';dest=folder/f"{kind}_{entry['id']}.npz";meta=dest.with_suffix('.json')
    if meta.exists():
        r=json.loads(meta.read_text(encoding='utf8'));assert r['candidate_sha256']==f['model_sha256'];return r
    if kind=='real':
        path=REAL/'streams'/f"{entry['id']}_matched.npz"
        with np.load(path,allow_pickle=False) as s:a={k:s[k].copy() for k in s.files}
        # Exactly the prior R374 matched input contract; command is not a feature.
        a['input_eligible']=a['valid']&np.isfinite(a['q']).all(1)
        baseline={'pred':a['real_new'],'ok':a['real_new_ok'],'permission':a['real_new_permission'],'support':a['real_new_support']}
    else:
        path=OLD/'virtual'/f"{entry['id']}.npz";assert sha(path)==entry['sha256']
        with np.load(path,allow_pickle=False) as s:a={k:s[k].copy() for k in s.files}
        a['safe']=a['valid']&np.isfinite(a['q']).all(1)&(a['q'][:,2]<=-.2);a['input_eligible']=a['valid']
        baseline=replay(a)
    candidate=replay(a,OUT/f'arm_{count}'/'best_model.npz');common=baseline['ok']&candidate['ok']
    scores={name:metrics(v['pred'],a['q'],a['t'],a['primary'],v['ok']) for name,v in [('baseline',baseline),('candidate',candidate)]}
    scores.update({name+'_common':metrics(v['pred'],a['q'],a['t'],a['primary'],common) for name,v in [('baseline',baseline),('candidate',candidate)]})
    for name,v in [('baseline',baseline),('candidate',candidate)]:
        scores[name]['safe_frames']=int(a['safe'].sum());scores[name]['safe_outputs']=int((a['safe']&v['ok']).sum())
        scores[name]['permission_primary']=quantile(v['permission'][a['primary']]);scores[name]['z_support_primary']=quantile(v['support'][a['primary']])
    changes={'primary_outputs_lost':int((a['primary']&baseline['ok']&~candidate['ok']).sum()),'primary_outputs_added':int((a['primary']&~baseline['ok']&candidate['ok']).sum())}
    for key in ('permission','support'):
        finite=a['primary']&np.isfinite(baseline[key])&np.isfinite(candidate[key]);changes[key+'_absolute_change']=quantile(np.abs(candidate[key][finite]-baseline[key][finite]))
    err=np.linalg.norm(candidate['pred']-a['q'],axis=1);ix=np.flatnonzero(a['primary']&candidate['ok']&np.isfinite(err));ix=ix[np.argsort(err[ix])[-10:][::-1]]
    worst=[{'index':int(i),'time_s':float(a['t'][i]),'reference_xyz_mm':a['q'][i].tolist(),'candidate_xyz_mm':candidate['pred'][i].tolist(),
        'baseline_xyz_mm':baseline['pred'][i].tolist() if baseline['ok'][i] else None,'permission':float(candidate['permission'][i]) if np.isfinite(candidate['permission'][i]) else None,
        'z_support':float(candidate['support'][i]) if np.isfinite(candidate['support'][i]) else None,'route':str(candidate['route'][i])} for i in ix]
    record={'id':entry['id'],'kind':kind,'device':entry.get('device','virtual'),'mode':entry.get('mode',entry.get('style')),
        'scores':scores,'changes':changes,'worst_accepted_primary_frames':worst,'candidate_sha256':f['model_sha256'],
        'input_sha256':sha(path),'seconds':time.perf_counter()-start}
    stream={k:a[k] for k in ('t','q','primary','safe','reset')}
    for name,v in [('baseline',baseline),('candidate',candidate)]:
        for key in ('pred','ok','permission','support'):stream[name+'_'+key]=v[key]
    np.savez_compressed(dest,**stream);dump(meta,record)
    print(count,kind,entry['id'],record['scores']['candidate'].get('xy_mean_mm'),'sec',round(record['seconds'],1),flush=True)
    return record

def render(count,report):
    f=report['freeze'];body=['<!doctype html><meta charset="utf-8"><title>R379 充分训练虚拟扩集</title><style>body{font:16px/1.65 system-ui;max-width:1100px;margin:32px auto;padding:24px;color:#16324a}table{border-collapse:collapse;width:100%;margin:16px 0}td,th{border:1px solid #cdd9e2;padding:10px;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#eaf1f8}.note{padding:16px;background:#fff3d7}h2{margin-top:30px}</style>',
        f'<h1>R379：{count}条丰富虚拟路径的充分训练研究副本</h1><p>从充分训练的R373 XY权重出发；完整遍历、同架构、同源侧验证、seed 7。Z头、状态识别、门控规则和磁图参数不变，但其数值可能与XY耦合。</p>',
        f'<p>训练样本 {f["training_samples"]:,}（虚拟 {f["virtual_samples"]:,}）；实际训练 {f["epochs_run"]} epoch，最佳 {f["best_epoch"]}；{f["optimizer_updates"]:,} 次更新。停止：最低学习率平台期。源侧验证指标 {f["baseline_validation_objective"]:.4f} → {f["best_validation_objective"]:.4f} mm。</p>',
        '<p class="note">下面是冻结后的回测，不是新的在线采集。已测真实复杂路径继续排除拟合与选模；Yellow只评价。不以去除平移后的改善替代原始误差，也不以拒绝输出计零误差。独立虚拟测试使用不同几何种子，但仍共享经验生成器，不等于真实器件收益。</p>']
    for name,r in report['groups'].items():
        body.append('<h2>'+html.escape(name)+'</h2><table><tr><th>完整链路指标</th><th>冻结R373</th><th>虚拟扩集XY研究副本</th></tr>')
        for k,label in [('coverage','有效覆盖'),('xy_mean_mm','原始XY平均误差 mm'),('z_mae_mm','原始Z MAE mm'),('xyz_mean_mm','原始XYZ平均误差 mm'),('xy_translation_removed_mm','逐Trial去常值XY诊断 mm')]:
            vals=[]
            for model in ('baseline','candidate'):
                v=r['macro'][model].get(k);vals.append('—' if v is None else f'{100*v:.2f}%' if k=='coverage' else f'{v:.4f}')
            body.append('<tr><td>'+label+'</td><td>'+vals[0]+'</td><td>'+vals[1]+'</td></tr>')
        body.append('</table><p>完整Trial等权平均；配对共同输出指标、Z跳变、许可与支持度、最差片段及独立网格偏移诊断见JSON。</p>')
    if (OUT/f'arm_{count}'/'coupling_offset_audit.html').exists():
        body.append('<p><a href="coupling_offset_audit.html">浅层、Z、许可和整体偏移的专门核对</a></p>')
    if (OUT/f'arm_{count}'/'figures/manifest.json').exists():
        body.append('<h2>同磁流轨迹对照：原始未配准坐标</h2><p>固定展示圆、米字、纯Z、真实复杂三维与每种虚拟路径族的第一个测试样本，不按误差挑选图。断点表示没有输出。</p>')
        for name in ('real_complex_two_views','real_standard_actions','virtual_fixed_examples','virtual_full_timeline_z'):
            body.append(f'<img style="width:100%;margin:12px 0" src="figures/{name}.png" alt="{name}">')
    body.append('<p><a href="evaluation_report.json">全部结果</a> · <a href="history.json">逐轮训练曲线数据</a> · <a href="freeze.json">权重冻结</a> · <a href="feature_manifest.json">训练数据来源和样本量</a></p>')
    (OUT/f'arm_{count}'/'index.html').write_text(''.join(body),encoding='utf8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--count',type=int,choices=(100,500,1000),default=100);ap.add_argument('--workers',type=int,default=3);a=ap.parse_args()
    f=frozen(a.count);folder=OUT/f'arm_{a.count}';(folder/'evaluation').mkdir(exist_ok=True)
    real=json.loads((REAL/'manifest.json').read_text(encoding='utf8'));catalog=json.loads((OLD/'catalog.json').read_text(encoding='utf8'))
    virtual=[r for r in catalog['trials'] if r['role']=='test'];assert len(virtual)==64
    protocol={'candidate_sha256':f['model_sha256'],'freeze_sha256':sha(folder/'freeze.json'),'real_ids':[r['id'] for r in real],'virtual_ids':[r['id'] for r in virtual],
        'input_cadence':'10Hz controlled causal replay; not native UI sampling parity','baseline':'R373 frozen full chain, no more real-only optimization',
        'changed':'both research XY routes only','unchanged_parameters':['Z networks','action head','permission policy','slow state','magnetic atlas','branch guard'],
        'full_chain_recomputed':True,'real_baseline':'exact cached R374 controlled stream outputs','source_selection':'completed earlier on three original source validation Trials',
        'raw_error_and_trial_translation_diagnostics_separate':True,'safe_refusal_not_zero_error':True,'target_tuning':False,'production_changed':False,
        'scope':'latest real test was previously inspected; this is fixed heldout replay, not new blind collection'}
    protocolfile=folder/'evaluation_protocol.json'
    if protocolfile.exists():assert json.loads(protocolfile.read_text(encoding='utf8'))==protocol
    else:dump(protocolfile,protocol)
    dump(folder/'evaluation_status.json',{'status':'RUNNING','pid':os.getpid()})
    tasks=[(a.count,'real',r) for r in real]+[(a.count,'virtual',r) for r in virtual]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:rows=list(pool.map(trial,tasks))
    groups={}
    for device in ('source','target','virtual'):
        rs=[r for r in rows if r['device']==device and r['mode']!='none'];groups[device]={'trials':len(rs),'macro':macro([r['scores'] for r in rs],['baseline','candidate','baseline_common','candidate_common'])}
        groups[device]['safe_outputs']={model:sum(r['scores'][model]['safe_outputs'] for r in rows if r['device']==device) for model in ('baseline','candidate')}
    # Independently collected same-session 1.4-mm grid offsets: diagnostic only.
    grid={}
    for device,tid in [('source','source_000002'),('target','target_000011')]:
        g=next(r for r in rows if r['id']==tid);records=[]
        for r in rows:
            if r['device']!=device or r['mode']=='none':continue
            with np.load(folder/'evaluation'/f"real_{r['id']}.npz",allow_pickle=False) as s:
                scores={}
                for model in ('baseline','candidate'):
                    bias=np.array(g['scores'][model]['bias_xyz_mm']);p=s[model+'_pred']-bias
                    scores[model]=metrics(p,s['q'],s['t'],s['primary'],s[model+'_ok'])
                records.append({'id':r['id'],'scores':scores})
        grid[device]={'calibration_trial_id':tid,'offsets':{m:g['scores'][m]['bias_xyz_mm'] for m in ('baseline','candidate')},'rows':records}
    report={'status':'COMPLETE','freeze':f,'groups':groups,'independent_grid_offset_diagnostic':grid,'records':rows,
        'production_changed':False,'xy_only':True,'target_tuned':False}
    dump(folder/'evaluation_report.json',report);render(a.count,report);dump(folder/'evaluation_status.json',{'status':'COMPLETE','pid':os.getpid(),'trials':len(rows)})
    print(json.dumps(groups,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
