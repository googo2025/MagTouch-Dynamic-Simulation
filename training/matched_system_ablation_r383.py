"""Frozen fully-trained scientific-module ablations on matched R374 fields.

This fills the complete F/M/G/N/U/Final sequence, for both measured B and the
same command path's virtual B. It is not a historical UI comparison and does
not select any augmentation checkpoint or refit a generator.
"""
from concurrent.futures import ProcessPoolExecutor
import argparse, html, json, os, sys, time
import numpy as np
from training.matched_real_virtual_r374 import ROOT, OUT as MATCHED, GEN
from training.research_formal_r365 import metrics, macro
from training.train_source_to_plateau_r372 import sha, dump
from training.had239_field_memory_depth_guard_runtime_r126 import DepthGuardFieldMemoryDecoder
from training.had239_guarded_neural_runtime_r166 import GuardedBranchMemoryDecoder
from training.had239_r133_coverage_xy_runtime import CoverageGuardedNeuralDecoder
sys.path.insert(0,str(ROOT/'code'))
from live_retrained_reconstruction_r373 import DirectTrainedEngineR373, PlateauCausalXY, COMPONENTS, frozen_components
from live_algorithm_ui_r92 import R92_PROFILE_PATH
from live_algorithm_ui_r126 import PROFILE_KEY
from replay_backtest import load_profile

OUT=ROOT/'analysis/r383_matched_full_system_ablation_20261002'
LABELS={'F':'Frozen static magnetic atlas', 'M':'Atlas + slow field memory',
        'G':'+ magnetic-evidence branch protection',
        'N':'+ fully-trained causal XY head',
        'U':'Full system without bounded Z increment',
        'Final':'Frozen fully-trained complete R373'}

def protocol():
    frozen_components();OUT.mkdir(exist_ok=True)
    source=json.loads((MATCHED/'manifest.json').read_text(encoding='utf8'))
    p={'schema':'R383-Matched-Scientific-Ablations-1','baseline_xy_z_sha256':COMPONENTS,
       'real_manifest_sha256':sha(MATCHED/'manifest.json'),
       'matched_protocol_sha256':sha(MATCHED/'protocol.json'),
       'generator_sha256':sha(GEN),'complete_real_trial_ids':[r['id'] for r in source],
       'component_labels':LABELS,'no_parameters_fitted_or_selected':True,
       'same_command_path_in_real_and_virtual':True,
       'shared_output_permission':'Fixed final R373 output mask within each real/virtual stream; real and virtual masks may differ',
       'component_coverage':'F/M/G/N may fail despite shared final permission; report own coverage AND common-all-stage frames',
       'N_weights':'R373 PlateauCausalXY explicitly replaces historical XY weights in isolated N decoder',
       'U':'Actual independent full R373 engine, gain0 only; accepted mask and XY checked against cached full system',
       'reference':'Logged command convention, not encoder or extreme-shallow actual-contact truth',
       'virtual':'Same frozen SOURCE-fitted generator for both source and target command paths, not a fitted target simulator',
       'cadence':'10Hz controlled R374 inputs; not native online outputs',
       'target_training_or_selection':False,'production_replaced':False,'hardware_operated':False,
       'no_posthoc_time_shift_or_coordinate_registration':True,
       'raw_and_constant_translation_diagnostics_separate':True}
    dest=OUT/'protocol.json'
    if dest.exists():assert json.loads(dest.read_text(encoding='utf8'))==p
    else:dump(dest,p)
    return source

def replay(a,kind):
    b=a['b'] if kind=='real' else a['virtual_b']
    prefix='real_new' if kind=='real' else 'virtual_new'
    accepted=a[prefix+'_ok'];full=a[prefix]
    p=load_profile(R92_PROFILE_PATH,PROFILE_KEY)
    u=DirectTrainedEngineR373(p,p['defaults']['threshold'],p['defaults']['min_cosine'],p['defaults']['smoothing_ms'],70.)
    u.final_refiner.selection=dict(u.final_refiner.selection,gain=0.)
    u.set_baseline(np.zeros(15))
    decoders={'F':DepthGuardFieldMemoryDecoder.from_report(),
              'M':DepthGuardFieldMemoryDecoder.from_report(),
              'G':GuardedBranchMemoryDecoder.from_report(),
              'N':CoverageGuardedNeuralDecoder.from_report()}
    decoders['F'].strength=0.
    decoders['N'].xy_model=PlateauCausalXY()
    np.testing.assert_array_equal(decoders['N'].xy_model.mean,u.memory_decoder.xy_model.mean)
    results={name:np.full_like(full,np.nan) for name in LABELS}
    results['Final']=full.copy();ok={name:np.zeros(len(full),bool) for name in LABELS}
    ok['Final']=accepted.copy();epoch=0
    eligible=a['valid']&np.isfinite(a['q']).all(1) # Identical logged R374 contract, not a position input.
    for i,t in enumerate(a['t']):
        if a['reset'][i]:
            epoch+=1;u.set_baseline(np.zeros(15))
            for d in decoders.values():d.reset()
        now=int(round(t*1e9))+10_000_000_000
        field=b[i] if eligible[i] else np.zeros(15)
        without=u.push(now,field,eligible=bool(eligible[i]))
        assert bool(without['valid'])==bool(accepted[i]),'Z increment unexpectedly changes output mask'
        ok['U'][i]=without['valid']
        if without['valid']:
            results['U'][i]=without['xyz']
            np.testing.assert_allclose(results['U'][i,:2],full[i,:2],rtol=0,atol=1e-12)
        for name,d in decoders.items():
            row=d.predict_frame(field,time_ns=now,b0_id=epoch,stream_id=0,
                                contact_active=bool(accepted[i]))
            if row is not None:
                value=np.asarray(row['xyz_mm'],float)
                assert accepted[i] and np.isfinite(value).all()
                results[name][i]=value;ok[name][i]=True
    return results,ok

def matched_mechanisms(rows):
    """Same exact timestamps across both field kinds and all six stages."""
    steps=(('F','M','xy_mean_mm'),('M','G','xy_mean_mm'),('G','N','xy_mean_mm'),
           ('N','U','z_mae_mm'),('U','Final','z_mae_mm'))
    result={}
    for device in ('source','target'):
        cases=[]
        for row in rows:
            if row['device']!=device or row['field_kind']!='real' or row['mode']=='none':continue
            paths=[OUT/'streams'/f"{kind}_{row['id']}.npz" for kind in ('real','virtual')]
            with np.load(paths[0],allow_pickle=False) as r,np.load(paths[1],allow_pickle=False) as v:
                for key in ('t','q','primary'):np.testing.assert_array_equal(r[key],v[key])
                joint=np.logical_and.reduce([s[m+'_ok'] for s in (r,v) for m in LABELS])
                scores={kind:{m:metrics(s[m],s['q'],s['t'],s['primary'],joint) for m in LABELS}
                        for kind,s in (('real',r),('virtual',v))}
            changes=[]
            for before,after,key in steps:
                d={}
                for kind in ('real','virtual'):
                    b=scores[kind][before].get(key);a=scores[kind][after].get(key)
                    d[kind]=None if a is None or b is None else float(a-b)
                changes.append({'before':before,'after':after,'metric':key,'delta_mm':d,
                    'direction_agrees':None if any(x is None for x in d.values()) else bool(np.sign(d['real'])==np.sign(d['virtual']))})
            cases.append({'id':row['id'],'mode':row['mode'],'joint_scores':scores,'changes':changes})
        stage=[]
        for before,after,key in steps:
            pairs=[next(x for x in c['changes'] if x['before']==before and x['after']==after) for c in cases]
            usable=[x for x in pairs if x['direction_agrees'] is not None]
            stage.append({'before':before,'after':after,'metric':key,'trials_with_joint_output':len(usable),
                          'same_direction_trials':sum(x['direction_agrees'] for x in usable),
                          'trial_equal_delta_mm':{kind:float(np.mean([x['delta_mm'][kind] for x in usable])) if usable else None
                                                 for kind in ('real','virtual')}})
        result[device]={'cases':cases,'stage_summary':stage,
                        'joint_mask':'Each Trial uses exact timestamps accepted by every stage in both field kinds; coverage retained per Trial',
                        'negative_delta_means_error_reduced':True,'target_is_descriptive_not_generator_validation':device=='target'}
    return result

def trial(task):
    entry,kind=task;frozen_components()
    source=MATCHED/'streams'/f"{entry['id']}_matched.npz"
    dest=OUT/'streams'/f"{kind}_{entry['id']}.npz";side=dest.with_suffix('.json')
    if side.exists():
        r=json.loads(side.read_text(encoding='utf8'))
        assert r['input_sha256']==sha(source) and r['protocol_sha256']==sha(OUT/'protocol.json')
        assert r['output_sha256']==sha(dest)
        return r
    start=time.perf_counter()
    with np.load(source,allow_pickle=False) as f:a={k:f[k].copy() for k in f.files}
    pred,ok=replay(a,kind);common=np.logical_and.reduce(list(ok.values()))
    scores={};common_scores={}
    for name in LABELS:
        scores[name]=metrics(pred[name],a['q'],a['t'],a['primary'],ok[name])
        scores[name]['safe_frames']=int(a['safe'].sum())
        scores[name]['safe_outputs']=int((a['safe']&ok[name]).sum())
        common_scores[name]=metrics(pred[name],a['q'],a['t'],a['primary'],common)
    shallow=np.isfinite(a['q']).all(1)&(a['q'][:,2]>0)&(a['q'][:,2]<.4)
    stream={k:a[k] for k in ('t','q','primary','safe','reset')}
    for name in LABELS:stream[name]=pred[name];stream[name+'_ok']=ok[name]
    np.savez_compressed(dest,**stream)
    r={'id':entry['id'],'device':entry['device'],'mode':entry['mode'],'field_kind':kind,
       'scores':scores,'all_common_scores':common_scores,'shared_final_permission_frames':int(ok['Final'].sum()),
       'nominal_shallow_frames':int(shallow.sum()),
       'nominal_shallow_outputs':{name:int((shallow&ok[name]).sum()) for name in LABELS},
       'input_sha256':sha(source),'protocol_sha256':sha(OUT/'protocol.json'),
       'output_sha256':sha(dest),'seconds':time.perf_counter()-start}
    dump(side,r);print('R383',kind,entry['id'],r['scores']['Final'].get('xy_mean_mm'),flush=True)
    return r

def render(report):
    body=['<!doctype html><meta charset="utf-8"><title>同路径真实与虚拟系统消融</title>',
          '<style>body{font:16px/1.7 system-ui;color:#16374c;max-width:1250px;margin:30px auto;padding:25px}table{width:100%;border-collapse:collapse;margin:18px 0}th,td{border:1px solid #cbd7df;padding:9px;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#eaf1f7}.note{padding:16px;background:#fff4db}</style>',
          '<h1>同一路径真实磁场与虚拟磁场的完整机制消融</h1>',
          '<p>冻结充分训练的R373及所有原有模块，按静态磁图、慢记忆、磁分支保护、因果XY、深度证据裁决与有界Z增量比较。不引入其他算法，不重新拟合；N使用充分训练的XY而非旧权重。</p>',
          '<p class="note">这是10 Hz受控回放，不是原生在线性能。每条磁流共享最终模型的许可状态，隔离定位模块贡献；不同模块仍可能缺少位置输出，因此同时报告各自覆盖率和全阶段共同输出。所有虚拟磁场均来自源侧经验生成器，目标动作路径的虚拟结果不是对Yellow器件已验证的物理仿真。</p>']
    for device in ('source','target'):
        for kind in ('real','virtual'):
            group=report['groups'][device+'_'+kind]
            body.append(f'<h2>{device} / {kind}</h2><table><tr><th>模块</th><th>覆盖率</th><th>原始XY mm</th><th>原始Z MAE mm</th><th>共同输出XY mm</th><th>安全抬离误输出</th></tr>')
            for name,label in LABELS.items():
                r=group['macro'][name];c=group['all_common_macro'][name]
                def f(v):return '—' if v is None else f'{v:.4f}'
                coverage=r.get('coverage')
                body.append(f'<tr><td>{name} {html.escape(label)}</td><td>'+('—' if coverage is None else f'{coverage:.2%}')+f'</td><td>{f(r.get("xy_mean_mm"))}</td><td>{f(r.get("z_mae_mm"))}</td><td>{f(c.get("xy_mean_mm"))}</td><td>{group["safe_outputs"][name]}</td></tr>')
            body.append('</table>')
    body.append('<p>完整Trial等权；逐动作原始误差、常值偏移诊断、恒深跳变及浅层输出见JSON。不会把拒绝输出当零误差，也不保证每一组件都在每一动作上改善。</p><p><a href="report.json">全部指标</a> · <a href="protocol.json">冻结协议与来源</a></p>')
    body.append('<h2>是否复现同一改进方向：严格共同时间帧</h2><p>以下每条Trial只用真实／虚拟两种磁流中全部六个阶段都输出的位置帧。表中负差值表示误差降低，完整共同输出覆盖和逐动作反例保留在JSON；目标路径虚拟磁场仍是源生成器输出。</p><p>N→U比较独立位置路径与整合系统深度路径，还包含主／临时位置状态生命周期差异，属于系统整合对照，不把这项差值归因于单个Z组件。U→Final才是每帧许可、XY完全相同、只关闭有界Z增量的严格消融。</p><table><tr><th>器件动作来源</th><th>组件变化</th><th>指标</th><th>真实误差差值 mm</th><th>虚拟误差差值 mm</th><th>方向一致Trial／有效Trial</th></tr>')
    for device,data in report['matched_mechanisms'].items():
        for s in data['stage_summary']:
            def f(v):return '—' if v is None else f'{v:+.4f}'
            body.append(f'<tr><td>{device}</td><td>{s["before"]} → {s["after"]}</td><td>{s["metric"]}</td><td>{f(s["trial_equal_delta_mm"]["real"])}</td><td>{f(s["trial_equal_delta_mm"]["virtual"])}</td><td>{s["same_direction_trials"]} / {s["trials_with_joint_output"]}</td></tr>')
    body.append('</table>')
    source=report['matched_mechanisms']['source']['stage_summary']
    memory=next(s for s in source if s['before']=='F' and s['after']=='M')
    causal=next(s for s in source if s['before']=='G' and s['after']=='N')
    increment=next(s for s in source if s['before']=='U' and s['after']=='Final')
    body.append('<h2>这组结果能说明什么</h2><p class="note">源器件的同路径对照复现了慢磁场记忆和因果XY的主要改进方向：'
        f'慢记忆使Trial等权XY误差分别降低{-memory["trial_equal_delta_mm"]["real"]:.3f} / {-memory["trial_equal_delta_mm"]["virtual"]:.3f} mm（真实／虚拟）；'
        f'因果XY分别降低{-causal["trial_equal_delta_mm"]["real"]:.3f} / {-causal["trial_equal_delta_mm"]["virtual"]:.3f} mm，方向一致{causal["same_direction_trials"]}/{causal["trials_with_joint_output"]}条。'
        f'但有界Z增量仅{increment["same_direction_trials"]}/{increment["trials_with_joint_output"]}条方向一致；分支保护在本组的平均差异很小。'
        '因此这是对主要静态—动态问题和部分解决机制的经验复现，不是所有组件／所有器件都被高保真复刻。目标器件的部分改进方向不一致，必须保留，不能把源生成器当成Yellow仿真真值。</p>')
    figures=OUT/'figures'
    if (figures/'manifest.json').exists():
        manifest=json.loads((figures/'manifest.json').read_text(encoding='utf8'))
        if manifest.get('status')=='VISUAL_QA_PASSED' and manifest['report_sha256']==sha(OUT/'report.json'):
            body.append('<h2>固定动作的真实／虚拟轨迹对照</h2><p>按动作和原始记录顺序固定示例；无误差挑图、坐标平移、平滑或时间对齐。全部图已检查，下面展示源器件：目标动作的完整数值仍保留在表和JSON中。</p>')
            captions={'source_xy_mechanism_progression':'八方向运动：静态磁图、慢记忆、分支保护与因果XY。',
                      'source_z_mechanism_progression':'垂直压入／撤出和恒深圆：完整时间流与Z分支。',
                      'same_source_path_real_virtual_xyz':'同一复杂三维指令路径：真实磁场与生成磁场，两种观察角度。'}
            for name,caption in captions.items():
                body.append(f'<figure style="margin:25px 0"><img style="width:100%" src="figures/{name}.png" alt="{caption}"><figcaption>{caption}</figcaption></figure>')
    (OUT/'index.html').write_text(''.join(body),encoding='utf8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    entries=protocol();(OUT/'streams').mkdir(exist_ok=True)
    if args.smoke:
        entry=next(r for r in entries if r['id']=='source_000002')
        rows=[trial((entry,kind)) for kind in ('real','virtual')]
        dump(OUT/'smoke_validation.json',{'status':'PASSED','input_id':entry['id'],
             'stream_count':len(rows),'U_mask_and_XY_equals_final_checked':True,
             'N_uses_R373_XY':True,'target_read':False})
        return
    dump(OUT/'status.json',{'status':'RUNNING','pid':os.getpid(),'total_streams':2*len(entries)})
    tasks=[(entry,kind) for entry in entries for kind in ('real','virtual')]
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for r in pool.map(trial,tasks):
            rows.append(r);dump(OUT/'status.json',{'status':'RUNNING','pid':os.getpid(),
                                                 'complete_streams':len(rows),'total_streams':len(tasks)})
    groups={}
    for device in ('source','target'):
        for kind in ('real','virtual'):
            all_rows=[r for r in rows if r['device']==device and r['field_kind']==kind]
            contact=[r for r in all_rows if r['mode']!='none']
            groups[device+'_'+kind]={'trials':len(contact),'macro':macro([r['scores'] for r in contact],LABELS),
                'all_common_macro':macro([r['all_common_scores'] for r in contact],LABELS),
                'safe_frames':sum(r['scores']['Final']['safe_frames'] for r in all_rows),
                'safe_outputs':{m:sum(r['scores'][m]['safe_outputs'] for r in all_rows) for m in LABELS}}
    report={'status':'COMPLETE','groups':groups,'records':rows,'labels':LABELS,
            'source_only_frozen_components':True,'production_replaced':False,'target_tuned':False,
            'protocol_sha256':sha(OUT/'protocol.json'),'scope':'Same-path controlled mechanistic ablation, not native online results or augmentation selection'}
    report['matched_mechanisms']=matched_mechanisms(rows)
    dump(OUT/'report.json',report);render(report)
    dump(OUT/'status.json',{'status':'COMPLETE','pid':os.getpid(),'complete_streams':len(rows)})

if __name__=='__main__':main()
