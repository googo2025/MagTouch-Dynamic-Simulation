"""Source-only active action proposals from fully fitted forward hypotheses.

No target/test errors, undertrained inverse ensemble, hardware commands or FEM
labels enter ranking. Disagreement is a hypothesis priority, not uncertainty.
"""
import json,html,time
import numpy as np
from training.run_forward_r361 import ROOT,load_row,dump,sha
from training.broad_trajectory_r370 import geometry,STYLES
from training.magnetic_forward_r361 import MagneticForward
from training.forward_complex_r363 import AxisForward

OUT=ROOT/'analysis/r381_source_active_candidates_20261002'
PATHS={'static':ROOT/'analysis/r361_command_forward_20261001/models/instantaneous.json',
       'joint_history':ROOT/'analysis/r361_command_forward_20261001/models/dynamic.json',
       'axis_history':ROOT/'analysis/r363_complex_forward_20261001/models/generator.json'}
MANIFEST=ROOT/'analysis/r361_command_forward_20261001/data_manifest.json'
NOISE=ROOT/'analysis/r363_complex_forward_20261001/synthetic_manifest.json'

def rank(rows):
    """Percentiles within a fixed candidate pool; do not use true XYZ errors."""
    keys=('field_disagreement_p95_noise_units','source_position_novelty_p95','history_fraction_p95')
    vals=np.array([[r[k] for k in keys] for r in rows])
    for i,r in enumerate(rows):r['priority_score']=float(np.mean([np.mean(vals[:,k]<=vals[i,k]) for k in range(len(keys))]))
    ordered=sorted(rows,key=lambda r:(-r['priority_score'],r['id']));chosen=[];styles={}
    for r in ordered:
        if r['qualified_source_envelope'] and styles.get(r['style'],0)<2:
            chosen.append(r);styles[r['style']]=styles.get(r['style'],0)+1
    return ordered,chosen

def nearest_position(q,reference):
    # A source-position coverage heuristic in millimetres; not dynamic-state
    # novelty and not a calibrated out-of-distribution probability.
    return np.concatenate([np.linalg.norm(p[:,None]-reference[None,:],axis=2).min(1) for p in np.array_split(q,max(1,(len(q)+127)//128))])

def geometry_svg(q,ident):
    # Orthographic mathematical coordinate projection, not a deformed magnet.
    az=np.deg2rad(-55);el=np.deg2rad(25);right=np.array([-np.sin(az),np.cos(az),0]);up=np.array([-np.sin(el)*np.cos(az),-np.sin(el)*np.sin(az),np.cos(el)])
    def p(v):return (310+34*float(np.dot(v,right)),210-34*float(np.dot(v,up)))
    parts=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 430"><rect width="640" height="430" fill="#f5f8fc"/>',
        f'<text x="20" y="28" font-family="Arial" font-size="18" fill="#19374d">{ident}: proposed command geometry</text>']
    for v in np.linspace(-5,5,6):
        for a,b in [(np.array([v,-5,0]),np.array([v,5,0])),(np.array([-5,v,0]),np.array([5,v,0]))]:
            x,y=p(a);xx,yy=p(b);parts.append(f'<path d="M{x:.2f} {y:.2f}L{xx:.2f} {yy:.2f}" fill="none" stroke="#dae4ef"/>')
    points=' '.join(f'{x:.2f},{y:.2f}' for x,y in map(p,q));parts.append(f'<polyline points="{points}" fill="none" stroke="#176fb5" stroke-width="1.5"/>')
    for name,v in [('X',np.array([5.5,0,0])),('Y',np.array([0,5.5,0])),('Z',np.array([0,0,3]))]:
        x,y=p(np.zeros(3));xx,yy=p(v);parts.append(f'<path d="M{x:.2f} {y:.2f}L{xx:.2f} {yy:.2f}" fill="none" stroke="#66839c"/><text x="{xx+5:.2f}" y="{yy:.2f}" font-family="Arial" font-size="14" fill="#19374d">{name} (mm)</text>')
    parts+=['<text x="20" y="405" font-family="Arial" font-size="12" fill="#19374d">Equal XYZ millimetre scale; known 5-mm single tip. Not a validated printer plan.</text></svg>']
    (OUT/'paths'/f'{ident}.svg').write_text(''.join(parts),encoding='utf8')

def render(report):
    body=['<!doctype html><meta charset="utf-8"><title>R381 源侧主动动作候选</title><style>body{font:16px/1.7 system-ui;max-width:1180px;padding:24px;margin:25px auto;color:#19384f}table{border-collapse:collapse;width:100%;margin:15px 0}th,td{padding:9px;border:1px solid #cbd9e3;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#eaf1f7}.note{background:#fff3da;padding:15px}.plots{display:grid;grid-template-columns:1fr 1fr;gap:18px}.plots img{width:100%}</style><h1>R381：只用源侧证据排序的新动作候选</h1><p>24条不同种子的自由三维路径；不是圆、米字轨迹参数的小幅改动。排序仅依据三种已闭式拟合的正向假设分歧、相对源训练位置的覆盖新颖度、预测历史成分。没有读取Yellow误差或真实保留复杂轨迹，也没有使用旧的短预算逆模型集合。</p><p class="note">排序是提出实验假设，不能保证哪个模型会错或哪个真实实验收益最大。正向模型仍是经验代理，空间包络通过不等于硬件安全。这里没有G-code，没有串口／打印机动作；实际执行前仍需触头、ContactZ、平台坐标、速度和安全高度审核。</p><h2>预筛选的多样动作</h2><div class="plots">']
    for r in report['selected']:body.append(f'<div><h3>{html.escape(r["id"])} | {html.escape(r["style"])}</h3><img src="paths/{r["id"]}.svg" alt="拟议XYZ几何路径"></div>')
    body+=['</div><h2>全部候选和排序依据</h2><table><tr><th>候选／路径族</th><th>源包络合格</th><th>正向分歧P95（噪声单位）</th><th>位置新颖度P95 mm</th><th>历史分量P95</th><th>池内优先分</th></tr>']
    for r in report['ranked']:
        body.append(f'<tr><td>{r["id"]}<br>{r["style"]}</td><td>{r["qualified_source_envelope"]}</td><td>{r["field_disagreement_p95_noise_units"]:.2f}</td><td>{r["source_position_novelty_p95"]:.3f}</td><td>{r["history_fraction_p95"]:.3f}</td><td>{r["priority_score"]:.3f}</td></tr>')
    body+=['</table><p>优先分为三项指标各自池内百分位的等权平均，每类最多两条；不含仿真真值定位误差。“源包络”要求至少90%的名义压入帧在生成器训练XYZ边界内，速度不超过2 mm/s，XY不超过±5 mm。并非已识别物理置信度。</p><h2>后续同流对比建议，不立即采集</h2><p>新路径与定点、纯Z、圆、米字、保留复杂三维动作分开汇报。冻结R373与虚拟扩集研究副本读取同一实测磁流，比原始XYZ、覆盖、无接触误输出、Z跳变及独立网格偏移；先源侧复核，再冻结后报告Yellow。实际采集需要用户另行安排。</p><p><a href="protocol.json">源侧协议和输入哈希</a> · <a href="report.json">完整候选参数与数据路径</a></p>']
    (OUT/'index.html').write_text(''.join(body),encoding='utf8')

def main():
    (OUT/'paths').mkdir(parents=True,exist_ok=True);manifest=json.loads(MANIFEST.read_text(encoding='utf8'))
    fitting=[r for r in manifest if r['role']=='train'];assert all(r['device']=='source' and r['mode']!='waypoint_xyz' for r in fitting)
    source=[load_row({**r,'path':ROOT/r['path']}) for r in fitting]
    reference=np.vstack([a['q'][a['primary']][::5] for a in source]);sigma=np.array(json.loads(NOISE.read_text(encoding='utf8'))['noise_sigma_uT'])
    models={name:AxisForward.load(p) if name=='axis_history' else MagneticForward.load(p) for name,p in PATHS.items()}
    protocol={'schema':'MagTouch-R381-SourceActiveProposals-1','candidate_count':24,'geometry_seed_base':3704000,
        'geometry_call':'R370 geometry(validation, 2000+i), seed3704000+i; roles rewritten as prospective_candidate, not fitting validation',
        'source_reference_trial_ids':[r['id'] for r in fitting],'source_reference_manifest_sha256':sha(MANIFEST),
        'forward_hypotheses_sha256':{k:sha(p) for k,p in PATHS.items()},'noise_source_sha256':sha(NOISE),
        'priority':['predicted field hypothesis disagreement P95 in frozen noise units','nearest SOURCE TRAIN XYZ distance P95','axis-history / predicted-field norm P95'],
        'rank_rule':'equal within-pool percentile mass, at most two per construction style',
        'parameter_selection_after_target_outcomes':False,'target_or_reserved_complex_loaded':False,'inverse_training_used':False,
        'uncertainty_calibrated':False,'physical_teacher_validated':False,'stage_plan_executable':False,'hardware_operated':False}
    p=OUT/'protocol.json'
    if p.exists():assert json.loads(p.read_text(encoding='utf8'))==protocol
    else:dump(p,protocol)
    rows=[]
    for i in range(24):
        ident=f'ACTIVE381_{i:03d}';q,t,info=geometry('validation',2000+i);info.update(role='prospective_candidate',geometry_id=ident)
        fields={k:m.rollout(q,t) for k,m in models.items()};contact=q[:,2]>=.4
        stack=np.stack(list(fields.values()))/sigma[None,None,:]
        disagreement=np.sqrt(np.mean(np.var(stack,axis=0),axis=1))
        inside=models['axis_history'].support(q);history=fields['axis_history']-models['axis_history'].static(q)
        history_fraction=np.linalg.norm(history,axis=1)/np.maximum(np.linalg.norm(fields['axis_history'],axis=1),50.)
        speed=np.linalg.norm(np.diff(q,axis=0),axis=1)/np.diff(t);novel=nearest_position(q[contact],reference)
        qualified=bool(inside[contact].mean()>=.9 and speed.max()<=2.0001 and np.abs(q[:,:2]).max()<=5.0001 and q[:,2].max()<=2.4001)
        dest=OUT/'paths'/f'{ident}.npz';np.savez_compressed(dest,q_mm=q,t_s=t,primary_nominal=contact,inside_source_envelope=inside,**{k+'_field_uT':v for k,v in fields.items()})
        r={'id':ident,**info,'qualified_source_envelope':qualified,'inside_source_fraction':float(inside[contact].mean()),
            'field_disagreement_p95_noise_units':float(np.quantile(disagreement[contact],.95)),
            'source_position_novelty_p95':float(np.quantile(novel,.95)),'history_fraction_p95':float(np.quantile(history_fraction[contact],.95)),
            'path':dest.relative_to(ROOT).as_posix(),'sha256':sha(dest)}
        rows.append(r);geometry_svg(q,ident);print(ident,r['style'],qualified,flush=True)
    ranked,selected=rank(rows);report={'status':'COMPLETE','ranked':ranked,'selected':selected,'source_only':True,
        'no_true_reconstruction_error_in_priority':True,'not_a_posterior_uncertainty':True,'not_executable_stage_plan':True,'hardware_operated':False}
    dump(OUT/'report.json',report);render(report)
    print('selected',len(selected),'families',sorted({r['style'] for r in selected}),flush=True)
if __name__=='__main__':main()
