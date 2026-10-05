"""Read-only diagnostic of completed R398 D acquisition, no hardware or fitting."""
from pathlib import Path
import csv, hashlib, html, json, re
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'data/S20261003_A/Plan_20261003_010839_R391_EXEC_75a70606cd72_1b7bb8701c02_2294d09__mag_5x5-had-239'
OUT=ROOT/'analysis/r400_d_real_quick_result_20261003'
PRIOR=ROOT/'analysis/r398_d_real_validation_plan_20261003'
CHANNELS=[f'S{i}_B{axis}_uT' for i in range(1,6) for axis in 'xyz']


def read(p):return json.loads(p.read_text(encoding='utf8'))
def dump(p,a):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(a,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def magnetic(path):
    a=pd.read_csv(path,usecols=['t_group_ns','valid_sensor_count']+CHANNELS)
    t=a.t_group_ns.to_numpy(np.int64);b=a[CHANNELS].to_numpy(float)
    good=(a.valid_sensor_count.to_numpy()==5)&np.isfinite(b).all(1)
    assert np.all(np.diff(t)>0)
    return t,b,good
def bins(t,b,width=.2,duration=20.):
    index=np.floor(t/width+1e-9).astype(int);time=[];values=[];counts=[]
    for i in range(int(duration/width)):
        take=(index==i)
        if not take.any():raise ValueError(f'Missing magnetic bin {i}')
        time.append((i+.5)*width);values.append(b[take].mean(0));counts.append(int(take.sum()))
    return np.array(time),np.array(values),counts


def svg_curves(curves,dest,include_prediction=True):
    width,height=1200,720;tags=sorted({r['tag'] for r in curves});colors=['#1468b3','#d97408','#188866']
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/>']
    for a,tag in enumerate(tags):
        for j,kind in enumerate(('history','control')):
            cc=[r for r in curves if r['tag']==tag and r['kind']==kind]
            left,top=75+j*590,45+a*350;w,h=485,245
            maximum=max(float(r['norm'].max()) for r in cc)
            if include_prediction:maximum=max(maximum,float(cc[0]['pred_norm'].max()))
            maximum*=1.08
            parts.append(f'<text x="{left}" y="{top-14}" font-size="17" font-family="Arial">{tag} | {kind}</text>')
            for value in np.linspace(0,maximum,5):
                y=top+h*(1-value/maximum)
                parts.append(f'<path d="M{left},{y}h{w}" stroke="#e2e8ed"/><text x="{left-8}" y="{y+5}" text-anchor="end" font-family="Arial" font-size="12">{value:.1f}</text>')
            for value in (0,5,10,15,20):
                x=left+w*value/20;parts.append(f'<text x="{x}" y="{top+h+18}" text-anchor="middle" font-family="Arial" font-size="12">{value}</text>')
            lines=[(r['t'],r['norm'],colors[k],f'Real repeat{k+1}',False) for k,r in enumerate(cc)]
            if include_prediction:lines.append((cc[0]['pred_t'],cc[0]['pred_norm'],'#222','Frozen prediction',True))
            for k,(tt,values,color,label,dashed) in enumerate(lines):
                points=' '.join(f'{left+w*t/20:.2f},{top+h*(1-v/maximum):.2f}' for t,v in zip(tt,values) if t<=20)
                dash='stroke-dasharray="7 5"' if dashed else ''
                parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2" {dash}/><text x="{left+k*120}" y="{top+h+57}" fill="{color}" font-family="Arial" font-size="12">{label}</text>')
            parts.append(f'<text x="{left+w/2}" y="{top+h+38}" text-anchor="middle" font-family="Arial" font-size="12">Seconds after recorded hold start</text>')
            parts.append(f'<text x="{left-48}" y="{top+h/2}" text-anchor="middle" transform="rotate(-90 {left-48} {top+h/2})" font-family="Arial" font-size="12">15D residual norm (uT)</text>')
    parts.append('</svg>');dest.write_text(''.join(parts),encoding='utf8')


def main():
    progress=read(PLAN/'plan_progress.json');assert progress['status']=='COMPLETED' and progress['completed_trials']==6
    trials=sorted(PLAN.glob('Trial_*'));assert len(trials)==6
    paths=[PLAN/'plan_progress.json',PLAN/'plan_definition.json']+[p/f for p in trials for f in ('magnetic_raw.csv','stage_commands.csv','events.csv','meta.json','batch_plan_item.json')]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths};allraw={};noise=[];tail_noise=[];raw_quality=[]
    for path in trials:
        t,b,good=magnetic(path/'magnetic_raw.csv');allraw[path.name]=(t,b,good)
        raw_quality.append(dict(trial=path.name,frames=len(t),valid_frames=int(good.sum()),
            median_period_ms=float(np.median(np.diff(t))/1e6),max_gap_ms=float(np.max(np.diff(t))/1e6)))
        if path.name.endswith('_none'):
            elapsed=(t-t[0])/1e9;take=good&(elapsed>=2)&(elapsed<30)
            _,bb,_=bins(elapsed[take]-2,b[take],duration=28)
            noise.append(bb-bb.mean(0))
            means=[]
            for start in np.arange(2,27,2.5):
                select=good&(elapsed>=start)&(elapsed<start+2.5);means.append(b[select].mean(0))
            tail_noise.extend(np.diff(means,axis=0))
    noise=np.vstack(noise);norm_noise=np.linalg.norm(noise,axis=1)
    noise_norm_sd=float(np.std(norm_noise,ddof=1));noise_norm_rms=float(np.sqrt(np.mean(np.sum(noise*noise,1))))
    tail_floor=float(np.sqrt(np.mean(np.sum(np.asarray(tail_noise)**2,1))))
    reps=[];curves=[];timing=[];b0s={};endpoints={};pred_late={}
    for path in trials:
        if path.name.endswith('_none'):continue
        item=read(path/'batch_plan_item.json');group=item['group'];kind='history' if 'D_HISTORY_' in group else 'control'
        tag=group.replace('D_HISTORY_','').replace('D_DIRECT_Z_','');cmd=rows(path/'stage_commands.csv')
        t,b,good=allraw[path.name];events=rows(path/'events.csv')
        b0event=next(r for r in events if r['event']=='DYNAMIC_B0_APPLIED')
        b0=np.asarray(json.loads(b0event['value'])['b0_5x3_uT']).reshape(15)
        b0s[(tag,kind)]=b0
        holds=[(i,r) for i,r in enumerate(cmd) if r['command']=='HOST_DWELL' and (r['phase'].startswith('waypoint_hold_') or r['phase']=='steady_hold')]
        assert len(holds)==3
        m400=sum(r['command'].strip().upper()=='M400' for r in cmd)
        prediction=np.load(PRIOR/'predictions'/f'{tag}_{kind}.npz',allow_pickle=False)
        ph=int(prediction['hold_start_index']);pt=prediction['hold_t'];pb=prediction['b'][ph:]
        predlate=pb[(pt>=16)&(pt<=20)].mean(0);prednorm=np.linalg.norm(pb-predlate,axis=1)
        pred_late[(tag,kind)]=predlate
        if kind=='history':
            endpoint=item['waypoints_xyz_mm'][-1]
            endpoints[(tag,kind)]=[endpoint['x_offset_mm'],endpoint['y_offset_mm'],endpoint['depth_mm']]
        else:endpoints[(tag,kind)]=[item['center_x_mm']-149.,item['center_y_mm']-147.,item['indent_mm']]
        for rep,(i,hold) in enumerate(holds,1):
            wait=cmd[i-1];assert wait['command']=='HOST_WAIT'
            start=int(wait['t_perf_ns']);end=int(hold['t_perf_ns'])
            # This UI records HOST_WAIT, not M400 completion. Expose that fact.
            age=(t-start)/1e9;take=good&(age>=0)&(age<20)
            ht,hb,counts=bins(age[take],b[take]);late=b[good&(age>=16)&(age<20)].mean(0)
            residual=hb-late;norm=np.linalg.norm(residual,axis=1)
            early=norm[0];peak_index=int(np.argmax(norm[ht<16]));peak=float(norm[peak_index]);rise=peak-early
            above=(ht>=.3)&(ht<16)&(norm-early>max(.05*early,3*noise_norm_sd))
            sustained=bool(np.any(above[:-1]&above[1:]));accepted=bool(sustained and ht[peak_index]>.2)
            tail1=b[good&(age>=15)&(age<17.5)].mean(0);tail2=b[good&(age>=17.5)&(age<20)].mean(0)
            tailshift=float(np.linalg.norm(tail2-tail1));stationary=bool(tailshift<=3*tail_floor)
            pred_at=np.interp(ht,pt,prednorm)
            pred_vector=np.column_stack([np.interp(ht,pt,(pb-predlate)[:,d]) for d in range(15)])
            simulated_rise=(pred_at[ht<16].max()-pred_at[0])/max(pred_at[0],1e-9)*100
            curve= dict(tag=tag,kind=kind,rep=rep,t=ht,field=hb,residual=residual,norm=norm,pred_t=pt,pred_norm=prednorm)
            curves.append(curve)
            row=dict(tag=tag,kind=kind,trial=path.name,repeat=rep,
                initial_residual_norm_uT=float(early),peak_residual_norm_uT=peak,peak_seconds=float(ht[peak_index]),
                rise_uT=float(rise),rise_percent=float(rise/max(early,1e-9)*100),
                overshoot_rule_pass=accepted,sustained_two_bins=sustained,tail_shift_uT=tailshift,tail_stationary_at_noise_floor=stationary,
                late_delta_B_uT=(late-b0).tolist(),late_absolute_B_uT=late.tolist(),
                noise_norm_sd_uT=noise_norm_sd,initial_to_unloaded_noise_rms_ratio=float(early/max(noise_norm_rms,1e-9)),
                binned_vector_curve_rmse_vs_frozen_prediction_uT=float(np.sqrt(np.mean((residual-pred_vector)**2))),
                frozen_predicted_rise_percent_at_bin_centers=float(simulated_rise),raw_hold_frames=int(take.sum()),minimum_bin_samples=min(counts),
                logged_hold_duration_seconds=(end-start)/1e9,m400_completion_logged=m400>0,
                timing_origin='recorded last HOST_WAIT end, not independently observed stage completion')
            reps.append(row);timing.append(dict(trial=path.name,repeat=rep,hold_start_ns=start,hold_end_ns=end,preceding_wait=wait['response'],m400_commands=m400))
    summaries=[]
    for tag in sorted({r['tag'] for r in reps}):
        item=dict(tag=tag)
        for kind in ('history','control'):
            rr=[r for r in reps if r['tag']==tag and r['kind']==kind]
            item[kind]=dict(repetitions=len(rr),overshoot_pass=sum(r['overshoot_rule_pass'] for r in rr),
                mean_rise_percent=float(np.mean([r['rise_percent'] for r in rr])),
                rise_percent_repeats=[round(r['rise_percent'],2) for r in rr],peak_seconds_repeats=[r['peak_seconds'] for r in rr],
                stationary_tails=sum(r['tail_stationary_at_noise_floor'] for r in rr))
        h=np.asarray([r['late_delta_B_uT'] for r in reps if r['tag']==tag and r['kind']=='history'])
        c=np.asarray([r['late_delta_B_uT'] for r in reps if r['tag']==tag and r['kind']=='control'])
        item['late_between_history_delta_field_norm_uT']=float(np.linalg.norm(h.mean(0)-c.mean(0)))
        item['between_condition_unloaded_b0_difference_uT']=float(np.linalg.norm(b0s[(tag,'history')]-b0s[(tag,'control')]))
        item['frozen_predicted_late_between_history_field_difference_uT']=float(np.linalg.norm(pred_late[(tag,'history')]-pred_late[(tag,'control')]))
        item['history_endpoint_mm']=endpoints[(tag,'history')];item['control_endpoint_mm']=endpoints[(tag,'control')]
        np.testing.assert_allclose(item['history_endpoint_mm'],item['control_endpoint_mm'],atol=1e-6)
        item['history_repeat_late_field_scatter_uT']=float(np.sqrt(np.mean(np.sum((h-h.mean(0))**2,1))))
        item['control_repeat_late_field_scatter_uT']=float(np.sqrt(np.mean(np.sum((c-c.mean(0))**2,1))))
        summaries.append(item)
    report=dict(status='COMPLETED_RAW_HOLD_DIAGNOSTIC',plan=PLAN.relative_to(ROOT).as_posix(),source_device='5*5-had-239',
        trials=6,contact_conditions=4,nested_repetitions=12,bin_seconds=.2,noise_norm_sd_uT=noise_norm_sd,
        noise_norm_rms_uT=noise_norm_rms,tail_noise_difference_rms_uT=tail_floor,
        summaries=summaries,repetitions=reps,raw_quality=raw_quality,timing=timing,input_hashes=hashes,
        material_time_constant_inferred=False,tail_reference_is_independent_static_truth=False,
        m400_available=False,timing_limit='No M400 ACK logged; t0 follows HOST_WAIT. Estimated arrival sensitivity remains to be checked.',
        no_fitting=True,no_hardware_operations=True,raw_files_unchanged=True)
    dump(OUT/'report.json',report)
    for path in paths:assert sha(path)==hashes[str(path.relative_to(ROOT))]
    np.savez_compressed(OUT/'hold_curves.npz',**{f'{i}_{k}':v for i,row in enumerate(curves) for k,v in row.items() if isinstance(v,np.ndarray)})
    svg_curves(curves,OUT/'hold_comparison.svg');svg_curves(curves,OUT/'hold_real_detail.svg',False)
    lines=[]
    for p in summaries:
        for kind,label in [('history','复杂历史'),('control','直接Z')]:
            s=p[kind];lines.append(f'<tr><td>{p["tag"]}</td><td>{label}</td><td>{s["overshoot_pass"]}/3</td><td>{s["rise_percent_repeats"]}</td><td>{s["peak_seconds_repeats"]}</td><td>{s["stationary_tails"]}/3</td></tr>')
    page='<meta charset="utf-8"><title>D快速实测分析</title><style>body{font:16px/1.7 system-ui;max-width:1100px;margin:30px auto}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:8px}img{width:100%}</style><h1>D快速实测：冻结仿真假设对照</h1>'
    page+=f'<p>6/6 Trial完成，4个条件各3次重复。0.2秒磁场均值后计算15维向量距；参照为每次停驻16–20秒的平均磁场，未视为已知静态真值。空载范数噪声SD={noise_norm_sd:.3f} μT，15维噪声RMS={noise_norm_rms:.3f} μT。</p>'
    page+='<table><tr><th>终点</th><th>到达历史</th><th>预设非单调规则通过</th><th>三次峰值上升%</th><th>峰值时刻s</th><th>尾部近噪声底</th></tr>'+''.join(lines)+'</table><h2>冻结仿真与实测：同一幅度单位</h2><img src="hold_comparison.svg"><h2>实测细节：各面板缩放已标注</h2><img src="hold_real_detail.svg">'
    page+='<h2>相同指令终点的晚期磁场差异</h2><table><tr><th>终点</th><th>实测ΔB差异 μT</th><th>两个条件B0差异 μT</th><th>冻结仿真预测差异 μT</th><th>历史/直接Z重复散布 μT</th></tr>'
    for s in summaries:
        page+=f'<tr><td>{s["tag"]}</td><td>{s["late_between_history_delta_field_norm_uT"]:.2f}</td><td>{s["between_condition_unloaded_b0_difference_uT"]:.2f}</td><td>{s["frozen_predicted_late_between_history_field_difference_uT"]:.2f}</td><td>{s["history_repeat_late_field_scatter_uT"]:.2f}/{s["control_repeat_late_field_scatter_uT"]:.2f}</td></tr>'
    page+='</table><p>晚期差异是另一个诊断，不把它替代为原先非单调峰值假设验证成功。端点为指令参考，尚未独立排除驱动回差、形变慢记忆与批次漂移。</p>'
    page+='<p>时间限制：本次日志无M400完成ACK，以最后HOST_WAIT结束作为停驻起点；未用命令发送时刻代替到达。模型与实测的绝对峰值幅度及时间仍需逐项检验。这里只验证磁响应假设，不是仿真增强定位模型的收益。</p><p><a href="report.json">完整逐重复数据与输入哈希</a></p>'
    (OUT/'index.html').write_text(page,encoding='utf8')
    print(json.dumps(dict(summaries=summaries,noise_norm_sd_uT=noise_norm_sd,noise_norm_rms_uT=noise_norm_rms),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
