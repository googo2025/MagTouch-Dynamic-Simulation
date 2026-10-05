"""Post-freeze diagnosis, never a checkpoint/parameter selector."""
import argparse,json,html
import numpy as np
from training.virtual_expansion_r379 import OUT,ROOT
from training.run_forward_r361 import dump,sha
from training.research_formal_r365 import macro

def change(before,after):return 100*(after/before-1) if before else None

def interpretation(groups):
    def raw_statement(device):
        g=groups[device]
        direction='decreases' if g['raw_xy_change_pct']<0 else 'increases' if g['raw_xy_change_pct']>0 else 'is unchanged'
        return f"Raw XY {direction} by {abs(g['raw_xy_change_pct']):.3f}% on this fixed post-freeze replay; offset-removed scores are separate diagnostics, not a substitute for raw error."
    return {'virtual':raw_statement('virtual')+' Test geometry seeds differ but the empirical generator is shared; this is not independent physical-teacher verification.',
            'source':raw_statement('source'),
            'target':raw_statement('target')+' No target tuning; one previously inspected target specimen, no new prospective acquisition.',
            'coupling':'Parameters unchanged does not imply numeric invariance: changed XY affects field support, permission and Z.'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--count',type=int,default=100);a=ap.parse_args()
    folder=OUT/f'arm_{a.count}';source=folder/'evaluation_report.json';report=json.loads(source.read_text(encoding='utf8'));groups={}
    for device,g in report['groups'].items():
        before=g['macro']['baseline'];after=g['macro']['candidate']
        rows=[r for r in report['records'] if r['device']==device];contact=[r for r in rows if r['mode']!='none']
        shallow={'before_outputs':0,'after_outputs':0,'nominal_frames':0,'lost':0,'added':0};jumps={m:{'pairs':0,'over_0p2_mm':0} for m in ('baseline','candidate')}
        for r in rows:
            with np.load(folder/'evaluation'/f"{r['kind']}_{r['id']}.npz",allow_pickle=False) as s:
                mask=np.isfinite(s['q']).all(1)&(s['q'][:,2]>0)&(s['q'][:,2]<.4)
                b=s['baseline_ok'];c=s['candidate_ok'];shallow['nominal_frames']+=int(mask.sum());shallow['before_outputs']+=int((mask&b).sum());shallow['after_outputs']+=int((mask&c).sum())
                shallow['lost']+=int((mask&b&~c).sum());shallow['added']+=int((mask&~b&c).sum())
            for m in jumps:
                v=r['scores'][m].get('constant_reference_z_steps',{});jumps[m]['pairs']+=v.get('pairs',0);jumps[m]['over_0p2_mm']+=v.get('over_0p2_mm',0)
        groups[device]={'trials':len(contact),'raw_xy_before_mm':before['xy_mean_mm'],'raw_xy_after_mm':after['xy_mean_mm'],
            'raw_xy_change_pct':change(before['xy_mean_mm'],after['xy_mean_mm']),
            'raw_xyz_change_pct':change(before['xyz_mean_mm'],after['xyz_mean_mm']),
            'diagnostic_trial_translation_removed_xy_change_pct':change(before['xy_translation_removed_mm'],after['xy_translation_removed_mm']),
            'z_mae_before_mm':before['z_mae_mm'],'z_mae_after_mm':after['z_mae_mm'],
            'coverage_before':before['coverage'],'coverage_after':after['coverage'],
            'primary_outputs_lost':sum(r['changes']['primary_outputs_lost'] for r in contact),
            'primary_outputs_added':sum(r['changes']['primary_outputs_added'] for r in contact),
            'safe_frames':sum(r['scores']['baseline']['safe_frames'] for r in rows),'safe_outputs':g['safe_outputs'],
            'nominal_shallow_output_changes':shallow,'constant_nominal_z_jumps':jumps,
            'per_trial_bias_delta_mm':[{'id':r['id'],'change':(np.array(r['scores']['candidate']['bias_xyz_mm'])-np.array(r['scores']['baseline']['bias_xyz_mm'])).tolist()} for r in contact],
            'per_trial_xy_results':[{'id':r['id'],'mode':r['mode'],'before_mm':r['scores']['baseline']['xy_mean_mm'],'after_mm':r['scores']['candidate']['xy_mean_mm'],
                'change_pct':change(r['scores']['baseline']['xy_mean_mm'],r['scores']['candidate']['xy_mean_mm'])} for r in contact]}
    grid={d:macro([r['scores'] for r in v['rows']],['baseline','candidate']) for d,v in report['independent_grid_offset_diagnostic'].items()}
    result={'status':'COMPLETE','count':a.count,'evaluation_report_sha256':sha(source),'freeze_sha256':sha(folder/'freeze.json'),
        'input_definition':'controlled 10Hz R374 magnetic replay / independent R370 virtual test; not new online acquisition',
        'groups':groups,'independent_same_session_grid_offset_diagnostic_macro':grid,
        'interpretation':interpretation(groups),
        'nominal_shallow_counts_are_not_actual_contact_recall':True,'simulation_safe_output_failures_are_not_real_hardware_failures':True,
        'production_replaced':False,'target_selected_parameters':False,'hypothesis_or_training_protocol_changed':False}
    dump(folder/'coupling_offset_audit.json',result)
    body=['<!doctype html><meta charset="utf-8"><title>R379 耦合与偏移核对</title><style>body{font:16px/1.7 system-ui;max-width:1100px;padding:25px;margin:25px auto;color:#173a50}table{border-collapse:collapse;width:100%;margin:15px 0}td,th{padding:9px;border:1px solid #cbd8e2;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#e9f0f6}.note{padding:15px;background:#fff3d7}pre{white-space:pre-wrap;background:#f1f5f8;padding:12px}</style><h1>冻结后的偏移、许可、浅层与Z耦合核对</h1><p>研究副本只改变XY权重；输入磁流、Z网络和许可规则相同。下表是受控10 Hz回放，不是新的原生在线采集。源／目标各7条接触Trial，虚拟测试64条完整路径；按Trial等权报告，不把帧当独立器件。</p><table><tr><th>数据</th><th>原始XY：前→后 mm</th><th>变化</th><th>去常值诊断变化</th><th>原始Z MAE：前→后 mm</th></tr>']
    for d,g in groups.items():
        body.append(f'<tr><td>{d}</td><td>{g["raw_xy_before_mm"]:.4f} → {g["raw_xy_after_mm"]:.4f}</td><td>{g["raw_xy_change_pct"]:+.2f}%</td><td>{g["diagnostic_trial_translation_removed_xy_change_pct"]:+.2f}%</td><td>{g["z_mae_before_mm"]:.4f} → {g["z_mae_after_mm"]:.4f}</td></tr>')
    body+=[f'</table><p class="note">本组源侧原始XY变化{groups["source"]["raw_xy_change_pct"]:+.2f}%，目标器件变化{groups["target"]["raw_xy_change_pct"]:+.2f}%。均以原始未配准结果为准，去平移诊断不能替代主要误差；目标器件只在冻结后评价，不扩展成所有器件保证。100／500／1000条的训练协议不根据目标结果改变。</p><h2>许可、浅层及安全抬离</h2><table><tr><th>数据</th><th>主评价输出：丢失／新增</th><th>名义浅层输出：前→后</th><th>安全窗口输出：前→后／总帧</th></tr>']
    for d,g in groups.items():
        sh=g['nominal_shallow_output_changes'];sf=g['safe_outputs']
        body.append(f'<tr><td>{d}</td><td>{g["primary_outputs_lost"]} / {g["primary_outputs_added"]}</td><td>{sh["before_outputs"]} → {sh["after_outputs"]} / {sh["nominal_frames"]}</td><td>{sf["baseline"]} → {sf["candidate"]} / {g["safe_frames"]}</td></tr>')
    body+=['</table><p>名义0–0.4 mm不是实际接触真值；以上浅层数量不能称接触召回率。虚拟撤出后仍有输出，说明闭环存在释放／残余场差异，不能凭虚拟XYZ误差降低宣布系统全部过关；真实记录的安全抬离输出另计，不能混淆。</p><h2>逐动作原始XY，禁止隐藏反例</h2><table><tr><th>器件／动作</th><th>前 mm</th><th>后 mm</th><th>变化</th></tr>']
    for d in ('source','target'):
        for r in groups[d]['per_trial_xy_results']:
            body.append(f'<tr><td>{d} / {r["mode"]}</td><td>{r["before_mm"]:.4f}</td><td>{r["after_mm"]:.4f}</td><td>{r["change_pct"]:+.2f}%</td></tr>')
    body+=['</table><p><a href="coupling_offset_audit.json">全量偏移和耦合记录</a> · <a href="evaluation_report.json">许可评分、支持度和最差帧</a> · <a href="index.html">训练数据与主结果</a></p>']
    (folder/'coupling_offset_audit.html').write_text(''.join(body),encoding='utf8')
    print({d:{k:v for k,v in g.items() if k in ('raw_xy_change_pct','diagnostic_trial_translation_removed_xy_change_pct','primary_outputs_lost','primary_outputs_added','safe_frames','safe_outputs','nominal_shallow_output_changes')} for d,g in groups.items()})
if __name__=='__main__':main()
