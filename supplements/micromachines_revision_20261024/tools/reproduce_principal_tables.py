"""Generate and check principal trajectory tables after the saved-output reduction."""
from pathlib import Path
import argparse,ast,csv,itertools,json,re,statistics,zipfile
import numpy as np

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--statistics',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();root=args.root.resolve();out=args.output.resolve();assert not out.exists();out.mkdir(parents=True)
 stats=args.statistics.resolve();verification=json.loads((stats/'verification.json').read_text());assert verification['status']=='PASS_PORTABLE_SAVED_OUTPUT_REDUCTION'
 rows=json.loads((stats/'original/reproduced_metrics.json').read_text());lookup={(r['trial_id'],r['model'].replace('_','')):r['metrics'] for r in rows}
 spec=json.loads((root/'analysis/r460_source_real_history_refinement_20261004/frozen_real_replay_spec.json').read_text());BATCH={'SOURCE':'Source 1','SOURCE_REPEAT':'Source 2','Yellow':'Target'}
 with zipfile.ZipFile(root/'submitted/File_S1_scripts.zip') as z:
  source=z.read('File_S1_scripts/control2.py').decode('utf8');tree=ast.parse(source);selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='cluster_stats'];namespace={'np':np,'itertools':itertools};exec(compile(ast.Module(body=selected,type_ignores=[]),'<published-original-cluster-stats>','exec'),namespace);cluster_stats=namespace['cluster_stats']
  source=z.read('File_S1_scripts/tables.py').decode('utf8');tree=ast.parse(source);selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['body','stat_rows']]
 def trials(batch):return [r for r in spec['entries'] if r['device']==batch and r['mode']=='waypoint_xyz']
 def rows_for(batch,key):
  model={'EMA0.03':'H0+EMA0.03','EMA0.05':'H0+EMA0.05','H2EMA':'H2+EMA0.05'}.get(key,key)
  return [lookup[(r['id'],model)] for r in trials(batch)]
 mean=lambda rr,key:float(np.mean([r[key] for r in rr]))
 namespace={'np':np,'A':{'H0':'R373','H1':'history_position','H2':'history_motion_hold_anchor'},'trials':trials,'rows_for':rows_for,'m':mean,'NUM':{},'BATCH':BATCH,'cluster_stats':cluster_stats}
 exec(compile(ast.Module(body=selected,type_ignores=[]),'<published-original-table-functions>','exec'),namespace)
 models=[('H0','H0'),('H1','H1'),('H2','H2'),('EMA0.03','H0+EMA\\textsubscript{0.03}'),('EMA0.05','H0+EMA\\textsubscript{0.05}'),('H2EMA','H2+EMA\\textsubscript{0.05}')]
 generated={'tab_source.tex':namespace['body'](('SOURCE','SOURCE_REPEAT'),models),'tab_target.tex':namespace['body'](('Yellow',),models)}
 MET=[('xy','$E_{xy}$ (mm)',1,'.3f'),('xy_p95','$E_{xy}$, 95th pct. (mm)',1,'.3f'),('cxy','$E^{\\mathrm c}_{xy}$ (mm)',1,'.3f'),('step_rms','Step RMS (mm)',1,'.3f'),('jump_0.2','Jump rate (\\%)',1,'.2f'),('dwell','Dwell RMS (mm)',1,'.3f')];lines=[];statsrows=[]
 for comp,(cand,ctrl) in [('H2 vs. H0',('H2','H0')),('H2 vs. H0+EMA\\textsubscript{0.03}$^{\\dagger}$',('H2','EMA0.03'))]:
  for batch in BATCH:
   st=namespace['stat_rows'](batch,cand,ctrl,MET)
   for i,(lab,v0,v1,rel,delta,ci,p,imp,fam) in enumerate(st):
    dec=2 if 'Jump' in lab else 3;lead=(f'\\multirow{{{len(st)}}}{{*}}{{\\shortstack[l]{{{comp}\\\\{BATCH[batch]}}}}}' if i==0 else '')
    lines.append(f'{lead} & {lab} & {v0:.{dec}f} & {v1:.{dec}f} & ${rel:+.1f}$ & $[{ci[0]:+.{dec}f},\\,{ci[1]:+.{dec}f}]$ & {p:.3f} & {imp}/12 & {fam}/6 \\\\')
    statsrows.append({'comparison':comp,'batch':batch,'metric':lab,'control':v0,'candidate':v1,'relative_change_percent':rel,'difference':delta,'ci_low':ci[0],'ci_high':ci[1],'p':p,'sequences_improved':imp,'families_improved':fam})
   lines.append('\\midrule')
 generated['tab_stats.tex']='\n'.join(lines[:-1]);checked={}
 for name,text in generated.items():
  expected=(root/'expected_tables'/name).read_text();assert re.sub(r'\s','',text)==re.sub(r'\s','',expected),('Original principal table differs',name)
  (out/name).write_text(text+'\n',encoding='utf8');checked[name]='all printed cells and group labels exact'
 with (out/'Table7_paired_full_precision.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(statsrows[0]));w.writeheader();w.writerows(statsrows)
 O=root/'analysis/micromachines_revision_validation_20261023';matched=json.loads((O/'matched_statistics_v1/report.json').read_text());archived=json.loads((O/'archived_validation_v1/statistics_v2/report.json').read_text())
 configs=['R373','H0_matched_L1','single_0p3_L1','single_1_L1','single_4_L1','multi_L1','multi_L2'];keys=['raw_xy_mean_mm','centered_xy_diagnostic_mm','xy_residual_step_rms_mm','xy_residual_jump_rate_percent','dwell_xy_wobble_rms_mm','coverage'];precision=[3,3,3,2,3,2]
 for table,batch in [('Table8','SOURCE'),('Table9','ARCHIVED9'),('TableS1','SOURCE_REPEAT'),('TableS2','Yellow')]:
  pool=archived['summaries'] if batch=='ARCHIVED9' else [r for r in matched['own_summaries'] if r['batch']==batch];actual=[]
  for config in configs:
   selected=[r for r in pool if r['arm']==config or r['arm'].startswith(config+'_seed')];values=[]
   for key,dec in zip(keys,precision):
    vv=[r[key]*(100 if key=='coverage' else 1) for r in selected];value=f'{statistics.mean(vv):.{dec}f}'
    if len(vv)>1 and key!='coverage':value+=' ± '+f'{statistics.stdev(vv):.{dec}f}'
    values.append(value)
   actual.append(values)
  text=(root/'expected_tables'/(table+'.md')).read_text();expected=[[c.strip() for c in line.split('|')[2:-1]] for line in text.splitlines() if line.startswith('|')][2:];assert actual==expected,('Added principal table differs',table)
  (out/(table+'.md')).write_text(text,encoding='utf8');checked[table]='all printed numerical cells exact, including order-seed SD'
 result={'status':'PASS_PRINCIPAL_TRAJECTORY_TABLES','tables':checked,'original_cluster_bootstrap_resamples':10000,'original_sign_flip_families':6,'new_inference':0,'new_training':0};(out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
