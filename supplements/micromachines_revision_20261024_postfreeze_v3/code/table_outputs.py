"""Render already verified numerical results into Supplementary Tables S5-S8."""
from statistics import median
def table(headers,rows):return '| '+' | '.join(headers)+' |\n|'+'|'.join('---' for _ in headers)+'|\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)+'\n'
def write_tables(out,result,study):
 a=result['common_equal_record_mean'];rows=[]
 for label,key,digits,factor in [('Raw XY mean / mm','raw_xy_mean_mm',3,1),('Residual-step RMS / mm','xy_residual_step_rms_mm',3,1),('Residual jumps / %','xy_residual_jump_rate_percent',2,1),('Common-output coverage / %','coverage',3,100)]:
  vals=[]
  for arm in ['H0','H2']:
   x=a[arm][key];vals.append(f"{factor*x['mean']:.{digits}f} ± {factor*x['SD']:.{digits}f}")
  rows.append([label,*vals])
 (out/'Table_S5.md').write_text('# Table S5\n\n'+table(['Metric','H0','H2'],rows))
 holds=[r for r in result['targeted']['hold_summaries'] if r['mask']=='common'];rows=[]
 for g in sorted({r['group'] for r in holds}):
  arms={r['arm']:r for r in holds if r['group']==g};h0,h2=arms['H0'],arms['H2'];rows.append([h0['depth_mm'],h0['nominal_speed_mm_s'],h0['evaluated_arrivals'],h0['output_frames'],f"{h0['xy_hold_wobble_rms_mm']:.3f} → {h2['xy_hold_wobble_rms_mm']:.3f}",f"{h0['between_direction_xy_spread_rms_mm']:.3f} → {h2['between_direction_xy_spread_rms_mm']:.3f}"])
 (out/'Table_S6.md').write_text('# Table S6\n\n'+table(['Depth / mm','Speed / (mm/s)','Arrivals','Common frames','Within-hold RMS H0 → H2 / mm','Between-direction spread H0 → H2 / mm'],rows))
 h0={r['uid']:r for r in result['own_record_statistics'] if r['arm']=='H0'};rows=[]
 for e in sorted(study['records'],key=lambda e:e['trial']):
  r=h0[e['uid']]['primary'];rows.append([e['trial'],e['nominal_depth_mm'],e['mode'],e['speed_mm_s'],f"{r['raw_xy_mean_mm']:.6f}",f"{100*r['coverage']:.3f}",'Yes' if e['uid'] in study['H0_comparable_uids'] else 'No'])
 (out/'Table_S7.md').write_text('# Table S7\n\n'+table(['Sequence','Depth / mm','Action','Speed / (mm/s)','H0 raw XY / mm','H0 coverage / %','H2 comparison'],rows))
 events={}
 for r in result['targeted']['response_rows']:events.setdefault((r['group'],r['uid'],r['window']),{})[r['arm']]=r
 delta=[v['H2']['excess_vs_command_reference_s']-v['H0']['excess_vs_command_reference_s'] for v in events.values() if v['H0']['crossing_observed'] and v['H2']['crossing_observed']]
 rows=[['Defined reversals',len(events)],['Jointly evaluable',len(delta)],['Unresolved retained',len(events)-len(delta)],['H2 earlier / later / tied',f"{sum(d<-1e-12 for d in delta)} / {sum(d>1e-12 for d in delta)} / {sum(abs(d)<=1e-12 for d in delta)}"],['Median paired H2 − H0 / ms',f'{1000*median(delta):.3f}']]
 (out/'Table_S8.md').write_text('# Table S8\n\n'+table(['Command-reference response statistic','Value'],rows)+'\nThis is a command/EST response proxy, not independently measured physical latency.\n')
