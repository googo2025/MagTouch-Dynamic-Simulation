"""Complete-evidence reduction for publication; no fitting or output alignment."""
from pathlib import Path
import hashlib
import itertools
import json
import re
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'analysis/r460_source_real_history_refinement_20261004'
BASE = OUT/'readback_closed_native/analysis/r460_source_real_history_refinement_20261004/new_trajectory_native_replay'
ARMS = ['R373', 'history_position', 'history_motion_hold_anchor']
BATCHES = ['SOURCE', 'SOURCE_REPEAT', 'Yellow']


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def average(vals):
    vals = [x for x in vals if x is not None]
    return float(np.mean(vals)) if vals else None


def cluster_uncertainty(base, candidate, families):
    # 12 observations are six geometry clusters with two speeds each.
    # Conditional paired resampling of these six chosen families, not a
    # population guarantee, seed variability, or independent-device inference.
    groups = sorted(set(families))
    assert len(groups) == 6
    b = np.array([np.mean([x for x, f in zip(base, families) if f == g]) for g in groups])
    c = np.array([np.mean([x for x, f in zip(candidate, families) if f == g]) for g in groups])
    d = c-b
    rng = np.random.default_rng(460)
    samples = rng.integers(0, 6, size=(10000, 6))
    delta = d[samples].mean(1)
    null = np.array([np.mean(d*np.array(signs)) for signs in itertools.product([-1, 1], repeat=6)])
    return dict(geometry_clusters=6, paired_difference_candidate_minus_baseline=float(d.mean()),
        descriptive_cluster_bootstrap_95_interval=np.quantile(delta, [.025, .975]).tolist(),
        two_sided_exact_cluster_sign_flip_p=float(np.mean(np.abs(null) >= abs(d.mean())-1e-12)),
        bootstrap_seed=460, bootstrap_replicates=10000, exact_sign_assignments=64,
        improved_geometry_clusters=int(np.count_nonzero(d < 0)),
        interpretation='Conditional six-family paired descriptive uncertainty; no multiplicity correction or random trajectory/device sampling; not population/seed CI')


def main():
    native = read(BASE/'report.json')
    audit = read(BASE/'audit_v2/report.json')
    assert audit['status'] == 'INDEPENDENT_NUMERIC_REDUCTION_PASSED'
    assert audit['native_report_sha256'] == sha(BASE/'report.json')
    assert len(native['rows']) == len(audit['rows']) == 144
    nidx = {(r['batch'], r['arm'], r['id']): r for r in native['rows']}
    rows = []
    for r in audit['rows']:
        p = nidx[r['batch'], r['arm'], r['id']]
        rows.append(dict(r, xyz=p['moving_xyz_shared'], safe_reference=p['safe_reference']))
    summaries = []
    comparisons = []
    for batch in BATCHES:
        for arm in ARMS:
            rr = [r for r in rows if r['batch'] == batch and r['arm'] == arm and r['mode'] == 'waypoint_xyz']
            assert len(rr) == 12
            s = next(x.copy() for x in native['summaries'] if x['batch'] == batch and x['arm'] == arm)
            aa = next(x for x in audit['summaries'] if x['batch'] == batch and x['arm'] == arm)
            s.update(dwell_shared_RMS_mm=aa['dwell_shared_RMS_mm'], dwell_own_RMS_mm=aa['dwell_own_RMS_mm'],
                eligible_zero_output_frames=sum(r['eligible_zero_output_frames'] for r in rr),
                raw_xyz_p95_trial_mean_mm=average([r['xyz'].get('xyz_p95_mm') for r in rr]),
                centered_xyz_diagnostic_mm=average([r['xyz'].get('translation_removed_xyz_diagnostic_mm') for r in rr]),
                rejected_moving_frames=sum(r['own']['reference_frames']-r['own']['output_frames'] for r in rr),
                own_reference_frames=sum(r['own']['reference_frames'] for r in rr),
                own_output_frames=sum(r['own']['output_frames'] for r in rr),
                amplitude_gain_trial_mean=[average([r['amplitude_axes'][i]['centered_projection_gain'] for r in rr]) for i in range(3)],
                amplitude_gain_trial_range=[[
                    min(r['amplitude_axes'][i]['centered_projection_gain'] for r in rr if r['amplitude_axes'][i]['centered_projection_gain'] is not None),
                    max(r['amplitude_axes'][i]['centered_projection_gain'] for r in rr if r['amplitude_axes'][i]['centered_projection_gain'] is not None)] for i in range(3)],
                reversal=aa['reversal'])
            summaries.append(s)
        baseline = [r for r in rows if r['batch'] == batch and r['arm'] == 'R373' and r['mode'] == 'waypoint_xyz']
        for arm in ARMS[1:]:
            candidates = {r['id']: r for r in rows if r['batch'] == batch and r['arm'] == arm}
            pairs = [(b, candidates[b['id']]) for b in baseline]
            families = [re.sub(r'_V[\d.]+$', '', b['group']) for b, _ in pairs]
            fields = dict(raw_XY=('movement','raw_xy_mean_mm'), centered_XY=('movement','centered_xy_diagnostic_mm'),
                raw_XYZ=('xyz','xyz_mean_mm'), jump_rate=('movement','xy_residual_jump_rate_percent'),
                shared_dwell_XY_RMS=('dwell_shared','mean_xy_wobble_rms_mm'))
            result = dict(batch=batch, arm=arm, trials=12, metrics={})
            for metric, (block, key) in fields.items():
                b = [p[0][block][key] for p in pairs]
                c = [p[1][block][key] for p in pairs]
                assert all(x is not None for x in b+c)
                result['metrics'][metric] = dict(baseline=average(b),candidate=average(c),
                    reduction_percent=100*(average(b)-average(c))/average(b),
                    improved_trials=sum(y < x for x,y in zip(b,c)),
                    unchanged_trials=sum(y == x for x,y in zip(b,c)),
                    uncertainty=cluster_uncertainty(b,c,families))
            comparisons.append(result)
    controls = [r for r in rows if r['mode'] != 'waypoint_xyz']
    safe = []
    for batch in BATCHES:
        for arm in ARMS:
            rr = [r for r in rows if r['batch'] == batch and r['arm'] == arm]
            safe.append(dict(batch=batch,arm=arm,**{
                k:sum(r['safe_reference'][k] for r in rr)
                for k in ['frames','outputs','without_external_hint_frames','without_external_hint_outputs',
                          'with_external_hint_frames','with_external_hint_outputs']}))
    completions = []
    for batch in BATCHES:
        for arm in ARMS[1:]:
            p = BASE/'native'/batch/arm/'completed.json'
            c = read(p)
            assert c['status'] == 'COMPLETED'
            completions.append(dict(batch=batch,arm=arm,groups=c['groups'],seconds=c['seconds'],contract=c['contract'],
                                    completed_sha256=sha(p)))
    data = dict(status='COMPLETE_FROZEN_NATIVE_REAL_COMPARISON_AND_INDEPENDENT_REDUCTION',
        source_reports=dict(native_sha256=sha(BASE/'report.json'),audit_sha256=sha(BASE/'audit_v2/report.json'),
                            input_manifest_sha256=sha(BASE/'input_manifest.json')),
        summaries=summaries,comparisons=comparisons,all_rows=rows,controls=controls,
        safe_reference_totals=safe,closed_six_jobs=completions,
        source_VAL_conditional_sensitivity=read(OUT/'source_val_conditional_sensitivity.json'),
        limits=['Retrospective frozen evaluation, not prospective blind experiment',
               'SOURCE_REPEAT is same source device; six geometry families repeated at two speeds and three batches',
               'References from stage commands/EST, not measured contact or optical ground truth',
               'No added XYZ output smoother; causal input EMA and original histories retained',
               'Centering fits one vector per Trial only for diagnosis, not applied to raw or deployed predictions',
               'Zero qualifying sharp reversal events: delay NOT estimated, NOT zero-delay evidence',
               'Conditional head Jacobian is not full causal magnetic-noise transfer',
               'One preset training seed; source-only selection; Yellow/new records not fitted'],
        scientific_training=False,hardware_operated=False,production_modified=False,server_deleted_files=[])
    out = OUT/'final_evidence_summary.json'
    out.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps(dict(summary_sha256=sha(out),comparisons=comparisons,
                         safe_reference_totals=safe,
                         amplitude=[dict(batch=s['batch'],arm=s['arm'],gain=s['amplitude_gain_trial_mean']) for s in summaries]),ensure_ascii=False))


if __name__ == '__main__':
    main()
