"""Portable, opt-in binding for immutable source study; never touch its results.

Only host/output execution guards are adapted. Every run records original SHA,
the guard removal and fresh output bindings. Scientific functions stay intact.
No mode operates hardware; train/replay require explicit CLI invocation.
"""
from pathlib import Path
import argparse
import hashlib
import inspect
import json
import math
import numpy as np
import os
import shutil
import sys
import textwrap
import types

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT/'analysis/r460_source_real_history_refinement_20261004'
sys.path[:0] = [str(ROOT), str(ROOT/'code'), str(ROOT/'training'), str(ROOT/'tools')]


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def remove_host_assert(function):
    source = textwrap.dedent(inspect.getsource(function))
    guard = "    assert socket.gethostname()=='googo-Super-Server'\n"
    assert source.count(guard) == 1, 'Unexpected source guard; retain originals'
    compiled = source.replace(guard, '', 1)
    ns = {}
    exec(compile(compiled, '<portable-host-binding>', 'exec'), function.__globals__, ns)
    return types.FunctionType(ns[function.__name__].__code__, function.__globals__)


def write(p, x): p.write_text(json.dumps(x, ensure_ascii=False, indent=2), encoding='utf8')


def compare_frozen(actual, expected):
    """Frozen audit contract: integer counts exact, floats rtol/atol1e-9."""
    result = dict(floating_tolerance=dict(rtol=1e-9, atol=1e-9), integer_counts_exact=True,
        amplitude_projection_float32_relative_bound=float(8*np.finfo(np.float32).eps),
        amplitude_projection_note='Original reference is float32; its variance dot product uses platform-dependent float32 reduction. Only this diagnostic gain uses8float32-epsilon relative comparison; underlying arrays and scientific formula unchanged.',
        nonidentical_floats=0, maximum_absolute_difference=0.0, amplitude_projection_differences=[])
    def walk(a, b, path):
        if isinstance(a, dict):
            assert isinstance(b, dict) and a.keys() == b.keys(), path
            for key in a:
                walk(a[key], b[key], path + '/' + key)
        elif isinstance(a, list):
            assert isinstance(b, list) and len(a) == len(b), path
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, path + '/' + str(i))
        elif isinstance(a, float) and isinstance(b, float):
            if math.isnan(a) and math.isnan(b):
                return
            amplitude_gain = '/amplitude_axes/' in path and path.endswith('/centered_projection_gain')
            rtol = result['amplitude_projection_float32_relative_bound'] if amplitude_gain else 1e-9
            assert math.isclose(a, b, rel_tol=rtol, abs_tol=1e-9), (path, a, b)
            if a != b:
                result['nonidentical_floats'] += 1
                result['maximum_absolute_difference'] = max(result['maximum_absolute_difference'], abs(a-b))
                if amplitude_gain:
                    result['amplitude_projection_differences'].append(dict(path=path, actual=a, frozen=b, absolute_difference=abs(a-b)))
        else:
            assert type(a) == type(b) and a == b, (path, a, b)
    walk(actual, expected, 'frozen')
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', required=True, choices=['audit','train','replay','summary'])
    p.add_argument('--output-name', required=True)
    p.add_argument('--max-epochs', type=int, default=2000)
    args = p.parse_args()
    assert args.output_name and all(c.isalnum() or c in '_-' for c in args.output_name)
    out = ROOT/'reproducibility_runs'/args.output_name
    assert not out.exists(), 'Choose a new output name; no overwrite or restart'
    out.mkdir(parents=True)
    os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
    receipt=dict(mode=args.mode, output=out.relative_to(ROOT).as_posix(), adapter_sha256=sha(Path(__file__)),
        adaptation='Exact server-host assertion or server_only execution gate removed; output directory changed. Frozen-result wrapper uses existing1e-9 audit tolerance and exact counts; only original float32 amplitude-projection diagnostic allows8float32-epsilon relative platform reduction differences. Scientific source files are not edited.',
        hardware=False, original_data_models_results_unchanged=True, status='STARTED')
    write(out/'portable_binding.json', receipt)
    if args.mode == 'audit':
        import audit_real_history_refinement_r460_v2 as v2
        m = v2.previous
        receipt['originals'] = {n:sha(ROOT/'tools'/n) for n in ['audit_real_history_refinement_r460.py','audit_real_history_refinement_r460_v2.py']}
        m.ROOT=ROOT; m.OUT=STUDY; m.EVAL=STUDY/'new_trajectory_native_replay'; m.DEST=out
        remove_host_assert(m.run)()
        expected=json.loads((m.EVAL/'audit_v2/report.json').read_text(encoding='utf8'))
        actual=json.loads((out/'report.json').read_text(encoding='utf8'))
        receipt['frozen_comparison'] = compare_frozen(
            dict(rows=actual['rows'],summaries=actual['summaries']),
            dict(rows=expected['rows'],summaries=expected['summaries']))
        receipt['numeric_rows_and_summaries_match_frozen_audit']=True
    elif args.mode == 'summary':
        import summarize_closed_native_r460 as m
        receipt['originals']={'tools/summarize_closed_native_r460.py':sha(ROOT/'tools/summarize_closed_native_r460.py')}
        m.BASE=STUDY/'new_trajectory_native_replay'; m.OUT=out
        shutil.copy2(STUDY/'source_val_conditional_sensitivity.json',out/'source_val_conditional_sensitivity.json')
        m.main()
        expected=json.loads((STUDY/'final_evidence_summary.json').read_text(encoding='utf8'))
        actual=json.loads((out/'final_evidence_summary.json').read_text(encoding='utf8'))
        receipt['frozen_comparison'] = compare_frozen(actual,expected)
        receipt['complete_summary_matches_frozen_evidence']=True
    else:
        from training import real_history_refinement_r460 as trainer
        receipt['originals']={'training/real_history_refinement_r460.py':sha(ROOT/'training/real_history_refinement_r460.py')}
        for n in ['protocol.json','data_manifest.json']:
            shutil.copy2(STUDY/n,out/n)
        shutil.copytree(STUDY/'inputs',out/'inputs')
        trainer.OUT=out; trainer.SOURCE=ROOT; trainer.REAL=ROOT/'analysis/r372_direct_source_plateau_20261001'
        trainer.server_only=lambda:None
        if args.mode=='train':
            # Original float64 CUDA training; no forced CPU or seed/LR change.
            for arm in trainer.ARMS:
                f=trainer.train(arm, resume=False, max_epochs=args.max_epochs)
                assert f['status']=='COMPLETED_PLATEAU','Budget truncation is not completed optimization'
        else:
            for arm in trainer.ARMS:
                (out/arm).mkdir()
                for n in ['best_model.npz','freeze.json']:
                    shutil.copy2(STUDY/arm/n,out/arm/n)
            import replay_real_history_refinement_r460 as m
            receipt['originals']['tools/replay_real_history_refinement_r460.py']=sha(ROOT/'tools/replay_real_history_refinement_r460.py')
            m.ROOT=ROOT; m.OUT=out; m.EVAL=out/'new_trajectory_native_replay'; m.EVAL.mkdir()
            for n in ['input_manifest.json','input_spec.json']:
                shutil.copy2(STUDY/'new_trajectory_native_replay'/n,m.EVAL/n)
            remove_host_assert(m.run)()
    receipt['status']='COMPLETE_PORTABLE_EXECUTION'
    write(out/'portable_binding.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__':main()
