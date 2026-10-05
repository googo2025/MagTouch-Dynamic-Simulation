"""Tolerance-only successor to the preserved R460 numeric reduction auditor.

The frozen native replay, predictions, references, masks and metrics do not
change. The first partial reduction found 8.30e-11 mm cancellation error when
comparing diff(x-q) with diff(x)-diff(q). Use sub-nanometre float tolerance;
integer denominators and jump counts must still match exactly. Preserve v1.
"""
from pathlib import Path
import argparse
import numpy as np
import audit_real_history_refinement_r460 as previous

V1 = Path(previous.__file__)
V1_SHA = 'be3b983bc7f6cc875e1949210eb74ee8ac839a3f5195b12c7ee0ff32b5d18f4c'
assert previous.sha(V1) == V1_SHA, 'Preserve original audit source'


def equal(a, b):
    if a is None or b is None:
        assert a is None and b is None
    elif isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
        assert a == b, 'Integer event/denominator counts must match exactly'
    else:
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9)


previous.equal = equal
previous.DEST = previous.EVAL/'audit_v2'
previous.CONTRACT = dict(previous.CONTRACT, schema='R460-Frozen-Reduction-Audit-2',
    numeric_tolerance=dict(atol=1e-9, rtol=1e-9, integer_counts_exact=True),
    preserved_v1_source_sha256=V1_SHA,
    reason='Equivalent residual-step expressions differed by8.30e-11mm during the partial source reduction; no changes to metrics, model, masks, alignment or predictions')
# The inherited launcher must execute this version rather than restart v1.
previous.__file__ = str(Path(__file__).resolve())


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--launch',action='store_true');p.add_argument('--wait-and-run',action='store_true');p.add_argument('--run',action='store_true');a=p.parse_args()
    if a.launch:previous.launch()
    elif a.wait_and_run:previous.wait_and_run()
    elif a.run:previous.run()
    else:p.error('Choose mode')
