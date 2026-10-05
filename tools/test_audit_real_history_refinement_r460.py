"""Hardware-free numerical fixtures for the frozen-output reduction audit."""
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from audit_real_history_refinement_r460 import independent_score,independent_dwell,amplitude,reversal_events,delay
from training.real_jump_audit_r402 import score,dwell_score


class AuditTests(unittest.TestCase):
    def test_independent_metrics_match_with_gaps_and_rejections(self):
        rng=np.random.default_rng(7);n=500
        t=np.arange(n,dtype=np.int64)*20_000_000;t[300:]+=400_000_000
        q=rng.normal(size=(n,3));x=q+.15*rng.normal(size=(n,3))+np.array([.5,-.2,.1])
        ok=np.ones(n,bool);ok[100:106]=False;x[410]=np.nan
        mask=np.ones(n,bool);mask[50:65]=False
        expected=score(x,q,ok,mask,t);actual=independent_score(x,q,ok,mask,t)
        for k,v in actual.items():np.testing.assert_allclose(v,expected[k],rtol=1e-12,atol=1e-12)

    def test_dwell_blocks_keep_gaps_phase_and_validity(self):
        t=np.arange(400,dtype=np.int64)*20_000_000;x=np.zeros((400,3))
        x[:,0]=np.sin(np.arange(400));ok=np.ones(400,bool);ok[100]=False
        mask=np.ones(400,bool);phase=np.array(['a']*200+['b']*200)
        expected=dwell_score(x,ok,mask,t,phase);actual=independent_dwell(x,ok,mask,t,phase)
        self.assertEqual(expected['blocks_count'],actual['blocks_count'])
        self.assertAlmostEqual(expected['mean_xy_wobble_rms_mm'],actual['mean_xy_wobble_rms_mm'])

    def test_amplitude_not_changed_by_constant_offset(self):
        q=np.column_stack([np.linspace(-1,1,100),np.linspace(-2,2,100),np.ones(100)])
        x=q*np.array([.8,1.2,1])+np.array([.5,-.3,0]);mask=np.ones(100,bool)
        a=amplitude(x,q,mask)
        self.assertAlmostEqual(a[0]['centered_projection_gain'],.8)
        self.assertAlmostEqual(a[1]['centered_projection_gain'],1.2)
        self.assertIsNone(a[2]['centered_projection_gain'])

    def test_reversal_known_delay_and_unresolved_are_preserved(self):
        s=np.arange(0,6,.01);t=np.rint(s*1e9).astype(np.int64)
        q=np.column_stack([np.where(s<=3,s-1.5,4.5-s),np.zeros(len(s)),np.ones(len(s))])
        selected=np.ones(len(s),bool);events=reversal_events(t,q,selected)
        self.assertEqual(len(events),1)
        shifted=np.maximum(s-.16,0)
        x=q.copy();x[:,0]=np.where(shifted<=3,shifted-1.5,4.5-shifted)
        d=delay(x,t,selected,events[0])
        self.assertEqual(d['status'],'RESOLVED')
        # Window/threshold detection has a fixed timing offset even for q.
        # Verify the known *additional* delay, not a fictitious exact encoder lag.
        reference_detection=delay(q,t,selected,events[0])
        self.assertAlmostEqual(d['delay_s']-reference_detection['delay_s'],.16,delta=.0200001)
        straight=q.copy();straight[:,0]=s-1.5
        unresolved=delay(straight,t,selected,events[0])
        self.assertIsNone(unresolved['delay_s'])


if __name__=='__main__':unittest.main()
