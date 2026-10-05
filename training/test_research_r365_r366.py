import unittest
import numpy as np
from training.research_formal_r365 import replay_fields, metrics
from training.research_ambiguity_r366 import causal_windows, peak_pair_distance


class ControlledResearchTests(unittest.TestCase):
    def test_history_only_reads_past_and_prefix(self):
        t=np.arange(50)*.1;b=np.random.default_rng(1).normal(size=(50,15));r=np.zeros(50,bool)
        x,v=causal_windows(b,t,r,np.arange(21,30),2.)
        b[30:]+=10000
        y,v2=causal_windows(b,t,r,np.arange(21,30),2.)
        np.testing.assert_array_equal(x,y);np.testing.assert_array_equal(v,v2)
    def test_reset_and_gaps_cannot_be_crossed(self):
        t=np.arange(50)*.1;b=np.zeros((50,15));r=np.zeros(50,bool);r[25]=True
        _,v=causal_windows(b,t,r,np.array([24,30,46]),2.)
        np.testing.assert_array_equal(v,[True,False,True])
    def test_longer_history_never_erases_current_evidence(self):
        x=np.zeros((2,45));x[1,:15]=4
        bank={'x':{'0.0':x[:,:15],'0.1':x[:,:30],'0.2':x}}
        self.assertEqual(peak_pair_distance(bank,[(0,1)],0.)[0],4.)
        self.assertEqual(peak_pair_distance(bank,[(0,1)],.2)[0],4.)
    def test_constant_translation_does_not_change_jitter(self):
        t=np.arange(20)*.1;q=np.tile([0.,0.,1.4],(20,1));p=q.copy();p[:,2]+=.02*np.sin(t)
        mask=np.ones(20,bool)
        a=metrics(p,q,t,mask,mask);b=metrics(p+[2,-3,.7],q,t,mask,mask)
        self.assertAlmostEqual(a['constant_reference_z_steps']['p95_mm'],b['constant_reference_z_steps']['p95_mm'])
        self.assertAlmostEqual(a['xyz_translation_removed_mm'],b['xyz_translation_removed_mm'])
    def test_rejected_outputs_are_not_zero_errors(self):
        q=np.tile([0.,0.,1.4],(10,1));p=q.copy();valid=np.zeros(10,bool)
        a=metrics(p,q,np.arange(10)*.1,np.ones(10,bool),valid)
        self.assertEqual(a['coverage'],0.);self.assertNotIn('xyz_mean_mm',a)
    def test_full_frozen_engine_prefix_causality(self):
        from training.had239_r133_coverage_xy_runtime import CoverageGuardedNeuralDecoder
        atlas=CoverageGuardedNeuralDecoder.from_report().atlas
        t=np.arange(35)*.1;q=np.tile([0.,0.,1.4],(35,1));q[:,0]=np.sin(t)
        b=atlas.forward(q);a=replay_fields(b,t);c=replay_fields(b[:20],t[:20])
        np.testing.assert_array_equal(a['valid'][:20],c['valid'])
        np.testing.assert_allclose(a['Final'][:20],c['Final'],equal_nan=True)
    def test_increment_ablation_preserves_xy_and_permission(self):
        from training.had239_r133_coverage_xy_runtime import CoverageGuardedNeuralDecoder
        atlas=CoverageGuardedNeuralDecoder.from_report().atlas
        t=np.arange(25)*.1;q=np.tile([0.,0.,1.4],(25,1));b=atlas.forward(q)
        a=replay_fields(b,t,components=True)
        np.testing.assert_array_equal(a['U'][:,:2],a['Final'][:,:2])


if __name__=='__main__':unittest.main()
