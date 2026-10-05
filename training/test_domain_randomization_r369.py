import unittest, tempfile
from pathlib import Path
import numpy as np
from training.domain_randomization_r369 import ResearchXY, protocol, measured, replay, geometry
from training.had239_r133_coverage_xy_runtime import CoverageCausalXYMLP

class DomainRandomizationTests(unittest.TestCase):
    def test_same_checkpoint_predict(self):
        x=np.random.default_rng(369).normal(size=(15,53))
        np.testing.assert_array_equal(ResearchXY().predict(x),CoverageCausalXYMLP().predict(x))
    def test_saved_roundtrip(self):
        m=ResearchXY();x=np.zeros((3,53))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'weights.npz';m.save(path)
            np.testing.assert_array_equal(m.predict(x),ResearchXY.load(path).predict(x))
    def test_no_target_training(self):
        p,rows=protocol()
        allowed={r['id'] for r in rows if r['device']=='source' and r['mode']!='waypoint_xyz'}
        self.assertTrue(set(p['source_train_ids']+p['source_validation_ids'])<=allowed)
        self.assertEqual(p['target_distribution_rows'],0)
        self.assertEqual(p['complex_real_fit_rows'],0)
    def test_reference_not_decoder_input(self):
        _,rows=protocol();a=measured(rows[0])
        cut={k:a[k][:80] for k in ('b','t','valid','reset')}
        out=replay(cut)
        changed=dict(cut,q=np.random.default_rng(5).uniform(-100,100,(80,3)),mode='fake_template')
        other=replay(changed)
        np.testing.assert_array_equal(out['accepted'],other['accepted'])
        np.testing.assert_allclose(out['pred'],other['pred'],equal_nan=True,atol=0)
    def test_paths_safe_box_and_monotone_time(self):
        rng=np.random.default_rng(369100)
        for i in range(48):
            q,t,_=geometry(i,rng)
            self.assertTrue((np.diff(t)>0).all())
            self.assertTrue((np.abs(q[:,:2])<5.5).all())
            self.assertTrue(((q[:,2]>=-1-1e-12)&(q[:,2]<=2.4+1e-12)).all())

if __name__=='__main__':unittest.main()
