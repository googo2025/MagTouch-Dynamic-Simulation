import unittest
import numpy as np
from training.broad_trajectory_r370 import geometry,STYLES,protocol,GEN,prior
from training.forward_complex_r363 import AxisForward

class BroadXYZTests(unittest.TestCase):
    def test_bounded_speed_and_positions(self):
        for role in ('train','validation','test'):
            for i in range(12):
                q,t,info=geometry(role,i)
                self.assertTrue(np.isfinite(q).all())
                self.assertTrue((np.diff(t)>0).all())
                self.assertLessEqual(np.max(np.linalg.norm(np.diff(q,axis=0),axis=1))/.1,2.+1e-8)
                self.assertTrue((np.abs(q[:,:2])<=5.+1e-8).all())
                self.assertTrue(((q[:,2]>=-1.-1e-8)&(q[:,2]<=2.4+1e-8)).all())
                self.assertEqual(info['style'],STYLES[i%4])
                self.assertEqual(q[-1,2],-1.)
    def test_reproducible_and_disjoint_roles(self):
        for i in range(8):
            a,_,_=geometry('train',i);b,_,_=geometry('train',i)
            np.testing.assert_array_equal(a,b)
            c,_,_=geometry('validation',i)
            self.assertFalse(a.shape==c.shape and np.array_equal(a,c))
    def test_source_only_protocol(self):
        p,rows=protocol()
        permitted={r['id'] for r in rows if r['device']=='source' and r['mode']!='waypoint_xyz'}
        self.assertTrue(set(p['source_train_ids']+p['source_validation_ids'])<=permitted)
        self.assertEqual(p['target_fit_rows'],0)
        self.assertEqual(p['actual_complex_selection_rows'],0)
    def test_causal_prefix_and_no_reference_input(self):
        q,t,_=geometry('train',7);q,t=q[:100],t[:100]
        b=AxisForward.load(GEN).rollout(q,t)
        a={'b':b,'t':t,'valid':np.ones(len(t),bool),'reset':np.r_[True,np.zeros(len(t)-1,bool)]}
        full=prior.replay(a)
        prefix={k:v[:60] for k,v in a.items()}
        first=prior.replay({**prefix,'q':np.full((60,3),999.),'mode':'not-a-decoder-input'})
        np.testing.assert_array_equal(first['accepted'],full['accepted'][:60])
        np.testing.assert_allclose(first['pred'],full['pred'][:60],atol=0,rtol=0,equal_nan=True)

if __name__=='__main__':unittest.main()
