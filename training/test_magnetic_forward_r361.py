import unittest
import numpy as np
from training.magnetic_forward_r361 import MagneticForward, design, spatial_features, accept_physics_export


class ForwardTests(unittest.TestCase):
    def setUp(self):
        self.t=np.arange(101)*.1
        self.q=np.column_stack((np.sin(self.t),np.cos(self.t),np.ones(101)))
        self.coef=np.random.default_rng(361).normal(size=(440,15))
        self.m=MagneticForward(self.coef,(.3,1.,4.))

    def test_prefix_causality(self):
        np.testing.assert_allclose(self.m.rollout(self.q,self.t)[:51],self.m.rollout(self.q[:51],self.t[:51]),atol=1e-12)

    def test_changed_future_has_no_effect_on_past(self):
        q=self.q.copy();q[51:]+=1.
        np.testing.assert_allclose(self.m.rollout(q,self.t)[:51],self.m.rollout(self.q,self.t)[:51],atol=1e-12)

    def test_no_field_argument(self):
        with self.assertRaises(TypeError): self.m.rollout(self.q,self.t,observed_B=np.zeros((101,15)))

    def test_unloaded_zero_static(self):
        q=self.q.copy();q[:,2]=-1
        np.testing.assert_array_equal(self.m.static(q),np.zeros((101,15)))
        np.testing.assert_array_equal(self.m.rollout(q,self.t),np.zeros((101,15)))

    def test_reset_equivalence(self):
        reset=np.zeros(101,bool);reset[51]=True
        # Exact timestamp history before a reset must not be reused.
        x=design(self.q,self.t,(.3,),reset)
        q=self.q.copy();q[:51]*=3
        np.testing.assert_allclose(x[51:],design(q,self.t,(.3,),reset)[51:])

    def test_physics_bridge_rejects_naive_multitouch(self):
        p={'schema':'MagTouch-Physics-Forward-1','t_s':[0,.1],
           'contacts_xyz_mm':np.zeros((2,2,3)).tolist(),'delta_b_uT':np.zeros((2,15)).tolist(),
           'provenance':{'construction':'sum_of_single_contacts'}}
        with self.assertRaises(ValueError): accept_physics_export(p)
        p['provenance']={}
        with self.assertRaises(ValueError): accept_physics_export(p)

    def test_invalid_time_rejected(self):
        with self.assertRaises(ValueError):self.m.rollout(self.q,self.t[::-1])
        invalid=self.t.copy();invalid[50]=np.nan
        with self.assertRaises(ValueError):self.m.rollout(self.q,invalid)

    def test_release_can_preserve_history(self):
        q=self.q.copy();q[50:,2]=-1
        # No hard zero after withdrawal: input is zero but states relax causally.
        p=self.m.rollout(q,self.t)
        self.assertGreater(float(np.linalg.norm(p[50])),0.)
        self.assertLess(float(np.linalg.norm(p[-1])),float(np.linalg.norm(p[50])))


if __name__=='__main__':unittest.main()
