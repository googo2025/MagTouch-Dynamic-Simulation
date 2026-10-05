import unittest
import numpy as np
from training.forward_complex_r363 import axis_history
from training.sparse_field_r364 import harmonic_basis,completion_features


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.t=np.arange(80)*.1
        self.q=np.column_stack((np.sin(self.t),np.cos(self.t),1+.2*np.sin(2*self.t)))
        self.r=np.zeros(80,bool);self.r[0]=True
    def test_axis_prefix_causality(self):
        x=axis_history(self.q,self.t,(.3,1),self.r)
        np.testing.assert_allclose(x[:40],axis_history(self.q[:40],self.t[:40],(.3,1),self.r[:40]))
    def test_axis_additive_partition(self):
        x=axis_history(self.q,self.t,(.3,),self.r)
        total=axis_history(self.q,self.t,(.3,),self.r,False)
        np.testing.assert_allclose(x[:,:110]+x[:,110:],total,atol=1e-12)
    def test_static_hold_generates_no_history(self):
        q=np.tile([2.,-1.,1.4],(80,1))
        np.testing.assert_array_equal(axis_history(q,self.t,(.3,1),self.r),0)
    def test_pure_z_does_not_create_lateral_history(self):
        q=self.q.copy();q[:,:2]=0
        np.testing.assert_allclose(axis_history(q,self.t,(.3,),self.r)[:,:110],0,atol=1e-12)
    def test_reset_excludes_past(self):
        r=self.r.copy();r[40]=True;q=self.q.copy();q[:40]+=4
        np.testing.assert_allclose(axis_history(q,self.t,(.3,),r)[40:],axis_history(self.q,self.t,(.3,),r)[40:])
    def test_held_node_never_enters_inputs(self):
        b=np.random.default_rng(1).normal(size=(80,15));observed=np.arange(12)
        x=completion_features(b,self.t,self.r,observed);b[:,12:]+=10000
        np.testing.assert_array_equal(x,completion_features(b,self.t,self.r,observed))
    def test_field_prefix_causality(self):
        b=np.random.default_rng(1).normal(size=(80,15));obs=np.arange(12)
        np.testing.assert_allclose(completion_features(b,self.t,self.r,obs)[:40],completion_features(b[:40],self.t[:40],self.r[:40],obs))
    def test_harmonic_gradient_is_trace_free_and_symmetric(self):
        eps=.01;j=[]
        for axis in range(3):
            d=np.zeros(3);d[axis]=eps
            j.append((harmonic_basis([d])[0]-harmonic_basis([-d])[0])/(2*eps))
        j=np.stack(j,axis=1)
        np.testing.assert_allclose(j,j.transpose(1,0,2),atol=1e-12)
        np.testing.assert_allclose(np.einsum('iik->k',j),0,atol=1e-12)


if __name__=='__main__':unittest.main()
