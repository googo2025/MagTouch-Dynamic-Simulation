import unittest
import numpy as np
from training import virtual_discovery_r387 as r

class RepresentationTests(unittest.TestCase):
    def setUp(self):
        self.b=np.random.default_rng(91).normal(size=(75,15));self.t=np.arange(75)*.1
        self.reset=np.r_[True,np.zeros(74,bool)]
    def test_prefix_causality(self):
        full=r.representation(self.b,self.t,self.reset,'multiscale_history')
        prefix=r.representation(self.b[:27],self.t[:27],self.reset[:27],'multiscale_history')
        np.testing.assert_array_equal(full[:27],prefix)
    def test_future_mutation_irrelevant(self):
        altered=self.b.copy();altered[30:]+=10000
        a=r.representation(self.b,self.t,self.reset,'multiscale_history')
        b=r.representation(altered,self.t,self.reset,'multiscale_history')
        np.testing.assert_array_equal(a[:30],b[:30])
    def test_disabled_slots(self):
        np.testing.assert_array_equal(r.representation(self.b,self.t,self.reset,'current')[:,15:],0)
        np.testing.assert_array_equal(r.representation(self.b,self.t,self.reset,'short_history')[:,30:],0)
    def test_reset_and_gap(self):
        reset=self.reset.copy();reset[30]=True
        x=r.representation(self.b,self.t,reset,'multiscale_history')
        np.testing.assert_array_equal(x[30,15:],0)
        t=self.t.copy();t[40:]+=1
        np.testing.assert_array_equal(r.representation(self.b,t,self.reset,'multiscale_history')[40,15:],0)
    def test_constant_field_zero_history(self):
        x=r.representation(np.ones_like(self.b),self.t,self.reset,'multiscale_history')
        np.testing.assert_allclose(x[:,15:],0,atol=1e-7)
    def test_identical_initialization(self):
        a,b=r.initial(),r.initial()
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
        self.assertEqual(sum(v.size for v in a),2051)
    def test_role_disjointness(self):
        roles={k:{str(r.path(k,i)) for i in range(n)} for k,n in r.COUNTS.items()}
        self.assertFalse(roles['train']&roles['validation']);self.assertFalse(roles['train']&roles['test'])
        self.assertFalse(roles['validation']&roles['test'])

if __name__=='__main__':unittest.main()
