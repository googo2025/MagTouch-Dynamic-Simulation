import unittest
import numpy as np
from training.virtual_expansion_r379 import weighted_gradients
from training.train_source_to_plateau_r372 import forward

class ExpansionTests(unittest.TestCase):
    def test_weighted_gradient(self):
        rng=np.random.default_rng(7)
        p=[rng.normal(0,.1,(3,4)),rng.normal(0,.1,4),rng.normal(0,.1,(4,4)),rng.normal(0,.1,4),rng.normal(0,.1,(4,2)),rng.normal(0,.1,2)]
        x=rng.normal(size=(8,3));y=rng.normal(size=(8,2));w=rng.uniform(.1,2,8);g=weighted_gradients(p,x,y,w)
        def loss():return np.mean(np.sum((forward(p,x)[2]-y)**2,axis=1)*w)+.0005*sum(np.sum(p[k]**2) for k in (0,2,4))
        for k in range(6):
            loc=(0,0) if p[k].ndim==2 else (0,);original=p[k][loc]
            p[k][loc]=original+1e-6;plus=loss();p[k][loc]=original-1e-6;minus=loss();p[k][loc]=original
            self.assertAlmostEqual(g[k][loc],(plus-minus)/2e-6,places=7)
    def test_full_traversal_has_no_sampling_replacement(self):
        order=np.random.default_rng(7).permutation(2571);parts=np.array_split(order,(len(order)+255)//256)
        self.assertEqual(len(np.unique(np.concatenate(parts))),2571)
        self.assertTrue(all(len(a)<=256 for a in parts))
if __name__=='__main__':unittest.main()
