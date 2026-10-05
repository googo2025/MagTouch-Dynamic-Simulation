import unittest
import numpy as np
from training.sequence_ambiguity_r380 import pair_peak,permitted
from training.research_ambiguity_r366 import causal_windows

class SequenceTests(unittest.TestCase):
    def test_history_peak_is_nested_not_diluted(self):
        x=np.zeros((2,21,15));x[1,0,:]=3.;x[1,10,:]=6.
        p=[(0,1)]
        self.assertEqual(pair_peak(x,p,0)[0],3.)
        self.assertEqual(pair_peak(x,p,4)[0],3.)
        self.assertEqual(pair_peak(x,p,10)[0],6.)

    def test_nominal_separation_and_same_trial_gap(self):
        q=np.array([[0,0,1],[2,0,1],[0,0,1.4],[.1,0,1.05]])
        permitted_pairs,same=permitted(q,np.zeros(4,int),np.array([0.,1.,5.,8.]))
        self.assertFalse(permitted_pairs[0,1])
        self.assertTrue(permitted_pairs[0,2])
        self.assertFalse(permitted_pairs[0,3])

    def test_causal_windows_exclude_reset_and_future(self):
        t=np.arange(35)*.1;b=np.repeat(t[:,None],15,axis=1);reset=np.zeros(35,bool);reset[[0,20]]=True
        values,good=causal_windows(b,t,reset,np.array([19,21,34]),1.)
        self.assertEqual(good.tolist(),[True,False,True])
        for row,endpoint in zip(values,[19,21,34]):self.assertLessEqual(row.max(),t[endpoint])

if __name__=='__main__':unittest.main()
