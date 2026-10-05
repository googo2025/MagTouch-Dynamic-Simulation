import unittest
import numpy as np
from tools.validate_depth_evaluation_r382 import independent_scores, check_scores
from training.research_formal_r365 import metrics
from training.audit_virtual_expansion_r379 import interpretation


class EvaluationArithmeticTests(unittest.TestCase):
    def test_rejected_output_is_not_zero_error(self):
        q=np.ones((4,3));pred=np.full((4,3),np.nan)
        r=independent_scores(pred,q,np.arange(4)*.1,np.ones(4,bool),np.zeros(4,bool))
        self.assertEqual(r['coverage'],0.)
        self.assertEqual(r['output_frames'],0)
        self.assertNotIn('z_mae_mm',r)
        check_scores(r,metrics(pred,q,np.arange(4)*.1,np.ones(4,bool),np.zeros(4,bool)))

    def test_constant_translation_does_not_remove_jump_counts(self):
        q=np.ones((5,3));pred=q.copy();pred[:,2]=[.7,.95,1.2,1.45,1.7]
        t=np.arange(5)*.1;mask=np.ones(5,bool)
        before=independent_scores(pred,q,t,mask,mask)
        after=independent_scores(pred+[3.,-2.,.4],q,t,mask,mask)
        self.assertEqual(before['constant_reference_z_steps'],after['constant_reference_z_steps'])
        self.assertEqual(before['constant_reference_z_steps']['over_0p2_mm'],4)
        self.assertNotEqual(before['xy_mean_mm'],after['xy_mean_mm'])
        self.assertAlmostEqual(before['xy_translation_removed_mm'],after['xy_translation_removed_mm'])

    def test_missing_outputs_are_not_connected_for_jump_statistics(self):
        q=np.ones((3,3));pred=q.copy();pred[1]=np.nan;pred[2,2]=2.
        ok=np.array([True,False,True]);mask=np.ones(3,bool)
        r=independent_scores(pred,q,np.arange(3)*.1,mask,ok)
        self.assertEqual(r['constant_reference_z_steps']['pairs'],0)
        self.assertEqual(r['constant_reference_z_steps']['over_0p2_mm'],0)
        self.assertAlmostEqual(r['z_mae_mm'],.5)

    def test_independent_arithmetic_matches_metric_contract(self):
        rng=np.random.default_rng(382)
        q=rng.normal(size=(200,3));q[:,2]=1.4;q[31]=np.nan
        pred=q+rng.normal(0,.2,size=q.shape);ok=rng.uniform(size=200)>.1
        pred[~ok]=np.nan;mask=rng.uniform(size=200)>.2
        t=np.cumsum(rng.uniform(.05,.35,size=200))
        check_scores(independent_scores(pred,q,t,mask,ok),metrics(pred,q,t,mask,ok))

    def test_report_interpretation_is_derived_not_previous_result(self):
        for sign in (-1,1):
            g={d:{'raw_xy_change_pct':sign*7.5} for d in ('source','target','virtual')}
            result=interpretation(g)
            expected='decreases' if sign<0 else 'increases'
            for d in g:self.assertIn(expected,result[d])


if __name__=='__main__':unittest.main()
