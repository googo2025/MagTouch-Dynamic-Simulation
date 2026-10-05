"""Contracts needed to treat virtual Trials like research recordings."""
import json,unittest
from pathlib import Path
import numpy as np
from training.virtual_experiment_lab_r368 import ROOT,OUT,load_virtual_trial,sha
from training.simulation_depth_r368 import frozen_context,decode_core,prepare_depth,depth_predictions,z_score


class VirtualLabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog=json.loads((OUT/'data_catalog.json').read_text(encoding='utf8'))
        cls.ctx=frozen_context()

    def sample(self,n=35):
        # A varying field so a missing history reset is observable.
        atlas=self.ctx[0];t=np.arange(n)*.1
        q=np.column_stack((.3*np.sin(t),.2*np.cos(t),1.4+.1*np.sin(2*t)))
        return {'q':q,'b':atlas.forward(q),'t':t,'primary':np.ones(n,bool),'gate':np.ones(n,bool)}

    def decode(self,a):
        atlas,basis,pick,xy=self.ctx[:4]
        decode_core(a,atlas,basis,pick,xy);prepare_depth(a,atlas,basis)
        return depth_predictions(a,*self.ctx[4:])[0]

    def test_reference_not_in_observation_contract(self):
        a=load_virtual_trial(ROOT/self.catalog['trials'][1]['path'])
        self.assertEqual(set(a['observation']),{'t_s','delta_b_uT','sample_valid','reset'})
        self.assertEqual(a['observation']['delta_b_uT'].shape[1],15)
        self.assertNotIn('q_mm',a['observation'])

    def test_stress_requires_explicit_opt_in(self):
        e=next(r for r in self.catalog['trials'] if r['mechanism_stress'])
        with self.assertRaisesRegex(ValueError,'explicit opt-in'):load_virtual_trial(ROOT/e['path'])
        self.assertTrue(load_virtual_trial(ROOT/e['path'],allow_stress=True)['meta']['mechanism_stress'])

    def test_exact_counterfactual_triplets_same_path_and_noise(self):
        for g in (0,53,108,188,215):
            entries=[r for r in self.catalog['trials'] if r['geometry_id']==g]
            data=[load_virtual_trial(ROOT/r['path'],allow_stress=True) for r in entries]
            for a in data[1:]:
                np.testing.assert_array_equal(a['reference']['q_mm'],data[0]['reference']['q_mm'])
                np.testing.assert_array_equal(a['observation']['t_s'],data[0]['observation']['t_s'])
            b=[a['observation']['delta_b_uT'] for a in data]
            np.testing.assert_allclose(b[0]+b[2]-2*b[1],0,atol=1e-10)
            self.assertEqual(len({e['noise_seed'] for e in entries}),1)

    def test_unique_geometry_count_not_pseudoreplication(self):
        rows=self.catalog['trials']
        self.assertEqual(len(rows),648);self.assertEqual(len({r['geometry_id'] for r in rows}),216)
        self.assertEqual(sum(r['frames'] for r in rows),340506)
        self.assertTrue(all(r['role']=='mechanism_test' for r in rows))

    def test_depth_inference_does_not_read_reference(self):
        a=self.sample();b={k:v.copy() for k,v in a.items()};b['q']+=100;b['primary'][:]=False
        pa,pb=self.decode(a),self.decode(b)
        for k in pa:np.testing.assert_array_equal(pa[k],pb[k])

    def test_causal_prefix_equal_for_all_depth_stages(self):
        a=self.sample();b={k:v[:20].copy() for k,v in a.items()}
        pa,pb=self.decode(a),self.decode(b)
        for k in pa:np.testing.assert_allclose(pa[k][:20],pb[k],atol=1e-12)

    def test_depth_ablations_have_identical_xy_and_permission(self):
        a=self.sample();a['gate'][8:10]=False;p=self.decode(a)
        for val in p.values():
            np.testing.assert_array_equal(np.isfinite(val).all(1),a['gate'])
            np.testing.assert_array_equal(val[:,:2],p['static_z'][:,:2])

    def test_stationary_points_not_counted_as_motion_jitter(self):
        a=self.sample();a['independent_frames']=True
        p=self.decode(a)['fast_slow'];s=z_score(a,p)
        self.assertEqual(s['constant_reference_z_steps']['pairs'],0)
        self.assertEqual(s['z_error_step_p95_mm'],0)

    def test_virtual_depth_models_shapes_and_no_real_training(self):
        from training.simulation_depth_r368 import DEPTH
        d=json.loads((DEPTH/'declared.json').read_text(encoding='utf8'))
        self.assertEqual(d['fit_real_frames'],0);self.assertFalse(d['formal_inverse_weights_loaded'])
        self.assertEqual(self.ctx[4].params[0].shape,(56,32))
        self.assertEqual(self.ctx[7].params[0].shape,(59,32))
        self.assertFalse(set(d['training_ids'])&set(d['validation_ids']))
        self.assertTrue(all(not k.startswith('VIRTUAL_') for k in d['training_ids']+d['validation_ids']))


if __name__=='__main__':unittest.main()
