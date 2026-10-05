import json, unittest
import numpy as np
from training.simulation_discovery_r367 import OUT, ForwardAtlas, prepare, guard_and_features, evaluate


class VirtualOnlyDiscoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        a = np.load(OUT / 'virtual_static_calibration.npz')
        cls.atlas = ForwardAtlas(a['q'], a['b'], 'unit_logamp')
        cls.basis = np.load(OUT / 'virtual_memory_basis.npz')['basis']

    def stream(self, n=25):
        t=np.arange(n)*.1;q=np.tile([0.,0.,1.4],(n,1));q[:,0]=.3*np.sin(t)
        b=self.atlas.forward(q)
        return {'b':b,'t':t,'q':q,'gate':np.ones(n,bool)}

    def decode(self,a):
        prepare(a,self.atlas,self.basis)
        guard_and_features(a,self.atlas,self.basis,.8,1.)
        return a

    def test_no_command_position_read_by_inverse(self):
        a=self.stream();b={k:v.copy() for k,v in a.items()};b['q']+=100
        a,b=self.decode(a),self.decode(b)
        for key in ('base','memory','guarded','h','x'):
            np.testing.assert_array_equal(a[key],b[key])

    def test_future_cannot_change_prefix(self):
        a=self.stream();b={k:v[:15].copy() for k,v in a.items()}
        a,b=self.decode(a),self.decode(b)
        for key in ('base','memory','guarded','h','x'):
            np.testing.assert_allclose(a[key][:15],b[key],atol=1e-12)

    def test_independent_stationary_frames_have_no_inherited_lags(self):
        a=self.stream();a['independent_frames']=True;a=self.decode(a)
        np.testing.assert_array_equal(a['x'][:,21:],0.)

    def test_permission_is_identical_across_position_ablations(self):
        a=self.stream();a['gate'][5:7]=False;a=self.decode(a)
        for key in ('base','memory','guarded'):
            np.testing.assert_array_equal(np.isfinite(a[key]).all(1),a['gate'])

    def test_release_clears_lag_state(self):
        a=self.stream();a['gate'][5:7]=False;a=self.decode(a)
        np.testing.assert_array_equal(a['x'][7,21:],0.)

    def test_z_unchanged_in_this_xy_core_ablation(self):
        a=self.decode(self.stream())
        np.testing.assert_array_equal(a['base'][:,2],a['memory'][:,2])
        np.testing.assert_array_equal(a['base'][:,2],a['guarded'][:,2])

    def test_complex_paths_never_enter_inverse_training_or_selection(self):
        d=json.loads((OUT/'replication_protocol.json').read_text(encoding='utf8'))
        train=set(d['training_ids']);valid=set(d['validation_ids']);test=set(d['complex_test_ids'])
        self.assertFalse(train&valid);self.assertFalse(test&(train|valid));self.assertEqual(len(test),60)
        self.assertEqual(d['learned_inverse_real_samples'],0)

    def test_same_network_shapes_not_arbitrary_readout(self):
        for name in ('causal_no_static_anchors','causal_static_anchors','current_only_static_anchors'):
            a=np.load(OUT/(name+'.npz'))
            self.assertEqual(a['w1'].shape,(53,64));self.assertEqual(a['w2'].shape,(64,64))
            self.assertEqual(a['w3'].shape,(64,2))


if __name__=='__main__':
    unittest.main()
