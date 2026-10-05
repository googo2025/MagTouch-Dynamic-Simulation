"""Training math/control tests, not historical-model training reproduction."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from training.train_source_to_plateau_r372 import Plateau, forward, gradients_xy, gradients_z, checkpoint, depth_kernel, fast_depth


class PlateauTrainingTests(unittest.TestCase):
    def test_exact_fifty_epoch_patience(self):
        scheduler = Plateau(.001)
        self.assertEqual(scheduler.observe(1.), 'continue')
        for _ in range(49):
            self.assertEqual(scheduler.observe(1.), 'continue')
        self.assertEqual(scheduler.observe(1.), 'reduce_lr')
        self.assertEqual(scheduler.lr, .0005)
        self.assertEqual(scheduler.bad_epochs, 0)
        self.assertIsNone(scheduler.reference)

    def test_lower_metric_is_improvement(self):
        s = Plateau(.001)
        s.observe(1.)
        for i in range(100):
            self.assertEqual(s.observe(.99**(i+1)), 'continue')
            self.assertEqual(s.bad_epochs, 0)
        self.assertEqual(s.reductions, 0)

    def test_small_improvements_accumulate(self):
        s = Plateau(.001)
        s.observe(1.)
        s.observe(.99995)
        self.assertEqual(s.bad_epochs, 1)
        s.observe(.9998)
        self.assertEqual(s.bad_epochs, 0)

    def test_floor_then_new_patience(self):
        s = Plateau(2e-6)
        s.observe(1.)
        for _ in range(50):
            event = s.observe(1.)
        self.assertEqual(event, 'reduce_lr')
        self.assertEqual(s.lr, 1e-6)
        self.assertEqual(s.observe(1.), 'continue')
        for _ in range(49):
            self.assertEqual(s.observe(1.), 'continue')
        self.assertEqual(s.observe(1.), 'early_stop_min_lr_plateau')

    def test_numeric_failure_is_not_early_stop(self):
        with self.assertRaises(ValueError):
            Plateau(.001).observe(float('nan'))

    def params(self):
        rng = np.random.default_rng(372)
        return [rng.normal(0, .1, (3, 4)), rng.normal(0, .1, 4),
                rng.normal(0, .1, (4, 4)), rng.normal(0, .1, 4),
                rng.normal(0, .1, (4, 2)), rng.normal(0, .1, 2)]

    def check_derivative(self, p, grads, loss):
        for k in range(6):
            loc = (0, 0) if p[k].ndim == 2 else (0,)
            original = p[k][loc]
            p[k][loc] = original+1e-6
            plus = loss()
            p[k][loc] = original-1e-6
            minus = loss()
            p[k][loc] = original
            self.assertAlmostEqual(grads[k][loc], (plus-minus)/2e-6, places=7)

    def test_xy_gradient_original_objective(self):
        rng = np.random.default_rng(7)
        p = self.params(); x = rng.normal(size=(5, 3)); y = rng.normal(size=(5, 2))
        grads = gradients_xy(p, x, y)
        def loss():
            return np.mean(np.sum((forward(p, x)[2]-y)**2, axis=1))+.0005*sum(np.sum(p[k]**2) for k in (0, 2, 4))
        self.check_derivative(p, grads, loss)

    def test_depth_gradient_original_objective(self):
        rng = np.random.default_rng(348)
        p = self.params(); p[4] = p[4][:, :1]; p[5] = p[5][:1]
        x = rng.normal(size=(8, 3)); y = rng.normal(size=(8, 1)); w = np.linspace(.5, 1.5, 8)
        ids = np.array([0, 3, 4, 7]); ia = np.array([0, 1, 2, 3]); ib = ia+1
        grads = gradients_z(p, x[ids], y[ids], w[ids], ia, ib, x, y, w)
        def loss():
            pred = forward(p, x)[2]
            delta = pred[ib]-pred[ia]-(y[ib]-y[ia])
            return (np.mean((pred[ids]-y[ids])**2*w[ids, None])
                    +.5*np.mean(pred[ids]**2*w[ids, None])
                    +.5*np.mean(delta**2*((w[ia]+w[ib])/2)[:, None])
                    +.0005*sum(np.sum(p[k]**2) for k in (0, 2, 4)))
        self.check_derivative(p, grads, loss)

    def test_checkpoint_keeps_rng_optimizer_and_controller(self):
        p = self.params(); rng = np.random.default_rng(7); rng.permutation(25)
        s = Plateau(.001); s.observe(.5); s.observe(.6)
        state = {'controller': vars(s), 'rng': rng.bit_generator.state, 'epoch': 2}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'state.npz'
            checkpoint(path, p, p, p, p, state)
            with np.load(path, allow_pickle=False) as f:
                loaded = json.loads(str(f['state_json']))
                np.testing.assert_array_equal(f['m0'], p[0])
                np.testing.assert_array_equal(f['best0'], p[0])
            restored = np.random.default_rng(0); restored.bit_generator.state = loaded['rng']
            np.testing.assert_array_equal(rng.permutation(200), restored.permutation(200))
            self.assertEqual(vars(Plateau(**loaded['controller'])), vars(s))

    def test_jit_matches_pure_depth_kernel(self):
        rng = np.random.default_rng(10); n = 80
        valid = np.ones(n, bool); valid[[0, 29]] = False
        z = np.linspace(.4, 3., 131); count = int(valid.sum())
        centre = rng.uniform(.5, 2.8, count)
        cost = (z[None, :]-centre[:, None])**2+.01
        args = (np.arange(n, dtype=np.int64)*100000000, valid, np.ones(n),
                np.r_[np.zeros(40, np.int64), np.ones(40, np.int64)], z, cost,
                rng.normal(size=(count, 16)), centre+.05, np.ones(count)*.2,
                4., .01, .45, 2., 8., .25)
        np.testing.assert_allclose(depth_kernel(*args), fast_depth(*args), atol=1e-10, rtol=0)


if __name__ == '__main__':
    unittest.main()
