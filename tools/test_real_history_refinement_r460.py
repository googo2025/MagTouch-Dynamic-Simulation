"""Offline mathematical fixtures only; no hardware or scientific fitting."""
import unittest
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'training'))
from real_history_refinement_r460 import (MultiscaleXYHistory, augment_existing,
    forward, batch_loss, torch_prediction, ARMS)
from training.had239_causal_xy_features_r165 import CausalXYFeatureHistory


class HistoryTests(unittest.TestCase):
    def fixture(self, n=24):
        rng = np.random.default_rng(460)
        t = np.arange(n, dtype=np.int64)*25_000_000+10_000_000_000
        phi = rng.normal(size=(n, 16))
        xyz = rng.normal(size=(n, 3))
        memory = rng.normal(size=(n, 2))
        return t, phi, xyz, memory

    def test_existing53_unchanged_and_prefix_causal(self):
        t, phi, xyz, memory = self.fixture()
        a, b = CausalXYFeatureHistory(), MultiscaleXYHistory()
        out = []
        for i in range(len(t)):
            kw = dict(time_ns=t[i], b0_id=0, stream_id=0, phi=phi[i], xyz=xyz[i], memory=memory[i])
            old, new = a.push(**kw), b.push(**kw)
            np.testing.assert_array_equal(old, new[:53]); out.append(new)
        b = MultiscaleXYHistory()
        for i in range(10):
            new = b.push(time_ns=t[i], b0_id=0, stream_id=0, phi=phi[i], xyz=xyz[i], memory=memory[i])
            np.testing.assert_array_equal(out[i], new)

    def test_resets_gap_b0_stream_time(self):
        for change in ('gap', 'b0', 'stream', 'time'):
            h = MultiscaleXYHistory()
            kw = dict(time_ns=1_000_000_000, b0_id=0, stream_id=0, phi=np.zeros(16), xyz=np.zeros(3), memory=np.zeros(2))
            h.push(**kw)
            kw.update(time_ns=1_025_000_000, phi=np.ones(16))
            self.assertGreater(np.linalg.norm(h.push(**kw)[53:]), 0)
            kw.update(time_ns=1_050_000_000, phi=np.full(16, 2.))
            if change == 'gap': kw['time_ns'] = 2_000_000_000
            if change == 'b0': kw['b0_id'] = 1
            if change == 'stream': kw['stream_id'] = 1
            if change == 'time': kw['time_ns'] = 1_025_000_000
            np.testing.assert_array_equal(h.push(**kw)[53:], np.zeros(48))

    def test_offline_and_incremental_identical(self):
        t, phi, xyz, memory = self.fixture()
        valid = np.ones(len(t), bool); valid[10] = False
        h = CausalXYFeatureHistory(); x = np.zeros((len(t), 53))
        for i in range(len(t)):
            if not valid[i]: h.reset(); continue
            x[i] = h.push(time_ns=t[i], b0_id=0, stream_id=0, phi=phi[i], xyz=xyz[i], memory=memory[i])
        ext = augment_existing(x, t, valid)
        np.testing.assert_allclose(ext[:, :53], x, atol=1e-12, rtol=0)
        np.testing.assert_array_equal(ext[11, 53:], np.zeros(48))

    def test_zero_extension_preserves_original_network(self):
        rng = np.random.default_rng(7)
        original = [rng.normal(size=s)*.03 for s in ((53, 64), (64,), (64, 64), (64,), (64, 2), (2,))]
        extended = [v.copy() for v in original]
        extended[0] = np.vstack([extended[0], np.zeros((48, 64))])
        x = rng.normal(size=(100, 101))
        np.testing.assert_allclose(forward(original, x[:, :53])[2], forward(extended, x)[2], atol=1e-14, rtol=0)

    def test_full_epoch_no_replacement(self):
        for n in (565, 7782, 14001):
            order = np.random.default_rng(7).permutation(n)
            batches = np.array_split(order, (n+255)//256)
            np.testing.assert_array_equal(np.sort(np.concatenate(batches)), np.arange(n))


class GradientTests(unittest.TestCase):
    def test_loss_autograd_finite_difference_and_motion_target(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Torch backend fixture runs on server before launch')
        rng = np.random.default_rng(7)
        ps = [torch.nn.Parameter(torch.tensor(rng.normal(size=s)*.03, dtype=torch.float64))
              for s in ((101, 4), (4,), (4, 4), (4,), (4, 2), (2,))]
        a = {k: torch.tensor(v, dtype=torch.float64) for k, v in dict(
            u=rng.normal(size=(5, 101)), prev_u=rng.normal(size=(5, 101)), y=rng.normal(size=(5, 2)),
            guard=rng.normal(size=(5, 2)), prev_guard=rng.normal(size=(5, 2)), dq=rng.normal(size=(5, 2)),
            weight=np.ones(5), pair_weight=np.array([1, 1, 0, 1, 0]),
            hold_weight=np.array([0, 1, 0, 0, 0]), anchor_weight=np.array([0, 0, 0, 0, 1])).items()}
        w = dict(increment=.25, hold=.25, anchor=.25)
        for arm in ARMS:
            for p in ps: p.grad = None
            batch_loss(ps, a, arm, w).backward()
            for pi, coord in ((0, (15, 2)), (2, (1, 1)), (4, (1, 0)), (5, (0,))):
                p = ps[pi]; analytic = p.grad[coord].item(); original = p[coord].item(); eps = 1e-6
                with torch.no_grad():
                    p[coord] = original+eps; plus = batch_loss(ps, a, arm, w).item()
                    p[coord] = original-eps; minus = batch_loss(ps, a, arm, w).item(); p[coord] = original
                self.assertAlmostEqual(analytic, (plus-minus)/(2*eps), places=6)
        # Correct reference movement has zero increment penalty, not shrink-to-zero.
        with torch.no_grad():
            a['dq'] = a['guard']+torch_prediction(ps, a['u'])-a['prev_guard']-torch_prediction(ps, a['prev_u'])
            a['hold_weight'].zero_(); a['anchor_weight'].zero_()
        self.assertAlmostEqual(batch_loss(ps, a, ARMS[0], w).item(), batch_loss(ps, a, ARMS[1], w).item(), places=12)


if __name__ == '__main__': unittest.main()
