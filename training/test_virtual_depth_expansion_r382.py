import unittest
import numpy as np
from training.virtual_depth_expansion_r382 import RecordingModel, RecordingAdjudicator
from training.unified_depth_runtime_r341 import UnifiedDepthAdjudicator, UnifiedDepthPolicy

class RecorderTests(unittest.TestCase):
    def test_predict_is_unmodified_and_copy_is_private(self):
        class Model:
            def predict(self, x):
                return np.asarray(x).sum(1, keepdims=True)
        m = RecordingModel(Model())
        x = np.arange(59, dtype=float)[None]
        np.testing.assert_array_equal(m.predict(x), Model().predict(x))
        x[0, 0] = -1
        self.assertEqual(m.last[0, 0], 0)

    def test_policy_state_and_reset_are_identical(self):
        p = UnifiedDepthPolicy()
        direct = UnifiedDepthAdjudicator(p)
        wrapped = RecordingAdjudicator(UnifiedDepthAdjudicator(p))
        z = np.linspace(.4, 3, 40)
        phi = np.zeros(16)
        for i in range(6):
            cost = 1+(z-1.2-.01*i)**2
            args = (10_000_000_000+i*100_000_000, z, cost, phi, 1.1, .2)
            self.assertEqual(direct.step(*args, token=(1, 0)), wrapped.step(*args, token=(1, 0)))
        direct.reset(); wrapped.reset()
        self.assertIsNone(wrapped.last)
        self.assertEqual(direct.step(*args, token=(2, 0)), wrapped.step(*args, token=(2, 0)))

if __name__ == '__main__':
    unittest.main()
