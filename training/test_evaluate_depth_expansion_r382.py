import copy
import unittest
import numpy as np
from training.evaluate_depth_expansion_r382 import check_freezes, pair_changes, replay_input


class DepthEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.xy = {'status': 'COMPLETED_PLATEAU', 'target_fit_or_selection': False,
                   'production_replaced': False, 'model_sha256': 'xy-frozen'}
        self.depth = {'status': 'COMPLETED_PLATEAU', 'target_fit_or_selection': False,
                      'production_replaced': False, 'model_sha256': 'z-frozen',
                      'selected_xy_sha256': 'xy-frozen', 'source_only': True,
                      'qualified_candidate': True}

    def test_no_budget_stop_or_unqualified_candidate_can_be_evaluated(self):
        check_freezes(self.xy, self.depth)
        for field, value in (('status', 'BUDGET_REACHED_NOT_CONVERGED'),
                             ('qualified_candidate', False),
                             ('target_fit_or_selection', True),
                             ('selected_xy_sha256', 'other-xy')):
            bad = copy.deepcopy(self.depth)
            bad[field] = value
            with self.assertRaises(AssertionError):
                check_freezes(self.xy, bad)

    def test_inference_needs_no_reference_and_never_uses_future_input(self):
        class Engine:
            def __init__(self):
                self.calls = []
                self.resets = 0
            def set_baseline(self, b):
                self.resets += 1
                np.testing.assert_array_equal(b, np.zeros(15))
            def push(self, now, b, *, eligible):
                self.calls.append((now, b.copy(), eligible))
                return {'valid': eligible, 'xyz': [b[0], b[1], b[2]],
                        'extra': {'localization_path': 'test',
                                  'localization_permission_score': .9 if eligible else None,
                                  'final_depth_evidence': {'source_support_distance': .2}}}
        e = Engine()
        b = np.arange(60, dtype=float).reshape(4, 15)
        b[1] = np.nan
        a = {'t': np.array([0., .1, .2, .3]), 'b': b,
             'input_eligible': np.array([True, False, True, True]),
             'reset': np.array([True, False, True, False])}
        output = replay_input(a, e)
        self.assertEqual(e.resets, 3)
        np.testing.assert_array_equal(output['ok'], a['input_eligible'])
        self.assertTrue(np.isnan(output['pred'][1]).all())
        np.testing.assert_array_equal(e.calls[1][1], np.zeros(15))
        for i in (0, 2, 3):
            np.testing.assert_array_equal(output['pred'][i], b[i, :3])
            self.assertEqual(e.calls[i][0], 10_000_000_000+i*100_000_000)

    def test_added_lost_counts_do_not_include_nonprimary_frames(self):
        a = {'primary': np.array([True, True, False, False])}
        before = {'ok': np.array([True, False, True, False]),
                  'permission': np.array([.8, .7, .5, .6]), 'support': np.ones(4)}
        after = {'ok': np.array([False, True, False, True]),
                 'permission': np.array([.9, .9, .5, .6]), 'support': np.ones(4)*1.1}
        result = pair_changes(a, before, after)
        self.assertEqual(result['primary_outputs_lost'], 1)
        self.assertEqual(result['primary_outputs_added'], 1)
        self.assertEqual(result['permission_absolute_change']['n'], 2)
        self.assertAlmostEqual(result['support_absolute_change']['median'], .1)


if __name__ == '__main__':
    unittest.main()
