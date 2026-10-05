"""Causality, streaming equivalence and selective XYZ contract tests."""
import unittest
import numpy as np
from training.unified_response_runtime_r330 import MagneticFeatureHistory, StreamingSourceResponse
from training.unified_tactile_policy_r330 import UnifiedOutputPolicy, LocalizationPolicy


def offline_response(time, field, eligible, runtime):
    from training.probe_r245_joint_contact_state import causal_state_features
    from training.audit_r263_magnetic_evidence_uncertain_depth import evidence_scores, confirm_evidence
    from training.probe_r276_condition_adaptation import run_scores
    from training.probe_r278_independent_direction import supplement
    from training.probe_r300_strength_conditioned_direction import replace
    x = causal_state_features(time, field, eligible)
    row = {"time": time, "field_uT": field, "eligible": eligible, "state_x": x,
           "amplitude_uT": np.linalg.norm(field, axis=1)}
    a, d = evidence_scores(row, runtime.null)
    row["response"] = confirm_evidence(row, a, d, runtime.null)
    scalar = x[:, list(range(10))+list(range(40, 43))+list(range(58, 61))]
    relative = np.column_stack((scalar, np.clip(scalar[:, 1:]/np.maximum(scalar[:, :1], .2), -10, 10)))
    from training.causal_response_sequence import sequence_features
    seq = sequence_features(row, "relative").reshape(len(time), -1)
    base = {"fresh": {"source": runtime.fresh_limit}, "change": {"source": runtime.change_limit}}
    p = runtime.protocols
    old = run_scores(row, [m.predict(scalar) for m in runtime.strength_models], base, p[276]["selected_policy"])
    old = supplement(row, old, runtime.direction278.predict(relative), p[278]["selected"]["policy"])
    old = supplement(row, old, runtime.direction282.predict(seq), p[282]["selected"]["policy"])
    return replace(row, old, runtime.direction300.predict(seq), runtime.context288.predict(seq),
                   p[300]["selected"]["policy"], p[288]["selected"], p[291]["selected"], p[293]["selected"])


class FeatureTests(unittest.TestCase):
    def test_matches_full_prefix_long_history_and_resets(self):
        from training.probe_r245_joint_contact_state import causal_state_features
        from training.causal_response_sequence import sequence_features
        rng = np.random.default_rng(330)
        t = np.arange(1700, dtype=np.int64)*25_000_000
        t[701:] += 400_000_000; t[801] = t[800]
        b = rng.normal(0, 4, (len(t), 15)).cumsum(0)
        ok = np.ones(len(t), bool); ok[900] = False
        expected = causal_state_features(t, b, ok)
        seq = sequence_features({"time": t, "state_x": expected, "eligible": ok}, "relative").reshape(len(t), -1)
        history = MagneticFeatureHistory()
        for i in range(len(t)):
            v = history.push(t[i], b[i], ok[i])
            np.testing.assert_allclose(v["state"], expected[i], atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(v["sequence"][0], seq[i], atol=1e-12, rtol=1e-12)
        self.assertLessEqual(len(history.times), 203)
        self.assertLessEqual(len(history.features), 63)

    def test_invalid_field_rejected(self):
        with self.assertRaises(ValueError): MagneticFeatureHistory().push(0, np.full(15, np.nan))


class ResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.runtime = StreamingSourceResponse()

    def test_streaming_matches_frozen_offline(self):
        rng = np.random.default_rng(330)
        t = np.arange(1700, dtype=np.int64)*25_000_000
        t[401:] += 400_000_000; t[1001] = t[1000]
        b = rng.normal(0, 8, (len(t), 15)).cumsum(0)
        b[300:400] *= .1; b[1100:] *= .05
        ok = np.ones(len(t), bool); ok[800:802] = False
        ref = offline_response(t, b, ok, self.runtime)
        self.runtime.reset()
        rows = [self.runtime.step(ti, bi, eligible=oi, reset_token=1) for ti, bi, oi in zip(t, b, ok)]
        for key in ("strength", "phase", "response"):
            np.testing.assert_array_equal([r[key] for r in rows], ref[key], err_msg=key)

    def test_baseline_token_resets_all_states(self):
        self.runtime.reset()
        b = np.arange(15)*20.
        for i in range(10): self.runtime.step(i*25_000_000, b, reset_token=1)
        result = self.runtime.step(250_000_000, b, reset_token=2)
        self.assertFalse(result["response"])
        self.assertEqual(len(self.runtime.history.times), 1)


class OutputTests(unittest.TestCase):
    def setUp(self): self.policy = UnifiedOutputPolicy()
    def response(self, **kw): return dict(eligible=True, response=True, strength=6, phase=3, **kw)

    def test_supported_valid_position_unchanged(self):
        xyz = [.2, -.4, 1.2]
        a = self.policy.decide(self.response(), xyz, decoder_valid=True)
        self.assertTrue(a["xyz_valid"]); self.assertEqual(a["xyz_mm"], xyz)
        self.assertTrue(a["trace_break"])
        self.assertFalse(self.policy.decide(self.response(), xyz, decoder_valid=True)["trace_break"])

    def test_direction_is_not_coordinate_permission(self):
        for phase in (2, 3, 4):
            v = self.policy.decide(dict(eligible=True, response=True, strength=2, phase=phase), [0, 0, .8], decoder_valid=True)
            self.assertFalse(v["xyz_valid"]); self.assertIsNone(v["xyz_mm"])

    def test_residue_blocked_even_with_direction_and_large_candidate(self):
        v = self.policy.decide(dict(eligible=True, response=True, strength=5, phase=4), [0, 0, 1.8], decoder_valid=True)
        self.assertEqual(v["system_state"], "residual"); self.assertIsNone(v["xyz_mm"])

    def test_decoder_invalid_not_rearmed(self):
        self.assertFalse(self.policy.decide(self.response(), [0, 0, .8], decoder_valid=False)["xyz_valid"])

    def test_domain_and_nonfinite_rejection(self):
        for xyz in ([0, 0, -.1], [0, 0, 3.2], [11, 0, 1], [0, np.nan, 1]):
            self.assertFalse(self.policy.decide(self.response(), xyz, decoder_valid=True)["xyz_valid"])
        with self.assertRaises(ValueError): LocalizationPolicy(depth_min_mm=2, depth_max_mm=1)

    def test_no_artificial_point_four_depth_gate(self):
        for z in (.25, .3):
            self.assertTrue(self.policy.decide(self.response(), [0, 0, z], decoder_valid=True)["xyz_valid"])

    def test_reentry_breaks_trace_and_does_not_hold_stale_coordinates(self):
        self.policy.decide(self.response(), [1, 1, 1], decoder_valid=True)
        rejected = self.policy.decide(dict(eligible=True, response=True, strength=2, phase=2), [1, 1, .5], decoder_valid=True)
        self.assertIsNone(rejected["xyz_mm"])
        accepted = self.policy.decide(self.response(), [2, 2, 1], decoder_valid=True)
        self.assertTrue(accepted["trace_break"])


if __name__ == "__main__": unittest.main()
