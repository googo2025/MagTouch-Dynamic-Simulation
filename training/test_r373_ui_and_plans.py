"""Hardware-free deployment parity and full-plan safety checks."""
import copy
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code'))
import magtouch_acquisition_ui_v1_3_4_R373 as ui
from live_retrained_reconstruction_r373 import (
    PRESET_NAME, GATE_NAME, COMPONENTS, FAMILY, TRAINING,
    DirectTrainedEngineR373, LiveRetrainedPredictorR373, frozen_components,
)
from live_fast_slow_reconstruction_r348 import (
    PRESET_NAME as OLD_PRESET, LiveFastSlowReconstructionPredictorR348,
    FastSlowReconstructionEngineR348,
)
from live_algorithm_ui_r92 import R92_PROFILE_PATH
from live_algorithm_ui_r126 import PROFILE_KEY
from replay_backtest import load_profile
from training.waypoint_xyz_protocol_r136 import validate_frozen_source_plan, path_sha256, compile_waypoint_path
from training.build_r373_retrained_online_plans import make, EXACT_GEOMETRY
from training.train_source_to_plateau_r372 import sha


class R373UIAndPlans(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = load_profile(R92_PROFILE_PATH, PROFILE_KEY)
        cls.d = cls.profile['defaults']

    def predictor(self):
        p = LiveRetrainedPredictorR373(self.profile, R92_PROFILE_PATH, PRESET_NAME,
            self.d['threshold'], self.d['min_cosine'], self.d['smoothing_ms'], 70., GATE_NAME)
        p.metadata['model_training_profile_id'] = 'magnetic_layer_v1_5*5-had-239'
        return p

    def host(self, plan, predictor):
        profile_id = plan['paper_target_profile_id']
        h = SimpleNamespace(live_predictor=predictor, live_predict_enabled=True,
            live_predict_lock=threading.Lock(), active_mag_profile_id=profile_id,
            maglayer_profiles={profile_id: {'calibration': {'contact_z_mm': plan['reference_contact_z_mm']},
                'magnetic_layer_id': 'magnetic_layer_v4', 'tactile_material_id': 'silicone_v4'}})
        h._grid_offsets = ui.core.MagTouchApp._grid_offsets
        return h

    def test_old_weights_and_old_selector_are_preserved(self):
        manifest = json.loads((TRAINING/'data_manifest.json').read_text(encoding='utf8'))
        from live_algorithm_ui_r131 import MODEL_SPECS
        self.assertIs(MODEL_SPECS[OLD_PRESET].predictor_type, LiveFastSlowReconstructionPredictorR348)
        self.assertIs(MODEL_SPECS[PRESET_NAME].predictor_type, LiveRetrainedPredictorR373)
        self.assertIs(ui.core.LiveAlgorithmPanel, ui.R373LiveAlgorithmPanel)
        for component in ('xy', 'z'):
            old = manifest[component]['old_model']
            self.assertEqual(sha(ROOT/old['path']), old['sha256'])

    def test_complete_model_keeps_shallow_and_z_protection(self):
        p = self.predictor()
        old = FastSlowReconstructionEngineR348(self.profile, self.d['threshold'],
            self.d['min_cosine'], self.d['smoothing_ms'], 70.)
        self.assertEqual(p.engine.response_observer.manifest, old.response_observer.manifest)
        self.assertEqual(p.engine.response_observer.protocols, old.response_observer.protocols)
        self.assertEqual(p.engine.permission.manifest, old.permission.manifest)
        self.assertEqual(p.engine.permission.selected, old.permission.selected)
        self.assertEqual(p.engine.final_refiner.manifest, old.final_refiner.manifest)
        self.assertEqual(p.engine.final_refiner.selection, old.final_refiner.selection)
        for key in ('threshold', 'min_cosine', 'contact_threshold', 'window_ns'):
            self.assertEqual(getattr(p.engine, key), getattr(old, key))
        np.testing.assert_array_equal(p.engine.memory_decoder.xy_model.mean,
            old.memory_decoder.xy_model.mean)
        np.testing.assert_array_equal(p.engine.memory_decoder.xy_model.scale,
            old.memory_decoder.xy_model.scale)
        np.testing.assert_array_equal(p.engine.final_refiner.model.mean,
            old.final_refiner.model.mean)
        np.testing.assert_array_equal(p.engine.final_refiner.model.scale,
            old.final_refiner.model.scale)
        self.assertEqual(p.family, FAMILY)
        self.assertEqual(p.metadata['contact_gate'], GATE_NAME)
        self.assertFalse(p.metadata['legacy_gate_applied'])
        self.assertTrue(p.metadata['shallow_response_weights_unchanged'])
        self.assertTrue(p.metadata['z_evidence_policy_unchanged'])
        self.assertIs(p.engine.response_observer.history, p.engine.permission.history)
        self.assertEqual(p.engine.final_refiner.selection['dimension'], 59)
        self.assertEqual(p.metadata['direct_training']['xy']['best_epoch'], 344)
        self.assertEqual(p.metadata['direct_training']['z']['best_epoch'], 117)
        self.assertEqual(p.metadata['xy_mlp_artifact_sha256'], COMPONENTS['xy']['model_sha256'])
        self.assertEqual(p.metadata['depth_increment_artifact_sha256'], COMPONENTS['z']['model_sha256'])

    def test_wrong_gate_or_modified_frozen_weight_is_rejected(self):
        with self.assertRaisesRegex(ValueError, '独立状态'):
            LiveRetrainedPredictorR373(self.profile, R92_PROFILE_PATH, PRESET_NAME,
                self.d['threshold'], self.d['min_cosine'], self.d['smoothing_ms'], 70., 'legacy')
        with patch('live_retrained_reconstruction_r373.sha', return_value='wrong'):
            with self.assertRaises(ValueError):
                frozen_components()

    def test_both_full_plans_pass_every_guard_without_hardware(self):
        p = self.predictor()
        for side in ('SOURCE', 'TARGET'):
            expected = make(side)
            plan = json.loads((ROOT/'config/paper_plans_r136'/f"{expected['plan_name']}.json").read_text(encoding='utf8'))
            self.assertEqual(plan, expected)
            self.assertEqual(len(plan['items']), 9)
            self.assertEqual([r['action'] for r in plan['items']],
                ['none', 'grid_points', 'grid_points', 'z_path', 'angle_scan', 'circle_scan', 'compound', 'waypoint_xyz', 'none'])
            definition = {'plan_name': plan['plan_name'], 'magnetic_layer_profile_id': plan['paper_target_profile_id'],
                'reference_center_printer_mm': plan['reference_center_printer_mm'],
                'reference_contact_z_mm': plan['reference_contact_z_mm'], 'items': plan['items']}
            self.assertEqual(validate_frozen_source_plan(definition), ui._plans['plans'][plan['plan_name']]['sha256'])
            h = self.host(plan, p)
            for r in plan['items']:
                self.assertIsNone(ui.core.MagTouchApp._validate_full_plan_required_hardware(h, r))
                self.assertTrue(ui.core.MagTouchApp._validate_action_cfg(h, r))
            w = plan['items'][7]
            self.assertEqual(path_sha256(w['waypoints_xyz_mm']), EXACT_GEOMETRY)
            self.assertEqual(len(w['waypoints_xyz_mm']), 29)
            self.assertEqual(w['indent_mm'], 2.4)
            self.assertEqual(w['repeats'], 2)
            self.assertEqual(w['speed_mm_s'], .7)
            self.assertEqual(compile_waypoint_path(w, machine_bounds=ui.core.A5S_MACHINE_BOUNDS_MM)['mixed_xyz_segments'], 28)
            self.assertEqual(plan['items'][5]['circle_radius_mm'], 4.25)
            self.assertIn('0.02,0.10,0.20,0.35', plan['items'][3]['z_depths_mm'])
            changed = copy.deepcopy(definition)
            changed['items'][0]['hold_s'] += .1
            with self.assertRaises(ValueError):
                validate_frozen_source_plan(changed)

    def test_wrong_model_hash_alignment_and_contact_z_fail_before_motion(self):
        plan = make('TARGET')
        p = self.predictor()
        h = self.host(plan, p)
        item = plan['items'][0]
        h.live_predictor = object()
        with self.assertRaisesRegex(ValueError, 'R373'):
            ui.core.MagTouchApp._validate_full_plan_required_hardware(h, item)
        h.live_predictor = p
        altered = {**item, 'required_z_increment_artifact_sha256': 'wrong'}
        with self.assertRaisesRegex(ValueError, 'SHA'):
            ui.core.MagTouchApp._validate_full_plan_required_hardware(h, altered)
        p.engine.xy_translation = np.array([1., 0.])
        with self.assertRaisesRegex(ValueError, '配准'):
            ui.core.MagTouchApp._validate_full_plan_required_hardware(h, item)
        p.engine.xy_translation = np.zeros(2)
        h.maglayer_profiles[h.active_mag_profile_id]['calibration']['contact_z_mm'] += .1
        with self.assertRaisesRegex(ValueError, 'Contact Z'):
            ui.core.MagTouchApp._validate_full_plan_required_hardware(h, item)

    def test_online_engine_matches_frozen_research_predictions(self):
        # Full-field causal replay, not just a forward pass on stored features.
        records = ('r354_source_000004', 'r354_source_000006', 'r355_target_000043')
        emitted = 0
        for ident in records:
            with np.load(ROOT/'analysis/r361_command_forward_20261001/cache'/f'{ident}.npz', allow_pickle=False) as a:
                fields, times, eligible, resets = (a[k].copy() for k in ('b', 't', 'valid', 'reset'))
            with np.load(TRAINING/'replay'/f'{ident}.npz', allow_pickle=False) as a:
                expected, accepted = a['new'].copy(), a['new_ok'].copy()
            e = DirectTrainedEngineR373(self.profile, self.d['threshold'], self.d['min_cosine'], self.d['smoothing_ms'], 70.)
            e.set_baseline(np.zeros(15))
            for i in range(min(600, len(times))):
                if resets[i]:
                    e.set_baseline(np.zeros(15))
                result = e.push(int(round(times[i]*1e9))+10_000_000_000,
                    fields[i] if eligible[i] else np.zeros(15), eligible=bool(eligible[i]))
                self.assertEqual(bool(result['valid']), bool(accepted[i]), (ident, i))
                self.assertEqual(result['family'], FAMILY)
                if accepted[i]:
                    emitted += 1
                    np.testing.assert_allclose(result['xyz'], expected[i], atol=1e-10, rtol=0)
                else:
                    self.assertIsNone(result['xyz'])
        self.assertGreater(emitted, 100)


if __name__ == '__main__':
    unittest.main()
