"""Real Tk variables/widgets regression; withdrawn window, no hardware access."""
import sys
import threading
import tkinter as tk
from tkinter import ttk
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code'))
import magtouch_acquisition_ui_v1_3_4_R373 as ui
from live_algorithm_ui import FIT
from live_retrained_reconstruction_r373 import PRESET_NAME, GATE_NAME


class ConfirmationPollingRegression(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.callback_errors = []
        self.root.report_callback_exception = lambda *error: self.callback_errors.append(error)
        self.polls = []
        self.autosaves = []
        self.host = SimpleNamespace(
            settings={'live_algorithm': {'mode': FIT, 'preset': PRESET_NAME}},
            recorder=None, full_plan_running=False,
            live_predictor=None, live_predict_enabled=False,
            live_predict_lock=threading.Lock(),
            live_predict_profile_confirm_var=tk.BooleanVar(self.root, False),
            after=lambda ms, callback: self.polls.append((ms, callback)),
            _schedule_settings_autosave=lambda: self.autosaves.append(True),
            _live_predict_clear_queues=lambda: None,
            _live_predict_clear_raw_output=lambda: None,
        )
        for name in ('live_predict_motion_var', 'live_predict_state_var',
                     'live_predict_baseline_var', 'live_predict_output_var'):
            setattr(self.host, name, tk.StringVar(self.root, ''))
        self.panel = ui.R373LiveAlgorithmPanel(
            self.host, self.root, ttk.Frame(self.root), ui.registry.R92_PROFILE_PATH)
        self.confirm = ttk.Checkbutton(self.root,
            variable=self.host.live_predict_profile_confirm_var)

    def tearDown(self):
        self.root.update_idletasks()
        self.root.destroy()

    def test_checkbox_persists_during_idle_polls(self):
        self.confirm.invoke()
        self.assertTrue(self.host.live_predict_profile_confirm_var.get())
        for _ in range(20):
            self.panel.poll()
            self.root.update_idletasks()
        self.assertTrue(self.host.live_predict_profile_confirm_var.get())
        self.assertEqual(self.autosaves, [])
        self.assertEqual(self.callback_errors, [])

    def test_loaded_model_and_enabled_state_survive_render_polls(self):
        marker = object()
        self.host.live_predictor = marker
        self.host.live_predict_enabled = True
        self.confirm.invoke()
        self.host.live_predict_baseline_var.set('B0 established')
        for _ in range(20):
            self.panel.poll()
        self.assertIs(self.host.live_predictor, marker)
        self.assertTrue(self.host.live_predict_enabled)
        self.assertTrue(self.host.live_predict_profile_confirm_var.get())
        self.assertEqual(self.host.live_predict_baseline_var.get(), 'B0 established')
        self.assertEqual(self.callback_errors, [])

    def test_repeated_same_value_write_reproduces_reported_failure(self):
        # Exact removed operation: Tk write traces fire even when text is equal.
        self.confirm.invoke()
        self.host.live_predictor = object()
        self.host.live_predict_enabled = True
        self.assertEqual(self.panel.vars['jump_policy'].get(), GATE_NAME)
        self.panel.vars['jump_policy'].set(GATE_NAME)
        self.assertFalse(self.host.live_predict_profile_confirm_var.get())
        self.assertIsNone(self.host.live_predictor)
        self.assertFalse(self.host.live_predict_enabled)

    def test_running_plan_refresh_does_not_touch_confirmation(self):
        marker = object()
        self.host.live_predictor = marker
        self.host.live_predict_enabled = True
        self.confirm.invoke()
        self.host.full_plan_running = True
        for _ in range(20):
            self.panel.poll()
        self.assertTrue(self.host.live_predict_profile_confirm_var.get())
        self.assertTrue(self.host.live_predict_enabled)
        self.assertIs(self.host.live_predictor, marker)
        self.assertEqual(self.callback_errors, [])

    def test_actual_setting_change_still_invalidates_confirmation(self):
        self.confirm.invoke()
        self.host.live_predictor = object()
        self.host.live_predict_enabled = True
        self.panel.vars['custom_path'].set('changed configuration')
        self.assertFalse(self.host.live_predict_profile_confirm_var.get())
        self.assertIsNone(self.host.live_predictor)
        self.assertFalse(self.host.live_predict_enabled)
        self.assertEqual(self.callback_errors, [])

    def test_unconfirmed_operator_still_cannot_begin_baseline(self):
        with patch.object(ui.core.messagebox, 'showwarning') as warning:
            ui.core.MagTouchApp._live_predict_begin_baseline(self.host)
        warning.assert_called_once()
        self.assertFalse(self.host.live_predict_enabled)

    def test_new_preset_is_visible_and_has_independent_locked_gate(self):
        self.assertIn(PRESET_NAME, self.panel.widgets['preset'].cget('values'))
        self.assertEqual(self.panel.vars['jump_policy'].get(), GATE_NAME)
        self.assertEqual(str(self.panel.widgets['jump_policy'].cget('state')), 'disabled')


if __name__ == '__main__':
    unittest.main()
