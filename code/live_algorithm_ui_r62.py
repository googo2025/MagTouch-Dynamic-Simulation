"""R62 adds the independent had-239 R61 runtime calibration to the UI.

The model has its own parameter file. Its canonical selector name is distinct
from R50, so saved five-point XY alignments cannot silently cross models.
"""

from pathlib import Path

from live_algorithm_ui_r53 import FIT, LiveAlgorithmPanel as _R53LiveAlgorithmPanel
from replay_backtest import JUMP_HYSTERESIS


R61_PRESET_NAME = "5×5｜had-239最终触头｜静态＋因果动态连续R61"
R61_PROFILE_KEY = "5x5_had239_core_continuous_r61"
R61_PROFILE_PATH = (
    Path(__file__).resolve().parent.parent
    / "config"
    / "had239_core_continuous_r61_runtime_20260924.json"
)


class LiveAlgorithmPanel(_R53LiveAlgorithmPanel):
    """Route only the R61 selector to its own immutable runtime JSON."""

    def restore(self, config):
        # The R40 base constructor calls restore after building its selector
        # from the R50 library and before the R53 display labels are installed.
        if R61_PROFILE_PATH.is_file():
            self.available_presets[R61_PRESET_NAME] = R61_PROFILE_KEY
        super().restore(config)

    def preset_defaults(self):
        if self.vars["preset"].get() != R61_PRESET_NAME:
            return super().preset_defaults()
        previous_path = self.profile_path
        self.profile_path = R61_PROFILE_PATH
        try:
            result = super().preset_defaults()
            # R61 holdout/runtime evidence depends on the 200 ms release-peak
            # window, which is active only under the hysteresis jump policy.
            self.vars["jump_policy"].set(JUMP_HYSTERESIS)
            return result
        finally:
            self.profile_path = previous_path

    def update_states(self):
        super().update_states()
        if self.vars["preset"].get() == R61_PRESET_NAME:
            self.widgets["jump_policy"].configure(state="disabled")

    def load_fit(self):
        if self.vars["preset"].get() != R61_PRESET_NAME:
            return super().load_fit()
        previous_path = self.profile_path
        self.profile_path = R61_PROFILE_PATH
        try:
            return super().load_fit()
        finally:
            self.profile_path = previous_path
