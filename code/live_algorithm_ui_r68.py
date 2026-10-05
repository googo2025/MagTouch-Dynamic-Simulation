"""R68 live selector: quarantine unvalidated old-1-mm-tip R63/R64/R65 gates."""

import json
from pathlib import Path
from tkinter import messagebox

from live_algorithm_ui_r62 import FIT, LiveAlgorithmPanel as _R62LiveAlgorithmPanel
from replay_backtest import CUSTOM


REVOKED_OLD1MM_PROFILE_ID = "magnetic_layer_v1_5*5-had-old1mmTouch"
REVOKED_R65_PRESET_NAME = "5×5｜had-239旧1 mm触头｜静态连续R65（试验）"


def unvalidated_old1mm_profile(path):
    """Detect bundled old-1-mm-tip profiles even when loaded as custom JSON."""
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    key = payload.get("default_profile")
    profile = (payload.get("profiles") or {}).get(key, payload)
    physical = profile.get("physical_configuration") or {}
    return (physical.get("profile_id") == REVOKED_OLD1MM_PROFILE_ID
            and not bool((profile.get("deployment_validation") or {}).get("passed")))


class LiveAlgorithmPanel(_R62LiveAlgorithmPanel):
    def restore(self, config):
        # A saved R67 selection must not silently fall back to another magnet's
        # preset, nor remain a loadable R65 setting after an application update.
        if config.get("preset") == REVOKED_R65_PRESET_NAME:
            config = dict(config)
            config.update(mode=FIT, preset=CUSTOM, custom_path="")
        return super().restore(config)

    def load_fit(self):
        if self.vars["preset"].get() == CUSTOM:
            candidate = self.vars["custom_path"].get().strip()
            if candidate and Path(candidate).is_file():
                try:
                    revoked = unvalidated_old1mm_profile(candidate)
                except (OSError, ValueError, TypeError, AttributeError):
                    revoked = False  # The ordinary loader reports malformed JSON.
                if revoked:
                    messagebox.showerror(
                        "旧1 mm触头模型已撤下",
                        "R63/R64/R65 的接触门控未通过整条空载回放，不能作为实时模型载入。"
                        "仅在装配一致且已有独立空载/接触证据时使用其他模型；"
                        "旧1 mm触头请等待新的完整验证。",
                    )
                    return None
        return super().load_fit()
