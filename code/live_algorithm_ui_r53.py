"""R53 display labels for the two R50 continuous static regressors.

The saved preset names remain unchanged so existing model-specific five-point
alignments and settings continue to apply to the same physical calibration.
"""

import tkinter as tk

from live_algorithm_ui import CUSTOM, FIT, LiveAlgorithmPanel as _LiveAlgorithmPanel


PRESET_DISPLAY_LABELS = {
    "5×5｜had-239最终触头｜静态全量R50":
        "5×5｜had-239最终触头｜静态连续回归全量R50",
    "5×5｜had-Yelow｜静态全量R50":
        "5×5｜had-Yelow｜静态连续回归全量R50",
}
PRESET_CANONICAL_LABELS = {display: canonical for canonical, display in PRESET_DISPLAY_LABELS.items()}


def display_preset_name(canonical: str) -> str:
    return PRESET_DISPLAY_LABELS.get(canonical, canonical)


def canonical_preset_name(display: str) -> str:
    return PRESET_CANONICAL_LABELS.get(display, display)


class LiveAlgorithmPanel(_LiveAlgorithmPanel):
    """Show descriptive labels without changing inference or calibration keys."""

    def __init__(self, host, parent, neural_box, profile_path):
        super().__init__(host, parent, neural_box, profile_path)
        canonical_var = self.vars["preset"]
        self.preset_display_var = tk.StringVar(
            self, value=display_preset_name(canonical_var.get()))
        self.widgets["preset"].configure(
            textvariable=self.preset_display_var,
            values=[display_preset_name(name) for name in self.available_presets] +
                   [CUSTOM],
            width=48,
        )
        self.preset_display_var.trace_add("write", self._display_preset_changed)
        canonical_var.trace_add("write", self._canonical_preset_changed)

    def _display_preset_changed(self, *_):
        selected = canonical_preset_name(self.preset_display_var.get())
        if self.vars["preset"].get() != selected:
            self.vars["preset"].set(selected)

    def _canonical_preset_changed(self, *_):
        displayed = display_preset_name(self.vars["preset"].get())
        if self.preset_display_var.get() != displayed:
            self.preset_display_var.set(displayed)
