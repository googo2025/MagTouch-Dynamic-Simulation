"""Experimental R348 preset; no hardware action occurs on import.

Launch explicitly with run_acquisition_ui_R348.bat. Existing R332 frozen-plan
preflight remains active, so an R348 run cannot impersonate R332 acceptance.
"""
from __future__ import annotations

import magtouch_acquisition_ui_v1_3_4_R332 as previous
import live_algorithm_ui_r131 as registry
from live_fast_slow_reconstruction_r348 import (
    GATE_NAME, PRESET_NAME, LiveFastSlowReconstructionPredictorR348)
from replay_backtest import JUMP_FAST
import time

core = previous.core
core.APP_NAME = "MagTouch Hardware Acquisition & Experiment System V1.3.4-R348"
core.BUILD_ID = "20260930-V1.3.4-R348-FAST-SLOW-DEPTH-EXPERIMENTAL"
core.__file__ = __file__
registry.MODEL_SPECS[PRESET_NAME] = registry.CrossModel(
    registry.PHYSICAL_ID, registry.PHYSICAL_ID, registry.R92_PROFILE_PATH,
    registry.PROFILE_KEY, LiveFastSlowReconstructionPredictorR348, 23.70)
registry._EXPERIMENTAL_SPECIAL += (PRESET_NAME,)


class R348TrialRecorder(core.TrialRecorder):
    """Keep both serial-dispatch and ACK times on future R348 recordings."""

    STAGE_FIELDS = core.TrialRecorder.STAGE_FIELDS + [
        "t_send_perf_ns", "t_ack_perf_ns"]


class R348PrinterController(core.PrinterController):
    def send_line(self, cmd: str, *, x=None, y=None, z=None, feed=None,
                  mode="manual", phase="", wait_ok=True, timeout_s=3.0,
                  strict_ack=False):
        # This is the inherited serial transaction with two extra timestamps.
        # Preserve its lock, emergency TX path, ACK check and software position.
        with self.lock:
            response = ""
            if self.dry_run:
                t_send = time.perf_counter_ns()
                self.log(f"Printer DRY> {cmd}")
                response = "DRY_RUN"
                t_ack = time.perf_counter_ns()
            else:
                if self.ser is None or not getattr(self.ser, "is_open", False):
                    raise RuntimeError("Printer serial port is not open")
                with self.tx_lock:
                    t_send = time.perf_counter_ns()
                    self.ser.write((cmd.strip() + "\n").encode("ascii", errors="ignore"))
                    self.ser.flush()
                response = self._read_ok(timeout_s=timeout_s) if wait_ok else "SENT"
                t_ack = time.perf_counter_ns()
                self.log(f"Printer> {cmd} | {response}")
            self.stage_callback({
                "command": cmd, "x_mm": x, "y_mm": y, "z_mm": z,
                "feed_mm_min": feed, "mode": mode, "phase": phase,
                "dry_run": self.dry_run, "response": response,
                "t_send_perf_ns": t_send, "t_ack_perf_ns": t_ack,
            })
            if strict_ack and isinstance(response, str) and response.startswith("TIMEOUT"):
                raise RuntimeError(f"Printer ACK timeout during {phase or mode}: {cmd} | {response}")
            if x is not None:
                self.virtual_position["x"] = float(x)
            if y is not None:
                self.virtual_position["y"] = float(y)
            if z is not None:
                self.virtual_position["z"] = float(z)
            return response


core.TrialRecorder = R348TrialRecorder
core.PrinterController = R348PrinterController


class R348LiveAlgorithmPanel(previous.R332LiveAlgorithmPanel):
    def __init__(self, host, parent, neural_box, profile_path):
        super().__init__(host, parent, neural_box, profile_path)
        core.ttk.Label(self, text=(
            "R348实验模式：源侧冻结的 R332 动作/磁定位许可、R343 共享因果磁历史与 "
            "R348 快慢深度增量；只改已许可 XYZ 的 Z，不改 XY 或许可。"
            "跨器件不强制磁片匹配、不自动配准；无旧阈值门、显示 EMA 或边沿裁剪。"
            "已完成同流离线回测，尚需此版本独立在线验证。"),
            foreground="#784000", wraplength=1270, justify="left"
        ).grid(row=14, column=0, columnspan=4, sticky="w", padx=4, pady=3)

    def restore(self, config):
        self.available_presets[PRESET_NAME] = registry.PROFILE_KEY
        result = super().restore(config)
        if self.vars["preset"].get() == PRESET_NAME:
            self.vars["jump_policy"].set(GATE_NAME)
        elif self.vars["jump_policy"].get() == GATE_NAME:
            self.vars["jump_policy"].set(JUMP_FAST)
        return result

    def preset_defaults(self):
        if self.vars["preset"].get() != PRESET_NAME:
            if self.vars["jump_policy"].get() == GATE_NAME:
                self.vars["jump_policy"].set(JUMP_FAST)
            return super().preset_defaults()
        profile = registry.load_profile(registry.R92_PROFILE_PATH, registry.PROFILE_KEY)
        for key in ("threshold", "min_cosine", "smoothing_ms"):
            self.vars[key].set(str(profile["defaults"][key]))
        self.vars["contact_threshold"].set("70")  # inactive legacy scaffold
        self.vars["jump_policy"].set(GATE_NAME)

    def update_states(self):
        super().update_states()
        if self.vars["preset"].get() == PRESET_NAME:
            self.widgets["jump_policy"].configure(values=(GATE_NAME,))
            for key in ("threshold", "contact_threshold", "min_cosine",
                        "smoothing_ms", "jump_policy"):
                self.widgets[key].configure(state="disabled")

    def load_fit(self):
        if self.vars["preset"].get() != PRESET_NAME:
            return super().load_fit()
        if self.vars["jump_policy"].get() != GATE_NAME:
            core.messagebox.showerror("R348 模式冲突", "R348不能叠加旧门控")
            return None
        predictor = super().load_fit()
        if predictor is not None:
            self.host.live_predict_state_var.set("R348 快慢深度实验模式已载入：" + predictor.model_summary)
            self.host.live_predict_baseline_var.set(
                "确认无载后重采 B0，先观察空载。R332旧验收计划不可换模型冒充同一实测。")
        return predictor


core.LiveAlgorithmPanel = R348LiveAlgorithmPanel
_previous_label = core.MagTouchApp._live_filter_label


def _r348_label(host):
    if isinstance(getattr(host, "live_predictor", None),
                  LiveFastSlowReconstructionPredictorR348):
        return "R348原始有效XYZ（快慢增量Z；无显示滤波/裁剪）"
    return _previous_label(host)


core.MagTouchApp._live_filter_label = _r348_label


def main():
    return previous.main()


if __name__ == "__main__":
    main()
