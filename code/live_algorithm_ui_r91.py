"""R91 adds a guarded R89 experimental selector without restoring R65."""
from __future__ import annotations

from pathlib import Path
from tkinter import messagebox, ttk

from live_algorithm_ui_r68 import FIT, LiveAlgorithmPanel as _R68LiveAlgorithmPanel
from live_old1mm_r89 import LiveR89Predictor
from replay_backtest import JUMP_FAST, load_profile


R89_PRESET_NAME = "5×5｜had-239旧1 mm触头｜因果XYZ R89（实验）"
R89_PROFILE_KEY = "5x5_had239_old1mm_grid_safe_causal_candidate_r89"
R89_MAGNETIC_PROFILE_ID = "magnetic_layer_v1_5*5-had-old1mmTouch"
R89_PROFILE_PATH = (
    Path(__file__).resolve().parent.parent / "config" /
    "had239_old1mm_grid_safe_causal_candidate_r89_20260925.json"
)


class LiveAlgorithmPanel(_R68LiveAlgorithmPanel):
    def __init__(self, host, parent, neural_box, profile_path):
        super().__init__(host, parent, neural_box, profile_path)
        ttk.Label(
            self,
            text=("R89 旧1 mm触头仅供新会话受控实测：已评估压深1.2–2.0 mm；"
                  "缺少独立长空载、圆周和 XYZ 联合运动验证。"
                  "B0 完成后先保持无载至少 0.5 s 以建立因果历史。"
                  "R65 仍已撤下，R89 不会自动载入或自动开始预测。"),
            foreground="#b71c1c", wraplength=1270, justify="left",
        ).grid(row=3, column=0, columnspan=4, sticky="w", padx=4, pady=2)

    def restore(self, config):
        if R89_PROFILE_PATH.is_file():
            self.available_presets[R89_PRESET_NAME] = R89_PROFILE_KEY
        return super().restore(config)

    def preset_defaults(self):
        if self.vars["preset"].get() != R89_PRESET_NAME:
            return super().preset_defaults()
        previous = self.profile_path
        self.profile_path = R89_PROFILE_PATH
        try:
            result = super().preset_defaults()
            self.vars["jump_policy"].set(JUMP_FAST)
            return result
        finally:
            self.profile_path = previous

    def update_states(self):
        super().update_states()
        if self.vars["preset"].get() == R89_PRESET_NAME:
            for key in ("threshold", "contact_threshold", "min_cosine",
                        "smoothing_ms", "jump_policy"):
                self.widgets[key].configure(state="disabled")

    def load_fit(self):
        if self.vars["preset"].get() != R89_PRESET_NAME:
            return super().load_fit()
        if self.busy() or self.vars["mode"].get() != FIT:
            return None
        active = str(getattr(self.host, "active_mag_profile_id", "") or "")
        if active != R89_MAGNETIC_PROFILE_ID:
            messagebox.showerror(
                "R89 物理配置不一致",
                f"R89 仅用于 {R89_MAGNETIC_PROFILE_ID}；当前为 {active or '未选择'}。"
                "请确认仍为 had-239 固定硅胶磁片＋原 1 mm 触头。",
            )
            return None
        if not messagebox.askyesno(
                "载入未独立验证的 R89？",
                "R89 是实验候选，仅有开发回看；尚未通过新会话长空载、圆周和 XYZ 运动测试。"
                "仅评估过 1.2–2.0 mm，范围外输出不保证可靠，浅压可能漏检。载入后仍须单独确认装配、"
                "采集无载 B0，再先做空载观察。是否继续载入？"):
            return None
        h = self.host
        try:
            profile = load_profile(R89_PROFILE_PATH, R89_PROFILE_KEY)
            predictor = LiveR89Predictor(
                profile, R89_PROFILE_PATH, R89_PRESET_NAME, key=R89_PROFILE_KEY,
                **{key: self.vars[key].get() for key in (
                    "threshold", "contact_threshold", "min_cosine",
                    "smoothing_ms", "jump_policy")},
            )
            if hasattr(h, "_live_alignment_apply_to_predictor"):
                h._live_alignment_apply_to_predictor(predictor, R89_PRESET_NAME)
            self.invalidate()
            with h.live_predict_lock:
                h.live_predictor = predictor
            for view in (h.live_predict_plot, h.live_predict_3d, h.exp_predict_3d):
                view.set_scale(predictor.visual_scale)
            h.live_predict_state_var.set("R89 实验候选已载入：" + predictor.model_summary)
            h.live_predict_begin_btn.configure(text="采集 1.0 s 静止 B0 并开始")
            h.live_predict_baseline_var.set(
                "R89：确认磁片/旧1 mm触头/传感器顺序；无载采集 B0，完成后先空载≥0.5 s。")
            return predictor
        except Exception as exc:
            self.invalidate()
            messagebox.showerror("R89 载入失败", str(exc))
            return None
