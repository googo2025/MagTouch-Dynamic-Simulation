"""Manually selected had-239/5 mm field-memory experiment for R126 UI."""
from __future__ import annotations

from tkinter import messagebox, ttk

from live_algorithm_ui_r92 import (
    FIT, R92_PROFILE_PATH, LiveAlgorithmPanel as _R92LiveAlgorithmPanel,
)
from live_field_memory_r126 import LiveR126Predictor
from replay_backtest import JUMP_FAST, load_profile
from training.had239_field_memory_depth_guard_runtime_r126 import R125_REPORT
from training.had239_field_memory_runtime_r123 import DEFAULT_REPORT


PRESET_NAME = "5×5｜had-239＋5 mm触头｜磁场记忆连续R126（实验）"
PROFILE_KEY = "5x5_had239_final5mm_old_formal_r92"
PHYSICAL_ID = "magnetic_layer_v1_5*5-had-239"


class LiveAlgorithmPanel(_R92LiveAlgorithmPanel):
    def __init__(self, host, parent, neural_box, profile_path):
        super().__init__(host, parent, neural_box, profile_path)
        ttk.Label(
            self,
            text=("R126 是 had-239＋最终 5 mm 触头的离线磁场记忆候选："
                  "运动回放已改善，但静态有小幅退化；使用快速 70 µT 临时接触门控，"
                  "尚无独立长空载误报验证。仅手动选择，确认实物后重新采集无载 B0，"
                  "先空载观察，再受控运动。旧 R92 模型仍可随时切回。"),
            foreground="#b71c1c", wraplength=1270, justify="left",
        ).grid(row=5, column=0, columnspan=4, sticky="w", padx=4, pady=2)

    def restore(self, config):
        if (R92_PROFILE_PATH.is_file() and DEFAULT_REPORT.is_file() and
                R125_REPORT.is_file()):
            self.available_presets[PRESET_NAME] = PROFILE_KEY
        result = super().restore(config)
        if self.vars["preset"].get() == PRESET_NAME:
            self.vars["contact_threshold"].set("70")
            self.vars["jump_policy"].set(JUMP_FAST)
        return result

    def preset_defaults(self):
        if self.vars["preset"].get() != PRESET_NAME:
            return super().preset_defaults()
        previous = self.profile_path
        self.profile_path = R92_PROFILE_PATH
        try:
            result = super().preset_defaults()
            self.vars["contact_threshold"].set("70")
            self.vars["jump_policy"].set(JUMP_FAST)
            return result
        finally:
            self.profile_path = previous

    def update_states(self):
        super().update_states()
        if self.vars["preset"].get() == PRESET_NAME:
            self.widgets["jump_policy"].configure(state="disabled")
            self.widgets["contact_threshold"].configure(state="disabled")

    def load_fit(self):
        if self.vars["preset"].get() != PRESET_NAME:
            return super().load_fit()
        if self.busy() or self.vars["mode"].get() != FIT:
            return None
        host = self.host
        active_id = str(getattr(host, "active_mag_profile_id", "") or "")
        if active_id != PHYSICAL_ID:
            messagebox.showerror(
                "R126 实物配置不一致",
                f"此模型只适用 {PHYSICAL_ID}＋最终 5 mm 触头；"
                f"当前磁层 Profile 为 {active_id or '未选择'}。")
            return None
        if not messagebox.askyesno(
                "载入尚未完成真实空载验证的 R126？",
                "R126 已在整 Trial 留出与后采全帧率记录中离线改善四条运动，"
                "但后采记录不是新盲测，静态 XY 小幅变差，真实长空载误报和实物推理尚未通过。"
                "UI 采用验证集选择的快速 70 µT 默认门控，不代表真实空载已验证。"
                "快速策略没有旧版释放锁定和自动释放后 B0；确认当前确为 had-239 固定硅胶磁片＋"
                "最终 5 mm 触头、五传感器顺序与 Contact Z，并准备重新采集无载 B0。"
                "请先空载观察，若出现误报立即停用本选项。是否继续？"):
            return None
        try:
            profile = load_profile(R92_PROFILE_PATH, PROFILE_KEY)
            if (profile.get("physical_profile_id") != PHYSICAL_ID or
                    abs(float(profile.get("training_contact_z_mm")) - 23.70)
                    > 1e-6):
                raise ValueError("R126 gate profile physical ID/Contact Z mismatch")
            predictor = LiveR126Predictor(
                profile, R92_PROFILE_PATH, PRESET_NAME,
                **{key: self.vars[key].get() for key in (
                    "threshold", "contact_threshold", "min_cosine",
                    "smoothing_ms", "jump_policy")},
            )
            if hasattr(host, "_live_alignment_apply_to_predictor"):
                host._live_alignment_apply_to_predictor(predictor, PRESET_NAME)
            self.invalidate()
            with host.live_predict_lock:
                host.live_predictor = predictor
            for view in (host.live_predict_plot, host.live_predict_3d,
                         host.exp_predict_3d):
                view.set_scale(predictor.visual_scale)
            host.live_predict_state_var.set(
                "R126 磁场记忆实验模型已载入：" + predictor.model_summary)
            host.live_predict_begin_btn.configure(text="采集 1.0 s 静止 B0 并开始")
            host.live_predict_baseline_var.set(
                "R126：确认 had-239＋最终 5 mm 触头；新 B0 后先观察长空载误报。")
            return predictor
        except Exception as exc:
            self.invalidate()
            messagebox.showerror("R126 载入失败", str(exc))
            return None
