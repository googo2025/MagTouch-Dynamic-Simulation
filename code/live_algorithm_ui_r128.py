"""R128 optional had-239/5 mm branch-aware model; preserves R127 old1mm."""
from __future__ import annotations

from tkinter import messagebox, ttk

from live_algorithm_ui_r127 import LiveAlgorithmPanel as _R127Panel
from live_algorithm_ui_r126 import (
    PHYSICAL_ID, PROFILE_KEY, R92_PROFILE_PATH,
)
from live_algorithm_ui_r91 import FIT
from live_field_memory_branch_r145 import LiveR128Predictor
from replay_backtest import JUMP_FAST, load_profile
from training.had239_branch_memory_runtime_r144 import AUDIT_REPORT
from training.had239_field_memory_depth_guard_runtime_r126 import R125_REPORT

PRESET_NAME = "5×5｜had-239＋5 mm触头｜快位置慢磁记忆R128（实验）"


class LiveAlgorithmPanel(_R127Panel):
    def __init__(self, host, parent, neural_box, profile_path):
        super().__init__(host, parent, neural_box, profile_path)
        ttk.Label(
            self,
            text=("R128：had-239＋5 mm 触头。连续磁图与 R126 相同，新增远距离"
                  "反演分支判别；材料记忆仍是独立慢状态。9/26 已有实测全帧率"
                  "回放显示八方向跳变减少、圆形时延基本不变；这些记录已用于开发，"
                  "尚非新的盲测。R127 旧 1 mm 模型仍单独保留。两种触头不可混用参数。"),
            foreground="#b71c1c", wraplength=1270, justify="left",
        ).grid(row=7, column=0, columnspan=4, sticky="w", padx=4, pady=2)

    def restore(self, config):
        if (R92_PROFILE_PATH.is_file() and R125_REPORT.is_file() and
                AUDIT_REPORT.is_file()):
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
            for key in ("threshold", "contact_threshold", "min_cosine",
                        "smoothing_ms", "jump_policy"):
                self.widgets[key].configure(state="disabled")

    def load_fit(self):
        if self.vars["preset"].get() != PRESET_NAME:
            return super().load_fit()
        if self.busy() or self.vars["mode"].get() != FIT:
            return None
        host = self.host
        active_id = str(getattr(host, "active_mag_profile_id", "") or "")
        if active_id != PHYSICAL_ID:
            messagebox.showerror(
                "R128 实物配置不一致",
                f"此选项只适用 {PHYSICAL_ID}＋5 mm 触头；"
                f"当前磁层 Profile 为 {active_id or '未选择'}。")
            return None
        if not messagebox.askyesno(
                "载入 R128 快位置／慢磁记忆实验模型？",
                "R128 与 R126 使用同一静态磁图及材料记忆参数，只在全局反演"
                "发生远距离跳支时作因果分支选择，不拟圆或平滑 XYZ。"
                "已有记录回放改善，但尚未完成新实测和独立长空载误报验证。"
                "确认实物是 had-239＋最终 5 mm 触头、Contact Z 对齐；"
                "请重新采集无载 B0，先空载观察，再做受控运动。是否继续？"):
            return None
        try:
            profile = load_profile(R92_PROFILE_PATH, PROFILE_KEY)
            if (profile.get("physical_profile_id") != PHYSICAL_ID or
                    abs(float(profile.get("training_contact_z_mm")) - 23.70) > 1e-6):
                raise ValueError("R128 物理 ID 或训练 Contact Z 不匹配")
            predictor = LiveR128Predictor(
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
                "R128 快位置／慢磁记忆模型已载入：" + predictor.model_summary)
            host.live_predict_begin_btn.configure(text="采集 1.0 s 静止 B0 并开始")
            host.live_predict_baseline_var.set(
                "R128：had-239＋5 mm 触头；新 B0 后先观察空载误报。")
            return predictor
        except Exception as exc:
            self.invalidate()
            messagebox.showerror("R128 载入失败", str(exc))
            return None
