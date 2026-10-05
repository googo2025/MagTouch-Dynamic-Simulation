"""R130 opt-in had-239/5 mm circle-recovery candidates; old presets unchanged."""
from __future__ import annotations

from tkinter import messagebox, ttk

from live_algorithm_ui_r129 import LiveAlgorithmPanel as _R129Panel
from live_algorithm_ui_r126 import PHYSICAL_ID, PROFILE_KEY, R92_PROFILE_PATH
from live_algorithm_ui_r91 import FIT
from live_field_memory_r130 import (
    LiveR130GuardPredictor, LiveR130NeuralPredictor,
)
from replay_backtest import JUMP_FAST, load_profile
from training.had239_branch_memory_runtime_r144 import AUDIT_REPORT
from training.had239_causal_xy_mlp_runtime_r165 import ARTIFACT, METADATA
from training.had239_field_memory_depth_guard_runtime_r126 import R125_REPORT


GUARD_NAME = "5×5｜had-239＋5 mm｜磁证据分支保护R130（实验）"
NEURAL_NAME = "5×5｜had-239＋5 mm｜分支保护＋因果XY网络R130（实验）"


class LiveAlgorithmPanel(_R129Panel):
    def __init__(self, host, parent, neural_box, profile_path):
        super().__init__(host, parent, neural_box, profile_path)
        ttk.Label(
            self,
            text=("R130 两个 had-239＋5 mm 候选：第一项仅用磁场证据阻止远距离错误分支；"
                  "第二项在其后加真实轨迹训练的 0.1/0.4 s 因果 XY 网络，Z/接触门控不变。"
                  "9/26 旧记录为开发与整 Trial 留出回测，不是新独立实测；"
                  "圆形仍有畸变。仅对该磁片和该触头使用，重新采无载 B0，先观察空载。"
                  "R128、R129 Yellow 和旧 1 mm 模型均保留。"),
            foreground="#b71c1c", wraplength=1270, justify="left",
        ).grid(row=9, column=0, columnspan=4, sticky="w", padx=4, pady=2)

    def restore(self, config):
        common = (R92_PROFILE_PATH.is_file() and R125_REPORT.is_file() and
                  AUDIT_REPORT.is_file())
        if common:
            self.available_presets[GUARD_NAME] = PROFILE_KEY
            if ARTIFACT.is_file() and METADATA.is_file():
                self.available_presets[NEURAL_NAME] = PROFILE_KEY
        result = super().restore(config)
        if self.vars["preset"].get() in (GUARD_NAME, NEURAL_NAME):
            self.vars["contact_threshold"].set("70")
            self.vars["jump_policy"].set(JUMP_FAST)
        return result

    def preset_defaults(self):
        if self.vars["preset"].get() not in (GUARD_NAME, NEURAL_NAME):
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
        if self.vars["preset"].get() in (GUARD_NAME, NEURAL_NAME):
            for key in ("threshold", "contact_threshold", "min_cosine",
                        "smoothing_ms", "jump_policy"):
                self.widgets[key].configure(state="disabled")

    def load_fit(self):
        selected = self.vars["preset"].get()
        if selected not in (GUARD_NAME, NEURAL_NAME):
            return super().load_fit()
        if self.busy() or self.vars["mode"].get() != FIT:
            return None
        host = self.host
        active_id = str(getattr(host, "active_mag_profile_id", "") or "")
        if active_id != PHYSICAL_ID:
            messagebox.showerror(
                "R130 实物配置不一致",
                f"此选项只适用 {PHYSICAL_ID}＋最终 5 mm 触头；"
                f"当前磁层 Profile 为 {active_id or '未选择'}。")
            return None
        is_neural = selected == NEURAL_NAME
        current_contact_z = (getattr(host, "maglayer_profiles", {})
                             .get(PHYSICAL_ID, {}).get("calibration", {})
                             .get("contact_z_mm"))
        contact_note = (f"当前 Profile Contact Z={current_contact_z} mm。"
                        if current_contact_z is not None else
                        "当前 Profile Contact Z 未读取到，请人工核对。")
        if not messagebox.askyesno(
                "载入 R130 实验模型？",
                ("R130 网络项用 9/26 had-239＋5 mm 的整 Trial 数据训练/验证，"
                 "测试 Trial 32 圆形 XY 约 1.06 mm、四次畸变约 0.48 mm；"
                 if is_neural else
                 "R130 保护项只在局部磁场误差超过全局解四倍时禁止锁错；"
                 "旧记录 Trial 32 圆形 XY 从 3.35 降到约 1.37 mm；") +
                "两项都未通过新的实时独立验证，圆形仍不完美。"
                "确认是 had-239＋最终 5 mm 触头；静态磁图训练 Contact Z=23.70 mm，"
                "网络轨迹训练会话 Contact Z=23.360001 mm。" + contact_note +
                "重新采集无载 B0，先观察空载，再点动、八方向和圆形。"
                "本操作不会自动启动试验台。是否继续？"):
            return None
        try:
            profile = load_profile(R92_PROFILE_PATH, PROFILE_KEY)
            if (profile.get("physical_profile_id") != PHYSICAL_ID or
                    abs(float(profile.get("training_contact_z_mm")) - 23.70) > 1e-6):
                raise ValueError("R130 物理 ID 或训练 Contact Z 不匹配")
            predictor_cls = (LiveR130NeuralPredictor if is_neural else
                             LiveR130GuardPredictor)
            predictor = predictor_cls(
                profile, R92_PROFILE_PATH, selected,
                **{key: self.vars[key].get() for key in (
                    "threshold", "contact_threshold", "min_cosine",
                    "smoothing_ms", "jump_policy")})
            predictor.metadata["runtime_profile_contact_z_mm"] = current_contact_z
            if hasattr(host, "_live_alignment_apply_to_predictor"):
                host._live_alignment_apply_to_predictor(predictor, selected)
            self.invalidate()
            with host.live_predict_lock:
                host.live_predictor = predictor
            for view in (host.live_predict_plot, host.live_predict_3d,
                         host.exp_predict_3d):
                view.set_scale(predictor.visual_scale)
            host.live_predict_state_var.set("R130 实验模型已载入：" + predictor.model_summary)
            host.live_predict_begin_btn.configure(text="采集 1.0 s 静止 B0 并开始")
            host.live_predict_baseline_var.set(
                "R130：had-239＋最终 5 mm 触头；新 B0 后先观察空载，"
                "新轨迹再作独立评价。")
            return predictor
        except Exception as exc:
            self.invalidate()
            messagebox.showerror("R130 载入失败", str(exc))
            return None
