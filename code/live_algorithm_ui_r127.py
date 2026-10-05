"""R127 UI: manually selected had-239 / old 1 mm field-memory candidate."""
from __future__ import annotations

from tkinter import messagebox, ttk

from live_algorithm_ui_r126 import LiveAlgorithmPanel as _R126LiveAlgorithmPanel
from live_algorithm_ui_r91 import FIT
from live_old1mm_field_memory_r134 import (
    LiveOld1mmFieldMemoryPredictor, R89_CONFIG, R89_KEY,
)
from replay_backtest import JUMP_FAST, load_profile
from training.fit_had239_old1mm_field_memory_r131 import PHYSICAL_ID
from training.had239_old1mm_field_memory_runtime_r132 import DEFAULT_REPORT

PRESET_NAME = "5×5｜had-239＋旧1 mm触头｜磁场记忆连续R127（实验）"


class LiveAlgorithmPanel(_R126LiveAlgorithmPanel):
    def __init__(self, host, parent, neural_box, profile_path):
        super().__init__(host, parent, neural_box, profile_path)
        ttk.Label(
            self,
            text=("R127 是 had-239＋原 1 mm 触头的独立磁场记忆模型，"
                  "与 R126 的 5 mm 参数严格分开。完整磁图 0.2–1.6 mm，"
                  "R89 因果门控训练范围 1.2–2.0 mm；联合已回测范围仅 1.2–1.6 mm。"
                  "1.0 mm 可能大量漏检，2.0 mm 不在完整磁图范围。"
                  "z1.6 网格转点运动回放仍见约 6 mm 瞬时跳变。"
                  "仅手动测试：先确认实物/Contact Z=16.00 mm，重新采集无载 B0，"
                  "先空载观察，再做受控运动。"),
            foreground="#b71c1c", wraplength=1270, justify="left",
        ).grid(row=6, column=0, columnspan=4, sticky="w", padx=4, pady=2)

    def restore(self, config):
        if R89_CONFIG.is_file() and DEFAULT_REPORT.is_file():
            self.available_presets[PRESET_NAME] = R89_KEY
        return super().restore(config)

    def preset_defaults(self):
        if self.vars["preset"].get() != PRESET_NAME:
            return super().preset_defaults()
        previous = self.profile_path
        self.profile_path = R89_CONFIG
        try:
            result = super().preset_defaults()
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
                "R127 实物配置不一致",
                f"此模型仅适用 {PHYSICAL_ID}＋旧 1 mm 触头；"
                f"当前磁层 Profile 为 {active_id or '未选择'}。")
            return None
        if not messagebox.askyesno(
                "载入 R127 旧 1 mm 磁场记忆实验模型？",
                "R127 已用 9 月 24 日同装配八方向记录训练，9 月 26 日圆/十字实测记录"
                "作离线回测；仍不是新的独立实时实验。完整磁图到 1.6 mm，"
                "门控在 1.0 mm 可能漏检，2.0 mm 不可作为本模型有效结果。"
                "z1.6 网格转点运动的离线回放仍见约 6 mm 瞬时跳变；"
                "跨日记录的已知抬离区有少量误输出，尚未通过真实长空载观察。"
                "请确认仍为 had-239＋原 1 mm 触头、Contact Z=16.00 mm，"
                "载入后重新采集无载 B0，先空载观察，再在 1.2–1.6 mm 做受控运动。"
                "是否继续？"):
            return None
        try:
            profile = load_profile(R89_CONFIG, R89_KEY)
            physical = profile["physical_configuration"]
            if (physical["profile_id"] != PHYSICAL_ID or
                    abs(float(physical["contact_z_mm"]) - 16.0) > 1e-6):
                raise ValueError("R127 物理 ID 或训练 Contact Z 不匹配")
            predictor = LiveOld1mmFieldMemoryPredictor(
                profile, R89_CONFIG, PRESET_NAME,
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
                "R127 旧 1 mm 磁场记忆模型已载入：" + predictor.model_summary)
            host.live_predict_begin_btn.configure(text="采集 1.0 s 静止 B0 并开始")
            host.live_predict_baseline_var.set(
                "R127：确认 had-239＋原 1 mm 触头，Contact Z=16.00 mm；"
                "新 B0 后先观察空载误报。")
            return predictor
        except Exception as exc:
            self.invalidate()
            messagebox.showerror("R127 载入失败", str(exc))
            return None
