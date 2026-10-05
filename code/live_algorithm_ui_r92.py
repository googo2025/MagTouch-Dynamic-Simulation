"""R92: expose the two faithful old-121-point-formula had-239 refits.

These are static continuous XY/Z regressors. They have no learned dynamic
residual and are explicit, physical-assembly-guarded test choices only.
"""
from __future__ import annotations

from pathlib import Path
from tkinter import messagebox, ttk

from live_algorithm_ui_r91 import FIT, LiveAlgorithmPanel as _R91LiveAlgorithmPanel
from live_normalized_fit import LiveNormalizedFitPredictor
from replay_backtest import JUMP_HYSTERESIS, load_profile


R92_PROFILE_PATH = (
    Path(__file__).resolve().parent.parent / "config" /
    "had239_legacy_formal_continuous_r92_20260926.json"
)
R92_FINAL5MM_PRESET_NAME = "5×5｜had-239＋5 mm触头｜旧121点正式连续R92（待实测）"
R92_OLD1MM_PRESET_NAME = "5×5｜had-239＋旧1 mm触头｜旧121点正式连续R92（待实测）"
R92_PRESETS = {
    R92_FINAL5MM_PRESET_NAME: {
        "key": "5x5_had239_final5mm_old_formal_r92",
        "physical_profile_id": "magnetic_layer_v1_5*5-had-239",
        "contact_z_mm": 23.70,
        "depth_range_mm": "0.4–3.0",
    },
    R92_OLD1MM_PRESET_NAME: {
        "key": "5x5_had239_old1mm_old_formal_r92",
        "physical_profile_id": "magnetic_layer_v1_5*5-had-old1mmTouch",
        "contact_z_mm": 16.00,
        "depth_range_mm": "0.5–2.0（部分压深仅33点）",
    },
}


class LiveAlgorithmPanel(_R91LiveAlgorithmPanel):
    def __init__(self, host, parent, neural_box, profile_path):
        super().__init__(host, parent, neural_box, profile_path)
        ttk.Label(
            self,
            text=("R92 两项是旧 5×5『121点正式连续』公式在 had-239 两种触头上分别重训；"
                  "没有动态残差。两者必须分别对应实际触头与磁层 Profile，"
                  "接触门控尚未通过独立长空载实测；先观察空载，再测试受控运动。"),
            foreground="#b71c1c", wraplength=1270, justify="left",
        ).grid(row=4, column=0, columnspan=4, sticky="w", padx=4, pady=2)

    def restore(self, config):
        if R92_PROFILE_PATH.is_file():
            for label, spec in R92_PRESETS.items():
                self.available_presets[label] = spec["key"]
        return super().restore(config)

    def preset_defaults(self):
        if self.vars["preset"].get() not in R92_PRESETS:
            return super().preset_defaults()
        previous = self.profile_path
        self.profile_path = R92_PROFILE_PATH
        try:
            result = super().preset_defaults()
            self.vars["jump_policy"].set(JUMP_HYSTERESIS)
            return result
        finally:
            self.profile_path = previous

    def update_states(self):
        super().update_states()
        if self.vars["preset"].get() in R92_PRESETS:
            self.widgets["jump_policy"].configure(state="disabled")

    def load_fit(self):
        preset = self.vars["preset"].get()
        if preset not in R92_PRESETS:
            return super().load_fit()
        if self.busy() or self.vars["mode"].get() != FIT:
            return None

        spec = R92_PRESETS[preset]
        host = self.host
        active_id = str(getattr(host, "active_mag_profile_id", "") or "")
        if active_id != spec["physical_profile_id"]:
            messagebox.showerror(
                "R92 实物配置不一致",
                f"此模型只允许 {spec['physical_profile_id']}；当前为 {active_id or '未选择'}。"
                "请先核对实际磁片、触头和磁层 Profile。",
            )
            return None

        try:
            profile = load_profile(R92_PROFILE_PATH, spec["key"])
            if (profile.get("physical_profile_id") != active_id or
                    profile.get("fit_variant") !=
                    "legacy_5x5_121pt_formal_exact_feature_and_hyperparameter_refit_r92" or
                    abs(float(profile.get("training_contact_z_mm")) -
                        spec["contact_z_mm"]) > 1e-6 or
                    profile.get("dynamic_model") is not None):
                raise ValueError("R92 参数文件的实物/算法/Contact Z 与 UI 注册项不一致")
            if not messagebox.askyesno(
                    "载入 R92 旧正式连续重训模型？",
                    f"当前模型仅用于 {active_id}；训练 Contact Z={spec['contact_z_mm']:.2f} mm，"
                    f"训练压深 {spec['depth_range_mm']} mm。"
                    "它是静态连续回归，没有动态残差；接触门限仅基于训练接触和合成零响应，"
                    "尚未通过独立长空载实测。请确认触头、接触平面、五传感器顺序和 MLX 配置，"
                    "载入后重新采集无载 B0，先观察空载，再做受控运动。是否载入？"):
                return None
            predictor = LiveNormalizedFitPredictor(
                profile, R92_PROFILE_PATH, preset,
                **{key: self.vars[key].get() for key in (
                    "threshold", "contact_threshold", "min_cosine",
                    "smoothing_ms", "jump_policy")},
            )
            predictor.metadata.update(
                physical_profile_id=active_id,
                training_contact_z_mm=spec["contact_z_mm"],
                deployment_qualified=False,
                validation_status="offline-only; live empty and dynamic tests pending",
            )
            if hasattr(host, "_live_alignment_apply_to_predictor"):
                host._live_alignment_apply_to_predictor(predictor, preset)
            self.invalidate()
            with host.live_predict_lock:
                host.live_predictor = predictor
            for view in (host.live_predict_plot, host.live_predict_3d,
                         host.exp_predict_3d):
                view.set_scale(predictor.visual_scale)
            host.live_predict_state_var.set("R92 旧正式连续重训模型已载入：" +
                                            predictor.model_summary)
            host.live_predict_begin_btn.configure(text="采集 1.0 s 静止 B0 并开始")
            host.live_predict_baseline_var.set(
                "R92：确认实物/Contact Z/传感器配置一致；无载采集新 B0，先观察空载误报。")
            return predictor
        except Exception as exc:
            self.invalidate()
            messagebox.showerror("R92 载入失败", str(exc))
            return None
