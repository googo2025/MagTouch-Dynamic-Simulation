"""R129: explicit zero-shot had-239 R128 -> had-Yelow transfer test.

The existing R128 and R127 choices are unchanged. This selector deliberately
uses the frozen had-239/5 mm coefficients on the distinct Yellow magnet, with
no output alignment or retraining. It does not touch acquisition hardware.
"""
from __future__ import annotations

import math
from tkinter import messagebox, ttk

from live_algorithm_ui_r128 import LiveAlgorithmPanel as _R128Panel
from live_algorithm_ui_r126 import PHYSICAL_ID, PROFILE_KEY, R92_PROFILE_PATH
from live_algorithm_ui_r91 import FIT
from live_field_memory_branch_r145 import LiveR128Predictor
from replay_backtest import JUMP_FAST, load_profile
from training.had239_branch_memory_runtime_r144 import AUDIT_REPORT
from training.had239_field_memory_depth_guard_runtime_r126 import R125_REPORT


YELLOW_ID = "magnetic_layer_v1_5*5-had-Yelow"
TRANSFER_NAME = (
    "5×5｜had-239 R128参数→had-Yelow｜零样本跨磁片R129（实验）"
)


class LiveAlgorithmPanel(_R128Panel):
    def __init__(self, host, parent, neural_box, profile_path):
        super().__init__(host, parent, neural_box, profile_path)
        ttk.Label(
            self,
            text=("R129 跨磁片选项：原样使用 had-239＋5 mm 的 R128 参数，在另一块 "
                  "had-Yelow 磁片上做零样本泛化测试；不重训、不自动借用 had-239 "
                  "五点校准、不补偿 Z。已知 R128 在最新 r6 圆形实测出现严重局部"
                  "分支锁定，不能视为 Yellow 已验证模型。仅选 Yellow Profile "
                  "并确认同为 5 mm 触头后手动载入。"),
            foreground="#b71c1c", wraplength=1270, justify="left",
        ).grid(row=8, column=0, columnspan=4, sticky="w", padx=4, pady=2)

    def restore(self, config):
        if (R92_PROFILE_PATH.is_file() and R125_REPORT.is_file() and
                AUDIT_REPORT.is_file()):
            self.available_presets[TRANSFER_NAME] = PROFILE_KEY
        result = super().restore(config)
        if self.vars["preset"].get() == TRANSFER_NAME:
            self.vars["contact_threshold"].set("70")
            self.vars["jump_policy"].set(JUMP_FAST)
        return result

    def preset_defaults(self):
        if self.vars["preset"].get() != TRANSFER_NAME:
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
        if self.vars["preset"].get() == TRANSFER_NAME:
            for key in ("threshold", "contact_threshold", "min_cosine",
                        "smoothing_ms", "jump_policy"):
                self.widgets[key].configure(state="disabled")

    def load_fit(self):
        if self.vars["preset"].get() != TRANSFER_NAME:
            return super().load_fit()
        if self.busy() or self.vars["mode"].get() != FIT:
            return None
        host = self.host
        active_id = str(getattr(host, "active_mag_profile_id", "") or "")
        if active_id != YELLOW_ID:
            messagebox.showerror(
                "R129 跨磁片测试装配不一致",
                f"此选项只允许 {YELLOW_ID}；当前为 {active_id or '未选择'}。"
                "had-239 请选原 R128，旧 1 mm 触头请选 R127。")
            return None
        profiles = getattr(host, "maglayer_profiles", {})
        source = profiles.get(PHYSICAL_ID, {})
        target = profiles.get(YELLOW_ID, {})
        try:
            target_contact_z = float(target["calibration"]["contact_z_mm"])
            if not math.isfinite(target_contact_z):
                raise ValueError("Yellow Contact Z 非有限值")
            if (not source or target.get("mlx90393") != source.get("mlx90393") or
                    target.get("acquisition") != source.get("acquisition") or
                    target.get("tactile_material_id") !=
                    source.get("tactile_material_id")):
                raise ValueError("两磁片的传感器采集配置或材料标识不一致")
        except (KeyError, TypeError, ValueError) as exc:
            messagebox.showerror("R129 跨磁片测试配置不完整", str(exc))
            return None
        if not messagebox.askyesno(
                "加载未经标定的 Yellow 跨磁片测试？",
                "这不是 had-Yelow 模型：将直接把 had-239＋5 mm 的 R128 磁图、"
                "磁记忆和 70 µT 接触门控原样用于另一块 had-Yelow 磁片。"
                "没有 Yellow 训练、五点 XY 对齐或 Z 输出补偿；"
                f"当前 Yellow Contact Z={target_contact_z:.3f} mm，"
                "had-239 训练 Contact Z=23.700 mm，切勿混作相同绝对标尺。"
                "R128 最新 r6 圆形实测曾发生严重错误分支锁定，"
                "可能输出明显错误的轨迹。确认仍是 5 mm 触头和五颗传感器顺序；"
                "重新采集无载 B0，先观察空载，再做受控点动和短轨迹。"
                "本选择不会自动控制试验台。是否继续？"):
            return None
        try:
            profile = load_profile(R92_PROFILE_PATH, PROFILE_KEY)
            if (profile.get("physical_profile_id") != PHYSICAL_ID or
                    abs(float(profile.get("training_contact_z_mm")) - 23.70)
                    > 1e-6):
                raise ValueError("冻结的 had-239 源参数 ID/Contact Z 不匹配")
            predictor = LiveR128Predictor(
                profile, R92_PROFILE_PATH, TRANSFER_NAME,
                **{key: self.vars[key].get() for key in (
                    "threshold", "contact_threshold", "min_cosine",
                    "smoothing_ms", "jump_policy")},
            )
            # A new preset key separates all saved output registrations; for a
            # true zero-shot test even a newly saved R129 alignment is not
            # automatically applied here.
            predictor.metadata.update({
                "algorithm": "had239_r128_zero_shot_on_had_yelow_r129",
                "fit_variant": "had239_r128_frozen_zero_shot_had_yelow",
                "generalization_test": True,
                "model_weights_zero_shot": True,
                "model_training_profile_id": PHYSICAL_ID,
                "inference_magnetic_layer_profile_id": YELLOW_ID,
                "model_training_contact_z_mm": 23.70,
                "target_session_contact_z_mm": target_contact_z,
                "cross_magnet_xy_alignment_on_load": False,
                "cross_magnet_z_compensation_on_load": False,
                "deployment_qualified": False,
            })
            predictor.model_summary = (
                f"{TRANSFER_NAME} | 冻结 had-239 R128 原参数；Yellow 零样本，"
                "无 XY/Z 跨磁片标定；不保证准确"
            )
            self.invalidate()
            with host.live_predict_lock:
                host.live_predictor = predictor
            for view in (host.live_predict_plot, host.live_predict_3d,
                         host.exp_predict_3d):
                view.set_scale(predictor.visual_scale)
            host.live_predict_state_var.set(
                "R129 Yellow 零样本跨磁片测试已载入：" +
                predictor.model_summary)
            host.live_predict_begin_btn.configure(text="采集 1.0 s 静止 B0 并开始")
            host.live_predict_baseline_var.set(
                "R129：Yellow＋5 mm；had-239 原参数，无跨磁片校准。"
                "新 B0 后先观察空载。")
            return predictor
        except Exception as exc:
            self.invalidate()
            messagebox.showerror("R129 跨磁片模型载入失败", str(exc))
            return None
