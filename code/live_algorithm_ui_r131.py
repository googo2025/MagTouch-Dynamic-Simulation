"""R131: opt-in cross-magnet loading without changing frozen model weights.

The older selectors keep their original same-profile behavior.  Only a
different active magnetic profile takes this explicit, unregistered test path.
Importing this module never starts acquisition or controls the stage.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, ttk

from live_algorithm_ui_r91 import (
    FIT, R89_MAGNETIC_PROFILE_ID, R89_PRESET_NAME, R89_PROFILE_KEY,
    R89_PROFILE_PATH,
)
from live_algorithm_ui_r92 import R92_PRESETS, R92_PROFILE_PATH
from live_algorithm_ui_r126 import (
    PHYSICAL_ID, PRESET_NAME as R126_NAME, PROFILE_KEY,
)
from live_algorithm_ui_r127 import PRESET_NAME as R127_NAME
from live_algorithm_ui_r128 import PRESET_NAME as R128_NAME
from live_algorithm_ui_r129 import (
    TRANSFER_NAME, YELLOW_ID,
)
from live_algorithm_ui_r130 import (
    GUARD_NAME, NEURAL_NAME, LiveAlgorithmPanel as _R130Panel,
)
from live_field_memory_r126 import LiveR126Predictor
from live_field_memory_branch_r145 import LiveR128Predictor
from live_field_memory_r130 import LiveR130GuardPredictor, LiveR130NeuralPredictor
from live_field_memory_r133 import LiveR133CoveragePredictor
from live_field_memory_r219 import LiveR219TightBranchPredictor
from live_field_memory_r223 import LiveR223AdaptiveBranchPredictor
from live_normalized_fit import LiveNormalizedFitPredictor
from live_old1mm_field_memory_r134 import (
    LiveOld1mmFieldMemoryPredictor, R89_CONFIG, R89_KEY,
)
from live_old1mm_r89 import LiveR89Predictor
from replay_backtest import JUMP_FAST, load_profile
from training.had239_r133_coverage_xy_runtime import (
    ARTIFACT as R133_ARTIFACT, METADATA as R133_METADATA,
)


COVERAGE_NAME = "5×5｜had-239＋5 mm｜D覆盖重训因果XY网络R133（实验）"
R219_NAME = "5×5｜had-239＋5 mm｜强磁证据分支＋R133因果XY R219（实验）"
R223_NAME = "5×5｜had-239＋5 mm｜浅层自适应分支＋R133因果XY R223（实验）"
_EXPERIMENTAL_SPECIAL = (GUARD_NAME, NEURAL_NAME, COVERAGE_NAME, R219_NAME,
                         R223_NAME)


@dataclass(frozen=True)
class CrossModel:
    source_profile_id: str
    original_target_id: str
    path: Path
    key: str
    predictor_type: type
    training_contact_z_mm: float
    needs_key: bool = False


OLD1MM_ID = R89_MAGNETIC_PROFILE_ID
_FIVE = CrossModel(PHYSICAL_ID, PHYSICAL_ID, R92_PROFILE_PATH,
                   PROFILE_KEY, LiveR128Predictor, 23.70)
MODEL_SPECS = {
    R89_PRESET_NAME: CrossModel(OLD1MM_ID, OLD1MM_ID, R89_PROFILE_PATH,
                               R89_PROFILE_KEY, LiveR89Predictor, 16.0, True),
    R126_NAME: CrossModel(PHYSICAL_ID, PHYSICAL_ID, R92_PROFILE_PATH,
                          PROFILE_KEY, LiveR126Predictor, 23.70),
    R127_NAME: CrossModel(OLD1MM_ID, OLD1MM_ID, R89_CONFIG, R89_KEY,
                          LiveOld1mmFieldMemoryPredictor, 16.0),
    R128_NAME: _FIVE,
    TRANSFER_NAME: CrossModel(PHYSICAL_ID, YELLOW_ID, R92_PROFILE_PATH,
                              PROFILE_KEY, LiveR128Predictor, 23.70),
    GUARD_NAME: CrossModel(PHYSICAL_ID, PHYSICAL_ID, R92_PROFILE_PATH,
                           PROFILE_KEY, LiveR130GuardPredictor, 23.70),
    NEURAL_NAME: CrossModel(PHYSICAL_ID, PHYSICAL_ID, R92_PROFILE_PATH,
                            PROFILE_KEY, LiveR130NeuralPredictor, 23.70),
    COVERAGE_NAME: CrossModel(PHYSICAL_ID, PHYSICAL_ID, R92_PROFILE_PATH,
                              PROFILE_KEY, LiveR133CoveragePredictor, 23.70),
    R219_NAME: CrossModel(PHYSICAL_ID, PHYSICAL_ID, R92_PROFILE_PATH,
                          PROFILE_KEY, LiveR219TightBranchPredictor, 23.70),
    R223_NAME: CrossModel(PHYSICAL_ID, PHYSICAL_ID, R92_PROFILE_PATH,
                          PROFILE_KEY, LiveR223AdaptiveBranchPredictor, 23.70),
}
for _label, _source in R92_PRESETS.items():
    MODEL_SPECS[_label] = CrossModel(
        _source["physical_profile_id"], _source["physical_profile_id"],
        R92_PROFILE_PATH, _source["key"], LiveNormalizedFitPredictor,
        _source["contact_z_mm"])


def _verify_frozen_source(profile: dict, spec: CrossModel) -> None:
    source = (profile.get("physical_profile_id") or
              profile.get("physical_configuration", {}).get("profile_id"))
    z = (profile.get("training_contact_z_mm") if
         "training_contact_z_mm" in profile else
         profile.get("physical_configuration", {}).get("contact_z_mm"))
    if source != spec.source_profile_id or z is None or not math.isclose(
            float(z), spec.training_contact_z_mm, abs_tol=1e-6):
        raise ValueError("冻结参数文件的源磁片或 Contact Z 与模型注册项不一致")
    if (spec.predictor_type is LiveNormalizedFitPredictor and
            profile.get("dynamic_model") is not None):
        raise ValueError("R92 静态模型参数意外包含动态模型")


class LiveAlgorithmPanel(_R130Panel):
    def __init__(self, host, parent, neural_box, profile_path):
        super().__init__(host, parent, neural_box, profile_path)
        # The inherited R130 banner predates both the prospective 9/27 runs
        # and opt-in cross-profile testing. Do not leave contradictory advice.
        for widget in self.grid_slaves(row=9, column=0):
            if isinstance(widget, ttk.Label):
                widget.configure(text=(
                    "R130：分支保护与因果 XY 网络的 9/27 新会话已有真实在线记录；"
                    "r4 圆形明显改善，但浅层门控、XZ 联动和跨磁片泛化仍需继续检验。"
                    "所有模型参数保持冻结，跨磁片仅作人工确认的实验测试。"))
        ttk.Label(
            self,
            text=("R131 跨磁片测试：上方旧版本文字中的 Profile 限制不再阻止模型载入。"
                  "不同磁片／触头可手动选任一已列模型；参数保持原样、不自动借用五点对齐或"
                  "补偿 Z，实际磁片和源磁片均写入推理元数据。跨片结果仅作实验，"
                  "必须重新采无载 B0 并先观察空载。已撤下的 R65 仍不恢复。"),
            foreground="#b71c1c", wraplength=1270, justify="left",
        ).grid(row=10, column=0, columnspan=4, sticky="w", padx=4, pady=2)
        ttk.Label(
            self,
            text=("R133-D 已有真实在线测试。R219 固定 1.25 倍否决在旧目标器件圆形轨迹"
                  "上退化，仅留作实验对照。R223 仅在磁图浅层加强否决，深层保持 R133；"
                  "旧数据离线筛查通过，需新独立 Trial 验证。Z 与接触门控均不变。"),
            foreground="#9c4b00", wraplength=1270, justify="left",
        ).grid(row=11, column=0, columnspan=4, sticky="w", padx=4, pady=2)

    def restore(self, config):
        if (R92_PROFILE_PATH.is_file() and R133_ARTIFACT.is_file() and
                R133_METADATA.is_file()):
            self.available_presets[COVERAGE_NAME] = PROFILE_KEY
            self.available_presets[R219_NAME] = PROFILE_KEY
            self.available_presets[R223_NAME] = PROFILE_KEY
        result = super().restore(config)
        if self.vars["preset"].get() in (COVERAGE_NAME, R219_NAME, R223_NAME):
            self.vars["contact_threshold"].set("70")
            self.vars["jump_policy"].set(JUMP_FAST)
        return result

    def preset_defaults(self):
        if self.vars["preset"].get() not in (COVERAGE_NAME, R219_NAME, R223_NAME):
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
        if self.vars["preset"].get() in (COVERAGE_NAME, R219_NAME, R223_NAME):
            for key in ("threshold", "contact_threshold", "min_cosine",
                        "smoothing_ms", "jump_policy"):
                self.widgets[key].configure(state="disabled")

    def load_fit(self):
        selected = self.vars["preset"].get()
        spec = MODEL_SPECS.get(selected)
        if spec is None:
            return super().load_fit()
        if self.busy() or self.vars["mode"].get() != FIT:
            return None
        host = self.host
        active_id = str(getattr(host, "active_mag_profile_id", "") or "")
        if (active_id == spec.original_target_id and
                selected not in _EXPERIMENTAL_SPECIAL):
            return super().load_fit()

        profiles = getattr(host, "maglayer_profiles", {})
        actual = profiles.get(active_id, {})
        actual_z = actual.get("calibration", {}).get("contact_z_mm")
        source = profiles.get(spec.source_profile_id, {})
        same_readout = bool(source and actual and
                            source.get("mlx90393") == actual.get("mlx90393") and
                            source.get("acquisition") == actual.get("acquisition"))
        cross = active_id != spec.source_profile_id
        version = ("R223" if selected == R223_NAME else
                   "R219" if selected == R219_NAME else
                   "R133" if selected == COVERAGE_NAME else "R131")
        if selected in (GUARD_NAME, NEURAL_NAME):
            tested_note = "R130 在 9/27 新会话已有真实在线测试，但仍属实验模型。"
        elif selected == COVERAGE_NAME:
            tested_note = "R133-D 网络已有 9/28 had-239 与 Yellow 真实在线复刻；Yellow 的静态、Z 与浅层覆盖仍需分项验收。"
        elif selected == R219_NAME:
            tested_note = "R219 固定 1.25 倍否决会使旧目标器件圆轨迹退化；仅作消融对照，不建议作为主测试。"
        elif selected == R223_NAME:
            tested_note = "R223 浅层自适应否决只在已查看数据上离线筛查；尚无新独立在线测试。"
        else:
            tested_note = ""
        native_r130 = (not cross and selected in (GUARD_NAME, NEURAL_NAME))
        registration_note = (
            "同片 R130 可沿用此模型已保存的输出对齐。" if native_r130
            else "本次不自动应用旧五点对齐或补偿 Z。")
        if not messagebox.askyesno(
                f"{version} 模型测试：确认继续载入？",
                f"所选模型：{selected}\n"
                f"训练磁片：{spec.source_profile_id}；原入口目标："
                f"{spec.original_target_id}；当前磁片：{active_id or '未选择'}。\n"
                f"训练 Contact Z={spec.training_contact_z_mm:.3f} mm；"
                f"当前 Contact Z={actual_z if actual_z is not None else '未知'} mm。\n"
                f"传感器采集配置与源 Profile {'相同' if same_readout else '不同或未知'}。"
                + tested_note + registration_note +
                "模型权重不变；错误触头／传感器顺序可能造成严重误差。"
                "请核对实物，载入后重新采集无载 B0，先观察空载，再做受控运动。"
                "本操作不会自动控制实验台。"):
            return None
        try:
            profile = load_profile(spec.path, spec.key)
            _verify_frozen_source(profile, spec)
            settings = {key: self.vars[key].get() for key in (
                "threshold", "contact_threshold", "min_cosine",
                "smoothing_ms", "jump_policy")}
            if spec.needs_key:
                settings["key"] = spec.key
            predictor = spec.predictor_type(profile, spec.path, selected,
                                            **settings)
            if native_r130 and hasattr(host, "_live_alignment_apply_to_predictor"):
                host._live_alignment_apply_to_predictor(predictor, selected)
            predictor.metadata.update({
                "cross_magnet_test_r131": cross,
                "model_training_profile_id": spec.source_profile_id,
                "model_original_target_profile_id": spec.original_target_id,
                "inference_magnetic_layer_profile_id": active_id,
                "model_training_contact_z_mm": spec.training_contact_z_mm,
                "target_session_contact_z_mm": actual_z,
                "cross_magnet_sensor_config_equal": same_readout,
                "deployment_qualified": False,
            })
            if cross:
                predictor.metadata.update({
                    "cross_magnet_xy_alignment_on_load": False,
                    "cross_magnet_z_compensation_on_load": False,
                })
                predictor.model_summary += (
                    f" | R131 跨片测试：{spec.source_profile_id} → "
                    f"{active_id or '未选择'}，无自动 XY/Z 配准")
            elif native_r130:
                predictor.metadata["validation_status"] = (
                    "S20260927_A prospective live tests recorded; still experimental")
            self.invalidate()
            with host.live_predict_lock:
                host.live_predictor = predictor
            for view in (host.live_predict_plot, host.live_predict_3d,
                         host.exp_predict_3d):
                view.set_scale(predictor.visual_scale)
            host.live_predict_state_var.set(f"{version} 实验模型已载入：" +
                                            predictor.model_summary)
            host.live_predict_begin_btn.configure(text="采集 1.0 s 静止 B0 并开始")
            host.live_predict_baseline_var.set(
                (f"{version} 跨片测试：模型源／实际磁片不匹配；无自动 XY/Z 配准。"
                 if cross else f"{version} 同源磁片测试：请核对该模型的输出对齐状态。") +
                "新 B0 后先观察空载与门控，再进行受控运动。")
            return predictor
        except Exception as exc:
            self.invalidate()
            messagebox.showerror(f"{version} 模型载入失败", str(exc))
            return None
