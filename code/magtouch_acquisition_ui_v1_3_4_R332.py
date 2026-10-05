"""Opt-in localization-evidence mode; keeps old modes and R136 motion guards.

Import is inert. Launching/collecting data is a separate user action. Old frozen
plans still require their original model and may not masquerade as R332 tests.
"""
from __future__ import annotations

import magtouch_acquisition_ui_v1_3_4_R330 as previous
import live_algorithm_ui_r131 as registry
from live_unified_tactile_r332 import GATE_NAME, PRESET_NAME, LiveEvidenceTactilePredictor
from replay_backtest import JUMP_FAST
from training.localization_permission_runtime_r332 import frozen_permission

core = previous.core
core.APP_NAME = "MagTouch Hardware Acquisition & Experiment System V1.3.4-R332"
core.BUILD_ID = "20260929-V1.3.4-R332-STATE-LOCALIZATION-EVIDENCE-EXPERIMENTAL"
core.__file__ = __file__
registry.MODEL_SPECS[PRESET_NAME] = registry.CrossModel(
    registry.PHYSICAL_ID, registry.PHYSICAL_ID, registry.R92_PROFILE_PATH,
    registry.PROFILE_KEY, LiveEvidenceTactilePredictor, 23.70)
registry._EXPERIMENTAL_SPECIAL += (PRESET_NAME,)
_PreviousPanel = previous.R330LiveAlgorithmPanel


class R332LiveAlgorithmPanel(_PreviousPanel):
    def __init__(self, host, parent, neural_box, profile_path):
        super().__init__(host, parent, neural_box, profile_path)
        core.ttk.Label(self, text=(
            "R332：在R330独立状态链路上增加源侧冻结的磁定位证据层。原有效XYZ不变；"
            "弱响应先计算不显示的候选位置，再按磁图拟合、局部可辨识性与历史证据选择输出。"
            "跨片不训练、不自动配准，不加显示平滑；0.25/0.3mm不是硬门槛。新在线实测待完成。"),
            foreground="#005b96", wraplength=1270, justify="left"
        ).grid(row=13, column=0, columnspan=4, sticky="w", padx=4, pady=3)

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
        self.vars["contact_threshold"].set("70")  # ignored legacy scaffolding
        self.vars["jump_policy"].set(GATE_NAME)

    def update_states(self):
        super().update_states()
        if self.vars["preset"].get() == PRESET_NAME:
            self.widgets["jump_policy"].configure(values=(GATE_NAME,))
            for key in ("threshold", "contact_threshold", "min_cosine", "smoothing_ms", "jump_policy"):
                self.widgets[key].configure(state="disabled")
            for label in self.grid_slaves(row=2, column=0):
                if isinstance(label, core.ttk.Label):
                    label.configure(text="R332独立状态＋定位证据：旧阈值/形状门/3帧释放锁不参与。原有效XYZ保留，新增点由源侧冻结许可层决定。")

    def load_fit(self):
        if self.vars["preset"].get() != PRESET_NAME:
            return super().load_fit()
        if self.vars["jump_policy"].get() != GATE_NAME:
            core.messagebox.showerror("R332 模式冲突", "请选择R332独立状态＋定位证据，不能混用旧门控")
            return None
        predictor = super().load_fit()
        if predictor is not None:
            self.host.live_predict_state_var.set("R332 独立定位证据模式已载入："+predictor.model_summary)
            self.host.live_predict_baseline_var.set(
                "明确无载后采B0，预热0.7秒；先观察空载。旧冻结计划不能换模型冒充同一实测。")
        return predictor


core.LiveAlgorithmPanel = R332LiveAlgorithmPanel
_previous_label = core.MagTouchApp._live_filter_label


def _evidence_label(host):
    if isinstance(getattr(host, "live_predictor", None), LiveEvidenceTactilePredictor):
        return "R332原始有效XYZ（磁定位证据，无显示滤波/裁剪）"
    return _previous_label(host)


core.MagTouchApp._live_filter_label = _evidence_label


def validate_evidence_plan_row(host, item):
    if item.get("system_protocol_r332") != "frozen_online_acceptance_v1":
        return
    value, manifest = frozen_permission()
    with host.live_predict_lock:
        predictor = host.live_predictor
        enabled = bool(host.live_predict_enabled)
    if not enabled or not isinstance(predictor, LiveEvidenceTactilePredictor):
        raise ValueError("R332验收计划必须启用R332独立定位证据模式")
    actual = predictor.metadata.get("localization_permission_protocol", {}).get("sha256")
    required = item.get("required_localization_permission_sha256")
    if not required or required != manifest["sha256"] or actual != required:
        raise ValueError("R332定位许可模型冻结SHA不一致；不能将换模型后的记录混为同一次验收")
    if predictor.metadata.get("contact_gate") != GATE_NAME:
        raise ValueError("R332验收计划不能叠加旧门控")


_previous_preflight = core.MagTouchApp._validate_full_plan_required_hardware
_previous_dynamic_b0 = core.MagTouchApp._capture_full_plan_dynamic_b0
_previous_execute = core.MagTouchApp._execute_stage_experiment


def _evidence_preflight(host, item):
    validate_evidence_plan_row(host, item)
    return _previous_preflight(host, item)


def _evidence_dynamic_b0(host, item, queue_index, total):
    # Recheck before this Trial's first safe-lift motion, not after acquisition.
    validate_evidence_plan_row(host, item)
    return _previous_dynamic_b0(host, item, queue_index, total)


def _evidence_execute(host, cfg):
    validate_evidence_plan_row(host, cfg)
    return _previous_execute(host, cfg)


core.MagTouchApp._validate_full_plan_required_hardware = _evidence_preflight
core.MagTouchApp._capture_full_plan_dynamic_b0 = _evidence_dynamic_b0
core.MagTouchApp._execute_stage_experiment = _evidence_execute


def main():
    return previous.main()


if __name__ == "__main__": main()
