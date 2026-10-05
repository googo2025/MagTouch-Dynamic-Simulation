"""Independent unified gating preset, preserving R136 safety and old models.

Import is inert: no GUI, serial device, acquisition or motion is started.
Existing frozen collection plans still demand their originally frozen model.
"""
from __future__ import annotations
import magtouch_acquisition_ui_v1_3_4_R136 as previous
import live_algorithm_ui_r131 as registry
from live_unified_tactile_r330 import GATE_NAME, PRESET_NAME, LiveUnifiedTactilePredictor
from replay_backtest import JUMP_FAST, JUMP_POLICIES

core = previous.core
core.APP_NAME = "MagTouch Hardware Acquisition & Experiment System V1.3.4-R330"
core.BUILD_ID = "20260929-V1.3.4-R330-UNIFIED-STATE-SELECTIVE-XYZ-EXPERIMENTAL"
core.__file__ = __file__
registry.MODEL_SPECS[PRESET_NAME] = registry.CrossModel(
    registry.PHYSICAL_ID, registry.PHYSICAL_ID, registry.R92_PROFILE_PATH,
    registry.PROFILE_KEY, LiveUnifiedTactilePredictor, 23.70)
registry._EXPERIMENTAL_SPECIAL += (PRESET_NAME,)
_PreviousPanel = core.LiveAlgorithmPanel


class R330LiveAlgorithmPanel(_PreviousPanel):
    def __init__(self, host, parent, neural_box, profile_path):
        super().__init__(host, parent, neural_box, profile_path)
        core.ttk.Label(self, text=(
            "R330 独立模式：源侧冻结浅动作识别＋原R133三维解码；不串联旧阈值门、形状门或3帧释放锁。"
            "仅磁响应支持充分时输出XYZ，其余输出压入/停驻/撤出/残余/不确定状态；不保留旧坐标。"
            "本模式显示原始有效坐标，不加EMA、不裁剪升降沿；跨片不自动配准。集成后的新在线实测待完成。"),
            foreground="#00695c", wraplength=1270, justify="left"
        ).grid(row=12, column=0, columnspan=4, sticky="w", padx=4, pady=3)

    def restore(self, config):
        self.available_presets[PRESET_NAME] = registry.PROFILE_KEY
        result = super().restore(config)
        if self.vars["preset"].get() == PRESET_NAME:
            self.vars["jump_policy"].set(GATE_NAME)
        elif self.vars["jump_policy"].get() == GATE_NAME:
            self.vars["jump_policy"].set(JUMP_FAST)
        return result

    def preset_defaults(self):
        selected = self.vars["preset"].get()
        if selected != PRESET_NAME:
            if self.vars["jump_policy"].get() == GATE_NAME:
                self.vars["jump_policy"].set(JUMP_FAST)
            return super().preset_defaults()
        profile = registry.load_profile(registry.R92_PROFILE_PATH, registry.PROFILE_KEY)
        for key in ("threshold", "min_cosine", "smoothing_ms"):
            self.vars[key].set(str(profile["defaults"][key]))
        self.vars["contact_threshold"].set("70")  # inactive legacy scaffolding only
        self.vars["jump_policy"].set(GATE_NAME)

    def update_states(self):
        super().update_states()
        unified = self.vars["preset"].get() == PRESET_NAME
        self.widgets["jump_policy"].configure(values=(GATE_NAME,) if unified else JUMP_POLICIES)
        if unified:
            for key in ("threshold", "contact_threshold", "min_cosine", "smoothing_ms", "jump_policy"):
                self.widgets[key].configure(state="disabled")
        for label in self.grid_slaves(row=2, column=0):
            if isinstance(label, core.ttk.Label):
                label.configure(text=(
                    "R330：旧阈值/相似度/平滑参数不参与推理；磁证据连续确认属于新状态模型内部，"
                    "不是旧3帧释放锁。切回旧算法后恢复原门控选择。" if unified else
                    "旧模式：快速门控使用阈值/形状；迟滞门控额外确认3帧释放并锁定残余。"
                    "R330请在模型列表单独选择统一状态模式，不能混用。"))

    def load_fit(self):
        if self.vars["preset"].get() != PRESET_NAME:
            return super().load_fit()
        if self.vars["jump_policy"].get() != GATE_NAME:
            core.messagebox.showerror("R330 模式冲突", "请选择独立统一状态模式，不能叠加旧门控")
            return None
        predictor = super().load_fit()
        if predictor is not None:
            self.host.live_predict_state_var.set("R330 独立统一状态模式已载入："+predictor.model_summary)
            self.host.live_predict_baseline_var.set(
                "显式无载B0后预热0.7秒；先观察空载。旧冻结计划仍需原模型，不能混称同一次实测。")
        return predictor


core.LiveAlgorithmPanel = R330LiveAlgorithmPanel


def unified_active(host):
    return isinstance(getattr(host, "live_predictor", None), LiveUnifiedTactilePredictor)


# The legacy UI has a second display-only gate/EMA stage. Scope its bypass
# strictly to R330, so saved XYZ and shown XYZ have the same validity policy.
_edge = core.MagTouchApp._live_edge_guard_ms
_trim = core.MagTouchApp._trim_live_falling_edge
_filter = core.MagTouchApp._live_trajectory_filter_apply
_label = core.MagTouchApp._live_filter_label
_display = core.MagTouchApp._live_display_xyz

def _unified_edge(host):
    return 0. if unified_active(host) else _edge(host)

def _unified_trim(host, end_ns, show_live_prediction, show_experiment):
    return 0 if unified_active(host) else _trim(host, end_ns, show_live_prediction, show_experiment)

def _unified_filter(host, t_ns, xyz):
    return tuple(float(v) for v in xyz) if unified_active(host) else _filter(host, t_ns, xyz)

def _unified_label(host):
    return "R330原始有效XYZ（无显示滤波/边沿裁剪）" if unified_active(host) else _label(host)

def _unified_display(host, prediction):
    if unified_active(host):
        phase = prediction.extra_outputs.get("action_phase", 0)
        phase_text = {0: "不确定", 1: "静默", 2: "压入", 3: "停驻", 4: "撤出", 5: "残余"}.get(phase, "未知")
        host.live_predict_motion_var.set("统一状态："+phase_text+"；磁支持定位 / XYZ有效")
        host.live_predict_output_var.set("原始磁场因果推理；不读取打印机位置/动作模板，不加旧阈值门或显示平滑。")
    return _display(host, prediction)

core.MagTouchApp._live_edge_guard_ms = _unified_edge
core.MagTouchApp._trim_live_falling_edge = _unified_trim
core.MagTouchApp._live_trajectory_filter_apply = _unified_filter
core.MagTouchApp._live_filter_label = _unified_label
core.MagTouchApp._live_display_xyz = _unified_display

def main():
    return previous.main()

if __name__ == "__main__": main()
