"""R40 algorithm selector. Uses existing acquisition and recording pipeline only."""
from pathlib import Path
from typing import Any
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from replay_backtest import (CUSTOM, DEFAULT_PRESET, load_profile,
                             normalize_preset_name, available_presets,
                             JUMP_POLICIES, JUMP_HYSTERESIS)
from live_normalized_fit import LiveNormalizedFitPredictor

FIT = '归一化拟合算法'
NN = '神经网络预测'


class LiveAlgorithmPanel(ttk.LabelFrame):
    def __init__(self, host: Any, parent, neural_box, profile_path):
        super().__init__(parent, text='实时预测算法 / Live algorithm — R40')
        self.host = host
        self.profile_path = Path(profile_path)
        self.available_presets = available_presets(self.profile_path)
        self.loading = False
        self.mutating = False
        self.vars = {k: tk.StringVar(self, value=v) for k,v in dict(
            mode=NN, preset=DEFAULT_PRESET, custom_path='', threshold='76.572',
            contact_threshold='76.572', min_cosine='0.8', smoothing_ms='250',
            jump_policy=JUMP_HYSTERESIS).items()}
        self.custom_values = dict(threshold='76.572', contact_threshold='76.572',
                                  min_cosine='0.8', smoothing_ms='250')
        self.widgets = {}
        self.columnconfigure(1, weight=1)
        self._field(self, 'mode', '预测大类', 0, 0, [FIT, NN], 22)
        self.fit_box = ttk.Frame(self)
        self.fit_box.grid(row=1, column=0, columnspan=4, sticky='ew')
        self.fit_box.columnconfigure(3, weight=1)
        self._field(self.fit_box, 'preset', '磁片与拟合类型', 0, 0,
                    [*self.available_presets, CUSTOM], 36)
        self._field(self.fit_box, 'custom_path', '自由设置标定JSON', 0, 2, width=32)
        self.browse = ttk.Button(self.fit_box, text='选择文件', command=self.choose)
        self.browse.grid(row=0, column=4, padx=3)
        params = ttk.Frame(self.fit_box); params.grid(row=1, column=0, columnspan=5, sticky='ew')
        for c,(k,label) in enumerate([('threshold','响应阈值µT'),('contact_threshold','临时接触阈值µT'),
                                      ('min_cosine','最低相似度'),('smoothing_ms','平滑ms')]):
            self._field(params,k,label,0,c*2,width=8)
        self.apply = ttk.Button(params,text='载入 / 应用拟合算法',command=self.load_fit)
        self.apply.grid(row=0,column=8,padx=8)
        self._field(params,'jump_policy','抑制跳变',1,0,JUMP_POLICIES,24)
        ttk.Label(self, text=('每种磁层均保留九点、121点与441点基线；R40新增0.5/0.8 mm工作区静态模型和因果历史动态补偿模型。'
                              '三种441点选项只在该磁层模型真实生成后出现，不会回退或冒充121点模型。'
                              '九点连续使用本磁层自己的粗线性XY＋全局Z，仅限±4 mm历史九点证据，不能视为20×20 mm正式验证。\n'
                              '“快速门控”逐帧使用阈值/形状；“迟滞门控”额外确认连续3帧释放并锁定残余；五点校准只修正输出XY平移。'),
                  foreground='#8b4513').grid(row=2,column=0,columnspan=4,sticky='w',padx=4)
        self.neural_widgets = [w for w in neural_box.winfo_children()
                               if int(w.grid_info().get('row',-1)) in (1,2,3)
                               and isinstance(w,(ttk.Entry,ttk.Combobox,ttk.Button))]
        self.restore(host.settings.get('live_algorithm',{}))
        self.saved = self.snapshot()
        for k,v in self.vars.items(): v.trace_add('write',lambda *_, key=k:self.changed(key))
        self.poll()

    def _field(self, parent, key, label, row, col, choices=None, width=12):
        ttk.Label(parent,text=label).grid(row=row,column=col,padx=4,pady=2)
        w = (ttk.Combobox(parent,textvariable=self.vars[key],values=choices,width=width,state='readonly')
             if choices else ttk.Entry(parent,textvariable=self.vars[key],width=width))
        w.grid(row=row,column=col+1,sticky='ew',padx=3,pady=2)
        self.widgets[key] = w

    def busy(self):
        return self.loading or self.host.recorder is not None or self.host.full_plan_running

    def snapshot(self):
        return {**{k:v.get() for k,v in self.vars.items()},'custom_values':dict(self.custom_values)}

    def restore(self, config):
        self.mutating = True
        try:
            for k,v in self.vars.items():
                if k in config: v.set(str(config[k]))
            self.vars['preset'].set(normalize_preset_name(self.vars['preset'].get()))
            if self.vars['mode'].get() not in (FIT,NN): self.vars['mode'].set(NN)
            if self.vars['preset'].get() not in (*self.available_presets,CUSTOM):
                fallback=(DEFAULT_PRESET if DEFAULT_PRESET in self.available_presets
                          else next(iter(self.available_presets),CUSTOM))
                self.vars['preset'].set(fallback)
            self.custom_values.update(config.get('custom_values',{}))
            if self.vars['preset'].get() != CUSTOM: self.preset_defaults()
            # Unlike immutable preset calibration parameters, the temporary
            # contact gate is deliberately user-tunable and must survive restart.
            if 'contact_threshold' in config:
                self.vars['contact_threshold'].set(str(config['contact_threshold']))
        finally: self.mutating = False
        self.saved = self.snapshot()

    def preset_defaults(self):
        preset = self.vars['preset'].get()
        if preset == CUSTOM:
            defaults = self.custom_values
        else:
            defaults = load_profile(self.profile_path, self.available_presets[preset]).get('defaults',{})
        for k in ('threshold','min_cosine','smoothing_ms'):
            if k in defaults: self.vars[k].set(str(defaults[k]))
        contact=defaults.get('contact_threshold',defaults.get('threshold',0))
        self.vars['contact_threshold'].set(str(contact))

    def invalidate(self):
        h = self.host
        with h.live_predict_lock:
            h.live_predict_enabled = False
            h.live_predictor = None
            h._live_predict_clear_queues()
        for name in ('live_predict_plot','live_predict_3d','exp_predict_3d'):
            view = getattr(h,name,None)
            if view is not None: view.clear()
        h._live_predict_clear_raw_output()
        h.live_predict_profile_confirm_var.set(False)
        h.live_predict_motion_var.set('状态 / State: --')
        h.live_predict_state_var.set('设置已改变，请载入对应算法，然后重新采集静止 B0')
        h.live_predict_baseline_var.set('B0：未建立')
        h.live_predict_output_var.set('等待载入算法与当前无载 B0')
        if hasattr(h,'exp_predict_info_var'): h.exp_predict_info_var.set('等待重新载入算法与 B0')

    def changed(self, key):
        if self.mutating: return
        if self.busy():
            self.restore(self.saved)
            return
        self.mutating = True
        try:
            if key == 'preset': self.preset_defaults()
            if self.vars['preset'].get() == CUSTOM:
                self.custom_values.update({k:self.vars[k].get() for k in self.custom_values})
        finally: self.mutating = False
        self.saved = self.snapshot()
        if key == 'preset' and hasattr(self.host,'_live_alignment_update_status'):
            self.host._live_alignment_update_status(self.vars['preset'].get())
        self.invalidate()
        self.update_states()
        self.host._schedule_settings_autosave()

    def choose(self):
        path = filedialog.askopenfilename(title='选择归一化拟合标定JSON',filetypes=[('JSON','*.json')])
        if path: self.vars['custom_path'].set(path)

    def update_states(self):
        busy = self.busy(); fit = self.vars['mode'].get() == FIT
        custom = self.vars['preset'].get() == CUSTOM
        for key,w in self.widgets.items():
            enabled = not busy and (key == 'mode' or (fit and (
                key in ('preset','contact_threshold','jump_policy') or custom)))
            w.configure(state=('readonly' if isinstance(w,ttk.Combobox) else 'normal') if enabled else 'disabled')
        self.browse.configure(state='normal' if fit and custom and not busy else 'disabled')
        self.apply.configure(state='normal' if fit and not busy else 'disabled')
        for w in self.neural_widgets:
            w.configure(state=('readonly' if isinstance(w,ttk.Combobox) else 'normal') if not fit and not busy else 'disabled')

    def poll(self):
        self.update_states()
        self.host.after(250,self.poll)

    def load_fit(self):
        if self.busy() or self.vars['mode'].get() != FIT: return
        h = self.host
        try:
            preset = self.vars['preset'].get()
            path = Path(self.vars['custom_path'].get()).expanduser() if preset == CUSTOM else self.profile_path
            profile = load_profile(path, None if preset == CUSTOM else self.available_presets[preset])
            predictor = LiveNormalizedFitPredictor(profile,path,preset,**{
                k:self.vars[k].get() for k in (
                    'threshold','contact_threshold','min_cosine','smoothing_ms','jump_policy')})
            if hasattr(h,'_live_alignment_apply_to_predictor'):
                h._live_alignment_apply_to_predictor(predictor,preset)
            self.invalidate()
            with h.live_predict_lock: h.live_predictor = predictor
            for view in (h.live_predict_plot,h.live_predict_3d,h.exp_predict_3d): view.set_scale(predictor.visual_scale)
            h.live_predict_state_var.set('拟合算法已载入：'+predictor.model_summary)
            h.live_predict_begin_btn.configure(text='采集 1.0 s 静止 B0 并开始')
            h.live_predict_baseline_var.set('先手动启动磁场采集；确认磁片/传感器配置一致，在未受力时采集 B0。')
        except Exception as exc:
            self.invalidate()
            messagebox.showerror('拟合算法载入失败',str(exc))
