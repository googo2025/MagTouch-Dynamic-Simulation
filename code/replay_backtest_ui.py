"""R39 replay-only UI. Workers never touch Tk or acquisition/printer objects."""
from __future__ import annotations
import bisect
import csv
import json
import math
import queue
import threading
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from replay_backtest import (RAW_FIELDS, CUSTOM, DEFAULT_PRESET, SCHEMA, load_profile,
                             normalize_preset_name, available_presets,
                             run_backtest, empty_record, record, number,
                             JUMP_POLICIES, JUMP_HYSTERESIS)
from contact_display import history_record, contact_window, display_point_visible
from neural_model_registry import (
    LIVE_NEURAL_MODEL_OPTIONS,
    neural_model_family,
    resolve_neural_model_selection,
)

FIT='归一化拟合算法'
NN='神经网络预测'
RECALC='从原始磁场重算'
HISTORY='历史预测记录（不重算）'
AUTO='自动：优先已有预测记录'
DEFAULTS=dict(mode=FIT,preset=DEFAULT_PRESET,custom_path='',threshold='76.572',contact_threshold='76.572',
               min_cosine='0.8',smoothing_ms='250',checkpoint='',family='auto',mapping='{}',
               device='cpu',source=AUTO,baseline_start='0',baseline_seconds='1',
               baseline_policy='动作前自动更新',visual='auto',
               jump_policy=JUMP_HYSTERESIS,edge_guard_ms='120')

class ReplayBacktestPanel(ttk.LabelFrame):
    def __init__(self,host,parent,plot,profile_path):
        super().__init__(parent,text='预测记录回放 / 原始磁场重算 — R39')
        self.host=host; self.plot=plot; self.profile_path=Path(profile_path)
        self.available_presets=available_presets(self.profile_path)
        self.vars={k:tk.StringVar(self,value=v) for k,v in DEFAULTS.items()}
        self.widgets={}; self.results=None; self.result_config=None
        self.recorded_results=[]
        self.worker=None; self.cancel_event=threading.Event(); self.mail=queue.Queue()
        self.generation=0; self.closed=False; self.mutating=False; self.play_after=False
        self.custom_values={k:DEFAULTS[k] for k in ('threshold','contact_threshold','min_cosine','smoothing_ms')}
        self.columnconfigure(0,weight=1)
        line=ttk.Frame(self);line.grid(row=0,column=0,sticky='ew');line.columnconfigure(1,weight=1)
        self._field(line,'mode','回测大类',0,0,[FIT,NN],width=24)
        self._field(line,'source','结果来源',1,0,[AUTO,HISTORY,RECALC],width=29)
        self.fit_box=ttk.LabelFrame(self,text=FIT);self.fit_box.grid(row=1,column=0,sticky='ew',padx=3)
        self.fit_box.columnconfigure(1,weight=1)
        self._field(self.fit_box,'preset','磁片与拟合类型',0,0,
                    [*self.available_presets,CUSTOM],width=36)
        self._field(self.fit_box,'custom_path','自定义标定JSON',1,0,width=30)
        self.custom_browse=ttk.Button(self.fit_box,text='选择文件',command=self.browse_profile)
        self.custom_browse.grid(row=1,column=2,padx=2)
        self.profile_export=ttk.Button(self.fit_box,text='导出参数模板',command=self.export_profile)
        self.profile_export.grid(row=0,column=2,padx=2)
        params=ttk.Frame(self.fit_box);params.grid(row=2,column=0,columnspan=3,sticky='ew')
        for col,(key,label) in enumerate([('threshold','响应阈值µT'),('contact_threshold','临时接触阈值µT'),
                                          ('min_cosine','相似度'),('smoothing_ms','平滑ms')]):
            self._field(params,key,label,0,col*2,width=8)
        self.nn_box=ttk.LabelFrame(self,text=NN);self.nn_box.grid(row=2,column=0,sticky='ew',padx=3)
        self.nn_box.columnconfigure(1,weight=1)
        self._field(self.nn_box,'checkpoint','模型选项 / .pt',1,0,
                    list(LIVE_NEURAL_MODEL_OPTIONS),width=42)
        self.widgets['checkpoint'].bind('<<ComboboxSelected>>',self.model_option_selected)
        self.model_browse=ttk.Button(self.nn_box,text='选择模型',command=self.browse_model)
        self.model_browse.grid(row=1,column=2,padx=2)
        opts=ttk.Frame(self.nn_box);opts.grid(row=2,column=0,columnspan=3,sticky='ew')
        self._field(opts,'family','类型',0,0,
                    ['auto','H1','H2-R15','H2-Spatial-R16','H2-Unified-R18','FinalTip-R56'],width=18)
        self._field(opts,'device','设备',0,2,['cpu','cuda:0'],width=9)
        self._field(self.nn_box,'mapping','输出映射JSON',3,0,width=30)
        common=ttk.Frame(self);common.grid(row=3,column=0,sticky='ew',padx=3,pady=3)
        self._field(common,'baseline_start','无载B0起点s',0,0,width=7)
        self._field(common,'baseline_seconds','时长s',0,2,width=7)
        self._field(common,'baseline_policy','B0策略',1,0,['仅指定无载窗口','动作前自动更新'],width=19)
        self._field(common,'visual','3D字段',1,2,['auto','xyz','XYZ','dXYZ'],width=8)
        self._field(common,'jump_policy','抑制跳变',2,0,JUMP_POLICIES,width=24)
        self._field(common,'edge_guard_ms','升/降沿抑制 ms',2,2,width=8)
        actions=ttk.Frame(self);actions.grid(row=4,column=0,sticky='ew')
        self.run_button=ttk.Button(actions,text='应用设置并回测',command=self.start)
        self.run_button.pack(side='left',padx=3)
        self.cancel_button=ttk.Button(actions,text='取消计算',command=self.cancel)
        self.cancel_button.pack(side='left',padx=3)
        self.export_button=ttk.Button(actions,text='导出回测CSV',command=self.export_results)
        self.export_button.pack(side='left',padx=3)
        self.status=tk.StringVar(self,value='请载入Trial；播放直接回放视频/磁场，算法重算请点击“应用设置并回测”。')
        ttk.Label(self,textvariable=self.status,wraplength=690,foreground='#0d47a1').grid(row=5,column=0,sticky='ew',padx=4)
        ttk.Label(self,text=('每种磁层均可回测九点和121点基线；完成本磁层1 mm标定后自动增加441点离散、连续、混合模型。'
                            '三种441点选项只在对应参数真实生成后显示，不会静默回退到121点。'
                            '九点连续只依据±4 mm历史九点，边缘/四角留一位置属于外推，不能替代密集冻结验证。'
                            '快速门控逐帧判别；迟滞门控增加连续3帧释放确认/锁定。\n'
                            'B0窗口须确认无载；动作前自动B0适用于计划实验，不代表已验证任意随机/持续触碰。'
                            '边沿抑制会裁掉每段接触头尾瞬态，不改原始预测和导出值；设为0可关闭。NN默认CPU，仅加载可信模型；不操作硬件。'),
                  wraplength=690,foreground='#8b4513').grid(row=6,column=0,sticky='ew',padx=4,pady=2)
        self.restore(host.settings.get('replay_backtest',{}))
        for key,v in self.vars.items(): v.trace_add('write',lambda *_,k=key:self.changed(k))
        self.poll_id=self.after(100,self.poll)

    def _field(self,parent,key,label,row,col,values=None,width=12):
        ttk.Label(parent,text=label).grid(row=row,column=col,sticky='w',padx=3,pady=2)
        w=(ttk.Combobox(parent,textvariable=self.vars[key],values=values,state='readonly',width=width)
           if values is not None else ttk.Entry(parent,textvariable=self.vars[key],width=width))
        w.grid(row=row,column=col+1,sticky='ew',padx=3,pady=2);self.widgets[key]=w

    def snapshot(self):
        view=self.host.replay_prediction_3d
        return {**{k:v.get() for k,v in self.vars.items()},'source_policy_version':24,'custom_values':dict(self.custom_values),
                'view':{k:getattr(view,k) for k in ('yaw_deg','pitch_deg','zoom','pan_x','pan_y')}}

    def restore(self,cfg):
        self.mutating=True
        try:
            for k,v in self.vars.items(): v.set(str(cfg.get(k,DEFAULTS[k])))
            if 'edge_guard_ms' not in cfg and 'onset_guard_ms' in cfg:
                self.vars['edge_guard_ms'].set(str(cfg['onset_guard_ms']))
            self.vars['preset'].set(normalize_preset_name(self.vars['preset'].get()))
            if cfg.get('source_policy_version')!=24 or self.vars['source'].get() not in (AUTO,HISTORY,RECALC):
                self.vars['source'].set(AUTO)
            if self.vars['mode'].get() not in (FIT,NN): self.vars['mode'].set(FIT)
            if self.vars['preset'].get() not in (*self.available_presets,CUSTOM):
                fallback=(DEFAULTS['preset'] if DEFAULTS['preset'] in self.available_presets
                          else next(iter(self.available_presets),CUSTOM))
                self.vars['preset'].set(fallback)
            self.custom_values.update(cfg.get('custom_values',{}))
            self.apply_preset()
            if 'contact_threshold' in cfg:
                self.vars['contact_threshold'].set(str(cfg['contact_threshold']))
            view=cfg.get('view',{})
            if view:self.host.replay_prediction_3d.set_view(*(view.get(k,d) for k,d in
                [('yaw_deg',-45),('pitch_deg',28),('zoom',1),('pan_x',0),('pan_y',0)]))
        finally:self.mutating=False
        self.update_states()
        self.invalidate()

    def apply_preset(self):
        name=self.vars['preset'].get()
        if name==CUSTOM:
            for k,v in self.custom_values.items(): self.vars[k].set(str(v))
        else:
            try:
                p=load_profile(self.profile_path,self.available_presets[name])
                for k,v in p['defaults'].items():
                    if k in self.vars:self.vars[k].set(str(v))
                if p.get('recommended_replay_baseline_policy'):
                    self.vars['baseline_policy'].set(str(p['recommended_replay_baseline_policy']))
                self.vars['contact_threshold'].set(str(
                    p['defaults'].get('contact_threshold',p['defaults'].get('threshold',0))))
            except Exception as e:self.status.set(f'磁片参数读取失败：{e}')

    def changed(self,key):
        if self.mutating:return
        self.mutating=True
        try:
            if key=='preset': self.apply_preset()
            elif key in self.custom_values and self.vars['preset'].get()==CUSTOM:
                self.custom_values[key]=self.vars[key].get()
        finally:self.mutating=False
        if key=='edge_guard_ms':
            self.update_states();self.set_scales()
            if self.host.replay_samples:self.show_index(int(self.host.replay_slider_var.get()))
            self.host._schedule_settings_autosave();return
        self.invalidate();self.update_states()
        if self.historical() and self.host.replay_samples:
            self.show_index(int(self.host.replay_slider_var.get()))
        self.host._schedule_settings_autosave()

    def update_states(self):
        historical=self.historical()
        fit=self.vars['mode'].get()==FIT and not historical
        custom=fit and self.vars['preset'].get()==CUSTOM
        nn=self.vars['mode'].get()==NN and not historical
        enabled=dict(preset=fit,custom_path=custom,threshold=custom,contact_threshold=fit,
                     min_cosine=custom,smoothing_ms=custom,
                     source=True,checkpoint=nn,family=nn,device=nn,mapping=nn,
                     baseline_start=fit or nn,baseline_seconds=fit or nn,
                      baseline_policy=fit,jump_policy=fit,visual=fit or nn,edge_guard_ms=True,
                     mode=not historical)
        for k,w in self.widgets.items():
            w.configure(state=('readonly' if isinstance(w,ttk.Combobox) else 'normal') if enabled[k] else 'disabled')
        self.custom_browse.configure(state='normal' if custom else 'disabled')
        self.profile_export.configure(state='normal' if fit else 'disabled')
        self.model_browse.configure(state='normal' if nn else 'disabled')
        busy=self.worker is not None and self.worker.is_alive()
        self.run_button.configure(state='disabled' if busy else 'normal')
        self.run_button.configure(text='显示已有预测记录' if historical else '应用设置并回测')
        self.cancel_button.configure(state='normal' if busy else 'disabled')
        self.export_button.configure(state='normal' if (self.recorded_results if historical else self.results) else 'disabled')

    def clear_display(self,message):
        self.host.replay_prediction_3d.clear();self.plot.clear()
        self.host.replay_prediction_info_var.set(message)

    def invalidate(self):
        self.generation+=1;self.cancel_event.set();self.results=None;self.result_config=None;self.play_after=False
        self.host.replay_pause()
        self.clear_display('设置或Trial已改变；等待所选方法重新计算。不会回退到其他模式的旧结果。')

    def on_trial_loaded(self):
        self.invalidate()
        self.recorded_results=[history_record(t,row,xyz) for t,row,xyz in self.host.replay_predictions]
        families=sorted({r['family'] for r in self.recorded_results})
        count=len(self.recorded_results)
        contact=sum(bool(r['extra'].get('display.contact')) for r in self.recorded_results)
        self.status.set(f'已扫描原预测记录：{count} 行，接触显示 {contact} 行；来源={" / ".join(families)}。'
                        +('当前直接回放原记录，不重算。' if self.historical() else '当前选择原始磁场重算。') if count
                        else '未发现可读取的 live_predictions.csv 记录；不会把回测结果冒充当时推理。可主动选择算法重算。')
        self.update_states();self.set_scales()

    def browse_profile(self):
        p=filedialog.askopenfilename(parent=self,filetypes=[('标定JSON','*.json')])
        if not p:return
        try:profile=load_profile(p)
        except Exception as e:messagebox.showerror('标定文件',str(e),parent=self);return
        self.vars['custom_path'].set(p)
        for k,v in profile.get('defaults',{}).items():
            if k in self.custom_values:self.vars[k].set(str(v))

    def browse_model(self):
        p=filedialog.askopenfilename(parent=self,filetypes=[('可信模型','*.pt *.pth'),('全部','*')])
        if p:
            self.vars['checkpoint'].set(p)
            self.vars['family'].set('auto')

    def model_option_selected(self,_event=None):
        selected=self.vars['checkpoint'].get().strip()
        if selected in LIVE_NEURAL_MODEL_OPTIONS:
            self.vars['family'].set('FinalTip-R56')

    def get_profile(self):
        name=self.vars['preset'].get()
        return (load_profile(self.vars['custom_path'].get()) if name==CUSTOM
                else load_profile(self.profile_path,self.available_presets[name]))

    def export_profile(self):
        try:p=dict(self.get_profile(),schema=SCHEMA)
        except Exception as e:messagebox.showerror('标定',str(e),parent=self);return
        path=filedialog.asksaveasfilename(parent=self,defaultextension='.json',initialfile='custom_replay_calibration.json')
        if path:
            try:
                if Path(path).resolve()==self.profile_path.resolve():raise ValueError('不能覆盖内置标定库')
                Path(path).write_text(json.dumps(p,ensure_ascii=False,indent=2),encoding='utf-8')
            except Exception as e:messagebox.showerror('导出',str(e),parent=self)

    def historical(self):
        source=self.vars['source'].get()
        return source==HISTORY or (source==AUTO and bool(getattr(self.host,'replay_predictions',[])))

    def ensure_ready_for_play(self):
        if self.historical():return True
        if self.results is not None:return True
        if self.worker is not None and self.worker.is_alive():
            self.play_after=True;return False
        self.start(play_after=True);return False

    def start(self,play_after=False):
        if self.worker is not None and self.worker.is_alive():return
        self.host.replay_pause()
        if self.historical():
            self.status.set('历史记录模式：只回放原 live_predictions.csv，不重算。')
            self.show_index(int(self.host.replay_slider_var.get()));return
        try:
            if not self.host.replay_samples:raise ValueError('请先选择并载入一个Trial')
            cfg={k:v.get() for k,v in self.vars.items()}
            for k,lo,hi in [('baseline_start',0,86400),('baseline_seconds',.2,60)]:number(cfg[k],k,lo,hi)
            number(cfg['edge_guard_ms'],'升/降沿轨迹抑制 ms',0,2000)
            profile=self.get_profile() if cfg['mode']==FIT else None
            if cfg['mode']==NN:
                selected=cfg['checkpoint']
                resolved=resolve_neural_model_selection(selected)
                if not resolved.is_file():raise ValueError('请指定现有可信模型 .pt 文件')
                cfg['checkpoint']=str(resolved)
                cfg['family']=neural_model_family(selected,cfg['family'])
                mapping=json.loads(cfg['mapping'])
                if not isinstance(mapping,dict):raise ValueError('输出映射必须是JSON对象')
            samples=self.host.replay_samples;times=self.host.replay_times;stages=self.host.replay_stages
            td=self.host.replay_trial_dir
        except Exception as e:messagebox.showerror('回测设置',str(e),parent=self);return
        self.invalidate();self.play_after=play_after;self.cancel_event=threading.Event()
        generation=self.generation;stop=self.cancel_event
        self.status.set('按时间顺序重算中（后台单任务，原始文件只读）…')
        def work():
            try:
                rr=run_backtest(samples,times,stages,td,cfg,profile,stop,
                    lambda n,total:self.mail.put((generation,'progress',(n,total))))
                self.mail.put((generation,'done',(rr,cfg)))
            except Exception as e:self.mail.put((generation,'error',str(e)))
        self.worker=threading.Thread(target=work,name='offline-replay-backtest',daemon=True)
        self.worker.start();self.update_states()

    def cancel(self):
        self.invalidate();self.status.set('已请求取消；等待当前计算退出。');self.update_states()

    def poll(self):
        if self.closed:return
        try:
            while True:
                generation,kind,payload=self.mail.get_nowait()
                if generation!=self.generation:continue
                if kind=='progress':self.status.set(f'后台回测：{payload[0]}/{payload[1]} 帧')
                elif kind=='error':
                    self.status.set(f'回测失败：{payload}');self.clear_display(f'回测失败：{payload}');self.play_after=False
                else:
                    self.results,self.result_config=payload
                    valid=sum(r['valid'] for r in self.results)
                    self.status.set(f'{self.result_config["mode"]} 完成：{valid}/{len(self.results)} 帧有效；可播放、拖动回看、导出。')
                    self.set_scales()
                    self.show_index(int(self.host.replay_slider_var.get()))
                    if self.play_after:
                        self.play_after=False;self.host.replay_play()
        except queue.Empty:pass
        if self.play_after and self.results is None and self.cancel_event.is_set() and not (self.worker and self.worker.is_alive()):
            self.start(play_after=True)
        self.update_states();self.poll_id=self.after(100,self.poll)

    def set_scales(self):
        rows=self.recorded_results if self.historical() else self.results
        values=[r['xyz'] for r in rows or [] if display_point_visible(r)[0]]
        if values:
            scale=[max(.1,max(abs(v[a]) for v in values)*1.10) for a in range(3)]
            self.host.replay_prediction_3d.set_scale(scale);self.plot.set_scale(scale)

    def edge_guard_ms(self):
        try:return number(self.vars['edge_guard_ms'].get(),'升/降沿轨迹抑制 ms',0,2000)
        except (TypeError,ValueError):return float(DEFAULTS['edge_guard_ms'])

    def show_index(self,index):
        edge_guard_ms=self.edge_guard_ms()
        history_points=self.host.prediction_trajectory_history_points()
        if self.historical():
            times=self.host.replay_times
            if not times:return
            t=times[max(0,min(len(times)-1,index))]
            k=bisect.bisect_right(self.host.replay_prediction_times,t)-1
            if k<0:self.clear_display('历史预测记录：当前时刻没有可用记录。');return
            r=self.recorded_results[k]
            visible=contact_window(
                self.recorded_results[max(0,k-history_points+1):k+1],
                edge_guard_ms=edge_guard_ms)
            if t-r['t_ns']>1_000_000_000:
                if visible and visible[-1] is not None:visible.append(None)
                r=dict(r,status='预测记录已过期，隐藏当前位置')
            source=HISTORY
        else:
            if not self.results or self.result_config is None:self.clear_display('尚未重算：确认设置后点击“应用设置并回测”。视频/磁场可独立播放。');return
            index=max(0,min(len(self.results)-1,index));r=self.results[index]
            visible=contact_window(
                self.results[max(0,index-history_points+1):index+1],
                edge_guard_ms=edge_guard_ms)
            source=self.result_config['mode']+' / 原始磁场重算'
        point_visible,display_reason=display_point_visible(r)
        if point_visible and not (visible and visible[-1] is not None):
            point_visible=False
            display_reason=f'显示抑制：接触边沿保护 {edge_guard_ms:g} ms（原始结果未修改）'
        if not point_visible:
            r=dict(r,status=str(r['status'])+' | '+display_reason)
        self.host.replay_prediction_3d.set_window(visible)
        # Numeric curves use only the latest contiguous contact segment.
        split=max((i for i,v in enumerate(visible) if v is None),default=-1)
        self.plot.values.clear();self.plot.values.extend(visible[split+1:]);self.plot.redraw()
        raw='  '.join(f'{k}={self.format_value(r["raw"].get(k))}' for k in RAW_FIELDS)
        extra=dict(r['extra']);extra.update({k:v for k,v in r['raw'].items() if k not in RAW_FIELDS})
        self.host.replay_prediction_info_var.set(f'{source} | {r["family"]} | {r["status"]}\n{raw}\n额外输出：{json.dumps(extra,ensure_ascii=False,default=str)}')

    @staticmethod
    def format_value(value):
        if value is None:return '--'
        if isinstance(value,float):return f'{value:.4f}'
        return json.dumps(value,ensure_ascii=False) if isinstance(value,(list,dict)) else str(value)

    def export_results(self):
        historical=self.historical()
        rows=self.recorded_results if historical else self.results
        if not rows:return
        config={'source':HISTORY,'display_policy':'R39 contact + bidirectional-edge display guard; raw fields unchanged',
                'edge_guard_ms':self.edge_guard_ms()} if historical else self.result_config
        trial=Path(self.host.replay_trial_dir).resolve()
        algorithm='recorded' if historical else ('fit' if (config or {}).get('mode')==FIT else 'neural')
        filename=f'replay_backtest_R39_{algorithm}_{datetime.now():%Y%m%d_%H%M%S_%f}.csv'
        path=filedialog.asksaveasfilename(parent=self,title='导出回测 CSV（默认当前 Trial）',
            defaultextension='.csv',filetypes=[('CSV','*.csv')],initialdir=str(trial),initialfile=filename)
        if not path:return
        try:
            dest=Path(path).resolve()
            # Derived results may live beside the original record, never replace it.
            protected={'magnetic_raw.csv','live_predictions.csv','events.csv','stage_commands.csv'}
            if dest.suffix.lower()!='.csv' or dest.name.lower() in protected or dest.name.lower().startswith('camera_'):
                raise ValueError('请使用新的回测结果 CSV 文件名；不能覆盖原始磁场、预测、事件或视频时间戳文件')
            fields=['t_ns','valid','display_contact','status','family',*RAW_FIELDS,'visual_x_mm','visual_y_mm','visual_z_mm','extra_json','settings_json']
            with dest.open('w',encoding='utf-8-sig',newline='') as f:
                w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
                for r in rows:
                    extra=dict(r['extra'],**{k:v for k,v in r['raw'].items() if k not in RAW_FIELDS})
                    values={k:(json.dumps(r['raw'].get(k),ensure_ascii=False) if isinstance(r['raw'].get(k),(dict,list)) else r['raw'].get(k)) for k in RAW_FIELDS}
                    w.writerow(dict(t_ns=r['t_ns'],valid=r['valid'],display_contact=r['extra'].get('display.contact'),status=r['status'],family=r['family'],**values,
                        **dict(zip(('visual_x_mm','visual_y_mm','visual_z_mm'),r['xyz'] or [None]*3)),
                        extra_json=json.dumps(extra,ensure_ascii=False,default=str),settings_json=json.dumps(config,ensure_ascii=False)))
            self.status.set(f'回测已导出：{dest}')
        except Exception as e:messagebox.showerror('导出失败',str(e),parent=self)

    def close(self):
        self.closed=True;self.cancel_event.set();self.after_cancel(self.poll_id)
