"""R36 offline causal replay engines. No hardware or Tk dependencies."""
from __future__ import annotations
import bisect
import copy
import json
import math
import re
from collections import deque
from pathlib import Path
from typing import Any
import numpy as np

RAW_FIELDS = ('x','y','z','dx','dy','mode5','Q','X','Y','Z','dX','dY','dZ')
PRESETS = {
    '19C｜九点离散':'19C_9pt_discrete',
    '19C｜九点连续（探索）':'19C_9pt_continuous',
    '19C｜121点正式离散':'19C_121pt_discrete',
    '19C｜121点正式连续':'19C',
    '19C｜121点正式混合':'19C_121pt_hybrid',
    '19C｜历史工作区0.5/0.8静态':'19C_working_0p5_0p8',
    '19C｜历史工作区0.5/0.8动态':'19C_working_dynamic_0p5_0p8',
    '19C｜441点1mm离散':'19C_441pt_1mm_discrete',
    '19C｜441点1mm连续':'19C_441pt_1mm_continuous',
    '19C｜441点1mm混合':'19C_441pt_1mm_hybrid',
    '19C｜同源1mm→9点离散':'19C_ss1mm_9pt_discrete',
    '19C｜同源1mm→9点连续':'19C_ss1mm_9pt_continuous',
    '19C｜同源1mm→121点离散':'19C_ss1mm_121pt_discrete',
    '19C｜同源1mm→121点连续':'19C_ss1mm_121pt_continuous',
    '19C｜同源1mm→121点混合':'19C_ss1mm_121pt_hybrid',
    '19C｜同源1mm→441点离散':'19C_ss1mm_441pt_discrete',
    '19C｜同源1mm→441点连续':'19C_ss1mm_441pt_continuous',
    '19C｜同源1mm→441点混合':'19C_ss1mm_441pt_hybrid',
    '19C｜同源1mm→工作区0.4–0.6':'19C_ss1mm_working_0p4_0p6',
    '5×5｜九点离散':'5x5_9pt_discrete',
    '5×5｜九点连续（探索）':'5x5_9pt_continuous',
    '5×5｜121点正式离散':'5x5_121pt_discrete',
    '5×5｜121点正式连续':'5x5',
    '5×5｜121点0.4–3mm连续（新磁体五面固定）':'5x5_depth_sweep_121pt_continuous',
    '5×5｜121点连续（更换触头后 silicone_v4）':'5x5_final_tip_depth_sweep_121pt_continuous',
    '5×5｜had-239最终触头｜静态全量R50':'5x5_had239_final_tip_static_r50',
    '5×5｜had-Yelow｜静态全量R50':'5x5_had_yelow_static_r50',
    '5×5｜更换触头后归一拟合＋因果动态岭回归（实验）':'5x5_final_tip_depth_sweep_121pt_dynamic_ridge',
    '5×5｜最终触头论文式归一拟合＋因果动态（全量R49）':'5x5_final_tip_paper_causal_full_r49',
    '5×5｜最终触头全量归一拟合＋论文因果动态（R49推荐）':'5x5_final_tip_selected_static_paper_causal_full_r49',
    '5×5｜121点正式混合':'5x5_121pt_hybrid',
    '5×5｜历史工作区0.5/0.8静态':'5x5_working_0p5_0p8',
    '5×5｜历史工作区0.5/0.8动态':'5x5_working_dynamic_0p5_0p8',
    '5×5｜441点1mm离散':'5x5_441pt_1mm_discrete',
    '5×5｜441点1mm连续':'5x5_441pt_1mm_continuous',
    '5×5｜441点1mm混合':'5x5_441pt_1mm_hybrid',
    '5×5｜同源1mm→9点离散':'5x5_ss1mm_9pt_discrete',
    '5×5｜同源1mm→9点连续':'5x5_ss1mm_9pt_continuous',
    '5×5｜同源1mm→121点离散':'5x5_ss1mm_121pt_discrete',
    '5×5｜同源1mm→121点连续':'5x5_ss1mm_121pt_continuous',
    '5×5｜同源1mm→121点混合':'5x5_ss1mm_121pt_hybrid',
    '5×5｜同源1mm→441点离散':'5x5_ss1mm_441pt_discrete',
    '5×5｜同源1mm→441点连续':'5x5_ss1mm_441pt_continuous',
    '5×5｜同源1mm→441点混合':'5x5_ss1mm_441pt_hybrid',
    '5×5｜同源1mm→工作区0.4–0.6':'5x5_ss1mm_working_0p4_0p6',
    'flot1650-4.5mm｜九点离散':'flot_4p5_9pt_discrete',
    'flot1650-4.5mm｜九点连续（探索）':'flot_4p5_9pt_continuous',
    'flot1650-4.5mm｜121点正式离散':'flot_4p5_121pt_discrete',
    'flot1650-4.5mm｜121点正式连续':'flot_4p5',
    'flot1650-4.5mm｜121点正式混合':'flot_4p5_121pt_hybrid',
    'flot1650-4.5mm｜历史工作区0.5/0.8静态':'flot_4p5_working_0p5_0p8',
    'flot1650-4.5mm｜历史工作区0.5/0.8动态':'flot_4p5_working_dynamic_0p5_0p8',
    'flot1650-4.5mm｜441点1mm离散':'flot_4p5_441pt_1mm_discrete',
    'flot1650-4.5mm｜441点1mm连续':'flot_4p5_441pt_1mm_continuous',
    'flot1650-4.5mm｜441点1mm混合':'flot_4p5_441pt_1mm_hybrid',
    'flot1650-2.5mm｜九点离散':'flot_2p5_9pt_discrete',
    'flot1650-2.5mm｜九点连续（探索）':'flot_2p5_9pt_continuous',
    'flot1650-2.5mm｜121点正式离散':'flot_2p5_121pt_discrete',
    'flot1650-2.5mm｜121点正式连续':'flot_2p5',
    'flot1650-2.5mm｜121点正式混合':'flot_2p5_121pt_hybrid',
    'flot1650-2.5mm｜历史工作区0.5/0.8静态':'flot_2p5_working_0p5_0p8',
    'flot1650-2.5mm｜历史工作区0.5/0.8动态':'flot_2p5_working_dynamic_0p5_0p8',
    'flot1650-2.5mm｜441点1mm离散':'flot_2p5_441pt_1mm_discrete',
    'flot1650-2.5mm｜441点1mm连续':'flot_2p5_441pt_1mm_continuous',
    'flot1650-2.5mm｜441点1mm混合':'flot_2p5_441pt_1mm_hybrid',
}
LEGACY_PRESET_ALIASES = {
    '19C':'19C｜九点离散', '5×5':'5×5｜九点离散',
    'flot1650-4.5mm':'flot1650-4.5mm｜九点离散',
    'flot1650-2.5mm':'flot1650-2.5mm｜121点正式连续',
}
DEFAULT_PRESET = 'flot1650-4.5mm｜九点离散'
CUSTOM = '自由设置 / Custom'
SCHEMA = 'MagTouch-NormalizedFit-1'
JUMP_FAST = '快速门控（阈值/形状）'
JUMP_HYSTERESIS = '迟滞门控（3帧释放锁定）'
JUMP_POLICIES = (JUMP_FAST, JUMP_HYSTERESIS)

def normalize_preset_name(value):
    name=str(value or '')
    return LEGACY_PRESET_ALIASES.get(name,name)

def available_presets(path):
    """Return only calibration options that are actually installed.

    R39 declares the 441-point option for every magnetic layer, but it must not
    silently fall back to a 121-point profile while a layer is still awaiting
    its own 1 mm acquisition.  The UI therefore exposes a 441-point option only
    after the builder has written that exact profile key into the calibration
    library.
    """
    try:
        document=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except Exception:
        return dict(PRESETS)
    installed=set(document.get('profiles',{})) | set(document.get('variants',{}))
    return {label:key for label,key in PRESETS.items() if key in installed}

def number(value, name, low, high):
    v=float(value)
    if not math.isfinite(v) or not low <= v <= high:
        raise ValueError(f'{name}须为{low}～{high}之间的有限数值')
    return v

def load_profile(path, key=None):
    doc=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if doc.get('schema') != SCHEMA:
        raise ValueError(f'标定JSON必须使用 schema={SCHEMA}')
    if key is not None:
        profiles=doc.get('profiles',{})
        selected=profiles.get(key)
        if selected is None:
            spec=doc.get('variants',{}).get(key)
            if not isinstance(spec,dict): raise ValueError(f'缺少磁片参数或算法变体 {key}')
            base=profiles.get(spec.get('base_profile'))
            if not isinstance(base,dict): raise ValueError(f"算法变体 {key} 的 base_profile 不存在")
            selected=copy.deepcopy(base)
            filt=spec.get('sample_filter') or {}
            samples=list(selected.get('samples') or [])
            def allowed(sample):
                checks=(('x','x_values'),('y','y_values'),('depth','depths'))
                return all(not filt.get(target) or any(
                    math.isclose(float(sample[source]),float(value),abs_tol=1e-9)
                    for value in filt[target]) for source,target in checks)
            selected['samples']=[sample for sample in samples if allowed(sample)]
            for field in spec.get('remove_fields') or []: selected.pop(str(field),None)
            selected.update(copy.deepcopy(spec.get('overrides') or {}))
            selected['variant_key']=key
            selected['variant_label']=spec.get('label',key)
        doc=selected
        if not isinstance(doc,dict): raise ValueError(f'缺少磁片参数 {key}')
    if not isinstance(doc.get('samples'),list) or len(doc['samples']) < 4:
        raise ValueError('标定至少需要4条samples，每个位置至少两个不同深度')
    for s in doc['samples']:
        if np.asarray(s['delta'],dtype=float).shape != (15,) or not np.isfinite(s['delta']).all():
            raise ValueError('每条delta必须包含15个有限数，按S1..S5的Bx/By/Bz排列，单位uT')
        for k in ('x','y','depth'):
            if not math.isfinite(float(s[k])): raise ValueError(f'非法标定字段{k}')
    xy_model=doc.get('continuous_xy_model')
    if xy_model is not None:
        method=xy_model.get('method')
        if method=='kernel_rbf':
            centers=np.asarray(xy_model.get('centers'),dtype=float)
            alpha=np.asarray(xy_model.get('alpha'),dtype=float)
            if centers.ndim!=2 or centers.shape[1]!=15:
                raise ValueError('kernel_rbf continuous_xy_model 必须为15维')
            if alpha.shape!=(len(centers),2) or not np.isfinite(centers).all() or not np.isfinite(alpha).all():
                raise ValueError('continuous_xy_model centers/alpha 维度或数值非法')
        elif method=='ridge_xy_unit_logamp':
            mean=np.asarray(xy_model.get('mean'),dtype=float)
            scale=np.asarray(xy_model.get('scale'),dtype=float)
            coef=np.asarray(xy_model.get('coef'),dtype=float)
            if xy_model.get('feature')!='unit_logamp' or mean.shape!=(32,) or scale.shape!=(32,):
                raise ValueError('ridge_xy_unit_logamp 必须包含32维 unit_logamp mean/scale')
            if coef.shape!=(33,2) or not all(np.isfinite(v).all() for v in (mean,scale,coef)):
                raise ValueError('ridge_xy_unit_logamp coef 必须为33×2有限矩阵')
        else:
            raise ValueError(f'不支持的 continuous_xy_model 方法：{method}')
    for name in ('continuous_z_model','contact_classifier'):
        model=doc.get(name)
        if model is None: continue
        mean=np.asarray(model.get('mean'),dtype=float)
        scale=np.asarray(model.get('scale'),dtype=float)
        coef=np.asarray(model.get('coef'),dtype=float)
        if model.get('method')!='ridge' or mean.ndim!=1 or scale.shape!=mean.shape:
            raise ValueError(f'{name} 必须为带 mean/scale 的 ridge')
        if coef.shape not in ((len(mean)+1,), (len(mean)+1,1)) or not all(
                np.isfinite(v).all() for v in (mean,scale,coef)):
            raise ValueError(f'{name} coef 维度或数值非法')
    hybrid=doc.get('hybrid_model')
    if hybrid is not None:
        if hybrid.get('method')!='topk_node_softmax_local_z':
            raise ValueError('hybrid_model方法必须为topk_node_softmax_local_z')
        node_xy=np.asarray(hybrid.get('node_xy'),dtype=float)
        node_z=np.asarray(hybrid.get('node_z_calibration'),dtype=float)
        banks=hybrid.get('node_templates')
        top_k=int(hybrid.get('top_k',0)); temperature=float(hybrid.get('temperature',0))
        if node_xy.ndim!=2 or node_xy.shape[1:]!=(2,) or node_z.shape!=(len(node_xy),2):
            raise ValueError('hybrid_model节点XY或局部Z参数维度非法')
        if not isinstance(banks,list) or len(banks)!=len(node_xy) or not 1<=top_k<=len(node_xy):
            raise ValueError('hybrid_model节点模板或top_k非法')
        if not math.isfinite(temperature) or temperature<=0:
            raise ValueError('hybrid_model temperature必须为正数')
        for bank in banks:
            array=np.asarray(bank,dtype=float)
            if array.ndim!=2 or array.shape[1]!=15 or not np.isfinite(array).all():
                raise ValueError('hybrid_model每个节点模板必须为N×15有限矩阵')
        if not np.isfinite(node_xy).all() or not np.isfinite(node_z).all() or np.any(node_z[:,0]<=1e-9):
            raise ValueError('hybrid_model节点或局部Z斜率非法')
    for name,outputs in (('working_range_xy_model',2),('working_range_z_model',1)):
        model=doc.get(name)
        if model is None: continue
        if model.get('method')!='kernel_ridge_fused':
            raise ValueError(f'{name} 方法必须为 kernel_ridge_fused')
        feature=str(model.get('feature',''))
        width={'shape':15,'fixed_fusion':32,'raw':15}.get(feature)
        mean=np.asarray(model.get('mean'),dtype=float)
        scale=np.asarray(model.get('scale'),dtype=float)
        centers=np.asarray(model.get('centers'),dtype=float)
        coef=np.asarray(model.get('coef'),dtype=float)
        output_mean=np.asarray(model.get('output_mean'),dtype=float).reshape(-1)
        if width is None or mean.shape!=(width,) or scale.shape!=(width,):
            raise ValueError(f'{name} 的 feature/mean/scale 维度非法')
        if centers.ndim!=2 or centers.shape[1]!=width or coef.shape!=(len(centers),outputs):
            raise ValueError(f'{name} 的 centers/coef 维度非法')
        if output_mean.shape!=(outputs,) or not all(np.isfinite(v).all() for v in (
                mean,scale,centers,coef,output_mean)) or np.any(scale<=0):
            raise ValueError(f'{name} 包含非法数值')
        for scalar in ('gamma','bandwidth','lambda'):
            if not math.isfinite(float(model.get(scalar,float('nan')))) or float(model[scalar])<=0:
                raise ValueError(f'{name}.{scalar} 必须为正数')
    dynamic=doc.get('dynamic_model')
    if dynamic is not None:
        mean=np.asarray(dynamic.get('mean'),float); scale=np.asarray(dynamic.get('scale'),float)
        coef=np.asarray(dynamic.get('coef'),float); lags=np.asarray(dynamic.get('lags_s'),float)
        if dynamic.get('method')!='causal_gated_history_ridge' or mean.ndim!=1:
            raise ValueError('dynamic_model 必须为 causal_gated_history_ridge')
        if scale.shape!=mean.shape or coef.shape!=(len(mean)+1,3) or lags.ndim!=1 or len(lags)<1:
            raise ValueError('dynamic_model 的 mean/scale/coef/lags_s 维度非法')
        if len(mean)!=34*(len(lags)+1) or np.any(lags<=0) or np.any(scale<=0) or not all(
                np.isfinite(v).all() for v in (mean,scale,coef,lags)):
            raise ValueError('dynamic_model 包含非法维度或数值')
        if float(dynamic.get('memory_tau_s',0))<=0 or float(dynamic.get('change_threshold',0))<=0:
            raise ValueError('dynamic_model 的 memory_tau_s/change_threshold 必须为正数')
    state=doc.get('dynamic_state_model')
    if state is not None:
        mean=np.asarray(state.get('mean'),float); scale=np.asarray(state.get('scale'),float)
        coef=np.asarray(state.get('coef'),float); intercept=np.asarray(state.get('intercept'),float)
        transition=np.asarray(state.get('transition'),float); names=list(state.get('state_names') or [])
        if state.get('method')!='causal_lda_markov' or len(names)<2:
            raise ValueError('dynamic_state_model 必须为 causal_lda_markov 且含状态名')
        if scale.shape!=mean.shape or coef.shape!=(len(mean),len(names)) or intercept.shape!=(len(names),):
            raise ValueError('dynamic_state_model 的回归参数维度非法')
        if transition.shape!=(len(names),len(names)) or np.any(scale<=0) or not all(
                np.isfinite(v).all() for v in (mean,scale,coef,intercept,transition)):
            raise ValueError('dynamic_state_model 包含非法维度或数值')
    return doc


def _feature_vector(kind, unit, amp):
    loga=math.log(max(float(amp),1e-6))
    if kind=='unit_logamp': return np.r_[unit,loga,loga*loga,unit*loga]
    if kind=='delta': return np.r_[unit*amp,loga]
    if kind=='amplitude_poly': return np.array([loga,loga*loga],float)
    if kind=='delta_logamp': return np.r_[unit*amp,loga,loga*loga]
    if kind=='unit_amp': return np.r_[unit,amp,math.sqrt(max(float(amp),0.0))]
    raise ValueError(f'不支持的特征 {kind}')


def _ridge_value(model, feature):
    mean=np.asarray(model['mean'],float); scale=np.asarray(model['scale'],float)
    coef=np.asarray(model['coef'],float).reshape(-1)
    value=np.r_[(np.asarray(feature,float)-mean)/scale,1.0]@coef
    return float(value)

def _ridge_xy_value(model, feature):
    mean=np.asarray(model['mean'],float); scale=np.asarray(model['scale'],float)
    coef=np.asarray(model['coef'],float)
    value=np.r_[(np.asarray(feature,float)-mean)/scale,1.0]@coef
    if np.asarray(value).shape!=(2,) or not np.isfinite(value).all():
        raise ValueError('连续XY岭回归输出非法')
    return np.asarray(value,float)

def _working_feature(kind, delta, unit, amp):
    """Feature family used by the R40 frozen working-range KRR models."""
    if kind=='shape': return np.asarray(unit,float)
    if kind=='raw': return np.asarray(delta,float)
    if kind=='fixed_fusion':
        loga=math.log(max(float(amp),1e-8))
        return np.r_[unit,loga,loga*loga,unit*loga]
    raise ValueError(f'不支持的R40工作区特征：{kind}')

def _working_krr_value(model, delta, unit, amp):
    feature=_working_feature(str(model['feature']),delta,unit,amp)
    mean=np.asarray(model['mean'],float); scale=np.asarray(model['scale'],float)
    query=(feature-mean)/scale; centers=np.asarray(model['centers'],float)
    sq=np.sum((centers-query[None,:])**2,axis=1)
    kernel=np.exp(-float(model['gamma'])*sq/float(model['bandwidth']))
    value=kernel@np.asarray(model['coef'],float)+np.asarray(model['output_mean'],float)
    return np.asarray(value,float).reshape(-1)

def _softmax(value):
    value=np.asarray(value,float); e=np.exp(value-np.max(value))
    return e/max(float(np.sum(e)),1e-12)

class NormalizedFit:
    def __init__(self, profile, trial_dir, threshold, min_cosine, smoothing_ms,
                 contact_threshold=None, jump_policy=JUMP_HYSTERESIS):
        self.threshold=number(threshold,'响应阈值 uT',0,1e6)
        self.contact_threshold=number(
            threshold if contact_threshold in (None,'') else contact_threshold,
            '临时接触阈值 uT',0,1e6)
        self.min_cosine=number(min_cosine,'最低相似度',-1,1)
        self.window_ns=int(number(smoothing_ms,'平滑窗口 ms',0,5000)*1e6)
        self.buffer=deque()
        self.b0=None
        self.anchor=None
        self.continuous_xy=profile.get('continuous_xy_model')
        self.continuous_z=profile.get('continuous_z_model')
        self.working_xy=profile.get('working_range_xy_model')
        self.working_z=profile.get('working_range_z_model')
        self.dynamic_model=profile.get('dynamic_model')
        self.dynamic_state_model=profile.get('dynamic_state_model')
        self.hybrid=profile.get('hybrid_model')
        self.contact_classifier=profile.get('contact_classifier')
        self.xy_semantics=profile.get('xy_semantics')
        self.fit_model_description=profile.get('fit_model_description')
        self.fit_variant=profile.get('fit_variant','legacy_discrete')
        self.evidence_tier=profile.get('evidence_tier','unspecified')
        self.calibration_domain=profile.get('calibration_domain_mm') or {}
        self.working_range_metadata=profile.get('working_range_mm') or {}
        self.release_hysteresis=profile.get('release_hysteresis') or {}
        self.jump_policy=str(jump_policy or JUMP_HYSTERESIS)
        if self.jump_policy not in JUMP_POLICIES:
            raise ValueError(f'不支持的抑制跳变方案：{self.jump_policy}')
        self.release_enabled=(self.jump_policy==JUMP_HYSTERESIS and
                              bool(self.release_hysteresis.get('enabled')))
        self.working_range=bool(self.working_xy is not None or self.working_z is not None)
        self.working_range_complete=bool(
            self.working_xy is not None and self.working_z is not None)
        self.continuous=bool(
            (self.working_xy is not None or self.continuous_xy is not None) and
            (self.working_z is not None or self.continuous_z is not None))
        if self.hybrid is not None and self.continuous:
            raise ValueError('单个算法变体不能同时启用continuous与hybrid模型')
        if self.dynamic_model is not None and not self.working_range_complete:
            raise ValueError('动态补偿只能叠加在完整的R40工作区静态模型上')
        # Optional output-coordinate reassembly calibration.  The transform is
        # deliberately downstream of the magnetic model and dynamic correction:
        # it changes neither B0 nor the learned magnetic response.
        self.xy_rotation_rad=0.0
        self.xy_rotation=np.eye(2,dtype=float)
        self.xy_translation=np.zeros(2,float)
        self.xy_translation_source='none'
        self.xy_translation_details={}
        self.dynamic_history=deque()
        self.dynamic_gate=0.0
        self.dynamic_last_t=None
        self.dynamic_state_prob=None
        self.contact_active=False; self.peak_amp=0.0
        self.release_latched=False; self.release_amp=0.0
        self.release_floor_amp=0.0; self.release_latched_since_ns=None
        self.release_rearm_ready=False
        self.release_drop_frames=0
        self.release_peak_history=deque()
        # Match both plan and trial to avoid excluding unrelated, similarly named data.
        td=Path(trial_dir)
        same_plan=td.parent.name == profile.get('source_plan')
        self.samples=[s for s in profile['samples'] if not (same_plan and s.get('trial')==td.name)]
        self.excluded=len(profile['samples'])-len(self.samples)
        if not self.samples: raise ValueError('排除当前Trial后没有标定样本')
        self.templates=np.array([s['delta'] for s in self.samples],float)
        self.xy=np.array([[s['x'],s['y']] for s in self.samples],float)
        norms=np.linalg.norm(self.templates,axis=1)
        if np.any(norms <= 1e-9): raise ValueError('标定存在零幅度模板')
        self.templates=self.templates/norms[:,None]
        if self.hybrid is not None:
            self.continuous_xy_method='topk_node_softmax_local_z'
            self.hybrid_node_xy=np.asarray(self.hybrid['node_xy'],float)
            self.hybrid_node_templates=[np.asarray(bank,float) for bank in self.hybrid['node_templates']]
            self.hybrid_node_z=np.asarray(self.hybrid['node_z_calibration'],float)
            self.hybrid_top_k=int(self.hybrid['top_k'])
            self.hybrid_temperature=float(self.hybrid['temperature'])
            self.confidence_templates=np.vstack(self.hybrid_node_templates)
        elif self.working_xy is not None:
            self.continuous_xy_method='kernel_ridge_fused_'+str(self.working_xy.get('feature'))
            self.confidence_templates=self.templates
        elif self.continuous:
            self.continuous_xy_method=str(self.continuous_xy.get('method'))
            if self.continuous_xy_method=='kernel_rbf':
                self.xy_centers=np.asarray(self.continuous_xy['centers'],float)
                self.xy_alpha=np.asarray(self.continuous_xy['alpha'],float)
                self.xy_gamma=float(self.continuous_xy['gamma'])
                self.confidence_templates=self.xy_centers/np.maximum(
                    np.linalg.norm(self.xy_centers,axis=1)[:,None],1e-12)
            elif self.continuous_xy_method=='ridge_xy_unit_logamp':
                self.confidence_templates=self.templates
            else:
                raise ValueError(f'不支持的连续XY方法：{self.continuous_xy_method}')
        else:
            self.continuous_xy_method='discrete_nearest_cosine'
            self.confidence_templates=self.templates
        self.calibration={}
        for p in np.unique(self.xy,axis=0):
            mask=np.all(self.xy==p,axis=1)
            z=np.array([s['depth'] for s,m in zip(self.samples,mask) if m],float)
            if len(np.unique(z)) < 2: raise ValueError(f'位置{p}排除当前Trial后不足两个标定深度')
            a,b=np.linalg.lstsq(np.column_stack((z,np.ones(len(z)))),norms[mask],rcond=None)[0]
            if not np.isfinite(a+b) or a <= 1e-6: raise ValueError(f'位置{p}的深度斜率必须为正')
            self.calibration[tuple(p)]=(float(a),float(b),float(z.min()),float(z.max()))
        self.dynamic_median_amp=max(float(np.median(norms)),1e-8)

    def _reset_dynamic(self):
        self.dynamic_history.clear(); self.dynamic_gate=0.0
        self.dynamic_last_t=None; self.dynamic_state_prob=None

    def set_xy_translation(self,x_mm=0.0,y_mm=0.0,source='manual',details=None):
        self.set_xy_rigid_transform(0.0,x_mm,y_mm,source,details)

    def set_xy_rigid_transform(self,rotation_deg=0.0,x_mm=0.0,y_mm=0.0,
                               source='manual',details=None):
        """Apply a 2-D rigid transform ``R @ xy + t`` to model outputs.

        This is an installation-coordinate correction only.  Scale and shear
        are intentionally unsupported because they would conceal height,
        tilt, preload, or deformation changes that need physical inspection.
        """
        angle=float(rotation_deg)
        offset=np.asarray([x_mm,y_mm],float)
        if not math.isfinite(angle) or abs(angle)>45.0:
            raise ValueError('XY旋转校准必须为±45°内有限数值')
        if offset.shape!=(2,) or not np.isfinite(offset).all() or np.any(np.abs(offset)>20):
            raise ValueError('XY整体平移必须为±20 mm内有限数值')
        radians=math.radians(angle)
        c,s=math.cos(radians),math.sin(radians)
        self.xy_rotation_rad=radians
        self.xy_rotation=np.asarray([[c,-s],[s,c]],dtype=float)
        self.xy_translation=offset
        self.xy_translation_source=str(source or 'manual')
        self.xy_translation_details=dict(details or {})

    def _dynamic_history_vector(self,t,current):
        if self.dynamic_model is None: return None
        lags=[float(v) for v in self.dynamic_model['lags_s']]
        if self.dynamic_last_t is not None and (t<=self.dynamic_last_t or t-self.dynamic_last_t>250_000_000):
            self._reset_dynamic()
        self.dynamic_last_t=int(t); self.dynamic_history.append((int(t),np.asarray(current,float)))
        horizon=max(lags)+.25
        while len(self.dynamic_history)>1 and self.dynamic_history[0][0] < t-int(horizon*1e9):
            self.dynamic_history.popleft()
        times=[row[0] for row in self.dynamic_history]
        values=[row[1] for row in self.dynamic_history]
        parts=[np.asarray(current,float)]
        for lag in lags:
            index=bisect.bisect_right(times,t-int(lag*1e9))-1
            if index<0:return None
            parts.append(np.asarray(current,float)-values[index])
        return np.concatenate(parts)

    def _dynamic_predict(self,t,delta,unit,amp,base_xyz):
        if self.dynamic_model is None:return np.asarray(base_xyz,float),None,0.0,False
        current=np.r_[delta/self.dynamic_median_amp,unit,math.log1p(amp),base_xyz]
        previous_t=self.dynamic_last_t
        history=self._dynamic_history_vector(t,current)
        if history is None:return np.asarray(base_xyz,float),None,0.0,False
        model=self.dynamic_model; lags=[float(v) for v in model['lags_s']]
        width=len(current); change=float(np.linalg.norm(history[width:width+15]))
        instant=float(np.clip(change/float(model['change_threshold'])-1.0,0.0,1.0))
        if previous_t is None:self.dynamic_gate=instant
        else:
            dt=max(0.0,min(.25,(int(t)-int(previous_t))/1e9))
            decay=math.exp(-dt/float(model['memory_tau_s']))
            self.dynamic_gate=max(self.dynamic_gate*decay,instant)
        mean=np.asarray(model['mean'],float); scale=np.asarray(model['scale'],float)
        correction=np.r_[1.0,(history-mean)/scale]@np.asarray(model['coef'],float)
        xyz=np.asarray(base_xyz,float)+correction*self.dynamic_gate
        state=None
        if self.dynamic_state_model is not None:
            energy=[history]
            for index,lag in enumerate(lags):
                d=history[width*(index+1):width*(index+2)]
                energy.append(np.array([np.linalg.norm(d[:15]),np.linalg.norm(d[15:30]),
                    abs(d[30]),np.linalg.norm(d[31:33]),abs(d[33])],float)/lag)
            q=np.concatenate(energy); sm=self.dynamic_state_model
            logits=(q-np.asarray(sm['mean'],float))/np.asarray(sm['scale'],float)@np.asarray(sm['coef'],float)+np.asarray(sm['intercept'],float)
            transition=np.asarray(sm['transition'],float)
            prior=(np.full(len(logits),1/len(logits)) if self.dynamic_state_prob is None
                   else self.dynamic_state_prob@transition)
            self.dynamic_state_prob=_softmax(logits+np.log(np.maximum(prior,1e-12)))
            names=list(sm['state_names'])
            # _dynamic_predict is reached only after the independent contact gate
            # accepted this frame.  Enforce that known constraint rather than
            # allowing a weak phase classifier (notably 19C) to contradict it.
            if 'noncontact' in names:
                index=names.index('noncontact');self.dynamic_state_prob[index]=0.0
                self.dynamic_state_prob/=max(float(np.sum(self.dynamic_state_prob)),1e-12)
            k=int(np.argmax(self.dynamic_state_prob))
            state={'label':names[k],'index':k,'probability':float(self.dynamic_state_prob[k]),
                   'probabilities':{name:float(value) for name,value in zip(names,self.dynamic_state_prob)},
                   'contact_constrained':True}
        return xyz,state,self.dynamic_gate,True

    def set_baseline(self,b):
        b=np.asarray(b,float)
        if b.shape != (15,) or not np.isfinite(b).all(): raise ValueError('非法B0')
        self.b0=b.copy(); self.buffer.clear(); self.anchor=None
        self.contact_active=False; self.peak_amp=0.0
        self.release_latched=False; self.release_amp=0.0
        self.release_floor_amp=0.0; self.release_latched_since_ns=None
        self.release_rearm_ready=False
        self.release_drop_frames=0
        self.release_peak_history.clear()
        self._reset_dynamic()

    def push(self,t,b):
        if self.b0 is None: return empty_record(t,'等待B0')
        # Do not let the smoothing window draw a residual tail after lift-off.
        current_amp=float(np.linalg.norm(np.asarray(b,float)-self.b0))
        if self.release_enabled:
            peak_window_ms=float(self.release_hysteresis.get('peak_window_ms',0.0))
            if peak_window_ms>0:
                self.release_peak_history.append((int(t),current_amp))
                cutoff=int(t-peak_window_ms*1e6)
                while (len(self.release_peak_history)>1 and
                       self.release_peak_history[0][0]<cutoff):
                    self.release_peak_history.popleft()
            rise=float(self.release_hysteresis.get('rearm_rise_uT',60.0))
            if self.release_latched:
                self.release_floor_amp=min(self.release_floor_amp,current_amp)
                floor_drop=float(self.release_hysteresis.get(
                    'rearm_floor_drop_uT',max(10.0,0.5*rise)))
                floor_ratio=float(self.release_hysteresis.get(
                    'rearm_floor_ratio',0.55))
                min_latched_ns=int(float(self.release_hysteresis.get(
                    'rearm_min_latched_ms',150.0))*1e6)
                elapsed_ns=(0 if self.release_latched_since_ns is None else
                            max(0,int(t)-int(self.release_latched_since_ns)))
                if (self.release_amp-self.release_floor_amp >= floor_drop and
                        self.release_floor_amp <= self.release_amp*floor_ratio):
                    self.release_rearm_ready=True
                legacy_rearm=False
                floor_rearm=False
                if current_amp < max(self.threshold,self.contact_threshold,1e-9):
                    self.release_latched=False; self.contact_active=False
                    self.buffer.clear(); self.anchor=None; self.peak_amp=0.0
                    self.release_amp=0.0; self.release_floor_amp=0.0
                    self.release_latched_since_ns=None; self.release_rearm_ready=False
                    self.release_drop_frames=0
                else:
                    legacy_rearm=current_amp >= self.release_amp+rise
                    floor_rearm=(self.release_rearm_ready and elapsed_ns >= min_latched_ns and
                                 current_amp >= self.release_floor_amp+rise)
                if self.release_latched and not (legacy_rearm or floor_rearm):
                    self.buffer.clear(); self.anchor=None; self._reset_dynamic()
                    return empty_record(t,'释放锁定 / 等待无载谷值后的明确接触上升',{
                        'amplitude_uT':current_amp,'release_latched':True,
                        'release_amplitude_uT':self.release_amp,
                        'release_floor_amplitude_uT':self.release_floor_amp,
                        'release_rearm_ready':self.release_rearm_ready,
                        'release_legacy_rearm':legacy_rearm,
                        'release_floor_rearm':floor_rearm,
                        'release_latched_ms':elapsed_ns/1e6,
                        'rearm_amplitude_uT':self.release_floor_amp+rise})
                elif self.release_latched:
                    self.release_latched=False; self.contact_active=False
                    self.buffer.clear(); self.anchor=None; self.peak_amp=0.0
                    self.release_amp=0.0; self.release_floor_amp=0.0
                    self.release_latched_since_ns=None; self.release_rearm_ready=False
                    self.release_drop_frames=0
            if self.contact_active:
                ratio=float(self.release_hysteresis.get('drop_ratio',0.65))
                min_drop=float(self.release_hysteresis.get('min_drop_uT',50.0))
                release_peak=(max(value for _,value in self.release_peak_history)
                              if peak_window_ms>0 else self.peak_amp)
                if current_amp < release_peak*ratio and release_peak-current_amp >= min_drop:
                    self.release_drop_frames+=1
                    if self.release_drop_frames >= int(self.release_hysteresis.get('confirm_frames',3)):
                        self.release_latched=True; self.release_amp=current_amp
                        self.release_floor_amp=current_amp
                        self.release_latched_since_ns=int(t)
                        self.release_rearm_ready=False
                        self.contact_active=False; self.buffer.clear(); self.anchor=None
                        self._reset_dynamic()
                        return empty_record(t,'检测到持续释放下降 / 隐藏抬起残余轨迹',{
                            'amplitude_uT':current_amp,'release_latched':True,
                            'peak_amplitude_uT':release_peak,'drop_ratio':current_amp/max(release_peak,1e-9)})
                else:
                    self.release_drop_frames=0
        if current_amp < max(self.threshold,1e-9):
            self.buffer.clear();self.anchor=None;self._reset_dynamic()
            self.contact_active=False; self.peak_amp=0.0
            self.release_drop_frames=0
            self.release_peak_history.clear()
            return empty_record(t,'当前帧低响应 / 无有效接触',{'amplitude_uT':current_amp})
        if current_amp < max(self.contact_threshold,1e-9):
            self.buffer.clear();self.anchor=None;self._reset_dynamic()
            self.contact_active=False; self.peak_amp=0.0
            self.release_drop_frames=0
            return empty_record(t,'低于临时接触阈值 / 判为非接触',{
                'amplitude_uT':current_amp,'contact_threshold_uT':self.contact_threshold,
                'contact_gate':'temporary current-frame amplitude threshold'})
        self.buffer.append((t,np.asarray(b,float)))
        while len(self.buffer)>1 and self.buffer[0][0] < t-self.window_ns: self.buffer.popleft()
        delta=np.median([v for _,v in self.buffer],axis=0)-self.b0
        amp=float(np.linalg.norm(delta))
        if amp < max(self.threshold,1e-9):
            self.anchor=None;self._reset_dynamic()
            return empty_record(t,'低响应 / 无有效接触',{'amplitude_uT':amp})
        if amp < max(self.contact_threshold,1e-9):
            self.anchor=None;self._reset_dynamic()
            return empty_record(t,'平滑响应低于临时接触阈值 / 判为非接触',{
                'amplitude_uT':amp,'contact_threshold_uT':self.contact_threshold,
                'contact_gate':'temporary smoothed-amplitude threshold'})
        unit=delta/amp
        sims=self.confidence_templates@unit
        k=int(np.argmax(sims)); score=float(sims[k])
        if score < self.min_cosine:
            self.anchor=None
            return empty_record(t,'模板相似度不足',{'cosine_similarity':score,'amplitude_uT':amp})
        contact_score=None
        if self.contact_classifier is not None:
            contact_score=_ridge_value(
                self.contact_classifier,
                _feature_vector(self.contact_classifier['feature'],unit,amp))
            score_threshold=float(self.contact_classifier['score_threshold'])
            if contact_score < score_threshold:
                self.anchor=None
                return empty_record(t,'接触判别分数不足 / 判为非接触',{
                    'amplitude_uT':amp,'contact_score':contact_score,
                    'contact_score_threshold':score_threshold,'cosine_similarity':score})
        hybrid_extra=None
        if self.hybrid is not None:
            node_scores=np.asarray([float(np.max(bank@unit)) for bank in self.hybrid_node_templates])
            order=np.argsort(node_scores)[::-1][:self.hybrid_top_k]
            selected=node_scores[order]
            weights=np.exp(np.clip(
                self.hybrid_temperature*(selected-selected[0]),-60.0,0.0))
            weights=weights/np.sum(weights)
            x,y=np.sum(self.hybrid_node_xy[order]*weights[:,None],axis=0).tolist()
            local=self.hybrid_node_z[order]
            a=float(np.sum(local[:,0]*weights)); b=float(np.sum(local[:,1]*weights))
            z=float((amp-b)/a)
            zlo=float(min(s['depth'] for s in self.samples)); zhi=float(max(s['depth'] for s in self.samples))
            score=float(selected[0])
            hybrid_extra={
                'hybrid_top_k':self.hybrid_top_k,
                'hybrid_temperature':self.hybrid_temperature,
                'hybrid_candidate_xy_mm':self.hybrid_node_xy[order].tolist(),
                'hybrid_candidate_scores':selected.tolist(),
                'hybrid_candidate_weights':weights.tolist(),
                'hybrid_top1_margin':float(selected[0]-selected[1]) if len(selected)>1 else None,
                'depth_fit_slope_uT_per_mm':a,
                'depth_fit_intercept_uT':b,
            }
        elif self.working_range:
            if self.working_xy is not None:
                x,y=_working_krr_value(self.working_xy,delta,unit,amp).tolist()
            elif self.continuous_xy_method=='kernel_rbf':
                sq=np.sum((unit[None,:]-self.xy_centers)**2,axis=1)
                x,y=(np.exp(-self.xy_gamma*sq)@self.xy_alpha).tolist()
            else:
                x,y=_ridge_xy_value(
                    self.continuous_xy,
                    _feature_vector(self.continuous_xy.get('feature','unit_logamp'),unit,amp)).tolist()
            if self.working_z is not None:
                z=float(_working_krr_value(self.working_z,delta,unit,amp)[0])
            else:
                z=_ridge_value(self.continuous_z,
                               _feature_vector(self.continuous_z['feature'],unit,amp))
            zlo=float(min(s['depth'] for s in self.samples)); zhi=float(max(s['depth'] for s in self.samples))
            a=b=float('nan')
        elif self.continuous:
            if self.continuous_xy_method=='kernel_rbf':
                sq=np.sum((unit[None,:]-self.xy_centers)**2,axis=1)
                x,y=(np.exp(-self.xy_gamma*sq)@self.xy_alpha).tolist()
            else:
                x,y=_ridge_xy_value(
                    self.continuous_xy,
                    _feature_vector(self.continuous_xy.get('feature','unit_logamp'),unit,amp)).tolist()
            z=_ridge_value(self.continuous_z,
                           _feature_vector(self.continuous_z['feature'],unit,amp))
            zlo=float(min(s['depth'] for s in self.samples)); zhi=float(max(s['depth'] for s in self.samples))
            a=b=float('nan')
        else:
            x,y=self.xy[k]; a,b,zlo,zhi=self.calibration[(x,y)]
            z=float((amp-b)/a)
        if z < 0:
            self.anchor=None
            return empty_record(t,'反算深度小于0，超出有效标定',{'estimated_z':z})
        uncorrected_xyz=np.array([x,y,z],float)
        xyz,dynamic_state,dynamic_gate,dynamic_ready=self._dynamic_predict(
            t,delta,unit,amp,uncorrected_xyz)
        if not np.isfinite(xyz).all() or xyz[2] < 0:
            self.anchor=None
            return empty_record(t,'动态修正后深度小于0或输出非法，超出有效标定',{
                'uncorrected_static_xyz_mm':uncorrected_xyz.tolist(),
                'dynamic_gate':dynamic_gate})
        unaligned_xyz=xyz.copy()
        xyz[:2]=self.xy_rotation@xyz[:2]+self.xy_translation
        x,y,z=xyz.tolist()
        if self.anchor is None: self.anchor=xyz.copy()
        self.contact_active=True; self.peak_amp=max(self.peak_amp,current_amp,amp)
        d=xyz-self.anchor
        raw=dict(zip(('x','y','z','dX','dY','dZ'),[float(x),float(y),z,*d.tolist()]))
        domain_warning=False
        if self.calibration_domain:
            xr=self.calibration_domain.get('x');yr=self.calibration_domain.get('y')
            domain_warning=bool((xr and not float(xr[0])<=x<=float(xr[1])) or
                                 (yr and not float(yr[0])<=y<=float(yr[1])))
        working_range_warning=False
        if self.working_range_metadata:
            wx=self.working_range_metadata.get('x');wy=self.working_range_metadata.get('y')
            wz=self.working_range_metadata.get('depth')
            working_range_warning=bool(
                (wx and not float(wx[0])<=x<=float(wx[1])) or
                (wy and not float(wy[0])<=y<=float(wy[1])) or
                (wz and not float(wz[0])<=z<=float(wz[1])))
        extra={
            'amplitude_uT':amp,'contact_threshold_uT':self.contact_threshold,
            'contact_gate':'temporary amplitude threshold','cosine_similarity':score,
            'anti_jump_policy':self.jump_policy,
            'calibration_depth_range_mm':[zlo,zhi],
            'depth_extrapolated':not zlo<=z<=zhi,'relative_reference':'first valid prediction since B0/no-contact',
            'xy_semantics':(self.xy_semantics or
                            ('continuous XY refined from local magnetic node candidates'
                             if self.hybrid is not None else
                             ('continuous XY calibrated from this magnetic-layer profile'
                              if self.continuous else
                              'discrete calibrated location; not validated continuous XY'))),
            'fit_model':(self.fit_model_description or
                         ('top-k node hybrid XY + weighted local Z'
                          if self.hybrid is not None else
                          ('RBF continuous XY + ridge Z' if self.continuous else
                           'nearest cosine XY + local linear Z'))),
            'fit_variant':self.fit_variant,'evidence_tier':self.evidence_tier,
            'continuous_xy_method':self.continuous_xy_method,
            'working_range_model':self.working_range,
            'working_range_mm':self.working_range_metadata or None,
            'outside_declared_working_range':working_range_warning,
            'calibration_domain_mm':self.calibration_domain,
            'xy_outside_calibration_domain':domain_warning,
            'excluded_calibration_samples':self.excluded,
            'uncorrected_static_xyz_mm':uncorrected_xyz.tolist(),
            'unaligned_xyz_mm':unaligned_xyz.tolist(),
            'xy_rotation_deg':math.degrees(self.xy_rotation_rad),
            'xy_translation_mm':self.xy_translation.tolist(),
            'xy_translation_source':self.xy_translation_source,
            'xy_translation_details':self.xy_translation_details,
            'dynamic_model_enabled':self.dynamic_model is not None,
            'dynamic_history_ready':dynamic_ready,
            'dynamic_gate':dynamic_gate}
        if dynamic_state is not None:
            raw['mode5']=dynamic_state
            extra['dynamic_state']=dynamic_state
        if contact_score is not None:
            extra.update(contact_score=contact_score,
                         contact_score_threshold=float(self.contact_classifier['score_threshold']),
                         contact_gate='shape-aware ridge classifier + amplitude floor')
        if hybrid_extra is not None:
            extra.update(hybrid_extra)
        elif not self.continuous:
            extra.update(depth_fit_slope_uT_per_mm=a,depth_fit_intercept_uT=b)
        return record(t,raw,xyz.tolist(),'NormalizedFit',extra)

def record(t,raw,xyz,family,extra=None) -> dict[str,Any]:
    fields={k:raw.get(k) for k in RAW_FIELDS}
    fields.update(raw)  # Future fields must never be discarded.
    valid=xyz is not None and len(xyz)==3 and all(v is not None and math.isfinite(float(v)) for v in xyz)
    return {'t_ns':int(t),'valid':valid,'raw':fields,'xyz':list(xyz) if xyz is not None and valid else None,'family':family,'extra':extra or {},'status':'有效' if valid else '三维适配输出无效'}

def empty_record(t,status,extra=None) -> dict[str,Any]:
    return {'t_ns':int(t),'valid':False,'raw':{k:None for k in RAW_FIELDS},'xyz':None,
            'family':'--','extra':extra or {},'status':status}

def vector(group,raw=False):
    keys=('x_raw','y_raw','z_raw') if raw else ('x_ut','y_ut','z_ut')
    try:
        if any(not group[i].get('valid',True) for i in range(5)): return None
        b=np.array([group[i][k] for i in range(5) for k in keys],float)
        return b if np.isfinite(b).all() else None
    except (KeyError,TypeError,ValueError): return None

def baseline_schedule(times,samples,stages,config):
    """Baselines use only past magnetic frames; no XYZ/depth labels are consumed."""
    start=times[0]+int(number(config['baseline_start'],'B0开始秒',0,86400)*1e9)
    end=start+int(number(config['baseline_seconds'],'B0时长秒',.2,60)*1e9)
    windows=[(end,start,end,'指定无载窗口')]
    if config.get('mode')=='归一化拟合算法' and config.get('baseline_policy')=='动作前自动更新':
        for t,row in stages:
            if str(row.get('command','')).startswith('G1') and re.fullmatch(
                r'press_indent|depth_\d+|preload_indent|grid_indent(?:_\d+)?|load|contact_onset',str(row.get('phase',''))):
                windows.append((int(t),int(t)-180_000_000,int(t)-20_000_000,'动作前间隙'))
    out=[]
    for effective,a,b,source in sorted(windows):
        lo,hi=bisect.bisect_left(times,a),bisect.bisect_right(times,b)
        vv=[v for s in samples[lo:hi] if (v:=vector(s)) is not None]
        window_times=times[lo:hi]
        continuous=(window_times and b-window_times[-1]<=100_000_000
                    and all(y-x<=250_000_000 for x,y in zip(window_times,window_times[1:])))
        if len(vv)>=5 and continuous: out.append((effective,np.median(vv,axis=0),source))
    if not out: raise ValueError('无可用B0：请选择至少含5帧有效数据的无载窗口')
    return out

def run_backtest(samples,times,stages,trial_dir,config,profile=None,cancel=None,progress=None,predictor_factory=None):
    from contact_display import contact_visible, prediction_contact
    if not samples or len(samples)!=len(times): raise ValueError('请先载入有效Trial')
    if times[0]<=0 or any(b<=a for a,b in zip(times,times[1:])): raise ValueError('磁场时间戳须为正且严格递增')
    schedules=baseline_schedule(times,samples,stages,config)
    fitting=config['mode']=='归一化拟合算法'
    if fitting:
        engine=NormalizedFit(profile,trial_dir,config['threshold'],config['min_cosine'],
                             config['smoothing_ms'],config.get('contact_threshold'),
                             config.get('jump_policy',JUMP_HYSTERESIS))
    else:
        if predictor_factory is None:
            from replay_neural_adapter import ReplayNeuralPredictor
            predictor_factory=ReplayNeuralPredictor
        engine=predictor_factory(config['checkpoint'],family=config.get('family','auto'),
                                 output_mapping=config.get('mapping','{}'),device=config.get('device','cpu'))
        # Seed B0 explicitly from the user's past-only no-load window, then build history.
        engine.reset_all()
    results=[]; j=0; ready=False; last=None; previous_t=None; base_name='--'
    for i,(t,g) in enumerate(zip(times,samples)):
        if cancel is not None and cancel.is_set(): raise InterruptedError('回测已取消')
        if previous_t is not None and t-previous_t>1_000_000_000:
            ready=False; last=None
        while j<len(schedules) and schedules[j][0]<=t:
            _,base,base_name=schedules[j]
            # Do not resurrect a baseline whose entire window predates a data gap.
            if previous_t is not None and t-previous_t>1_000_000_000 and schedules[j][0]<=previous_t:
                j+=1
                continue
            if isinstance(engine,NormalizedFit): engine.set_baseline(base)
            else:
                engine.reset_all()
                engine.impl.b0=base.reshape(5,3).astype(np.float32)
            ready=True; last=None; j+=1
        b=vector(g)
        previous_t=t
        if not ready:
            r=empty_record(t,'等待指定B0 / 数据间断后需重新校准')
        elif b is None:
            r=empty_record(t,'磁场无效 / 缺传感器'); last=None
            if isinstance(engine,NormalizedFit): engine.buffer.clear(); engine.anchor=None
            else:
                if engine.impl.b0 is None:raise RuntimeError('神经网络 B0 尚未建立')
                saved=engine.impl.b0.copy(); engine.reset_all(); engine.impl.b0=saved
        elif isinstance(engine,NormalizedFit):
            r=engine.push(t,b)
        elif vector(g,raw=True) is None:
            r=empty_record(t,'缺原始计数：神经网络Q输入不可伪造'); last=None
            if engine.impl.b0 is None:raise RuntimeError('神经网络 B0 尚未建立')
            saved=engine.impl.b0.copy(); engine.reset_all(); engine.impl.b0=saved
        else:
            pp=engine.push_group(g,t)
            if pp:
                p=pp[-1]
                last=record(p.t_ns,p.raw_fields,p.visual_xyz_mm,p.family,p.extra_outputs)
                last['extra']=dict(last['extra'],prediction_t_ns=int(p.t_ns),visual_sources=list(p.visual_sources))
                last['valid']=last['valid'] and bool(p.interaction_valid)
                visible,reason=prediction_contact(p)
                last['extra'].update({'display.contact':visible,'display.contact_reason':reason})
                last['status']='有效' if last['valid'] else '模型输出无效'
            r=dict(last) if last is not None else empty_record(t,'神经网络历史窗口预热')
            r['t_ns']=int(t)
        r['extra']=dict(r['extra'],baseline_source=base_name)
        # Optional visualization selection never renames original raw fields.
        visual=config.get('visual','auto')
        if r['valid'] and visual!='auto':
            names={'xyz':('x','y','z'),'XYZ':('X','Y','Z'),'dXYZ':('dX','dY','dZ')}[visual]
            vals=[r['raw'].get(k) for k in names]
            if any(v is None or not isinstance(v,(int,float)) or not math.isfinite(v) for v in vals):
                r=dict(r,valid=False,xyz=None,status='当前模型不提供所选三维字段')
            else: r['xyz']=[float(v) for v in vals]
        visible,reason=contact_visible(r['raw'],r['extra'],r['valid'],family=r['family'])
        r['extra'].update({'display.contact':visible,'display.contact_reason':reason})
        results.append(r)
        if progress is not None and i%100==0: progress(i+1,len(times))
    return results
