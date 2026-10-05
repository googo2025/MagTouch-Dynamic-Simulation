"""Replay-only R18 adapter; reuse the established causal R15 feature construction."""
import json
from pathlib import Path
import numpy as np
import torch
from training.live_multimodel_inference import (
    UnifiedLivePredictor, _LiveBase, _LiveH2R15Predictor, _torch_load_cross_platform,
    normalize_output_mapping, RAW_FIELD_ORDER, _to_python, UnifiedPrediction)

class R18ReplayPredictor(_LiveH2R15Predictor):
    family='H2-Unified-R18'

    def __init__(self,path,record,output_mapping,device):
        try:
            from training.model_unified_magtouch_r18_hw5 import UnifiedMagTouchHW5
        except ImportError:  # standalone/incremental-package validation
            from model_unified_magtouch_r18_hw5 import UnifiedMagTouchHW5
        self.checkpoint_schema=str(record.get('schema',''))
        if self.checkpoint_schema not in ('MagTouch-Unified-R18-checkpoint-1.0','MagTouch-Unified-R18-checkpoint-1.1'):
            raise ValueError(f'不支持的R18 checkpoint schema: {self.checkpoint_schema or "缺失"}')
        self.legacy_bad_absolute=self.checkpoint_schema.endswith('1.0')
        self.torch=torch;self.path=Path(path);self.args=record['args']
        self.feature=str(self.args.get('feature','R1'))
        if self.feature not in ('R0','R1','R2','R3'):raise ValueError('不支持的R18特征')
        self.history=int(record.get('history_frames',self.args.get('history',64)));self.seq_len=max(1,self.history)
        self.use_q=bool(self.args.get('use_q',True));self.motion_epsilon_mm=float(self.args.get('motion_epsilon_mm',.02))
        self.objective=record.get('objective','unified')
        if self.objective not in ('unified','coordinate_only'):raise ValueError('未知R18训练目标')
        user=json.loads(output_mapping or '{}') if isinstance(output_mapping,str) else dict(output_mapping or {})
        # R18's native coordinates are lower-case; V16's uppercase convention is not reused.
        user['h2_absolute']=['x','y','z']
        self.output_mapping=normalize_output_mapping(user)
        self.scaler_median=np.asarray(record['scaler_median'],np.float32)
        self.scaler_scale=np.asarray(record['scaler_scale'],np.float32)
        self.fixed_reg_scale=np.asarray(record['fixed_reg_scale'],np.float32)
        self.relative_reg_scale=np.asarray(record['relative_reg_scale'],np.float32);self.reg_scale=self.relative_reg_scale
        if self.scaler_median.shape!=self.scaler_scale.shape or np.any(self.scaler_scale<=0):raise ValueError('R18标准化参数无效')
        self.mode_names=list(record.get('mode_names',['noncontact','press','slide','hold','release']))
        if self.mode_names!=['noncontact','press','slide','hold','release']:raise ValueError('R18运动模式定义不兼容')
        self.coordinate_frame=record.get('coordinate_frame',{})
        self.device=torch.device(device or 'cpu')
        self.model=UnifiedMagTouchHW5(self.scaler_median.shape[-1],self.feature,self.seq_len,
            int(self.args.get('d_model',128)),int(self.args.get('layers',3)),4,.1,self.use_q,len(self.mode_names)).to(self.device)
        self.model.load_state_dict(record['state_dict']);self.model.eval()
        self.source_dataset=str(self.args.get('dataset',''))
        _LiveBase.__init__(self)

    def _infer(self):
        x,q=self._features()
        if x.shape[1:]!=self.scaler_median.shape:raise ValueError('R18输入特征与训练标准化维度不一致')
        x=(x-self.scaler_median[None,...])/self.scaler_scale[None,...]
        with torch.inference_mode():
            out=self.model(torch.from_numpy(x[None,...]).to(self.device),torch.from_numpy(q[None,...]).to(self.device))
            fixed=out.fixed_xyz_norm.float().cpu().numpy()[0]*self.fixed_reg_scale
            rel=out.dxyz_norm.float().cpu().numpy()[0]*self.relative_reg_scale
            probs=torch.softmax(out.mode_logits.float(),dim=-1).cpu().numpy()[0]
        raw={};extra={'model.coordinate_frame':_to_python(self.coordinate_frame),'model.objective':self.objective,
            'model.extras':_to_python(out.extras),'adapter.input_Q_last':_to_python(q[-1]),
            'adapter.input_Q_mean':float(q[-1].mean()),'adapter.input_Q_min':float(q[-1].min())}
        if self.legacy_bad_absolute:
            extra['model.legacy_fixed_xyz_unsafe_mm']=_to_python(fixed)
            extra['adapter.absolute_output_suppressed']='R18-1.0绝对XY标签按Trial归零；x/y/z已安全屏蔽，仅提供dXYZ与mode5。'
        else:
            for name,value in zip(self.output_mapping['h2_absolute'],fixed):(raw if name in RAW_FIELD_ORDER else extra)[name]=float(value)
        for name,value in zip(self.output_mapping['h2_reg'],rel):(raw if name in RAW_FIELD_ORDER else extra)[name]=float(value)
        idx=int(probs.argmax());valid=True
        if self.objective=='unified':
            raw['mode5']={'index':idx,'label':self.mode_names[idx],'probability':float(probs[idx])}
            extra['model.mode5_probabilities']=dict(zip(self.mode_names,probs.tolist()))
            valid=idx!=0
            extra['adapter.validity']='Predicted noncontact hides 3D only; all raw coordinate heads remain available.'
        else:extra['adapter.validity']='coordinate_only: mode head untrained; no learned contact gate'
        p=self._mapped_prediction(raw,extra,q)
        return UnifiedPrediction(p.family,p.raw_fields,p.extra_outputs,p.visual_xyz_mm,p.visual_sources,p.norm_mm,
            p.moving,p.q_mean,p.q_min,p.history_frames,p.history_required,p.t_ns,valid)

class ReplayNeuralPredictor:
    def __init__(self,path,family='auto',output_mapping=None,device='cpu'):
        path=Path(path).expanduser().resolve()
        checkpoint=_torch_load_cross_platform(torch,path)
        is_r18=isinstance(checkpoint,dict) and str(checkpoint.get('schema','')).startswith('MagTouch-Unified-R18')
        if is_r18:
            if family not in ('auto','H2-Unified-R18'):raise ValueError('模型是R18，所选类型不匹配')
            self.impl=R18ReplayPredictor(path,checkpoint,output_mapping,device)
        else:
            if family=='H2-Unified-R18':raise ValueError('所选文件不是R18联合模型')
            self.impl=UnifiedLivePredictor(path,family,output_mapping,device).impl

    def reset_all(self):self.impl.reset_all()
    def push_group(self,data,t_ns):return self.impl.push_group(data,t_ns)
