#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R15 history-aware relative MagTouch model for the real 5-sensor hardware.

Main task: past/current magnetic-change sequence -> current contact-relative dXYZ.
No magnetic-threshold-derived contact label is predicted by the main model.
History=0 is represented internally as one current frame (sequence length 1).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import torch
from torch import nn
from model_magtouch_ra_hw5 import HW5_SENSOR_XY,AlternatingTransformer,CoordinatePositionEncoder,AttnPool

@dataclass
class RelativeOutput:
    dxyz_norm: torch.Tensor
    extras: Dict[str,torch.Tensor]

class DeterministicAxisQGate(nn.Module):
    def __init__(self,feature:str,min_gain:float=.03,gate_power:float=2.5):
        super().__init__();self.feature=feature;self.min_gain=float(min_gain);self.gate_power=float(gate_power);self.signal_blocks={'R0':1,'R1':2,'R2':3,'R3':3}[feature]
    def forward(self,x,q):
        g=self.min_gain+(1-self.min_gain)*q.clamp(0,1).pow(self.gate_power);y=x.clone()
        for bi in range(self.signal_blocks):
            s=3*bi;y[...,s:s+3]=y[...,s:s+3]*g
        return y

class RelativeTactileHW5(nn.Module):
    def __init__(self,in_dim,feature='R1',timesteps=64,d_model=128,layers=3,heads=4,dropout=.1,use_q=True):
        super().__init__();self.use_q=bool(use_q);self.feature=feature;timesteps=max(1,int(timesteps))
        self.register_buffer('sensor_xy',HW5_SENSOR_XY.clone(),persistent=True);self.qgate=DeterministicAxisQGate(feature) if use_q else None
        self.proj=nn.Sequential(nn.Linear(in_dim,d_model),nn.GELU(),nn.Dropout(dropout),nn.Linear(d_model,d_model));self.qproj=nn.Linear(3,d_model) if use_q else None
        self.coord=CoordinatePositionEncoder(d_model);self.time=nn.Parameter(torch.randn(1,timesteps,1,d_model)*.02)
        self.trunk=AlternatingTransformer(d_model,layers,heads,dropout,self.sensor_xy,16.0);self.norm=nn.LayerNorm(d_model);self.pool=AttnPool(d_model,heads)
        self.xyz=nn.Sequential(nn.Linear(d_model,d_model),nn.GELU(),nn.Dropout(dropout),nn.Linear(d_model,3))
    def forward(self,x,q_axis):
        if x.ndim!=4 or x.shape[2]!=5:raise ValueError(f'Expected [B,T,5,D], got {tuple(x.shape)}')
        if q_axis.shape[-2:]!=(5,3):raise ValueError(f'Expected Q [B,T,5,3], got {tuple(q_axis.shape)}')
        y=self.qgate(x,q_axis) if self.qgate is not None else x;h=self.proj(y)+self.time[:,:x.shape[1]]+self.coord(self.sensor_xy).view(1,1,5,-1)
        if self.qproj is not None:h=h+self.qproj(q_axis)
        h=self.norm(self.trunk(h));p=self.pool(h)
        return RelativeOutput(self.xyz(p),{'pooled':p})
