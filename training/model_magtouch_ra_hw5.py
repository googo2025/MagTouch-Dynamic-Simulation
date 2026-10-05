#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MagTouch-RA-HW5 R13: 5-node hardware adaptation with axis-level reliability.

Key R13 change
--------------
The physical 19C fault is axis-local (S4-Y wraps while S4-X/Z remain usable).
Therefore reliability is represented as Q[B,T,5,3], not one scalar per sensor.
The learned frontend gates XYZ-aligned feature blocks axis-by-axis, while a
3-axis reliability embedding informs the spatiotemporal trunk.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import torch
from torch import nn
import torch.nn.functional as F

HW5_SENSOR_XY = torch.tensor([
    [-7.594, +7.594],
    [-7.594, -7.594],
    [+7.594, +7.594],
    [+7.594, -7.594],
    [ 0.000, +1.406],
], dtype=torch.float32)

@dataclass
class ModelOutput:
    reg: torch.Tensor
    cls_logits: torch.Tensor
    reliability: Optional[torch.Tensor]   # [B,T,5,3] in R13
    extras: Dict[str, torch.Tensor]

class FeedForward(nn.Module):
    def __init__(self,d_model,dropout=.1):
        super().__init__();h=4*d_model
        self.net=nn.Sequential(nn.Linear(d_model,h),nn.GELU(),nn.Dropout(dropout),nn.Linear(h,d_model),nn.Dropout(dropout))
    def forward(self,x):return self.net(x)

class TemporalBlock(nn.Module):
    def __init__(self,d_model,heads=4,dropout=.1):
        super().__init__();self.n1=nn.LayerNorm(d_model);self.a=nn.MultiheadAttention(d_model,heads,dropout=dropout,batch_first=True);self.n2=nn.LayerNorm(d_model);self.ff=FeedForward(d_model,dropout)
    def forward(self,x):
        h=self.n1(x);y,_=self.a(h,h,h,need_weights=False);x=x+y;return x+self.ff(self.n2(x))

def geometry_mask(coords:torch.Tensor,edge_threshold_mm:float=16.0):
    d=torch.cdist(coords,coords);ok=d<=edge_threshold_mm
    mask=torch.full_like(d,float('-inf'));mask[ok]=0.0
    return mask

class SensorBlock(nn.Module):
    def __init__(self,d_model,heads=4,dropout=.1,sensor_xy=HW5_SENSOR_XY,edge_threshold_mm=16.0):
        super().__init__();self.n1=nn.LayerNorm(d_model);self.a=nn.MultiheadAttention(d_model,heads,dropout=dropout,batch_first=True);self.n2=nn.LayerNorm(d_model);self.ff=FeedForward(d_model,dropout);self.register_buffer('mask',geometry_mask(sensor_xy,edge_threshold_mm),persistent=True)
    def forward(self,x):
        h=self.n1(x);y,_=self.a(h,h,h,attn_mask=self.mask,need_weights=False);x=x+y;return x+self.ff(self.n2(x))

class AlternatingTransformer(nn.Module):
    def __init__(self,d_model,layers=4,heads=4,dropout=.1,sensor_xy=HW5_SENSOR_XY,edge_threshold_mm=16.0):
        super().__init__();self.temp=nn.ModuleList([TemporalBlock(d_model,heads,dropout) for _ in range(layers)]);self.sensor=nn.ModuleList([SensorBlock(d_model,heads,dropout,sensor_xy,edge_threshold_mm) for _ in range(layers)])
    def forward(self,x):
        b,t,n,c=x.shape
        for tb,sb in zip(self.temp,self.sensor):
            xt=x.permute(0,2,1,3).reshape(b*n,t,c);xt=tb(xt);x=xt.reshape(b,n,t,c).permute(0,2,1,3)
            xs=x.reshape(b*t,n,c);xs=sb(xs);x=xs.reshape(b,t,n,c)
        return x

class AxisReliabilityFrontend(nn.Module):
    """Predict X/Y/Z quality independently for each sensor-time token.

    All HW5 feature layouts are XYZ-aligned blocks (D = 3*k).  A three-axis
    gate is repeated over those blocks, so an S4-Y fault suppresses only the Y
    channels belonging to S4 instead of weakening the whole sensor node.
    """
    def __init__(self,in_dim,hidden=96,gate_power=2.5,min_gain=.03):
        super().__init__()
        if in_dim % 3:
            raise ValueError(f'AxisReliabilityFrontend requires D=3*k, got {in_dim}')
        self.in_dim=int(in_dim);self.blocks=self.in_dim//3;self.gate_power=gate_power;self.min_gain=min_gain
        self.rel=nn.Sequential(nn.Linear(in_dim*4,hidden),nn.GELU(),nn.Dropout(.05),nn.Linear(hidden,hidden//2),nn.GELU(),nn.Linear(hidden//2,3))
    def forward(self,x):
        spatial_med=x.sort(dim=2).values[:,:,2:3,:]
        spatial_dev=(x-spatial_med).abs()
        prev=torch.cat([x[:,0:1],x[:,:-1]],dim=1)
        nxt=torch.cat([x[:,1:],x[:,-1:]],dim=1)
        local=(prev+x+nxt)/3
        temporal=(x-local).abs();delta=(x-prev).abs()
        rel_logits=self.rel(torch.cat([x,spatial_dev,temporal,delta],dim=-1))  # B,T,5,3
        rel=torch.sigmoid(rel_logits)
        gate_axis=self.min_gain+(1-self.min_gain)*rel.clamp(1e-4,1).pow(self.gate_power)
        gate_full=gate_axis.unsqueeze(-2).expand(*gate_axis.shape[:-1],self.blocks,3).reshape(*gate_axis.shape[:-1],self.in_dim)
        return x*gate_full,rel,rel_logits

class CoordinatePositionEncoder(nn.Module):
    def __init__(self,d_model,pitch_mm=15.188):super().__init__();self.pitch=pitch_mm;self.net=nn.Sequential(nn.Linear(2,d_model),nn.GELU(),nn.Linear(d_model,d_model))
    def forward(self,xy):return self.net(xy/max(self.pitch,1e-6))

class AttnPool(nn.Module):
    def __init__(self,d_model,heads=4):super().__init__();self.q=nn.Parameter(torch.randn(1,1,d_model)*.02);self.a=nn.MultiheadAttention(d_model,heads,batch_first=True)
    def forward(self,h):
        tok=h.reshape(h.size(0),h.size(1)*h.size(2),h.size(3));q=self.q.expand(h.size(0),-1,-1);p,_=self.a(q,tok,tok,need_weights=False);return p[:,0]

class MultiTaskHeads(nn.Module):
    def __init__(self,d_model,dropout=.1):
        super().__init__();self.xy=nn.Sequential(nn.Linear(d_model,d_model),nn.GELU(),nn.Dropout(dropout),nn.Linear(d_model,2));self.z=nn.Sequential(nn.Linear(d_model,d_model//2),nn.GELU(),nn.Linear(d_model//2,1));self.slip=nn.Sequential(nn.Linear(d_model,d_model),nn.GELU(),nn.Dropout(dropout),nn.Linear(d_model,2));self.cls=nn.Sequential(nn.Linear(d_model,d_model),nn.GELU(),nn.Dropout(dropout),nn.Linear(d_model,5))
    def forward(self,p):return torch.cat([self.xy(p),self.z(p),self.slip(p)],-1),self.cls(p)

class MagTouchRAHardware5(nn.Module):
    def __init__(self,in_dim,timesteps=32,d_model=160,layers=4,heads=4,dropout=.1,use_reliability=True,pitch_mm=15.188,edge_threshold_mm=16.0):
        super().__init__();self.use_reliability=use_reliability;self.register_buffer('sensor_xy',HW5_SENSOR_XY.clone(),persistent=True)
        self.front=AxisReliabilityFrontend(in_dim,max(64,d_model//2)) if use_reliability else None
        self.proj=nn.Sequential(nn.Linear(in_dim,d_model),nn.GELU(),nn.Dropout(dropout),nn.Linear(d_model,d_model))
        self.relproj=nn.Linear(3,d_model) if use_reliability else None
        self.coord=CoordinatePositionEncoder(d_model,pitch_mm);self.time=nn.Parameter(torch.randn(1,timesteps,1,d_model)*.02)
        self.trunk=AlternatingTransformer(d_model,layers,heads,dropout,self.sensor_xy,edge_threshold_mm);self.norm=nn.LayerNorm(d_model);self.pool=AttnPool(d_model,heads);self.heads=MultiTaskHeads(d_model,dropout)
    def forward(self,x):
        if x.ndim!=4 or x.shape[2]!=5:raise ValueError(f'HW5 expects [B,T,5,D], got {tuple(x.shape)}')
        if self.front is None:
            aligned=x;rel=torch.ones((*x.shape[:3],3),device=x.device,dtype=x.dtype);rel_logits=None
        else:
            aligned,rel,rel_logits=self.front(x)
        h=self.proj(aligned)+self.time[:,:x.shape[1]]+self.coord(self.sensor_xy).view(1,1,5,-1)
        if self.relproj is not None:h=h+self.relproj(rel)
        # Do NOT collapse XYZ reliability into one post-trunk scalar gate.  That
        # would again punish healthy S4-X/Z when only S4-Y is faulty.
        h=self.norm(self.trunk(h));p=self.pool(h);reg,cls=self.heads(p)
        extras={'pooled':p,'reliability_node_mean':rel.mean(dim=-1)}
        if rel_logits is not None:extras['reliability_logits']=rel_logits
        return ModelOutput(reg,cls,rel,extras)

def multitask_loss(out,reg_target,cls_target,contact_mask,rel_target=None,class_weight=None,
                   objective='multitask',lambda_cls=1.0,lambda_xy=1.0,lambda_z=.8,
                   lambda_slip=1.0,lambda_rel=.5):
    """Loss for fair mode-only comparison or full multitask training.

    objective='mode_only' optimizes only 5-class mode recognition.  The learned
    reliability frontend can still adapt end-to-end through classification,
    but no regression/reliability auxiliary target changes the comparison.
    objective='multitask' adds x/y/z/dx/dy and axis-level reliability losses.
    """
    cm=contact_mask.bool();lc=F.cross_entropy(out.cls_logits,cls_target,weight=class_weight)
    if cm.any():
        pr=out.reg[cm];tg=reg_target[cm];lxy=F.smooth_l1_loss(pr[:,:2],tg[:,:2]);lz=F.smooth_l1_loss(pr[:,2:3],tg[:,2:3]);ls=F.smooth_l1_loss(pr[:,3:5],tg[:,3:5])
    else:
        z=out.reg.sum()*0;lxy=lz=ls=z
    rel_logits=out.extras.get('reliability_logits') if out.extras is not None else None
    if rel_target is None or out.reliability is None or rel_logits is None or lambda_rel<=0:
        lr=lc*0
    else:
        lr=F.binary_cross_entropy_with_logits(rel_logits,rel_target.to(dtype=rel_logits.dtype))
    if objective=='mode_only':
        loss=lambda_cls*lc
    elif objective=='multitask':
        loss=lambda_cls*lc+lambda_xy*lxy+lambda_z*lz+lambda_slip*ls+lambda_rel*lr
    else:
        raise ValueError(f'Unknown objective: {objective}')
    return loss,{'cls':float(lc.detach()),'xy':float(lxy.detach()),'z':float(lz.detach()),'slip':float(ls.detach()),'rel':float(lr.detach())}
