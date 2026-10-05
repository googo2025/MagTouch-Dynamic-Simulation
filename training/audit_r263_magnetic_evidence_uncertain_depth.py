#!/usr/bin/env python3
"""Audit raw magnetic evidence without declaring nominal shallow Z contact.

Null statistics use earlier unloaded/lift-before-first-contact observations.
Nominal Z only groups the diagnostic AFTER magnetic evidence is computed.
Evidence absence is indeterminate contact, not a negative physical label.
"""
from __future__ import annotations

import json

import numpy as np

from training.audit_r226_shallow_history_gate import ROOT
from training.audit_r227_assisted_contact_gate import load_rows
from training.probe_r245_joint_contact_state import prepare
from training.probe_r228_shallow_inverse import reference

OUT=ROOT/"analysis/r263_magnetic_evidence_uncertain_depth_20260929"
FREE=("unloaded_trial","precontact_lift")
LAGS=(100_000_000,300_000_000,700_000_000)


def historical_changes(time,field,eligible):
    time=np.asarray(time,np.int64);field=np.asarray(field,float);eligible=np.asarray(eligible,bool)
    if (field.shape!=(len(time),15) or eligible.shape!=(len(time),) or
            not np.isfinite(field).all() or np.any(np.diff(time)<0)):
        raise ValueError("finite chronological 15-channel observations required")
    if not len(time):
        return np.empty((0,len(LAGS),15))
    reset=~eligible.copy();reset[0]=True
    reset[1:] |= ~eligible[:-1] | (np.diff(time)>250_000_000) | (np.diff(time)<=0)
    start=np.maximum.accumulate(np.where(reset,np.arange(len(time)),0))
    blocks=[]
    for lag in LAGS:
        ids=np.minimum(np.maximum(np.searchsorted(time,time-lag,side="right")-1,start),np.arange(len(time)))
        blocks.append(field-field[ids])
    return np.stack(blocks,axis=1)


def null_fit(rows,device):
    fields,weights=[] ,[]
    for row in rows:
        if row["partition"]!="development" or row["device"]!=device:
            continue
        use=row["eligible"]&~row["state_validate"]&np.isin(row["labels"],FREE)
        ids=np.flatnonzero(use)[::4]
        if len(ids):
            fields.append(row["field_uT"][ids]);weights.append(np.full(len(ids),1./len(ids)))
    x=np.vstack(fields);w=np.concatenate(weights);w/=w.sum()
    center=np.sum(w[:,None]*x,axis=0)
    dx=x-center
    cov=np.einsum("n,ni,nj->ij",w,dx,dx)
    cov=.8*cov+.2*np.diag(np.diag(cov))
    eig,basis=np.linalg.eigh(cov)
    floor=max(1.,float(np.median(eig))*.02)
    whitener=(basis/np.sqrt(np.maximum(eig,floor)))@basis.T
    model={"center":center,"whitener":whitener,"null_samples":len(x),"eigen_floor_uT2":floor}
    static,moving=[],[]
    # Set thresholds with training-only noise; held-out noise remains a check.
    for row in rows:
        if row["partition"]!="development" or row["device"]!=device:
            continue
        a,d=evidence_scores(row,model)
        use=row["eligible"]&~row["state_validate"]&np.isin(row["labels"],FREE)
        ids=np.flatnonzero(use)[::4]
        static.extend(a[ids].tolist());moving.extend(d[ids].tolist())
    model["absolute_limit"]=float(np.percentile(static,99.5))
    model["change_limit"]=float(np.percentile(moving,99.5))
    return model


def evidence_scores(row,model):
    whiten=model["whitener"]
    absolute=np.linalg.norm((row["field_uT"]-model["center"])@whiten,axis=1)
    changes=historical_changes(row["time"],row["field_uT"],row["eligible"])
    dynamic=np.linalg.norm(changes@whiten,axis=2).max(axis=1)
    return absolute,dynamic


def confirm_evidence(row,absolute,dynamic,model):
    result=np.zeros(len(absolute),bool)
    run=0
    for i in range(len(result)):
        if not row["eligible"][i] or (i and (row["time"][i]-row["time"][i-1]>250_000_000 or row["time"][i]<=row["time"][i-1])):
            run=0
        if not row["eligible"][i]:
            continue
        high=absolute[i]>model["absolute_limit"] or dynamic[i]>model["change_limit"]
        run=run+1 if high else 0
        result[i]=run>=3
    return result


def ramp_diagnostics(row,evidence,absolute,dynamic,model):
    if row["mode"]=="none":
        return []
    q=reference(row);time=row["time"]
    dz=np.zeros(len(q));dz[1:]=np.diff(np.nan_to_num(q[:,2],nan=-100.))
    rising=row["eligible"]&np.isfinite(q).all(axis=1)&(dz>1e-4)&(q[:,2]>=-.5)&(q[:,2]<=.8)
    ids=np.flatnonzero(rising)
    records=[]
    for block in np.split(ids,np.flatnonzero(np.diff(ids)>1)+1):
        if len(block)<5 or not np.any(q[block,2]<.2) or not np.any(q[block,2]>=.4):
            continue
        # Report first confirmed evidence, not retrospectively backdated onset.
        hits=block[evidence[block]]
        first=int(hits[0]) if len(hits) else None
        records.append({"first_index":int(block[0]),"last_index":int(block[-1]),
            "nominal_xy_at_entry_mm":q[block[0],:2].tolist(),
            "confirmation_nominal_z_mm":None if first is None else float(q[first,2]),
            "confirmation_time_ns":None if first is None else int(time[first]),
            "nominal_shallow_frames":int(np.sum((q[block,2]>=0.)&(q[block,2]<=.2))),
            "nominal_shallow_with_evidence":int(np.sum((q[block,2]>=0.)&(q[block,2]<=.2)&evidence[block]))})
    return records


def main():
    if (OUT/"report.json").exists():
        raise FileExistsError(OUT/"report.json")
    rows=load_rows();prepare(rows)
    models={d:null_fit(rows,d) for d in ("source","target")}
    OUT.mkdir(parents=True,exist_ok=True)
    trials=[];pooled={}
    for row in rows:
        model=models[row["device"]]
        a,d=evidence_scores(row,model)
        confirmed=confirm_evidence(row,a,d,model)
        trial=int(row["path"].name.split("_")[1])
        np.savez_compressed(OUT/f"trial_{trial:06d}_evidence.npz",time_ns=row["time"],
            absolute_score=a,change_score=d,magnetic_response_evidence=confirmed,
            contact_truth=np.full(len(a),-1,np.int8))
        item={"trial":row["path"].name,"device":row["device"],"partition":row["partition"],"stages":{}}
        group=pooled.setdefault(row["partition"]+"_"+row["device"],{})
        for label in ("shallow_descent","shallow_ascent","deep_contact","deep_still",
                      "lift_residual","postcontact_lift","precontact_lift","unloaded_trial"):
            mask=row["eligible"]&(row["labels"]==label)
            if not mask.any():
                continue
            count={"frames":int(mask.sum()),"with_magnetic_evidence":int(np.sum(mask&confirmed)),
                   "insufficient_evidence":int(np.sum(mask&~confirmed))}
            item["stages"][label]=count
            slot=group.setdefault(label,{k:0 for k in count})
            for key,value in count.items():
                slot[key]+=value
        val=row["eligible"]&row["state_validate"]&np.isin(row["labels"],FREE)
        item["heldout_free"]= {"n":int(val.sum()),"evidence":int(np.sum(val&confirmed))}
        item["ramps"]=ramp_diagnostics(row,confirmed,a,d,model)
        trials.append(item)
    records={}
    for device in models:
        model=models[device]
        np.savez_compressed(OUT/f"{device}_null_model.npz",**{k:v for k,v in model.items()})
        records[device]={k:float(v) if np.isscalar(v) else v.tolist() for k,v in model.items()}
    report={"schema":"MagTouch-R263-MagneticEvidenceUncertainDepth-1",
        "status":"offline raw magnetic audit; not a contact-truth classifier or deployed XYZ gate",
        "principle":"nominal 0-0.2mm neither guarantees contact nor guarantees measurable magnetic deformation",
        "statistic":"noise-whitened 15D absolute residual OR 0.1/0.3/0.7s change, three consecutive causal confirmations",
        "threshold":"99.5 percentile earlier training null observations only",
        "warning":"field response can be contact, residual memory, environmental/motor disturbance; absent response does NOT prove no contact",
        "scoring":"stage depths only group diagnostics after field evidence; R208 viewed; no trajectory alignment",
        "models":records,"pooled":pooled,"trials":trials}
    (OUT/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    for group,m in pooled.items():
        print(group,{k:m.get(k) for k in ("shallow_descent","deep_contact","lift_residual","unloaded_trial")},flush=True)


if __name__=="__main__":
    main()
