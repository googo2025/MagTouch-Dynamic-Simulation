#!/usr/bin/env python3
"""Joint causal nominal contact/depth-state model; offline development only.

Four states distinguish free baseline, lifted magnetic residue, shallow
indentation and atlas-supported indentation. Commands supply training/scoring
labels, never features. R208 is already viewed and remains retrospective.
"""
from __future__ import annotations

import json

import numpy as np

from training.audit_r226_shallow_history_gate import ROOT, _auc
from training.audit_r227_assisted_contact_gate import INTEREST, load_rows, vote
from training.probe_r228_shallow_inverse import reference

OUT = ROOT / "analysis/r245_joint_contact_state_20260929"
VALID_TRIALS = {9, 12, 21, 24, 27, 31}
STATE_NAMES = ("unloaded", "lifted_residue", "shallow_nominal", "deep_nominal")


def causal_state_features(time, field, eligible):
    time = np.asarray(time,np.int64)
    field = np.asarray(field,float)
    eligible = np.asarray(eligible,bool)
    if (not len(time) or field.shape != (len(time),15) or
            eligible.shape != (len(time),) or np.any(np.diff(time)<0) or
            not np.isfinite(field).all()):
        raise ValueError("state features require finite chronological 15D field")
    amp = np.linalg.norm(field,axis=1)
    reset = ~eligible
    reset = reset.copy()
    reset[0] = True
    reset[1:] |= (~eligible[:-1]) | (np.diff(time) > 250_000_000) | (np.diff(time) <= 0)
    start = np.maximum.accumulate(np.where(reset,np.arange(len(time)),0))
    previous, scalars = [], [amp/100.]
    for lag in (100_000_000,300_000_000,700_000_000,2_000_000_000,5_000_000_000):
        ids = np.minimum(np.maximum(np.searchsorted(time,time-lag,side="right")-1,start),
                         np.arange(len(time)))
        prior = field[ids]
        delta = field-prior
        three = ((amp-np.linalg.norm(prior,axis=1))/100.,
                 np.linalg.norm(delta,axis=1)/100.,
                 np.sum(field*delta,axis=1)/(100.*np.maximum(amp,1.)))
        previous.append((delta,three))
        if len(previous) <= 3:
            scalars.extend(three)
    parts = [np.column_stack(scalars),field/np.maximum(amp[:,None],1.),
             previous[1][0]/100.]
    for delta,three in previous[3:]:
        parts.extend((np.column_stack(three),delta/100.))
    return np.column_stack(parts)


def class_labels(row):
    out = np.full(len(row["time"]), -1, int)
    out[np.isin(row["labels"], ("unloaded_trial", "precontact_lift"))] = 0
    out[np.isin(row["labels"], ("lift_residual", "postcontact_lift"))] = 1
    out[np.isin(row["labels"], ("shallow_descent", "shallow_ascent"))] = 2
    out[np.isin(row["labels"], ("deep_contact", "deep_still"))] = 3
    return out


def prepare(rows):
    for row in rows:
        row["state_x"] = causal_state_features(row["time"],row["field_uT"],row["eligible"])
        row["state_y"] = class_labels(row)
        trial = int(row["path"].name.split("_")[1])
        row["state_validate"] = np.full(len(row["time"]), trial in VALID_TRIALS)
        if row["mode"] == "grid_points" and row["partition"] == "development":
            ref = reference(row)
            xy = np.nan_to_num(np.round(ref[:, :2]), nan=100).astype(int)
            # Complete grid positions, including their lift observations,
            # are held out together; this is not an independent-device test.
            row["state_validate"] = (3 * xy[:, 0] + 7 * xy[:, 1]) % 5 == 4


def dataset(rows, validate=False, full_development=False):
    xs, ys, ws = [], [], []
    for row in rows:
        if row["partition"] != "development":
            continue
        part = (np.ones(len(row["time"]), bool) if full_development else
                row["state_validate"] if validate else ~row["state_validate"])
        for cls in range(int(row["state_y"].max())+1):
            ids = np.flatnonzero(row["eligible"] & part & (row["state_y"] == cls))
            if not len(ids):
                continue
            ids = ids[::max(1, int(np.ceil(len(ids) / 500)))]
            xs.append(row["state_x"][ids])
            ys.append(np.full(len(ids), cls))
            # Neither thousands of lifted frames nor a single grid can
            # dominate the loss; grid diversity still gets more weight.
            total = min(3., np.sqrt(len(ids)) / 5.)
            ws.append(np.full(len(ids), total / len(ids)))
    x, y, w = np.vstack(xs), np.concatenate(ys), np.concatenate(ws)
    for cls in range(int(y.max())+1):
        mask = y == cls
        w[mask] /= w[mask].sum()
    w *= len(w) / w.sum()
    return x, y, w


def softmax(logits):
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


class StateMLP:
    def __init__(self, width=32, seed=245, n_classes=4):
        self.width = int(width)
        self.seed = seed
        self.n_classes = int(n_classes)

    def predict(self, x):
        v = np.clip((x - self.mean) / self.scale, -8., 8.)
        for w, b in zip(self.params[::2], self.params[1::2]):
            v = v @ w + b
            if w is not self.params[-2]:
                v = np.tanh(v)
        return softmax(v)

    def fit(self, x, y, weight, validation=None, epochs=120):
        rng = np.random.default_rng(self.seed)
        self.mean = np.average(x, axis=0, weights=weight)
        self.scale = np.maximum(np.sqrt(np.average((x-self.mean)**2,
                                      axis=0, weights=weight)), .05)
        v = np.clip((x-self.mean)/self.scale, -8., 8.)
        sizes = [x.shape[1]] + ([self.width, self.width] if self.width else []) + [self.n_classes]
        params = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            params.extend((rng.normal(0., np.sqrt(1./a), (a,b)), np.zeros(b)))
        self.params = params
        first = [np.zeros_like(p) for p in params]
        second = [np.zeros_like(p) for p in params]
        best, chosen, best_epoch, step = np.inf, None, 0, 0
        self.history = []
        for epoch in range(1, epochs+1):
            for ids in np.array_split(rng.permutation(len(x)), max(1, int(np.ceil(len(x)/256)))):
                activations = [v[ids]]
                for layer, (w,b) in enumerate(zip(params[::2],params[1::2])):
                    value = activations[-1] @ w + b
                    activations.append(softmax(value) if layer == len(sizes)-2 else np.tanh(value))
                delta = activations[-1].copy()
                delta[np.arange(len(ids)), y[ids]] -= 1.
                delta *= weight[ids, None] / len(ids)
                gradients = [None] * len(params)
                for layer in range(len(sizes)-2, -1, -1):
                    w = params[2*layer]
                    gradients[2*layer] = activations[layer].T @ delta + .001*w
                    gradients[2*layer+1] = delta.sum(axis=0)
                    if layer:
                        delta = (delta @ w.T) * (1.-activations[layer]**2)
                step += 1
                for p,g,m,s in zip(params,gradients,first,second):
                    m *= .9
                    m += .1*g
                    s *= .999
                    s += .001*g*g
                    p -= .001*(m/(1.-.9**step))/(np.sqrt(s/(1.-.999**step))+1e-8)
            if epoch % 5 == 0:
                if validation is None:
                    loss = float(epoch == epochs) * -1.
                else:
                    vx,vy,vw = validation
                    prob = self.predict(vx)
                    loss = float(np.average(-np.log(np.maximum(prob[np.arange(len(vy)),vy], 1e-9)), weights=vw))
                self.history.append({"epoch":epoch,"weighted_validation_cross_entropy":loss})
                if loss < best:
                    best, best_epoch = loss, epoch
                    chosen = [p.copy() for p in params]
        self.params = chosen
        self.best_epoch = best_epoch
        self.best_loss = best
        return self

    def save(self, path):
        data = {"mean":self.mean,"scale":self.scale,"width":np.array(self.width),
                "n_classes":np.array(self.n_classes)}
        data.update({f"p{i}":p for i,p in enumerate(self.params)})
        np.savez_compressed(path, **data)


def thresholds(rows, model):
    negative, deep, free = [], [], []
    for row in rows:
        if row["partition"] != "development":
            continue
        probability = model.predict(row["state_x"])
        use = row["eligible"] & row["state_validate"]
        negative.extend((probability[use & (row["state_y"] >= 0) & (row["state_y"] < 2),2:].sum(axis=1)).tolist())
        free.extend((probability[use & (row["state_y"] == 0),2:].sum(axis=1)).tolist())
        deep.extend(probability[use & (row["state_y"] == 3),2].tolist())
    return {"contact":max(.5, float(np.percentile(negative,99.5)),
                          float(np.percentile(free,99.9))),
            "shallow":max(.7,float(np.percentile(deep,99.5)))}


def evaluate(rows, model, limits):
    trials, pooled = [], {}
    for row in rows:
        prob = model.predict(row["state_x"])
        contact_score = prob[:,2:].sum(axis=1)
        accepted = vote(contact_score,limits["contact"],row["eligible"],
                        required=2,time=row["time"])
        shallow = accepted & (prob[:,2] >= limits["shallow"])
        protected = accepted | (row["eligible"] & row["ui_valid"] &
                                (row["amplitude_uT"] >= 140.))
        trial = {"trial":row["path"].name,"device":row["device"],
                 "partition":row["partition"],"mode":row["mode"],"labels":{}}
        dest = pooled.setdefault(row["partition"]+"_"+row["device"],{})
        for label in INTEREST:
            mask = row["eligible"] & (row["labels"] == label)
            if not mask.any():
                continue
            counts = {"n":int(mask.sum()),"ui":int(np.sum(mask&row["ui_valid"])),
                      "contact":int(np.sum(mask&accepted)),
                      "protected":int(np.sum(mask&protected)),
                      "shallow_identified":int(np.sum(mask&shallow))}
            trial["labels"][label] = counts
            slot = dest.setdefault(label,{key:0 for key in counts})
            for key,value in counts.items():
                slot[key] += value
        if row["mode"] == "grid_points":
            trial["nominal_shallow_grid"] = trial["labels"].get("shallow_descent")
        trials.append(trial)
    return {"trials":trials,"pooled":pooled}


def main():
    if (OUT/"report.json").exists():
        raise FileExistsError(OUT/"report.json")
    rows = load_rows()
    prepare(rows)
    tx,ty,tw = dataset(rows)
    vx,vy,vw = dataset(rows,validate=True)
    results, models = {}, {}
    OUT.mkdir(parents=True,exist_ok=True)
    for width in (0,32,64):
        model = StateMLP(width).fit(tx,ty,tw,(vx,vy,vw))
        limits = thresholds(rows,model)
        results[str(width)] = {"best_epoch":model.best_epoch,
                              "validation_loss":model.best_loss,"limits":limits,
                              "history":model.history,**evaluate(rows,model,limits)}
        model.save(OUT/f"state_model_width{width}.npz")
        models[width] = model
        print("width",width,"epoch",model.best_epoch,"thresholds",limits,flush=True)
        for key,value in results[str(width)]["pooled"].items():
            print(key,{label:value.get(label) for label in
                  ("shallow_descent","deep_contact","lift_residual","postcontact_lift","unloaded_trial")},flush=True)
    chosen = min(models,key=lambda width:models[width].best_loss)
    report = {"schema":"MagTouch-R245-JointContactState-1",
              "status":"retrospective offline prototype, no UI/hardware/paper changes",
              "classes":STATE_NAMES,"feature_count":tx.shape[1],
              "training_samples":len(tx),"validation_samples":len(vx),
              "validation_trials":sorted(VALID_TRIALS),
              "grid_validation":"entire XY node hash group 4 held out for loss selection",
              "warning":"R208 inspected; shallow states are nominal instruction labels, not physically verified contact",
              "selected_by_validation_loss_width":chosen,"results":results}
    (OUT/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
