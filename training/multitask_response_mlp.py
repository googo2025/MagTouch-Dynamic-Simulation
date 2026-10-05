"""Small causal-feature shared MLP with separately calibrated action heads.

The caller supplies SOURCE-only historical features and protocol-relative
labels. This module never reads commands, device labels, future or XYZ.
"""
from __future__ import annotations
import numpy as np
from training.probe_r245_joint_contact_state import softmax


def task_targets(joint_y, weight):
    y = np.asarray(joint_y, int); w = np.asarray(weight, float)
    if y.ndim != 1 or w.shape != y.shape or not len(y) or np.any((y < 0) | (y > 3)):
        raise ValueError("four SOURCE protocol classes required")
    if not np.isfinite(w).all() or np.any(w <= 0): raise ValueError("positive finite weights required")
    direction = np.where(y == 3, 2, y); context = np.where(y == 3, 0, 1)
    weights = []
    for target, classes in ((direction, 3), (context, 2)):
        values = w.copy()
        for c in range(classes):
            mask = target == c
            if not mask.any(): raise ValueError("missing SOURCE task class")
            values[mask] /= values[mask].sum()
        values *= len(values)/values.sum(); weights.append(values)
    return direction, context, weights[0], weights[1]


class MultiTaskMLP:
    def __init__(self, width=32, seed=292, context_weight=1.):
        if int(width) <= 0 or not np.isfinite(context_weight) or context_weight <= 0:
            raise ValueError("positive width and context weight required")
        self.width = int(width); self.seed = int(seed); self.context_weight = float(context_weight)

    def initialize(self, inputs, rng):
        sizes = ((inputs, self.width), (self.width, self.width), (self.width, 3), (self.width, 2))
        self.params = []
        for a, b in sizes:
            self.params.extend((rng.normal(0, np.sqrt(1./a), (a, b)), np.zeros(b)))

    def forward(self, x):
        w1, b1, w2, b2, wd, bd, wc, bc = self.params
        h1 = np.tanh(x@w1+b1); h2 = np.tanh(h1@w2+b2)
        return h1, h2, softmax(h2@wd+bd), softmax(h2@wc+bc)

    def loss_gradients(self, x, yd, yc, dw, cw, regularization=.001):
        """Exact minibatch sum(w*CE)/N plus weight-only L2 gradient."""
        h1, h2, pd, pc = self.forward(x); n = len(x); ii = np.arange(n)
        loss = float(np.mean(-dw*np.log(np.maximum(pd[ii, yd], 1e-12))) +
                     self.context_weight*np.mean(-cw*np.log(np.maximum(pc[ii, yc], 1e-12))))
        loss += .5*regularization*sum(float(np.square(p).sum()) for p in self.params[::2])
        dd = pd.copy(); dd[ii, yd] -= 1.; dd *= dw[:, None]/n
        dc = pc.copy(); dc[ii, yc] -= 1.; dc *= self.context_weight*cw[:, None]/n
        w1, _, w2, _, wd, _, wc, _ = self.params
        delta2 = (dd@wd.T+dc@wc.T)*(1.-h2*h2)
        delta1 = (delta2@w2.T)*(1.-h1*h1)
        gradients = [x.T@delta1+regularization*w1, delta1.sum(0),
            h1.T@delta2+regularization*w2, delta2.sum(0),
            h2.T@dd+regularization*wd, dd.sum(0), h2.T@dc+regularization*wc, dc.sum(0)]
        return loss, gradients

    def predict(self, x):
        v = np.clip((x-self.mean)/self.scale, -8., 8.)
        _, _, pd, pc = self.forward(v)
        return {"direction": pd, "context": pc}

    def evaluate_loss(self, x, y, weight):
        yd, yc, dw, cw = task_targets(y, weight); p = self.predict(x); ii = np.arange(len(y))
        direction = float(np.mean(-dw*np.log(np.maximum(p["direction"][ii, yd], 1e-12))))
        context = float(np.mean(-cw*np.log(np.maximum(p["context"][ii, yc], 1e-12))))
        return {"direction_ce": direction, "context_ce": context,
                "combined_ce": direction+self.context_weight*context}

    def fit(self, x, y, weight, validation, epochs=80):
        if epochs < 5: raise ValueError("at least one SOURCE validation checkpoint required")
        yd, yc, dw, cw = task_targets(y, weight); normal_weight = .5*(dw+cw)
        if not np.isfinite(x).all(): raise ValueError("finite SOURCE features required")
        self.mean = np.average(x, axis=0, weights=normal_weight)
        self.scale = np.maximum(np.sqrt(np.average((x-self.mean)**2, axis=0, weights=normal_weight)), .05)
        v = np.clip((x-self.mean)/self.scale, -8., 8.)
        rng = np.random.default_rng(self.seed); self.initialize(x.shape[1], rng)
        first = [np.zeros_like(p) for p in self.params]; second = [np.zeros_like(p) for p in self.params]
        best, chosen, best_epoch, step = np.inf, None, 0, 0; self.history = []
        for epoch in range(1, epochs+1):
            for ids in np.array_split(rng.permutation(len(x)), max(1, int(np.ceil(len(x)/256)))):
                _, gradients = self.loss_gradients(v[ids], yd[ids], yc[ids], dw[ids], cw[ids])
                step += 1
                for p, g, m, s in zip(self.params, gradients, first, second):
                    m *= .9; m += .1*g; s *= .999; s += .001*g*g
                    p -= .001*(m/(1.-.9**step))/(np.sqrt(s/(1.-.999**step))+1e-8)
            if epoch % 5 == 0:
                loss = self.evaluate_loss(*validation); self.history.append(dict(loss, epoch=epoch))
                if loss["combined_ce"] < best:
                    best = loss["combined_ce"]; best_epoch = epoch; chosen = [p.copy() for p in self.params]
        self.params = chosen; self.best_loss = best; self.best_epoch = best_epoch
        return self

    def save(self, path):
        data = {"mean": self.mean, "scale": self.scale, "width": np.array(self.width),
                "context_weight": np.array(self.context_weight), "seed": np.array(self.seed)}
        data.update({f"p{i}": p for i, p in enumerate(self.params)}); np.savez_compressed(path, **data)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            model = cls(int(z["width"]), int(z["seed"]), float(z["context_weight"]))
            model.mean = z["mean"].copy(); model.scale = z["scale"].copy()
            model.params = [z[f"p{i}"].copy() for i in range(8)]
        if not all(np.isfinite(v).all() for v in (model.mean, model.scale, *model.params)):
            raise ValueError("nonfinite frozen multitask model")
        if np.any(model.scale <= 0): raise ValueError("positive normalization required")
        return model
