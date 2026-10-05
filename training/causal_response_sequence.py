"""Bounded causal magnetic-history sampling and a small NumPy GRU.

Inputs are magnetic features and acquisition time only. Fixed past-time taps
are clamped to the latest eligible contiguous acquisition segment; padding
repeats its first observation and explicitly has zero elapsed time. No future
frames, action template, command, contact truth or device identifier is read.
"""
from __future__ import annotations
import numpy as np
from training.probe_r278_independent_direction import features as magnetic_features
from training.probe_r245_joint_contact_state import softmax


def causal_taps(time, eligible, steps=16, step_s=.1):
    time = np.asarray(time, np.int64); eligible = np.asarray(eligible, bool)
    if time.ndim != 1 or not len(time) or eligible.shape != time.shape or np.any(np.diff(time) < 0):
        raise ValueError("chronological timestamps and eligibility required")
    if not isinstance(steps, (int, np.integer)) or steps < 2 or not np.isfinite(step_s) or step_s <= 0:
        raise ValueError("positive past sampling interval and >=2 steps required")
    gap = np.diff(time); reset = ~eligible.copy(); reset[0] = True
    reset[1:] |= ~eligible[:-1] | (gap <= 0) | (gap > 250_000_000)
    start = np.maximum.accumulate(np.where(reset, np.arange(len(time)), 0))
    lag = np.arange(steps-1, -1, -1, dtype=np.int64)*int(round(step_s*1e9))
    ids = np.searchsorted(time, time[:, None]-lag[None, :], side="right")-1
    ids = np.minimum(np.maximum(ids, start[:, None]), np.arange(len(time))[:, None])
    # Duplicate clocks must use this acquisition, not another equal-time frame.
    ids[:, -1] = np.arange(len(time))
    return ids


def sequence_features(row, variant="relative", steps=16, step_s=.1, ids=None):
    current = magnetic_features(row, "relative")
    if variant == "ratio": current = current[:, 16:]
    elif variant != "relative": raise ValueError("known sequence feature variant required")
    taps = causal_taps(row["time"], row["eligible"], steps, step_s)
    if ids is not None: taps = taps[np.asarray(ids, int)]
    elapsed = np.zeros(taps.shape, float)
    elapsed[:, 1:] = np.diff(row["time"][taps], axis=1)/1e9/step_s
    return np.concatenate((current[taps], np.clip(elapsed, 0., 5.)[..., None]), axis=2)


def sigmoid(x):
    return 1./(1.+np.exp(-np.clip(x, -40., 40.)))


class CausalGRU:
    """Last-observation classifier; bounded magnetic past, not streaming BPTT."""
    def __init__(self, width=16, seed=282, n_classes=3):
        self.width = int(width); self.seed = int(seed); self.n_classes = int(n_classes)

    def initialize(self, inputs):
        rng = np.random.default_rng(self.seed); h = self.width
        p = []
        for gate in range(3):
            p.extend((rng.normal(0., np.sqrt(1./inputs), (inputs, h)),
                      rng.normal(0., np.sqrt(1./h), (h, h)),
                      np.ones(h) if gate == 0 else np.zeros(h)))
        p.extend((rng.normal(0., np.sqrt(1./h), (h, self.n_classes)), np.zeros(self.n_classes)))
        self.params = p

    def forward(self, v, cache=False):
        p = self.params; h = np.zeros((len(v), self.width)); states = []
        for k in range(v.shape[1]):
            x = v[:, k]; old = h
            z = sigmoid(x@p[0]+old@p[1]+p[2]); r = sigmoid(x@p[3]+old@p[4]+p[5])
            candidate = np.tanh(x@p[6]+(r*old)@p[7]+p[8])
            h = z*old+(1.-z)*candidate
            if cache: states.append((x, old, z, r, candidate))
        prob = softmax(h@p[9]+p[10])
        return (prob, h, states) if cache else prob

    def loss_gradient(self, v, y, weight, decay=.001):
        p = self.params; prob, h, states = self.forward(v, True)
        loss = float(np.sum(weight*-np.log(np.maximum(prob[np.arange(len(y)), y], 1e-12)))/len(y))
        loss += .5*decay*sum(float(np.sum(a*a)) for a in p if a.ndim == 2)
        g = [np.zeros_like(a) for a in p]
        delta = prob.copy(); delta[np.arange(len(y)), y] -= 1.; delta *= weight[:, None]/len(y)
        g[9] = h.T@delta; g[10] = delta.sum(0); dh = delta@p[9].T
        for x, old, z, r, candidate in reversed(states):
            dn = dh*(1.-z); dz = dh*(old-candidate); dh_old = dh*z
            an = dn*(1.-candidate*candidate)
            g[6] += x.T@an; g[7] += (r*old).T@an; g[8] += an.sum(0)
            dr_old = an@p[7].T; dr = dr_old*old; dh_old += dr_old*r
            ar = dr*r*(1.-r)
            g[3] += x.T@ar; g[4] += old.T@ar; g[5] += ar.sum(0); dh_old += ar@p[4].T
            az = dz*z*(1.-z)
            g[0] += x.T@az; g[1] += old.T@az; g[2] += az.sum(0); dh = dh_old+az@p[1].T
        for i, a in enumerate(p):
            if a.ndim == 2: g[i] += decay*a
        return loss, g

    def normalized(self, x):
        x = np.asarray(x, float)
        if x.ndim != 3 or x.shape[2] != len(self.mean) or not np.isfinite(x).all():
            raise ValueError("finite batch x past-steps x fitted-features required")
        return np.clip((x-self.mean)/self.scale, -8., 8.)

    def predict(self, x):
        value = self.normalized(x)
        return np.vstack([self.forward(value[i:i+512]) for i in range(0, len(value), 512)])

    def fit(self, x, y, weight, validation=None, epochs=80):
        rng = np.random.default_rng(self.seed); x = np.asarray(x, float)
        if x.ndim != 3 or not len(x) or not np.isfinite(x).all(): raise ValueError("finite sequence samples required")
        self.mean = np.average(x.mean(1), axis=0, weights=weight)
        self.scale = np.maximum(np.sqrt(np.average(((x-self.mean)**2).mean(1), axis=0, weights=weight)), .05)
        v = self.normalized(x); self.initialize(x.shape[2]); params = self.params
        first = [np.zeros_like(p) for p in params]; second = [np.zeros_like(p) for p in params]
        best = np.inf; chosen = None; self.history = []; step = 0
        for epoch in range(1, epochs+1):
            for ids in np.array_split(rng.permutation(len(x)), max(1, int(np.ceil(len(x)/256)))):
                _, gradients = self.loss_gradient(v[ids], y[ids], weight[ids]); step += 1
                norm = np.sqrt(sum(float(np.sum(g*g)) for g in gradients))
                if norm > 5.: gradients = [g*5./norm for g in gradients]
                for p, g, m, s in zip(params, gradients, first, second):
                    m *= .9; m += .1*g; s *= .999; s += .001*g*g
                    p -= .001*(m/(1.-.9**step))/(np.sqrt(s/(1.-.999**step))+1e-8)
            if epoch % 5 == 0:
                if validation is None: loss = float(epoch == epochs)*-1.
                else:
                    vx, vy, vw = validation; prob = self.predict(vx)
                    loss = float(np.average(-np.log(np.maximum(prob[np.arange(len(vy)), vy], 1e-9)), weights=vw))
                self.history.append({"epoch": epoch, "weighted_validation_cross_entropy": loss})
                print("GRU epoch", epoch, "validation CE", round(loss, 6), flush=True)
                if loss < best: best = loss; chosen = [p.copy() for p in params]; self.best_epoch = epoch
        self.params = chosen; self.best_loss = best
        return self

    def save(self, path):
        data = {"mean": self.mean, "scale": self.scale, "width": np.array(self.width),
                "n_classes": np.array(self.n_classes), "architecture": np.array("bounded_causal_GRU")}
        data.update({f"p{i}": p for i, p in enumerate(self.params)})
        np.savez_compressed(path, **data)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as a:
            if str(a["architecture"]) != "bounded_causal_GRU": raise ValueError("not a frozen GRU")
            model = cls(int(a["width"]), n_classes=int(a["n_classes"]))
            model.mean = a["mean"].copy(); model.scale = a["scale"].copy()
            model.params = [a[f"p{i}"].copy() for i in range(11)]
        return model
