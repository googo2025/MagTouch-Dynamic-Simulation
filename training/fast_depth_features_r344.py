"""Causal vertical innovation after removing predicted lateral field change."""
from bisect import bisect_right
from collections import deque
import numpy as np


class FastDepthContext:
    def __init__(self, atlas):
        self.atlas = atlas; self.reset()

    def reset(self):
        self.history = deque(); self.token = None

    def push(self, time_ns, x56, *, token=0):
        x = np.asarray(x56, float)
        if x.shape != (56,) or not np.isfinite(x).all():
            raise ValueError('Finite causal 56D input required')
        now = int(time_ns); q = x[:3]*np.array([10.,10.,3.]); phi = x[3:19]
        if (self.token != token or (self.history and
                not 0 < (now-self.history[-1][0])/1e9 <= .25)):
            self.reset()
        self.token = token
        times = [v[0] for v in self.history]
        index = bisect_right(times, now-100_000_000)-1
        extra = np.zeros(3)
        if index >= 0:
            previous_t, previous_q, previous_phi = self.history[index]
            dt = (now-previous_t)/1e9
            if 0 < dt <= .25:
                current_xy = np.r_[q[:2],previous_q[2]]
                probes = np.vstack([current_xy,previous_q,current_xy,current_xy])
                probes[2,2] += .02; probes[3,2] -= .02
                f = self.atlas.features(self.atlas.forward(probes))
                delta = phi-previous_phi-(f[0]-f[1]); tangent = (f[2]-f[3])/.04
                dot = float(delta@tangent); norm = float(tangent@tangent)
                signed = dot/max(norm,1e-9)/dt
                alignment = abs(dot)/max(np.linalg.norm(delta)*np.sqrt(norm),1e-9)
                extra = np.array([np.arcsinh(np.clip(signed,-20,20)),
                                  min(float(alignment),1.),np.log1p(min(abs(signed),20.))])
        self.history.append((now,q.copy(),phi.copy()))
        while self.history and now-self.history[0][0] > 350_000_000:
            self.history.popleft()
        return np.r_[x,extra]


def augment(atlas, time, x, valid, epoch):
    result = np.zeros((len(time),59)); h = FastDepthContext(atlas)
    for i,t in enumerate(time):
        if not valid[i]: h.reset(); continue
        result[i] = h.push(t,x[i],token=int(epoch[i]))
    return result
