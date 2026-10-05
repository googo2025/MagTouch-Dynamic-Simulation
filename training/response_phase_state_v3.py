"""Causal re-loading confirmation and normal-response continuity, shadow only."""
from __future__ import annotations

import numpy as np
from training.response_phase_state import PhasePolicy, ResponsePhaseObserver


class RearmPhaseObserver(ResponsePhaseObserver):
    def reset(self):
        super().reset()
        self.active_anchor = False
        self.last_withdrawal_time = None
        self.token = None

    def step(self, time_ns, context_scores, direction_scores, *, response,
             fresh_rise=False, amplitude_uT=0., eligible=True, reset_token=None):
        c = self.check_scores(context_scores, 4)
        d = self.check_scores(direction_scores, 3)
        now = int(time_ns)
        if not np.isfinite(amplitude_uT) or amplitude_uT < 0.:
            raise ValueError("nonnegative finite magnetic amplitude required")
        if not eligible:
            self.reset()
            return {"state": 0, "proposal": 0, "carried": False, "continuity_guard": False}
        if (self.token != reset_token or (self.last_time is not None and
            (now <= self.last_time or now - self.last_time > self.policy.gap_s * 1e9))):
            self.reset()
        self.token = reset_token
        if self.active_anchor and response and d[1] >= self.policy.direction_limit and d[1] > d[0] and d[1] > d[2]:
            self.last_withdrawal_time = now
        had_withdrawal = self.last_withdrawal_time is not None and now - self.last_withdrawal_time <= 2_000_000_000
        adjusted = c.copy()
        continuity_guard = False
        if response and self.state in (2, 3, 6) and amplitude_uT >= 140. and d[1] < self.policy.direction_limit:
            # Existing large-signal preservation criterion, conditioned on a
            # previously recognized action. It cannot arm from residue alone.
            adjusted = np.array([0., 0., 1., 0.])
            continuity_guard = True
        elif self.active_anchor and response and not had_withdrawal and d[1] < self.policy.direction_limit and c[0] < .5:
            adjusted = np.array([0., 0., 1., 0.])
            continuity_guard = True
        elif self.active_anchor and response and had_withdrawal and d[1] >= self.policy.direction_limit and c[2] < self.policy.supported_limit:
            adjusted = np.array([0., 1., 0., 0.])
        elif not self.active_anchor and response and fresh_rise and d[0] >= .85 and d[0] > d[2] and c[1] >= .15 and c[2] < self.policy.supported_limit:
            adjusted = np.array([0., 1., 0., 0.])
        result = super().step(now, adjusted, d, response=response, eligible=True)
        if result["state"] == 6:
            self.active_anchor = True
        if result["state"] in (1, 5):
            self.active_anchor = False
        if result["state"] == 4:
            self.last_withdrawal_time = now
        result["continuity_guard"] = continuity_guard
        return result


def replay_rearm_phase(time, context, direction, response, fresh_rise,
                       amplitude, eligible, policy=PhasePolicy()):
    time = np.asarray(time, np.int64)
    context, direction = np.asarray(context, float), np.asarray(direction, float)
    n = len(time)
    if context.shape != (n, 4) or direction.shape != (n, 3) or any(np.asarray(v).shape != (n,) for v in (response, fresh_rise, amplitude, eligible)):
        raise ValueError("aligned state histories required")
    observer = RearmPhaseObserver(policy)
    arrays = [np.zeros(n, np.int8), np.zeros(n, np.int8), np.zeros(n, bool), np.zeros(n, bool)]
    for i in range(n):
        r = observer.step(time[i], context[i], direction[i], response=response[i], fresh_rise=fresh_rise[i],
                          amplitude_uT=amplitude[i], eligible=eligible[i])
        for a, name in zip(arrays, ("state", "proposal", "carried", "continuity_guard")):
            a[i] = r[name]
    return tuple(arrays)
