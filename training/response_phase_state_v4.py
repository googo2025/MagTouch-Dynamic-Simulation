"""Source-selected re-entry evidence guard, offline response observer only.

It distinguishes direction-like magnetic responses, not physical contact.
No command, device identity, future data, or XYZ is an input.
"""
from __future__ import annotations

import numpy as np
from training.response_phase_state_v3 import RearmPhaseObserver


class EvidenceRearmObserver(RearmPhaseObserver):
    def reset(self):
        super().reset()
        self.last_fresh_rise = None

    def step(self, time_ns, context_scores, direction_scores, *, response,
             fresh_rise=False, amplitude_uT=0., eligible=True, reset_token=None):
        now = int(time_ns)
        if (not eligible or self.token != reset_token or
            (self.last_time is not None and
             (now <= self.last_time or now-self.last_time > self.policy.gap_s*1e9))):
            self.reset()
        if eligible and fresh_rise:
            self.last_fresh_rise = now
        recent_rise = self.last_fresh_rise is not None and now-self.last_fresh_rise <= 500_000_000
        was_residue = self.state == 5
        result = super().step(now, context_scores, direction_scores, response=response,
            fresh_rise=bool(recent_rise), amplitude_uT=amplitude_uT,
            eligible=eligible, reset_token=reset_token)
        # Block the generic weak->loading shortcut specifically after residue.
        # Do not gate supported responses, or the first weak loading from quiet.
        rejected = bool(eligible and was_residue and not recent_rise and
                        result["proposal"] == 2)
        if rejected:
            self.state, self.pending = 5, 0
            self.pending_start, self.pending_frames = None, 0
            result.update(state=5, proposal=0, carried=True)
        result["reentry_rejected"] = rejected
        return result


def replay_evidence_rearm(time, context, direction, response, fresh, amplitude,
                          eligible, policy):
    n = len(time)
    observer = EvidenceRearmObserver(policy)
    state, rejected = np.zeros(n, np.int8), np.zeros(n, bool)
    for i in range(n):
        v = observer.step(time[i], context[i], direction[i], response=response[i],
            fresh_rise=fresh[i], amplitude_uT=amplitude[i], eligible=eligible[i])
        state[i], rejected[i] = v["state"], v["reentry_rejected"]
    return state, rejected
