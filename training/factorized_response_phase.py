"""Direction observer separate from response strength; no contact/XYZ output.

Conditional direction scores only rank loading versus withdrawal *within an
action-like response*. They are never interpreted as contact probabilities.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from training.response_phase_state import ResponsePhaseObserver


@dataclass(frozen=True)
class DirectionPolicy:
    conditional_limit: float = .85
    activity_limit: float = .35
    confirm_s: float = .03
    residue_tail_s: float = .10
    gap_s: float = .25
    reuse_confirmed_direction: bool = False

    def __post_init__(self):
        if not 0. < self.conditional_limit <= 1. or not 0. < self.activity_limit <= 1.:
            raise ValueError("normalized direction score limits required")
        if any(not np.isfinite(v) or v < 0. for v in (self.confirm_s, self.residue_tail_s, self.gap_s)):
            raise ValueError("finite nonnegative durations required")


class FactorizedPhaseObserver:
    def __init__(self, policy=DirectionPolicy()):
        self.policy = policy
        self.reset()

    def reset(self):
        self.last_time = None
        self.last_active_time = None
        self.pending = 0
        self.pending_start = None
        self.pending_frames = 0
        self.phase = 0
        self.token = None

    def step(self, time_ns, strength_state, direction_scores, conditional_scores,
             *, response, signed_amplitude_change_uT, change_limit_uT,
             eligible=True, reset_token=None):
        d = ResponsePhaseObserver.check_scores(direction_scores, 3)
        binary = ResponsePhaseObserver.check_scores(conditional_scores, 2)
        if strength_state not in range(7):
            raise ValueError("known response strength state required")
        if not np.isfinite(signed_amplitude_change_uT) or not np.isfinite(change_limit_uT) or change_limit_uT < 0.:
            raise ValueError("finite signed change and nonnegative activity limit required")
        now = int(time_ns)
        if not eligible:
            self.reset()
            return {"phase": 0, "proposal": 0, "residual_conflict": False}
        if self.token != reset_token or (self.last_time is not None and
            (now <= self.last_time or now - self.last_time > self.policy.gap_s * 1e9)):
            self.reset()
        self.token = reset_token
        self.last_time = now
        if response and strength_state in (2, 3, 4, 6):
            self.last_active_time = now
        near_active = self.last_active_time is not None and now - self.last_active_time <= self.policy.residue_tail_s * 1e9
        activity = d[0] + d[1] >= self.policy.activity_limit
        signed = signed_amplitude_change_uT
        loading = activity and binary[0] >= self.policy.conditional_limit and signed > change_limit_uT
        withdrawal = activity and binary[1] >= self.policy.conditional_limit and signed < -change_limit_uT
        proposal, conflict = 0, False
        if not response:
            proposal = 1 if strength_state == 1 else 0
        elif strength_state == 5:
            # A short recent-action tail may be direction-like even when the
            # context head already favors memory. Preserve that disagreement;
            # it is not evidence that contact persists and never arms XYZ.
            if near_active and withdrawal:
                proposal, conflict = 4, True
            else:
                proposal = 5
        elif strength_state in (2, 3, 4, 6):
            if strength_state == 2 or loading:
                proposal = 2
            elif strength_state == 4 or withdrawal:
                proposal = 4
            else:
                proposal = 3
        if proposal != self.pending:
            self.pending = proposal
            self.pending_start = now
            self.pending_frames = 1
        else:
            self.pending_frames += 1
        # Input strength already includes a causal quiet confirmation.
        reused = (self.policy.reuse_confirmed_direction and
                  strength_state in (2, 4) and proposal == strength_state)
        if proposal in (0, 1, 5) or reused:
            self.phase = proposal
        elif self.pending_frames >= 2 and now - self.pending_start >= self.policy.confirm_s * 1e9:
            self.phase = proposal
        else:
            self.phase = 0
        return {"phase": self.phase, "proposal": proposal,
                "residual_conflict": bool(conflict and self.phase == 4)}


def replay_factorized(time, strength, direction, conditional, response, change,
                      change_limit, eligible, policy=DirectionPolicy()):
    n = len(time)
    if np.asarray(direction).shape != (n, 3) or np.asarray(conditional).shape != (n, 2) or any(
            np.asarray(a).shape != (n,) for a in (time, strength, response, change, eligible)):
        raise ValueError("aligned factorized response histories required")
    phase, proposal, conflict = np.zeros(n, np.int8), np.zeros(n, np.int8), np.zeros(n, bool)
    observer = FactorizedPhaseObserver(policy)
    for i in range(n):
        result = observer.step(time[i], int(strength[i]), direction[i], conditional[i],
                               response=response[i], signed_amplitude_change_uT=change[i],
                               change_limit_uT=change_limit, eligible=eligible[i])
        phase[i], proposal[i], conflict[i] = result["phase"], result["proposal"], result["residual_conflict"]
    return phase, proposal, conflict
