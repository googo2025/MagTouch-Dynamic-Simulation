"""Hardware-independent causal magnetic-response state; not a contact gate.

The state never creates/vetoes XYZ and never declares physical contact truth.
Neural scores describe response contexts, not calibrated physical probabilities.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

STATE_NAMES = ("indeterminate", "quiet_response", "weak_loading_like",
               "weak_hold_like", "weak_withdrawal_like", "residual_like",
               "supported_response")


@dataclass(frozen=True)
class PhasePolicy:
    weak_limit: float = .75
    supported_limit: float = .55
    residual_limit: float = .65
    quiet_limit: float = .70
    direction_limit: float = .60
    confirm_s: float = .03
    quiet_confirm_s: float = .20
    hold_s: float = .25
    residue_hold_s: float = 1.
    gap_s: float = .25

    def __post_init__(self):
        for name in ("weak_limit", "supported_limit", "residual_limit", "quiet_limit", "direction_limit"):
            if not np.isfinite(getattr(self, name)) or not 0. < getattr(self, name) <= 1.:
                raise ValueError(f"invalid score limit {name}")
        for name in ("confirm_s", "quiet_confirm_s", "hold_s", "residue_hold_s", "gap_s"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0.:
                raise ValueError(f"invalid duration {name}")


class ResponsePhaseObserver:
    def __init__(self, policy=PhasePolicy()):
        self.policy = policy
        self.reset()

    def reset(self):
        self.last_time = None
        self.state = 0
        self.pending = 0
        self.pending_start = None
        self.pending_frames = 0
        self.last_supported_time = None
        self.last_weak_time = None
        self.last_residue_time = None

    @staticmethod
    def check_scores(scores, length):
        p = np.asarray(scores, float)
        if p.shape != (length,) or not np.isfinite(p).all() or np.any(p < 0.) or abs(p.sum() - 1.) > 1e-5:
            raise ValueError("finite normalized nonnegative head scores required")
        return p

    def step(self, time_ns, context_scores, direction_scores, *, response, eligible=True):
        c = self.check_scores(context_scores, 4)
        d = self.check_scores(direction_scores, 3)
        now = int(time_ns)
        if not eligible:
            self.reset()
            return {"state": 0, "proposal": 0, "carried": False}
        if self.last_time is not None and (now <= self.last_time or now - self.last_time > self.policy.gap_s * 1e9):
            self.reset()
        self.last_time = now
        p = self.policy
        proposal = 0
        if not response:
            if c[0] >= p.quiet_limit:
                proposal = 1
        elif c[2] >= p.supported_limit and c[2] >= c[1] and c[2] >= c[3]:
            proposal = 6
        elif c[3] >= p.residual_limit:
            proposal = 5
        elif c[1] >= p.weak_limit and c[1] - c[3] >= .25:
            direction = int(d.argmax())
            proposal = (2 if direction == 0 else 4 if direction == 1 else 3) if d[direction] >= p.direction_limit else 3
        # A new loading-like response or clear supported response can re-enter
        # before the field becomes quiet. Residue is not armed by amplitude.
        if self.state == 5 and proposal in (3, 4):
            proposal = 0
        if proposal != self.pending:
            self.pending = proposal
            self.pending_start = now
            self.pending_frames = 1
        else:
            self.pending_frames += 1
        delay = p.quiet_confirm_s if proposal == 1 else p.confirm_s
        confirmed = (proposal != 0 and self.pending_frames >= 2 and self.pending_start is not None
                     and now - self.pending_start >= delay * 1e9)
        carried = False
        if confirmed:
            self.state = proposal
            if proposal == 6:
                self.last_supported_time = now
            elif proposal in (2, 3, 4):
                self.last_weak_time = now
            elif proposal == 5:
                self.last_residue_time = now
        elif self.state == 5 and response and self.last_residue_time is not None and now - self.last_residue_time <= p.residue_hold_s * 1e9:
            carried = True
        elif self.state in (2, 3, 4) and response and self.last_weak_time is not None and now - self.last_weak_time <= p.hold_s * 1e9:
            self.state = 3
            carried = True
        else:
            self.state = 0
        return {"state": self.state, "proposal": proposal, "carried": carried}


def replay_phase(time, context, direction, response, eligible, policy=PhasePolicy()):
    time = np.asarray(time, np.int64)
    if context.shape != (len(time), 4) or direction.shape != (len(time), 3) or np.asarray(response).shape != (len(time),) or np.asarray(eligible).shape != (len(time),):
        raise ValueError("aligned time and head arrays required")
    observer = ResponsePhaseObserver(policy)
    state = np.zeros(len(time), np.int8)
    proposal = np.zeros(len(time), np.int8)
    carried = np.zeros(len(time), bool)
    for i in range(len(time)):
        result = observer.step(time[i], context[i], direction[i], response=response[i], eligible=eligible[i])
        state[i], proposal[i], carried[i] = result["state"], result["proposal"], result["carried"]
    return state, proposal, carried
