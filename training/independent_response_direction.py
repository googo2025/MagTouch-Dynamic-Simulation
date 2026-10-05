"""Causal action-response supplementation without changing strength/XYZ.

The directional classifier includes quiet and residual negative examples.
Its scores are not contact probabilities. A confirmed supplementary phase
may disagree with strength context; preserve that disagreement explicitly.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from training.response_phase_state import ResponsePhaseObserver


@dataclass(frozen=True)
class SupplementPolicy:
    score_limit: float = .9
    confirm_s: float = .03
    permit_residue: bool = False
    require_signed_change: bool = True
    gap_s: float = .25

    def __post_init__(self):
        if not np.isfinite(self.score_limit) or not 0. < self.score_limit <= 1.:
            raise ValueError("finite normalized classifier score threshold required")
        if any(not np.isfinite(v) or v < 0. for v in (self.confirm_s, self.gap_s)):
            raise ValueError("nonnegative finite causal durations required")


class SupplementaryDirectionObserver:
    def __init__(self, policy=SupplementPolicy()):
        self.policy = policy; self.reset()

    def reset(self):
        self.last_time = None; self.token = None
        self.pending = 0; self.pending_start = None; self.pending_frames = 0

    def step(self, time_ns, baseline_phase, strength, direction_scores, *, response,
             signed_change_uT, eligible=True, reset_token=None):
        scores = ResponsePhaseObserver.check_scores(direction_scores, 3)
        if baseline_phase not in range(6) or strength not in range(7):
            raise ValueError("known magnetic-response states required")
        if not np.isfinite(signed_change_uT):
            raise ValueError("finite magnetic change required")
        now = int(time_ns)
        if not eligible:
            self.reset()
            return {"phase": 0, "supplemented": False, "context_conflict": False}
        if self.token != reset_token or (self.last_time is not None and
            (now <= self.last_time or now-self.last_time > self.policy.gap_s*1e9)):
            self.reset()
        self.token, self.last_time = reset_token, now
        cls = int(scores.argmax()); proposed = 2 if cls == 0 else 4 if cls == 1 else 0
        allowed = bool(response and proposed and scores[cls] >= self.policy.score_limit and
            (strength != 5 or self.policy.permit_residue))
        if self.policy.require_signed_change and proposed:
            allowed &= signed_change_uT > 0. if proposed == 2 else signed_change_uT < 0.
        # Preserve all previously declared action phases, including any
        # documented baseline errors. Supplementing cannot silently erase them.
        if baseline_phase in (2, 4) or not allowed:
            self.pending, self.pending_start, self.pending_frames = 0, None, 0
            return {"phase": baseline_phase, "supplemented": False, "context_conflict": False}
        if proposed != self.pending:
            self.pending, self.pending_start, self.pending_frames = proposed, now, 1
        else:
            self.pending_frames += 1
        confirmed = self.pending_frames >= 2 and now-self.pending_start >= self.policy.confirm_s*1e9
        return {"phase": proposed if confirmed else baseline_phase,
            "supplemented": bool(confirmed),
            "context_conflict": bool(confirmed and strength not in (2, 3, 4, 6))}


def replay_supplement(time, baseline_phase, strength, scores, response, change, eligible,
                      policy=SupplementPolicy()):
    n = len(time)
    if np.asarray(scores).shape != (n, 3) or any(np.asarray(v).shape != (n,) for v in (
        time, baseline_phase, strength, response, change, eligible)):
        raise ValueError("aligned causal magnetic-response histories required")
    time, scores = np.asarray(time, np.int64), np.asarray(scores, float)
    baseline_phase, strength = np.asarray(baseline_phase, np.int8), np.asarray(strength, np.int8)
    response, eligible, change = np.asarray(response, bool), np.asarray(eligible, bool), np.asarray(change, float)
    if not np.isfinite(scores).all() or np.any(scores < 0.) or np.any(np.abs(scores.sum(1)-1.) > 1e-5):
        raise ValueError("finite normalized classifier scores required")
    if not np.isin(baseline_phase, np.arange(6)).all() or not np.isin(strength, np.arange(7)).all() or not np.isfinite(change).all():
        raise ValueError("known states and finite change required")
    if not n:
        return np.zeros(0, np.int8), np.zeros(0, bool), np.zeros(0, bool)
    cls = scores.argmax(1)
    proposed = np.where(cls == 0, 2, np.where(cls == 1, 4, 0)).astype(np.int8)
    allowed = eligible & response & (proposed != 0) & (scores[np.arange(n), cls] >= policy.score_limit)
    allowed &= (strength != 5) | policy.permit_residue
    allowed &= ~np.isin(baseline_phase, (2, 4))
    if policy.require_signed_change:
        allowed &= np.where(proposed == 2, change > 0., change < 0.)
    proposed[~allowed] = 0
    # These are contiguous causal runs, equivalent to the streaming observer.
    reset = np.r_[True, (proposed[1:] != proposed[:-1]) | ~eligible[1:] | ~eligible[:-1] |
        (np.diff(time) <= 0) | (np.diff(time) > policy.gap_s*1e9)]
    starts = np.maximum.accumulate(np.where(reset, np.arange(n), 0))
    added = allowed & (np.arange(n)-starts >= 1) & (time-time[starts] >= policy.confirm_s*1e9)
    phase = baseline_phase.copy(); phase[~eligible] = 0; phase[added] = proposed[added]
    conflict = added & ~np.isin(strength, (2, 3, 4, 6))
    return phase, added, conflict
