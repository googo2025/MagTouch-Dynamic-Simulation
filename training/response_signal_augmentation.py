"""Causal low-frequency response-gain proxy with unchanged fast residual.

This is augmentation, not a physically identified signal/noise decomposition.
Unlike multiplying the whole trace, it does not intentionally scale all fast
acquisition fluctuations. No labels or target statistics enter the transform.
"""
from __future__ import annotations
import numpy as np
from training.probe_r245_joint_contact_state import causal_state_features
from training.audit_r263_magnetic_evidence_uncertain_depth import evidence_scores, confirm_evidence


def causal_response_component(time, field, eligible, tau_s=.05):
    t = np.asarray(time, np.int64); b = np.asarray(field, float); ok = np.asarray(eligible, bool)
    if t.ndim != 1 or not len(t) or b.shape != (len(t), 15) or ok.shape != t.shape or not np.isfinite(b).all() or np.any(np.diff(t) < 0):
        raise ValueError("finite chronological 15D field and eligibility required")
    if not np.isfinite(tau_s) or tau_s <= 0: raise ValueError("positive response-component time constant required")
    slow = np.empty_like(b)
    for i in range(len(t)):
        reset = i == 0 or not ok[i] or not ok[i-1] or t[i] <= t[i-1] or t[i]-t[i-1] > 250_000_000
        if reset: slow[i] = b[i]
        else:
            rho = np.exp(-(int(t[i])-int(t[i-1]))/1e9/tau_s)
            slow[i] = rho*slow[i-1]+(1.-rho)*b[i]
    return slow


def signal_gain_field(time, field, eligible, gain, tau_s=.05):
    if not np.isfinite(gain) or gain <= 0: raise ValueError("finite positive magnetic response gain required")
    field = np.asarray(field, float)
    if gain == 1.:
        # Validate even the identity case; preserve its values exactly.
        causal_response_component(time, field, eligible, tau_s)
        return field.copy()
    slow = causal_response_component(time, field, eligible, tau_s)
    return field+(gain-1.)*slow


def augmented_signal_row(row, gain, null, tau_s=.05):
    value = dict(row)
    value["field_uT"] = signal_gain_field(row["time"], row["field_uT"], row["eligible"], gain, tau_s)
    value["state_x"] = causal_state_features(value["time"], value["field_uT"], value["eligible"])
    a, d = evidence_scores(value, null); value["response"] = confirm_evidence(value, a, d, null)
    value["amplitude_uT"] = np.linalg.norm(value["field_uT"], axis=1)
    return value
