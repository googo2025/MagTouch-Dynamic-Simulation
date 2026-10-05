"""One output contract for action evidence and selective XYZ reconstruction."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class LocalizationPolicy:
    # Bounds on emitted coordinates, NOT on commanded contact depth. Magnetic
    # support decides validity; .4 mm atlas training is not a suppression line.
    depth_min_mm: float = 0.
    depth_max_mm: float = 3.
    xy_limit_mm: float = 10.

    def __post_init__(self):
        if not (np.isfinite((self.depth_min_mm, self.depth_max_mm, self.xy_limit_mm)).all()
                and 0 <= self.depth_min_mm < self.depth_max_mm and self.xy_limit_mm > 0):
            raise ValueError("Finite ordered localization domain required")


class UnifiedOutputPolicy:
    """Never arm XYZ from loading/withdrawal recognition alone.

    No smoothing, template fitting, stale-coordinate hold or shallow XYZ labels.
    Magnetic support is a necessary check, not a calibrated error probability.
    """
    def __init__(self, policy=LocalizationPolicy()):
        self.policy = policy
        self.reset()

    def reset(self):
        self.was_valid = False

    def decide(self, response, candidate_xyz=None, *, decoder_valid=False):
        reasons = []
        strength, phase = int(response.get("strength", 0)), int(response.get("phase", 0))
        if strength not in range(7) or phase not in range(6):
            raise ValueError("Known response states required")
        if not response.get("eligible", False): reasons.append("invalid_magnetic_input")
        if not response.get("response", False): reasons.append("unconfirmed_magnetic_response")
        if strength != 6: reasons.append("localization_support_not_confirmed")
        if not decoder_valid: reasons.append("trajectory_decoder_invalid")
        xyz = None if candidate_xyz is None else np.asarray(candidate_xyz, float)
        if xyz is None or xyz.shape != (3,) or not np.isfinite(xyz).all():
            reasons.append("no_finite_candidate")
        elif (abs(xyz[0]) > self.policy.xy_limit_mm or abs(xyz[1]) > self.policy.xy_limit_mm
              or not self.policy.depth_min_mm-1e-9 <= xyz[2] <= self.policy.depth_max_mm+1e-9):
            reasons.append("outside_localization_domain")
        valid = not reasons
        state = ("position" if valid else "invalid_input" if not response.get("eligible", False)
                 else "residual" if strength == 5
                 else {2: "loading_unlocalized", 3: "weak_hold_unlocalized", 4: "withdrawal_unlocalized"}.get(phase,
                 "quiet" if strength == 1 else "uncertain"))
        result = {"system_state": state, "xyz_valid": valid,
                  "xyz_mm": xyz.tolist() if valid else None,
                  "action_phase": phase, "magnetic_strength": strength,
                  "rejection_reasons": reasons,
                  "trace_break": not valid or not self.was_valid,
                  "localization_confidence_semantics": "selective evidence policy; not a calibrated error probability"}
        self.was_valid = valid
        return result
