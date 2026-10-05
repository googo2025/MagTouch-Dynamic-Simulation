"""Causal depth-of-atlas-support branch evidence, experimental R223.

The global inverse's *current* depth is the sole scheduler input.  Platform
commands, future observations, trajectory labels, and output smoothing are
not used.  The rule is an engineering hypothesis for the poorly constrained
0.4-mm lower atlas boundary, not an identified material constitutive law.
"""
from __future__ import annotations

import numpy as np

from training.evidence_guarded_branch_r162 import EvidenceGuardedBranchInverse


def ratio_at_global_depth(global_depth_mm):
    fraction = np.clip((float(global_depth_mm)-.6)/.4, 0., 1.)
    return float(1.25 + 2.75*fraction)


class ShallowAdaptiveEvidenceBranch(EvidenceGuardedBranchInverse):
    def predict_frame(self, *args, **kwargs):
        global_xyz = kwargs.get("global_xyz")
        if global_xyz is None:
            raise ValueError("R223 requires the current global magnetic inverse")
        xyz = np.asarray(global_xyz, float).reshape(3)
        if not np.isfinite(xyz).all():
            raise ValueError("R223 global inverse must be finite")
        self.max_local_cost_ratio = ratio_at_global_depth(xyz[2])
        return super().predict_frame(*args, **kwargs)
