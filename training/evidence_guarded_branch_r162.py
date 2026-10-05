"""Causal branch selection with an absolute magnetic-evidence veto.

The R138 spatial prior can otherwise keep a tracker on an implausible local
branch after a wrong turn.  This class makes no geometric assumption about a
line, circle, speed, or commanded path.
"""
from __future__ import annotations

import numpy as np

from training.branch_aware_inverse_r138 import BranchAwareInverse


class EvidenceGuardedBranchInverse(BranchAwareInverse):
    def __init__(self, *args, max_local_cost_ratio: float = 2.5, **kwargs):
        self.max_local_cost_ratio = float(max_local_cost_ratio)
        if not np.isfinite(self.max_local_cost_ratio) or self.max_local_cost_ratio < 1:
            raise ValueError("max_local_cost_ratio must be finite and >= 1")
        super().__init__(*args, **kwargs)

    def predict_frame(self, *args, **kwargs):
        row = super().predict_frame(*args, **kwargs)
        if row is None:
            self.last_veto = False
            return None
        local = row["local_field_cost"]
        global_ = row["global_field_cost"]
        if (row["used_local_branch"] and
                local > self.max_local_cost_ratio * max(global_, 1e-9)):
            # Reject the high-residual continuation and update state to the
            # selected branch immediately. No output smoothing or hindsight.
            selected = np.asarray(row["global_xyz_mm"], float).copy()
            self.previous = selected.copy()
            row["xyz_mm"] = selected
            row["used_local_branch"] = False
            row["global_branch_switch_accepted"] = True
            row["local_branch_vetoed_by_field"] = True
        else:
            row["local_branch_vetoed_by_field"] = False
        self.last_veto = row["local_branch_vetoed_by_field"]
        return row
