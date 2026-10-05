"""One magnetic stream, unchanged R373 primary and frozen virtual-XY shadow.

No device or stage API. The shadow cannot alter the primary model, B0, output,
motion, display transformation or checkpoint selection.
"""
from __future__ import annotations
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from live_retrained_reconstruction_r373 import (
    ROOT, COMPONENTS, PlateauCausalXY, DirectTrainedEngineR373, LiveRetrainedPredictorR373,
    sha, GATE_NAME,
)

ARM = ROOT / "analysis/r379_full_virtual_expansion_20261002/arm_1000"
CANDIDATE_SHA = "2eadbba142db96654e7a4c215c536f414b05cee14b505f13d16705cfda97533e"
CANDIDATE_FAMILY = "VirtualExpandedXY1000_R385_R373Z"


def candidate_contract():
    f = json.loads((ARM / "freeze.json").read_text(encoding="utf-8"))
    if (f["status"] != "COMPLETED_PLATEAU" or f["best_epoch"] != 143 or f["epochs_run"] != 933
            or f["target_fit_or_selection"] or f["full_xyz_retrained"]
            or f["model_sha256"] != CANDIDATE_SHA or sha(ARM / "best_model.npz") != CANDIDATE_SHA):
        raise ValueError("R385扩集候选与预先固定的1000条/第143轮权重不一致")
    return {"primary_xy_sha256":COMPONENTS["xy"]["model_sha256"],
            "candidate_xy_sha256":CANDIDATE_SHA, "shared_z_sha256":COMPONENTS["z"]["model_sha256"],
            "virtual_training_paths":1000, "candidate_selected_epoch":143, "epochs_completed":933,
            "changed":"both XY routes only; full chain independently recomputed",
            "same_input_timestamps_baseline_and_causal_hints":True,
            "selection":"fixed largest coverage arm before prospective collection; not chosen using new real results",
            "candidate_is_not_production_replacement":True}


class VirtualXY(PlateauCausalXY):
    def __init__(self):
        candidate_contract()
        # Reuse the original array/normalization checks, then replace only the
        # same-architecture frozen weights. No optimizer or training is imported.
        super().__init__()
        with np.load(ARM / "best_model.npz", allow_pickle=False) as a:
            for key in ("mean","scale","w1","b1","w2","b2","w3","b3"):
                v = a[key].copy()
                if v.shape != getattr(self,key).shape or not np.isfinite(v).all():
                    raise ValueError("Invalid candidate array: " + key)
                if key in ("mean","scale"):
                    np.testing.assert_allclose(v,getattr(self,key),atol=1e-12,rtol=0)
                v.flags.writeable = False
                setattr(self,key,v)


class VirtualEngine(DirectTrainedEngineR373):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.memory_decoder.xy_model = VirtualXY()
        self.candidate_decoder.xy_model = VirtualXY()
        self.reset_unified()

    def push(self,*args,**kwargs):
        r = super().push(*args,**kwargs)
        r["family"] = CANDIDATE_FAMILY
        r["extra"].update(fit_variant="R373_with_frozen1000virtual_XY_only", xy_model_sha256=CANDIDATE_SHA,
                           z_increment_model_sha256=COMPONENTS["z"]["model_sha256"])
        return r


class VirtualPredictor(LiveRetrainedPredictorR373):
    family = CANDIDATE_FAMILY
    def __init__(self,profile,path,preset,threshold,min_cosine,smoothing_ms,
                 contact_threshold=None,jump_policy=GATE_NAME):
        super().__init__(profile,path,preset,threshold,min_cosine,smoothing_ms,contact_threshold,jump_policy)
        self.engine = VirtualEngine(profile,threshold,min_cosine,smoothing_ms,contact_threshold)
        self.metadata.update(xy_mlp_artifact_sha256=CANDIDATE_SHA,
                             paired_contract=candidate_contract(), validation_status="prospective result pending")
        self.begin_baseline()


class PairedPredictorR385(LiveRetrainedPredictorR373):
    """Still an exact R373 primary for the inherited frozen preflight contract."""
    def __init__(self,*args,**kwargs):
        self.shadow = None
        self._suspend_shadow_baseline = False
        super().__init__(*args,**kwargs)
        self.shadow = VirtualPredictor(*args,**kwargs)
        self.metadata["prospective_paired_r385"] = candidate_contract()
        self.model_summary += " | R385：同一输入同步记录1000条虚拟扩集XY研究副本（Z为原R373）"
        self.begin_baseline()

    def begin_baseline(self,seconds=1.):
        super().begin_baseline(seconds)
        if self.shadow is not None and not self._suspend_shadow_baseline:
            self.shadow.begin_baseline(seconds)

    def set_no_contact_hint(self,value,reason=""):
        super().set_no_contact_hint(value,reason)
        if self.shadow is not None:
            self.shadow.set_no_contact_hint(value,reason)

    def request_confirmed_baseline(self,seconds=.4,reason="pre_action_b0"):
        # The base implementation calls begin_baseline; do not reset the
        # shadow's required no-contact confirmation before its own request.
        self._suspend_shadow_baseline = True
        try:
            super().request_confirmed_baseline(seconds,reason)
            if self.shadow is not None:
                self.shadow.request_confirmed_baseline(seconds,reason)
        finally:
            self._suspend_shadow_baseline = False

    def set_xy_rigid_transform(self,*args,**kwargs):
        super().set_xy_rigid_transform(*args,**kwargs)
        if self.shadow is not None:
            self.shadow.set_xy_rigid_transform(*args,**kwargs)

    def push_group(self,data,t_ns):
        primary = super().push_group(data,t_ns)
        secondary = self.shadow.push_group(data,t_ns)
        if len(primary) != len(secondary) or self.ready != self.shadow.ready:
            raise RuntimeError("双模型时间/B0状态不同步，不能继续比较")
        if self.ready:
            np.testing.assert_allclose(self.b0,self.shadow.b0,atol=0,rtol=0)
        output = []
        for a,b in zip(primary,secondary):
            if a.t_ns != b.t_ns:
                raise RuntimeError("双模型输出时间不一致")
            extra = dict(a.extra_outputs)
            extra["prospective_shadow_r385"] = asdict(b)
            output.append(replace(a,extra_outputs=extra))
        return output
