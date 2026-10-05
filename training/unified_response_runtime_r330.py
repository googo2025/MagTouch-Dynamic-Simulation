"""Streaming port of the Source-selected R300 response observer.

No fitting, commands, device identifiers, trajectory templates or future input.
This observer supplies magnetic state evidence, not permission to emit XYZ.
"""
from __future__ import annotations

from bisect import bisect_right
import hashlib
import json
from pathlib import Path
import numpy as np

from training.response_phase_state import PhasePolicy
from training.response_phase_state_v4 import EvidenceRearmObserver
from training.factorized_response_phase import DirectionPolicy, FactorizedPhaseObserver
from training.independent_response_direction import SupplementPolicy, SupplementaryDirectionObserver

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = {
    275: "r275_source_only_zero_shot_20260929",
    276: "r276_source_condition_adaptation_20260929",
    278: "r278_source_independent_direction_20260929",
    282: "r282_source_causal_sequence_20260929",
    288: "r288_source_ordered_action_context_20260929",
    291: "r291_source_supported_context_release_20260929",
    293: "r293_source_decoupled_support_context_20260929",
    300: "r300_source_strength_conditioned_direction_20260929",
}
SCALAR_IDS = list(range(10))+list(range(40, 43))+list(range(58, 61))
LAGS_NS = (100_000_000, 300_000_000, 700_000_000, 2_000_000_000, 5_000_000_000)
GAP_NS = 250_000_000


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_protocol(version):
    path = ROOT/"analysis"/FOLDERS[version]/"freeze.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    # Deployment checks code/model dependencies, not hundreds of raw files.
    for item in value.get("tracked", value.get("selected_files", [])):
        if digest(ROOT/item["path"]) != item["sha256"]:
            raise ValueError("Frozen response dependency changed: "+item["path"])
    for key in ("base_protocol", "network_protocol", "direction_protocol"):
        if key in value and digest(ROOT/value[key]) != value[key+"_sha256"]:
            raise ValueError("Frozen response protocol changed: "+value[key])
    for key in ("target_fit_rows", "target_selection_rows", "target_fit_selection_rows", "source_test_selection_rows", "SOURCE_test_selection_rows"):
        if value.get(key, 0) != 0:
            raise ValueError("Response deployment requires Source-only fitting/selection")
    return value, {"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path)}


class FrozenMLP:
    """Load only finite immutable NumPy classifier parameters."""
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as a:
            self.mean, self.scale = a["mean"].copy(), a["scale"].copy()
            keys = sorted((k for k in a.files if k.startswith("p") and k[1:].isdigit()), key=lambda k: int(k[1:]))
            self.params = [a[k].copy() for k in keys]
        if self.mean.ndim != 1 or self.scale.shape != self.mean.shape or np.any(self.scale <= 0):
            raise ValueError("Invalid frozen normalizer")
        expected = len(self.mean)
        if not self.params or len(self.params)%2:
            raise ValueError("Invalid frozen MLP")
        for w, b in zip(self.params[::2], self.params[1::2]):
            if w.ndim != 2 or w.shape[0] != expected or b.shape != (w.shape[1],):
                raise ValueError("Invalid frozen layer shape")
            expected = w.shape[1]
        if not all(np.isfinite(p).all() for p in [self.mean, self.scale]+self.params):
            raise ValueError("Nonfinite frozen parameters")

    def predict(self, x):
        v = np.clip((np.asarray(x, float).reshape(-1, len(self.mean))-self.mean)/self.scale, -8., 8.)
        for i, (w, b) in enumerate(zip(self.params[::2], self.params[1::2])):
            v = v@w+b
            if i < len(self.params)//2-1:
                v = np.tanh(v)
        exp = np.exp(v-v.max(axis=1, keepdims=True))
        return exp/exp.sum(axis=1, keepdims=True)


class MagneticFeatureHistory:
    """Exact finite causal lags/taps, bounded by acquisition time."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.times, self.fields, self.feature_times, self.features = [], [], [], []
        self.last_time = None
        self.was_eligible = False

    def push(self, time_ns, field, eligible=True):
        t = int(time_ns); b = np.asarray(field, float)
        if b.shape != (15,) or not np.isfinite(b).all():
            raise ValueError("Finite 15D delta-B required")
        reset = (not eligible or not self.was_eligible or self.last_time is None or
                 t <= self.last_time or t-self.last_time > GAP_NS)
        if reset:
            self.reset()
        self.last_time, self.was_eligible = t, bool(eligible)
        self.times.append(t); self.fields.append(b.copy())
        amp = float(np.linalg.norm(b)); blocks, scalars = [], [amp/100.]
        for lag in LAGS_NS:
            j = max(0, bisect_right(self.times, t-lag)-1)
            delta = b-self.fields[j]
            three = ((amp-np.linalg.norm(self.fields[j]))/100.,
                     np.linalg.norm(delta)/100., np.dot(b, delta)/(100.*max(amp, 1.)))
            blocks.append((delta, three))
            if len(blocks) <= 3:
                scalars.extend(three)
        state = np.concatenate((np.asarray(scalars), b/max(amp, 1.), blocks[1][0]/100.,
                                np.asarray(blocks[3][1]), blocks[3][0]/100.,
                                np.asarray(blocks[4][1]), blocks[4][0]/100.))
        scalar = state[SCALAR_IDS]
        relative = np.r_[scalar, np.clip(scalar[1:]/max(scalar[0], .2), -10., 10.)]
        self.feature_times.append(t); self.features.append(relative)
        ids = [max(0, bisect_right(self.feature_times, t-lag)-1)
               for lag in range(1_500_000_000, -1, -100_000_000)]
        ids[-1] = len(self.feature_times)-1
        tap_times = np.asarray([self.feature_times[j] for j in ids], np.int64)
        elapsed = np.r_[0., np.diff(tap_times)/100_000_000.]
        seq = np.column_stack(([self.features[j] for j in ids], np.clip(elapsed, 0., 5.))).reshape(1, -1)
        # Retain the predecessor at each oldest lag; never truncate a lag early.
        for times, values, horizon in ((self.times, self.fields, 5_000_000_000),
                                       (self.feature_times, self.features, 1_500_000_000)):
            remove = max(0, bisect_right(times, t-horizon)-1)
            if remove:
                del times[:remove]; del values[:remove]
        return {"state": state, "scalar": scalar, "relative": relative, "sequence": seq,
                "changes": np.asarray([v[0] for v in blocks[:3]]), "amplitude_uT": amp,
                "reset": reset}


class CausalConfirmation:
    def __init__(self, confirm_s):
        self.confirm_s = float(confirm_s); self.reset()

    def reset(self):
        self.start, self.frames, self.last_time = None, 0, None

    def step(self, t, active):
        if not active or (self.last_time is not None and (t <= self.last_time or t-self.last_time > GAP_NS)):
            self.reset()
        self.last_time = t
        if not active:
            return False
        if self.start is None:
            self.start = t
        self.frames += 1
        return self.frames >= 2 and t-self.start >= self.confirm_s*1e9


class StreamingSourceResponse:
    """R300 is selected by Source validation, not by its Yellow score."""
    def __init__(self):
        protocols = {}; self.manifest = []
        for version in FOLDERS:
            protocols[version], entry = frozen_protocol(version); self.manifest.append(entry)
        self.protocols = protocols
        source = protocols[275]
        with np.load(ROOT/source["null"], allow_pickle=False) as a:
            self.null = {k: a[k].copy() for k in a.files}
        self.fresh_limit = float(source["fresh_rise_limit_uT"])
        self.change_limit = float(source["activity_change_limit_uT"])
        self.strength_models = [FrozenMLP(ROOT/f) for f in protocols[276]["models"]]
        self.direction278 = FrozenMLP(ROOT/protocols[278]["selected"]["file"])
        self.direction282 = FrozenMLP(ROOT/protocols[282]["selected"]["file"])
        self.direction300 = FrozenMLP(ROOT/protocols[300]["selected"]["file"])
        self.context288 = FrozenMLP(ROOT/protocols[288]["selected"]["file"])
        if protocols[276]["variant"] != "scalar" or protocols[282]["selected"]["spec"]["architecture"] != "MLP":
            raise ValueError("Unsupported frozen response architecture")
        p = protocols[276]["selected_policy"]
        self.strength_observer = EvidenceRearmObserver(PhasePolicy(**p["strength"]))
        self.phase_observer = FactorizedPhaseObserver(DirectionPolicy(**p["direction"]))
        self.supp278 = SupplementaryDirectionObserver(SupplementPolicy(**protocols[278]["selected"]["policy"]))
        self.supp282 = SupplementaryDirectionObserver(SupplementPolicy(**protocols[282]["selected"]["policy"]))
        self.supp300 = SupplementaryDirectionObserver(SupplementPolicy(
            score_limit=protocols[300]["selected"]["policy"]["supported_limit"],
            confirm_s=.03, permit_residue=True, require_signed_change=False))
        self.context_new = CausalConfirmation(.03)
        self.context_veto = CausalConfirmation(protocols[288]["selected"]["confirm_s"])
        self.release291 = CausalConfirmation(protocols[291]["selected"]["confirm_s"])
        self.release293 = CausalConfirmation(protocols[293]["selected"]["confirm_s"])
        self.history = MagneticFeatureHistory(); self.reset()

    def reset(self):
        if hasattr(self, "history"):
            self.history.reset()
        for name in ("strength_observer", "phase_observer", "supp278", "supp282", "supp300",
                     "context_new", "context_veto", "release291", "release293"):
            if hasattr(self, name): getattr(self, name).reset()
        self.evidence_run = 0; self.token = None

    def step(self, time_ns, delta_b_uT, *, eligible=True, reset_token=None):
        t = int(time_ns); b = np.asarray(delta_b_uT, float)
        if b.shape != (15,) or not np.isfinite(b).all():
            eligible = False; b = np.zeros(15)
        if self.token != reset_token:
            self.reset(); self.token = reset_token
        x = self.history.push(t, b, eligible)
        if x["reset"]: self.evidence_run = 0
        a = float(np.linalg.norm((b-self.null["center"])@self.null["whitener"]))
        d = float(np.linalg.norm(x["changes"]@self.null["whitener"], axis=1).max())
        high = eligible and (a > self.null["absolute_limit"] or d > self.null["change_limit"])
        self.evidence_run = self.evidence_run+1 if high else 0
        response = self.evidence_run >= 3
        c, direction, binary = [m.predict(x["scalar"])[0] for m in self.strength_models]
        strength = self.strength_observer.step(t, c, direction, response=response,
            fresh_rise=100.*x["state"][1] > self.fresh_limit, amplitude_uT=x["amplitude_uT"],
            eligible=eligible, reset_token=reset_token)["state"]
        p = self.protocols[276]["selected_policy"]
        phase = self.phase_observer.step(t, strength, direction, binary, response=response,
            signed_amplitude_change_uT=x["state"][1]*100., change_limit_uT=self.change_limit*p["change_multiplier"],
            eligible=eligible, reset_token=reset_token)["phase"]
        for observer, model, feature in ((self.supp278, self.direction278, x["relative"]),
                                         (self.supp282, self.direction282, x["sequence"])):
            phase = observer.step(t, phase, strength, model.predict(feature)[0], response=response,
                signed_change_uT=x["state"][1]*100., eligible=eligible, reset_token=reset_token)["phase"]
        base_phase = phase
        scores = self.direction300.predict(x["sequence"])[0]
        policy = self.protocols[300]["selected"]["policy"]
        limit = policy["supported_limit"] if strength == 6 else policy["other_limit"]
        if max(scores[:2]) < limit: scores = np.array([0., 0., 1.])
        added = self.supp300.step(t, phase, strength, scores, response=response,
            signed_change_uT=x["state"][1]*100., eligible=eligible, reset_token=reset_token)
        phase = added["phase"]
        residual = float(self.context288.predict(x["sequence"])[0, 0])
        denied = self.context_new.step(t, eligible and residual >= policy["new_direction_residual_limit"])
        if added["supplemented"] and denied: phase = base_phase
        context = self.protocols[288]["selected"]
        veto = self.context_veto.step(t, eligible and residual >= context["limit"] and strength != 6)
        if veto and phase in (2, 3, 4): phase = 5 if response else 1
        r291 = self.protocols[291]["selected"]
        release1 = self.release291.step(t, eligible and residual >= r291["limit"])
        if release1 and strength == 6: strength = phase = 5 if response else 1
        r293 = self.protocols[293]["selected"]
        release2 = self.release293.step(t, eligible and residual >= r293["limit"])
        changed = release2 and strength == 6
        if changed:
            strength = 5 if response else 1
            if phase not in (0, 2, 4): phase = strength
        if not eligible: phase = strength = 0
        return {"strength": int(strength), "phase": int(phase), "response": bool(response),
                "amplitude_uT": x["amplitude_uT"], "context_residual_score": residual,
                "support_released": bool(release1 or changed), "eligible": bool(eligible),
                "absolute_evidence": a, "dynamic_evidence": d,
                "direction_scores": scores.tolist(), "history_reset": bool(x["reset"])}
