"""Causal live adapter for the calibrated R36 fit engine. No device access."""
from __future__ import annotations
import hashlib
import math
from collections import deque
from pathlib import Path
import numpy as np
from replay_backtest import (NormalizedFit, empty_record, vector, number,
                             JUMP_HYSTERESIS)
from training.live_multimodel_inference import UnifiedPrediction


class LiveNormalizedFitPredictor:
    family = 'NormalizedFit'
    seq_len = history = 1
    motion_epsilon_mm = 0.02
    output_mapping = {'visual': {'x': 'x', 'y': 'y', 'z': 'z'}}
    visual_scale = (5.0, 5.0, 1.0)

    def __init__(self, profile, path, preset, threshold, min_cosine, smoothing_ms,
                 contact_threshold=None, jump_policy=JUMP_HYSTERESIS):
        self.path = Path(path)
        # Keep the immutable calibration payload available to the UI's R46
        # source-fingerprint calibration.  That calibration compares directly
        # with samples[].delta and never asks the XY predictor for a position.
        self.profile = profile
        self.engine = NormalizedFit(profile, Path('__live__'), threshold, min_cosine,
                                    smoothing_ms, contact_threshold, jump_policy)
        continuous = self.engine.continuous
        xy_semantics = profile.get('xy_semantics') or (
            'continuous XY calibrated from this magnetic-layer profile'
            if continuous else 'discrete calibrated XY; not continuous localization')
        self.metadata = dict(algorithm='normalized_fit', preset=preset,
                             calibration_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
                             source_plan=profile.get('source_plan'), threshold=float(threshold),
                             contact_threshold=float(self.engine.contact_threshold),
                             min_cosine=float(min_cosine), smoothing_ms=float(smoothing_ms),
                             xy_semantics=xy_semantics,
                             fit_variant=profile.get('fit_variant','legacy_discrete'),
                             evidence_tier=profile.get('evidence_tier','unspecified'),
                             calibration_domain_mm=profile.get('calibration_domain_mm'),
                             baseline_semantics=profile.get('baseline_semantics','trial_b0'),
                             anti_jump_policy=self.engine.jump_policy,
                             auto_b0_after_release=bool(
                                 profile.get('baseline_semantics')=='pre_action_b0' and
                                 self.engine.release_enabled),
                             random_touch_validation=profile.get(
                                 'random_touch_validation','not_validated'),
                             working_range_mm=profile.get('working_range_mm'),
                             dynamic_model=bool(profile.get('dynamic_model')),
                             dynamic_state_model=bool(profile.get('dynamic_state_model')),
                             relative_reference='first valid prediction since B0/no-contact')
        method = (profile.get('fit_model_description') or
                  ('连续XY + 连续Z + 接触门控' if continuous
                   else '离散XY + 标定深度Z'))
        self.model_summary = (f'{preset} | {method} | '
                              f'临时接触阈值={self.engine.contact_threshold:g}µT | '
                              f'{self.engine.jump_policy} | dXYZ=本次接触首点差值')
        self.begin_baseline()

    def delta_from_group(self, data):
        """Return the current 15-channel raw delta-B relative to this session B0."""
        b = vector(data)
        if b is None or self.engine.b0 is None:
            return None
        delta = np.asarray(b, dtype=float).reshape(-1) - np.asarray(
            self.engine.b0, dtype=float).reshape(-1)
        if delta.shape != (15,) or not np.isfinite(delta).all():
            return None
        return delta.copy()

    def set_xy_translation(self,x_mm=0.0,y_mm=0.0,source='manual',details=None):
        """Apply an output-coordinate translation without changing raw data or B0."""
        self.set_xy_rigid_transform(0.0,x_mm,y_mm,source,details)

    def set_xy_rigid_transform(self,rotation_deg=0.0,x_mm=0.0,y_mm=0.0,
                               source='manual',details=None):
        """Apply an output-only 2-D rigid reassembly transform."""
        self.engine.set_xy_rigid_transform(rotation_deg,x_mm,y_mm,source,details)
        self.metadata['xy_rotation_deg']=float(rotation_deg)
        self.metadata['xy_translation_mm']=[float(x_mm),float(y_mm)]
        self.metadata['xy_translation_source']=str(source or 'manual')

    @property
    def xy_translation(self):
        return tuple(float(v) for v in self.engine.xy_translation)

    @property
    def xy_rotation_deg(self):
        return math.degrees(float(self.engine.xy_rotation_rad))

    def begin_baseline(self, seconds=1.0):
        self.seconds = number(seconds, 'B0秒', .2, 60)
        self.baseline_target_frames = max(5, round(self.seconds * 40))
        self.b0 = None
        self.engine.b0 = None
        self.engine.buffer.clear()
        self.engine.anchor = None
        self.engine._reset_dynamic()
        self._frames = deque(maxlen=12000)
        self._release_frames = deque(maxlen=24)
        self._baseline_noise_p95_uT = None
        self._baseline_step_p95_uT = None
        self._auto_b0_attempts = 0
        self._auto_b0_rejections = 0
        self._no_contact_hint = None
        self._no_contact_hint_reason = 'magnetic-only fallback'
        self._controlled_baseline_reason = ''
        self._start = self._previous = None
        self._fault = ''

    @property
    def ready(self):
        return self.b0 is not None and not self._fault

    @property
    def baseline_progress(self):
        elapsed = 0 if self._start is None or self._previous is None else (self._previous-self._start)/1e9
        n = self.baseline_target_frames if self.ready else min(self.baseline_target_frames-1, int(elapsed*40))
        return n, self.baseline_target_frames

    def _prediction(self, r):
        valid = bool(r['valid'])
        xyz = (float(r['xyz'][0]), float(r['xyz'][1]), float(r['xyz'][2])) if valid else (float('nan'), float('nan'), float('nan'))
        return UnifiedPrediction(
            family=self.family, raw_fields=r['raw'],
            extra_outputs=dict(r['extra'], status=r['status'], preset=self.metadata['preset'],
                               needs_baseline=bool(self._fault)),
            visual_xyz_mm=xyz, visual_sources=('x','y','z'),
            norm_mm=math.sqrt(sum(v*v for v in xyz)) if valid else float('nan'),
            moving=False, q_mean=float('nan'), q_min=float('nan'),
            history_frames=1, history_required=1, t_ns=r['t_ns'], interaction_valid=valid)

    def set_no_contact_hint(self, value, reason=''):
        """Accept an optional *causal* hint from the experiment state machine.

        True means the commanded motion has completed at a known lifted height;
        False means motion/contact may be present; None is a random-touch stream
        with no external state.  Labels or future samples are never consumed.
        """
        self._no_contact_hint=(None if value is None else bool(value))
        self._no_contact_hint_reason=str(reason or '')
        if self._no_contact_hint is False:
            self._release_frames.clear()

    def request_confirmed_baseline(self, seconds=.4, reason='pre_action_b0'):
        """Start a baseline capture only after the motion controller confirms lift."""
        if self._no_contact_hint is not True:
            raise RuntimeError('动作前B0只能在运动完成且已确认无接触时采集')
        saved_reason=str(reason or 'pre_action_b0')
        self.begin_baseline(seconds)
        self._no_contact_hint=True
        self._no_contact_hint_reason=saved_reason
        self._controlled_baseline_reason=saved_reason

    def push_group(self, data, t_ns):
        t = int(t_ns)
        if self._previous is not None and (t <= self._previous or t-self._previous > 1_000_000_000):
            if self.ready:
                self._fault = '时间戳逆序/数据间断：请重新采集无载 B0'
            else:
                self._frames.clear(); self._start = None
        self._previous = t
        if self._fault:
            return [self._prediction(empty_record(t, self._fault))]
        b = vector(data)
        if b is None:
            self.engine.buffer.clear(); self.engine.anchor = None
            if not self.ready:
                self._frames.clear(); self._start = None
                return []
            return [self._prediction(empty_record(t, '磁场无效 / 缺传感器'))]
        baseline_completed_reason=''
        if not self.ready:
            if self._start is None: self._start = t
            self._frames.append(b)
            if t-self._start >= self.seconds*1e9 and len(self._frames) >= 5:
                values=np.asarray(self._frames)
                base = np.median(values, axis=0)
                residual=np.linalg.norm(values-base,axis=1)
                steps=(np.linalg.norm(np.diff(values,axis=0),axis=1)
                       if len(values)>1 else np.zeros(1,float))
                self._baseline_noise_p95_uT=float(np.percentile(residual,95))
                self._baseline_step_p95_uT=float(np.percentile(steps,95))
                self.b0 = base.reshape(5,3)
                self.engine.set_baseline(base)
                self._frames.clear()
                baseline_completed_reason=self._controlled_baseline_reason
                self._controlled_baseline_reason=''
            else:
                return []
        r=self.engine.push(t,b)
        if baseline_completed_reason:
            r=empty_record(t,'动作前确认无载窗口已更新 B0',{
                'controlled_pre_action_b0':True,
                'controlled_pre_action_b0_reason':baseline_completed_reason,
                'baseline_noise_p95_uT':self._baseline_noise_p95_uT,
                'baseline_step_p95_uT':self._baseline_step_p95_uT})
        if (self.metadata.get('auto_b0_after_release') and
                r.get('extra',{}).get('release_latched') and
                self._no_contact_hint is not False):
            self._release_frames.append((t,b.copy()))
            required_ns=(350_000_000 if self._no_contact_hint is True else 800_000_000)
            required_frames=(16 if self._no_contact_hint is True else 28)
            if (len(self._release_frames)>=required_frames and
                    self._release_frames[-1][0]-self._release_frames[0][0]>=required_ns):
                values=np.asarray([v for _,v in list(self._release_frames)[-20:]])
                base=np.median(values,axis=0)
                residual=np.linalg.norm(values-base,axis=1)
                stability=float(np.percentile(residual,95))
                half=max(2,len(values)//2)
                drift=float(np.linalg.norm(
                    np.median(values[-half:],axis=0)-np.median(values[:half],axis=0)))
                noise_ref=max(10.0,float(self._baseline_noise_p95_uT or 0.0),
                              0.75*float(self._baseline_step_p95_uT or 0.0))
                stability_limit=min(80.0,max(25.0,2.0*noise_ref))
                drift_limit=min(60.0,max(18.0,1.5*noise_ref))
                delta=base-self.engine.b0
                candidate_amp=float(np.linalg.norm(delta))
                if candidate_amp>1e-9:
                    candidate_cosine=float(np.max(
                        self.engine.confidence_templates@(delta/candidate_amp)))
                else:
                    candidate_cosine=-1.0
                contact_cosine_limit=(self.engine.min_cosine if self._no_contact_hint is True
                                      else max(-1.0,self.engine.min_cosine-0.05))
                contact_like=bool(
                    candidate_amp>=self.engine.contact_threshold and
                    candidate_cosine>=contact_cosine_limit)
                floor=float(r.get('extra',{}).get(
                    'release_floor_amplitude_uT',candidate_amp))
                release_amp=float(r.get('extra',{}).get(
                    'release_amplitude_uT',candidate_amp))
                floor_consistent=(candidate_amp <= max(
                    floor+stability_limit,release_amp))
                self._auto_b0_attempts+=1
                if (stability<=stability_limit and drift<=drift_limit and
                        floor_consistent and not contact_like):
                    self.b0=base.reshape(5,3)
                    self.engine.set_baseline(base)
                    self._release_frames.clear()
                    r=empty_record(t,'释放后确认稳定无载并自动更新 B0',{
                        'auto_b0_after_release':True,
                        'baseline_stability_p95_uT':stability,
                        'baseline_stability_limit_uT':stability_limit,
                        'baseline_drift_uT':drift,
                        'baseline_drift_limit_uT':drift_limit,
                        'baseline_candidate_amplitude_uT':candidate_amp,
                        'baseline_candidate_cosine':candidate_cosine,
                        'baseline_candidate_contact_cosine_limit':contact_cosine_limit,
                        'external_no_contact_hint':self._no_contact_hint,
                        'external_no_contact_hint_reason':self._no_contact_hint_reason,
                        'auto_b0_attempts':self._auto_b0_attempts,
                        'auto_b0_rejections':self._auto_b0_rejections})
                else:
                    self._auto_b0_rejections+=1
                    r['extra'].update({
                        'auto_b0_candidate_rejected':True,
                        'baseline_stability_p95_uT':stability,
                        'baseline_stability_limit_uT':stability_limit,
                        'baseline_drift_uT':drift,
                        'baseline_drift_limit_uT':drift_limit,
                        'baseline_candidate_amplitude_uT':candidate_amp,
                        'baseline_candidate_cosine':candidate_cosine,
                        'baseline_candidate_contact_cosine_limit':contact_cosine_limit,
                        'baseline_candidate_contact_like':contact_like,
                        'baseline_candidate_floor_consistent':floor_consistent,
                        'external_no_contact_hint':self._no_contact_hint,
                        'external_no_contact_hint_reason':self._no_contact_hint_reason,
                        'auto_b0_attempts':self._auto_b0_attempts,
                        'auto_b0_rejections':self._auto_b0_rejections})
        elif r.get('valid') or self._no_contact_hint is False:
            self._release_frames.clear()
        return [self._prediction(r)]
