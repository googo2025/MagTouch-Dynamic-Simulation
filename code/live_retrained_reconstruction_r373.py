"""R372 plateau-trained weights in the complete, unchanged R348 system.

Only the causal XY and bounded incremental-Z weights change. Existing shallow
response states, localization permission, slow memory and Z evidence protection
stay active. Import/construction do not connect to hardware.
"""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
import numpy as np

from live_fast_slow_reconstruction_r348 import (
    FastSlowReconstructionEngineR348, LiveFastSlowReconstructionPredictorR348,
    GATE_NAME as OLD_GATE,
)
from training.causal_depth_network_r337 import DepthMLP
from training.fast_slow_runtime_r348 import FREEZE as OLD_DEPTH_FREEZE

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT/'analysis/r372_direct_source_plateau_20261001'
FAMILY = 'DirectPlateauReconstructionR373'
PRESET_NAME = '5×5｜源5 mm模型｜充分训练XY＋Z R373（实测）'
GATE_NAME = '统一状态＋磁定位证据＋充分训练XY/Z（R373，独立模式）'
COMPONENTS = {
    'xy': {'seed': 7, 'freeze_sha256': '672a1a40eb7a810fcad40e523c29d86285f3ff631b3ff2e78ed9fd1e1463bc7b',
           'model_sha256': '65f3015e8c053b1c806fe6c56cb53e862a8b97c4f1f882cb8f8fbf7b694305ff'},
    'z': {'seed': 348, 'freeze_sha256': '65cecaab6f9953af80d6c889ee1f0c9043845ea1ef316c8a8afc7092d2627c39',
          'model_sha256': '26eca2f30c07caf03a2a018b64ca632991657cfdc12653e4e81bb9bf43e259c7'},
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_components():
    result = {}
    for component, expected in COMPONENTS.items():
        folder = TRAINING/component
        if sha(folder/'freeze.json') != expected['freeze_sha256']:
            raise ValueError('R372 training freeze changed: '+component)
        f = json.loads((folder/'freeze.json').read_text(encoding='utf8'))
        if (f['status'] != 'COMPLETED_PLATEAU' or f['seed'] != expected['seed']
                or f['test_used_for_fitting_or_selection'] or f['model'] != 'best_model.npz'
                or f['model_sha256'] != expected['model_sha256']
                or sha(folder/'best_model.npz') != expected['model_sha256']):
            raise ValueError('R372 frozen source weights invalid: '+component)
        result[component] = f
    return result


class PlateauCausalXY:
    def __init__(self):
        frozen_components()
        with np.load(TRAINING/'xy/best_model.npz', allow_pickle=False) as a:
            for key in ('mean', 'scale', 'w1', 'b1', 'w2', 'b2', 'w3', 'b3'):
                setattr(self, key, a[key].copy())
        shapes = {'mean': (53,), 'scale': (53,), 'w1': (53, 64), 'b1': (64,),
                  'w2': (64, 64), 'b2': (64,), 'w3': (64, 2), 'b3': (2,)}
        for key, shape in shapes.items():
            value = getattr(self, key)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError('Invalid retrained XY array: '+key)
            value.flags.writeable = False
        if np.any(self.scale <= 0):
            raise ValueError('Invalid retrained XY scaling')

    def predict(self, feature):
        x = (np.asarray(feature, float).reshape(-1, 53)-self.mean)/self.scale
        return np.tanh(np.tanh(x@self.w1+self.b1)@self.w2+self.b2)@self.w3+self.b3


class DirectTrainedEngineR373(FastSlowReconstructionEngineR348):
    def __init__(self, *args, **kwargs):
        self.training_freezes = frozen_components()
        super().__init__(*args, **kwargs)
        main, provisional = PlateauCausalXY(), PlateauCausalXY()
        np.testing.assert_allclose(main.mean, self.memory_decoder.xy_model.mean, atol=1e-12, rtol=0)
        np.testing.assert_allclose(main.scale, self.memory_decoder.xy_model.scale, atol=1e-12, rtol=0)
        self.memory_decoder.xy_model = main
        self.candidate_decoder.xy_model = provisional
        model = DepthMLP.load(TRAINING/'z/best_model.npz')
        shapes = ((59, 32), (32,), (32, 32), (32,), (32, 1), (1,))
        if (model.mean.shape != (59,) or model.scale.shape != (59,)
                or not np.isfinite(model.mean).all() or not np.isfinite(model.scale).all()
                or np.any(model.scale <= 0)):
            raise ValueError('Invalid retrained Z scaling')
        for p, shape in zip(model.params, shapes):
            if p.shape != shape or not np.isfinite(p).all():
                raise ValueError('Invalid retrained Z weights')
            p.flags.writeable = False
        np.testing.assert_allclose(model.mean, self.final_refiner.model.mean, atol=1e-12, rtol=0)
        np.testing.assert_allclose(model.scale, self.final_refiner.model.scale, atol=1e-12, rtol=0)
        if self.final_refiner.selection['dimension'] != 59 or self.final_refiner.selection['gain'] != 1:
            raise ValueError('Original incremental-Z architecture/gain changed')
        self.final_refiner.model = model
        self.reset_unified()

    def push(self, t, b, *, eligible=True):
        result = super().push(t, b, eligible=eligible)
        result['family'] = FAMILY
        extra = result['extra']
        # Remove historical claims that XYZ/acceptance are numerically invariant:
        # changed XY also enters the unchanged localization-evidence rules.
        for key in ('output_permission_unchanged', 'original_xy_preserved', 'original_xyz_preserved'):
            extra.pop(key, None)
        extra.update(fit_variant='source_original_data_plateau_xy_z_r372',
            final_adjudication_version='R373-R372-plateau-weights-R348-system',
            contact_gate=GATE_NAME, xy_weights_unchanged=False,
            output_permission_rules_unchanged=True, shallow_response_weights_unchanged=True,
            z_evidence_policy_unchanged=True, slow_memory_unchanged=True,
            incremental_z_preserves_input_xy=True, no_future_or_command_input=True,
            xy_model_sha256=COMPONENTS['xy']['model_sha256'],
            z_increment_model_sha256=COMPONENTS['z']['model_sha256'])
        return result


class LiveRetrainedPredictorR373(LiveFastSlowReconstructionPredictorR348):
    family = FAMILY

    def __init__(self, profile, path, preset, threshold, min_cosine, smoothing_ms,
                 contact_threshold=None, jump_policy=GATE_NAME):
        if jump_policy != GATE_NAME:
            raise ValueError('R373必须使用独立状态＋磁定位证据，不能叠加旧阈值/三帧释放门')
        super().__init__(profile, path, preset, threshold, min_cosine, smoothing_ms,
                         contact_threshold, jump_policy=OLD_GATE)
        self.engine = DirectTrainedEngineR373(profile, threshold, min_cosine, smoothing_ms, contact_threshold)
        self.metadata.update(algorithm='source_original_data_plateau_xy_z_r372',
            fit_variant='source_original_data_plateau_xy_z_r372', contact_gate=GATE_NAME,
            anti_jump_policy=GATE_NAME, xy_mlp_artifact_sha256=COMPONENTS['xy']['model_sha256'],
            xy_mlp_metadata_sha256=COMPONENTS['xy']['freeze_sha256'],
            xy_mlp_artifact='analysis/r372_direct_source_plateau_20261001/xy/best_model.npz',
            r372_xy_freeze_sha256=COMPONENTS['xy']['freeze_sha256'],
            r372_z_freeze_sha256=COMPONENTS['z']['freeze_sha256'],
            depth_increment_freeze_sha256=COMPONENTS['z']['freeze_sha256'],
            depth_increment_artifact='analysis/r372_direct_source_plateau_20261001/z/best_model.npz',
            depth_increment_artifact_sha256=COMPONENTS['z']['model_sha256'],
            original_depth_increment_policy_freeze_sha256=sha(OLD_DEPTH_FREEZE),
            direct_training={c: {k: f[k] for k in ('seed', 'epochs_run', 'best_epoch', 'stop_reason')}
                             for c, f in self.engine.training_freezes.items()},
            shallow_response_weights_unchanged=True, z_evidence_policy_unchanged=True,
            no_target_fitting_or_selection_in_this_retraining=True,
            validation_status='source plateau training and 30 retrospective real replays complete; new online run pending',
            runtime_source_sha256=sha(Path(__file__)), deployment_qualified=False)
        self.model_summary = (f'{preset} | 原数据充分训练XY＋Z增量；'
            '保留浅压状态、磁定位许可、慢记忆和Z保护；无旧门控、显示平滑或自动输出配准')
        self.begin_baseline()
