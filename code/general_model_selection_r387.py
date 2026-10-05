"""Native model inventory/factory for R387 A/B inference. No hardware or Tk I/O.

The factory does not combine predictions, change weights, apply saved alignment,
or homogenize baseline algorithms. Each returned predictor is the original class.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from live_normalized_fit import LiveNormalizedFitPredictor
from neural_model_registry import LIVE_NEURAL_MODEL_OPTIONS, neural_model_family
from replay_backtest import CUSTOM, JUMP_HYSTERESIS, load_profile


PARAM_NAMES = ("threshold", "contact_threshold", "min_cosine", "smoothing_ms", "jump_policy")
VIRTUAL_LABEL = "FIT｜虚拟扩集1000条XY＋原R373 Z（研究副本）"
CUSTOM_FIT_LABEL = "FIT｜自由设置标定 JSON"
CUSTOM_NEURAL_LABEL = "NN｜自由选择 .pt checkpoint"


class _Value:
    """Python-only variable; never calls Tcl or fires live UI traces."""
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


def _defaults(preset, profile, cls):
    fallback = dict(threshold=76.572, contact_threshold=76.572,
                    min_cosine=.8, smoothing_ms=250, jump_policy=JUMP_HYSTERESIS)
    supplied = profile.get("defaults", {})
    fallback.update({k: supplied[k] for k in PARAM_NAMES if k in supplied})
    fallback["contact_threshold"] = supplied.get("contact_threshold", supplied.get("threshold", fallback["threshold"]))
    # These are the native preset_defaults policies, not a new common gate.
    # Calling the UI methods repeatedly would reread a 59-MB library for every
    # option; inventory must not reconstruct models or block on that work.
    from replay_backtest import JUMP_FAST
    from live_field_memory_r126 import LiveR126Predictor
    native = importlib.import_module(cls.__module__)
    if cls.__name__ in {"LiveUnifiedTactilePredictor", "LiveEvidenceTactilePredictor",
                        "LiveFastSlowReconstructionPredictorR348", "LiveRetrainedPredictorR373",
                        "VirtualPredictor"}:
        fallback.update(contact_threshold=70, jump_policy=native.GATE_NAME)
    elif issubclass(cls, LiveR126Predictor):
        fallback.update(contact_threshold=70, jump_policy=JUMP_FAST)
    elif cls.__name__ in {"LiveR89Predictor", "LiveOld1mmFieldMemoryPredictor"}:
        fallback["jump_policy"] = JUMP_FAST
    return fallback


def _profile_info(path, key, documents):
    """Read each library once, extract only selector defaults/provenance.

    Actual model validation remains load_profile() at load time. Variants use
    the exact base-profile/overrides lookup, without copying large sample arrays.
    """
    from replay_backtest import SCHEMA
    path = Path(path)
    if str(path) not in documents:
        document = json.loads(path.read_text(encoding="utf-8-sig"))
        if document.get("schema") != SCHEMA:
            raise ValueError("Unsupported calibration schema: " + str(path))
        documents[str(path)] = document
    document = documents[str(path)]
    if key is None:
        selected = document
    else:
        selected = document.get("profiles", {}).get(key)
        if selected is None:
            variant = document.get("variants", {}).get(key)
            if not isinstance(variant, dict):
                raise ValueError("Missing installed profile: " + str(key))
            base = document.get("profiles", {}).get(variant.get("base_profile"))
            if not isinstance(base, dict):
                raise ValueError("Missing variant base profile: " + str(key))
            selected = {**base, **variant.get("overrides", {})}
    fields = ("defaults", "physical_profile_id", "physical_configuration", "training_contact_z_mm")
    return {k: copy.deepcopy(selected[k]) for k in fields if k in selected}


def _type_name(cls):
    return cls.__module__ + ":" + cls.__name__


def _native_classes():
    # Runtime imports are file-only, never UI/hardware startup. R385's registry
    # monkeypatch must not turn the native R373 choice into a hidden pair.
    import live_algorithm_ui_r131 as registry
    from live_retrained_reconstruction_r373 import LiveRetrainedPredictorR373
    from live_paired_virtual_validation_r385 import VirtualPredictor
    classes = {_type_name(s.predictor_type): s.predictor_type for s in registry.MODEL_SPECS.values()}
    classes.update({_type_name(c): c for c in (LiveNormalizedFitPredictor, LiveRetrainedPredictorR373, VirtualPredictor)})
    return classes


def inventory(panel):
    """Return serializable selections covering all installed original options.

    Call on the UI owner thread: reading the real panel's current variables is
    intentional. Each library is read once; no predictor is constructed here.
    Optional explicit settings can be supplied by replacing spec['params']; the
    original predictor's own frozen-setting checks remain in force at load.
    """
    import live_algorithm_ui_r131 as registry
    from live_algorithm_ui_r62 import R61_PRESET_NAME, R61_PROFILE_PATH
    from live_retrained_reconstruction_r373 import PRESET_NAME, LiveRetrainedPredictorR373
    from live_paired_virtual_validation_r385 import VirtualPredictor
    result, documents = {}, {}
    for preset, key in panel.available_presets.items():
        if preset == CUSTOM:
            continue
        registration = registry.MODEL_SPECS.get(preset)
        path = Path(registration.path if registration else panel.profile_path)
        if preset == R61_PRESET_NAME:
            path = R61_PROFILE_PATH
        if not path.is_file():
            continue
        cls = registration.predictor_type if registration else LiveNormalizedFitPredictor
        if preset == PRESET_NAME:
            cls = LiveRetrainedPredictorR373
        profile = _profile_info(path, registration.key if registration else key, documents)
        label = "FIT｜" + preset
        spec = dict(kind="fit", label=label, preset=preset, path=str(path),
                    key=registration.key if registration else key,
                    predictor_type=_type_name(cls), params=_defaults(preset, profile, cls),
                    needs_key=bool(registration and registration.needs_key), custom=False,
                    source_profile_id=registration.source_profile_id if registration else
                    profile.get("physical_profile_id", profile.get("physical_configuration", {}).get("profile_id")),
                    training_contact_z_mm=registration.training_contact_z_mm if registration else
                    profile.get("training_contact_z_mm", profile.get("physical_configuration", {}).get("contact_z_mm")))
        result[label] = spec
    native = result.get("FIT｜" + PRESET_NAME)
    if native:
        candidate = copy.deepcopy(native)
        candidate.update(label=VIRTUAL_LABEL, predictor_type=_type_name(VirtualPredictor),
                         research_only=True, preset=PRESET_NAME)
        result[VIRTUAL_LABEL] = candidate
    current = {k: panel.vars[k].get() for k in PARAM_NAMES}
    result[CUSTOM_FIT_LABEL] = dict(kind="fit", label=CUSTOM_FIT_LABEL, preset=CUSTOM,
        path=panel.vars.get("custom_path", _Value("")).get(), key=None,
        predictor_type=_type_name(LiveNormalizedFitPredictor), params=current,
        needs_key=False, custom=True)
    for name, path in LIVE_NEURAL_MODEL_OPTIONS.items():
        label = "NN｜" + name
        result[label] = dict(kind="neural", label=label, preset=name, path=str(path),
                             neural_family=neural_model_family(name), custom=False)
    result[CUSTOM_NEURAL_LABEL] = dict(kind="neural", label=CUSTOM_NEURAL_LABEL,
        preset=CUSTOM_NEURAL_LABEL, path="", neural_family="auto", custom=True)
    # Catch accidental nonserializable classes/Paths before these become config.
    json.dumps(result, ensure_ascii=False, allow_nan=False)
    return result


def load(spec, common_mapping=None, neural_family="auto"):
    """Construct exactly one native model, with independent mutable history/B0."""
    selected = copy.deepcopy(dict(spec))
    path_text = str(selected.get("path", "")).strip()
    if not path_text:
        raise ValueError("请先为自由设置模型选择文件")
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if selected.get("kind") == "neural":
        from training.live_multimodel_inference import UnifiedLivePredictor
        family = neural_family if selected.get("custom") else selected.get("neural_family", neural_family)
        mapping = common_mapping if common_mapping is not None else selected.get("output_mapping")
        return UnifiedLivePredictor(path, family=family, output_mapping=mapping)
    if selected.get("kind") != "fit":
        raise ValueError("Unknown model kind")
    if selected.get("custom"):
        from live_algorithm_ui_r68 import unvalidated_old1mm_profile
        if unvalidated_old1mm_profile(path):
            raise ValueError("已撤下的旧1mm R63/R64/R65门控不能通过自由JSON重新启用")
    classes = _native_classes()
    type_name = selected.get("predictor_type")
    if type_name not in classes:
        raise ValueError("Model class is not an installed native predictor: " + str(type_name))
    profile = load_profile(path, selected.get("key"))
    # Profiles of registered models retain the same frozen-source check as the
    # original loader. Comparison does not silently rewrite their magnet ID.
    if not selected.get("custom") and selected.get("source_profile_id"):
        actual = profile.get("physical_profile_id") or profile.get("physical_configuration", {}).get("profile_id")
        if actual and actual != selected["source_profile_id"]:
            raise ValueError("Model source profile changed since selection")
        expected_z = selected.get("training_contact_z_mm")
        actual_z = profile.get("training_contact_z_mm", profile.get("physical_configuration", {}).get("contact_z_mm"))
        if expected_z is not None and actual_z is not None and abs(float(actual_z) - float(expected_z)) > 1e-6:
            raise ValueError("Frozen training Contact-Z changed since selection")
    params = {k: selected["params"][k] for k in PARAM_NAMES}
    if selected.get("needs_key"):
        params["key"] = selected["key"]
    return classes[type_name](profile, path, selected["preset"], **params)


def describe(predictor, spec):
    """Lossless model provenance; same input does not imply identical native B0."""
    path = Path(spec["path"]).expanduser().resolve()
    neural = spec.get("kind") == "neural"
    return json.loads(json.dumps(dict(
        selection=copy.deepcopy(spec), checkpoint_or_profile_path=str(path),
        checkpoint_or_profile_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        native_predictor_type=_type_name(type(predictor)), family=str(predictor.family),
        model_summary=str(predictor.model_summary),
        history_frames_required=int(predictor.seq_len),
        output_mapping=dict(predictor.output_mapping),
        visual_adapter_scale_mm=[float(v) for v in predictor.visual_scale],
        baseline_semantics=("native 40-Hz interpolated-grid median; native history warm-up" if neural
                            else "native acquisition-group median over explicit unloaded time window"),
        prediction_timebase=("native 25-ms grid" if neural else "native magnetic-group timestamps"),
        equality_of_b0_with_other_model_not_assumed=True,
        automatic_output_alignment_applied_by_factory=False,
        algorithm_metadata=copy.deepcopy(getattr(predictor, "metadata", {})),
    ), ensure_ascii=False, default=str))
