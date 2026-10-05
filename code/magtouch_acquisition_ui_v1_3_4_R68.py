# -*- coding: utf-8 -*-
"""
MagTouch Hardware Acquisition & Experiment System V1.3.4-R68
R68 withdraws the unvalidated old-1-mm-tip R65 real-time selector after
observed no-contact false positives, without changing existing R62 options.
R53 clarified that the two R50 static models are continuous XYZ regressors.
It keeps the R52 magnet-labelled records and center-Contact-Z motion policy.
Four-point raw magnetic onset Z differences above the reference tolerance are
reported but no longer stop five-point alignment; motion safety limits still apply.
V1.3.4-R5 continues from the validated V1.3.4 experiment-motion base. All helper tools are launched with the exact same Python interpreter as the running main UI (sys.executable). The unreliable USB M112
emergency-stop button is removed; physical printer power-off is the defined emergency action.
This revision fixes synchronized video replay (including legacy Trials whose camera frame IDs
were recorded from a continuously running preview counter), while preserving the validated
V1.3.4 experiment action logic.

Windows 11 / Python 3.10+

Main functions
--------------
1. 5x MLX90393 grouped acquisition via CH341 I2C.
2. Optional top/side camera preview + synchronized recording; cameras may remain disabled for current stage-only validation.
3. Trial/session management and reproducible metadata.
4. Optional generic Marlin-style 3D-printer G-code controller (dry-run by default).
5. Unified PC master timeline for magnetics, cameras, printer commands, and manual events.
6. Formal raw-data preservation: raw sensor counts and converted uT are both saved.
7. Integrated experiment-data browser, QC management and synchronized magnetic/video/printer replay.
8. Full experiment-plan generator with editable queue, presets, save/load, and one-click sequential acquisition.
9. Every Full Plan run is stored as one self-contained Plan_* folder containing its plan files and all Trial_* subfolders.
10. Case-preserving H1/H2-R15 live inference with raw/extra outputs plus adapted plots and an inference-only 3-D view.
11. Ubuntu Bash/conda/Python training-command export with persistent paths, GPU IDs, CPU allocation, and parameters.

This application intentionally does NOT use video as model input, labels, physical
ground truth, or post-hoc validation evidence.  Optional video is only a surface
inspection/recording aid and is excluded from the R15 inference route.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import queue
import re
import sys
import shutil
import subprocess
import bisect
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

try:
    import cv2
except Exception as e:  # pragma: no cover
    cv2 = None
    CV2_IMPORT_ERROR = e
else:
    CV2_IMPORT_ERROR = None

try:
    from PIL import Image, ImageTk
except Exception as e:  # pragma: no cover
    Image = None
    ImageTk = None
    PIL_IMPORT_ERROR = e
else:
    PIL_IMPORT_ERROR = None

try:
    import serial  # pyserial, optional unless printer control is used
except Exception:
    serial = None

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.database_manager import scan_trials  # noqa: E402

from magtouch_driver import (  # noqa: E402
    CH341Bridge,
    CH341Error,
    MLX90393,
    MultiMLX90393Sampler,
    SimulatedFiveSensorSampler,
)
from training.live_multimodel_inference import (
    RAW_FIELD_ORDER,
    UnifiedLivePredictor,
    default_output_mapping_json,
    normalize_output_mapping,
)
from training.ubuntu_training_export import (
    DEFAULT_UBUNTU_TRAINING,
    RECOMMENDED_HARDWARE,
    build_ubuntu_training_script,
    normalize_ubuntu_training_config,
)


from replay_backtest_ui import ReplayBacktestPanel
from live_algorithm_ui_r68 import LiveAlgorithmPanel, FIT
from live_normalized_fit import LiveNormalizedFitPredictor
from contact_display import prediction_contact
from installation_fingerprint import compare_fingerprint, robust_fingerprint
from source_fingerprint_calibration import fit_source_fingerprint_alignment
from contact_z_calibration import (
    flatten_group_ut,
    raw_baseline_statistics,
    raw_contact_window,
)
from neural_model_registry import (
    LIVE_NEURAL_MODEL_OPTIONS,
    neural_model_family,
    resolve_neural_model_selection,
)

APP_NAME = "MagTouch Hardware Acquisition & Experiment System V1.3.4-R68"
BUILD_ID = "20260925-V1.3.4-R68-WITHDRAW-OLD1MM-R65"

# A5S physical travel used as a final software backstop for all absolute experiment moves.
# These are deliberately not inferred from G-code state.
A5S_MACHINE_BOUNDS_MM = {"x": (0.0, 305.0), "y": (0.0, 305.0), "z": (0.0, 320.0)}
AUTO_MAX_INDENT_MM = 3.0
AUTO_MAX_XY_OFFSET_MM = 25.0
AUTO_MAX_XY_SPEED_MM_S = 30.0
AUTO_MAX_Z_SPEED_MM_S = 10.0
AUTO_MAX_HOLD_S = 60.0
AUTO_MAX_REPEATS = 500
AUTO_MAX_REPEAT_INTERVAL_S = 600.0
AUTO_MAX_GRID_POINTS = 1024
PRE_ACTION_B0_SECONDS = 0.35
PRE_ACTION_B0_DWELL_S = 0.50


def center_z_plane_assessment(center_z_mm, local_z_mm, nominal_depth_mm,
                              reference_tolerance_mm):
    """Describe a local magnetic-onset Z difference without treating it as geometry."""
    center=float(center_z_mm)
    local=float(local_z_mm)
    depth=float(nominal_depth_mm)
    tolerance=float(reference_tolerance_mm)
    if not all(math.isfinite(value) for value in (center,local,depth,tolerance)):
        raise ValueError("五点接触Z比较包含非有限数值。")
    if depth<0 or tolerance<0:
        raise ValueError("五点接触Z比较的压深或参考容差不能为负。")
    deviation=local-center
    estimated_depth=depth+deviation
    return {
        "deviation_from_center_mm":deviation,
        "estimated_local_indent_depth_mm":estimated_depth,
        "outside_reference_tolerance":abs(deviation)>tolerance,
        "exceeds_indent_safety_limit":estimated_depth>AUTO_MAX_INDENT_MM,
    }

# Only these automatic experiment phases are allowed to issue an explicit Z target.
# Lateral scan phases MUST omit Z; this catches the exact class of error that caused
# the dangerous repeated-downward-motion incident.
AUTO_Z_ALLOWED_PHASE_PREFIXES = (
    "safe_z", "press_approach", "press_indent", "preload_approach",
    "preload_indent", "retract", "depth_", "approach", "contact_onset", "xyz_compound",
    "grid_approach", "grid_indent", "grid_retract",
    "grid_reference_safe", "grid_reference_approach",
    "grid_reference_indent", "grid_reference_retract", "grid_reference_z_probe", "load",
    "unload_to_contact", "release", "final_safe_z", "path_", "dynamic_b0_safe_z"
)
DEFAULT_ADDRS = [0x0C, 0x0D, 0x0E, 0x0F, 0x10]

MAG_VIEW_LABELS = {
    "X / Bx": "x",
    "Y / By": "y",
    "Z / Bz": "z",
    "|B| / Magnitude": "m",
    "全部XYZ / All XYZ": "all",
}
MAG_VIEW_CODES = {v:k for k,v in MAG_VIEW_LABELS.items()}

EXPERIMENT_ACTION_LABELS = {
    "none": "none / 无接触基线",
    "press": "press / 单次法向按压",
    "press_cycle": "press_cycle / 周期法向点压",
    "slide_x": "slide_x / X向切向滑动",
    "slide_y": "slide_y / Y向切向滑动",
    "slide_xy": "slide_xy / 任意XY方向滑动",
    "compound": "compound / XYZ同步复合运动",
    "z_sweep": "z_sweep / Z阶梯压入",
    "z_path": "z_path / Z路径依赖",
    "x_sweep": "x_sweep / X位移阶梯扫描",
    "y_sweep": "y_sweep / Y位移阶梯扫描",
    "cross_scan": "cross_scan / 十字切向扫描",
    "angle_scan": "angle_scan / 多角度径向滑动",
    "circle_scan": "circle_scan / 圆周轨迹扫描",
    "grid_points": "grid_points / XY网格点触",
    "load_hold_release": "load_hold_release / 加载-保持-释放",
}
TRIAL_MODE_LABELS = dict(EXPERIMENT_ACTION_LABELS)
TRIAL_MODE_CODES = {v: k for k, v in TRIAL_MODE_LABELS.items()}
STAGE_ACTION_LABELS = dict(EXPERIMENT_ACTION_LABELS)
STAGE_ACTION_CODES = {v: k for k, v in STAGE_ACTION_LABELS.items()}

SETTINGS_PATH = HERE.parent / "config" / "magtouch_experiment_config.json"
MAGNETIC_LAYER_PROFILES_PATH = HERE.parent / "config" / "magnetic_layer_profiles.json"

def load_magnetic_layer_profiles(path=MAGNETIC_LAYER_PROFILES_PATH):
    path=Path(path)
    if not path.exists():
        return {"schema_version":"MagTouch-Magnetic-Layer-Profiles-1.0","default_profile_id":"","profiles":{}}
    try:
        d=json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        d=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(d.get("profiles"),dict):
        raise ValueError("magnetic_layer_profiles.json 缺少 profiles 对象")
    return d

def save_magnetic_layer_profiles(doc,path=MAGNETIC_LAYER_PROFILES_PATH):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(doc,ensure_ascii=False,indent=2),encoding="utf-8")
    tmp.replace(path)

DEFAULT_RUNTIME_SETTINGS = {
    "config_version": 2,
    "printer": {
        "port": "COM77", "baud": 115200, "dry_run": False,
        "require_home_before_auto_motion": True,
        "sensor_center_x_mm": 149.0, "sensor_center_y_mm": 147.0,
        "surface_contact_z_mm": 11.1, "safe_lift_mm": 3.0, "approach_clearance_mm": 1.0,
        "jog_step_mm": 1.0, "xy_speed_mm_s": 3.0, "z_speed_mm_s": 2.0,
        "home_z_raise_enabled": True, "home_z_raise_mm": 20.0, "g92_z_mm": 0.0,
        "auto_fan_off": True, "auto_save_parameters": True, "live_position_refresh": True,
    },
    "point_touch": {"depth_mm":0.10,"hold_s":0.50,"repeat":1,"interval_s":0.50,"z_speed_mm_s":0.50},
    "stage_experiment": {
        "action":"press","indent_mm":0.20,"dx_mm":2.0,"dy_mm":2.0,"speed_mm_s":1.0,"z_speed_mm_s":0.50,"hold_s":0.5,"repeat":1,
        "z_sweep_depths_mm":"0.10,0.20,0.30,0.50",
        "xy_sweep_values_mm":"-3,-2,-1,0,1,2,3",
        "angle_count":8,"circle_radius_mm":2.0,"circle_points":24,
        "grid_half_x_mm":4.0,"grid_half_y_mm":4.0,"grid_step_mm":2.0,
    },
    "trial": {"auto_id":True,"repeats":1,"repeat_interval_s":1.0,"auto_summary":True},
    "magnetic": {
        "ch341_index":0,"i2c_clock":"400 kHz","sample_period_ms":20,"include_temperature_each_cycle":False,
        "addresses":["0x0C","0x0D","0x0E","0x0F","0x10"],
    },
    "camera": {"top_enable":False,"side_enable":False,"top_index":0,"side_index":1,"width":1280,"height":720,"fps":60.0},
    # This is UI/inference setup only.  B0 is deliberately not persisted: it must
    # be recaptured for the current, unloaded local state on every session.
    "live_prediction": {
        "checkpoint_path":"", "model_family":"auto", "profile_confirmed":False,
        "output_mapping_json":default_output_mapping_json(),
        "display_invert_x":False, "display_invert_y":False, "edge_guard_ms":120.0,
        "trajectory_filter_mode":"关闭 / Off", "trajectory_filter_ema_alpha":0.25,
        "trajectory_filter_q":1.0, "trajectory_filter_r_xy":0.16,
        "trajectory_filter_r_z":0.01,
        "trajectory_history_points":4000,
        "xy_alignment_by_preset":{},
        "alignment_basis":"源数据ΔB指纹（推荐）",
        "alignment_z_threshold_uT":15.0,
        "alignment_z_step_mm":0.02,
        "alignment_z_max_search_mm":1.0,
        "view_yaw_deg":-45.0, "view_pitch_deg":28.0,
        "view_zoom":1.0, "view_pan_x":0.0, "view_pan_y":0.0,
        "experiment_view_yaw_deg":-45.0, "experiment_view_pitch_deg":28.0,
        "experiment_view_zoom":1.0, "experiment_view_pan_x":0.0,
        "experiment_view_pan_y":0.0,
    },
    "training": {
        "plan_path":"", "export_output_path":"", "training_export_path":"", "h1_features_path":"",
        "h2_relative_path":"", "r18_training_export_path":"", "r18_relative_path":"", "r18_labels_path":"",
        "feature":"F0", "policy":"raw15", "baseline_model":"transformer", "task":"mode5",
        "epochs":20, "baseline_batch":48, "gpu":0, "d_model":128, "layers":3, "hw5_batch":24,
        "h1_objective":"mode_only", "deterministic":True, "seed":20260910,
        "h2_feature":"R1", "h2_history":"64", "h2_use_q":True,
        "r18_feature":"R1", "r18_history":"64", "r18_endpoint_min_history":"128",
        "r18_objective":"unified", "r18_sample_policy":"common", "r18_use_q":True,
    },
    "ubuntu_training": dict(DEFAULT_UBUNTU_TRAINING),
}

def _deep_merge(base, override):
    out=json.loads(json.dumps(base))
    if isinstance(override, dict):
        for k,v in override.items():
            if isinstance(v,dict) and isinstance(out.get(k),dict):
                out[k]=_deep_merge(out[k],v)
            else:
                out[k]=v
    return out

def load_runtime_settings(path=SETTINGS_PATH):
    path=Path(path)
    cfg=json.loads(json.dumps(DEFAULT_RUNTIME_SETTINGS))
    if path.exists():
        try:
            cfg=_deep_merge(cfg,json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            pass
    return cfg

def save_runtime_settings(data,path=SETTINGS_PATH):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Reproducible geometry / metadata
# ---------------------------------------------------------------------------

DEFAULT_GEOMETRY = {
    "prototype": "MagTouch 2x2+1 hardware prototype V1",
    "coordinate_system": {
        "origin": "geometric center of four corner sensors",
        "x_positive": "right",
        "y_positive": "up in top-view image / array plane",
        "z_positive": "out of sensor plane toward tactile surface",
        "units": "mm",
    },
    "corner_pitch_mm": 15.188,
    "sensors": {
        "S1": {"x_mm": -7.594, "y_mm": 7.594, "z_mm": 0.0, "role": "upper-left"},
        "S2": {"x_mm": -7.594, "y_mm": -7.594, "z_mm": 0.0, "role": "lower-left"},
        "S3": {"x_mm": 7.594, "y_mm": 7.594, "z_mm": 0.0, "role": "upper-right"},
        "S4": {"x_mm": 7.594, "y_mm": -7.594, "z_mm": 0.0, "role": "lower-right"},
        "S5": {"x_mm": 0.0, "y_mm": 1.406, "z_mm": 0.0, "role": "offset-center"},
    },
    "important_note": "S5 physical +1.406 mm y-offset is retained; do not idealize to (0,0).",
    "printer_reference": {
        "sensor_center_printer_x_mm": 149.0,
        "sensor_center_printer_y_mm": 147.0,
        "surface_contact_printer_z_mm": 11.1,
        "note": "Measured A5S coordinates: nozzle just touches the upper surface at (149,147,11.1). Printer-axis to sensor-axis sign mapping remains an experiment calibration item."
    },
}


@dataclass
class TrialParameters:
    session_id: str
    trial_id: str
    command_mode: str
    x_mm: float
    y_mm: float
    z_indent_mm: float
    dx_mm: float
    dy_mm: float
    speed_mm_s: float
    hold_s: float
    repeat_count: int
    repeat_interval_s: float
    indenter_id: str
    material_id: str
    magnetic_layer_id: str
    auto_summary: str
    notes: str


def _folder_num(value: float, digits: int = 3) -> str:
    """Compact filename-safe decimal, e.g. -0.5 -> m0p5."""
    x=float(value)
    sign="m" if x < 0 else ""
    txt=f"{abs(x):.{digits}f}".rstrip("0").rstrip(".") or "0"
    return sign + txt.replace(".","p")


def _safe_slug(text: str, max_len: int = 96) -> str:
    text=str(text).strip()
    text=re.sub(r"[^A-Za-z0-9_.+\-]+","-",text)
    text=re.sub(r"-+","-",text).strip("-_.")
    return (text or "trial")[:max_len]


def _magnet_filename_suffix(profile_id: str) -> str:
    """Keep the selected profile distinguishable using Windows-safe characters.

    '*' is illegal in Windows paths, so the common 5*5 profile prefix becomes
    5x5. The full, unmodified profile ID remains in meta.json.
    """
    name=str(profile_id or "unknown").strip().replace("*", "x").replace("×", "x")
    common_prefix="magnetic_layer_v1_5x5-"
    if name.startswith(common_prefix):
        name="5x5-"+name[len(common_prefix):]
    slug=_safe_slug(name, max_len=max(40,len(name)+1))
    # Retain the distinguishing tail (for example, '-Yelow') for long custom IDs.
    if len(slug)>40:
        slug=slug[-40:].lstrip("-_.") or "unknown"
    return "mag_"+slug


def _trial_folder_name(params: TrialParameters, folder_tag: str = "",
                       magnet_profile_id: str = "") -> str:
    magnet_suffix=_magnet_filename_suffix(magnet_profile_id or params.magnetic_layer_id)
    # Preserve the prior filename-length budget when adding the final suffix.
    summary_budget=max(24,78-len(magnet_suffix)-2)
    suffix=_safe_slug(params.auto_summary or params.command_mode, max_len=summary_budget)
    tag=_safe_slug(folder_tag, max_len=24) if str(folder_tag).strip() else ""
    tagged=f"{tag}_{suffix}" if tag else suffix
    return f"Trial_{params.trial_id}_{tagged}__{magnet_suffix}"


class SafeCSVWriter:
    def __init__(self, path: Path, fieldnames: list[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fp = open(self.path, "w", newline="", encoding="utf-8-sig", buffering=1)
        self.writer = csv.DictWriter(self.fp, fieldnames=fieldnames, extrasaction="ignore")
        self.writer.writeheader()
        self.lock = threading.Lock()

    def write(self, row: dict[str, Any]):
        with self.lock:
            self.writer.writerow(row)

    def close(self):
        with self.lock:
            try:
                self.fp.flush()
                self.fp.close()
            except Exception:
                pass


class TrialRecorder:
    """Owns all files for one trial and accepts data from worker threads."""

    MAG_FIELDS = [
        "session_id", "trial_id", "cycle_id", "t_group_ns", "t_return_ns",
        "t_wall_ns", "trigger_span_ms", "conversion_wait_ms", "valid_sensor_count",
        "error_json",
    ]
    for _sid in range(1, 6):
        MAG_FIELDS.extend([
            f"S{_sid}_trigger_ns", f"S{_sid}_read_ns", f"S{_sid}_status",
            f"S{_sid}_x_raw", f"S{_sid}_y_raw", f"S{_sid}_z_raw",
            f"S{_sid}_Bx_uT", f"S{_sid}_By_uT", f"S{_sid}_Bz_uT", f"S{_sid}_Bmag_uT",
            f"S{_sid}_temp_C",
        ])

    EVENT_FIELDS = ["session_id", "trial_id", "t_perf_ns", "t_wall_ns", "event", "value", "source"]
    STAGE_FIELDS = [
        "session_id", "trial_id", "t_perf_ns", "t_wall_ns", "command",
        "x_mm", "y_mm", "z_mm", "feed_mm_min", "mode", "phase", "dry_run", "response"
    ]
    # Keep every currently supported case-sensitive model output as a stable
    # column.  A missing output is an empty CSV cell, never a fabricated zero.
    # Custom mappings and future/nested outputs are retained losslessly in the
    # two JSON columns at the end of each row.
    PREDICTION_FIELDS = [
        "session_id", "trial_id", "prediction_index", "t_prediction_ns",
        "t_recorded_perf_ns", "model_family", "history_frames", "history_required",
        "interaction_valid", "normal_state", "tangential_motion", "moving",
        "speed_mm_s", "norm_mm", "q_mean", "q_min",
        "visual_x_mm", "visual_y_mm", "visual_z_mm",
        "visual_source_x", "visual_source_y", "visual_source_z",
    ] + list(RAW_FIELD_ORDER) + ["raw_fields_json", "extra_outputs_json"]

    def __init__(self, root: Path, params: TrialParameters, geometry: dict,
                 acquisition_meta: dict, parent_dir: Optional[Path] = None,
                 folder_tag: str = ""):
        self.params = params
        self.geometry = geometry
        self.root = Path(root)
        # Manual Trial: root/Session/Trial_*
        # Full Plan Trial: root/Session/Plan_*/Trial_*
        self.parent_dir = Path(parent_dir) if parent_dir is not None else (self.root / params.session_id)
        self.folder_tag = _safe_slug(folder_tag, max_len=24) if str(folder_tag).strip() else ""
        self.magnet_profile_id = str(acquisition_meta.get("magnetic_layer_profile_id") or "").strip()
        self.magnet_filename_suffix = _magnet_filename_suffix(
            self.magnet_profile_id or params.magnetic_layer_id)
        self.trial_dir = self.parent_dir / _trial_folder_name(
            params, self.folder_tag, self.magnet_profile_id)
        if self.trial_dir.exists():
            raise FileExistsError(f"Trial directory already exists: {self.trial_dir}")
        self.trial_dir.mkdir(parents=True, exist_ok=False)
        (self.trial_dir / "processed").mkdir(exist_ok=True)

        self.t0_perf_ns = time.perf_counter_ns()
        self.t0_wall_ns = time.time_ns()
        self.meta_path = self.trial_dir / "meta.json"
        self.meta = {
            "schema_version": "MagTouch-HW-RAW-0.1",
            "app": APP_NAME,
            "created_local": datetime.now().isoformat(timespec="seconds"),
            "trial_parameters": asdict(params),
            "geometry": geometry,
            "acquisition": acquisition_meta,
            "timebase": {
                "master": "time.perf_counter_ns",
                "trial_t0_perf_ns": self.t0_perf_ns,
                "trial_t0_wall_ns": self.t0_wall_ns,
                "wall_clock": "time.time_ns",
            },
            "status": "recording",
            "container": {
                "parent_dir": str(self.parent_dir),
                "plan_folder": self.parent_dir.name if self.parent_dir.name.startswith("Plan_") else None,
                "folder_tag": self.folder_tag or None,
                "magnet_filename_suffix": self.magnet_filename_suffix,
            },
        }
        self._write_meta()
        self.mag_writer = SafeCSVWriter(self.trial_dir / "magnetic_raw.csv", self.MAG_FIELDS)
        self.event_writer = SafeCSVWriter(self.trial_dir / "events.csv", self.EVENT_FIELDS)
        self.stage_writer = SafeCSVWriter(self.trial_dir / "stage_commands.csv", self.STAGE_FIELDS)
        self.prediction_writer = None
        self.prediction_rows_written = 0
        self.closed = False
        self.write_event("TRIAL_START", params.command_mode, "recorder")

    def _write_meta(self):
        tmp = self.meta_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.meta_path)

    def write_event(self, event: str, value: Any = "", source: str = "manual"):
        if self.closed:
            return
        self.event_writer.write({
            "session_id": self.params.session_id,
            "trial_id": self.params.trial_id,
            "t_perf_ns": time.perf_counter_ns(),
            "t_wall_ns": time.time_ns(),
            "event": event,
            "value": value,
            "source": source,
        })

    def write_stage(self, row: dict[str, Any]):
        if self.closed:
            return
        base = {
            "session_id": self.params.session_id,
            "trial_id": self.params.trial_id,
            "t_perf_ns": time.perf_counter_ns(),
            "t_wall_ns": time.time_ns(),
        }
        base.update(row)
        self.stage_writer.write(base)

    def write_magnetic(self, data: dict[int, dict], meta: dict):
        if self.closed:
            return
        row = {
            "session_id": self.params.session_id,
            "trial_id": self.params.trial_id,
            "cycle_id": meta.get("cycle"),
            "t_group_ns": meta.get("trigger_base_ns"),
            "t_return_ns": meta.get("sample_return_ns", time.perf_counter_ns()),
            "t_wall_ns": time.time_ns(),
            "trigger_span_ms": meta.get("trigger_span_ms"),
            "conversion_wait_ms": meta.get("conversion_wait_ms"),
            "valid_sensor_count": len(data),
            "error_json": json.dumps(meta.get("errors", {}), ensure_ascii=False),
        }
        for i in range(5):
            d = data.get(i)
            prefix = f"S{i+1}_"
            if d is None:
                continue
            row.update({
                prefix + "trigger_ns": d.get("trigger_ns"),
                prefix + "read_ns": d.get("read_ns"),
                prefix + "status": d.get("status"),
                prefix + "x_raw": d.get("x_raw"),
                prefix + "y_raw": d.get("y_raw"),
                prefix + "z_raw": d.get("z_raw"),
                prefix + "Bx_uT": d.get("x_ut"),
                prefix + "By_uT": d.get("y_ut"),
                prefix + "Bz_uT": d.get("z_ut"),
                prefix + "Bmag_uT": d.get("mag_ut"),
                prefix + "temp_C": d.get("temp_c"),
            })
        self.mag_writer.write(row)

    @staticmethod
    def _prediction_csv_value(value: Any):
        if value is None:
            return ""
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (str, int, float)):
            return value
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)

    def enable_prediction_recording(self, model_meta: dict[str, Any]):
        """Add the optional timestamp-aligned prediction stream to this Trial."""
        if self.closed:
            raise RuntimeError("Cannot enable prediction recording on a closed Trial")
        if self.prediction_writer is not None:
            return
        self.prediction_writer = SafeCSVWriter(
            self.trial_dir / "live_predictions.csv", self.PREDICTION_FIELDS
        )
        self.prediction_rows_written = 0
        self.meta["live_prediction_recording"] = {
            "schema": "MagTouch-LIVE-PREDICTION-1.0",
            "file": "live_predictions.csv",
            "started_local": datetime.now().isoformat(timespec="seconds"),
            "started_perf_ns": time.perf_counter_ns(),
            "fixed_case_sensitive_output_fields": list(RAW_FIELD_ORDER),
            "missing_output_semantics": "Empty CSV cell: the loaded model did not emit this field; never interpret as numeric zero.",
            "unknown_output_policy": "raw_fields_json preserves all mapped raw outputs; extra_outputs_json preserves all additional/nested outputs.",
            "time_alignment": "Join t_prediction_ns to magnetic_raw.csv t_group_ns on the shared perf_counter_ns timebase; predictions use the inference 40 Hz grid and may be interpolated from adjacent magnetic groups.",
            **json.loads(json.dumps(model_meta, ensure_ascii=False, default=str)),
        }
        self._write_meta()

    def write_prediction(self, prediction: Any):
        if self.closed or self.prediction_writer is None:
            return
        raw_fields = dict(getattr(prediction, "raw_fields", {}) or {})
        extra_outputs = dict(getattr(prediction, "extra_outputs", {}) or {})
        visual = tuple(getattr(prediction, "visual_xyz_mm", (None, None, None)))
        sources = tuple(getattr(prediction, "visual_sources", (None, None, None)))
        row = {
            "session_id": self.params.session_id,
            "trial_id": self.params.trial_id,
            "prediction_index": self.prediction_rows_written,
            "t_prediction_ns": getattr(prediction, "t_ns", None),
            "t_recorded_perf_ns": time.perf_counter_ns(),
            "model_family": getattr(prediction, "family", None),
            "history_frames": getattr(prediction, "history_frames", None),
            "history_required": getattr(prediction, "history_required", None),
            "interaction_valid": getattr(prediction, "interaction_valid", None),
            "normal_state": getattr(prediction, "normal_state", None),
            "tangential_motion": getattr(prediction, "tangential_motion", None),
            "moving": getattr(prediction, "moving", None),
            "speed_mm_s": getattr(prediction, "speed_mm_s", None),
            "norm_mm": getattr(prediction, "norm_mm", None),
            "q_mean": getattr(prediction, "q_mean", None),
            "q_min": getattr(prediction, "q_min", None),
            "visual_x_mm": visual[0] if len(visual) > 0 else None,
            "visual_y_mm": visual[1] if len(visual) > 1 else None,
            "visual_z_mm": visual[2] if len(visual) > 2 else None,
            "visual_source_x": sources[0] if len(sources) > 0 else None,
            "visual_source_y": sources[1] if len(sources) > 1 else None,
            "visual_source_z": sources[2] if len(sources) > 2 else None,
            "raw_fields_json": json.dumps(raw_fields, ensure_ascii=False, separators=(",", ":"), default=str),
            "extra_outputs_json": json.dumps(extra_outputs, ensure_ascii=False, separators=(",", ":"), default=str),
        }
        for name in RAW_FIELD_ORDER:
            row[name] = self._prediction_csv_value(raw_fields.get(name))
        self.prediction_writer.write(row)
        self.prediction_rows_written += 1

    def update_prediction_recording_stats(self, input_groups_dropped: Optional[int] = None):
        info = self.meta.get("live_prediction_recording")
        if isinstance(info, dict):
            info["rows_written"] = int(self.prediction_rows_written)
            if input_groups_dropped is not None:
                info["input_groups_dropped_by_live_queue"] = int(input_groups_dropped)

    def register_camera(self, name: str, info: dict):
        self.meta.setdefault("camera_recordings", {})[name] = info
        self._write_meta()

    def close(self, status: str = "completed"):
        if self.closed:
            return
        self.write_event("TRIAL_STOP", status, "recorder")
        self.closed = True
        self.mag_writer.close()
        self.event_writer.close()
        self.stage_writer.close()
        self.update_prediction_recording_stats()
        if self.prediction_writer is not None:
            self.prediction_writer.close()
        self.meta["status"] = status
        self.meta["closed_local"] = datetime.now().isoformat(timespec="seconds")
        self.meta["duration_s"] = (time.perf_counter_ns() - self.t0_perf_ns) / 1e9
        self._write_meta()


# ---------------------------------------------------------------------------
# Camera worker
# ---------------------------------------------------------------------------

class CameraWorker:
    def __init__(self, name: str, index: int, width: int, height: int, fps: float,
                 preview_queue: queue.Queue, log):
        self.name = name
        self.index = int(index)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.preview_queue = preview_queue
        self.log = log
        self.cap = None
        self.thread = None
        self.stop_event = threading.Event()
        self.record_lock = threading.Lock()
        self.record_writer = None
        self.ts_writer = None
        self.record_info = None
        self.frame_id = 0
        # Video files restart at frame 0 for every Trial. Keep this counter separate
        # from the continuously-running camera capture frame_id. This fixes replay of
        # recordings that begin after the camera has already previewed many frames.
        self.record_frame_id = 0
        self.last_frame_ns = 0
        self.actual_width = 0
        self.actual_height = 0
        self.actual_fps = 0.0

    @property
    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def start(self):
        if cv2 is None:
            raise RuntimeError(f"OpenCV is unavailable: {CV2_IMPORT_ERROR}")
        if self.running:
            return
        cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {self.name} camera index {self.index}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self.cap = cap
        self.actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or self.width)
        self.actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or self.height)
        self.actual_fps = float(cap.get(cv2.CAP_PROP_FPS) or self.fps)
        self.stop_event.clear()
        self.frame_id = 0
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        self.log(f"{self.name}: camera started index={self.index}, {self.actual_width}x{self.actual_height}, requested {self.fps:g} fps")

    def stop(self):
        self.stop_recording()
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None
        self.thread = None
        self.log(f"{self.name}: camera stopped")

    def start_recording(self, trial_dir: Path):
        if not self.running:
            return None
        if cv2 is None:
            raise RuntimeError("OpenCV is unavailable; camera recording cannot start")
        with self.record_lock:
            if self.record_writer is not None:
                return self.record_info
            video_path = Path(trial_dir) / f"camera_{self.name}.mp4"
            ts_path = Path(trial_dir) / f"camera_{self.name}_timestamps.csv"
            fourcc = cv2.VideoWriter.fourcc(*"mp4v")
            use_fps = self.actual_fps if self.actual_fps > 1 else self.fps
            writer = cv2.VideoWriter(str(video_path), fourcc, use_fps,
                                     (self.actual_width, self.actual_height))
            if not writer.isOpened():
                raise RuntimeError(f"Cannot open VideoWriter: {video_path}")
            ts_writer = SafeCSVWriter(ts_path, ["frame_id", "capture_frame_id", "t_perf_ns", "t_wall_ns"])
            self.record_writer = writer
            self.ts_writer = ts_writer
            self.record_frame_id = 0
            self.record_info = {
                "camera_index": self.index,
                "video": video_path.name,
                "timestamps": ts_path.name,
                "width": self.actual_width,
                "height": self.actual_height,
                "nominal_fps": use_fps,
                "timestamp_semantics": "perf_counter_ns captured immediately after VideoCapture.read returned",
            }
            self.log(f"{self.name}: recording -> {video_path.name}")
            return dict(self.record_info)

    def stop_recording(self):
        with self.record_lock:
            if self.record_writer is not None:
                try:
                    self.record_writer.release()
                except Exception:
                    pass
            if self.ts_writer is not None:
                self.ts_writer.close()
            if self.record_writer is not None:
                self.log(f"{self.name}: recording stopped")
            self.record_writer = None
            self.ts_writer = None
            self.record_info = None

    def _loop(self):
        preview_every = 2
        if cv2 is None:
            return
        while not self.stop_event.is_set():
            cap = self.cap
            if cap is None:
                break
            ok, frame = cap.read()
            t_perf_ns = time.perf_counter_ns()
            t_wall_ns = time.time_ns()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            self.last_frame_ns = t_perf_ns
            frame_id = self.frame_id
            self.frame_id += 1
            with self.record_lock:
                if self.record_writer is not None and self.ts_writer is not None:
                    video_frame_id = self.record_frame_id
                    self.record_frame_id += 1
                    self.record_writer.write(frame)
                    self.ts_writer.write({
                        # frame_id is the frame number INSIDE this Trial video file.
                        # capture_frame_id is the camera's continuous preview counter.
                        "frame_id": video_frame_id,
                        "capture_frame_id": frame_id,
                        "t_perf_ns": t_perf_ns,
                        "t_wall_ns": t_wall_ns,
                    })
            if frame_id % preview_every == 0:
                try:
                    small = frame
                    max_w = 620
                    if small.shape[1] > max_w:
                        scale = max_w / small.shape[1]
                        small = cv2.resize(small, (max_w, int(small.shape[0] * scale)))
                    self.preview_queue.put_nowait((self.name, small.copy(), t_perf_ns))
                except queue.Full:
                    pass


# ---------------------------------------------------------------------------
# Optional printer controller
# ---------------------------------------------------------------------------

class PrinterController:
    def __init__(self, log, stage_callback):
        self.log = log
        self.stage_callback = stage_callback
        self.ser = None
        self.port = ""
        self.baud = 115200
        self.dry_run = True
        self.lock = threading.RLock()
        # Held only during serial writes, never while waiting for a response.
        # This permits STOP/E-STOP bytes to be transmitted promptly.
        self.tx_lock = threading.Lock()
        self.virtual_position = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.homed_axes: set[str] = set()
        self.abort_event: Optional[threading.Event] = None

    @property
    def connected(self):
        return self.dry_run or (self.ser is not None and getattr(self.ser, "is_open", False))

    def connect(self, port: str, baud: int, dry_run: bool):
        self.close()
        self.port = port.strip()
        self.baud = int(baud)
        self.dry_run = bool(dry_run)
        self.homed_axes.clear()
        if self.dry_run:
            self.log("Printer: DRY-RUN mode enabled; no serial commands will be sent")
            return
        if serial is None:
            raise RuntimeError("pyserial is not installed. Run: pip install pyserial")
        if not self.port:
            raise ValueError("Printer COM port is empty")
        self.ser = serial.Serial(self.port, self.baud, timeout=1.0, write_timeout=1.0)
        time.sleep(2.0)
        try:
            self.ser.reset_input_buffer()
        except Exception:
            pass
        self.log(f"Printer connected: {self.port} @ {self.baud}")
        self.send_line("G90", mode="setup", phase="absolute")
        fw=self.send_line("M115", mode="setup", phase="firmware")
        self.log(f"Printer firmware handshake: {fw}")
        self.query_position()

    def close(self):
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
        self.ser = None

    def _read_ok(self, timeout_s=3.0):
        if self.dry_run:
            return "DRY_RUN"
        connection = self.ser
        if connection is None:
            raise RuntimeError("Printer serial connection is not open")
        deadline = time.monotonic() + max(0.2, float(timeout_s))
        lines = []
        while time.monotonic() < deadline:
            raw = connection.readline()
            if not raw:
                continue
            text = raw.decode(errors="replace").strip()
            lines.append(text)
            if text.lower().startswith("ok"):
                return " | ".join(lines[-4:])
        return "TIMEOUT: " + " | ".join(lines[-4:])

    def send_line(self, cmd: str, *, x=None, y=None, z=None, feed=None,
                  mode="manual", phase="", wait_ok=True, timeout_s=3.0, strict_ack=False):
        with self.lock:
            response = ""
            if self.dry_run:
                self.log(f"Printer DRY> {cmd}")
                response = "DRY_RUN"
            else:
                if self.ser is None or not getattr(self.ser, "is_open", False):
                    raise RuntimeError("Printer serial port is not open")
                with self.tx_lock:
                    self.ser.write((cmd.strip() + "\n").encode("ascii", errors="ignore"))
                    self.ser.flush()
                response = self._read_ok(timeout_s=timeout_s) if wait_ok else "SENT"
                self.log(f"Printer> {cmd} | {response}")
            self.stage_callback({
                "command": cmd,
                "x_mm": x, "y_mm": y, "z_mm": z,
                "feed_mm_min": feed,
                "mode": mode,
                "phase": phase,
                "dry_run": self.dry_run,
                "response": response,
            })
            if strict_ack and isinstance(response, str) and response.startswith("TIMEOUT"):
                raise RuntimeError(f"Printer ACK timeout during {phase or mode}: {cmd} | {response}")
            # Only advance the software position after the command has been accepted.
            if x is not None:
                self.virtual_position["x"] = float(x)
            if y is not None:
                self.virtual_position["y"] = float(y)
            if z is not None:
                self.virtual_position["z"] = float(z)
            return response

    def query_position(self):
        resp=self.send_line("M114", mode="manual", phase="query_position")
        if self.dry_run:
            return dict(self.virtual_position)
        m=re.search(r"X:\s*(-?\d+(?:\.\d+)?)\s+Y:\s*(-?\d+(?:\.\d+)?)\s+Z:\s*(-?\d+(?:\.\d+)?)", resp, re.I)
        if m:
            self.virtual_position.update(x=float(m.group(1)),y=float(m.group(2)),z=float(m.group(3)))
        return dict(self.virtual_position)

    def query_position_live(self):
        """Best-effort, non-blocking M114 poll for UI live position.

        It never waits for the normal printer command lock. If a Home/Jog/Trial motion
        is using the serial channel, None is returned and the next 1 Hz poll retries.
        Live polls are deliberately not written to stage_commands.csv and do not spam
        the run log.
        """
        if self.dry_run:
            return dict(self.virtual_position)
        if self.ser is None or not getattr(self.ser, "is_open", False):
            return None
        if not self.lock.acquire(blocking=False):
            return None
        try:
            try:
                self.ser.reset_input_buffer()
            except Exception:
                pass
            with self.tx_lock:
                self.ser.write(b"M114\n")
                self.ser.flush()
            resp=self._read_ok(timeout_s=1.2)
            m=re.search(r"X:\s*(-?\d+(?:\.\d+)?)\s+Y:\s*(-?\d+(?:\.\d+)?)\s+Z:\s*(-?\d+(?:\.\d+)?)", resp, re.I)
            if m:
                self.virtual_position.update(x=float(m.group(1)), y=float(m.group(2)), z=float(m.group(3)))
                return dict(self.virtual_position)
            return None
        finally:
            self.lock.release()

    def home(self, axes="XYZ"):
        axes=''.join(a for a in str(axes).upper() if a in 'XYZ') or 'XYZ'
        cmd="G28" if axes=="XYZ" else "G28 "+" ".join(axes)
        # G28 does not need a following M400. Avoid firmware-blocking waits.
        self.send_line(cmd, mode="manual", phase="home", timeout_s=180.0, strict_ack=True)
        self.homed_axes.update(axes)
        return self.query_position()

    def jog(self, axis, delta_mm, speed_mm_s):
        axis=str(axis).upper()
        if axis not in 'XYZ': raise ValueError('axis must be X/Y/Z')
        feed=max(0.05,float(speed_mm_s))*60.0
        self.send_line("G91", mode="manual", phase="relative")
        try:
            self.send_line(f"G1 {axis}{float(delta_mm):.4f} F{feed:.1f}", feed=feed, mode="manual", phase="jog", strict_ack=True)
            est=abs(float(delta_mm))/max(0.05,float(speed_mm_s))
            self._host_motion_wait(est, phase="jog_wait")
        finally:
            self.send_line("G90", mode="manual", phase="absolute")
        return self.query_position()

    def send_urgent(self, cmd: str, phase: str, repeat: int = 1):
        """Write a stop-class command immediately, without waiting on the normal ACK lock."""
        cmd = str(cmd).strip()
        response = "SENT_URGENT"
        if self.dry_run:
            self.log(f"Printer DRY URGENT> {cmd} x{max(1,int(repeat))}")
        else:
            if self.ser is None or not getattr(self.ser, "is_open", False):
                raise RuntimeError("Printer serial port is not open")
            payload = ("\n" + cmd + "\n").encode("ascii", errors="ignore")
            for i in range(max(1, int(repeat))):
                with self.tx_lock:
                    self.ser.write(payload)
                    self.ser.flush()
                if repeat > 1 and i + 1 < repeat:
                    time.sleep(0.02)
            self.log(f"Printer URGENT> {cmd} x{max(1,int(repeat))} | sent without ACK wait")
        self.stage_callback({"command":cmd,"x_mm":None,"y_mm":None,"z_mm":None,
                             "feed_mm_min":None,"mode":"urgent","phase":phase,
                             "dry_run":self.dry_run,"response":response})
        return response

    def quick_stop(self):
        # Old A5S firmware may not implement M410. M84 is a conservative fallback.
        # This intentionally invalidates coordinates and therefore requires HOME before reuse.
        self.send_urgent("M410", "quick_stop")
        time.sleep(0.03)
        return self.send_urgent("M84", "quick_stop_fallback")

    def fan_off(self):
        return self.send_line("M107", mode="manual", phase="fan_off")

    def emergency_stop(self):
        # M112 exists on old Marlin; repeat transmission for robustness.
        return self.send_urgent("M112", "emergency_stop", repeat=3)

    @staticmethod
    def _within_bounds(axis: str, value: float) -> bool:
        lo,hi=A5S_MACHINE_BOUNDS_MM[axis]
        return lo <= float(value) <= hi

    def _guard_absolute_target(self, x=None, y=None, z=None, mode="manual", phase="move"):
        for axis,val in (("x",x),("y",y),("z",z)):
            if val is None:
                continue
            if not math.isfinite(float(val)):
                raise RuntimeError(f"Unsafe non-finite {axis.upper()} target in {phase}: {val}")
            if not self._within_bounds(axis,float(val)):
                lo,hi=A5S_MACHINE_BOUNDS_MM[axis]
                raise RuntimeError(f"Unsafe {axis.upper()} target {float(val):.3f} mm outside A5S bounds [{lo:.1f},{hi:.1f}] during {phase}")
        # Phase-level Z firewall for automatic experiment actions.  Any future accidental
        # 'scan G1 X...Y...Z...' command is rejected before a byte reaches the printer.
        if mode in EXPERIMENT_ACTION_LABELS and z is not None:
            if not any(str(phase).startswith(prefix) for prefix in AUTO_Z_ALLOWED_PHASE_PREFIXES):
                raise RuntimeError(f"SAFETY_Z_PHASE_BLOCK: action={mode}, phase={phase}, attempted Z={float(z):.3f} mm")

    def _host_motion_wait(self, est_s: float, phase: str = "motion_wait"):
        """Interruptible host-side wait that leaves the A5S Marlin parser responsive."""
        est_s = max(0.0, float(est_s))
        margin = max(0.20, min(1.50, 0.15 * est_s + 0.15))
        wait_s = est_s + margin
        deadline = time.monotonic() + wait_s
        while True:
            if self.abort_event is not None and self.abort_event.is_set():
                raise RuntimeError("MOTION_ABORTED")
            remain = deadline - time.monotonic()
            if remain <= 0:
                break
            if self.abort_event is not None:
                self.abort_event.wait(min(0.02, remain))
            else:
                time.sleep(min(0.02, remain))
        self.stage_callback({"command":"HOST_WAIT","x_mm":None,"y_mm":None,"z_mm":None,
                             "feed_mm_min":None,"mode":"host_wait","phase":phase,
                             "dry_run":self.dry_run,"response":f"EST={est_s:.3f}s WAIT={wait_s:.3f}s"})

    def move_abs(self, x=None, y=None, z=None, speed_mm_s=2.0, mode="manual", phase="move"):
        if self.abort_event is not None and self.abort_event.is_set():
            raise RuntimeError("MOTION_ABORTED")
        self._guard_absolute_target(x=x,y=y,z=z,mode=mode,phase=phase)
        # SAFETY: every absolute move re-asserts G90. A previous jog/abort/manual G91
        # must never make an experiment's absolute XYZ targets become relative moves.
        auto = mode in EXPERIMENT_ACTION_LABELS
        self.send_line("G90", mode=mode, phase=phase + "_absolute_guard", strict_ack=auto)
        if self.abort_event is not None and self.abort_event.is_set():
            raise RuntimeError("MOTION_ABORTED")
        speed=max(0.05,float(speed_mm_s))
        feed = speed * 60.0
        prev=dict(self.virtual_position)
        tx=prev["x"] if x is None else float(x); ty=prev["y"] if y is None else float(y); tz=prev["z"] if z is None else float(z)
        dist=math.sqrt((tx-prev["x"])**2+(ty-prev["y"])**2+(tz-prev["z"])**2)
        est_s=dist/speed if speed>0 else 0.0
        parts = ["G1"]
        if x is not None: parts.append(f"X{float(x):.4f}")
        if y is not None: parts.append(f"Y{float(y):.4f}")
        if z is not None: parts.append(f"Z{float(z):.4f}")
        parts.append(f"F{feed:.1f}")
        self.send_line(" ".join(parts), x=x, y=y, z=z, feed=feed, mode=mode, phase=phase, strict_ack=auto)
        if self.abort_event is not None and self.abort_event.is_set():
            raise RuntimeError("MOTION_ABORTED")
        # NEVER place M400 behind automatic movement on this A5S firmware. Without
        # EMERGENCY_PARSER an executing M400 prevents the firmware from parsing M410/M112.
        self._host_motion_wait(est_s, phase=phase + "_host_wait")

    def dwell(self, seconds: float, mode="manual", phase="hold"):
        if self.abort_event is not None and self.abort_event.is_set():
            raise RuntimeError("MOTION_ABORTED")
        seconds=max(0.0,float(seconds))
        deadline=time.monotonic()+seconds
        while True:
            if self.abort_event is not None and self.abort_event.is_set():
                raise RuntimeError("MOTION_ABORTED")
            remain=deadline-time.monotonic()
            if remain<=0: break
            if self.abort_event is not None:
                self.abort_event.wait(min(0.02,remain))
            else:
                time.sleep(min(0.02,remain))
        self.stage_callback({"command":"HOST_DWELL","x_mm":None,"y_mm":None,"z_mm":None,
                             "feed_mm_min":None,"mode":mode,"phase":phase,
                             "dry_run":self.dry_run,"response":f"{seconds:.3f}s"})

    def execute_trial(self, params: TrialParameters, contact_z_abs: float, safe_lift_mm: float):
        """Execute a conservative single-contact trajectory.

        IMPORTANT: contact_z_abs must be physically calibrated by the user. The code does
        not infer contact height. Positive z_indent_mm means moving downward by reducing
        printer Z: z_target = contact_z_abs - z_indent_mm.
        """
        mode = params.command_mode
        safe_z = contact_z_abs + abs(float(safe_lift_mm))
        z_press = contact_z_abs - max(0.0, params.z_indent_mm)
        x0, y0 = params.x_mm, params.y_mm
        speed = max(0.1, params.speed_mm_s)

        self.move_abs(z=safe_z, speed_mm_s=speed, mode=mode, phase="safe_z")
        self.move_abs(x=x0, y=y0, speed_mm_s=speed, mode=mode, phase="xy_start")
        self.dwell(0.15, mode=mode, phase="pre_contact")

        if mode == "none":
            self.dwell(max(0.5, params.hold_s), mode=mode, phase="none_hold")
            return

        self.move_abs(z=z_press, speed_mm_s=speed, mode=mode, phase="preload_indent")
        self.dwell(params.hold_s, mode=mode, phase="press_hold")

        if mode == "slide_x":
            self.move_abs(x=x0 + params.dx_mm, y=y0, speed_mm_s=speed,
                          mode=mode, phase="slide_x")
        elif mode == "slide_y":
            self.move_abs(x=x0, y=y0 + params.dy_mm, speed_mm_s=speed,
                          mode=mode, phase="slide_y")
        elif mode == "compound":
            # Legacy helper behaves as preload + XY compound translation; Z is intentionally
            # omitted here so a lateral phase can never issue another downward target.
            self.move_abs(x=x0 + params.dx_mm, y=y0 + params.dy_mm,
                          speed_mm_s=speed, mode=mode, phase="compound_xy_legacy")
        elif mode == "press":
            pass
        else:
            self.log(f"Printer: unknown core mode '{mode}', executing press-only trajectory")

        self.dwell(params.hold_s, mode=mode, phase="post_motion_hold")
        self.move_abs(z=safe_z, speed_mm_s=speed, mode=mode, phase="release")


# ---------------------------------------------------------------------------
# Simple 5-channel live plot (magnitude by default)
# ---------------------------------------------------------------------------

class MultiMagPlot(tk.Canvas):
    """Five-sensor magnetic plot with selectable Bx/By/Bz/|B| or all XYZ.

    Data are retained per axis so switching view never restarts acquisition or loses
    the recent history. In All-XYZ mode sensor identity is encoded by color while
    axis identity is encoded by line style.
    """
    COLORS = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854"]
    AXIS_DASH = {"x": None, "y": (6, 3), "z": (2, 2)}
    AXIS_LABEL = {"x":"Bx", "y":"By", "z":"Bz", "m":"|B|"}

    def __init__(self, master, maxlen=400, view_mode="m", **kwargs):
        super().__init__(master, bg="#101318", highlightthickness=0, **kwargs)
        from collections import deque
        self.maxlen=int(maxlen)
        self.view_mode=view_mode if view_mode in ("x","y","z","m","all") else "m"
        self.data=[{a:deque(maxlen=self.maxlen) for a in ("x","y","z","m")} for _ in range(5)]
        self.bind("<Configure>", lambda e: self.redraw())

    def set_mode(self, mode):
        code=MAG_VIEW_LABELS.get(str(mode), str(mode))
        if code not in ("x","y","z","m","all"):
            code="m"
        self.view_mode=code
        self.redraw()

    @staticmethod
    def _sensor_values(d):
        if d is None:
            return {a:float("nan") for a in ("x","y","z","m")}
        def f(key):
            try:return float(d.get(key,float("nan")))
            except Exception:return float("nan")
        return {"x":f("x_ut"),"y":f("y_ut"),"z":f("z_ut"),"m":f("mag_ut")}

    def add(self, data: dict[int, dict], redraw=True):
        for i in range(5):
            vals=self._sensor_values(data.get(i))
            for a,v in vals.items(): self.data[i][a].append(v)
        if redraw:self.redraw()

    def clear(self, redraw=True):
        for sensor in self.data:
            for q in sensor.values(): q.clear()
        if redraw:self.redraw()

    def set_window(self, samples):
        """Replace visible history with iterable of live-like {sensor_index:dict} samples."""
        self.clear(redraw=False)
        for sample in samples:
            self.add(sample,redraw=False)
        self.redraw()

    def _selected_series(self):
        if self.view_mode=="all":
            return [(i,a,self.data[i][a]) for i in range(5) for a in ("x","y","z")]
        return [(i,self.view_mode,self.data[i][self.view_mode]) for i in range(5)]

    def redraw(self):
        self.delete("all")
        w,h=max(100,self.winfo_width()),max(80,self.winfo_height())
        pl,pr,pt,pb=58,15,38,26
        x0,x1,y0,y1=pl,w-pr,h-pb,pt
        self.create_rectangle(x0,y1,x1,y0,outline="#4c566a")
        selected=self._selected_series()
        finite=[v for _i,_a,q in selected for v in q if math.isfinite(v)]
        mode_title="全部 XYZ / All XYZ" if self.view_mode=="all" else self.AXIS_LABEL.get(self.view_mode,self.view_mode)
        self.create_text(8,6,anchor="nw",fill="#d8dee9",text=f"5× {mode_title} 实时曲线 (µT)")
        # Sensor legend
        for i in range(5):
            self.create_text(165+i*68,6,anchor="nw",fill=self.COLORS[i],text=f"S{i+1}")
        if self.view_mode=="all":
            self.create_text(8,22,anchor="nw",fill="#aeb7c2",text="线型 / Axis: Bx=实线  By=长虚线  Bz=点线")
        if len(finite)<2:
            self.create_text((x0+x1)/2,(y0+y1)/2,fill="#7f8c9d",text="等待数据… / Waiting for data…")
            return
        lo,hi=min(finite),max(finite)
        if hi-lo<1e-9: lo-=1;hi+=1
        margin=(hi-lo)*0.08
        lo-=margin;hi+=margin
        self.create_text(5,y1,anchor="nw",fill="#8893a2",text=f"{hi:.1f}")
        self.create_text(5,y0-12,anchor="nw",fill="#8893a2",text=f"{lo:.1f}")
        for i,a,q in selected:
            arr=list(q);n=len(arr)
            if n<2:continue
            # Split at NaNs so missing points do not create false bridges.
            segment=[]
            def draw_seg(seg):
                if len(seg)>=4:
                    kw={"fill":self.COLORS[i],"width":1.35}
                    dash=self.AXIS_DASH.get(a) if self.view_mode=="all" else None
                    if dash:kw["dash"]=dash
                    self.create_line(*seg,**kw)
            for j,v in enumerate(arr):
                if not math.isfinite(v):
                    draw_seg(segment);segment=[];continue
                px=x0+(x1-x0)*j/max(1,n-1)
                py=y0-(y0-y1)*(v-lo)/(hi-lo)
                segment.extend([px,py])
            draw_seg(segment)


class LiveDxyzPlot(tk.Canvas):
    """Compact live visualization for the three selected model outputs."""
    COLORS = ("#4fc3f7", "#ffb74d", "#81c784")
    NAMES: tuple[str, str, str] = ("ΔX", "ΔY", "ΔZ")

    def __init__(self, master, maxlen=180, **kwargs):
        super().__init__(master, bg="#101318", highlightthickness=0, **kwargs)
        from collections import deque
        self.values = deque(maxlen=int(maxlen))
        self.axis_scale = [1.0, 1.0, 1.0]
        self.display_title = "模型实时预测 / Live model prediction (mm)"
        self.empty_title = "载入 R15 模型并采集静止基线后显示预测 / Load model + capture B0"
        self.bind("<Configure>", lambda _event: self.redraw())

    def set_scale(self, scale):
        try:
            self.axis_scale = [max(0.10, abs(float(value))) for value in scale]
        except Exception:
            self.axis_scale = [1.0, 1.0, 1.0]
        self.redraw()

    def clear(self):
        self.values.clear()
        self.redraw()

    def remove_tail(self,count,redraw=True):
        """Remove already drawn samples after a retrospectively confirmed release."""
        for _ in range(max(0,int(count))):
            if not self.values:break
            self.values.pop()
        if redraw:self.redraw()

    def add(self, dx, dy, dz, redraw=True):
        self.values.append((float(dx), float(dy), float(dz)))
        if redraw:
            self.redraw()

    def redraw(self):
        self.delete("all")
        w, h = max(260, self.winfo_width()), max(180, self.winfo_height())
        left, right, top, bottom = 62, 16, 28, 18
        row_h = (h - top - bottom) / 3.0
        self.create_text(8, 7, anchor="nw", fill="#d8dee9",
                         text=getattr(self,"display_title","模型实时预测 / Live model prediction (mm)"))
        if not self.values:
            self.create_text(w / 2, h / 2, fill="#7f8c9d",
                             text=getattr(self,"empty_title","载入 R15 模型并采集静止基线后显示预测 / Load model + capture B0"))
            return
        values = list(self.values)
        for axis in range(3):
            y0 = top + axis * row_h
            y1 = y0 + row_h - 5
            mid = (y0 + y1) / 2.0
            scale = self.axis_scale[axis]
            self.create_rectangle(left, y0, w - right, y1, outline="#4c566a")
            self.create_line(left, mid, w - right, mid, fill="#64748b", dash=(3, 3))
            self.create_text(8, mid - 7, anchor="nw", fill=self.COLORS[axis],
                             text=f"{self.NAMES[axis]}  ±{scale:.2f}")
            points = []
            for index, row in enumerate(values):
                x = left + (w - right - left) * index / max(1, len(values) - 1)
                value = max(-scale, min(scale, row[axis]))
                y = mid - (mid - y0 - 4) * value / scale
                points.extend((x, y))
            if len(points) >= 4:
                self.create_line(*points, fill=self.COLORS[axis], width=1.7)
            latest = values[-1][axis]
            self.create_text(w - right - 3, y0 + 3, anchor="ne", fill=self.COLORS[axis],
                             text=f"{latest:+.4f} mm")


class LiveDxyz3DView(tk.Canvas):
    """Interactive isometric view containing model inference only."""
    AXIS_COLORS = ("#ef5350", "#66bb6a", "#42a5f5")

    def __init__(self, master, maxlen=100, **kwargs):
        super().__init__(master, bg="#101318", highlightthickness=0, **kwargs)
        from collections import deque
        self.values = deque(maxlen=int(maxlen))
        self.span_mm = 1.0
        self.yaw_deg = -45.0
        self.pitch_deg = 28.0
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._drag = None
        self._pan_drag = None
        self.on_view_changed: Optional[Callable[[], None]] = None
        self.display_title = "三维模型推理 / 3D model inference"
        self.display_subtitle = "仅显示模型输出；不添加人工锚点或打印机计划参考"
        self.bind("<Configure>", lambda _event: self.redraw())
        self.bind("<ButtonPress-1>", self._drag_begin)
        self.bind("<B1-Motion>", self._drag_move)
        self.bind("<ButtonRelease-1>", self._drag_end)
        self.bind("<ButtonPress-3>", self._pan_begin)
        self.bind("<B3-Motion>", self._pan_move)
        self.bind("<ButtonRelease-3>", self._pan_end)
        self.bind("<MouseWheel>", self._zoom_wheel)
        self.bind("<Button-4>", lambda _event: self._zoom_step(1))
        self.bind("<Button-5>", lambda _event: self._zoom_step(-1))

    def _drag_begin(self, event):
        self._drag = (event.x, event.y, self.yaw_deg, self.pitch_deg)

    def _drag_move(self, event):
        if self._drag is None:
            return
        x0, y0, yaw0, pitch0 = self._drag
        self.yaw_deg = yaw0 + (event.x - x0) * 0.5
        self.pitch_deg = max(-5.0, min(78.0, pitch0 + (event.y - y0) * 0.35))
        self.redraw()

    def _drag_end(self, _event):
        if self._drag is None:
            return
        self._drag = None
        if callable(self.on_view_changed):
            self.on_view_changed()

    def _pan_begin(self, event):
        self._pan_drag = (event.x, event.y, self.pan_x, self.pan_y)

    def _pan_move(self, event):
        if self._pan_drag is None:
            return
        x0, y0, pan_x0, pan_y0 = self._pan_drag
        self.pan_x = pan_x0 + event.x - x0
        self.pan_y = pan_y0 + event.y - y0
        self.redraw()

    def _pan_end(self, _event):
        if self._pan_drag is None:
            return
        self._pan_drag = None
        if callable(self.on_view_changed):
            self.on_view_changed()

    def _zoom_wheel(self, event):
        if getattr(event, "delta", 0):
            self._zoom_step(1 if event.delta > 0 else -1)
        return "break"

    def _zoom_step(self, direction):
        factor = 1.12 if direction > 0 else 1.0 / 1.12
        self.zoom = max(0.25, min(8.0, self.zoom * factor))
        self.redraw()
        if callable(self.on_view_changed):
            self.on_view_changed()

    def set_scale(self, scale):
        try:
            self.span_mm = max(0.10, max(abs(float(value)) for value in scale))
        except Exception:
            self.span_mm = 1.0
        self.redraw()

    def set_view(self, yaw_deg, pitch_deg, zoom=1.0, pan_x=0.0, pan_y=0.0):
        """Restore a saved user viewpoint without treating it as measurement data."""
        try:
            self.yaw_deg = float(yaw_deg)
            self.pitch_deg = max(-5.0, min(78.0, float(pitch_deg)))
            self.zoom = max(0.25, min(8.0, float(zoom)))
            self.pan_x = float(pan_x)
            self.pan_y = float(pan_y)
        except (TypeError, ValueError):
            return
        self.redraw()

    def clear(self):
        self.values.clear()
        self.redraw()

    @property
    def maxlen(self):
        return int(self.values.maxlen or 0)

    def set_maxlen(self, maxlen, redraw=True):
        """Resize display-only trajectory history while preserving recent points."""
        from collections import deque
        value = int(maxlen)
        if not 50 <= value <= 50000:
            raise ValueError("三维轨迹历史点数须为 50～50000")
        if value == self.maxlen:
            return
        self.values = deque(list(self.values)[-value:], maxlen=value)
        if redraw:
            self.redraw()

    def remove_tail(self,count,redraw=True):
        """Remove only the latest contact points; never cross a segment gap."""
        for _ in range(max(0,int(count))):
            if not self.values or self.values[-1] is None:break
            self.values.pop()
        if redraw:self.redraw()

    def add(self, dx, dy, dz, redraw=True):
        self.values.append((float(dx), float(dy), float(dz)))
        if redraw:
            self.redraw()

    def add_gap(self, redraw=True):
        """Preserve contact segments, hide the marker, never join across air travel."""
        if self.values and self.values[-1] is not None:
            self.values.append(None)
        if redraw:self.redraw()

    def set_window(self, values):
        """Replace the visible trajectory in one redraw (used by replay)."""
        self.values.clear()
        for row in values:
            if row is None:
                if self.values and self.values[-1] is not None:self.values.append(None)
                continue
            try:
                point=tuple(float(value) for value in row[:3])
                if len(point)==3 and all(math.isfinite(v) for v in point):self.values.append(point)
            except (TypeError, ValueError):
                continue
        self.redraw()

    def _project(self, point, center, pixels_per_mm):
        # World convention mirrors R15 labels: +Z is predicted indentation/down.
        x, y, z = (float(v) for v in point)
        yaw = math.radians(self.yaw_deg)
        pitch = math.radians(self.pitch_deg)
        xr = x * math.cos(yaw) - y * math.sin(yaw)
        yr = x * math.sin(yaw) + y * math.cos(yaw)
        # Screen y grows down, so positive R15 Z (indentation) is displayed down.
        return center[0] + xr * pixels_per_mm, center[1] + (yr * math.sin(pitch) + z * math.cos(pitch)) * pixels_per_mm

    @staticmethod
    def _arrow(canvas, x0, y0, x1, y1, color, width=2):
        canvas.create_line(x0, y0, x1, y1, fill=color, width=width, arrow="last", arrowshape=(9, 11, 4))

    def redraw(self):
        self.delete("all")
        w, h = max(260, self.winfo_width()), max(220, self.winfo_height())
        center = (w * 0.50 + self.pan_x, h * 0.52 + self.pan_y)
        pixels_per_mm = min(w, h) * 0.30 * self.zoom / max(self.span_mm, 1e-6)
        self.create_text(8, 7, anchor="nw", fill="#d8dee9",
                         text=getattr(self,"display_title","三维模型推理 / 3D model inference"))
        self.create_text(8, 24, anchor="nw", fill="#8fa3b8",
                         text=getattr(self,"display_subtitle","仅显示模型输出；不添加人工锚点或打印机计划参考"))
        self.create_text(8, 41, anchor="nw", fill="#7f8c9d",
                         text="黄点=当前接触，蓝线=接触轨迹；离开接触则断开，不连接原点。")
        origin = self._project((0, 0, 0), center, pixels_per_mm)
        # A small XY reference grid makes radial/arc motion visible in 3D.
        grid_extent = self.span_mm
        for fraction in (-1.0, -0.5, 0.5, 1.0):
            x0, y0 = self._project((-grid_extent, fraction * grid_extent, 0), center, pixels_per_mm)
            x1, y1 = self._project(( grid_extent, fraction * grid_extent, 0), center, pixels_per_mm)
            self.create_line(x0, y0, x1, y1, fill="#263746", dash=(2, 4))
            x0, y0 = self._project((fraction * grid_extent, -grid_extent, 0), center, pixels_per_mm)
            x1, y1 = self._project((fraction * grid_extent,  grid_extent, 0), center, pixels_per_mm)
            self.create_line(x0, y0, x1, y1, fill="#263746", dash=(2, 4))
        axes = ((self.span_mm, 0, 0), (0, self.span_mm, 0), (0, 0, self.span_mm))
        labels = ("+X", "+Y", "+Z ↓压入")
        for endpoint, color, label in zip(axes, self.AXIS_COLORS, labels):
            ex, ey = self._project(endpoint, center, pixels_per_mm)
            self._arrow(self, origin[0], origin[1], ex, ey, color, 2)
            self.create_text(ex, ey, anchor="sw", fill=color, text=label)
        self.create_oval(origin[0] - 4, origin[1] - 4, origin[0] + 4, origin[1] + 4,
                         fill="#eceff1", outline="")
        if not self.values:
            self.create_text(w / 2, h * 0.77, fill="#7f8c9d",
                             text="等待有效接触预测 / Waiting for inferred contact")
            return
        trail = list(self.values)
        non_gap_count = sum(point is not None for point in trail)
        # Long experiment histories are stored in full but rendered with a bounded
        # point budget.  Each contact segment keeps its endpoints, so a 121-point
        # experiment remains visible without thousands of per-frame Canvas items.
        stride = max(1, math.ceil(non_gap_count / 1600))
        segment = []
        def draw_segment(points):
            if len(points) < 2:
                return
            sampled = points[::stride]
            if sampled[-1] != points[-1]:
                sampled.append(points[-1])
            projected = []
            for point in sampled:
                projected.extend(self._project(point, center, pixels_per_mm))
            if len(projected) >= 4:
                self.create_line(*projected, fill="#4c8bb8", width=2)
        for point in trail:
            if point is None:
                draw_segment(segment); segment = []
            else:
                segment.append(point)
        draw_segment(segment)
        self.create_text(w - 10, 26, anchor="ne", fill="#7f8c9d",
                         text=f"历史 {non_gap_count}/{self.maxlen} 点")
        latest = trail[-1]
        if latest is None:
            self.create_text(w/2,h*.90,fill="#7f8c9d",text="非接触/释放/无效输出：保留接触轨迹，不显示当前位置")
            return
        ex, ey = self._project(latest, center, pixels_per_mm)
        radius = 7
        self.create_oval(ex - radius, ey - radius, ex + radius, ey + radius, fill="#ffd54f", outline="#fff8e1")
        self.create_text(ex + 10, ey - 10, anchor="sw", fill="#fff8e1",
                         text=f"XYZ=({latest[0]:+.3f}, {latest[1]:+.3f}, {latest[2]:+.3f}) mm")
        rxy = math.hypot(latest[0], latest[1])
        theta_deg = math.degrees(math.atan2(latest[1], latest[0])) if rxy > 1e-9 else 0.0
        self.create_text(w - 10, 9, anchor="ne", fill="#9bd4ff",
                         text=f"rXY={rxy:.3f} mm   θ={theta_deg:+.1f}°")


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class MagTouchApp(tk.Tk):
    def __init__(self, simulate=False):
        super().__init__()
        self.title(APP_NAME)
        self.geometry("1600x1000")
        self.minsize(1280, 780)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.simulate = bool(simulate)
        self.settings_path = SETTINGS_PATH
        self.settings = load_runtime_settings(self.settings_path)
        self.maglayer_profiles_path = MAGNETIC_LAYER_PROFILES_PATH
        self.maglayer_profiles_doc = load_magnetic_layer_profiles(self.maglayer_profiles_path)
        self.maglayer_profiles = self.maglayer_profiles_doc.get("profiles",{})
        self.active_mag_profile_id = str(self.settings.get("magnetic",{}).get(
            "active_layer_profile_id", self.maglayer_profiles_doc.get("default_profile_id","")
        ))
        if self.active_mag_profile_id not in self.maglayer_profiles and self.maglayer_profiles:
            self.active_mag_profile_id = next(iter(self.maglayer_profiles))

        self.geometry_config = json.loads(json.dumps(DEFAULT_GEOMETRY))
        self.bridge = CH341Bridge()
        self.sampler = None
        self.mag_thread = None
        self.mag_stop = threading.Event()
        self.mag_queue: queue.Queue = queue.Queue(maxsize=100)
        self.last_mag = None
        # True sensor-domain history used by automatic contact-Z probing.  Each row
        # is (perf_counter_ns, 15-channel S1..S5 XYZ in uT), before any ML model.
        self.raw_magnetic_recent = deque(maxlen=600)
        self.mag_fingerprint_history = deque(maxlen=120)
        self.mag_fingerprint_reference = None
        self.actual_mag_periods = []
        self.last_mag_group_ns = None

        # Online H1/H2-R15 inference is kept separate from acquisition. The worker only
        # consumes copies of live magnetic groups and never issues printer motion.
        self.live_predictor: Optional[UnifiedLivePredictor | LiveNormalizedFitPredictor] = None
        self.live_predict_lock = threading.RLock()
        self.live_predict_enabled = False
        self.live_contact_segment = 0
        self.live_contact_previous = None
        self.live_contact_onset_ns = None
        self.live_contact_last_end_ns = None
        self.live_contact_trimmed_end_ns = None
        self.live_display_point_times = deque(maxlen=160)
        self.live_alignment_recent = deque(maxlen=24)
        # Raw, B0-subtracted 15-channel groups used by R47 source-data
        # fingerprint calibration.  This cache is independent of predicted XY.
        self.live_source_fingerprint_recent = deque(maxlen=160)
        self.live_alignment_auto_running = False
        self.live_contact_shown_segment = -1
        # Display-only trajectory filter.  It is intentionally downstream of
        # prediction recording so live_predictions.csv always retains raw model output.
        self.live_trajectory_filter_state: dict[str, Any] = {
            "signature": None, "last_t_ns": None, "ema": None, "axes": None,
        }
        self.live_predict_queue: queue.Queue = queue.Queue(maxsize=80)
        self.live_predict_results: queue.Queue = queue.Queue(maxsize=8)
        self.live_predict_record_recorder: Optional[TrialRecorder] = None
        self.live_predict_record_source = ""
        self.live_predict_dropped_groups = 0
        self.live_predict_min_input_ns = 0
        self.live_predict_stop = threading.Event()
        self.live_predict_thread = threading.Thread(target=self._live_predict_loop,
                                                     name="MagTouchLiveMultiModel", daemon=True)
        self.live_predict_thread.start()

        self.preview_queue: queue.Queue = queue.Queue(maxsize=6)
        self.cameras: dict[str, CameraWorker] = {}
        self.preview_labels = {}
        self.preview_images = {}
        # Experiment page owns its own lightweight preview surfaces. Only the active
        # page is rendered; acquisition/recording continues independently in background.
        self.exp_preview_labels = {}
        self.exp_preview_images = {}

        self.recorder: Optional[TrialRecorder] = None
        self.trial_lock = threading.RLock()

        self.printer = PrinterController(self.log, self._record_stage)
        self._live_stage_safe_height = False
        self._live_stage_pending_safe_height = False
        self.printer_thread = None
        self.z_minus_confirmed_once = False
        self.contact_action_confirmed_once = False
        self.motion_abort_event = threading.Event()
        self.printer.abort_event = self.motion_abort_event
        self.batch_abort_event = threading.Event()
        # Full acquisition-plan scheduler. Pause/soft-stop are honored between Trials.
        # Emergency stop on this A5S is physical printer power-off, not a USB M112 button.
        self.full_plan_items = []
        # Frozen geometry snapshot of the generated/loaded Full Plan.
        # A Plan is generated from the current Printer settings, then keeps that
        # reference center/contact geometry even if Printer settings are edited later.
        self.full_plan_reference_center = None
        self.full_plan_reference_contact_z = None
        self.full_plan_thread = None
        self.full_plan_pause_event = threading.Event()
        self.full_plan_soft_stop_event = threading.Event()
        self.full_plan_running = False
        self.full_plan_dynamic_b0_enabled = False
        self.full_plan_path = HERE.parent / "config" / "batch_plan_current.json"
        # When a Full Plan is running, all generated Trial_* directories are kept
        # inside one Plan_* run folder under the current Session for easy management.
        self.active_plan_run_dir: Optional[Path] = None
        self.active_plan_run_id: Optional[str] = None
        self.active_plan_run_meta_path: Optional[Path] = None
        # Printer position live monitor. 1 Hz is enough for UI feedback and avoids
        # stealing serial bandwidth from experimental motion commands.
        self._printer_pos_poll_inflight = False
        self._printer_pos_poll_after_id = None

        self._training_proc: Optional[subprocess.Popen[str]] = None
        self._training_proc_label = ""

        self._build_ui()
        self._install_parameter_autosave()
        # Restore the selected checkpoint automatically when it still exists.  This
        # deliberately does not restore B0 or start inference: B0 is a fresh local
        # no-load reference and must be captured for each physical session.
        self.after(550, self._live_predict_autoload_saved_model)
        # Session/root changes automatically rescan existing Trial_* folders.
        try:
            self.session_var.trace_add("write",lambda *_:self.after(250,lambda:self.scan_next_trial_id(silent=True) if self.auto_trial_id_var.get() else None))
            self.data_root_var.trace_add("write",lambda *_:self.after(250,lambda:self.scan_next_trial_id(silent=True) if self.auto_trial_id_var.get() else None))
            self.after(300,lambda:self.scan_next_trial_id(silent=True) if self.auto_trial_id_var.get() else None)
        except Exception: pass
        self._update_home_state()
        self.after(25, self._drain_queues)
        self.after(1000, self._update_rates)
        self._printer_pos_poll_after_id = self.after(1000, self._live_printer_position_tick)
        if self.simulate:
            self.log("SIMULATION mode: magnetic hardware is simulated")
        self.log(f"Project Python / 当前项目解释器: {sys.executable}")

    # ------------------------- UI -------------------------
    def _build_ui(self):
        style = ttk.Style(self)
        try: style.theme_use("clam")
        except Exception: pass
        style.configure("TButton", padding=(8, 5))
        style.configure("TLabelframe", padding=5)

        top = ttk.Frame(self, padding=7)
        top.pack(fill="x")
        ttk.Label(top, text=APP_NAME, font=("Segoe UI", 14, "bold")).pack(side="left")
        ttk.Label(top, text=f"  Build {BUILD_ID}  |  {Path(__file__).name}", foreground="#555555").pack(side="left", padx=(10,0))
        # A5S old Marlin does not provide a trustworthy USB emergency stop.
        # Keep only a software sequence-stop control; real emergency stop is physical power-off.
        tk.Button(top, text="停止实验序列 / STOP SEQUENCE", command=self.printer_quick_stop,
                  bg="#ef6c00", fg="white", activebackground="#c43e00", activeforeground="white",
                  font=("Segoe UI", 10, "bold"), padx=12, pady=4).pack(side="right", padx=(4,8))
        ttk.Label(top, text="急停=物理断电 / Emergency = POWER OFF", foreground="#b71c1c",
                  font=("Segoe UI", 9, "bold")).pack(side="right", padx=8)
        self.record_state_var = tk.StringVar(value="IDLE")
        ttk.Label(top, textvariable=self.record_state_var, font=("Segoe UI", 11, "bold")).pack(side="right", padx=12)

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=6, pady=(0, 4))
        self.tab_acq = ttk.Frame(self.nb)
        self.tab_predict = ttk.Frame(self.nb)
        self.tab_maglayer = ttk.Frame(self.nb)
        self.tab_cam = ttk.Frame(self.nb)
        self.tab_exp = ttk.Frame(self.nb)
        self.tab_plan = ttk.Frame(self.nb)
        self.tab_data = ttk.Frame(self.nb)
        self.tab_analysis = ttk.Frame(self.nb)
        self.tab_train = ttk.Frame(self.nb)
        self.tab_ubuntu = ttk.Frame(self.nb)
        self.tab_printer = ttk.Frame(self.nb)
        self.tab_info = ttk.Frame(self.nb)
        self.nb.add(self.tab_acq, text="磁场采集 / Magnetics")
        self.nb.add(self.tab_predict, text="实时预测 / Live Prediction")
        self.nb.add(self.tab_maglayer, text="磁层配置 / Magnetic Layer")
        self.nb.add(self.tab_cam, text="相机/视频 / Camera (Optional)")
        self.nb.add(self.tab_exp, text="实验 / Trial")
        self.nb.add(self.tab_plan, text="全部实验采集 / Full Plan")
        self.nb.add(self.tab_data, text="数据管理/回放 / Data & Replay")
        self.nb.add(self.tab_analysis, text="研究分析 / Research Analysis")
        self.nb.add(self.tab_train, text="导出/训练 / Export & Training")
        self.nb.add(self.tab_ubuntu, text="Ubuntu 训练导出 / Ubuntu Training")
        self.nb.add(self.tab_printer, text="3D打印机 / Printer")
        self.nb.add(self.tab_info, text="几何/说明 / Geometry")

        self._build_acq_tab()
        self._build_live_prediction_tab()
        self._build_maglayer_tab()
        self._build_camera_tab()
        self._build_experiment_tab()
        self._build_data_tab()
        self._build_analysis_tab()
        self._build_training_tab()
        self._build_ubuntu_training_tab()
        self._build_printer_tab()
        self._build_full_plan_tab()
        self._build_info_tab()
        self._apply_selected_magnetic_layer_profile(disconnect=False,quiet=True)
        self.nb.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        log_frame = ttk.LabelFrame(self, text="运行日志 / Runtime Log")
        log_frame.pack(fill="x", padx=7, pady=(0, 7))
        self.logbox = ScrolledText(log_frame, height=7, font=("Consolas", 9))
        self.logbox.pack(fill="both", expand=True)


    def _research_database_scan(self):
        import tkinter.filedialog as fd
        p=fd.askdirectory()
        if not p:
            return
        records=scan_trials(p) if scan_trials else []
        print("Research database:", len(records), "plans")

    def _build_acq_tab(self):
        tab = self.tab_acq
        ctl = ttk.LabelFrame(tab, text="CH341 / MLX90393")
        ctl.pack(fill="x", padx=8, pady=8)
        mcfg=self.settings.get("magnetic",{})
        self.dev_index_var = tk.IntVar(value=int(mcfg.get("ch341_index",0)))
        self.i2c_clock_var = tk.StringVar(value=str(mcfg.get("i2c_clock","400 kHz")))
        self.mag_interval_var = tk.IntVar(value=int(mcfg.get("sample_period_ms",20)))
        self.include_temp_var = tk.BooleanVar(value=bool(mcfg.get("include_temperature_each_cycle",False)))
        ttk.Label(ctl, text="CH341设备索引 / Device #").grid(row=0, column=0, sticky="w")
        ttk.Entry(ctl, textvariable=self.dev_index_var, width=6).grid(row=0, column=1, padx=3)
        ttk.Label(ctl, text="I²C时钟 / Clock").grid(row=0, column=2, padx=(12,3))
        ttk.Combobox(ctl, textvariable=self.i2c_clock_var, values=["100 kHz","400 kHz","750 kHz"], width=10, state="readonly").grid(row=0, column=3)
        ttk.Label(ctl, text="目标采样周期 / Target period (ms)").grid(row=0, column=4, padx=(12,3))
        ttk.Entry(ctl, textvariable=self.mag_interval_var, width=7).grid(row=0, column=5)
        ttk.Checkbutton(ctl, text="每周期读取温度 / Temperature each cycle (slower)", variable=self.include_temp_var).grid(row=0, column=6, padx=10)
        ttk.Button(ctl, text="连接/初始化 / Connect", command=self.connect_magnetics).grid(row=0, column=7, padx=4)
        ttk.Button(ctl, text="开始采样 / Start", command=self.start_magnetics).grid(row=0, column=8, padx=4)
        ttk.Button(ctl, text="停止采样 / Stop", command=self.stop_magnetics).grid(row=0, column=9, padx=4)
        self.active_mag_profile_summary_var=tk.StringVar(value="磁层配置未加载")
        ttk.Label(ctl,textvariable=self.active_mag_profile_summary_var,foreground="#0d47a1",wraplength=1450,justify="left").grid(
            row=1,column=0,columnspan=10,sticky="w",padx=3,pady=(5,1))

        addr_frame = ttk.LabelFrame(tab, text="5个传感器地址 / 5 Sensor I²C Addresses（S1~S5物理位置由几何配置定义）")
        addr_frame.pack(fill="x", padx=8, pady=(0,8))
        self.addr_vars=[]
        configured_addrs=self.settings.get("magnetic",{}).get("addresses",[f"0x{a:02X}" for a in DEFAULT_ADDRS])
        for i, a in enumerate(DEFAULT_ADDRS):
            ttk.Label(addr_frame, text=f"S{i+1}").grid(row=0,column=i*2,padx=(8,2),pady=4)
            v=tk.StringVar(value=str(configured_addrs[i]) if i < len(configured_addrs) else f"0x{a:02X}"); self.addr_vars.append(v)
            ttk.Entry(addr_frame,textvariable=v,width=7).grid(row=0,column=i*2+1,padx=(0,10))

        stat = ttk.LabelFrame(tab, text="实时状态 / Live Status")
        stat.pack(fill="x", padx=8, pady=(0,8))
        headers=["节点 / Node","地址 / Address","Bx µT","By µT","Bz µT","|B| µT","状态 / Status","SM触发Δt / Trigger Δt"]
        for c,h in enumerate(headers):
            ttk.Label(stat,text=h,font=("Segoe UI",9,"bold")).grid(row=0,column=c,padx=5,pady=3,sticky="w")
        self.sensor_vars=[]
        for i in range(5):
            d={k:tk.StringVar(value="--") for k in ["addr","x","y","z","m","st","dt"]}
            self.sensor_vars.append(d)
            ttk.Label(stat,text=f"S{i+1}").grid(row=i+1,column=0,padx=5,sticky="w")
            for c,k in enumerate(["addr","x","y","z","m","st","dt"],start=1):
                ttk.Label(stat,textvariable=d[k],width=15).grid(row=i+1,column=c,padx=4,sticky="w")

        plot_head=ttk.Frame(tab)
        plot_head.pack(fill="x",padx=10,pady=(0,2))
        self.mag_summary_var=tk.StringVar(value="未采样")
        ttk.Label(plot_head,textvariable=self.mag_summary_var).pack(side="left")
        self.mag_view_var=tk.StringVar(value="|B| / Magnitude")
        ttk.Label(plot_head,text="曲线 / Curves:").pack(side="right",padx=(8,2))
        cb=ttk.Combobox(plot_head,textvariable=self.mag_view_var,values=list(MAG_VIEW_LABELS.keys()),state="readonly",width=20)
        cb.pack(side="right")
        self.mag_plot=MultiMagPlot(tab,height=420,view_mode="m")
        cb.bind("<<ComboboxSelected>>",lambda _e:self.mag_plot.set_mode(self.mag_view_var.get()))
        self.mag_plot.pack(fill="both",expand=True,padx=8,pady=8)

    def _build_live_prediction_tab(self):
        """Case-preserving H1/H2-R15/V16 display. It never moves A5S."""
        outer = self.tab_predict
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)
        canvas = tk.Canvas(outer, highlightthickness=0)
        canvas.grid(row=1, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        scroll.grid(row=1, column=1, sticky="ns")
        canvas.configure(yscrollcommand=scroll.set)
        tab = ttk.Frame(canvas)
        window = canvas.create_window((0,0), window=tab, anchor="nw")
        tab.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window, width=e.width,
                    height=max(e.height, tab.winfo_reqheight())))
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(2, weight=10, minsize=300)
        tab.rowconfigure(3, weight=2, minsize=115)
        live_cfg = self.settings.get("live_prediction", {})

        notice = ttk.LabelFrame(tab, text="神经网络设置与公共 B0 控制 / Neural settings and common baseline controls")
        notice.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        notice.columnconfigure(1, weight=1)
        ttk.Label(
            notice,
            text=("输入仅为实时五传感器磁场；先载入所选算法并捕获无载 B0。神经网络以 40 Hz 构造历史；拟合按原始采集帧处理。"
                  "上方原始字段严格保留大小写：H1 保留小写输出，H2/R56 绝对坐标使用 X/Y/Z、"
                  "相对坐标使用 dX/dY/dZ；旧 H2 仅有相对输出时仍可兼容。"
                  "只有下方曲线和 3D 视图经过显式适配。本页绝不向打印机发送运动指令。"),
            foreground="#0d47a1", wraplength=1250, justify="left",
        ).grid(row=0, column=0, columnspan=4, sticky="w", padx=7, pady=(6, 8))

        self.live_predict_model_var = tk.StringVar(value=str(live_cfg.get("checkpoint_path", "")))
        self.live_predict_model_family_var = tk.StringVar(value=str(live_cfg.get("model_family", "auto")))
        self.live_predict_output_mapping_var = tk.StringVar(
            value=str(live_cfg.get("output_mapping_json", default_output_mapping_json()))
        )
        self.live_predict_state_var = tk.StringVar(value="未载入模型 / No checkpoint loaded")
        self.live_predict_baseline_var = tk.StringVar(value="B0：未采集 / not captured")
        self.live_predict_output_var = tk.StringVar(value="等待模型与静止基线 / Waiting for model and B0")
        self.live_predict_profile_confirm_var = tk.BooleanVar(value=bool(live_cfg.get("profile_confirmed", False)))
        self.live_predict_invert_x_var = tk.BooleanVar(value=bool(live_cfg.get("display_invert_x", False)))
        self.live_predict_invert_y_var = tk.BooleanVar(value=bool(live_cfg.get("display_invert_y", False)))
        self.live_predict_edge_guard_var = tk.StringVar(value=str(
            live_cfg.get("edge_guard_ms",live_cfg.get("onset_guard_ms",120.0))))
        self.live_filter_mode_var = tk.StringVar(value=str(
            live_cfg.get("trajectory_filter_mode", "关闭 / Off")))
        self.live_filter_ema_alpha_var = tk.StringVar(value=str(
            live_cfg.get("trajectory_filter_ema_alpha", 0.25)))
        self.live_filter_q_var = tk.StringVar(value=str(
            live_cfg.get("trajectory_filter_q", 1.0)))
        self.live_filter_r_xy_var = tk.StringVar(value=str(
            live_cfg.get("trajectory_filter_r_xy", 0.16)))
        self.live_filter_r_z_var = tk.StringVar(value=str(
            live_cfg.get("trajectory_filter_r_z", 0.01)))
        self.live_trajectory_history_var = tk.StringVar(value=str(
            live_cfg.get("trajectory_history_points", 4000)))
        self.live_filter_status_var = tk.StringVar(value="显示滤波：关闭；原始预测与CSV不变。")
        saved_alignment=live_cfg.get("xy_alignment_by_preset",{})
        self.live_alignment_by_preset=(json.loads(json.dumps(saved_alignment))
                                       if isinstance(saved_alignment,dict) else {})
        self.live_alignment_ref_var=tk.StringVar(value="中心 (0,0)")
        self.live_alignment_basis_var=tk.StringVar(value=str(
            live_cfg.get("alignment_basis","源数据ΔB指纹（推荐）")))
        self.live_alignment_mode_var=tk.StringVar(value=str(
            live_cfg.get("alignment_mode","刚体旋转+平移（推荐）")))
        self.live_alignment_auto_depth_var=tk.StringVar(
            value=str(live_cfg.get("alignment_auto_depth_mm",0.5)))
        self.live_alignment_z_threshold_var=tk.StringVar(
            value=str(live_cfg.get("alignment_z_threshold_uT",15.0)))
        self.live_alignment_z_step_var=tk.StringVar(
            value=str(live_cfg.get("alignment_z_step_mm",0.02)))
        self.live_alignment_z_max_search_var=tk.StringVar(
            value=str(live_cfg.get("alignment_z_max_search_mm",1.0)))
        self.live_alignment_check_x_var=tk.StringVar(value="6.0")
        self.live_alignment_check_y_var=tk.StringVar(value="6.0")
        self.live_alignment_status_var=tk.StringVar(
            value="重新装配校准：优先直接匹配训练源数据ΔB；预测XY对齐仅作二级对照。")
        ttk.Label(notice, text="模型选项 / checkpoint (.pt)").grid(row=1, column=0, sticky="e", padx=5, pady=4)
        self.live_predict_model_combo = ttk.Combobox(
            notice, textvariable=self.live_predict_model_var,
            values=list(LIVE_NEURAL_MODEL_OPTIONS), state="normal", width=54,
        )
        self.live_predict_model_combo.grid(row=1, column=1, sticky="ew", padx=4, pady=4)
        self.live_predict_model_combo.bind(
            "<<ComboboxSelected>>", self._live_predict_model_option_selected
        )
        ttk.Button(notice, text="选择 / Browse", command=self._live_predict_choose_model).grid(row=1, column=2, padx=3)
        self.live_predict_load_btn = ttk.Button(notice, text="载入模型 / Load", command=self._live_predict_load_model)
        self.live_predict_load_btn.grid(row=1, column=3, padx=3)
        ttk.Label(notice, text="模型族 / Family").grid(row=2, column=0, sticky="e", padx=5, pady=3)
        ttk.Combobox(
            notice, textvariable=self.live_predict_model_family_var,
            values=["auto", "H1", "H2-R15", "H2-Spatial-R16", "H2-Unified-R18", "FinalTip-R56"], state="readonly", width=18,
        ).grid(row=2, column=1, sticky="w", padx=4, pady=3)
        ttk.Checkbutton(
            notice,
            text="确认当前磁层、5颗传感器顺序、MLX配置与训练/标定一致",
            variable=self.live_predict_profile_confirm_var,
        ).grid(row=2, column=2, columnspan=2, sticky="w", padx=6, pady=3)
        ttk.Label(notice, text="输出映射 JSON").grid(row=3, column=0, sticky="e", padx=5, pady=3)
        ttk.Entry(notice, textvariable=self.live_predict_output_mapping_var).grid(
            row=3, column=1, sticky="ew", padx=4, pady=3
        )
        ttk.Button(notice, text="校验 / Validate", command=self._live_predict_validate_mapping).grid(row=3, column=2, padx=3)
        ttk.Button(notice, text="默认 / Reset", command=self._live_predict_reset_mapping).grid(row=3, column=3, padx=3)
        self.live_predict_begin_btn = ttk.Button(notice, text="采集静止 B0 并开始", command=self._live_predict_begin_baseline)
        self.live_predict_begin_btn.grid(
            row=4, column=2, padx=3, pady=3
        )
        ttk.Button(notice, text="暂停预测", command=self._live_predict_pause).grid(row=4, column=3, padx=3, pady=3)
        ttk.Label(notice, textvariable=self.live_predict_state_var, foreground="#1b5e20", wraplength=1250,
                  justify="left").grid(row=4, column=0, columnspan=2, sticky="w", padx=6, pady=(4, 1))
        ttk.Label(notice, textvariable=self.live_predict_baseline_var, foreground="#6a1b9a", wraplength=1250,
                  justify="left").grid(row=5, column=0, columnspan=4, sticky="w", padx=6, pady=(1, 5))
        display_axes=ttk.Frame(notice)
        display_axes.grid(row=6,column=0,columnspan=4,sticky="w",padx=6,pady=(1,3))
        ttk.Label(display_axes,text="观察方向 / Display axes（仅曲线与3D，不修改原始/记录数据）：").pack(side="left")
        ttk.Checkbutton(display_axes,text="X 反向显示",variable=self.live_predict_invert_x_var,
                        command=self._live_predict_display_axes_changed).pack(side="left",padx=5)
        ttk.Checkbutton(display_axes,text="Y 反向显示",variable=self.live_predict_invert_y_var,
                        command=self._live_predict_display_axes_changed).pack(side="left",padx=5)
        ttk.Label(display_axes,text="升/降沿轨迹抑制 ms").pack(side="left",padx=(18,3))
        edge_entry=ttk.Entry(display_axes,textvariable=self.live_predict_edge_guard_var,width=7)
        edge_entry.pack(side="left")
        edge_entry.bind("<FocusOut>",lambda _e:self._live_predict_display_axes_changed())
        edge_entry.bind("<Return>",lambda _e:self._live_predict_display_axes_changed())
        ttk.Label(display_axes,text="（仅显示；0=关闭）",foreground="#6d4c41").pack(side="left",padx=3)
        ttk.Label(display_axes,text="3D轨迹历史点数").pack(side="left",padx=(16,3))
        history_entry=ttk.Entry(display_axes,textvariable=self.live_trajectory_history_var,width=8)
        history_entry.pack(side="left")
        history_entry.bind(
            "<FocusOut>",lambda _e:self._live_trajectory_history_apply(show_error=False))
        history_entry.bind("<Return>",lambda _e:self._live_trajectory_history_apply())
        ttk.Button(display_axes,text="应用",command=self._live_trajectory_history_apply).pack(
            side="left",padx=3)
        trajectory_filter=ttk.LabelFrame(
            notice,text="轨迹显示滤波（热生效，不修改原始预测/CSV） / Display trajectory filter")
        trajectory_filter.grid(row=7,column=0,columnspan=4,sticky="ew",padx=6,pady=(2,3))
        ttk.Label(trajectory_filter,text="方法").pack(side="left",padx=(5,2))
        filter_combo=ttk.Combobox(
            trajectory_filter,textvariable=self.live_filter_mode_var,state="readonly",width=22,
            values=["关闭 / Off","指数平滑 / EMA","卡尔曼匀速 / Kalman CV"])
        filter_combo.pack(side="left")
        filter_combo.bind("<<ComboboxSelected>>",lambda _e:self._live_filter_hot_apply())
        ttk.Label(trajectory_filter,text="EMA α").pack(side="left",padx=(10,2))
        ema_entry=ttk.Entry(trajectory_filter,textvariable=self.live_filter_ema_alpha_var,width=6)
        ema_entry.pack(side="left")
        ttk.Label(trajectory_filter,text="Kalman Q").pack(side="left",padx=(10,2))
        q_entry=ttk.Entry(trajectory_filter,textvariable=self.live_filter_q_var,width=7)
        q_entry.pack(side="left")
        ttk.Label(trajectory_filter,text="Rxy").pack(side="left",padx=(8,2))
        rxy_entry=ttk.Entry(trajectory_filter,textvariable=self.live_filter_r_xy_var,width=7)
        rxy_entry.pack(side="left")
        ttk.Label(trajectory_filter,text="Rz").pack(side="left",padx=(8,2))
        rz_entry=ttk.Entry(trajectory_filter,textvariable=self.live_filter_r_z_var,width=7)
        rz_entry.pack(side="left")
        for entry in (ema_entry,q_entry,rxy_entry,rz_entry):
            entry.bind("<FocusOut>",lambda _e:self._live_filter_hot_apply(show_error=False))
            entry.bind("<Return>",lambda _e:self._live_filter_hot_apply())
        ttk.Button(trajectory_filter,text="应用/热生效",command=self._live_filter_hot_apply).pack(side="left",padx=6)
        ttk.Button(trajectory_filter,text="重置滤波状态",command=self._live_filter_reset_button).pack(side="left",padx=2)
        ttk.Label(trajectory_filter,textvariable=self.live_filter_status_var,
                  foreground="#1565c0").pack(side="left",padx=8)

        alignment=ttk.LabelFrame(
            notice,text="重新装配快速校准 / Source ΔB fingerprint + output alignment")
        alignment.grid(row=8,column=0,columnspan=4,sticky="ew",padx=6,pady=(2,3))
        ttk.Label(alignment,text="校准依据").pack(side="left",padx=(5,2))
        basis_combo=ttk.Combobox(
            alignment,textvariable=self.live_alignment_basis_var,state="readonly",width=27,
            values=("源数据ΔB指纹（推荐）","模型预测XY（二级对照）"))
        basis_combo.pack(side="left")
        basis_combo.bind("<<ComboboxSelected>>",lambda _e:self._live_alignment_basis_changed())
        ttk.Label(alignment,text="输出对齐").pack(side="left",padx=(5,2))
        ttk.Combobox(alignment,textvariable=self.live_alignment_mode_var,state="readonly",width=22,
                     values=("刚体旋转+平移（推荐）","仅中心平移（对照）")).pack(side="left")
        ttk.Label(alignment,text="校准点").pack(side="left",padx=(5,2))
        self.live_alignment_ref_combo=ttk.Combobox(
            alignment,textvariable=self.live_alignment_ref_var,state="readonly",width=16,
            values=list(self._live_alignment_default_reference_map()))
        self.live_alignment_ref_combo.pack(side="left")
        ttk.Button(alignment,text="记录当前稳定样本",command=self._live_alignment_record_reference).pack(side="left",padx=3)
        ttk.Button(alignment,text="计算/应用五点校准",command=self._live_alignment_compute).pack(side="left",padx=3)
        ttk.Label(alignment,text="自动压深 mm").pack(side="left",padx=(10,2))
        ttk.Entry(alignment,textvariable=self.live_alignment_auto_depth_var,width=5).pack(side="left")
        self.live_alignment_auto_btn=ttk.Button(
            alignment,text="一键中心Z＋五点校准",command=self._live_alignment_auto_start)
        self.live_alignment_auto_btn.pack(side="left",padx=4)
        ttk.Label(alignment,text="检查真值XY").pack(side="left",padx=(12,2))
        ttk.Entry(alignment,textvariable=self.live_alignment_check_x_var,width=6).pack(side="left")
        ttk.Entry(alignment,textvariable=self.live_alignment_check_y_var,width=6).pack(side="left",padx=(2,0))
        ttk.Button(alignment,text="记录检查点",command=self._live_alignment_record_check).pack(side="left",padx=3)
        ttk.Button(alignment,text="清零本模型校准",command=self._live_alignment_reset).pack(side="left",padx=3)
        z_contact=ttk.Frame(notice)
        z_contact.grid(row=9,column=0,columnspan=4,sticky="w",padx=9,pady=(0,3))
        ttk.Label(z_contact,text="中心Z原始ΔB探测：最低阈值 µT").pack(side="left")
        ttk.Entry(z_contact,textvariable=self.live_alignment_z_threshold_var,width=7).pack(side="left",padx=(2,8))
        ttk.Label(z_contact,text="Z步长 mm").pack(side="left")
        ttk.Entry(z_contact,textvariable=self.live_alignment_z_step_var,width=6).pack(side="left",padx=(2,8))
        ttk.Label(z_contact,text="向下最大搜索 mm").pack(side="left")
        ttk.Entry(z_contact,textvariable=self.live_alignment_z_max_search_var,width=6).pack(side="left",padx=(2,8))
        ttk.Label(z_contact,text="连续4帧确认；四方向Z差超参考容差只提醒；仍按中心Contact Z运动；后续需做曲面补偿",
                  foreground="#6d4c41").pack(side="left")
        ttk.Label(notice,textvariable=self.live_alignment_status_var,foreground="#6d4c41",
                  wraplength=1250,justify="left").grid(row=10,column=0,columnspan=4,sticky="w",padx=9,pady=(0,3))
        self.live_predict_record_state_var = tk.StringVar(
            value="预测记录：自动（任何 Trial 启动时若实时预测已启用，将自动加入 live_predictions.csv）"
        )
        ttk.Label(notice, textvariable=self.live_predict_record_state_var, foreground="#ad1457",
                  wraplength=1250, justify="left").grid(row=11, column=0, columnspan=4,
                                                         sticky="w", padx=6, pady=(2, 6))
        ttk.Label(
            notice,
            text=("R50 两项是静态连续 XYZ 回归：had-239 工作区 0.4–3.0 mm；"
                  "had-Yelow 工作区 0.8–3.0 mm。新增 had-239 R61 为静态底座＋因果动态残差，"
                  "只适用于同一 had-239 最终触头及本次 Core 接触面状态；其 Z 原点修正已包含在模型中，勿重复补偿。"
                  "R61 目前仅有整 Trial 留出回放，尚无新实机盲测。R67 的旧1 mm触头R65实时选项"
                  "因空载误报与浅压深漏检已在R68撤下；请勿继续用R67做实时试验。"
                  "切换其他模型后须重采无载 B0，"
                  "分别检查中心 Contact Z 与推理后五点校准；超出训练压深的轨迹能力以新实测为准。"),
            foreground="#8b4513", wraplength=1250, justify="left",
        ).grid(row=12, column=0, columnspan=4, sticky="w", padx=6, pady=(0, 6))

        values = ttk.LabelFrame(tab, text="原始模型输出（大小写保真）与兼容适配 / Raw outputs and visual adapter")
        values.grid(row=1, column=0, sticky="ew", padx=8, pady=4)
        values.columnconfigure(0, weight=0, minsize=570)
        values.columnconfigure(1, weight=1)
        compact_values=ttk.Frame(values)
        compact_values.grid(row=0,column=0,sticky="nw",padx=(4,6),pady=3)
        for column in range(3):
            compact_values.columnconfigure(column, weight=1)
        self.live_predict_raw_vars = {
            name: tk.StringVar(value=f"{name}: --") for name in RAW_FIELD_ORDER
        }
        self.live_predict_motion_var = tk.StringVar(value="状态 / State: --")
        for index, name in enumerate(RAW_FIELD_ORDER):
            row, column = divmod(index, 3)
            variable = self.live_predict_raw_vars[name]
            ttk.Label(compact_values, textvariable=variable, font=("Segoe UI", 10, "bold")).grid(
                row=row, column=column, sticky="w", padx=7, pady=2
            )
        raw_rows = (len(RAW_FIELD_ORDER) + 2) // 3
        ttk.Label(compact_values, textvariable=self.live_predict_motion_var,
                  font=("Segoe UI", 10, "bold"),wraplength=560,justify="left").grid(
            row=raw_rows, column=0, columnspan=3, sticky="w", padx=7, pady=(4,2)
        )
        ttk.Label(compact_values, textvariable=self.live_predict_output_var, foreground="#37474f",
                  wraplength=560,justify="left").grid(
            row=raw_rows+1,column=0,columnspan=3,sticky="w",padx=7,pady=(2,4))
        raw_frame=ttk.LabelFrame(values,text="额外输出 / Extra outputs")
        raw_frame.grid(row=0,column=1,sticky="nsew",padx=(2,5),pady=3)
        raw_frame.columnconfigure(0,weight=1);raw_frame.rowconfigure(0,weight=1)
        self.live_predict_extra_text = ScrolledText(raw_frame, height=9, wrap="word", font=("Consolas", 8))
        self.live_predict_extra_text.grid(row=0,column=0,sticky="nsew",padx=4,pady=4)
        self.live_predict_extra_text.insert("1.0", "--")
        self.live_predict_extra_text.configure(state="disabled")

        view_box = ttk.LabelFrame(tab, text="主视图：仅显示模型三维推理 / Main view: model inference only")
        view_box.grid(row=2, column=0, sticky="nsew", padx=8, pady=(4, 3))
        view_box.columnconfigure(0, weight=1)
        view_box.rowconfigure(0, weight=1)
        self.live_predict_3d = LiveDxyz3DView(
            view_box, maxlen=self.prediction_trajectory_history_points())
        self.live_predict_3d.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.live_predict_3d.set_view(
            live_cfg.get("view_yaw_deg", -45.0), live_cfg.get("view_pitch_deg", 28.0),
            live_cfg.get("view_zoom", 1.0), live_cfg.get("view_pan_x", 0.0), live_cfg.get("view_pan_y", 0.0),
        )
        self.live_predict_3d.on_view_changed = self._schedule_settings_autosave

        plot_box = ttk.LabelFrame(tab, text="小图：兼容三维预测历史 / Compact adapted XYZ history")
        plot_box.grid(row=3, column=0, sticky="nsew", padx=8, pady=(3, 8))
        plot_box.columnconfigure(0, weight=1)
        plot_box.rowconfigure(0, weight=1)
        self.live_predict_plot = LiveDxyzPlot(plot_box, height=110)
        self.live_predict_plot.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.live_algorithm_panel = LiveAlgorithmPanel(
            self, outer, notice, PROJECT_ROOT / "config/replay_fit_presets_r50.json")
        self.live_algorithm_panel.grid(row=0,column=0,columnspan=2,sticky="ew",padx=8,pady=4)
        self._live_filter_hot_apply(show_error=False)

    @staticmethod
    def _live_alignment_default_reference_map():
        return {
            "中心 (0,0)":(0.0,0.0), "左 (-8,0)":(-8.0,0.0),
            "右 (+8,0)":(8.0,0.0), "下 (0,-8)":(0.0,-8.0),
            "上 (0,+8)":(0.0,8.0),
        }

    @staticmethod
    def _live_alignment_point_label(direction,x,y):
        def f(v):
            return "0" if abs(float(v))<1e-9 else f"{float(v):+g}"
        return f"{direction} ({f(x)},{f(y)})"

    @classmethod
    def _live_alignment_model_reference_map(cls,predictor):
        """Choose center + four axis references from the selected model's own samples."""
        if not isinstance(predictor,LiveNormalizedFitPredictor):
            return cls._live_alignment_default_reference_map()
        try:
            points={(round(float(row[0]),6),round(float(row[1]),6))
                    for row in predictor.engine.xy.tolist()}
        except Exception:
            return cls._live_alignment_default_reference_map()
        if len(points)<5:
            raise RuntimeError("所选拟合模型的训练坐标不足5点，不能执行五点自动校准。")
        center=min(points,key=lambda p:math.hypot(p[0],p[1]))
        scale=max(1.0,max(max(abs(x),abs(y)) for x,y in points))
        axis_tol=max(1e-6,0.02*scale)
        horizontal=[p for p in points if abs(p[1]-center[1])<=axis_tol]
        vertical=[p for p in points if abs(p[0]-center[0])<=axis_tol]
        left=[p for p in horizontal if p[0]<center[0]-axis_tol]
        right=[p for p in horizontal if p[0]>center[0]+axis_tol]
        down=[p for p in vertical if p[1]<center[1]-axis_tol]
        up=[p for p in vertical if p[1]>center[1]+axis_tol]
        if not all((left,right,down,up)):
            raise RuntimeError("所选模型训练坐标缺少中心轴四方向样本，不能安全自动选择五点。")
        def inner_target(rows,axis,sign):
            extent=max(abs((p[axis]-center[axis])) for p in rows)
            target=center[axis]+sign*0.8*extent
            return min(rows,key=lambda p:abs(p[axis]-target))
        chosen=[("中心",center),
                ("左",inner_target(left,0,-1)),("右",inner_target(right,0,1)),
                ("下",inner_target(down,1,-1)),("上",inner_target(up,1,1))]
        return {cls._live_alignment_point_label(name,*p):(float(p[0]),float(p[1]))
                for name,p in chosen}

    @classmethod
    def _live_source_fingerprint_reference_map(cls,predictor):
        """Use interior five points so a shifted installation stays inside the source map."""
        if not isinstance(predictor,LiveNormalizedFitPredictor):
            return cls._live_alignment_default_reference_map()
        try:
            points={(round(float(row[0]),6),round(float(row[1]),6))
                    for row in predictor.engine.xy.tolist()}
            center=min(points,key=lambda p:math.hypot(p[0],p[1]))
            min_x=min(p[0] for p in points);max_x=max(p[0] for p in points)
            min_y=min(p[1] for p in points);max_y=max(p[1] for p in points)
            arm_x=0.5*min(center[0]-min_x,max_x-center[0])
            arm_y=0.5*min(center[1]-min_y,max_y-center[1])
            if arm_x<0.75 or arm_y<0.75:
                raise RuntimeError("源数据空间跨度不足，不能布置中心＋四方向内点。")
            # Half-millimetre coordinates are practical for the printer while
            # retaining margin for the unknown installation translation.
            arm_x=max(0.5,round(arm_x*2.0)/2.0)
            arm_y=max(0.5,round(arm_y*2.0)/2.0)
            chosen=[("中心",center),("左",(center[0]-arm_x,center[1])),
                    ("右",(center[0]+arm_x,center[1])),
                    ("下",(center[0],center[1]-arm_y)),
                    ("上",(center[0],center[1]+arm_y))]
            return {cls._live_alignment_point_label(name,*point):
                    (float(point[0]),float(point[1])) for name,point in chosen}
        except Exception:
            return cls._live_alignment_model_reference_map(predictor)

    def _live_alignment_reference_map(self,predictor=None):
        if predictor is None:
            with self.live_predict_lock:
                predictor=self.live_predictor
        if self._live_alignment_basis_key(self.live_alignment_basis_var.get())=="source_delta_b":
            return self._live_source_fingerprint_reference_map(predictor)
        return self._live_alignment_model_reference_map(predictor)

    def _live_alignment_refresh_reference_ui(self,predictor=None):
        refs=self._live_alignment_reference_map(predictor)
        values=list(refs)
        self.live_alignment_ref_combo.configure(values=values)
        if self.live_alignment_ref_var.get() not in refs:
            self.live_alignment_ref_var.set(values[0])
        return refs

    def _live_alignment_basis_changed(self):
        self.live_alignment_recent.clear();self.live_source_fingerprint_recent.clear()
        try:self._live_alignment_refresh_reference_ui()
        except Exception as exc:self.live_alignment_status_var.set(f"校准依据切换失败：{exc}")
        else:self._live_alignment_update_status()
        self._schedule_settings_autosave()

    def _live_alignment_preset(self):
        panel=getattr(self,"live_algorithm_panel",None)
        return str(panel.vars['preset'].get()) if panel is not None else ""

    def _live_alignment_state(self,preset=None,create=True) -> Optional[dict[str,Any]]:
        key=str(preset or self._live_alignment_preset())
        state=self.live_alignment_by_preset.get(key)
        if not isinstance(state,dict) and create:
            state={"offset_mm":[0.0,0.0],"calibration_points":{},"check_points":[]}
            self.live_alignment_by_preset[key]=state
        return state

    @staticmethod
    def _live_alignment_mode_key(value):
        return "translation" if "仅中心" in str(value or "") else "rigid"

    @staticmethod
    def _live_alignment_basis_key(value):
        return "prediction_xy" if "预测XY" in str(value or "") else "source_delta_b"

    @staticmethod
    def _live_alignment_transform_xy(state,x,y):
        offset=state.get("offset_mm",[0.0,0.0])
        angle=math.radians(float(state.get("rotation_deg",0.0) or 0.0))
        c,s=math.cos(angle),math.sin(angle)
        return (c*float(x)-s*float(y)+float(offset[0]),
                s*float(x)+c*float(y)+float(offset[1]))

    def _live_alignment_required_state(self,preset=None) -> dict[str,Any]:
        state=self._live_alignment_state(preset,create=True)
        if not isinstance(state,dict):
            raise RuntimeError("无法建立当前模型的五点校准状态。")
        return state

    def _live_alignment_recent_xy(self):
        preset=self._live_alignment_preset()
        rows=[row for row in self.live_alignment_recent if row[1]==preset]
        if not rows:
            raise RuntimeError("当前接触尚无可用的未平移拟合输出；请稳定接触后再记录。")
        segment=rows[-1][2];end_t=rows[-1][0]
        rows=[row for row in rows if row[2]==segment and end_t-row[0]<=500_000_000][-12:]
        if len(rows)<3:
            raise RuntimeError("稳定输出不足3帧；请保持接触约0.2秒后再记录。")
        return (float(statistics.median(row[3] for row in rows)),
                float(statistics.median(row[4] for row in rows)),len(rows))

    def _live_source_fingerprint_recent_delta(self,preset=None,since_ns=None,
                                               window_ns=320_000_000,min_frames=6):
        """Reduce a stable hold window without consulting predicted XY/contact state."""
        key=str(preset or self._live_alignment_preset())
        rows=[row for row in list(self.live_source_fingerprint_recent)
              if row[1]==key and (since_ns is None or int(row[0])>=int(since_ns))]
        if not rows:
            raise RuntimeError("当前点尚无B0差分原始磁场；请建立B0并稳定按压后再记录。")
        end_t=int(rows[-1][0])
        rows=[row for row in rows if end_t-int(row[0])<=int(window_ns)][-18:]
        if len(rows)<int(min_frames):
            raise RuntimeError(f"稳定原始磁场不足{min_frames}帧；请保持按压约0.3秒。")
        columns=list(zip(*(row[2] for row in rows)))
        delta=[float(statistics.median(values)) for values in columns]
        amplitude=math.sqrt(sum(value*value for value in delta))
        residual=[]
        for row in rows:
            residual.append(math.sqrt(sum((float(v)-m)**2 for v,m in zip(row[2],delta))))
        ordered=sorted(residual)
        p95=ordered[min(len(ordered)-1,max(0,math.ceil(0.95*len(ordered))-1))]
        limit=max(5.0,0.08*amplitude)
        if amplitude<=5.0:
            raise RuntimeError(f"当前ΔB幅值仅 {amplitude:.2f} µT，不能确认已稳定接触。")
        if p95>limit:
            raise RuntimeError(
                f"当前窗口仍在变化：ΔB稳定残差P95={p95:.2f} µT > {limit:.2f} µT；"
                "请避开升降沿并继续保持。")
        return delta,len(rows),p95,amplitude

    def _live_alignment_apply_to_predictor(self,predictor,preset=None,update_status=True):
        if not isinstance(predictor,LiveNormalizedFitPredictor):return
        key=str(preset or self._live_alignment_preset())
        state=self._live_alignment_state(key,create=False) or {}
        records=state.get("calibration_points") or {}
        migrated=False
        supported=("center_locked_translation_v1","rigid_rotation_translation_v1",
                   "source_delta_b_fingerprint_rigid_v1")
        if len(records)>=5 and state.get("calibration_strategy") not in supported:
            expected={name:tuple(row.get("reference_xy_mm",(0.0,0.0)))
                      for name,row in records.items()}
            try:
                mode=state.get("calibration_mode","rigid")
                self._live_alignment_compute_state(state,expected,mode);migrated=True
            except Exception:pass
        offset=state.get("offset_mm",[0.0,0.0])
        try:x,y=float(offset[0]),float(offset[1])
        except Exception:x=y=0.0
        angle=float(state.get("rotation_deg",0.0) or 0.0)
        if (state.get("calibration_strategy")=="source_delta_b_fingerprint_rigid_v1" and
                state.get("fingerprint_quality")=="fail"):
            angle=x=y=0.0
        source=("source_delta_b_fingerprint" if
                state.get("calibration_strategy")=="source_delta_b_fingerprint_rigid_v1"
                else ("five_point_prediction_xy" if records else "none"))
        predictor.set_xy_rigid_transform(angle,x,y,source,{
            "preset":key,"calibration_points":len(state.get("calibration_points") or {}),
            "calibration_strategy":state.get("calibration_strategy"),
            "calibration_rms_mm":state.get("calibration_rms_mm"),
            "check_rms_mm":state.get("check_rms_mm"),
            "fingerprint_quality":state.get("fingerprint_quality"),
            "fingerprint_median_cosine":state.get("fingerprint_median_cosine"),
            "fingerprint_minimum_cosine":state.get("fingerprint_minimum_cosine"),
            "fingerprint_amplitude_scale":state.get("fingerprint_amplitude_scale")})
        if update_status:
            self._live_alignment_refresh_reference_ui(predictor)
            self._live_alignment_update_status(key)
        if migrated:self._schedule_settings_autosave()

    def _live_alignment_update_status(self,preset=None):
        key=str(preset or self._live_alignment_preset())
        state=self._live_alignment_state(key,create=False) or {}
        records=state.get("calibration_points") or {}
        supported=("center_locked_translation_v1","rigid_rotation_translation_v1",
                   "source_delta_b_fingerprint_rigid_v1")
        if len(records)>=5 and state.get("calibration_strategy") not in supported:
            expected={name:tuple(row.get("reference_xy_mm",(0.0,0.0)))
                      for name,row in records.items()}
            try:
                self._live_alignment_compute_state(state,expected,state.get("calibration_mode","rigid"))
                self._schedule_settings_autosave()
            except Exception:pass
        offset=state.get("offset_mm",[0.0,0.0]);n=len(state.get("calibration_points") or {})
        if state.get("calibration_strategy")=="source_delta_b_fingerprint_rigid_v1":
            strategy="源数据15通道ΔB指纹匹配"
        elif state.get("calibration_strategy")=="rigid_rotation_translation_v1":
            strategy="五点刚体旋转＋平移"
        elif state.get("calibration_strategy")=="center_locked_translation_v1":
            strategy="仅中心平移；四方向检验"
        else:
            strategy="尚未计算"
        text=(f"{key}｜五点 {n}/5｜XY平移=({float(offset[0]):+.3f}, {float(offset[1]):+.3f}) mm"
              f"｜旋转={float(state.get('rotation_deg',0.0) or 0.0):+.3f}°｜{strategy}")
        if state.get("calibration_rms_mm") is not None:
            text+=f"｜校准残差RMS={float(state['calibration_rms_mm']):.3f} mm"
        if state.get("check_rms_mm") is not None:
            text+=f"｜独立检查RMS={float(state['check_rms_mm']):.3f} mm，最大={float(state.get('check_max_mm',0)):.3f} mm"
        if state.get("fingerprint_quality"):
            text+=(f"｜源指纹={state['fingerprint_quality']}"
                   f"，cos中位/最小={float(state.get('fingerprint_median_cosine',float('nan'))):.3f}/"
                   f"{float(state.get('fingerprint_minimum_cosine',float('nan'))):.3f}"
                   f"，幅值比={float(state.get('fingerprint_amplitude_scale',float('nan'))):.3f}")
            if state.get("fingerprint_quality")=="fail":text+="（未应用）"
        if state.get("alignment_warning"):
            text+="｜警告："+str(state['alignment_warning'])
        zcal=state.get("z_contact_calibration")
        if (isinstance(zcal,dict) and zcal.get("committed") and
                zcal.get("center_contact_z_mm") is not None):
            text+=(f"｜ContactZ={float(zcal['center_contact_z_mm']):.3f} mm"
                   f"（原始ΔB阈值={float(zcal.get('threshold_ut',float('nan'))):.1f} µT）")
            if zcal.get("validation_max_abs_mm") is not None:
                text+=f"｜四点Z最大差={float(zcal['validation_max_abs_mm']):.3f} mm"
            if zcal.get("flatness_warning"):
                text+="｜警告：四方向磁场越阈Z不一致；后续仍用中心Z，无曲面补偿"
        self.live_alignment_status_var.set(text)

    def _live_alignment_record_reference(self):
        try:
            with self.live_predict_lock:predictor=self.live_predictor
            if not isinstance(predictor,LiveNormalizedFitPredictor):
                raise RuntimeError("五点校准只适用于归一化拟合算法；请先载入对应磁片拟合模型。")
            expected=self._live_alignment_refresh_reference_ui(predictor)
            label=self.live_alignment_ref_var.get();reference=expected[label]
            preset=self._live_alignment_preset()
            state=self._live_alignment_required_state(preset)
            basis=self._live_alignment_basis_key(self.live_alignment_basis_var.get())
            row={"reference_xy_mm":list(reference),
                 "calibration_basis":basis,
                 "captured_utc":datetime.utcnow().isoformat(timespec="seconds")+"Z"}
            if basis=="source_delta_b":
                delta,n,p95,amplitude=self._live_source_fingerprint_recent_delta(preset)
                row.update(current_delta_uT=delta,median_frames=n,
                           stability_p95_uT=p95,amplitude_uT=amplitude,
                           indent_depth_mm=float(self.live_alignment_auto_depth_var.get()))
            else:
                x,y,n=self._live_alignment_recent_xy()
                row.update(predicted_unaligned_xy_mm=[x,y],median_frames=n)
            state.setdefault("calibration_points",{})[label]=row
            state["calibration_basis"]=basis
            self._live_alignment_update_status(preset);self._schedule_settings_autosave()
        except Exception as exc:messagebox.showwarning("五点快速校准",str(exc))

    @staticmethod
    def _live_source_fingerprint_compute_state(state,expected,predictor,depth_mm):
        records=state.get("calibration_points") or {}
        missing=[name for name in expected if name not in records]
        if missing:raise RuntimeError("还缺少校准点："+"、".join(missing))
        captures=[]
        for name in expected:
            row=records[name]
            if not isinstance(row.get("current_delta_uT"),list):
                raise RuntimeError(f"{name} 不是原始ΔB源指纹样本；请清零后按当前校准依据重新采集。")
            captured_depth=float(row.get("indent_depth_mm",depth_mm))
            if abs(captured_depth-float(depth_mm))>1e-6:
                raise RuntimeError(
                    f"{name} 的采集压深为 {captured_depth:g} mm，与当前 {float(depth_mm):g} mm 不同；"
                    "请清零后统一压深重采。")
            captures.append({
                "label":name,
                "reference_xy_mm":list(map(float,row["reference_xy_mm"])),
                "current_delta_uT":list(map(float,row["current_delta_uT"])),
            })
        result=fit_source_fingerprint_alignment(
            predictor.profile,captures,float(depth_mm),
            translation_limit_mm=6.0,rotation_limit_deg=12.0)
        state.update(
            offset_mm=list(result["offset_mm"]),rotation_deg=float(result["rotation_deg"]),
            calibration_strategy=result["strategy"],calibration_mode="rigid",
            calibration_basis="source_delta_b",calibration_rms_mm=None,
            fingerprint_quality=result["quality"],
            fingerprint_loss=float(result["loss"]),
            fingerprint_median_cosine=float(result["median_cosine"]),
            fingerprint_minimum_cosine=float(result["minimum_cosine"]),
            fingerprint_amplitude_scale=float(result["amplitude_scale"]),
            fingerprint_amplitude_scale_spread=float(
                result["amplitude_scale_median_relative_spread"]),
            fingerprint_vector_relative_residual=float(
                result["median_vector_relative_residual"]),
            fingerprint_source_node_count=int(result["source_node_count"]),
            fingerprint_source_grid_spacing_mm=float(result["source_grid_spacing_mm"]),
            fingerprint_depth_mm=float(result["depth_mm"]),
            fingerprint_depth_interpolated_node_count=int(
                result["depth_interpolated_node_count"]),
            fingerprint_point_matches=result["point_matches"],
            calibration_applied=result["quality"]!="fail",
            check_points=[],check_rms_mm=None,check_max_mm=None,
            alignment_warning="；".join(result.get("warnings") or []))
        return result

    @staticmethod
    def _live_alignment_compute_state(state,expected,mode="rigid"):
        records=state.get("calibration_points") or {}
        missing=[name for name in expected if name not in records]
        if missing:raise RuntimeError("还缺少校准点："+"、".join(missing))
        center_name=min(expected,key=lambda name:math.hypot(
            float(expected[name][0]),float(expected[name][1])))
        names=list(expected)
        predicted=[tuple(map(float,records[name]['predicted_unaligned_xy_mm'])) for name in names]
        references=[tuple(map(float,records[name]['reference_xy_mm'])) for name in names]
        mode="translation" if str(mode)=="translation" else "rigid"
        if mode=="translation":
            center_index=names.index(center_name)
            theta=0.0
            ox=references[center_index][0]-predicted[center_index][0]
            oy=references[center_index][1]-predicted[center_index][1]
            strategy="center_locked_translation_v1"
        else:
            px=sum(v[0] for v in predicted)/len(predicted)
            py=sum(v[1] for v in predicted)/len(predicted)
            qx=sum(v[0] for v in references)/len(references)
            qy=sum(v[1] for v in references)/len(references)
            a=sum((p[0]-px)*(q[0]-qx)+(p[1]-py)*(q[1]-qy)
                  for p,q in zip(predicted,references))
            b=sum((p[0]-px)*(q[1]-qy)-(p[1]-py)*(q[0]-qx)
                  for p,q in zip(predicted,references))
            if abs(a)+abs(b)<1e-9:raise RuntimeError("五点空间跨度不足，不能求旋转。")
            theta=math.atan2(b,a)
            c,s=math.cos(theta),math.sin(theta)
            ox=qx-(c*px-s*py);oy=qy-(s*px+c*py)
            strategy="rigid_rotation_translation_v1"
        c,s=math.cos(theta),math.sin(theta)
        errors=[]
        for pred,ref in zip(predicted,references):
            aligned=(c*pred[0]-s*pred[1]+ox,s*pred[0]+c*pred[1]+oy)
            error=math.hypot(aligned[0]-ref[0],aligned[1]-ref[1])
            errors.append(error)
        rms=math.sqrt(sum(v*v for v in errors)/len(errors))
        pxc=sum(v[0] for v in predicted)/len(predicted);pyc=sum(v[1] for v in predicted)/len(predicted)
        qxc=sum(v[0] for v in references)/len(references);qyc=sum(v[1] for v in references)/len(references)
        p_energy=sum((v[0]-pxc)**2+(v[1]-pyc)**2 for v in predicted)
        q_energy=sum((v[0]-qxc)**2+(v[1]-qyc)**2 for v in references)
        scale_ratio=math.sqrt(q_energy/p_energy) if p_energy>1e-9 else float('nan')
        offsets=[[ref[0]-pred[0],ref[1]-pred[1]] for pred,ref in zip(predicted,references)]
        spread=max(math.hypot(v[0]-ox,v[1]-oy) for v in offsets)
        for key in list(state):
            if key.startswith("fingerprint_"):
                state.pop(key,None)
        state.update(offset_mm=[ox,oy],calibration_rms_mm=rms,calibration_max_offset_spread_mm=spread,
                     rotation_deg=math.degrees(theta),calibration_scale_ratio=scale_ratio,
                     calibration_origin_point=center_name,
                     calibration_strategy=strategy,calibration_mode=mode,
                     calibration_basis="prediction_xy",calibration_applied=True,
                     check_points=[],check_rms_mm=None,check_max_mm=None)
        warnings=[]
        if rms>1.0:warnings.append("五点刚体校准残差较大")
        if math.isfinite(scale_ratio) and abs(scale_ratio-1.0)>0.05:
            warnings.append(f"空间尺度比={scale_ratio:.3f}，疑似高度/倾斜/预压变化；未做仿射缩放")
        state['alignment_warning']="；".join(warnings)
        return ox,oy,math.degrees(theta),rms

    def _live_alignment_compute(self):
        try:
            preset=self._live_alignment_preset();state=self._live_alignment_required_state(preset)
            with self.live_predict_lock:predictor=self.live_predictor
            if not isinstance(predictor,LiveNormalizedFitPredictor):
                raise RuntimeError("请先载入对应磁片的归一化拟合模型。")
            expected=self._live_alignment_refresh_reference_ui(predictor)
            basis=self._live_alignment_basis_key(self.live_alignment_basis_var.get())
            if basis=="source_delta_b":
                result=self._live_source_fingerprint_compute_state(
                    state,expected,predictor,float(self.live_alignment_auto_depth_var.get()))
                if result["quality"]=="fail":
                    self._live_alignment_update_status(preset);self._schedule_settings_autosave()
                    raise RuntimeError("源指纹匹配质量不足，已保存诊断但未应用坐标校准："+
                                       "；".join(result.get("warnings") or []))
            else:
                mode=self._live_alignment_mode_key(self.live_alignment_mode_var.get())
                self._live_alignment_compute_state(state,expected,mode)
            self._live_alignment_apply_to_predictor(predictor,preset)
            self.live_alignment_recent.clear();self.live_source_fingerprint_recent.clear()
            self._schedule_settings_autosave()
        except Exception as exc:messagebox.showwarning("五点快速校准",str(exc))

    def _live_alignment_record_check(self):
        try:
            preset=self._live_alignment_preset();state=self._live_alignment_required_state(preset)
            if len(state.get("calibration_points") or {})<5:
                raise RuntimeError("请先完成五点校准并计算平移，再记录独立检查点。")
            ref=(float(self.live_alignment_check_x_var.get()),float(self.live_alignment_check_y_var.get()))
            if not all(math.isfinite(v) and -10<=v<=10 for v in ref):
                raise RuntimeError("检查点XY须在 -10～+10 mm 范围内。")
            x,y,n=self._live_alignment_recent_xy()
            pred=self._live_alignment_transform_xy(state,x,y)
            error=math.hypot(pred[0]-ref[0],pred[1]-ref[1])
            checks=state.setdefault("check_points",[]);checks.append({
                "reference_xy_mm":list(ref),"predicted_xy_mm":list(pred),"error_mm":error,
                "median_frames":n,"captured_utc":datetime.utcnow().isoformat(timespec="seconds")+"Z"})
            del checks[:-50]
            state['check_rms_mm']=math.sqrt(sum(float(v['error_mm'])**2 for v in checks)/len(checks))
            state['check_max_mm']=max(float(v['error_mm']) for v in checks)
            if state['check_rms_mm']>1.0 or state['check_max_mm']>2.0:
                state['alignment_warning']="独立检查误差仍大，整体平移不能解释装配变化；建议重新装配或完整重标定。"
            self._live_alignment_apply_to_predictor(self.live_predictor,preset)
            self._schedule_settings_autosave()
        except Exception as exc:messagebox.showwarning("独立检查点",str(exc))

    def _live_alignment_reset(self):
        preset=self._live_alignment_preset()
        self.live_alignment_by_preset[preset]={"offset_mm":[0.0,0.0],"rotation_deg":0.0,
            "calibration_basis":self._live_alignment_basis_key(self.live_alignment_basis_var.get()),
            "calibration_points":{},"check_points":[]}
        with self.live_predict_lock:predictor=self.live_predictor
        if isinstance(predictor,LiveNormalizedFitPredictor):predictor.set_xy_rigid_transform(0,0,0,"none",{})
        self.live_alignment_recent.clear();self.live_source_fingerprint_recent.clear()
        self._live_alignment_update_status(preset);self._schedule_settings_autosave()

    def _live_alignment_wait_unaligned_xy(self,preset,since_ns,timeout_s=3.0,min_frames=5):
        deadline=time.monotonic()+float(timeout_s)
        while time.monotonic()<deadline:
            if self.motion_abort_event.is_set():
                raise RuntimeError("五点自动校准已停止。")
            rows=[row for row in list(self.live_alignment_recent)
                  if row[1]==preset and int(row[0])>=int(since_ns)]
            if rows:
                segment=rows[-1][2];end_t=rows[-1][0]
                rows=[row for row in rows if row[2]==segment and end_t-row[0]<=600_000_000][-15:]
                if len(rows)>=min_frames:
                    return (float(statistics.median(row[3] for row in rows)),
                            float(statistics.median(row[4] for row in rows)),len(rows))
            time.sleep(0.04)
        raise TimeoutError("接触后3秒内未获得足够稳定的未平移预测；请检查B0、阈值、磁片选择和压深。")

    def _live_source_fingerprint_wait_delta(self,preset,since_ns,timeout_s=3.0):
        deadline=time.monotonic()+float(timeout_s);last_error=""
        while time.monotonic()<deadline:
            if self.motion_abort_event.is_set():
                raise RuntimeError("五点自动源指纹校准已停止。")
            try:
                return self._live_source_fingerprint_recent_delta(
                    preset,since_ns=since_ns,window_ns=320_000_000,min_frames=6)
            except RuntimeError as exc:
                last_error=str(exc)
            time.sleep(0.04)
        raise TimeoutError("接触后3秒内未获得稳定原始ΔB窗口；"+(last_error or "请检查B0和磁场采集。"))

    def _live_alignment_wait_baseline(self,predictor,timeout_s=5.0):
        deadline=time.monotonic()+float(timeout_s)
        while time.monotonic()<deadline:
            if self.motion_abort_event.is_set():raise RuntimeError("五点自动校准已停止。")
            with self.live_predict_lock:
                if predictor is not self.live_predictor or not self.live_predict_enabled:
                    raise RuntimeError("自动校准期间所选模型被更换或预测被暂停。")
                if predictor.ready:return
            time.sleep(0.04)
        raise TimeoutError("自动校准B0采集超时；请检查磁场采集是否持续运行。")

    def _live_alignment_wait_raw_frames(self,since_ns,min_frames=8,timeout_s=1.5):
        """Wait for new complete 15-channel sensor frames, independent of the model."""
        deadline=time.monotonic()+float(timeout_s)
        rows=[]
        while time.monotonic()<deadline:
            if self.motion_abort_event.is_set():
                raise RuntimeError("中心Z原始磁场探测已停止。")
            rows=[(int(t_ns),tuple(values)) for t_ns,values in list(self.raw_magnetic_recent)
                  if int(t_ns)>=int(since_ns)]
            if len(rows)>=int(min_frames):
                return rows
            time.sleep(0.02)
        raise TimeoutError(
            f"原始磁场窗口超时：仅收到 {len(rows)}/"
            f"{int(min_frames)} 帧；请检查5颗传感器采样。"
        )

    def _live_alignment_capture_raw_contact_baseline(self,minimum_threshold_ut):
        started=time.perf_counter_ns()
        self.printer.dwell(0.40,mode="grid_points",phase="grid_reference_raw_b0")
        rows=self._live_alignment_wait_raw_frames(started,min_frames=12,timeout_s=2.0)
        stats=raw_baseline_statistics([values for _,values in rows],minimum_threshold_ut)
        stats["started_perf_ns"]=int(started)
        stats["completed_perf_ns"]=int(rows[-1][0])
        return stats

    def _live_alignment_probe_raw_contact_z(self,label,baseline,start_z,end_z,step_mm,
                                             z_speed,confirm_frames=4):
        """Step down and return the M114 Z of the first confirmed raw ΔB crossing."""
        start_z=float(start_z);end_z=float(end_z);step=abs(float(step_mm))
        if not (math.isfinite(start_z) and math.isfinite(end_z) and
                math.isfinite(step) and step>0 and start_z>end_z):
            raise ValueError("原始接触Z探测范围或步长无效。")
        count=int(math.ceil((start_z-end_z)/step))+1
        for index in range(count):
            if self.motion_abort_event.is_set():
                raise RuntimeError("中心Z原始磁场探测已停止。")
            z=max(end_z,start_z-index*step)
            self.printer.move_abs(z=z,speed_mm_s=min(max(0.05,float(z_speed)),0.5),
                                  mode="grid_points",phase="grid_reference_z_probe")
            sampled_after=time.perf_counter_ns()
            rows=self._live_alignment_wait_raw_frames(
                sampled_after,min_frames=max(6,int(confirm_frames)),timeout_s=1.3)
            verdict=raw_contact_window(
                [values for _,values in rows],baseline["baseline_ut"],
                baseline["threshold_ut"],confirm_frames)
            self.after(0,lambda l=label,p=index+1,n=count,zv=z,a=verdict["last_amplitude_ut"],
                       th=baseline["threshold_ut"]:self.live_alignment_status_var.set(
                f"原始接触Z探测 {l}｜{p}/{n}｜Z={zv:.3f} mm｜ΔB={a:.1f}/{th:.1f} µT"))
            if verdict["triggered"]:
                if index==0:
                    raise RuntimeError(
                        f"{label} 在Z探测起点 {z:.3f} mm 已连续越阈；起点可能已接触，"
                        "为避免把预压位置误当接触面，本次不写入Contact Z。"
                    )
                reported=self.printer.query_position()
                reported_z=float(reported.get("z",z))
                if not math.isfinite(reported_z):reported_z=z
                result=dict(verdict)
                result.update(
                    label=str(label),commanded_z_mm=float(z),reported_z_mm=reported_z,
                    start_z_mm=start_z,end_z_mm=end_z,step_mm=step,
                    baseline_frames=int(baseline["frames"]),
                    baseline_noise_p95_ut=float(baseline["noise_p95_ut"]),
                    adaptive_threshold_ut=float(baseline["adaptive_threshold_ut"]),
                )
                self.log("AUTO CONTACT-Z RAW THRESHOLD: "+json.dumps(result,ensure_ascii=False))
                return result
        raise RuntimeError(
            f"{label} 从 {start_z:.3f} 降到 {end_z:.3f} mm 仍未出现连续"
            f"{int(confirm_frames)}帧原始ΔB越阈；未修改Contact Z。"
        )

    def _live_alignment_commit_contact_z(self,contact_z,z_meta):
        """Commit validated Contact Z on the Tk thread and persist it with the profile."""
        done=threading.Event();box={}
        def commit():
            profile=None;old_calibration=None;old_contact=None
            try:
                value=float(contact_z)
                old_contact=float(self.contact_z_var.get())
                self.contact_z_var.set(value)
                profile=self.maglayer_profiles.get(self.active_mag_profile_id)
                if isinstance(profile,dict):
                    old_calibration=json.loads(json.dumps(profile.get("calibration",{})))
                    calibration=profile.setdefault("calibration",{})
                    calibration["contact_z_mm"]=value
                    calibration["contact_z_source"]="raw_15ch_threshold_center_plus_four_warning_r52"
                    calibration["contact_z_calibrated_utc"]=datetime.utcnow().isoformat(timespec="seconds")+"Z"
                    calibration["contact_z_validation"]=json.loads(json.dumps(z_meta))
                    self.maglayer_profiles_doc["profiles"]=self.maglayer_profiles
                    save_magnetic_layer_profiles(
                        self.maglayer_profiles_doc,self.maglayer_profiles_path)
                if hasattr(self,"mag_hw_vars") and "contact_z_mm" in self.mag_hw_vars:
                    self.mag_hw_vars["contact_z_mm"].set(value)
                self.settings=self._settings_snapshot()
                save_runtime_settings(self.settings,self.settings_path)
            except Exception as exc:
                try:
                    if old_contact is not None:self.contact_z_var.set(old_contact)
                    if isinstance(profile,dict) and old_calibration is not None:
                        profile["calibration"]=old_calibration
                        self.maglayer_profiles_doc["profiles"]=self.maglayer_profiles
                        save_magnetic_layer_profiles(
                            self.maglayer_profiles_doc,self.maglayer_profiles_path)
                    if (hasattr(self,"mag_hw_vars") and "contact_z_mm" in self.mag_hw_vars
                            and old_contact is not None):
                        self.mag_hw_vars["contact_z_mm"].set(old_contact)
                except Exception as rollback_exc:
                    self.log(f"Contact Z rollback ERROR: {rollback_exc}")
                box["error"]=exc
            finally:
                done.set()
        self.after(0,commit)
        if not done.wait(8.0):
            raise RuntimeError("写入Contact Z超时；未确认配置已保存。")
        if box.get("error") is not None:
            raise RuntimeError(f"写入Contact Z失败：{box['error']}")

    def _live_alignment_auto_finish(self,preset,error=None):
        self.live_alignment_auto_running=False
        self.live_alignment_auto_btn.state(["!disabled"])
        if error:
            self.live_alignment_status_var.set(f"{preset}｜一键自动五点校准失败：{error}")
            self._schedule_settings_autosave()
            messagebox.showerror("一键自动五点校准",str(error))
        else:
            self._live_alignment_update_status(preset)
            self._schedule_settings_autosave()
            state=self._live_alignment_state(preset,create=False) or {}
            if state.get("calibration_strategy")=="source_delta_b_fingerprint_rigid_v1":
                strategy=(f"源数据ΔB指纹匹配（{state.get('fingerprint_quality','?')}）"
                          f"，cos中位={float(state.get('fingerprint_median_cosine',float('nan'))):.3f}")
            elif state.get("calibration_strategy")=="rigid_rotation_translation_v1":
                strategy="预测XY旋转＋平移（二级）"
            else:strategy="预测XY仅中心平移（二级）"
            zcal=state.get("z_contact_calibration") or {}
            z_text=(f"Contact Z={float(zcal['center_contact_z_mm']):.3f} mm，"
                    f"四点最大差={float(zcal.get('validation_max_abs_mm',0.0)):.3f} mm；"
                    if zcal.get("center_contact_z_mm") is not None else "")
            if zcal.get("flatness_warning"):
                messagebox.showwarning(
                    "一键自动五点校准：接触Z差异提醒",
                    f"五点校准已完成：{z_text}{strategy}。\n\n"
                    f"四方向磁场越阈Z差超出参考 ±{float(zcal.get('validation_tolerance_mm',0.15)):.3f} mm，"
                    "但本版只提醒，不因此中断。后续实验继续以中心 Contact Z 计算压深，"
                    "未做接触面曲面补偿；不同XY处实际压深可能不一致。"
                    "后续需采集并验证接触面图，再实现轨迹Z补偿。"
                )
            else:
                messagebox.showinfo("一键自动五点校准",
                                    f"中心原始磁场Z标定＋四方向验证已完成：{z_text}{strategy}。")

    def _live_alignment_auto_start(self):
        """User-authorized printer routine; never called by startup or analysis code."""
        if self.live_alignment_auto_running:
            messagebox.showwarning("一键自动五点校准","自动校准已在运行。")
            return
        if self.full_plan_running or self.recorder is not None:
            messagebox.showwarning("一键自动五点校准","请先结束 Full Plan 和当前 Trial。")
            return
        with self.live_predict_lock:predictor=self.live_predictor
        if not isinstance(predictor,LiveNormalizedFitPredictor):
            messagebox.showwarning("一键自动五点校准","请先在“归一化拟合算法”中载入要校准的磁片/模型。")
            return
        if not self.live_predict_profile_confirm_var.get():
            messagebox.showwarning("一键自动五点校准","请先确认当前磁片、五颗传感器顺序和MLX配置与所选标定模型一致。")
            return
        try:
            preset=self._live_alignment_preset()
            refs=self._live_alignment_reference_map(predictor)
            depth=float(self.live_alignment_auto_depth_var.get())
            minimum_z_threshold=float(self.live_alignment_z_threshold_var.get())
            z_probe_step=float(self.live_alignment_z_step_var.get())
            z_max_search=float(self.live_alignment_z_max_search_var.get())
            calibration_basis=self._live_alignment_basis_key(self.live_alignment_basis_var.get())
            alignment_mode=self._live_alignment_mode_key(self.live_alignment_mode_var.get())
            if not math.isfinite(depth) or not 0.05<=depth<=AUTO_MAX_INDENT_MM:
                raise ValueError(f"自动校准压深必须为 0.05～{AUTO_MAX_INDENT_MM:g} mm。")
            if not math.isfinite(minimum_z_threshold) or not 5.0<=minimum_z_threshold<=5000.0:
                raise ValueError("中心Z原始ΔB最低阈值必须为 5～5000 µT。")
            if not math.isfinite(z_probe_step) or not 0.01<=z_probe_step<=0.10:
                raise ValueError("中心Z探测步长必须为 0.01～0.10 mm。")
            if not math.isfinite(z_max_search) or not 0.2<=z_max_search<=2.0:
                raise ValueError("中心Z向下最大搜索必须为 0.2～2.0 mm。")
            domain_map=predictor.metadata.get("calibration_domain_mm")
            domain=(domain_map.get("depth") if isinstance(domain_map,dict) else None)
            if isinstance(domain,(list,tuple)) and len(domain)>=2:
                lo,hi=float(min(domain)),float(max(domain))
                if not lo-1e-6<=depth<=hi+1e-6:
                    raise ValueError(f"自动校准压深 {depth:g} mm 超出所选模型训练范围 {lo:g}～{hi:g} mm。")
            center_x=float(self.sensor_center_x_var.get());center_y=float(self.sensor_center_y_var.get())
            contact=float(self.contact_z_var.get())
            safe_lift=abs(float(self.safe_lift_var.get()))
            approach_gap=abs(float(self.approach_clearance_var.get()))
            safe=contact+safe_lift
            probe_start=min(safe,contact+max(0.50,approach_gap))
            probe_end=contact-z_max_search
            xy_speed=max(0.2,float(self.xy_speed_var.get()));z_speed=max(0.2,float(self.z_speed_var.get()))
            # Guard the complete range of any candidate Contact Z before motion.
            for phase,z in (("grid_reference_safe",probe_start+safe_lift),
                            ("grid_reference_z_probe",probe_end),
                            ("grid_reference_indent",probe_end-z_max_search-depth)):
                self.printer._guard_absolute_target(z=z,mode="grid_points",phase=phase)
            for ref_x,ref_y in refs.values():
                self.printer._guard_absolute_target(
                    x=center_x+ref_x,y=center_y+ref_y,mode="grid_points",phase="grid_reference_xy")
        except Exception as exc:
            messagebox.showerror("一键自动五点校准参数",str(exc));return
        self.start_magnetics()
        if self.mag_thread is None or not self.mag_thread.is_alive():
            messagebox.showerror("一键自动五点校准","磁场采集未能启动，未执行打印机动作。")
            return
        if not self._ensure_printer_connected() or not self._require_home_for_auto():return
        if not self._confirm_contact_motion(
                f"自动重装校准：中心先用15通道原始ΔB探测Contact Z（最低阈值"
                f" {minimum_z_threshold:g} µT，步长 {z_probe_step:g} mm，最多向下"
                f" {z_max_search:g} mm），四方向复测Z后再以压深 {depth:g} mm采五点"):
            return
        if self.printer_thread and self.printer_thread.is_alive():
            messagebox.showwarning("一键自动五点校准","已有自动动作正在运行。")
            return
        with self.live_predict_lock:
            self.live_predict_enabled=True
            self.live_predict_min_input_ns=time.perf_counter_ns()
            self._live_predict_clear_queues()
        self.live_alignment_recent.clear()
        self.live_source_fingerprint_recent.clear()
        self.live_alignment_auto_running=True
        self.live_alignment_auto_btn.state(["disabled"])
        basis_text=("训练源数据15通道ΔB直接匹配" if calibration_basis=="source_delta_b"
                    else "模型预测XY二级对齐")
        self.live_alignment_status_var.set(
            f"{preset}｜中心原始磁场Contact Z探测准备中｜{basis_text}｜四方向随后验证Z。")
        self.motion_abort_event.clear()

        def job():
            error=None
            final_safe=safe
            try:
                self.printer.move_abs(z=safe,speed_mm_s=z_speed,mode="grid_points",phase="grid_reference_safe")
                self.printer.move_abs(x=center_x,y=center_y,speed_mm_s=xy_speed,
                                      mode="grid_points",phase="grid_reference_xy")
                state=self._live_alignment_required_state(preset)
                state.clear();state.update(offset_mm=[0.0,0.0],rotation_deg=0.0,
                    calibration_basis=calibration_basis,calibration_points={},check_points=[])
                with self.live_predict_lock:
                    predictor.set_xy_rigid_transform(0.0,0.0,0.0,"calibration_in_progress",{})
                center_label=min(refs,key=lambda name:math.hypot(
                    float(refs[name][0]),float(refs[name][1])))
                ordered=[(center_label,refs[center_label])]+[
                    (name,point) for name,point in refs.items() if name!=center_label]
                total=len(ordered);candidate_contact=None;z_validations=[]
                validation_tolerance=max(0.15,4.0*z_probe_step)
                for index,(label,(ref_x,ref_y)) in enumerate(ordered,1):
                    if self.motion_abort_event.is_set():raise RuntimeError("五点自动校准已停止。")
                    self.after(0,lambda i=index,l=label:self.live_alignment_status_var.set(
                        f"{preset}｜Z探测/五点校准 {i}/{total}：{l}"))
                    point_safe=((contact if candidate_contact is None else candidate_contact)+safe_lift)
                    self.printer.move_abs(z=point_safe,speed_mm_s=z_speed,mode="grid_points",phase="grid_reference_safe")
                    self.printer.move_abs(x=center_x+ref_x,y=center_y+ref_y,speed_mm_s=xy_speed,
                                          mode="grid_points",phase="grid_reference_xy")
                    raw_b0=self._live_alignment_capture_raw_contact_baseline(
                        minimum_z_threshold)
                    if candidate_contact is None:
                        local_start=probe_start;local_end=probe_end
                    else:
                        local_start=min(point_safe,candidate_contact+max(0.50,approach_gap))
                        local_end=candidate_contact-z_max_search
                    detected=self._live_alignment_probe_raw_contact_z(
                        label,raw_b0,local_start,local_end,z_probe_step,z_speed,4)
                    local_contact=float(detected["reported_z_mm"])
                    if candidate_contact is None:
                        candidate_contact=local_contact
                        final_safe=candidate_contact+safe_lift
                        state["z_contact_calibration"]={
                            "method":"raw_15ch_delta_threshold_center_plus_four_validation_v1",
                            "committed":False,
                            "old_contact_z_mm":contact,
                            "center_contact_z_mm":candidate_contact,
                            "compensation_mm":candidate_contact-contact,
                            "minimum_threshold_ut":minimum_z_threshold,
                            "threshold_ut":float(raw_b0["threshold_ut"]),
                            "baseline_noise_p95_ut":float(raw_b0["noise_p95_ut"]),
                            "probe_step_mm":z_probe_step,
                            "max_search_mm":z_max_search,
                            "confirm_frames":4,
                            "center_detection":detected,
                            "validation_tolerance_mm":validation_tolerance,
                            "validation_points":z_validations,
                            "flatness_policy":"warn_only_fixed_center_z_r52",
                            "surface_z_compensation_applied":False,
                            "flatness_warnings":[],
                        }
                    else:
                        assessment=center_z_plane_assessment(
                            candidate_contact,local_contact,depth,validation_tolerance)
                        deviation=assessment["deviation_from_center_mm"]
                        validation={"label":label,"contact_z_mm":local_contact,
                                    "deviation_from_center_mm":deviation,
                                    "threshold_ut":float(raw_b0["threshold_ut"]),
                                    "detection":detected}
                        validation.update(assessment)
                        z_validations.append(validation)
                        state["z_contact_calibration"]["validation_points"]=z_validations
                        if assessment["exceeds_indent_safety_limit"]:
                            raise RuntimeError(
                                f"{label} 按中心Z压入将使越阈Z估算压深"
                                f" {assessment['estimated_local_indent_depth_mm']:.3f} mm，"
                                f"超过 {AUTO_MAX_INDENT_MM:g} mm 安全上限；未执行该点压入。"
                            )
                        if assessment["outside_reference_tolerance"]:
                            warning=(
                                f"{label} 磁场越阈Z={local_contact:.3f} mm，"
                                f"与中心 {candidate_contact:.3f} mm 相差 {deviation:+.3f} mm，"
                                f"超过参考 ±{validation_tolerance:.3f} mm；"
                                "仅提醒并继续按中心Contact Z运动，未进行曲面补偿。"
                            )
                            state["z_contact_calibration"]["flatness_warnings"].append(warning)
                            self.log("AUTO CONTACT-Z PLANE WARNING: "+warning)
                            self.after(0,lambda w=warning:self.live_alignment_status_var.set(w))
                    approach=candidate_contact+approach_gap
                    zpress=candidate_contact-depth
                    self.printer.move_abs(z=approach,speed_mm_s=z_speed,
                                          mode="grid_points",phase="grid_reference_approach")
                    self._capture_pre_action_b0("grid_points",f"grid_reference_b0_{index}")
                    self._live_alignment_wait_baseline(predictor)
                    self.live_alignment_recent.clear();self.live_source_fingerprint_recent.clear()
                    capture_start=time.perf_counter_ns()
                    self.printer.move_abs(z=zpress,speed_mm_s=z_speed,
                                          mode="grid_points",phase="grid_reference_indent")
                    self.printer.dwell(0.55,mode="grid_points",phase="grid_reference_hold")
                    row={"reference_xy_mm":[ref_x,ref_y],
                         "calibration_basis":calibration_basis,
                         "capture_mode":"automatic_raw_contact_z_then_five_point",
                         "indent_depth_mm":depth,
                         "center_contact_z_mm":candidate_contact,
                         "local_validation_contact_z_mm":local_contact,
                         "local_contact_deviation_mm":local_contact-candidate_contact,
                         "captured_utc":datetime.utcnow().isoformat(timespec="seconds")+"Z"}
                    if calibration_basis=="source_delta_b":
                        delta,n,p95,amplitude=self._live_source_fingerprint_wait_delta(
                            preset,capture_start)
                        row.update(current_delta_uT=delta,median_frames=n,
                                   stability_p95_uT=p95,amplitude_uT=amplitude)
                    else:
                        x,y,n=self._live_alignment_wait_unaligned_xy(preset,capture_start)
                        row.update(predicted_unaligned_xy_mm=[x,y],median_frames=n)
                    state["calibration_points"][label]=row
                    self.printer.move_abs(z=approach,speed_mm_s=z_speed,
                                          mode="grid_points",phase="grid_reference_retract")
                if calibration_basis=="source_delta_b":
                    result=self._live_source_fingerprint_compute_state(
                        state,refs,predictor,depth)
                    if result["quality"]=="fail":
                        raise RuntimeError("源指纹匹配质量不足，未应用："+
                                           "；".join(result.get("warnings") or []))
                else:
                    self._live_alignment_compute_state(state,refs,alignment_mode)
                max_z_delta=max((abs(float(row["deviation_from_center_mm"]))
                                 for row in z_validations),default=0.0)
                z_meta=state["z_contact_calibration"]
                flatness_warning=bool(z_meta["flatness_warnings"])
                z_meta.update(validation_max_abs_mm=max_z_delta,
                              validation_points_complete=len(z_validations)==4,
                              validation_passed=len(z_validations)==4 and not flatness_warning,
                              flatness_warning=flatness_warning,
                              calibration_accepted_with_warning=flatness_warning,
                              committed=False,
                              completed_utc=datetime.utcnow().isoformat(timespec="seconds")+"Z")
                if len(z_validations)!=4:
                    raise RuntimeError(f"Contact Z四方向验证数量异常：{len(z_validations)}/4。")
                persisted_z_meta=json.loads(json.dumps(z_meta));persisted_z_meta["committed"]=True
                self._live_alignment_commit_contact_z(candidate_contact,persisted_z_meta)
                z_meta["committed"]=True
                state["automatic_calibration"]={
                    "reference_points":{k:list(v) for k,v in refs.items()},
                    "indent_depth_mm":depth,
                    "contact_z_calibration":json.loads(json.dumps(z_meta)),
                    "completed_utc":datetime.utcnow().isoformat(timespec="seconds")+"Z"}
                with self.live_predict_lock:
                    self._live_alignment_apply_to_predictor(predictor,preset,update_status=False)
            except Exception as exc:
                error=str(exc);self.log(f"Automatic five-point alignment ERROR: {exc}")
            finally:
                try:
                    self.printer.move_abs(z=final_safe,speed_mm_s=z_speed,mode="grid_points",phase="final_safe_z")
                    self.printer.move_abs(x=center_x,y=center_y,speed_mm_s=xy_speed,
                                          mode="grid_points",phase="grid_reference_xy")
                    self._set_printer_position_async(self.printer.query_position())
                except Exception as retract_exc:
                    self.log(f"Automatic five-point final safe return ERROR: {retract_exc}")
                    if error is None:error=f"最终安全返回失败：{retract_exc}"
                self.after(0,lambda e=error:self._live_alignment_auto_finish(preset,e))
        self.printer_thread=threading.Thread(target=job,name="MagTouchAutoFivePoint",daemon=True)
        self.printer_thread.start()

    def _live_predict_display_axes_changed(self):
        """Display transform only: never alter model fields or recorded predictions."""
        try:self.live_predict_edge_guard_var.set(f"{self._live_edge_guard_ms():g}")
        except Exception:return
        for view in (self.live_predict_3d,self.exp_predict_3d):view.clear()
        self.live_predict_plot.clear()
        self.live_display_point_times.clear()
        self.live_contact_shown_segment=-1
        self._schedule_settings_autosave()

    def prediction_trajectory_history_points(self):
        """Validated display capacity shared by live, experiment and replay 3-D views."""
        variable=getattr(self,"live_trajectory_history_var",None)
        raw=variable.get() if variable is not None else self.settings.get(
            "live_prediction",{}).get("trajectory_history_points",4000)
        try:
            value=int(str(raw).strip())
        except (TypeError,ValueError) as exc:
            raise ValueError("三维轨迹历史点数须为整数") from exc
        if not 50<=value<=50000:
            raise ValueError("三维轨迹历史点数须为 50～50000")
        return value

    def _live_trajectory_history_apply(self,show_error=True):
        """Hot-apply display history only; raw predictions and CSV are unchanged."""
        try:
            value=self.prediction_trajectory_history_points()
            self.live_trajectory_history_var.set(str(value))
            for name in ("live_predict_3d","exp_predict_3d","replay_prediction_3d"):
                view=getattr(self,name,None)
                if view is not None:view.set_maxlen(value)
            self._schedule_settings_autosave()
            return value
        except Exception as exc:
            if show_error:messagebox.showwarning("三维轨迹历史",str(exc))
            fallback=4000
            self.live_trajectory_history_var.set(str(fallback))
            for name in ("live_predict_3d","exp_predict_3d","replay_prediction_3d"):
                view=getattr(self,name,None)
                if view is not None:view.set_maxlen(fallback)
            return fallback

    def _clear_prediction_trajectory_history(self):
        for name in ("live_predict_3d","exp_predict_3d","replay_prediction_3d"):
            view=getattr(self,name,None)
            if view is not None:view.clear()
        plot=getattr(self,"live_predict_plot",None)
        if plot is not None:plot.clear()
        self.live_display_point_times.clear()
        self.live_contact_shown_segment=-1

    @staticmethod
    def _live_filter_mode_key(value):
        text=str(value or "").lower()
        if "ema" in text or "指数" in text:return "ema"
        if "kalman" in text or "卡尔曼" in text:return "kalman_cv"
        return "off"

    def _live_filter_parameters(self):
        alpha=float(self.live_filter_ema_alpha_var.get())
        q=float(self.live_filter_q_var.get())
        rxy=float(self.live_filter_r_xy_var.get())
        rz=float(self.live_filter_r_z_var.get())
        if not math.isfinite(alpha) or not 0.01<=alpha<=1.0:
            raise ValueError("EMA α 必须在 0.01～1.0；越小越平滑、滞后越大。")
        if not math.isfinite(q) or not 1e-6<=q<=1e4:
            raise ValueError("Kalman Q 必须在 1e-6～1e4；越大越跟随快速运动。")
        if not math.isfinite(rxy) or not 1e-6<=rxy<=100.0:
            raise ValueError("Kalman Rxy 必须在 1e-6～100 mm²；越大越平滑。")
        if not math.isfinite(rz) or not 1e-6<=rz<=100.0:
            raise ValueError("Kalman Rz 必须在 1e-6～100 mm²；越大越平滑。")
        mode=self._live_filter_mode_key(self.live_filter_mode_var.get())
        return mode,alpha,q,rxy,rz

    def _live_trajectory_filter_reset(self):
        self.live_trajectory_filter_state.update(
            signature=None,last_t_ns=None,ema=None,axes=None)

    def _live_filter_clear_views(self):
        """Clear every prediction view that has already been constructed.

        The live-prediction tab is built before the experiment tab.  Startup
        therefore must not assume that ``exp_predict_3d`` exists yet.
        """
        for attribute in ("live_predict_plot","live_predict_3d","exp_predict_3d"):
            view=getattr(self,attribute,None)
            clear=getattr(view,"clear",None)
            if callable(clear):
                clear()

    def _live_filter_hot_apply(self,show_error=True):
        try:
            mode,alpha,q,rxy,rz=self._live_filter_parameters()
        except (TypeError,ValueError) as exc:
            if show_error:messagebox.showwarning("轨迹显示滤波",str(exc))
            return False
        label={"off":"关闭","ema":f"EMA α={alpha:g}",
               "kalman_cv":f"Kalman-CV Q={q:g}, Rxy={rxy:g}, Rz={rz:g}"}[mode]
        self.live_filter_status_var.set(f"显示滤波：{label}；原始预测与CSV不变。")
        self._live_trajectory_filter_reset()
        self._live_filter_clear_views()
        self.live_display_point_times.clear();self.live_contact_shown_segment=-1
        # During startup the replay/experiment tabs do not exist yet.  A user
        # hot-apply happens after all tabs have been built and is persisted.
        if hasattr(self,"replay_backtest_panel"):
            self._schedule_settings_autosave()
        return True

    def _live_filter_reset_button(self):
        self._live_trajectory_filter_reset()
        self._live_filter_clear_views()
        self.live_display_point_times.clear();self.live_contact_shown_segment=-1
        self.live_filter_status_var.set(self.live_filter_status_var.get().split("；")[0]+"；状态已重置，等待下一有效点。")

    def _live_trajectory_filter_apply(self,t_ns,xyz):
        """Filter displayed XYZ only; caller has already recorded the raw prediction."""
        try:mode,alpha,q,rxy,rz=self._live_filter_parameters()
        except (TypeError,ValueError):return tuple(float(value) for value in xyz)
        signature=(mode,alpha,q,rxy,rz)
        state=self.live_trajectory_filter_state
        if state.get("signature")!=signature:
            self._live_trajectory_filter_reset();state=self.live_trajectory_filter_state
            state["signature"]=signature
        measurement=tuple(float(value) for value in xyz)
        if mode=="off":
            state["last_t_ns"]=int(t_ns)
            return measurement
        previous_t=state.get("last_t_ns")
        dt=(0.025 if previous_t is None else (int(t_ns)-int(previous_t))/1e9)
        state["last_t_ns"]=int(t_ns)
        if not math.isfinite(dt) or dt<=0.0 or dt>0.5:
            state["ema"]=None;state["axes"]=None;dt=0.025
        if mode=="ema":
            previous=state.get("ema")
            result=(measurement if previous is None else tuple(
                alpha*value+(1.0-alpha)*float(old) for value,old in zip(measurement,previous)))
            state["ema"]=result
            return result
        axes=state.get("axes")
        if axes is None:
            axes=[]
            for value,r in zip(measurement,(rxy,rxy,rz)):
                axes.append({"x":value,"v":0.0,"p00":4.0*r,"p01":0.0,
                             "p10":0.0,"p11":max(q,1.0)})
            state["axes"]=axes
            return measurement
        result=[]
        dt2=dt*dt;dt3=dt2*dt;dt4=dt2*dt2
        for axis,value,r in zip(axes,measurement,(rxy,rxy,rz)):
            xp=axis["x"]+dt*axis["v"];vp=axis["v"]
            p00=axis["p00"]+dt*(axis["p01"]+axis["p10"])+dt2*axis["p11"]+q*dt4/4.0
            p01=axis["p01"]+dt*axis["p11"]+q*dt3/2.0
            p10=axis["p10"]+dt*axis["p11"]+q*dt3/2.0
            p11=axis["p11"]+q*dt2
            innovation=value-xp;s=max(1e-12,p00+r);k0=p00/s;k1=p10/s
            axis["x"]=xp+k0*innovation;axis["v"]=vp+k1*innovation
            new_p00=(1.0-k0)*p00;new_p01=(1.0-k0)*p01
            new_p10=p10-k1*p00;new_p11=p11-k1*p01
            cross=0.5*(new_p01+new_p10)
            axis.update(p00=max(1e-12,new_p00),p01=cross,p10=cross,p11=max(1e-12,new_p11))
            result.append(axis["x"])
        return tuple(result)

    def _live_filter_label(self):
        mode=self._live_filter_mode_key(self.live_filter_mode_var.get())
        return {"off":"未滤波","ema":"EMA显示滤波","kalman_cv":"Kalman-CV显示滤波"}[mode]

    def _live_display_xyz(self,prediction):
        x=float(prediction.dx_mm);y=float(prediction.dy_mm);z=float(prediction.dz_mm)
        return ((-x if self.live_predict_invert_x_var.get() else x),
                (-y if self.live_predict_invert_y_var.get() else y),z)

    def _live_edge_guard_ms(self):
        try:value=float(self.live_predict_edge_guard_var.get())
        except (TypeError,ValueError):return 120.0
        if not math.isfinite(value):return 120.0
        return min(2000.0,max(0.0,value))

    def _trim_live_falling_edge(self,end_ns,show_live_prediction,show_experiment):
        """Retroactively drop display points belonging to the confirmed falling edge."""
        if end_ns is None or end_ns==self.live_contact_trimmed_end_ns:return 0
        cutoff=int(end_ns)-int(self._live_edge_guard_ms()*1e6)
        count=0
        while self.live_display_point_times and self.live_display_point_times[-1]>=cutoff:
            self.live_display_point_times.pop();count+=1
        if count:
            self.live_predict_plot.remove_tail(count,redraw=show_live_prediction)
            self.live_predict_3d.remove_tail(count,redraw=show_live_prediction)
            self.exp_predict_3d.remove_tail(count,redraw=show_experiment)
        self.live_contact_trimmed_end_ns=int(end_ns)
        return count

    def _live_predict_validate_mapping(self, quiet=False):
        try:
            mapping = normalize_output_mapping(self.live_predict_output_mapping_var.get())
            normalized = json.dumps(mapping, ensure_ascii=False, separators=(",", ":"))
            self.live_predict_output_mapping_var.set(normalized)
            if not quiet:
                self.live_predict_state_var.set("输出映射有效；原始字段名称仍按模型族大小写显示。")
            return mapping
        except Exception as exc:
            if not quiet:
                messagebox.showwarning("输出映射 / Output mapping", str(exc))
            return None

    def _live_predict_reset_mapping(self):
        self.live_predict_output_mapping_var.set(default_output_mapping_json())
        self._live_predict_validate_mapping(quiet=False)

    @staticmethod
    def _live_predict_flat_numbers(value):
        out=[]
        def visit(item):
            if isinstance(item,(list,tuple)):
                for child in item: visit(child)
            elif isinstance(item,(int,float)):
                out.append(float(item))
        visit(value)
        return out

    def _live_predict_format_raw(self, name, value):
        if value is None:
            return f"{name}: --"
        if name in {"x","y","z","dx","dy","X","Y","Z","dX","dY","dZ"} and isinstance(value,(int,float)):
            return f"{name}: {float(value):+.4f} mm"
        if name == "mode5" and isinstance(value,dict):
            return f"mode5: {value.get('label','?')} [{value.get('index','?')}] p={float(value.get('probability',0.0)):.3f}"
        if name == "Q":
            numbers=self._live_predict_flat_numbers(value)
            if numbers:
                return f"Q: mean={sum(numbers)/len(numbers):.3f} min={min(numbers):.3f}"
        text=json.dumps(value,ensure_ascii=False,separators=(",",":")) if isinstance(value,(dict,list,tuple)) else str(value)
        return f"{name}: {text[:80]}{'…' if len(text)>80 else ''}"

    def _live_predict_show_extras(self, extras):
        """Show only unmapped/additional outputs; fixed raw fields are already at left."""
        try:
            text=json.dumps(extras or {},ensure_ascii=False,indent=2,default=str)
        except Exception:
            text=repr(extras)
        self.live_predict_extra_text.configure(state="normal")
        self.live_predict_extra_text.delete("1.0","end")
        self.live_predict_extra_text.insert("1.0",text or "--")
        self.live_predict_extra_text.configure(state="disabled")

    def _live_predict_clear_raw_output(self):
        for name,variable in self.live_predict_raw_vars.items():
            variable.set(f"{name}: --")
        self._live_predict_show_extras({})

    def _live_predict_resolve_model_selection(self, value=None):
        selected = str(self.live_predict_model_var.get() if value is None else value).strip()
        return resolve_neural_model_selection(selected)

    def _live_predict_model_option_selected(self, _event=None):
        if self.live_predict_model_var.get().strip() in LIVE_NEURAL_MODEL_OPTIONS:
            self.live_predict_model_family_var.set("FinalTip-R56")
            if hasattr(self, "live_algorithm_panel"):
                self.live_algorithm_panel.invalidate()
            self._schedule_settings_autosave()

    def _live_predict_choose_model(self):
        current = self._live_predict_resolve_model_selection()
        initial = str(current if current.exists() else HERE.parent / "data")
        path = filedialog.askopenfilename(
            title="选择 MagTouch PyTorch checkpoint（含 R56）",
            initialdir=initial if Path(initial).exists() else str(HERE.parent),
            filetypes=[("MagTouch PyTorch checkpoint", "*.pt"), ("All files", "*.*")],
        )
        if path:
            self.live_predict_model_var.set(path)
            self.live_predict_model_family_var.set("auto")

    def _live_predict_autoload_saved_model(self):
        if self.live_algorithm_panel.vars['mode'].get() == FIT:
            self.live_predict_state_var.set("已恢复拟合设置；请载入算法，确认配置后手动采集 B0。")
            return
        path_text = self.live_predict_model_var.get().strip()
        if not path_text:
            return
        if self._live_predict_resolve_model_selection(path_text).is_file():
            self._live_predict_load_model()
        else:
            self.live_predict_state_var.set("已恢复 checkpoint 路径，但文件当前不存在 / Saved checkpoint not found")

    def _live_predict_load_model(self):
        if self.live_algorithm_panel.vars['mode'].get() == FIT:
            return
        if self.live_algorithm_panel.busy():
            messagebox.showwarning("实时预测", "Trial/Full Plan 正在运行，不能更换模型；请先结束当前实验。")
            return
        selected_model = self.live_predict_model_var.get().strip()
        path = self._live_predict_resolve_model_selection(selected_model)
        if not path.is_file():
            messagebox.showwarning("实时预测", "请先选择存在的 MagTouch .pt checkpoint。")
            return
        mapping=self._live_predict_validate_mapping(quiet=True)
        if mapping is None:
            messagebox.showwarning("实时预测", "输出映射 JSON 无效，请先点击“校验”。")
            return
        family=neural_model_family(selected_model, self.live_predict_model_family_var.get())
        self.live_algorithm_panel.invalidate()
        self.live_algorithm_panel.loading = True
        self.live_algorithm_panel.update_states()
        self.live_predict_load_btn.state(["disabled"])
        self.live_predict_state_var.set("正在载入模型 / Loading checkpoint...")

        def worker():
            try:
                # Standard package import: both Python and Pylance resolve this
                # from the project root, rather than relying on a temporary path.
                from training.live_multimodel_inference import UnifiedLivePredictor
                predictor = UnifiedLivePredictor(
                    path,
                    family=family,
                    output_mapping=mapping,
                )
                with self.live_predict_lock:
                    self.live_predictor = predictor
                    self.live_predict_enabled = False
                    self.live_contact_previous = None
                    self.live_contact_onset_ns = None
                    self.live_contact_last_end_ns = None
                    self.live_contact_trimmed_end_ns = None
                    self.live_display_point_times.clear()
                    self._live_trajectory_filter_reset()
                    self._live_predict_clear_queues()
                def done():
                    self.live_predict_plot.set_scale(predictor.visual_scale)
                    self.live_predict_plot.clear()
                    self.live_predict_3d.set_scale(predictor.visual_scale)
                    self.live_predict_3d.clear()
                    self.exp_predict_3d.set_scale(predictor.visual_scale)
                    self.exp_predict_3d.clear()
                    self.exp_predict_info_var.set(
                        f"模型已载入 [{predictor.family}]；等待静止 B0 / Waiting for B0"
                    )
                    self._live_predict_clear_raw_output()
                    baseline_s = float(getattr(
                        predictor, "recommended_baseline_seconds",
                        max(1.0, predictor.seq_len / 40.0),
                    ))
                    self.live_predict_begin_btn.configure(text=f"采集 {baseline_s:.1f} s 静止 B0 并开始")
                    self.live_predict_baseline_var.set(f"B0：请在未受力、无明显磁变化时点击“采集 {baseline_s:.1f} s 静止 B0 并开始”")
                    self.live_predict_state_var.set(f"模型已载入 [{predictor.family}] / " + predictor.model_summary)
                    self.live_predict_output_var.set(
                        "在线模型尚未开始；请确认磁层/传感器配置一致，再捕获实时静止基线。"
                    )
                self.after(0, done)
            except Exception as exc:
                self.after(0, lambda error=repr(exc): messagebox.showerror("实时预测模型载入失败", error))
                self.after(0, lambda: self.live_predict_state_var.set("模型载入失败 / Load failed"))
            finally:
                def finish_load():
                    self.live_algorithm_panel.loading = False
                    self.live_algorithm_panel.update_states()
                self.after(0, finish_load)
        threading.Thread(target=worker, name="MagTouchLoadMultiModel", daemon=True).start()

    def _live_predict_begin_baseline(self):
        if self.full_plan_running:
            messagebox.showwarning("实时预测", "Full Plan 运行中；每个 Trial 的动态 B0 由计划自动管理，不能手动重置。")
            return
        if not self.live_predict_profile_confirm_var.get():
            messagebox.showwarning(
                "实时预测", "请先确认当前磁层、传感器物理顺序和 MLX90393 配置与训练模型一致。"
            )
            return
        with self.live_predict_lock:
            predictor = self.live_predictor
            if predictor is None:
                messagebox.showwarning("实时预测", "请先载入所选拟合算法或神经网络模型。")
                return
            baseline_seconds = float(getattr(
                predictor, "recommended_baseline_seconds",
                max(1.0, predictor.seq_len / 40.0),
            ))
            self.live_predict_min_input_ns=time.perf_counter_ns()
            predictor.begin_baseline(seconds=baseline_seconds)
            self.live_predict_enabled = True
            self.live_contact_previous = None
            self.live_contact_onset_ns = None
            self.live_contact_last_end_ns = None
            self.live_contact_trimmed_end_ns = None
            self.live_display_point_times.clear()
            self.live_alignment_recent.clear()
            self.live_source_fingerprint_recent.clear()
            self._live_trajectory_filter_reset()
            self._live_predict_clear_queues()
        self.live_predict_plot.clear()
        self.live_predict_3d.clear()
        self.exp_predict_3d.clear()
        self.exp_predict_info_var.set("正在建立静止 B0；预测历史尚未输出 / Capturing baseline")
        self._live_predict_clear_raw_output()
        # A Trial may have been started before the operator enabled inference.
        # Attach the optional stream now so any later model outputs still enter that
        # same Trial instead of requiring a second recording workflow.
        with self.trial_lock:
            recorder=self.recorder
            if recorder is not None and recorder.prediction_writer is None:
                try:
                    self._live_predict_enable_recording(
                        recorder,"automatic_prediction_enabled_during_trial"
                    )
                except Exception as exc:
                    self.log(f"ERROR auto-attaching prediction to active Trial: {exc}")
                    messagebox.showerror("自动附加预测记录失败",str(exc))
        self.live_predict_baseline_var.set(f"B0：正在采集 0/{predictor.baseline_target_frames} 帧；请保持未受力静止")
        self.live_predict_state_var.set(f"在线预测已启用：先建立当前局部静止基线 B0（H{predictor.history} 需要 {baseline_seconds:.1f} s）")

    def _live_predict_model_metadata(self, source):
        with self.live_predict_lock:
            predictor = self.live_predictor
            if predictor is None:
                raise RuntimeError("No live prediction model is loaded")
            return {
                "recording_source": str(source),
                "checkpoint_path": str(getattr(predictor, "path", self.live_predict_model_var.get())),
                "model_family": str(predictor.family),
                "model_summary": str(predictor.model_summary),
                "history_frames_required": int(predictor.seq_len),
                "output_mapping": dict(predictor.output_mapping),
                "visual_adapter_scale_mm": [float(value) for value in predictor.visual_scale],
                "algorithm_metadata": dict(getattr(predictor, "metadata", {})),
            }

    def _live_predict_enable_recording(self, recorder, source):
        """Attach model output recording to the already-active atomic Trial."""
        with self.live_predict_lock:
            predictor = self.live_predictor
            if predictor is None or not self.live_predict_enabled:
                raise RuntimeError("请先载入模型并启用实时预测 / Load and enable live prediction first")
        newly_enabled=recorder.prediction_writer is None
        recorder.enable_prediction_recording(self._live_predict_model_metadata(source))
        if newly_enabled:
            recorder.write_event("LIVE_PREDICTION_RECORD_START", source, "live_prediction")
        self.live_predict_record_recorder = recorder
        self.live_predict_record_source = str(source)
        self.live_predict_dropped_groups = 0
        self.live_predict_record_state_var.set(
            f"● 当前 Trial 自动记录预测 [{predictor.family}]：{recorder.trial_dir.name} | live_predictions.csv"
        )
        if newly_enabled:
            self.log(f"Live prediction recording auto-attached: {recorder.trial_dir}")

    def _live_predict_pause(self):
        if self.live_predict_record_recorder is not None or self.full_plan_running:
            messagebox.showwarning("实时预测", "预测记录或 Full Plan 正在进行；请先结束当前实验。")
            return
        with self.live_predict_lock:
            self.live_predict_enabled = False
            self.live_contact_previous = None
            self.live_contact_onset_ns = None
            self.live_contact_last_end_ns = None
            self.live_contact_trimmed_end_ns = None
            self.live_display_point_times.clear()
            self.live_alignment_recent.clear()
            self.live_source_fingerprint_recent.clear()
            self._live_trajectory_filter_reset()
            self._live_predict_clear_queues()
        self.live_predict_state_var.set("在线预测已暂停；磁场采集和实验控制不受影响。")
        self.exp_predict_info_var.set("实时预测已暂停；保留最后三维轨迹 / Prediction paused")

    def _live_predict_clear_queues(self):
        for q in (self.live_predict_queue, self.live_predict_results):
            try:
                while True:
                    q.get_nowait()
            except queue.Empty:
                pass

    def _live_predict_submit(self, data, meta):
        with self.live_predict_lock:
            enabled = bool(self.live_predict_enabled and self.live_predictor is not None)
        if not enabled:
            return
        copied = {int(index): dict(value) for index, value in data.items()}
        input_t_ns=int(meta.get("trigger_base_ns") or time.perf_counter_ns())
        if input_t_ns < int(self.live_predict_min_input_ns):
            return
        item = (copied, input_t_ns)
        try:
            self.live_predict_queue.put_nowait(item)
        except queue.Full:
            # Retain recent data so a slow CPU model cannot stall acquisition.
            if self.live_predict_record_recorder is not None:
                self.live_predict_dropped_groups += 1
            try:
                self.live_predict_queue.get_nowait()
                self.live_predict_queue.put_nowait(item)
            except queue.Empty:
                pass

    def _live_predict_publish(self, payload):
        try:
            self.live_predict_results.put_nowait(payload)
        except queue.Full:
            try:
                self.live_predict_results.get_nowait()
                self.live_predict_results.put_nowait(payload)
            except queue.Empty:
                pass

    def _live_predict_loop(self):
        while not self.live_predict_stop.is_set():
            predictor = None
            try:
                data, t_ns = self.live_predict_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                with self.live_predict_lock:
                    predictor = self.live_predictor if self.live_predict_enabled else None
                    if predictor is None:
                        continue
                    if isinstance(predictor,LiveNormalizedFitPredictor) and predictor.ready:
                        delta=predictor.delta_from_group(data)
                        if delta is not None:
                            self.live_source_fingerprint_recent.append((
                                int(t_ns),str(predictor.metadata.get("preset","")),
                                tuple(float(value) for value in delta.tolist())))
                    predictions = predictor.push_group(data, t_ns)
                    annotated=[]
                    for prediction in predictions:
                        visible,reason=prediction_contact(prediction)
                        previous=self.live_contact_previous
                        discontinuous=(previous is None or previous[0] is not predictor
                            or prediction.t_ns-previous[1]>1_000_000_000 or prediction.t_ns<=previous[1])
                        previous_visible=(False if previous is None or len(previous)<3 else bool(previous[2]))
                        if visible and (discontinuous or not previous_visible):
                            self.live_contact_segment+=1
                            self.live_contact_onset_ns=int(prediction.t_ns)
                        elif not visible:
                            if previous_visible:self.live_contact_last_end_ns=int(prediction.t_ns)
                            self.live_contact_onset_ns=None
                        contact_age_ms=(0.0 if not visible or self.live_contact_onset_ns is None else
                                        max(0.0,(int(prediction.t_ns)-self.live_contact_onset_ns)/1e6))
                        self.live_contact_previous=(predictor,prediction.t_ns,visible)
                        annotated.append(replace(prediction,extra_outputs=dict(prediction.extra_outputs,
                            **{"display.contact":visible,"display.contact_reason":reason,
                               "display.contact_segment":self.live_contact_segment,
                               "display.contact_age_ms":contact_age_ms,
                               "display.contact_last_end_ns":self.live_contact_last_end_ns})))
                    predictions=annotated
                    progress = predictor.baseline_progress
                    ready = predictor.ready
                if predictions:
                    with self.trial_lock:
                        recorder = self.live_predict_record_recorder
                        if recorder is not None and recorder is self.recorder:
                            for prediction in predictions:
                                recorder.write_prediction(prediction)
                    self._live_predict_publish({"kind": "prediction", "value": predictions[-1], "predictor": predictor})
                elif not ready:
                    self._live_predict_publish({"kind": "baseline", "progress": progress, "predictor": predictor})
            except Exception as exc:
                self.live_contact_segment+=1
                self._live_predict_publish({"kind": "error", "error": repr(exc), "predictor": predictor})

    def _live_predict_drain_results(self, show_live_prediction, show_experiment):
        latest = None
        try:
            while True:
                latest = self.live_predict_results.get_nowait()
        except queue.Empty:
            pass
        if latest is None:
            return
        if latest.get("predictor", self.live_predictor) is not self.live_predictor:
            return
        if not self.live_predict_enabled:
            return
        if latest.get("kind") == "error":
            self.live_predict_state_var.set("在线预测错误：" + latest.get("error", "unknown"))
            self.exp_predict_info_var.set("在线预测错误：" + latest.get("error", "unknown"))
            self.live_predict_plot.clear()
            self.live_predict_3d.add_gap(redraw=show_live_prediction)
            self.exp_predict_3d.add_gap(redraw=show_experiment)
            self.live_alignment_recent.clear()
            self._live_trajectory_filter_reset()
            return
        if latest.get("kind") == "baseline":
            self.live_predict_plot.clear()
            self.live_predict_3d.add_gap(redraw=show_live_prediction)
            self.exp_predict_3d.add_gap(redraw=show_experiment)
            got, total = latest.get("progress", (0, 40))
            self.live_predict_baseline_var.set(f"B0：正在采集 {got}/{total} 帧；请保持未受力静止")
            self.exp_predict_info_var.set(f"B0：正在采集 {got}/{total} 帧；三维推理将在基线完成后显示")
            self.live_alignment_recent.clear()
            self.live_source_fingerprint_recent.clear()
            self._live_trajectory_filter_reset()
            return
        prediction = latest.get("value")
        if prediction is None:
            return
        self.live_predict_baseline_var.set("B0：已建立（当前会话局部静止基线）")
        for name in RAW_FIELD_ORDER:
            self.live_predict_raw_vars[name].set(
                self._live_predict_format_raw(name,prediction.raw_fields.get(name))
            )
        self._live_predict_show_extras(prediction.extra_outputs)
        visible,contact_reason=prediction_contact(prediction)
        contact_end_ns=prediction.extra_outputs.get("display.contact_last_end_ns")
        trimmed=self._trim_live_falling_edge(contact_end_ns,show_live_prediction,show_experiment)
        segment=prediction.extra_outputs.get("display.contact_segment",self.live_contact_shown_segment)
        if segment != self.live_contact_shown_segment:
            self.live_predict_3d.add_gap(redraw=False);self.exp_predict_3d.add_gap(redraw=False)
            self.live_predict_plot.clear()
            self.live_display_point_times.clear()
            self.live_alignment_recent.clear()
            self._live_trajectory_filter_reset()
            self.live_contact_shown_segment=segment
        if not visible:
            trim_note=(f" | 已回删下降沿 {trimmed} 点" if trimmed else "")
            status = str(prediction.extra_outputs.get("status", ""))+" | "+contact_reason+trim_note
            self.live_predict_motion_var.set("状态：" + str(status))
            self.live_predict_output_var.set("无有效接触/输出；不显示旧坐标。" + str(status))
            self.exp_predict_info_var.set(str(status))
            if prediction.extra_outputs.get("needs_baseline"):
                self.live_predict_baseline_var.set("B0 已失效：请在无载时重新采集")
            self.live_predict_plot.clear()
            self.live_predict_3d.add_gap(redraw=show_live_prediction)
            self.exp_predict_3d.add_gap(redraw=show_experiment)
            self.live_alignment_recent.clear()
            self._live_trajectory_filter_reset()
            return
        edge_guard_ms=self._live_edge_guard_ms()
        contact_age_ms=float(prediction.extra_outputs.get("display.contact_age_ms",edge_guard_ms))
        if contact_age_ms < edge_guard_ms:
            status=(f"接触升沿显示抑制 {contact_age_ms:.0f}/{edge_guard_ms:g} ms；"
                    "原始预测仍照常记录")
            self.live_predict_motion_var.set("状态："+status)
            self.live_predict_output_var.set(status+"；用于去除磁场升沿造成的虚假XY长尾。")
            self.exp_predict_info_var.set(status)
            self.live_predict_plot.clear()
            self.live_predict_3d.add_gap(redraw=show_live_prediction)
            self.exp_predict_3d.add_gap(redraw=show_experiment)
            self._live_trajectory_filter_reset()
            return
        state = "超阈值 / above threshold" if prediction.moving else "阈值内 / within threshold"
        sources="/".join(prediction.visual_sources)
        with self.live_predict_lock:
            predictor = self.live_predictor
            epsilon_text = f"{predictor.motion_epsilon_mm:.3f}" if predictor is not None else "--"
        self.live_predict_motion_var.set(f"兼容 XYZ={sources} | Derived: {state}")
        self.live_predict_output_var.set(
            f"|ΔP|={prediction.norm_mm:.4f} mm；派生状态阈值={epsilon_text} mm（不是动作分类器）；"
            f"输入轴 Q mean/min={prediction.q_mean:.3f}/{prediction.q_min:.3f}；"
            "仅显示当前模型的实际推理输出；不是打印机计划、打印机读回值或硅胶内部真实形变。"
        )
        raw_display_xyz=self._live_display_xyz(prediction)
        display_x,display_y,display_z=self._live_trajectory_filter_apply(
            prediction.t_ns,raw_display_xyz)
        filter_label=self._live_filter_label()
        self.live_predict_plot.add(display_x,display_y,display_z,redraw=show_live_prediction)
        self.live_predict_3d.add(display_x,display_y,display_z,redraw=show_live_prediction)
        self.exp_predict_3d.add(display_x,display_y,display_z,redraw=show_experiment)
        self.live_display_point_times.append(int(prediction.t_ns))
        if prediction.family == "NormalizedFit":
            unaligned=prediction.extra_outputs.get("unaligned_xyz_mm")
            if isinstance(unaligned,(list,tuple)) and len(unaligned)>=3:
                try:
                    ux,uy,uz=(float(unaligned[0]),float(unaligned[1]),float(unaligned[2]))
                    if all(math.isfinite(v) for v in (ux,uy,uz)):
                        self.live_alignment_recent.append((int(prediction.t_ns),
                            self._live_alignment_preset(),segment,ux,uy,uz))
                except (TypeError,ValueError):pass
        self.exp_predict_info_var.set(
            f"[{prediction.family}] Display XYZ=({display_x:+.4f}, {display_y:+.4f}, "
            f"{display_z:+.4f}) mm | adapter={'/'.join(prediction.visual_sources)} | "
            f"{filter_label} | Q mean/min={prediction.q_mean:.3f}/{prediction.q_min:.3f}"
        )
        if prediction.family == "NormalizedFit":
            fit_model=str(prediction.extra_outputs.get("fit_model","归一化拟合"))
            evidence=str(prediction.extra_outputs.get("evidence_tier","unspecified"))
            mode=prediction.raw_fields.get("mode5")
            if isinstance(mode,dict):
                motion_text=(f"动态状态={mode.get('label','?')} "
                             f"p={float(mode.get('probability',0.0)):.3f}")
            else:motion_text="静态拟合：无运动状态分类"
            self.live_predict_motion_var.set(f"拟合有效：{fit_model}；{motion_text}")
            warning=""
            if prediction.extra_outputs.get("depth_extrapolated"):
                warning += " 当前 Z 超出标定深度范围。"
            if prediction.extra_outputs.get("xy_outside_calibration_domain"):
                warning += " 当前 XY 超出该拟合的标定范围。"
            if prediction.extra_outputs.get("outside_declared_working_range"):
                warning += " 当前输出超出所选模型声明的工作区。"
            if "exploratory" in evidence:
                warning += " 当前为九点探索模型，不能作为20×20 mm正式验证结果。"
            translation=prediction.extra_outputs.get("xy_translation_mm")
            rotation=float(prediction.extra_outputs.get("xy_rotation_deg",0.0) or 0.0)
            translation_text=""
            if isinstance(translation,(list,tuple)) and len(translation)>=2:
                translation_text=(f" XY校准=({float(translation[0]):+.3f},"
                                  f"{float(translation[1]):+.3f})mm，旋转={rotation:+.3f}°。")
            self.live_predict_output_var.set(
                f"dXYZ=本次有效接触首点差值，非 R15 标签；{filter_label}；"
                "动态模型可输出mode5，Q未提供。" + translation_text + warning)
            self.exp_predict_info_var.set(
                f"[拟合] 显示XYZ=({display_x:+.4f}, {display_y:+.4f}, "
                f"{display_z:+.4f}) mm | {filter_label} | {fit_model}{translation_text}")

    def _build_maglayer_tab(self):
        tab=self.tab_maglayer
        outer=ttk.Frame(tab,padding=10); outer.pack(fill="both",expand=True)
        head=ttk.LabelFrame(outer,text="磁层配置档案 / Magnetic Layer Profile")
        head.pack(fill="x",pady=(0,8)); head.columnconfigure(1,weight=1)
        self.mag_profile_id_var=tk.StringVar(value=self.active_mag_profile_id)
        ttk.Label(head,text="配置ID / Profile ID").grid(row=0,column=0,sticky="e",padx=5,pady=4)
        self.mag_profile_combo=ttk.Combobox(head,textvariable=self.mag_profile_id_var,values=list(self.maglayer_profiles.keys()),state="normal")
        self.mag_profile_combo.grid(row=0,column=1,sticky="ew",padx=5,pady=4)
        self.mag_profile_combo.bind("<<ComboboxSelected>>",lambda _e:self._load_selected_mag_profile_to_editor())
        ttk.Button(head,text="载入 / Load",command=self._load_selected_mag_profile_to_editor).grid(row=0,column=2,padx=4)
        ttk.Button(head,text="保存 / Save",command=self._save_mag_profile_from_editor).grid(row=0,column=3,padx=4)
        ttk.Button(head,text="应用到当前实验 / Apply",command=self._apply_selected_magnetic_layer_profile).grid(row=0,column=4,padx=4)
        ttk.Button(head,text="重读配置文件 / Reload",command=self._reload_mag_profiles).grid(row=0,column=5,padx=4)

        form=ttk.LabelFrame(outer,text="一次设置，后续 Trial 自动带入 / Configure once, auto-propagate")
        form.pack(fill="x",pady=(0,8)); form.columnconfigure(1,weight=1); form.columnconfigure(3,weight=1)
        self.mag_profile_fields={}
        specs=[
            ("display_name","名称 / Name",str),("magnetic_layer_id","磁层ID / Layer ID",str),
            ("tactile_material_id","关联柔性材料 / Material ID",str),("magnetic_material","磁性材料 / Magnetic material",str),
            ("magnet_spec","磁片规格 / Magnet spec",str),("layer_thickness_mm","磁层厚度 mm",str),
            ("sensor_to_magnetic_layer_gap_mm","传感器-磁层距离 mm",str),
        ]
        r=0;c=0
        for key,label,_typ in specs:
            v=tk.StringVar(value=""); self.mag_profile_fields[key]=v
            ttk.Label(form,text=label).grid(row=r,column=c,sticky="e",padx=(6,2),pady=3)
            ttk.Entry(form,textvariable=v).grid(row=r,column=c+1,sticky="ew",padx=(0,8),pady=3)
            c=2 if c==0 else 0
            if c==0:r+=1
        if c!=0:r+=1
        ttk.Label(form,text="排布说明 / Arrangement").grid(row=r,column=0,sticky="ne",padx=(6,2),pady=3)
        self.mag_arrangement_text=ScrolledText(form,height=4,wrap="word"); self.mag_arrangement_text.grid(row=r,column=1,columnspan=3,sticky="ew",padx=(0,8),pady=3);r+=1
        ttk.Label(form,text="备注 / Notes").grid(row=r,column=0,sticky="ne",padx=(6,2),pady=3)
        self.mag_notes_text=ScrolledText(form,height=3,wrap="word"); self.mag_notes_text.grid(row=r,column=1,columnspan=3,sticky="ew",padx=(0,8),pady=3);r+=1

        hw=ttk.LabelFrame(outer,text="MLX90393 + 采集 + 安全阈值 / Sensor & Acquisition")
        hw.pack(fill="x");
        int_fields=[("gain","GAIN",7),("res_x","RES X",0),("res_y","RES Y",0),("res_z","RES Z",0),("filter","DIG_FILT",0),("osr","OSR",0),("osr2","OSR2",0),("raw_warn_abs","Raw预警",28000),("raw_critical_abs","Raw严重",32000),("sample_period_ms","目标周期ms",20)]
        self.mag_hw_vars={}
        for idx,(key,label,default) in enumerate(int_fields):
            v=tk.IntVar(value=default);self.mag_hw_vars[key]=v
            row=idx//5;col=(idx%5)*2
            ttk.Label(hw,text=label).grid(row=row,column=col,sticky="e",padx=(6,2),pady=3)
            ttk.Entry(hw,textvariable=v,width=8).grid(row=row,column=col+1,sticky="w",padx=(0,8),pady=3)
        self.mag_hw_vars["tcmp"]=tk.BooleanVar(value=False)
        ttk.Checkbutton(hw,text="TCMP",variable=self.mag_hw_vars["tcmp"]).grid(row=2,column=0,columnspan=2,sticky="w",padx=6,pady=3)
        self.mag_hw_vars["include_temperature_each_cycle"]=tk.BooleanVar(value=False)
        ttk.Checkbutton(hw,text="每周期温度",variable=self.mag_hw_vars["include_temperature_each_cycle"]).grid(row=2,column=2,columnspan=2,sticky="w",padx=6,pady=3)
        self.mag_hw_vars["i2c_clock"]=tk.StringVar(value="400 kHz")
        ttk.Label(hw,text="I²C Clock").grid(row=2,column=4,sticky="e",padx=(6,2));ttk.Combobox(hw,textvariable=self.mag_hw_vars["i2c_clock"],values=["100 kHz","400 kHz","750 kHz"],state="readonly",width=10).grid(row=2,column=5,sticky="w")
        self.mag_hw_vars["datasheet_profile"]=tk.StringVar(value="legacy_2015")
        ttk.Label(hw,text="换算表 / Datasheet profile").grid(row=2,column=6,sticky="e",padx=(6,2));ttk.Combobox(hw,textvariable=self.mag_hw_vars["datasheet_profile"],values=["legacy_2015","current_2023"],state="readonly",width=14).grid(row=2,column=7,sticky="w")
        self.mag_hw_vars["target_max_utilization"]=tk.DoubleVar(value=0.80)
        ttk.Label(hw,text="目标最大量程占用").grid(row=2,column=8,sticky="e",padx=(6,2));ttk.Entry(hw,textvariable=self.mag_hw_vars["target_max_utilization"],width=8).grid(row=2,column=9,sticky="w")
        self.mag_hw_vars["contact_z_mm"]=tk.DoubleVar(value=13.8)
        ttk.Label(hw,text="该装配接触面Z mm").grid(row=3,column=0,sticky="e",padx=(6,2));ttk.Entry(hw,textvariable=self.mag_hw_vars["contact_z_mm"],width=9).grid(row=3,column=1,sticky="w")
        ttk.Label(hw,text="更换磁层/柔性层后需重新标定；应用Profile时自动带入实验Contact Z",foreground="#555").grid(row=3,column=2,columnspan=8,sticky="w",padx=6)

        fp=ttk.LabelFrame(outer,text="安装指纹参考 / Installation Fingerprint（仅提示，不阻止实验）")
        fp.pack(fill="x",pady=(8,0));fp.columnconfigure(7,weight=1)
        self.mag_fingerprint_enabled_var=tk.BooleanVar(value=True)
        self.mag_fingerprint_warn_l2_var=tk.DoubleVar(value=500.0)
        self.mag_fingerprint_warn_channel_var=tk.DoubleVar(value=300.0)
        self.mag_fingerprint_status_var=tk.StringVar(value="尚未检测；请在确认无载时点击“比较当前无载”")
        ttk.Checkbutton(fp,text="启用参考提示",variable=self.mag_fingerprint_enabled_var).grid(row=0,column=0,padx=5,pady=3,sticky="w")
        ttk.Label(fp,text="L2提示阈值 µT").grid(row=0,column=1,sticky="e");ttk.Entry(fp,textvariable=self.mag_fingerprint_warn_l2_var,width=9).grid(row=0,column=2,padx=3)
        ttk.Label(fp,text="单通道阈值 µT").grid(row=0,column=3,sticky="e");ttk.Entry(fp,textvariable=self.mag_fingerprint_warn_channel_var,width=9).grid(row=0,column=4,padx=3)
        ttk.Button(fp,text="比较当前无载 / Compare",command=self._compare_current_installation_fingerprint).grid(row=0,column=5,padx=4)
        ttk.Button(fp,text="当前无载设为参考 / Capture",command=self._capture_current_installation_fingerprint).grid(row=0,column=6,padx=4)
        ttk.Label(fp,textvariable=self.mag_fingerprint_status_var,foreground="#8b4513",wraplength=1420,justify="left").grid(row=1,column=0,columnspan=8,sticky="w",padx=6,pady=(2,5))
        self._load_selected_mag_profile_to_editor()

    def _selected_mag_profile(self):
        pid=str(getattr(self,"mag_profile_id_var",tk.StringVar(value=self.active_mag_profile_id)).get()).strip()
        if not pid: raise ValueError("磁层配置ID不能为空")
        p=self.maglayer_profiles.get(pid)
        if not isinstance(p,dict): raise ValueError(f"未找到磁层配置: {pid}")
        return pid,p

    @staticmethod
    def _opt_float_text(v):
        s=str(v).strip()
        if s in ("","None","null","NA","N/A"): return None
        return float(s)

    def _load_selected_mag_profile_to_editor(self):
        try:
            pid,p=self._selected_mag_profile()
        except Exception:
            return
        for key,v in self.mag_profile_fields.items(): v.set("" if p.get(key) is None else str(p.get(key,"")))
        self.mag_arrangement_text.delete("1.0","end");self.mag_arrangement_text.insert("1.0",str(p.get("arrangement_description","")))
        self.mag_notes_text.delete("1.0","end");self.mag_notes_text.insert("1.0",str(p.get("notes","")))
        m=p.get("mlx90393",{});a=p.get("acquisition",{});s=p.get("safety",{})
        for key in ("gain","res_x","res_y","res_z","filter","osr","osr2"):
            self.mag_hw_vars[key].set(int(m.get(key,0)))
        self.mag_hw_vars["tcmp"].set(bool(m.get("tcmp",False)))
        self.mag_hw_vars["datasheet_profile"].set(str(m.get("datasheet_profile","legacy_2015")))
        self.mag_hw_vars["sample_period_ms"].set(int(a.get("sample_period_ms",20)))
        self.mag_hw_vars["i2c_clock"].set(str(a.get("i2c_clock","400 kHz")))
        self.mag_hw_vars["include_temperature_each_cycle"].set(bool(a.get("include_temperature_each_cycle",False)))
        self.mag_hw_vars["raw_warn_abs"].set(int(s.get("raw_warn_abs",28000)))
        self.mag_hw_vars["raw_critical_abs"].set(int(s.get("raw_critical_abs",32000)))
        self.mag_hw_vars["target_max_utilization"].set(float(s.get("target_max_utilization",0.80)))
        cal=p.get("calibration",{})
        self.mag_hw_vars["contact_z_mm"].set(float(cal.get("contact_z_mm",13.8)))
        fingerprint=p.get("installation_fingerprint",{}) if isinstance(p.get("installation_fingerprint",{}),dict) else {}
        self.mag_fingerprint_enabled_var.set(bool(fingerprint.get("enabled",True)))
        self.mag_fingerprint_warn_l2_var.set(float(fingerprint.get("warn_l2_uT",500.0)))
        self.mag_fingerprint_warn_channel_var.set(float(fingerprint.get("warn_max_channel_uT",300.0)))
        self.mag_fingerprint_reference=fingerprint.get("reference_b0_5x3_uT")
        self.mag_fingerprint_status_var.set(
            "已载入参考；尚未比较（仅提示）" if self.mag_fingerprint_reference is not None
            else "该Profile尚无安装指纹参考；不影响实验运行")

    def _save_mag_profile_from_editor(self):
        try:
            pid=self.mag_profile_id_var.get().strip()
            if not pid: raise ValueError("请填写配置ID")
            old=dict(self.maglayer_profiles.get(pid,{}) or {})
            p=old
            for key,v in self.mag_profile_fields.items():
                if key in ("layer_thickness_mm","sensor_to_magnetic_layer_gap_mm"):
                    p[key]=self._opt_float_text(v.get())
                else:p[key]=v.get().strip()
            p["arrangement_description"]=self.mag_arrangement_text.get("1.0","end").strip()
            p["notes"]=self.mag_notes_text.get("1.0","end").strip()
            p["mlx90393"]={
                "datasheet_profile":self.mag_hw_vars["datasheet_profile"].get(),"gain":int(self.mag_hw_vars["gain"].get()),
                "res_x":int(self.mag_hw_vars["res_x"].get()),"res_y":int(self.mag_hw_vars["res_y"].get()),"res_z":int(self.mag_hw_vars["res_z"].get()),
                "filter":int(self.mag_hw_vars["filter"].get()),"osr":int(self.mag_hw_vars["osr"].get()),"osr2":int(self.mag_hw_vars["osr2"].get()),"tcmp":bool(self.mag_hw_vars["tcmp"].get())}
            p["acquisition"]={"i2c_clock":self.mag_hw_vars["i2c_clock"].get(),"sample_period_ms":int(self.mag_hw_vars["sample_period_ms"].get()),"include_temperature_each_cycle":bool(self.mag_hw_vars["include_temperature_each_cycle"].get())}
            p["safety"]={"raw_warn_abs":int(self.mag_hw_vars["raw_warn_abs"].get()),"raw_critical_abs":int(self.mag_hw_vars["raw_critical_abs"].get()),"target_max_utilization":float(self.mag_hw_vars["target_max_utilization"].get())}
            p.setdefault("calibration",{})
            p["calibration"]["contact_z_mm"]=float(self.mag_hw_vars["contact_z_mm"].get())
            previous_fp=old.get("installation_fingerprint",{}) if isinstance(old.get("installation_fingerprint",{}),dict) else {}
            p["installation_fingerprint"]={
                **previous_fp,
                "enabled":bool(self.mag_fingerprint_enabled_var.get()),
                "enforcement":"advisory",
                "reference_b0_5x3_uT":self.mag_fingerprint_reference,
                "warn_l2_uT":float(self.mag_fingerprint_warn_l2_var.get()),
                "warn_max_channel_uT":float(self.mag_fingerprint_warn_channel_var.get()),
                "blocks_experiment":False,
            }
            self.maglayer_profiles[pid]=p;self.maglayer_profiles_doc["profiles"]=self.maglayer_profiles
            if not self.maglayer_profiles_doc.get("default_profile_id"):self.maglayer_profiles_doc["default_profile_id"]=pid
            save_magnetic_layer_profiles(self.maglayer_profiles_doc,self.maglayer_profiles_path)
            self.mag_profile_combo.configure(values=list(self.maglayer_profiles.keys()))
            self.log(f"磁层配置已保存: {pid}")
        except Exception as e:
            messagebox.showerror("保存磁层配置",str(e));self.log(f"Save magnetic layer profile ERROR: {e}")

    def _reload_mag_profiles(self):
        try:
            self.maglayer_profiles_doc=load_magnetic_layer_profiles(self.maglayer_profiles_path);self.maglayer_profiles=self.maglayer_profiles_doc.get("profiles",{})
            self.mag_profile_combo.configure(values=list(self.maglayer_profiles.keys()))
            if self.mag_profile_id_var.get() not in self.maglayer_profiles and self.maglayer_profiles:
                self.mag_profile_id_var.set(next(iter(self.maglayer_profiles)))
            self._load_selected_mag_profile_to_editor();self.log("磁层配置文件已重新读取")
        except Exception as e: messagebox.showerror("磁层配置",str(e))

    @staticmethod
    def _magnetic_group_fingerprint_vector(data):
        vector=[]
        for index in range(5):
            row=data.get(index) if isinstance(data,dict) else None
            if not row:return None
            try:vector.append([float(row["x_ut"]),float(row["y_ut"]),float(row["z_ut"])])
            except (KeyError,TypeError,ValueError):return None
        return vector

    def _recent_unloaded_fingerprint(self):
        if len(self.mag_fingerprint_history)<10:
            raise RuntimeError("有效历史不足10帧；请先开始磁场采样，并确认当前完全无载")
        return robust_fingerprint(list(self.mag_fingerprint_history))

    def _compare_current_installation_fingerprint(self):
        try:
            pid,p=self._selected_mag_profile();current=self._recent_unloaded_fingerprint()
            configured=dict(p.get("installation_fingerprint",{}) or {})
            configured.update(enabled=bool(self.mag_fingerprint_enabled_var.get()),
                              reference_b0_5x3_uT=self.mag_fingerprint_reference,
                              warn_l2_uT=float(self.mag_fingerprint_warn_l2_var.get()),
                              warn_max_channel_uT=float(self.mag_fingerprint_warn_channel_var.get()),
                              enforcement="advisory")
            result=compare_fingerprint(current,configured)
            self.mag_fingerprint_status_var.set(f"{pid} | {result['message']}")
            self.log("Installation fingerprint advisory: "+json.dumps(result,ensure_ascii=False))
        except Exception as e:
            self.mag_fingerprint_status_var.set(f"安装指纹未完成：{e}（不阻止实验）")

    def _capture_current_installation_fingerprint(self):
        try:
            self.mag_fingerprint_reference=self._recent_unloaded_fingerprint()
            self.mag_fingerprint_status_var.set("已从最近无载窗口捕获参考；点击磁层Profile“保存”后写入配置。R38仍仅提示。")
        except Exception as e:self.mag_fingerprint_status_var.set(f"捕获失败：{e}（不阻止实验）")

    def _apply_selected_magnetic_layer_profile(self,disconnect=True,quiet=False):
        try:
            pid,p=self._selected_mag_profile()
            m=p.get("mlx90393",{});a=p.get("acquisition",{})
            # Profile is an acquisition invariant. Do not change it while sampling/recording.
            if disconnect and self.mag_thread and self.mag_thread.is_alive():
                raise RuntimeError("正在磁场采样，不能切换磁层配置。请先停止采样/Trial。")
            self.active_mag_profile_id=pid
            self.mag_fingerprint_history.clear()
            if hasattr(self,"i2c_clock_var"):self.i2c_clock_var.set(str(a.get("i2c_clock",self.i2c_clock_var.get())))
            if hasattr(self,"mag_interval_var"):self.mag_interval_var.set(int(a.get("sample_period_ms",self.mag_interval_var.get())))
            if hasattr(self,"include_temp_var"):self.include_temp_var.set(bool(a.get("include_temperature_each_cycle",self.include_temp_var.get())))
            if hasattr(self,"material_var"):self.material_var.set(str(p.get("tactile_material_id","")))
            if hasattr(self,"maglayer_var"):self.maglayer_var.set(str(p.get("magnetic_layer_id",pid)))
            _cal=p.get("calibration",{})
            if hasattr(self,"contact_z_var") and _cal.get("contact_z_mm") is not None:
                self.contact_z_var.set(float(_cal.get("contact_z_mm")))
            if hasattr(self,"active_mag_profile_summary_var"):
                self.active_mag_profile_summary_var.set(
                    f"磁层={pid} | {p.get('display_name','')} | GAIN={m.get('gain')} RES=({m.get('res_x')},{m.get('res_y')},{m.get('res_z')}) "
                    f"FILT={m.get('filter')} OSR={m.get('osr')} OSR2={m.get('osr2')} | {a.get('i2c_clock')} / target {a.get('sample_period_ms')} ms "
                    f"| ContactZ={_cal.get('contact_z_mm','--')} mm | 安装指纹=仅参考/不强制"
                )
            if disconnect and self.sampler is not None:
                try:self.bridge.close()
                except Exception:pass
                self.sampler=None
                if not quiet:self.log("磁层配置已切换；磁传感器连接已释放，请重新 Connect 以写入并核对新配置。")
            self.save_settings(silent=True)
            if not quiet:self.log(f"已应用磁层配置: {pid}")
        except Exception as e:
            if not quiet:messagebox.showerror("应用磁层配置",str(e))
            self.log(f"Apply magnetic layer profile ERROR: {e}")

    def _build_camera_tab(self):
        tab=self.tab_cam
        ctl=ttk.LabelFrame(tab,text="相机设置 / Camera Settings（当前阶段可全部不启用 / optional now）")
        ctl.pack(fill="x",padx=8,pady=8)
        self.cam_vars={}
        ccfg=self.settings.get("camera",{})
        for row,name,default_idx in [(0,"top",0),(1,"side",1)]:
            vars_={
                "enable":tk.BooleanVar(value=bool(ccfg.get(f"{name}_enable",False))),
                "index":tk.IntVar(value=int(ccfg.get(f"{name}_index",default_idx))),
                "width":tk.IntVar(value=int(ccfg.get("width",1280))),
                "height":tk.IntVar(value=int(ccfg.get("height",720))),
                "fps":tk.DoubleVar(value=float(ccfg.get("fps",60.0))),
            }
            self.cam_vars[name]=vars_
            ttk.Checkbutton(ctl,text=("顶视 / Top" if name=="top" else "侧视 / Side")+" 启用 / Enable",variable=vars_["enable"]).grid(row=row,column=0,padx=4,pady=3)
            ttk.Label(ctl,text="设备索引 / Index").grid(row=row,column=1); ttk.Entry(ctl,textvariable=vars_["index"],width=5).grid(row=row,column=2)
            ttk.Label(ctl,text="宽 / W").grid(row=row,column=3); ttk.Entry(ctl,textvariable=vars_["width"],width=6).grid(row=row,column=4)
            ttk.Label(ctl,text="高 / H").grid(row=row,column=5); ttk.Entry(ctl,textvariable=vars_["height"],width=6).grid(row=row,column=6)
            ttk.Label(ctl,text="帧率 / FPS").grid(row=row,column=7); ttk.Entry(ctl,textvariable=vars_["fps"],width=7).grid(row=row,column=8)
        ttk.Button(ctl,text="启动相机 / Start",command=self.start_cameras).grid(row=0,column=10,rowspan=2,padx=10)
        ttk.Button(ctl,text="停止相机 / Stop",command=self.stop_cameras).grid(row=0,column=11,rowspan=2,padx=4)
        self.camera_summary_var=tk.StringVar(value="相机未启动 / Cameras stopped")
        ttk.Label(ctl,textvariable=self.camera_summary_var).grid(row=0,column=12,rowspan=2,padx=12)

        prev=ttk.Frame(tab); prev.pack(fill="both",expand=True,padx=8,pady=8)
        prev.columnconfigure(0,weight=1); prev.columnconfigure(1,weight=1); prev.rowconfigure(1,weight=1)
        for col,name in enumerate(["top","side"]):
            ttk.Label(prev,text=f"Camera-{name}",font=("Segoe UI",10,"bold")).grid(row=0,column=col)
            lab=ttk.Label(prev,text="未启动",anchor="center")
            lab.grid(row=1,column=col,sticky="nsew",padx=5,pady=5)
            self.preview_labels[name]=lab

    def _build_experiment_tab(self):
        tab=self.tab_exp
        tab.columnconfigure(0,weight=1)
        tab.rowconfigure(0,weight=1)

        # Horizontal split:
        # LEFT  = live experiment monitor (2 cameras + one magnetic plot)
        # RIGHT = all experiment parameters/settings in one independently scrollable pane.
        split=ttk.PanedWindow(tab,orient="horizontal")
        split.grid(row=0,column=0,sticky="nsew",padx=6,pady=6)

        monitor=ttk.Frame(split)
        controls_host=ttk.Frame(split,width=590)
        split.add(monitor,weight=3)
        split.add(controls_host,weight=2)
        self.experiment_split=split
        self.experiment_left_ratio=0.60
        split.bind("<Configure>",self._enforce_experiment_split_ratio,add="+")
        self.after_idle(self._enforce_experiment_split_ratio)

        # ---------------- LEFT: monitor ----------------
        monitor.columnconfigure(0,weight=1)
        monitor.rowconfigure(0,weight=3)
        monitor.rowconfigure(1,weight=4)
        monitor.rowconfigure(2,weight=5)

        cam_box=ttk.LabelFrame(monitor,text="实验视频监视 / Experiment Cameras（可选 / Optional）")
        cam_box.grid(row=0,column=0,sticky="nsew",padx=(0,6),pady=(0,5))
        cam_box.columnconfigure(0,weight=1); cam_box.columnconfigure(1,weight=1)
        cam_box.rowconfigure(1,weight=1)
        for col,name,title in [(0,"top","顶视相机 / Top Camera"),(1,"side","侧视相机 / Side Camera")]:
            ttk.Label(cam_box,text=title,font=("Segoe UI",9,"bold")).grid(row=0,column=col,pady=(4,2))
            # A Canvas has a stable requested size: changing PhotoImage dimensions on
            # every frame cannot feed back into the geometry manager and resize the
            # complete left monitor pane (a tk.Label would do that).
            lab=tk.Canvas(cam_box,width=320,height=118,bg="#111318",
                          highlightthickness=1,highlightbackground="#6d737b",bd=0)
            lab.grid(row=1,column=col,sticky="nsew",padx=4,pady=(0,4))
            lab.bind(
                "<Configure>",
                lambda e,c=lab:(
                    c.coords("camera_frame",e.width//2,e.height//2),
                    c.coords("camera_placeholder",e.width//2,e.height//2),
                ),
                add="+",
            )
            self._set_camera_placeholder(lab,"未启用 / Disabled")
            self.exp_preview_labels[name]=lab

        mag_box=ttk.LabelFrame(monitor,text="实验磁场监视 / Experiment Magnetic Monitor")
        mag_box.grid(row=1,column=0,sticky="nsew",padx=(0,6),pady=(5,0))
        mag_box.columnconfigure(0,weight=1); mag_box.rowconfigure(1,weight=1)
        self.exp_mag_summary_var=tk.StringVar(value="等待磁场数据 / Waiting for magnetic data")
        ttk.Label(mag_box,textvariable=self.exp_mag_summary_var).grid(row=0,column=0,sticky="w",padx=8,pady=(4,0))
        self.exp_mag_view_var=tk.StringVar(value="|B| / Magnitude")
        exp_cb=ttk.Combobox(mag_box,textvariable=self.exp_mag_view_var,values=list(MAG_VIEW_LABELS.keys()),state="readonly",width=20)
        exp_cb.grid(row=0,column=1,sticky="e",padx=6,pady=(4,0))
        self.exp_mag_plot=MultiMagPlot(mag_box,height=330,maxlen=400,view_mode="m")
        exp_cb.bind("<<ComboboxSelected>>",lambda _e:self.exp_mag_plot.set_mode(self.exp_mag_view_var.get()))
        self.exp_mag_plot.grid(row=1,column=0,columnspan=2,sticky="nsew",padx=4,pady=4)

        pred_box=ttk.LabelFrame(
            monitor,text="实验实时三维推理 / Experiment Live 3D Inference（左键旋转、滚轮缩放、右键平移）"
        )
        pred_box.grid(row=2,column=0,sticky="nsew",padx=(0,6),pady=(5,0))
        pred_box.columnconfigure(0,weight=1);pred_box.rowconfigure(0,weight=1)
        self.exp_predict_3d=LiveDxyz3DView(
            pred_box,maxlen=self.prediction_trajectory_history_points(),height=310)
        self.exp_predict_3d.grid(row=0,column=0,sticky="nsew",padx=4,pady=4)
        live_cfg=self.settings.get("live_prediction",{})
        self.exp_predict_3d.set_view(
            live_cfg.get("experiment_view_yaw_deg",live_cfg.get("view_yaw_deg",-45.0)),
            live_cfg.get("experiment_view_pitch_deg",live_cfg.get("view_pitch_deg",28.0)),
            live_cfg.get("experiment_view_zoom",live_cfg.get("view_zoom",1.0)),
            live_cfg.get("experiment_view_pan_x",live_cfg.get("view_pan_x",0.0)),
            live_cfg.get("experiment_view_pan_y",live_cfg.get("view_pan_y",0.0)),
        )
        self.exp_predict_3d.on_view_changed=self._schedule_settings_autosave
        self.exp_predict_info_var=tk.StringVar(
            value="等待实时推理；本视图与“实时预测”页使用同一个模型输出流"
        )
        trajectory_ctl=ttk.Frame(pred_box)
        trajectory_ctl.grid(row=1,column=0,sticky="ew",padx=6,pady=(0,3))
        ttk.Label(trajectory_ctl,text="历史预测轨迹点数（50～50000）").pack(side="left")
        exp_history_entry=ttk.Entry(
            trajectory_ctl,textvariable=self.live_trajectory_history_var,width=9)
        exp_history_entry.pack(side="left",padx=(4,2))
        exp_history_entry.bind(
            "<FocusOut>",lambda _e:self._live_trajectory_history_apply(show_error=False))
        exp_history_entry.bind("<Return>",lambda _e:self._live_trajectory_history_apply())
        ttk.Button(trajectory_ctl,text="应用",command=self._live_trajectory_history_apply).pack(
            side="left",padx=2)
        ttk.Button(trajectory_ctl,text="清空轨迹",command=self._clear_prediction_trajectory_history).pack(
            side="left",padx=2)
        ttk.Label(trajectory_ctl,text="仅影响显示，不修改预测CSV",foreground="#6d4c41").pack(
            side="left",padx=6)
        ttk.Label(pred_box,textvariable=self.exp_predict_info_var,foreground="#0d47a1",
                  wraplength=850,justify="left").grid(row=2,column=0,sticky="ew",padx=6,pady=(0,5))

        ttk.Label(
            monitor,
            text="显示策略 / Render policy：当前页只刷新本页的2路视频与磁场曲线；"
                 "原‘相机/视频’和‘磁场采集’页面停止绘图刷新，但底层采集、时间戳与Trial写盘继续运行。",
            foreground="#455a64",wraplength=900,justify="left"
        ).grid(row=3,column=0,sticky="ew",padx=4,pady=(5,0))

        # ---------------- RIGHT: independently scrollable controls ----------------
        controls_host.columnconfigure(0,weight=1); controls_host.rowconfigure(0,weight=1)
        right_canvas=tk.Canvas(controls_host,highlightthickness=0,width=565)
        right_vsb=ttk.Scrollbar(controls_host,orient="vertical",command=right_canvas.yview)
        right_canvas.configure(yscrollcommand=right_vsb.set)
        right_canvas.grid(row=0,column=0,sticky="nsew")
        right_vsb.grid(row=0,column=1,sticky="ns")
        right_body=ttk.Frame(right_canvas)
        right_win=right_canvas.create_window((0,0),window=right_body,anchor="nw")
        right_body.bind("<Configure>",lambda _e:right_canvas.configure(scrollregion=right_canvas.bbox("all")))
        right_canvas.bind("<Configure>",lambda e:right_canvas.itemconfigure(right_win,width=e.width))
        def _wheel(event):
            if getattr(event,"delta",0):
                right_canvas.yview_scroll(int(-event.delta/120),"units")
            return "break"
        def _wheel_on(_e): right_canvas.bind_all("<MouseWheel>",_wheel)
        def _wheel_off(_e): right_canvas.unbind_all("<MouseWheel>")
        right_canvas.bind("<Enter>",_wheel_on); right_canvas.bind("<Leave>",_wheel_off)
        right_body.bind("<Enter>",_wheel_on); right_body.bind("<Leave>",_wheel_off)
        self.experiment_controls_canvas=right_canvas

        rootf=ttk.LabelFrame(right_body,text="数据目录与试验编号 / Data Directory & Trial IDs")
        rootf.pack(fill="x",padx=5,pady=(3,7))
        self.data_root_var=tk.StringVar(value=str((HERE.parent / "data").resolve()))
        self.session_var=tk.StringVar(value=datetime.now().strftime("S%Y%m%d_A"))
        self.trial_var=tk.StringVar(value="000001")
        tcfg=self.settings.get("trial",{})
        self.auto_trial_id_var=tk.BooleanVar(value=bool(tcfg.get("auto_id",True)))
        self.trial_repeat_var=tk.IntVar(value=max(1,int(tcfg.get("repeats", tcfg.get("count",1)))))
        self.repeat_interval_var=tk.DoubleVar(value=float(tcfg.get("repeat_interval_s", tcfg.get("inter_trial_s",1.0))))
        self.auto_summary_enabled_var=tk.BooleanVar(value=bool(tcfg.get("auto_summary",True)))
        self.auto_summary_var=tk.StringVar(value="")
        ttk.Label(rootf,text="数据根目录 / Root").grid(row=0,column=0,sticky="e",padx=(6,2),pady=3)
        ttk.Entry(rootf,textvariable=self.data_root_var,width=36).grid(row=0,column=1,padx=3,sticky="ew")
        ttk.Button(rootf,text="选择 / Browse",command=self.choose_root).grid(row=0,column=2,padx=3)
        ttk.Label(rootf,text="实验会话 / Session").grid(row=1,column=0,sticky="e",padx=(6,2),pady=3)
        ttk.Entry(rootf,textvariable=self.session_var,width=20).grid(row=1,column=1,sticky="ew",padx=3)
        ttk.Label(rootf,text="试验编号 / Trial ID").grid(row=2,column=0,sticky="e",padx=(6,2),pady=3)
        ttk.Entry(rootf,textvariable=self.trial_var,width=12).grid(row=2,column=1,sticky="w",padx=3)
        ttk.Button(rootf,text="扫描下一编号 / Scan Next ID",command=self.scan_next_trial_id).grid(row=2,column=2,padx=3,pady=3,sticky="e")
        ttk.Checkbutton(rootf,text="自动扫描编号 / Auto ID",variable=self.auto_trial_id_var).grid(row=3,column=1,sticky="w",padx=3,pady=2)
        ttk.Checkbutton(rootf,text="自动参数摘要 / Auto summary",variable=self.auto_summary_enabled_var).grid(row=3,column=2,sticky="e",padx=3,pady=2)
        rootf.columnconfigure(1,weight=1)

        pf=ttk.LabelFrame(right_body,text="Trial 参数 / Trial Parameters（灰色 = 当前模式不使用 / grey = unused）")
        pf.pack(fill="x",padx=5,pady=(0,7))
        self.mode_var=tk.StringVar(value=TRIAL_MODE_LABELS["press"])
        self.x_var=tk.DoubleVar(value=149.0); self.y_var=tk.DoubleVar(value=147.0)
        self.z_var=tk.DoubleVar(value=0.1); self.dx_var=tk.DoubleVar(value=2.0); self.dy_var=tk.DoubleVar(value=0.0)
        self.speed_var=tk.DoubleVar(value=1.0); self.hold_var=tk.DoubleVar(value=0.5)
        self.indenter_var=tk.StringVar(value="flat_d6mm")
        self.material_var=tk.StringVar(value="")
        self.maglayer_var=tk.StringVar(value="")
        self.notes_var=tk.StringVar(value="")
        self.trial_z_depths_var=tk.StringVar(value="0.10,0.20,0.30,0.50")
        self.trial_xy_sweep_var=tk.StringVar(value="-3,-2,-1,0,1,2,3")
        self.trial_angle_count_var=tk.IntVar(value=8)
        self.trial_circle_radius_var=tk.DoubleVar(value=2.0)
        self.trial_circle_points_var=tk.IntVar(value=24)
        self.trial_grid_half_x_var=tk.DoubleVar(value=4.0)
        self.trial_grid_half_y_var=tk.DoubleVar(value=4.0)
        self.trial_grid_step_var=tk.DoubleVar(value=2.0)
        self.trial_param_widgets={}
        def add_trial_param(key,label_text,var,row,col,width=14,combo_values=None,colspan=1):
            lab=ttk.Label(pf,text=label_text); lab.grid(row=row,column=col,padx=(7,2),pady=3,sticky="e")
            if combo_values is not None:
                w=ttk.Combobox(pf,textvariable=var,values=combo_values,width=25,state="readonly")
                w.bind("<<ComboboxSelected>>",lambda _e:self._update_trial_param_states())
            else:
                w=ttk.Entry(pf,textvariable=var,width=width)
            w.grid(row=row,column=col+1,columnspan=colspan,padx=(0,8),pady=3,sticky="ew" if colspan>1 else "w")
            self.trial_param_widgets[key]=(lab,w)
            return w

        # Two-column parameter matrix: short enough to keep recording controls visible
        # while preserving the mode-dependent grey/enable logic.
        add_trial_param("mode","实验模式 / Mode",self.mode_var,0,0,combo_values=list(TRIAL_MODE_LABELS.values()),colspan=3)
        add_trial_param("x","中心X / Center X (mm)",self.x_var,1,0)
        add_trial_param("y","中心Y / Center Y (mm)",self.y_var,1,2)
        add_trial_param("z","压入 / Indent z (mm)",self.z_var,2,0)
        add_trial_param("speed","速度 / Speed (mm/s)",self.speed_var,2,2)
        add_trial_param("dx","X位移 / dx (mm)",self.dx_var,3,0)
        add_trial_param("dy","Y位移 / dy (mm)",self.dy_var,3,2)
        add_trial_param("hold","保持 / Hold (s)",self.hold_var,4,0)
        add_trial_param("repeat","Trial内重复 / Repeats",self.trial_repeat_var,4,2)
        add_trial_param("repeat_interval","重复间隔 / Repeat interval (s)",self.repeat_interval_var,5,0)
        add_trial_param("angle_count","方向数 / Angle count",self.trial_angle_count_var,5,2)
        add_trial_param("z_depths","Z阶梯 / Z depths (mm)",self.trial_z_depths_var,6,0,width=22)
        add_trial_param("xy_values","XY阶梯 / Sweep values (mm)",self.trial_xy_sweep_var,6,2,width=22)
        add_trial_param("circle_radius","圆周半径 / Circle radius (mm)",self.trial_circle_radius_var,7,0)
        add_trial_param("circle_points","圆周点数 / Circle points",self.trial_circle_points_var,7,2)
        add_trial_param("grid_half_x","网格半宽X / Grid half-X (mm)",self.trial_grid_half_x_var,8,0)
        add_trial_param("grid_half_y","网格半宽Y / Grid half-Y (mm)",self.trial_grid_half_y_var,8,2)
        add_trial_param("grid_step","网格步长 / Grid step (mm)",self.trial_grid_step_var,9,0)
        add_trial_param("indenter","压头 / Indenter ID",self.indenter_var,9,2)
        add_trial_param("material","材料 / Material ID（来自磁层配置）",self.material_var,10,0)
        add_trial_param("maglayer","磁层 / Layer ID（来自磁层配置）",self.maglayer_var,10,2)
        for _k in ("material","maglayer"):
            try:self.trial_param_widgets[_k][1].configure(state="disabled")
            except Exception:pass
        ttk.Label(pf,text="自动实验摘要 / Auto summary").grid(row=11,column=0,padx=(7,2),pady=3,sticky="e")
        summary_ent=ttk.Entry(pf,textvariable=self.auto_summary_var,state="readonly")
        summary_ent.grid(row=11,column=1,columnspan=3,sticky="ew",padx=(0,8))
        ttk.Label(pf,text="人工备注 / Manual notes (optional)").grid(row=12,column=0,padx=(7,2),pady=3,sticky="e")
        ttk.Entry(pf,textvariable=self.notes_var).grid(row=12,column=1,columnspan=3,sticky="ew",padx=(0,8))
        self.trial_param_hint_var=tk.StringVar(value="")
        ttk.Label(pf,textvariable=self.trial_param_hint_var,foreground="#1b5e20",
                  wraplength=520,justify="left").grid(row=13,column=0,columnspan=4,sticky="w",padx=8,pady=(3,6))
        pf.columnconfigure(1,weight=1); pf.columnconfigure(3,weight=1)
        for _v in (self.mode_var,self.x_var,self.y_var,self.z_var,self.dx_var,self.dy_var,self.speed_var,self.hold_var,
                   self.trial_z_depths_var,self.trial_xy_sweep_var,self.trial_angle_count_var,self.trial_circle_radius_var,
                   self.trial_circle_points_var,self.trial_grid_half_x_var,self.trial_grid_half_y_var,self.trial_grid_step_var,self.trial_repeat_var,self.repeat_interval_var):
            try:_v.trace_add("write",lambda *_:self._update_auto_summary())
            except Exception:pass
        self._update_auto_summary()

        btn=ttk.LabelFrame(right_body,text="录制与事件 / Recording & Events")
        btn.pack(fill="x",padx=5,pady=(0,7))
        ttk.Button(btn,text="开始Trial / Start",command=self.start_trial).grid(row=0,column=0,padx=4,pady=4,sticky="ew")
        ttk.Button(btn,text="停止Trial / Stop",command=self.stop_trial).grid(row=0,column=1,padx=4,sticky="ew")
        ttk.Button(btn,text="录制+自动动作 / Record + Auto",command=self.auto_execute_trial).grid(row=1,column=0,columnspan=2,padx=4,pady=4,sticky="ew")
        for j,(label,event) in enumerate([
            ("接触 / CONTACT","CONTACT"),("滑动 / SLIDE","SLIDE"),
            ("释放 / RELEASE","RELEASE"),("备注 / NOTE","NOTE")
        ]):
            ttk.Button(btn,text=label,command=lambda e=event:self.manual_event(e)).grid(
                row=2+j//2,column=j%2,padx=4,pady=3,sticky="ew")
        self.trial_status_var=tk.StringVar(value="无活动 Trial / No active trial")
        ttk.Label(btn,textvariable=self.trial_status_var,wraplength=510,justify="left").grid(
            row=4,column=0,columnspan=2,sticky="w",padx=7,pady=4)
        btn.columnconfigure(0,weight=1); btn.columnconfigure(1,weight=1)

        expl=ttk.LabelFrame(right_body,text="数据原则 / Data Rules")
        expl.pack(fill="x",padx=5,pady=(0,10))
        txt=(
            "1) 五路同时采集 / Always acquire all 5 sensors\n"
            "S1~S5 必须在同一次物理 Trial 中同时记录。1×1、2×2、2×2+1 只能在后处理中裁剪通道，"
            "不能分别重做实验后直接比较。\n\n"
            "2) 原始数据不可逆保留 / Preserve raw data\n"
            "magnetic_raw.csv 同时保存 raw ADC、µT、每颗芯片 trigger/read 时间、状态字和组触发跨度。"
            "采集阶段不做不可逆置零、滤波、裁剪或归一化。\n\n"
            "3) 当前标签等级 / Current label level\n"
            "暂不使用摄像头时，A5S G-code 轨迹属于 commanded motion（命令运动）标签。当前实测传感器中心"
            "(149,147)，上表面 Contact Z=11.1 mm；z_indent = Contact Z - Z_target。"
            "可用于初步网络验证，但不能宣称为真实柔性层 observed deformation。\n\n"
            "4) 时间轴 / Timebase\n"
            "自动实验不使用会阻塞旧Marlin解析器的 M400/G4；采用可中断主机侧等待。磁场、G-code、事件统一使用 perf_counter_ns()。"
            "M114 只用于位置查询/检查，不作为高频形变真值。\n\n"
            "5) Trial 是最小不可拆实验单元 / Trial is atomic\n"
            "同一 Trial 连续帧高度相关。Train/Validation/Test 必须按完整 Trial、Session、日期或物理样件划分，"
            "禁止把同一轨迹的帧随机拆到不同集合。\n\n"
            "6) 配置快照 / Configuration snapshot\n"
            "每个 Trial 的 meta.json 保存中心坐标、Contact Z、动作类型、压入/位移/速度/保持、传感器几何、"
            "I²C配置和软件版本，保证可复现。\n\n"
            "7) 灰色参数 / Grey parameters\n"
            "灰色项不是当前实验控制变量，动作执行时忽略；切换到需要该变量的实验模式后才启用。\n\n"
            "8) 异常数据 / Failed trials\n"
            "通信错误、M410、M112、人工停止的 Trial 不在原始层删除，保留并标记 aborted/error，后续由 QC 决定是否进入训练。\n\n"
            "9) Trial内重复 / Repeats inside one Trial\n"
            "实验页的 Repeats 表示同一个 Trial 目录内重复执行同一完整动作 N 次；Repeat interval 表示相邻动作之间的间隔。"
            "每次重复都在 events.csv 中记录 ACTION_REPEAT_BEGIN/END 和 repeat index。一个 Trial 仍是训练/验证/测试划分的最小单元。\n\n"
            "10) 视频边界 / Video boundary\n"
            "视频只可保留为表面记录/检查，不进入R15模型输入、标签、真值核验或结论证据链；"
            "磁场变化历史→接触相对平台/压头位移是当前唯一的R15学习链。"
        )
        ttk.Label(expl,text=txt,wraplength=520,justify="left").pack(fill="x",padx=8,pady=8)

        self._update_trial_param_states()

    def _build_data_tab(self):
        """Experiment data browser, safe management and synchronized replay."""
        tab=self.tab_data
        tab.columnconfigure(0,weight=1);tab.rowconfigure(0,weight=1)
        split=ttk.PanedWindow(tab,orient="horizontal")
        split.grid(row=0,column=0,sticky="nsew",padx=6,pady=6)
        left_host=ttk.Frame(split,width=720);right_host=ttk.Frame(split)
        split.add(left_host,weight=2);split.add(right_host,weight=4)
        left_host.columnconfigure(0,weight=1);left_host.rowconfigure(0,weight=1)
        left_canvas=tk.Canvas(left_host,highlightthickness=0,width=720)
        left_scroll=ttk.Scrollbar(left_host,orient="vertical",command=left_canvas.yview)
        left_canvas.configure(yscrollcommand=left_scroll.set)
        left_canvas.grid(row=0,column=0,sticky="nsew");left_scroll.grid(row=0,column=1,sticky="ns")
        left=ttk.Frame(left_canvas);left_window=left_canvas.create_window((0,0),window=left,anchor="nw")
        def resize_replay_left(_event=None):
            left_canvas.itemconfigure(left_window,width=left_canvas.winfo_width(),
                height=max(left.winfo_reqheight(),left_canvas.winfo_height()))
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))
        left.bind("<Configure>",resize_replay_left);left_canvas.bind("<Configure>",resize_replay_left)
        left.columnconfigure(0,weight=1);left.rowconfigure(2,weight=1,minsize=170)

        # R9: the entire replay/metadata pane gets a visible right-side vertical
        # scrollbar. This is especially useful on 14/16-inch notebooks where the
        # camera + magnetic plot + metadata cannot all fit vertically at once.
        right_host.columnconfigure(0,weight=1); right_host.rowconfigure(0,weight=1)
        data_canvas=tk.Canvas(right_host,highlightthickness=0,borderwidth=0)
        data_vsb=ttk.Scrollbar(right_host,orient="vertical",command=data_canvas.yview)
        data_canvas.configure(yscrollcommand=data_vsb.set)
        data_canvas.grid(row=0,column=0,sticky="nsew")
        data_vsb.grid(row=0,column=1,sticky="ns")
        right=ttk.Frame(data_canvas)
        data_win=data_canvas.create_window((0,0),window=right,anchor="nw")
        right.bind("<Configure>",lambda _e:data_canvas.configure(scrollregion=data_canvas.bbox("all")))
        data_canvas.bind("<Configure>",lambda e:data_canvas.itemconfigure(data_win,width=e.width))
        def _data_wheel(event):
            if getattr(event,"delta",0):
                data_canvas.yview_scroll(int(-event.delta/120),"units")
            return "break"
        def _data_wheel_on(_e): data_canvas.bind_all("<MouseWheel>",_data_wheel)
        def _data_wheel_off(_e): data_canvas.unbind_all("<MouseWheel>")
        right_host.bind("<Enter>",_data_wheel_on); right_host.bind("<Leave>",_data_wheel_off)
        data_canvas.bind("<Enter>",_data_wheel_on); data_canvas.bind("<Leave>",_data_wheel_off)
        self.data_right_canvas=data_canvas
        self.data_right_scrollbar=data_vsb
        right.columnconfigure(0,weight=1)
        right.rowconfigure(0,minsize=260); right.rowconfigure(2,minsize=390); right.rowconfigure(3,minsize=360)

        src=ttk.LabelFrame(left,text="数据源 / Data Source")
        src.grid(row=0,column=0,sticky="ew",padx=(0,5),pady=(0,5));src.columnconfigure(1,weight=1)
        ttk.Label(src,text="Root").grid(row=0,column=0,padx=4,pady=4)
        self.data_manage_root_var=tk.StringVar(value=self.data_root_var.get())
        ttk.Entry(src,textvariable=self.data_manage_root_var).grid(row=0,column=1,sticky="ew",padx=4,pady=4)
        ttk.Button(src,text="使用实验Root / Use Trial Root",command=lambda:self._data_use_trial_root()).grid(row=0,column=2,padx=3)
        ttk.Button(src,text="选择 / Browse",command=self._data_choose_root).grid(row=0,column=3,padx=3)
        ttk.Button(src,text="刷新索引 / Refresh",command=self.refresh_data_index).grid(row=0,column=4,padx=3)

        filt=ttk.Frame(left);filt.grid(row=1,column=0,sticky="ew",padx=(0,5),pady=(0,4));filt.columnconfigure(1,weight=1)
        ttk.Label(filt,text="筛选 / Filter").grid(row=0,column=0,padx=(2,4))
        self.data_filter_var=tk.StringVar(value="")
        fe=ttk.Entry(filt,textvariable=self.data_filter_var);fe.grid(row=0,column=1,sticky="ew")
        fe.bind("<Return>",lambda _e:self.refresh_data_index())

        # R9: hierarchical experiment browser. One complete acquisition Plan is one
        # collapsible folder; Trials are children instead of a 137-row flat list.
        cols=("mode","status","repeats","duration")
        tree_box=ttk.Frame(left);tree_box.grid(row=2,column=0,sticky="nsew",padx=(0,5));tree_box.columnconfigure(0,weight=1);tree_box.rowconfigure(0,weight=1)
        self.data_tree=ttk.Treeview(tree_box,columns=cols,show="tree headings",selectmode="browse")
        self.data_tree.heading("#0",text="实验文件夹 / Experiment → Trial")
        self.data_tree.column("#0",width=285,minwidth=180,anchor="w",stretch=True)
        headings={"mode":"Mode","status":"Status","repeats":"Rep","duration":"Duration(s)"}
        widths={"mode":105,"status":72,"repeats":42,"duration":72}
        for c in cols:
            self.data_tree.heading(c,text=headings[c]);self.data_tree.column(c,width=widths[c],anchor="w",stretch=(c=="mode"))
        ys=ttk.Scrollbar(tree_box,orient="vertical",command=self.data_tree.yview);xs=ttk.Scrollbar(tree_box,orient="horizontal",command=self.data_tree.xview)
        self.data_tree.configure(yscrollcommand=ys.set,xscrollcommand=xs.set)
        self.data_tree.grid(row=0,column=0,sticky="nsew");ys.grid(row=0,column=1,sticky="ns");xs.grid(row=1,column=0,sticky="ew")
        self.data_tree.bind("<<TreeviewSelect>>",self._data_tree_selected)
        self.data_tree.bind("<Double-1>",self._data_tree_double_click)
        self.data_trial_paths={}
        self.data_folder_paths={}

        manage=ttk.LabelFrame(left,text="选择与管理 / Selection & Management")
        manage.grid(row=3,column=0,sticky="ew",padx=(0,5),pady=(5,0));manage.columnconfigure(1,weight=1)
        self.data_selected_var=tk.StringVar(value="未选择 / No selection")
        ttk.Label(manage,textvariable=self.data_selected_var,wraplength=470,justify="left").grid(row=0,column=0,columnspan=4,sticky="w",padx=5,pady=3)
        ttk.Button(manage,text="载入回放 / Load Replay",command=self.load_selected_trial_for_replay).grid(row=1,column=0,padx=3,pady=3,sticky="ew")
        ttk.Label(manage,text="视频回放支持旧Trial帧号自动修复；无时间戳时按FPS近似同步 / Legacy video replay auto-fixed",foreground="#1b5e20",wraplength=330,justify="left").grid(row=5,column=0,columnspan=2,padx=3,pady=(5,2),sticky="w")
        ttk.Button(manage,text="打开文件夹 / Open Folder",command=self.open_selected_trial_folder).grid(row=1,column=1,padx=3,pady=3,sticky="ew")
        ttk.Label(
            manage,
            text="数据页只负责浏览 / QC / 回放；分析导出、TRAINING_EXPORT 与训练已移到“导出/训练”页面。",
            foreground="#555",wraplength=470,justify="left"
        ).grid(row=3,column=0,columnspan=4,padx=3,pady=4,sticky="w")
        self.data_qc_var=tk.StringVar(value="pending")
        ttk.Combobox(manage,textvariable=self.data_qc_var,values=["pending","include","exclude"],state="readonly",width=10).grid(row=2,column=0,padx=3,pady=3)
        ttk.Button(manage,text="保存QC / Save QC",command=self.save_selected_trial_qc).grid(row=2,column=1,padx=3,pady=3,sticky="ew")
        ttk.Button(manage,text="移至_TRASH / Move to Trash",command=self.trash_selected_trial).grid(row=2,column=2,columnspan=2,padx=3,pady=3,sticky="ew")
        for c in range(4):manage.columnconfigure(c,weight=1)

        # Right: synchronized cameras + magnetic replay
        cam=ttk.LabelFrame(right,text="回放视频 / Replay Cameras（若该Trial有视频 / if recorded）")
        cam.grid(row=0,column=0,sticky="nsew",pady=(0,5)); cam.columnconfigure(0,weight=1,uniform="replay_camera"); cam.columnconfigure(1,weight=1,uniform="replay_camera"); cam.rowconfigure(0,weight=1)
        self.replay_camera_labels={};self.replay_camera_images={}
        for col,name,title in [(0,"top","顶视 / Top"),(1,"side","侧视 / Side")]:
            box=ttk.Frame(cam);box.grid(row=0,column=col,sticky="nsew",padx=3,pady=3); box.columnconfigure(0,weight=1); box.rowconfigure(1,weight=1)
            ttk.Label(box,text=title).grid(row=0,column=0)
            lab=tk.Canvas(box,width=1,height=210,bg="#111318",highlightthickness=0)
            lab.grid(row=1,column=0,sticky="nsew")
            self.replay_camera_labels[name]=lab

        phead=ttk.Frame(right);phead.grid(row=1,column=0,sticky="ew",pady=(0,3));phead.columnconfigure(0,weight=1)
        self.replay_info_var=tk.StringVar(value="请选择Trial并载入回放 / Select a Trial and load replay")
        ttk.Label(phead,textvariable=self.replay_info_var).grid(row=0,column=0,sticky="w")
        self.replay_view_var=tk.StringVar(value="|B| / Magnitude")
        ttk.Label(phead,text="曲线 / Curves:").grid(row=0,column=1,padx=(8,2))
        rcb=ttk.Combobox(phead,textvariable=self.replay_view_var,values=list(MAG_VIEW_LABELS.keys()),state="readonly",width=20)
        rcb.grid(row=0,column=2);rcb.bind("<<ComboboxSelected>>",lambda _e:self.replay_plot.set_mode(self.replay_view_var.get()))

        replay_box=ttk.LabelFrame(right,text="磁场/动作同步回放 / Magnetic + Motion Replay")
        replay_box.grid(row=2,column=0,sticky="nsew");replay_box.columnconfigure(0,weight=1);replay_box.rowconfigure(0,weight=1)
        self.replay_plot=MultiMagPlot(replay_box,height=380,maxlen=500,view_mode="m")
        self.replay_plot.grid(row=0,column=0,sticky="nsew",padx=4,pady=4)
        self.replay_slider_var=tk.DoubleVar(value=0)
        self.replay_slider=ttk.Scale(replay_box,from_=0,to=1,variable=self.replay_slider_var,command=self._replay_slider_changed)
        self.replay_slider.grid(row=1,column=0,sticky="ew",padx=8,pady=(0,3))
        ctl=ttk.Frame(replay_box);ctl.grid(row=2,column=0,sticky="ew",padx=6,pady=(0,5));ctl.columnconfigure(5,weight=1)
        ttk.Button(ctl,text="▶ 播放 / Play",command=self.replay_play).grid(row=0,column=0,padx=2)
        ttk.Button(ctl,text="Ⅱ 暂停 / Pause",command=self.replay_pause).grid(row=0,column=1,padx=2)
        ttk.Button(ctl,text="■ 回到开始 / Start",command=self.replay_stop).grid(row=0,column=2,padx=2)
        ttk.Label(ctl,text="速度 / Speed").grid(row=0,column=3,padx=(8,2))
        self.replay_speed_var=tk.DoubleVar(value=1.0)
        ttk.Combobox(ctl,textvariable=self.replay_speed_var,values=["0.25","0.5","1.0","2.0","5.0","10.0"],state="readonly",width=6).grid(row=0,column=4)
        self.replay_state_var=tk.StringVar(value="")
        ttk.Label(ctl,textvariable=self.replay_state_var,wraplength=700,justify="left").grid(row=0,column=5,sticky="w",padx=8)

        prediction_box=ttk.LabelFrame(
            right, text="预测三维回放 / Prediction 3D Replay（左键旋转、滚轮缩放、右键拖动）"
        )
        prediction_box.grid(row=3,column=0,sticky="nsew",pady=(5,0))
        prediction_box.columnconfigure(0,weight=1); prediction_box.rowconfigure(0,weight=1)
        self.replay_prediction_3d=LiveDxyz3DView(
            prediction_box,maxlen=self.prediction_trajectory_history_points(),height=330)
        self.replay_prediction_3d.display_title="所选方法回测 / Selected backtest output"
        self.replay_prediction_3d.display_subtitle="显示所选离散/连续拟合或模型输出；探索性九点连续结果不等于20×20 mm正式验证"
        self.replay_prediction_3d.on_view_changed=self._schedule_settings_autosave
        self.replay_prediction_3d.grid(row=0,column=0,sticky="nsew",padx=4,pady=4)
        live_cfg=self.settings.get("live_prediction",{})
        self.replay_prediction_3d.set_view(
            live_cfg.get("view_yaw_deg",-45.0),live_cfg.get("view_pitch_deg",28.0),
            live_cfg.get("view_zoom",1.0),live_cfg.get("view_pan_x",0.0),live_cfg.get("view_pan_y",0.0),
        )
        self.replay_prediction_info_var=tk.StringVar(value="该 Trial 尚未载入预测记录 / No prediction recording loaded")
        self.replay_prediction_text=ScrolledText(prediction_box,height=5,wrap="word",font=("Consolas",9))
        self.replay_prediction_text.grid(row=1,column=0,sticky="ew",padx=6,pady=(0,5))
        def update_replay_output(*_):
            self.replay_prediction_text.configure(state="normal")
            self.replay_prediction_text.delete("1.0","end")
            self.replay_prediction_text.insert("1.0",self.replay_prediction_info_var.get())
            self.replay_prediction_text.configure(state="disabled")
        self.replay_prediction_info_var.trace_add("write",update_replay_output)
        update_replay_output()
        self.replay_backtest_plot=LiveDxyzPlot(prediction_box,height=205,maxlen=160)
        self.replay_backtest_plot.NAMES=("X / dX","Y / dY","Z / dZ")
        self.replay_backtest_plot.display_title="回测三维适配曲线 / Backtest adapted outputs (mm)"
        self.replay_backtest_plot.empty_title="暂无有效回测输出 / No valid backtest output"
        self.replay_backtest_plot.grid(row=2,column=0,sticky="ew",padx=4,pady=4)

        details=ttk.LabelFrame(right,text="Trial元数据/事件 / Metadata & Current Event")
        details.grid(row=4,column=0,sticky="ew",pady=(5,0))
        self.data_detail=ScrolledText(details,height=6,font=("Consolas",8),wrap="word")
        self.data_detail.pack(fill="both",expand=True)
        self.data_detail.configure(state="disabled")

        self.replay_samples=[];self.replay_times=[];self.replay_events=[];self.replay_stages=[]
        self.replay_predictions=[];self.replay_prediction_times=[]
        self.replay_trial_dir=None;self.replay_playing=False;self.replay_tick_after=None;self.replay_median_dt_ms=20.0
        self.replay_caps={};self.replay_camera_times={}
        self.replay_backtest_panel=ReplayBacktestPanel(self,left,self.replay_backtest_plot,
            PROJECT_ROOT/"config"/"replay_fit_presets_r50.json")
        self.replay_backtest_panel.grid(row=4,column=0,sticky="ew",pady=(4,0))
        self.after(500,lambda:self.refresh_data_index(silent=True))

    def _build_analysis_tab(self):
        """Visible local research-data analysis page.

        This page deliberately keeps analysis separate from acquisition and
        training.  The analyzer reads original Plan folders in a worker process,
        never opens/copies video, and writes only a new result directory plus a
        small result ZIP.
        """
        tab = self.tab_analysis
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=1)

        intro = ttk.LabelFrame(tab, text="MagTouch-RA 研究数据整理与分析 / Research Data Analysis")
        intro.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        intro.columnconfigure(1, weight=1)
        ttk.Label(intro, text="推荐流程：先扫描整个工程生成小型研究快照，再分析 H1/H2/H3、QC、训练与验证结果；原始 Plan 和视频始终只读。",
                  foreground="#0d47a1", wraplength=1200, justify="left").grid(
                      row=0, column=0, columnspan=4, sticky="w", padx=6, pady=(5, 8))

        self.analysis_plan1_var = tk.StringVar(value="")
        self.analysis_plan2_var = tk.StringVar(value="")
        self.analysis_out_var = tk.StringVar(value="")
        self.analysis_snapshot_var = tk.StringVar(value="")
        self.analysis_hash_var = tk.BooleanVar(value=False)
        self.analysis_state_var = tk.StringVar(value="等待选择 Plan / Select Plan")

        ttk.Label(intro, text="Plan 1（19C）").grid(row=1, column=0, sticky="e", padx=5, pady=4)
        ttk.Entry(intro, textvariable=self.analysis_plan1_var).grid(row=1, column=1, sticky="ew", padx=4, pady=4)
        ttk.Button(intro, text="选择 / Browse", command=lambda: self._analysis_choose_plan(self.analysis_plan1_var)).grid(row=1, column=2, padx=4)
        ttk.Button(intro, text="使用数据页当前 Plan", command=self._analysis_use_selected_plan).grid(row=1, column=3, padx=4)

        ttk.Label(intro, text="Plan 2（可选 H2 5×5）").grid(row=2, column=0, sticky="e", padx=5, pady=4)
        ttk.Entry(intro, textvariable=self.analysis_plan2_var).grid(row=2, column=1, sticky="ew", padx=4, pady=4)
        ttk.Button(intro, text="选择 / Browse", command=lambda: self._analysis_choose_plan(self.analysis_plan2_var)).grid(row=2, column=2, padx=4)
        ttk.Button(intro, text="清空 Plan 2", command=lambda: self.analysis_plan2_var.set("")).grid(row=2, column=3, padx=4)

        ttk.Label(intro, text="输出目录（留空自动生成）").grid(row=3, column=0, sticky="e", padx=5, pady=4)
        ttk.Entry(intro, textvariable=self.analysis_out_var).grid(row=3, column=1, sticky="ew", padx=4, pady=4)
        ttk.Button(intro, text="选择输出目录", command=self._analysis_choose_output).grid(row=3, column=2, padx=4)
        ttk.Checkbutton(intro, text="计算输入文件 SHA-256（较慢）", variable=self.analysis_hash_var).grid(row=3, column=3, padx=4, sticky="w")

        ttk.Label(intro, text="研究快照（推荐）").grid(row=4, column=0, sticky="e", padx=5, pady=4)
        ttk.Entry(intro, textvariable=self.analysis_snapshot_var).grid(row=4, column=1, sticky="ew", padx=4, pady=4)
        ttk.Button(intro, text="选择 ZIP", command=self._analysis_choose_snapshot).grid(row=4, column=2, padx=4)
        self.analysis_collect_btn = ttk.Button(intro, text="1. 扫描当前工程", command=self._analysis_collect_snapshot)
        self.analysis_collect_btn.grid(row=4, column=3, padx=4)

        controls = ttk.Frame(intro)
        controls.grid(row=5, column=0, columnspan=4, sticky="ew", padx=4, pady=(5, 6))
        self.analysis_snapshot_run_btn = ttk.Button(controls, text="2. 分析研究快照（推荐）", command=self._analysis_run_snapshot)
        self.analysis_snapshot_run_btn.pack(side="left", padx=2)
        self.analysis_run_btn = ttk.Button(controls, text="单 Plan 深入分析（可选）", command=self._analysis_run)
        self.analysis_run_btn.pack(side="left", padx=2)
        ttk.Label(controls, textvariable=self.analysis_state_var, foreground="#1b5e20",
                  wraplength=950, justify="left").pack(side="left", padx=12, fill="x", expand=True)

        logf = ttk.LabelFrame(tab, text="分析日志 / Analysis Log")
        logf.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        logf.columnconfigure(0, weight=1)
        logf.rowconfigure(0, weight=1)
        self.analysis_log = ScrolledText(logf, height=24, font=("Consolas", 9), wrap="word")
        self.analysis_log.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        ttk.Button(logf, text="清空日志", command=lambda: self.analysis_log.delete("1.0", "end")).grid(
            row=1, column=0, sticky="e", padx=4, pady=3)

    def _analysis_log_append(self, message):
        def write():
            if hasattr(self, "analysis_log"):
                self.analysis_log.insert("end", str(message).rstrip() + "\n")
                self.analysis_log.see("end")
        if threading.current_thread() is threading.main_thread():
            write()
        else:
            self.after(0, write)

    def _analysis_choose_plan(self, variable):
        initial = ""
        if hasattr(self, "data_manage_root_var"):
            initial = self.data_manage_root_var.get()
        if not initial and hasattr(self, "data_root_var"):
            initial = self.data_root_var.get()
        p = filedialog.askdirectory(title="选择原始 Plan 文件夹", initialdir=initial or None)
        if p:
            variable.set(p)

    def _analysis_choose_output(self):
        initial = self.analysis_plan1_var.get()
        if not initial and hasattr(self, "data_root_var"):
            initial = self.data_root_var.get()
        p = filedialog.askdirectory(title="选择输出父目录（实际结果目录会自动新建）",
                                    initialdir=initial or None)
        if p:
            self.analysis_out_var.set(str(Path(p) / "MagTouch_RESEARCH_ANALYSIS_NEW"))

    def _analysis_choose_snapshot(self):
        p = filedialog.askopenfilename(title="选择 MagTouch 研究快照 ZIP",
                                       initialdir=str(HERE.parent.parent),
                                       filetypes=[("Research snapshot ZIP", "MagTouch_RESEARCH_SNAPSHOT_*.zip"),
                                                  ("ZIP files", "*.zip"), ("All files", "*.*")])
        if p:
            self.analysis_snapshot_var.set(p)

    def _analysis_start_process(self, cmd, button, running_text, success_title, success_text, on_success=None):
        button.state(["disabled"])
        self.analysis_state_var.set(running_text)
        self._analysis_log_append("COMMAND: " + " ".join(f'\"{x}\"' if " " in x else x for x in cmd))

        def worker():
            try:
                proc = subprocess.Popen(cmd, cwd=str(HERE.parent), stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, bufsize=1)
                assert proc.stdout is not None
                output = []
                for line in proc.stdout:
                    output.append(line.rstrip())
                    self._analysis_log_append(line)
                code = proc.wait()
                if code != 0:
                    raise RuntimeError(f"分析程序退出码 {code}")
                if on_success:
                    self.after(0, on_success, output)
                self.after(0, lambda: self.analysis_state_var.set("分析完成 / Completed"))
                self.after(0, lambda: messagebox.showinfo(success_title, success_text))
            except Exception as exc:
                self._analysis_log_append("ERROR: " + repr(exc))
                self.after(0, lambda: self.analysis_state_var.set("分析失败，请查看日志 / Failed"))
                self.after(0, lambda error=str(exc): messagebox.showerror("研究分析失败", error))
            finally:
                self.after(0, lambda: button.state(["!disabled"]))
        threading.Thread(target=worker, name="MagTouchResearchAnalysis", daemon=True).start()

    def _analysis_collect_snapshot(self):
        tool = HERE.parent / "tools" / "magtouch_collect_research_snapshot.py"
        if not tool.exists():
            messagebox.showerror("研究分析", f"未找到研究快照采集器：\n{tool}")
            return
        cmd = [sys.executable, "-u", str(tool), "--root", str(HERE.parent), "--out", str(HERE.parent.parent)]

        def choose_output(lines):
            for line in reversed(lines):
                prefix = "[DONE] Upload only this ZIP:"
                if line.startswith(prefix):
                    self.analysis_snapshot_var.set(line[len(prefix):].strip())
                    self.analysis_state_var.set("研究快照已生成，请点击第 2 步分析")
                    return
        self._analysis_start_process(cmd, self.analysis_collect_btn, "正在扫描当前工程 / Collecting snapshot...",
                                     "研究快照完成", "快照已生成并已自动填入路径。请点击“2. 分析研究快照”。", choose_output)

    def _analysis_run_snapshot(self):
        snapshot = self.analysis_snapshot_var.get().strip()
        if not snapshot or not Path(snapshot).is_file():
            messagebox.showwarning("研究分析", "请先扫描当前工程，或选择有效的 MagTouch_RESEARCH_SNAPSHOT_*.zip。")
            return
        tool = HERE.parent / "tools" / "analyze_magtouch_research_snapshot.py"
        if not tool.exists():
            messagebox.showerror("研究分析", f"未找到研究快照分析器：\n{tool}")
            return
        cmd = [sys.executable, "-u", str(tool), "--snapshot", snapshot]
        self._analysis_start_process(cmd, self.analysis_snapshot_run_btn, "正在整理研究快照 / Analyzing snapshot...",
                                     "研究快照整理完成", "已生成中英双语详细研究报告、表格速览、QC/特征/R15 对比表和缺口清单。请查看日志中的结果 ZIP。")

    def _analysis_use_selected_plan(self):
        try:
            selected = self._selected_data_path()
        except Exception:
            selected = None
        if not selected:
            messagebox.showwarning("研究分析", "请先在“数据管理/回放”页选择一个 Trial 或 Plan。")
            return
        p = Path(selected)
        if p.name.lower().startswith("trial_"):
            p = p.parent
        while p != p.parent and not p.name.lower().startswith("plan_"):
            p = p.parent
        if not p.name.lower().startswith("plan_"):
            messagebox.showwarning("研究分析", f"当前选择不是 Plan 文件夹：\n{selected}")
            return
        self.analysis_plan1_var.set(str(p))

    def _analysis_run(self):
        plan1 = self.analysis_plan1_var.get().strip()
        plan2 = self.analysis_plan2_var.get().strip()
        if not plan1 or not Path(plan1).is_dir():
            messagebox.showwarning("研究分析", "请先选择有效的 Plan 1 文件夹。")
            return
        if plan2 and not Path(plan2).is_dir():
            messagebox.showwarning("研究分析", "Plan 2 路径不存在，请清空或重新选择。")
            return
        tool = HERE.parent / "tools" / "magtouch_research_analysis.py"
        if not tool.exists():
            messagebox.showerror("研究分析", f"未找到分析程序：\n{tool}")
            return
        cmd = [sys.executable, "-u", str(tool), "--plan", plan1]
        if plan2:
            cmd += ["--plan", plan2]
        if self.analysis_out_var.get().strip():
            cmd += ["--out", self.analysis_out_var.get().strip()]
        if self.analysis_hash_var.get():
            cmd.append("--hash")
        self.analysis_run_btn.state(["disabled"])
        self.analysis_state_var.set("分析进行中 / Running...")
        self._analysis_log_append("COMMAND: " + " ".join(f'\"{x}\"' if " " in x else x for x in cmd))

        def worker():
            result_lines = []
            try:
                proc = subprocess.Popen(cmd, cwd=str(HERE.parent), stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True, bufsize=1)
                assert proc.stdout is not None
                for line in proc.stdout:
                    result_lines.append(line.rstrip())
                    self._analysis_log_append(line)
                code = proc.wait()
                if code != 0:
                    raise RuntimeError(f"分析程序退出码 {code}")
                self.after(0, lambda: self.analysis_state_var.set("分析完成 / Completed"))
                self.after(0, lambda: messagebox.showinfo("研究分析完成", "分析已完成。请查看日志中的结果目录和 ZIP。"))
            except Exception as exc:
                self._analysis_log_append("ERROR: " + repr(exc))
                self.after(0, lambda: self.analysis_state_var.set("分析失败，请查看日志 / Failed"))
                self.after(0, lambda: messagebox.showerror("研究分析失败", str(exc)))
            finally:
                self.after(0, lambda: self.analysis_run_btn.state(["!disabled"]))
        threading.Thread(target=worker, name="MagTouchResearchAnalysis", daemon=True).start()

    def _build_training_tab(self):
        """Integrated H1 export / feature build / Pilot training UI.

        Heavy work is executed in background threads or child processes while all
        Tk updates are marshalled back to the UI thread. PyTorch is intentionally
        NOT imported during application startup, so acquisition-only environments
        remain usable.
        """
        host=self.tab_train
        host.columnconfigure(0,weight=1); host.rowconfigure(0,weight=1)
        # R9: Export/Training is intentionally a long form. Put it inside a canvas
        # with a permanent right-side scrollbar instead of squeezing controls on a
        # notebook-height laptop display.
        train_canvas=tk.Canvas(host,highlightthickness=0,borderwidth=0)
        train_vsb=ttk.Scrollbar(host,orient="vertical",command=train_canvas.yview)
        train_canvas.configure(yscrollcommand=train_vsb.set)
        train_canvas.grid(row=0,column=0,sticky="nsew")
        train_vsb.grid(row=0,column=1,sticky="ns")
        tab=ttk.Frame(train_canvas)
        train_win=train_canvas.create_window((0,0),window=tab,anchor="nw")
        tab.bind("<Configure>",lambda _e:train_canvas.configure(scrollregion=train_canvas.bbox("all")))
        train_canvas.bind("<Configure>",lambda e:train_canvas.itemconfigure(train_win,width=e.width))
        def _train_wheel(event):
            if getattr(event,"delta",0):
                train_canvas.yview_scroll(int(-event.delta/120),"units")
            return "break"
        def _train_wheel_on(_e): train_canvas.bind_all("<MouseWheel>",_train_wheel)
        def _train_wheel_off(_e): train_canvas.unbind_all("<MouseWheel>")
        host.bind("<Enter>",_train_wheel_on); host.bind("<Leave>",_train_wheel_off)
        train_canvas.bind("<Enter>",_train_wheel_on); train_canvas.bind("<Leave>",_train_wheel_off)
        self.training_canvas=train_canvas
        self.training_scrollbar=train_vsb
        tab.columnconfigure(0,weight=1)

        training_cfg=self.settings.get("training",{})
        exp=ttk.LabelFrame(tab,text="① 数据导出 / Export（全部集中在本页面）")
        exp.grid(row=0,column=0,sticky="ew",padx=8,pady=(8,4)); exp.columnconfigure(1,weight=1)
        self.training_plan_var=tk.StringVar(value=str(training_cfg.get("plan_path","")))
        self.training_export_out_var=tk.StringVar(value=str(training_cfg.get("export_output_path","")))
        ttk.Label(exp,text="原始 Plan").grid(row=0,column=0,sticky="w",padx=5,pady=4)
        ttk.Entry(exp,textvariable=self.training_plan_var).grid(row=0,column=1,sticky="ew",padx=4,pady=4)
        ttk.Button(exp,text="选择 Plan",command=self.training_choose_plan).grid(row=0,column=2,padx=3,pady=4)
        ttk.Button(exp,text="使用数据页当前实验",command=self.training_use_selected_plan).grid(row=0,column=3,padx=3,pady=4)
        ttk.Button(exp,text="自动找最新 Plan",command=self.training_use_latest_plan).grid(row=0,column=4,padx=3,pady=4)
        ttk.Label(exp,text="输出目录").grid(row=1,column=0,sticky="w",padx=5,pady=4)
        ttk.Entry(exp,textvariable=self.training_export_out_var).grid(row=1,column=1,sticky="ew",padx=4,pady=4)
        ttk.Button(exp,text="选择输出",command=self.training_choose_export_out).grid(row=1,column=2,padx=3,pady=4)
        self.training_analysis_btn=ttk.Button(exp,text="轻量分析导出 / QC",command=self.analysis_export_integrated)
        self.training_analysis_btn.grid(row=1,column=3,padx=3,pady=4,sticky="ew")
        self.training_export_btn=ttk.Button(exp,text="完整训练导出 / TRAINING_EXPORT",command=self.training_export_integrated)
        self.training_export_btn.grid(row=1,column=4,padx=3,pady=4,sticky="ew")
        self.training_export_state_var=tk.StringVar(value="等待导出 / Idle")
        ttk.Label(exp,textvariable=self.training_export_state_var,foreground="#1b5e20",wraplength=1200,justify="left").grid(row=2,column=0,columnspan=5,sticky="w",padx=5,pady=(2,5))

        feat=ttk.LabelFrame(tab,text="② 构建 40 Hz / T=32 / F0–F5 特征")
        feat.grid(row=1,column=0,sticky="ew",padx=8,pady=4); feat.columnconfigure(1,weight=1)
        self.training_source_var=tk.StringVar(value=str(training_cfg.get("training_export_path","")))
        self.training_features_var=tk.StringVar(value=str(training_cfg.get("h1_features_path","")))
        ttk.Label(feat,text="TRAINING_EXPORT目录").grid(row=0,column=0,sticky="w",padx=5,pady=4)
        ttk.Entry(feat,textvariable=self.training_source_var).grid(row=0,column=1,sticky="ew",padx=4,pady=4)
        ttk.Button(feat,text="选择",command=self.training_choose_export_source).grid(row=0,column=2,padx=3,pady=4)
        ttk.Button(feat,text="生成 F0–F5 构建命令",command=self.training_build_features).grid(row=0,column=3,padx=3,pady=4)
        ttk.Label(feat,text="特征目录").grid(row=1,column=0,sticky="w",padx=5,pady=4)
        ttk.Entry(feat,textvariable=self.training_features_var).grid(row=1,column=1,sticky="ew",padx=4,pady=4)
        ttk.Button(feat,text="选择",command=self.training_choose_features).grid(row=1,column=2,padx=3,pady=4)
        ttk.Label(feat,text="每 Trial 独立置零；40 Hz 重采样；量程预警(85%)、Reliability近限(95%)、critical 与 wrap 分层；Train/Val/Test 不跨 Trial。",foreground="#555").grid(row=2,column=0,columnspan=4,sticky="w",padx=5,pady=(1,5))

        gpu=ttk.LabelFrame(tab,text="③ 当前项目 Python / GPU")
        gpu.grid(row=2,column=0,sticky="ew",padx=8,pady=4); gpu.columnconfigure(0,weight=1)
        self.training_gpu_state_var=tk.StringVar(value=f"Python: {sys.executable} | 尚未检测 PyTorch/CUDA")
        ttk.Label(gpu,textvariable=self.training_gpu_state_var,wraplength=1250,justify="left").grid(row=0,column=0,sticky="w",padx=5,pady=4)
        ttk.Button(gpu,text="检测 GPU 并自动推荐5070/笔记本参数",command=self.training_detect_gpu).grid(row=0,column=1,padx=4,pady=4)

        run=ttk.LabelFrame(tab,text="④ H1 Pilot 单GPU训练命令（R13：严格复现 + 轴级Reliability；只生成不启动）")
        run.grid(row=3,column=0,sticky="ew",padx=8,pady=4)
        labels=[("特征",0),("策略",2),("基线模型",4),("任务",6),("Epoch",8),("Batch",10),("GPU",12)]
        for txt,c in labels: ttk.Label(run,text=txt).grid(row=0,column=c,sticky="e",padx=(5,2),pady=4)
        self.training_feature_var=tk.StringVar(value=str(training_cfg.get("feature","F0")))
        self.training_policy_var=tk.StringVar(value=str(training_cfg.get("policy","raw15")))
        self.training_model_var=tk.StringVar(value=str(training_cfg.get("baseline_model","transformer")))
        self.training_task_var=tk.StringVar(value=str(training_cfg.get("task","mode5")))
        self.training_epochs_var=tk.IntVar(value=int(training_cfg.get("epochs",20)))
        self.training_batch_var=tk.IntVar(value=int(training_cfg.get("baseline_batch",48)))
        self.training_gpu_var=tk.IntVar(value=int(training_cfg.get("gpu",0)))
        ttk.Combobox(run,textvariable=self.training_feature_var,values=[f"F{i}" for i in range(6)],state="readonly",width=5).grid(row=0,column=1)
        ttk.Combobox(run,textvariable=self.training_policy_var,values=["raw15","clean14","mask15"],state="readonly",width=9).grid(row=0,column=3)
        ttk.Combobox(run,textvariable=self.training_model_var,values=["mlp","tcn","transformer"],state="readonly",width=12).grid(row=0,column=5)
        ttk.Combobox(run,textvariable=self.training_task_var,values=["mode5","z","dx","dy","x","y"],state="readonly",width=8).grid(row=0,column=7)
        ttk.Entry(run,textvariable=self.training_epochs_var,width=6).grid(row=0,column=9)
        ttk.Entry(run,textvariable=self.training_batch_var,width=6).grid(row=0,column=11)
        ttk.Entry(run,textvariable=self.training_gpu_var,width=4).grid(row=0,column=13)
        ttk.Button(run,text="生成基线命令",command=self.training_run_baseline).grid(row=0,column=14,padx=4)
        ttk.Button(run,text="生成 MagTouch-RA-HW5 命令",command=self.training_run_hw5).grid(row=0,column=15,padx=4)
        self.training_hw5_dmodel_var=tk.IntVar(value=int(training_cfg.get("d_model",128)))
        self.training_hw5_layers_var=tk.IntVar(value=int(training_cfg.get("layers",3)))
        self.training_hw5_batch_var=tk.IntVar(value=int(training_cfg.get("hw5_batch",24)))
        self.training_hw5_objective_var=tk.StringVar(value=str(training_cfg.get("h1_objective","mode_only")))
        self.training_deterministic_var=tk.BooleanVar(value=bool(training_cfg.get("deterministic",True)))
        self.training_seed_var=tk.IntVar(value=int(training_cfg.get("seed",20260910)))
        ttk.Label(run,text="HW5: d_model").grid(row=1,column=0,sticky="e",padx=(5,2),pady=4)
        ttk.Entry(run,textvariable=self.training_hw5_dmodel_var,width=6).grid(row=1,column=1)
        ttk.Label(run,text="layers").grid(row=1,column=2,sticky="e")
        ttk.Entry(run,textvariable=self.training_hw5_layers_var,width=5).grid(row=1,column=3)
        ttk.Label(run,text="batch").grid(row=1,column=4,sticky="e")
        ttk.Entry(run,textvariable=self.training_hw5_batch_var,width=6).grid(row=1,column=5)
        ttk.Label(run,text="objective").grid(row=1,column=6,sticky="e",padx=(8,2))
        ttk.Combobox(run,textvariable=self.training_hw5_objective_var,values=["mode_only","multitask"],state="readonly",width=12).grid(row=1,column=7)
        ttk.Checkbutton(run,text="严格固定seed",variable=self.training_deterministic_var).grid(row=1,column=8,columnspan=2,sticky="w",padx=8)
        ttk.Label(run,text="Seed").grid(row=1,column=10,sticky="e",padx=(4,2))
        ttk.Entry(run,textvariable=self.training_seed_var,width=10).grid(row=1,column=11)
        ttk.Label(run,text="mode_only=公平分类；multitask=分类+回归+轴级Q。",foreground="#555").grid(row=1,column=12,columnspan=4,sticky="w",padx=6)
        ttk.Label(run,text="最后生成的命令（可直接粘贴到独立 CMD；Ctrl+C 停止训练）：").grid(row=2,column=0,columnspan=16,sticky="w",padx=5,pady=(6,2))
        self.training_command_text=ScrolledText(run,height=5,font=("Consolas",9),wrap="word")
        self.training_command_text.grid(row=3,column=0,columnspan=16,sticky="ew",padx=5,pady=2)
        cmdbar=ttk.Frame(run);cmdbar.grid(row=4,column=0,columnspan=16,sticky="ew",padx=5,pady=(2,5))
        self.training_last_command_path_var=tk.StringVar(value="尚未生成命令")
        ttk.Button(cmdbar,text="复制最后命令",command=self.training_copy_last_command).pack(side="left",padx=(0,4))
        ttk.Button(cmdbar,text="打开命令文件夹",command=self.training_open_command_dir).pack(side="left",padx=4)
        ttk.Button(cmdbar,text="生成R13核心4组实验",command=self.training_generate_r13_core_matrix).pack(side="left",padx=4)
        ttk.Label(cmdbar,textvariable=self.training_last_command_path_var,foreground="#1b5e20",wraplength=900,justify="left").pack(side="left",padx=8,fill="x",expand=True)
        ttk.Label(run,text="安全策略：UI只生成命令和 .bat，不启动训练 Python。R13传统Pilot流程完整保留，供传统实验/历史基线对比。",foreground="#7a3e00",wraplength=1250,justify="left").grid(row=5,column=0,columnspan=16,sticky="w",padx=5,pady=(1,5))

        rel=ttk.LabelFrame(tab,text="⑤ R15 相对触觉训练：传统采集不变 → 高阈值确认/向前回溯对齐 → 历史ΔB ↔ 当前ΔXYZ")
        rel.grid(row=4,column=0,sticky="ew",padx=8,pady=4); rel.columnconfigure(1,weight=1)
        self.training_relative_var=tk.StringVar(value=str(training_cfg.get("h2_relative_path","")))
        self.training_relative_feature_var=tk.StringVar(value=str(training_cfg.get("h2_feature","R1")))
        self.training_relative_history_var=tk.StringVar(value=str(training_cfg.get("h2_history","64")))
        self.training_relative_use_q_var=tk.BooleanVar(value=bool(training_cfg.get("h2_use_q",True)))
        ttk.Label(rel,text="相对触觉数据目录").grid(row=0,column=0,sticky="w",padx=5,pady=4)
        ttk.Entry(rel,textvariable=self.training_relative_var).grid(row=0,column=1,sticky="ew",padx=4,pady=4)
        ttk.Button(rel,text="生成相对数据命令",command=self.training_build_relative).grid(row=0,column=2,padx=3,pady=4)
        ttk.Button(rel,text="选择相对数据",command=self.training_choose_relative).grid(row=0,column=3,padx=3,pady=4)
        ttk.Label(rel,text="特征 / 历史帧").grid(row=1,column=0,sticky="e",padx=(5,2),pady=4)
        sub=ttk.Frame(rel);sub.grid(row=1,column=1,sticky="w",padx=4)
        ttk.Combobox(sub,textvariable=self.training_relative_feature_var,values=["R0","R1","R2","R3"],state="readonly",width=6).pack(side="left")
        ttk.Combobox(sub,textvariable=self.training_relative_history_var,values=["0","16","32","64","128"],state="readonly",width=6).pack(side="left",padx=(8,0))
        ttk.Label(sub,text="0=仅当前帧；64≈1.6s@40Hz").pack(side="left",padx=(6,0))
        ttk.Checkbutton(rel,text="轴级Q仅用于量程/wrap可靠性",variable=self.training_relative_use_q_var).grid(row=1,column=1,sticky="e",padx=8)
        ttk.Button(rel,text="生成相对训练命令",command=self.training_run_relative).grid(row=1,column=2,padx=3,pady=4)
        ttk.Button(rel,text="生成R15特征核心4组",command=self.training_generate_r15_relative_matrix).grid(row=1,column=3,padx=3,pady=4)
        ttk.Button(rel,text="生成R15历史5组（R1+Q）",command=self.training_generate_r15_history_matrix).grid(row=2,column=3,padx=3,pady=4)
        ttk.Label(rel,text="R15主监督只有ΔXYZ：磁阈值只在数据集生成时做事件确认和向前回溯，不作为contact训练标签。接触前历史完整保留且ΔXYZ=0，模型自己从历史序列学习动作何时开始。历史消融0/16/32/64/128；严格的‘无历史’比较建议R0/R1，因为R2/R3本身含dΔB/dt。",foreground="#1b5e20",wraplength=1180,justify="left").grid(row=2,column=0,columnspan=3,sticky="w",padx=5,pady=(1,5))

        unified=ttk.LabelFrame(tab,text="⑥ R18 统一模型：指定 TRAINING_EXPORT + R15相对数据 + R18标签 → XYZ+dXYZ+mode5")
        unified.grid(row=5,column=0,sticky="ew",padx=8,pady=4); unified.columnconfigure(1,weight=1)
        self.training_r18_source_var=tk.StringVar(value=str(training_cfg.get("r18_training_export_path",training_cfg.get("training_export_path",""))))
        self.training_r18_relative_var=tk.StringVar(value=str(training_cfg.get("r18_relative_path",training_cfg.get("h2_relative_path",""))))
        r18_labels_value=str(training_cfg.get("r18_labels_path",""))
        if r18_labels_value.endswith("_UNIFIED_LABELS_R18"):
            r18_labels_value += "_1_ABSXYZ"
        self.training_r18_labels_var=tk.StringVar(value=r18_labels_value)
        self.training_r18_feature_var=tk.StringVar(value=str(training_cfg.get("r18_feature","R1")))
        self.training_r18_history_var=tk.StringVar(value=str(training_cfg.get("r18_history","64")))
        self.training_r18_endpoint_history_var=tk.StringVar(value=str(training_cfg.get("r18_endpoint_min_history","128")))
        self.training_r18_objective_var=tk.StringVar(value=str(training_cfg.get("r18_objective","unified")))
        self.training_r18_sample_policy_var=tk.StringVar(value=str(training_cfg.get("r18_sample_policy","common")))
        self.training_r18_use_q_var=tk.BooleanVar(value=bool(training_cfg.get("r18_use_q",True)))
        ttk.Label(unified,text="TRAINING_EXPORT目录").grid(row=0,column=0,sticky="e",padx=5,pady=3)
        ttk.Entry(unified,textvariable=self.training_r18_source_var).grid(row=0,column=1,sticky="ew",padx=5,pady=3)
        ttk.Button(unified,text="选择目录",command=lambda:self.training_choose_r18_directory(self.training_r18_source_var,"选择 R18 对应 TRAINING_EXPORT 目录")).grid(row=0,column=2,padx=4,pady=3)
        ttk.Label(unified,text="R15相对数据目录").grid(row=1,column=0,sticky="e",padx=5,pady=3)
        ttk.Entry(unified,textvariable=self.training_r18_relative_var).grid(row=1,column=1,sticky="ew",padx=5,pady=3)
        ttk.Button(unified,text="选择目录",command=lambda:self.training_choose_r18_directory(self.training_r18_relative_var,"选择 R18 使用的 R15 RELATIVE_TACTILE 目录")).grid(row=1,column=2,padx=4,pady=3)
        ttk.Label(unified,text="R18统一标签目录").grid(row=2,column=0,sticky="e",padx=5,pady=3)
        ttk.Entry(unified,textvariable=self.training_r18_labels_var).grid(row=2,column=1,sticky="ew",padx=5,pady=3)
        ttk.Button(unified,text="选择目录",command=lambda:self.training_choose_r18_directory(self.training_r18_labels_var,"选择 R18 UNIFIED_LABELS 目录")).grid(row=2,column=2,padx=4,pady=3)
        ttk.Button(unified,text="生成R18标签命令",command=self.training_build_r18_labels).grid(row=2,column=3,padx=4,pady=3)
        opts=ttk.Frame(unified);opts.grid(row=3,column=0,columnspan=4,sticky="w",padx=5,pady=3)
        ttk.Label(opts,text="Feature").pack(side="left")
        ttk.Combobox(opts,textvariable=self.training_r18_feature_var,values=["R0","R1","R2","R3"],state="readonly",width=6).pack(side="left",padx=(3,8))
        ttk.Label(opts,text="History").pack(side="left")
        ttk.Combobox(opts,textvariable=self.training_r18_history_var,values=["0","16","32","64","128"],state="readonly",width=6).pack(side="left",padx=(3,8))
        ttk.Label(opts,text="Endpoint min history").pack(side="left")
        ttk.Combobox(opts,textvariable=self.training_r18_endpoint_history_var,values=["0","16","32","64","128"],state="readonly",width=6).pack(side="left",padx=(3,8))
        ttk.Label(opts,text="Objective").pack(side="left")
        ttk.Combobox(opts,textvariable=self.training_r18_objective_var,values=["coordinate_only","unified"],state="readonly",width=14).pack(side="left",padx=(3,8))
        ttk.Label(opts,text="Sample policy").pack(side="left")
        ttk.Combobox(opts,textvariable=self.training_r18_sample_policy_var,values=["common","native"],state="readonly",width=9).pack(side="left",padx=(3,8))
        ttk.Checkbutton(opts,text="轴级Q",variable=self.training_r18_use_q_var).pack(side="left",padx=5)
        actions=ttk.Frame(unified);actions.grid(row=4,column=0,columnspan=4,sticky="w",padx=5,pady=(3,6))
        ttk.Button(actions,text="生成R18标签验证命令",command=self.training_validate_r18_labels).pack(side="left",padx=(0,5))
        ttk.Button(actions,text="生成R18统一训练命令",command=self.training_run_r18).pack(side="left",padx=5)
        ttk.Label(actions,text="所有目录均可独立指定并自动保存；UI只生成命令，不直接启动训练。",foreground="#1b5e20").pack(side="left",padx=8)

        logf=ttk.LabelFrame(tab,text="训练/导出日志（不会一闪而过）")
        logf.grid(row=6,column=0,sticky="nsew",padx=8,pady=(4,8)); logf.columnconfigure(0,weight=1); logf.rowconfigure(0,weight=1)
        self.training_log=ScrolledText(logf,height=14,font=("Consolas",9),wrap="word")
        self.training_log.grid(row=0,column=0,sticky="nsew")
        ttk.Button(logf,text="清空日志",command=lambda:self.training_log.delete("1.0","end")).grid(row=1,column=0,sticky="e",pady=3)

    def _training_log_append(self,msg):
        text=str(msg).rstrip("\r\n")
        def _do():
            if hasattr(self,"training_log"):
                self.training_log.insert("end",text+"\n"); self.training_log.see("end")
            self.log("[H1] "+text)
        if threading.current_thread() is threading.main_thread(): _do()
        else: self.after(0,_do)

    def training_choose_plan(self):
        p=filedialog.askdirectory(title="选择原始 Plan_... 文件夹",initialdir=self.data_manage_root_var.get() or self.data_root_var.get())
        if p:
            self.training_plan_var.set(p)
            if not self.training_export_out_var.get(): self.training_export_out_var.set(str(Path(p).parent))

    def training_use_selected_plan(self):
        """Use the Plan folder selected on Data & Replay page."""
        p=self._selected_data_path() if hasattr(self,"data_tree") else None
        if p is None:
            messagebox.showwarning("使用数据页当前实验","请先在“数据管理/回放”页选择一个 Plan 文件夹或其中的 Trial。")
            return
        q=Path(p)
        if re.match(r"^Trial_\d+",q.name,re.I):
            q=q.parent
        while q != q.parent and not q.name.startswith("Plan_"):
            q=q.parent
        if not q.name.startswith("Plan_"):
            messagebox.showwarning("使用数据页当前实验",f"当前选择不属于 Plan_* 实验文件夹：\n{p}")
            return
        self.training_plan_var.set(str(q)); self.training_export_out_var.set(str(q.parent))
        self._training_log_append(f"从数据页带入 Plan: {q}")

    def training_use_latest_plan(self):
        root=Path(self.data_manage_root_var.get() or self.data_root_var.get()).expanduser()
        plans=[]
        if root.exists():
            for p in root.rglob("Plan_*"):
                if p.is_dir() and list(p.rglob("Trial_*")):
                    try: plans.append((p.stat().st_mtime,p))
                    except Exception: pass
        if not plans:
            messagebox.showwarning("自动找最新 Plan","当前 Data Root 下未找到含 Trial_* 的 Plan_* 文件夹。")
            return
        p=max(plans,key=lambda x:x[0])[1]
        self.training_plan_var.set(str(p)); self.training_export_out_var.set(str(p.parent))
        self._training_log_append(f"自动选择最新 Plan: {p}")

    def training_choose_export_out(self):
        p=filedialog.askdirectory(title="选择 TRAINING_EXPORT 输出目录",initialdir=self.training_export_out_var.get() or self.data_root_var.get())
        if p:self.training_export_out_var.set(p)

    def training_choose_export_source(self):
        p=filedialog.askdirectory(title="选择 TRAINING_EXPORT 解压/导出目录",initialdir=self.training_source_var.get() or self.training_export_out_var.get() or self.data_root_var.get())
        if p:self.training_source_var.set(p)

    def training_choose_features(self):
        p=filedialog.askdirectory(title="选择 H1 F0-F5 特征数据目录",initialdir=self.training_features_var.get() or self.training_source_var.get() or self.data_root_var.get())
        if p:self.training_features_var.set(p)

    def training_choose_r18_directory(self,variable,title):
        initial=variable.get() or self.training_source_var.get() or self.data_root_var.get() or str(HERE.parent)
        p=filedialog.askdirectory(title=title,initialdir=initial)
        if p:variable.set(p)

    def training_export_from_data_tab(self):
        # One click from Data tab: use selected Trial's parent Plan when possible.
        plan=None
        try:
            sel=self.data_tree.selection()
            if sel:
                td=self.data_trial_paths.get(sel[0])
                if td:
                    q=Path(td).parent
                    if q.name.startswith("Plan_"): plan=q
        except Exception: pass
        if plan is not None:
            self.training_plan_var.set(str(plan)); self.training_export_out_var.set(str(plan.parent))
        self.nb.select(self.tab_train)
        if plan is None and not self.training_plan_var.get(): self.training_use_latest_plan()
        if self.training_plan_var.get(): self.training_export_integrated()

    def _load_analysis_export_module(self):
        import importlib.util
        tool=HERE.parent/"tools"/"export_plan_for_analysis.py"
        if not tool.exists(): raise FileNotFoundError(f"未找到轻量分析导出工具: {tool}")
        spec=importlib.util.spec_from_file_location("magtouch_export_analysis_runtime",tool)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载分析导出模块: {tool}")
        mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        return mod

    def analysis_export_integrated(self):
        """Run lightweight QC export inside this page; no extra exporter window/bat."""
        if getattr(self,"_analysis_export_running",False):
            messagebox.showinfo("轻量分析导出","已有轻量分析导出任务正在运行。")
            return
        try:
            plan=Path(self.training_plan_var.get()).expanduser().resolve()
            if not plan.exists(): raise ValueError("请先选择有效的原始 Plan 文件夹。")
            out=Path(self.training_export_out_var.get() or plan.parent).expanduser().resolve(); out.mkdir(parents=True,exist_ok=True)
        except Exception as e:
            messagebox.showerror("轻量分析导出",str(e)); return
        self._analysis_export_running=True; self.training_analysis_btn.state(["disabled"])
        self.training_export_state_var.set("正在生成轻量分析包；视频不会复制……")
        self._training_log_append(f"ANALYSIS_EXPORT start | Plan={plan} | Out={out}")
        def worker():
            try:
                mod=self._load_analysis_export_module()
                result=mod.export_plan(plan,out,log=self._training_log_append)
                def done():
                    self._analysis_export_running=False; self.training_analysis_btn.state(["!disabled"])
                    self.training_export_state_var.set(f"轻量分析导出完成：{result}")
                    messagebox.showinfo("轻量分析导出完成",f"QC ZIP 已生成：\n{result}")
                self.after(0,done)
            except Exception as e:
                self._training_log_append(f"ANALYSIS_EXPORT ERROR: {e!r}")
                def fail(error=repr(e)):
                    self._analysis_export_running=False; self.training_analysis_btn.state(["!disabled"]); self.training_export_state_var.set("轻量分析导出失败，请看下方日志")
                    messagebox.showerror("轻量分析导出失败",error)
                self.after(0,fail)
        threading.Thread(target=worker,name="MagTouchAnalysisExport",daemon=True).start()

    def _load_training_export_module(self):
        import importlib.util
        tool=HERE.parent/"tools"/"export_plan_for_training.py"
        if not tool.exists(): raise FileNotFoundError(f"未找到训练导出工具: {tool}")
        spec=importlib.util.spec_from_file_location("magtouch_export_training_runtime",tool)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载训练导出模块: {tool}")
        mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        return mod

    def training_export_integrated(self):
        if getattr(self,"_training_export_running",False):
            messagebox.showinfo("TRAINING_EXPORT","已有导出任务正在运行。")
            return
        try:
            plan=Path(self.training_plan_var.get()).expanduser().resolve()
            if not plan.exists(): raise ValueError("请先选择有效的原始 Plan 文件夹。")
            out=Path(self.training_export_out_var.get() or plan.parent).expanduser().resolve()
            out.mkdir(parents=True,exist_ok=True)
        except Exception as e:
            messagebox.showerror("TRAINING_EXPORT",str(e)); return
        self._training_export_running=True; self.training_export_btn.state(["disabled"])
        self.training_export_state_var.set("正在导出完整 magnetic_raw.csv；视频不会复制……")
        self._training_log_append(f"TRAINING_EXPORT start | Plan={plan} | Out={out}")
        def worker():
            try:
                mod=self._load_training_export_module()
                result=mod.export_plan(plan,out,True,log=self._training_log_append)
                folder=Path(result).with_suffix("") if str(result).lower().endswith(".zip") else Path(result)
                def done():
                    self._training_export_running=False; self.training_export_btn.state(["!disabled"])
                    self.training_export_state_var.set(f"完成：{result}")
                    self.training_source_var.set(str(folder))
                    messagebox.showinfo("TRAINING_EXPORT完成",f"训练数据导出完成。\n\nZIP:\n{result}\n\n特征构建目录已自动指向：\n{folder}")
                self.after(0,done)
            except Exception as e:
                self._training_log_append(f"TRAINING_EXPORT ERROR: {e!r}")
                def fail(error=repr(e)):
                    self._training_export_running=False; self.training_export_btn.state(["!disabled"]); self.training_export_state_var.set("导出失败，请看下方日志")
                    messagebox.showerror("TRAINING_EXPORT失败",error)
                self.after(0,fail)
        threading.Thread(target=worker,name="MagTouchTrainingExport",daemon=True).start()

    def _system_memory_status(self):
        """Return (used_percent, available_GB). Works on Windows without psutil."""
        try:
            import psutil
            vm=psutil.virtual_memory()
            return float(vm.percent), float(vm.available)/(1024**3)
        except Exception:
            pass
        if os.name=="nt":
            try:
                import ctypes
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_=[("dwLength",ctypes.c_ulong),("dwMemoryLoad",ctypes.c_ulong),("ullTotalPhys",ctypes.c_ulonglong),("ullAvailPhys",ctypes.c_ulonglong),("ullTotalPageFile",ctypes.c_ulonglong),("ullAvailPageFile",ctypes.c_ulonglong),("ullTotalVirtual",ctypes.c_ulonglong),("ullAvailVirtual",ctypes.c_ulonglong),("sullAvailExtendedVirtual",ctypes.c_ulonglong)]
                st=MEMORYSTATUSEX();st.dwLength=ctypes.sizeof(st)
                if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                    return float(st.dwMemoryLoad), float(st.ullAvailPhys)/(1024**3)
            except Exception:
                pass
        return None,None

    def _training_memory_preflight(self):
        used,avail=self._system_memory_status()
        if used is None or avail is None:return True
        self._training_log_append(f"系统内存预检：used={used:.1f}% | available={avail:.2f} GB")
        if used>=90.0 or avail<1.5:
            return messagebox.askyesno("系统内存很紧张",f"当前系统内存已使用 {used:.1f}% ，可用约 {avail:.2f} GB。\n\n这可能主要来自浏览器、PyCharm或其他程序，并不一定是 MagTouch。\n建议先关闭不需要的软件再训练。\n\n仍然继续吗？")
        return True

    def _run_training_subprocess(self,cmd,label,on_success=None):
        current_proc = self._training_proc
        if current_proc is not None and current_proc.poll() is None:
            messagebox.showwarning("训练正在运行",f"当前已有任务 PID={current_proc.pid} 在运行。\n请先停止当前训练。")
            return
        if label not in ("构建 F0-F5",) and not self._training_memory_preflight():
            return
        # -u + PYTHONUNBUFFERED ensures epoch/cache logs appear in the UI immediately.
        cmd=[str(x) for x in cmd]
        if cmd and Path(cmd[0]).name.lower().startswith("python") and (len(cmd)<2 or cmd[1]!="-u"):
            cmd.insert(1,"-u")
        self._training_log_append(label+"\nCMD: "+subprocess.list2cmdline(cmd))
        def worker():
            proc=None
            try:
                flags=0
                if os.name=="nt":
                    if hasattr(subprocess,"CREATE_NO_WINDOW"):flags|=subprocess.CREATE_NO_WINDOW
                    if hasattr(subprocess,"CREATE_NEW_PROCESS_GROUP"):flags|=subprocess.CREATE_NEW_PROCESS_GROUP
                env=os.environ.copy();env["PYTHONUNBUFFERED"]="1";env["PYTHONIOENCODING"]="utf-8"
                proc=subprocess.Popen(cmd,cwd=str(HERE.parent),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding="utf-8",errors="replace",bufsize=1,creationflags=flags,env=env)
                self._training_proc=proc;self._training_proc_label=label
                self._training_log_append(f"[PROCESS] {label} started | PID={proc.pid}")
                assert proc.stdout is not None
                for line in iter(proc.stdout.readline,''):
                    if line:self._training_log_append(line)
                rc=proc.wait()
                self._training_log_append(f"{label} finished, returncode={rc}")
                if rc==0 and on_success:self.after(0,on_success)
                elif rc!=0 and not getattr(self,"_training_stop_requested",False):
                    self.after(0,lambda:messagebox.showerror(label,f"运行失败，returncode={rc}\n请查看训练日志。"))
            except Exception as e:
                self._training_log_append(f"{label} ERROR: {e!r}")
                if not getattr(self,"_training_stop_requested",False):self.after(0,lambda err=repr(e), title=label:messagebox.showerror(title,err))
            finally:
                if getattr(self,"_training_proc",None) is proc:self._training_proc=None
                self._training_stop_requested=False
        threading.Thread(target=worker,name="MagTouchTrainingProcess",daemon=True).start()

    def training_stop_current(self):
        proc=getattr(self,"_training_proc",None)
        if proc is None or proc.poll() is not None:
            self._training_log_append("当前没有正在运行的训练/特征子进程。")
            return
        pid=proc.pid;self._training_stop_requested=True
        self._training_log_append(f"[STOP] 请求终止整个训练进程树 PID={pid} …")
        def killer():
            try:
                if os.name=="nt":
                    flags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess,"CREATE_NO_WINDOW") else 0
                    r=subprocess.run(["taskkill","/PID",str(pid),"/T","/F"],capture_output=True,text=True,errors="replace",creationflags=flags,timeout=15)
                    self._training_log_append(f"[STOP] taskkill returncode={r.returncode} | {(r.stdout+r.stderr).strip()}")
                else:
                    try:
                        os.killpg(os.getpgid(pid),15)
                    except Exception:
                        proc.terminate()
                try:proc.wait(timeout=5)
                except Exception:
                    try:proc.kill()
                    except Exception:pass
            except Exception as e:
                self._training_log_append(f"[STOP] ERROR: {e!r}")
                self.after(0,lambda:messagebox.showerror("停止训练",f"自动停止失败。\nPID={pid}\n\n请用管理员 CMD：\ntaskkill /PID {pid} /T /F"))
        threading.Thread(target=killer,name="MagTouchTrainingKiller",daemon=True).start()

    def _normalize_external_python_cmd(self, cmd):
        """Return a visible/unbuffered command line without launching it."""
        out=[str(x) for x in cmd]
        if out and Path(out[0]).name.lower().startswith("python") and (len(out)<2 or out[1]!="-u"):
            out.insert(1,"-u")
        return out

    def _training_command_dir(self):
        d=HERE.parent/"generated_commands"
        d.mkdir(parents=True,exist_ok=True)
        return d

    def _write_external_training_command(self, cmd, label, slug):
        """R13 safety model: generate/copy/save a command, never start training from the UI."""
        used,avail=self._system_memory_status()
        if used is not None:
            self._training_log_append(f"[COMMAND SNAPSHOT] 当前系统内存 used={used:.1f}% | available={avail:.2f} GB；这里只生成命令，不启动训练。")
        cmd=self._normalize_external_python_cmd(cmd)
        cmdline=subprocess.list2cmdline(cmd)
        root=HERE.parent.resolve()
        stamp=datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_slug=re.sub(r"[^A-Za-z0-9_.-]+","_",str(slug)).strip("_") or "H1_COMMAND"
        command_dir=self._training_command_dir()
        bat_path=command_dir/f"{stamp}_{safe_slug}.bat"
        last_path=command_dir/"RUN_LAST_H1_COMMAND.bat"
        bat=(
            "@echo off\r\n"
            "setlocal\r\n"
            "chcp 65001 >nul\r\n"
            f"title MagTouch H1 R15 - {safe_slug}\r\n"
            "set PYTHONUTF8=1\r\n"
            "set PYTHONIOENCODING=utf-8\r\n"
            f"cd /d \"{root}\"\r\n"
            f"echo [MagTouch R15] {label}\r\n"
            f"echo [Python] {sys.executable}\r\n"
            "echo [STOP] Press Ctrl+C in this console to stop training.\r\n"
            "echo.\r\n"
            f"{cmdline}\r\n"
            "set RC=%ERRORLEVEL%\r\n"
            "echo.\r\n"
            "echo [DONE] returncode=%RC%\r\n"
            "echo Press any key to close this window.\r\n"
            "pause >nul\r\n"
            "exit /b %RC%\r\n"
        )
        bat_path.write_text(bat,encoding="utf-8-sig")
        last_path.write_text(bat,encoding="utf-8-sig")
        self.training_command_text.delete("1.0","end")
        self.training_command_text.insert("1.0",cmdline)
        self.training_last_command_path_var.set(str(bat_path))
        try:
            self.clipboard_clear();self.clipboard_append(cmdline);self.update_idletasks()
        except Exception:
            pass
        self._training_log_append(f"[R15 COMMAND ONLY] {label}\nCMD: {cmdline}\nBAT: {bat_path}\n已复制到剪贴板；UI未启动任何训练Python进程。")
        messagebox.showinfo("训练命令已生成",f"已生成并复制命令，但尚未启动训练。\n\nBAT：\n{bat_path}\n\n可在独立CMD粘贴命令运行，或双击BAT。\n停止训练：在该控制台按 Ctrl+C。")
        return bat_path

    def training_copy_last_command(self):
        if not hasattr(self,"training_command_text"):
            return
        text=self.training_command_text.get("1.0","end").strip()
        if not text:
            messagebox.showinfo("复制命令","尚未生成训练命令。")
            return
        self.clipboard_clear();self.clipboard_append(text);self.update_idletasks()
        self._training_log_append("最后训练命令已复制到剪贴板。")

    def training_open_command_dir(self):
        d=self._training_command_dir()
        try:
            if os.name=="nt": os.startfile(str(d))
            else: subprocess.Popen(["xdg-open",str(d)])
        except Exception as e:
            messagebox.showerror("打开命令文件夹",str(e))

    def training_build_features(self):
        src=Path(self.training_source_var.get()).expanduser()
        if not src.exists():
            messagebox.showerror("构建F0-F5","请选择 TRAINING_EXPORT 目录（不是 ANALYSIS_EXPORT）。"); return
        out=src.parent/(src.name+"_H1_FEATURES")
        self.training_features_var.set(str(out))
        cmd=[sys.executable,str(HERE.parent/"training"/"build_h1_dataset.py"),str(src),"--out",str(out),"--hz","40"]
        self._write_external_training_command(cmd,"构建 F0-F5",f"BUILD_F0_F5_{src.name}")

    def training_detect_gpu(self):
        code=("import sys; print('Python:',sys.executable); "
              "import torch; print('PyTorch:',torch.__version__); print('CUDA available:',torch.cuda.is_available()); "
              "print('CUDA build:',torch.version.cuda); "
              "[(print(f'GPU {i}: {torch.cuda.get_device_name(i)} | VRAM={torch.cuda.get_device_properties(i).total_memory/1024**3:.2f} GB')) for i in range(torch.cuda.device_count())]")
        cmd=[sys.executable,"-c",code]
        self._training_log_append("GPU检测中……")
        def worker():
            try:
                flags=subprocess.CREATE_NO_WINDOW if os.name=="nt" and hasattr(subprocess,"CREATE_NO_WINDOW") else 0
                r=subprocess.run(cmd,cwd=str(HERE.parent),capture_output=True,text=True,encoding="utf-8",errors="replace",creationflags=flags)
                text=(r.stdout+"\n"+r.stderr).strip(); self._training_log_append(text)
                # Query VRAM separately in a tiny JSON-ish line for robust parameter recommendation.
                q="import torch; print(0 if not torch.cuda.is_available() else int(torch.cuda.get_device_properties(0).total_memory))"
                rr=subprocess.run([sys.executable,"-c",q],capture_output=True,text=True,encoding="utf-8",errors="replace",creationflags=flags)
                try:vram=int(rr.stdout.strip())/1024**3
                except Exception:vram=0.0
                def done():
                    self.training_gpu_state_var.set(text.replace("\n"," | ") if text else "GPU检测无输出")
                    if vram>0:
                        if vram<=8.5: b,hw,dm,ly=48,24,128,3
                        elif vram<=12.5: b,hw,dm,ly=64,32,128,3
                        else: b,hw,dm,ly=96,48,160,4
                        self.training_batch_var.set(b);self.training_hw5_batch_var.set(hw);self.training_hw5_dmodel_var.set(dm);self.training_hw5_layers_var.set(ly)
                        self._training_log_append(f"按 GPU0 实际显存 {vram:.2f} GB 推荐：baseline batch={b}; HW5 batch={hw}, d_model={dm}, layers={ly}")
                self.after(0,done)
            except Exception as e:
                self._training_log_append(f"GPU检测失败: {e!r}"); self.after(0,lambda err=repr(e):messagebox.showerror("GPU检测",err))
        threading.Thread(target=worker,name="MagTouchGPUCheck",daemon=True).start()

    def _features_root_for_training(self):
        p=Path(self.training_features_var.get()).expanduser()
        if not p.exists() or not (p/"splits_stratified_trial.json").exists():
            raise ValueError("请先构建或选择有效的 H1_FEATURES 目录。")
        return p

    def training_run_baseline(self):
        try:root=self._features_root_for_training()
        except Exception as e:messagebox.showerror("生成基线命令",str(e));return
        cmd=[sys.executable,str(HERE.parent/"training"/"train_h1_baselines.py"),str(root),"--feature",self.training_feature_var.get(),"--policy",self.training_policy_var.get(),"--model",self.training_model_var.get(),"--task",self.training_task_var.get(),"--epochs",str(self.training_epochs_var.get()),"--batch",str(self.training_batch_var.get()),"--gpu",str(self.training_gpu_var.get()),"--seed",str(self.training_seed_var.get())]
        cmd.append("--deterministic" if self.training_deterministic_var.get() else "--no-deterministic")
        slug=f"BASELINE_{self.training_model_var.get()}_{self.training_feature_var.get()}_{self.training_policy_var.get()}_{self.training_task_var.get()}"
        self._write_external_training_command(cmd,"H1 baseline training",slug)

    def training_run_hw5(self):
        try:root=self._features_root_for_training()
        except Exception as e:messagebox.showerror("生成HW5命令",str(e));return
        cmd=[sys.executable,str(HERE.parent/"training"/"train_magtouch_ra_hw5.py"),str(root),"--feature",self.training_feature_var.get(),"--policy",self.training_policy_var.get(),"--objective",self.training_hw5_objective_var.get(),"--epochs",str(self.training_epochs_var.get()),"--batch",str(self.training_hw5_batch_var.get()),"--gpu",str(self.training_gpu_var.get()),"--d_model",str(self.training_hw5_dmodel_var.get()),"--layers",str(self.training_hw5_layers_var.get()),"--seed",str(self.training_seed_var.get())]
        cmd.append("--deterministic" if self.training_deterministic_var.get() else "--no-deterministic")
        slug=f"HW5_{self.training_feature_var.get()}_{self.training_policy_var.get()}_{self.training_hw5_objective_var.get()}_dm{self.training_hw5_dmodel_var.get()}_L{self.training_hw5_layers_var.get()}"
        self._write_external_training_command(cmd,"MagTouch-RA-HW5 training",slug)

    def training_generate_r13_core_matrix(self):
        """Generate, but do not launch, the fixed R13 four-run Pilot matrix."""
        try:
            root=self._features_root_for_training()
        except Exception as e:
            messagebox.showerror("生成R13核心实验",str(e));return
        py=sys.executable
        tr=HERE.parent/"training"
        seed=str(self.training_seed_var.get())
        common_det=["--deterministic"] if self.training_deterministic_var.get() else ["--no-deterministic"]
        cmds=[
            [py,str(tr/"train_h1_baselines.py"),str(root),"--feature","F4","--policy","raw15","--model","transformer","--task","mode5","--epochs",str(self.training_epochs_var.get()),"--batch",str(self.training_batch_var.get()),"--gpu",str(self.training_gpu_var.get()),"--seed",seed]+common_det,
            [py,str(tr/"train_magtouch_ra_hw5.py"),str(root),"--feature","F4","--policy","raw15","--objective","mode_only","--epochs",str(self.training_epochs_var.get()),"--batch",str(self.training_hw5_batch_var.get()),"--gpu",str(self.training_gpu_var.get()),"--d_model",str(self.training_hw5_dmodel_var.get()),"--layers",str(self.training_hw5_layers_var.get()),"--seed",seed]+common_det,
            [py,str(tr/"train_magtouch_ra_hw5.py"),str(root),"--feature","F4","--policy","raw15","--objective","multitask","--epochs",str(self.training_epochs_var.get()),"--batch",str(self.training_hw5_batch_var.get()),"--gpu",str(self.training_gpu_var.get()),"--d_model",str(self.training_hw5_dmodel_var.get()),"--layers",str(self.training_hw5_layers_var.get()),"--seed",seed]+common_det,
            [py,str(tr/"train_magtouch_ra_hw5.py"),str(root),"--feature","F5","--policy","mask15","--objective","multitask","--epochs",str(self.training_epochs_var.get()),"--batch",str(self.training_hw5_batch_var.get()),"--gpu",str(self.training_gpu_var.get()),"--d_model",str(self.training_hw5_dmodel_var.get()),"--layers",str(self.training_hw5_layers_var.get()),"--seed",seed]+common_det,
        ]
        stamp=datetime.now().strftime("%Y%m%d_%H%M%S")
        d=self._training_command_dir();p=d/f"{stamp}_R13_CORE_4RUN_MATRIX.bat"
        lines=["@echo off","setlocal","chcp 65001 >nul","title MagTouch H1 R13 - CORE 4 RUN MATRIX","set PYTHONUTF8=1","set PYTHONIOENCODING=utf-8","set CUBLAS_WORKSPACE_CONFIG=:4096:8",f'cd /d "{HERE.parent.resolve()}"',"echo [R13] Core 4-run Pilot matrix","echo [STOP] Ctrl+C stops the current command.","echo."]
        names=["1/4 Transformer F4 raw15 mode5","2/4 HW5 F4 raw15 mode_only","3/4 HW5 F4 raw15 multitask","4/4 HW5 F5 mask15 multitask"]
        for name,cmd in zip(names,cmds):
            lines += [f"echo ===== {name} =====",subprocess.list2cmdline(self._normalize_external_python_cmd(cmd)),"if errorlevel 1 goto :fail","echo."]
        lines += ["echo [DONE] R13 core 4-run matrix completed.","pause","exit /b 0",":fail","echo [FAILED] One experiment returned non-zero. Matrix stopped.","pause","exit /b 1"]
        p.write_text("\r\n".join(lines)+"\r\n",encoding="utf-8-sig")
        self.training_last_command_path_var.set(str(p));self.training_command_text.delete("1.0","end");self.training_command_text.insert("1.0",str(p))
        self._training_log_append(f"[R13 MATRIX GENERATED] {p}\n只生成，不启动。4组按顺序串行运行，任一报错即停止。")
        messagebox.showinfo("R13核心4组实验已生成",f"只生成，未启动。\n\n{p}\n\n双击BAT后4组实验串行执行；Ctrl+C可停止当前训练。")

    def training_choose_relative(self):
        p=filedialog.askdirectory(initialdir=self.training_relative_var.get() or self.training_source_var.get() or str(HERE.parent))
        if p:self.training_relative_var.set(p)

    def _relative_root_for_training(self):
        p=Path(self.training_relative_var.get()).expanduser()
        if not p.exists() or not (p/"relative_manifest.json").exists() or not (p/"splits_trial.json").exists():
            raise ValueError("请先从 TRAINING_EXPORT 构建或选择有效的 R15 RELATIVE_TACTILE 数据目录。")
        return p

    def training_build_relative(self):
        src=Path(self.training_source_var.get()).expanduser()
        if not src.exists():
            messagebox.showerror("构建R15相对触觉数据","请选择传统 TRAINING_EXPORT 目录。传统采集、传统分析和F0-F5流程全部保留。")
            return
        out=src.parent/(src.name+"_RELATIVE_TACTILE_R15")
        self.training_relative_var.set(str(out))
        cmd=[sys.executable,str(HERE.parent/"training"/"build_relative_tactile_dataset.py"),str(src),"--out",str(out),"--hz","40","--seed",str(self.training_seed_var.get()),"--episode-pre-seconds","4.0"]
        self._write_external_training_command(cmd,"R15 build history-ready relative tactile dataset",f"BUILD_R15_RELATIVE_{src.name}")

    def training_run_relative(self):
        try:root=self._relative_root_for_training()
        except Exception as e:messagebox.showerror("生成R15相对训练命令",str(e));return
        history=str(self.training_relative_history_var.get())
        cmd=[sys.executable,str(HERE.parent/"training"/"train_relative_tactile_hw5.py"),str(root),"--feature",self.training_relative_feature_var.get(),"--history",history,"--epochs",str(self.training_epochs_var.get()),"--batch",str(self.training_hw5_batch_var.get()),"--gpu",str(self.training_gpu_var.get()),"--d_model",str(self.training_hw5_dmodel_var.get()),"--layers",str(self.training_hw5_layers_var.get()),"--seed",str(self.training_seed_var.get())]
        cmd.append("--use-q" if self.training_relative_use_q_var.get() else "--no-use-q");cmd.append("--deterministic" if self.training_deterministic_var.get() else "--no-deterministic")
        slug=f"RELATIVE_{self.training_relative_feature_var.get()}_H{history}_Q{int(self.training_relative_use_q_var.get())}_dm{self.training_hw5_dmodel_var.get()}_L{self.training_hw5_layers_var.get()}"
        self._write_external_training_command(cmd,"R15 history-aware relative tactile training",slug)

    def _write_r15_matrix(self,title,filename,specs):
        try:root=self._relative_root_for_training()
        except Exception as e:messagebox.showerror(title,str(e));return
        py=sys.executable;tr=HERE.parent/"training";seed=str(self.training_seed_var.get());det=["--deterministic"] if self.training_deterministic_var.get() else ["--no-deterministic"]
        base=["--epochs",str(self.training_epochs_var.get()),"--batch",str(self.training_hw5_batch_var.get()),"--gpu",str(self.training_gpu_var.get()),"--d_model",str(self.training_hw5_dmodel_var.get()),"--layers",str(self.training_hw5_layers_var.get()),"--seed",seed]+det
        cmds=[]
        for spec in specs:
            name,feat,history,qflag,*extra=spec
            cmds.append([py,str(tr/"train_relative_tactile_hw5.py"),str(root),"--feature",feat,"--history",str(history),qflag]+extra+base)
        stamp=datetime.now().strftime("%Y%m%d_%H%M%S");d=self._training_command_dir();p=d/f"{stamp}_{filename}.bat"
        lines=["@echo off","setlocal","chcp 65001 >nul",f"title {title}","set PYTHONUTF8=1","set PYTHONIOENCODING=utf-8","set CUBLAS_WORKSPACE_CONFIG=:4096:8",f'cd /d "{HERE.parent.resolve()}"',"echo [R15] Main target: magnetic-change history to relative dXYZ. No threshold-derived contact target.","echo [STOP] Ctrl+C stops the current command.","echo."]
        for spec,cmd in zip(specs,cmds):
            name=spec[0];lines += [f"echo ===== {name} =====",subprocess.list2cmdline(self._normalize_external_python_cmd(cmd)),"if errorlevel 1 goto :fail","echo."]
        lines += ["echo [DONE] R15 matrix completed.","pause","exit /b 0",":fail","echo [FAILED] One experiment returned non-zero. Matrix stopped.","pause","exit /b 1"]
        p.write_text("\r\n".join(lines)+"\r\n",encoding="utf-8-sig");self.training_last_command_path_var.set(str(p));self.training_command_text.delete("1.0","end");self.training_command_text.insert("1.0",str(p));self._training_log_append(f"[R15 MATRIX GENERATED] {p}\n只生成，不启动。")
        messagebox.showinfo(title,f"只生成，未启动。\n\n{p}\n\n双击BAT后串行执行；Ctrl+C可停止当前训练。")

    def training_generate_r15_relative_matrix(self):
        h=int(self.training_relative_history_var.get())
        specs=[("1/4 R0 ΔB only / no-Q", "R0",h,"--no-use-q"),("2/4 R1 ΔB+spatial / axis-Q", "R1",h,"--use-q"),("3/4 R2 + derivative+common / axis-Q", "R2",h,"--use-q"),("4/4 R3 + explicit Q / axis-Q", "R3",h,"--use-q")]
        self._write_r15_matrix("MagTouch R15 - RELATIVE FEATURE 4 RUN MATRIX","R15_RELATIVE_FEATURE_4RUN_MATRIX",specs)

    def training_generate_r15_history_matrix(self):
        # R1 intentionally has no precomputed temporal derivative. This isolates
        # sequence-history length itself; Q remains fixed for hardware range/wrap robustness.
        specs=[(f"{i+1}/5 R1+Q history={h}","R1",h,"--use-q","--endpoint-min-history","128") for i,h in enumerate((0,16,32,64,128))]
        self._write_r15_matrix("MagTouch R15 - HISTORY 0 16 32 64 128","R15_HISTORY_5RUN_R1_Q",specs)

    def _r18_selected_paths(self,require_labels=False):
        source=Path(self.training_r18_source_var.get()).expanduser()
        relative=Path(self.training_r18_relative_var.get()).expanduser()
        labels=Path(self.training_r18_labels_var.get()).expanduser()
        if not source.exists() or not (source/"trials").is_dir():
            raise ValueError("请选择与本次模型对应、且包含 trials/ 的 TRAINING_EXPORT 目录。")
        if not relative.exists() or not (relative/"relative_manifest.json").is_file() or not (relative/"splits_trial.json").is_file():
            raise ValueError("请选择有效的 R15 RELATIVE_TACTILE 数据目录。")
        if require_labels and (not labels.exists() or not (labels/"unified_manifest.json").is_file()):
            raise ValueError("请选择有效的 R18 UNIFIED_LABELS 目录，或先生成 R18 标签。")
        if require_labels:
            try:
                label_manifest=json.loads((labels/"unified_manifest.json").read_text(encoding="utf-8-sig"))
            except Exception as exc:
                raise ValueError(f"无法读取 R18 标签清单：{exc}") from exc
            if label_manifest.get("schema")!="MagTouch-UNIFIED-LABELS-R18-1.1":
                raise ValueError(
                    "该标签不是修正后的 R18-1.1。旧 R18-1.0 会把每个 Trial 的落点减掉，"
                    "导致绝对 XY 塌缩；请重新生成到新的 R18_1_ABSXYZ 目录。"
                )
        return source,relative,labels

    def training_build_r18_labels(self):
        try:
            source,relative,labels=self._r18_selected_paths(require_labels=False)
            old_schema=False
            if (labels/"unified_manifest.json").is_file():
                try:
                    old_schema=json.loads((labels/"unified_manifest.json").read_text(encoding="utf-8-sig")).get("schema")!="MagTouch-UNIFIED-LABELS-R18-1.1"
                except Exception:
                    old_schema=True
            if not str(labels).strip() or str(labels)=="." or labels.name.endswith("_UNIFIED_LABELS_R18") or old_schema:
                labels=source.parent/(source.name+"_UNIFIED_LABELS_R18_1_ABSXYZ")
                self.training_r18_labels_var.set(str(labels))
        except Exception as e:
            messagebox.showerror("生成R18标签命令",str(e));return
        cmd=[sys.executable,str(HERE.parent/"training"/"build_unified_labels_r18.py"),str(relative),"--source",str(source),"--out",str(labels)]
        self._write_external_training_command(cmd,"R18.1 corrected global-XYZ+dXYZ+mode5 labels",f"BUILD_R18_1_ABSXYZ_LABELS_{source.name}")

    def training_validate_r18_labels(self):
        try:
            _source,relative,labels=self._r18_selected_paths(require_labels=True)
        except Exception as e:
            messagebox.showerror("生成R18标签验证命令",str(e));return
        report=labels/"validation_r18.json"
        cmd=[sys.executable,str(HERE.parent/"training"/"validate_unified_labels_r18.py"),str(relative),"--labels",str(labels),"--json-out",str(report)]
        self._write_external_training_command(cmd,"Validate R18.1 global-XYZ labels",f"VALIDATE_R18_1_LABELS_{labels.name}")

    def training_run_r18(self):
        try:
            _source,relative,labels=self._r18_selected_paths(require_labels=True)
        except Exception as e:
            messagebox.showerror("生成R18统一训练命令",str(e));return
        qflag="--use-q" if self.training_r18_use_q_var.get() else "--no-use-q"
        det="--deterministic" if self.training_deterministic_var.get() else "--no-deterministic"
        cmd=[sys.executable,str(HERE.parent/"training"/"train_unified_magtouch_r18_hw5.py"),str(relative),
             "--labels",str(labels),"--objective",self.training_r18_objective_var.get(),
             "--sample-policy",self.training_r18_sample_policy_var.get(),"--feature",self.training_r18_feature_var.get(),
             "--history",self.training_r18_history_var.get(),"--endpoint-min-history",self.training_r18_endpoint_history_var.get(),
             qflag,"--epochs",str(self.training_epochs_var.get()),"--batch",str(self.training_hw5_batch_var.get()),
             "--gpu",str(self.training_gpu_var.get()),"--d_model",str(self.training_hw5_dmodel_var.get()),
             "--layers",str(self.training_hw5_layers_var.get()),"--seed",str(self.training_seed_var.get()),det]
        slug=(f"R18_{self.training_r18_objective_var.get()}_{self.training_r18_sample_policy_var.get()}_"
              f"{self.training_r18_feature_var.get()}_H{self.training_r18_history_var.get()}_Q{int(self.training_r18_use_q_var.get())}")
        self._write_external_training_command(cmd,"R18.1 guarded global-XYZ+dXYZ+mode5 training",slug)

    def launch_plan_analysis_exporter(self):
        """Launch the large-Plan lightweight exporter with this exact project's Python runtime."""
        tool = HERE.parent / "tools" / "export_plan_for_analysis.py"
        if not tool.exists():
            messagebox.showerror(
                "Plan分析导出 / Plan Export",
                f"未找到工具 / Tool not found:\n{tool}"
            )
            return
        try:
            # Critical rule: always reuse the exact interpreter that is already running MagTouch.
            # This works for venv/conda/system Python/PyCharm alike and avoids py -3.10 / PATH mismatches.
            cmd = [sys.executable, str(tool)]
            subprocess.Popen(cmd, cwd=str(HERE.parent))
            self.log(
                "Plan analysis exporter launched with current project runtime: "
                f"{sys.executable}"
            )
        except Exception as e:
            self.log(f"Plan analysis exporter launch error: {e}")
            messagebox.showerror("Plan分析导出 / Plan Export", str(e))

    def _data_use_trial_root(self):
        self.data_manage_root_var.set(self.data_root_var.get());self.refresh_data_index()

    def _data_choose_root(self):
        p=filedialog.askdirectory(initialdir=self.data_manage_root_var.get() or self.data_root_var.get())
        if p:self.data_manage_root_var.set(p);self.refresh_data_index()

    def refresh_data_index(self,silent=False):
        if not hasattr(self,"data_tree"): return
        root=Path(self.data_manage_root_var.get()).expanduser()
        filt=self.data_filter_var.get().strip().lower()
        for iid in self.data_tree.get_children(): self.data_tree.delete(iid)
        self.data_trial_paths={}; self.data_folder_paths={}
        trial_count=0; plan_count=0; session_count=0
        if root.exists():
            # Build a stable Session -> Plan -> Trial hierarchy.  For legacy data with
            # Trial_* directly under a Session, create a synthetic "Legacy Trials" folder.
            sessions=[p for p in root.iterdir() if p.is_dir() and p.name!="_TRASH"]
            # If Root itself is a Plan folder, expose it directly without inventing a session.
            if root.name.startswith("Plan_"):
                sessions=[root]
            for session in sorted(sessions,key=lambda p:p.name.lower()):
                if session.name.startswith("Plan_"):
                    plan_dirs=[session]; legacy=[]; session_parent=""
                else:
                    plan_dirs=sorted([p for p in session.iterdir() if p.is_dir() and p.name.startswith("Plan_")],key=lambda p:p.name.lower())
                    legacy=sorted([p for p in session.iterdir() if p.is_dir() and re.match(r"^Trial_\d+",p.name,re.I)],key=lambda p:p.name.lower())
                    session_parent=None
                groups=[]
                for plan in plan_dirs:
                    tds=sorted([p for p in plan.rglob("Trial_*") if p.is_dir() and re.match(r"^Trial_\d+",p.name,re.I)],key=lambda p:p.name.lower())
                    if tds: groups.append((plan.name,plan,tds))
                if legacy: groups.append(("Legacy Trials / 旧版Trial",session,legacy))
                if not groups: continue
                # Filter is applied to group/trial names and metadata; keep matching descendants.
                rendered=[]
                for gname,gpath,tds in groups:
                    items=[]
                    for td in tds:
                        meta={}
                        try: meta=json.loads((td/"meta.json").read_text(encoding="utf-8"))
                        except Exception: pass
                        tp=meta.get("trial_parameters",{}) if isinstance(meta,dict) else {}
                        mode=tp.get("command_mode",""); status=meta.get("status","unknown") if isinstance(meta,dict) else "unknown"
                        rep=tp.get("repeat_count",1); dur=meta.get("duration_s","") if isinstance(meta,dict) else ""
                        hay=(session.name+" "+gname+" "+td.name+" "+str(mode)+" "+str(status)).lower()
                        if filt and filt not in hay: continue
                        items.append((td,mode,status,rep,dur))
                    if items or (filt and filt in (session.name+" "+gname).lower()): rendered.append((gname,gpath,items))
                if not rendered: continue
                if session.name.startswith("Plan_"):
                    session_iid=""
                else:
                    session_iid=f"session:{session_count}"; session_count+=1
                    self.data_folder_paths[session_iid]=session
                    self.data_tree.insert("","end",iid=session_iid,text=f"📁 {session.name}",values=("","", "", ""),open=True)
                for gname,gpath,items in rendered:
                    plan_iid=f"plan:{plan_count}"; plan_count+=1; self.data_folder_paths[plan_iid]=gpath
                    label=f"📂 {gname}   ({len(items)} Trials)"
                    self.data_tree.insert(session_iid,"end",iid=plan_iid,text=label,values=("","", "", ""),open=False)
                    for td,mode,status,rep,dur in items:
                        iid=f"trial:{trial_count}"; trial_count+=1; self.data_trial_paths[iid]=td
                        dtext=f"{dur:.2f}" if isinstance(dur,(int,float)) else dur
                        self.data_tree.insert(plan_iid,"end",iid=iid,text=f"  {td.name}",values=(mode,status,rep,dtext))
        if not silent: self.log(f"Data index refreshed: root={root} plans={plan_count} trials={trial_count}")

    def _selected_data_path(self):
        if not hasattr(self,"data_tree"): return None
        sel=self.data_tree.selection()
        if not sel: return None
        iid=sel[0]
        return self.data_trial_paths.get(iid) or self.data_folder_paths.get(iid)

    def _selected_trial_path(self):
        if not hasattr(self,"data_tree"): return None
        sel=self.data_tree.selection()
        return self.data_trial_paths.get(sel[0]) if sel else None

    def _data_tree_double_click(self,event):
        iid=self.data_tree.identify_row(event.y)
        if not iid: return
        self.data_tree.selection_set(iid); self.data_tree.focus(iid)
        if iid in self.data_trial_paths:
            self.load_selected_trial_for_replay(); return "break"
        if iid in self.data_folder_paths:
            self.data_tree.item(iid,open=not bool(self.data_tree.item(iid,"open"))); return "break"

    def _data_tree_selected(self,_e=None):
        p=self._selected_data_path()
        if not p: return
        td=self._selected_trial_path()
        if td is None:
            # Folder selection: show compact experiment summary instead of pretending it is a Trial.
            n=sum(1 for x in Path(p).rglob("Trial_*") if x.is_dir() and re.match(r"^Trial_\d+",x.name,re.I))
            self.data_selected_var.set(f"实验文件夹 / Folder: {p}   |   Trial={n}")
            self.data_qc_var.set("pending")
            info={"folder":str(p),"trial_count":n,"hint":"展开文件夹并选择 Trial 子项后可载入同步回放。"}
            self.data_detail.configure(state="normal"); self.data_detail.delete("1.0","end"); self.data_detail.insert("1.0",json.dumps(info,ensure_ascii=False,indent=2)); self.data_detail.configure(state="disabled")
            return
        self.data_selected_var.set(str(td))
        meta={}
        try: meta=json.loads((td/"meta.json").read_text(encoding="utf-8"))
        except Exception as e: meta={"meta_error":str(e)}
        qc={}
        try: qc=json.loads((td/"qc_user.json").read_text(encoding="utf-8"))
        except Exception: pass
        self.data_qc_var.set(qc.get("status","pending"))
        txt=json.dumps({"folder":td.name,"meta":meta,"qc_user":qc},ensure_ascii=False,indent=2)
        self.data_detail.configure(state="normal");self.data_detail.delete("1.0","end");self.data_detail.insert("1.0",txt);self.data_detail.configure(state="disabled")

    def open_selected_trial_folder(self):
        p=self._selected_data_path()
        if not p: return
        try:
            if os.name=="nt": os.startfile(str(p))
            elif sys.platform=="darwin": subprocess.Popen(["open",str(p)])
            else: subprocess.Popen(["xdg-open",str(p)])
        except Exception as e: messagebox.showerror("Open Folder",str(e))

    def save_selected_trial_qc(self):
        td=self._selected_trial_path()
        if not td:return
        payload={"status":self.data_qc_var.get(),"updated_local":datetime.now().isoformat(timespec="seconds"),"app":APP_NAME}
        (td/"qc_user.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
        self.log(f"QC saved: {td.name} -> {payload['status']}");self.refresh_data_index(silent=True)

    def trash_selected_trial(self):
        td=self._selected_trial_path()
        if not td:return
        if self.recorder is not None and Path(self.recorder.trial_dir)==td:
            messagebox.showwarning("Move to Trash","活动Trial不能移动 / Active Trial cannot be moved");return
        if not messagebox.askyesno("移动到回收区 / Move to _TRASH",f"不会永久删除。确认移动？\n{td}"):
            return
        root=Path(self.data_manage_root_var.get()).expanduser();dest=root/"_TRASH"/td.parent.name/td.name;dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():dest=dest.with_name(dest.name+"_"+datetime.now().strftime("%H%M%S"))
        shutil.move(str(td),str(dest));self.log(f"Trial moved to _TRASH: {dest}");self.refresh_data_index()

    @staticmethod
    def _csv_float(row,key,default=float("nan")):
        try:
            v=row.get(key,"")
            return float(v) if v not in (None,"") else default
        except Exception:return default

    def _close_replay_caps(self):
        for cap in getattr(self,"replay_caps",{}).values():
            try:cap.release()
            except Exception:pass
        self.replay_caps={};self.replay_camera_times={}

    def load_selected_trial_for_replay(self):
        td=self._selected_trial_path()
        if not td:
            if self._selected_data_path() is not None:
                messagebox.showinfo("Replay","当前选择的是实验文件夹。请展开并选择一个 Trial 子文件夹后再载入回放。")
            return
        mag=td/"magnetic_raw.csv"
        if not mag.exists():messagebox.showerror("Replay",f"缺少 magnetic_raw.csv\n{mag}");return
        self.replay_pause();self._close_replay_caps();samples=[];times=[]
        try:
            with open(mag,"r",encoding="utf-8-sig",newline="") as f:
                for row in csv.DictReader(f):
                    d={}
                    for i in range(1,6):
                        x=self._csv_float(row,f"S{i}_Bx_uT");y=self._csv_float(row,f"S{i}_By_uT");z=self._csv_float(row,f"S{i}_Bz_uT");m=self._csv_float(row,f"S{i}_Bmag_uT")
                        d[i-1]={"x_ut":x,"y_ut":y,"z_ut":z,"mag_ut":m,
                            "x_raw":self._csv_float(row,f"S{i}_x_raw"),
                            "y_raw":self._csv_float(row,f"S{i}_y_raw"),
                            "z_raw":self._csv_float(row,f"S{i}_z_raw"),
                            "valid":str(row.get(f"S{i}_valid","1")).lower() not in ("0","false")
                                and float(row.get("valid_sensor_count") or 5)>=5}
                    samples.append(d)
                    try:times.append(int(row.get("t_group_ns") or row.get("t_return_ns") or 0))
                    except Exception:times.append(0)
        except Exception as e:messagebox.showerror("Replay",str(e));return
        self.replay_samples=samples;self.replay_times=times;self.replay_trial_dir=td
        # Event and stage timelines
        self.replay_events=[];self.replay_stages=[]
        for fn,target in [("events.csv",self.replay_events),("stage_commands.csv",self.replay_stages)]:
            p=td/fn
            if p.exists():
                try:
                    with open(p,"r",encoding="utf-8-sig",newline="") as f:
                        for row in csv.DictReader(f):
                            try:t=int(float(row.get("t_perf_ns") or 0))
                            except Exception:t=0
                            target.append((t,row))
                    target.sort(key=lambda x:x[0])
                except Exception as e:self.log(f"Replay read {fn} error: {e}")
        diffs=[(b-a)/1e6 for a,b in zip(times,times[1:]) if a>0 and b>a]
        self.replay_median_dt_ms=statistics.median(diffs) if diffs else 20.0
        self.replay_slider.configure(to=max(1,len(samples)-1));self.replay_slider_var.set(0)
        # Optional fixed-schema prediction stream. Its perf_counter_ns timestamp is
        # synchronized with magnetic_raw.csv and therefore follows the same slider.
        self.replay_predictions=[];self.replay_prediction_times=[]
        pred_path=td/"live_predictions.csv"
        if pred_path.exists():
            try:
                with open(pred_path,"r",encoding="utf-8-sig",newline="") as f:
                    for row in csv.DictReader(f):
                        try:t_pred=int(float(row.get("t_prediction_ns") or 0))
                        except Exception:t_pred=0
                        xyz=tuple(self._csv_float(row,key) for key in ("visual_x_mm","visual_y_mm","visual_z_mm"))
                        # Keep invalid samples as timeline breaks, never hold a stale
                        # valid position across a recorded no-contact interval.
                        self.replay_predictions.append((t_pred,row,xyz))
                self.replay_predictions.sort(key=lambda item:item[0])
                self.replay_prediction_times=[item[0] for item in self.replay_predictions]
                if self.replay_predictions:
                    maxima=[max((abs(item[2][axis]) for item in self.replay_predictions
                                if all(math.isfinite(v) for v in item[2])),default=.1) for axis in range(3)]
                    self.replay_prediction_3d.set_scale([max(0.10,value*1.10) for value in maxima])
                    self.replay_prediction_info_var.set(
                        f"已载入 {len(self.replay_predictions)} 行预测；滑块与磁场、事件和三维轨迹共用时间轴。"
                    )
                else:
                    self.replay_prediction_3d.clear()
                    self.replay_prediction_info_var.set("live_predictions.csv 存在，但没有有效三维适配输出。")
            except Exception as e:
                self.replay_prediction_3d.clear()
                self.replay_prediction_info_var.set(f"预测记录读取失败：{e}")
                self.log(f"Replay prediction read warning: {e}")
        else:
            self.replay_prediction_3d.clear()
            self.replay_prediction_info_var.set("该 Trial 没有 live_predictions.csv（普通历史 Trial 可继续回放磁场）。")
        # Optional camera files and timestamp timelines.
        # V1.3.4-R1 fixes legacy recordings where timestamp frame_id was the continuously
        # running camera counter while each Trial MP4 itself started from frame 0.
        replay_video_notes=[]
        if cv2 is not None:
            meta_obj={}
            try:
                mp=td/"meta.json"
                if mp.exists(): meta_obj=json.loads(mp.read_text(encoding="utf-8"))
            except Exception as e:
                self.log(f"Replay meta read warning: {e}")
            camera_meta=meta_obj.get("camera_recordings",{}) if isinstance(meta_obj,dict) else {}
            trial_t0=int(((meta_obj.get("timebase") or {}).get("trial_t0_perf_ns") or 0)) if isinstance(meta_obj,dict) else 0
            for name in ("top","side"):
                cmeta=camera_meta.get(name,{}) if isinstance(camera_meta,dict) else {}
                # First trust meta.json; then support current and older/common filenames.
                video_candidates=[]; ts_candidates=[]
                if isinstance(cmeta,dict):
                    if cmeta.get("video"): video_candidates.append(td/str(cmeta.get("video")))
                    if cmeta.get("timestamps"): ts_candidates.append(td/str(cmeta.get("timestamps")))
                video_candidates += [td/f"camera_{name}.mp4",td/f"camera_{name}.avi",td/f"camera_{name}.mkv"]
                ts_candidates += [td/f"camera_{name}_timestamps.csv",td/f"camera_{name}_timestamp.csv"]
                # Also accept files copied/renamed by older builds.
                try:
                    video_candidates += sorted([x for x in td.glob(f"*{name}*") if x.suffix.lower() in (".mp4",".avi",".mkv",".mov")])
                    ts_candidates += sorted([x for x in td.glob(f"*{name}*.csv") if "time" in x.name.lower()])
                except Exception: pass
                vp=next((x for x in video_candidates if x.exists() and x.is_file()),None)
                tp=next((x for x in ts_candidates if x.exists() and x.is_file()),None)
                if vp is None:
                    replay_video_notes.append(f"{name}:no-video")
                    continue
                cap=cv2.VideoCapture(str(vp))
                if not cap.isOpened():
                    replay_video_notes.append(f"{name}:open-failed")
                    self.log(f"Replay {name}: cannot open video {vp}")
                    cap.release()
                    continue
                self.replay_caps[name]=cap
                frame_count=max(0,int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
                fps=float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                lst=[]; legacy_normalized=False
                if tp is not None:
                    try:
                        raw_rows=[]
                        with open(tp,"r",encoding="utf-8-sig",newline="") as f:
                            for seq,row in enumerate(csv.DictReader(f)):
                                try:
                                    t=int(float(row.get("t_perf_ns") or 0))
                                    raw_id=int(float(row.get("frame_id") or seq))
                                    raw_rows.append((t,raw_id,seq))
                                except Exception: pass
                        # Old V1.x recorded capture-global frame IDs. If those IDs do not fit
                        # inside this Trial video, normalize by timestamp-row order (0..N-1).
                        for t,raw_id,seq in raw_rows:
                            fid=raw_id
                            if frame_count>0 and (raw_id < 0 or raw_id >= frame_count):
                                fid=seq; legacy_normalized=True
                            lst.append((t,fid))
                    except Exception as e:
                        self.log(f"Replay {name} timestamp read warning: {e}")
                # If video exists but timestamp CSV is absent/unusable, allow approximate replay
                # from Trial start + video FPS rather than hiding an existing recording.
                if not lst and frame_count>0:
                    if fps <= 0.1: fps=30.0
                    tbase=trial_t0 or (next((x for x in self.replay_times if x>0),0))
                    lst=[(int(tbase + i*1e9/fps),i) for i in range(frame_count)]
                    replay_video_notes.append(f"{name}:approx-ts,{frame_count}f")
                else:
                    note=f"{name}:{len(lst)}ts/{frame_count}f"
                    if legacy_normalized: note += ",legacy-fixed"
                    replay_video_notes.append(note)
                self.replay_camera_times[name]=lst
                self.log(f"Replay {name}: video={vp.name}, timestamps={(tp.name if tp else 'synthetic')}, frames={frame_count}, timeline={len(lst)}, legacy_fixed={legacy_normalized}")
        video_note=" | video="+", ".join(replay_video_notes) if replay_video_notes else ""
        self.replay_info_var.set(f"{td.parent.name}/{td.name} | samples={len(samples)} | median Δt={self.replay_median_dt_ms:.2f} ms{video_note}")
        self.replay_backtest_panel.on_trial_loaded()
        self._replay_show_index(0)

    def _replay_slider_changed(self,value):
        if not self.replay_samples:return
        try:i=max(0,min(len(self.replay_samples)-1,int(round(float(value)))))
        except Exception:i=0
        self._replay_show_index(i)

    @staticmethod
    def _latest_timeline_item(items,t_ns):
        if not items:return None
        ts=[x[0] for x in items];k=bisect.bisect_right(ts,t_ns)-1
        return items[k][1] if k>=0 else None

    def _replay_show_index(self,i):
        if not self.replay_samples:return
        i=max(0,min(len(self.replay_samples)-1,int(i)))
        start=max(0,i-self.replay_plot.maxlen+1)
        self.replay_plot.set_window(self.replay_samples[start:i+1])
        try:self.replay_slider_var.set(i)
        except Exception:pass
        t=self.replay_times[i] if i<len(self.replay_times) else 0
        t0=next((x for x in self.replay_times if x>0),t)
        rel=(t-t0)/1e9 if t and t0 else 0.0
        ev=self._latest_timeline_item(self.replay_events,t);st=self._latest_timeline_item(self.replay_stages,t)
        evtxt=(ev.get("event","")+":"+str(ev.get("value",""))) if ev else "--"
        sttxt=(st.get("phase","")+" "+st.get("command","")) if st else "--"
        self.replay_state_var.set(f"#{i+1}/{len(self.replay_samples)}  t={rel:.3f}s  Event={evtxt}  Stage={sttxt}")
        self.replay_backtest_panel.show_index(i)
        self._replay_show_cameras(t)

    def _replay_show_cameras(self,t_ns):
        for name,lab in self.replay_camera_labels.items():
            arr=self.replay_camera_times.get(name,[]);cap=self.replay_caps.get(name)
            if not arr or cap is None or cv2 is None:
                reason = "缺 OpenCV，无法回放" if cv2 is None else "无可读取录像 / No readable video"
                self._set_camera_placeholder(lab, reason)
                continue
            if Image is None or ImageTk is None:
                self._set_camera_placeholder(lab, "缺 Pillow，无法显示录像")
                continue
            ts=[x[0] for x in arr];k=bisect.bisect_left(ts,t_ns)
            if k>=len(arr):k=len(arr)-1
            elif k>0 and abs(ts[k-1]-t_ns)<abs(ts[k]-t_ns):k-=1
            frame_id=arr[k][1]
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES,frame_id);ok,frame=cap.read()
                if ok:
                    lab.delete("camera_placeholder")
                    self._render_camera_frame(name,frame,lab,self.replay_camera_images,"replay_"+name)
                else:
                    self._set_camera_placeholder(lab, f"录像解码失败：帧 {frame_id}")
            except Exception as exc:
                self._set_camera_placeholder(lab, f"录像读取错误：{str(exc)[:70]}")

    def replay_play(self):
        if not self.replay_samples:return
        # Recorded video/magnetics must play even without a model or usable B0.
        # Recalculation is explicitly requested with Apply and Backtest.
        self.replay_playing=True;self._replay_schedule_tick()

    def replay_pause(self):
        self.replay_playing=False
        tick_after = getattr(self,"replay_tick_after",None)
        if tick_after is not None:
            try:self.after_cancel(tick_after)
            except Exception:pass
            self.replay_tick_after=None

    def replay_stop(self):
        self.replay_pause();self.replay_slider_var.set(0);self._replay_show_index(0) if self.replay_samples else None

    def _replay_schedule_tick(self):
        if not self.replay_playing:return
        try:i=int(round(self.replay_slider_var.get()))
        except Exception:i=0
        if i>=len(self.replay_samples)-1:
            self.replay_pause();return
        self._replay_show_index(i+1)
        speed=max(0.05,float(self.replay_speed_var.get()))
        delay=max(10,min(500,int(self.replay_median_dt_ms/speed)))
        self.replay_tick_after=self.after(delay,self._replay_schedule_tick)

    def _build_ubuntu_training_tab(self):
        """Mirror the Windows training page while emitting Ubuntu Bash only."""
        host=self.tab_ubuntu
        host.columnconfigure(0,weight=1);host.rowconfigure(0,weight=1)
        canvas=tk.Canvas(host,highlightthickness=0,borderwidth=0)
        scrollbar=ttk.Scrollbar(host,orient="vertical",command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0,column=0,sticky="nsew");scrollbar.grid(row=0,column=1,sticky="ns")
        tab=ttk.Frame(canvas);window=canvas.create_window((0,0),window=tab,anchor="nw")
        tab.bind("<Configure>",lambda _e:canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",lambda e:canvas.itemconfigure(window,width=e.width))
        def wheel(event):
            if getattr(event,"delta",0):canvas.yview_scroll(int(-event.delta/120),"units")
            return "break"
        def wheel_on(_e):canvas.bind_all("<MouseWheel>",wheel)
        def wheel_off(_e):canvas.unbind_all("<MouseWheel>")
        host.bind("<Enter>",wheel_on);host.bind("<Leave>",wheel_off)
        canvas.bind("<Enter>",wheel_on);canvas.bind("<Leave>",wheel_off)
        self.ubuntu_canvas=canvas;self.ubuntu_scrollbar=scrollbar
        tab.columnconfigure(0,weight=1)

        cfg=normalize_ubuntu_training_config(self.settings.get("ubuntu_training",{}))
        self.ubuntu_project_root_var=tk.StringVar(value=str(cfg["project_root"]))
        self.ubuntu_plan_path_var=tk.StringVar(value=str(cfg["plan_path"]))
        self.ubuntu_export_output_var=tk.StringVar(value=str(cfg["export_output_path"]))
        self.ubuntu_training_export_var=tk.StringVar(value=str(cfg["training_export_path"]))
        self.ubuntu_h1_features_var=tk.StringVar(value=str(cfg["h1_features_path"]))
        self.ubuntu_h2_relative_var=tk.StringVar(value=str(cfg["h2_relative_path"]))
        self.ubuntu_h2_spatial_labels_var=tk.StringVar(value=str(cfg["h2_spatial_labels_path"]))
        ubuntu_r18_labels=str(cfg["h2_unified_labels_path"])
        if ubuntu_r18_labels.endswith("_UNIFIED_LABELS_R18"):
            ubuntu_r18_labels += "_1_ABSXYZ"
        self.ubuntu_h2_unified_labels_var=tk.StringVar(value=ubuntu_r18_labels)
        self.ubuntu_conda_env_var=tk.StringVar(value=str(cfg["conda_env"]))
        self.ubuntu_gpus_var=tk.StringVar(value=str(cfg["gpus"]))
        self.ubuntu_cpu_count_var=tk.StringVar(value=str(cfg["cpu_count"]))
        self.ubuntu_gpu_schedule_var=tk.StringVar(value=str(cfg["gpu_schedule"]))
        self.ubuntu_seeds_var=tk.StringVar(value=str(cfg["seeds"]))
        self.ubuntu_h1_feature_var=tk.StringVar(value=str(cfg["h1_feature"]))
        self.ubuntu_h1_policy_var=tk.StringVar(value=str(cfg["h1_policy"]))
        self.ubuntu_baseline_model_var=tk.StringVar(value=str(cfg["baseline_model"]))
        self.ubuntu_baseline_task_var=tk.StringVar(value=str(cfg["baseline_task"]))
        self.ubuntu_epochs_var=tk.StringVar(value=str(cfg["epochs"]))
        self.ubuntu_baseline_batch_var=tk.StringVar(value=str(cfg["baseline_batch"]))
        self.ubuntu_hw5_batch_var=tk.StringVar(value=str(cfg["hw5_batch"]))
        self.ubuntu_dmodel_var=tk.StringVar(value=str(cfg["d_model"]))
        self.ubuntu_layers_var=tk.StringVar(value=str(cfg["layers"]))
        self.ubuntu_h1_objective_var=tk.StringVar(value=str(cfg["h1_objective"]))
        self.ubuntu_deterministic_var=tk.BooleanVar(value=bool(cfg["deterministic"]))
        self.ubuntu_h2_feature_var=tk.StringVar(value=str(cfg["h2_feature"]))
        self.ubuntu_h2_history_var=tk.StringVar(value=str(cfg["h2_history"]))
        self.ubuntu_h2_endpoint_history_var=tk.StringVar(value=str(cfg["h2_endpoint_min_history"]))
        self.ubuntu_h2_use_q_var=tk.BooleanVar(value=bool(cfg["h2_use_q"]))
        self.ubuntu_h2_spatial_objective_var=tk.StringVar(value=str(cfg["h2_spatial_objective"]))
        self.ubuntu_h2_sample_policy_var=tk.StringVar(value=str(cfg["h2_sample_policy"]))
        self.ubuntu_h2_unified_objective_var=tk.StringVar(value=str(cfg["h2_unified_objective"]))
        self.ubuntu_h2_unified_sample_policy_var=tk.StringVar(value=str(cfg["h2_unified_sample_policy"]))
        self.ubuntu_extra_args_var=tk.StringVar(value=str(cfg["extra_args"]))
        self.ubuntu_status_var=tk.StringVar(value="尚未生成 / Not generated")
        self.ubuntu_last_command_kind="h2_train"

        head=ttk.LabelFrame(tab,text="Ubuntu 训练导出 / Windows training-page parity")
        head.grid(row=0,column=0,sticky="ew",padx=8,pady=(8,4));head.columnconfigure(0,weight=1)
        ttk.Label(head,text=f"推荐配置 / Recommended: {RECOMMENDED_HARDWARE}",font=("Segoe UI",13,"bold"),foreground="#0d47a1").grid(row=0,column=0,sticky="w",padx=8,pady=(7,3))
        ttk.Label(head,text="功能与 Windows 训练页逐项对应；区别仅为 Ubuntu /home 路径、Bash/conda/Python 命令及多 GPU/CPU 调度。所有按钮只生成脚本，不启动训练。",foreground="#7a3e00",wraplength=1250,justify="left").grid(row=1,column=0,sticky="w",padx=8,pady=(2,7))

        export=ttk.LabelFrame(tab,text="① 数据导出 / Export command")
        export.grid(row=1,column=0,sticky="ew",padx=8,pady=4);export.columnconfigure(1,weight=1);export.columnconfigure(4,weight=1)
        ttk.Label(export,text="Ubuntu 项目根目录").grid(row=0,column=0,sticky="e",padx=5,pady=3)
        ttk.Entry(export,textvariable=self.ubuntu_project_root_var).grid(row=0,column=1,sticky="ew",padx=5,pady=3)
        ttk.Button(export,text="选择目录",command=lambda:self._ubuntu_choose_directory(self.ubuntu_project_root_var,"选择 Ubuntu 项目根目录")).grid(row=0,column=2,padx=4,pady=3)
        ttk.Label(export,text="conda 环境").grid(row=0,column=3,sticky="e",padx=5,pady=3)
        ttk.Entry(export,textvariable=self.ubuntu_conda_env_var).grid(row=0,column=4,sticky="ew",padx=5,pady=3)
        ttk.Label(export,text="原始 Plan").grid(row=1,column=0,sticky="e",padx=5,pady=3)
        ttk.Entry(export,textvariable=self.ubuntu_plan_path_var).grid(row=1,column=1,sticky="ew",padx=5,pady=3)
        ttk.Button(export,text="选择 Plan",command=lambda:self._ubuntu_choose_directory(self.ubuntu_plan_path_var,"选择 Ubuntu 原始 Plan 目录")).grid(row=1,column=2,padx=4,pady=3)
        ttk.Label(export,text="输出目录").grid(row=1,column=3,sticky="e",padx=5,pady=3)
        ttk.Entry(export,textvariable=self.ubuntu_export_output_var).grid(row=1,column=4,sticky="ew",padx=5,pady=3)
        ttk.Button(export,text="选择输出",command=lambda:self._ubuntu_choose_directory(self.ubuntu_export_output_var,"选择 Ubuntu 导出目录")).grid(row=1,column=5,padx=4,pady=3)
        buttons=ttk.Frame(export);buttons.grid(row=2,column=0,columnspan=6,sticky="w",padx=5,pady=(3,6))
        ttk.Button(buttons,text="生成轻量分析导出 / QC",command=lambda:self._ubuntu_generate_commands("analysis_export")).pack(side="left",padx=(0,5))
        ttk.Button(buttons,text="生成完整训练导出 / TRAINING_EXPORT",command=lambda:self._ubuntu_generate_commands("training_export")).pack(side="left",padx=5)

        features=ttk.LabelFrame(tab,text="② 构建 40 Hz / T=32 / F0–F5 特征")
        features.grid(row=2,column=0,sticky="ew",padx=8,pady=4);features.columnconfigure(1,weight=1)
        ttk.Label(features,text="TRAINING_EXPORT目录").grid(row=0,column=0,sticky="e",padx=5,pady=3)
        ttk.Entry(features,textvariable=self.ubuntu_training_export_var).grid(row=0,column=1,sticky="ew",padx=5,pady=3)
        ttk.Button(features,text="选择目录",command=lambda:self._ubuntu_choose_directory(self.ubuntu_training_export_var,"选择 Ubuntu TRAINING_EXPORT 目录")).grid(row=0,column=2,padx=4,pady=3)
        ttk.Label(features,text="H1 特征目录").grid(row=1,column=0,sticky="e",padx=5,pady=3)
        ttk.Entry(features,textvariable=self.ubuntu_h1_features_var).grid(row=1,column=1,sticky="ew",padx=5,pady=3)
        ttk.Button(features,text="选择目录",command=lambda:self._ubuntu_choose_directory(self.ubuntu_h1_features_var,"选择 Ubuntu H1 特征目录")).grid(row=1,column=2,padx=4,pady=3)
        ttk.Button(features,text="生成 F0–F5 构建命令",command=lambda:self._ubuntu_generate_commands("build_h1_features")).grid(row=0,column=3,rowspan=2,padx=5,pady=3)
        ttk.Label(features,text="与 Windows 页一致：40 Hz 重采样、Trial 级划分、F0–F5；这里只把命令换为 Ubuntu 路径。",foreground="#555").grid(row=2,column=0,columnspan=4,sticky="w",padx=5,pady=(1,5))

        compute=ttk.LabelFrame(tab,text="③ Ubuntu GPU / CPU 调度")
        compute.grid(row=3,column=0,sticky="ew",padx=8,pady=4);compute.columnconfigure(1,weight=1);compute.columnconfigure(3,weight=1)
        ttk.Label(compute,text="GPU IDs（逗号分隔）").grid(row=0,column=0,sticky="e",padx=5,pady=3)
        ttk.Entry(compute,textvariable=self.ubuntu_gpus_var).grid(row=0,column=1,sticky="ew",padx=5,pady=3)
        ttk.Label(compute,text="CPU 总核心数").grid(row=0,column=2,sticky="e",padx=5,pady=3)
        ttk.Entry(compute,textvariable=self.ubuntu_cpu_count_var).grid(row=0,column=3,sticky="ew",padx=5,pady=3)
        ttk.Label(compute,text="串行 seeds").grid(row=1,column=0,sticky="e",padx=5,pady=3)
        ttk.Entry(compute,textvariable=self.ubuntu_seeds_var).grid(row=1,column=1,sticky="ew",padx=5,pady=3)
        ttk.Label(compute,text="性能优先安全模式：任何时候只运行 1 个训练进程，独占 CPU 线程；GPU IDs 按顺序轮换，但绝不并发。",foreground="#1b5e20",wraplength=900,justify="left").grid(row=1,column=2,columnspan=2,sticky="w",padx=5,pady=3)
        ttk.Label(compute,text="GPU 调度").grid(row=2,column=0,sticky="e",padx=5,pady=3)
        ttk.Combobox(compute,textvariable=self.ubuntu_gpu_schedule_var,values=["serial","full_parallel"],state="readonly",width=16).grid(row=2,column=1,sticky="w",padx=5,pady=3)
        ttk.Label(compute,text="serial=单任务安全模式；full_parallel=不同实验分配到全部所选 GPU，CPU 自动均分。",foreground="#7a3e00",wraplength=900,justify="left").grid(row=2,column=2,columnspan=2,sticky="w",padx=5,pady=3)

        h1=ttk.LabelFrame(tab,text="④ H1 Pilot：基线 / MagTouch-RA-HW5 / R13 核心矩阵")
        h1.grid(row=4,column=0,sticky="ew",padx=8,pady=4)
        labels=[("特征",0),("策略",2),("基线模型",4),("任务",6),("Epoch",8),("Baseline Batch",10)]
        for text,column in labels:ttk.Label(h1,text=text).grid(row=0,column=column,sticky="e",padx=(5,2),pady=4)
        ttk.Combobox(h1,textvariable=self.ubuntu_h1_feature_var,values=[f"F{i}" for i in range(6)],state="readonly",width=5).grid(row=0,column=1)
        ttk.Combobox(h1,textvariable=self.ubuntu_h1_policy_var,values=["raw15","clean14","mask15"],state="readonly",width=9).grid(row=0,column=3)
        ttk.Combobox(h1,textvariable=self.ubuntu_baseline_model_var,values=["mlp","tcn","transformer"],state="readonly",width=12).grid(row=0,column=5)
        ttk.Combobox(h1,textvariable=self.ubuntu_baseline_task_var,values=["mode5","z","dx","dy","x","y"],state="readonly",width=8).grid(row=0,column=7)
        ttk.Entry(h1,textvariable=self.ubuntu_epochs_var,width=6).grid(row=0,column=9)
        ttk.Entry(h1,textvariable=self.ubuntu_baseline_batch_var,width=7).grid(row=0,column=11)
        ttk.Label(h1,text="HW5 d_model").grid(row=1,column=0,sticky="e",padx=(5,2),pady=4)
        ttk.Entry(h1,textvariable=self.ubuntu_dmodel_var,width=6).grid(row=1,column=1)
        ttk.Label(h1,text="layers").grid(row=1,column=2,sticky="e")
        ttk.Entry(h1,textvariable=self.ubuntu_layers_var,width=5).grid(row=1,column=3)
        ttk.Label(h1,text="HW5 batch").grid(row=1,column=4,sticky="e")
        ttk.Entry(h1,textvariable=self.ubuntu_hw5_batch_var,width=6).grid(row=1,column=5)
        ttk.Label(h1,text="objective").grid(row=1,column=6,sticky="e")
        ttk.Combobox(h1,textvariable=self.ubuntu_h1_objective_var,values=["mode_only","multitask"],state="readonly",width=12).grid(row=1,column=7)
        ttk.Checkbutton(h1,text="严格固定 seed",variable=self.ubuntu_deterministic_var).grid(row=1,column=8,columnspan=2,sticky="w",padx=7)
        h1buttons=ttk.Frame(h1);h1buttons.grid(row=2,column=0,columnspan=12,sticky="w",padx=5,pady=(3,6))
        ttk.Button(h1buttons,text="生成基线命令",command=lambda:self._ubuntu_generate_commands("h1_baseline")).pack(side="left",padx=(0,5))
        ttk.Button(h1buttons,text="生成 MagTouch-RA-HW5 命令",command=lambda:self._ubuntu_generate_commands("h1_hw5")).pack(side="left",padx=5)
        ttk.Button(h1buttons,text="生成 R13 核心4组实验",command=lambda:self._ubuntu_generate_commands("h1_core_matrix")).pack(side="left",padx=5)

        h2=ttk.LabelFrame(tab,text="⑤ H2-R15：数据构建 / 相对训练 / 特征与历史矩阵")
        h2.grid(row=5,column=0,sticky="ew",padx=8,pady=4);h2.columnconfigure(1,weight=1)
        ttk.Label(h2,text="H2 相对数据目录").grid(row=0,column=0,sticky="e",padx=5,pady=3)
        ttk.Entry(h2,textvariable=self.ubuntu_h2_relative_var).grid(row=0,column=1,sticky="ew",padx=5,pady=3)
        ttk.Button(h2,text="选择目录",command=lambda:self._ubuntu_choose_directory(self.ubuntu_h2_relative_var,"选择 Ubuntu H2 相对数据目录")).grid(row=0,column=2,padx=4)
        ttk.Button(h2,text="生成相对数据命令",command=lambda:self._ubuntu_generate_commands("build_h2_relative")).grid(row=0,column=3,padx=4)
        sub=ttk.Frame(h2);sub.grid(row=1,column=0,columnspan=4,sticky="w",padx=5,pady=3)
        ttk.Label(sub,text="Feature").pack(side="left")
        ttk.Combobox(sub,textvariable=self.ubuntu_h2_feature_var,values=["R0","R1","R2","R3"],state="readonly",width=6).pack(side="left",padx=(3,8))
        ttk.Label(sub,text="History").pack(side="left")
        ttk.Combobox(sub,textvariable=self.ubuntu_h2_history_var,values=["0","16","32","64","128"],state="readonly",width=6).pack(side="left",padx=(3,8))
        ttk.Label(sub,text="Endpoint min history").pack(side="left")
        ttk.Combobox(sub,textvariable=self.ubuntu_h2_endpoint_history_var,values=["0","16","32","64","128"],state="readonly",width=6).pack(side="left",padx=(3,8))
        ttk.Checkbutton(sub,text="轴级 Q",variable=self.ubuntu_h2_use_q_var).pack(side="left",padx=5)
        h2buttons=ttk.Frame(h2);h2buttons.grid(row=2,column=0,columnspan=4,sticky="w",padx=5,pady=(3,6))
        ttk.Button(h2buttons,text="生成相对训练命令",command=lambda:self._ubuntu_generate_commands("h2_train")).pack(side="left",padx=(0,5))
        ttk.Button(h2buttons,text="生成 R15 特征核心4组",command=lambda:self._ubuntu_generate_commands("h2_feature_matrix")).pack(side="left",padx=5)
        ttk.Button(h2buttons,text="生成 R15 历史5组",command=lambda:self._ubuntu_generate_commands("h2_history_matrix")).pack(side="left",padx=5)

        spatial=ttk.LabelFrame(h2,text="V16 固定坐标 X/Y/Z + R15 相对 dX/dY/dZ（严格串行）")
        spatial.grid(row=3,column=0,columnspan=4,sticky="ew",padx=5,pady=(4,6));spatial.columnconfigure(1,weight=1)
        ttk.Label(spatial,text="固定坐标标签目录").grid(row=0,column=0,sticky="e",padx=5,pady=3)
        ttk.Entry(spatial,textvariable=self.ubuntu_h2_spatial_labels_var).grid(row=0,column=1,sticky="ew",padx=5,pady=3)
        ttk.Button(spatial,text="选择目录",command=lambda:self._ubuntu_choose_directory(self.ubuntu_h2_spatial_labels_var,"选择 Ubuntu V16 固定坐标标签目录")).grid(row=0,column=2,padx=4)
        ttk.Button(spatial,text="生成标签命令",command=lambda:self._ubuntu_generate_commands("build_h2_spatial_labels")).grid(row=0,column=3,padx=4)
        opts=ttk.Frame(spatial);opts.grid(row=1,column=0,columnspan=4,sticky="w",padx=5,pady=3)
        ttk.Label(opts,text="Objective").pack(side="left")
        ttk.Combobox(opts,textvariable=self.ubuntu_h2_spatial_objective_var,values=["fixed_only","relative_only","joint"],state="readonly",width=14).pack(side="left",padx=(3,10))
        ttk.Label(opts,text="Sample policy").pack(side="left")
        ttk.Combobox(opts,textvariable=self.ubuntu_h2_sample_policy_var,values=["common","native"],state="readonly",width=9).pack(side="left",padx=(3,10))
        ttk.Label(opts,text="论文公平比较使用 common；坐标系为 printer_centered_surface_v1，尚未标定到传感器几何轴。",foreground="#7a3e00").pack(side="left",padx=5)
        spatial_buttons=ttk.Frame(spatial);spatial_buttons.grid(row=2,column=0,columnspan=4,sticky="w",padx=5,pady=(3,6))
        ttk.Button(spatial_buttons,text="生成 V16 单项训练",command=lambda:self._ubuntu_generate_commands("h2_spatial_train")).pack(side="left",padx=(0,5))
        ttk.Button(spatial_buttons,text="生成 V16 三目标对照",command=lambda:self._ubuntu_generate_commands("h2_spatial_objective_matrix")).pack(side="left",padx=5)
        ttk.Button(spatial_buttons,text="生成 V16 历史5组",command=lambda:self._ubuntu_generate_commands("h2_spatial_history_matrix")).pack(side="left",padx=5)

        unified=ttk.LabelFrame(h2,text="R18 统一输出 XYZ + dXYZ + mode5（数据集可独立指定）")
        unified.grid(row=4,column=0,columnspan=4,sticky="ew",padx=5,pady=(4,6));unified.columnconfigure(1,weight=1)
        ttk.Label(unified,text="R18统一标签目录").grid(row=0,column=0,sticky="e",padx=5,pady=3)
        ttk.Entry(unified,textvariable=self.ubuntu_h2_unified_labels_var).grid(row=0,column=1,sticky="ew",padx=5,pady=3)
        ttk.Button(unified,text="选择目录",command=lambda:self._ubuntu_choose_directory(self.ubuntu_h2_unified_labels_var,"选择 Ubuntu R18 UNIFIED_LABELS 目录")).grid(row=0,column=2,padx=4)
        ttk.Button(unified,text="生成R18标签命令",command=lambda:self._ubuntu_generate_commands("build_h2_unified_labels")).grid(row=0,column=3,padx=4)
        unified_opts=ttk.Frame(unified);unified_opts.grid(row=1,column=0,columnspan=4,sticky="w",padx=5,pady=3)
        ttk.Label(unified_opts,text="Objective").pack(side="left")
        ttk.Combobox(unified_opts,textvariable=self.ubuntu_h2_unified_objective_var,values=["coordinate_only","unified"],state="readonly",width=14).pack(side="left",padx=(3,10))
        ttk.Label(unified_opts,text="Sample policy").pack(side="left")
        ttk.Combobox(unified_opts,textvariable=self.ubuntu_h2_unified_sample_policy_var,values=["common","native"],state="readonly",width=9).pack(side="left",padx=(3,10))
        ttk.Label(unified_opts,text="TRAINING_EXPORT 与 R15相对数据目录使用本页上方各自选择值。",foreground="#1b5e20").pack(side="left",padx=5)
        unified_buttons=ttk.Frame(unified);unified_buttons.grid(row=2,column=0,columnspan=4,sticky="w",padx=5,pady=(3,6))
        ttk.Button(unified_buttons,text="生成R18标签验证",command=lambda:self._ubuntu_generate_commands("validate_h2_unified_labels")).pack(side="left",padx=(0,5))
        ttk.Button(unified_buttons,text="生成R18统一训练",command=lambda:self._ubuntu_generate_commands("h2_unified_train")).pack(side="left",padx=5)

        preview=ttk.LabelFrame(tab,text="⑥ 生成的 Bash/conda/Python 脚本（只生成，不启动）")
        preview.grid(row=6,column=0,sticky="nsew",padx=8,pady=4);preview.columnconfigure(0,weight=1);preview.rowconfigure(0,weight=1)
        self.ubuntu_script_text=ScrolledText(preview,height=16,wrap="none",font=("Consolas",9))
        self.ubuntu_script_text.grid(row=0,column=0,sticky="nsew",padx=5,pady=5)
        actions=ttk.Frame(preview);actions.grid(row=1,column=0,sticky="ew",padx=5,pady=(1,5))
        ttk.Button(actions,text="重新生成当前脚本",command=lambda:self._ubuntu_generate_commands()).pack(side="left",padx=(0,4))
        ttk.Button(actions,text="复制 / Copy",command=self._ubuntu_copy_commands).pack(side="left",padx=4)
        ttk.Button(actions,text="另存 .sh / Save As",command=self._ubuntu_save_script).pack(side="left",padx=4)
        ttk.Button(actions,text="打开命令文件夹",command=self.training_open_command_dir).pack(side="left",padx=4)
        ttk.Label(actions,textvariable=self.ubuntu_status_var,foreground="#1b5e20",wraplength=850,justify="left").pack(side="left",padx=12,fill="x",expand=True)
        ttk.Label(preview,text="额外 Python 参数（仅追加到训练命令）").grid(row=2,column=0,sticky="w",padx=5)
        ttk.Entry(preview,textvariable=self.ubuntu_extra_args_var).grid(row=3,column=0,sticky="ew",padx=5,pady=(2,6))
        self._ubuntu_generate_commands("h2_train",show_error=False,auto_save=False)

    @staticmethod
    def _ubuntu_path_from_dialog(path):
        """Return a POSIX path, including conversion from Windows WSL shares."""
        text=str(path or "").strip()
        if not text:return None
        normalized=text.replace("\\","/")
        if normalized.startswith("/") and not normalized.startswith("//"):
            return normalized.rstrip("/") or "/"
        parts=[part for part in normalized.split("/") if part]
        if len(parts)>=3 and parts[0].lower() in {"wsl$","wsl.localhost"}:
            return "/"+"/".join(parts[2:])
        drive_match=re.fullmatch(r"([A-Za-z]):(?:/(.*))?",normalized)
        if drive_match:
            suffix=(drive_match.group(2) or "").strip("/")
            return f"/mnt/{drive_match.group(1).lower()}"+(f"/{suffix}" if suffix else "")
        return None

    def _ubuntu_choose_directory(self,variable,title):
        current=str(variable.get()).strip()
        initial=current if current and Path(current).is_dir() else (str(HERE.parent) if os.name=="nt" else "/home")
        selected=filedialog.askdirectory(title=title,initialdir=initial,mustexist=True)
        if not selected:return
        ubuntu_path=self._ubuntu_path_from_dialog(selected)
        if ubuntu_path is None:
            messagebox.showwarning(
                "Ubuntu 路径",
                "所选网络目录无法转换为 Ubuntu 路径。\n\n"
                "请通过 \\\\wsl$\\发行版\\... 或 \\\\wsl.localhost\\发行版\\... 进入 Ubuntu 目录，"
                "也可以直接在输入框填写 /home/... 路径。普通盘符目录会转换成 /mnt/盘符/...。",
            )
            return
        variable.set(ubuntu_path)

    def _ubuntu_training_config_from_ui(self):
        return {
            "project_root":self.ubuntu_project_root_var.get(),"plan_path":self.ubuntu_plan_path_var.get(),
            "export_output_path":self.ubuntu_export_output_var.get(),"training_export_path":self.ubuntu_training_export_var.get(),
            "h1_features_path":self.ubuntu_h1_features_var.get(),"h2_relative_path":self.ubuntu_h2_relative_var.get(),"h2_spatial_labels_path":self.ubuntu_h2_spatial_labels_var.get(),
            "h2_unified_labels_path":self.ubuntu_h2_unified_labels_var.get(),
            "conda_env":self.ubuntu_conda_env_var.get(),"gpus":self.ubuntu_gpus_var.get(),
            "cpu_count":self.ubuntu_cpu_count_var.get(),"gpu_schedule":self.ubuntu_gpu_schedule_var.get(),"seeds":self.ubuntu_seeds_var.get(),
            "h1_feature":self.ubuntu_h1_feature_var.get(),"h1_policy":self.ubuntu_h1_policy_var.get(),
            "baseline_model":self.ubuntu_baseline_model_var.get(),"baseline_task":self.ubuntu_baseline_task_var.get(),
            "epochs":self.ubuntu_epochs_var.get(),"baseline_batch":self.ubuntu_baseline_batch_var.get(),
            "hw5_batch":self.ubuntu_hw5_batch_var.get(),"d_model":self.ubuntu_dmodel_var.get(),
            "layers":self.ubuntu_layers_var.get(),"h1_objective":self.ubuntu_h1_objective_var.get(),
            "deterministic":self.ubuntu_deterministic_var.get(),"h2_feature":self.ubuntu_h2_feature_var.get(),
            "h2_history":self.ubuntu_h2_history_var.get(),"h2_endpoint_min_history":self.ubuntu_h2_endpoint_history_var.get(),
            "h2_use_q":self.ubuntu_h2_use_q_var.get(),"h2_spatial_objective":self.ubuntu_h2_spatial_objective_var.get(),"h2_sample_policy":self.ubuntu_h2_sample_policy_var.get(),
            "h2_unified_objective":self.ubuntu_h2_unified_objective_var.get(),"h2_unified_sample_policy":self.ubuntu_h2_unified_sample_policy_var.get(),"extra_args":self.ubuntu_extra_args_var.get(),
        }

    def _ubuntu_auto_save_script(self,script,kind):
        command_dir=self._training_command_dir()
        stamp=datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_kind=re.sub(r"[^A-Za-z0-9_.-]+","_",str(kind)).strip("_").upper() or "COMMAND"
        path=command_dir/f"{stamp}_UBUNTU_{safe_kind}.sh"
        last_path=command_dir/"RUN_LAST_UBUNTU_COMMAND.sh"
        for target in (path,last_path):
            target.write_text(script,encoding="utf-8",newline="\n")
            try:target.chmod(target.stat().st_mode | 0o111)
            except Exception:pass
        self.ubuntu_last_script_path=path
        return path

    def _ubuntu_generate_commands(self,command_kind=None,show_error=True,auto_save=True):
        if command_kind is not None:self.ubuntu_last_command_kind=str(command_kind)
        kind=getattr(self,"ubuntu_last_command_kind","h2_train")
        labels={
            "analysis_export":"轻量分析导出","training_export":"完整训练导出","build_h1_features":"F0–F5 构建",
            "h1_baseline":"H1 基线训练","h1_hw5":"MagTouch-RA-HW5","h1_core_matrix":"R13 核心4组",
            "build_h2_relative":"H2 相对数据构建","h2_train":"H2-R15 训练",
            "h2_feature_matrix":"R15 特征4组","h2_history_matrix":"R15 历史5组",
            "build_h2_spatial_labels":"V16 固定坐标标签构建","h2_spatial_train":"V16 双坐标训练",
            "h2_spatial_objective_matrix":"V16 三目标公平对照","h2_spatial_history_matrix":"V16 历史5组",
            "build_h2_unified_labels":"R18统一标签构建","validate_h2_unified_labels":"R18标签验证",
            "h2_unified_train":"R18统一XYZ+dXYZ+mode5训练",
        }
        try:
            cfg=normalize_ubuntu_training_config(self._ubuntu_training_config_from_ui())
            script=build_ubuntu_training_script(cfg,kind)
            self.ubuntu_script_text.delete("1.0","end");self.ubuntu_script_text.insert("1.0",script)
            saved=self._ubuntu_auto_save_script(script,kind) if auto_save else None
            suffix=f" | 已自动保存：{saved}" if saved is not None else " | 预览（尚未自动保存）"
            schedule="全资源并行" if cfg["gpu_schedule"]=="full_parallel" else "严格串行"
            self.ubuntu_status_var.set(f"已生成：{labels.get(kind,kind)} | GPU IDs={cfg['gpus']} | CPU={cfg['cpu_count']} | {schedule}{suffix}")
            self._schedule_settings_autosave()
            return script
        except Exception as exc:
            self.ubuntu_status_var.set("参数无效 / Invalid parameters")
            if show_error:messagebox.showwarning("Ubuntu 训练导出",str(exc))
            return None

    def _ubuntu_copy_commands(self):
        script=self._ubuntu_generate_commands(show_error=True)
        if script is None:return
        self.clipboard_clear();self.clipboard_append(script);self.update_idletasks()
        self.ubuntu_status_var.set(f"完整 Bash 脚本已复制；自动保存：{getattr(self,'ubuntu_last_script_path','--')}")

    def _ubuntu_save_script(self):
        script=self._ubuntu_generate_commands(show_error=True,auto_save=False)
        if script is None:return
        initial=HERE.parent/"generated_commands";initial.mkdir(parents=True,exist_ok=True)
        kind=getattr(self,"ubuntu_last_command_kind","h2_train").upper()
        path=filedialog.asksaveasfilename(title="保存 Ubuntu 训练 Bash 脚本",initialdir=str(initial),initialfile=f"RUN_UBUNTU_{kind}.sh",defaultextension=".sh",filetypes=[("Bash script","*.sh"),("All files","*.*")])
        if not path:return
        Path(path).write_text(script,encoding="utf-8",newline="\n")
        self.ubuntu_status_var.set(f"已保存 / Saved: {path}")

    def _build_printer_tab(self):
        tab=self.tab_printer

        # Scrollable body: the bottom global log remains fixed outside the Notebook.
        canvas=tk.Canvas(tab, highlightthickness=0)
        vsb=ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        body=ttk.Frame(canvas)
        win=canvas.create_window((0,0), window=body, anchor="nw")
        body.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(win, width=e.width))
        self.printer_scroll_canvas=canvas

        f=ttk.LabelFrame(body,text="JGAurora A5S USB / Marlin XYZ平台 / XYZ Stage（COM77 @ 115200 已验证）")
        f.pack(fill="x",padx=8,pady=8)
        pcfg=self.settings.get("printer",{})
        self.printer_port_var=tk.StringVar(value=str(pcfg.get("port","COM77")))
        self.printer_baud_var=tk.IntVar(value=int(pcfg.get("baud",115200)))
        self.printer_dry_var=tk.BooleanVar(value=bool(pcfg.get("dry_run",False)))
        self.contact_z_var=tk.DoubleVar(value=float(pcfg.get("surface_contact_z_mm",11.1)))
        self.safe_lift_var=tk.DoubleVar(value=float(pcfg.get("safe_lift_mm",3.0)))
        self.printer_pos_var=tk.StringVar(value="X/Y/Z: -- / -- / --")
        self.printer_home_state_var=tk.StringVar(value="Home状态: --")
        self.jog_step_var=tk.DoubleVar(value=float(pcfg.get("jog_step_mm",1.0)))
        self.xy_speed_var=tk.DoubleVar(value=float(pcfg.get("xy_speed_mm_s",3.0)))
        self.z_speed_var=tk.DoubleVar(value=float(pcfg.get("z_speed_mm_s",2.0)))
        self.auto_fan_off_var=tk.BooleanVar(value=bool(pcfg.get("auto_fan_off",True)))
        self.require_home_auto_var=tk.BooleanVar(value=bool(pcfg.get("require_home_before_auto_motion",True)))
        self.auto_save_parameters_var=tk.BooleanVar(value=bool(pcfg.get("auto_save_parameters",True)))
        self.printer_live_pos_var=tk.BooleanVar(value=bool(pcfg.get("live_position_refresh",True)))

        ttk.Label(f,text="串口 / COM").grid(row=0,column=0,padx=(6,2)); ttk.Entry(f,textvariable=self.printer_port_var,width=10).grid(row=0,column=1)
        ttk.Label(f,text="波特率 / Baud").grid(row=0,column=2,padx=(10,2)); ttk.Entry(f,textvariable=self.printer_baud_var,width=10).grid(row=0,column=3)
        ttk.Checkbutton(f,text="干运行 / DRY RUN",variable=self.printer_dry_var).grid(row=0,column=4,padx=8)
        ttk.Button(f,text="连接 / Connect",command=self.connect_printer).grid(row=0,column=5,padx=5)
        ttk.Button(f,text="读位置 / M114",command=self.query_printer_position).grid(row=0,column=6,padx=5)
        ttk.Label(f,textvariable=self.printer_pos_var,font=("Consolas",10,"bold")).grid(row=0,column=7,padx=10,sticky="w")
        ttk.Checkbutton(f,text="实时位置 / Live M114 (1 Hz)",variable=self.printer_live_pos_var,command=lambda:self.save_settings(silent=True)).grid(row=0,column=8,padx=(6,4),sticky="w")
        ttk.Checkbutton(f,text="参数自动保存 / Auto-save",variable=self.auto_save_parameters_var,command=lambda:self.save_settings(silent=True)).grid(row=1,column=0,columnspan=2,sticky="w",padx=6,pady=(4,5))
        ttk.Button(f,text="关风扇 / M107",command=self.printer_fan_off).grid(row=1,column=2,padx=5)
        ttk.Button(f,text="保存 / Save",command=self.save_settings).grid(row=1,column=3,padx=5)
        ttk.Button(f,text="重载 / Reload",command=self.reload_settings).grid(row=1,column=4,padx=5)
        ttk.Label(f,textvariable=self.printer_home_state_var).grid(row=1,column=5,columnspan=3,sticky="w",padx=8,pady=(4,5))

        # Homing / coordinate management
        homef=ttk.LabelFrame(body,text="回零与坐标 / Homing & Coordinate Setup")
        homef.pack(fill="x",padx=8,pady=(0,8))
        self.z_home_raise_enabled_var=tk.BooleanVar(value=bool(pcfg.get("home_z_raise_enabled",True)))
        self.z_home_raise_var=tk.DoubleVar(value=float(pcfg.get("home_z_raise_mm",20.0)))
        self.g92_z_var=tk.DoubleVar(value=float(pcfg.get("g92_z_mm",0.0)))
        ttk.Button(homef,text="Home X",command=lambda:self.printer_home("X")).grid(row=0,column=0,padx=4,pady=5)
        ttk.Button(homef,text="Home Y",command=lambda:self.printer_home("Y")).grid(row=0,column=1,padx=4)
        ttk.Button(homef,text="Home Z（先移开样品）",command=lambda:self.printer_home("Z")).grid(row=0,column=2,padx=4)
        ttk.Button(homef,text="Home XYZ",command=lambda:self.printer_home("XYZ")).grid(row=0,column=3,padx=5)
        ttk.Checkbutton(homef,text="自动实验前必须Home XYZ / Require Home",variable=self.require_home_auto_var).grid(row=0,column=4,columnspan=3,sticky="w",padx=(12,4))

        ttk.Checkbutton(homef,text="含Z回零后自动抬升 / Auto raise after Z Home",variable=self.z_home_raise_enabled_var).grid(row=1,column=0,columnspan=2,padx=(6,2),sticky="w")
        ttk.Label(homef,text="抬升到 / Raise to Z=").grid(row=1,column=2,sticky="e"); ttk.Entry(homef,textvariable=self.z_home_raise_var,width=8).grid(row=1,column=3,sticky="w")
        ttk.Label(homef,text="设置当前Z / Set current Z, G92=").grid(row=1,column=4,padx=(10,2),sticky="e")
        ttk.Entry(homef,textvariable=self.g92_z_var,width=8).grid(row=1,column=5,sticky="w")
        ttk.Button(homef,text="应用 / Apply",command=self.set_current_z_coordinate).grid(row=1,column=6,padx=5)
        ttk.Label(homef,text="说明 / Note：自动动作只保留 Home/限位保护；Contact Z 直接使用配置文件/界面参数，不强制再次标定。",
                  foreground="#1b5e20").grid(row=2,column=0,columnspan=7,sticky="w",padx=5,pady=(2,5))

        # Basic jog
        basic=ttk.LabelFrame(body,text="XYZ基本运动控制 / Basic Jog Control")
        basic.pack(fill="x",padx=8,pady=(0,8))
        ttk.Label(basic,text="步长 / Step (mm)").grid(row=0,column=0,padx=(6,2),pady=5)
        ttk.Combobox(basic,textvariable=self.jog_step_var,values=["0.05","0.1","0.2","0.5","1.0","2.0","5.0"],width=7,state="readonly").grid(row=0,column=1)
        ttk.Label(basic,text="XY速度 / XY speed (mm/s)").grid(row=0,column=2,padx=(12,2)); ttk.Entry(basic,textvariable=self.xy_speed_var,width=7).grid(row=0,column=3)
        ttk.Label(basic,text="Z速度 / Z speed (mm/s)").grid(row=0,column=4,padx=(12,2)); ttk.Entry(basic,textvariable=self.z_speed_var,width=7).grid(row=0,column=5)
        for j,(txt,axis,sgn) in enumerate([("X -","X",-1),("X +","X",1),("Y -","Y",-1),("Y +","Y",1),("Z +（远离样品）","Z",1),("Z -（靠近样品）","Z",-1)]):
            ttk.Button(basic,text=txt,command=lambda a=axis,s=sgn:self.printer_jog(a,s)).grid(row=1,column=j,padx=5,pady=6)

        # Sensor reference center
        center=ttk.LabelFrame(body,text="MagTouch传感器中心与接触面 / Sensor Center & Contact Surface（A5S实测）")
        center.pack(fill="x",padx=8,pady=(0,8))
        self.sensor_center_x_var=tk.DoubleVar(value=float(pcfg.get("sensor_center_x_mm",149.0)))
        self.sensor_center_y_var=tk.DoubleVar(value=float(pcfg.get("sensor_center_y_mm",147.0)))
        self.approach_clearance_var=tk.DoubleVar(value=float(pcfg.get("approach_clearance_mm",1.0)))
        ttk.Label(center,text="中心X / Center X").grid(row=0,column=0,padx=(6,2),pady=5); ttk.Entry(center,textvariable=self.sensor_center_x_var,width=9).grid(row=0,column=1)
        ttk.Label(center,text="中心Y / Center Y").grid(row=0,column=2,padx=(8,2)); ttk.Entry(center,textvariable=self.sensor_center_y_var,width=9).grid(row=0,column=3)
        ttk.Label(center,text="接触面Z / Contact Z").grid(row=0,column=4,padx=(8,2)); ttk.Entry(center,textvariable=self.contact_z_var,width=9).grid(row=0,column=5)
        ttk.Label(center,text="安全抬升 / Safe lift").grid(row=0,column=6,padx=(8,2)); ttk.Entry(center,textvariable=self.safe_lift_var,width=8).grid(row=0,column=7)
        ttk.Label(center,text="接触前间隙 / Approach gap").grid(row=0,column=8,padx=(8,2)); ttk.Entry(center,textvariable=self.approach_clearance_var,width=8).grid(row=0,column=9)
        ttk.Button(center,text="去中心安全位 / Center Safe",command=lambda:self.go_sensor_center("safe")).grid(row=1,column=0,columnspan=2,padx=5,pady=6,sticky="ew")
        ttk.Button(center,text="去中心接触前 / Center Approach",command=lambda:self.go_sensor_center("approach")).grid(row=1,column=2,columnspan=2,padx=5,sticky="ew")
        ttk.Button(center,text="到接触面Z / Go Contact Z",command=lambda:self.go_sensor_center("contact")).grid(row=1,column=4,columnspan=2,padx=5,sticky="ew")
        ttk.Button(center,text="释放到安全Z / Release",command=self.release_to_safe_z).grid(row=1,column=6,columnspan=2,padx=5,sticky="ew")
        ttk.Button(center,text="同步中心到Trial / Sync Center",command=self.sync_center_to_trial).grid(row=1,column=8,columnspan=2,padx=5,sticky="ew")
        ttk.Label(center,text="当前基准参数：中心(149,147)，上表面Z=11.1 mm。参数以配置文件/界面值为准；压入深度 = 接触面Z - Ztarget。",
                  foreground="#1b5e20").grid(row=2,column=0,columnspan=10,sticky="w",padx=6,pady=(0,5))

        # Point touch parameters
        point=ttk.LabelFrame(body,text="点触参数与动作测试 / Point Touch Setup")
        point.pack(fill="x",padx=8,pady=(0,8))
        ptcfg=self.settings.get("point_touch",{})
        self.point_depth_var=tk.DoubleVar(value=float(ptcfg.get("depth_mm",0.10)))
        self.point_hold_var=tk.DoubleVar(value=float(ptcfg.get("hold_s",0.50)))
        self.point_repeat_var=tk.IntVar(value=int(ptcfg.get("repeat",1)))
        self.point_interval_var=tk.DoubleVar(value=float(ptcfg.get("interval_s",0.50)))
        self.point_speed_var=tk.DoubleVar(value=float(ptcfg.get("z_speed_mm_s",0.50)))
        for c,(lab,var,w) in enumerate([("压入深度 / Depth (mm)",self.point_depth_var,8),("保持时间 / Hold (s)",self.point_hold_var,8),("重复次数 / Repeats",self.point_repeat_var,6),("间隔 / Interval (s)",self.point_interval_var,8),("Z速度 / Z speed (mm/s)",self.point_speed_var,8)]):
            ttk.Label(point,text=lab).grid(row=0,column=c*2,padx=(6 if c==0 else 10,2),pady=5); ttk.Entry(point,textvariable=var,width=w).grid(row=0,column=c*2+1)
        ttk.Button(point,text="单次点触 / Single Touch",command=lambda:self.run_point_touch(False)).grid(row=1,column=0,columnspan=2,padx=5,pady=6,sticky="ew")
        ttk.Button(point,text="重复点触 / Repeated Touch",command=lambda:self.run_point_touch(False,force_repeat=True)).grid(row=1,column=2,columnspan=2,padx=5,sticky="ew")
        ttk.Button(point,text="记录磁场+点触 / Record + Touch",command=lambda:self.run_point_touch(True)).grid(row=1,column=4,columnspan=2,padx=5,sticky="ew")
        ttk.Button(point,text="停止实验序列 / Stop sequence",command=self.printer_quick_stop).grid(row=1,column=6,columnspan=2,padx=5,sticky="ew")
        ttk.Button(point,text="关闭风扇 / M107",command=self.printer_fan_off).grid(row=1,column=8,columnspan=2,padx=5,sticky="ew")

        # Experiment action generator
        exp=ttk.LabelFrame(body,text="实验动作设置 / Experiment Action Setup（灰色项 = 当前动作不用 / grey = unused）")
        exp.pack(fill="x",padx=8,pady=(0,8))
        ecfg=self.settings.get("stage_experiment",{})
        action_code=str(ecfg.get("action","press"))
        self.stage_action_var=tk.StringVar(value=STAGE_ACTION_LABELS.get(action_code,STAGE_ACTION_LABELS["press"]))
        self.stage_indent_var=tk.DoubleVar(value=float(ecfg.get("indent_mm",0.20)))
        self.stage_dx_var=tk.DoubleVar(value=float(ecfg.get("dx_mm",2.0)))
        self.stage_dy_var=tk.DoubleVar(value=float(ecfg.get("dy_mm",2.0)))
        self.stage_speed_var=tk.DoubleVar(value=float(ecfg.get("speed_mm_s",1.0)))
        self.stage_z_speed_var=tk.DoubleVar(value=float(ecfg.get("z_speed_mm_s",self.settings.get("point_touch",{}).get("z_speed_mm_s",0.50))))
        self.stage_hold_var=tk.DoubleVar(value=float(ecfg.get("hold_s",0.5)))
        self.stage_repeat_var=tk.IntVar(value=int(ecfg.get("repeat",1)))
        self.stage_depths_var=tk.StringVar(value=str(ecfg.get("z_sweep_depths_mm","0.10,0.20,0.30,0.50")))
        self.grid_half_x_var=tk.DoubleVar(value=float(ecfg.get("grid_half_x_mm",4.0)))
        self.grid_half_y_var=tk.DoubleVar(value=float(ecfg.get("grid_half_y_mm",4.0)))
        self.grid_step_var=tk.DoubleVar(value=float(ecfg.get("grid_step_mm",2.0)))
        self.stage_xy_sweep_var=tk.StringVar(value=str(ecfg.get("xy_sweep_values_mm","-3,-2,-1,0,1,2,3")))
        self.stage_angle_count_var=tk.IntVar(value=int(ecfg.get("angle_count",8)))
        self.stage_circle_radius_var=tk.DoubleVar(value=float(ecfg.get("circle_radius_mm",2.0)))
        self.stage_circle_points_var=tk.IntVar(value=int(ecfg.get("circle_points",24)))
        self.stage_param_widgets={}

        def add_param(key, label_text, var, row, col, width=10, colspan=1):
            lab=ttk.Label(exp,text=label_text); lab.grid(row=row,column=col,padx=(7,2),pady=4,sticky="e")
            ent=ttk.Entry(exp,textvariable=var,width=width); ent.grid(row=row,column=col+1,columnspan=colspan,padx=(0,8),pady=4,sticky="w")
            self.stage_param_widgets[key]=(lab,ent)
            return lab,ent

        ttk.Label(exp,text="实验动作 / Action").grid(row=0,column=0,padx=(7,2),pady=5,sticky="e")
        act_cb=ttk.Combobox(exp,textvariable=self.stage_action_var,values=list(STAGE_ACTION_LABELS.values()),width=27,state="readonly")
        act_cb.grid(row=0,column=1,padx=(0,10),sticky="w")
        act_cb.bind("<<ComboboxSelected>>",lambda _e:self._update_stage_param_states())
        add_param("indent","压入 / Indent (mm)",self.stage_indent_var,0,2)
        add_param("dx","X位移 / dx (mm)",self.stage_dx_var,0,4)

        add_param("dy","Y位移 / dy (mm)",self.stage_dy_var,1,0)
        add_param("xy_speed","XY速度 / XY speed (mm/s)",self.stage_speed_var,1,2)
        add_param("z_speed","Z速度 / Z speed (mm/s)",self.stage_z_speed_var,1,4)

        add_param("hold","保持 / Hold (s)",self.stage_hold_var,2,0)
        add_param("repeat","重复 / Repeats",self.stage_repeat_var,2,2)
        lab=ttk.Label(exp,text="Z阶梯深度 / Z sweep depths (mm)"); lab.grid(row=2,column=4,padx=(7,2),pady=4,sticky="e")
        ent=ttk.Entry(exp,textvariable=self.stage_depths_var,width=27); ent.grid(row=2,column=5,padx=(0,8),sticky="w")
        self.stage_param_widgets["z_depths"]=(lab,ent)

        add_param("grid_half_x","网格半宽X / Grid half-X (mm)",self.grid_half_x_var,3,0)
        add_param("grid_half_y","网格半宽Y / Grid half-Y (mm)",self.grid_half_y_var,3,2)
        add_param("grid_step","网格步长 / Grid step (mm)",self.grid_step_var,3,4)
        add_param("xy_values","XY阶梯值 / Sweep values (mm)",self.stage_xy_sweep_var,4,0,width=28)
        add_param("angle_count","方向数 / Angle count",self.stage_angle_count_var,4,2)
        add_param("circle_radius","圆半径 / Circle radius (mm)",self.stage_circle_radius_var,4,4)
        add_param("circle_points","圆周点数 / Circle points",self.stage_circle_points_var,5,0)

        self.stage_param_hint_var=tk.StringVar(value="")
        ttk.Label(exp,textvariable=self.stage_param_hint_var,foreground="#1b5e20",wraplength=1250,justify="left").grid(row=6,column=0,columnspan=6,sticky="w",padx=8,pady=(2,5))
        ttk.Button(exp,text="动作测试 / Test Motion",command=lambda:self.run_stage_experiment(False)).grid(row=7,column=0,padx=5,pady=6,sticky="ew")
        ttk.Button(exp,text="记录磁场+动作 / Record + Motion",command=lambda:self.run_stage_experiment(True)).grid(row=7,column=1,columnspan=2,padx=5,sticky="ew")
        ttk.Button(exp,text="终止实验序列 / Stop sequence",command=self.printer_quick_stop).grid(row=7,column=3,padx=5,sticky="ew")
        ttk.Button(exp,text="释放安全Z / Release",command=self.release_to_safe_z).grid(row=7,column=4,columnspan=2,padx=5,sticky="ew")
        self._update_stage_param_states()

        # Stop/fan controls inside page; global stop controls are also always visible in the header.
        stop=ttk.LabelFrame(body,text="停止与振动控制 / Stop & Vibration Control")
        stop.pack(fill="x",padx=8,pady=(0,8))
        ttk.Button(stop,text="停止实验序列 / Stop sequence",command=self.printer_quick_stop).pack(side="left",padx=6,pady=6)
        ttk.Button(stop,text="关闭可控风扇 / M107",command=self.printer_fan_off).pack(side="left",padx=6)
        ttk.Checkbutton(stop,text="Trial前自动M107 / Auto fan-off before Trial",variable=self.auto_fan_off_var).pack(side="left",padx=10)
        ttk.Label(stop,text="急停请直接物理断电 / Emergency: switch printer power OFF",foreground="#b71c1c").pack(side="left",padx=14)

        # Manual console
        jog=ttk.LabelFrame(body,text="手动G-code / Manual Console")
        jog.pack(fill="x",padx=8,pady=(0,8))
        self.gcode_var=tk.StringVar(value="M114")
        ttk.Entry(jog,textvariable=self.gcode_var,width=70).grid(row=0,column=0,padx=5,pady=5)
        ttk.Button(jog,text="发送 / Send",command=self.send_manual_gcode).grid(row=0,column=1,padx=5)
        ttk.Button(jog,text="M400 已禁用 / Disabled",state="disabled").grid(row=0,column=2,padx=5)
        ttk.Button(jog,text="关风扇 / M107",command=self.printer_fan_off).grid(row=0,column=3,padx=5)
        ttk.Button(jog,text="读位置 / M114",command=self.query_printer_position).grid(row=0,column=4,padx=5)

        ttk.Label(body,text=(
            "当前阶段可以完全不启用摄像头：Trial 会保存 magnetic_raw.csv + stage_commands.csv + events.csv。"
            "打印机坐标只作为 commanded motion；后续加入视觉后再产生更高等级 observed deformation 真值。"
        ),wraplength=1450,justify="left").pack(anchor="w",padx=12,pady=(0,10))

    def _build_full_plan_tab(self):
        """Build one-page full experiment plan generator and one-click scheduler.

        One queue row = one independent Trial directory.  `repeats` stays the user's
        original meaning: repeated complete motions *inside that Trial*.
        """
        tab=self.tab_plan
        tab.columnconfigure(0,weight=1); tab.rowconfigure(1,weight=1)

        top=ttk.LabelFrame(tab,text="全部实验数据采集计划 / Full Experiment Acquisition Plan")
        top.grid(row=0,column=0,sticky="ew",padx=7,pady=7)
        self.plan_preset_var=tk.StringVar(value="Pilot / 初步训练")
        self.plan_name_var=tk.StringVar(value="MagTouch_H1_Pilot")
        self.plan_status_var=tk.StringVar(value="尚未生成计划 / No plan generated")
        ttk.Label(top,text="预设 / Preset").grid(row=0,column=0,padx=4,pady=3,sticky="e")
        preset_cb=ttk.Combobox(top,textvariable=self.plan_preset_var,state="readonly",width=36,
                      values=["Quick / 快速验证","Pilot / 初步训练","Core / 核心训练",
                              "FixedSilicone / 固定硅胶0.2-3.0",
                              "PureFit-2mm / 20×20连续XY",
                              "PureFit-1mm / 20×20高密度",
                              "Backtest / 预测回测","H3-19C / 圆形可靠性",
                             "H3-5x5 / 方形动态","H3-flot / 平面弱信号"])
        preset_cb.grid(row=0,column=1,padx=4,sticky="w")
        preset_cb.bind("<<ComboboxSelected>>",lambda _e:self._apply_plan_preset_fields(self.plan_preset_var.get()))
        ttk.Label(top,text="计划名称 / Plan name").grid(row=0,column=2,padx=(14,4),sticky="e")
        ttk.Entry(top,textvariable=self.plan_name_var,width=22).grid(row=0,column=3,padx=4,sticky="w")
        ttk.Button(top,text="生成推荐计划 / Generate",command=self.generate_full_plan).grid(row=0,column=4,padx=5)
        ttk.Button(top,text="保存计划 / Save",command=self.save_full_plan).grid(row=0,column=5,padx=3)
        ttk.Button(top,text="载入计划 / Load",command=self.load_full_plan).grid(row=0,column=6,padx=3)
        ttk.Label(top,textvariable=self.plan_status_var,foreground="#1b5e20").grid(row=1,column=0,columnspan=7,sticky="w",padx=6,pady=(3,1))

        # User-editable generator knobs. Presets fill these; users may modify then regenerate.
        gen=ttk.LabelFrame(top,text="计划生成参数 / Generator Parameters（修改后点 Generate）")
        gen.grid(row=2,column=0,columnspan=7,sticky="ew",padx=3,pady=(7,3))
        self.plan_position_pattern_var=tk.StringVar(value="Cross5 / 十字5点")
        self.plan_z_list_var=tk.StringVar(value="0.2,0.5")
        self.plan_speed_list_var=tk.StringVar(value="0.5,1.0")
        self.plan_slide_list_var=tk.StringVar(value="2.5")
        self.plan_default_repeats_var=tk.IntVar(value=4)
        self.plan_angle_count_var=tk.IntVar(value=8)
        labels=[
            ("位置分布 / Positions",self.plan_position_pattern_var,0,["Center / 中心","Cross5 / 十字5点","Grid3x3 / 3×3网格"]),
            ("Z深度列表 / Z depths mm",self.plan_z_list_var,2,None),
            ("速度列表 / Speeds mm/s",self.plan_speed_list_var,4,None),
            ("切向幅值 / Slide amplitudes mm",self.plan_slide_list_var,6,None),
        ]
        for text,var,col,vals in labels:
            ttk.Label(gen,text=text).grid(row=0,column=col,padx=(5,2),pady=4,sticky="e")
            if vals:
                ttk.Combobox(gen,textvariable=var,state="readonly",values=vals,width=18).grid(row=0,column=col+1,padx=2,sticky="w")
            else:
                ttk.Entry(gen,textvariable=var,width=18).grid(row=0,column=col+1,padx=2,sticky="w")
        ttk.Label(gen,text="Trial内重复 / Repeats in Trial").grid(row=1,column=0,padx=(5,2),sticky="e")
        ttk.Spinbox(gen,from_=1,to=100,textvariable=self.plan_default_repeats_var,width=8).grid(row=1,column=1,sticky="w")
        ttk.Label(gen,text="角度数 / Angle count").grid(row=1,column=2,padx=(8,2),sticky="e")
        ttk.Spinbox(gen,from_=4,to=36,textvariable=self.plan_angle_count_var,width=8).grid(row=1,column=3,sticky="w")
        self.plan_group_vars={}
        groups=[("baseline","基线 / Baseline"),("z_sweep","Z阶梯 / Z sweep"),("press","法向 / Press"),
                ("slide","XY切向 / Slide"),("angle","多角度 / Angle"),("compound","复合 / Compound"),
                ("hold","加载保持释放 / Hold-release")]
        for i,(code,label) in enumerate(groups):
            v=tk.BooleanVar(value=True); self.plan_group_vars[code]=v
            ttk.Checkbutton(gen,text=label,variable=v).grid(row=1+(i//4),column=4+(i%4),sticky="w",padx=3,pady=2)

        body=ttk.PanedWindow(tab,orient="horizontal")
        body.grid(row=1,column=0,sticky="nsew",padx=7,pady=(0,7))
        qf=ttk.LabelFrame(body,text="计划队列 / Plan Queue（1行 = 1个独立Trial；Rep = Trial内部动作重复）")
        sf=ttk.LabelFrame(body,text="执行控制 / Run Control")
        body.add(qf,weight=5); body.add(sf,weight=2)
        qf.columnconfigure(0,weight=1); qf.rowconfigure(0,weight=1)
        cols=("idx","group","mode","xy","z","move","speed","rep","status")
        self.plan_tree=ttk.Treeview(qf,columns=cols,show="headings",selectmode="browse")
        headings={"idx":"#","group":"组 / Group","mode":"实验 / Mode","xy":"执行XY / Trial XY","z":"Z / mm","move":"dx/dy / mm","speed":"速度 / Speed","rep":"Rep","status":"状态 / Status"}
        widths={"idx":42,"group":110,"mode":125,"xy":120,"z":68,"move":110,"speed":78,"rep":50,"status":105}
        for c in cols:self.plan_tree.heading(c,text=headings[c]);self.plan_tree.column(c,width=widths[c],anchor="center")
        ysb=ttk.Scrollbar(qf,orient="vertical",command=self.plan_tree.yview); self.plan_tree.configure(yscrollcommand=ysb.set)
        self.plan_tree.grid(row=0,column=0,sticky="nsew"); ysb.grid(row=0,column=1,sticky="ns")
        self.plan_tree.bind("<Double-1>",lambda _e:self.edit_selected_plan_item())
        qbtn=ttk.Frame(qf); qbtn.grid(row=1,column=0,columnspan=2,sticky="ew",pady=4)
        ttk.Button(qbtn,text="编辑选中 / Edit",command=self.edit_selected_plan_item).pack(side="left",padx=3)
        ttk.Button(qbtn,text="复制 / Duplicate",command=self.duplicate_selected_plan_item).pack(side="left",padx=3)
        ttk.Button(qbtn,text="删除 / Delete",command=self.delete_selected_plan_item).pack(side="left",padx=3)
        ttk.Button(qbtn,text="全部重置为待运行 / Reset status",command=self.reset_full_plan_status).pack(side="left",padx=3)

        sf.columnconfigure(0,weight=1)
        self.plan_progress_var=tk.StringVar(value="IDLE")
        ttk.Label(sf,textvariable=self.plan_progress_var,font=("Segoe UI",11,"bold"),wraplength=330,justify="left").grid(row=0,column=0,sticky="ew",padx=8,pady=7)
        self.plan_progress=ttk.Progressbar(sf,mode="determinate",maximum=100)
        self.plan_progress.grid(row=1,column=0,sticky="ew",padx=8,pady=5)
        ttk.Button(sf,text="一键开始全部采集 / START ALL",command=self.start_full_plan).grid(row=2,column=0,sticky="ew",padx=8,pady=(12,4))
        self.plan_pause_btn=ttk.Button(sf,text="Trial间暂停 / Pause between Trials",command=self.toggle_full_plan_pause)
        self.plan_pause_btn.grid(row=3,column=0,sticky="ew",padx=8,pady=4)
        ttk.Button(sf,text="当前Trial后停止 / Stop after current Trial",command=self.stop_full_plan_after_current).grid(row=4,column=0,sticky="ew",padx=8,pady=4)
        tk.Button(sf,text="停止实验序列 / STOP SEQUENCE",command=self.printer_quick_stop,bg="#ef6c00",fg="white",font=("Segoe UI",10,"bold")).grid(row=5,column=0,sticky="ew",padx=8,pady=4)
        ttk.Separator(sf).grid(row=6,column=0,sticky="ew",padx=8,pady=10)
        ttk.Label(sf,text="数据归档 / Archive\n每次 START ALL 都创建一个独立 Plan_* 文件夹；本次计划的 plan_definition / progress / run_meta 与所有 Trial_* 全部放在其中。\n\n启用实时预测时，每个 Trial 前自动移动到安全非接触位置，稳定后重新采集动态 B0；B0 成功才开始该 Trial。不会在受力动作中连续更新 B0。\n\n建议顺序 / Recommended\n1) Quick 快速验证\n2) Pilot 初步训练\n3) 检查神经网络可学习性\n4) Core 核心训练\n5) 视觉真值加入后再做论文级多Session数据\n\n当前无摄像头时，标签属于 commanded motion，不要称为真实柔性层 observed deformation。",
                  wraplength=330,justify="left").grid(row=7,column=0,sticky="nw",padx=8,pady=4)

        # If a saved current plan exists, restore it; otherwise generate the Pilot preset.
        self.after(100,lambda:self.load_full_plan(path=self.full_plan_path,silent=True) if self.full_plan_path.exists() else self.generate_full_plan())

    @staticmethod
    def _num_list(text):
        return [float(v.strip()) for v in str(text).replace(";",",").split(",") if v.strip()]

    def _plan_positions(self,pattern):
        if str(pattern).startswith("Center"): offsets=[(0.0,0.0)]
        elif str(pattern).startswith("Grid3x3"): offsets=[(x,y) for y in (-4.0,0.0,4.0) for x in (-4.0,0.0,4.0)]
        else: offsets=[(0.0,0.0),(-4.0,0.0),(4.0,0.0),(0.0,-4.0),(0.0,4.0)]
        x0=float(self.sensor_center_x_var.get()); y0=float(self.sensor_center_y_var.get())
        return [(x0+dx,y0+dy) for dx,dy in offsets]

    def _plan_item(self,group,action,x,y,indent=0.2,dx=0.0,dy=0.0,speed=1.0,zspeed=0.5,hold=0.5,repeats=1,repeat_interval=0.5,**extra):
        # recovery_s is separate from steady hold for H3 dynamics.  Legacy plans
        # without this field retain the old behaviour (recovery duration = hold).
        recovery_s=extra.pop("recovery_s",hold)
        d={
            "group":group,"action":action,"center_x_mm":float(x),"center_y_mm":float(y),
            "contact_z_mm":float(self.contact_z_var.get()),"safe_lift_mm":float(self.safe_lift_var.get()),
            "approach_clearance_mm":float(self.approach_clearance_var.get()),
            "indent_mm":float(indent),"dx_mm":float(dx),"dy_mm":float(dy),"speed_mm_s":float(speed),"z_speed_mm_s":float(zspeed),
            "hold_s":float(hold),"recovery_s":float(recovery_s),"repeats":int(repeats),"repeat_interval_s":float(repeat_interval),
            "z_depths_mm":"","xy_sweep_values_mm":"","angle_count":int(self.plan_angle_count_var.get()),
            "circle_radius_mm":2.0,"circle_points":24,"grid_half_x_mm":4.0,"grid_half_y_mm":4.0,"grid_step_mm":2.0,
            "status":"PENDING",
        }
        d.update(extra); return d

    def _apply_plan_preset_fields(self,preset):
        p=str(preset)
        if p.startswith("FixedSilicone"):
            self.plan_name_var.set("MagTouch_FixedSilicone_121_DepthSweep_0p2_3p0")
            self.plan_position_pattern_var.set("Center / 中心")
            self.plan_z_list_var.set(
                "0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0"
            )
            self.plan_speed_list_var.set("2.0")
            self.plan_slide_list_var.set("2.0")
            self.plan_default_repeats_var.set(1)
            self.plan_angle_count_var.set(8)
            self.plan_status_var.set(
                "固定硅胶专用协议：中心(使用当前校准值)、20×20 mm、2 mm间隔、11×11=121点；"
                "0.2~3.0 mm按0.2 mm递增，共15个独立Trial/1815次接触。"
                "固定 flat_d6mm + silicone_v4 + magnetic_layer_v4，速度/Z速度2 mm/s、保持0.5 s、间隔1.2 s。"
            )
        elif p.startswith("PureFit-1mm"):
            self.plan_name_var.set("MagTouch_PureFit_20x20_1mm_Dense")
            self.plan_position_pattern_var.set("Center / 中心"); self.plan_z_list_var.set("0.2,0.5,0.8")
            self.plan_speed_list_var.set("10.0"); self.plan_slide_list_var.set("1.0"); self.plan_default_repeats_var.set(2); self.plan_angle_count_var.set(8)
            self.plan_status_var.set(
                "R40沿用全系列固定协议：20×20 mm、1 mm间隔、21×21=441位置、3深度、2次反向重复，共2646次有效接触；"
                "适用于19C、5×5、flot1650-4.5 mm、flot1650-2.5 mm，但每种磁层必须独立采集和建模。"
                "2 mm网格121点预声明为旧模型对照，其余320点用于升级前冻结验证；分21个Trial逐块重采B0。"
            )
        elif p.startswith("PureFit-2mm"):
            self.plan_name_var.set("MagTouch_PureFit_20x20_2mm_Interpolation")
            self.plan_position_pattern_var.set("Center / 中心"); self.plan_z_list_var.set("0.2,0.5,0.8")
            self.plan_speed_list_var.set("10.0"); self.plan_slide_list_var.set("2.0"); self.plan_default_repeats_var.set(2); self.plan_angle_count_var.set(8)
            self.plan_status_var.set(
                "R27固定协议：20×20 mm、2 mm间隔、11×11=121位置、3深度、2次反向重复；"
                "4 mm锚点36个用于拟合，另外85点仅用于插值验证。分9个Trial逐块重采B0，并插入中心漂移参考。"
            )
        elif p.startswith("Backtest"):
            self.plan_name_var.set("MagTouch_H3_LivePrediction_Backtest")
            self.plan_position_pattern_var.set("Grid3x3 / 3×3网格"); self.plan_z_list_var.set("0.2,0.5,0.8")
            self.plan_speed_list_var.set("5.0"); self.plan_slide_list_var.set("3.0"); self.plan_default_repeats_var.set(4); self.plan_angle_count_var.set(8)
            self.plan_status_var.set(
                "预测回测固定协议：46个独立 Trial；必须先载入模型并建立 B0。每个 Trial 同步保存 live_predictions.csv，且标记 exclude_from_training。"
            )
        elif p.startswith("Quick"):
            self.plan_name_var.set("MagTouch_H0_Quick")
            self.plan_position_pattern_var.set("Center / 中心"); self.plan_z_list_var.set("0.2,0.5")
            self.plan_speed_list_var.set("0.5,1.0"); self.plan_slide_list_var.set("2.0"); self.plan_default_repeats_var.set(4); self.plan_angle_count_var.set(8)
        elif p.startswith("Core"):
            self.plan_name_var.set("MagTouch_H2_Core")
            self.plan_position_pattern_var.set("Grid3x3 / 3×3网格"); self.plan_z_list_var.set("0.2,0.5,0.8")
            self.plan_speed_list_var.set("0.5,1.0,2.0"); self.plan_slide_list_var.set("1.5,3.0"); self.plan_default_repeats_var.set(4); self.plan_angle_count_var.set(8)
        elif p.startswith("H3-19C"):
            self.plan_name_var.set("MagTouch_H3_CrossSession_Dynamics-19C")
            self.plan_position_pattern_var.set("Cross5 / 十字5点"); self.plan_z_list_var.set("0.3,0.5,0.8,1.0")
            self.plan_speed_list_var.set("5.0"); self.plan_slide_list_var.set("3.0"); self.plan_default_repeats_var.set(3); self.plan_angle_count_var.set(8)
            self.plan_status_var.set("H3-19C固定补充协议：48 Trials；建议使用19C安全测量Profile（历史经验 GAIN=5）。通用Generator参数仅作显示，不改变H3固定矩阵。")
        elif p.startswith("H3-5x5"):
            self.plan_name_var.set("MagTouch_H3_CrossSession_Dynamics-5-5")
            self.plan_position_pattern_var.set("Cross5 / 十字5点"); self.plan_z_list_var.set("0.3,0.5,0.6")
            self.plan_speed_list_var.set("0.5,1,2,3,5,8,10"); self.plan_slide_list_var.set("3.0"); self.plan_default_repeats_var.set(3); self.plan_angle_count_var.set(8)
            self.plan_status_var.set("H3-5x5固定补充协议：57 Trials；建议沿用5x5健康Profile（历史经验 GAIN=5）。通用Generator参数仅作显示。")
        elif p.startswith("H3-flot"):
            self.plan_name_var.set("MagTouch_H3_CrossSession_Dynamics-flot1650")
            self.plan_position_pattern_var.set("Cross5 / 十字5点"); self.plan_z_list_var.set("0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50")
            self.plan_speed_list_var.set("0.5,1,3,5,10"); self.plan_slide_list_var.set("3.0"); self.plan_default_repeats_var.set(1); self.plan_angle_count_var.set(8)
            self.plan_status_var.set("H3-flot固定补充协议：99 Trials；弱信号样件建议沿用高灵敏度且不饱和Profile（历史经验 GAIN=7）。通用Generator参数仅作显示。")
        else:
            self.plan_name_var.set("MagTouch_H1_Pilot")
            self.plan_position_pattern_var.set("Cross5 / 十字5点"); self.plan_z_list_var.set("0.2,0.5")
            self.plan_speed_list_var.set("0.5,1.0"); self.plan_slide_list_var.set("2.5"); self.plan_default_repeats_var.set(4); self.plan_angle_count_var.set(8)

    def _h3_current_profile_gain(self):
        try:
            pid,p=self._selected_mag_profile()
            gain=(p.get("mlx90393",{}) or {}).get("gain")
            return (pid,int(gain)) if gain is not None else (None,None)
        except Exception:
            return None,None

    def _generate_fixed_silicone_depth_plan_items(self,center):
        """Build the fixed-silicone 121-point depth ladder requested for V4.

        Every depth is an independent Trial so it has its own recorder files,
        recoverable queue status, and (when live inference is enabled) dynamic B0.
        Hardware identity fields are frozen into every row and copied into Trial
        metadata when the row starts; this prevents a stale silicone_v3 UI value
        from silently labelling the new five-face-fixed specimen.
        """
        cx,cy=float(center[0]),float(center[1])
        depths=tuple(round(i*0.2,1) for i in range(1,16))
        items=[]
        for depth in depths:
            items.append(self._plan_item(
                f"FIXED_SILICONE_Z{int(round(depth*100)):03d}",
                "grid_points",cx,cy,
                indent=depth,speed=2.0,zspeed=2.0,hold=0.5,
                repeats=1,repeat_interval=1.2,
                grid_half_x_mm=10.0,grid_half_y_mm=10.0,grid_step_mm=2.0,
                grid_reference_interval=0,
                indenter_id="flat_d6mm",
                material_id="silicone_v4",
                magnetic_layer_id="magnetic_layer_v4",
                required_magnetic_profile_id=str(self.active_mag_profile_id),
                required_tactile_material_id="silicone_v4",
                required_magnetic_layer_id="magnetic_layer_v4",
                study_phase="FIXED_SILICONE_121_DEPTH_SWEEP",
                research_role="fixed_silicone_continuous_depth_training",
                boundary_condition="five_face_fixed",
                depth_ladder_start_mm=0.2,
                depth_ladder_stop_mm=3.0,
                depth_ladder_step_mm=0.2,
                grid_total_positions=121,
                planned_contacts=121,
                requires_dynamic_b0=False,
                b0_policy="dynamic per-Trial B0 when live inference is enabled; raw acquisition remains allowed",
            ))
        if len(items)!=15:
            raise RuntimeError(
                f"FixedSilicone protocol internal count error: expected 15, got {len(items)}"
            )
        return items

    def _generate_purefit_2mm_plan_items(self, center):
        """Build the fixed R27 continuous-XY interpolation acquisition protocol.

        The 11x11 grid is split into three Y blocks at every depth.  Every block is
        an independent Trial so the existing Full-Plan dynamic-B0 mechanism samples
        a fresh unloaded baseline before roughly 33--44 grid contacts instead of only
        once for the complete 121-point scan.  Repeat 2 is executed in the reverse
        path order by ``_execute_stage_experiment`` to reduce path-order bias.

        The 4 mm anchor lattice is declared *before acquisition*: offsets
        {-10,-6,-2,2,6,10} on both axes are calibration points (36); the remaining
        85 locations are a locked unseen interpolation test set.  GRID_POINT events
        carry this role so downstream builders must not silently mix the two sets.
        """
        cx,cy=float(center[0]),float(center[1])
        y_blocks=(
            (-10.0,-8.0,-6.0,-4.0),
            (-2.0,0.0,2.0,4.0),
            (6.0,8.0,10.0),
        )
        items=[]
        for depth in (0.2,0.5,0.8):
            for block_index,ys in enumerate(y_blocks,1):
                item=self._plan_item(
                    f"PF2_Z{int(round(depth*100)):03d}_B{block_index}",
                    "grid_points",cx,cy,
                    indent=depth,speed=10.0,zspeed=2.0,hold=0.20,
                    repeats=2,repeat_interval=0.20,
                    grid_half_x_mm=10.0,grid_half_y_mm=10.0,grid_step_mm=2.0,
                    grid_y_offsets_mm=",".join(f"{v:g}" for v in ys),
                    grid_reference_interval=22,
                    grid_anchor_origin_mm=-10.0,grid_anchor_step_mm=4.0,
                    grid_split_policy="predeclared_anchor4mm_vs_unseen2mm",
                    study_phase="PUREFIT_CONTINUOUS_XY_2MM",
                    research_role="continuous_xy_interpolation_dataset",
                    calibration_role="anchor_4mm_only",
                    validation_role="unseen_2mm_interleaved_only",
                    grid_total_positions=121,grid_anchor_positions=36,
                    grid_unseen_positions=85,grid_total_contacts=726,
                    exclude_unseen_from_fit=True,
                    do_not_merge_split_before_final_evaluation=True,
                    requires_dynamic_b0=True,
                    b0_policy="fresh unloaded B0 before every Y block Trial",
                    drift_reference_policy="extra center contact every 22 planned grid contacts; QC only",
                    block_index=block_index,block_count=3,
                )
                items.append(item)
        if len(items)!=9:
            raise RuntimeError(f"PureFit-2mm protocol internal count error: expected 9, got {len(items)}")
        return items

    def _generate_purefit_1mm_plan_items(self,center):
        """Build a 21x21 grid split into safe three-row Trials at three depths."""
        cx,cy=float(center[0]),float(center[1])
        all_y=[float(v) for v in range(-10,11)]
        y_blocks=[tuple(all_y[i:i+3]) for i in range(0,len(all_y),3)]
        items=[]
        for depth in (0.2,0.5,0.8):
            for block_index,ys in enumerate(y_blocks,1):
                items.append(self._plan_item(
                    f"PF1_Z{int(round(depth*100)):03d}_B{block_index}","grid_points",cx,cy,
                    indent=depth,speed=10.0,zspeed=2.0,hold=0.20,repeats=2,repeat_interval=0.20,
                    grid_half_x_mm=10.0,grid_half_y_mm=10.0,grid_step_mm=1.0,
                    grid_y_offsets_mm=",".join(f"{v:g}" for v in ys),grid_reference_interval=42,
                    grid_anchor_origin_mm=-10.0,grid_anchor_step_mm=2.0,
                    grid_split_policy="predeclared_anchor2mm_vs_unseen1mm",
                    study_phase="PUREFIT_DENSE_XY_1MM",research_role="dense_continuous_xy_dataset",
                    calibration_role="anchor_2mm_only",validation_role="unseen_1mm_interleaved_only",
                    grid_total_positions=441,grid_anchor_positions=121,grid_unseen_positions=320,
                    grid_total_contacts=2646,exclude_unseen_from_fit=True,
                    do_not_merge_split_before_final_evaluation=True,requires_dynamic_b0=True,
                    b0_policy="fresh unloaded B0 before every three-row Trial",
                    drift_reference_policy="extra center contact every 42 planned grid contacts; QC only",
                    block_index=block_index,block_count=len(y_blocks),
                ))
        if len(items)!=21:
            raise RuntimeError(f"PureFit-1mm protocol internal count error: expected 21, got {len(items)}")
        return items

    def _generate_prediction_backtest_plan_items(self, center):
        """Independent online-model validation matrix; never use it for training."""
        cx,cy=(float(center[0]),float(center[1]))
        positions=[(cx+dx,cy+dy) for dy in (-4.0,0.0,4.0) for dx in (-4.0,0.0,4.0)]
        points={"C":(cx,cy),"L":(cx-4.0,cy),"R":(cx+4.0,cy)}
        items=[]
        def add(group,action,x,y,**kwargs):
            tags={
                "study_phase":"H3_BACKTEST",
                "research_role":"independent_live_prediction_backtest",
                "record_live_predictions":True,
                "exclude_from_training":True,
                "prediction_folder_tag":"PRED_BACKTEST",
                "prediction_output_policy":"case_sensitive_raw_plus_visual_adapter",
            }
            tags.update(kwargs)
            items.append(self._plan_item(group,action,x,y,**tags))

        # Three no-contact drift references around the spatial validation block.
        for name in ("C","L","R"):
            add("BT_A_BASELINE","none",*points[name],hold=20.0,repeats=1,
                repeat_interval=0.0,backtest_position=name)
        # Path/history sensitivity at left, center and right.
        for name in ("L","C","R"):
            add("BT_B_Z_SWEEP","z_sweep",*points[name],indent=1.0,zspeed=0.5,
                hold=0.3,repeats=4,z_depths_mm="0.10,0.20,0.30,0.50,0.80,1.00",
                backtest_position=name)
        # Absolute XYZ generalization over a 3x3 position grid and three depths.
        for x,y in positions:
            for z in (0.2,0.5,0.8):
                add("BT_C_ABSOLUTE_GRID","press",x,y,indent=z,speed=5.0,zspeed=5.0,
                    hold=0.5,repeats=4)
        # Signed tangential directions and compound quadrants at a shared normal load.
        for action,dx,dy in (("slide_x",3.0,0.0),("slide_x",-3.0,0.0),
                             ("slide_y",0.0,3.0),("slide_y",0.0,-3.0)):
            add("BT_D_SIGNED_SHEAR",action,cx,cy,indent=0.6,dx=dx,dy=dy,
                speed=3.0,zspeed=0.5,hold=0.3,repeats=4)
        for dx,dy in ((2.0,2.0),(2.0,-2.0),(-2.0,2.0),(-2.0,-2.0)):
            add("BT_E_COMPOUND","compound",cx,cy,indent=0.6,dx=dx,dy=dy,
                speed=3.0,zspeed=0.5,hold=0.3,repeats=4)
        for z in (0.3,0.6):
            for hold in (0.5,2.0):
                add("BT_F_HOLD","load_hold_release",cx,cy,indent=z,speed=1.0,
                    zspeed=0.5,hold=hold,recovery_s=5.0,repeats=4)
        add("BT_G_ANGLE","angle_scan",cx,cy,indent=0.6,dx=3.0,speed=3.0,
            zspeed=0.5,hold=0.15,repeats=4,angle_count=8)
        if len(items)!=46:
            raise RuntimeError(f"Backtest protocol internal count error: expected 46, got {len(items)}")
        return items

    def _generate_h3_plan_items(self,preset,center):
        """Build the fixed H3 cross-specimen supplement matrix.

        H3 intentionally shares physical actions, not an artificial common GAIN.
        Each specimen keeps the acquisition profile appropriate for its real field
        magnitude.  All downstream comparisons use calibrated physical units (uT).
        """
        p=str(preset)
        if p.startswith("H3-19C"):
            specimen="19C"; expected_gain=5
        elif p.startswith("H3-5x5"):
            specimen="5x5"; expected_gain=5
        elif p.startswith("H3-flot"):
            specimen="flot1650"; expected_gain=7
        else:
            raise ValueError(f"Not an H3 preset: {preset}")

        cx,cy=float(center[0]),float(center[1])
        P={
            "C":(cx,cy), "L":(cx-4.0,cy), "R":(cx+4.0,cy),
            "D":(cx,cy-4.0), "U":(cx,cy+4.0),
        }
        items=[]
        def add(block,group,action,x,y,**kw):
            tags={
                "study_phase":"H3", "specimen_family":specimen,
                "research_block":block, "recommended_gain_sel":expected_gain,
                "comparison_unit":"uT",
            }
            tags.update(kw)
            items.append(self._plan_item(group,action,x,y,**tags))

        # ---------------- shared reference protocol: 24 Trials ----------------
        # Correct center baseline + spatial no-contact references.
        for k in range(3):
            add("COMMON_BASELINE","H3A_BASELINE","none",*P["C"],hold=30.0,repeats=1,repeat_interval=0,h3_repeat_index=k+1)
        for pos_name in ("L","R","D","U"):
            add("COMMON_BASELINE","H3A_BASELINE","none",*P[pos_name],hold=30.0,repeats=1,repeat_interval=0,h3_position=pos_name)

        # Same physical depth ruler for cross-session/cross-specimen comparison.
        for k in range(3):
            add("COMMON_Z_SWEEP","H3B_ZSWEEP","z_sweep",*P["C"],indent=1.0,zspeed=0.5,hold=0.3,repeats=1,
                z_depths_mm="0.10,0.20,0.30,0.50,0.80,1.00",h3_repeat_index=k+1)

        # Standard normal press: 3 depths x 2 independent Trials.
        for z in (0.2,0.5,0.8):
            for k in range(2):
                add("COMMON_PRESS","H3C_PRESS","press",*P["C"],indent=z,speed=5.0,zspeed=5.0,hold=0.5,repeats=3,repeat_interval=1.0,h3_repeat_index=k+1)

        # Signed shear: +X, -X, +Y, -Y.
        add("COMMON_SHEAR","H3D_SHEAR","slide_x",*P["C"],indent=0.6,dx=+3.0,speed=3.0,zspeed=0.5,hold=0.3,repeats=3)
        add("COMMON_SHEAR","H3D_SHEAR","slide_x",*P["C"],indent=0.6,dx=-3.0,speed=3.0,zspeed=0.5,hold=0.3,repeats=3)
        add("COMMON_SHEAR","H3D_SHEAR","slide_y",*P["C"],indent=0.6,dy=+3.0,speed=3.0,zspeed=0.5,hold=0.3,repeats=3)
        add("COMMON_SHEAR","H3D_SHEAR","slide_y",*P["C"],indent=0.6,dy=-3.0,speed=3.0,zspeed=0.5,hold=0.3,repeats=3)

        # Hold/recovery uses separate 20 s recovery instead of reusing steady hold.
        for z in (0.3,0.6):
            add("COMMON_RECOVERY","H3E_HOLD","load_hold_release",*P["C"],indent=z,speed=1.0,zspeed=0.5,hold=5.0,recovery_s=20.0,repeats=1)

        # Two standardized compound trajectories.
        add("COMMON_COMPOUND","H3F_COMPOUND","compound",*P["C"],indent=0.6,dx=+2.0,dy=+2.0,speed=3.0,zspeed=0.5,hold=0.3,repeats=2)
        add("COMMON_COMPOUND","H3F_COMPOUND","compound",*P["C"],indent=0.6,dx=-2.0,dy=-2.0,speed=3.0,zspeed=0.5,hold=0.3,repeats=2)

        # ---------------- specimen-specific blocks ----------------
        if specimen=="19C":
            # 24 Trials: original vulnerable spatial sectors, now under safe profile.
            for pos_name in ("C","R","D"):
                for z in (0.3,0.5,0.8,1.0):
                    for k in range(2):
                        add("19C_RELIABILITY_BOUNDARY","H3R_19C","press",*P[pos_name],indent=z,speed=5.0,zspeed=5.0,hold=0.5,repeats=3,repeat_interval=1.0,
                            h3_position=pos_name,h3_repeat_index=k+1)

        elif specimen=="5x5":
            # 14 Trials: loading-rate scan at the same final commanded state.
            for sp in (0.5,1.0,2.0,3.0,5.0,8.0,10.0):
                for k in range(2):
                    add("5X5_RATE_SCAN","H3DYN_RATE","press",*P["C"],indent=0.5,speed=sp,zspeed=sp,hold=0.5,repeats=3,repeat_interval=1.0,h3_repeat_index=k+1)
            # 10 Trials: steady hold vs recovery time constants.
            for z in (0.3,0.6):
                for h in (0.5,2.0,5.0,10.0,30.0):
                    add("5X5_HOLD_RECOVERY","H3DYN_HOLD","load_hold_release",*P["C"],indent=z,speed=1.0,zspeed=0.5,hold=h,recovery_s=20.0,repeats=1)
            # 9 Trials: same/related final states reached through different Z histories.
            paths=(
                ("A_up","0.30,0.60,0.00"),
                ("B_down","0.60,0.30,0.00"),
                ("C_reload","0.60,0.00,0.60,0.00"),
            )
            for path_name,path_text in paths:
                for k in range(3):
                    add("5X5_PATH_DEPENDENCE","H3DYN_PATH","z_path",*P["C"],indent=0.6,speed=1.0,zspeed=1.0,hold=1.0,recovery_s=20.0,repeats=1,
                        z_depths_mm=path_text,h3_path_name=path_name,h3_repeat_index=k+1)

        elif specimen=="flot1650":
            # 40 Trials: dense shallow-depth boundary with independent repeats.
            for z in (0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50):
                for k in range(5):
                    add("FLOT_SHALLOW_BOUNDARY","H3W_SHALLOW","press",*P["C"],indent=z,speed=5.0,zspeed=5.0,hold=0.5,repeats=1,h3_repeat_index=k+1)
            # 20 Trials: weak signal x loading speed.
            for z in (0.20,0.30):
                for sp in (0.5,1.0,3.0,5.0,10.0):
                    for k in range(2):
                        add("FLOT_WEAK_RATE","H3W_RATE","press",*P["C"],indent=z,speed=sp,zspeed=sp,hold=0.5,repeats=1,h3_repeat_index=k+1)
            # 10 Trials: weak-state hold and long recovery.
            for z in (0.20,0.30):
                for h in (0.5,2.0,5.0,10.0,30.0):
                    add("FLOT_WEAK_HISTORY","H3W_HOLD","load_hold_release",*P["C"],indent=z,speed=1.0,zspeed=0.5,hold=h,recovery_s=20.0,repeats=1)
            # 5 Trials: ten cycles per near-threshold depth for detection probability.
            for z in (0.10,0.15,0.20,0.25,0.30):
                add("FLOT_DETECTION_REPEAT","H3W_REPEAT","press_cycle",*P["C"],indent=z,speed=3.0,zspeed=3.0,hold=0.3,repeats=10,repeat_interval=1.0)

        expected={"19C":48,"5x5":57,"flot1650":99}[specimen]
        if len(items)!=expected:
            raise RuntimeError(f"H3 protocol internal count error: {specimen} expected {expected}, got {len(items)}")

        # Inform, but never silently change, the actual acquisition profile.
        pid,actual_gain=self._h3_current_profile_gain()
        if actual_gain is not None and actual_gain!=expected_gain:
            messagebox.showwarning(
                "H3 magnetic profile check",
                f"{specimen} H3计划建议 GAIN_SEL={expected_gain}（依据既有实测工作区），但当前磁层Profile {pid!r} 为 GAIN_SEL={actual_gain}。\n\n"
                "计划已经生成，但系统不会自动修改GAIN。请在真正START ALL前确认你希望使用的磁层Profile；跨磁体比较使用换算后的真实 μT，而不是比较GAIN。"
            )
        return items

    def generate_full_plan(self):
        preset=self.plan_preset_var.get()
        if str(preset).startswith("FixedSilicone"):
            try:
                center=(float(self.sensor_center_x_var.get()),float(self.sensor_center_y_var.get()))
                self.full_plan_reference_center=center
                self.full_plan_reference_contact_z=float(self.contact_z_var.get())
                items=self._generate_fixed_silicone_depth_plan_items(center)
                for item in items:
                    self._validate_action_cfg(item)
                    self._validate_full_plan_required_hardware(item)
                self.full_plan_items=items
                self.refresh_full_plan_tree();self._update_full_plan_stats()
                self.save_full_plan(path=self.full_plan_path,silent=True)
            except Exception as e:
                messagebox.showerror("固定硅胶计划生成失败 / FixedSilicone plan",str(e))
            return
        if str(preset).startswith("PureFit-1mm"):
            try:
                center=(float(self.sensor_center_x_var.get()),float(self.sensor_center_y_var.get()))
                self.full_plan_reference_center=center
                self.full_plan_reference_contact_z=float(self.contact_z_var.get())
                items=self._generate_purefit_1mm_plan_items(center)
                for item in items:self._validate_action_cfg(item)
                self.full_plan_items=items
                self.refresh_full_plan_tree();self._update_full_plan_stats()
                self.save_full_plan(path=self.full_plan_path,silent=True)
            except Exception as e:
                messagebox.showerror("1 mm高密度计划生成失败 / PureFit-1mm plan",str(e))
            return
        if str(preset).startswith("PureFit-2mm"):
            try:
                center=(float(self.sensor_center_x_var.get()),float(self.sensor_center_y_var.get()))
                self.full_plan_reference_center=center
                self.full_plan_reference_contact_z=float(self.contact_z_var.get())
                items=self._generate_purefit_2mm_plan_items(center)
                for item in items:self._validate_action_cfg(item)
                self.full_plan_items=items
                self.refresh_full_plan_tree();self._update_full_plan_stats()
                self.save_full_plan(path=self.full_plan_path,silent=True)
            except Exception as e:
                messagebox.showerror("2 mm插值计划生成失败 / PureFit-2mm plan",str(e))
            return
        if str(preset).startswith("Backtest"):
            try:
                center=(float(self.sensor_center_x_var.get()),float(self.sensor_center_y_var.get()))
                self.full_plan_reference_center=center
                self.full_plan_reference_contact_z=float(self.contact_z_var.get())
                items=self._generate_prediction_backtest_plan_items(center)
                for item in items:self._validate_action_cfg(item)
                self.full_plan_items=items
                self.refresh_full_plan_tree();self._update_full_plan_stats()
                self.save_full_plan(path=self.full_plan_path,silent=True)
            except Exception as e:
                messagebox.showerror("预测回测计划生成失败 / Backtest plan",str(e))
            return
        # H3 presets are fixed scientific supplement matrices.  They always snapshot
        # the CURRENT calibrated Printer center/contact geometry at Generate time.
        if str(preset).startswith("H3-"):
            try:
                center=(float(self.sensor_center_x_var.get()),float(self.sensor_center_y_var.get()))
                self.full_plan_reference_center=(float(center[0]),float(center[1]))
                self.full_plan_reference_contact_z=float(self.contact_z_var.get())
                items=self._generate_h3_plan_items(preset,center)
                for it in items:self._validate_action_cfg(it)
                self.full_plan_items=items
                self.refresh_full_plan_tree(); self._update_full_plan_stats()
                self.save_full_plan(path=self.full_plan_path,silent=True)
            except Exception as e:
                messagebox.showerror("H3计划生成失败 / H3 plan generation",str(e))
            return
        # Preset selection fills recommended fields, but Generate respects any edits the
        # user has made afterwards.
        try:
            zvals=self._num_list(self.plan_z_list_var.get()); speeds=self._num_list(self.plan_speed_list_var.get()); amps=[abs(v) for v in self._num_list(self.plan_slide_list_var.get()) if abs(v)>1e-9]
            if not zvals or not speeds or not amps: raise ValueError("Z/speed/slide lists cannot be empty")
        except Exception as e:
            messagebox.showerror("计划参数 / Plan parameters",str(e)); return
        positions=self._plan_positions(self.plan_position_pattern_var.get())
        # IMPORTANT: the Plan reference center is the calibrated Printer/Sensor Center,
        # NOT positions[0].  For Grid3x3, positions[0] is the (-4,-4) corner.
        # Using positions[0] here previously moved A_BASELINE (and any center-only
        # branch) to the lower-left corner instead of the configured center.
        center=(float(self.sensor_center_x_var.get()),float(self.sensor_center_y_var.get()))
        self.full_plan_reference_center=(float(center[0]),float(center[1]))
        self.full_plan_reference_contact_z=float(self.contact_z_var.get())
        reps=max(1,int(self.plan_default_repeats_var.get())); items=[]
        g=self.plan_group_vars
        quick=str(preset).startswith("Quick"); core=str(preset).startswith("Core")

        # Baseline: separate Trials make drift/zero checks explicit.
        if g["baseline"].get():
            nbase=1 if quick else (3 if core else 2); hold=10.0 if quick else 30.0
            for _ in range(nbase): items.append(self._plan_item("A_BASELINE","none",*center,hold=hold,repeats=1,repeat_interval=0))
        # Z sweep: one Trial per physical position; all depths remain in the same Trial.
        if g["z_sweep"].get():
            zs="0.10,0.20,0.30,0.50" if quick else ("0.10,0.20,0.30,0.50,0.80,1.00" if core else "0.10,0.20,0.30,0.50,0.80")
            for x,y in positions: items.append(self._plan_item("B_Z_SWEEP","z_sweep",x,y,zspeed=0.5,hold=0.3,repeats=2 if not core else 3,z_depths_mm=zs))
        # Press: systematic z x speed mapping.
        if g["press"].get():
            use_pos=positions; use_z=zvals; use_sp=speeds
            for x,y in use_pos:
                for z in use_z:
                    for sp in use_sp: items.append(self._plan_item("C_PRESS","press",x,y,indent=z,speed=sp,zspeed=sp,hold=0.5,repeats=reps))
        # Tangential +/-X and +/-Y. Core deliberately uses 5-position cross to limit combinatorial explosion.
        if g["slide"].get():
            slide_pos=positions if not core else self._plan_positions("Cross5 / 十字5点")
            slide_z=[0.3,0.6] if not quick else [0.3]
            slide_sp=[1.0,3.0] if core else ([1.0] if quick else [1.5])
            for action in ("slide_x","slide_y"):
                for x,y in slide_pos:
                    for z in slide_z:
                        for a in amps:
                            for sign in (-1.0,1.0):
                                for sp in slide_sp:
                                    items.append(self._plan_item("D_SLIDE",action,x,y,indent=z,dx=sign*a if action=="slide_x" else 0.0,dy=sign*a if action=="slide_y" else 0.0,speed=sp,zspeed=0.5,hold=0.3,repeats=3 if not core else 4))
        # 360-degree directional observability.
        if g["angle"].get():
            angle_pos=[center] if quick else (self._plan_positions("Cross5 / 十字5点") if core else positions)
            angle_z=[0.5] if quick else [0.3,0.6]
            angle_amp=amps[-1]
            angle_sp=[1.0] if not core else [1.0,3.0]
            for x,y in angle_pos:
                for z in angle_z:
                    for sp in angle_sp:
                        items.append(self._plan_item("E_ANGLE","angle_scan",x,y,indent=z,dx=angle_amp,speed=sp,zspeed=0.5,hold=0.15,repeats=1,angle_count=int(self.plan_angle_count_var.get())))
        # Compound XYZ with four XY quadrants.
        if g["compound"].get():
            comp_pos=[center] if quick else (self._plan_positions("Cross5 / 十字5点") if core else positions)
            comp_z=[0.5] if quick else [0.3,0.6]
            comp_sp=[1.0] if not core else [1.0,3.0]
            a=1.5 if quick else 2.0
            for x,y in comp_pos:
                for z in comp_z:
                    for sx,sy in ((a,a),(a,-a),(-a,a),(-a,-a)):
                        for sp in comp_sp:
                            items.append(self._plan_item("F_COMPOUND","compound",x,y,indent=z,dx=sx,dy=sy,speed=sp,zspeed=0.5,hold=0.3,repeats=2 if not core else 3))
        # Viscoelastic/dynamic recovery proxy without a force sensor.
        if g["hold"].get():
            hold_pos=[center] if quick else (self._plan_positions("Cross5 / 十字5点") if core else positions)
            hold_z=[0.5] if quick else [0.3,0.6]
            hold_times=[2.0] if quick else [0.5,2.0]
            for x,y in hold_pos:
                for z in hold_z:
                    for h in hold_times: items.append(self._plan_item("G_HOLD","load_hold_release",x,y,indent=z,speed=1.0,zspeed=0.5,hold=h,repeats=2 if not core else 3))
        self.full_plan_items=items
        self.refresh_full_plan_tree(); self._update_full_plan_stats()
        self.save_full_plan(path=self.full_plan_path,silent=True)

    def _plan_action_events(self,item):
        r=max(1,int(item.get("repeats",1))); a=item.get("action")
        if a in ("z_sweep","z_path"): return r*max(1,len(self._num_list(item.get("z_depths_mm",""))))
        if a=="angle_scan": return r*max(1,int(item.get("angle_count",8)))
        if a=="circle_scan": return r*max(1,int(item.get("circle_points",24)))
        if a=="grid_points":
            try:
                xs=self._grid_offsets(item.get("grid_half_x_mm",4),item.get("grid_step_mm",2))
                y_text=str(item.get("grid_y_offsets_mm","")).strip()
                ys=self._num_list(y_text) if y_text else self._grid_offsets(item.get("grid_half_y_mm",4),item.get("grid_step_mm",2))
                n=r*len(xs)*len(ys)
                interval=max(0,int(item.get("grid_reference_interval",0)))
                if interval>0:n+=r*((len(xs)*len(ys)-1)//interval)
                return n
            except Exception:return r
        return r

    def _plan_reference_geometry(self):
        """Return the frozen Full-Plan reference center/contact geometry.

        New plans store this snapshot at Generate time.  Legacy plans that do not
        contain the snapshot fall back to the current Printer settings.
        """
        c=getattr(self,"full_plan_reference_center",None)
        try:
            if c is None or len(c)!=2: raise ValueError
            cx=float(c[0]); cy=float(c[1])
        except Exception:
            cx=float(self.sensor_center_x_var.get()); cy=float(self.sensor_center_y_var.get())
        cz=getattr(self,"full_plan_reference_contact_z",None)
        try:
            if cz is None: raise ValueError("No stored contact Z")
            cz=float(cz)
        except Exception: cz=float(self.contact_z_var.get())
        return (cx,cy),cz

    def _update_full_plan_stats(self):
        n=len(self.full_plan_items); ev=sum(self._plan_action_events(i) for i in self.full_plan_items)
        hz=1000.0/max(1.0,float(self.mag_interval_var.get()))
        # This is explicitly a rough estimate; true duration is written from measured timestamps.
        est_h=max(0.01,(ev*4.0 + n*2.0)/3600.0)
        (cx,cy),cz=self._plan_reference_geometry()
        self.plan_status_var.set(f"计划 / Plan: {self.plan_name_var.get()} | 参考中心=({cx:.1f},{cy:.1f}) | ContactZ={cz:.2f} | 独立Trials={n} | 约动作事件={ev} | 当前磁采样≈{hz:.1f} Hz | 粗略时长≈{est_h:.1f} h")

    def refresh_full_plan_tree(self):
        if not hasattr(self,"plan_tree"): return
        self.plan_tree.delete(*self.plan_tree.get_children())
        for idx,it in enumerate(self.full_plan_items,1):
            mv=f"{it.get('dx_mm',0):g}/{it.get('dy_mm',0):g}"
            self.plan_tree.insert("", "end", iid=str(idx-1), values=(idx,it.get("group",""),it.get("action",""),f"{it.get('center_x_mm',0):.1f},{it.get('center_y_mm',0):.1f}",f"{it.get('indent_mm',0):g}",mv,f"{it.get('speed_mm_s',0):g}",it.get("repeats",1),it.get("status","PENDING")))

    def _selected_plan_index(self):
        sel=self.plan_tree.selection() if hasattr(self,"plan_tree") else ()
        return int(sel[0]) if sel else None

    def edit_selected_plan_item(self):
        idx=self._selected_plan_index()
        if idx is None or not (0<=idx<len(self.full_plan_items)): return
        item=self.full_plan_items[idx]; win=tk.Toplevel(self); win.title(f"编辑计划条目 / Edit Plan Item #{idx+1}"); win.transient(self); win.grab_set()
        fields=[("action",str),("center_x_mm",float),("center_y_mm",float),("indent_mm",float),("dx_mm",float),("dy_mm",float),("speed_mm_s",float),("z_speed_mm_s",float),("hold_s",float),("recovery_s",float),("repeats",int),("repeat_interval_s",float),("z_depths_mm",str),("xy_sweep_values_mm",str),("angle_count",int),("circle_radius_mm",float),("circle_points",int),("grid_y_offsets_mm",str),("grid_reference_interval",int)]
        vars_={}
        for r,(k,tp) in enumerate(fields):
            ttk.Label(win,text=k).grid(row=r,column=0,sticky="e",padx=6,pady=2); v=tk.StringVar(value=str(item.get(k,""))); vars_[k]=(v,tp); ttk.Entry(win,textvariable=v,width=28).grid(row=r,column=1,padx=6,pady=2)
        def save():
            try:
                for k,(v,tp) in vars_.items(): item[k]=tp(v.get())
                self._validate_action_cfg(item); item["status"]="PENDING"; self.refresh_full_plan_tree(); self._update_full_plan_stats(); self.save_full_plan(path=self.full_plan_path,silent=True); win.destroy()
            except Exception as e: messagebox.showerror("Edit",str(e),parent=win)
        ttk.Button(win,text="保存 / Save",command=save).grid(row=len(fields),column=0,columnspan=2,pady=8)

    def duplicate_selected_plan_item(self):
        idx=self._selected_plan_index()
        if idx is None:return
        cp=json.loads(json.dumps(self.full_plan_items[idx])); cp["status"]="PENDING"; self.full_plan_items.insert(idx+1,cp); self.refresh_full_plan_tree();self._update_full_plan_stats()

    def delete_selected_plan_item(self):
        idx=self._selected_plan_index()
        if idx is None:return
        self.full_plan_items.pop(idx); self.refresh_full_plan_tree();self._update_full_plan_stats()

    def reset_full_plan_status(self):
        for i in self.full_plan_items:i["status"]="PENDING"
        self.refresh_full_plan_tree();self._update_full_plan_stats()

    def save_full_plan(self,path=None,silent=False):
        try:
            if path is None:
                path=filedialog.asksaveasfilename(title="保存全部实验计划",defaultextension=".json",filetypes=[("JSON","*.json")],initialdir=str((HERE.parent/"config").resolve()),initialfile=f"{_safe_slug(self.plan_name_var.get(),50)}.json")
                if not path:return
            path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
            (cx,cy),cz=self._plan_reference_geometry()
            obj={"format":"MagTouch Full Acquisition Plan V1.2","plan_name":self.plan_name_var.get(),"saved_at":datetime.now().isoformat(),
                 "reference_center_printer_mm":[cx,cy],"reference_contact_z_mm":cz,"items":self.full_plan_items}
            path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8")
            if not silent:self.log(f"Full plan saved: {path}")
        except Exception as e:
            if not silent:messagebox.showerror("Save Plan",str(e))

    def load_full_plan(self,path=None,silent=False):
        try:
            if path is None:
                path=filedialog.askopenfilename(title="载入全部实验计划",filetypes=[("JSON","*.json")],initialdir=str((HERE.parent/"config").resolve()))
                if not path:return
            obj=json.loads(Path(path).read_text(encoding="utf-8")); items=obj.get("items",[])
            if not isinstance(items,list):raise ValueError("items must be a list")
            self.full_plan_items=items; self.plan_name_var.set(obj.get("plan_name",Path(path).stem))
            ref=obj.get("reference_center_printer_mm")
            try:
                self.full_plan_reference_center=(float(ref[0]),float(ref[1]))
            except Exception:
                # Backward compatibility for legacy plans: old files had no plan-level
                # reference center.  Use the current calibrated Printer center rather
                # than items[0], because Grid3x3 items[0] is a corner.
                self.full_plan_reference_center=(float(self.sensor_center_x_var.get()),float(self.sensor_center_y_var.get()))
            try:self.full_plan_reference_contact_z=float(obj.get("reference_contact_z_mm"))
            except Exception:self.full_plan_reference_contact_z=float(self.contact_z_var.get())
            self.refresh_full_plan_tree();self._update_full_plan_stats()
            if not silent:self.log(f"Full plan loaded: {path}")
        except Exception as e:
            if not silent:messagebox.showerror("Load Plan",str(e))

    def _apply_full_plan_item_ui(self,item):
        self.mode_var.set(TRIAL_MODE_LABELS.get(item["action"],TRIAL_MODE_LABELS["press"])); self._update_trial_param_states()
        # Apply the geometry frozen into this Plan item before Trial metadata is
        # snapshotted.  This keeps recorded metadata and commanded motion identical
        # even if Printer settings were edited after the Plan was generated.
        if "contact_z_mm" in item:self.contact_z_var.set(float(item["contact_z_mm"]))
        if "safe_lift_mm" in item:self.safe_lift_var.set(float(item["safe_lift_mm"]))
        if "approach_clearance_mm" in item:self.approach_clearance_var.set(float(item["approach_clearance_mm"]))
        self.x_var.set(float(item["center_x_mm"]));self.y_var.set(float(item["center_y_mm"]));self.z_var.set(float(item.get("indent_mm",0)))
        self.dx_var.set(float(item.get("dx_mm",0)));self.dy_var.set(float(item.get("dy_mm",0)));self.speed_var.set(float(item.get("speed_mm_s",1)));self.hold_var.set(float(item.get("hold_s",0.5)))
        self.trial_repeat_var.set(max(1,int(item.get("repeats",1))));self.repeat_interval_var.set(max(0.0,float(item.get("repeat_interval_s",0))))
        self.trial_z_depths_var.set(str(item.get("z_depths_mm","")));self.trial_xy_sweep_var.set(str(item.get("xy_sweep_values_mm","")))
        self.trial_angle_count_var.set(max(1,int(item.get("angle_count",8))));self.trial_circle_radius_var.set(float(item.get("circle_radius_mm",2.0)));self.trial_circle_points_var.set(int(item.get("circle_points",24)))
        self.trial_grid_half_x_var.set(float(item.get("grid_half_x_mm",4.0)));self.trial_grid_half_y_var.set(float(item.get("grid_half_y_mm",4.0)));self.trial_grid_step_var.set(float(item.get("grid_step_mm",2.0)))
        if "indenter_id" in item:self.indenter_var.set(str(item["indenter_id"]))
        if "material_id" in item:self.material_var.set(str(item["material_id"]))
        if "magnetic_layer_id" in item:self.maglayer_var.set(str(item["magnetic_layer_id"]))
        self._update_trial_param_states();self._update_auto_summary()

    def _validate_full_plan_required_hardware(self,item):
        """Reject a fixed-hardware row when the active magnetic profile is wrong."""
        required_profile=str(item.get("required_magnetic_profile_id","")).strip()
        required_layer=str(item.get("required_magnetic_layer_id","")).strip()
        required_material=str(item.get("required_tactile_material_id","")).strip()
        if not required_profile and not required_layer and not required_material:
            return
        profile=self.maglayer_profiles.get(self.active_mag_profile_id,{}) or {}
        actual_layer=str(profile.get("magnetic_layer_id","")).strip()
        actual_material=str(profile.get("tactile_material_id","")).strip()
        mismatches=[]
        if required_profile and self.active_mag_profile_id!=required_profile:
            mismatches.append(
                f"Profile要求 {required_profile}，当前为 {self.active_mag_profile_id}"
            )
        if required_layer and actual_layer!=required_layer:
            mismatches.append(f"磁层要求 {required_layer}，当前Profile为 {actual_layer or '(空)'}")
        if required_material and actual_material!=required_material:
            mismatches.append(f"材料要求 {required_material}，当前Profile为 {actual_material or '(空)'}")
        if mismatches:
            raise ValueError(
                f"固定硅胶计划与当前磁层Profile {self.active_mag_profile_id!r} 不一致："
                + "；".join(mismatches)
                + "。请先在‘磁层配置’页选择V4 Profile，再生成/启动计划。"
            )

    def _set_plan_item_status(self,idx,status):
        if 0<=idx<len(self.full_plan_items):self.full_plan_items[idx]["status"]=status
        self.refresh_full_plan_tree()

    def _create_plan_run_folder(self, pending_indices):
        """Create one self-contained folder for this Full Plan execution.

        Layout:
          data/Session/Plan_YYYYMMDD_HHMMSS_<plan_name>__mag_<profile>/
              plan_definition.json
              plan_run_meta.json
              plan_progress.json
              Trial_000123_.../
              Trial_000124_.../
        """
        session_dir=self._session_dir()
        session_dir.mkdir(parents=True,exist_ok=True)
        stamp=datetime.now().strftime("%Y%m%d_%H%M%S")
        profile_id=str(self.active_mag_profile_id or "").strip()
        magnet_suffix=_magnet_filename_suffix(profile_id or self.maglayer_var.get())
        pname_budget=max(16,60-len(magnet_suffix)-2)
        pname=_safe_slug(self.plan_name_var.get().strip() or "FullPlan",pname_budget)
        base=f"Plan_{stamp}_{pname}__{magnet_suffix}"
        plan_dir=session_dir/base
        n=2
        while plan_dir.exists():
            plan_dir=session_dir/f"Plan_{stamp}_{pname}_{n:02d}__{magnet_suffix}"; n+=1
        plan_dir.mkdir(parents=True,exist_ok=False)
        run_id=plan_dir.name
        definition={
            "format":"MagTouch Full Acquisition Plan Run V1.2",
            "plan_name":self.plan_name_var.get(),
            "plan_run_id":run_id,
            "magnetic_layer_profile_id":profile_id,
            "magnet_filename_suffix":magnet_suffix,
            "created_local":datetime.now().isoformat(timespec="seconds"),
            "session_id":self.session_var.get().strip(),
            "data_root":str(Path(self.data_root_var.get()).expanduser()),
            "pending_queue_indices":[int(i)+1 for i in pending_indices],
            "reference_center_printer_mm":list(self._plan_reference_geometry()[0]),
            "reference_contact_z_mm":self._plan_reference_geometry()[1],
            "dynamic_b0_per_trial":bool(self.full_plan_dynamic_b0_enabled),
            "dynamic_b0_policy":"robust pre-action unloaded-window calibration; never updated continuously during contact motion",
            "items":self.full_plan_items,
        }
        (plan_dir/"plan_definition.json").write_text(json.dumps(definition,ensure_ascii=False,indent=2),encoding="utf-8")
        # Also preserve the exact current-plan JSON as used at start time.
        try:
            if self.full_plan_path.exists():
                shutil.copy2(self.full_plan_path,plan_dir/"plan_source_snapshot.json")
        except Exception:
            pass
        meta={
            "format":"MagTouch Full Plan Run Metadata V1.2",
            "app":APP_NAME,
            "plan_run_id":run_id,
            "magnetic_layer_profile_id":profile_id,
            "magnet_filename_suffix":magnet_suffix,
            "plan_name":self.plan_name_var.get(),
            "session_id":self.session_var.get().strip(),
            "status":"RUNNING",
            "started_local":datetime.now().isoformat(timespec="seconds"),
            "started_perf_ns":time.perf_counter_ns(),
            "planned_trials":len(pending_indices),
            "completed_trials":0,
            "aborted_trials":0,
            # Backward-compatible fields now describe the frozen Plan reference
            # geometry rather than whichever Printer settings happen to be visible
            # when START ALL is pressed.
            "center_printer_mm":list(self._plan_reference_geometry()[0]),
            "contact_z_mm":self._plan_reference_geometry()[1],
            "reference_center_printer_mm":list(self._plan_reference_geometry()[0]),
            "reference_contact_z_mm":self._plan_reference_geometry()[1],
            "dynamic_b0_per_trial":bool(self.full_plan_dynamic_b0_enabled),
            "dynamic_b0_policy":"safe Z + trial XY + 2 s settle + max(2 s, model history) B0 window",
        }
        meta_path=plan_dir/"plan_run_meta.json"
        meta_path.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
        self.active_plan_run_dir=plan_dir
        self.active_plan_run_id=run_id
        self.active_plan_run_meta_path=meta_path
        return plan_dir

    def _write_plan_run_progress(self, completed, total, current_idx=None, status="RUNNING", error=None):
        plan_dir=self.active_plan_run_dir
        if not plan_dir:return
        payload={
            "plan_run_id":self.active_plan_run_id,
            "plan_name":self.plan_name_var.get(),
            "status":status,
            "updated_local":datetime.now().isoformat(timespec="seconds"),
            "completed_trials":int(completed),
            "planned_trials":int(total),
            "current_queue_index":None if current_idx is None else int(current_idx)+1,
            "queue_status":[{"index":i+1,"group":it.get("group"),"action":it.get("action"),"status":it.get("status","PENDING")} for i,it in enumerate(self.full_plan_items)],
        }
        if error:payload["error"]=str(error)
        try:(plan_dir/"plan_progress.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
        except Exception:pass
        # Keep run metadata short and human-friendly.
        if self.active_plan_run_meta_path:
            try:
                meta=json.loads(self.active_plan_run_meta_path.read_text(encoding="utf-8"))
                meta["status"]=status;meta["completed_trials"]=int(completed);meta["updated_local"]=payload["updated_local"]
                if status in ("COMPLETED","STOPPED","ABORTED","ERROR"):
                    meta["finished_local"]=payload["updated_local"]
                if error:meta["error"]=str(error)
                self.active_plan_run_meta_path.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding="utf-8")
            except Exception:pass

    def _capture_full_plan_dynamic_b0(self,item,queue_index,total):
        """Capture one fresh unloaded B0 immediately before a predicted Plan Trial.

        B0 is deliberately refreshed only in an explicit non-contact window. A
        continuous/EMA update during press, hold or shear would absorb the tactile
        signal itself and invalidate slow-response and hysteresis measurements.
        """
        action=str(item.get("action","press"))
        x=float(item["center_x_mm"]);y=float(item["center_y_mm"])
        contact=float(item["contact_z_mm"])
        safe=contact+abs(float(item["safe_lift_mm"]))
        xy_speed=max(0.2,float(item.get("speed_mm_s",1.0)))
        z_speed=max(0.2,float(item.get("z_speed_mm_s",0.5)))
        settle_s=2.0
        if self.batch_abort_event.is_set() or self.motion_abort_event.is_set():
            raise RuntimeError("B0 calibration aborted before positioning")

        self.after(0,lambda:self.plan_progress_var.set(
            f"B0 PREP {queue_index+1}/{total} | safe non-contact ({x:.3f},{y:.3f},{safe:.3f}) mm"
        ))
        self.printer.move_abs(z=safe,speed_mm_s=z_speed,mode=action,phase="dynamic_b0_safe_z")
        self.printer.move_abs(x=x,y=y,speed_mm_s=xy_speed,mode=action,phase="dynamic_b0_trial_xy")
        self.printer.dwell(settle_s,mode=action,phase="dynamic_b0_mechanical_settle")

        started_perf_ns=time.perf_counter_ns()
        with self.live_predict_lock:
            predictor=self.live_predictor
            if predictor is None or not self.live_predict_enabled:
                raise RuntimeError("Live predictor became unavailable before dynamic B0")
            history_required=int(predictor.seq_len)
            baseline_seconds=max(2.0,history_required/40.0)
            self._live_predict_clear_queues()
            self.live_predict_min_input_ns=started_perf_ns
            predictor.begin_baseline(seconds=baseline_seconds)
            target=int(predictor.baseline_target_frames)
            family=str(predictor.family)

        self.after(0,lambda:self.live_predict_plot.clear())
        self.after(0,lambda:self.live_predict_3d.clear())
        self.after(0,lambda:self.exp_predict_3d.clear())
        deadline=time.monotonic()+max(15.0,baseline_seconds*4.0)
        last_progress=None
        while True:
            if (self.batch_abort_event.is_set() or self.motion_abort_event.is_set() or
                    self.full_plan_soft_stop_event.is_set()):
                raise RuntimeError("B0 calibration aborted/stopped")
            with self.live_predict_lock:
                if predictor is not self.live_predictor or not self.live_predict_enabled:
                    raise RuntimeError("Live predictor changed or stopped during dynamic B0")
                progress=tuple(predictor.baseline_progress)
                ready=bool(predictor.ready)
            if progress!=last_progress:
                got,total_frames=progress
                message=(f"DYNAMIC B0 {queue_index+1}/{total} | {got}/{total_frames} frames | "
                         f"{family} | keep unloaded")
                self.after(0,lambda m=message:self.plan_progress_var.set(m))
                self.after(0,lambda m=message:self.exp_predict_info_var.set(m))
                last_progress=progress
            if ready:
                break
            if time.monotonic()>=deadline:
                raise TimeoutError(
                    f"Dynamic B0 timeout: progress={progress[0]}/{progress[1]}, "
                    f"expected {baseline_seconds:.2f}s calibration window"
                )
            time.sleep(0.05)

        completed_perf_ns=time.perf_counter_ns()
        with self.live_predict_lock:
            b0_value=getattr(predictor,"b0",None)
            if b0_value is not None and hasattr(b0_value,"tolist"):
                b0_value=b0_value.tolist()
            elif isinstance(b0_value,tuple):
                b0_value=list(b0_value)
            progress=tuple(predictor.baseline_progress)
        info={
            "schema":"MagTouch-DYNAMIC-B0-1.0",
            "policy":"per-Trial robust unloaded median; no continuous update during action",
            "model_family":family,
            "queue_index":int(queue_index)+1,
            "trial_center_xy_mm":[x,y],
            "safe_noncontact_z_mm":safe,
            "mechanical_settle_s":settle_s,
            "baseline_window_s":baseline_seconds,
            "baseline_frames":int(progress[0]),
            "baseline_target_frames":target,
            "history_frames_required":history_required,
            "started_perf_ns":started_perf_ns,
            "completed_perf_ns":completed_perf_ns,
            "duration_s":(completed_perf_ns-started_perf_ns)/1e9,
            "b0_5x3_uT":b0_value,
        }
        self.after(0,lambda:self.exp_predict_info_var.set(
            f"动态 B0 完成 [{family}]；即将开始 Trial #{queue_index+1}"
        ))
        return info

    @staticmethod
    def _full_plan_final_status(completed,total,batch_aborted=False,
                                motion_aborted=False,soft_stopped=False,failure=None):
        """Resolve one terminal state without overwriting a worker ERROR as STOPPED."""
        if failure:
            return "ERROR"
        if batch_aborted or motion_aborted:
            return "ABORTED"
        if int(completed)>=int(total) and not soft_stopped:
            return "COMPLETED"
        return "STOPPED"

    def start_full_plan(self):
        if self.full_plan_running or (self.full_plan_thread and self.full_plan_thread.is_alive()):
            messagebox.showwarning("Full Plan","全部实验计划正在运行 / Plan is already running");return
        pending=[i for i,it in enumerate(self.full_plan_items) if it.get("status","PENDING")!="DONE"]
        if not pending:messagebox.showinfo("Full Plan","没有待运行条目；可先 Reset status");return
        # Validate the ENTIRE queue before creating a folder or issuing any motion.
        for idx in pending:
            try:
                self._validate_action_cfg(self.full_plan_items[idx])
                self._validate_full_plan_required_hardware(self.full_plan_items[idx])
            except Exception as e:
                messagebox.showerror("Full Plan 安全预检失败 / Safety preflight failed",f"计划条目 #{idx+1} 不安全，已禁止启动：\n{e}")
                return
        if self.recorder is not None:
            messagebox.showwarning("Full Plan","已有活动 Trial；请先停止记录再启动计划。")
            return
        prediction_items=[idx for idx in pending if self.full_plan_items[idx].get("record_live_predictions")]
        b0_required_items=[idx for idx in pending if self.full_plan_items[idx].get("requires_dynamic_b0")]
        with self.live_predict_lock:
            predictor=self.live_predictor
            prediction_active=bool(predictor is not None and self.live_predict_enabled)
        if prediction_items:
            if not prediction_active:
                messagebox.showwarning(
                    "预测回测安全预检 / Prediction backtest preflight",
                    "该计划要求同步记录模型输出。请先在“实时预测”页载入正确 checkpoint、确认配置并启用预测。每个 Trial 的 B0 将由计划自动重新采集；系统尚未连接或移动打印机。"
                )
                return
        if b0_required_items and not prediction_active:
            messagebox.showwarning(
                "高密度网格分块B0预检 / Dynamic-B0 preflight",
                "该1/2 mm高密度计划要求每个Y分块Trial开始前采集新的无载荷B0。请先在“实时预测”页启用实时算法（本协议建议选择归一化拟合并载入对应磁片参数）。\n\n"
                "系统尚未连接或移动打印机；未建立实时算法时禁止开始该计划，避免产生无法可靠比较的长时漂移数据。"
            )
            return
        if not self._ensure_printer_connected():return
        if not self._require_home_for_auto():return
        (pcx,pcy),pcz=self._plan_reference_geometry()
        b0_notice=("\n\n实时预测已启用：每个 Trial 前强制执行安全非接触定位、2 s 稳定和动态 B0；校准失败立即停止。"
                   if prediction_active else "\n\n实时预测未启用：本计划不执行模型 B0 校准，只进行原始采集。")
        if not messagebox.askyesno("一键开始全部采集 / Start Full Plan",f"将连续执行 {len(pending)} 个独立 Trial。\n每个 Trial 内按照 Rep 重复完整动作。\n\n计划参考中心 / Plan center=({pcx:.3f},{pcy:.3f}), Contact Z={pcz:.3f} mm\n当前Printer设置=({self.sensor_center_x_var.get():.3f},{self.sensor_center_y_var.get():.3f}), Contact Z={self.contact_z_var.get():.3f} mm{b0_notice}\n\n执行时以已生成Plan中冻结的Trial几何参数为准。\n请确认压头/样品安装、软件限位和急停均正常。是否开始？"):return
        if prediction_active:
            self.start_magnetics()
            if self.mag_thread is None or not self.mag_thread.is_alive():
                messagebox.showerror("动态 B0","磁场采集未能启动，禁止执行需要动态 B0 的计划。")
                return
        self.full_plan_dynamic_b0_enabled=prediction_active
        self.batch_abort_event.clear();self.motion_abort_event.clear();self.full_plan_soft_stop_event.clear();self.full_plan_pause_event.clear();self.full_plan_running=True
        self.plan_pause_btn.configure(text="Trial间暂停 / Pause between Trials")
        try:
            plan_dir=self._create_plan_run_folder(pending)
            self.plan_progress_var.set(f"PLAN FOLDER: {plan_dir.name}")
            self.log(f"Full plan run folder created: {plan_dir}")
            self._write_plan_run_progress(0,len(pending),status="RUNNING")
        except Exception as e:
            self.full_plan_running=False
            self.full_plan_dynamic_b0_enabled=False
            messagebox.showerror("Full Plan folder",f"无法创建本次实验计划文件夹 / Cannot create plan folder:\n{e}")
            return
        if bool(self.auto_fan_off_var.get()):
            try:self.printer.fan_off()
            except Exception:pass
        total=len(pending)
        def worker():
            completed=0
            failure=None
            try:
                for idx in pending:
                    if self.batch_abort_event.is_set() or self.motion_abort_event.is_set() or self.full_plan_soft_stop_event.is_set():break
                    while self.full_plan_pause_event.is_set() and not self.batch_abort_event.is_set():time.sleep(0.1)
                    if self.batch_abort_event.is_set():break
                    item=self.full_plan_items[idx]
                    b0_info=(self._capture_full_plan_dynamic_b0(item,idx,total)
                             if self.full_plan_dynamic_b0_enabled else None)
                    # Synchronize all Trial metadata on the Tk thread.
                    ready=threading.Event();box={}
                    def prep():
                        try:
                            self.scan_next_trial_id(silent=True);self._apply_full_plan_item_ui(item)
                            self.start_trial(parent_dir=plan_dir,folder_tag=str(item.get("prediction_folder_tag", "")))
                            if self.recorder is None:raise RuntimeError("Trial recorder did not start")
                            if item.get("record_live_predictions") and self.recorder.prediction_writer is None:
                                raise RuntimeError("Backtest Trial failed to attach live prediction recording")
                            if b0_info is not None:
                                self.recorder.meta["dynamic_b0_calibration"]=b0_info
                                self.recorder._write_meta()
                                self.recorder.write_event(
                                    "DYNAMIC_B0_APPLIED",json.dumps(b0_info,ensure_ascii=False),"planner"
                                )
                            box["rec"]=self.recorder;self._set_plan_item_status(idx,"RUNNING")
                        except Exception as exc:
                            box["error"]=exc
                        finally:ready.set()
                    self.after(0,prep)
                    if not ready.wait(8.0):raise RuntimeError("UI timeout while starting plan Trial")
                    if box.get("error") is not None:raise RuntimeError(str(box["error"]))
                    rec=box.get("rec")
                    if rec is None:raise RuntimeError("Trial recorder did not start")
                    rec.write_event("FULL_PLAN_ITEM",json.dumps({"plan":self.plan_name_var.get(),"queue_index":idx+1,"item":item},ensure_ascii=False),"planner")
                    try:(rec.trial_dir/"batch_plan_item.json").write_text(json.dumps(item,ensure_ascii=False,indent=2),encoding="utf-8")
                    except Exception:pass
                    self.after(0,lambda idx=idx,total=total:self.plan_progress_var.set(f"RUNNING {completed+1}/{total} | queue #{idx+1} | {self.full_plan_items[idx].get('group')} / {self.full_plan_items[idx].get('action')}"))
                    self._write_plan_run_progress(completed,total,current_idx=idx,status="RUNNING")
                    self._execute_stage_experiment(item)
                    if self.motion_abort_event.is_set() or self.batch_abort_event.is_set():
                        self.after(0,lambda:self.stop_trial("aborted"));self.after(0,lambda idx=idx:self._set_plan_item_status(idx,"ABORTED"));break
                    done=threading.Event()
                    def finish():
                        try:self.stop_trial("completed");self._set_plan_item_status(idx,"DONE")
                        finally:done.set()
                    self.after(0,finish);done.wait(8.0);completed+=1
                    self.after(0,lambda c=completed,t=total:self.plan_progress.configure(value=100.0*c/max(1,t)))
                    self.save_full_plan(path=self.full_plan_path,silent=True)
                    self._write_plan_run_progress(completed,total,current_idx=idx,status="RUNNING")
                    if self.full_plan_soft_stop_event.is_set():break
            except Exception as e:
                failure=str(e)
                self.log(f"FULL PLAN ERROR: {e}")
                self._write_plan_run_progress(completed,total,status="ERROR",error=e)
                try:self.after(0,lambda:self.stop_trial("aborted"))
                except Exception:pass
            finally:
                self.full_plan_running=False
                final_status=self._full_plan_final_status(
                    completed,total,
                    batch_aborted=self.batch_abort_event.is_set(),
                    motion_aborted=self.motion_abort_event.is_set(),
                    soft_stopped=self.full_plan_soft_stop_event.is_set(),
                    failure=failure,
                )
                self._write_plan_run_progress(completed,total,status=final_status,error=failure)
                run_dir=self.active_plan_run_dir
                status_text=(
                    f"{final_status} | completed={completed}/{total} | "
                    f"{run_dir.name if run_dir else ''}"
                    + (f"\n原因 / Reason: {failure}" if failure else "")
                )
                self.after(0,lambda msg=status_text:self.plan_progress_var.set(msg))
                if failure:
                    self.after(0,lambda msg=failure:messagebox.showerror(
                        "全部实验计划失败 / Full Plan failed",msg
                    ))
                self.save_full_plan(path=self.full_plan_path,silent=True)
                self.active_plan_run_dir=None;self.active_plan_run_id=None;self.active_plan_run_meta_path=None
                self.full_plan_dynamic_b0_enabled=False
        self.full_plan_thread=threading.Thread(target=worker,daemon=True);self.full_plan_thread.start()

    def toggle_full_plan_pause(self):
        if not self.full_plan_running:return
        if self.full_plan_pause_event.is_set():
            self.full_plan_pause_event.clear();self.plan_pause_btn.configure(text="Trial间暂停 / Pause between Trials");self.log("Full plan resumed")
        else:
            self.full_plan_pause_event.set();self.plan_pause_btn.configure(text="继续计划 / Resume Plan");self.log("Full plan pause requested; takes effect between Trials")

    def stop_full_plan_after_current(self):
        if not self.full_plan_running:return
        self.full_plan_soft_stop_event.set();self.log("Full plan soft-stop requested: stop after current Trial")

    def _build_info_tab(self):
        tab=self.tab_info
        geo=ttk.LabelFrame(tab,text="第一版 2×2+1 实测几何")
        geo.pack(fill="x",padx=8,pady=8)
        text=(
            "四角相邻传感器中心距 = 15.188 mm；原点为四角几何中心。\n"
            "S1=(-7.594,+7.594), S2=(-7.594,-7.594), S3=(+7.594,+7.594), "
            "S4=(+7.594,-7.594), S5=(0,+1.406) mm。\n"
            "A5S实测传感器中心/上表面接触坐标 = (149.000, 147.000, 11.100) mm。\n"
            "S5 的 +1.406 mm 实际偏移必须保留；打印机轴与传感器物理轴的符号映射将在空间扫描中进一步标定。"
        )
        ttk.Label(geo,text=text,justify="left",wraplength=1450).pack(anchor="w",padx=8,pady=8)
        self.geometry_text=ScrolledText(tab,height=28,font=("Consolas",9))
        self.geometry_text.pack(fill="both",expand=True,padx=8,pady=8)
        self.geometry_text.insert("1.0",json.dumps(self.geometry_config,ensure_ascii=False,indent=2))

    @staticmethod
    def _set_param_enabled(pair, enabled: bool):
        """Enable/disable a (label, input-widget) pair so unused parameters are visibly grey."""
        if not pair:
            return
        lab, widget = pair
        try:
            if enabled:
                lab.state(["!disabled"])
                widget.state(["!disabled"])
            else:
                lab.state(["disabled"])
                widget.state(["disabled"])
        except Exception:
            try: widget.configure(state="normal" if enabled else "disabled")
            except Exception: pass

    def _trial_mode_code(self):
        raw=self.mode_var.get().strip()
        return TRIAL_MODE_CODES.get(raw, raw.split(" / ",1)[0])

    def _stage_action_code(self):
        raw=self.stage_action_var.get().strip()
        return STAGE_ACTION_CODES.get(raw, raw.split(" / ",1)[0])

    def _update_trial_param_states(self):
        mode=self._trial_mode_code()
        allowed={
            "none":{"x","y","hold"},
            "press":{"x","y","z","speed","hold"},
            "press_cycle":{"x","y","z","speed","hold"},
            "slide_x":{"x","y","z","dx","speed","hold"},
            "slide_y":{"x","y","z","dy","speed","hold"},
            "slide_xy":{"x","y","z","dx","dy","speed","hold"},
            "compound":{"x","y","z","dx","dy","speed","hold"},
            "z_sweep":{"x","y","speed","hold","z_depths"},
            "z_path":{"x","y","speed","hold","z_depths"},
            "x_sweep":{"x","y","z","speed","hold","xy_values"},
            "y_sweep":{"x","y","z","speed","hold","xy_values"},
            "cross_scan":{"x","y","z","dx","dy","speed","hold"},
            "angle_scan":{"x","y","z","dx","speed","hold","angle_count"},
            "circle_scan":{"x","y","z","speed","hold","circle_radius","circle_points"},
            "grid_points":{"x","y","z","speed","hold","grid_half_x","grid_half_y","grid_step"},
            "load_hold_release":{"x","y","z","speed","hold"},
        }.get(mode,{"x","y","z","dx","dy","speed","hold"})
        if mode != "none":
            allowed = set(allowed) | {"repeat","repeat_interval"}
        for key,pair in getattr(self,"trial_param_widgets",{}).items():
            if key in {"mode","indenter","material","maglayer"}: continue
            self._set_param_enabled(pair,key in allowed)
        hints={
            "none":"无接触基线 / Baseline：仅到中心安全位置并保持，记录静态磁场。",
            "press":"单次按压 / Press：中心点单次法向压入。",
            "press_cycle":"周期点压 / Press Cycle：在同一个 Trial 内按 Repeats 连续执行压入-释放；Repeat interval 控制循环间隔。",
            "slide_x":"X向滑动 / Slide-X：预压后沿 X 位移 dx。",
            "slide_y":"Y向滑动 / Slide-Y：预压后沿 Y 位移 dy。",
            "slide_xy":"任意XY滑动 / Slide-XY：预压后按 dx,dy 合成方向滑动。",
            "compound":"XYZ同步复合 / Compound：从接触前位置以 G1 同时改变 X/Y/Z，区别于先压后滑。",
            "z_sweep":"Z阶梯 / Z Sweep：按 Z depths 列表逐级点压。",
            "z_path":"Z路径依赖 / Z Path：按 Z depths 顺序连续到达各深度，不在各点之间抬离，用于历史/路径依赖。",
            "x_sweep":"X阶梯 / X Sweep：固定压入后按 Sweep values 做多位置X位移。",
            "y_sweep":"Y阶梯 / Y Sweep：固定压入后按 Sweep values 做多位置Y位移。",
            "cross_scan":"十字扫描 / Cross Scan：固定压入后依次 ±X、±Y。",
            "angle_scan":"多角度扫描 / Angle Scan：dx 作为半径，按 Angle count 个方向径向滑动。",
            "circle_scan":"圆周扫描 / Circle Scan：固定压入后沿指定半径和点数走圆周轨迹。",
            "grid_points":"网格点触 / Grid Points：XY网格逐点执行相同法向点触。",
            "load_hold_release":"加载保持释放 / Load-Hold-Release：专门采集加载、稳态保持和回弹过程。",
        }
        if hasattr(self,"trial_param_hint_var"):
            self.trial_param_hint_var.set(hints.get(mode,""))
        self._update_auto_summary()

    def _update_stage_param_states(self):
        action=self._stage_action_code()
        allowed={
            "none":{"hold","repeat"},
            "press":{"indent","z_speed","hold","repeat"},
            "press_cycle":{"indent","z_speed","hold","repeat"},
            "slide_x":{"indent","dx","xy_speed","z_speed","hold","repeat"},
            "slide_y":{"indent","dy","xy_speed","z_speed","hold","repeat"},
            "slide_xy":{"indent","dx","dy","xy_speed","z_speed","hold","repeat"},
            "compound":{"indent","dx","dy","xy_speed","z_speed","hold","repeat"},
            "z_sweep":{"z_speed","hold","repeat","z_depths"},
            "z_path":{"z_speed","hold","repeat","z_depths"},
            "x_sweep":{"indent","xy_speed","z_speed","hold","repeat","xy_values"},
            "y_sweep":{"indent","xy_speed","z_speed","hold","repeat","xy_values"},
            "cross_scan":{"indent","dx","dy","xy_speed","z_speed","hold","repeat"},
            "angle_scan":{"indent","dx","xy_speed","z_speed","hold","repeat","angle_count"},
            "circle_scan":{"indent","xy_speed","z_speed","hold","repeat","circle_radius","circle_points"},
            "grid_points":{"indent","xy_speed","z_speed","hold","repeat","grid_half_x","grid_half_y","grid_step"},
            "load_hold_release":{"indent","z_speed","hold","repeat"},
        }.get(action,set())
        for key,pair in getattr(self,"stage_param_widgets",{}).items():
            self._set_param_enabled(pair,key in allowed)
        hints={
            "none":"无接触基线 / Baseline：到中心安全位保持，不接触样品。",
            "press":"单次法向按压 / Press：仅改变 Z。",
            "press_cycle":"周期点压 / Press Cycle：按 Repeats 连续执行压入-释放循环。",
            "slide_x":"X向滑动 / Slide-X：先压入，再沿 X 移动 dx。",
            "slide_y":"Y向滑动 / Slide-Y：先压入，再沿 Y 移动 dy。",
            "slide_xy":"任意XY滑动 / Slide-XY：先压入，再按 dx+dy 合成方向移动。",
            "compound":"XYZ同步复合 / Compound：从接触前位置用一个 G1 同时改变 X/Y/Z。",
            "z_sweep":"Z阶梯压入 / Z Sweep：按 Z深度列表逐级点压。",
            "z_path":"Z路径依赖 / Z Path：按Z深度列表连续运行，不在各深度之间抬离。",
            "x_sweep":"X阶梯扫描 / X Sweep：固定压入后按 XY阶梯列表沿X扫描。",
            "y_sweep":"Y阶梯扫描 / Y Sweep：固定压入后按 XY阶梯列表沿Y扫描。",
            "cross_scan":"十字切向扫描 / Cross Scan：固定压入后按 ±X、±Y 扫描。",
            "angle_scan":"多角度径向滑动 / Angle Scan：dx 作为半径，按方向数覆盖 360°。",
            "circle_scan":"圆周轨迹 / Circle Scan：固定压入后沿圆周连续采样方向响应。",
            "grid_points":"XY网格点触 / Grid Points：网格逐点按压，用于空间响应图。",
            "load_hold_release":"加载-保持-释放 / Load-Hold-Release：专门记录加载瞬态、保持、回弹。",
        }
        if hasattr(self,"stage_param_hint_var"):
            self.stage_param_hint_var.set(hints.get(action,""))

    def _install_parameter_autosave(self):
        """Debounced auto-save for experiment/printer parameters.

        User-entered geometry and motion values are configuration, not temporary UI state.
        The callback is intentionally debounced so typing a number does not rewrite JSON
        on every keystroke.
        """
        self._autosave_after_id=None
        vars_to_watch=[]
        names=[
            "printer_port_var","printer_baud_var","printer_dry_var","require_home_auto_var",
            "sensor_center_x_var","sensor_center_y_var","contact_z_var","safe_lift_var","approach_clearance_var",
            "jog_step_var","xy_speed_var","z_speed_var","z_home_raise_enabled_var","z_home_raise_var","g92_z_var","auto_fan_off_var",
            "point_depth_var","point_hold_var","point_repeat_var","point_interval_var","point_speed_var",
            "stage_action_var","stage_indent_var","stage_dx_var","stage_dy_var","stage_speed_var","stage_z_speed_var","stage_hold_var","stage_repeat_var",
            "stage_depths_var","stage_xy_sweep_var","stage_angle_count_var","stage_circle_radius_var","stage_circle_points_var","grid_half_x_var","grid_half_y_var","grid_step_var",
            "mode_var","x_var","y_var","z_var","dx_var","dy_var","speed_var","hold_var","trial_z_depths_var","trial_xy_sweep_var","trial_angle_count_var","trial_circle_radius_var","trial_circle_points_var","trial_grid_half_x_var","trial_grid_half_y_var","trial_grid_step_var","trial_repeat_var","repeat_interval_var","auto_trial_id_var","auto_summary_enabled_var",
            "dev_index_var","i2c_clock_var","mag_interval_var","include_temp_var",
            # The chosen checkpoint and the explicit configuration-match acknowledgement
            # are reusable UI parameters.  The per-session B0 remains intentionally live.
            "live_predict_model_var","live_predict_model_family_var","live_predict_output_mapping_var","live_predict_profile_confirm_var",
            "training_plan_var","training_export_out_var","training_source_var","training_features_var",
            "training_feature_var","training_policy_var","training_model_var","training_task_var","training_epochs_var","training_batch_var","training_gpu_var",
            "training_hw5_dmodel_var","training_hw5_layers_var","training_hw5_batch_var","training_hw5_objective_var","training_deterministic_var","training_seed_var",
            "training_relative_var","training_relative_feature_var","training_relative_history_var","training_relative_use_q_var",
            "training_r18_source_var","training_r18_relative_var","training_r18_labels_var","training_r18_feature_var","training_r18_history_var",
            "training_r18_endpoint_history_var","training_r18_objective_var","training_r18_sample_policy_var","training_r18_use_q_var",
            "ubuntu_project_root_var","ubuntu_plan_path_var","ubuntu_export_output_var",
            "ubuntu_training_export_var","ubuntu_h1_features_var","ubuntu_h2_relative_var","ubuntu_h2_spatial_labels_var","ubuntu_h2_unified_labels_var","ubuntu_conda_env_var",
            "ubuntu_gpus_var","ubuntu_cpu_count_var","ubuntu_gpu_schedule_var","ubuntu_seeds_var",
            "ubuntu_h1_feature_var","ubuntu_h1_policy_var","ubuntu_baseline_model_var","ubuntu_baseline_task_var",
            "ubuntu_epochs_var","ubuntu_baseline_batch_var","ubuntu_hw5_batch_var","ubuntu_dmodel_var","ubuntu_layers_var",
            "ubuntu_h1_objective_var","ubuntu_deterministic_var","ubuntu_h2_feature_var","ubuntu_h2_history_var",
            "ubuntu_h2_endpoint_history_var","ubuntu_h2_use_q_var","ubuntu_h2_spatial_objective_var","ubuntu_h2_sample_policy_var",
            "ubuntu_h2_unified_objective_var","ubuntu_h2_unified_sample_policy_var","ubuntu_extra_args_var",
        ]
        for name in names:
            v=getattr(self,name,None)
            if v is not None: vars_to_watch.append(v)
        vars_to_watch.extend(getattr(self,"addr_vars",[]))
        for name in ("top","side"):
            for v in getattr(self,"cam_vars",{}).get(name,{}).values(): vars_to_watch.append(v)
        def changed(*_):
            self._schedule_settings_autosave()
        for v in vars_to_watch:
            try:v.trace_add("write",changed)
            except Exception:pass

    def _schedule_settings_autosave(self):
        """Queue one configuration write after a completed UI adjustment."""
        if not getattr(self,"auto_save_parameters_var",tk.BooleanVar(value=True)).get():
            return
        autosave_after = getattr(self,"_autosave_after_id",None)
        if autosave_after is not None:
            try:self.after_cancel(autosave_after)
            except Exception:pass
        self._autosave_after_id=self.after(800,lambda:self.save_settings(silent=True))

    # ------------------------- persistent configuration -------------------------
    def _settings_snapshot(self):
        camera={
            "top_enable":bool(self.cam_vars["top"]["enable"].get()), "side_enable":bool(self.cam_vars["side"]["enable"].get()),
            "top_index":int(self.cam_vars["top"]["index"].get()), "side_index":int(self.cam_vars["side"]["index"].get()),
            "width":int(self.cam_vars["top"]["width"].get()), "height":int(self.cam_vars["top"]["height"].get()),
            "fps":float(self.cam_vars["top"]["fps"].get()),
        }
        return {
            "config_version":2,
            "replay_backtest":self.replay_backtest_panel.snapshot(),
            "printer":{
                "port":self.printer_port_var.get().strip(), "baud":int(self.printer_baud_var.get()), "dry_run":bool(self.printer_dry_var.get()),
                "require_home_before_auto_motion":bool(self.require_home_auto_var.get()),
                "sensor_center_x_mm":float(self.sensor_center_x_var.get()), "sensor_center_y_mm":float(self.sensor_center_y_var.get()),
                "surface_contact_z_mm":float(self.contact_z_var.get()), "safe_lift_mm":float(self.safe_lift_var.get()),
                "approach_clearance_mm":float(self.approach_clearance_var.get()), "jog_step_mm":float(self.jog_step_var.get()),
                "xy_speed_mm_s":float(self.xy_speed_var.get()), "z_speed_mm_s":float(self.z_speed_var.get()),
                "home_z_raise_enabled":bool(self.z_home_raise_enabled_var.get()), "home_z_raise_mm":float(self.z_home_raise_var.get()),
                "g92_z_mm":float(self.g92_z_var.get()), "auto_fan_off":bool(self.auto_fan_off_var.get()), "auto_save_parameters":bool(self.auto_save_parameters_var.get()),
                "live_position_refresh":bool(self.printer_live_pos_var.get()),
            },
            "point_touch":{
                "depth_mm":float(self.point_depth_var.get()), "hold_s":float(self.point_hold_var.get()), "repeat":int(self.point_repeat_var.get()),
                "interval_s":float(self.point_interval_var.get()), "z_speed_mm_s":float(self.point_speed_var.get()),
            },
            "stage_experiment":{
                "action":self._stage_action_code(), "indent_mm":float(self.stage_indent_var.get()), "dx_mm":float(self.stage_dx_var.get()),
                "dy_mm":float(self.stage_dy_var.get()), "speed_mm_s":float(self.stage_speed_var.get()), "z_speed_mm_s":float(self.stage_z_speed_var.get()), "hold_s":float(self.stage_hold_var.get()),
                "repeat":int(self.stage_repeat_var.get()), "z_sweep_depths_mm":self.stage_depths_var.get(),
                "xy_sweep_values_mm":self.stage_xy_sweep_var.get(), "angle_count":int(self.stage_angle_count_var.get()),
                "circle_radius_mm":float(self.stage_circle_radius_var.get()), "circle_points":int(self.stage_circle_points_var.get()),
                "grid_half_x_mm":float(self.grid_half_x_var.get()), "grid_half_y_mm":float(self.grid_half_y_var.get()), "grid_step_mm":float(self.grid_step_var.get()),
            },
            "trial":{"auto_id":bool(self.auto_trial_id_var.get()),"repeats":max(1,int(self.trial_repeat_var.get())),"repeat_interval_s":max(0.0,float(self.repeat_interval_var.get())),"auto_summary":bool(self.auto_summary_enabled_var.get())},
            "magnetic":{
                "active_layer_profile_id":self.active_mag_profile_id,
                "ch341_index":int(self.dev_index_var.get()), "i2c_clock":self.i2c_clock_var.get(), "sample_period_ms":int(self.mag_interval_var.get()),
                "include_temperature_each_cycle":bool(self.include_temp_var.get()), "addresses":[v.get() for v in self.addr_vars],
            },
            "camera":camera,
            "live_algorithm": self.live_algorithm_panel.snapshot(),
            "live_prediction":{
                "checkpoint_path":self.live_predict_model_var.get().strip(),
                "model_family":self.live_predict_model_family_var.get(),
                 "output_mapping_json":self.live_predict_output_mapping_var.get().strip(),
                 "profile_confirmed":bool(self.live_predict_profile_confirm_var.get()),
                 "display_invert_x":bool(self.live_predict_invert_x_var.get()),
                 "display_invert_y":bool(self.live_predict_invert_y_var.get()),
                 "edge_guard_ms":self._live_edge_guard_ms(),
                 "trajectory_filter_mode":self.live_filter_mode_var.get(),
                 "trajectory_filter_ema_alpha":float(self.live_filter_ema_alpha_var.get()),
                 "trajectory_filter_q":float(self.live_filter_q_var.get()),
                 "trajectory_filter_r_xy":float(self.live_filter_r_xy_var.get()),
                  "trajectory_filter_r_z":float(self.live_filter_r_z_var.get()),
                  "trajectory_history_points":self.prediction_trajectory_history_points(),
                  "alignment_basis":self.live_alignment_basis_var.get(),
                  "alignment_mode":self.live_alignment_mode_var.get(),
                  "alignment_auto_depth_mm":float(self.live_alignment_auto_depth_var.get()),
                  "alignment_z_threshold_uT":float(self.live_alignment_z_threshold_var.get()),
                  "alignment_z_step_mm":float(self.live_alignment_z_step_var.get()),
                  "alignment_z_max_search_mm":float(self.live_alignment_z_max_search_var.get()),
                 "xy_alignment_by_preset":json.loads(json.dumps(self.live_alignment_by_preset)),
                "view_yaw_deg":float(self.live_predict_3d.yaw_deg),
                "view_pitch_deg":float(self.live_predict_3d.pitch_deg),
                "view_zoom":float(self.live_predict_3d.zoom),
                "view_pan_x":float(self.live_predict_3d.pan_x),
                "view_pan_y":float(self.live_predict_3d.pan_y),
                "experiment_view_yaw_deg":float(self.exp_predict_3d.yaw_deg),
                "experiment_view_pitch_deg":float(self.exp_predict_3d.pitch_deg),
                "experiment_view_zoom":float(self.exp_predict_3d.zoom),
                "experiment_view_pan_x":float(self.exp_predict_3d.pan_x),
                "experiment_view_pan_y":float(self.exp_predict_3d.pan_y),
            },
            "training":{
                "plan_path":self.training_plan_var.get().strip(), "export_output_path":self.training_export_out_var.get().strip(),
                "training_export_path":self.training_source_var.get().strip(), "h1_features_path":self.training_features_var.get().strip(),
                "h2_relative_path":self.training_relative_var.get().strip(),
                "r18_training_export_path":self.training_r18_source_var.get().strip(),
                "r18_relative_path":self.training_r18_relative_var.get().strip(), "r18_labels_path":self.training_r18_labels_var.get().strip(),
                "feature":self.training_feature_var.get(), "policy":self.training_policy_var.get(),
                "baseline_model":self.training_model_var.get(), "task":self.training_task_var.get(),
                "epochs":int(self.training_epochs_var.get()), "baseline_batch":int(self.training_batch_var.get()), "gpu":int(self.training_gpu_var.get()),
                "d_model":int(self.training_hw5_dmodel_var.get()), "layers":int(self.training_hw5_layers_var.get()),
                "hw5_batch":int(self.training_hw5_batch_var.get()), "h1_objective":self.training_hw5_objective_var.get(),
                "deterministic":bool(self.training_deterministic_var.get()), "seed":int(self.training_seed_var.get()),
                "h2_feature":self.training_relative_feature_var.get(), "h2_history":self.training_relative_history_var.get(),
                "h2_use_q":bool(self.training_relative_use_q_var.get()),
                "r18_feature":self.training_r18_feature_var.get(), "r18_history":self.training_r18_history_var.get(),
                "r18_endpoint_min_history":self.training_r18_endpoint_history_var.get(),
                "r18_objective":self.training_r18_objective_var.get(), "r18_sample_policy":self.training_r18_sample_policy_var.get(),
                "r18_use_q":bool(self.training_r18_use_q_var.get()),
            },
            "ubuntu_training":normalize_ubuntu_training_config(self._ubuntu_training_config_from_ui()),
        }

    def save_settings(self, silent=False):
        try:
            self.settings=self._settings_snapshot()
            save_runtime_settings(self.settings,self.settings_path)
            if not silent:
                self.log(f"参数已保存：{self.settings_path}")
        except Exception as e:
            if not silent: messagebox.showerror("保存参数",str(e))
            self.log(f"Save settings error: {e}")

    def reload_settings(self):
        try:
            self.settings=load_runtime_settings(self.settings_path)
            p=self.settings["printer"]; pt=self.settings["point_touch"]; ex=self.settings["stage_experiment"]; mg=self.settings["magnetic"]; cc=self.settings["camera"]
            self.printer_port_var.set(p["port"]); self.printer_baud_var.set(p["baud"]); self.printer_dry_var.set(p["dry_run"]); self.require_home_auto_var.set(p["require_home_before_auto_motion"])
            self.sensor_center_x_var.set(p["sensor_center_x_mm"]); self.sensor_center_y_var.set(p["sensor_center_y_mm"]); self.contact_z_var.set(p["surface_contact_z_mm"]); self.safe_lift_var.set(p["safe_lift_mm"]); self.approach_clearance_var.set(p["approach_clearance_mm"])
            self.jog_step_var.set(p["jog_step_mm"]); self.xy_speed_var.set(p["xy_speed_mm_s"]); self.z_speed_var.set(p["z_speed_mm_s"]); self.z_home_raise_enabled_var.set(p["home_z_raise_enabled"]); self.z_home_raise_var.set(p["home_z_raise_mm"]); self.g92_z_var.set(p["g92_z_mm"]); self.auto_fan_off_var.set(p["auto_fan_off"]); self.auto_save_parameters_var.set(p.get("auto_save_parameters",True)); self.printer_live_pos_var.set(p.get("live_position_refresh",True))
            self.point_depth_var.set(pt["depth_mm"]); self.point_hold_var.set(pt["hold_s"]); self.point_repeat_var.set(pt["repeat"]); self.point_interval_var.set(pt["interval_s"]); self.point_speed_var.set(pt["z_speed_mm_s"])
            self.stage_action_var.set(STAGE_ACTION_LABELS.get(ex.get("action","press"),STAGE_ACTION_LABELS["press"])); self.stage_indent_var.set(ex["indent_mm"]); self.stage_dx_var.set(ex["dx_mm"]); self.stage_dy_var.set(ex["dy_mm"]); self.stage_speed_var.set(ex["speed_mm_s"]); self.stage_z_speed_var.set(ex.get("z_speed_mm_s",pt.get("z_speed_mm_s",0.5))); self.stage_hold_var.set(ex["hold_s"]); self.stage_repeat_var.set(ex["repeat"]); self.stage_depths_var.set(ex["z_sweep_depths_mm"]); self.stage_xy_sweep_var.set(ex.get("xy_sweep_values_mm","-3,-2,-1,0,1,2,3")); self.stage_angle_count_var.set(ex.get("angle_count",8)); self.stage_circle_radius_var.set(ex.get("circle_radius_mm",2.0)); self.stage_circle_points_var.set(ex.get("circle_points",24)); self.grid_half_x_var.set(ex["grid_half_x_mm"]); self.grid_half_y_var.set(ex["grid_half_y_mm"]); self.grid_step_var.set(ex["grid_step_mm"]); self._update_stage_param_states()
            tr=self.settings.get("trial",{}); self.auto_trial_id_var.set(tr.get("auto_id",True)); self.trial_repeat_var.set(tr.get("repeats",tr.get("count",1))); self.repeat_interval_var.set(tr.get("repeat_interval_s",tr.get("inter_trial_s",1.0))); self.auto_summary_enabled_var.set(tr.get("auto_summary",True))
            self.dev_index_var.set(mg["ch341_index"]); self.i2c_clock_var.set(mg["i2c_clock"]); self.mag_interval_var.set(mg["sample_period_ms"]); self.include_temp_var.set(mg["include_temperature_each_cycle"])
            _pid=str(mg.get("active_layer_profile_id",self.active_mag_profile_id));
            if _pid in self.maglayer_profiles:
                self.active_mag_profile_id=_pid; self.mag_profile_id_var.set(_pid); self._load_selected_mag_profile_to_editor(); self._apply_selected_magnetic_layer_profile(disconnect=True,quiet=True)
            for i,v in enumerate(self.addr_vars):
                if i < len(mg["addresses"]): v.set(mg["addresses"][i])
            for name in ("top","side"):
                self.cam_vars[name]["enable"].set(cc.get(f"{name}_enable",False)); self.cam_vars[name]["index"].set(cc.get(f"{name}_index",0 if name=="top" else 1)); self.cam_vars[name]["width"].set(cc.get("width",1280)); self.cam_vars[name]["height"].set(cc.get("height",720)); self.cam_vars[name]["fps"].set(cc.get("fps",60.0))
            live=self.settings.get("live_prediction",{})
            if not self.live_algorithm_panel.busy():
                self.live_algorithm_panel.restore(self.settings.get("live_algorithm",{}))
                self.live_algorithm_panel.invalidate()
                self.live_algorithm_panel.update_states()
            self.live_predict_model_var.set(str(live.get("checkpoint_path","")))
            self.live_predict_model_family_var.set(str(live.get("model_family","auto")))
            self.live_predict_output_mapping_var.set(str(live.get("output_mapping_json",default_output_mapping_json())))
            self.live_predict_profile_confirm_var.set(bool(live.get("profile_confirmed",False)))
            self.live_predict_invert_x_var.set(bool(live.get("display_invert_x",False)))
            self.live_predict_invert_y_var.set(bool(live.get("display_invert_y",False)))
            self.live_predict_edge_guard_var.set(str(
                live.get("edge_guard_ms",live.get("onset_guard_ms",120.0))))
            self.live_filter_mode_var.set(str(live.get("trajectory_filter_mode","关闭 / Off")))
            self.live_filter_ema_alpha_var.set(str(live.get("trajectory_filter_ema_alpha",0.25)))
            self.live_filter_q_var.set(str(live.get("trajectory_filter_q",1.0)))
            self.live_filter_r_xy_var.set(str(live.get("trajectory_filter_r_xy",0.16)))
            self.live_filter_r_z_var.set(str(live.get("trajectory_filter_r_z",0.01)))
            self.live_trajectory_history_var.set(str(live.get("trajectory_history_points",4000)))
            self._live_trajectory_history_apply(show_error=False)
            self._live_filter_hot_apply(show_error=False)
            self.live_alignment_basis_var.set(str(
                live.get("alignment_basis","源数据ΔB指纹（推荐）")))
            self.live_alignment_mode_var.set(str(
                live.get("alignment_mode","刚体旋转+平移（推荐）")))
            self.live_alignment_auto_depth_var.set(str(live.get("alignment_auto_depth_mm",0.5)))
            self.live_alignment_z_threshold_var.set(str(live.get("alignment_z_threshold_uT",15.0)))
            self.live_alignment_z_step_var.set(str(live.get("alignment_z_step_mm",0.02)))
            self.live_alignment_z_max_search_var.set(str(live.get("alignment_z_max_search_mm",1.0)))
            alignment=live.get("xy_alignment_by_preset",{})
            self.live_alignment_by_preset=(json.loads(json.dumps(alignment))
                                           if isinstance(alignment,dict) else {})
            self._live_alignment_update_status()
            self.live_predict_3d.set_view(
                live.get("view_yaw_deg",-45.0), live.get("view_pitch_deg",28.0),
                live.get("view_zoom",1.0), live.get("view_pan_x",0.0), live.get("view_pan_y",0.0),
            )
            self.exp_predict_3d.set_view(
                live.get("experiment_view_yaw_deg",live.get("view_yaw_deg",-45.0)),
                live.get("experiment_view_pitch_deg",live.get("view_pitch_deg",28.0)),
                live.get("experiment_view_zoom",live.get("view_zoom",1.0)),
                live.get("experiment_view_pan_x",live.get("view_pan_x",0.0)),
                live.get("experiment_view_pan_y",live.get("view_pan_y",0.0)),
            )
            train=self.settings.get("training",{})
            self.training_plan_var.set(str(train.get("plan_path","")));self.training_export_out_var.set(str(train.get("export_output_path","")))
            self.training_source_var.set(str(train.get("training_export_path","")));self.training_features_var.set(str(train.get("h1_features_path","")))
            self.training_relative_var.set(str(train.get("h2_relative_path","")))
            self.training_feature_var.set(str(train.get("feature","F0")));self.training_policy_var.set(str(train.get("policy","raw15")))
            self.training_model_var.set(str(train.get("baseline_model","transformer")));self.training_task_var.set(str(train.get("task","mode5")))
            self.training_epochs_var.set(int(train.get("epochs",20)));self.training_batch_var.set(int(train.get("baseline_batch",48)));self.training_gpu_var.set(int(train.get("gpu",0)))
            self.training_hw5_dmodel_var.set(int(train.get("d_model",128)));self.training_hw5_layers_var.set(int(train.get("layers",3)));self.training_hw5_batch_var.set(int(train.get("hw5_batch",24)))
            self.training_hw5_objective_var.set(str(train.get("h1_objective","mode_only")));self.training_deterministic_var.set(bool(train.get("deterministic",True)));self.training_seed_var.set(int(train.get("seed",20260910)))
            self.training_relative_feature_var.set(str(train.get("h2_feature","R1")));self.training_relative_history_var.set(str(train.get("h2_history","64")));self.training_relative_use_q_var.set(bool(train.get("h2_use_q",True)))
            self.training_r18_source_var.set(str(train.get("r18_training_export_path",train.get("training_export_path",""))))
            self.training_r18_relative_var.set(str(train.get("r18_relative_path",train.get("h2_relative_path",""))))
            r18_labels_value=str(train.get("r18_labels_path",""))
            if r18_labels_value.endswith("_UNIFIED_LABELS_R18"):r18_labels_value += "_1_ABSXYZ"
            self.training_r18_labels_var.set(r18_labels_value)
            self.training_r18_feature_var.set(str(train.get("r18_feature","R1")));self.training_r18_history_var.set(str(train.get("r18_history","64")))
            self.training_r18_endpoint_history_var.set(str(train.get("r18_endpoint_min_history","128")));self.training_r18_objective_var.set(str(train.get("r18_objective","unified")))
            self.training_r18_sample_policy_var.set(str(train.get("r18_sample_policy","common")));self.training_r18_use_q_var.set(bool(train.get("r18_use_q",True)))
            ubuntu=normalize_ubuntu_training_config(self.settings.get("ubuntu_training",{}))
            ubuntu_r18_labels=str(ubuntu["h2_unified_labels_path"])
            if ubuntu_r18_labels.endswith("_UNIFIED_LABELS_R18"):ubuntu_r18_labels += "_1_ABSXYZ"
            self.ubuntu_h2_unified_labels_var.set(ubuntu_r18_labels)
            self.ubuntu_h2_unified_objective_var.set(str(ubuntu["h2_unified_objective"]));self.ubuntu_h2_unified_sample_policy_var.set(str(ubuntu["h2_unified_sample_policy"]))
            self.ubuntu_project_root_var.set(str(ubuntu["project_root"]));self.ubuntu_plan_path_var.set(str(ubuntu["plan_path"]));self.ubuntu_export_output_var.set(str(ubuntu["export_output_path"]));self.ubuntu_training_export_var.set(str(ubuntu["training_export_path"]));self.ubuntu_h1_features_var.set(str(ubuntu["h1_features_path"]));self.ubuntu_h2_relative_var.set(str(ubuntu["h2_relative_path"]));self.ubuntu_h2_spatial_labels_var.set(str(ubuntu["h2_spatial_labels_path"]));self.ubuntu_conda_env_var.set(str(ubuntu["conda_env"]));self.ubuntu_gpus_var.set(str(ubuntu["gpus"]));self.ubuntu_cpu_count_var.set(str(ubuntu["cpu_count"]));self.ubuntu_gpu_schedule_var.set(str(ubuntu["gpu_schedule"]));self.ubuntu_seeds_var.set(str(ubuntu["seeds"]));self.ubuntu_h1_feature_var.set(str(ubuntu["h1_feature"]));self.ubuntu_h1_policy_var.set(str(ubuntu["h1_policy"]));self.ubuntu_baseline_model_var.set(str(ubuntu["baseline_model"]));self.ubuntu_baseline_task_var.set(str(ubuntu["baseline_task"]));self.ubuntu_epochs_var.set(str(ubuntu["epochs"]));self.ubuntu_baseline_batch_var.set(str(ubuntu["baseline_batch"]));self.ubuntu_hw5_batch_var.set(str(ubuntu["hw5_batch"]));self.ubuntu_dmodel_var.set(str(ubuntu["d_model"]));self.ubuntu_layers_var.set(str(ubuntu["layers"]));self.ubuntu_h1_objective_var.set(str(ubuntu["h1_objective"]));self.ubuntu_deterministic_var.set(bool(ubuntu["deterministic"]));self.ubuntu_h2_feature_var.set(str(ubuntu["h2_feature"]));self.ubuntu_h2_history_var.set(str(ubuntu["h2_history"]));self.ubuntu_h2_endpoint_history_var.set(str(ubuntu["h2_endpoint_min_history"]));self.ubuntu_h2_use_q_var.set(bool(ubuntu["h2_use_q"]));self.ubuntu_h2_spatial_objective_var.set(str(ubuntu["h2_spatial_objective"]));self.ubuntu_h2_sample_policy_var.set(str(ubuntu["h2_sample_policy"]));self.ubuntu_extra_args_var.set(str(ubuntu["extra_args"]));self._ubuntu_generate_commands(show_error=False)
            self.replay_backtest_panel.restore(self.settings.get("replay_backtest",{}))
            self.log(f"参数已重新载入：{self.settings_path}")
        except Exception as e:
            messagebox.showerror("重新载入参数",str(e)); self.log(f"Reload settings error: {e}")

    def _update_home_state(self):
        homes=''.join(a for a in 'XYZ' if a in self.printer.homed_axes) or '--'
        try:self.printer_home_state_var.set(f"Home状态（本程序记录）：{homes} | 自动动作强制Home={'ON' if self.require_home_auto_var.get() else 'OFF'}")
        except Exception:pass

    def _require_home_for_auto(self):
        if not bool(self.require_home_auto_var.get()):
            return True
        missing=[a for a in 'XYZ' if a not in self.printer.homed_axes]
        if missing:
            messagebox.showwarning("自动动作前需Home","当前自动动作要求先完成 Home XYZ。\n缺少："+','.join(missing)+"\n\n接触面Z不需要再按钮标定，直接使用已保存/输入的参数即可。")
            return False
        return True

    # ------------------------- helpers -------------------------
    def log(self,msg):
        stamp=datetime.now().strftime("%H:%M:%S.%f")[:-3]
        line=f"[{stamp}] {msg}\n"
        try:
            self.logbox.insert("end",line); self.logbox.see("end")
        except Exception:
            print(line,end="")

    def _parse_addresses(self):
        vals=[]
        for v in self.addr_vars:
            s=v.get().strip().lower()
            vals.append(int(s,16) if s.startswith("0x") else int(s))
        if len(set(vals))!=5:
            raise ValueError("5个I²C地址必须互不相同")
        return vals

    def connect_magnetics(self):
        try:
            addrs=self._parse_addresses()
            pid,p=self._selected_mag_profile()
            mcfg=p.get("mlx90393",{})
            if self.simulate:
                self.sampler=SimulatedFiveSensorSampler(addrs)
            else:
                clock_map={"100 kHz":100_000,"400 kHz":400_000,"750 kHz":750_000}
                dp=str(mcfg.get("datasheet_profile","legacy_2015"))
                sensor_profile=MLX90393.PROFILE_CURRENT_2023 if dp=="current_2023" else MLX90393.PROFILE_LEGACY_2015
                self.bridge.open(self.dev_index_var.get(),clock_map[self.i2c_clock_var.get()])
                self.sampler=MultiMLX90393Sampler(self.bridge,addrs,profile=sensor_profile)
            configs=self.sampler.initialize(reset=True,config=mcfg)
            expected={k:mcfg.get(k) for k in ("gain","res_x","res_y","res_z","filter","osr","osr2","tcmp") if k in mcfg}
            mismatches=[]
            for c in configs:
                for k,v in expected.items():
                    if k in c and c.get(k)!=v:mismatches.append(f"S{int(c.get('index',0))+1}:{k} expected={v} actual={c.get(k)}")
            if mismatches:
                raise RuntimeError("磁层配置写入后读回不一致 / sensor config verify failed: "+"; ".join(mismatches))
            self.log("Magnetic sensors initialized & profile verified: "+json.dumps(configs,ensure_ascii=False))
            for i,a in enumerate(addrs): self.sensor_vars[i]["addr"].set(f"0x{a:02X}")
        except Exception as e:
            messagebox.showerror("磁传感器初始化失败",str(e)); self.log(f"ERROR magnetic connect: {e}")

    def start_magnetics(self):
        if self.mag_thread and self.mag_thread.is_alive(): return
        if self.sampler is None:
            self.connect_magnetics()
            if self.sampler is None: return
        self.mag_stop.clear()
        interval=max(1,int(self.mag_interval_var.get()))/1000.0
        include_temp=bool(self.include_temp_var.get())
        sampler = self.sampler
        if sampler is None:
            return
        def loop():
            while not self.mag_stop.is_set():
                t0=time.monotonic()
                try:
                    data,meta=sampler.group_sample(include_temperature=include_temp)
                    raw_vector=flatten_group_ut(data)
                    if raw_vector is not None:
                        raw_t_ns=int(meta.get("trigger_base_ns") or time.perf_counter_ns())
                        self.raw_magnetic_recent.append((raw_t_ns,raw_vector))
                    try:self.mag_queue.put_nowait((data,meta))
                    except queue.Full: pass
                    with self.trial_lock:
                        if self.recorder is not None:
                            self.recorder.write_magnetic(data,meta)
                except Exception as e:
                    self.mag_queue.put((None,{"fatal":str(e)})); break
                elapsed=time.monotonic()-t0
                self.mag_stop.wait(max(0.0,interval-elapsed))
        self.mag_thread=threading.Thread(target=loop,daemon=True); self.mag_thread.start()
        self.log(f"Magnetic preview started; requested period={interval*1000:.1f} ms")

    def stop_magnetics(self):
        self.mag_stop.set()
        self.log("Magnetic sampling stop requested")

    def start_cameras(self):
        enabled=[name for name,vars_ in self.cam_vars.items() if vars_["enable"].get()]
        if not enabled:
            self.camera_summary_var.set("相机未启用（当前阶段允许）")
            return
        if cv2 is None or ImageTk is None:
            messagebox.showerror("Camera dependencies",f"OpenCV/Pillow unavailable: {CV2_IMPORT_ERROR or PIL_IMPORT_ERROR}")
            return
        for name,vars_ in self.cam_vars.items():
            if not vars_["enable"].get(): continue
            if name in self.cameras and self.cameras[name].running: continue
            try:
                worker=CameraWorker(name,vars_["index"].get(),vars_["width"].get(),vars_["height"].get(),vars_["fps"].get(),self.preview_queue,self.log)
                worker.start(); self.cameras[name]=worker
            except Exception as e:
                self.log(f"Camera {name} ERROR: {e}")
                messagebox.showwarning(f"Camera {name}",str(e))

    def stop_cameras(self):
        for w in list(self.cameras.values()):
            try:w.stop()
            except Exception as e:self.log(f"Camera stop error: {e}")
        self.cameras.clear()

    def _build_trial_summary(self):
        mode=self._trial_mode_code()
        parts=[mode]
        try:
            if mode not in ("none",):
                if mode not in ("z_sweep",): parts.append("z"+_folder_num(self.z_var.get()))
            if mode in ("slide_x","slide_xy","compound","cross_scan","angle_scan"): parts.append("dx"+_folder_num(self.dx_var.get()))
            if mode in ("slide_y","slide_xy","compound","cross_scan"): parts.append("dy"+_folder_num(self.dy_var.get()))
            if mode=="z_sweep": parts.append("zs"+_safe_slug(self.trial_z_depths_var.get(),36))
            if mode in ("x_sweep","y_sweep"): parts.append("vals"+_safe_slug(self.trial_xy_sweep_var.get(),42))
            if mode=="angle_scan": parts.append("n"+str(int(self.trial_angle_count_var.get())))
            if mode=="circle_scan": parts.extend(["r"+_folder_num(self.trial_circle_radius_var.get()),"n"+str(int(self.trial_circle_points_var.get()))])
            if mode=="grid_points": parts.extend(["hx"+_folder_num(self.trial_grid_half_x_var.get()),"hy"+_folder_num(self.trial_grid_half_y_var.get()),"st"+_folder_num(self.trial_grid_step_var.get())])
            if mode not in ("none",): parts.append("v"+_folder_num(self.speed_var.get()))
            parts.append("h"+_folder_num(self.hold_var.get()))
            if mode != "none":
                parts.append("rep"+str(max(1,int(self.trial_repeat_var.get()))))
                if float(self.repeat_interval_var.get()) > 0:
                    parts.append("ri"+_folder_num(self.repeat_interval_var.get()))
            # Keep the trial self-describing without requiring manual notes.
            if self.indenter_var.get().strip(): parts.append(_safe_slug(self.indenter_var.get().strip(),18))
            if self.material_var.get().strip(): parts.append(_safe_slug(self.material_var.get().strip(),18))
            if self.maglayer_var.get().strip(): parts.append(_safe_slug(self.maglayer_var.get().strip(),18))
        except Exception:
            pass
        return _safe_slug("_".join(parts),110)

    def _update_auto_summary(self):
        if hasattr(self,"auto_summary_var"):
            try:self.auto_summary_var.set(self._build_trial_summary())
            except Exception:self.auto_summary_var.set(self._trial_mode_code())

    def _session_dir(self):
        return Path(self.data_root_var.get()).expanduser() / self.session_var.get().strip()

    def _existing_trial_ids(self):
        """Return Trial IDs anywhere below the current Session.

        V1.2 stores Full-Plan Trials inside Session/Plan_*/Trial_*, so ID scanning
        must be recursive. Manual Session/Trial_* directories remain supported.
        """
        out=[]
        d=self._session_dir()
        if not d.exists(): return out
        for p in d.rglob("Trial_*"):
            if not p.is_dir(): continue
            m=re.match(r"^Trial_(\d+)(?:_|$)",p.name,re.I)
            if m:
                try: out.append(int(m.group(1)))
                except Exception: pass
        return sorted(set(out))

    def scan_next_trial_id(self, silent=False):
        ids=self._existing_trial_ids()
        n=(max(ids)+1) if ids else 1
        self.trial_var.set(f"{n:06d}")
        if not silent:
            self.log(f"Trial ID scan: session={self.session_var.get().strip()} existing={len(ids)} next={n:06d}")
        return n

    def _ensure_free_trial_id(self):
        requested=self.trial_var.get().strip()
        try:n=int(requested)
        except Exception:n=1
        ids=set(self._existing_trial_ids())
        if n in ids:
            if bool(self.auto_trial_id_var.get()):
                n=(max(ids)+1) if ids else 1
                self.trial_var.set(f"{n:06d}")
                self.log(f"Trial ID {requested} already exists; auto-selected {n:06d}")
            else:
                raise FileExistsError(f"Trial ID {requested} already exists in current Session")
        return f"{n:06d}"

    def _current_trial_params(self):
        return TrialParameters(
            session_id=self.session_var.get().strip(), trial_id=self.trial_var.get().strip(),
            command_mode=self._trial_mode_code(), x_mm=float(self.x_var.get()), y_mm=float(self.y_var.get()),
            z_indent_mm=float(self.z_var.get()), dx_mm=float(self.dx_var.get()), dy_mm=float(self.dy_var.get()),
            speed_mm_s=float(self.speed_var.get()), hold_s=float(self.hold_var.get()),
            repeat_count=max(1,int(self.trial_repeat_var.get())), repeat_interval_s=max(0.0,float(self.repeat_interval_var.get())),
            indenter_id=self.indenter_var.get().strip(), material_id=self.material_var.get().strip(),
            magnetic_layer_id=self.maglayer_var.get().strip(), auto_summary=(self._build_trial_summary() if self.auto_summary_enabled_var.get() else self._trial_mode_code()), notes=self.notes_var.get().strip(),
        )

    def _acquisition_meta(self):
        addrs=self._parse_addresses()
        cfg=[]
        try: cfg=self.sampler.current_config() if self.sampler else []
        except Exception as e: cfg=[{"error":str(e)}]
        return {
            "simulate":self.simulate,
            "sensor_addresses7":addrs,
            "sensor_configs":cfg,
            "magnetic_layer_profile_id":self.active_mag_profile_id,
            "magnetic_layer_profile":json.loads(json.dumps(self.maglayer_profiles.get(self.active_mag_profile_id,{}))),
            "i2c_clock":self.i2c_clock_var.get(),
            "requested_magnetic_period_ms":int(self.mag_interval_var.get()),
            "include_temperature_per_cycle":bool(self.include_temp_var.get()),
            "camera_requested":{name:{k:v.get() for k,v in vars_.items()} for name,vars_ in self.cam_vars.items()},
            "printer_reference": {
                "sensor_center_x_mm": float(getattr(self, "sensor_center_x_var", tk.DoubleVar(value=149.0)).get()),
                "sensor_center_y_mm": float(getattr(self, "sensor_center_y_var", tk.DoubleVar(value=147.0)).get()),
                "contact_z_mm": float(getattr(self, "contact_z_var", tk.DoubleVar(value=11.1)).get()),
            },
        }

    def start_trial(self, parent_dir: Optional[Path] = None, folder_tag: str = ""):
        with self.trial_lock:
            if self.recorder is not None:
                messagebox.showwarning("Trial","已有活动 Trial"); return
            try:
                self._ensure_free_trial_id()
                self._update_auto_summary()
                params=self._current_trial_params()
                if not params.session_id or not params.trial_id: raise ValueError("Session/Trial ID不能为空")
                if self.sampler is None:
                    self.connect_magnetics()
                self.start_magnetics()
                self.start_cameras()
                with self.live_predict_lock:
                    auto_prediction=bool(self.live_predictor is not None and self.live_predict_enabled)
                effective_folder_tag=folder_tag or ("WITH_PRED" if auto_prediction else "")
                rec=TrialRecorder(Path(self.data_root_var.get()),params,self.geometry_config,
                                  self._acquisition_meta(), parent_dir=parent_dir,
                                  folder_tag=effective_folder_tag)
                self.recorder=rec
                if auto_prediction:
                    source="full_plan_backtest" if folder_tag=="PRED_BACKTEST" else "automatic_trial_recording"
                    self._live_predict_enable_recording(rec,source)
                for name,w in self.cameras.items():
                    if w.running:
                        info=w.start_recording(rec.trial_dir)
                        if info: rec.register_camera(name,info)
                self.record_state_var.set("● RECORDING")
                self.trial_status_var.set(f"Recording {params.session_id}/Trial_{params.trial_id}: {params.command_mode}")
                self.log(f"Trial started: {rec.trial_dir}")
            except Exception as e:
                self.recorder=None
                messagebox.showerror("开始 Trial 失败",str(e)); self.log(f"ERROR start trial: {e}")

    def stop_trial(self,status="completed"):
        with self.trial_lock:
            if self.recorder is None: return
            rec=self.recorder
            for w in self.cameras.values():
                try:w.stop_recording()
                except Exception as e:self.log(f"Camera record stop error: {e}")
            if rec is self.live_predict_record_recorder:
                rec.write_event("LIVE_PREDICTION_RECORD_STOP", status, "live_prediction")
                rec.update_prediction_recording_stats(self.live_predict_dropped_groups)
            rec.close(status=status)
            self.recorder=None
            if rec is self.live_predict_record_recorder:
                rows=rec.prediction_rows_written
                dropped=self.live_predict_dropped_groups
                self.live_predict_record_recorder=None
                self.live_predict_record_source=""
                self.live_predict_dropped_groups=0
                if hasattr(self,"live_predict_record_state_var"):
                    self.live_predict_record_state_var.set(
                        f"预测记录：自动 | 上一 Trial rows={rows} | dropped input groups={dropped} | {rec.trial_dir.name}"
                    )
            self.record_state_var.set("IDLE")
            self.trial_status_var.set(f"Trial stopped: {rec.trial_dir}")
            self.log(f"Trial closed: {rec.trial_dir}")
            try:self.refresh_data_index(silent=True)
            except Exception:pass

    def manual_event(self,event):
        with self.trial_lock:
            if self.recorder:
                value=self.notes_var.get().strip() if event=="NOTE" else ""
                self.recorder.write_event(event,value,"manual")
                self.log(f"Event marked: {event} {value}")

    def connect_printer(self):
        try:
            self.printer.connect(self.printer_port_var.get(),self.printer_baud_var.get(),self.printer_dry_var.get())
            p=self.printer.query_position()
            self._update_printer_position(p)
            self.log(f"A5S ready: {self.printer_port_var.get()} @ {self.printer_baud_var.get()}")
            self._update_home_state(); self.save_settings(silent=True)
        except Exception as e:
            messagebox.showerror("Printer connect",str(e)); self.log(f"Printer ERROR: {e}")

    def _update_printer_position(self,p):
        if isinstance(p,dict):
            self.printer_pos_var.set(f"X/Y/Z: {p.get('x',0):.3f} / {p.get('y',0):.3f} / {p.get('z',0):.3f}")

    def query_printer_position(self):
        # Manual M114 should never block Tk. It also enables the 1 Hz live monitor,
        # matching the user's expectation that position keeps refreshing after read.
        try:
            self.printer_live_pos_var.set(True)
        except Exception:
            pass
        def task():
            try:
                p=self.printer.query_position()
                self._set_printer_position_async(p)
            except Exception as e:
                self.log(f"M114 error: {e}")
        threading.Thread(target=task,daemon=True).start()

    def _live_printer_position_tick(self):
        """Refresh reported A5S XYZ at 1 Hz while the serial channel is idle.

        During Home/Jog/automatic Trial motion, the motion thread owns the serial
        channel and live polling simply skips that tick. Segment completion callbacks
        still update the position, so acquisition is never delayed by display polling.
        """
        try:
            enabled=bool(getattr(self,"printer_live_pos_var",tk.BooleanVar(value=False)).get())
            busy=(self.printer_thread is not None and self.printer_thread.is_alive()) or bool(self.full_plan_running)
            if enabled and self.printer.connected and not busy and not self._printer_pos_poll_inflight:
                self._printer_pos_poll_inflight=True
                def worker():
                    try:
                        p=self.printer.query_position_live()
                        if p is not None:
                            self._set_printer_position_async(p)
                    except Exception:
                        # Live monitor is best-effort; manual M114 remains the diagnostic path.
                        pass
                    finally:
                        self._printer_pos_poll_inflight=False
                threading.Thread(target=worker,daemon=True).start()
        finally:
            try:
                self._printer_pos_poll_after_id=self.after(1000,self._live_printer_position_tick)
            except Exception:
                self._printer_pos_poll_after_id=None

    def _ensure_printer_connected(self):
        if self.printer.connected:
            return True
        self.connect_printer()
        return bool(self.printer.connected)

    def _set_printer_position_async(self, p):
        try:
            self.after(0, lambda q=dict(p): self._update_printer_position(q))
        except Exception:
            pass

    def printer_home(self,axes):
        axes=str(axes).upper()
        extra=""
        if "Z" in axes:
            extra=("\n\n警告：A5S 的 G28 Z 会执行真实Z归零运动。"
                   "如果样品/夹具位于归零路径，请先移开，避免压头碰撞。")
        if not messagebox.askyesno("Home安全确认",f"确认执行 Home {axes}？\n请确保压头、传感器和夹具不会发生碰撞。{extra}"):
            return
        if not self._ensure_printer_connected():
            return
        auto_raise=bool(self.z_home_raise_enabled_var.get()) if hasattr(self,"z_home_raise_enabled_var") else False
        raise_z=float(self.z_home_raise_var.get()) if hasattr(self,"z_home_raise_var") else 20.0
        zspeed=float(self.z_speed_var.get())
        self.motion_abort_event.clear()
        def task():
            try:
                p=self.printer.home(axes)
                if "Z" in axes and auto_raise and not self.motion_abort_event.is_set():
                    self.log(f"Home {axes} completed; raising Z to configured safe height {raise_z:.3f} mm")
                    self.printer.move_abs(z=raise_z,speed_mm_s=zspeed,mode="home",phase="post_home_safe_z")
                    p=self.printer.query_position()
                self._set_printer_position_async(p)
                self.after(0,self._update_home_state); self.after(0,lambda:self.save_settings(silent=True))
            except Exception as e:self.log(f"Home {axes} error: {e}")
        threading.Thread(target=task,daemon=True).start()

    def set_current_z_coordinate(self):
        if not self._ensure_printer_connected(): return
        z=float(self.g92_z_var.get())
        if not messagebox.askyesno("G92 Z坐标设置",f"确认把打印机当前物理位置定义为 Z={z:.3f} mm？\n这会改变当前坐标解释，但不会移动机械轴。"):
            return
        def task():
            try:
                self.printer.send_line(f"G92 Z{z:.4f}",mode="setup",phase="set_z_coordinate",strict_ack=True)
                # G92 changes the coordinate interpretation used by Contact Z.  Treat Z as
                # un-homed for automatic experiments until the user homes again.
                self.printer.homed_axes.discard("Z")
                p=self.printer.query_position(); self._set_printer_position_async(p)
                self.after(0,self._update_home_state)
                self.log(f"Current Z coordinate set by G92: Z={z:.4f} mm; automatic experiment Z-home state invalidated for safety")
            except Exception as e:self.log(f"G92 Z error: {e}")
        threading.Thread(target=task,daemon=True).start()

    def printer_jog(self,axis,sign):
        if not self._ensure_printer_connected(): return
        step=float(self.jog_step_var.get())*float(sign)
        if axis=="Z" and sign<0 and not self.z_minus_confirmed_once:
            if not messagebox.askyesno("Z-安全确认",f"Z- {abs(step):.3f} mm 通常意味着靠近样品。\n本次程序运行仅提醒这一次，确认继续？"):
                return
            self.z_minus_confirmed_once=True
            self.log("Z- safety confirmation accepted for this program session")
        speed=float(self.z_speed_var.get() if axis=="Z" else self.xy_speed_var.get())
        def task():
            try:
                p=self.printer.jog(axis,step,speed); self._set_printer_position_async(p)
            except Exception as e:self.log(f"Jog {axis}{step:+.3f} error: {e}")
        threading.Thread(target=task,daemon=True).start()

    def _confirm_contact_motion(self, description="接触/压入动作"):
        if self.contact_action_confirmed_once:
            return True
        ok=messagebox.askyesno(
            "首次接触动作安全确认",
            f"即将执行：{description}\n\n"
            f"已设置传感器中心 X={float(self.sensor_center_x_var.get()):.3f}, "
            f"Y={float(self.sensor_center_y_var.get()):.3f}, 接触面 Z={float(self.contact_z_var.get()):.3f} mm。\n"
            "请确认压头、样品和限位均安全。这个提示本次程序运行只确认一次。"
        )
        if ok:
            self.contact_action_confirmed_once=True
            self.log("Contact-motion safety confirmation accepted for this program session")
        return ok

    def go_sensor_center(self, target="safe"):
        if not self._ensure_printer_connected(): return
        if not self._require_home_for_auto(): return
        self.save_settings(silent=True)
        target=str(target)
        if target=="contact" and not self._confirm_contact_motion("移动到传感器中心并到上表面接触高度"):
            return
        x=float(self.sensor_center_x_var.get()); y=float(self.sensor_center_y_var.get())
        contact=float(self.contact_z_var.get()); safe=contact+abs(float(self.safe_lift_var.get()))
        approach=contact+abs(float(self.approach_clearance_var.get()))
        xyspeed=float(self.xy_speed_var.get()); zspeed=float(self.z_speed_var.get())
        if self.printer_thread and self.printer_thread.is_alive():
            messagebox.showwarning("Printer","已有自动动作正在运行"); return
        self.motion_abort_event.clear()
        def task():
            try:
                self.printer.move_abs(z=safe,speed_mm_s=zspeed,mode="manual",phase="center_safe_z")
                if self.motion_abort_event.is_set(): return
                self.printer.move_abs(x=x,y=y,speed_mm_s=xyspeed,mode="manual",phase="center_xy")
                if self.motion_abort_event.is_set(): return
                if target=="approach":
                    self.printer.move_abs(z=approach,speed_mm_s=zspeed,mode="manual",phase="center_approach")
                elif target=="contact":
                    self.printer.move_abs(z=contact,speed_mm_s=min(zspeed,0.8),mode="manual",phase="center_contact")
                p=self.printer.query_position(); self._set_printer_position_async(p)
            except Exception as e:self.log(f"Go sensor center error: {e}")
        self.printer_thread=threading.Thread(target=task,daemon=True); self.printer_thread.start()

    def release_to_safe_z(self):
        if not self._ensure_printer_connected(): return
        safe=float(self.contact_z_var.get())+abs(float(self.safe_lift_var.get()))
        zspeed=float(self.z_speed_var.get())
        def task():
            try:
                self.printer.move_abs(z=safe,speed_mm_s=zspeed,mode="manual",phase="release_safe_z")
                self._set_printer_position_async(self.printer.query_position())
            except Exception as e:self.log(f"Release safe Z error: {e}")
        threading.Thread(target=task,daemon=True).start()

    def sync_center_to_trial(self):
        self.x_var.set(float(self.sensor_center_x_var.get()))
        self.y_var.set(float(self.sensor_center_y_var.get()))
        self.z_var.set(float(self.stage_indent_var.get()) if hasattr(self,"stage_indent_var") else 0.1)
        self.log(f"Trial center synced: X={self.x_var.get():.3f}, Y={self.y_var.get():.3f}, contact Z={self.contact_z_var.get():.3f}")

    def _motion_event(self,event,value=""):
        with self.trial_lock:
            if self.recorder:
                self.recorder.write_event(event,value,"printer")

    def _capture_pre_action_b0(self, action, phase):
        """Record a causal unloaded window and align live fit B0 with training."""
        activated=False
        with self.live_predict_lock:
            predictor=self.live_predictor if self.live_predict_enabled else None
            if isinstance(predictor,LiveNormalizedFitPredictor):
                predictor.request_confirmed_baseline(PRE_ACTION_B0_SECONDS,phase)
                activated=True
        payload={"phase":phase,"capture_seconds":PRE_ACTION_B0_SECONDS,
                 "dwell_seconds":PRE_ACTION_B0_DWELL_S,
                 "live_normalized_fit_updated":activated,
                 "policy":"confirmed lifted pose; causal samples only"}
        self._motion_event("PRE_ACTION_B0_WINDOW",json.dumps(payload,ensure_ascii=False))
        self.printer.dwell(PRE_ACTION_B0_DWELL_S,mode=action,phase=phase+"_hold")

    def _run_motion_thread(self, target, name="motion"):
        if self.printer_thread and self.printer_thread.is_alive():
            messagebox.showwarning("Printer","已有自动动作正在运行"); return False
        self.motion_abort_event.clear()
        def wrapped():
            try: target()
            except Exception as e:
                self.log(f"{name} ERROR: {e}")
                self._motion_event("MOTION_ERROR",str(e))
        self.printer_thread=threading.Thread(target=wrapped,daemon=True)
        self.printer_thread.start()
        return True

    def _point_touch_sequence(self, depth, hold, repeats, interval, zspeed):
        x=float(self.sensor_center_x_var.get()); y=float(self.sensor_center_y_var.get())
        contact=float(self.contact_z_var.get()); safe=contact+abs(float(self.safe_lift_var.get()))
        approach=contact+abs(float(self.approach_clearance_var.get()))
        xyspeed=float(self.xy_speed_var.get())
        depth=max(0.0,float(depth)); repeats=max(1,int(repeats))
        self.printer.move_abs(z=safe,speed_mm_s=max(zspeed,0.2),mode="point_touch",phase="safe_z")
        self.printer.move_abs(x=x,y=y,speed_mm_s=xyspeed,mode="point_touch",phase="center_xy")
        for k in range(repeats):
            if self.motion_abort_event.is_set(): break
            self._motion_event("POINT_TOUCH_BEGIN",json.dumps({"repeat":k+1,"depth_mm":depth},ensure_ascii=False))
            self.printer.move_abs(z=approach,speed_mm_s=zspeed,mode="point_touch",phase="approach")
            self._capture_pre_action_b0("point_touch","point_touch_pre_action_b0")
            self.printer.move_abs(z=contact,speed_mm_s=zspeed,mode="point_touch",phase="contact")
            self.printer.move_abs(z=contact-depth,speed_mm_s=zspeed,mode="point_touch",phase="indent")
            self.printer.dwell(hold,mode="point_touch",phase="hold")
            self.printer.move_abs(z=approach,speed_mm_s=zspeed,mode="point_touch",phase="retract")
            self._motion_event("POINT_TOUCH_END",str(k+1))
            if interval>0 and k<repeats-1: self.printer.dwell(interval,mode="point_touch",phase="interval")
        self.printer.move_abs(z=safe,speed_mm_s=max(zspeed,0.2),mode="point_touch",phase="final_safe_z")
        self._set_printer_position_async(self.printer.query_position())

    def run_point_touch(self, record=False, force_repeat=False):
        if not self._ensure_printer_connected(): return
        if not self._require_home_for_auto(): return
        self.save_settings(silent=True)
        if not self._confirm_contact_motion("点触测试（会从接触面继续向 Z- 压入）"): return
        depth=float(self.point_depth_var.get()); hold=float(self.point_hold_var.get())
        repeats=int(self.point_repeat_var.get()) if (force_repeat or record) else 1
        interval=float(self.point_interval_var.get()); zspeed=float(self.point_speed_var.get())
        if depth<0 or depth>3.0:
            messagebox.showerror("点触参数","压入深度必须在 0~3 mm 的当前软件安全范围内"); return
        if record:
            self.mode_var.set(TRIAL_MODE_LABELS["press"]); self._update_trial_param_states(); self.sync_center_to_trial(); self.z_var.set(depth); self.speed_var.set(zspeed); self.hold_var.set(hold)
            self.start_trial()
            if self.recorder is None:return
        rec_ref=self.recorder
        def job():
            try:
                if bool(self.auto_fan_off_var.get()): self.printer.fan_off()
                self._motion_event("POINT_TOUCH_CONFIG",json.dumps({"depth_mm":depth,"hold_s":hold,"repeats":repeats,"interval_s":interval,"z_speed_mm_s":zspeed},ensure_ascii=False))
                self._point_touch_sequence(depth,hold,repeats,interval,zspeed)
                self.log(f"Point touch completed: depth={depth:.3f} mm, repeats={repeats}")
                if record and rec_ref is not None:
                    self.after(0,lambda:self.stop_trial("completed")); self.after(100,self.increment_trial)
            except Exception as e:
                self.log(f"Point touch ERROR: {e}")
                if record and rec_ref is not None:self.after(0,lambda:self.stop_trial("aborted"))
        self._run_motion_thread(job,"point_touch")

    @staticmethod
    def _grid_offsets(half_span, step):
        half=abs(float(half_span)); step=abs(float(step))
        if step<=0: raise ValueError("Grid step must be >0")
        vals=[]; v=-half
        while v<=half+1e-9:
            vals.append(round(v,6)); v+=step
        if not vals or abs(vals[-1]-half)>1e-6: vals.append(half)
        return vals

    def _stage_experiment_config(self):
        return {
            "action":self._stage_action_code(),
            "center_x_mm":float(self.sensor_center_x_var.get()),
            "center_y_mm":float(self.sensor_center_y_var.get()),
            "contact_z_mm":float(self.contact_z_var.get()),
            "safe_lift_mm":float(self.safe_lift_var.get()),
            "approach_clearance_mm":float(self.approach_clearance_var.get()),
            "indent_mm":float(self.stage_indent_var.get()),
            "dx_mm":float(self.stage_dx_var.get()),
            "dy_mm":float(self.stage_dy_var.get()),
            "speed_mm_s":float(self.stage_speed_var.get()),
            "z_speed_mm_s":float(self.stage_z_speed_var.get()),
            "hold_s":float(self.stage_hold_var.get()),
            "repeats":max(1,int(self.stage_repeat_var.get())),
            "z_depths_mm":self.stage_depths_var.get().strip(),
            "xy_sweep_values_mm":self.stage_xy_sweep_var.get().strip(),
            "angle_count":max(1,int(self.stage_angle_count_var.get())),
            "circle_radius_mm":max(0.0,float(self.stage_circle_radius_var.get())),
            "circle_points":max(4,int(self.stage_circle_points_var.get())),
            "grid_half_x_mm":float(self.grid_half_x_var.get()),
            "grid_half_y_mm":float(self.grid_half_y_var.get()),
            "grid_step_mm":float(self.grid_step_var.get()),
        }

    def _trial_action_config(self):
        return {
            "action":self._trial_mode_code(),
            "center_x_mm":float(self.x_var.get()),"center_y_mm":float(self.y_var.get()),
            "contact_z_mm":float(self.contact_z_var.get()),"safe_lift_mm":float(self.safe_lift_var.get()),
            "approach_clearance_mm":float(self.approach_clearance_var.get()),
            "indent_mm":float(self.z_var.get()),"dx_mm":float(self.dx_var.get()),"dy_mm":float(self.dy_var.get()),
            "speed_mm_s":float(self.speed_var.get()),"z_speed_mm_s":float(self.stage_z_speed_var.get() if hasattr(self,"stage_z_speed_var") else self.z_speed_var.get()),
            "hold_s":float(self.hold_var.get()),"repeats":max(1,int(self.trial_repeat_var.get())),"repeat_interval_s":max(0.0,float(self.repeat_interval_var.get())),
            "z_depths_mm":self.trial_z_depths_var.get().strip(),"xy_sweep_values_mm":self.trial_xy_sweep_var.get().strip(),
            "angle_count":max(1,int(self.trial_angle_count_var.get())),"circle_radius_mm":max(0.0,float(self.trial_circle_radius_var.get())),
            "circle_points":max(4,int(self.trial_circle_points_var.get())),
            "grid_half_x_mm":float(self.trial_grid_half_x_var.get()),"grid_half_y_mm":float(self.trial_grid_half_y_var.get()),"grid_step_mm":float(self.trial_grid_step_var.get()),
        }

    def _execute_stage_experiment(self,cfg):
        action=cfg["action"]; x0=cfg["center_x_mm"]; y0=cfg["center_y_mm"]; contact=cfg["contact_z_mm"]
        safe=contact+abs(cfg["safe_lift_mm"]); approach=contact+abs(cfg["approach_clearance_mm"])
        depth=max(0.0,cfg.get("indent_mm",0.0)); zpress=contact-depth
        speed=max(0.05,float(cfg.get("speed_mm_s",1.0)))
        zspeed=max(0.05,float(cfg.get("z_speed_mm_s",0.5)))
        hold=max(0.0,float(cfg.get("hold_s",0.0))); recovery=max(0.0,float(cfg.get("recovery_s",hold))); reps=max(1,int(cfg.get("repeats",1))); repeat_interval=max(0.0,float(cfg.get("repeat_interval_s",0.0)))

        def chk():
            if self.motion_abort_event.is_set(): raise RuntimeError("MOTION_ABORTED")
        def at_center_safe():
            chk(); self.printer.move_abs(z=safe,speed_mm_s=max(zspeed,0.2),mode=action,phase="safe_z")
            chk(); self.printer.move_abs(x=x0,y=y0,speed_mm_s=max(speed,0.2),mode=action,phase="center_xy")
        def preload(phase="preload"):
            chk(); self.printer.move_abs(z=approach,speed_mm_s=zspeed,mode=action,phase=phase+"_approach")
            self._capture_pre_action_b0(action,phase+"_b0")
            chk(); self.printer.move_abs(z=zpress,speed_mm_s=zspeed,mode=action,phase=phase+"_indent")
            if hold>0: self.printer.dwell(hold,mode=action,phase=phase+"_hold")
        def values(text):
            return [float(v.strip()) for v in str(text).replace(";",",").split(",") if v.strip()]

        # Re-run the full validator immediately before issuing any command.
        # This is intentional redundancy: UI/pre-plan validation can never be the only safety layer.
        envelope=self._validate_action_cfg(cfg)
        self._motion_event("SAFETY_PREFLIGHT_PASS",json.dumps(envelope,ensure_ascii=False))
        self.log(f"SAFETY PREFLIGHT PASS | X[{envelope['x_min']:.3f},{envelope['x_max']:.3f}] "
                 f"Y[{envelope['y_min']:.3f},{envelope['y_max']:.3f}] Z[{envelope['z_min']:.3f},{envelope['z_max']:.3f}] mm")

        # Planned Z envelope: lateral scan phases are forbidden from issuing Z targets.
        # Only preload/retract/Z-sweep/grid/compound may change Z.
        if action in ("z_sweep","z_path"):
            _zs = values(cfg.get("z_depths_mm", ""))
            _max_depth = max(_zs) if _zs else 0.0
        else:
            _max_depth = depth
        z_min_planned = contact - max(0.0, _max_depth)
        z_max_planned = safe
        self._motion_event("Z_ENVELOPE", json.dumps({
            "contact_z_mm": contact, "z_min_planned_mm": z_min_planned,
            "z_max_planned_mm": z_max_planned, "action": action
        }, ensure_ascii=False))
        self.log(f"Motion Z envelope [{z_min_planned:.3f}, {z_max_planned:.3f}] mm; lateral phases issue XY only")

        # Absolute-coordinate contract for every experiment.
        self.printer.send_line("G90", mode=action, phase="experiment_absolute_mode", strict_ack=True)
        at_center_safe()
        if action=="none":
            if hold>0:self.printer.dwell(hold,mode=action,phase="baseline_hold")
        else:
            for r in range(reps):
                chk(); self._motion_event("ACTION_REPEAT_BEGIN",json.dumps({"action":action,"repeat":r+1},ensure_ascii=False))
                if action in ("press","press_cycle"):
                    preload("press"); chk(); self.printer.move_abs(z=approach,speed_mm_s=zspeed,mode=action,phase="retract")
                elif action in ("slide_x","slide_y","slide_xy"):
                    preload("preload")
                    xt=x0+(cfg.get("dx_mm",0.0) if action in ("slide_x","slide_xy") else 0.0)
                    yt=y0+(cfg.get("dy_mm",0.0) if action in ("slide_y","slide_xy") else 0.0)
                    chk(); self.printer.move_abs(x=xt,y=yt,speed_mm_s=speed,mode=action,phase=action)
                    if hold>0:self.printer.dwell(hold,mode=action,phase="post_slide_hold")
                    chk(); self.printer.move_abs(z=approach,speed_mm_s=zspeed,mode=action,phase="retract")
                    chk(); self.printer.move_abs(x=x0,y=y0,speed_mm_s=speed,mode=action,phase="recenter")
                elif action=="compound":
                    # Correct compound semantics: establish contact at center first, then apply
                    # ONLY requested dx/dy and indent together from the physical contact point.
                    chk(); self.printer.move_abs(z=approach,speed_mm_s=zspeed,mode=action,phase="approach")
                    self._capture_pre_action_b0(action,"compound_pre_action_b0")
                    chk(); self.printer.move_abs(z=contact,speed_mm_s=min(zspeed,0.8),mode=action,phase="contact_onset")
                    xt=x0+cfg.get("dx_mm",0.0); yt=y0+cfg.get("dy_mm",0.0)
                    self.log(f"COMPOUND TARGET | contact center=({x0:.3f},{y0:.3f},{contact:.3f}) -> final=({xt:.3f},{yt:.3f},{zpress:.3f}) mm")
                    chk(); self.printer.move_abs(x=xt,y=yt,z=zpress,speed_mm_s=min(speed,zspeed),mode=action,phase="xyz_compound")
                    if hold>0:self.printer.dwell(hold,mode=action,phase="compound_hold")
                    # Lift fully clear before any XY return, preventing drag across the sample.
                    chk(); self.printer.move_abs(z=safe,speed_mm_s=zspeed,mode=action,phase="retract")
                    chk(); self.printer.move_abs(x=x0,y=y0,speed_mm_s=speed,mode=action,phase="recenter")
                elif action=="z_sweep":
                    ds=values(cfg.get("z_depths_mm",""))
                    if not ds: raise ValueError("Z阶梯深度列表为空")
                    for i,d in enumerate(ds):
                        chk();
                        if d<0 or d>3.0: raise ValueError(f"Z sweep depth out of 0~3 mm safety range: {d}")
                        self._motion_event("Z_SWEEP_POINT",json.dumps({"index":i,"depth_mm":d},ensure_ascii=False))
                        self._capture_pre_action_b0(action,f"depth_{i}_pre_action_b0")
                        self.printer.move_abs(z=contact-d,speed_mm_s=zspeed,mode=action,phase=f"depth_{i}")
                        if hold>0:self.printer.dwell(hold,mode=action,phase=f"hold_{i}")
                        self.printer.move_abs(z=approach,speed_mm_s=zspeed,mode=action,phase=f"retract_{i}")
                elif action=="z_path":
                    ds=values(cfg.get("z_depths_mm",""))
                    if not ds: raise ValueError("Z路径深度列表为空")
                    # Establish the same contact-relative zero before following the path.
                    chk(); self.printer.move_abs(z=approach,speed_mm_s=zspeed,mode=action,phase="path_approach")
                    self._capture_pre_action_b0(action,"path_pre_action_b0")
                    chk(); self.printer.move_abs(z=contact,speed_mm_s=min(zspeed,0.8),mode=action,phase="path_contact_zero")
                    for i,d in enumerate(ds):
                        chk()
                        if d<0 or d>3.0: raise ValueError(f"Z path depth out of 0~3 mm safety range: {d}")
                        self._motion_event("Z_PATH_POINT",json.dumps({"index":i,"depth_mm":d},ensure_ascii=False))
                        self.printer.move_abs(z=contact-d,speed_mm_s=zspeed,mode=action,phase=f"path_{i}")
                        if hold>0:self.printer.dwell(hold,mode=action,phase=f"path_hold_{i}")
                    # Recovery is measured at the contact-relative zero, then the indenter is lifted clear.
                    chk(); self.printer.move_abs(z=contact,speed_mm_s=zspeed,mode=action,phase="path_return_contact")
                    if recovery>0:self.printer.dwell(recovery,mode=action,phase="path_recovery")
                    chk(); self.printer.move_abs(z=approach,speed_mm_s=zspeed,mode=action,phase="path_release")
                elif action in ("x_sweep","y_sweep"):
                    preload("preload")
                    vals=values(cfg.get("xy_sweep_values_mm",""))
                    if not vals: raise ValueError("XY阶梯列表为空")
                    for i,v in enumerate(vals):
                        chk(); xt=x0+(v if action=="x_sweep" else 0.0); yt=y0+(v if action=="y_sweep" else 0.0)
                        self._motion_event("XY_SWEEP_POINT",json.dumps({"index":i,"value_mm":v,"axis":action[0]},ensure_ascii=False))
                        self.printer.move_abs(x=xt,y=yt,speed_mm_s=speed,mode=action,phase=f"sweep_{i}")
                        if hold>0:self.printer.dwell(hold,mode=action,phase=f"hold_{i}")
                    self.printer.move_abs(z=approach,speed_mm_s=zspeed,mode=action,phase="retract")
                    self.printer.move_abs(x=x0,y=y0,speed_mm_s=speed,mode=action,phase="recenter")
                elif action=="cross_scan":
                    preload("preload")
                    pts=[(cfg.get("dx_mm",0),0),(0,0),(-cfg.get("dx_mm",0),0),(0,0),(0,cfg.get("dy_mm",0)),(0,0),(0,-cfg.get("dy_mm",0)),(0,0)]
                    for i,(ox,oy) in enumerate(pts):
                        chk(); self._motion_event("CROSS_POINT",json.dumps({"index":i,"ox":ox,"oy":oy},ensure_ascii=False))
                        self.printer.move_abs(x=x0+ox,y=y0+oy,speed_mm_s=speed,mode=action,phase=f"cross_{i}")
                    self.printer.move_abs(z=approach,speed_mm_s=zspeed,mode=action,phase="retract")
                elif action=="angle_scan":
                    preload("preload"); radius=abs(float(cfg.get("dx_mm",2.0))); n=max(1,int(cfg.get("angle_count",8)))
                    for i in range(n):
                        chk(); th=2*math.pi*i/n; ox=radius*math.cos(th); oy=radius*math.sin(th)
                        self._motion_event("ANGLE_POINT",json.dumps({"index":i,"deg":360*i/n,"ox":ox,"oy":oy},ensure_ascii=False))
                        self.printer.move_abs(x=x0+ox,y=y0+oy,speed_mm_s=speed,mode=action,phase=f"angle_{i}")
                        if hold>0:self.printer.dwell(hold,mode=action,phase=f"hold_{i}")
                        self.printer.move_abs(x=x0,y=y0,speed_mm_s=speed,mode=action,phase=f"angle_return_{i}")
                    self.printer.move_abs(z=approach,speed_mm_s=zspeed,mode=action,phase="retract")
                elif action=="circle_scan":
                    preload("preload"); radius=abs(float(cfg.get("circle_radius_mm",2.0))); n=max(4,int(cfg.get("circle_points",24)))
                    self.printer.move_abs(x=x0+radius,y=y0,speed_mm_s=speed,mode=action,phase="circle_start")
                    for i in range(1,n+1):
                        chk(); th=2*math.pi*i/n
                        self.printer.move_abs(x=x0+radius*math.cos(th),y=y0+radius*math.sin(th),speed_mm_s=speed,mode=action,phase=f"circle_{i}")
                    if hold>0:self.printer.dwell(hold,mode=action,phase="circle_hold")
                    self.printer.move_abs(z=approach,speed_mm_s=zspeed,mode=action,phase="retract")
                    self.printer.move_abs(x=x0,y=y0,speed_mm_s=speed,mode=action,phase="recenter")
                elif action=="grid_points":
                    xs=self._grid_offsets(cfg["grid_half_x_mm"],cfg["grid_step_mm"])
                    y_text=str(cfg.get("grid_y_offsets_mm","")).strip()
                    ys=values(y_text) if y_text else self._grid_offsets(cfg["grid_half_y_mm"],cfg["grid_step_mm"])
                    point_count=len(xs)*len(ys)
                    if point_count>AUTO_MAX_GRID_POINTS:
                        raise ValueError(
                            f"Grid points {point_count} exceed safety cap {AUTO_MAX_GRID_POINTS}; "
                            "split the grid into multiple Trials")
                    base_points=[]
                    for row_i,oy in enumerate(ys):
                        row_xs=xs if row_i%2==0 else list(reversed(xs))
                        for ox in row_xs:
                            base_points.append((len(base_points),ox,oy))
                    # Repeat 2 follows the exact reverse path.  This balances drift and
                    # path-order effects without changing the predeclared train/test role.
                    run_points=base_points if r%2==0 else list(reversed(base_points))
                    ref_interval=max(0,int(cfg.get("grid_reference_interval",0)))
                    anchor_origin=float(cfg.get("grid_anchor_origin_mm",-10.0))
                    anchor_step=float(cfg.get("grid_anchor_step_mm",4.0))
                    def is_anchor(v):
                        if anchor_step<=0:return False
                        q=(float(v)-anchor_origin)/anchor_step
                        return abs(q-round(q))<=1e-7
                    for seq_i,(canonical_i,ox,oy) in enumerate(run_points,1):
                        split_role=("anchor_calibration" if is_anchor(ox) and is_anchor(oy)
                                    else "unseen_interpolation_validation")
                        chk(); self.printer.move_abs(z=approach,speed_mm_s=zspeed,mode=action,phase="grid_approach")
                        self.printer.move_abs(x=x0+ox,y=y0+oy,speed_mm_s=speed,mode=action,phase="grid_xy")
                        self._capture_pre_action_b0(action,"grid_pre_action_b0")
                        point_payload={
                            "index":canonical_i,"sequence_index":seq_i-1,"repeat":r+1,
                            "offset_x_mm":ox,"offset_y_mm":oy,
                            "x_mm":x0+ox,"y_mm":y0+oy,"depth_mm":depth,
                            "split_role":split_role,
                            "split_policy":cfg.get("grid_split_policy","unspecified"),
                        }
                        self._motion_event("GRID_POINT",json.dumps(point_payload,ensure_ascii=False))
                        self.printer.move_abs(z=zpress,speed_mm_s=zspeed,mode=action,phase="grid_indent")
                        if hold>0:self.printer.dwell(hold,mode=action,phase="grid_hold")
                        self.printer.move_abs(z=approach,speed_mm_s=zspeed,mode=action,phase="grid_retract")
                        self._motion_event("GRID_POINT_END",json.dumps(point_payload,ensure_ascii=False))
                        if ref_interval>0 and seq_i<len(run_points) and seq_i%ref_interval==0:
                            # An extra center contact is a drift/QC reference only.  It
                            # is explicitly tagged and must never enter the 121-point fit.
                            chk(); self.printer.move_abs(z=safe,speed_mm_s=zspeed,mode=action,phase="grid_reference_safe")
                            self.printer.move_abs(x=x0,y=y0,speed_mm_s=speed,mode=action,phase="grid_reference_xy")
                            self.printer.move_abs(z=approach,speed_mm_s=zspeed,mode=action,phase="grid_reference_approach")
                            self._capture_pre_action_b0(action,"grid_reference_pre_action_b0")
                            ref_payload={"after_sequence_index":seq_i-1,"repeat":r+1,
                                         "x_mm":x0,"y_mm":y0,"depth_mm":depth,
                                         "role":"drift_monitor_exclude_from_fit"}
                            self._motion_event("GRID_REFERENCE_BEGIN",json.dumps(ref_payload,ensure_ascii=False))
                            self.printer.move_abs(z=zpress,speed_mm_s=zspeed,mode=action,phase="grid_reference_indent")
                            if hold>0:self.printer.dwell(hold,mode=action,phase="grid_reference_hold")
                            self.printer.move_abs(z=approach,speed_mm_s=zspeed,mode=action,phase="grid_reference_retract")
                            self._motion_event("GRID_REFERENCE_END",json.dumps(ref_payload,ensure_ascii=False))
                    # Finish each grid repeat at the known center before the next repeat.
                    self.printer.move_abs(z=safe,speed_mm_s=zspeed,mode=action,phase="safe_z_grid_end")
                    self.printer.move_abs(x=x0,y=y0,speed_mm_s=speed,mode=action,phase="grid_recenter_safe")
                elif action=="load_hold_release":
                    chk(); self.printer.move_abs(z=approach,speed_mm_s=zspeed,mode=action,phase="approach")
                    self._capture_pre_action_b0(action,"load_pre_action_b0")
                    self.printer.move_abs(z=zpress,speed_mm_s=zspeed,mode=action,phase="load")
                    if hold>0:self.printer.dwell(hold,mode=action,phase="steady_hold")
                    self.printer.move_abs(z=contact,speed_mm_s=zspeed,mode=action,phase="unload_to_contact")
                    if recovery>0:self.printer.dwell(recovery,mode=action,phase="recovery_hold")
                    self.printer.move_abs(z=approach,speed_mm_s=zspeed,mode=action,phase="release")
                else:
                    raise ValueError(f"Unknown experiment action: {action}")
                self._motion_event("ACTION_REPEAT_END",str(r+1))
                if r < reps-1 and repeat_interval > 0:
                    self._motion_event("ACTION_REPEAT_INTERVAL",json.dumps({"after_repeat":r+1,"seconds":repeat_interval},ensure_ascii=False))
                    self.printer.dwell(repeat_interval,mode=action,phase="repeat_interval")
        if not self.motion_abort_event.is_set():
            self.printer.move_abs(z=safe,speed_mm_s=max(zspeed,0.2),mode=action,phase="final_safe_z")
            self._set_printer_position_async(self.printer.query_position())
        else:
            self._motion_event("MOTION_ABORTED","M410/software abort")

    def run_stage_experiment(self, record=False):
        if not self._ensure_printer_connected(): return
        if not self._require_home_for_auto(): return
        self.save_settings(silent=True)
        cfg=self._stage_experiment_config()
        action=cfg["action"]
        if action in ("press","press_cycle","slide_x","slide_y","slide_xy","compound","x_sweep","y_sweep","cross_scan","angle_scan","circle_scan","grid_points","load_hold_release"):
            if cfg["indent_mm"]<0 or cfg["indent_mm"]>3.0:
                messagebox.showerror("实验参数 / Experiment Parameters","压入深度 / Indent depth 必须在 0~3 mm 当前软件安全范围内"); return
        if action=="z_sweep":
            try:
                depths=[float(v.strip()) for v in cfg["z_depths_mm"].replace(";",",").split(",") if v.strip()]
            except Exception:
                messagebox.showerror("实验参数 / Experiment Parameters","Z阶梯深度列表 / Z sweep depths 格式错误"); return
            if not depths or any(d<0 or d>3.0 for d in depths):
                messagebox.showerror("实验参数 / Experiment Parameters","Z阶梯深度 / Z sweep depth 每个值必须在 0~3 mm"); return
        try:self._validate_action_cfg(cfg)
        except Exception as e:
            messagebox.showerror("实验参数 / Experiment Parameters",str(e)); return
        if action!="none" and not self._confirm_contact_motion(f"实验动作 {cfg['action']}（按已保存 Contact Z 参数执行）"): return
        if record:
            # Synchronize Printer-page action/config into Trial metadata/UI. Manual Notes remain optional.
            self.mode_var.set(TRIAL_MODE_LABELS.get(cfg["action"],TRIAL_MODE_LABELS["press"])); self._update_trial_param_states()
            self.x_var.set(cfg["center_x_mm"]); self.y_var.set(cfg["center_y_mm"]); self.z_var.set(cfg["indent_mm"])
            self.dx_var.set(cfg["dx_mm"]); self.dy_var.set(cfg["dy_mm"]); self.speed_var.set(cfg["speed_mm_s"]); self.hold_var.set(cfg["hold_s"])
            self.trial_z_depths_var.set(cfg.get("z_depths_mm","")); self.trial_xy_sweep_var.set(cfg.get("xy_sweep_values_mm",""))
            self.trial_angle_count_var.set(cfg.get("angle_count",8)); self.trial_circle_radius_var.set(cfg.get("circle_radius_mm",2.0)); self.trial_circle_points_var.set(cfg.get("circle_points",24))
            self.trial_grid_half_x_var.set(cfg.get("grid_half_x_mm",4.0)); self.trial_grid_half_y_var.set(cfg.get("grid_half_y_mm",4.0)); self.trial_grid_step_var.set(cfg.get("grid_step_mm",2.0))
            self._update_trial_param_states(); self._update_auto_summary()
            self.start_trial()
            if self.recorder is None:return
        rec_ref=self.recorder
        def job():
            try:
                if bool(self.auto_fan_off_var.get()):
                    self.printer.fan_off(); self.log("Experiment preflight: M107 sent to reduce fan vibration")
                self._motion_event("EXPERIMENT_CONFIG",json.dumps(cfg,ensure_ascii=False))
                self._motion_event("EXPERIMENT_START",cfg["action"])
                self._execute_stage_experiment(cfg)
                self._motion_event("EXPERIMENT_END",cfg["action"])
                self.log(f"Experiment action completed: {cfg['action']}")
                if record and rec_ref is not None:
                    self.after(0,lambda:self.stop_trial("completed")); self.after(100,self.increment_trial)
            except Exception as e:
                self.log(f"Experiment action ERROR: {e}")
                self._motion_event("EXPERIMENT_ERROR",str(e))
                if record and rec_ref is not None:self.after(0,lambda:self.stop_trial("aborted"))
        self._run_motion_thread(job,"stage_experiment")

    def printer_quick_stop(self):
        # V1.3.4-R1: this is deliberately NOT presented as an emergency stop.
        # It stops the host-side experiment sequence and makes one best-effort M410 request.
        # On this A5S old Marlin, only physical power-off is treated as emergency stop.
        self.motion_abort_event.set(); self.batch_abort_event.set(); self.full_plan_soft_stop_event.set()
        self.record_state_var.set("⚠ SEQUENCE STOP REQUESTED")
        def task():
            try:
                self.printer.send_urgent("M410", "sequence_stop_best_effort", repeat=1)
                self.log("Experiment sequence stop requested; M410 sent best-effort. For emergency, use physical power OFF.")
                self.after(0,lambda:self.record_state_var.set("IDLE / SEQUENCE STOPPED" if self.recorder is None else "● RECORDING / SEQUENCE STOPPED"))
            except Exception as e:
                self.log(f"Sequence stop request error: {e}")
        threading.Thread(target=task,daemon=True).start()

    def printer_fan_off(self):
        try:self.printer.fan_off(); self.log("M107 sent: controllable/part-cooling fan OFF request")
        except Exception as e:self.log(f"M107 error: {e}")

    def printer_emergency_stop(self):
        if not messagebox.askyesno("M112紧急停止","确认发送 M112？通常需要重启/断电恢复。"):
            return
        self.printer_emergency_stop_immediate()

    def printer_emergency_stop_immediate(self):
        self.motion_abort_event.set(); self.batch_abort_event.set(); self.full_plan_soft_stop_event.set()
        self.record_state_var.set("⚠ EMERGENCY STOP REQUESTED")
        self.printer.homed_axes.clear(); self._update_home_state()
        def task():
            try:
                self.printer.emergency_stop()
                self.log("!!! M112 E-STOP x3 SENT. POWER-CYCLE / RESET and HOME XYZ required. !!!")
                self.after(0,lambda:self.record_state_var.set("⚠ E-STOP / POWER-CYCLE REQUIRED"))
            except Exception as e:
                self.log(f"M112 error: {e}")
        threading.Thread(target=task,daemon=True).start()

    def _record_stage(self,row):
        command=str(row.get("command", ""))
        phase=str(row.get("phase", ""))
        safe_prefixes=("safe_z","approach","press_approach","preload_approach","retract",
                       "grid_approach","grid_retract","grid_reference_safe",
                       "grid_reference_approach","final_safe_z","path_approach","path_release",
                       "dynamic_b0_safe_z","recenter")
        hint=None
        if command.startswith("G1 "):
            if row.get("z_mm") is not None:
                self._live_stage_pending_safe_height=any(
                    phase.startswith(prefix) for prefix in safe_prefixes)
                if not self._live_stage_pending_safe_height:
                    self._live_stage_safe_height=False
            else:
                self._live_stage_pending_safe_height=self._live_stage_safe_height
            hint=False
        elif command=="HOST_WAIT":
            self._live_stage_safe_height=bool(self._live_stage_pending_safe_height)
            hint=self._live_stage_safe_height
        if hint is not None:
            with self.live_predict_lock:
                predictor=self.live_predictor
                if isinstance(predictor,LiveNormalizedFitPredictor):
                    predictor.set_no_contact_hint(hint,phase)
        with self.trial_lock:
            if self.recorder:
                self.recorder.write_stage(row)

    def _printer_send(self,cmd):
        try:
            text=str(cmd).strip()
            upper0=text.upper()
            if upper0.startswith("M400") or upper0.startswith("G4"):
                messagebox.showwarning(
                    "阻塞命令已禁用 / Blocking command disabled",
                    "本A5S旧版Marlin上，M400/G4可能阻塞命令解析并导致软件STOP/E-STOP不能及时处理。\n"
                    "V1.3.4自动实验已改为可中断的主机侧等待，因此UI中不再允许手动发送 M400/G4。"
                )
                self.log(f"SAFETY BLOCK manual blocking command: {text}")
                return
            self.printer.send_line(text)
            upper=text.upper()
            if upper.startswith("G92") or upper.startswith("M206") or upper.startswith("G10"):
                self.printer.homed_axes.clear()
                self._update_home_state()
                self.log("SAFETY: manual coordinate/offset command detected; automatic Home state invalidated. Home XYZ again before automatic experiments.")
        except Exception as e:messagebox.showerror("G-code",str(e))

    def send_manual_gcode(self): self._printer_send(self.gcode_var.get())

    def _validate_action_cfg(self,cfg):
        """Full automatic-motion preflight. No experiment command is sent until this passes."""
        action=str(cfg.get("action",""))
        if action not in EXPERIMENT_ACTION_LABELS:
            raise ValueError(f"未知实验动作 / Unknown action: {action}")

        def finite(name, default=0.0):
            v=float(cfg.get(name,default))
            if not math.isfinite(v): raise ValueError(f"{name} must be finite")
            return v
        x0=finite("center_x_mm"); y0=finite("center_y_mm"); contact=finite("contact_z_mm")
        safe_lift=finite("safe_lift_mm"); approach_gap=finite("approach_clearance_mm")
        indent=finite("indent_mm"); dx=finite("dx_mm"); dy=finite("dy_mm")
        speed=finite("speed_mm_s",1.0); zspeed=finite("z_speed_mm_s",0.5)
        hold=finite("hold_s",0.0); recovery=finite("recovery_s",hold); repeat_interval=finite("repeat_interval_s",0.0)
        reps=int(cfg.get("repeats",1))

        if not (0.0 <= safe_lift <= 30.0): raise ValueError("Safe lift must be 0~30 mm")
        if not (0.0 <= approach_gap <= safe_lift): raise ValueError("Approach gap must satisfy 0 <= gap <= Safe lift")
        if not (0.05 <= speed <= AUTO_MAX_XY_SPEED_MM_S): raise ValueError(f"XY speed must be 0.05~{AUTO_MAX_XY_SPEED_MM_S:g} mm/s")
        if not (0.05 <= zspeed <= AUTO_MAX_Z_SPEED_MM_S): raise ValueError(f"Z speed must be 0.05~{AUTO_MAX_Z_SPEED_MM_S:g} mm/s")
        if not (0.0 <= hold <= AUTO_MAX_HOLD_S): raise ValueError(f"Hold must be 0~{AUTO_MAX_HOLD_S:g} s")
        if not (0.0 <= recovery <= AUTO_MAX_REPEAT_INTERVAL_S): raise ValueError(f"Recovery must be 0~{AUTO_MAX_REPEAT_INTERVAL_S:g} s")
        if not (1 <= reps <= AUTO_MAX_REPEATS): raise ValueError(f"Repeats must be 1~{AUTO_MAX_REPEATS}")
        if not (0.0 <= repeat_interval <= AUTO_MAX_REPEAT_INTERVAL_S): raise ValueError(f"Repeat interval must be 0~{AUTO_MAX_REPEAT_INTERVAL_S:g} s")
        for name,val in (("dx_mm",dx),("dy_mm",dy)):
            if abs(val)>AUTO_MAX_XY_OFFSET_MM: raise ValueError(f"{name} magnitude must be <= {AUTO_MAX_XY_OFFSET_MM:g} mm")

        # Resolve the deepest legal Z used by this action.
        max_depth=0.0
        if action in ("z_sweep","z_path"):
            ds=[float(v.strip()) for v in cfg.get("z_depths_mm","").replace(";",",").split(",") if v.strip()]
            if not ds or any((not math.isfinite(d)) or d<0 or d>AUTO_MAX_INDENT_MM for d in ds):
                raise ValueError(f"Z depth/path values must all be in 0~{AUTO_MAX_INDENT_MM:g} mm")
            if len(ds)>100: raise ValueError("Z depth/path list has >100 points")
            max_depth=max(ds)
        elif action!="none":
            if not (0.0 <= indent <= AUTO_MAX_INDENT_MM):
                raise ValueError(f"Indent depth must be 0~{AUTO_MAX_INDENT_MM:g} mm")
            max_depth=indent

        # Build the XY target envelope actually used by each action.
        action_points=1
        xs=[x0]; ys=[y0]
        if action=="slide_x": xs.append(x0+dx)
        elif action=="slide_y": ys.append(y0+dy)
        elif action in ("slide_xy","compound"):
            xs.append(x0+dx); ys.append(y0+dy)
        elif action=="x_sweep":
            vals=[float(v.strip()) for v in cfg.get("xy_sweep_values_mm","").replace(";",",").split(",") if v.strip()]
            if not vals: raise ValueError("XY sweep values list is empty")
            if len(vals)>200: raise ValueError("X sweep has >200 points")
            if any((not math.isfinite(v)) or abs(v)>AUTO_MAX_XY_OFFSET_MM for v in vals): raise ValueError(f"X sweep offsets must be within ±{AUTO_MAX_XY_OFFSET_MM:g} mm")
            xs += [x0+v for v in vals]
        elif action=="y_sweep":
            vals=[float(v.strip()) for v in cfg.get("xy_sweep_values_mm","").replace(";",",").split(",") if v.strip()]
            if not vals: raise ValueError("XY sweep values list is empty")
            if len(vals)>200: raise ValueError("Y sweep has >200 points")
            if any((not math.isfinite(v)) or abs(v)>AUTO_MAX_XY_OFFSET_MM for v in vals): raise ValueError(f"Y sweep offsets must be within ±{AUTO_MAX_XY_OFFSET_MM:g} mm")
            ys += [y0+v for v in vals]
        elif action=="cross_scan":
            xs += [x0+dx,x0-dx]; ys += [y0+dy,y0-dy]
        elif action=="angle_scan":
            n=int(cfg.get("angle_count",0)); radius=abs(dx)
            if not (1 <= n <= 360): raise ValueError("Angle count must be 1~360")
            if radius>AUTO_MAX_XY_OFFSET_MM: raise ValueError(f"Angle-scan radius must be <= {AUTO_MAX_XY_OFFSET_MM:g} mm")
            xs += [x0-radius,x0+radius]; ys += [y0-radius,y0+radius]
        elif action=="circle_scan":
            radius=finite("circle_radius_mm") ; n=int(cfg.get("circle_points",24))
            if not (0 < radius <= AUTO_MAX_XY_OFFSET_MM): raise ValueError(f"Circle radius must be >0 and <= {AUTO_MAX_XY_OFFSET_MM:g} mm")
            if not (4 <= n <= 720): raise ValueError("Circle points must be 4~720")
            xs += [x0-radius,x0+radius]; ys += [y0-radius,y0+radius]
        elif action=="grid_points":
            hx=finite("grid_half_x_mm"); hy=finite("grid_half_y_mm"); step=finite("grid_step_mm")
            if not (0 <= hx <= AUTO_MAX_XY_OFFSET_MM and 0 <= hy <= AUTO_MAX_XY_OFFSET_MM): raise ValueError(f"Grid half-spans must be 0~{AUTO_MAX_XY_OFFSET_MM:g} mm")
            if step<=0: raise ValueError("Grid step must be >0")
            gx=self._grid_offsets(hx,step)
            y_text=str(cfg.get("grid_y_offsets_mm","")).strip()
            gy=([float(v.strip()) for v in y_text.replace(";",",").split(",") if v.strip()]
                if y_text else self._grid_offsets(hy,step))
            if not gy: raise ValueError("Grid Y offset block is empty")
            if any((not math.isfinite(v)) or abs(v)>hy+1e-7 for v in gy):
                raise ValueError("Grid Y block offsets must stay inside the configured half-span")
            if len(set(round(v,7) for v in gy))!=len(gy):
                raise ValueError("Grid Y block contains duplicate offsets")
            action_points=len(gx)*len(gy)
            if action_points>AUTO_MAX_GRID_POINTS:
                raise ValueError(
                    f"Grid points {action_points} exceed safety cap {AUTO_MAX_GRID_POINTS}; "
                    "split the grid into multiple Trials")
            ref_interval=int(cfg.get("grid_reference_interval",0))
            if ref_interval<0: raise ValueError("Grid reference interval must be >=0")
            anchor_step=float(cfg.get("grid_anchor_step_mm",4.0))
            if anchor_step<=0: raise ValueError("Grid anchor step must be >0")
            xs += [x0+v for v in gx]; ys += [y0+v for v in gy]

        safe=contact+safe_lift; approach=contact+approach_gap; zmin=contact-max_depth
        for axis,val in (("X min",min(xs)),("X max",max(xs))):
            lo,hi=A5S_MACHINE_BOUNDS_MM["x"]
            if not lo<=val<=hi: raise ValueError(f"{axis}={val:.3f} mm outside A5S X bounds [{lo},{hi}]")
        for axis,val in (("Y min",min(ys)),("Y max",max(ys))):
            lo,hi=A5S_MACHINE_BOUNDS_MM["y"]
            if not lo<=val<=hi: raise ValueError(f"{axis}={val:.3f} mm outside A5S Y bounds [{lo},{hi}]")
        lo,hi=A5S_MACHINE_BOUNDS_MM["z"]
        for name,val in (("contact Z",contact),("deepest Z",zmin),("approach Z",approach),("safe Z",safe)):
            if not lo<=val<=hi: raise ValueError(f"{name}={val:.3f} mm outside A5S Z bounds [{lo},{hi}]")

        # Save an explicit preflight envelope for the run log / metadata.
        return {"x_min":min(xs),"x_max":max(xs),"y_min":min(ys),"y_max":max(ys),
                "z_min":zmin,"z_max":safe,"max_depth":max_depth,
                "points_per_repeat":action_points,
                "planned_contacts":action_points*reps}

    def auto_execute_trial(self):
        self._auto_execute_n_trials(1)

    def auto_execute_trial_batch(self):
        # Compatibility alias retained for old callbacks/scripts. V0.9 no longer creates
        # multiple independent Trial directories from the experiment page.
        self.auto_execute_trial()

    def _auto_execute_n_trials(self,n):
        if self.printer_thread and self.printer_thread.is_alive():
            messagebox.showwarning("Printer","自动动作正在运行"); return
        if not self.printer.connected:
            self.connect_printer()
            if not self.printer.connected:return
        if not self._require_home_for_auto(): return
        try:
            cfg=self._trial_action_config(); self._validate_action_cfg(cfg)
        except Exception as e:
            messagebox.showerror("实验参数 / Experiment Parameters",str(e)); return
        if cfg["action"]!="none" and not self._confirm_contact_motion(f"实验动作 {cfg['action']}（按已保存 Contact Z 参数执行）"): return
        self.save_settings(silent=True); self.batch_abort_event.clear(); self.motion_abort_event.clear(); self._update_auto_summary()
        inter=0.0
        def job():
            for batch_i in range(n):
                if self.batch_abort_event.is_set() or self.motion_abort_event.is_set(): break
                try:
                    # Resolve a truly free ID immediately before each independent Trial.
                    id_done=threading.Event()
                    self.after(0,lambda:(self.scan_next_trial_id(silent=True),id_done.set()))
                    if not id_done.wait(3.0): raise RuntimeError("UI timeout while resolving next Trial ID")
                    done=threading.Event(); box={}
                    def start_on_ui():
                        try:self.start_trial(); box["rec"]=self.recorder
                        finally:done.set()
                    self.after(0,start_on_ui)
                    if not done.wait(5.0): raise RuntimeError("UI timeout while starting Trial recorder")
                    rec=box.get("rec")
                    if rec is None: raise RuntimeError("Trial recorder did not start")
                    rec.write_event("TRIAL_RUN",json.dumps({"index":batch_i+1,"count":n},ensure_ascii=False),"auto")
                    rec.write_event("AUTO_MOTION_START",cfg["action"],"printer")
                    if bool(self.auto_fan_off_var.get()): self.printer.fan_off()
                    self._motion_event("EXPERIMENT_CONFIG",json.dumps(cfg,ensure_ascii=False))
                    self._execute_stage_experiment(cfg)
                    if self.motion_abort_event.is_set() or self.batch_abort_event.is_set():
                        self.after(0,lambda:self.stop_trial("aborted")); break
                    rec.write_event("AUTO_MOTION_END",cfg["action"],"printer")
                    time.sleep(0.25)
                    stop_done=threading.Event()
                    def stop_on_ui():
                        try:self.stop_trial("completed")
                        finally:stop_done.set()
                    self.after(0,stop_on_ui); stop_done.wait(5.0)
                    if batch_i < n-1 and inter>0:
                        deadline=time.monotonic()+inter
                        while time.monotonic()<deadline:
                            if self.batch_abort_event.is_set():break
                            time.sleep(min(0.05,max(0,deadline-time.monotonic())))
                except Exception as e:
                    self.log(f"AUTO/BATCH motion ERROR: {e}")
                    try:self._motion_event("AUTO_MOTION_ERROR",str(e))
                    except Exception:pass
                    self.after(0,lambda:self.stop_trial("aborted"))
                    break
            self.log(f"Auto Trial finished/stopped: requested={n}")
        self._run_motion_thread(job,"trial_batch")

    def _active_notebook_tab(self):
        try:
            return self.nametowidget(self.nb.select())
        except Exception:
            return None

    @staticmethod
    def _experiment_split_position(width,ratio=0.60,min_left=520,min_right=565):
        """Return a stable left-pane width while preserving usable controls."""
        width=max(1,int(width))
        target=int(round(width*float(ratio)))
        if width >= int(min_left)+int(min_right):
            target=max(int(min_left),min(target,width-int(min_right)))
        return max(1,min(target,width-1 if width>1 else 1))

    def _enforce_experiment_split_ratio(self,event=None):
        """Keep the experiment monitor/controls split at 60:40 after resizes."""
        split=getattr(self,"experiment_split",None)
        if split is None:
            return
        try:
            width=int(getattr(event,"width",0) or split.winfo_width())
            if width < 64:
                return
            desired=self._experiment_split_position(
                width,getattr(self,"experiment_left_ratio",0.60)
            )
            if abs(int(split.sashpos(0))-desired)>1:
                split.sashpos(0,desired)
        except (tk.TclError,AttributeError,TypeError,ValueError):
            pass

    @staticmethod
    def _set_camera_placeholder(surface,text):
        """Show status text without changing a fixed camera Canvas geometry."""
        if isinstance(surface,tk.Canvas):
            try:
                surface.delete("camera_frame")
                surface.delete("camera_placeholder")
                width=max(1,int(surface.winfo_width()),int(float(surface.cget("width"))))
                height=max(1,int(surface.winfo_height()),int(float(surface.cget("height"))))
                surface.create_text(
                    width//2,height//2,text=text,fill="#aeb7c2",
                    anchor="center",tags=("camera_placeholder",)
                )
            except (tk.TclError,ValueError,TypeError):
                pass
            return
        try:
            surface.configure(image="",text=text)
        except (tk.TclError,AttributeError):
            pass

    def _on_tab_changed(self, _event=None):
        """Update passive status text only. Rendering is routed in _drain_queues."""
        active=self._active_notebook_tab()
        if active is self.tab_data:
            try:self.refresh_data_index(silent=True)
            except Exception:pass
            if self.replay_samples:
                self.after_idle(lambda: self._replay_show_index(int(self.replay_slider_var.get())))
        if active is self.tab_exp:
            try:self.exp_predict_3d.redraw()
            except Exception:pass
            for name,lab in self.exp_preview_labels.items():
                enabled = bool(self.cam_vars.get(name,{}).get("enable").get()) if name in self.cam_vars else False
                if not enabled:
                    try:
                        self._set_camera_placeholder(lab,"未启用 / Disabled")
                        self.exp_preview_images.pop(name,None)
                    except Exception:
                        pass
        # Do not stop acquisition/recording when changing pages.
        # Only UI rendering target changes.

    def _render_camera_frame(self, name, frame, label, image_store, store_key):
        """Fit the COMPLETE frame inside the actual widget; never crop by forcing a minimum image size."""
        if cv2 is None or Image is None or ImageTk is None or label is None:
            return
        try:
            label.update_idletasks()
            lw=int(label.winfo_width()); lh=int(label.winfo_height())
            # During first geometry pass Tk may report 1x1.  Use the allocated parent
            # size if available, but never generate an image larger than visible area.
            if lw < 32 or lh < 32:
                par=label.master
                lw=max(lw,int(par.winfo_width()) if par is not None else 0)
                lh=max(lh,int(par.winfo_height()-24) if par is not None else 0)
            if lw < 32 or lh < 32:
                return
            h,w=frame.shape[:2]
            if w<=0 or h<=0: return
            margin=4
            avail_w=max(1,lw-margin*2); avail_h=max(1,lh-margin*2)
            scale=min(avail_w/w,avail_h/h)
            nw=max(1,int(round(w*scale))); nh=max(1,int(round(h*scale)))
            interp=cv2.INTER_AREA if scale<1.0 else cv2.INTER_LINEAR
            resized=cv2.resize(frame,(nw,nh),interpolation=interp) if (nw!=w or nh!=h) else frame
            rgb=cv2.cvtColor(resized,cv2.COLOR_BGR2RGB)
            photo=ImageTk.PhotoImage(Image.fromarray(rgb))
            image_store[store_key]=photo
            if isinstance(label,tk.Canvas):
                label.delete("camera_placeholder")
                items=label.find_withtag("camera_frame")
                if items:
                    label.itemconfigure(items[0],image=photo)
                    label.coords(items[0],lw//2,lh//2)
                else:
                    label.create_image(
                        lw//2,lh//2,image=photo,anchor="center",
                        tags=("camera_frame",)
                    )
            else:
                label.configure(image=photo,text="",anchor="center")
        except Exception as e:
            self.log(f"Camera preview render error ({name}): {e}")

    def choose_root(self):
        path=filedialog.askdirectory(initialdir=self.data_root_var.get() or str(HERE.parent))
        if path:
            self.data_root_var.set(path)
            if getattr(self,"auto_trial_id_var",None) is not None and self.auto_trial_id_var.get():
                self.scan_next_trial_id(silent=True)

    def increment_trial(self):
        # Always resolve against folders already on disk; avoids duplicate IDs after restart/crash.
        return self.scan_next_trial_id(silent=True)

    def _drain_queues(self):
        active=self._active_notebook_tab()
        show_exp = active is self.tab_exp
        show_mag = active is self.tab_acq
        show_cam = active is self.tab_cam
        show_predict = active is self.tab_predict

        # Always drain queues so acquisition threads never stall. Only the currently
        # visible page receives expensive canvas/image redraws.
        try:
            while True:
                data,meta=self.mag_queue.get_nowait()
                if data is None:
                    self.log("Magnetic fatal: "+str(meta.get("fatal"))); self.stop_magnetics(); continue
                self.last_mag=(data,meta)
                fingerprint_vector=self._magnetic_group_fingerprint_vector(data)
                if fingerprint_vector is not None:self.mag_fingerprint_history.append(fingerprint_vector)
                self._live_predict_submit(data, meta)
                base=meta.get("trigger_base_ns",0)
                if base:
                    if self.last_mag_group_ns:
                        self.actual_mag_periods.append((base-self.last_mag_group_ns)/1e9)
                        self.actual_mag_periods=self.actual_mag_periods[-100:]
                    self.last_mag_group_ns=base

                if show_exp:
                    self.exp_mag_plot.add(data)
                    self.exp_mag_summary_var.set(
                        f"Cycle {meta.get('cycle')} | 触发跨度 / trigger span={meta.get('trigger_span_ms',0):.3f} ms | "
                        f"有效 / valid={len(data)}/5"
                    )
                elif show_mag:
                    self.mag_plot.add(data)
                    for i in range(5):
                        d=data.get(i); v=self.sensor_vars[i]
                        if not d:
                            v["st"].set("MISS");continue
                        v["x"].set(f"{d['x_ut']:.3f}");v["y"].set(f"{d['y_ut']:.3f}")
                        v["z"].set(f"{d['z_ut']:.3f}");v["m"].set(f"{d['mag_ut']:.3f}")
                        v["st"].set(f"0x{int(d['status']):02X}")
                        v["dt"].set(f"{(d['trigger_ns']-base)/1e6:.3f} ms" if base else "--")
                    self.mag_summary_var.set(
                        f"Cycle {meta.get('cycle')} | trigger span={meta.get('trigger_span_ms',0):.3f} ms | valid={len(data)}/5"
                    )
                # On every other page: no plot/table redraw. Raw acquisition and Trial
                # recording already occurred in the worker thread.
        except queue.Empty:
            pass

        self._live_predict_drain_results(show_predict,show_exp)

        try:
            while True:
                name,frame,t_ns=self.preview_queue.get_nowait()
                if show_exp:
                    lab=self.exp_preview_labels.get(name)
                    self._render_camera_frame(name,frame,lab,self.exp_preview_images,name)
                elif show_cam:
                    lab=self.preview_labels.get(name)
                    self._render_camera_frame(name,frame,lab,self.preview_images,name)
                # On all other tabs we intentionally drop preview rendering while the
                # camera worker/recorder may continue to capture and timestamp frames.
        except queue.Empty:
            pass
        self.after(25,self._drain_queues)

    def _update_rates(self):
        mag_hz=0.0
        if self.actual_mag_periods:
            mean=sum(self.actual_mag_periods)/len(self.actual_mag_periods)
            mag_hz=1.0/mean if mean>0 else 0.0
        cam_parts=[]
        now=time.perf_counter_ns()
        for name,w in self.cameras.items():
            age=(now-w.last_frame_ns)/1e9 if w.last_frame_ns else 999
            cam_parts.append(f"{name}:{'OK' if age<1 else 'STALE'}")
        self.camera_summary_var.set(f"Mag actual≈{mag_hz:.1f} Hz | "+" ".join(cam_parts))
        self.after(1000,self._update_rates)

    def on_close(self):
        self.replay_backtest_panel.close()
        # R13: training/build commands are not launched by the UI. Keep legacy child cleanup as a safety fallback.
        try:
            proc=getattr(self,"_training_proc",None)
            if proc is not None and proc.poll() is None:
                pid=proc.pid
                if os.name=="nt":
                    flags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess,"CREATE_NO_WINDOW") else 0
                    subprocess.run(["taskkill","/PID",str(pid),"/T","/F"],capture_output=True,creationflags=flags,timeout=8)
                else:
                    try:os.killpg(os.getpgid(pid),15)
                    except Exception:proc.terminate()
        except Exception as e:
            try:self.log(f"Training child cleanup on close failed: {e}")
            except Exception:pass
        try:
            if self._printer_pos_poll_after_id is not None:
                self.after_cancel(self._printer_pos_poll_after_id)
                self._printer_pos_poll_after_id=None
        except Exception:
            pass
        try:self.save_settings(silent=True)
        except Exception:pass
        try:self.stop_trial("aborted")
        except Exception:pass
        try:self.stop_magnetics()
        except Exception:pass
        try:
            self.live_predict_stop.set()
            self.live_predict_enabled=False
        except Exception:
            pass
        try:self.stop_cameras()
        except Exception:pass
        try:self.replay_pause(); self._close_replay_caps()
        except Exception:pass
        try:self.printer.close()
        except Exception:pass
        try:self.bridge.close()
        except Exception:pass
        self.destroy()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--simulate",action="store_true",help="Use synthetic 5-sensor data; useful for UI/data-path testing without CH341")
    ap.add_argument("--headless-smoke-test",action="store_true",help="Check imports/config/simulated sampler without creating a window or accessing hardware")
    ap.add_argument("--smoke-test-ms",type=int,default=0,help="With --simulate, open the real UI and close it automatically after this many milliseconds")
    args=ap.parse_args()
    if args.headless_smoke_test:
        assert callable(scan_trials), "Project database scanner did not import"
        sampler=SimulatedFiveSensorSampler(DEFAULT_ADDRS)
        info=sampler.initialize(reset=False)
        data,meta=sampler.group_sample(include_temperature=True)
        assert len(info)==5 and len(data)==5 and "trigger_base_ns" in meta
        normalize_output_mapping(DEFAULT_RUNTIME_SETTINGS["live_prediction"]["output_mapping_json"])
        build_ubuntu_training_script(DEFAULT_RUNTIME_SETTINGS["ubuntu_training"])
        print("HEADLESS_SMOKE_OK sensors=5 config_version="+str(DEFAULT_RUNTIME_SETTINGS["config_version"]))
        return
    app=MagTouchApp(simulate=args.simulate)
    if args.smoke_test_ms>0:
        if not args.simulate:
            raise SystemExit("--smoke-test-ms requires --simulate")
        app.after(max(100,int(args.smoke_test_ms)),app.on_close)
    app.mainloop()


if __name__=="__main__":
    main()
