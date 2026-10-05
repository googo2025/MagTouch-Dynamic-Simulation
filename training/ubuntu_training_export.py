#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure helpers for the full Windows-parity Ubuntu training-command UI."""
from __future__ import annotations

import json
import re
import shlex
from typing import Any, Mapping, Sequence


RECOMMENDED_HARDWARE = "4×NVIDIA RTX 4090 | 124 GB RAM | 80-core CPU"

DEFAULT_UBUNTU_TRAINING = {
    "project_root": "/home/magtouch/MagTouch_Hardware_Experiment_V0_1_COMPLETE",
    "plan_path": "/home/magtouch/data/Plan_MagTouch",
    "export_output_path": "/home/magtouch/data/exports",
    "training_export_path": "/home/magtouch/data/TRAINING_EXPORT",
    "h1_features_path": "/home/magtouch/data/TRAINING_EXPORT_H1_FEATURES",
    "h2_relative_path": "/home/magtouch/data/TRAINING_EXPORT_RELATIVE_TACTILE_R15",
    "h2_spatial_labels_path": "/home/magtouch/data/TRAINING_EXPORT_SPATIAL_LABELS_V16",
    "h2_unified_labels_path": "/home/magtouch/data/TRAINING_EXPORT_UNIFIED_LABELS_R18_1_ABSXYZ",
    "conda_env": "magtouch",
    "gpus": "0,1,2,3",
    "cpu_count": 80,
    "gpu_schedule": "serial",
    "seeds": "20260910,20260911,20260912,20260913",
    "h1_feature": "F5",
    "h1_policy": "mask15",
    "baseline_model": "transformer",
    "baseline_task": "mode5",
    "epochs": 30,
    "baseline_batch": 48,
    "hw5_batch": 24,
    "d_model": 128,
    "layers": 3,
    "h1_objective": "multitask",
    "deterministic": True,
    "h2_feature": "R1",
    "h2_history": 64,
    "h2_endpoint_min_history": 128,
    "h2_use_q": True,
    "h2_spatial_objective": "joint",
    "h2_sample_policy": "common",
    "h2_unified_objective": "unified",
    "h2_unified_sample_policy": "common",
    "extra_args": "",
}

UBUNTU_COMMAND_KINDS = (
    "analysis_export",
    "training_export",
    "build_h1_features",
    "h1_baseline",
    "h1_hw5",
    "h1_core_matrix",
    "build_h2_relative",
    "h2_train",
    "h2_feature_matrix",
    "h2_history_matrix",
    "build_h2_spatial_labels",
    "h2_spatial_train",
    "h2_spatial_objective_matrix",
    "h2_spatial_history_matrix",
    "build_h2_unified_labels",
    "validate_h2_unified_labels",
    "h2_unified_train",
)


def _positive_int(value: Any, name: str, allow_zero: bool = False) -> int:
    number = int(value)
    if number < (0 if allow_zero else 1):
        raise ValueError(f"{name} must be {'non-negative' if allow_zero else 'positive'}")
    return number


def _bool_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


def normalize_ubuntu_training_config(config: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(DEFAULT_UBUNTU_TRAINING)
    out.update(dict(config))
    for name in (
        "project_root", "plan_path", "export_output_path", "training_export_path",
        "h1_features_path", "h2_relative_path", "h2_spatial_labels_path", "h2_unified_labels_path",
    ):
        text = str(out[name]).strip()
        if not text.startswith("/") or "\\" in text:
            raise ValueError(f"{name} must be an absolute Ubuntu path such as /home/...")
        out[name] = text.rstrip("/") or "/"

    env = str(out["conda_env"]).strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", env):
        raise ValueError("conda_env may contain only letters, digits, '.', '_' and '-'")
    out["conda_env"] = env

    gpus = [item.strip() for item in str(out["gpus"]).split(",") if item.strip()]
    if not gpus or not all(item.isdigit() for item in gpus) or len(set(gpus)) != len(gpus):
        raise ValueError("gpus must be a comma-separated list of unique non-negative GPU IDs")
    out["gpus"] = ",".join(gpus)
    out["gpu_schedule"] = str(out["gpu_schedule"]).strip().lower()
    if out["gpu_schedule"] not in {"serial", "full_parallel"}:
        raise ValueError("gpu_schedule must be serial or full_parallel")
    seeds = [item.strip() for item in str(out["seeds"]).split(",") if item.strip()]
    if not seeds or not all(re.fullmatch(r"-?\d+", item) for item in seeds):
        raise ValueError("seeds must be a comma-separated integer list")
    if len(seeds) < len(gpus):
        base = int(seeds[-1])
        seeds.extend(str(base + index + 1) for index in range(len(gpus) - len(seeds)))
    out["seeds"] = ",".join(seeds)

    out["cpu_count"] = _positive_int(out["cpu_count"], "cpu_count")
    if out["cpu_count"] > 4096:
        raise ValueError("cpu_count must not exceed 4096")
    for name in ("epochs", "baseline_batch", "hw5_batch", "d_model", "layers"):
        out[name] = _positive_int(out[name], name)
    out["h2_history"] = _positive_int(out["h2_history"], "h2_history", allow_zero=True)
    out["h2_endpoint_min_history"] = _positive_int(
        out["h2_endpoint_min_history"], "h2_endpoint_min_history", allow_zero=True
    )
    for name in ("h2_history", "h2_endpoint_min_history"):
        if out[name] not in {0, 16, 32, 64, 128}:
            raise ValueError(f"{name} must be one of 0, 16, 32, 64, 128")

    out["h1_feature"] = str(out["h1_feature"]).strip()
    out["h1_policy"] = str(out["h1_policy"]).strip()
    out["baseline_model"] = str(out["baseline_model"]).strip()
    out["baseline_task"] = str(out["baseline_task"]).strip()
    out["h1_objective"] = str(out["h1_objective"]).strip()
    out["h2_feature"] = str(out["h2_feature"]).strip()
    out["h2_spatial_objective"] = str(out["h2_spatial_objective"]).strip()
    out["h2_sample_policy"] = str(out["h2_sample_policy"]).strip()
    out["h2_unified_objective"] = str(out["h2_unified_objective"]).strip()
    out["h2_unified_sample_policy"] = str(out["h2_unified_sample_policy"]).strip()
    if out["h1_feature"] not in {f"F{i}" for i in range(6)}:
        raise ValueError("h1_feature must be F0..F5")
    if out["h1_policy"] not in {"raw15", "clean14", "mask15"}:
        raise ValueError("h1_policy must be raw15, clean14, or mask15")
    if out["baseline_model"] not in {"mlp", "tcn", "transformer"}:
        raise ValueError("baseline_model must be mlp, tcn, or transformer")
    if out["baseline_task"] not in {"mode5", "z", "dx", "dy", "x", "y"}:
        raise ValueError("baseline_task is invalid")
    if out["h1_objective"] not in {"mode_only", "multitask"}:
        raise ValueError("h1_objective must be mode_only or multitask")
    if out["h2_feature"] not in {"R0", "R1", "R2", "R3"}:
        raise ValueError("h2_feature must be R0..R3")
    if out["h2_spatial_objective"] not in {"fixed_only", "relative_only", "joint"}:
        raise ValueError("h2_spatial_objective must be fixed_only, relative_only, or joint")
    if out["h2_sample_policy"] not in {"common", "native"}:
        raise ValueError("h2_sample_policy must be common or native")
    if out["h2_unified_objective"] not in {"coordinate_only", "unified"}:
        raise ValueError("h2_unified_objective must be coordinate_only or unified")
    if out["h2_unified_sample_policy"] not in {"common", "native"}:
        raise ValueError("h2_unified_sample_policy must be common or native")
    out["deterministic"] = _bool_value(out["deterministic"])
    out["h2_use_q"] = _bool_value(out["h2_use_q"])
    out["extra_args"] = str(out["extra_args"]).strip()
    return out


def default_ubuntu_training_json() -> str:
    return json.dumps(DEFAULT_UBUNTU_TRAINING, ensure_ascii=False, separators=(",", ":"))


def _command(parts: Sequence[Any], extra_args: str = "") -> str:
    def quote(item: Any) -> str:
        text = str(item)
        if re.fullmatch(r"\$[A-Z][A-Z0-9_]*", text):
            return f'"{text}"'
        return shlex.quote(text)
    command = " ".join(quote(item) for item in parts)
    return command + ((" " + extra_args) if extra_args else "")


def _common_header(cfg: Mapping[str, Any], title: str, active_gpu_jobs: int) -> list[str]:
    active_gpu_jobs = max(1, int(active_gpu_jobs))
    return [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"# {title}",
        f"# Recommended host: {RECOMMENDED_HARDWARE}",
        "# Generated only; this UI never starts Ubuntu training.",
        f"PROJECT_ROOT={shlex.quote(str(cfg['project_root']))}",
        f"PLAN_ROOT={shlex.quote(str(cfg['plan_path']))}",
        f"EXPORT_OUTPUT={shlex.quote(str(cfg['export_output_path']))}",
        f"TRAINING_EXPORT={shlex.quote(str(cfg['training_export_path']))}",
        f"H1_FEATURES={shlex.quote(str(cfg['h1_features_path']))}",
        f"H2_RELATIVE={shlex.quote(str(cfg['h2_relative_path']))}",
        f"H2_SPATIAL_LABELS={shlex.quote(str(cfg['h2_spatial_labels_path']))}",
        f"H2_UNIFIED_LABELS={shlex.quote(str(cfg['h2_unified_labels_path']))}",
        f"GPUS=({' '.join(shlex.quote(item) for item in str(cfg['gpus']).split(','))})",
        f"SEEDS=({' '.join(shlex.quote(item) for item in str(cfg['seeds']).split(','))})",
        f"CPU_COUNT={int(cfg['cpu_count'])}",
        f"GPU_JOB_COUNT={active_gpu_jobs}",
        "CPU_THREADS_PER_JOB=$((CPU_COUNT / GPU_JOB_COUNT))",
        "if (( CPU_THREADS_PER_JOB < 1 )); then CPU_THREADS_PER_JOB=1; fi",
        'export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 CUBLAS_WORKSPACE_CONFIG=:4096:8',
        "_magtouch_on_exit() {",
        "  local rc=$?",
        "  trap - EXIT",
        "  echo",
        '  if (( rc == 0 )); then echo "[DONE] MagTouch Ubuntu command completed."; else echo "[FAILED] returncode=$rc"; fi',
        '  echo "Logs: $PROJECT_ROOT/training_logs"',
        '  if [[ -t 0 && "${MAGTOUCH_NO_PAUSE:-0}" != "1" ]]; then read -r -p "Press Enter to close..." _ || true; fi',
        '  exit "$rc"',
        "}",
        "trap _magtouch_on_exit EXIT",
        "# Some conda activate.d hooks read optional variables such as LD_LIBRARY_PATH.",
        "# Temporarily disable nounset only for conda initialization, then restore it.",
        "set +u",
        'source "$(conda info --base)/etc/profile.d/conda.sh"',
        f"conda activate {shlex.quote(str(cfg['conda_env']))}",
        "set -u",
        'cd "$PROJECT_ROOT"',
        'mkdir -p "$PROJECT_ROOT/training_logs" "$EXPORT_OUTPUT"',
        'echo "GPU IDs=${GPUS[*]} | CPU total=$CPU_COUNT | active GPU jobs=$GPU_JOB_COUNT | CPU threads/job=$CPU_THREADS_PER_JOB"',
        "python --version",
        'python -c "import torch; print(\'PyTorch\', torch.__version__); print(\'CUDA\', torch.version.cuda); print(\'GPUs\', torch.cuda.device_count())"',
        "",
    ]


def _cpu_only_line(cfg: Mapping[str, Any], parts: Sequence[Any]) -> str:
    prefix = (
        f"OMP_NUM_THREADS={int(cfg['cpu_count'])} MKL_NUM_THREADS={int(cfg['cpu_count'])} "
        f"OPENBLAS_NUM_THREADS={int(cfg['cpu_count'])} NUMEXPR_NUM_THREADS={int(cfg['cpu_count'])}"
    )
    return prefix + " " + _command(parts)


def _training_job_command(cfg: Mapping[str, Any], gpu: str, parts: Sequence[Any]) -> str:
    prefix = (
        f"CUDA_VISIBLE_DEVICES={shlex.quote(gpu)} "
        'OMP_NUM_THREADS="$CPU_THREADS_PER_JOB" MKL_NUM_THREADS="$CPU_THREADS_PER_JOB" '
        'OPENBLAS_NUM_THREADS="$CPU_THREADS_PER_JOB" NUMEXPR_NUM_THREADS="$CPU_THREADS_PER_JOB"'
    )
    return prefix + " " + _command(parts, str(cfg["extra_args"]))


def _live_logged_command(command: str, log: str, label: str) -> str:
    """Stream prefixed worker output to the terminal while retaining a raw log."""
    # `set -o pipefail` in the common header makes the whole pipeline fail if
    # Python fails, even when tee and sed exit successfully. Python is invoked
    # with -u and GNU sed uses -u, so epoch/resource lines appear immediately.
    sed_expression = shlex.quote(f"s/^/[{label}] /")
    return f'{command} 2>&1 | tee "{log}" | sed -u {sed_expression}'


def _parallel_seed_jobs(
    cfg: Mapping[str, Any], title: str, command_factory,
) -> list[str]:
    gpus = str(cfg["gpus"]).split(",")
    seeds = str(cfg["seeds"]).split(",")
    parallel = cfg["gpu_schedule"] == "full_parallel"
    lines = _common_header(cfg, title, len(gpus) if parallel else 1)
    lines.append(
        "# Full-resource mode: one independent seed job per selected GPU."
        if parallel else
        "# Strict serial mode: never run two training processes at once."
    )
    if parallel:
        lines.append("PIDS=()")
    for index, gpu in enumerate(gpus):
        seed = seeds[index]
        parts, slug = command_factory(seed)
        log = f'$PROJECT_ROOT/training_logs/{slug}_seed{seed}.log'
        command = _live_logged_command(
            _training_job_command(cfg, gpu, parts), log, f"{slug}_seed{seed}"
        )
        lines.extend([
            f'echo "[START] GPU={gpu} seed={seed} log={log}"',
            command + (" &" if parallel else ""),
        ])
        if parallel:
            lines.append('PIDS+=("$!")')
        else:
            lines.append(f'echo "[DONE] GPU={gpu} seed={seed}"')
    if parallel:
        lines.extend([
            "status=0",
            'for pid in "${PIDS[@]}"; do wait "$pid" || status=1; done',
            'if (( status != 0 )); then echo "[FAILED] one or more GPU jobs failed"; exit "$status"; fi',
            'echo "[DONE] all parallel seed jobs completed"',
        ])
    else:
        lines.append('echo "[DONE] all serial seed jobs completed"')
    lines.append("")
    return lines


def _matrix_script(
    cfg: Mapping[str, Any], title: str, jobs: Sequence[tuple[str, Sequence[Any]]],
) -> list[str]:
    gpus = str(cfg["gpus"]).split(",")
    seeds = str(cfg["seeds"]).split(",")
    parallel = cfg["gpu_schedule"] == "full_parallel"
    worker_count = min(len(gpus), len(jobs)) if parallel else 1
    lines = _common_header(cfg, title, worker_count)
    lines.append(
        "# Full-resource mode: matrix variants run concurrently across selected GPUs; each GPU receives a sequential queue."
        if parallel else
        "# Strict serial mode: one matrix job at a time; selected GPUs are used round-robin."
    )
    if parallel:
        lines.append("PIDS=()")
        for worker in range(worker_count):
            gpu = gpus[worker]
            lines.append(f"worker_{worker}() {{")
            for job_index in range(worker, len(jobs), worker_count):
                label, base_parts = jobs[job_index]
                seed = seeds[0]
                parts = list(base_parts) + ["--seed", seed]
                log_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_").lower()
                log = f'$PROJECT_ROOT/training_logs/{log_slug}_seed{seed}.log'
                lines.extend([
                    f'  echo "[START] {label} | GPU={gpu} seed={seed} | log={log}"',
                    "  " + _live_logged_command(_training_job_command(cfg, gpu, parts), log, label),
                    f'  echo "[DONE] {label} | GPU={gpu} seed={seed}"',
                ])
            lines.extend(["}", f"worker_{worker} &", 'PIDS+=("$!")'])
        lines.extend([
            "status=0",
            'for pid in "${PIDS[@]}"; do wait "$pid" || status=1; done',
            'if (( status != 0 )); then echo "[FAILED] one or more matrix workers failed"; exit "$status"; fi',
            'echo "[DONE] full-resource matrix completed"',
            "",
        ])
        return lines
    for job_index, (label, base_parts) in enumerate(jobs):
        gpu = gpus[job_index % len(gpus)]
        # Ablation variables must not be confounded with seed.  A matrix uses
        # one common seed; repeated-seed runs are generated by the single-run
        # command and are also executed serially.
        seed = seeds[0]
        parts = list(base_parts) + ["--seed", seed]
        log_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_").lower()
        log = f'$PROJECT_ROOT/training_logs/{log_slug}_seed{seed}.log'
        lines.extend([
            f'echo "[START] {label} | GPU={gpu} seed={seed} | log={log}"',
            _live_logged_command(_training_job_command(cfg, gpu, parts), log, label),
            f'echo "[DONE] {label} | GPU={gpu} seed={seed}"',
        ])
    lines.extend(['echo "[DONE] serial matrix completed"', ""])
    return lines


def build_ubuntu_training_script(config: Mapping[str, Any], command_kind: str = "h2_train") -> str:
    """Generate one Windows-page-equivalent Ubuntu Bash workflow."""
    cfg = normalize_ubuntu_training_config(config)
    kind = str(command_kind).strip()
    if kind not in UBUNTU_COMMAND_KINDS:
        raise ValueError("Unsupported Ubuntu command kind: " + kind)
    det = ["--deterministic"] if cfg["deterministic"] else ["--no-deterministic"]

    if kind == "analysis_export":
        lines = _common_header(cfg, "MagTouch lightweight QC export", 1)
        lines += [_cpu_only_line(cfg, ["python", "-u", "tools/export_plan_for_analysis.py", "$PLAN_ROOT", "$EXPORT_OUTPUT"]), ""]
    elif kind == "training_export":
        lines = _common_header(cfg, "MagTouch full TRAINING_EXPORT", 1)
        lines += [_cpu_only_line(cfg, ["python", "-u", "tools/export_plan_for_training.py", "--plan", "$PLAN_ROOT", "--out", "$EXPORT_OUTPUT"]), ""]
    elif kind == "build_h1_features":
        lines = _common_header(cfg, "Build 40 Hz H1 F0-F5 features", 1)
        lines += [_cpu_only_line(cfg, ["python", "-u", "training/build_h1_dataset.py", "$TRAINING_EXPORT", "--out", "$H1_FEATURES", "--hz", "40", "--seed", str(cfg["seeds"]).split(",")[0]]), ""]
    elif kind == "build_h2_relative":
        lines = _common_header(cfg, "Build H2-R15 history-ready relative dataset", 1)
        lines += [_cpu_only_line(cfg, ["python", "-u", "training/build_relative_tactile_dataset.py", "$TRAINING_EXPORT", "--out", "$H2_RELATIVE", "--hz", "40", "--seed", str(cfg["seeds"]).split(",")[0], "--episode-pre-seconds", "4.0"]), ""]
    elif kind == "build_h2_spatial_labels":
        lines = _common_header(cfg, "Build V16 fixed-coordinate labels", 1)
        lines += [_cpu_only_line(cfg, ["python", "-u", "training/build_spatial_labels_v16.py", "$H2_RELATIVE", "--source", "$TRAINING_EXPORT", "--out", "$H2_SPATIAL_LABELS"]), ""]
    elif kind == "build_h2_unified_labels":
        lines = _common_header(cfg, "Build R18 unified XYZ+dXYZ+mode5 labels", 1)
        lines += [_cpu_only_line(cfg, ["python", "-u", "training/build_unified_labels_r18.py", "$H2_RELATIVE", "--source", "$TRAINING_EXPORT", "--out", "$H2_UNIFIED_LABELS"]), ""]
    elif kind == "validate_h2_unified_labels":
        lines = _common_header(cfg, "Validate R18 unified labels", 1)
        lines += [_cpu_only_line(cfg, ["python", "-u", "training/validate_unified_labels_r18.py", "$H2_RELATIVE", "--labels", "$H2_UNIFIED_LABELS", "--json-out", "$H2_UNIFIED_LABELS/validation_r18.json"]), ""]
    elif kind == "h1_baseline":
        def baseline(seed: str):
            return (["python", "-u", "training/train_h1_baselines.py", "$H1_FEATURES", "--feature", cfg["h1_feature"], "--policy", cfg["h1_policy"], "--model", cfg["baseline_model"], "--task", cfg["baseline_task"], "--epochs", cfg["epochs"], "--batch", cfg["baseline_batch"], "--gpu", "0", "--seed", seed] + det, "h1_baseline")
        lines = _parallel_seed_jobs(cfg, "H1 baseline training", baseline)
    elif kind == "h1_hw5":
        def hw5(seed: str):
            return (["python", "-u", "training/train_magtouch_ra_hw5.py", "$H1_FEATURES", "--feature", cfg["h1_feature"], "--policy", cfg["h1_policy"], "--objective", cfg["h1_objective"], "--epochs", cfg["epochs"], "--batch", cfg["hw5_batch"], "--gpu", "0", "--d_model", cfg["d_model"], "--layers", cfg["layers"], "--seed", seed] + det, "h1_hw5")
        lines = _parallel_seed_jobs(cfg, "MagTouch-RA-HW5 training", hw5)
    elif kind == "h2_train":
        def h2(seed: str):
            qflag = "--use-q" if cfg["h2_use_q"] else "--no-use-q"
            return (["python", "-u", "training/train_relative_tactile_hw5.py", "$H2_RELATIVE", "--feature", cfg["h2_feature"], "--history", cfg["h2_history"], "--endpoint-min-history", cfg["h2_endpoint_min_history"], qflag, "--epochs", cfg["epochs"], "--batch", cfg["hw5_batch"], "--gpu", "0", "--d_model", cfg["d_model"], "--layers", cfg["layers"], "--seed", seed] + det, "h2_relative")
        lines = _parallel_seed_jobs(cfg, "H2-R15 history-aware relative training", h2)
    elif kind == "h2_spatial_train":
        def spatial(seed: str):
            qflag = "--use-q" if cfg["h2_use_q"] else "--no-use-q"
            return (["python", "-u", "training/train_spatial_relative_hw5.py", "$H2_RELATIVE", "--spatial-labels", "$H2_SPATIAL_LABELS", "--objective", cfg["h2_spatial_objective"], "--sample-policy", cfg["h2_sample_policy"], "--feature", cfg["h2_feature"], "--history", cfg["h2_history"], "--endpoint-min-history", cfg["h2_endpoint_min_history"], qflag, "--epochs", cfg["epochs"], "--batch", cfg["hw5_batch"], "--gpu", "0", "--d_model", cfg["d_model"], "--layers", cfg["layers"], "--seed", seed] + det, "h2_spatial_v16")
        lines = _parallel_seed_jobs(cfg, "H2 V16 dual-coordinate training", spatial)
    elif kind == "h2_unified_train":
        def unified(seed: str):
            qflag = "--use-q" if cfg["h2_use_q"] else "--no-use-q"
            return (["python", "-u", "training/train_unified_magtouch_r18_hw5.py", "$H2_RELATIVE", "--labels", "$H2_UNIFIED_LABELS", "--objective", cfg["h2_unified_objective"], "--sample-policy", cfg["h2_unified_sample_policy"], "--feature", cfg["h2_feature"], "--history", cfg["h2_history"], "--endpoint-min-history", cfg["h2_endpoint_min_history"], qflag, "--epochs", cfg["epochs"], "--batch", cfg["hw5_batch"], "--gpu", "0", "--d_model", cfg["d_model"], "--layers", cfg["layers"], "--seed", seed] + det, "h2_unified_r18")
        lines = _parallel_seed_jobs(cfg, "H2 R18 unified XYZ+dXYZ+mode5 training", unified)
    else:
        common_h1 = ["--epochs", cfg["epochs"], "--gpu", "0"] + det
        common_hw5 = ["--batch", cfg["hw5_batch"], "--d_model", cfg["d_model"], "--layers", cfg["layers"]]
        if kind == "h1_core_matrix":
            jobs = [
                ("1_4_H1_baseline_F4_raw15_mode5", ["python", "-u", "training/train_h1_baselines.py", "$H1_FEATURES", "--feature", "F4", "--policy", "raw15", "--model", "transformer", "--task", "mode5", "--batch", cfg["baseline_batch"]] + common_h1),
                ("2_4_HW5_F4_raw15_mode_only", ["python", "-u", "training/train_magtouch_ra_hw5.py", "$H1_FEATURES", "--feature", "F4", "--policy", "raw15", "--objective", "mode_only"] + common_hw5 + common_h1),
                ("3_4_HW5_F4_raw15_multitask", ["python", "-u", "training/train_magtouch_ra_hw5.py", "$H1_FEATURES", "--feature", "F4", "--policy", "raw15", "--objective", "multitask"] + common_hw5 + common_h1),
                ("4_4_HW5_F5_mask15_multitask", ["python", "-u", "training/train_magtouch_ra_hw5.py", "$H1_FEATURES", "--feature", "F5", "--policy", "mask15", "--objective", "multitask"] + common_hw5 + common_h1),
            ]
        else:
            base_h2 = ["python", "-u", "training/train_relative_tactile_hw5.py", "$H2_RELATIVE"]
            common_h2 = ["--epochs", cfg["epochs"], "--batch", cfg["hw5_batch"], "--gpu", "0", "--d_model", cfg["d_model"], "--layers", cfg["layers"]] + det
            if kind == "h2_feature_matrix":
                jobs = [
                    ("1_4_R0_no_Q", base_h2 + ["--feature", "R0", "--history", cfg["h2_history"], "--no-use-q"] + common_h2),
                    ("2_4_R1_axis_Q", base_h2 + ["--feature", "R1", "--history", cfg["h2_history"], "--use-q"] + common_h2),
                    ("3_4_R2_axis_Q", base_h2 + ["--feature", "R2", "--history", cfg["h2_history"], "--use-q"] + common_h2),
                    ("4_4_R3_axis_Q", base_h2 + ["--feature", "R3", "--history", cfg["h2_history"], "--use-q"] + common_h2),
                ]
            elif kind == "h2_history_matrix":
                jobs = [
                    (f"{index + 1}_5_R1_Q_history_{history}", base_h2 + ["--feature", "R1", "--history", history, "--endpoint-min-history", "128", "--use-q"] + common_h2)
                    for index, history in enumerate((0, 16, 32, 64, 128))
                ]
            else:
                base_spatial = ["python", "-u", "training/train_spatial_relative_hw5.py", "$H2_RELATIVE", "--spatial-labels", "$H2_SPATIAL_LABELS"]
                common_spatial = ["--sample-policy", "common", "--feature", cfg["h2_feature"], "--batch", cfg["hw5_batch"], "--gpu", "0", "--d_model", cfg["d_model"], "--layers", cfg["layers"], "--epochs", cfg["epochs"], "--use-q"] + det
                if kind == "h2_spatial_objective_matrix":
                    jobs = [
                        ("1_3_V16_fixed_only", base_spatial + ["--objective", "fixed_only", "--history", cfg["h2_history"], "--endpoint-min-history", cfg["h2_endpoint_min_history"]] + common_spatial),
                        ("2_3_V16_relative_only", base_spatial + ["--objective", "relative_only", "--history", cfg["h2_history"], "--endpoint-min-history", cfg["h2_endpoint_min_history"]] + common_spatial),
                        ("3_3_V16_joint", base_spatial + ["--objective", "joint", "--history", cfg["h2_history"], "--endpoint-min-history", cfg["h2_endpoint_min_history"]] + common_spatial),
                    ]
                else:
                    jobs = [
                        (f"{index + 1}_5_V16_joint_history_{history}", base_spatial + ["--objective", "joint", "--history", history, "--endpoint-min-history", "128"] + common_spatial)
                        for index, history in enumerate((0, 16, 32, 64, 128))
                    ]
        lines = _matrix_script(cfg, kind, jobs)
    return "\n".join(lines)
