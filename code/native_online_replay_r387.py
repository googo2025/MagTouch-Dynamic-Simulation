"""Replay the ORIGINAL live predictor and its ordered protocol, without devices.

There is deliberately no 10-Hz decimation, new interpolation, invented B0
window, time alignment or output registration in this module. Neural models
retain the interpolation implemented by their own native online adapter.

New native_stream.jsonl journals contain the actual lock-serialized calls,
including unloaded samples before Trial creation. Legacy CSV archives cannot
prove that ordering or reconstruct an omitted initial baseline; they are
qualified explicitly, rather than silently declared equivalent to online.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

JOURNAL_SCHEMA = "MagTouch-NATIVE-INFERENCE-JOURNAL-R387-1"


class ReplayProvenanceError(ValueError):
    """A missing protocol is not repaired by guessing a baseline or history."""


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def implementation_sources(predictor):
    """Hash loaded native classes/components, not just their numerical weights."""
    found, seen = {}, set()
    def visit(value, depth=0):
        if depth > 8 or id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, dict):
            for item in value.values():
                if hasattr(item, "__dict__") or isinstance(item, dict): visit(item, depth+1)
            return
        if not hasattr(value, "__dict__"):
            return
        local_class = False
        for cls in type(value).__mro__:
            module = sys.modules.get(cls.__module__)
            filename = getattr(module, "__file__", None)
            if filename:
                path = Path(filename).resolve()
                if path.suffix == ".py" and path.is_relative_to(ROOT):
                    found[str(path.relative_to(ROOT)).replace("\\", "/")] = sha256(path)
                    local_class = True
        if local_class:
            for item in vars(value).values():
                if hasattr(item, "__dict__") or isinstance(item, dict): visit(item, depth+1)
    visit(predictor)
    for filename in (Path(__file__), ROOT/"code/general_model_selection_r387.py",
                     ROOT/"code/contact_display.py"):
        found[str(filename.resolve().relative_to(ROOT)).replace("\\", "/")] = sha256(filename)
    return found


def _applies(record, target):
    requested = record.get("target", "both")
    return requested in (None, "both", "all") or requested == target or (
        isinstance(requested, list) and target in requested)


def _csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        yield from csv.DictReader(source)


class NativeReplaySession:
    """One independent native model; consume calls in recorded execution order.

    ``consume(record)`` returns complete native predictions as dictionaries.
    No wall-clock sleep is required: native state is driven by recorded t_ns.
    A's optional display annotations duplicate the original live loop only;
    they do not change any coordinate, raw field or contact permission.
    """
    def __init__(self, predictor, target="A", annotate_display=None, allow_unsupported_controls=False, honor_output_record_markers=True):
        self.predictor = predictor
        self.target = target
        self.annotate_display = target == "A" if annotate_display is None else bool(annotate_display)
        self.trial = None
        self.group_count = 0
        self.output_count = 0
        self.controls = []
        self.unsupported_controls = []
        self.allow_unsupported_controls = bool(allow_unsupported_controls)
        self.gaps = []
        self.previous_input_ns = None
        self.last_seq = None
        self.baseline_called = False
        self.display_segment = 0
        self.outputs_by_time = {}
        self.output_record_markers = 0
        self.honor_output_record_markers = bool(honor_output_record_markers)
        self._reset_display()

    def _reset_display(self):
        self.display_previous = None
        self.display_onset_ns = None
        self.display_last_end_ns = None

    def _annotate(self, prediction):
        from contact_display import prediction_contact
        visible, reason = prediction_contact(prediction)
        previous = self.display_previous
        discontinuous = previous is None or prediction.t_ns - previous[0] > 1_000_000_000 or prediction.t_ns <= previous[0]
        was_visible = False if previous is None else previous[1]
        if visible and (discontinuous or not was_visible):
            self.display_segment += 1
            self.display_onset_ns = int(prediction.t_ns)
        elif not visible:
            if was_visible:
                self.display_last_end_ns = int(prediction.t_ns)
            self.display_onset_ns = None
        age = 0. if not visible or self.display_onset_ns is None else max(0., (prediction.t_ns-self.display_onset_ns)/1e6)
        self.display_previous = (int(prediction.t_ns), visible)
        return replace(prediction, extra_outputs=dict(prediction.extra_outputs, **{
            "display.contact": visible, "display.contact_reason": reason,
            "display.contact_segment": self.display_segment,
            "display.contact_age_ms": age,
            "display.contact_last_end_ns": self.display_last_end_ns}))

    def consume(self, record):
        """Supported ops match the R387 live journal; unknown ops fail visibly."""
        if "seq" in record:
            seq = int(record["seq"])
            if self.last_seq is not None and seq <= self.last_seq:
                raise ReplayProvenanceError("Journal execution sequence is not strictly increasing")
            if self.last_seq is not None and seq != self.last_seq + 1:
                self.gaps.append({"kind": "missing_journal_operations", "after_seq": self.last_seq, "next_seq": seq})
            self.last_seq = seq
        op = record.get("op", record.get("kind"))
        if op in ("header", "end", "footer", "snapshot", "marker"):
            return []
        if op == "trial_start":
            self.trial = record.get("path", record.get("trial_dir", record.get("trial_id")))
            return []
        if op == "trial_end":
            self.trial = None
            return []
        if op == "output_recorded":
            if self.honor_output_record_markers and _applies(record, self.target):
                candidates = self.outputs_by_time.get(int(record["t_ns"]), [])
                unmarked = [row for row in candidates if not row.get("online_recorded", False)]
                if not unmarked:
                    raise ReplayProvenanceError("Output-recorded marker has no corresponding native emitted prediction")
                row = unmarked[-1]
                row["trial"] = record.get("trial_dir", record.get("path", record.get("trial_id")))
                row["online_recorded"] = True
                row["online_prediction_index"] = record.get("prediction_index")
                self.output_record_markers += 1
            return []
        if op == "model_error":
            self.gaps.append({"kind": "recorded_model_error", "record": copy.deepcopy(record)})
            if _applies(record, self.target):
                self.display_segment += 1
            return []
        if not _applies(record, self.target):
            return []
        if op == "begin_baseline":
            self.predictor.begin_baseline(seconds=float(record["seconds"]))
            self.baseline_called = True
            if record.get("display_reset", record.get("reset_display", False)):
                self._reset_display()
        elif op == "display_reset":
            self._reset_display()
        elif op in ("set_no_contact_hint", "no_contact_hint"):
            method = getattr(self.predictor, "set_no_contact_hint", None)
            if method is None:
                if self.allow_unsupported_controls:
                    self.unsupported_controls.append(copy.deepcopy(record))
                    return []
                raise ReplayProvenanceError("Recorded hint calls an unsupported native model API")
            method(record.get("value"), str(record.get("reason", "")))
        elif op in ("request_confirmed_baseline", "confirmed_baseline"):
            method = getattr(self.predictor, "request_confirmed_baseline", None)
            if method is None:
                if self.allow_unsupported_controls:
                    self.unsupported_controls.append(copy.deepcopy(record))
                    return []
                raise ReplayProvenanceError("Recorded confirmed B0 calls an unsupported native model API")
            method(seconds=float(record["seconds"]), reason=str(record.get("reason", "pre_action_b0")))
            self.baseline_called = True
            # Unlike the manual/full-Trial begin operation, the original
            # pre-action request does not reset the UI's contact bookkeeping.
        elif op in ("set_xy_rigid_transform", "xy_transform"):
            self.predictor.set_xy_rigid_transform(
                rotation_deg=float(record.get("rotation_deg", 0)),
                x_mm=float(record.get("x_mm", 0)), y_mm=float(record.get("y_mm", 0)),
                source=str(record.get("source", "manual")), details=record.get("details"))
        elif op == "group":
            if not self.baseline_called:
                raise ReplayProvenanceError("Input history begins without an explicit recorded initial B0/reset call")
            t_ns = int(record["t_ns"])
            if self.previous_input_ns is not None:
                delta = t_ns-self.previous_input_ns
                if delta <= 0 or delta > 1_000_000_000:
                    self.gaps.append({"kind": "native_input_time_gap_or_reversal", "before_ns": self.previous_input_ns, "after_ns": t_ns})
            self.previous_input_ns = t_ns
            self.group_count += 1
            data = {int(k): copy.deepcopy(v) for k, v in record["data"].items()}
            predictions = self.predictor.push_group(data, t_ns)
            if self.annotate_display:
                predictions = [self._annotate(p) for p in predictions]
            self.output_count += len(predictions)
            rows = [dict(asdict(p), trial=self.trial, online_recorded=False) for p in predictions]
            for row in rows:
                self.outputs_by_time.setdefault(int(row["t_ns"]), []).append(row)
            return rows
        else:
            raise ReplayProvenanceError("Unsupported native journal operation: " + str(op))
        self.controls.append(copy.deepcopy(record))
        return []


def _hash_fields(value, prefix=""):
    result = {}
    if isinstance(value, dict):
        for k, v in value.items():
            key = prefix + "." + k if prefix else k
            if "sha256" in k and isinstance(v, str):
                result[key] = v
            else:
                result.update(_hash_fields(v, key))
    return result


def _read_journal(path):
    with Path(path).open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReplayProvenanceError(f"Incomplete/corrupt journal line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ReplayProvenanceError(f"Journal line {line_number} is not an operation")
            yield record


def replay_journal(path, target="A", spec=None, compare_csv=None):
    """Replay a complete recorded native protocol, validating immutable models.

    An explicitly different ``spec`` is allowed for model comparison, not for
    claiming parity with the old online model. Calls unsupported by that new
    adapter are errors, never silently reinterpreted as a different B0 policy.
    """
    from general_model_selection_r387 import load, describe
    records = iter(_read_journal(path))
    header = next(records, None)
    if not header or header.get("op") != "header":
        raise ReplayProvenanceError("The native journal must begin with its model/protocol header")
    if header.get("schema") != JOURNAL_SCHEMA:
        raise ReplayProvenanceError("Unknown native journal schema")
    recorded = header.get("models", {}).get(target)
    if not recorded:
        raise ReplayProvenanceError("Target model is absent from the native journal: " + target)
    selected = copy.deepcopy(spec if spec is not None else recorded["selection"])
    same_model = spec is None or selected == recorded["selection"]
    predictor = load(selected, common_mapping=recorded.get("output_mapping") if same_model else selected.get("output_mapping"))
    description = describe(predictor, selected)
    source_hashes = implementation_sources(predictor)
    if same_model:
        before, now = _hash_fields(recorded), _hash_fields(description)
        changed = {k: [v, now.get(k)] for k, v in before.items() if v != now.get(k)}
        if changed:
            raise ReplayProvenanceError("Recorded model/profile/code artifacts changed: " + json.dumps(changed, ensure_ascii=False))
        recorded_sources = header.get("implementation_sources", {}).get(target, {})
        changed_sources = {k: [v, source_hashes.get(k)] for k, v in recorded_sources.items() if v != source_hashes.get(k)}
        if changed_sources:
            raise ReplayProvenanceError("Recorded native implementation changed: " + json.dumps(changed_sources, ensure_ascii=False))
    session = NativeReplaySession(predictor, target,
        annotate_display=header.get("display_annotations", {}).get(target, target == "A"),
        allow_unsupported_controls=not same_model, honor_output_record_markers=same_model)
    session.consume(header)
    rows = []
    terminated = False
    for record in records:
        rows.extend(session.consume(record))
        terminated = record.get("op") in ("end", "footer") or terminated
    report = {
        "schema": "MagTouch-NativeReplay-Report-R387-1", "mode": "ordered_native_journal",
        "source_journal": str(Path(path).resolve()), "source_sha256": sha256(path),
        "target": target, "model": description, "same_model_as_recorded_online": same_model,
        "native_implementation_sources_sha256": source_hashes,
        "recorded_implementation_hashes_available": bool(header.get("implementation_sources", {}).get(target)),
        "actual_native_input_groups": session.group_count, "native_prediction_rows": len(rows),
        "journal_terminated": terminated, "protocol_calls": session.controls, "gaps": session.gaps,
        "unsupported_controls_not_native_to_comparison_model": session.unsupported_controls,
        "actual_output_record_markers": session.output_record_markers,
        "no_external_resampling": True, "no_output_alignment": True, "no_wall_clock_sleep_required": True,
        "qualification": "Native algorithm and serialized input/control order reproduced; equality to archived online outputs is separately measured, not assumed.",
    }
    if compare_csv is not None:
        selected_rows = rows
        if session.output_record_markers:
            trial_path = str(Path(compare_csv).resolve().parent).replace("\\", "/").lower()
            selected_rows = [r for r in rows if r.get("online_recorded") and str(r.get("trial", "")).replace("\\", "/").lower().rstrip("/") == trial_path]
            if not selected_rows:
                trial_name = Path(compare_csv).resolve().parent.name.lower()
                names = {str(r.get("trial", "")).replace("\\", "/").rstrip("/") for r in rows if r.get("online_recorded") and str(r.get("trial", "")).replace("\\", "/").rstrip("/").split("/")[-1].lower() == trial_name}
                if len(names) != 1:
                    raise ReplayProvenanceError("The selected online CSV cannot be uniquely identified by output_recorded Trial markers")
                selected_rows = [r for r in rows if r.get("online_recorded") and str(r.get("trial", "")).replace("\\", "/").rstrip("/") in names]
            report["comparison_trial_selected_by_output_markers"] = str(Path(compare_csv).resolve().parent)
        report["online_comparison"] = compare_online_csv(selected_rows, compare_csv)
    return {"metadata": report, "rows": rows}


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def compare_online_csv(rows, csv_path):
    """Exact timestamp join only. No fitted translation/time shift/nearest join."""
    by_time = {}
    duplicate = 0
    for row in rows:
        t = int(row["t_ns"])
        if t in by_time:
            duplicate += 1
        by_time[t] = row
    matched, missing, mismatch_valid, max_abs, sum_distance, finite = 0, 0, 0, 0., 0., 0
    online_count = 0
    for old in _csv_rows(csv_path):
        online_count += 1
        new = by_time.get(int(old["t_prediction_ns"]))
        if new is None:
            missing += 1
            continue
        matched += 1
        old_valid = str(old.get("interaction_valid", "")).lower() in ("1", "true")
        mismatch_valid += old_valid != bool(new["interaction_valid"])
        old_xyz = [_float(old.get("visual_"+a+"_mm")) for a in "xyz"]
        new_xyz = [_float(v) for v in new["visual_xyz_mm"]]
        if all(math.isfinite(v) for v in old_xyz+new_xyz):
            differences = [abs(a-b) for a, b in zip(old_xyz, new_xyz)]
            max_abs = max(max_abs, *differences)
            sum_distance += math.sqrt(sum(v*v for v in differences))
            finite += 1
        elif any(math.isfinite(a) != math.isfinite(b) for a, b in zip(old_xyz, new_xyz)):
            max_abs = float("inf")
    return dict(source_csv=str(Path(csv_path).resolve()), source_sha256=sha256(csv_path),
                online_rows=online_count, exact_timestamp_matches=matched, online_rows_not_reproduced=missing,
                replay_rows_not_in_online=max(0, len(rows)-matched), duplicate_replay_timestamps=duplicate,
                validity_disagreements=mismatch_valid, finite_xyz_pairs=finite,
                max_absolute_axis_difference_mm=max_abs, mean_output_distance_mm=sum_distance/finite if finite else None,
                posthoc_time_or_coordinate_alignment=False,
                xyz_and_validity_parity=(missing == 0 and matched == len(rows) and duplicate == 0 and mismatch_valid == 0 and max_abs <= 1e-10))


def raw_csv_group(row):
    """Restore exactly the saved native sensor fields and trigger timestamp."""
    data = {}
    fields = {"x_ut": "Bx_uT", "y_ut": "By_uT", "z_ut": "Bz_uT", "mag_ut": "Bmag_uT",
              "x_raw": "x_raw", "y_raw": "y_raw", "z_raw": "z_raw", "temp_c": "temp_C",
              "trigger_ns": "trigger_ns", "read_ns": "read_ns", "status": "status"}
    for index in range(5):
        prefix = f"S{index+1}_"
        if not any(row.get(prefix+axis, "") not in (None, "") for axis in ("Bx_uT", "By_uT", "Bz_uT")):
            continue
        sensor = {}
        for key, field in fields.items():
            value = row.get(prefix+field)
            if value not in (None, ""):
                sensor[key] = int(value) if key in ("trigger_ns", "read_ns", "status", "x_raw", "y_raw", "z_raw") else float(value)
        data[index] = sensor
    return dict(op="group", t_ns=int(row["t_group_ns"]), data=data,
                available_ns=int(row.get("t_return_ns") or row["t_group_ns"]), cycle_id=row.get("cycle_id"))


class ArchivedStageHints:
    """The exact R68+R136 stage hint policy, not motion-derived XYZ labels."""
    SAFE_PREFIXES = ("safe_z", "approach", "press_approach", "preload_approach", "retract",
                     "grid_approach", "grid_retract", "grid_reference_safe", "grid_reference_approach",
                     "final_safe_z", "path_approach", "path_release", "dynamic_b0_safe_z", "recenter")
    WAYPOINT_SAFE = {"waypoint_safe_z", "waypoint_retract_safe", "waypoint_final_safe_z"}

    def __init__(self):
        self.safe = False
        self.pending = False

    def consume(self, row):
        command, phase = str(row.get("command", "")), str(row.get("phase", ""))
        hint = None
        if command.startswith("G1 "):
            if row.get("z_mm") not in (None, ""):
                self.pending = any(phase.startswith(p) for p in self.SAFE_PREFIXES)
                if not self.pending:
                    self.safe = False
            else:
                self.pending = self.safe
            hint = False
        elif command == "HOST_WAIT":
            self.safe = bool(self.pending)
            hint = self.safe
        if command.startswith("G1 ") and phase in self.WAYPOINT_SAFE and row.get("z_mm") not in (None, ""):
            self.pending = True
        return None if hint is None else dict(op="set_no_contact_hint", value=hint, reason=phase)


def replay_legacy_trial(trial_dir, spec, begin_baseline_seconds=None, compare_csv=None):
    """Causal archived replay, explicit qualification when old provenance is missing.

    A legacy initial dynamic B0 precedes magnetic_raw.csv. Its saved median is
    NOT a complete native state (noise statistics/history are missing). Without
    an explicitly supplied initial unloaded capture protocol, replay starts at
    the first archived confirmed pre-action B0 and omits the earlier prefix.
    """
    from general_model_selection_r387 import load, describe
    path = Path(trial_dir).resolve()
    exclusions = ROOT / "analysis/r387_soft_state_research_map_20261002/exclusions.json"
    if exclusions.is_file():
        # Exclusions are not optional. Prefix paths remain excluded after copies.
        document = json.loads(exclusions.read_text(encoding="utf-8-sig"))
        def strings(x):
            if isinstance(x, str): yield x
            elif isinstance(x, dict):
                for v in x.values(): yield from strings(v)
            elif isinstance(x, list):
                for v in x: yield from strings(v)
        for value in strings(document):
            normalized = value.replace("\\", "/").rstrip("/").lower()
            if "plan_" in normalized and normalized in str(path).replace("\\", "/").lower():
                raise ReplayProvenanceError("This acquisition batch is explicitly invalidated: " + value)
    predictor = load(spec)
    session = NativeReplaySession(predictor)
    raw_path, event_path, stage_path = (path / n for n in ("magnetic_raw.csv", "events.csv", "stage_commands.csv"))
    timeline = []
    hints = ArchivedStageHints()
    if stage_path.is_file() and hasattr(predictor, "set_no_contact_hint"):
        for row in _csv_rows(stage_path):
            call = hints.consume(row)
            if call is not None:
                timeline.append((int(row["t_perf_ns"]), 0, call))
    if event_path.is_file():
        for row in _csv_rows(event_path):
            if row.get("event") == "PRE_ACTION_B0_WINDOW":
                payload = json.loads(row["value"])
                if payload.get("live_normalized_fit_updated"):
                    timeline.append((int(row["t_perf_ns"]), 1, dict(op="request_confirmed_baseline",
                                    seconds=float(payload["capture_seconds"]), reason=payload["phase"])))
    raw_records = [raw_csv_group(row) for row in _csv_rows(raw_path)]
    if begin_baseline_seconds is not None:
        if not raw_records:
            raise ReplayProvenanceError("No raw groups")
        timeline.append((raw_records[0]["available_ns"]-1, -1,
                         dict(op="begin_baseline", seconds=float(begin_baseline_seconds))))
    for record in raw_records:
        timeline.append((record["available_ns"], 2, record))
    timeline.sort(key=lambda item: (item[0], item[1]))
    rows, skipped = [], 0
    for _, _, record in timeline:
        if record["op"] == "group" and not session.baseline_called:
            skipped += 1
            continue
        rows.extend(session.consume(record))
    if not session.baseline_called:
        raise ReplayProvenanceError("Trial lacks the initial unloaded raw B0 window and any reconstructable confirmed pre-action B0; no baseline guessed. Supply its native journal or an explicit verified unloaded prefix protocol.")
    report = dict(schema="MagTouch-NativeReplay-Report-R387-1", mode="legacy_actual_time_native_adapter",
                  trial_dir=str(path), model=describe(predictor, spec), native_prediction_rows=len(rows),
                  native_implementation_sources_sha256=implementation_sources(predictor),
                  raw_groups=len(raw_records), raw_prefix_groups_omitted=skipped, gaps=session.gaps,
                  protocol_calls=session.controls, input_hashes={p.name: sha256(p) for p in (raw_path, event_path, stage_path) if p.is_file()},
                  actual_recorded_trigger_intervals_preserved=True, no_external_resampling=True,
                  missing_online_processed_input_subset=True, control_worker_execution_order_unknown=True,
                  initial_window_explicitly_supplied=begin_baseline_seconds is not None,
                  saved_b0_median_not_used_as_full_state=True,
                  qualification="Native algorithms and archived causal controls; legacy input/control execution provenance is incomplete, so this is not certified bit-equivalent native online inference.")
    if compare_csv is not None:
        report["online_comparison"] = compare_online_csv(rows, compare_csv)
    return dict(metadata=report, rows=rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--journal", type=Path)
    mode.add_argument("--trial", type=Path)
    parser.add_argument("--target", choices=("A", "B"), default="A")
    parser.add_argument("--model-spec", type=Path, help="Serialized native model selection JSON; required for legacy archive")
    parser.add_argument("--begin-baseline-seconds", type=float, help="Legacy only: explicitly verified unloaded prefix capture, never inferred automatically")
    parser.add_argument("--compare-csv", type=Path, help="Original native output CSV, joined at EXACT prediction timestamps")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    spec = json.loads(args.model_spec.read_text(encoding="utf-8-sig")) if args.model_spec else None
    if args.trial and spec is None:
        parser.error("--trial requires --model-spec; a generic fit replacement is forbidden")
    result = replay_journal(args.journal, args.target, spec, args.compare_csv) if args.journal else replay_legacy_trial(args.trial, spec, args.begin_baseline_seconds, args.compare_csv)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "native_predictions.jsonl").open("w", encoding="utf-8") as output:
        for row in result["rows"]:
            output.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")
    (args.output_dir / "native_replay_report.json").write_text(json.dumps(result["metadata"], ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps({k: v for k, v in result["metadata"].items() if k in ("mode", "native_prediction_rows", "qualification", "online_comparison")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
