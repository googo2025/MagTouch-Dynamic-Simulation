"""Sequential, device-free native model comparisons for CLOSED acquisition Trials.

The default uses the actual serialized R387/R390 input/control journal, including
pre-Trial B0 and history. It does not manufacture a B0 from a saved median, fill
input groups dropped online, switch to R39, resample, align outputs, or train.
An offline candidate in a single-A journal follows target A's protocol, but its
new emissions are NOT filtered by A's online-recorded markers.
"""
from __future__ import annotations

import argparse
import collections
import copy
import csv
import gc
import hashlib
import json
import re
from pathlib import Path

import native_online_replay_r387 as native


SCHEMA = "MagTouch-OfflineComparison-R390-1"
ReplayProvenanceError = native.ReplayProvenanceError


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _normalized(value):
    return str(value or "").replace("\\", "/").rstrip("/").lower()


def _trial_ref(record):
    return record.get("trial_dir", record.get("path", record.get("trial_id", "")))


def _matches_trial(value, trial):
    value = _normalized(value)
    requested = _normalized(trial)
    return value == requested or value.rsplit("/", 1)[-1] == trial.name.lower()


def _assert_not_excluded(trial):
    exclusions = native.ROOT / "analysis/r387_soft_state_research_map_20261002/exclusions.json"
    if not exclusions.is_file():
        return

    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for child in value.values():
                yield from strings(child)
        elif isinstance(value, list):
            for child in value:
                yield from strings(child)

    for value in strings(_json(exclusions)):
        prefix = _normalized(value)
        if "plan_" in prefix and prefix in _normalized(trial):
            raise ReplayProvenanceError("Acquisition batch is explicitly invalidated: " + value)


def _journal_path(trial, meta):
    live = meta.get("live_prediction_recording", {})
    comparison = live.get("native_comparison_r387", meta.get("native_comparison_r387", {}))
    value = (meta.get("native_inference_journal_r387")
             or live.get("native_inference_journal_r387") or comparison.get("journal"))
    if not value:
        raise ReplayProvenanceError(
            "No ordered native journal. Legacy raw CSV/B0 metadata cannot certify native replay; no R39 fallback.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        candidates = [trial / path, native.ROOT / path]
        path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
    path = path.resolve()
    if not path.is_file():
        raise ReplayProvenanceError("Recorded native journal is missing: " + str(path))
    return path


def _source_csvs(trial, header, meta):
    """Only existing native online files establish recorded marker membership."""
    result = {}
    a_name = meta.get("live_prediction_recording", {}).get("file", "live_predictions.csv")
    names = {"A": a_name, "B": "live_predictions_virtual1000.csv"}
    for target, description in header.get("models", {}).items():
        if target not in names:
            continue
        path = (trial / names[target]).resolve()
        if not path.is_relative_to(trial):
            raise ReplayProvenanceError("Online CSV must belong to the selected Trial")
        if not path.is_file():
            continue
        timestamps = collections.Counter()
        with path.open(encoding="utf-8-sig", newline="") as source:
            for row in csv.DictReader(source):
                timestamps[int(row["t_prediction_ns"])] += 1
                family = row.get("model_family")
                if family and family != description.get("family"):
                    raise ReplayProvenanceError("Archived CSV model family differs from native journal header")
        result[target] = {"path": path, "timestamps": timestamps,
                          "rows": sum(timestamps.values()), "sha256": native.sha256(path)}
    if "A" not in result:
        raise ReplayProvenanceError("A's archived native CSV is required to verify the closed Trial journal prefix")
    reported = meta.get("live_prediction_recording", {}).get("rows_written")
    if reported is not None and int(reported) != result["A"]["rows"]:
        raise ReplayProvenanceError("Closed Trial metadata and A's CSV row counts disagree")
    return result


def _capture_protocol_prefix(trial, journal, meta):
    """Stop at this Trial's close and all archived markers, not an active tail.

    Keeping the exact prefix is necessary: baseline/history may start before
    Trial creation. A completed Trial can be replayed while later Trials append
    to the same journal, without reading those later CSVs or requiring run end.
    """
    payload = bytearray()
    started = closed = False
    references = set()
    markers = collections.defaultdict(collections.Counter)
    previous_seq = None
    with journal.open("rb") as source:
        first = source.readline()
        try:
            header = json.loads(first.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReplayProvenanceError("Invalid native journal header") from exc
        if header.get("op") != "header" or header.get("schema") != native.JOURNAL_SCHEMA:
            raise ReplayProvenanceError("Unknown native journal schema/header")
        if header.get("seq") != 0:
            raise ReplayProvenanceError("Native journal must preserve the original sequence starting at header 0")
        if "A" not in header.get("models", {}):
            raise ReplayProvenanceError("Native comparison requires the actually recorded A protocol")
        csvs = _source_csvs(trial, header, meta)
        payload.extend(first)
        previous_seq = header.get("seq")
        for line_number, line in enumerate(source, 2):
            if not line.strip():
                payload.extend(line)
                continue
            try:
                record = json.loads(line.decode("utf-8-sig"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ReplayProvenanceError(f"Incomplete/corrupt journal before selected Trial closure, line {line_number}") from exc
            if not isinstance(record, dict):
                raise ReplayProvenanceError("Native journal operation is not an object")
            seq = record.get("seq")
            if previous_seq is not None and (seq is None or int(seq) != int(previous_seq) + 1):
                raise ReplayProvenanceError("Native journal prefix has missing/reordered operations")
            previous_seq = seq
            payload.extend(line)
            op = record.get("op", record.get("kind"))
            reference = _trial_ref(record)
            if op in ("trial_start", "trial_end", "output_recorded") and _matches_trial(reference, trial):
                references.add(_normalized(reference))
                if len(references) != 1:
                    raise ReplayProvenanceError("Selected Trial basename is ambiguous in the native journal")
                if op == "trial_start":
                    if started:
                        raise ReplayProvenanceError("Duplicate selected Trial start")
                    started = True
                elif op == "trial_end":
                    if not started:
                        raise ReplayProvenanceError("Selected Trial ends before its start")
                    if record.get("status", "completed") != "completed":
                        raise ReplayProvenanceError("Journal Trial closure is not completed")
                    closed = True
                else:
                    target = record.get("target", "A")
                    if target in csvs:
                        markers[target][int(record["t_ns"])] += 1
            if closed and all(markers[target] == item["timestamps"] for target, item in csvs.items()):
                return bytes(payload), header, csvs, dict(
                    selected_trial_reference=next(iter(references)),
                    prefix_bytes=len(payload), prefix_sha256=hashlib.sha256(payload).hexdigest(),
                    last_native_sequence=seq, selected_trial_start_observed=started,
                    selected_trial_completed_end_observed=closed,
                    archived_output_markers_verified={target: item["rows"] for target, item in csvs.items()},
                    later_trials_and_active_journal_tail_not_required=True)
    raise ReplayProvenanceError("Journal lacks the completed Trial boundary or exact archived output markers")


def _select_protocol(label, spec, header):
    recorded = header["models"]
    if spec is None:
        target = label if label in recorded else "A"
        return target, None, True
    targets = [key for key, description in recorded.items() if spec == description.get("selection")]
    target = label if label in targets else (targets[0] if targets else "A")
    selected = copy.deepcopy(spec)
    if not targets and selected.get("kind") == "neural" and selected.get("custom"):
        # The old factory takes custom=True's family from a separate keyword,
        # which replay_journal does not pass. Freeze an OFFLINE candidate's
        # explicit family in the selection instead of silently using auto.
        # Recorded models stay byte/object-identical so their marker/hash
        # semantics remain those of their actual original loader.
        original_sha = hashlib.sha256(json.dumps(selected, sort_keys=True, ensure_ascii=False,
                                                separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
        selected.update(custom=False, custom_checkpoint=True,
                        neural_family=selected.get("neural_family", "auto"),
                        selection_normalization_r390=dict(
                            reason="Explicit custom-checkpoint family frozen for the native factory; no implicit auto fallback",
                            original_selection_sha256=original_sha,
                            original_custom=True,
                            original_neural_family=spec.get("neural_family", "auto")))
    return target, selected, bool(targets)


def _selected_rows(rows, trial, same_model):
    if same_model:
        # Real recording assignment may differ from the active Trial at input
        # consumption because outputs are buffered. Honor actual markers only
        # when reproducing the same online model.
        return [row for row in rows if row.get("online_recorded") and _matches_trial(row.get("trial"), trial)]
    # A new candidate has new emissions, including possible adapter grid times.
    # Its outputs are assigned by the actual active-Trial journal protocol,
    # never deleted for failing to match A's online-recorded marker set.
    return [row for row in rows if _matches_trial(row.get("trial"), trial)]


def run_comparison(trial_path, specs, output_dir, mode="recorded_native"):
    """Run named native models one after another, in a NEW output directory.

    ``specs`` is an insertion-ordered mapping, e.g. ``{'A': None, 'B': specB}``.
    ``None`` reproduces the recorded model matching its label, or recorded A.
    A matching recorded selection validates native artifact/source hashes and
    actual output markers. A different selection follows the recorded A input
    and control protocol and is qualified as an OFFLINE counterfactual, not a
    newly observed online run. No full-raw/legacy-native equivalence is offered.
    """
    if mode != "recorded_native":
        raise ReplayProvenanceError(
            "Only recorded_native is implemented. Full raw has a different input subset; legacy B0/control order is incomplete.")
    if not isinstance(specs, dict) or not specs:
        raise ValueError("specs must be a nonempty ordered mapping of label to native selection or None")
    frozen_specs = copy.deepcopy(specs)
    for label, spec in frozen_specs.items():
        if not isinstance(label, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", label):
            raise ValueError("Use simple safe model labels such as A and B")
        if spec is not None and not isinstance(spec, dict):
            raise ValueError("Each model entry must be a native selection object or None")
    json.dumps(frozen_specs, ensure_ascii=False, allow_nan=False)
    trial = Path(trial_path).expanduser().resolve()
    _assert_not_excluded(trial)
    meta_path = trial / "meta.json"
    meta = _json(meta_path)
    if meta.get("status") != "completed":
        raise ReplayProvenanceError("Only completed/closed Trials may be compared offline")
    journal = _journal_path(trial, meta)
    output = Path(output_dir).expanduser().resolve()
    raw_root = (native.ROOT / "data").resolve()
    if output == trial or output.is_relative_to(trial) or output.is_relative_to(journal.parent) or output.is_relative_to(raw_root):
        raise ReplayProvenanceError("Offline outputs must be outside original Trial/journal/raw-data directories")
    if output.exists():
        raise FileExistsError("Output directory must be NEW: " + str(output))
    source_files = [trial / name for name in ("meta.json", "identity.json", "batch_plan_item.json",
                    "magnetic_raw.csv", "events.csv", "stage_commands.csv") if (trial / name).is_file()]
    hashes = {str(path): native.sha256(path) for path in source_files}
    prefix, header, csvs, prefix_report = _capture_protocol_prefix(trial, journal, meta)
    hashes.update({str(item["path"]): item["sha256"] for item in csvs.values()})
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "native_protocol_prefix.jsonl"
    snapshot.write_bytes(prefix)
    report = dict(schema=SCHEMA, status="RUNNING", mode=mode, source_trial=str(trial),
                  comparison_tool_sha256=native.sha256(Path(__file__)),
                  source_files_sha256=hashes, source_journal=str(journal),
                  source_journal_hash_scope="Exact immutable prefix through this closed Trial, NOT a hash of the later active run",
                  native_protocol_snapshot=str(snapshot), native_protocol_prefix=prefix_report,
                  requested_specs=frozen_specs, execution_order=list(frozen_specs),
                  models_run_sequentially=True, hardware_operations=[], training_operations=[],
                  no_legacy_R39_substitution=True, no_external_resampling=True,
                  no_added_output_alignment=True, no_output_alignment=True,
                  full_raw_input_replacement=False, models={},
                  qualification="Recorded native input/control replay; offline candidates are counterfactuals, not new online measurements.")
    results = {}
    try:
        for label, selected in frozen_specs.items():
            target, selected, same = _select_protocol(label, selected, header)
            if same and target not in csvs:
                raise ReplayProvenanceError("Missing archived CSV for the recorded model: " + target)
            result = native.replay_journal(snapshot, target=target, spec=selected,
                                           compare_csv=csvs[target]["path"] if same else None)
            if result["metadata"].get("unsupported_controls_not_native_to_comparison_model"):
                raise ReplayProvenanceError("Comparison model cannot reproduce recorded native controls; no controls silently dropped")
            rows = _selected_rows(result["rows"], trial, same)
            metadata = result["metadata"]
            transforms = [call for call in metadata.get("protocol_calls", [])
                          if call.get("op") in ("xy_transform", "set_xy_rigid_transform")]
            nonidentity = any(any(float(call.get(key, 0.0)) != 0.0
                                 for key in ("rotation_deg", "x_mm", "y_mm")) for call in transforms)
            metadata.update(offline_comparison_label=label, selected_trial=str(trial),
                            selected_trial_prediction_rows=len(rows), protocol_target=target,
                            selected_trial_closed=True, whole_run_termination_not_required=True,
                            candidate_is_new_native_online_result=False,
                            selected_row_policy="actual_output_recorded_markers" if same else "active_trial_journal_protocol_not_A_output_markers",
                            fully_replayed_recorded_controls=True,
                            recorded_output_transforms_replayed=transforms,
                            no_added_output_alignment=True, no_output_alignment=not nonidentity,
                            exact_online_parity_claimed=bool(same and metadata.get("online_comparison", {}).get("xyz_and_validity_parity")))
            if not same:
                metadata["qualification"] = (
                    "Different native model, same recorded A input/control execution protocol. New emissions retained; "
                    "not an online measurement, not equality to A's outputs, not full-raw replay.")
            model_dir = output / label
            model_dir.mkdir(exist_ok=False)
            with (model_dir / "native_predictions.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")
            (model_dir / "native_replay_report.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
            report["models"][label] = metadata
            report["no_output_alignment"] = report["no_output_alignment"] and not nonidentity
            results[label] = {"metadata": metadata, "rows": rows, "output_dir": str(model_dir)}
            del result
            gc.collect()  # Do not retain a predictor while constructing the next one.
        changed = [path for path, digest in hashes.items() if native.sha256(path) != digest]
        if changed:
            raise ReplayProvenanceError("Closed source files changed during comparison: " + repr(changed))
        report["status"] = "COMPLETED"
        report["original_source_files_unchanged"] = True
    except Exception as exc:
        report.update(status="FAILED", error_type=type(exc).__name__, error=str(exc),
                      qualification="Incomplete/failed comparison; partial artifacts retained, no completion or online equivalence claim.")
        raise
    finally:
        (output / "comparison_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    return {"metadata": report, "models": results, "output_dir": str(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--specs", type=Path, required=True, help="JSON mapping, e.g. {A:null,B:native_selection}")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("recorded_native",), default="recorded_native")
    args = parser.parse_args(argv)
    result = run_comparison(args.trial, _json(args.specs), args.output_dir, args.mode)
    print(json.dumps(dict(status=result["metadata"]["status"], output_dir=result["output_dir"],
                         models={label: len(value["rows"]) for label, value in result["models"].items()}), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
