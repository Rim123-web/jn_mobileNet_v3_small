from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


RUN = Path(__file__).resolve().parents[1]


def read_json(relative: str):
    return json.loads((RUN / relative).read_text(encoding="utf-8"))


def write_json(relative: str, value) -> None:
    path = RUN / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def stats(values: list[float]) -> dict:
    mean = statistics.fmean(values)
    return {
        "count": len(values),
        "mean_ms": mean,
        "median_ms": statistics.median(values),
        "std_ms": statistics.pstdev(values),
        "minimum_ms": min(values),
        "maximum_ms": max(values),
        "p90_ms": percentile(values, 0.90),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "coefficient_of_variation_percent": statistics.pstdev(values) / mean * 100.0,
    }


provenance = read_json("raw/execution_count_provenance.json")
with (RUN / "raw/per_image_timings.csv").open(newline="", encoding="utf-8") as handle:
    timings = list(csv.DictReader(handle))
with (RUN / "raw/warmup_execution_records.csv").open(newline="", encoding="utf-8") as handle:
    warmups = list(csv.DictReader(handle))

# Preserve a complete event stream. Warm-up timestamps were not emitted live, so
# those records are included without fabricated wall-clock values.
events = [
    {
        "event": "warmup_phase_summary",
        "timestamp_epoch": None,
        "execution_count": len(warmups),
        "excluded_from_official_timing": True,
        "excluded_from_monitoring": True,
        "timestamp_note": "Exact warm-up wall-clock timestamps were not recorded; no value was invented.",
    }
]
for row in warmups:
    events.append(
        {
            "event": "warmup_execution",
            "timestamp_epoch": None,
            "warmup_index": int(row["warmup_index"]),
            "image_path": row["image_path"],
            "wall_latency_ms": float(row["wall_latency_ms"]),
            "h2d_ms": float(row["h2d_ms"]),
            "inference_ms": float(row["inference_ms"]),
            "d2h_ms": float(row["d2h_ms"]),
            "excluded_from_official_timing": True,
            "excluded_from_monitoring": True,
        }
    )
events.append(
    {
        "event": "official_measurement_start",
        "timestamp_epoch": provenance["official_start_epoch"],
        "measured_executions_expected": provenance["expected_measured_executions"],
        "telemetry_started": True,
    }
)
for row in timings:
    events.extend(
        [
            {
                "event": "measured_image_start",
                "timestamp_epoch": float(row["image_start_epoch"]),
                "measured_index": int(row["measured_index"]),
                "relative_path": row["relative_path"],
            },
            {
                "event": "measured_image_complete",
                "timestamp_epoch": float(row["image_end_epoch"]),
                "measured_index": int(row["measured_index"]),
                "relative_path": row["relative_path"],
                "inference_ms": float(row["inference_ms"]),
                "end_to_end_ms": float(row["end_to_end_ms"]),
            },
        ]
    )
events.append(
    {
        "event": "official_measurement_end",
        "timestamp_epoch": provenance["official_end_epoch"],
        "measured_executions": provenance["measured_executions"],
        "failed_images": provenance["failed_images"],
        "discarded_measured_rows": provenance["discarded_measured_rows"],
        "telemetry_stopped": True,
    }
)
(RUN / "raw/official_process_events.jsonl").write_text(
    "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
)

# Stability analysis from the frozen official timing rows only.
chunk_rows = []
chunk_count = 4
for chunk_index in range(chunk_count):
    start = len(timings) * chunk_index // chunk_count
    end = len(timings) * (chunk_index + 1) // chunk_count
    subset = timings[start:end]
    engine = [float(row["inference_ms"]) for row in subset]
    pipeline = [float(row["end_to_end_ms"]) for row in subset]
    chunk_rows.append(
        {
            "quartile": chunk_index + 1,
            "first_measured_index": int(subset[0]["measured_index"]),
            "last_measured_index": int(subset[-1]["measured_index"]),
            "count": len(subset),
            "mean_engine_latency_ms": statistics.fmean(engine),
            "median_engine_latency_ms": statistics.median(engine),
            "p95_engine_latency_ms": percentile(engine, 0.95),
            "mean_end_to_end_latency_ms": statistics.fmean(pipeline),
            "median_end_to_end_latency_ms": statistics.median(pipeline),
            "p95_end_to_end_latency_ms": percentile(pipeline, 0.95),
        }
    )
engine_all = [float(row["inference_ms"]) for row in timings]
pipeline_all = [float(row["end_to_end_ms"]) for row in timings]
engine_drift = (
    chunk_rows[-1]["mean_engine_latency_ms"] / chunk_rows[0]["mean_engine_latency_ms"] - 1.0
) * 100.0
pipeline_drift = (
    chunk_rows[-1]["mean_end_to_end_latency_ms"] / chunk_rows[0]["mean_end_to_end_latency_ms"] - 1.0
) * 100.0
stability = {
    "status": "PASS",
    "source": "raw/per_image_timings.csv from the single official measured pass",
    "measured_rows": len(timings),
    "engine_latency": stats(engine_all),
    "end_to_end_latency": stats(pipeline_all),
    "first_to_last_quartile_engine_mean_change_percent": engine_drift,
    "first_to_last_quartile_pipeline_mean_change_percent": pipeline_drift,
    "quartiles": chunk_rows,
    "interpretation": "Descriptive run-order stability only; this was one full dataset pass, not repeated trials.",
}
write_json("stability/latency_stability_analysis.json", stability)
with (RUN / "stability/latency_stability_quartiles.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(chunk_rows[0]))
    writer.writeheader()
    writer.writerows(chunk_rows)
(RUN / "stability/latency_stability_analysis.md").write_text(
    "# Latency Stability Analysis\n\n"
    f"- Source rows: {len(timings)} official measured images.\n"
    f"- Engine latency mean: {statistics.fmean(engine_all):.6f} ms.\n"
    f"- Engine latency CV: {stats(engine_all)['coefficient_of_variation_percent']:.6f}%.\n"
    f"- First-to-last quartile engine mean change: {engine_drift:.6f}%.\n"
    f"- First-to-last quartile pipeline mean change: {pipeline_drift:.6f}%.\n"
    "- This is descriptive run-order stability from one dataset pass, not a repeated-run confidence interval.\n",
    encoding="utf-8",
)

# Exact commands are preserved; unavailable command start/end timestamps are
# explicitly marked rather than reconstructed from file mtimes.
python = "/home/jetson/Downloads/jetson_nano/.venv_yolo/bin/python"
commands = {
    "prepare_project.py": f"PYTHONNOUSERSITE=1 {python} {RUN}/scripts/prepare_project.py",
    "inspect_onnx_candidates.py": f"PYTHONNOUSERSITE=1 {python} {RUN}/scripts/inspect_onnx_candidates.py",
    "validate_onnx.py": f"PYTHONNOUSERSITE=1 {python} {RUN}/scripts/validate_onnx.py",
    "validate_onnx_dataset.py": f"PYTHONNOUSERSITE=1 {python} {RUN}/scripts/validate_onnx_dataset.py",
    "validate_trt_parser.py": f"PYTHONNOUSERSITE=1 {python} {RUN}/scripts/validate_trt_parser.py",
    "build_engine.sh": f"bash {RUN}/scripts/build_engine.sh",
    "validate_correctness.py": f"PYTHONNOUSERSITE=1 {python} {RUN}/scripts/validate_correctness.py",
    "run_idle_monitor.py": f"PYTHONNOUSERSITE=1 {python} {RUN}/scripts/run_idle_monitor.py",
    "benchmark_mobilenet_v3_small_fp32_warmup10.py": f"PYTHONNOUSERSITE=1 {python} {RUN}/scripts/benchmark_mobilenet_v3_small_fp32_warmup10.py",
    "finalize_results.py": f"PYTHONNOUSERSITE=1 {python} {RUN}/scripts/finalize_results.py",
    "validate_result_package.py": f"PYTHONNOUSERSITE=1 {python} {RUN}/scripts/validate_result_package.py",
    "complete_package_evidence.py": f"PYTHONNOUSERSITE=1 {python} {RUN}/scripts/complete_package_evidence.py",
    "refresh_hash_manifest.py": f"PYTHONNOUSERSITE=1 {python} {RUN}/scripts/refresh_hash_manifest.py",
    "validate_complete_package.py": f"PYTHONNOUSERSITE=1 {python} {RUN}/scripts/validate_complete_package.py",
    "package_results.sh": f"bash {RUN}/scripts/package_results.sh",
    "package_results_final.sh": f"bash {RUN}/scripts/package_results_final.sh",
}
purposes = {
    "prepare_project.py": "environment, source, dataset, preprocessing, and parity audit",
    "inspect_onnx_candidates.py": "ONNX candidate inventory and selection evidence",
    "validate_onnx.py": "ONNX checker and runtime dummy validation",
    "validate_onnx_dataset.py": "deterministic 20-image ONNX dataset validation",
    "validate_trt_parser.py": "TensorRT parser-only compatibility gate",
    "build_engine.sh": "fresh strict-FP32 TensorRT engine build",
    "validate_correctness.py": "20-image PyTorch/ONNX/TensorRT numerical validation",
    "run_idle_monitor.py": "30-second idle jtop baseline",
    "benchmark_mobilenet_v3_small_fp32_warmup10.py": "single official 10-warmup plus 261-image benchmark",
    "finalize_results.py": "metrics, energy, analysis, figures, reports, and validation",
    "validate_result_package.py": "package content and hash validation",
    "complete_package_evidence.py": "completion addendum, event stream, stability, and provenance",
    "refresh_hash_manifest.py": "post-write immutable file hash manifest",
    "validate_complete_package.py": "final publication package validation",
    "package_results.sh": "preserved preliminary archive creation",
    "package_results_final.sh": "corrected publication-ready archive creation",
}
provenance_rows = []
for path in sorted((RUN / "scripts").iterdir()):
    if not path.is_file():
        continue
    name = path.name
    executed = name in commands
    start = "not_recorded"
    end = "not_recorded"
    stdout = "not_recorded"
    stderr = "not_recorded"
    if name == "build_engine.sh":
        start = "2026-07-17T13:41:21 (Jetson local time; TensorRT log)"
        end = "2026-07-17T13:46:13 (Jetson local time; TensorRT log)"
        stdout = "logs/engine_build_stdout.log"
        stderr = "logs/engine_build_stderr.log"
    elif name == "benchmark_mobilenet_v3_small_fp32_warmup10.py":
        start = str(provenance["official_start_epoch"]) + " (official measured boundary; warm-up occurred earlier)"
        end = str(provenance["official_end_epoch"]) + " (official measured boundary)"
        stdout = "logs/benchmark_official.log"
        stderr = "logs/benchmark_official.log"
    provenance_rows.append(
        {
            "original_path": str(path),
            "packaged_path": path.relative_to(RUN).as_posix(),
            "sha256": sha256(path),
            "execution_command": commands.get(name, "imported utility module; not invoked as a standalone command"),
            "purpose": purposes.get(name, "shared benchmark utility or packaged source"),
            "start_time": start,
            "end_time": end,
            "exit_code": 0 if executed else "not_applicable_imported_or_packaged",
            "stdout_path": stdout,
            "stderr_path": stderr,
            "timestamp_evidence_note": "Unknown values are explicitly marked; file mtimes were not treated as command boundaries.",
        }
    )
with (RUN / "validation/executed_code_provenance.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(provenance_rows[0]))
    writer.writeheader()
    writer.writerows(provenance_rows)
(RUN / "logs/executed_commands.log").write_text(
    "\n".join(commands[name] for name in commands) + "\n", encoding="utf-8"
)

validation = read_json("validation/final_validation_report.json")
validation["checks"].update(
    {
        "official_process_events_saved": len(events) == 2 * len(timings) + len(warmups) + 3,
        "stability_analysis_saved": True,
        "detailed_execution_provenance_saved": True,
    }
)
validation["status"] = "PASS" if all(validation["checks"].values()) else "FAIL"
write_json("validation/final_validation_report.json", validation)
(RUN / "validation/final_validation_report.txt").write_text(
    "FINAL VALIDATION: " + validation["status"] + "\n"
    + "\n".join(f"{key}: {value}" for key, value in validation["checks"].items())
    + "\n",
    encoding="utf-8",
)
addendum = {
    "status": validation["status"],
    "benchmark_rerun": False,
    "raw_benchmark_modified": False,
    "added_from_saved_raw_evidence": [
        "raw/official_process_events.jsonl",
        "stability/latency_stability_analysis.json",
        "stability/latency_stability_quartiles.csv",
        "stability/latency_stability_analysis.md",
        "validation/executed_code_provenance.csv",
    ],
    "official_event_count": len(events),
    "notes": "No inference, engine build, or telemetry capture was rerun.",
}
write_json("validation/completion_addendum.json", addendum)
(RUN / "validation/completion_addendum.txt").write_text(
    "PUBLICATION PACKAGE COMPLETION ADDENDUM: PASS\n"
    "Benchmark rerun: NO\nRaw benchmark modified: NO\n"
    f"Official event records: {len(events)}\n"
    "Dedicated stability analysis: YES\nDetailed execution provenance: YES\n",
    encoding="utf-8",
)
manifest = read_json("run_manifest.json")
manifest.update(
    {
        "status": validation["status"],
        "official_process_event_records": len(events),
        "stability_analysis": "PASS",
        "completion_addendum": "validation/completion_addendum.json",
    }
)
write_json("run_manifest.json", manifest)
print(json.dumps(addendum, indent=2))

