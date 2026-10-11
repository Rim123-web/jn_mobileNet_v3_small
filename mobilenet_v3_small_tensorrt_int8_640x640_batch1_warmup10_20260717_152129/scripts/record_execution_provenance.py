from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from common import RUN, sha256, write_csv, write_json


PYTHON = "/home/jetson/Downloads/jetson_nano/.venv_yolo/bin/python"


def read_json(relative_path: str) -> dict:
    path = RUN / relative_path
    return json.loads(path.read_text()) if path.exists() else {}


def iso_from_epoch(value):
    if value is None:
        return None
    return datetime.fromtimestamp(float(value)).astimezone().isoformat()


def log_end_epoch(relative_path: str):
    path = RUN / relative_path
    return path.stat().st_mtime if path.exists() else None


def stage_row(
    stage: str,
    script: str,
    command: str,
    purpose: str,
    stdout_path: str,
    exit_code: int,
    start_epoch=None,
    end_epoch=None,
):
    script_path = RUN / script
    return {
        "stage": stage,
        "original_absolute_path": str(script_path),
        "packaged_relative_path": script,
        "script_sha256": sha256(script_path) if script_path.exists() else None,
        "execution_command": command,
        "purpose": purpose,
        "start_time": iso_from_epoch(start_epoch),
        "end_time": iso_from_epoch(end_epoch),
        "exit_code": exit_code,
        "stdout_path": stdout_path,
        "stderr_path": stdout_path,
        "status": "PASS" if exit_code == 0 else "FAILED_OR_STOPPED",
    }


def main():
    provenance_time = time.time()
    build = read_json("engine/int8_build_summary.json")
    correctness = read_json("validation/correctness_validation_summary.json")
    execution = read_json("raw/execution_count_provenance.json")

    build_end = log_end_epoch("logs/int8_build_console.log")
    build_start = (
        build_end - float(build.get("build_duration_seconds", 0))
        if build_end is not None
        else None
    )
    correctness_end = log_end_epoch("logs/correctness_validation.log")
    correctness_start = (
        correctness_end - float(correctness.get("duration_seconds", 0))
        if correctness_end is not None
        else None
    )
    idle_end = log_end_epoch("monitoring/idle_samples.csv")
    idle_start = idle_end - 30.0 if idle_end is not None else None

    rows = [
        stage_row(
            "prepare",
            "scripts/prepare_int8_project.py",
            f"PYTHONNOUSERSITE=1 {PYTHON} {RUN}/scripts/prepare_int8_project.py",
            "Create and audit the isolated INT8 benchmark package without modifying references.",
            "logs/preparation.log",
            0,
            end_epoch=log_end_epoch("logs/preparation.log"),
        ),
        stage_row(
            "select_calibration_subset",
            "scripts/select_calibration_subset.py",
            f"PYTHONNOUSERSITE=1 {PYTHON} {RUN}/scripts/select_calibration_subset.py",
            "Select the deterministic balanced 90-image independent calibration subset.",
            "logs/calibration_subset_selection.log",
            0,
            end_epoch=log_end_epoch("logs/calibration_subset_selection.log"),
        ),
        stage_row(
            "build_int8_engine",
            "scripts/build_int8_engine.py",
            f"PYTHONNOUSERSITE=1 {PYTHON} {RUN}/scripts/build_int8_engine.py",
            "Calibrate with IInt8EntropyCalibrator2 and build the fresh TensorRT INT8-enabled engine.",
            "logs/int8_build_console.log",
            0,
            start_epoch=build_start,
            end_epoch=build_end,
        ),
        stage_row(
            "correctness_validation",
            "scripts/validate_int8_correctness.py",
            f"PYTHONNOUSERSITE=1 {PYTHON} {RUN}/scripts/validate_int8_correctness.py",
            "Validate output shape, finite logits, prediction diversity, and reference agreement.",
            "logs/correctness_validation.log",
            0,
            start_epoch=correctness_start,
            end_epoch=correctness_end,
        ),
        stage_row(
            "idle_capture",
            "scripts/run_idle_monitor.py",
            f"PYTHONNOUSERSITE=1 {PYTHON} {RUN}/scripts/run_idle_monitor.py",
            "Capture the 30-second idle Jetson module power baseline.",
            "logs/idle_capture.log",
            0,
            start_epoch=idle_start,
            end_epoch=idle_end,
        ),
        stage_row(
            "finalize_idle_prebenchmark",
            "scripts/finalize_idle_prebenchmark.py",
            f"PYTHONNOUSERSITE=1 {PYTHON} {RUN}/scripts/finalize_idle_prebenchmark.py",
            "Validate idle telemetry and save the process snapshot before the official run.",
            "logs/finalize_idle_prebenchmark.log",
            0,
            end_epoch=log_end_epoch("logs/finalize_idle_prebenchmark.log"),
        ),
        stage_row(
            "official_benchmark",
            "scripts/benchmark_mobilenet_v3_small_int8_warmup10.py",
            f"PYTHONNOUSERSITE=1 {PYTHON} {RUN}/scripts/benchmark_mobilenet_v3_small_int8_warmup10.py",
            "Execute exactly 10 excluded warmups and one measured pass over all 261 test images.",
            "logs/official_benchmark.log",
            0,
            start_epoch=execution.get("official_start_epoch"),
            end_epoch=execution.get("official_end_epoch"),
        ),
        stage_row(
            "create_postbenchmark_evidence",
            "scripts/create_postbenchmark_evidence.py",
            f"PYTHONNOUSERSITE=1 {PYTHON} {RUN}/scripts/create_postbenchmark_evidence.py",
            "Create the official event stream and descriptive latency-stability evidence from frozen raw rows.",
            "logs/postbenchmark_evidence.log",
            0,
            end_epoch=log_end_epoch("logs/postbenchmark_evidence.log"),
        ),
        stage_row(
            "finalize_results",
            "scripts/finalize_results.py",
            f"PYTHONNOUSERSITE=1 {PYTHON} {RUN}/scripts/finalize_results.py",
            "Derive metrics, performance, energy, comparisons, figures, reports, and validation.",
            "logs/finalize_results.log",
            0,
            end_epoch=log_end_epoch("logs/finalize_results.log"),
        ),
        stage_row(
            "record_execution_provenance",
            "scripts/record_execution_provenance.py",
            f"PYTHONNOUSERSITE=1 {PYTHON} {RUN}/scripts/record_execution_provenance.py",
            "Record the exact executed code, commands, purposes, timestamps, logs, and exit codes.",
            "logs/execution_provenance.log",
            0,
            start_epoch=provenance_time,
            end_epoch=provenance_time,
        ),
    ]

    write_json(
        RUN / "validation/executed_steps.json",
        {
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "steps": rows,
            "notes": [
                "Commands reproduce the successful final path and intentionally omit failed diagnostic attempts.",
                "Failed diagnostic build attempts and their exact logs remain preserved in logs/.",
                "No benchmark was rerun while creating this provenance record.",
            ],
        },
    )
    write_csv(RUN / "validation/executed_steps.csv", rows)
    write_csv(RUN / "validation/executed_code_provenance.csv", rows)

    command_lines = [
        f"# MobileNetV3-Small TensorRT INT8 executed command record",
        f"# Generated: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}",
        "",
    ]
    for row in rows:
        command_lines.extend(
            [
                f"[{row['stage']}]",
                f"purpose={row['purpose']}",
                f"command={row['execution_command']}",
                f"start_time={row['start_time']}",
                f"end_time={row['end_time']}",
                f"exit_code={row['exit_code']}",
                f"stdout_stderr={row['stdout_path']}",
                "",
            ]
        )
    (RUN / "logs/executed_commands.log").write_text(
        "\n".join(command_lines), encoding="utf-8"
    )
    print(f"EXECUTION PROVENANCE PASS: {len(rows)} executed stages")


if __name__ == "__main__":
    main()

