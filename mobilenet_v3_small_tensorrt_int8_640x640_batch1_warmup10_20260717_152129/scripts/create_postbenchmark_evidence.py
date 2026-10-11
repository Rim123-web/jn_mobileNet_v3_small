#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import statistics

from common import RUN, write_csv, write_json


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


provenance = json.loads(
    (RUN / "raw/execution_count_provenance.json").read_text(encoding="utf-8")
)
with (RUN / "raw/per_image_timings.csv").open(newline="", encoding="utf-8") as handle:
    timings = list(csv.DictReader(handle))
with (RUN / "raw/warmup_execution_records.csv").open(newline="", encoding="utf-8") as handle:
    warmups = list(csv.DictReader(handle))

events = [{
    "event": "warmup_phase_summary",
    "timestamp_epoch": None,
    "execution_count": len(warmups),
    "excluded_from_official_timing": True,
    "excluded_from_monitoring": True,
    "timestamp_note": "Warm-up wall-clock timestamps were not recorded; no values were invented.",
}]
for row in warmups:
    events.append({
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
    })
events.append({
    "event": "official_measurement_start",
    "timestamp_epoch": provenance["official_start_epoch"],
    "measured_executions_expected": provenance["expected_measured_executions"],
    "telemetry_started": True,
})
for row in timings:
    events.extend((
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
    ))
events.append({
    "event": "official_measurement_end",
    "timestamp_epoch": provenance["official_end_epoch"],
    "measured_executions": provenance["measured_executions"],
    "failed_images": provenance["failed_images"],
    "discarded_measured_rows": provenance["discarded_measured_rows"],
    "telemetry_stopped": True,
})
(RUN / "raw/official_process_events.jsonl").write_text(
    "".join(json.dumps(row) + "\n" for row in events), encoding="utf-8"
)

quartiles = []
for index in range(4):
    start = len(timings) * index // 4
    end = len(timings) * (index + 1) // 4
    subset = timings[start:end]
    engine = [float(row["inference_ms"]) for row in subset]
    pipeline = [float(row["end_to_end_ms"]) for row in subset]
    quartiles.append({
        "quartile": index + 1,
        "first_measured_index": int(subset[0]["measured_index"]),
        "last_measured_index": int(subset[-1]["measured_index"]),
        "count": len(subset),
        "mean_engine_latency_ms": statistics.fmean(engine),
        "median_engine_latency_ms": statistics.median(engine),
        "p95_engine_latency_ms": percentile(engine, 0.95),
        "mean_end_to_end_latency_ms": statistics.fmean(pipeline),
        "median_end_to_end_latency_ms": statistics.median(pipeline),
        "p95_end_to_end_latency_ms": percentile(pipeline, 0.95),
    })
write_csv(RUN / "stability/latency_stability_quartiles.csv", quartiles)
stability = {
    "status": "PASS",
    "source": "raw/per_image_timings.csv from the single official measured pass",
    "measured_rows": len(timings),
    "official_event_count": len(events),
    "quartiles": quartiles,
    "first_to_last_engine_mean_change_percent": 100 * (
        quartiles[-1]["mean_engine_latency_ms"]
        / quartiles[0]["mean_engine_latency_ms"] - 1
    ),
    "interpretation": "Descriptive run-order stability from one full dataset pass.",
}
write_json(RUN / "stability/latency_stability_analysis.json", stability)
(RUN / "stability/latency_stability_analysis.md").write_text(
    "# Latency Stability\n\n"
    f"- Official measured rows: {len(timings)}\n"
    f"- Official event records: {len(events)}\n"
    f"- First-to-last engine mean change: {stability['first_to_last_engine_mean_change_percent']:.6f}%\n"
    "- This is descriptive run-order stability, not a repeated-run confidence interval.\n",
    encoding="utf-8",
)
expected = 2 * len(timings) + len(warmups) + 3
if len(events) != expected:
    raise SystemExit(f"Official event count mismatch: {len(events)} != {expected}")
print(f"POSTBENCHMARK EVIDENCE PASS: events={len(events)}")
