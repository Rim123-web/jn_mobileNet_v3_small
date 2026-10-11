#!/usr/bin/env python3
import csv
import json
import statistics
import subprocess
from pathlib import Path

from common import RUN, write_json


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


idle_path = RUN / "monitoring/idle_samples.csv"
status_path = RUN / "monitoring/idle_capture_status.json"
rows = list(csv.DictReader(idle_path.open(newline="", encoding="utf-8")))
status = json.loads(status_path.read_text(encoding="utf-8"))

power_field = next(
    (name for name in ("power_tot_w", "power_total_w", "total_power_w") if rows and name in rows[0]),
    None,
)
power_values = [as_float(row.get(power_field)) for row in rows] if power_field else []
power_values = [value for value in power_values if value is not None]

idle_summary = {
    "status": "PASS" if status.get("status") == "PASS" and len(rows) >= 60 else "FAIL",
    "duration_seconds": status.get("duration_seconds"),
    "sample_count": len(rows),
    "power_field": power_field,
    "mean_power_w": statistics.mean(power_values) if power_values else None,
    "median_power_w": statistics.median(power_values) if power_values else None,
    "min_power_w": min(power_values) if power_values else None,
    "max_power_w": max(power_values) if power_values else None,
    "note": "Power statistics are finalized from the raw idle CSV by finalize_results.py.",
}
write_json(RUN / "monitoring/idle_summary.json", idle_summary)
write_json(
    RUN / "monitoring/idle_capture_validation.json",
    {
        "status": idle_summary["status"],
        "idle_samples_exists": idle_path.exists(),
        "idle_status_exists": status_path.exists(),
        "minimum_30_seconds": float(status.get("duration_seconds", 0)) >= 30.0,
        "minimum_60_samples": len(rows) >= 60,
        "nonempty_columns": bool(rows and rows[0]),
    },
)

result = subprocess.run(
    ["ps", "-eo", "pid=,ppid=,etimes=,%cpu=,%mem=,args="],
    check=True,
    text=True,
    stdout=subprocess.PIPE,
)
process_rows = []
for line in result.stdout.splitlines():
    parts = line.strip().split(None, 5)
    if len(parts) == 6:
        process_rows.append(dict(zip(("pid", "ppid", "elapsed_s", "cpu_percent", "memory_percent", "command"), parts)))

process_path = RUN / "environment/processes_before_official_run.csv"
with process_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=("pid", "ppid", "elapsed_s", "cpu_percent", "memory_percent", "command"))
    writer.writeheader()
    writer.writerows(process_rows)

if idle_summary["status"] != "PASS":
    raise SystemExit("IDLE PREBENCHMARK EVIDENCE FAIL")
print("IDLE PREBENCHMARK EVIDENCE PASS")
