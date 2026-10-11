from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


RUN = Path(__file__).resolve().parents[1]
required = [
    "raw/official_process_events.jsonl",
    "stability/latency_stability_analysis.json",
    "stability/latency_stability_quartiles.csv",
    "stability/latency_stability_analysis.md",
    "validation/completion_addendum.json",
    "validation/executed_code_provenance.csv",
    "validation/file_hash_manifest.csv",
]
missing = [name for name in required if not (RUN / name).is_file()]
if missing:
    raise SystemExit("MISSING COMPLETION FILES: " + ", ".join(missing))

report = json.loads((RUN / "validation/final_validation_report.json").read_text())
if report.get("status") != "PASS" or not all(report.get("checks", {}).values()):
    raise SystemExit("FINAL VALIDATION REPORT IS NOT PASS")
events = [json.loads(line) for line in (RUN / "raw/official_process_events.jsonl").read_text().splitlines()]
if len(events) != 535:
    raise SystemExit(f"OFFICIAL EVENT COUNT INVALID: {len(events)}")
with (RUN / "validation/executed_code_provenance.csv").open(newline="", encoding="utf-8") as handle:
    provenance = list(csv.DictReader(handle))
required_columns = {
    "original_path", "packaged_path", "sha256", "execution_command", "purpose",
    "start_time", "end_time", "exit_code", "stdout_path", "stderr_path",
}
if not provenance or not required_columns.issubset(provenance[0]):
    raise SystemExit("EXECUTION PROVENANCE SCHEMA INVALID")
script_count = len([path for path in (RUN / "scripts").iterdir() if path.is_file()])
if len(provenance) != script_count:
    raise SystemExit(f"EXECUTION PROVENANCE ROW COUNT INVALID: {len(provenance)} != {script_count}")

with (RUN / "validation/file_hash_manifest.csv").open(newline="", encoding="utf-8") as handle:
    manifest = list(csv.DictReader(handle))
failures = []
for row in manifest:
    path = RUN / row["relative_path"]
    if not path.is_file():
        failures.append("missing:" + row["relative_path"])
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != row["sha256"]:
        failures.append("hash:" + row["relative_path"])
if failures:
    raise SystemExit("HASH VALIDATION FAILED: " + ", ".join(failures[:20]))
print(
    f"COMPLETE PACKAGE VALIDATION PASS: events={len(events)} "
    f"provenance_rows={len(provenance)} manifest_rows={len(manifest)}"
)

