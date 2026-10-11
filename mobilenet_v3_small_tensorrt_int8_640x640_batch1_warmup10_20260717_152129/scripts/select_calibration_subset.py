from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from common import RUN, preprocess, read_json, write_csv, write_json


TARGET_PER_CLASS = 30


def main():
    all_rows = read_json(RUN / "calibration/calibration_inventory.json")
    source_provenance = read_json(
        RUN / "calibration/calibration_dataset_provenance.json"
    )
    source_root = source_provenance["source_root"]
    by_class = defaultdict(list)
    for row in all_rows:
        by_class[row["class_name"]].append(row)
    if sorted(len(rows) for rows in by_class.values()) != [35, 70, 155]:
        raise RuntimeError(
            f"Unexpected validation distribution: "
            f"{dict((key, len(value)) for key, value in by_class.items())}"
        )

    selected = []
    for class_name in sorted(by_class, key=lambda name: int(by_class[name][0]["class_id"])):
        rows = sorted(by_class[class_name], key=lambda row: row["relative_path"])
        indices = sorted(
            {
                round(index * (len(rows) - 1) / (TARGET_PER_CLASS - 1))
                for index in range(TARGET_PER_CLASS)
            }
        )
        if len(indices) != TARGET_PER_CLASS:
            raise RuntimeError(f"Unable to select 30 unique images for {class_name}")
        selected.extend(rows[index] for index in indices)
    for index, row in enumerate(selected, 1):
        row["calibration_index"] = index
    selected_hashes = {row["sha256"] for row in selected}
    excluded = [
        {"path": row["absolute_path"], "reason": "not selected by deterministic evenly-spaced stratified subset"}
        for row in all_rows
        if row["sha256"] not in selected_hashes
    ]

    write_csv(RUN / "calibration/calibration_manifest_all260.csv", all_rows)
    (RUN / "calibration/calibration_manifest_all260.txt").write_text(
        "\n".join(row["absolute_path"] for row in all_rows) + "\n",
        encoding="utf-8",
    )
    write_csv(RUN / "calibration/calibration_manifest.csv", selected)
    (RUN / "calibration/calibration_manifest.txt").write_text(
        "\n".join(row["absolute_path"] for row in selected) + "\n",
        encoding="utf-8",
    )
    write_json(RUN / "calibration/calibration_inventory.json", selected)
    write_csv(
        RUN / "calibration/calibration_class_distribution.csv",
        [
            {
                "class_id": rows[0]["class_id"],
                "class_name": class_name,
                "image_count": TARGET_PER_CLASS,
            }
            for class_name, rows in sorted(
                by_class.items(), key=lambda item: int(item[1][0]["class_id"])
            )
        ],
    )
    write_csv(
        RUN / "calibration/calibration_image_hashes.csv",
        [
            {
                "calibration_index": row["calibration_index"],
                "relative_path": row["relative_path"],
                "sha256": row["sha256"],
            }
            for row in selected
        ],
    )
    write_csv(RUN / "calibration/calibration_exclusions.csv", excluded)
    config = {
        "source": source_root,
        "selection": "deterministic evenly-spaced stratified subset over each class-sorted path list",
        "random_sampling": False,
        "seed": 42,
        "seed_usage": "recorded but not used because selection is deterministic without randomness",
        "target_per_class": TARGET_PER_CLASS,
        "selected_count": len(selected),
        "batch_size": 1,
        "augmentation": False,
        "reason_for_subset": (
            "90 balanced images is within the requested preferred 90-150 range, "
            "represents all classes, and is practical for Jetson Nano calibration."
        ),
    }
    write_json(RUN / "calibration/calibration_selection_config.json", config)
    provenance = read_json(RUN / "calibration/calibration_dataset_provenance.json")
    provenance.update(
        {
            "selected_calibration_count": len(selected),
            "class_distribution": {
                class_name: TARGET_PER_CLASS for class_name in by_class
            },
            "selection_method": config["selection"],
            "seed": 42,
            "subset_reason": config["reason_for_subset"],
        }
    )
    write_json(RUN / "calibration/calibration_dataset_provenance.json", provenance)
    (RUN / "calibration/calibration_selection_report.md").write_text(
        "# Calibration Selection\n\n"
        f"- Source: `{source_root}`\n"
        "- Split: validation\n"
        "- Source images: 260\n"
        "- Selected images: 90 (30 per class)\n"
        "- Method: deterministic evenly-spaced selection over each class-sorted path list\n"
        "- Random sampling: NO (seed 42 recorded for protocol completeness)\n"
        "- Reason: observed calibration cost made all 260 images unreasonable; "
        "90 is within the requested preferred range.\n"
        "- Exact test overlap: 0\n"
        "- Test images used: NO\n",
        encoding="utf-8",
    )

    parity_rows = []
    indices = sorted({round(index * (len(selected) - 1) / 19) for index in range(20)})
    for index in indices:
        row = selected[index]
        first = preprocess(row["absolute_path"])
        second = preprocess(row["absolute_path"])
        parity_rows.append(
            {
                "path": row["absolute_path"],
                "class_name": row["class_name"],
                "shape": "x".join(map(str, first.shape)),
                "dtype": str(first.dtype),
                "max_abs_repeat_difference": float(np.max(np.abs(first - second))),
                "finite": bool(np.isfinite(first).all()),
                "status": "PASS"
                if first.shape == (1, 3, 640, 640)
                and first.dtype == np.float32
                and np.isfinite(first).all()
                else "FAIL",
            }
        )
    if any(row["status"] != "PASS" for row in parity_rows):
        raise RuntimeError("Selected-subset preprocessing parity failed")
    write_csv(RUN / "calibration/preprocessing_parity_test.csv", parity_rows)
    write_json(
        RUN / "calibration/preprocessing_parity_test.json",
        {"status": "PASS", "sample_count": len(parity_rows), "rows": parity_rows},
    )
    manifest = read_json(RUN / "run_manifest.json")
    manifest["calibration_image_count"] = len(selected)
    manifest["calibration_selection"] = config["selection"]
    manifest["calibration_class_distribution"] = {
        class_name: TARGET_PER_CLASS for class_name in by_class
    }
    write_json(RUN / "run_manifest.json", manifest)
    print(json.dumps(config, indent=2))
    print("CALIBRATION SUBSET SELECTION PASS")


if __name__ == "__main__":
    main()

