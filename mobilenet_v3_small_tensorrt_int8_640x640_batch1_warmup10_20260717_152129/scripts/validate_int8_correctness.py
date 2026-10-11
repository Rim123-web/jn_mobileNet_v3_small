from __future__ import annotations

import csv
import json
import math
import time

import numpy as np

from common import (
    FP16_REFERENCE,
    FP32_REFERENCE,
    PACKAGED_ENGINE,
    RUN,
    load_image_order,
    preprocess,
    read_json,
    sha256,
    write_csv,
    write_json,
)
from trt_runtime import TensorRTRunner


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def cosine_similarity(left, right):
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 1.0


def main():
    start = time.time()
    build = read_json(RUN / "engine/int8_build_summary.json")
    if build["status"] != "PASS":
        raise RuntimeError("INT8 build summary is not PASS")
    if build["historical_test_fallback_cache_used"]:
        raise RuntimeError("Historical test-fallback cache was used")
    if not build["fresh_cache"] or not build["int8_builder_flag"]:
        raise RuntimeError("Fresh INT8 calibration evidence is incomplete")

    samples = load_image_order()
    indices = sorted({round(index * (len(samples) - 1) / 19) for index in range(20)})
    selected = [samples[index] for index in indices]
    onnx_outputs = {
        int(row["index"]): np.asarray(row["logits"], dtype=np.float32).reshape(-1)
        for row in read_json(RUN / "validation/onnx_validation_outputs.json")
    }
    fp32_predictions = {
        row["relative_path"]: row
        for row in read_csv(FP32_REFERENCE / "raw/per_image_predictions.csv")
    }
    fp16_predictions = {
        row["relative_path"]: row
        for row in read_csv(FP16_REFERENCE / "raw/per_image_predictions.csv")
    }

    rows = []
    observed_classes = set()
    with TensorRTRunner(PACKAGED_ENGINE) as runner:
        bindings = runner.binding_info
        if tuple(runner.input_shape) != (1, 3, 640, 640):
            raise RuntimeError(f"INT8 engine input shape mismatch: {bindings}")
        for position, dataset_index in enumerate(indices):
            sample = selected[position]
            tensor = preprocess(sample["path"])
            output, _ = runner.infer(tensor, timed=False)
            logits = np.asarray(output, dtype=np.float32).reshape(-1)
            if logits.shape != (3,):
                raise RuntimeError(f"Unexpected output shape: {logits.shape}")
            if not np.isfinite(logits).all():
                raise RuntimeError(f"NaN/Inf output at {sample['path']}")
            int8_top1 = int(np.argmax(logits))
            observed_classes.add(int8_top1)
            fp32 = fp32_predictions[sample["relative_path"]]
            fp16 = fp16_predictions[sample["relative_path"]]
            onnx_logits = onnx_outputs[dataset_index]
            rows.append(
                {
                    "sample_index": dataset_index,
                    "image_path": sample["path"],
                    "relative_path": sample["relative_path"],
                    "true_class_id": sample["class_id"],
                    "onnx_top1": int(np.argmax(onnx_logits)),
                    "fp32_top1": int(fp32["predicted_class_id"]),
                    "fp16_top1": int(fp16["predicted_class_id"]),
                    "int8_top1": int8_top1,
                    "int8_matches_onnx": int(int8_top1 == int(np.argmax(onnx_logits))),
                    "int8_matches_fp32": int(
                        int8_top1 == int(fp32["predicted_class_id"])
                    ),
                    "int8_matches_fp16": int(
                        int8_top1 == int(fp16["predicted_class_id"])
                    ),
                    "int8_correct": int(int8_top1 == sample["class_id"]),
                    "onnx_int8_max_abs_error": float(
                        np.max(np.abs(onnx_logits - logits))
                    ),
                    "onnx_int8_mean_abs_error": float(
                        np.mean(np.abs(onnx_logits - logits))
                    ),
                    "onnx_int8_cosine_similarity": cosine_similarity(
                        onnx_logits, logits
                    ),
                    "int8_logits": json.dumps(logits.tolist()),
                }
            )

    agreement_fp32 = float(np.mean([row["int8_matches_fp32"] for row in rows]))
    agreement_fp16 = float(np.mean([row["int8_matches_fp16"] for row in rows]))
    agreement_onnx = float(np.mean([row["int8_matches_onnx"] for row in rows]))
    sample_accuracy = float(np.mean([row["int8_correct"] for row in rows]))
    finite = all(
        math.isfinite(row["onnx_int8_max_abs_error"])
        and math.isfinite(row["onnx_int8_cosine_similarity"])
        for row in rows
    )
    collapse = len(observed_classes) < 2
    status = (
        "PASS"
        if finite
        and not collapse
        and agreement_fp32 >= 0.90
        and agreement_fp16 >= 0.90
        and sample_accuracy >= 0.80
        else "FAIL"
    )
    metadata = {
        "status": status,
        "engine_path": str(PACKAGED_ENGINE),
        "engine_sha256": sha256(PACKAGED_ENGINE),
        "engine_size_bytes": PACKAGED_ENGINE.stat().st_size,
        "bindings": bindings,
        "input_shape_verified": True,
        "batch_size": 1,
        "builder_int8_enabled": build["int8_builder_flag"],
        "tf32_disabled": build["tf32_disabled"],
        "calibration_used": True,
        "calibration_algorithm": build["calibration_algorithm"],
        "calibration_cache_path": build["calibration_cache_path"],
        "calibration_cache_sha256": build["calibration_cache_sha256"],
        "historical_test_fallback_cache_used": False,
        "input_binding_dtype": next(
            item["dtype"] for item in bindings if item["is_input"]
        ),
        "output_binding_dtypes": [
            item["dtype"] for item in bindings if not item["is_input"]
        ],
        "internal_execution_precision": "TensorRT INT8-enabled execution with calibrated internal layers and permitted higher-precision fallbacks",
        "fp16_fallback_status": (
            "Possible where selected by TensorRT; see layer_precision_report.txt."
        ),
        "fp32_fallback_status": (
            "Possible for unsupported or accuracy-sensitive layers and binding reformats; see layer_precision_report.txt."
        ),
        "layer_precision_summary": build["layer_precision_summary"],
    }
    write_json(RUN / "engine/engine_metadata.json", metadata)
    (RUN / "engine/engine_inspector.txt").write_text(
        json.dumps(bindings, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(RUN / "validation/source_onnx_int8_comparison.csv", rows)
    write_csv(RUN / "validation/fp32_fp16_int8_output_comparison.csv", rows)
    write_csv(RUN / "validation/onnx_fp32_fp16_int8_output_comparison.csv", rows)
    summary = {
        "status": status,
        "sample_count": len(rows),
        "sampling": "20 evenly spaced deterministic indices over the sorted 261-image list",
        "output_shape": [1, 3],
        "outputs_finite": finite,
        "observed_predicted_classes": sorted(observed_classes),
        "prediction_collapse": collapse,
        "sample_accuracy": sample_accuracy,
        "int8_onnx_top1_agreement": agreement_onnx,
        "int8_fp32_top1_agreement": agreement_fp32,
        "int8_fp16_top1_agreement": agreement_fp16,
        "minimum_onnx_int8_cosine_similarity": min(
            row["onnx_int8_cosine_similarity"] for row in rows
        ),
        "maximum_onnx_int8_abs_error": max(
            row["onnx_int8_max_abs_error"] for row in rows
        ),
        "duration_seconds": time.time() - start,
        "validation_process_releases_context_on_exit": True,
    }
    write_json(RUN / "validation/correctness_validation_summary.json", summary)
    write_csv(
        RUN / "validation/int8_output_statistics.csv",
        [
            {
                "sample_count": len(rows),
                "outputs_finite": finite,
                "observed_class_count": len(observed_classes),
                "prediction_collapse": collapse,
                "sample_accuracy": sample_accuracy,
                "onnx_agreement": agreement_onnx,
                "fp32_agreement": agreement_fp32,
                "fp16_agreement": agreement_fp16,
                "minimum_cosine_similarity": summary[
                    "minimum_onnx_int8_cosine_similarity"
                ],
                "maximum_abs_error": summary["maximum_onnx_int8_abs_error"],
            }
        ],
    )
    (RUN / "validation/correctness_validation_report.md").write_text(
        "# INT8 Correctness Validation\n\n"
        f"- Status: **{status}**\n"
        f"- Samples: {len(rows)}\n"
        f"- Output finite: {finite}\n"
        f"- Output shape: `[1,3]`\n"
        f"- Observed prediction classes: {sorted(observed_classes)}\n"
        f"- Prediction collapse: {collapse}\n"
        f"- Sample accuracy: {sample_accuracy:.3%}\n"
        f"- Agreement with ONNX: {agreement_onnx:.3%}\n"
        f"- Agreement with FP32: {agreement_fp32:.3%}\n"
        f"- Agreement with FP16: {agreement_fp16:.3%}\n"
        "- This smoke/correctness pass is separate from the official measured process.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    if status != "PASS":
        raise RuntimeError(f"INT8 correctness validation failed: {summary}")
    print("INT8 CORRECTNESS VALIDATION PASS")


if __name__ == "__main__":
    main()

