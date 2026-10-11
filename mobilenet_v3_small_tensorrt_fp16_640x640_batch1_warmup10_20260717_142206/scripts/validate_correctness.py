from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from common import FP32_REFERENCE, PACKAGED_ENGINE, RUN, load_image_order, preprocess, write_csv, write_json
from trt_runtime import TensorRTRunner


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else 1.0


def inspect_layers(runner: TensorRTRunner) -> tuple[str, list[dict], str]:
    try:
        inspector = runner.engine.create_engine_inspector()
        fmt = runner.trt.LayerInformationFormat.JSON
        try:
            raw = inspector.get_engine_information(fmt)
        except TypeError:
            raw = inspector.get_engine_information(runner.context, fmt)
        parsed = json.loads(raw)
        items = parsed if isinstance(parsed, list) else parsed.get("Layers", parsed.get("layers", []))
        rows = []
        for index, item in enumerate(items):
            text = json.dumps(item, sort_keys=True)
            rows.append({
                "layer_index": index,
                "layer_name": item.get("Name", item.get("name", f"layer_{index}")),
                "layer_type": item.get("LayerType", item.get("layer_type", "NA")),
                "precision": str(item.get("Precision", item.get("precision", "NA"))),
                "output_types": str(item.get("OutputTypes", item.get("output_types", "NA"))),
                "fp32_fallback": "YES" if "Float" in text or "FP32" in text else "NO_OR_NOT_EXPLICIT",
            })
        return raw, rows, "TensorRT EngineInspector JSON"
    except Exception as exc:
        return (
            json.dumps({"status": "NOT_INSPECTABLE", "error": repr(exc)}, indent=2),
            [{
                "layer_index": "NA", "layer_name": "NOT_INSPECTABLE", "layer_type": "NA",
                "precision": "NA", "output_types": "NA", "fp32_fallback": "UNKNOWN",
            }],
            f"TensorRT 8.0.1 EngineInspector unavailable: {exc!r}",
        )


def worker(engine_path: Path, output_path: Path, inspect: bool) -> None:
    samples = load_image_order()
    indices = sorted({round(i * (len(samples) - 1) / 19) for i in range(20)})
    with TensorRTRunner(engine_path) as runner:
        if tuple(runner.input_shape) != (1, 3, 640, 640):
            raise RuntimeError(f"Engine shape mismatch: {runner.binding_info}")
        if not all("FLOAT" in item["dtype"].upper() for item in runner.binding_info):
            raise RuntimeError(f"Expected FP32 host bindings: {runner.binding_info}")
        raw, fallback_rows, inspector_note = inspect_layers(runner) if inspect else ("", [], "not requested")
        outputs = []
        for index in indices:
            value, _ = runner.infer(preprocess(samples[index]["path"]), timed=False)
            outputs.append({"sample_index": index, "logits": np.asarray(value, dtype=np.float32).reshape(-1).tolist()})
        payload = {
            "engine_path": str(engine_path), "bindings": runner.binding_info, "cudart": runner.cudart_path,
            "outputs": outputs, "engine_inspector_raw": raw, "fallback_rows": fallback_rows,
            "inspector_note": inspector_note,
        }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"WORKER PASS: {engine_path}")


def run_worker(engine: Path, output: Path, inspect: bool, label: str) -> dict:
    command = [sys.executable, str(Path(__file__).resolve()), "--worker-engine", str(engine), "--worker-output", str(output)]
    if inspect:
        command.append("--inspect")
    environment = os.environ.copy(); environment["PYTHONNOUSERSITE"] = "1"
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment)
    (RUN / f"logs/correctness_{label}_worker_stdout.log").write_text(result.stdout, encoding="utf-8")
    (RUN / f"logs/correctness_{label}_worker_stderr.log").write_text(result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"{label} validation worker failed rc={result.returncode}: {result.stderr}")
    return json.loads(output.read_text())


def main() -> None:
    started = time.time()
    samples = load_image_order()
    indices = sorted({round(i * (len(samples) - 1) / 19) for i in range(20)})
    onnx_rows = {
        row["index"]: np.asarray(row["logits"], dtype=np.float32)
        for row in json.loads((RUN / "validation/onnx_validation_outputs.json").read_text())
    }
    fp32_engine = FP32_REFERENCE / "engine/mobilenet_v3_small_tensorrt_fp32_1x3x640x640.engine"
    fp32_payload = run_worker(fp32_engine, RUN / "validation/fp32_worker_outputs.json", False, "fp32")
    fp16_payload = run_worker(PACKAGED_ENGINE, RUN / "validation/fp16_worker_outputs.json", True, "fp16")
    fp32_outputs = {row["sample_index"]: np.asarray(row["logits"], dtype=np.float32) for row in fp32_payload["outputs"]}
    fp16_outputs = {row["sample_index"]: np.asarray(row["logits"], dtype=np.float32) for row in fp16_payload["outputs"]}

    (RUN / "engine/engine_inspector.txt").write_text(fp16_payload["engine_inspector_raw"] + "\n", encoding="utf-8")
    fallback_rows = fp16_payload["fallback_rows"]
    write_csv(RUN / "engine/fallback_layer_report.csv", fallback_rows)
    explicit_fp32_fallbacks = sum(row["fp32_fallback"] == "YES" for row in fallback_rows)
    write_json(RUN / "engine/engine_metadata.json", {
        "status": "PASS", "engine_path": str(PACKAGED_ENGINE), "bindings": fp16_payload["bindings"],
        "input_shape_verified": True, "input_output_fp32_bindings_verified": True,
        "fp16_builder_enabled": True, "int8_enabled": False, "tf32_enabled": False,
        "calibration_used": False, "dla_used": False,
        "execution_wording": "TensorRT FP16-enabled execution with FP32 input/output bindings",
        "layer_inspector_basis": fp16_payload["inspector_note"], "inspected_layer_rows": len(fallback_rows),
        "explicit_fp32_fallback_rows": explicit_fp32_fallbacks, "cudart": fp16_payload["cudart"],
    })
    (RUN / "engine/layer_precision_report.txt").write_text(
        "TensorRT FP16-enabled build configuration\nFP16 builder flag enabled: YES\n"
        "TF32 enabled: NO\nINT8 enabled: NO\nCalibration used: NO\nDLA enabled: NO\n"
        "Input/output bindings: FP32 (preserves FP32 host preprocessing protocol)\n"
        f"Layer inspection basis: {fp16_payload['inspector_note']}\n"
        f"Inspector rows: {len(fallback_rows)}\nExplicit FP32 fallback rows: {explicit_fp32_fallbacks}\n"
        "Do not interpret FP32 bindings as proof that internal execution is FP32.\n",
        encoding="utf-8",
    )

    rows, output_stats = [], []
    for index in indices:
        sample = samples[index]
        onnx_logits, fp32_logits, fp16_logits = onnx_rows[index], fp32_outputs[index], fp16_outputs[index]
        row = {
            "sample_index": index, "image_path": sample["path"], "true_class_id": sample["class_id"],
            "onnx_top1": int(np.argmax(onnx_logits)), "fp32_tensorrt_top1": int(np.argmax(fp32_logits)),
            "fp16_tensorrt_top1": int(np.argmax(fp16_logits)),
            "fp32_fp16_top1_agree": int(np.argmax(fp32_logits) == np.argmax(fp16_logits)),
            "onnx_fp16_top1_agree": int(np.argmax(onnx_logits) == np.argmax(fp16_logits)),
            "fp32_fp16_max_abs_error": float(np.max(np.abs(fp32_logits - fp16_logits))),
            "fp32_fp16_mean_abs_error": float(np.mean(np.abs(fp32_logits - fp16_logits))),
            "onnx_fp16_max_abs_error": float(np.max(np.abs(onnx_logits - fp16_logits))),
            "onnx_fp16_mean_abs_error": float(np.mean(np.abs(onnx_logits - fp16_logits))),
            "fp32_fp16_cosine_similarity": cosine(fp32_logits, fp16_logits),
            "onnx_fp16_cosine_similarity": cosine(onnx_logits, fp16_logits),
            "fp16_output_finite": int(np.isfinite(fp16_logits).all()),
        }
        rows.append(row)
        for backend, values in (("ONNX", onnx_logits), ("TensorRT FP32", fp32_logits), ("TensorRT FP16-enabled", fp16_logits)):
            output_stats.append({
                "sample_index": index, "backend": backend, "minimum": float(values.min()),
                "maximum": float(values.max()), "mean": float(values.mean()), "std": float(values.std()),
                "finite": int(np.isfinite(values).all()), "top1": int(np.argmax(values)),
            })
    fp32_agreement = sum(row["fp32_fp16_top1_agree"] for row in rows) / len(rows)
    onnx_agreement = sum(row["onnx_fp16_top1_agree"] for row in rows) / len(rows)
    minimum_cosine = min(row["fp32_fp16_cosine_similarity"] for row in rows)
    all_finite = all(row["fp16_output_finite"] for row in rows)
    unique_predictions = len({row["fp16_tensorrt_top1"] for row in rows})
    status = "PASS" if all_finite and fp32_agreement == 1.0 and onnx_agreement == 1.0 and minimum_cosine >= 0.999 and unique_predictions >= 2 else "FAIL"
    summary = {
        "status": status, "sample_count": len(rows),
        "sampling": "20 evenly spaced deterministic indices over the sorted 261-image list",
        "all_outputs_finite": all_finite, "fp32_fp16_top1_agreement": fp32_agreement,
        "onnx_fp16_top1_agreement": onnx_agreement, "minimum_fp32_fp16_cosine_similarity": minimum_cosine,
        "maximum_fp32_fp16_absolute_error": max(row["fp32_fp16_max_abs_error"] for row in rows),
        "mean_fp32_fp16_absolute_error": float(np.mean([row["fp32_fp16_mean_abs_error"] for row in rows])),
        "unique_fp16_predictions": unique_predictions, "duration_seconds": time.time() - started,
        "validation_process_model": "one fresh subprocess per TensorRT engine; aggregation occurs after both contexts exit",
    }
    write_csv(RUN / "validation/onnx_fp16_tensorrt_output_comparison.csv", rows)
    write_csv(RUN / "validation/fp32_fp16_output_comparison.csv", rows)
    write_csv(RUN / "validation/output_statistics.csv", output_stats)
    write_json(RUN / "validation/correctness_validation_summary.json", summary)
    write_json(RUN / "validation/framework_output_summary.json", summary)
    (RUN / "validation/correctness_validation_report.md").write_text(
        "# FP16 Correctness Validation\n\n" f"- Status: **{status}**\n- Images: {len(rows)} deterministic samples\n"
        f"- FP32/FP16 top-1 agreement: {fp32_agreement:.6f}\n- ONNX/FP16 top-1 agreement: {onnx_agreement:.6f}\n"
        f"- Minimum FP32/FP16 cosine similarity: {minimum_cosine:.9f}\n"
        f"- Maximum FP32/FP16 absolute logit error: {summary['maximum_fp32_fp16_absolute_error']:.9g}\n"
        "- Each TensorRT engine ran in a separate short-lived subprocess; no CUDA context was reused.\n",
        encoding="utf-8",
    )
    if status != "PASS":
        raise RuntimeError(f"Correctness validation failed: {summary}")
    print("FP16 CORRECTNESS VALIDATION PASS")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-engine", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()
    if args.worker_engine:
        if args.worker_output is None:
            parser.error("--worker-output is required with --worker-engine")
        worker(args.worker_engine, args.worker_output, args.inspect)
    else:
        main()
