from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from common import PACKAGED_ENGINE, PACKAGED_MODEL, RUN, load_image_order, load_model, preprocess, write_csv, write_json
from trt_runtime import TensorRTRunner


def main():
    start = time.time()
    samples = load_image_order()
    indices = sorted({round(i * (len(samples) - 1) / 19) for i in range(20)})
    onnx_rows = {
        row["index"]: row
        for row in json.loads((RUN / "validation/onnx_validation_outputs.json").read_text())
    }
    model, names, _ = load_model(PACKAGED_MODEL)
    rows = []
    with torch.inference_mode(), TensorRTRunner(PACKAGED_ENGINE) as runner:
        if tuple(runner.input_shape) != (1, 3, 640, 640):
            raise RuntimeError(f"Engine shape mismatch: {runner.binding_info}")
        if not all("FLOAT" in item["dtype"].upper() for item in runner.binding_info):
            raise RuntimeError(f"Engine binding dtype is not FP32: {runner.binding_info}")
        write_json(
            RUN / "engine/engine_metadata.json",
            {
                "status": "PASS",
                "engine_path": str(PACKAGED_ENGINE),
                "bindings": runner.binding_info,
                "input_shape_verified": True,
                "input_output_fp32_verified": True,
                "fp16_enabled": False,
                "int8_enabled": False,
                "tf32_enabled": False,
                "verification_basis": "strict build command plus runtime binding inspection",
                "cudart": runner.cudart_path,
            },
        )
        (RUN / "engine/engine_inspector.txt").write_text(
            json.dumps(runner.binding_info, indent=2) + "\n", encoding="utf-8"
        )
        (RUN / "engine/layer_precision_report.txt").write_text(
            "Strict FP32 build configuration\n"
            "TF32 disabled: YES\n"
            "FP16 enabled: NO\n"
            "INT8 enabled: NO\n"
            "DLA enabled: NO\n"
            "Input/output bindings: FP32\n"
            "TensorRT 8.0.1 detailed layer inspector: unavailable in this environment.\n",
            encoding="utf-8",
        )
        for index in indices:
            sample = samples[index]
            tensor = preprocess(sample["path"])
            pytorch_logits = model(torch.from_numpy(tensor)).detach().cpu().numpy().reshape(-1)
            trt_output, _ = runner.infer(tensor, timed=False)
            trt_logits = trt_output.reshape(-1)
            onnx_logits = np.asarray(onnx_rows[index]["logits"], dtype=np.float32)
            rows.append(
                {
                    "sample_index": index,
                    "image_path": sample["path"],
                    "true_class_id": sample["class_id"],
                    "pytorch_top1": int(np.argmax(pytorch_logits)),
                    "onnx_top1": int(np.argmax(onnx_logits)),
                    "tensorrt_top1": int(np.argmax(trt_logits)),
                    "pytorch_onnx_max_abs_error": float(np.max(np.abs(pytorch_logits - onnx_logits))),
                    "pytorch_onnx_mean_abs_error": float(np.mean(np.abs(pytorch_logits - onnx_logits))),
                    "pytorch_tensorrt_max_abs_error": float(np.max(np.abs(pytorch_logits - trt_logits))),
                    "pytorch_tensorrt_mean_abs_error": float(np.mean(np.abs(pytorch_logits - trt_logits))),
                    "onnx_tensorrt_max_abs_error": float(np.max(np.abs(onnx_logits - trt_logits))),
                    "all_top1_agree": int(
                        np.argmax(pytorch_logits) == np.argmax(onnx_logits) == np.argmax(trt_logits)
                    ),
                }
            )
    all_agree = all(row["all_top1_agree"] for row in rows)
    max_error = max(row["pytorch_tensorrt_max_abs_error"] for row in rows)
    status = "PASS" if all_agree and max_error <= 1e-3 else "FAIL"
    summary = {
        "status": status,
        "sample_count": len(rows),
        "sampling": "20 evenly spaced deterministic indices over the sorted 261-image list",
        "all_top1_agree": all_agree,
        "max_pytorch_tensorrt_abs_error": max_error,
        "mean_pytorch_tensorrt_abs_error": float(
            np.mean([row["pytorch_tensorrt_mean_abs_error"] for row in rows])
        ),
        "tolerance": 1e-3,
        "duration_seconds": time.time() - start,
    }
    write_csv(RUN / "validation/framework_output_comparison.csv", rows)
    write_json(RUN / "validation/framework_output_summary.json", summary)
    (RUN / "validation/correctness_validation_report.md").write_text(
        "# Correctness Validation\n\n"
        f"- Status: **{status}**\n"
        f"- Images: {len(rows)} deterministic samples\n"
        f"- PyTorch/ONNX/TensorRT top-1 agreement: {all_agree}\n"
        f"- Maximum PyTorch/TensorRT absolute logit error: {max_error:.9g}\n"
        "- This validation ran in a separate process from the official benchmark.\n",
        encoding="utf-8",
    )
    if status != "PASS":
        raise RuntimeError(f"Correctness validation failed: {summary}")
    print("CORRECTNESS VALIDATION PASS")


if __name__ == "__main__":
    main()
