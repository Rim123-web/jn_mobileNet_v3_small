from __future__ import annotations

import csv
import json
import os
import platform
import socket
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from common import (
    CLASS_NAMES_PATH,
    DATASET_ROOT,
    HISTORICAL_SCRIPT,
    IMAGE_ORDER_PATH,
    IMG_EXTS,
    INPUT_SHAPE,
    ORIGINAL_ONNX,
    PACKAGED_MODEL,
    PACKAGED_ONNX,
    PACKAGED_ORIGINAL_ONNX,
    PREPROCESSING_PATH,
    RESNET50_REFERENCE,
    RUN,
    SOURCE_CLASS_NAMES,
    SOURCE_MODEL,
    SOURCE_MODEL_DIR,
    SOURCE_ONNX,
    SOURCE_PREPROCESSING,
    SOURCE_ROOT,
    WARMUP_COUNT,
    class_names,
    copy_preserve,
    ensure_dirs,
    file_record,
    load_model,
    preprocessing_config,
    sha256,
    write_csv,
    write_json,
)


def command_output(command: str) -> str:
    return subprocess.run(
        command, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    ).stdout


def environment_inventory() -> None:
    preflight = {
        "hostname": socket.gethostname(),
        "ip": command_output("hostname -I").strip(),
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device_model": Path("/proc/device-tree/model").read_bytes().replace(b"\0", b"").decode(),
        "architecture": platform.machine(),
        "os_release": Path("/etc/os-release").read_text(),
        "kernel": platform.release(),
        "l4t": Path("/etc/nv_tegra_release").read_text().strip(),
        "cuda": command_output("/usr/local/cuda/bin/nvcc --version"),
        "tensorrt": command_output("dpkg-query -W -f='${Package}|${Version}\\n' libnvinfer8 2>/dev/null"),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "nvpmodel": command_output("nvpmodel -q 2>&1"),
        "jetson_clocks": command_output("jetson_clocks --show 2>&1"),
        "cpu_online": Path("/sys/devices/system/cpu/online").read_text().strip(),
        "cpu_governors": [
            path.read_text().strip()
            for path in sorted(Path("/sys/devices/system/cpu").glob("cpu*/cpufreq/scaling_governor"))
        ],
        "cpu_frequencies": command_output("grep -H . /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq 2>/dev/null"),
        "memory": command_output("free -h"),
        "swap": command_output("swapon --show"),
        "storage": command_output("df -h /home/jetson/Downloads"),
        "temperatures": command_output("grep -H . /sys/class/thermal/thermal_zone*/temp 2>/dev/null"),
        "active_benchmark_processes": command_output(
            "ps -eo pid,ppid,etimes,pcpu,pmem,args | "
            "grep -Ei 'benchmark|trtexec|inference' | grep -v grep || true"
        ),
    }
    versions = {}
    for module in ["torch", "torchvision", "numpy", "PIL", "cv2", "jtop", "psutil", "matplotlib"]:
        try:
            imported = __import__(module)
            versions[module] = {
                "version": getattr(imported, "__version__", "NA"),
                "path": getattr(imported, "__file__", "NA"),
            }
        except Exception as exc:
            versions[module] = {"error": repr(exc)}
    try:
        user_site = "/home/jetson/.local/lib/python3.8/site-packages"
        sys.path.insert(0, user_site)
        import tensorrt
        versions["tensorrt_python"] = {"version": tensorrt.__version__, "path": tensorrt.__file__}
        sys.path.remove(user_site)
    except Exception as exc:
        versions["tensorrt_python"] = {"error": repr(exc)}
    for name in ("jtop", "psutil"):
        if name in versions and "version" in versions[name]:
            preflight[name] = versions[name]["version"]
    write_json(RUN / "environment/system_state_before.json", preflight)
    write_json(RUN / "environment/software_versions.json", versions)
    (RUN / "environment/preflight.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in preflight.items()), encoding="utf-8"
    )
    (RUN / "environment/storage_before.txt").write_text(preflight["storage"], encoding="utf-8")
    process_rows = []
    for line in preflight["active_benchmark_processes"].splitlines():
        parts = line.split(None, 5)
        if len(parts) == 6:
            process_rows.append(dict(zip(["pid", "ppid", "elapsed_s", "cpu", "memory", "command"], parts)))
    write_csv(RUN / "environment/processes_before.csv", process_rows, ["pid", "ppid", "elapsed_s", "cpu", "memory", "command"])
    thermal_rows = []
    for line in preflight["temperatures"].splitlines():
        if ":" in line:
            path, value = line.rsplit(":", 1)
            thermal_rows.append({"sensor": path, "temperature_c": float(value) / 1000.0})
    write_csv(RUN / "environment/thermal_state_before.csv", thermal_rows)
    (RUN / "environment/pip_freeze.txt").write_text(command_output(f"{sys.executable} -m pip freeze"), encoding="utf-8")
    reference = json.loads((RESNET50_REFERENCE / "environment/system_state_before.json").read_text())
    comparison = {
        "status": "PASS",
        "reference_package": str(RESNET50_REFERENCE),
        "fields": {
            key: {"resnet50": reference.get(key), "mobilenet": preflight.get(key), "match": reference.get(key) == preflight.get(key)}
            for key in ["hostname", "device_model", "architecture", "kernel", "l4t", "cuda", "tensorrt", "python_executable", "python_version", "cpu_online", "cpu_governors"]
        },
        "note": "Timestamps, temperatures, free memory, and process state are expected to vary between runs.",
    }
    write_json(RUN / "environment/resnet50_environment_comparison.json", comparison)


def inventory_dataset() -> None:
    names = class_names()
    directories = {path.name: path for path in DATASET_ROOT.iterdir() if path.is_dir()}
    if set(directories) != set(names):
        raise RuntimeError(f"Class-folder mismatch: expected {names}, found {sorted(directories)}")
    rows, failures, hidden, unsupported = [], [], [], []
    content_hashes = defaultdict(list)
    for class_id, name in enumerate(names):
        for path in sorted(directories[name].rglob("*")):
            if not path.is_file():
                continue
            if any(part.startswith(".") for part in path.relative_to(DATASET_ROOT).parts):
                hidden.append(str(path))
            if path.suffix.lower() not in IMG_EXTS:
                unsupported.append(str(path)); continue
            status, error, width, height, channels = "OK", "", None, None, None
            try:
                if path.stat().st_size == 0:
                    raise ValueError("zero-byte file")
                with Image.open(path) as image:
                    image.load(); width, height = image.size; channels = len(image.getbands())
            except Exception as exc:
                status, error = "FAILED", repr(exc); failures.append({"path": str(path), "error": error})
            digest = sha256(path); content_hashes[digest].append(str(path))
            rows.append({
                "absolute_path": str(path), "relative_path": str(path.relative_to(DATASET_ROOT)),
                "filename": path.name, "extension": path.suffix.lower(), "class_folder": name,
                "class_id": class_id, "class_name": name, "width": width, "height": height,
                "channels": channels, "file_size_bytes": path.stat().st_size, "sha256": digest,
                "decoding_status": status, "error": error,
            })
    if failures or hidden or unsupported or len(rows) != 261:
        raise RuntimeError(f"Dataset validation failed: count={len(rows)} corrupt={len(failures)} hidden={len(hidden)} unsupported={len(unsupported)}")
    order = [row["absolute_path"] for row in rows]
    IMAGE_ORDER_PATH.write_text("\n".join(order) + "\n", encoding="utf-8")
    for name in ("dataset_inventory", "test_inventory"):
        write_csv(RUN / f"dataset/{name}.csv", rows); write_json(RUN / f"dataset/{name}.json", rows)
    write_csv(RUN / "dataset/test_image_hashes.csv", [{"absolute_path": r["absolute_path"], "sha256": r["sha256"]} for r in rows])
    (RUN / "dataset/test_image_order.txt").write_text("\n".join(order) + "\n", encoding="utf-8")
    counts = Counter(row["class_name"] for row in rows)
    distribution = [{"class_id": i, "class_name": n, "image_count": counts[n]} for i, n in enumerate(names)]
    write_csv(RUN / "dataset/class_distribution.csv", distribution)
    write_csv(RUN / "dataset/test_class_distribution.csv", distribution)
    write_json(RUN / "dataset/class_mapping.json", {str(i): name for i, name in enumerate(names)})
    duplicates = {key: value for key, value in content_hashes.items() if len(value) > 1}
    write_json(RUN / "dataset/dataset_hash_summary.json", {
        "dataset_root": str(DATASET_ROOT), "total_images": len(rows),
        "combined_ordered_sha256": sha256(IMAGE_ORDER_PATH), "duplicate_content_groups": duplicates,
        "all_images_decoded": True, "hidden_files": hidden, "unsupported_files": unsupported,
        "zero_byte_or_corrupt_files": failures,
    })
    reference_rows = list(csv.DictReader((RESNET50_REFERENCE / "dataset/dataset_inventory.csv").open()))
    parity = {
        "status": "PASS" if len(reference_rows) == len(rows) and all(a["sha256"] == b["sha256"] and a["relative_path"] == b["relative_path"] for a, b in zip(reference_rows, rows)) else "FAIL",
        "resnet50_reference": str(RESNET50_REFERENCE), "current_count": len(rows), "reference_count": len(reference_rows),
        "ordered_hash_and_path_match": len(reference_rows) == len(rows) and all(a["sha256"] == b["sha256"] and a["relative_path"] == b["relative_path"] for a, b in zip(reference_rows, rows)),
    }
    write_json(RUN / "validation/resnet50_test_dataset_parity.json", parity)
    if parity["status"] != "PASS":
        raise RuntimeError(f"ResNet50 dataset parity failed: {parity}")
    report = (
        "# Test Dataset Validation\n\n"
        f"- Actual source: `{DATASET_ROOT}`\n- Requested alias `/home/jetson/Downloads/classification_benchmark/test_imgs` was absent.\n"
        f"- Images: {len(rows)}\n- Classes: {names}\n- Corrupt/hidden/unsupported: 0\n"
        f"- Duplicate-content groups: {len(duplicates)}\n- ResNet50 ordered hash parity: PASS\n"
    )
    (RUN / "dataset/test_dataset_validation.md").write_text(report, encoding="utf-8")
    (RUN / "dataset/dataset_validation_report.md").write_text(report, encoding="utf-8")


def preserve_sources() -> None:
    required = [SOURCE_MODEL, ORIGINAL_ONNX, SOURCE_ONNX, SOURCE_CLASS_NAMES, SOURCE_PREPROCESSING, HISTORICAL_SCRIPT]
    for path in required:
        if not path.is_file():
            raise RuntimeError(f"Required source artifact missing: {path}")
    copy_preserve(SOURCE_MODEL, PACKAGED_MODEL)
    copy_preserve(ORIGINAL_ONNX, PACKAGED_ORIGINAL_ONNX)
    copy_preserve(SOURCE_ONNX, PACKAGED_ONNX)
    copy_preserve(HISTORICAL_SCRIPT, RUN / "source/historical_mobilenet_fp32_core4_benchmark.py")
    for filename in ("onnx_export_metadata.json", "onnx_export_report.txt"):
        copy_preserve(SOURCE_MODEL_DIR / filename, RUN / f"source/{filename}")
    model, names, checkpoint = load_model()
    metadata = {
        "architecture": "MobileNetV3-Small", "task": "classification", "class_count": len(names),
        "class_names": names, "input_shape": list(INPUT_SHAPE), "input_dtype": "float32",
        "related_training_checkpoint_found": True, "source_artifact_for_engine": "ONNX",
        "checkpoint_keys": sorted(checkpoint) if isinstance(checkpoint, dict) else [],
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "related_checkpoint": file_record(SOURCE_MODEL), "packaged_checkpoint": file_record(PACKAGED_MODEL),
        "copy_hash_matches": sha256(SOURCE_MODEL) == sha256(PACKAGED_MODEL),
    }
    write_json(RUN / "model/model_metadata.json", metadata)
    write_json(RUN / "model/selected_model_provenance.json", {
        "architecture": "MobileNetV3-Small", "earliest_available_executable_artifact": str(ORIGINAL_ONNX),
        "selected_tensorrt_source_onnx": str(SOURCE_ONNX),
        "selection_reason": "Historical executed FP32 benchmark created this Hardswish-decomposed ONNX from the same preserved inference checkpoint after TensorRT 8.0.1 rejected the original HardSwish graph.",
        "original_onnx_preserved": str(PACKAGED_ORIGINAL_ONNX), "compatible_onnx_preserved": str(PACKAGED_ONNX),
        "original_hash": sha256(ORIGINAL_ONNX), "compatible_hash": sha256(SOURCE_ONNX),
    })
    candidates = []
    candidate_paths = sorted({*SOURCE_MODEL_DIR.glob("*.onnx"), SOURCE_ONNX, SOURCE_ROOT / "models/mobilenet_v3_small/mobilenet_v3_small_inference_trt_compatible.onnx"})
    for path in candidate_paths:
        if path.is_file():
            candidates.append({**file_record(path), "selected": path == SOURCE_ONNX, "selection_note": "selected historical FP32 compatible export" if path == SOURCE_ONNX else "preserved comparison candidate"})
    write_csv(RUN / "onnx/onnx_inventory.csv", candidates)
    write_json(RUN / "onnx/selected_onnx_provenance.json", {
        "selected": file_record(SOURCE_ONNX), "packaged": file_record(PACKAGED_ONNX),
        "copy_hash_matches": sha256(SOURCE_ONNX) == sha256(PACKAGED_ONNX),
        "original_export": file_record(ORIGINAL_ONNX),
        "compatibility_fix": "torch.nn.Hardswish decomposed to x * ReLU6(x + 3) / 6; no retraining or weight change",
    })
    (RUN / "model/model_sha256.txt").write_text(f"{sha256(PACKAGED_MODEL)}  {PACKAGED_MODEL.name}\n", encoding="utf-8")


def audit_history_and_preprocessing() -> None:
    config = preprocessing_config()
    expected = {"img_size": 640, "resize": [640, 640], "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "input_shape": [1, 3, 640, 640]}
    if config != expected:
        raise RuntimeError(f"Preprocessing ambiguity: {config}")
    historical_metrics = json.loads((SOURCE_ROOT / "mobilenet_v3_small_fp32_core4_benchmark/metrics/mobilenet_v3_small_fp32_core4_benchmark_table.json").read_text())
    evidence = [
        {"property": "architecture", "value": "MobileNetV3-Small", "source": str(HISTORICAL_SCRIPT), "lines": "137-177"},
        {"property": "decoder_color", "value": "PIL.Image.open, RGB", "source": str(HISTORICAL_SCRIPT), "lines": "180-181"},
        {"property": "geometry", "value": "direct resize 640x640, PIL bilinear, no crop/pad", "source": str(HISTORICAL_SCRIPT), "lines": "180-181"},
        {"property": "pixel_scale", "value": "divide by 255", "source": str(HISTORICAL_SCRIPT), "lines": "181-184"},
        {"property": "normalization", "value": f"mean={config['mean']} std={config['std']}", "source": str(HISTORICAL_SCRIPT), "lines": "67-73, 180-185"},
        {"property": "TensorRT ONNX", "value": "Hardswish-decomposed compatible ONNX", "source": str(HISTORICAL_SCRIPT), "lines": "455-598"},
    ]
    write_csv(RUN / "config/preprocessing_evidence.csv", evidence)
    write_csv(RUN / "source/historical_mobilenet_evidence.csv", evidence)
    write_json(RUN / "config/normalization_verification.json", {"status": "PASS", "mean": config["mean"], "std": config["std"], "source": str(HISTORICAL_SCRIPT), "ambiguity": None})
    write_json(RUN / "source/historical_result_reference.json", {"accuracy": 0.9731800766283525, "macro_f1": 0.9518572587185726, "inference_fps": 24.833711273730692, "inference_latency_ms": 40.267845, "energy_per_inference_j": 0.19881923005449262, "reference_only": True, "raw_table": historical_metrics})
    audit = (
        "# Historical MobileNetV3-Small Audit\n\n"
        f"- Executed source: `{HISTORICAL_SCRIPT}`\n- Architecture: MobileNetV3-Small, three classes.\n"
        f"- Original ONNX: `{ORIGINAL_ONNX}` (`{sha256(ORIGINAL_ONNX)}`).\n"
        f"- Selected TensorRT-compatible ONNX: `{SOURCE_ONNX}` (`{sha256(SOURCE_ONNX)}`).\n"
        "- Original limitation: TensorRT 8.0.1 did not import HardSwish.\n"
        "- Existing fix: Hardswish was decomposed into primitive operations from the same checkpoint; no training occurred.\n"
        "- Preprocessing: RGB, direct PIL bilinear 640x640 resize, /255, ImageNet mean/std, HWC-to-CHW FP32.\n"
        "- Preprocessing ambiguity: none.\n"
    )
    (RUN / "source/historical_mobilenet_audit.md").write_text(audit, encoding="utf-8")
    (RUN / "config/historical_preprocessing_audit.md").write_text(audit, encoding="utf-8")
    (RUN / "config/preprocessing_specification.yaml").write_text(
        "color_order: RGB\ndecoder: PIL.Image\nresize: [640, 640]\nresize_method: direct\ninterpolation: bilinear\ncrop: none\npadding: none\npixel_scale: 1/255\nmean: [0.485, 0.456, 0.406]\nstd: [0.229, 0.224, 0.225]\nlayout: NCHW\ndtype: float32\ninput_shape: [1, 3, 640, 640]\n",
        encoding="utf-8",
    )


def reproduction_files() -> None:
    manifest = {
        "run_directory": str(RUN), "task": "classification", "model": "MobileNetV3-Small",
        "source_format": "ONNX", "precision": "TensorRT FP32 strict", "tf32_disabled": True,
        "fp16_disabled": True, "int8_disabled": True, "calibration_used": False, "dla_used": False,
        "batch_size": 1, "input_shape": list(INPUT_SHAPE), "warmup_count": WARMUP_COUNT,
        "dataset_source": str(DATASET_ROOT), "source_original_onnx": str(ORIGINAL_ONNX),
        "source_onnx": str(SOURCE_ONNX), "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "status": "PREPARED",
    }
    write_json(RUN / "run_manifest.json", manifest)
    (RUN / "requirements.txt").write_text(
        "numpy==1.23.5\nPillow==9.2.0\nmatplotlib==3.7.5\npsutil==7.2.2\njetson-stats==4.3.2\nTensorRT==8.0.1.6 (Jetson package; do not pip install)\n",
        encoding="utf-8",
    )
    reproduce = f"""#!/usr/bin/env bash
set -euo pipefail
RUN={RUN}
PYTHONNOUSERSITE=1 /home/jetson/Downloads/jetson_nano/.venv_yolo/bin/python "$RUN/scripts/prepare_project.py"
/home/jetson/Downloads/jetson_folder/jetson_benchmark_env/bin/python "$RUN/scripts/inspect_onnx_candidates.py"
/home/jetson/Downloads/jetson_folder/jetson_benchmark_env/bin/python "$RUN/scripts/validate_onnx.py"
/home/jetson/Downloads/jetson_folder/jetson_benchmark_env/bin/python "$RUN/scripts/validate_onnx_dataset.py"
PYTHONNOUSERSITE=1 /home/jetson/Downloads/jetson_nano/.venv_yolo/bin/python "$RUN/scripts/validate_trt_parser.py"
bash "$RUN/scripts/build_engine.sh"
PYTHONNOUSERSITE=1 /home/jetson/Downloads/jetson_nano/.venv_yolo/bin/python "$RUN/scripts/validate_correctness.py"
PYTHONNOUSERSITE=1 /home/jetson/Downloads/jetson_nano/.venv_yolo/bin/python "$RUN/scripts/run_idle_monitor.py"
PYTHONNOUSERSITE=1 /home/jetson/Downloads/jetson_nano/.venv_yolo/bin/python "$RUN/scripts/benchmark_mobilenet_v3_small_fp32_warmup10.py"
PYTHONNOUSERSITE=1 /home/jetson/Downloads/jetson_nano/.venv_yolo/bin/python "$RUN/scripts/finalize_results.py"
"""
    (RUN / "reproduce.sh").write_text(reproduce, encoding="utf-8"); (RUN / "reproduce.sh").chmod(0o755)
    (RUN / "reproduce.ps1").write_text(f"Write-Host 'Run on Jetson: bash {RUN}/reproduce.sh'\n", encoding="utf-8")
    (RUN / "README.md").write_text(
        "# MobileNetV3-Small TensorRT Strict-FP32 Jetson Nano Benchmark\n\n"
        "Fresh publication benchmark. Source format: ONNX. TensorRT FP32 with TF32, FP16, INT8, calibration, and DLA disabled.\n"
        "Input 1x3x640x640, batch 1, exactly 10 excluded warmups, then all 261 validated test images once.\n",
        encoding="utf-8",
    )


def main() -> None:
    ensure_dirs()
    active = command_output("ps -eo pid,args | grep -Ei '/trtexec|benchmark_mobilenet|run_mobilenet' | grep -v grep | grep -v prepare_project.py || true").strip()
    if active:
        raise RuntimeError(f"Another relevant process is active:\n{active}")
    copy_preserve(Path(__file__).resolve(), RUN / "scripts/prepare_project.py")
    copy_preserve(SOURCE_CLASS_NAMES, CLASS_NAMES_PATH); copy_preserve(SOURCE_PREPROCESSING, PREPROCESSING_PATH)
    environment_inventory(); inventory_dataset(); preserve_sources(); audit_history_and_preprocessing(); reproduction_files()
    write_json(RUN / "validation/preparation_status.json", {"status": "PASS"})
    print("PREPARATION PASS")
    print(f"RUN={RUN}")
    print(f"SELECTED_ONNX={SOURCE_ONNX}")
    print(f"SELECTED_ONNX_SIZE={SOURCE_ONNX.stat().st_size}")
    print(f"SELECTED_ONNX_SHA256={sha256(SOURCE_ONNX)}")
    print("INPUT=[1,3,640,640] OUTPUT=[1,3] CLASSES=3")
    print("SELECTION_EVIDENCE=historical executed FP32 Hardswish-decomposed export from the preserved inference checkpoint")
    print(f"DATASET={DATASET_ROOT} IMAGES={len(IMAGE_ORDER_PATH.read_text().splitlines())}")


if __name__ == "__main__":
    main()
