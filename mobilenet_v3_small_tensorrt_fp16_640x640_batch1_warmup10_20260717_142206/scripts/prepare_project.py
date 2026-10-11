from __future__ import annotations

import ast
import csv
import hashlib
import json
import platform
import shutil
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
    FP32_REFERENCE,
    IMAGE_ORDER_PATH,
    IMG_EXTS,
    INPUT_SHAPE,
    PACKAGED_MODEL,
    PACKAGED_ONNX,
    PACKAGED_ORIGINAL_ONNX,
    PREPROCESSING_PATH,
    RUN,
    WARMUP_COUNT,
    class_names,
    copy_preserve,
    ensure_dirs,
    file_record,
    load_model,
    sha256,
    write_csv,
    write_json,
)


FP32_ONNX = FP32_REFERENCE / "onnx/mobilenet_v3_small_source.onnx"
FP32_ORIGINAL_ONNX = FP32_REFERENCE / "onnx/mobilenet_v3_small_original_export.onnx"
FP32_MODEL = FP32_REFERENCE / "model/mobilenet_v3_small_inference_only.pt"
FP32_CLASS_NAMES = FP32_REFERENCE / "config/class_names.json"
FP32_PREPROCESSING = FP32_REFERENCE / "config/preprocessing.json"
FP32_BENCHMARK = FP32_REFERENCE / "scripts/benchmark_mobilenet_v3_small_fp32_warmup10.py"


def command_output(command: str) -> str:
    return subprocess.run(
        command, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    ).stdout


def function_source(path: Path, function_name: str) -> str:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            return ast.get_source_segment(text, node) or ""
    raise RuntimeError(f"Function {function_name} not found in {path}")


def validate_fp32_reference() -> dict:
    required = [
        FP32_REFERENCE / "validation/final_validation_report.json",
        FP32_REFERENCE / "validation/file_hash_manifest.csv",
        FP32_ONNX,
        FP32_MODEL,
        FP32_CLASS_NAMES,
        FP32_PREPROCESSING,
        FP32_REFERENCE / "summary/article_results.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"FP32 reference files missing: {missing}")
    report = json.loads((FP32_REFERENCE / "validation/final_validation_report.json").read_text())
    if report.get("status") != "PASS" or not all(report.get("checks", {}).values()):
        raise RuntimeError("FP32 reference package is not fully validated PASS")
    with (FP32_REFERENCE / "validation/file_hash_manifest.csv").open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    failures = []
    for row in manifest:
        path = FP32_REFERENCE / row["relative_path"]
        if not path.is_file() or sha256(path) != row["sha256"]:
            failures.append(row["relative_path"])
    if failures:
        raise RuntimeError(f"FP32 reference hash validation failed: {failures[:10]}")
    article = json.loads((FP32_REFERENCE / "summary/article_results.json").read_text())
    result = {
        "status": "PASS",
        "reference_package": str(FP32_REFERENCE),
        "validation_status": report["status"],
        "manifest_rows_verified": len(manifest),
        "manifest_failures": failures,
        "onnx": file_record(FP32_ONNX),
        "engine": article["engine"],
        "headline_results": {
            "accuracy": article["metrics"]["accuracy"],
            "macro_f1": article["metrics"]["macro_f1"],
            "engine_fps": article["performance"]["engine_fps"],
            "engine_latency_ms": article["performance"]["mean_engine_latency_ms"],
            "wall_clock_pipeline_fps": article["performance"]["wall_clock_pipeline_fps"],
            "gross_energy_per_image_j": article["power_energy"]["gross_energy_per_image_j"],
        },
    }
    write_json(RUN / "validation/fp32_reference_package.json", result)
    return result


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
            "ps -eo pid,ppid,etimes,pcpu,pmem,args | grep -Ei 'benchmark|trtexec|inference' | grep -v grep || true"
        ),
    }
    versions = {}
    for module in ["torch", "torchvision", "numpy", "PIL", "cv2", "jtop", "psutil", "matplotlib"]:
        try:
            imported = __import__(module)
            versions[module] = {"version": getattr(imported, "__version__", "NA"), "path": getattr(imported, "__file__", "NA")}
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
    preflight["jtop"] = versions.get("jtop", {}).get("version", "NA")
    preflight["psutil"] = versions.get("psutil", {}).get("version", "NA")
    write_json(RUN / "environment/system_state_before.json", preflight)
    write_json(RUN / "environment/software_versions.json", versions)
    (RUN / "environment/preflight.txt").write_text("\n".join(f"{key}: {value}" for key, value in preflight.items()), encoding="utf-8")
    (RUN / "environment/storage_before.txt").write_text(preflight["storage"], encoding="utf-8")
    processes = []
    for line in preflight["active_benchmark_processes"].splitlines():
        parts = line.split(None, 5)
        if len(parts) == 6:
            processes.append(dict(zip(["pid", "ppid", "elapsed_s", "cpu", "memory", "command"], parts)))
    write_csv(RUN / "environment/processes_before.csv", processes, ["pid", "ppid", "elapsed_s", "cpu", "memory", "command"])
    thermal = []
    for line in preflight["temperatures"].splitlines():
        if ":" in line:
            path, value = line.rsplit(":", 1)
            thermal.append({"sensor": path, "temperature_c": float(value) / 1000.0})
    write_csv(RUN / "environment/thermal_state_before.csv", thermal)
    (RUN / "environment/pip_freeze.txt").write_text(command_output(f"{sys.executable} -m pip freeze"), encoding="utf-8")
    reference = json.loads((FP32_REFERENCE / "environment/system_state_before.json").read_text())
    fields = {}
    for key in ["hostname", "device_model", "architecture", "kernel", "l4t", "cuda", "tensorrt", "python_executable", "python_version", "cpu_online", "cpu_governors", "nvpmodel"]:
        fields[key] = {"fp32": reference.get(key), "fp16": preflight.get(key), "match": reference.get(key) == preflight.get(key)}
    comparison = {
        "status": "PASS" if all(value["match"] for key, value in fields.items() if key not in {"nvpmodel"}) else "FAIL",
        "reference_package": str(FP32_REFERENCE),
        "fields": fields,
        "note": "Timestamps, temperatures, free memory, frequencies, and process state are expected to vary.",
    }
    write_json(RUN / "environment/fp32_environment_comparison.json", comparison)
    if comparison["status"] != "PASS":
        raise RuntimeError(f"Material FP32/FP16 environment mismatch: {comparison}")


def inventory_dataset() -> None:
    names = class_names()
    folders = {path.name: path for path in DATASET_ROOT.iterdir() if path.is_dir()}
    if set(folders) != set(names):
        raise RuntimeError(f"Class folder mismatch: expected {names}, found {sorted(folders)}")
    rows, failures, hidden, unsupported = [], [], [], []
    hashes = defaultdict(list)
    for class_id, name in enumerate(names):
        for path in sorted(folders[name].rglob("*")):
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
            digest = sha256(path); hashes[digest].append(str(path))
            rows.append({
                "absolute_path": str(path), "relative_path": str(path.relative_to(DATASET_ROOT)),
                "filename": path.name, "extension": path.suffix.lower(), "class_folder": name,
                "class_id": class_id, "class_name": name, "width": width, "height": height,
                "channels": channels, "file_size_bytes": path.stat().st_size, "sha256": digest,
                "decoding_status": status, "error": error,
            })
    if failures or hidden or unsupported:
        raise RuntimeError(f"Dataset invalid: corrupt={failures} hidden={hidden} unsupported={unsupported}")
    order = [row["absolute_path"] for row in rows]
    IMAGE_ORDER_PATH.write_text("\n".join(order) + "\n", encoding="utf-8")
    write_csv(RUN / "dataset/test_inventory.csv", rows); write_json(RUN / "dataset/test_inventory.json", rows)
    write_csv(RUN / "dataset/dataset_inventory.csv", rows); write_json(RUN / "dataset/dataset_inventory.json", rows)
    write_csv(RUN / "dataset/test_image_hashes.csv", [{"absolute_path": row["absolute_path"], "sha256": row["sha256"]} for row in rows])
    (RUN / "dataset/test_image_order.txt").write_text("\n".join(order) + "\n", encoding="utf-8")
    counts = Counter(row["class_name"] for row in rows)
    distribution = [{"class_id": i, "class_name": name, "image_count": counts[name]} for i, name in enumerate(names)]
    write_csv(RUN / "dataset/test_class_distribution.csv", distribution)
    write_csv(RUN / "dataset/class_distribution.csv", distribution)
    write_json(RUN / "dataset/class_mapping.json", {str(i): name for i, name in enumerate(names)})
    write_json(RUN / "dataset/dataset_hash_summary.json", {
        "dataset_root": str(DATASET_ROOT), "total_images": len(rows),
        "duplicate_content_groups": {key: value for key, value in hashes.items() if len(value) > 1},
        "all_images_decoded": True, "hidden_files": hidden, "unsupported_files": unsupported,
        "zero_byte_or_corrupt_files": failures,
    })
    with (FP32_REFERENCE / "dataset/test_inventory.csv").open(newline="", encoding="utf-8") as handle:
        reference = list(csv.DictReader(handle))
    match = len(reference) == len(rows) and all(
        old["relative_path"] == new["relative_path"]
        and old["sha256"] == new["sha256"]
        and int(old["class_id"]) == int(new["class_id"])
        and old["class_name"] == new["class_name"]
        for old, new in zip(reference, rows)
    )
    parity = {
        "status": "PASS" if match else "FAIL", "reference_package": str(FP32_REFERENCE),
        "fp32_count": len(reference), "fp16_count": len(rows),
        "ordered_relative_path_hash_class_match": match,
    }
    write_json(RUN / "validation/fp32_fp16_dataset_parity.json", parity)
    if not match:
        raise RuntimeError(f"FP32/FP16 dataset parity failed: {parity}")
    (RUN / "dataset/test_dataset_validation.md").write_text(
        "# FP16 Test Dataset Validation\n\n"
        f"- Actual dataset source: `{DATASET_ROOT}`\n- Images: {len(rows)}\n- Classes: {names}\n"
        "- Corrupt, hidden, zero-byte, unsupported: 0\n- Ordered SHA-256/class parity with FP32: PASS\n",
        encoding="utf-8",
    )


def preserve_sources_and_verify_protocol(reference: dict) -> None:
    for path in [FP32_ONNX, FP32_ORIGINAL_ONNX, FP32_MODEL, FP32_CLASS_NAMES, FP32_PREPROCESSING]:
        if not path.is_file():
            raise RuntimeError(f"Required FP32 source artifact missing: {path}")
    copy_preserve(FP32_ONNX, PACKAGED_ONNX)
    copy_preserve(FP32_ORIGINAL_ONNX, PACKAGED_ORIGINAL_ONNX)
    copy_preserve(FP32_MODEL, PACKAGED_MODEL)
    for source, destination in [
        (FP32_REFERENCE / "source/onnx_export_metadata.json", RUN / "source/onnx_export_metadata.json"),
        (FP32_REFERENCE / "source/onnx_export_report.txt", RUN / "source/onnx_export_report.txt"),
        (FP32_REFERENCE / "source/historical_mobilenet_audit.md", RUN / "source/historical_mobilenet_audit.md"),
        (FP32_REFERENCE / "config/preprocessing_evidence.csv", RUN / "config/preprocessing_evidence.csv"),
    ]:
        if source.is_file():
            copy_preserve(source, destination)
    model, names, checkpoint = load_model(PACKAGED_MODEL)
    write_json(RUN / "model/model_metadata.json", {
        "architecture": "MobileNetV3-Small", "task": "classification", "class_count": len(names),
        "class_names": names, "input_shape": list(INPUT_SHAPE), "input_dtype": "float32",
        "source_artifact_for_engine": "exact FP32 reference ONNX", "parameter_count": sum(p.numel() for p in model.parameters()),
        "checkpoint_keys": sorted(checkpoint), "fp32_reference_model": file_record(FP32_MODEL),
        "packaged_model": file_record(PACKAGED_MODEL), "copy_hash_matches": sha256(FP32_MODEL) == sha256(PACKAGED_MODEL),
    })
    write_json(RUN / "model/selected_model_provenance.json", {
        "architecture": "MobileNetV3-Small", "selected_onnx": str(FP32_ONNX),
        "selection_reason": "Exact validated ONNX from the completed FP32 final package",
        "fp32_reference_package": str(FP32_REFERENCE), "copy_hash_matches": sha256(FP32_ONNX) == sha256(PACKAGED_ONNX),
    })
    write_json(RUN / "onnx/selected_onnx_provenance.json", {
        "selected": file_record(FP32_ONNX), "packaged": file_record(PACKAGED_ONNX),
        "copy_hash_matches": sha256(FP32_ONNX) == sha256(PACKAGED_ONNX),
        "fp32_reference_package": str(FP32_REFERENCE), "reexported": False,
    })
    write_json(RUN / "validation/fp32_fp16_onnx_parity.json", {
        "status": "PASS" if sha256(FP32_ONNX) == sha256(PACKAGED_ONNX) else "FAIL",
        "fp32_onnx_sha256": sha256(FP32_ONNX), "fp16_packaged_onnx_sha256": sha256(PACKAGED_ONNX),
        "size_match": FP32_ONNX.stat().st_size == PACKAGED_ONNX.stat().st_size,
    })
    current_common = RUN / "scripts/common.py"
    fp32_common = FP32_REFERENCE / "scripts/common.py"
    preprocessing_match = function_source(current_common, "preprocess_detailed") == function_source(fp32_common, "preprocess_detailed")
    benchmark_match = sha256(RUN / "scripts/benchmark_mobilenet_v3_small_fp16_warmup10.py") == sha256(FP32_BENCHMARK)
    monitor_match = sha256(RUN / "scripts/monitor.py") == sha256(FP32_REFERENCE / "scripts/monitor.py")
    runtime_match = sha256(RUN / "scripts/trt_runtime.py") == sha256(FP32_REFERENCE / "scripts/trt_runtime.py")
    config = json.loads(PREPROCESSING_PATH.read_text())
    fp32_config = json.loads(FP32_PREPROCESSING.read_text())
    precheck = {
        "status": "PASS" if all([preprocessing_match, benchmark_match, monitor_match, runtime_match, config == fp32_config]) else "FAIL",
        "fp32_reference_package": str(FP32_REFERENCE), "fp32_validation": reference["validation_status"],
        "onnx_sha256_match": sha256(FP32_ONNX) == sha256(PACKAGED_ONNX),
        "preprocessing_config_match": config == fp32_config,
        "preprocess_detailed_function_source_match": preprocessing_match,
        "official_benchmark_script_content_match": benchmark_match,
        "monitor_script_hash_match": monitor_match,
        "tensorrt_runtime_script_hash_match": runtime_match,
        "dataset_parity": json.loads((RUN / "validation/fp32_fp16_dataset_parity.json").read_text())["status"],
        "class_mapping_match": (RUN / "config/class_names.json").read_bytes() == FP32_CLASS_NAMES.read_bytes(),
        "input_shape": list(INPUT_SHAPE), "batch_size": 1, "warmup_count": WARMUP_COUNT,
        "experimental_variable": "TensorRT FP16 builder flag/internal tactic precision",
    }
    write_json(RUN / "validation/fp32_fp16_protocol_precheck.json", precheck)
    (RUN / "validation/fp32_fp16_protocol_precheck.md").write_text(
        "# FP32/FP16 Protocol Precheck\n\n" + "\n".join(f"- {key}: {value}" for key, value in precheck.items()) + "\n",
        encoding="utf-8",
    )
    write_json(RUN / "config/preprocessing_parity.json", {
        "status": "PASS" if preprocessing_match and config == fp32_config else "FAIL",
        "mean": config["mean"], "std": config["std"], "color_order": "RGB", "pixel_scaling": "1/255",
        "resize": "direct 640x640", "interpolation": "PIL bilinear", "crop": "none", "padding": "none",
        "source": str(FP32_PREPROCESSING), "function_source_match": preprocessing_match,
    })
    (RUN / "config/preprocessing_specification.yaml").write_text(
        "color_order: RGB\ndecoder: PIL.Image\nresize: [640, 640]\nresize_method: direct\ninterpolation: bilinear\n"
        "crop: none\npadding: none\npixel_scale: 1/255\nmean: [0.485, 0.456, 0.406]\nstd: [0.229, 0.224, 0.225]\n"
        "layout: NCHW\ndtype: float32\ninput_shape: [1, 3, 640, 640]\n",
        encoding="utf-8",
    )
    if precheck["status"] != "PASS":
        raise RuntimeError(f"FP32/FP16 protocol precheck failed: {precheck}")


def reproduction_files() -> None:
    manifest = {
        "run_directory": str(RUN), "task": "classification", "model": "MobileNetV3-Small",
        "source_format": "ONNX", "precision": "TensorRT FP16-enabled", "fp16_enabled": True,
        "tf32_disabled": True, "int8_disabled": True, "calibration_used": False, "dla_used": False,
        "batch_size": 1, "input_shape": list(INPUT_SHAPE), "warmup_count": WARMUP_COUNT,
        "dataset_source": str(DATASET_ROOT), "fp32_reference": str(FP32_REFERENCE),
        "source_onnx": str(FP32_ONNX), "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "status": "PREPARED",
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
PYTHONNOUSERSITE=1 /home/jetson/Downloads/jetson_nano/.venv_yolo/bin/python "$RUN/scripts/validate_onnx.py"
PYTHONNOUSERSITE=1 /home/jetson/Downloads/jetson_nano/.venv_yolo/bin/python "$RUN/scripts/validate_onnx_dataset.py"
PYTHONNOUSERSITE=1 /home/jetson/Downloads/jetson_nano/.venv_yolo/bin/python "$RUN/scripts/validate_trt_parser.py"
bash "$RUN/scripts/build_engine.sh"
PYTHONNOUSERSITE=1 /home/jetson/Downloads/jetson_nano/.venv_yolo/bin/python "$RUN/scripts/validate_correctness.py"
PYTHONNOUSERSITE=1 /home/jetson/Downloads/jetson_nano/.venv_yolo/bin/python "$RUN/scripts/run_idle_monitor.py"
PYTHONNOUSERSITE=1 /home/jetson/Downloads/jetson_nano/.venv_yolo/bin/python "$RUN/scripts/benchmark_mobilenet_v3_small_fp16_warmup10.py"
PYTHONNOUSERSITE=1 /home/jetson/Downloads/jetson_nano/.venv_yolo/bin/python "$RUN/scripts/finalize_results.py"
"""
    (RUN / "reproduce.sh").write_text(reproduce, encoding="utf-8"); (RUN / "reproduce.sh").chmod(0o755)
    (RUN / "reproduce.ps1").write_text(f"Write-Host 'Run on Jetson: bash {RUN}/reproduce.sh'\n", encoding="utf-8")
    (RUN / "README.md").write_text(
        "# MobileNetV3-Small TensorRT FP16-enabled Jetson Nano Benchmark\n\n"
        "Fresh publication benchmark using the exact validated FP32 ONNX, dataset, preprocessing, timing, and telemetry protocol.\n"
        "Input 1x3x640x640, batch 1, exactly 10 excluded warm-ups, then all 261 validated images once. INT8, calibration, and DLA are disabled.\n",
        encoding="utf-8",
    )


def main() -> None:
    ensure_dirs()
    active = command_output("ps -eo pid,args | grep -Ei '/trtexec|benchmark_mobilenet|run_mobilenet' | grep -v grep | grep -v prepare_project.py || true").strip()
    if active:
        raise RuntimeError(f"Another relevant process is active:\n{active}")
    reference = validate_fp32_reference()
    copy_preserve(FP32_CLASS_NAMES, CLASS_NAMES_PATH)
    copy_preserve(FP32_PREPROCESSING, PREPROCESSING_PATH)
    environment_inventory()
    inventory_dataset()
    preserve_sources_and_verify_protocol(reference)
    reproduction_files()
    write_json(RUN / "validation/preparation_status.json", {"status": "PASS"})
    print("FP16 PREPARATION PASS")
    print(f"RUN={RUN}")
    print(f"FP32_REFERENCE={FP32_REFERENCE}")
    print(f"SELECTED_ONNX={FP32_ONNX}")
    print(f"SELECTED_ONNX_SIZE={FP32_ONNX.stat().st_size}")
    print(f"SELECTED_ONNX_SHA256={sha256(FP32_ONNX)}")
    print("INPUT=[1,3,640,640] OUTPUT=[1,3] CLASSES=3")
    print(f"DATASET={DATASET_ROOT} IMAGES={len(IMAGE_ORDER_PATH.read_text().splitlines())}")
    print("FP32_FP16_PROTOCOL_PRECHECK=PASS")


if __name__ == "__main__":
    main()
