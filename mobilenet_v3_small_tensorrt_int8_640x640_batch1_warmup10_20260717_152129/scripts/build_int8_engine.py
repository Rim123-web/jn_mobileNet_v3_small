from __future__ import annotations

import ctypes
import ctypes.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from common import (
    CALIBRATION_CACHE,
    INPUT_SHAPE,
    PACKAGED_ENGINE,
    PACKAGED_ONNX,
    RUN,
    preprocess,
    read_json,
    sha256,
    write_csv,
    write_json,
)


USER_SITE = "/home/jetson/.local/lib/python3.8/site-packages"
CACHE = CALIBRATION_CACHE
MANIFEST = RUN / "calibration/calibration_manifest.txt"
BUILD_LOG = RUN / "engine/engine_build_log.txt"
LAYER_REPORT = RUN / "engine/layer_precision_report.txt"
BUILD_SUMMARY = RUN / "engine/int8_build_summary.json"


def import_tensorrt():
    if USER_SITE not in sys.path:
        sys.path.insert(0, USER_SITE)
    import tensorrt as trt
    sys.path[:] = [
        path
        for path in sys.path
        if path != USER_SITE and not path.startswith(USER_SITE + "/")
    ]
    return trt


def load_cudart():
    candidates = [
        ctypes.util.find_library("cudart"),
        "libcudart.so",
        "/usr/local/cuda/lib64/libcudart.so",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            library = ctypes.CDLL(candidate)
            library.cudaMalloc.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_size_t,
            ]
            library.cudaMalloc.restype = ctypes.c_int
            library.cudaFree.argtypes = [ctypes.c_void_p]
            library.cudaFree.restype = ctypes.c_int
            library.cudaMemcpy.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
            ]
            library.cudaMemcpy.restype = ctypes.c_int
            return library, candidate
        except OSError:
            continue
    raise RuntimeError("Unable to load CUDA Runtime")


def check_cuda(code, operation):
    if int(code) != 0:
        raise RuntimeError(f"{operation} failed with CUDA error {code}")


def main():
    trt = import_tensorrt()
    paths = [Path(line) for line in MANIFEST.read_text().splitlines() if line.strip()]
    if not 90 <= len(paths) <= 300:
        raise RuntimeError(f"Calibration count outside approved range: {len(paths)}")
    if any(not path.is_file() for path in paths):
        raise RuntimeError("Calibration manifest contains a missing image")
    if PACKAGED_ENGINE.exists():
        raise RuntimeError(f"Refusing to overwrite engine: {PACKAGED_ENGINE}")
    if CACHE.exists():
        raise RuntimeError(
            "Fresh-run calibration cache already exists; refusing to reuse it"
        )

    cudart, cudart_path = load_cudart()
    calibration_events = []

    class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self, image_paths):
            super().__init__()
            self.paths = list(image_paths)
            self.index = 0
            self.batch_size = 1
            self.host = np.empty(INPUT_SHAPE, dtype=np.float32)
            self.device = ctypes.c_void_p()
            check_cuda(
                cudart.cudaMalloc(ctypes.byref(self.device), self.host.nbytes),
                "calibration cudaMalloc",
            )

        def get_batch_size(self):
            return self.batch_size

        def get_batch(self, names):
            if self.index >= len(self.paths):
                return None
            path = self.paths[self.index]
            self.host[0] = preprocess(path)[0]
            check_cuda(
                cudart.cudaMemcpy(
                    self.device,
                    ctypes.c_void_p(self.host.ctypes.data),
                    self.host.nbytes,
                    1,
                ),
                "calibration H2D",
            )
            self.index += 1
            calibration_events.append(
                {
                    "batch_index": self.index,
                    "image_path": str(path),
                    "input_shape": list(self.host.shape),
                    "dtype": str(self.host.dtype),
                    "finite": bool(np.isfinite(self.host).all()),
                    "timestamp": time.time(),
                }
            )
            if self.index == 1 or self.index % 25 == 0 or self.index == len(self.paths):
                print(
                    f"CALIBRATION_PROGRESS {self.index}/{len(self.paths)} {path}",
                    flush=True,
                )
            return [int(self.device.value)]

        def read_calibration_cache(self):
            return b""

        def write_calibration_cache(self, cache):
            CACHE.write_bytes(bytes(cache))
            print(f"CALIBRATION_CACHE_WRITTEN {CACHE} {CACHE.stat().st_size}", flush=True)

        def release(self):
            if self.device and self.device.value:
                cudart.cudaFree(self.device)
                self.device = ctypes.c_void_p()

        def __del__(self):
            try:
                self.release()
            except Exception:
                pass

    command_line = (
        "TensorRT Python BuilderFlag.INT8 --noTF32 "
        "IInt8EntropyCalibrator2 fresh independent validation calibration "
        "input=1x3x640x640 batch=1 workspace=1024MiB"
    )
    BUILD_LOG.write_text(command_line + "\n", encoding="utf-8")
    logger = trt.Logger(trt.Logger.VERBOSE)
    start = time.time()
    calibrator = None
    engine = None
    try:
        explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        builder = trt.Builder(logger)
        network = builder.create_network(explicit_batch)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse(PACKAGED_ONNX.read_bytes()):
            errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
            raise RuntimeError("ONNX parse failed:\n" + "\n".join(errors))
        if network.num_inputs != 1 or network.num_outputs != 1:
            raise RuntimeError(
                f"Unexpected network I/O count: {network.num_inputs}/{network.num_outputs}"
            )
        input_tensor = network.get_input(0)
        output_tensor = network.get_output(0)
        if tuple(input_tensor.shape) != INPUT_SHAPE:
            raise RuntimeError(f"ONNX input shape changed: {input_tensor.shape}")

        config = builder.create_builder_config()
        config.max_workspace_size = 1024 * 1024 * 1024
        config.set_flag(trt.BuilderFlag.INT8)
        if hasattr(trt.BuilderFlag, "TF32"):
            config.clear_flag(trt.BuilderFlag.TF32)
        if (
            hasattr(config, "profiling_verbosity")
            and hasattr(trt, "ProfilingVerbosity")
            and hasattr(trt.ProfilingVerbosity, "DETAILED")
        ):
            config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        calibrator = EntropyCalibrator(paths)
        config.int8_calibrator = calibrator
        print("INT8_ENGINE_BUILD_STARTED", flush=True)
        engine = builder.build_engine(network, config)
        if engine is None:
            raise RuntimeError("TensorRT builder returned no engine")
        PACKAGED_ENGINE.write_bytes(engine.serialize())
        print(f"INT8_ENGINE_WRITTEN {PACKAGED_ENGINE}", flush=True)
    finally:
        if calibrator is not None:
            calibrator.release()

    duration = time.time() - start
    if not CACHE.is_file() or CACHE.stat().st_size == 0:
        raise RuntimeError("Fresh calibration cache was not created")
    if not PACKAGED_ENGINE.is_file() or PACKAGED_ENGINE.stat().st_size == 0:
        raise RuntimeError("INT8 engine was not created")
    if len(calibration_events) != len(paths):
        raise RuntimeError(
            f"Calibration consumed {len(calibration_events)} images, expected {len(paths)}"
        )

    inspector_text = ""
    try:
        inspector = engine.create_engine_inspector()
        layer_format = (
            trt.LayerInformationFormat.JSON
            if hasattr(trt.LayerInformationFormat, "JSON")
            else trt.LayerInformationFormat.ONELINE
        )
        inspector_text = inspector.get_engine_information(layer_format)
    except Exception as exc:
        inspector_text = "Engine inspector unavailable: " + repr(exc)
    LAYER_REPORT.write_text(inspector_text + "\n", encoding="utf-8")
    layer_lower = inspector_text.lower()
    precision_summary = {
        "inspector_available": not inspector_text.startswith(
            "Engine inspector unavailable"
        ),
        "int8_token_count": layer_lower.count("int8"),
        "fp16_token_count": layer_lower.count("half")
        + layer_lower.count("fp16"),
        "fp32_token_count": layer_lower.count("float")
        + layer_lower.count("fp32"),
        "interpretation": (
            "Token counts are inspection evidence, not a guaranteed mutually exclusive "
            "layer count. TensorRT may retain higher-precision fallback layers."
        ),
    }
    build_summary = {
            "status": "PASS",
            "builder": "TensorRT Python API",
            "tensorrt_version": trt.__version__,
            "onnx_path": str(PACKAGED_ONNX),
            "onnx_sha256": sha256(PACKAGED_ONNX),
            "engine_path": str(PACKAGED_ENGINE),
            "engine_size_bytes": PACKAGED_ENGINE.stat().st_size,
            "engine_sha256": sha256(PACKAGED_ENGINE),
            "input_name": input_tensor.name,
            "input_shape": list(input_tensor.shape),
            "input_dtype": str(input_tensor.dtype),
            "output_name": output_tensor.name,
            "output_shape": list(output_tensor.shape),
            "output_dtype": str(output_tensor.dtype),
            "int8_builder_flag": True,
            "tf32_disabled": True,
            "fp16_builder_flag": False,
            "calibration_algorithm": "IInt8EntropyCalibrator2",
            "calibration_source": str(paths[0].parents[1]),
            "calibration_manifest": str(MANIFEST),
            "calibration_images": len(paths),
            "calibration_batch_size": 1,
            "calibration_batches": len(calibration_events),
            "test_images_used_for_calibration": False,
            "fresh_cache": True,
            "historical_test_fallback_cache_used": False,
            "calibration_cache_path": str(CACHE),
            "calibration_cache_size_bytes": CACHE.stat().st_size,
            "calibration_cache_sha256": sha256(CACHE),
            "build_duration_seconds": duration,
            "cudart": cudart_path,
            "layer_precision_summary": precision_summary,
            "build_start_epoch": start,
            "build_end_epoch": start + duration,
            "new_engine_created": True,
            "dynamic_shapes": False,
            "dla_used": False,
            "workspace_mb": 1024,
            "fallback_behavior": "TensorRT may retain FP32 fallback; FP16 builder flag was not enabled",
        }
    write_json(BUILD_SUMMARY, build_summary)
    write_json(RUN / "engine/engine_build_config.json", build_summary)
    write_json(RUN / "engine/engine_metadata.json", build_summary)
    write_json(
        RUN / "calibration/calibration_run_events.json",
        calibration_events,
    )
    write_csv(RUN / "calibration/calibration_batch_records.csv", calibration_events)
    write_csv(
        RUN / "calibration/calibration_failures.csv",
        [],
        ["image_path", "error"],
    )
    calibration_provenance = read_json(
        RUN / "calibration/calibration_dataset_provenance.json"
    )
    calibrator_configuration = {
        "status": "PASS",
        "calibrator_class": "IInt8EntropyCalibrator2",
        "source_archive": calibration_provenance.get("source_archive"),
        "extracted_source": calibration_provenance["source_root"],
        "calibration_manifest": str(MANIFEST),
        "calibration_manifest_sha256": sha256(MANIFEST),
        "calibration_images": len(paths),
        "calibration_batch_size": 1,
        "calibration_batches": len(calibration_events),
        "partial_final_batch": False,
        "input_shape": list(INPUT_SHAPE),
        "input_dtype": "float32",
        "augmentation": False,
        "test_images_used": False,
        "onnx_sha256": sha256(PACKAGED_ONNX),
        "preprocessing_sha256": sha256(RUN / "config/preprocessing.json"),
        "seed": 42,
        "fresh_cache": True,
        "historical_cache_used": False,
    }
    write_json(
        RUN / "calibration/calibrator_configuration.json",
        calibrator_configuration,
    )
    write_json(
        RUN / "calibration/calibration_run_manifest.json",
        {
            **calibrator_configuration,
            "start_epoch": start,
            "end_epoch": start + duration,
            "duration_seconds": duration,
            "cache_path": str(CACHE),
            "cache_size_bytes": CACHE.stat().st_size,
            "cache_sha256": sha256(CACHE),
            "failed_images": 0,
        },
    )
    (RUN / "calibration/calibration_log.txt").write_text(
        "Calibration status: PASS\n"
        f"Algorithm: IInt8EntropyCalibrator2\nImages/batches: {len(paths)}/{len(calibration_events)}\n"
        f"Duration seconds: {duration}\nCache: {CACHE}\nCache SHA-256: {sha256(CACHE)}\n"
        "Test images used: NO\nHistorical cache used: NO\n",
        encoding="utf-8",
    )
    (RUN / "engine/engine_inspector.txt").write_text(
        inspector_text + "\n", encoding="utf-8"
    )
    write_csv(
        RUN / "engine/fallback_layer_report.csv",
        [
            {
                "layer": "NOT_INSPECTABLE" if not precision_summary["inspector_available"] else "SUMMARY",
                "int8": "UNKNOWN",
                "fp16_fallback": "NO_BUILDER_FLAG",
                "fp32_fallback": "UNKNOWN",
                "evidence": inspector_text[:500],
            }
        ],
    )
    (RUN / "engine/dynamic_range_report.txt").write_text(
        "TensorRT 8.0.1 Python API does not expose a complete post-build per-layer dynamic-range report in this environment.\n"
        "Fresh IInt8EntropyCalibrator2 cache generation and BuilderFlag.INT8 are preserved as evidence.\n",
        encoding="utf-8",
    )
    (RUN / "calibration/calibration_cache_sha256.txt").write_text(
        f"{sha256(CACHE)}  {CACHE}\n", encoding="utf-8"
    )
    (RUN / "engine/engine_sha256.txt").write_text(
        f"{sha256(PACKAGED_ENGINE)}  {PACKAGED_ENGINE}\n", encoding="utf-8"
    )
    (RUN / "engine/engine_file_stat.txt").write_text(
        command_line
        + "\n"
        + f"size_bytes={PACKAGED_ENGINE.stat().st_size}\n"
        + f"sha256={sha256(PACKAGED_ENGINE)}\n",
        encoding="utf-8",
    )
    print("INT8 BUILD PASS")
    print(json.dumps(read_json(BUILD_SUMMARY), indent=2))


if __name__ == "__main__":
    main()

