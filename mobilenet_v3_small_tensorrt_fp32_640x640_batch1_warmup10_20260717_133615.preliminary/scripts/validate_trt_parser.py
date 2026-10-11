from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy  # Load the clean venv NumPy before exposing only the TensorRT package.


RUN = Path(__file__).resolve().parents[1]
USER_SITE = "/home/jetson/.local/lib/python3.8/site-packages"
sys.path.insert(0, USER_SITE)
import tensorrt as trt
sys.path.remove(USER_SITE)

CANDIDATES = [
    Path("/home/jetson/Downloads/classification_jetson/mobilenet_v3_small_inference_models/mobilenet_v3_small_inference.onnx"),
    RUN / "onnx/mobilenet_v3_small_source.onnx",
]
logger = trt.Logger(trt.Logger.ERROR)
rows = []
for path in CANDIDATES:
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    success = parser.parse(path.read_bytes())
    errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
    rows.append({
        "path": str(path), "parse_success": bool(success), "errors": errors,
        "inputs": [{"name": network.get_input(i).name, "shape": list(network.get_input(i).shape), "dtype": str(network.get_input(i).dtype)} for i in range(network.num_inputs)],
        "outputs": [{"name": network.get_output(i).name, "shape": list(network.get_output(i).shape), "dtype": str(network.get_output(i).dtype)} for i in range(network.num_outputs)],
    })
(RUN / "logs/trt_parser_validation.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
(RUN / "logs/onnx_validation_stdout.log").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
(RUN / "logs/onnx_validation_stderr.log").write_text("\n".join(error for row in rows for error in row["errors"]) + "\n", encoding="utf-8")
selected = rows[-1]
if not selected["parse_success"] or selected["inputs"][0]["shape"] != [1, 3, 640, 640] or selected["outputs"][0]["shape"] != [1, 3]:
    raise RuntimeError(f"Selected ONNX TensorRT parse failed: {selected}")
print(json.dumps(rows, indent=2))
print("TENSORRT PARSER VALIDATION PASS")
