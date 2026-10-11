#!/usr/bin/env bash
set -euo pipefail
RUN="$(cd "$(dirname "$0")" && pwd)"
PYTHONNOUSERSITE=1 /home/jetson/Downloads/jetson_nano/.venv_yolo/bin/python "$RUN/scripts/prepare_int8_project.py"
PYTHONPATH=/home/jetson/.local/lib/python3.8/site-packages /home/jetson/Downloads/jetson_nano/.venv_yolo/bin/python "$RUN/scripts/build_int8_engine.py"
PYTHONPATH=/home/jetson/.local/lib/python3.8/site-packages /home/jetson/Downloads/jetson_nano/.venv_yolo/bin/python "$RUN/scripts/validate_int8_correctness.py"
PYTHONNOUSERSITE=1 /home/jetson/Downloads/jetson_nano/.venv_yolo/bin/python "$RUN/scripts/run_idle_monitor.py"
PYTHONPATH=/home/jetson/.local/lib/python3.8/site-packages /home/jetson/Downloads/jetson_nano/.venv_yolo/bin/python "$RUN/scripts/benchmark_mobilenet_v3_small_int8_warmup10.py"
PYTHONNOUSERSITE=1 /home/jetson/Downloads/jetson_nano/.venv_yolo/bin/python "$RUN/scripts/finalize_results.py"
