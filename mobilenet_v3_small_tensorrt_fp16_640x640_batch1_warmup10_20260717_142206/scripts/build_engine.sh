#!/usr/bin/env bash
set -euo pipefail
RUN="$(cd "$(dirname "$0")/.." && pwd)"
ONNX="$RUN/onnx/mobilenet_v3_small_source.onnx"
ENGINE="$RUN/engine/mobilenet_v3_small_tensorrt_fp16_1x3x640x640.engine"
LOG="$RUN/engine/engine_build_log.txt"
STDOUT="$RUN/logs/engine_build_stdout.log"
STDERR="$RUN/logs/engine_build_stderr.log"
CACHE="$RUN/engine/timing_cache.bin"
if [[ -e "$ENGINE" ]]; then
  echo "Refusing to overwrite existing engine: $ENGINE" >&2
  exit 3
fi
START_EPOCH="$(date +%s.%N)"
cat > "$RUN/engine/engine_build_config.json" <<EOF
{
  "builder": "/usr/src/tensorrt/bin/trtexec",
  "source_onnx": "$ONNX",
  "output_engine": "$ENGINE",
  "workspace_mb": 1024,
  "input_shape": [1, 3, 640, 640],
  "batch_size": 1,
  "fp16": true,
  "int8": false,
  "tf32": false,
  "dla": false,
  "dynamic_shapes": false,
  "input_io_format": "fp32:chw",
  "output_io_format": "fp32:chw",
  "execution_mode": "TensorRT FP16-enabled with FP32 host bindings",
  "build_only": true,
  "timing_cache": "$CACHE"
}
EOF
CMD=(/usr/src/tensorrt/bin/trtexec
  "--onnx=$ONNX"
  "--saveEngine=$ENGINE"
  "--workspace=1024"
  "--fp16"
  "--noTF32"
  "--inputIOFormats=fp32:chw"
  "--outputIOFormats=fp32:chw"
  "--timingCacheFile=$CACHE"
  "--buildOnly")
printf '%q ' "${CMD[@]}" > "$LOG"
printf '\n' >> "$LOG"
printf '%q ' "${CMD[@]}" >> "$RUN/logs/executed_commands.log"
printf '\n' >> "$RUN/logs/executed_commands.log"
"${CMD[@]}" > >(tee -a "$STDOUT" "$LOG") 2> >(tee -a "$STDERR" "$LOG" >&2)
test -s "$ENGINE"
sha256sum "$ENGINE" > "$RUN/engine/engine_sha256.txt"
END_EPOCH="$(date +%s.%N)"
python3 - "$RUN/engine/engine_build_config.json" "$ENGINE" "$START_EPOCH" "$END_EPOCH" <<'PY'
import hashlib,json,sys
from pathlib import Path
config_path, engine_path, start, end = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
config=json.loads(config_path.read_text())
h=hashlib.sha256()
with engine_path.open('rb') as f:
    for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
config.update({"status":"PASS","build_start_epoch":start,"build_end_epoch":end,"build_duration_seconds":end-start,"engine_size_bytes":engine_path.stat().st_size,"engine_sha256":h.hexdigest(),"new_engine_created":True})
config_path.write_text(json.dumps(config,indent=2)+'\n')
PY
echo "ENGINE BUILD PASS"
