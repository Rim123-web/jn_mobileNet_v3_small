# Methodology

- Device: NVIDIA Jetson Nano Developer Kit / Tegra X1.
- Model: MobileNetV3-Small classification, TensorRT FP16-enabled (`--fp16`, `--noTF32`, no INT8).
- Source artifact: the exact ONNX from the validated FP32 final package; no re-export occurred.
- Host input/output bindings remain FP32 for preprocessing parity; internal layer precision and fallback evidence are recorded separately.
- Input: batch 1, RGB `[1,3,640,640]`.
- Preprocessing: PIL decode, RGB, direct bilinear resize, divide by 255,
  mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`, HWC-to-CHW.
- Warm-up: exactly 10 engine executions, excluded from all official timing and monitoring.
- Measured pass: 261 validated images exactly once.
- CPU stages: `time.perf_counter_ns()`.
- H2D, TensorRT, D2H: CUDA events with synchronization at stage boundaries.
- Power boundary: Jetson module-level jtop Power TOT (POM_5V_IN), sampled at 0.5 s.
- Energy: trapezoidal integration of timestamped Power TOT samples.
- FP32-versus-FP16 protocol parity: PASS.
