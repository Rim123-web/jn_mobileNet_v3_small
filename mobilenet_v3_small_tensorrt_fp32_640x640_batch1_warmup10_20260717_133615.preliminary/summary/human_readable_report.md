# Publication Results: MobileNetV3-Small TensorRT FP32 on Jetson Nano

| Item | Result |
|---|---:|
| Images | 261 |
| Accuracy | 0.973180 |
| Macro precision | 0.947315 |
| Macro recall | 0.957672 |
| Macro F1 | 0.951857 |
| Weighted F1 | 0.973423 |
| Engine latency | 33.146 ms |
| End-to-end latency | 117.627 ms |
| Engine FPS | 30.170 |
| Transfer-inclusive FPS | 27.904 |
| Wall-clock pipeline FPS | 8.472 |
| Average Power TOT | 5.215 W |
| Gross energy/image | 0.617 J |
| Idle-corrected energy/image | 0.297 J |

# Methodology

- Device: NVIDIA Jetson Nano Developer Kit / Tegra X1.
- Model: MobileNetV3-Small classification, TensorRT strict FP32 (`--noTF32`, no FP16, no INT8).
- Source artifact: existing Hardswish-decomposed ONNX, with the original export preserved separately.
- Input: batch 1, RGB `[1,3,640,640]`.
- Preprocessing: PIL decode, RGB, direct bilinear resize, divide by 255,
  mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`, HWC-to-CHW.
- Warm-up: exactly 10 engine executions, excluded from all official timing and monitoring.
- Measured pass: 261 validated images exactly once.
- CPU stages: `time.perf_counter_ns()`.
- H2D, TensorRT, D2H: CUDA events with synchronization at stage boundaries.
- Power boundary: Jetson module-level jtop Power TOT (POM_5V_IN), sampled at 0.5 s.
- Energy: trapezoidal integration of timestamped Power TOT samples.
- ResNet50 comparison protocol parity: PASS.
