# Publication Results: MobileNetV3-Small TensorRT INT8-Enabled Execution on Jetson Nano

| Item | Result |
|---|---:|
| Images | 261 |
| Accuracy | 0.973180 |
| Macro precision | 0.947315 |
| Macro recall | 0.957672 |
| Macro F1 | 0.951857 |
| Weighted F1 | 0.973423 |
| Engine latency | 33.557 ms |
| End-to-end latency | 140.615 ms |
| Engine FPS | 29.800 |
| Transfer-inclusive FPS | 27.447 |
| Wall-clock pipeline FPS | 7.091 |
| Average Power TOT | 5.102 W |
| Gross energy/image | 0.712 J |
| Idle-corrected energy/image | 0.303 J |
| FP32/INT8 prediction agreement | 100.000000% |
| INT8 engine speedup over FP32 | 0.988x |
| INT8 pipeline speedup over FP32 | 0.837x |

# Methodology

- Device: NVIDIA Jetson Nano Developer Kit / Tegra X1.
- Model: MobileNetV3-Small classification, TensorRT INT8-enabled execution (BuilderFlag.INT8, TF32 disabled, FP16 builder flag disabled).
- Calibration: IInt8EntropyCalibrator2, source `/home/jetson/Downloads/classification_benchmark/mobileNet_v3_small/calibration_sources/val_mobilenet_fce132c3a7172452/val`, split validation.
- Calibration images: 90, batch size 1, 90 batches.
- Calibration/test exact hash overlap: 0.
- Historical test-fallback cache used: NO.
- Calibration cache: `/home/jetson/Downloads/classification_benchmark/mobileNet_v3_small/mobilenet_v3_small_tensorrt_int8_640x640_batch1_warmup10_20260717_152129/calibration/mobilenet_v3_small_int8_calibration.cache` (8eb23a265e3d43b13dadd85dffc86895b5d14530d498596a9fb99549e0af9aef).
- Input/output binding dtypes: DataType.FLOAT / ['DataType.FLOAT'].
- Input: batch 1, RGB `[1,3,640,640]`.
- Preprocessing: PIL decode, RGB, direct bilinear resize, divide by 255,
  mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`, HWC-to-CHW.
- Warm-up: exactly 10 engine executions, excluded from all official timing and monitoring.
- Measured pass: 261 validated images exactly once.
- CPU stages: `time.perf_counter_ns()`.
- H2D, TensorRT, D2H: CUDA events with synchronization at stage boundaries.
- Power boundary: Jetson module-level jtop Power TOT (POM_5V_IN), sampled at 0.5 s.
- Energy: trapezoidal integration of timestamped Power TOT samples.
- Comparison references: validated FP32 package `/home/jetson/Downloads/classification_benchmark/mobileNet_v3_small/mobilenet_v3_small_tensorrt_fp32_640x640_batch1_warmup10_20260717_133615` and FP16 package `/home/jetson/Downloads/classification_benchmark/mobileNet_v3_small/mobilenet_v3_small_tensorrt_fp16_640x640_batch1_warmup10_20260717_142206`.
