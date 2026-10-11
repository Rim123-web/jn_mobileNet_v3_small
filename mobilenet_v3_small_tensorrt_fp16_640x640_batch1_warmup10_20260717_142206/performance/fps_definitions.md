# FPS Definitions

- **engine_fps**: 1000 / mean TensorRT execution-only CUDA-event latency
- **transfer_inclusive_fps**: 1000 / mean(H2D + TensorRT + D2H)
- **latency_based_pipeline_fps**: 1000 / mean complete file-open-to-classification latency
- **wall_clock_pipeline_fps**: N / official measured process wall time
