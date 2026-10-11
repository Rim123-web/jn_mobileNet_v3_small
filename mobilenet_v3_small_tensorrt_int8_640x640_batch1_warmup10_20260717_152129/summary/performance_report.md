# Performance

{
  "measured_images": 261,
  "warmup_count": 10,
  "mean_engine_latency_ms": 33.557010811407444,
  "mean_transfer_inclusive_latency_ms": 36.43423539936771,
  "mean_end_to_end_latency_ms": 140.6151934137931,
  "engine_fps": 29.800032119072352,
  "transfer_inclusive_fps": 27.44671293465251,
  "latency_based_pipeline_fps": 7.111607044178122,
  "official_measured_wall_time_seconds": 36.80787607499951,
  "wall_clock_pipeline_fps": 7.0908736887775845,
  "engine_percentiles_ms": {
    "p50": 33.52692794799805,
    "p90": 33.600887298583984,
    "p95": 33.631874084472656,
    "p99": 33.694863891601564
  },
  "end_to_end_percentiles_ms": {
    "p50": 112.016359,
    "p90": 164.287358,
    "p95": 214.336803,
    "p99": 781.9259497999994
  },
  "fps_definitions": {
    "engine_fps": "1000 / mean TensorRT execution-only CUDA-event latency",
    "transfer_inclusive_fps": "1000 / mean(H2D + TensorRT + D2H)",
    "latency_based_pipeline_fps": "1000 / mean complete file-open-to-classification latency",
    "wall_clock_pipeline_fps": "N / official measured process wall time"
  }
}
