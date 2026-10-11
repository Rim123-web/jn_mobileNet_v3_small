# Performance

{
  "measured_images": 261,
  "warmup_count": 10,
  "mean_engine_latency_ms": 33.14579384171643,
  "mean_transfer_inclusive_latency_ms": 35.83714222171526,
  "mean_end_to_end_latency_ms": 117.6272600421456,
  "engine_fps": 30.169740534059137,
  "transfer_inclusive_fps": 27.904010699660564,
  "latency_based_pipeline_fps": 8.501430702727431,
  "official_measured_wall_time_seconds": 30.80849886299984,
  "wall_clock_pipeline_fps": 8.471688320830646,
  "engine_percentiles_ms": {
    "p50": 33.11317825317383,
    "p90": 33.20635223388672,
    "p95": 33.2314567565918,
    "p99": 33.73277206420898
  },
  "end_to_end_percentiles_ms": {
    "p50": 111.862761,
    "p90": 160.838537,
    "p95": 194.49233,
    "p99": 236.1881907999998
  },
  "fps_definitions": {
    "engine_fps": "1000 / mean TensorRT execution-only CUDA-event latency",
    "transfer_inclusive_fps": "1000 / mean(H2D + TensorRT + D2H)",
    "latency_based_pipeline_fps": "1000 / mean complete file-open-to-classification latency",
    "wall_clock_pipeline_fps": "N / official measured process wall time"
  }
}
