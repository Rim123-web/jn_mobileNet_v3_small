# FP16 Correctness Validation

- Status: **PASS**
- Images: 20 deterministic samples
- FP32/FP16 top-1 agreement: 1.000000
- ONNX/FP16 top-1 agreement: 1.000000
- Minimum FP32/FP16 cosine similarity: 0.999975985
- Maximum FP32/FP16 absolute logit error: 0.0981712341
- Each TensorRT engine ran in a separate short-lived subprocess; no CUDA context was reused.
