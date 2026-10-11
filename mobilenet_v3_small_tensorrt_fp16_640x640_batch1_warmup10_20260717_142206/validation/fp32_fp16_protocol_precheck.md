# FP32/FP16 Protocol Precheck

- status: PASS
- fp32_reference_package: /home/jetson/Downloads/classification_benchmark/mobileNet_v3_small/mobilenet_v3_small_tensorrt_fp32_640x640_batch1_warmup10_20260717_133615
- fp32_validation: PASS
- onnx_sha256_match: True
- preprocessing_config_match: True
- preprocess_detailed_function_source_match: True
- official_benchmark_script_content_match: True
- monitor_script_hash_match: True
- tensorrt_runtime_script_hash_match: True
- dataset_parity: PASS
- class_mapping_match: True
- input_shape: [1, 3, 640, 640]
- batch_size: 1
- warmup_count: 10
- experimental_variable: TensorRT FP16 builder flag/internal tactic precision
