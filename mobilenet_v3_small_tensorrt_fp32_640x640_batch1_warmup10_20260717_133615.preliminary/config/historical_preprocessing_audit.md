# Historical MobileNetV3-Small Audit

- Executed source: `/home/jetson/Downloads/classification_jetson/mobilenet_v3_small_fp32_core4_benchmark/scripts/run_mobilenet_v3_small_fp32_core4_benchmark.py`
- Architecture: MobileNetV3-Small, three classes.
- Original ONNX: `/home/jetson/Downloads/classification_jetson/mobilenet_v3_small_inference_models/mobilenet_v3_small_inference.onnx` (`c88ff27f9e94950204921d529cb7c49c16a35c60907c983a7b281e8f77782ed8`).
- Selected TensorRT-compatible ONNX: `/home/jetson/Downloads/classification_jetson/mobilenet_v3_small_fp32_core4_benchmark/exports/mobilenet_v3_small_inference_fp32_trt_compatible.onnx` (`28ef2fe9519b184d0e3c41690726c8911157a103c6ec66ccf8b86387de1d47e7`).
- Original limitation: TensorRT 8.0.1 did not import HardSwish.
- Existing fix: Hardswish was decomposed into primitive operations from the same checkpoint; no training occurred.
- Preprocessing: RGB, direct PIL bilinear 640x640 resize, /255, ImageNet mean/std, HWC-to-CHW FP32.
- Preprocessing ambiguity: none.
