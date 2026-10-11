# Calibration Selection

- Source: `/home/jetson/Downloads/classification_benchmark/mobileNet_v3_small/calibration_sources/val_mobilenet_fce132c3a7172452/val`
- Split: validation
- Source images: 260
- Selected images: 90 (30 per class)
- Method: deterministic evenly-spaced selection over each class-sorted path list
- Random sampling: NO (seed 42 recorded for protocol completeness)
- Reason: observed calibration cost made all 260 images unreasonable; 90 is within the requested preferred range.
- Exact test overlap: 0
- Test images used: NO
