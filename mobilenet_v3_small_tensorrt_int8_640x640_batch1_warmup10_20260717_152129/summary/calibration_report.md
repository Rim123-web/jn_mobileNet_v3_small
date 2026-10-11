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

# Calibration/Test Overlap Verification

- Exact SHA-256 overlap: **0**
- Decoded-pixel overlap: **0**
- Conservative dHash near-duplicate candidates: **0**
- Mandatory exact-hash blocker: PASS

# Safe Validation Archive Extraction

- Archive: `/home/jetson/Downloads/val_mobilenet.zip`
- SHA-256: `fce132c3a7172452a18b068c9630591517bbe1de93f140bee580669df4323a15`
- Entries/files: 260/260
- Path traversal, absolute path, symlink, device, and duplicate checks: PASS
- Immutable extracted root: `/home/jetson/Downloads/classification_benchmark/mobileNet_v3_small/calibration_sources/val_mobilenet_fce132c3a7172452`
- Original ZIP unchanged: YES

# Validation Source Quality

- Class mapping: exact names, unambiguous
- All three classes represented: YES
- Corrupt, zero-byte, unsupported, and duplicate images: 0
- Non-image or excluded calibration samples: 0
- Domain: the independent validation split supplied for this model
