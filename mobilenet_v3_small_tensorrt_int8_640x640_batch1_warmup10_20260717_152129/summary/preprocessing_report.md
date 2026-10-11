# Preprocessing

- Validation source: `/home/jetson/Downloads/classification_benchmark/mobileNet_v3_small/calibration_sources/val_mobilenet_fce132c3a7172452/val`
- Calibration images: 260
- Class distribution: {'class_1_asymptomatic': 155, 'class_2_mild_symptoms': 70, 'class_3_symptomatic_compromised': 35}
- Test overlap: 0
- Mean: `[0.485, 0.456, 0.406]`
- Std: `[0.229, 0.224, 0.225]`
- RGB, divide by 255, direct PIL bilinear resize, HWC-to-CHW
- Final shape: `[1,3,640,640]`
- FP32/FP16/INT8 preprocessing parity: PASS
