DEFAULT_CONFIG = {
    "MODEL_NAME": "EfficientNetB0", # "MobileNetV2", "ResNet50", "EfficientNetB0" 중 선택
    "OPTIMIZER_NAME": "SGD",        # "SGD" 또는 "Adam"
    "MOMENTUM": 0.9,                # SGD 전용
    "WEIGHT_DECAY": 1e-4,           # 가중치 감쇠 (L2 규제)
    "BATCH_SIZE": 32,
    "NUM_WORKERS": 2,
    "NUM_EPOCHS": 200,
    "LEARNING_RATE": 0.0001,
    "EARLY_STOP_PATIENCE": 10,
    "EAR_DATA_PATH": "ears.npy",
    "LABEL_DATA_PATH": "labels.npy",
    "ENSEMBLE_WEIGHT_VISION": 0.75, # 비전 모델 비중
    "ENSEMBLE_WEIGHT_EAR": 0.25,    # EAR 수치 비중
    "SCHEDULER_FACTOR": 0.1,
    "SCHEDULER_PATIENCE": 3,
    "MODEL_PREFIX": "Best_VisionGuard_v2",
}
