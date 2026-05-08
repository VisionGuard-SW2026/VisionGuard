DEFAULT_CONFIG = {
    "MODEL_NAME": "MobileNetV2",    # "MobileNetV2" 또는 "ResNet50"
    "OPTIMIZER_NAME": "SGD",        # "SGD" 또는 "Adam"
    "MOMENTUM": 0.9,                # SGD 전용
    "WEIGHT_DECAY": 1e-4,           # 가중치 감쇠 (L2 규제)
    "BATCH_SIZE": 32,
    "NUM_WORKERS": 6,
    "NUM_EPOCHS": 200,
    "LEARNING_RATE": 0.0001,
    "EARLY_STOP_PATIENCE": 5,
    "SCHEDULER_FACTOR": 0.1,
    "SCHEDULER_PATIENCE": 3,
    "MODEL_PREFIX": "Best_VisionGuard_v2",
}
