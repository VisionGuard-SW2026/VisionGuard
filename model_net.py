import torch.nn as nn
from torchvision import models
from config import DEFAULT_CONFIG as config

def get_model(model_name = config, num_classes=2):
    """
    설정값에 따라 최적화된 졸음 판별 모델을 반환합니다.
    model_name이 None일 경우 config.py의 기본 설정을 따릅니다.
    """
    if model_name == "MobileNetV2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.last_channel, 2)

    elif model_name == "ResNet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    
    elif model_name == "EfficientNetB0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 2)
    
    else:
        raise ValueError(f"지원하지 않는 모델 이름입니다: {model_name}")
    
    return model