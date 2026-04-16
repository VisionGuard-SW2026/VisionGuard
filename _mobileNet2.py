import torch
import torchvision.models as models

# 1. MobileNetV2 모델 불러오기 (사전 학습된 가중치 포함)
model = models.mobilenet_v2(weights='DEFAULT')

# 2. 마지막 출력 레이어를 2개(정상, 졸음)로 변경 
model.classifier[1] = torch.nn.Linear(model.last_channel, 2)

# 3. GPU 사용 설정 (RTX 40 시리즈 활용!)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"현재 사용 중인 장치: {device}")