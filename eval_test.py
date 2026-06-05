"""학습이 완전히 끝난 후 실행하는 최종 평가 스크립트"""

import torch
from loader import test_loader  # 우리가 연동해둔 test_loader를 가져옵니다.

# 1. 가장 성적이 좋았던 최종 모델 가중치 로드
model = load_best_safetensors("EfficientNetB0_Best_VisionGuard_v2(97.40%).safetensors")
model.eval() # 🧠 중요: 나 이제 시험 볼 거니까 드롭아웃이나 배치 정규화 가동 중지해!

correct = 0
total = 0

# 2. 수능 시험 시작 (test_loader 한 바퀴 돌리기)
with torch.no_grad(): # 역전파(Backprop) 엔진 꺼서 RTX 4070 메모리 아끼기
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

final_test_acc = correct / total
print(f"🎯 Vision Guard 최종 실전 정확도 (Test Accuracy): {final_test_acc:.2%}")