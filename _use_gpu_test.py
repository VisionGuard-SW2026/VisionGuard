import torch

# 1. CUDA 사용 가능 여부 확인
gpu_available = torch.cuda.is_available()
print(f"GPU 가속 가능: {gpu_available}")

# 2. 연결된 GPU 이름 확인
if gpu_available:
    print(f"사용 중인 장치: {torch.cuda.get_device_name(0)}")