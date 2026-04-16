import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import copy
from tqdm import tqdm
from PIL import ImageFile

# 이미지 로드 오류 방지
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 1. 환경 설정 및 하이퍼파라미터
# 상혁님의 새 데이터셋 경로로 설정합니다.
data_dir = r"C:\Users\임상혁\Desktop\VisionGuard\VG Data\데이터 전처리 파일\dataset_final_v2"
batch_size = 64  # RTX 4070의 12GB VRAM을 고려하여 64로 상향 조정
num_epochs = 20
patience = 5     # Early Stopping 기준
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 데이터 전처리 (MobileNetV2 최적화)
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(), # 데이터 다양성 확보
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # 조명 변화 대응 (실제도로 환경)
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 3. 데이터셋 로드
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

print(f"학습 데이터: {dataset_sizes['train']}장, 검증 데이터: {dataset_sizes['val']}장")
print(f"클래스: {class_names}")

# 4. 모델 설정 (MobileNetV2 Pre-trained 사용)
model = models.mobilenet_v2(pretrained=True)
# 마지막 레이어를 2개 클래스(normal, drowsy)로 변경
model.classifier[1] = nn.Linear(model.last_channel, 2)
model = model.to(device)

# 5. 손실 함수 및 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 6. 학습 함수
def train_model(model, criterion, optimizer, num_epochs=25, patience=3):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    early_stop_counter = 0

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # tqdm으로 실시간 진행률 표시
            pbar = tqdm(dataloaders[phase], desc=f'{phase} phase')
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                pbar.set_postfix(loss=loss.item())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 검증 단계에서 성능 개선 확인 및 조기 종료 체크
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, 'best_vision_guard_v2.pth')
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1

        if early_stop_counter >= patience:
            print("성능 개선 없음. 조기 종료 발동!")
            break

    print(f'\n최종 검증 정확도: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    # 1. GPU 및 모델 초기 설정 (기존과 동일)
    print(f"학습 재개 준비! 사용 GPU: {torch.cuda.get_device_name(0)}")
    
    # 모델 구조 정의
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    model = model.to(device)

    # 2. [핵심] 저장된 10 에포크 가중치 불러오기
    checkpoint_path = 'best_vision_guard_v2.pth'
    if os.path.exists(checkpoint_path):
        print(f"가중치 로드 중: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path))
        print("학습 내용을 성공적으로 불러왔습니다.")
    else:
        print("저장된 모델 파일을 찾을 수 없어 처음부터 시작합니다.")

    # 3. 옵티마이저 설정 (이전과 동일한 lr 권장)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    # 4. 학습 함수 호출
    # num_epochs를 30으로 두면, 11회차부터 30회차까지 진행됩니다.
    model_ft = train_model(model, criterion, optimizer, num_epochs=20, patience=5)