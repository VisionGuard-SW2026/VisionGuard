import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import copy
import warnings
from tqdm import tqdm
from PIL import ImageFile

# 1. 환경 설정 및 안정성 확보
# 불필요한 UserWarning 차단 (MobileNetV2 weights 관련 등)
warnings.filterwarnings("ignore", category=UserWarning)
# 이미지 로드 중 스트림 끊김(Truncated) 에러 방지
ImageFile.LOAD_TRUNCATED_IMAGES = True

def train_model(model, criterion, optimizer, num_epochs=25, patience=5, dataset_sizes=None, dataloaders=None, device=None):
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

            # tqdm 진행바 설정
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

            # 검증 단계: 성능 개선 시 가중치 저장 및 조기 종료 체크
            if phase == 'val':
                if epoch_acc > best_acc:
                    # 이전 베스트 파일이 있다면 삭제 (폴더가 지저분해지는 것을 방지)
                    old_checkpoint = f'best_vision_guard_v2({best_acc:.2%}).pth'
                    if os.path.exists(old_checkpoint):
                        os.remove(old_checkpoint)
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

                    # 새로운 파일명 생성: best_vision_guard_v2(88.77%).pth 형식
                    # .2%는 0.8877을 88.77%로 변환해줍니다.
                    current_filename = f'best_vision_guard_v2({best_acc:.2%}).pth'

                    # 성능이 개선될 때마다 'best_vision_guard_v2.pth' 갱신
                    torch.save(best_model_wts, current_filename)
                    print(f"새로운 베스트 모델 저장 완료! (Acc: {best_acc:.4f})")
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f"\n성능 개선 없음 ({patience} epochs). 조기 종료 발동!")
            break

    print(f'\n최종 검증 정확도: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    # --- [Windows 멀티프로세싱 중복 출력 방지를 위해 메인 블록 내부에서 실행] ---
    
    # 데이터 경로 및 하이퍼파라미터
    data_dir = r"C:\Users\임상혁\Desktop\VisionGuard\VG Data\데이터 전처리 파일\dataset_final_v2"
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"학습 재개 준비! 사용 GPU: {torch.cuda.get_device_name(0)}")

    # 2. 데이터 전처리 및 로드
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}
    # num_workers=4 설정으로 990 PRO 속도 활용
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    
    print(f"학습 데이터: {dataset_sizes['train']}장, 검증 데이터: {dataset_sizes['val']}장")
    print(f"클래스 분류: {image_datasets['train'].classes}")

    # 3. 모델 설정 (MobileNetV2)
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    model = model.to(device)

    # 4. 이전 학습 가중치 불러오기 (Resume) - 파일명 유동적 대응
    prefix = "best_vision_guard_v2"
    checkpoint_path = None

    # 현재 폴더에서 해당 접두사로 시작하는 가장 최신 .pth 파일 찾기
    for file in os.listdir('.'):
        if file.startswith(prefix) and file.endswith(".pth"):
            checkpoint_path = file
            break # 가장 먼저 발견된 파일을 타겟으로 설정
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        # 보안 경고 해결을 위해 weights_only=True 권장 (PyTorch 최신버전 대응)
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        print(f"가중치 로드 성공: {checkpoint_path} (이전 학습 내용을 이어서 시작합니다.)")
    else:
        print("저장된 모델이 없습니다. 처음부터 학습을 시작합니다.")

    # 5. 손실 함수 및 옵티마이저 (lr=0.0001 유지)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 6. 학습 시작 (최대 200 에포크, 조기종료 5회 적용)
    model_ft = train_model(
        model, criterion, optimizer, 
        num_epochs=200, patience=5, 
        dataset_sizes=dataset_sizes, 
        dataloaders=dataloaders, 
        device=device
    )