import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import copy
import warnings
import re # 파일명에서 숫자 추출을 위해 추가
from tqdm import tqdm
from PIL import ImageFile

# 1. 환경 설정 및 안정성 확보
warnings.filterwarnings("ignore", category=UserWarning)
ImageFile.LOAD_TRUNCATED_IMAGES = True

def train_model(model, criterion, optimizer, num_epochs=25, patience=5, dataset_sizes=None, dataloaders=None, device=None, initial_best_acc=0.0, scheduler=None):
    best_model_wts = copy.deepcopy(model.state_dict())
    # 불러온 파일의 정확도를 초기 최고점으로 설정
    best_acc = initial_best_acc 
    early_stop_counter = 0

    print(f"\n학습 시작 기준 정확도: {best_acc:.2%}")

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

            # 검증 단계: 이전 최고 기록(best_acc)을 넘을 때만 저장
            if phase == 'val':
                # 스케줄러에게 현재 검증 정확도를 알려줍니다.
                if scheduler is not None:
                    scheduler.step(epoch_acc)
                
                if epoch_acc > best_acc:
                    # 기존 베스트 파일들 삭제
                    prefix = "best_vision_guard_v2"
                    for file in os.listdir('.'):
                        if file.startswith(prefix) and file.endswith(".pth"):
                            try:
                                os.remove(file)
                            except:
                                pass

                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    
                    # 새 파일명 저장
                    current_filename = f'best_vision_guard_v2({best_acc:.2%}).pth'
                    torch.save(best_model_wts, current_filename)
                    
                    print(f"🔥 신기록 달성! 모델 저장 완료: {current_filename}")
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    print(f"현재 성적({epoch_acc:.2%})이 기존 최고 성적({best_acc:.2%})에 미치지 못해 저장하지 않습니다.")

        if early_stop_counter >= patience:
            print(f"\n성능 개선 없음 ({patience} epochs). 조기 종료!")
            break

    return model

if __name__ == '__main__':
    data_dir = r"C:\Users\임상혁\Desktop\VisionGuard\VG Data\데이터 전처리 파일\dataset_final_v2"
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. 데이터셋 로드 설정
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),       # 고개 꺾임 대비 (최대 15도)
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)), # 초점 흐려짐 대비
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2), # 더 강한 조명 변화
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
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    # 3. 가중치 탐색 및 최고 정확도 추출 로직
    prefix = "best_vision_guard_v2"
    checkpoint_path = None
    best_acc_from_file = 0.0

    # 현재 폴더 내 파일 중 가장 높은 정확도를 가진 파일 찾기
    for file in os.listdir('.'):
        if file.startswith(prefix) and file.endswith(".pth"):
            # 정규표현식으로 (88.77%) 형태에서 숫자 추출
            match = re.search(r"\((\d+\.?\d*)%\)", file)
            if match:
                acc_val = float(match.group(1)) / 100.0
                if acc_val > best_acc_from_file:
                    best_acc_from_file = acc_val
                    checkpoint_path = file

    # 4. 모델 설정 및 로드
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    model = model.to(device)

    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        print(f"가중치 로드 성공: {checkpoint_path} (기존 기록 {best_acc_from_file:.2%}부터 시작)")
    else:
        print("기존 가중치 파일이 없습니다. 0.0%부터 학습을 시작합니다.")

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    
    # 스케줄러 정의: 3에포크 동안 정확도 안 오르면 lr을 1/10로 감소
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)

    # 5. 학습 시작 (추출한 정확도를 인자로 전달)
    model_ft = train_model(
        model, criterion, optimizer,
        num_epochs=200,
        patience=10,
        dataset_sizes=dataset_sizes,
        dataloaders=dataloaders,
        device=device,
        initial_best_acc=best_acc_from_file,
        scheduler=scheduler
    )