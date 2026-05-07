import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import copy
import warnings
import re # 파일명에서 숫자 추출을 위해 추가
from pathlib import Path
from tqdm import tqdm
from PIL import ImageFile
from config import DEFAULT_CONFIG
from dotenv import load_dotenv

# 1. 환경 설정 및 안정성 확보
warnings.filterwarnings("ignore", category=UserWarning)
ImageFile.LOAD_TRUNCATED_IMAGES = True


def train_model(
        model,
        criterion,
        optimizer,
        num_epochs,
        patience,
        dataset_sizes,
        dataloaders,
        device,
        initial_best_acc,
        scheduler,
        model_prefix,
    ):
    best_model_wts = copy.deepcopy(model.state_dict())
    # 불러온 파일의 정확도를 초기 최고점으로 설정
    best_acc = initial_best_acc 
    early_stop_counter = 0

    print(f"\n학습 시작 기준 정확도: {best_acc:.2%}")

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'valid']:
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
            if phase == 'valid':
                # 스케줄러에게 현재 검증 정확도를 알려줍니다.
                if scheduler is not None:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"⏱️ 스케줄러 점검: 검증 정확도 {epoch_acc:.2%} 기준으로 학습률을 확인합니다. (현재 lr={current_lr:.6f})")
                    scheduler.step(epoch_acc)
                
                if epoch_acc > best_acc:
                    previous_best_acc = best_acc
                    # 기존 베스트 파일들 삭제
                    prefix = model_prefix
                    for file in os.listdir('.'):
                        if file.startswith(prefix) and file.endswith(".pth"):
                            try:
                                os.remove(file)
                            except:
                                pass

                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    
                    # 새 파일명 저장
                    current_filename = f'{model_prefix}({best_acc:.2%}).pth'
                    torch.save(best_model_wts, current_filename)
                    
                    print(f"🔥 신기록 달성! 기존 성적 {previous_best_acc:.2%}에서 신기록 성적 {best_acc:.2%}로 갱신하여 모델 저장 완료: {current_filename}")
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    print(f"현재 성적({epoch_acc:.2%})이 기존 최고 성적({best_acc:.2%})에 미치지 못해 저장하지 않습니다.")

        if early_stop_counter >= patience:
            print(f"\n성능 개선 없음 ({patience} epochs). 조기 종료!")
            break

    return model

def predict_drowsiness(image_tensor, ear_value):
    """
    학습된 모델과 EAR 수치를 결합하여 최종 판단을 내리는 함수 (실전용)
    """
    # 1. 비전 모델 예측
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        prob = torch.softmax(output, dim=1)
        vision_score = prob[0][0].item() # 졸음(drowsy) 클래스 확률

    # 2. EAR 기반 보정 (EAR이 낮을수록 졸음 확률 가산)
    ear_threshold = 0.22
    ear_bonus = 0.2 if ear_value < ear_threshold else 0.0
    
    # 3. 최종 판단
    final_score = (vision_score * 0.8) + ear_bonus
    
    return "Drowsy" if final_score > 0.6 else "Normal"

if __name__ == '__main__':
    load_dotenv()
    config = DEFAULT_CONFIG
    # 데이터 경로는 VG_DATA_ROOT 기준으로만 구성합니다.
    # (선택) VG_DATASET_REL: VG Data 루트 + 상대경로 (기본값: 데이터 전처리 파일\\dataset_final_v2)
    vg_data_root_raw = os.getenv("VG_DATA_ROOT", "").strip().strip('"').strip("'")
    if not vg_data_root_raw:
        raise ValueError(
            "필수 환경변수 'VG_DATA_ROOT'가 설정되지 않았습니다. "
            ".env 또는 실행 환경에 값을 넣어주세요. "
        )
    vg_dataset_rel = (
        os.getenv("VG_DATASET_REL", r"데이터 전처리 파일\dataset_final_v2")
        .strip()
        .strip('"')
        .strip("'")
    )
    data_dir = Path(vg_data_root_raw) / vg_dataset_rel

    train_dir = data_dir / "train"
    valid_dir = data_dir / "valid"

    if not train_dir.exists() or not valid_dir.exists():
        raise ValueError(
            "학습 데이터 폴더를 찾지 못했습니다. 아래 경로에 'train', 'valid' 폴더가 있어야 합니다.\n"
            f"- 계산된 경로: {data_dir}\n"
            f"- 확인: train={train_dir.exists()}, valid={valid_dir.exists()}\n"
            "- 해결: VG_DATA_ROOT / VG_DATASET_REL(상대경로)을 올바르게 지정하세요."
        )
    batch_size = config["BATCH_SIZE"]
    num_workers = config["NUM_WORKERS"]
    num_epochs = config["NUM_EPOCHS"]
    early_stop_patience = config["EARLY_STOP_PATIENCE"]
    learning_rate = config["LEARNING_RATE"]
    scheduler_factor = config["SCHEDULER_FACTOR"]
    scheduler_patience = config["SCHEDULER_PATIENCE"]
    model_prefix = config["MODEL_PREFIX"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. 데이터셋 로드 설정
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((448, 448)),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(15),       # 고개 꺾임 대비 (최대 15도)
            # transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)), # 초점 흐려짐 대비
            # transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2), # 조명 변화
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {
        x: datasets.ImageFolder(str(data_dir / x), data_transforms[x]) for x in ["train", "valid"]
    }
    dataloaders = {
        x: DataLoader(
            image_datasets[x],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False
        )
        for x in ["train", "valid"]
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "valid"]}

    # 3. 가중치 탐색 및 최고 정확도 추출 로직
    prefix = model_prefix
    checkpoint_path = None
    best_acc_from_file = 0.0
    model_out_dir = Path(".")

    # 현재 폴더 내 파일 중 가장 높은 정확도를 가진 파일 찾기
    for file in model_out_dir.iterdir():
        if file.is_file() and file.name.startswith(prefix) and file.suffix == ".pth":
            # 정규표현식으로 (88.77%) 형태에서 숫자 추출
            match = re.search(r"\((\d+\.?\d*)%\)", file.name)
            if match:
                acc_val = float(match.group(1)) / 100.0
                if acc_val > best_acc_from_file:
                    best_acc_from_file = acc_val
                    checkpoint_path = file

    # 4. 모델 설정 및 로드
    # model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    # model.classifier[1] = nn.Linear(model.last_channel, 2)
    # model = model.to(device)

    # 기존 MobileNet_V2 대신 ResNet50을 사용
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model = model.to(device)

    if checkpoint_path:
        model.load_state_dict(torch.load(str(checkpoint_path), weights_only=True))
        print(f"가중치 로드 성공: {checkpoint_path} (기존 기록 {best_acc_from_file:.2%}부터 시작)")
    else:
        print("기존 가중치 파일이 없습니다. 0.0%부터 학습을 시작합니다.")

    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 데이터 비율을 고려하여 졸음에 1.6배 가중치 부여
    weights = torch.tensor([2, 1.0], device=device) # [drowsy, normal] 순서
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    # 스케줄러: 3에포크 동안 정확도가 안 오르면 lr을 1/10로 감소
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=scheduler_factor,
        patience=scheduler_patience,
    )

    # 5. 학습 시작 (추출한 정확도를 인자로 전달)
    model_ft = train_model(
        model, criterion, optimizer,
        num_epochs=num_epochs,
        patience=early_stop_patience,
        dataset_sizes=dataset_sizes,
        dataloaders=dataloaders,
        device=device,
        initial_best_acc=best_acc_from_file,
        scheduler=scheduler,
        model_prefix=model_prefix,
    )