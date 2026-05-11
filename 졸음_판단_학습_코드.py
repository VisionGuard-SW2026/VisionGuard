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
    best_acc = initial_best_acc 
    # 표준 Early Stopping을 위해 최소 손실값을 추적합니다.
    best_loss = float('inf') 
    early_stop_counter = 0

    print(f"\n학습 시작 기준 정확도: {best_acc:.2%}")

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 12)

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
                # 1. 스케줄러 업데이트 (정확도 기반)
                if scheduler is not None:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"⏱️ 스케줄러 점검: 검증 정확도 {epoch_acc:.2%} 기준으로 학습률을 확인합니다. (현재 lr={current_lr:.6f})")
                    scheduler.step(epoch_acc)
                
                # 2. 모델 저장 조건 (최고 정확도 경신 시 저장)
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
                    
                # 3. 표준 Early Stopping 조건 (Validation Loss 기반)
                # 정확도는 오르지 않아도 Loss가 낮아지고 있다면 모델은 계속 학습 중입니다.
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    early_stop_counter = 0 # 손실이 개선되면 카운터 초기화
                    print(f"✅ 검증 손실 개선됨 ({epoch_loss:.4f}). Early Stopping 카운터 초기화.")
                else:
                    early_stop_counter += 1 # 손실 개선 실패 시 카운트 증가
                    print(f"⚠️ 검증 손실 개선 실패. Early Stopping 카운터: {early_stop_counter}/{patience}")

        if early_stop_counter >= patience:
            print(f"\n🛑 조기 종료: 검증 손실이 {patience} 에포크 동안 개선되지 않았습니다.")
            break

    return model

def predict_drowsiness(image_tensor, ear_value, config):
    """
    비전 모델(Vision Guard)과 EAR(수치)을 결합한 앙상블 판단 함수
    """
    # 1. 비전 모델 예측 (CNN의 판단)
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        prob = torch.softmax(output, dim=1)
        # 졸음(drowsy) 클래스가 0번 인덱스라고 가정
        vision_score = prob[0][0].item() 

    # 2. EAR 기반 가중치 계산 (Step 4 앙상블 로직)
    # EAR 임계값 기준은 기존과 동일하게 0.22를 유지합니다.
    ear_threshold = 0.22
    
    # 모델의 신뢰도와 EAR의 수치를 8:2 비율로 혼합하거나, 
    # EAR이 임계값보다 낮을 때 확실한 가중치를 부여합니다.
    if ear_value < ear_threshold:
        # 눈이 감겼을 가능성이 매우 높으므로 비전 점수에 강한 가중치 합산
        # 0.2의 보너스는 89% 고지에서 90%를 뚫어줄 결정적 한 방이 됩니다.
        ear_influence = 0.25 
    else:
        ear_influence = 0.0

    # 3. 최종 앙상블 점수 산출
    # 비전 모델의 판단 75% + EAR의 수치적 증거 25% 조합
    final_score = (vision_score * 0.75) + ear_influence
    
    # 4. 최종 판정 (기준치 0.6)
    is_drowsy = final_score > 0.6
    
    return {
        "status": "Drowsy" if is_drowsy else "Normal",
        "score": final_score,
        "vision_part": vision_score,
        "ear_part": ear_influence
    }

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
    
    # 모델 설정 로드
    batch_size = config["BATCH_SIZE"]
    num_workers = config["NUM_WORKERS"]
    num_epochs = config["NUM_EPOCHS"]
    early_stop_patience = config["EARLY_STOP_PATIENCE"]
    learning_rate = config["LEARNING_RATE"]
    scheduler_factor = config["SCHEDULER_FACTOR"]
    scheduler_patience = config["SCHEDULER_PATIENCE"]
    model_name = config["MODEL_NAME"]
    model_prefix = f"{model_name}_{config['MODEL_PREFIX']}"

    # 옵티마이저 설정 로드
    optimizer_name = config.get("OPTIMIZER_NAME")
    weight_decay = config.get("WEIGHT_DECAY")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. 데이터셋 로드 설정
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(488),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(15),       # 고개 꺾임 대비 (최대 15도)
            # transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)), # 초점 흐려짐 대비
            # transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2), # 조명 변화
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(488),
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
            pin_memory=True
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

    # 4. 모델 설정 및 동적 로드
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
    model = model.to(device)
    
    if checkpoint_path:
        model.load_state_dict(torch.load(str(checkpoint_path), weights_only=True))
        print(f"가중치 로드 성공: {checkpoint_path} (기존 기록 {best_acc_from_file:.2%}부터 시작)")
    else:
        print("기존 가중치 파일이 없습니다. 0.0%부터 학습을 시작합니다.")

    # 5. 옵티마이저 동적 로드
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
        momentum = config.get("MOMENTUM", 0.9)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"지원하지 않는 옵티마이저 이름입니다: {optimizer_name}")

    # 데이터 비율을 고려하여 졸음에 1.6배 가중치 부여
    weights = torch.tensor([2, 1.0], device=device) # [drowsy, normal] 순서
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    # 스케줄러: 3 Epoch 동안 정확도가 안 오르면 lr을 1/10로 감소
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
