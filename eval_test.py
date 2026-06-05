import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import os
import re
import sys  
from dotenv import load_dotenv
from config import DEFAULT_CONFIG
from model_net import get_model
from tqdm import tqdm

def evaluate_test_set():
    print("=" * 60)
    print("🎯 Vision Guard 다중 모델 실전 데이터셋(Test Set) 통합 검증 가동")
    print("=" * 60)

    # 1. 환경 변수 및 설정 로드
    load_dotenv()
    config = DEFAULT_CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. VG_DATA_ROOT 환경 변수 기반 Weight 폴더 경로 계산
    vg_data_root_raw = os.getenv("VG_DATA_ROOT", "").strip().strip('"').strip("'")
    if not vg_data_root_raw:
        print("🚨 [환경변수 에러] 필수 환경변수 'VG_DATA_ROOT'가 설정되지 않았습니다.")
        return
        
    model_out_dir = Path(vg_data_root_raw) / "Weight"
    model_out_dir.mkdir(parents=True, exist_ok=True)

    # 3. 대상 모델 후보군 수집 (아키텍처/파일명 필터링 전면 해제)
    valid_models = []

    # Weight 폴더 내의 모든 .safetensors 및 .pth 파일을 무조건 수집
    for file in model_out_dir.iterdir():
        if file.is_file() and file.suffix in [".safetensors", ".pth"]:
            # 파일명에서 정확도 양식 (XX.XX%) 추출 시도
            match = re.search(r"\((\d+\.?\d*)%\)", file.name)
            recorded_acc = f"{float(match.group(1)):.2%}" if match else "기록 없음"
            
            valid_models.append({
                "path": file,
                "name": file.name,
                "suffix": file.suffix,
                "recorded_acc": recorded_acc
            })

    if not valid_models:
        print(f"🚨 [에러] '{model_out_dir}' 폴더 내에서 .safetensors 또는 .pth 파일을 찾을 수 없습니다.")
        return

    print(f"📂 가중치 저장소 경로: {model_out_dir}")
    print(f"✅ 총 {len(valid_models)}개의 모델 후보군을 발견했습니다.")
    print("-" * 60)

    # 4. 환경 변수 기반 데이터셋 경로 계산 및 데이터로더 빌드
    vg_dataset_rel = os.getenv("VG_DATASET_REL", r"dataset").strip().strip('"').strip("'")
    data_dir = Path(vg_data_root_raw) / vg_dataset_rel
    test_dir = data_dir / "test"

    if not test_dir.exists():
        print(f"🚨 [경로 에러] Test 데이터셋 폴더를 찾을 수 없습니다: {test_dir}")
        return

    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        test_dataset = datasets.ImageFolder(str(test_dir), transform=test_transform)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config["BATCH_SIZE"], 
            shuffle=False, 
            num_workers=config["NUM_WORKERS"], 
            pin_memory=True
        )
    except Exception as e:
        print(f"🚨 [데이터로더 에러] 데이터셋 로드 실패: {e}")
        return

    # 5. 다중 모델 검증 루프 가동
    skip_prompt = False  
    total_samples = len(test_dataset)

    for idx, model_info in enumerate(valid_models, 1):
        print(f"\n[모델 {idx}/{len(valid_models)}] 후보 검증 준비")
        print(f"📄 파일명: {model_info['name']}")
        print(f"📊 파일 기록 성적: {model_info['recorded_acc']}")

        # 대화형 프롬프트 제어 구역
        if not skip_prompt:
            user_input = input("▶️ 이 모델을 검증하시겠습니까? (Y: 진행 / N: 패스 / ay: 전체 자동진행 / q: 종료): ").strip().lower()
            
            if user_input == 'q':
                print("\n🛑 사용자의 요청으로 모델 검증 프로그램을 강제 종료합니다.")
                sys.exit(0)
            elif user_input == 'n':
                print("⏭️ 해당 모델 검증을 패스하고 다음으로 넘어갑니다.")
                continue
            elif user_input == 'ay':
                print("🚀 'All Yes'가 활성화되었습니다. 이후 모든 모델은 확인 없이 자동 검증을 진행합니다.")
                skip_prompt = True
            elif user_input != 'y' and user_input != '':
                print("⚠️ 잘못된 입력입니다. 기본값인 [Y]로 간주하고 진행합니다.")

        # 6. 파일명에서 모델 종류(아키텍처)를 동적으로 추론하여 백엔드 빌드
        # 파일명이 'MobileNetV2_...'로 시작하면 MobileNetV2 모델 구조를 동적으로 생성합니다.
        current_model_name = config["MODEL_NAME"] # 기본값 복사
        for arch in ["EfficientNetB0", "MobileNetV2", "ResNet50"]:
            if model_info['name'].startswith(arch):
                current_model_name = arch
                break

        try:
            model = get_model(current_model_name).to(device)
            
            if model_info['suffix'] == ".safetensors":
                from safetensors.torch import load_file
                weights = load_file(str(model_info['path']))
            else:
                ckpt = torch.load(str(model_info['path']), map_location=device)
                weights = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
            
            model.load_state_dict(weights)
            model.eval()
            print(f"🤖 모델 구조 매핑 완료: {current_model_name}")
        except Exception as e:
            print(f"🚨 [모델 로드 에러] '{model_info['name']}' 가중치 주입 실패: {e}")
            print("   - model_net.py에 해당 모델 구조가 정의되어 있는지 확인하세요.")
            continue

        # 7. 수능 평가 추론 연산 수행
        running_corrects = 0
        print(f"🔍 총 {total_samples}개의 격리된 순수 Test 이미지 평가 진행 중...")
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="채점 중", dynamic_ncols=True):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)

        # 8. 개별 모델 결과 리포팅
        final_test_acc = (running_corrects.double() / total_samples).item()
        print("-" * 60)
        print(f"📊 [검증 완료 결과 - {model_info['name']}]")
        print(f"   - 기존 기록 성적 (Valid Accuracy) : {model_info['recorded_acc']}")
        print(f"   - 실전 채점 성적 (Test Accuracy)  : {final_test_acc:.2%}")
        print("-" * 60)

    print("\n🏁 모든 모델의 최종 인수 테스트 검증 절차가 종료되었습니다.")

if __name__ == '__main__':
    evaluate_test_set()