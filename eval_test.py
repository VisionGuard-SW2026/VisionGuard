import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import os
from dotenv import load_dotenv
from config import DEFAULT_CONFIG
from model_net import get_model

def evaluate_test_set():
    print("=" * 60)
    print("🎯 Vision Guard 최종 실전 데이터셋(Test Set) 평가 가동")
    print("=" * 60)

    # 1. 환경 변수 및 설정 로드
    load_dotenv()
    config = DEFAULT_CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 파일에서 가중치 탐색 및 자동 매칭
    model_name = config["MODEL_NAME"]
    model_prefix = f"{model_name}_{config['MODEL_PREFIX']}"
    
    checkpoint_path = None
    best_acc_from_file = "0.00%"
    model_out_dir = Path(".")

    # 현재 폴더에서 가장 높은 성적의 .safetensors 가중치 파일 자동 탐색
    import re
    max_acc = -1.0
    for file in model_out_dir.iterdir():
        if file.is_file() and file.name.startswith(model_prefix) and file.suffix == ".safetensors":
            match = re.search(r"\((\d+\.?\d*)%\)", file.name)
            if match:
                acc_val = float(match.group(1))
                if acc_val > max_acc:
                    max_acc = acc_val
                    checkpoint_path = file
                    best_acc_from_file = f"{acc_val:.2%}"

    if not checkpoint_path:
        print("🚨 [에러] 폴더 내에서 .safetensors 가중치 파일을 찾을 수 없습니다.")
        print("   - 파일명이 'EfficientNetB0_Best_VisionGuard_v2(99.45%).safetensors' 구조인지 확인하세요.")
        return

    print(f"📦 로드할 최종 가중치 파일: {checkpoint_path.name} (기록된 성적: {best_acc_from_file})")

    # 3. 환경 변수 기반 데이터셋 경로 계산
    vg_data_root_raw = os.getenv("VG_DATA_ROOT", "").strip().strip('"').strip("'")
    vg_dataset_rel = os.getenv("VG_DATASET_REL", r"dataset").strip().strip('"').strip("'")
    data_dir = Path(vg_data_root_raw) / vg_dataset_rel
    test_dir = data_dir / "test"

    if not test_dir.exists():
        print(f"🚨 [경로 에러] Test 데이터셋 폴더를 찾을 수 없습니다: {test_dir}")
        return

    # 4. 수능 시험(Test Set) 전용 정규화 데이터로더 선언
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
            shuffle=False,  # 🧠 순수 평가이므로 셔플 원천 차단
            num_workers=config["NUM_WORKERS"], 
            pin_memory=True
        )
    except Exception as e:
        print(f"🚨 [데이터로더 에러] 데이터셋 로드 실패: {e}")
        return

    # 5. 모델 빌드 및 가중치 주입 (safetensors 가속 로드)
    from safetensors.torch import load_file
    model = get_model(model_name).to(device)
    
    try:
        weights = load_file(str(checkpoint_path))
        model.load_state_dict(weights)
        print("✅ 가중치 메모리 매핑 완료. 최종 인퍼런스 모드 가동.")
    except Exception as e:
        print(f"🚨 [가중치 로드 에러] .safetensors 파일 디코딩 실패: {e}")
        return

    # 6. 최종 수능 평가 루프 돌리기 (추론 연산)
    model.eval()
    running_corrects = 0
    total_samples = len(test_dataset)

    print(f"\n🔍 총 {total_samples}개의 오염되지 않은 순수 Test 이미지 평가 진행 중...")
    
    with torch.no_grad(): # 역전파 경사하강 연산 정지하여 RTX 4070 VRAM 최적화
        for inputs, labels in tqdm(test_loader, desc="최종 채점 중", dynamic_ncols=True):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == labels.data)

    # 7. 최종 결과 레포팅
    final_test_acc = (running_corrects.double() / total_samples).item()
    print("\n" + "=" * 60)
    print(f"📊 [최종 인수 테스트 채점 결과]")
    print(f"   - 모의고사 성적 (Valid Accuracy) : {best_acc_from_file}")
    print(f"   - 실제 수능 성적 (Test Accuracy)  : {final_test_acc:.2%}")
    print("=" * 60)
    
    if final_test_acc >= 0.95:
        print("✨ [결론] 일반화 성능 완벽 검증 완료. 실전 배포 가능 등급.")
    else:
        print("⚠️ [결론] 데이터 누수(Data Leakage) 징후 감지됨.")
        print("   - 의심 원인: 무작위 프레임 셔플로 인해 Train 데이터의 특징이 시험 문제에 유출되었을 수 있음.")

if __name__ == '__main__':
    evaluate_test_set()