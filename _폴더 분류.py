import numpy as np
import os
import shutil
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

# 1. 경로 설정
CURRENT_DIR = Path(__file__).resolve().parent
load_dotenv()
root_str = os.getenv("VG_DATA_ROOT", "")
if not root_str:
    raise ValueError("VG_DATA_ROOT가 .env에 없습니다.")
VG_DATA_ROOT = Path(root_str).expanduser()
BASE_PATH = VG_DATA_ROOT / "기본 데이터 파일" / "졸음운전 예방을 위한 운전자 상태 정보 영상"
SAVE_ROOT = VG_DATA_ROOT / "데이터 전처리 파일" / "dataset_final_v2"

# [핵심 수정] 실제 폴더명인 '[원천]bbox'로 경로를 지정해야 합니다.
IMG_DIRS = [
    BASE_PATH / "Training" / "[원천]bbox(실제도로환경)",
    BASE_PATH / "Training" / "[원천]bbox(통제환경)",
    BASE_PATH / "Training" / "[원천]keypoint(준통제환경)",
    BASE_PATH / "Validation" / "[원천]bbox(실제도로환경)",
    BASE_PATH / "Validation" / "[원천]bbox(통제환경)",
    BASE_PATH / "Validation" / "[원천]keypoint(준통제환경)",
]

# 2. 정답지 로드
true_labels = np.load(CURRENT_DIR / "labels.npy")

# 3. 모든 이미지 파일 목록 확보 (정렬 유지)
print("모든 이미지 경로를 수집 중...")
all_images = []
for d in IMG_DIRS:
    if d.exists():
        # 하위 폴더까지 jpg 파일 수집 후 정렬
        found_images = sorted(list(d.rglob("*.jpg")))
        all_images.extend(found_images)
        print(f"{d.name}에서 {len(found_images)}장 발견!")
    else:
        print(f"경고: 폴더를 찾을 수 없습니다 -> {d}")

# 4. 재분류 시작
limit = min(len(all_images), len(true_labels))
if limit == 0:
    print("이미지 또는 라벨을 찾지 못했습니다. 경로를 다시 확인해주세요!")
else:
    print(f"총 {limit}장의 사진을 검증 및 재분류합니다.")

    for i in tqdm(range(limit)):
        img_path = all_images[i]
        img_name = img_path.name
        
        # [파일명 우선 검증]
        if "정상주시" in img_name:
            label_name = "normal"
        elif "졸음재현" in img_name or "하품재현" in img_name:
            label_name = "drowsy"
        else:
            # 키워드가 없는 경우는 npy 값을 따름
            label_valid = true_labels[i]
            label_name = "normal" if label_valid == 0 else "drowsy"
        
        # [8:2 분할]
        if hash(img_name) % 10 < 8:
            phase = "train"
        else:
            phase = "valid"
        
        target_dir = SAVE_ROOT / phase / label_name
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # 파일 복사
        shutil.copy(img_path, target_dir / img_name)

    print(f"\n[완료] {SAVE_ROOT}에 데이터셋 구축이 끝났습니다.")