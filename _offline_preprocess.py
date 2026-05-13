import os
import cv2
from pathlib import Path
from tqdm import tqdm
from _face_detector import crop_features_v2, safe_save_img
from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor
import numpy as np

load_dotenv()

# 경로 설정
root_str = os.getenv("VG_DATA_ROOT", "").strip().strip('"').strip("'")
SOURCE_DIR = Path(root_str) / "데이터 전처리 파일" / "dataset_final_v2"
TARGET_DIR = Path("C:/VG/dataset_eyes_v1") # 가공된 데이터 저장소 고정

def process_worker(task):
    """개별 이미지를 처리하는 워커 함수"""
    img_p, target_file = task
    
    if target_file.exists():
        return True

    # _face_detector.py의 정밀 눈 크롭 로직 활용
    processed_img = crop_features_v2(img_p, output_size=(224, 224))
    
    if processed_img is not None:
        return safe_save_img(str(target_file), processed_img)
    else:
        # 얼굴/눈 인식 실패 시, 차선책으로 중앙 크롭 후 한글 대응 저장
        try:
            img_array = np.fromfile(str(img_p), np.uint8)
            raw_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if raw_img is not None:
                h, w, _ = raw_img.shape
                side = min(h, w)
                start_x, start_y = (w - side) // 2, (h - side) // 2
                fallback = raw_img[start_y:start_y+side, start_x:start_x+side]
                fallback = cv2.resize(fallback, (224, 224))
                return safe_save_img(str(target_file), fallback)
        except:
            pass
    return False

def preprocess_all_data():
    if not SOURCE_DIR.exists():
        print(f"원본 경로를 찾을 수 없습니다: {SOURCE_DIR}")
        return

    print(f"🚀 오프라인 전처리 시작: {SOURCE_DIR} -> {TARGET_DIR}")
    
    # 모든 작업 리스트 수집
    all_tasks = []
    for phase in ["train", "valid"]:
        for label in ["normal", "drowsy"]:
            src_path = SOURCE_DIR / phase / label
            img_files = list(src_path.glob("*.jpg"))
            for img_p in img_files:
                target_file = TARGET_DIR / phase / label / img_p.name
                all_tasks.append((img_p, target_file))

    # 멀티프로세싱 실행 (병렬 코어 활용)
    # max_workers는 코어 수에 맞춰 조정하세요 (6~8개 권장)
    with ProcessPoolExecutor(max_workers=6) as executor:
        list(tqdm(executor.map(process_worker, all_tasks), total=len(all_tasks), desc="🔥 병렬 전처리 진행 중"))

if __name__ == "__main__":
    preprocess_all_data()
    print(f"\n✅ 전처리 완료! 이제 'dataset_eyes_v1' 폴더를 사용하세요.")