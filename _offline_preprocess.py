import os
import cv2
from pathlib import Path
from tqdm import tqdm
from _face_detector import crop_features_v2
from dotenv import load_dotenv

load_dotenv()

# 경로 설정
root_str = os.getenv("VG_DATA_ROOT", "").strip().strip('"').strip("'")
SOURCE_DIR = Path(root_str) / "데이터 전처리 파일" / "dataset_final_v2"
TARGET_DIR = Path(root_str) / "데이터 전처리 파일" / "dataset_eyes_v1" # 가공된 데이터 저장소

def preprocess_all_data():
    if not SOURCE_DIR.exists():
        print(f"원본 경로를 찾을 수 없습니다: {SOURCE_DIR}")
        return

    print(f"🚀 오프라인 전처리 시작: {SOURCE_DIR} -> {TARGET_DIR}")
    
    for phase in ["train", "valid"]:
        for label in ["normal", "drowsy"]:
            src_path = SOURCE_DIR / phase / label
            dst_path = TARGET_DIR / phase / label
            dst_path.mkdir(parents=True, exist_ok=True)

            img_files = list(src_path.glob("*.jpg"))
            print(f"\n[{phase}/{label}] {len(img_files)}장 처리 중...")

            for img_p in tqdm(img_files):
                # 이미 가공된 파일이 있다면 건너뛰기 (중단 후 재시작 가능)
                target_file = dst_path / img_p.name
                if target_file.exists():
                    continue

                # _face_detector.py의 정밀 눈 크롭 로직 활용
                # EfficientNetB0 최적화 해상도인 224x224로 저장
                processed_img = crop_features_v2(img_p, output_size=(224, 224))
                
                if processed_img is not None:
                    cv2.imwrite(str(target_file), processed_img)
                else:
                    # 얼굴/눈 인식 실패 시, 차선책으로 중앙 크롭 후 저장
                    raw_img = cv2.imread(str(img_p))
                    if raw_img is not None:
                        h, w, _ = raw_img.shape
                        side = min(h, w)
                        start_x, start_y = (w - side) // 2, (h - side) // 2
                        fallback = raw_img[start_y:start_y+side, start_x:start_x+side]
                        fallback = cv2.resize(fallback, (224, 224))
                        cv2.imwrite(str(target_file), fallback)

if __name__ == "__main__":
    preprocess_all_data()
    print(f"\n✅ 전처리 완료! 이제 'dataset_eyes_v1' 폴더를 사용하세요.")