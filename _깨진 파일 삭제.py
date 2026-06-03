import os
from pathlib import Path

from PIL import Image
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

vg_data_root_raw = os.getenv("VG_DATA_ROOT", "").strip().strip('"').strip("'")
if not vg_data_root_raw:
    raise ValueError(
        "필수 환경변수 'VG_DATA_ROOT'가 설정되지 않았습니다. "
        ".env 파일의 설정을 확인해주세요."
    )

vg_dataset_rel = os.getenv("VG_DATASET_REL", r"dataset").strip().strip('"').strip("'")
data_dir = Path(vg_data_root_raw) / vg_dataset_rel

print("불량 이미지 검사 시작...")
for root, dirs, files in os.walk(data_dir):
    for file in tqdm(files):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    img.verify() # 파일이 손상되었는지 검사
            except Exception:
                print(f"\n불량 파일 발견 및 삭제: {file_path}")
                os.remove(file_path) # 읽을 수 없는 파일 삭제
print("검사 완료! 이제 학습 코드를 다시 실행해 보세요.")