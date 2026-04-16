import os
from PIL import Image
from tqdm import tqdm

data_dir = r"C:\Users\임상혁\Desktop\VisionGuard\VG Data\데이터 전처리 파일\dataset_final_v2"

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