# 파일명: clean_and_split_dataset.py

import json
import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

def get_float_coord(coord_value):
    """문자열이든 정수든 안전하게 좌표를 float으로 변환하는 방어 함수"""
    try:
        return float(coord_value)
    except (ValueError, TypeError):
        return 0.0

def process_and_split_dataset(json_raw_dir, img_raw_dir, output_data_dir, split_ratio=(0.8, 0.1, 0.1)):
    """
    1단계: JSON 분석 후 데이터 실분포(0.45~0.48)에 맞춰 졸음 데이터 대거 구출 및 정제
    2단계: 8:1:1 비율로 Train, Valid, Test 폴더로 분할하여 복사
    """
    json_path_base = Path(json_raw_dir)
    img_path_base = Path(img_raw_dir)
    out_path = Path(output_data_dir)
    
    # [핵심 수정] 실제 데이터셋 분포를 반영한 현실적인 종횡비 임계값 재설정
    AMBIGUOUS_MAX_CLOSE = 0.48  # 눈을 감았을 때(False)의 상한선을 높여 졸음 데이터 대거 구출
    AMBIGUOUS_MIN_OPEN = 0.45   # 눈을 떴을 때(True)의 하한선을 낮춰 정상 데이터 유연하게 수용

    normal_files = []
    drowsy_files = []
    error_count = 0
    filtered_count = 0
    
    first_error_printed = False 

    print("\n🔍 [Step 1] 원본 데이터 탐색 및 무결성 검증 진행 중...")
    json_files = list(json_path_base.rglob("*.json"))
    
    for json_path in tqdm(json_files, desc="데이터 정제 중", dynamic_ncols=True):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            img_name = data["FileInfo"]["FileName"]
            
            # [구조 치환] [라벨] 폴더 구조를 [원천] 폴더 구조로 자동 치환하여 이미지 짝커플 매칭
            rel_path = json_path.parent.relative_to(json_path_base)
            rel_path_templated = str(rel_path).replace("[라벨]", "[원천]")
            img_path = img_path_base / rel_path_templated / img_name

            if not img_path.exists():
                error_count += 1
                if not first_error_printed:
                    print(f"\n🚨 [디버그] 이미지 매칭 실패 예시")
                    print(f"   - JSON 위치: {json_path}")
                    print(f"   - 찾으려던 이미지 경로: {img_path}")
                    first_error_printed = True
                continue

            bbox = data["ObjectInfo"]["BoundingBox"]
            leye = bbox.get("Leye", {})
            reye = bbox.get("Reye", {})
            
            # 눈 데이터가 비정상적이거나 가려짐(isVisible=False) 상태면 탈락
            if not leye.get("isVisible", False) or not reye.get("isVisible", False):
                filtered_count += 1
                continue
            
            # 좌표 데이터 타입 예외 처리 및 가공
            lx1, ly1, lx2, ly2 = map(get_float_coord, leye.get("Position", [0,0,0,0]))
            rx1, ry1, rx2, ry2 = map(get_float_coord, reye.get("Position", [0,0,0,0]))
            
            # [버그 수정] 오른눈 h 계산 시 ly1으로 잘못 매핑되어 있던 휴먼 에러 완벽 수정 (ry1으로 교정)
            l_w, l_h = lx2 - lx1, ly2 - ly1
            r_w, r_h = rx2 - rx1, ry2 - ry1
            
            l_ratio = l_h / l_w if l_w > 0 else 0
            r_ratio = r_h / r_w if r_w > 0 else 0
            avg_ratio = (l_ratio + r_ratio) / 2
            
            if "Opened" not in leye or "Opened" not in reye:
                filtered_count += 1
                continue
                
            l_open, r_open = leye["Opened"], reye["Opened"]
            
            # 수밀리초 단위 완벽 필터링 검증 기전
            if l_open and r_open and avg_ratio >= AMBIGUOUS_MIN_OPEN:
                normal_files.append((img_path, json_path))
            elif not l_open and not r_open and avg_ratio <= AMBIGUOUS_MAX_CLOSE:
                drowsy_files.append((img_path, json_path))
            else:
                filtered_count += 1

        except Exception as e:
            error_count += 1
            if not first_error_printed:
                print(f"\n🚨 [디버그] 내부 구문 파싱 에러: {e} (파일: {json_path.name})")
                first_error_printed = True

    print(f"\n📊 [정제 결과 - B 방안 적용 후]")
    print(f" - 채택된 정상(Normal) 데이터: {len(normal_files)}개")
    print(f" - 채택된 졸음(Drowsy) 데이터: {len(drowsy_files)}개")
    print(f" - 제거된 애매한 데이터: {filtered_count}개")
    print(f" - 오류 및 경로 미매칭 데이터: {error_count}개")

    # ---------------------------------------------------------
    
    print("\n🔀 [Step 2] 데이터 셔플 및 분할 (Train/Valid/Test) 시작...")
    
    random.seed(42) 
    random.shuffle(normal_files)
    random.shuffle(drowsy_files)
    
    def split_and_copy(files, class_name):
        total = len(files)
        if total == 0: return
        
        train_idx = int(total * split_ratio[0])
        valid_idx = train_idx + int(total * split_ratio[1])
        
        splits = {
            "train": files[:train_idx],
            "valid": files[train_idx:valid_idx],
            "test": files[valid_idx:]
        }
        
        for split_name, split_files in splits.items():
            target_dir = out_path / split_name / class_name
            target_dir.mkdir(parents=True, exist_ok=True)
            
            for img_p, json_p in tqdm(split_files, desc=f"{split_name}/{class_name} 복사 중", dynamic_ncols=True):
                shutil.copy(str(img_p), str(target_dir / img_p.name))
                shutil.copy(str(json_p), str(target_dir / json_p.name))

    split_and_copy(normal_files, "normal")
    split_and_copy(drowsy_files, "drowsy")
    
    print("\n🚀 데이터 파이프라인 구축 완료! 정제 완료된 데이터셋으로 학습을 진행하세요.")

if __name__ == "__main__":
    print("=" * 60)
    print("🛡️ Vision Guard 데이터셋 통합 정제 및 분할 파이프라인 (자동 치환형)")
    print("=" * 60)

    json_input_dir = input("▶️ [입력] JSON 파일들이 있는 최상위 폴더 경로:\n👉 ").strip()
    img_input_dir = input("\n▶️ [입력] JPG 파일들이 있는 최상위 폴더 경로:\n👉 ").strip()
    out_input_dir = input("\n▶️ [출력] 최종 저장할 폴더 경로:\n👉 ").strip()

    if os.path.exists(json_input_dir) and os.path.exists(img_input_dir):
        process_and_split_dataset(json_input_dir, img_input_dir, out_input_dir)
    else:
        print("\n🚨 [입력 오류] 존재하지 않는 경로가 있습니다. 공백이나 오타를 확인해 주세요.")