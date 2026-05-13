import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import os

# 전역 변수로 landmarker를 설정하여 모델을 한 번만 로드
_LANDMARKER = None

def get_landmarker():
    global _LANDMARKER
    if _LANDMARKER is None:
        # 현재 파일(_face_detector.py)과 같은 위치에 있는 .task 파일의 절대 경로를 계산합니다.
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'face_landmarker.task')
        
        if not os.path.exists(model_path):
            print(f"❌ 에러: {model_path} 파일을 찾을 수 없습니다!")
            return None
            
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            num_faces=1
        )
        _LANDMARKER = vision.FaceLandmarker.create_from_options(options)
    return _LANDMARKER

def crop_features_v2(image_path, output_size=(224, 224)):
    # 윈도우 한글 경로 대응: cv2.imread 대신 np.fromfile 사용
    try:
        img_array = np.fromfile(str(image_path), np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"❌ 이미지 로드 실패: {image_path}, 에러: {e}")
        return None

    if image is None: return None
    h, w, _ = image.shape
    
    # RGB 변환 및 MediaPipe 이미지 객체 생성 (더 안정적인 방식)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # [핵심] Windows free 에러 방지: 메모리를 한 덩어리로 정렬하고 강제 복사
    rgb_image = np.ascontiguousarray(rgb_image.copy())

    try:
        mp_image = mp.Image.create_from_numpy(rgb_image)
    except AttributeError:
        # 버전 차이로 위 메서드가 없을 경우를 대비한 Fallback
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    # 로드된 모델 사용
    landmarker = get_landmarker()
    if landmarker is None: return None
    
    detection_result = landmarker.detect(mp_image)
    
    if not detection_result.face_landmarks:
        # 얼굴을 못 찾으면 스킵 (이게 너무 많으면 문제)
        return None

    # 눈(EYE) + 입(MOUTH) 인덱스
    EYE_INDICES = [33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380]
    MOUTH_INDICES = [61, 291, 13, 14, 81, 311, 178, 402]
    COMBINED_INDICES = EYE_INDICES + MOUTH_INDICES
    
    landmarks = detection_result.face_landmarks[0]
    x_coords = [int(landmarks[i].x * w) for i in COMBINED_INDICES]
    y_coords = [int(landmarks[i].y * h) for i in COMBINED_INDICES]
    
    # 1. 최소/최대 좌표 계산
    xmin, xmax = min(x_coords), max(x_coords)
    ymin, ymax = min(y_coords), max(y_coords)
    
    # 2. 정사각형 영역 및 여백 설정 (1.4배)
    width, height = xmax - xmin, ymax - ymin
    side_length = int(max(width, height) * 1.4) # 여백 포함
    
    # 3. 중심점 기준 좌표 계산
    center_x, center_y = (xmin + xmax) // 2, (ymin + ymax) // 2
    
    start_x = center_x - side_length // 2
    start_y = center_y - side_length // 2
    
    # 4. 이미지 경계를 벗어나지 않도록 보정 (중요!)
    final_xmin = max(0, start_x)
    final_ymin = max(0, start_y)
    final_xmax = min(w, final_xmin + side_length)
    final_ymax = min(h, final_ymin + side_length)
    
    # 만약 오른쪽/아래가 잘린다면 왼쪽/위쪽을 더 확보해서 정사각형 유지
    if final_xmax - final_xmin < side_length:
        final_xmin = max(0, final_xmax - side_length)
    if final_ymax - final_ymin < side_length:
        final_ymin = max(0, final_ymax - side_length)

    crop_img = image[final_ymin:final_ymax, final_xmin:final_xmax]
    
    if crop_img.size == 0: return None
    return cv2.resize(crop_img, output_size)

def safe_save_img(save_path, img):
    """한글 경로를 포함한 환경에서도 안전하게 이미지를 저장합니다."""
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        extension = os.path.splitext(save_path)[1]
        result, encoded_img = cv2.imencode(extension, img)
        if result:
            encoded_img.tofile(str(save_path))
            return True
    except Exception as e:
        print(f"❌ 저장 실패: {save_path}, 에러: {e}")
    return False