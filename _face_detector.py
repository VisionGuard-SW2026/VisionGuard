import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np

# 전역 변수로 landmarker를 설정하여 모델을 한 번만 로드
_LANDMARKER = None

def get_landmarker(model_path='face_landmarker.task'):
    """모델을 메모리에 한 번만 로드하여 재사용"""
    global _LANDMARKER
    if _LANDMARKER is None:
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            num_faces=1
        )
        _LANDMARKER = vision.FaceLandmarker.create_from_options(options)
    return _LANDMARKER

def crop_features_v2(image_path, output_size=(224, 224)):
    """
    MediaPipe Tasks API를 사용하여 눈과 입 영역을 정밀 크롭
    """
    image = cv2.imread(str(image_path))
    if image is None: return None
    h, w, _ = image.shape
    
    # RGB 변환 및 MediaPipe 이미지 객체 생성
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # 로드된 모델 사용
    landmarker = get_landmarker()
    detection_result = landmarker.detect(mp_image)
    
    if not detection_result.face_landmarks:
        return None

    # 특징점 좌표 수집 (눈 + 입)
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