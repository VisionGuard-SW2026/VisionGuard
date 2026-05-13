import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np

# 전역 변수로 landmarker를 설정하여 모델을 한 번만 로드합니다.
_LANDMARKER = None

def get_landmarker(model_path='face_landmarker.task'):
    """모델을 메모리에 한 번만 로드하여 재사용합니다."""
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
    MediaPipe Tasks API를 사용하여 눈과 입 영역을 정밀 크롭합니다.
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
    
    # 바운딩 박스 계산 및 여백 부여
    xmin, xmax, ymin, ymax = min(x_coords), max(x_coords), min(y_coords), max(y_coords)
    width, height = xmax - xmin, ymax - ymin
    side_length = int(max(width, height) * 1.4) # 여백 포함
    
    center_x, center_y = (xmin + xmax) // 2, (ymin + ymax) // 2
    
    final_xmin = max(0, center_x - side_length // 2)
    final_ymin = max(0, center_y - side_length // 2)
    
    crop_img = image[final_ymin:final_ymin+side_length, final_xmin:final_xmin+side_length]
    
    if crop_img.size == 0: return None
    return cv2.resize(crop_img, output_size)