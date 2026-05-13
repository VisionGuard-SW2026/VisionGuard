import mediapipe as mp
import cv2
import numpy as np

def crop_features_v2(image_path, output_size=(224, 224)):
    """
    이미지에서 눈과 입 주변 영역을 검출하여 정사각형 형태로 정밀 크롭합니다.

    Args:
        image_path (str or Path): 원본 이미지 파일의 경로.
        output_size (tuple): 모델 입력에 맞게 리사이즈할 크기 (기본값: 224x224).

    Returns:
        numpy.ndarray: 가공된 (output_size) 크기의 이미지 배열. 
                      얼굴 인식 실패 시 None을 반환합니다.

    Logic:
        1. MediaPipe FaceMesh를 사용하여 얼굴의 468개 특징점 좌표를 추출합니다.
        2. 눈 주변(EYE_INDICES)과 입 주변(MOUTH_INDICES) 좌표만 선택적으로 수집합니다.
        3. 선택된 모든 좌표를 포함하는 최소 사각형(Bounding Box)을 계산합니다.
        4. 움직임 변화를 학습에 반영하기 위해 상하좌우 20%의 여백(Margin)을 추가합니다.
        5. 이미지 왜곡을 방지하기 위해 긴 쪽을 기준으로 정사각형(Square) 비율을 맞춥니다.
        6. 최종 영역을 잘라내어 지정된 해상도로 리사이즈합니다.
    """
    mp_face_mesh = mp.solutions.face_mesh
    
    # 눈 주변 랜드마크 인덱스 (왼쪽/오른쪽 눈 전체)
    EYE_INDICES = [33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380]
    # 하품 감지용 입 주변 랜드마크 인덱스
    MOUTH_INDICES = [61, 291, 13, 14, 81, 311, 178, 402]
    COMBINED_INDICES = EYE_INDICES + MOUTH_INDICES
    
    image = cv2.imread(str(image_path))
    if image is None: return None
    
    h, w, _ = image.shape
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        # BGR 이미지를 RGB로 변환하여 MediaPipe 처리
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if not results.multi_face_landmarks:
            return None
            
        landmarks = results.multi_face_landmarks[0].landmark
        
        # 픽셀 좌표로 변환하여 리스트 생성
        x_coords = [int(landmarks[i].x * w) for i in COMBINED_INDICES]
        y_coords = [int(landmarks[i].y * h) for i in COMBINED_INDICES]
        
        xmin, xmax = min(x_coords), max(x_coords)
        ymin, ymax = min(y_coords), max(y_coords)
        
        # 특징점들을 아우르는 사각형의 크기 계산
        width = xmax - xmin
        height = ymax - ymin
        
        # 여백 부여: 눈꺼풀의 깜빡임과 입의 벌어짐이 잘리지 않도록 함
        margin_x = int(width * 0.2)
        margin_y = int(height * 0.2)
        
        xmin, xmax = max(0, xmin - margin_x), min(w, xmax + margin_x)
        ymin, ymax = max(0, ymin - margin_y), min(h, ymax + margin_y)
        
        # 정사각형 비율로 보정하여 리사이즈 시 왜곡 방지
        new_width = xmax - xmin
        new_height = ymax - ymin
        side_length = max(new_width, new_height)
        
        center_x, center_y = (xmin + xmax) // 2, (ymin + ymax) // 2
        
        # 이미지 경계를 벗어나지 않도록 최종 좌표 결정
        final_xmin = max(0, center_x - side_length // 2)
        final_xmax = min(w, final_xmin + side_length)
        final_ymin = max(0, center_y - side_length // 2)
        final_ymax = min(h, final_ymin + side_length)
        
        crop_img = image[final_ymin:final_ymax, final_xmin:final_xmax]
        
        # 최종 모델 입력 규격에 맞춰 리사이즈
        return cv2.resize(crop_img, output_size)