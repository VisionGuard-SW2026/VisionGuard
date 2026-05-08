import mediapipe as mp
import cv2
import numpy as np

def crop_eyes_only(image_path, output_size=(224, 224)):
    mp_face_mesh = mp.solutions.face_mesh
    # 눈 주변 랜드마크 인덱스 (왼쪽/오른쪽 눈 전체를 아우르는 범위)
    EYE_INDICES = [33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380]
    
    image = cv2.imread(str(image_path))
    if image is None: return None
    
    h, w, _ = image.shape
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if not results.multi_face_landmarks:
            return None # 얼굴 인식 실패 시 제외
            
        landmarks = results.multi_face_landmarks[0].landmark
        
        # 눈 랜드마크들의 좌표 추출
        x_coords = [int(landmarks[i].x * w) for i in EYE_INDICES]
        y_coords = [int(landmarks[i].y * h) for i in EYE_INDICES]
        
        # 눈 부위 바운딩 박스 계산 (20% 정도 여유 있게 크롭)
        xmin, xmax = min(x_coords), max(x_coords)
        ymin, ymax = min(y_coords), max(y_coords)
        
        # 가로세로 비율을 유지하며 여백 추가
        offset_x = int((xmax - xmin) * 0.2)
        offset_y = int((ymax - ymin) * 0.5)
        
        cropped = image[max(0, ymin-offset_y):min(h, ymax+offset_y), 
                       max(0, xmin-offset_x):min(w, xmax+offset_x)]
        
        return cv2.resize(cropped, output_size)