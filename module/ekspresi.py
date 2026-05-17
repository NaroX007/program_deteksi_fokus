import os
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import Counter, deque

# ======================
# MODEL PATH & INIT
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "model", "face_landmarker.task"))

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)
landmarker = FaceLandmarker.create_from_options(options)

timestamp_ms = 0

# ======================
# THRESHOLD & CONFIG
# ======================
THRESHOLD = {
    'smile_curve'   : 0.018,
    'brow_raise'    : 0.22,
    'eye_squint'    : 0.23,
    'uncomfortable_curve'         : -0.014,
    'uncomfortable_MAR'           : 0.30,
    'brow_furrow'                 : 0.16,
    'uncomfortable_min_indicators': 2,
    'min_confidence': 0.55,
}

LM = {
    'mouth_left': 61, 'mouth_right': 291, 'mouth_top': 13, 'mouth_bottom': 14,
    'l_eye_top': 159, 'l_eye_bot': 145, 'l_eye_left': 33, 'l_eye_right': 133,
    'r_eye_top': 386, 'r_eye_bot': 374, 'r_eye_left': 362, 'r_eye_right': 263,
    'l_brow': [70, 63, 105, 66, 107], 'r_brow': [336, 296, 334, 293, 300],
    'face_left': 234, 'face_right': 454,
}

expr_buffer = deque(maxlen=12)

def extract_features(landmarks, w, h):
    points = {idx: (int(lm.x * w), int(lm.y * h)) for idx, lm in enumerate(landmarks)}
    
    ml = np.array(points[LM['mouth_left']]); mr = np.array(points[LM['mouth_right']])
    mt = np.array(points[LM['mouth_top']]); mb = np.array(points[LM['mouth_bottom']])

    mouth_width = np.linalg.norm(mr - ml) + 1e-6
    mouth_height = np.linalg.norm(mb - mt)
    mar = mouth_height / mouth_width

    center_y = (mt[1] + mb[1]) / 2
    corner_avg_y = (points[LM['mouth_left']][1] + points[LM['mouth_right']][1]) / 2
    raw_curve = center_y - corner_avg_y

    def ear(top, bot, left, right):
        v = np.linalg.norm(np.array(top) - np.array(bot))
        h = np.linalg.norm(np.array(left) - np.array(right)) + 1e-6
        return v / h

    ear_left = ear(points[LM['l_eye_top']], points[LM['l_eye_bot']], points[LM['l_eye_left']], points[LM['l_eye_right']])
    ear_right = ear(points[LM['r_eye_top']], points[LM['r_eye_bot']], points[LM['r_eye_left']], points[LM['r_eye_right']])
    ear_avg = (ear_left + ear_right) / 2

    l_brow_y = np.mean([points[i][1] for i in LM['l_brow']])
    r_brow_y = np.mean([points[i][1] for i in LM['r_brow']])
    l_eye_y = points[LM['l_eye_top']][1]
    r_eye_y = points[LM['r_eye_top']][1]
    raw_brow = ((l_eye_y - l_brow_y) + (r_eye_y - r_brow_y)) / 2

    face_w = np.linalg.norm(np.array(points[LM['face_left']]) - np.array(points[LM['face_right']])) + 1e-6

    return {
        'MAR': mar,
        'mouth_curve': raw_curve / face_w,
        'EAR_avg': ear_avg,
        'brow_raise_avg': raw_brow / face_w
    }

def classify(features):
    smile_score = 0.0
    unc_score = 0.0
    unc_count = 0

    if features['mouth_curve'] > THRESHOLD['smile_curve']: smile_score += 2.5
    if features['EAR_avg'] < THRESHOLD['eye_squint']: smile_score += 1.0
    if features['brow_raise_avg'] > THRESHOLD['brow_raise']: smile_score += 1.0

    if features['mouth_curve'] < THRESHOLD['uncomfortable_curve']: unc_score += 2.5; unc_count += 1
    if features['MAR'] > THRESHOLD['uncomfortable_MAR']: unc_score += 2.0; unc_count += 1
    if features['brow_raise_avg'] < THRESHOLD['brow_furrow']: unc_score += 1.0; unc_count += 1

    if unc_count < THRESHOLD['uncomfortable_min_indicators']: unc_score = 0.0

    if unc_score >= 3.0 and unc_score > smile_score: return "TIDAK NYAMAN", min(unc_score / 5.5, 1.0)
    elif smile_score >= 2.0 and smile_score > unc_score: return "SENANG", min(smile_score / 4.5, 1.0)
    else: return "NETRAL", 1.0

def analyze_expression(frame):
    global timestamp_ms
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    timestamp_ms += 33
    
    if not result.face_landmarks:
        return {
            "label": "UNKNOWN", 
            "not_focus": False, 
            "reason": "NO FACE",
            "landmarks": None
        }
        
    landmarks = result.face_landmarks[0]
    h, w, _ = frame.shape
    
    features = extract_features(landmarks, w, h)
    expression, conf = classify(features)
    
    if conf >= THRESHOLD['min_confidence']:
        expr_buffer.append(expression)
        
    smoothed = Counter(expr_buffer).most_common(1)[0][0] if expr_buffer else "UNKNOWN"
    is_not_focus = (smoothed == "TIDAK NYAMAN")
    
    return {
        "label": smoothed,
        "not_focus": is_not_focus,
        "reason": "Ekspresi Negatif" if is_not_focus else "OK",
        "landmarks": result.face_landmarks[0] if result.face_landmarks else None
    }
