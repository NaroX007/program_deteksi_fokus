import os
import time
from collections import deque, Counter

import cv2
import mediapipe as mp

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

# ======================
# THRESHOLD (DIKALIBRASI)
# ======================
YAW_THRESHOLD = 0.20        # Kiri/Kanan
PITCH_UP_THRESHOLD = 0.11   # Atas
PITCH_DOWN_THRESHOLD = 0.15 # Bawah

# Ambang batas gerakan sedikit dipertajam
MOVE_START_THRESHOLD = 0.060 
MOVE_STOP_THRESHOLD = 0.040
MOVE_CONFIRM_TIME = 0.2      # Waktu konfirmasi bahwa ini beneran bergerak
MOVE_RELEASE_TIME = 0.4      # Waktu delay sebelum dianggap berhenti
MOVE_SECONDS_THRESHOLD = 5.0 # Batas waktu goyang/gerak

# ======================
# STATE & SMOOTHING
# ======================
timestamp_ms = 0
# History diperpanjang agar bisa menangkap ayunan penuh kiri-ke-kanan
nose_history = deque(maxlen=25) 
direction_buffer = deque(maxlen=10)

movement_start_time = None
movement_confirm_time = None
movement_stop_time = None
is_moving = False
movement_counted = False

def get_head_pose(landmarks):
    nose = landmarks[1]; left = landmarks[234]
    right = landmarks[454]; chin = landmarks[152]
    forehead = landmarks[10]

    face_width = abs(right.x - left.x) or 1e-6
    face_height = abs(chin.y - forehead.y) or 1e-6

    center_x = (left.x + right.x) / 2
    center_y = (forehead.y + chin.y) / 2 

    yaw = (nose.x - center_x) / face_width
    pitch = (nose.y - center_y) / face_height 
    
    return yaw, pitch

def classify_direction(yaw, pitch):
    if pitch <= -PITCH_UP_THRESHOLD: return "ATAS"
    if pitch >= PITCH_DOWN_THRESHOLD: return "BAWAH"
    if yaw <= -YAW_THRESHOLD: return "KIRI"
    if yaw >= YAW_THRESHOLD: return "KANAN"
    return "NETRAL" 

def get_head_movement(x, y):
    global is_moving, movement_start_time
    global movement_confirm_time, movement_stop_time, movement_counted

    nose_history.append((x, y))
    if len(nose_history) < 10: return False, False

    xs = [p[0] for p in nose_history]
    ys = [p[1] for p in nose_history]
    motion = max(max(xs) - min(xs), max(ys) - min(ys))
    now = time.time()

    if not is_moving:
        if motion > MOVE_START_THRESHOLD:
            if movement_confirm_time is None: movement_confirm_time = now
            elif now - movement_confirm_time >= MOVE_CONFIRM_TIME:
                is_moving = True
                movement_start_time = now
                movement_counted = False
        else:
            movement_confirm_time = None
        return False, False

    duration = now - movement_start_time if movement_start_time is not None else 0

    if motion < MOVE_STOP_THRESHOLD:
        if movement_stop_time is None: movement_stop_time = now
        elif now - movement_stop_time >= MOVE_RELEASE_TIME:
            is_moving = False
            movement_confirm_time = None; movement_stop_time = None; movement_start_time = None
            movement_counted = False
            return False, False
    else:
        movement_stop_time = None

    if duration >= MOVE_SECONDS_THRESHOLD:
        if not movement_counted:
            movement_counted = True
        return True, True # is_moving=True, event=True (sudah > 5s)
        
    return True, False # is_moving=True, event=False (masih < 5s)

def analyze_head(frame):
    global timestamp_ms

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    timestamp_ms += 33

    if not result.face_landmarks:
        direction_buffer.append("NO FACE")
        smooth_dir = Counter(direction_buffer).most_common(1)[0][0]
        return {
            "direction": smooth_dir, 
            "bad_direction": False, 
            "movement_event": False, 
            "yaw": 0, 
            "pitch": 0,
            "landmarks": None
        }

    lm = result.face_landmarks[0]
    nose = lm[1]
    
    yaw, pitch = get_head_pose(lm)
    raw_dir = classify_direction(yaw, pitch)
    
    direction_buffer.append(raw_dir)
    smooth_dir = Counter(direction_buffer).most_common(1)[0][0]

    # Cek apakah kepala sedang goyang/bergerak
    is_moving_now, movement_event = get_head_movement(nose.x, nose.y)

    # ==========================================
    # LOGIKA PRIORITAS GERAKAN
    # ==========================================
    if is_moving_now:
        # Tulis di layar bahwa sedang "MOVING"
        smooth_dir = "BERGERAK"
        # Matikan sementara alarm menoleh agar timer gerak 5 detik bisa berjalan!
        bad_direction = False 
    else:
        # Jika kepala diam, baru cek apakah posisinya sedang melihat ke tempat terlarang
        bad_direction = smooth_dir in ["KIRI", "KANAN", "ATAS"]

    return {
        "direction": smooth_dir,
        "bad_direction": bad_direction,
        "movement_event": movement_event,
        "yaw": yaw,
        "pitch": pitch,
        "landmarks": result.face_landmarks[0] if result.face_landmarks else None
    }
