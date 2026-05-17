import os
import cv2
import mediapipe as mp
import time

# ======================
# MODEL PATH & INIT
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "model", "pose_landmarker_lite.task"))

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1
)
landmarker = PoseLandmarker.create_from_options(options)

timestamp_ms = 0

# ======================
# THRESHOLDS & STATE (DIKALIBRASI UNTUK DUDUK MEJA)
# ======================
LEFT_SHOULDER = 11; RIGHT_SHOULDER = 12
LEFT_HIP = 23; RIGHT_HIP = 24

# Toleransi visibilitas diturunkan (0.4) agar lebih aman untuk pakaian longgar/hijab
VISIBILITY_THRESHOLD = 0.4 

# Batas pundak dinaikkan (0.25) karena saat menunduk/condong ke meja, pundak akan naik di frame
SHOULDER_Y_MIN = 0.25 

HOLD_TIME = 1.2

candidate_label = None
candidate_start_time = 0
stable_label = "UNKNOWN"

def analyze_body(frame):
    global timestamp_ms, candidate_label, candidate_start_time, stable_label
    now = time.time()
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    timestamp_ms += 33
    
    raw_label = "UNKNOWN"
    
    if result.pose_landmarks:
        lm = result.pose_landmarks[0]
        
        l_sh = lm[LEFT_SHOULDER]; r_sh = lm[RIGHT_SHOULDER]
        l_hip = lm[LEFT_HIP]; r_hip = lm[RIGHT_HIP]
        
        # 1. Cek apakah kedua bahu terlihat di layar (sesuai threshold baru)
        shoulders_visible = (l_sh.visibility > VISIBILITY_THRESHOLD and r_sh.visibility > VISIBILITY_THRESHOLD)
        
        # 2. Ambil posisi rata-rata bahu secara vertikal (0.0 = pucuk atas frame, 1.0 = dasar frame)
        shoulder_y = (l_sh.y + r_sh.y) / 2
        
        # 3. Cek pinggul. Model AI kadang berhalusinasi menebak pinggul di balik meja.
        # Kita hanya anggap dia berdiri JIKA pinggul terlihat TINGGI/NAIK di layar (y < 0.75).
        hip_is_high = False
        if (l_hip.visibility > VISIBILITY_THRESHOLD and l_hip.y < 0.75) or \
           (r_hip.visibility > VISIBILITY_THRESHOLD and r_hip.y < 0.75):
            hip_is_high = True
            
        # 4. LOGIKA KEPUTUSAN DUDUK DI MEJA
        if shoulders_visible and shoulder_y > SHOULDER_Y_MIN and not hip_is_high:
            raw_label = "DUDUK"
        else:
            raw_label = "TIDAK DUDUK"
            
    else:
        # Jika badan anak keluar sama sekali dari frame kamera
        raw_label = "TIDAK DUDUK"

    # Stabilisasi agar status tidak berkedip-kedip saat bergerak sedikit
    if raw_label != candidate_label:
        candidate_label = raw_label
        candidate_start_time = now

    if now - candidate_start_time >= HOLD_TIME:
        stable_label = candidate_label

    active_not_focus = (stable_label == "TIDAK DUDUK")

    return {
        "posture": stable_label,
        "active_not_focus": active_not_focus,
        "reason": "Postur Berubah" if active_not_focus else "OK",
        "landmarks": result.pose_landmarks[0] if result.pose_landmarks else None
    }
