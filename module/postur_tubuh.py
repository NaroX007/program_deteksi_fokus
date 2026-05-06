import mediapipe as mp
import cv2
import time
import os

# =============================
# PATH MODEL
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "..", "model", "pose_landmarker_lite.task")
)

# =============================
# INIT MEDIAPIPE
# =============================
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

# =============================
# CONFIG
# =============================
# Visibility diturunkan sedikit agar lebih toleran terhadap jarak jauh
VIS_THR = 0.30 
HOLD_TIME = 1.2

# =============================
# STATE
# =============================
candidate_label = None
candidate_start_time = 0
stable_label = "UNKNOWN"
stand_event_fired = False

def analyze_body(frame):
    global timestamp_ms
    global candidate_label, candidate_start_time, stable_label, stand_event_fired

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    timestamp_ms += 33
    now = time.time()

    raw_label = "UNKNOWN"

    if result.pose_landmarks:
        lm = result.pose_landmarks[0]

        # Ambil poin utama
        nose = lm[0]
        shoulder = lm[11] # Bahu kiri sebagai sampel
        hip = lm[23]      # Pinggul kiri sebagai sampel
        knee = lm[25]     # Lutut kiri sebagai sampel

        # Cek apakah poin tersebut "Valid" (terlihat dan di dalam frame)
        def check(point):
            return point.visibility > VIS_THR and 0.0 <= point.y <= 1.0

        face_visible = check(nose)
        shoulder_visible = check(shoulder)
        hip_visible = check(hip)
        knee_visible = check(knee)

        # ==========================================
        # LOGIKA DEFAULT (SIMPEL)
        # ==========================================

        # 1. Jika Paha & Lutut terlihat jelas (Utamakan Berdiri)
        if hip_visible and knee_visible:
            # Perbandingan vertikal sederhana: jika jarak pinggul ke lutut signifikan
            if abs(knee.y - hip.y) > 0.15:
                raw_label = "BERDIRI"
            else:
                raw_label = "DUDUK"

        # 2. Jika hanya badan atas (Kasus jarak dekat/kepotong)
        elif shoulder_visible:
            if face_visible:
                # Bahu ada + Wajah ada = DUDUK
                raw_label = "DUDUK"
            else:
                # Bahu ada + Wajah TIDAK ada (kepotong ke atas) = BERDIRI
                raw_label = "BERDIRI"
        
        else:
            raw_label = "UNKNOWN"

    # =============================
    # STABILISASI (Agar tidak lompat-lompat)
    # =============================
    if raw_label != candidate_label:
        candidate_label = raw_label
        candidate_start_time = now

    # Jika label bertahan selama HOLD_TIME, baru update stable_label
    if now - candidate_start_time >= HOLD_TIME:
        stable_label = candidate_label

    # =============================
    # EVENT
    # =============================
    active_not_focus = (stable_label == "BERDIRI")
    stand_event = False

    if stable_label == "BERDIRI":
        if not stand_event_fired:
            stand_event = True
            stand_event_fired = True
    else:
        stand_event_fired = False

    return {
        "posture": stable_label,
        "active_not_focus": active_not_focus,
        "stand_event": stand_event
    }
