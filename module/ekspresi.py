import mediapipe as mp
import cv2
import time
import os
from collections import deque, Counter

# ===== PATH MODEL =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "model", "face_landmarker.task"))

# ===== INIT =====
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    output_face_blendshapes=True,
    num_faces=1,

    # 🔽 DITURUNKAN → lebih toleran (kacamata, lighting)
    min_face_detection_confidence=0.4,
    min_face_presence_confidence=0.4,
    min_tracking_confidence=0.4
)

landmarker = FaceLandmarker.create_from_options(options)

timestamp = 0

# ===== SMOOTHING =====
history = deque(maxlen=10)

# ===== HOLD TIME =====
last_label = None
label_start_time = 0
HOLD_TIME = 1.5  # detik


def get_majority_label():
    if len(history) == 0:
        return "UNKNOWN"
    return Counter(history).most_common(1)[0][0]


def analyze_expression(frame):
    global timestamp, last_label, label_start_time

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )

    result = landmarker.detect_for_video(mp_image, timestamp)
    timestamp += 33

    # ===== NO FACE =====
    if not result.face_blendshapes:
        history.append("UNKNOWN")
        return {
            "label": "UNKNOWN",
            "not_focus": False,
            "reason": "NO_FACE"
        }

    bs = result.face_blendshapes[0]
    scores = {b.category_name: b.score for b in bs}

    # ===== FITUR =====
    sad = scores.get("mouthFrownLeft", 0) + scores.get("mouthFrownRight", 0)
    brow = scores.get("browDownLeft", 0) + scores.get("browDownRight", 0)
    eye = scores.get("eyeSquintLeft", 0) + scores.get("eyeSquintRight", 0)
    smile = scores.get("mouthSmileLeft", 0) + scores.get("mouthSmileRight", 0)

    # 🔥 TAMBAHAN: DETEKSI MENGUAP
    yawn = scores.get("jawOpen", 0)

    # ===== LOGIKA RAW =====
    if yawn > 0.6:
        raw_label = "TIDAK FOKUS"
        reason = "MENGUAP"

    elif (sad > 0.5 and brow > 0.4) or (brow > 0.6 and eye > 0.4):
        raw_label = "TIDAK FOKUS"
        reason = "EKSPRESI_NEGATIF"

    else:
        raw_label = "NETRAL"
        reason = "NORMAL"

    # ===== SMOOTHING =====
    history.append(raw_label)
    smooth_label = get_majority_label()

    # ===== HOLD TIME =====
    now = time.time()

    if smooth_label != last_label:
        last_label = smooth_label
        label_start_time = now

    duration = now - label_start_time

    if smooth_label == "TIDAK FOKUS" and duration >= HOLD_TIME:
        final_label = "TIDAK FOKUS"
    else:
        final_label = "NETRAL"

    return {
        "label": final_label,
        "not_focus": final_label == "TIDAK FOKUS",
        "reason": reason
    }
