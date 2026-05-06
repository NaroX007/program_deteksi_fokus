import os
import time
from collections import deque

import cv2
import mediapipe as mp

# ======================
# MODEL PATH
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "model", "face_landmarker.task"))

# ======================
# MEDIAPIPE INIT
# ======================
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
# THRESHOLD
# ======================
YAW_THRESHOLD = 0.20
PITCH_UP_THRESHOLD = 0.18
PITCH_DOWN_THRESHOLD = 0.45

MOVE_START_THRESHOLD = 0.050
MOVE_STOP_THRESHOLD = 0.035
MOVE_CONFIRM_TIME = 0.6
MOVE_RELEASE_TIME = 0.7
MOVE_SECONDS_THRESHOLD = 10.0

DIRECTION_HOLD_TIME = 5.0

# ======================
# STATE
# ======================
nose_history = deque(maxlen=15)

timestamp_ms = 0

movement_start_time = None
movement_confirm_time = None
movement_stop_time = None
is_moving = False
movement_counted = False

current_direction = None
direction_start_time = None
direction_counted = False


def get_head_pose(landmarks):
    nose = landmarks[1]
    left = landmarks[234]
    right = landmarks[454]
    chin = landmarks[152]
    forehead = landmarks[10]

    face_width = abs(right.x - left.x) or 1e-6
    face_height = abs(chin.y - forehead.y) or 1e-6

    center_x = (left.x + right.x) / 2
    yaw = (nose.x - center_x) / face_width

    pitch = (nose.y - forehead.y) / face_height
    return yaw, pitch


def classify_direction(yaw, pitch):
    """
    Fokus tidak fokus:
    - LEFT
    - RIGHT
    - UP
    Jika masih CENTER / DOWN, belum dianggap tidak fokus dari arah kepala.
    """
    if pitch <= -PITCH_UP_THRESHOLD:
        return "UP"
    if yaw <= -YAW_THRESHOLD:
        return "LEFT"
    if yaw >= YAW_THRESHOLD:
        return "RIGHT"
    return "CENTER"


def get_head_movement(x, y):
    global is_moving, movement_start_time
    global movement_confirm_time, movement_stop_time, movement_counted

    nose_history.append((x, y))

    if len(nose_history) < 10:
        return "STABLE", False, False, 0

    xs = [p[0] for p in nose_history]
    ys = [p[1] for p in nose_history]
    motion = max(max(xs) - min(xs), max(ys) - min(ys))
    now = time.time()

    # Belum bergerak
    if not is_moving:
        if motion > MOVE_START_THRESHOLD:
            if movement_confirm_time is None:
                movement_confirm_time = now
            elif now - movement_confirm_time >= MOVE_CONFIRM_TIME:
                is_moving = True
                movement_start_time = now
                movement_counted = False
        else:
            movement_confirm_time = None

        return "STABLE", False, False, 0

    # Sedang bergerak
    duration = now - movement_start_time if movement_start_time is not None else 0

    if motion < MOVE_STOP_THRESHOLD:
        if movement_stop_time is None:
            movement_stop_time = now
        elif now - movement_stop_time >= MOVE_RELEASE_TIME:
            is_moving = False
            movement_confirm_time = None
            movement_stop_time = None
            movement_start_time = None
            movement_counted = False
            return "STABLE", False, False, 0
    else:
        movement_stop_time = None

    # Baru dihitung jika sudah 10 detik
    if duration >= MOVE_SECONDS_THRESHOLD:
        if not movement_counted:
            movement_counted = True
            return "MOVING_10S", True, True, duration
        return "MOVING_10S", True, False, duration

    return "MOVING", False, False, duration


def update_direction(raw_dir):
    global current_direction, direction_start_time, direction_counted

    now = time.time()

    if raw_dir == "CENTER":
        current_direction = None
        direction_start_time = None
        direction_counted = False
        return "CENTER", False, False, 0

    if raw_dir != current_direction:
        current_direction = raw_dir
        direction_start_time = now
        direction_counted = False
        return f"{raw_dir}_PENDING", False, False, 0

    duration = now - direction_start_time if direction_start_time is not None else 0

    if duration >= DIRECTION_HOLD_TIME:
        if not direction_counted:
            direction_counted = True
            return f"{raw_dir}_5S", True, True, duration
        return f"{raw_dir}_5S", True, False, duration

    return f"{raw_dir}_PENDING", False, False, duration


def analyze_head(frame):
    global timestamp_ms

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )

    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    timestamp_ms += 33

    if not result.face_landmarks:
        return {
            "focused": False,
            "movement": "NO_FACE",
            "movement_event": False,
            "movement_duration": 0,
            "direction": "NO_FACE",
            "direction_event": False,
            "direction_duration": 0,
            "yaw": 0,
            "pitch": 0,
            "reason": "NO_FACE"
        }

    lm = result.face_landmarks[0]

    nose = lm[1]

    # posisi hidung untuk deteksi gerakan
    nose_x = nose.x
    nose_y = nose.y

    yaw, pitch = get_head_pose(lm)
    raw_dir = classify_direction(yaw, pitch)

    movement, movement_active, movement_event, movement_duration = get_head_movement(nose_x, nose_y)
    direction, direction_active, direction_event, direction_duration = update_direction(raw_dir)

    looking_down = pitch > PITCH_DOWN_THRESHOLD
    facing_forward = abs(yaw) < 0.10

    # Fokus hanya jika:
    # - kepala menunduk ke meja
    # - tidak ada gerakan kepala yang sudah melewati 10 detik
    # - tidak ada arah kiri/kanan/atas yang sudah melewati 5 detik
    focused = looking_down and not direction_active

    if movement_event:
        reason = "HEAD_MOVING_10S"
    elif direction_event:
        reason = f"HEAD_{raw_dir}_5S"
    elif not looking_down:
        reason = "NOT_LOOKING_DOWN"
    elif not facing_forward and raw_dir == "CENTER":
        reason = "HEAD_NOT_CENTER"
    else:
        reason = "OK"

    return {
        "focused": focused,
        "movement": movement,
        "movement_event": movement_event,
        "movement_duration": movement_duration,
        "direction": direction,
        "direction_event": direction_event,
        "direction_duration": direction_duration,
        "yaw": yaw,
        "pitch": pitch,
        "looking_down": looking_down,
        "reason": reason
    }
