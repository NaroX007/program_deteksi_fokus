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

# ============================================================
# KONFIGURASI THRESHOLD — UBAH DI SINI UNTUK KALIBRASI
# ============================================================

# --- THRESHOLD ARAH PANDANGAN ---
YAW_THRESHOLD = 0.20        # Kiri/Kanan (makin kecil = makin sensitif)
PITCH_UP_THRESHOLD = 0.11   # Atas (mendongak)
PITCH_DOWN_THRESHOLD = 0.15 # Bawah (menunduk)

# --- THRESHOLD GERAKAN KEPALA ---
# Berapa lama gerakan kepala harus berlangsung sebelum dianggap off-task
MOVE_SECONDS_THRESHOLD = 5.0   # ← UBAH DI SINI (detik). Default 5.0

# Velocity threshold (sensitivitas deteksi gerakan)
MOVE_START_THRESHOLD = 0.030   # Velocity rata-rata untuk mulai deteksi gerakan
MOVE_STOP_THRESHOLD = 0.005    # Velocity di bawah ini = kepala diam
MOVE_CONFIRM_TIME = 0.5        # Waktu konfirmasi gerakan (detik)
MOVE_RELEASE_TIME = 0.3        # Waktu delay sebelum dianggap berhenti

# --- DEBOUNCE ARAH ---
# Saat arah berubah (mis. TENGAH → KIRI), sistem menahan arah lama
# selama DIRECTION_HOLD_TIME detik sebelum mengubah tampilan
# Tujuan: mencegah flicker "KIRI → BERGERAK → KIRI"
DIRECTION_HOLD_TIME = 0.3      # ← UBAH DI SINI (detik). 0.3 = 300ms delay
                                # Makin besar = makin stabil tapi kurang responsif
                                # Makin kecil = makin responsif tapi mudah flicker

# --- BUFFER SIZE ---
nose_history = deque(maxlen=10)
direction_buffer = deque(maxlen=5)  # Voting mayoritas untuk arah

# ============================================================
# STATE GERAKAN KEPALA
# ============================================================
movement_start_time = None
movement_confirm_time = None
movement_stop_time = None
is_moving = False
movement_counted = False

# ============================================================
# STATE DEBOUNCE ARAH
# ============================================================
confirmed_direction = "NETRAL"   # Arah yang sudah dikonfirmasi (ditampilkan)
direction_hold_start = None      # Waktu mulai hold arah baru
pending_direction = None         # Arah baru yang sedang di-hold

# ============================================================
# STATE TIMESTAMP
# ============================================================
timestamp_ms = 0


def get_head_pose(landmarks):
    """
    Hitung yaw dan pitch berdasarkan posisi landmark wajah.

    Yaw = (nose.x - center_x) / face_width
        - Negatif = menoleh kiri
        - Positif = menoleh kanan

    Pitch = (nose.y - center_y) / face_height
        - Negatif = mendongak ke atas
        - Positif = menunduk ke bawah
    """
    nose = landmarks[1]
    left = landmarks[234]
    right = landmarks[454]
    chin = landmarks[152]
    forehead = landmarks[10]

    face_width = abs(right.x - left.x) or 1e-6
    face_height = abs(chin.y - forehead.y) or 1e-6

    center_x = (left.x + right.x) / 2
    center_y = (forehead.y + chin.y) / 2

    yaw = (nose.x - center_x) / face_width
    pitch = (nose.y - center_y) / face_height

    return yaw, pitch


def classify_direction(yaw, pitch):
    """
    Klasifikasi arah pandangan ke 5 kategori.

    Prioritas: ATAS/BAWAH dulu (karena pitch lebih spesifik),
    baru KIRI/KANAN.
    """
    if pitch <= -PITCH_UP_THRESHOLD: return "ATAS"
    if pitch >= PITCH_DOWN_THRESHOLD: return "BAWAH"
    if yaw <= -YAW_THRESHOLD: return "KIRI"
    if yaw >= YAW_THRESHOLD: return "KANAN"
    return "NETRAL"


def get_head_movement(x, y):
    """
    Deteksi gerakan kepala berdasarkan VELOCITY (perubahan posisi per frame).

    State Machine:
        DIAM → (velocity > START dan tahan CONFIRM_TIME) → BERGERAK
        BERGERAK → (velocity < STOP dan tahan RELEASE_TIME) → DIAM
        BERGERAK → (durasi ≥ MOVE_SECONDS) → movement_event = True (off-task)

    Returns:
        (is_moving_now: bool, movement_event: bool, avg_velocity: float)
        - is_moving_now: True jika kepala sedang bergerak saat ini
        - movement_event: True jika gerakan sudah berlangsung > MOVE_SECONDS_THRESHOLD
        - avg_velocity: rata-rata velocity 5 frame terakhir (untuk debug/display)
    """
    global is_moving, movement_start_time
    global movement_confirm_time, movement_stop_time, movement_counted

    nose_history.append((x, y))
    if len(nose_history) < 3:
        return False, False, 0.0

    # Hitung velocity (perubahan posisi) antar frame
    # Pakai Chebyshev distance = max(|dx|, |dy|)
    recent = list(nose_history)
    velocities = []
    for i in range(1, len(recent)):
        dx = abs(recent[i][0] - recent[i-1][0])
        dy = abs(recent[i][1] - recent[i-1][1])
        velocities.append(max(dx, dy))

    # Rata-rata velocity 5 frame terakhir (moving average N=5)
    recent_velocities = velocities[-5:] if len(velocities) >= 5 else velocities
    avg_velocity = sum(recent_velocities) / len(recent_velocities) if recent_velocities else 0

    now = time.time()

    # STATE: SEDANG DIAM — cek apakah mulai bergerak
    if not is_moving:
        if avg_velocity > MOVE_START_THRESHOLD:
            if movement_confirm_time is None:
                movement_confirm_time = now
            elif now - movement_confirm_time >= MOVE_CONFIRM_TIME:
                # Konfirmasi gerakan dimulai
                is_moving = True
                movement_start_time = now
                movement_counted = False
        else:
            # Velocity turun di bawah threshold → reset confirm timer
            movement_confirm_time = None
        return False, False, round(avg_velocity, 4)

    # STATE: SEDANG BERGERAK — cek apakah berhenti
    duration = now - movement_start_time if movement_start_time is not None else 0

    if avg_velocity < MOVE_STOP_THRESHOLD:
        if movement_stop_time is None:
            movement_stop_time = now
        elif now - movement_stop_time >= MOVE_RELEASE_TIME:
            # Konfirmasi berhenti
            is_moving = False
            movement_confirm_time = None
            movement_stop_time = None
            movement_start_time = None
            movement_counted = False
            return False, False, round(avg_velocity, 4)
    else:
        # Masih bergerak, reset stop timer
        movement_stop_time = None

    # Cek apakah gerakan sudah > MOVE_SECONDS_THRESHOLD (off-task event)
    if duration >= MOVE_SECONDS_THRESHOLD:
        if not movement_counted:
            movement_counted = True
        return True, True, round(avg_velocity, 4)  # is_moving=True, event=True

    return True, False, round(avg_velocity, 4)  # is_moving=True, event=False


def get_debounced_direction(raw_direction):
    """
    Time-based hold untuk arah pandangan.

    Saat raw_direction berubah, sistem menahan arah lama selama
    DIRECTION_HOLD_TIME detik sebelum mengubah confirmed_direction.

    Tujuan: mencegah flicker saat arah berubah cepat
    (mis. TENGAH → KIRI → TENGAH → KIRI dalam hitungan milidetik)
    """
    global confirmed_direction, direction_hold_start, pending_direction

    now = time.time()

    if raw_direction == confirmed_direction:
        # Arah sama dengan yang sudah dikonfirmasi → reset hold
        direction_hold_start = None
        pending_direction = None
        return confirmed_direction

    # Arah berbeda dari yang dikonfirmasi
    if pending_direction != raw_direction:
        # Arah baru berbeda dari yang sedang di-hold → mulai hold baru
        pending_direction = raw_direction
        direction_hold_start = now
    elif direction_hold_start is not None:
        # Cek apakah hold time sudah terpenuhi
        elapsed = now - direction_hold_start
        if elapsed >= DIRECTION_HOLD_TIME:
            # KONFIRMASI perubahan arah
            confirmed_direction = raw_direction
            direction_hold_start = None
            pending_direction = None

    return confirmed_direction


def analyze_head(frame):
    """
    Analisis wajah untuk mendeteksi arah pandangan dan gerakan kepala.

    Args:
        frame: RGB image (numpy array)

    Returns:
        dict dengan field:
            - direction: arah pandangan (TENGAH/KIRI/KANAN/ATAS/BAWAH/NO FACE)
            - bad_direction: True jika arah menyimpang (off-task)
            - movement_event: True jika gerakan >5 detik
            - is_moving: True jika kepala sedang bergerak saat ini
            - velocity: rata-rata velocity 5 frame terakhir
            - yaw: nilai yaw ternormalisasi
            - pitch: nilai pitch ternormalisasi
            - landmarks: list landmark MediaPipe (atau None)
    """
    global timestamp_ms

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    timestamp_ms += 33

    # ============================================================
    # KASUS 1: TIDAK ADA WAJAH TERDETEKSI
    # ============================================================
    if not result.face_landmarks:
        direction_buffer.append("NO FACE")
        smooth_dir = Counter(direction_buffer).most_common(1)[0][0]
        return {
            "direction": smooth_dir,
            "bad_direction": False,
            "movement_event": False,
            "is_moving": False,
            "velocity": 0.0,
            "yaw": 0,
            "pitch": 0,
            "landmarks": None
        }

    # ============================================================
    # KASUS 2: WAJAH TERDETEKSI
    # ============================================================
    lm = result.face_landmarks[0]
    nose = lm[1]

    yaw, pitch = get_head_pose(lm)
    raw_dir = classify_direction(yaw, pitch)

    # Smoothing Level 1: Voting mayoritas dari 5 frame terakhir
    direction_buffer.append(raw_dir)
    smooth_dir = Counter(direction_buffer).most_common(1)[0][0]

    # Smoothing Level 2: Time-based hold (mencegah flicker cepat)
    smooth_dir = get_debounced_direction(smooth_dir)

    # Deteksi gerakan kepala
    is_moving_now, movement_event, avg_velocity = get_head_movement(nose.x, nose.y)

    # ============================================================
    # LOGIKA PRIORITAS GERAKAN (REVISI: OPSI G + OVERRIDE MENUNDUK)
    # ============================================================
    # Cek yaw dan pitch secara TERPISAH (tidak hanya andalkan smooth_dir)
    # Tujuan: tangani kasus anak menoleh + menunduk secara bersamaan
    #
    # Skenario yang ditangani:
    #   - Anak menunduk melihat kartu di meja → FOKUS (meski menoleh sedikit)
    #   - Anak menoleh dengan kepala tegak → OFF-TASK
    #   - Anak mendongak → OFF-TASK
    #   - Anak menggerakkan kepala >5s + menunduk → FOKUS (Opsi G)
    #   - Anak menggerakkan kepala >5s + menoleh tegak → OFF-TASK (Opsi G)
    #
    # ⭐ OVERRIDE TAMBAHAN (baru):
    #   - Jika anak menunduk (pitch_down), movement_event di-FORCE ke False
    #   - Alasan: Anak sedang fokus bermain kartu di meja → gerakan kepala
    #     yang terjadi selama menunduk TIDAK dihitung sebagai off-task
    #   - Trigger "Kepala Bergerak > 5 detik" tidak akan aktif di fusion tracker
    # ============================================================

    yaw_deviation = (yaw <= -YAW_THRESHOLD) or (yaw >= YAW_THRESHOLD)
    pitch_down = pitch >= PITCH_DOWN_THRESHOLD   # Menunduk = fokus ke kartu
    pitch_up = pitch <= -PITCH_UP_THRESHOLD      # Mendongak = off-task

    # ⭐ OVERRIDE: Jika anak menunduk, gerakan kepala TIDAK dihitung sebagai off-task
    # Ini menonaktifkan trigger "Kepala Bergerak > 5 detik" di fusion tracker
    # sehingga anak yang asyik bermain kartu (menunduk + gerak kepala) tetap FOKUS
    if pitch_down and movement_event:
        movement_event = False  # Override: anak menunduk = fokus ke kartu

    if is_moving_now:
        if movement_event:
            # OPSI G: Gerakan >5 detik — cek kombinasi velocity + arah + durasi
            # Pengecualian: anak menunduk = sedang melihat permainan di meja
            if pitch_down:
                bad_direction = False  # Menunduk = FOKUS ke kartu
            elif pitch_up:
                bad_direction = True   # Mendongak = OFF-TASK
            else:
                # Pitch netral — cek yaw (menoleh ekstrim tanpa menunduk)
                bad_direction = yaw_deviation
        else:
            # Gerakan masih <5 detik, belum off-task
            bad_direction = False
    else:
        # Kepala diam — cek posisi arah pandangan saat ini
        if pitch_down:
            bad_direction = False  # Menunduk = FOKUS ke permainan
        elif pitch_up:
            bad_direction = True   # Mendongak = OFF-TASK
        else:
            # Pitch netral — cek yaw
            bad_direction = yaw_deviation

    return {
        "direction": smooth_dir,
        "bad_direction": bad_direction,
        "movement_event": movement_event,
        "is_moving": is_moving_now,
        "velocity": avg_velocity,
        "yaw": yaw,
        "pitch": pitch,
        "landmarks": result.face_landmarks[0] if result.face_landmarks else None
    }


def reset_state():
    """
    Reset semua state modul kepala.
    Panggil di awal setiap sesi baru untuk menghindari data residual.
    """
    global is_moving, movement_start_time, movement_confirm_time
    global movement_stop_time, movement_counted
    global confirmed_direction, direction_hold_start, pending_direction
    global nose_history, direction_buffer
    global timestamp_ms

    nose_history.clear()
    direction_buffer.clear()
    is_moving = False
    movement_start_time = None
    movement_confirm_time = None
    movement_stop_time = None
    movement_counted = False
    confirmed_direction = "NETRAL"
    direction_hold_start = None
    pending_direction = None
    timestamp_ms = 0
