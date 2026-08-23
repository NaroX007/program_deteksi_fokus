import cv2
import mediapipe as mp
import time
from collections import deque

# =========================================
# MEDIAPIPE - Pose (legacy solutions API)
# =========================================
mp_pose = mp.solutions.pose

# =========================================
# LANDMARK INDEX
# =========================================
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

# =========================================
# THRESHOLD (SATU ATURAN MUTLAK)
# =========================================
SHOULDER_THRESHOLD = 0.25
VISIBILITY_THRESHOLD = 0.5

# =========================================
# SMOOTHING CONFIG (Frame-based)
# =========================================
SMOOTH_WINDOW = 5
STABILITY_FRAMES = 3
STABILITY_THRESHOLD = 0.6

# =========================================
# TIME-BASED HOLD CONFIG
# =========================================
HOLD_TIME = 1.5
MIN_CONFIDENCE_RATIO = 0.7

# =========================================
# MODULE-LEVEL STATE (persistent across frames)
# =========================================
shoulder_y_buffer = deque(maxlen=SMOOTH_WINDOW)
posture_history = deque(maxlen=SMOOTH_WINDOW)

confirmed_posture = "TIDAK_DUDUK"
hold_start_time = None
hold_buffer = deque()

# =========================================
# POSE MODEL INSTANCE (module-level, lazy init)
# =========================================
_pose_instance = None


def _get_pose():
    """Lazy-load Pose instance agar tidak di-load saat import."""
    global _pose_instance
    if _pose_instance is None:
        _pose_instance = mp_pose.Pose(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
    return _pose_instance


def reset_state():
    """
    Reset semua state (buffer, hold timer, confirmed posture).
    Panggil fungsi ini di awal setiap sesi baru.
    """
    global confirmed_posture, hold_start_time
    shoulder_y_buffer.clear()
    posture_history.clear()
    hold_buffer.clear()
    confirmed_posture = "TIDAK_DUDUK"
    hold_start_time = None


# =========================================
# MAIN ANALYZER (untuk dipanggil dari main_tes / main / benchmark)
# =========================================
def analyze_body(frame):
    """
    Menerima frame BGR dari OpenCV, mengembalikan hasil analisis postur tubuh.

    Menggunakan SATU ATURAN MUTLAK berdasarkan posisi Y bahu (shoulder_y).
    - shoulder_y > 0.25 -> DUDUK
    - shoulder_y <= 0.25 -> TIDAK_DUDUK

    Returns:
        dict dengan keys:
            posture (str)        : DUDUK / TIDAK_DUDUK (status final)
            posture_raw (str)    : Status raw sebelum smoothing
            posture_smoothed (str): Status setelah smoothing, sebelum time-hold
            active_not_focus (bool): True jika posture == TIDAK_DUDUK
            landmarks (obj)     : landmark MediaPipe Pose (list of landmark objects)
            reason (str)        : keterangan singkat
            debug (dict)        : nilai-nilai debug (shoulder_y, facing_camera, dll)
    """
    global confirmed_posture, hold_start_time

    current_time = time.time()
    pose = _get_pose()

    # Process frame (TANPA mirror, sesuai kode original teman)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    # Default values
    posture_raw = "NO_DETECTION"
    posture_smoothed = "NO_DETECTION"
    posture_final = confirmed_posture

    shoulder_y_raw = 0.0
    shoulder_y_smoothed = 0.0
    facing_camera = False
    pending_change = False
    hold_progress = 0.0

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # =========================================
        # VISIBILITY & FACING CAMERA
        # =========================================
        left_shoulder_vis = landmarks[LEFT_SHOULDER].visibility
        right_shoulder_vis = landmarks[RIGHT_SHOULDER].visibility

        facing_camera = (
            left_shoulder_vis > VISIBILITY_THRESHOLD and
            right_shoulder_vis > VISIBILITY_THRESHOLD
        )

        # =========================================
        # shoulder_y (RAW)
        # =========================================
        left_shoulder_y = landmarks[LEFT_SHOULDER].y
        right_shoulder_y = landmarks[RIGHT_SHOULDER].y
        shoulder_y_raw = (left_shoulder_y + right_shoulder_y) / 2

        # =========================================
        # SMOOTHING SHOULDER Y
        # =========================================
        if shoulder_y_raw > 0:
            shoulder_y_buffer.append(shoulder_y_raw)

        shoulder_y_smoothed = (sum(shoulder_y_buffer) / len(shoulder_y_buffer)
                               if shoulder_y_buffer else shoulder_y_raw)

        # =========================================
        # SATU ATURAN MUTLAK
        # =========================================
        if shoulder_y_smoothed > SHOULDER_THRESHOLD:
            posture_raw = 'DUDUK'
        else:
            posture_raw = 'TIDAK_DUDUK'

        # =========================================
        # SMOOTHING STATUS POSTURE (Voting Mayoritas)
        # =========================================
        posture_history.append(posture_raw)

        if len(posture_history) >= STABILITY_FRAMES:
            duduk_count = list(posture_history).count('DUDUK')
            tidak_count = list(posture_history).count('TIDAK_DUDUK')

            if duduk_count / len(posture_history) >= STABILITY_THRESHOLD:
                posture_smoothed = 'DUDUK'
            elif tidak_count / len(posture_history) >= STABILITY_THRESHOLD:
                posture_smoothed = 'TIDAK_DUDUK'
            else:
                posture_smoothed = 'DUDUK' if duduk_count > tidak_count else 'TIDAK_DUDUK'
        else:
            posture_smoothed = posture_raw

        # =========================================
        # TIME-BASED HOLD LOGIC
        # =========================================
        if posture_smoothed == confirmed_posture:
            hold_start_time = None
            hold_buffer.clear()
            posture_final = confirmed_posture
            pending_change = False
            hold_progress = 1.0
        else:
            if hold_start_time is None:
                hold_start_time = current_time

            hold_buffer.append((current_time, posture_smoothed))

            while hold_buffer and (current_time - hold_buffer[0][0]) > HOLD_TIME:
                hold_buffer.popleft()

            elapsed = current_time - hold_start_time
            hold_progress = min(elapsed / HOLD_TIME, 1.0)

            if len(hold_buffer) > 0:
                new_count = sum(1 for _, p in hold_buffer if p == posture_smoothed)
                confidence_ratio = new_count / len(hold_buffer)
            else:
                confidence_ratio = 0

            if elapsed >= HOLD_TIME and confidence_ratio >= MIN_CONFIDENCE_RATIO:
                confirmed_posture = posture_smoothed
                hold_start_time = None
                hold_buffer.clear()
                posture_final = confirmed_posture
                pending_change = False
                hold_progress = 1.0
            else:
                posture_final = confirmed_posture
                pending_change = True

    # =========================================
    # RETURN HASIL
    # =========================================
    active_not_focus = (posture_final == "TIDAK_DUDUK")

    return {
        "posture": posture_final,
        "posture_raw": posture_raw,
        "posture_smoothed": posture_smoothed,
        "active_not_focus": active_not_focus,
        "reason": "Postur Berubah" if active_not_focus else "OK",
        "landmarks": results.pose_landmarks.landmark if results.pose_landmarks else None,
        "debug": {
            "facing_camera": facing_camera,
            "shoulder_y_raw": round(shoulder_y_raw, 4),
            "shoulder_y_smoothed": round(shoulder_y_smoothed, 4),
            "pending_change": pending_change,
            "hold_progress": round(hold_progress, 3),
        }
    }


# =========================================
# STANDALONE TEST SCRIPT (untuk pengujian akurasi)
# =========================================
if __name__ == "__main__":
    import csv
    import os
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    import pandas as pd

    CSV_FILE = "hasil_uji.csv"

    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "actual", "predicted_raw", "predicted_smoothed",
                "predicted_final", "facing_camera",
                "shoulder_y_raw", "shoulder_y_smoothed",
                "pending_change", "hold_progress"
            ])

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("=" * 60)
    print("SISTEM DETEKSI DUDUK - SATU ATURAN MUTLAK")
    print("=" * 60)
    print("Tekan: d=DUDUK | t=TIDAK_DUDUK | q=Quit & hitung akurasi")
    print("=" * 60)

    reset_state()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = frame.copy()
        result = analyze_body(frame)

        posture = result["posture"]
        posture_raw = result["posture_raw"]
        posture_smoothed = result["posture_smoothed"]
        posture_final = result["posture"]
        pending_change = result["debug"]["pending_change"]
        hold_progress = result["debug"]["hold_progress"]
        shoulder_y_raw = result["debug"]["shoulder_y_raw"]
        shoulder_y_smoothed = result["debug"]["shoulder_y_smoothed"]
        facing_camera = result["debug"]["facing_camera"]

        # Color
        if posture == 'DUDUK':
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        # Draw landmarks (bahu saja)
        if result["landmarks"]:
            landmarks = result["landmarks"]
            h, w, _ = image.shape
            for idx in [LEFT_SHOULDER, RIGHT_SHOULDER]:
                lm = landmarks[idx]
                if lm.visibility > 0.5:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(image, (cx, cy), 10, (0, 255, 0), -1)

            ls = landmarks[LEFT_SHOULDER]
            rs = landmarks[RIGHT_SHOULDER]
            if ls.visibility > 0.5 and rs.visibility > 0.5:
                x1, y1 = int(ls.x * w), int(ls.y * h)
                x2, y2 = int(rs.x * w), int(rs.y * h)
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # UI Text
        cv2.putText(image, f'POSTUR : {posture}',
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)
        cv2.putText(image, f'RULE: shoulder_y > {SHOULDER_THRESHOLD} -> DUDUK',
                    (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(image, f'SHOULDER Y: {shoulder_y_raw:.3f} -> {shoulder_y_smoothed:.3f}',
                    (30, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(image, f'FACING CAMERA: {facing_camera}',
                    (30, 258), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(image, f'Posture RAW      : {posture_raw}',
                    (30, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if posture_raw == 'DUDUK' else (0, 0, 255), 2)
        cv2.putText(image, f'Posture SMOOTHED : {posture_smoothed}',
                    (30, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if posture_smoothed == 'DUDUK' else (0, 0, 255), 2)

        if pending_change:
            final_text = f'Posture FINAL    : {posture_final}  [HOLDING: {hold_progress*100:.0f}%]'
            final_color = (0, 165, 255)
        else:
            final_text = f'Posture FINAL    : {posture_final}  [CONFIRMED]'
            final_color = (0, 255, 0) if posture_final == 'DUDUK' else (0, 0, 255)

        cv2.putText(image, final_text,
                    (30, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.55, final_color, 2)

        cv2.putText(image, 'd=DUDUK | t=TIDAK_DUDUK | q=Quit',
                    (30, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Deteksi Duduk - Shoulder Y Only", image)

        key = cv2.waitKey(10) & 0xFF

        if key == ord('d') or key == ord('t'):
            actual = "DUDUK" if key == ord('d') else "TIDAK_DUDUK"
            with open(CSV_FILE, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S"), actual, posture_raw, posture_smoothed, posture_final,
                    facing_camera, shoulder_y_raw, shoulder_y_smoothed, pending_change, hold_progress
                ])
            print(f"Saved -> {actual} | Raw: {posture_raw} | Smoothed: {posture_smoothed} | Final: {posture_final}")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Hitung Akurasi
    print("\n" + "=" * 60)
    print("MENGHITUNG AKURASI")
    print("=" * 60)

    df = pd.read_csv(CSV_FILE)
    df = df[df["predicted_final"] != "NO_DETECTION"]

    if len(df) > 0:
        y_true = df["actual"]
        y_pred_final = df["predicted_final"]
        accuracy_final = accuracy_score(y_true, y_pred_final) * 100
        print(f"Akurasi FINAL    : {accuracy_final:.2f}%")
        print("Confusion Matrix (FINAL):")
        print(confusion_matrix(y_true, y_pred_final))
        print(classification_report(y_true, y_pred_final))
