import cv2
import time

from module.kepala import analyze_head
from module.postur_tubuh import analyze_body
from module.ekspresi import analyze_expression

from core.database import Database
from core.fusion import decide

# ======================
# INIT
# ======================
db = Database()
session_id = db.start_session()

cap = cv2.VideoCapture(0)

# ======================
# COUNTER
# ======================
head_move_count = 0
head_direction_count = 0
posture_stand_count = 0
expression_negative_count = 0

prev_posture = None
prev_expression = None

# ======================
# DURASI
# ======================
on_task_time = 0
off_task_time = 0

prev_time = time.time()

print("Session started... (Press Q / ESC / Ctrl+C to stop)")

# ======================
# LOOP
# ======================
try:
    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        delta_time = current_time - prev_time
        prev_time = current_time

        # ======================
        # ANALISIS
        # ======================
        head = analyze_head(frame)
        body = analyze_body(frame)
        expr = analyze_expression(frame)

        posture = body.get("posture", "UNKNOWN")
        expression = expr.get("label", "UNKNOWN")

        # ======================
        # EVENT LOG
        # ======================
        if head.get("movement_event", False):
            head_move_count += 1
            db.log_event(session_id, "HEAD_MOVE", "MOVING_10S")

        if head.get("direction_event", False):
            direction = head.get("direction", "UNKNOWN")
            head_direction_count += 1
            db.log_event(session_id, "HEAD_DIRECTION", direction)

        if posture == "BERDIRI" and prev_posture != "BERDIRI":
            posture_stand_count += 1
            db.log_event(session_id, "POSTURE", "BERDIRI")

        prev_posture = posture

        if expression == "TIDAK FOKUS" and prev_expression != "TIDAK FOKUS":
            expression_negative_count += 1
            db.log_event(session_id, "EXPRESSION", "NEGATIVE")

        prev_expression = expression

        # ======================
        # FUSION (tetap dipakai untuk durasi)
        # ======================
        is_focus = True

        if not head.get("focused", True):
            is_focus = False

        if posture == "BERDIRI":
            is_focus = False

        if expression == "TIDAK FOKUS":
            is_focus = False

        # ======================
        # HITUNG DURASI
        # ======================
        if is_focus:
            on_task_time += delta_time
        else:
            off_task_time += delta_time

        # ======================
        # DISPLAY (UPDATED)
        # ======================
        head_text = head.get("direction", "CENTER")

        # kalau lagi movement kuat
        if head.get("movement_event", False):
            head_text = "MOVING_10S"

        cv2.putText(frame, f"HEAD : {head_text}", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.putText(frame, f"POSTURE : {posture}", (20,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        cv2.putText(frame, f"EXPRESSION : {expression}", (20,120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,150,255), 2)

        cv2.imshow("Focus System", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

except KeyboardInterrupt:
    print("\nProgram dihentikan dengan Ctrl+C")

# ======================
# FINAL RESULT
# ======================
finally:
    final, ratio = decide(on_task_time, off_task_time)

    print("\n=== HASIL AKHIR ===")
    print(f"On-task: {on_task_time:.2f}s")
    print(f"Off-task: {off_task_time:.2f}s")
    print(f"Focus Ratio: {ratio*100:.2f}%")
    print(f"FINAL: {final}")

    db.save_summary(
        session_id,
        head_move_count,
        head_direction_count,
        posture_stand_count,
        expression_negative_count,
        on_task_time,
        off_task_time,
        ratio,
        final
    )

    db.end_session(session_id)
    cap.release()
    cv2.destroyAllWindows()
