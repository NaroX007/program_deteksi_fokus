"""
main_tes.py - Sistem Deteksi Fokus Anak (STANDALONE MODE)
=========================================================
Mode standalone untuk pengujian tanpa koneksi web/permainan.
Menggunakan ID anak dan sesi dummy (TEST_ANAK_01, TEST_SESSION_01).

Cara pakai:
    python main_tes.py

Fitur:
- Deteksi real-time: arah kepala, gerakan kepala, postur tubuh, ekspresi
- Filter durasi minimum berdasarkan jenis trigger
- Pencatatan ke SQLite lokal (tanpa kirim Firebase)
- Bounding box wajah & tubuh untuk identifikasi subjek
- UI dengan status, sebab, FPS, dan legenda warna

Keyboard:
    [T] - Toggle titik landmark postur tubuh
    [K] - Toggle titik landmark kepala
    [E] - Toggle titik landmark ekspresi
    [Q] - Quit
"""

import cv2
import time
import threading
from datetime import datetime
from collections import deque

from module.kepala import analyze_head
from module.postur_tubuh import analyze_body
from module.ekspresi import analyze_expression

from core.database import Database
from core.fusion import FocusTracker
from core.firebase import send_session_data
from core.focus_api import (
    get_latest_active_session,
    save_focus_result,
)


# ============================================================
# KONFIGURASI SISTEM
# ============================================================
HEADLESS_MODE = False
STANDALONE_MODE = True           # Mode Standalone (Tanpa Koneksi Web/Permainan)
PROCESS_EVERY_N_FRAMES = 2        # Proses AI tiap 2 frame (responsivitas vs CPU)
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

MIN_OFF_TASK_DURATION = 3.0       # Filter durasi minimum untuk arah/postur/ekspresi
# ============================================================


class ThreadedCamera:
    """
    Kamera dengan thread terpisah untuk membaca frame.
    Buffer diatur ke 1 untuk meminimalkan delay (default V4L2: 2-5 frame).
    """

    def __init__(self, src=0, width=640, height=480):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            self.ret, self.frame = self.stream.read()

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()


# ============================================================
# HELPER: Text dengan background
# ============================================================
def draw_text_with_bg(frame, text, pos, font_scale=0.5, color=(255, 255, 255),
                      bg_color=(0, 0, 0), thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX,
                      padding=4):
    """Gambar text dengan background rectangle agar terlihat di semua warna."""
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    cv2.rectangle(frame,
                  (x - padding, y - text_h - padding),
                  (x + text_w + padding, y + padding),
                  bg_color, -1)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness,
                cv2.LINE_AA)


# ============================================================
# HELPER: Bounding box di sekitar landmark
# ============================================================
def draw_bounding_box(frame, landmarks, color=(0, 255, 0), label="TRACKED"):
    """
    Gambar bounding box di sekitar landmark yang terdeteksi.
    Berguna saat ada 2 objek di frame untuk identifikasi subjek utama.
    """
    if not landmarks:
        return

    h_frame, w_frame, _ = frame.shape
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    padding_ratio = 0.02
    x_pad = (x_max - x_min) * padding_ratio
    y_pad = (y_max - y_min) * padding_ratio

    x_min_px = int(max(0, (x_min - x_pad) * w_frame))
    x_max_px = int(min(w_frame, (x_max + x_pad) * w_frame))
    y_min_px = int(max(0, (y_min - y_pad) * h_frame))
    y_max_px = int(min(h_frame, (y_max + y_pad) * h_frame))

    cv2.rectangle(frame, (x_min_px, y_min_px), (x_max_px, y_max_px), color, 2)

    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame,
                  (x_min_px, max(0, y_min_px - label_h - 8)),
                  (x_min_px + label_w + 8, y_min_px),
                  color, -1)
    cv2.putText(frame, label,
                (x_min_px + 4, y_min_px - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    corner_len = 12
    cv2.line(frame, (x_min_px, y_min_px), (x_min_px + corner_len, y_min_px), color, 3)
    cv2.line(frame, (x_min_px, y_min_px), (x_min_px, y_min_px + corner_len), color, 3)
    cv2.line(frame, (x_max_px, y_min_px), (x_max_px - corner_len, y_min_px), color, 3)
    cv2.line(frame, (x_max_px, y_min_px), (x_max_px, y_min_px + corner_len), color, 3)
    cv2.line(frame, (x_min_px, y_max_px), (x_min_px + corner_len, y_max_px), color, 3)
    cv2.line(frame, (x_min_px, y_max_px), (x_min_px, y_max_px - corner_len), color, 3)
    cv2.line(frame, (x_max_px, y_max_px), (x_max_px - corner_len, y_max_px), color, 3)
    cv2.line(frame, (x_max_px, y_max_px), (x_max_px, y_max_px - corner_len), color, 3)


def get_min_duration(triggers):
    """
    Tentukan batas minimal durasi berdasarkan jenis trigger.
    - Gerakan kepala >5s: 1.0 detik (karena 5 detik sudah dihitung di modul kepala)
    - Arah/postur/ekspresi: 3.0 detik (MIN_OFF_TASK_DURATION)
    """
    if "Kepala Bergerak > 5 detik" in triggers:
        return 1.0
    return MIN_OFF_TASK_DURATION


# ============================================================
# LOOP AI UTAMA
# ============================================================
def run_ai_loop(anak_id, session_id):
    print(f"\n[AI START] Kamera menyala. Memulai deteksi untuk Sesi: {session_id}")

    db = Database()
    local_session_id = db.start_session()
    session_start_iso = datetime.now().isoformat()
    real_start_time = time.time()

    tracker = FocusTracker()
    cam = ThreadedCamera(src=0, width=CAMERA_WIDTH, height=CAMERA_HEIGHT).start()
    time.sleep(1.0)

    frame_count = 0
    ai_frame_count = 0

    # AI FPS — rolling window 30 frame untuk akurasi tinggi
    ai_frame_times = deque(maxlen=30)
    ai_fps = 0

    is_focused = True
    head = {}
    body = {}
    expr = {}

    show_t = False
    show_k = False
    show_e = False

    session_is_active = [True]

    def monitor_session():
        """Monitor status sesi dari web (skip jika standalone mode)."""
        if STANDALONE_MODE:
            return
        while session_is_active[0]:
            time.sleep(2.0)
            try:
                latest_session = get_latest_active_session()

                if latest_session is None:
                    print("\n[INFO] Sesi telah dihentikan dari Web! (Tidak ada sesi aktif)")
                    session_is_active[0] = False
                    break

                current_latest_id = latest_session.get("id") or latest_session.get("session_id")
                if current_latest_id != session_id:
                    print(f"\n[INFO] Sesi {session_id} telah selesai! (Sesi tidak lagi aktif)")
                    session_is_active[0] = False
                    break

            except Exception as e:
                print(f"[API ERROR] Gagal mengecek sesi web: {e}")

    threading.Thread(target=monitor_session, daemon=True).start()

    try:
        while session_is_active[0]:
            ret, shared_frame = cam.read()
            if not ret:
                break

            frame = shared_frame.copy()
            frame = cv2.flip(frame, 1)

            h_frame, w_frame, _ = frame.shape
            frame_count += 1

            # ========================================================
            # Pemrosesan AI tiap N frame
            # ========================================================
            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                head = analyze_head(rgb_frame)
                body = analyze_body(rgb_frame)
                expr = analyze_expression(rgb_frame)

                # Kompabilitas modul ekspresi baru & lama
                if 'label' not in expr and 'kelas' in expr:
                    expr['label'] = expr['kelas']

                is_focused, completed_event = tracker.update(head, body, expr)

                # Filter durasi berdasarkan jenis trigger
                if completed_event:
                    min_duration = get_min_duration(completed_event['triggers'])
                    if completed_event['duration_sec'] > min_duration:
                        print(f"[LOG] Off-Task: {completed_event['duration_sec']}s | {completed_event['triggers']}")
                        db.log_off_task_event(local_session_id, completed_event)

                # Update AI FPS (rolling window 30 frame)
                ai_frame_count += 1
                ai_now = time.time()
                ai_frame_times.append(ai_now)
                if len(ai_frame_times) >= 2:
                    duration = ai_frame_times[-1] - ai_frame_times[0]
                    if duration > 0:
                        ai_fps = (len(ai_frame_times) - 1) / duration

            # ========================================================
            # TAMPILAN UI
            # ========================================================
            if not HEADLESS_MODE:
                # --- Titik landmark (saat T/K/E ditekan) ---
                if show_t and body.get('landmarks'):
                    for idx, lm in enumerate(body['landmarks']):
                        if idx >= 11:
                            cv2.circle(frame, (int(lm.x * w_frame), int(lm.y * h_frame)),
                                       5, (255, 0, 255), -1)

                if show_k and head.get('landmarks'):
                    titik_kepala = [1, 10, 152, 234, 454]
                    for idx, lm in enumerate(head['landmarks']):
                        if idx in titik_kepala:
                            cv2.circle(frame, (int(lm.x * w_frame), int(lm.y * h_frame)),
                                       4, (0, 0, 255), -1)

                if show_e and expr.get('landmarks'):
                    titik_ekspresi = [61, 291, 13, 14, 159, 145, 33, 133, 386, 374,
                                      362, 263, 70, 63, 105, 66, 107, 336, 296, 334, 293, 300]
                    for idx, lm in enumerate(expr['landmarks']):
                        if idx in titik_ekspresi:
                            cv2.circle(frame, (int(lm.x * w_frame), int(lm.y * h_frame)),
                                       2, (0, 255, 255), -1)

                # --- Bounding box wajah & tubuh ---
                if head.get('landmarks'):
                    draw_bounding_box(frame, head['landmarks'],
                                      color=(0, 255, 0), label="WAJAH DIIKUTI")

                if body.get('landmarks'):
                    draw_bounding_box(frame, body['landmarks'],
                                      color=(255, 0, 255), label="TUBUH DIIKUTI")

                # --- Status utama ---
                person_detected = bool(head.get('landmarks') or body.get('landmarks'))

                if not person_detected:
                    draw_text_with_bg(frame, "STATUS: TIDAK ADA ORANG", (20, 40),
                                      font_scale=0.8, color=(0, 165, 255),
                                      bg_color=(0, 0, 0), thickness=2)
                    draw_text_with_bg(frame, "Sebab: Subjek tidak terdeteksi", (20, 75),
                                      font_scale=0.6, color=(0, 165, 255),
                                      bg_color=(0, 0, 0), thickness=2)
                else:
                    if is_focused:
                        status_text = "STATUS: ON-TASK"
                        status_color = (0, 255, 0)
                    else:
                        status_text = "STATUS: OFF-TASK"
                        status_color = (0, 0, 255)

                    draw_text_with_bg(frame, status_text, (20, 40),
                                      font_scale=0.8, color=status_color,
                                      bg_color=(0, 0, 0), thickness=2)

                    # Detail sebab
                    info_parts = []
                    info_parts.append(f"Arah ({head.get('direction', 'UNKNOWN')})")

                    if head.get('movement_event'):
                        info_parts.append("Gerakan >5s")

                    info_parts.append(f"Postur ({body.get('posture', 'UNKNOWN')})")

                    detail_ekspresi = expr.get('expression', expr.get('label', 'UNKNOWN'))
                    info_parts.append(f"Ekspresi ({detail_ekspresi})")

                    reason_text = " | ".join(info_parts)
                    draw_text_with_bg(frame, f"Sebab: {reason_text}", (20, 75),
                                      font_scale=0.55, color=status_color,
                                      bg_color=(0, 0, 0), thickness=2)

                # --- FPS ---
                draw_text_with_bg(frame, f"FPS: {int(ai_fps)}", (540, 40),
                                  font_scale=0.7, color=(0, 255, 255),
                                  bg_color=(0, 0, 0), thickness=2)

                # --- Legenda warna bounding box ---
                legend_y = CAMERA_HEIGHT - 60
                draw_text_with_bg(frame, "Legenda:", (CAMERA_WIDTH - 130, legend_y),
                                  font_scale=0.4, color=(255, 255, 255),
                                  bg_color=(0, 0, 0), thickness=1)
                cv2.rectangle(frame, (CAMERA_WIDTH - 130, legend_y + 5),
                              (CAMERA_WIDTH - 115, legend_y + 20), (0, 255, 0), -1)
                draw_text_with_bg(frame, "Wajah", (CAMERA_WIDTH - 110, legend_y + 18),
                                  font_scale=0.4, color=(255, 255, 255),
                                  bg_color=(0, 0, 0), thickness=1)
                cv2.rectangle(frame, (CAMERA_WIDTH - 70, legend_y + 5),
                              (CAMERA_WIDTH - 55, legend_y + 20), (255, 0, 255), -1)
                draw_text_with_bg(frame, "Tubuh", (CAMERA_WIDTH - 50, legend_y + 18),
                                  font_scale=0.4, color=(255, 255, 255),
                                  bg_color=(0, 0, 0), thickness=1)

                # --- Hint keyboard ---
                draw_text_with_bg(frame, "[T]ubuh | [K]epala | [E]kspresi | [Q]uit",
                                  (20, CAMERA_HEIGHT - 20),
                                  font_scale=0.5, color=(255, 255, 255),
                                  bg_color=(0, 0, 0), thickness=1)

                cv2.imshow("Sistem Deteksi", frame)

                # --- Input keyboard ---
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    session_is_active[0] = False
                    break
                elif key == ord('t'):
                    show_t = not show_t
                elif key == ord('k'):
                    show_k = not show_k
                elif key == ord('e'):
                    show_e = not show_e

    finally:
        # ========================================================
        # SAAT SESI BERHENTI — Simpan data & kirim ke Firebase
        # ========================================================
        print("\n[AI STOP] Sesi berakhir. Menyimpan data...")

        # Catat event off-task terakhir jika masih berlangsung
        if not tracker.is_focused:
            durasi = time.time() - tracker.off_task_start_time
            triggers_list = list(tracker.current_triggers)
            min_duration = get_min_duration(triggers_list)

            if durasi > min_duration:
                final_event = {
                    "start_time": time.strftime("%H:%M:%S", time.localtime(tracker.off_task_start_time)),
                    "duration_sec": round(durasi, 2),
                    "triggers": triggers_list
                }
                db.log_off_task_event(local_session_id, final_event)

        # Hitung hasil akhir sesi
        events = db.get_session_events(local_session_id)
        total_duration = time.time() - real_start_time
        total_off_task = sum(e["duration_sec"] for e in events)
        on_task_sec = max(0, total_duration - total_off_task)

        persentase = round((on_task_sec / total_duration) * 100, 2) if total_duration > 0 else 0

        kategori_db = "Fokus" if persentase >= 75 else "Tidak Fokus"
        kategori_layar = "On-Task" if persentase >= 75 else "Off-Task"

        # Simpan ke SQLite lokal
        db.end_session(local_session_id, total_duration, persentase, kategori_db)
        cam.stop()
        if not HEADLESS_MODE:
            cv2.destroyAllWindows()

        if not STANDALONE_MODE:
            # Kirim hasil ke API Web
            try:
                save_focus_result(anak_id, session_id, kategori_db)
                print(f"[API] Hasil sesi dikirim ke Web: {kategori_db}")
            except Exception as e:
                print(f"[API WARNING] Gagal menyimpan hasil akhir ke web: {e}")

            # Kirim data lengkap ke Firebase
            payload = {
                "session_id": session_id,
                "anak_id": anak_id,
                "session_date": session_start_iso,
                "total_duration_sec": round(total_duration, 2),
                "persentase_fokus": persentase,
                "kategori": kategori_db,
                "events": events
            }

            try:
                send_session_data(payload)
                print(f"[FIREBASE] Data sesi dikirim ke Firebase")
            except Exception as e:
                print(f"[API WARNING] Gagal mengirim ke Firebase: {e}")

        print(f"[FINISH] Persentase: {persentase}% | Kategori: {kategori_layar}")
        print(f"[FINISH] Total durasi: {round(total_duration, 1)}s | Off-task: {round(total_off_task, 1)}s")
        print(f"[FINISH] Sesi {session_id} selesai. Kembali ke standby.\n")


# ============================================================
# MAIN LOOP — Standby dan jalankan sesi
# ============================================================
def main():
    print("=== SISTEM ANALISIS FOKUS AKTIF ===")

    if STANDALONE_MODE:
        print("[INFO] Berjalan dalam Mode Standalone (Tanpa Koneksi Web/Permainan)")
        try:
            run_ai_loop(anak_id="TEST_ANAK_01", session_id="TEST_SESSION_01")
        except KeyboardInterrupt:
            print("\nSistem dihentikan secara manual (Ctrl+C).")
        return

    last_session_id = None
    while True:
        try:
            session_data = get_latest_active_session()

            if not session_data:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"Standby... Menunggu Permainan Dimulai...    ", end="\r")
                time.sleep(2)
                continue

            session_id = session_data.get("id") or session_data.get("session_id")
            if not session_id:
                time.sleep(2)
                continue

            if session_id == last_session_id:
                time.sleep(2)
                continue

            anak_id = session_data.get("anak_id")
            print(f"\n[SESSION ACTIVE] Session ID: {session_id} | Anak ID: {anak_id}")
            last_session_id = session_id

            run_ai_loop(anak_id, session_id)
            time.sleep(3)

        except KeyboardInterrupt:
            print("\n[INFO] Sistem dihentikan oleh user (Ctrl+C)")
            break
        except Exception as e:
            print(f"\n[ERROR] Error di loop utama: {e}")
            time.sleep(5)


if __name__ == "__main__":
    main()
