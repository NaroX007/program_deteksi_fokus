import cv2
import time
import threading
from datetime import datetime

# Import modul deteksi AI Anda
from module.kepala import analyze_head
from module.postur_tubuh import analyze_body
from module.ekspresi import analyze_expression

# Import inti sistem
from core.database import Database
from core.fusion import FocusTracker
from core.firebase import send_session_data
from core.focus_api import (
    get_current_user,
    get_active_anak,
    get_active_session,
    get_latest_active_session,
    save_focus_result
)

# ==========================================
# KONFIGURASI SISTEM
# ==========================================
HEADLESS_MODE = False        # Set ke True jika dijalankan tanpa monitor
PROCESS_EVERY_N_FRAMES = 3   # Optimasi FPS
CAMERA_WIDTH = 640           # Resolusi optimal Raspi 5
CAMERA_HEIGHT = 480

# --- PENGATURAN TOLERANSI GERAKAN ---
# Sistem baru menganggap anak Off-Task jika gerakannya kacau lebih dari 3 detik.
MIN_OFF_TASK_DURATION = 3.0  
# ==========================================

class ThreadedCamera:
    def __init__(self, src=0, width=640, height=480):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
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
    prev_time = time.time()
    fps_smooth = 0

    is_focused = True
    head = {}
    body = {}
    expr = {}

    show_t = False  
    show_k = False  
    show_e = False  

    session_is_active = [True] 

    def monitor_session():
        while session_is_active[0]:
            time.sleep(2.0)
            if get_active_session(anak_id) is None:
                print("\n[INFO] Sesi telah dihentikan dari Web!")
                session_is_active[0] = False
                break

    threading.Thread(target=monitor_session, daemon=True).start()

    try:
        while session_is_active[0]:
            ret, shared_frame = cam.read()
            if not ret: break
            
            frame = shared_frame.copy()
            frame = cv2.flip(frame, 1)

            h_frame, w_frame, _ = frame.shape
            frame_count += 1

            now = time.time()
            if (now - prev_time) > 0:
                fps_smooth = (0.9 * fps_smooth) + (0.1 * (1 / (now - prev_time)))
            prev_time = now

            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                head = analyze_head(rgb_frame)
                body = analyze_body(rgb_frame)
                expr = analyze_expression(rgb_frame)

                is_focused, completed_event = tracker.update(head, body, expr)

                # FILTER DURASI GERAKAN EKSTREM
                if completed_event and completed_event['duration_sec'] > MIN_OFF_TASK_DURATION:
                    print(f"[LOG] Off-Task Ekstrem: {completed_event['duration_sec']}s | {completed_event['triggers']}")
                    db.log_off_task_event(local_session_id, completed_event)

            if not HEADLESS_MODE:
                if show_t and body.get('landmarks'):
                    for idx, lm in enumerate(body['landmarks']):
                        if idx >= 11: 
                            cv2.circle(frame, (int(lm.x * w_frame), int(lm.y * h_frame)), 5, (255, 0, 255), -1) 
                
                if show_k and head.get('landmarks'):
                    titik_kepala = [1, 10, 152, 234, 454]
                    for idx, lm in enumerate(head['landmarks']):
                        if idx in titik_kepala:
                            cv2.circle(frame, (int(lm.x * w_frame), int(lm.y * h_frame)), 4, (0, 0, 255), -1) 
                
                if show_e and expr.get('landmarks'):
                    titik_ekspresi = [61, 291, 13, 14, 159, 145, 33, 133, 386, 374, 362, 263, 70, 63, 105, 66, 107, 336, 296, 334, 293, 300]
                    for idx, lm in enumerate(expr['landmarks']):
                        if idx in titik_ekspresi:
                            cv2.circle(frame, (int(lm.x * w_frame), int(lm.y * h_frame)), 2, (0, 255, 255), -1) 

                # TAMPILAN UI TETAP ON-TASK / OFF-TASK
                if is_focused:
                    cv2.putText(frame, "STATUS: ON-TASK", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "STATUS: OFF-TASK", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    live_triggers = []
                    if head.get('bad_direction'): live_triggers.append(f"Arah ({head.get('direction', 'UNKNOWN')})")
                    if head.get('movement_event'): live_triggers.append("Gerakan Ekstrem")
                    if body.get('active_not_focus'): live_triggers.append(f"Postur ({body.get('posture', 'UNKNOWN')})")
                    if expr.get('not_focus'): live_triggers.append(f"Ekspresi ({expr.get('label', 'UNKNOWN')})")
                    
                    reason_text = " | ".join(live_triggers)
                    cv2.putText(frame, f"Sebab: {reason_text}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                cv2.putText(frame, f"FPS: {int(fps_smooth)}", (540, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, "Keyboard: [T]ubuh | [K]epala | [E]kspresi | [Q]uit", (20, CAMERA_HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow("Sistem Deteksi", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'): 
                    session_is_active[0] = False
                    break
                elif key == ord('t'): show_t = not show_t
                elif key == ord('k'): show_k = not show_k
                elif key == ord('e'): show_e = not show_e

    finally:
        if not tracker.is_focused:
            durasi = time.time() - tracker.off_task_start_time
            if durasi > MIN_OFF_TASK_DURATION:
                final_event = {
                    "start_time": time.strftime("%H:%M:%S", time.localtime(tracker.off_task_start_time)),
                    "duration_sec": round(durasi, 2),
                    "triggers": list(tracker.current_triggers)
                }
                db.log_off_task_event(local_session_id, final_event)

        # KALKULASI AKHIR SESI
        events = db.get_session_events(local_session_id)
        total_duration = time.time() - real_start_time
        total_off_task = sum(e["duration_sec"] for e in events)
        on_task_sec = max(0, total_duration - total_off_task)
        
        persentase = round((on_task_sec / total_duration) * 100, 2) if total_duration > 0 else 0
        
        # PEMISAHAN VARIABEL DATABASE DAN TAMPILAN (PILIHAN B)
        kategori_db = "Fokus" if persentase >= 75 else "Tidak Fokus"
        kategori_layar = "On-Task" if persentase >= 75 else "Off-Task"

        try:
            # Kirim data aman ke API Web teman Anda
            save_focus_result(anak_id, session_id, kategori_db)
        except Exception as e:
            print(f"[API WARNING] Gagal menyimpan hasil akhir ke web: {e}")

        # Simpan ke local database dengan data aman
        db.end_session(local_session_id, total_duration, persentase, kategori_db)
        cam.stop()
        if not HEADLESS_MODE: cv2.destroyAllWindows()

        payload = {
            "session_id": session_id,
            "anak_id": anak_id,
            "session_date": session_start_iso,
            "total_duration_sec": round(total_duration, 2),
            "persentase_fokus": persentase,
            "kategori": kategori_db, # Kirim data aman ke Firebase
            "events": events
        }
        
        try:
            send_session_data(payload)
        except Exception as e:
            print(f"[API WARNING] Gagal mengirim ke Firebase: {e}")
            
        # Cetak menggunakan bahasa akademis di Terminal
        print(f"[FINISH] Persentase: {persentase}% | Kategori: {kategori_layar}\n")

def main():
    print("=== SISTEM ANALISIS FOKUS AKTIF ===")
    last_session_id = None

    while True:
        try:
            session_data = get_latest_active_session()

            if not session_data:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Standby... Menunggu Permainan Dimulai...    ", end="\r")
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
            break
        except Exception as e:
            print(f"\nError di loop utama: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
