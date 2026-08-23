import time

class FocusTracker:
    def __init__(self):
        self.is_focused = True
        self.off_task_start_time = 0
        self.current_triggers = set()

    def update(self, head_data, body_data, expr_data):
        sekarang = time.time()
        triggers = []

        # ==========================================
        # CEK KEHADIRAN SUBJEK
        # ==========================================
        person_detected = bool(head_data.get('landmarks') or body_data.get('landmarks'))

        if not person_detected:
            # Jika tidak ada orang, langsung catat sebagai pemicu utama
            triggers.append("Subjek Tidak Terdeteksi")
        else:
            # Aturan Tidak Fokus lainnya hanya dievaluasi JIKA subjek terdeteksi
            
            # 1. Kepala melihat Kiri/Kanan/Atas (langsung, tanpa timer 5s)
            if head_data.get("bad_direction", False):
                arah = head_data.get('direction', '')
                triggers.append(f"Arah Kepala ({arah})")
                
            # 2. Kepala terlalu banyak gerak selama > 5 detik
            if head_data.get("movement_event", False):
                triggers.append("Kepala Bergerak > 5 detik")
            
            # 3. Postur selain duduk
            if body_data.get("active_not_focus", False):
                triggers.append("Tidak Duduk")
            
            # 4. Ekspresi tidak nyaman
            if expr_data.get("not_focus", False):
                # Kompatibilitas ganda: ambil 'expression' (modul baru) atau 'label' (modul lama)
                detail_ekspresi = expr_data.get("expression", expr_data.get("label", "Tidak Nyaman"))
                triggers.append(f"Ekspresi ({detail_ekspresi})")

        # ==========================================
        # KALKULASI STATUS FOKUS
        # ==========================================
        kondisi_sedang_tidak_fokus = len(triggers) > 0
        event_selesai = None

        if self.is_focused and kondisi_sedang_tidak_fokus:
            # MULAI TIDAK FOKUS
            self.is_focused = False
            self.off_task_start_time = sekarang
            self.current_triggers = set(triggers)
        
        elif not self.is_focused:
            if kondisi_sedang_tidak_fokus:
                # PERBARUI ALASAN (jika ada pemicu tambahan saat sedang tidak fokus)
                self.current_triggers.update(triggers)
            else:
                # KEMBALI FOKUS -> Catat Durasi ke dalam Log Event
                durasi = sekarang - self.off_task_start_time
                event_selesai = {
                    "start_time": time.strftime("%H:%M:%S", time.localtime(self.off_task_start_time)),
                    "duration_sec": round(durasi, 2),
                    "triggers": list(self.current_triggers)
                }
                # Reset status kembali fokus
                self.is_focused = True
                self.current_triggers = set()

        return self.is_focused, event_selesai
