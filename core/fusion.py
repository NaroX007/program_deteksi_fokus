import time

class FocusTracker:
    def __init__(self):
        self.is_focused = True
        self.off_task_start_time = 0
        self.current_triggers = set()

    def update(self, head_data, body_data, expr_data):
        sekarang = time.time()
        triggers = []

        # Aturan Tidak Fokus:
        # 1. Kepala melihat Kiri/Kanan/Atas (langsung, tanpa timer 5s)
        if head_data.get("bad_direction", False):
            triggers.append(f"Arah Kepala ke {head_data.get('direction', '')})")
            
        # 2. Kepala terlalu banyak gerak selama > 5 detik
        if head_data.get("movement_event", False):
            triggers.append("Kepala Bergerak > 5 detik)")
        
        # 3. Postur selain duduk
        if body_data.get("active_not_focus", False):
            triggers.append("Tidak Duduk")
        
        # 4. Ekspresi tidak nyaman
        if expr_data.get("not_focus", False):
            triggers.append("Tidak Nyaman")

        kondisi_sedang_tidak_fokus = len(triggers) > 0
        event_selesai = None

        if self.is_focused and kondisi_sedang_tidak_fokus:
            # MULAI TIDAK FOKUS
            self.is_focused = False
            self.off_task_start_time = sekarang
            self.current_triggers = set(triggers)
        
        elif not self.is_focused:
            if kondisi_sedang_tidak_fokus:
                self.current_triggers.update(triggers)
            else:
                # KEMBALI FOKUS -> Catat Durasi
                durasi = sekarang - self.off_task_start_time
                event_selesai = {
                    "start_time": time.strftime("%H:%M:%S", time.localtime(self.off_task_start_time)),
                    "duration_sec": round(durasi, 2),
                    "triggers": list(self.current_triggers)
                }
                self.is_focused = True
                self.current_triggers = set()

        return self.is_focused, event_selesai
