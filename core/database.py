import sqlite3
import time
import json


class Database:
    def __init__(self, path="output/fokus.db"):
        self.conn = sqlite3.connect(path)
        self.cursor = self.conn.cursor()
        self.cursor.execute("PRAGMA journal_mode=WAL;")
        self.create_tables()

    def create_tables(self):
        # Tabel session dengan field yang sudah di-rename
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS session (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_start TEXT,
            session_end TEXT,
            total_duration_sec REAL,
            persentase_fokus REAL,
            kategori TEXT
        )
        """)

        # Tabel off_task_events dengan field yang sudah di-rename
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS off_task_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            off_task_start TEXT,
            duration_sec REAL,
            triggers TEXT,
            FOREIGN KEY (session_id) REFERENCES session(id)
        )
        """)
        self.conn.commit()

    def start_session(self):
        """Catat sesi baru, return session_id."""
        session_start = time.strftime("%Y-%m-%d %H:%M:%S")
        self.cursor.execute(
            "INSERT INTO session (session_start) VALUES (?)",
            (session_start,)
        )
        self.conn.commit()
        return self.cursor.lastrowid

    def end_session(self, session_id, total_duration_sec=0,
                    persentase_fokus=0, kategori="UNKNOWN"):
        """Update sesi dengan hasil akhir."""
        session_end = time.strftime("%Y-%m-%d %H:%M:%S")
        self.cursor.execute("""
            UPDATE session
            SET session_end=?, total_duration_sec=?, persentase_fokus=?, kategori=?
            WHERE id=?
        """, (session_end, total_duration_sec, persentase_fokus,
              kategori, session_id))
        self.conn.commit()

    def log_off_task_event(self, session_id, event_data):
        """
        Catat event off-task ke database.

        event_data berisi:
            - start_time: waktu mulai (dari fusion.py)
            - duration_sec: durasi event
            - triggers: list of triggers
        """
        triggers_json = json.dumps(event_data["triggers"])
        self.cursor.execute("""
            INSERT INTO off_task_events (session_id, off_task_start, duration_sec, triggers)
            VALUES (?, ?, ?, ?)
        """, (session_id, event_data["start_time"],
              event_data["duration_sec"], triggers_json))
        self.conn.commit()

    def get_session_events(self, session_id):
        """Ambil semua event off-task untuk sesi tertentu."""
        self.cursor.execute("""
            SELECT off_task_start, duration_sec, triggers
            FROM off_task_events
            WHERE session_id=?
            ORDER BY off_task_start
        """, (session_id,))
        rows = self.cursor.fetchall()
        events = []
        for row in rows:
            events.append({
                "start_time": row[0],
                "duration_sec": row[1],
                "triggers": json.loads(row[2])
            })
        return events

    def get_session_summary(self, session_id):
        """Ambil ringkasan sesi."""
        self.cursor.execute("""
            SELECT session_start, session_end, total_duration_sec,
                   persentase_fokus, kategori
            FROM session
            WHERE id=?
        """, (session_id,))
        row = self.cursor.fetchone()
        if row:
            return {
                "session_start": row[0],
                "session_end": row[1],
                "total_duration_sec": row[2],
                "persentase_fokus": row[3],
                "kategori": row[4]
            }
        return None

    def close(self):
        """Tutup koneksi database."""
        self.conn.close()
