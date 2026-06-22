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
        # Tabel session diupdate dengan 3 kolom baru
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS session (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_time TEXT,
            end_time TEXT,
            total_duration_sec REAL,
            persentase_fokus REAL,
            kategori TEXT
        )
        """)
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS off_task_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            start_time TEXT,
            duration_sec REAL,
            triggers TEXT
        )
        """)
        self.conn.commit()

    def start_session(self):
        start_time = time.strftime("%Y-%m-%d %H:%M:%S")
        self.cursor.execute("INSERT INTO session (start_time) VALUES (?)", (start_time,))
        self.conn.commit()
        return self.cursor.lastrowid

    # Fungsi end_session diupdate untuk menerima dan menyimpan hasil kalkulasi AI
    def end_session(self, session_id, total_duration_sec=0, persentase_fokus=0, kategori="UNKNOWN"):
        end_time = time.strftime("%Y-%m-%d %H:%M:%S")
        self.cursor.execute("""
            UPDATE session 
            SET end_time=?, total_duration_sec=?, persentase_fokus=?, kategori=? 
            WHERE id=?
        """, (end_time, total_duration_sec, persentase_fokus, kategori, session_id))
        self.conn.commit()

    def log_off_task_event(self, session_id, event_data):
        triggers_json = json.dumps(event_data["triggers"])
        self.cursor.execute("""
        INSERT INTO off_task_events (session_id, start_time, duration_sec, triggers)
        VALUES (?, ?, ?, ?)
        """, (session_id, event_data["start_time"], event_data["duration_sec"], triggers_json))
        self.conn.commit()

    def get_session_events(self, session_id):
        self.cursor.execute("SELECT start_time, duration_sec, triggers FROM off_task_events WHERE session_id=?", (session_id,))
        rows = self.cursor.fetchall()
        events = []
        for row in rows:
            events.append({
                "start_time": row[0],
                "duration_sec": row[1],
                "triggers": json.loads(row[2])
            })
        return events
