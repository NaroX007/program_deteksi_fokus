import sqlite3
import time

class Database:
    def __init__(self, path="output/fokus.db"):
        self.conn = sqlite3.connect(path)
        self.cursor = self.conn.cursor()

        # optimasi untuk performa
        self.cursor.execute("PRAGMA journal_mode=WAL;")
        self.cursor.execute("PRAGMA synchronous=NORMAL;")

        self.create_tables()

    # =========================
    # BUAT TABEL
    # =========================
    def create_tables(self):

        # session
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS session (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_time TEXT,
            end_time TEXT
        )
        """)

        # event log
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            timestamp REAL,
            type TEXT,
            detail TEXT
        )
        """)

        # summary (UPDATED)
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,

            head_move_count INTEGER,
            head_direction_count INTEGER,
            posture_stand_count INTEGER,
            expression_negative_count INTEGER,

            on_task_time REAL,
            off_task_time REAL,
            on_task_ratio REAL,

            final_status TEXT
        )
        """)

        self.conn.commit()

    # =========================
    # SESSION
    # =========================
    def start_session(self):
        start_time = time.strftime("%Y-%m-%d %H:%M:%S")
        self.cursor.execute(
            "INSERT INTO session (start_time) VALUES (?)",
            (start_time,)
        )
        self.conn.commit()
        return self.cursor.lastrowid

    def end_session(self, session_id):
        end_time = time.strftime("%Y-%m-%d %H:%M:%S")
        self.cursor.execute(
            "UPDATE session SET end_time=? WHERE id=?",
            (end_time, session_id)
        )
        self.conn.commit()

    # =========================
    # EVENT LOG
    # =========================
    def log_event(self, session_id, event_type, detail):
        self.cursor.execute("""
        INSERT INTO events (session_id, timestamp, type, detail)
        VALUES (?, ?, ?, ?)
        """, (session_id, time.time(), event_type, detail))
        self.conn.commit()

    # =========================
    # SIMPAN SUMMARY (UPDATED)
    # =========================
    def save_summary(self, session_id,
                     head_move_count,
                     head_direction_count,
                     posture_stand_count,
                     expression_negative_count,
                     on_task_time,
                     off_task_time,
                     on_task_ratio,
                     final_status):

        self.cursor.execute("""
        INSERT INTO summary (
            session_id,
            head_move_count,
            head_direction_count,
            posture_stand_count,
            expression_negative_count,
            on_task_time,
            off_task_time,
            on_task_ratio,
            final_status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            head_move_count,
            head_direction_count,
            posture_stand_count,
            expression_negative_count,
            on_task_time,
            off_task_time,
            on_task_ratio,
            final_status
        ))

        self.conn.commit()
